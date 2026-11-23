"""Tests for OccamRoute FastAPI KPI & live routing observability dashboard."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from occamroute.dashboard.app import create_app, export_standalone_html


def test_dashboard_api_endpoints_and_live_routing(tmp_path: Path) -> None:
    # Build isolated project root with minimal Python file so test is fast and self-contained
    src_pkg = tmp_path / "src" / "occamroute"
    src_pkg.mkdir(parents=True)
    (src_pkg / "sample.py").write_text(
        '''def helper(x: int) -> int:
    return x + 1

def pipeline(x: int) -> int:
    if x > 0:
        return helper(x)
    return 0
''',
        encoding="utf-8",
    )

    test_app = create_app(project_root=tmp_path)
    client = TestClient(test_app)

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "OccamRoute" in r_root.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_kpis = client.get("/api/kpis")
    assert r_kpis.status_code == 200
    kpis_payload = r_kpis.json()
    assert "kpis" in kpis_payload
    assert "pgr_pct" in kpis_payload["kpis"]
    assert "torch_roc_auc" in kpis_payload["kpis"]

    r_route = client.post(
        "/api/route",
        json={
            "query": "Trace `pipeline` calling `helper`.",
            "token_budget": 900,
            "max_symbols": 4,
            "k_hop": 2,
            "threshold": 0.50,
            "backend": "torch",
        },
    )
    assert r_route.status_code == 200
    route_data = r_route.json()
    assert route_data["context_tokens"] <= 900
    assert len(route_data["features"]) == 17
    assert route_data["selected_decision"]["selected_tier"] in ("weak_local", "strong_frontier")

    r_tel = client.get("/api/telemetry")
    assert r_tel.status_code == 200
    assert len(r_tel.json()) >= 1

    out_html = export_standalone_html(tmp_path / "dash.html", project_root=tmp_path)
    assert out_html.exists()
    assert "__OCCAMROUTE_INITIAL_KPIS__" in out_html.read_text(encoding="utf-8")
