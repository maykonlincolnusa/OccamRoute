"""Unit tests for OccamRoute Phase 3 real-world tools (Git Diff Impact, MCP Server, CLI)."""
from __future__ import annotations

import json
from pathlib import Path

from occamroute.cli import main as cli_main
from occamroute.graph_builder import RepoGraphBuilder
from occamroute.impact_analyzer import GitDiffImpactAnalyzer
from occamroute.mcp_server import OccamRouteMCPServer


def _create_repo(tmp_path: Path) -> Path:
    (tmp_path / "core.py").write_text(
        '''def compute_tax(amount: float) -> float:
    if amount <= 0:
        return 0.0
    return amount * 0.15
''',
        encoding="utf-8",
    )
    (tmp_path / "billing.py").write_text(
        '''from core import compute_tax

def build_invoice(subtotal: float) -> float:
    return subtotal + compute_tax(subtotal)
''',
        encoding="utf-8",
    )
    return tmp_path


def test_git_diff_impact_analyzer_blast_radius(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    graph = RepoGraphBuilder().build_from_directory(repo)
    analyzer = GitDiffImpactAnalyzer(graph)

    patch = """diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
@@ -1,4 +1,5 @@
 def compute_tax(amount: float) -> float:
     if amount <= 0:
         return 0.0
-    return amount * 0.15
+    return amount * 0.18
"""
    report = analyzer.analyze_patch(patch, token_budget=1000)
    assert report.changed_files == ["core.py"]
    assert "core.py::compute_tax" in report.changed_symbols
    assert "billing.py::build_invoice" in report.impacted_callers
    assert report.context_slice.context_tokens <= 1000
    assert "OccamRoute PR Impact" in report.to_markdown()


def test_mcp_server_protocol_and_tools(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    server = OccamRouteMCPServer(repo_dir=repo, threshold=0.30)

    init_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "occamroute"

    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert list_resp is not None
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert tool_names == {"occamroute_slice_context", "occamroute_diff_impact", "occamroute_repo_map"}

    slice_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "occamroute_slice_context",
                "arguments": {"query": "How does `build_invoice` call `compute_tax`?", "token_budget": 800},
            },
        }
    )
    assert slice_resp is not None
    payload = json.loads(slice_resp["result"]["content"][0]["text"])
    assert payload["context_tokens"] <= 800
    assert "billing.py::build_invoice" in payload["ordered_symbols"]


def test_cli_map_slice_and_impact(tmp_path: Path, capsys) -> None:
    repo = _create_repo(tmp_path)
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(
        """diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
@@ -2,2 +2,2 @@ def compute_tax(amount: float) -> float:
-    if amount <= 0:
+    if amount < 0:
""",
        encoding="utf-8",
    )

    assert cli_main(["map", "--repo", str(repo), "--top", "2"]) == 0
    assert "num_nodes" in capsys.readouterr().out

    assert cli_main(["slice", "Explain `compute_tax`", "--repo", str(repo), "--json"]) == 0
    sliced = json.loads(capsys.readouterr().out)
    assert sliced["selected_tier"] in ("weak_local", "strong_frontier")

    assert cli_main(["impact", "--repo", str(repo), "--patch-file", str(patch_file), "--markdown"]) == 0
    assert "core.py::compute_tax" in capsys.readouterr().out
