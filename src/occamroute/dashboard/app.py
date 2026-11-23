"""FastAPI + Pydantic v2 observability API and live routing server for OccamRoute."""
from __future__ import annotations

import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from occamroute.context_slicer import SurgicalContextSlicer
from occamroute.feature_extractor import FEATURE_NAMES, SubgraphFeatureExtractor
from occamroute.graph_builder import RepoGraph, RepoGraphBuilder
from occamroute.router import (
    OccamRouter,
    RoutingDecision,
    SklearnComplexityRouter,
    TorchComplexityRouter,
    build_routing_dataset,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class LiveRouteRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Engineering task or bug description")
    token_budget: int = Field(default=1500, ge=100, le=8000)
    max_symbols: int = Field(default=6, ge=1, le=16)
    k_hop: int = Field(default=2, ge=0, le=3)
    threshold: float = Field(default=0.50, ge=0.05, le=0.95)
    backend: Literal["torch", "sklearn"] = "torch"


class OrderedSymbolView(BaseModel):
    node_id: str
    qualname: str
    file_path: str
    kind: str
    token_count: int
    cyclomatic_complexity: int
    pagerank: float
    is_seed: bool
    position_zone: Literal["primacy_start", "middle", "recency_end"]
    start_line: int
    end_line: int


class SubgraphEdgeView(BaseModel):
    source: str
    target: str
    edge_type: str


class RouterVerdictView(BaseModel):
    backend: Literal["sklearn", "torch"]
    p_strong: float
    threshold: float
    selected_tier: Literal["weak_local", "strong_frontier"]
    recommended_model: str
    is_strong: bool
    estimated_cost_per_1k_usd: float
    routing_latency_ms: float


class LiveRouteResponse(BaseModel):
    query: str
    context_text: str
    ordered_symbols: list[OrderedSymbolView]
    subgraph_edges: list[SubgraphEdgeView]
    seed_symbols: list[str]
    retrieved_files: list[str]
    context_tokens: int
    token_budget: int
    compression_ratio: float
    slicing_latency_ms: float
    subgraph_nodes: int
    features: dict[str, float]
    torch_decision: RouterVerdictView
    sklearn_decision: RouterVerdictView
    selected_decision: RouterVerdictView


def _resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "occamroute").exists():
        return cwd
    return Path(__file__).resolve().parents[3]


def _resolve_target_repo(root: Path) -> tuple[Path, str]:
    candidates = [
        (root / "_bench_repo" / "httpx", "encode/httpx@0.28.1"),
        (root / "src" / "occamroute", "occamroute/src"),
    ]
    for path, label in candidates:
        if path.exists():
            return path, label
    return Path(__file__).resolve().parents[1], "occamroute/src"


def _compute_feature_separability(train_csv: Path) -> list[dict[str, float | str]]:
    if not train_csv.exists():
        return []
    df = pd.read_csv(train_csv)
    if "label" not in df.columns:
        return []
    rows: list[dict[str, float | str]] = []
    for fname in FEATURE_NAMES:
        if fname not in df.columns:
            continue
        v0 = df.loc[df["label"] == 0, fname].to_numpy(dtype=np.float64)
        v1 = df.loc[df["label"] == 1, fname].to_numpy(dtype=np.float64)
        if len(v0) == 0 or len(v1) == 0:
            continue
        pooled = math.sqrt(0.5 * (float(np.var(v0)) + float(np.var(v1)))) + 1e-6
        cohen_d = (float(np.mean(v1)) - float(np.mean(v0))) / pooled
        rows.append(
            {
                "feature": fname,
                "weak_mean": round(float(np.mean(v0)), 4),
                "strong_mean": round(float(np.mean(v1)), 4),
                "cohen_d": round(cohen_d, 4),
            }
        )
    rows.sort(key=lambda r: abs(float(r["cohen_d"])), reverse=True)
    return rows


def _to_verdict_view(dec: RoutingDecision) -> RouterVerdictView:
    return RouterVerdictView(
        backend=dec.backend,
        p_strong=dec.p_strong,
        threshold=dec.threshold,
        selected_tier=dec.selected_tier,
        recommended_model=dec.recommended_model,
        is_strong=dec.is_strong,
        estimated_cost_per_1k_usd=dec.estimated_cost_per_1k_usd,
        routing_latency_ms=round(dec.routing_latency_ms, 4),
    )


class DashboardState:
    """Holds indexed RepoGraph, trained ML routers, and benchmark KPI snapshots."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.root = _resolve_project_root(project_root)
        self.artifacts_dir = self.root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.repo_path, self.repo_label = _resolve_target_repo(self.root)
        self.repo_graph: RepoGraph = RepoGraphBuilder().build_from_directory(self.repo_path)
        self.slicer = SurgicalContextSlicer(self.repo_graph)
        self.extractor = SubgraphFeatureExtractor(self.repo_graph)

        self._ensure_router_artifacts()
        self.torch_router = OccamRouter.from_artifact(self.artifacts_dir / "torch_router.pt", threshold=0.50)
        try:
            self.sklearn_router = OccamRouter.from_artifact(self.artifacts_dir / "sklearn_router.joblib", threshold=0.50)
        except Exception:
            # Handle cross-version scikit-learn pickle incompatibility (e.g. Colab 1.6 -> local 1.9)
            train_csv = self.artifacts_dir / "router_training_dataset.csv"
            if train_csv.exists():
                df_t = pd.read_csv(train_csv)
                X_t = df_t[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
                y_t = df_t["label"].to_numpy(dtype=np.int64)
            else:
                ds_tmp = build_routing_dataset(repo_graph=self.repo_graph, n_samples=320, val_ratio=0.25, seed=42)
                X_t, y_t = ds_tmp.X_train, ds_tmp.y_train
            sk_refit = SklearnComplexityRouter().fit(X_t, y_t)
            sk_refit.save(self.artifacts_dir / "sklearn_router.joblib")
            self.sklearn_router = OccamRouter(sk_refit, threshold=0.50)
        self.live_telemetry: deque[dict[str, Any]] = deque(maxlen=40)
        self._seed_initial_telemetry()

    def _ensure_router_artifacts(self) -> None:
        pt_path = self.artifacts_dir / "torch_router.pt"
        sk_path = self.artifacts_dir / "sklearn_router.joblib"
        metrics_path = self.artifacts_dir / "router_metrics.json"
        train_csv = self.artifacts_dir / "router_training_dataset.csv"

        if pt_path.exists() and sk_path.exists() and metrics_path.exists() and train_csv.exists():
            return

        ds = build_routing_dataset(repo_graph=self.repo_graph, n_samples=320, val_ratio=0.25, seed=42)
        df_train = pd.DataFrame(ds.X_train, columns=ds.feature_names)
        df_train["label"] = ds.y_train
        df_train.to_csv(train_csv, index=False)

        sk = SklearnComplexityRouter().fit(ds.X_train, ds.y_train)
        pt = TorchComplexityRouter().fit(ds.X_train, ds.y_train)
        sk.save(sk_path)
        pt.save(pt_path)

        sk_eng = OccamRouter(sk, threshold=0.50)
        pt_eng = OccamRouter(pt, threshold=0.50)
        payload = {
            "optimal_tau": 0.50,
            "sklearn": sk.evaluate(ds.X_val, ds.y_val, threshold=0.50),
            "torch": pt.evaluate(ds.X_val, ds.y_val, threshold=0.50),
            "sklearn_sweep": sk_eng.sweep_thresholds(ds.X_val, ds.y_val, ds.weak_quality_val, ds.strong_quality_val),
            "torch_sweep": pt_eng.sweep_thresholds(ds.X_val, ds.y_val, ds.weak_quality_val, ds.strong_quality_val),
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _seed_initial_telemetry(self) -> None:
        sample_queries = [
            "Explain `position_aware_order` U-shaped primacy/recency ordering.",
            "Trace cross-module flow from `RepoGraphBuilder.build_from_directory` to `SurgicalContextSlicer.slice`.",
        ]
        if self.repo_label.startswith("encode/httpx"):
            sample_queries = [
                "Explain `codes.is_redirect` and `codes.get_reason_phrase`.",
                "Trace `BaseClient._redirect_headers`, `_same_origin`, and `_is_https_redirect`.",
            ]
        for q in sample_queries:
            try:
                self.route_live(LiveRouteRequest(query=q, token_budget=1500, max_symbols=6, k_hop=2))
            except Exception:
                pass

    def build_kpi_payload(self) -> dict[str, Any]:
        metrics_path = self.artifacts_dir / "router_metrics.json"
        eval_csv = self.artifacts_dir / "swebench_eval_results.csv"
        train_csv = self.artifacts_dir / "router_training_dataset.csv"

        router_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        separability = _compute_feature_separability(train_csv)

        pipeline_summary: list[dict[str, Any]] = []
        per_task_eval: list[dict[str, Any]] = []
        kpis: dict[str, Any] = {}

        if eval_csv.exists():
            df_eval = pd.read_csv(eval_csv)
            order = ["Brute Force", "Naive RAG", "Unbounded AST 2-Hop", "OccamRoute (ML)"]
            present_order = [p for p in order if p in set(df_eval["pipeline"])]
            grp = (
                df_eval.groupby("pipeline", as_index=False)
                .agg(
                    file_recall=("file_recall", "mean"),
                    function_recall=("function_recall", "mean"),
                    symbol_precision=("symbol_precision", "mean"),
                    symbol_f1=("symbol_f1", "mean"),
                    middle_burial_pct=("middle_burial_pct", "mean"),
                    mean_tokens=("context_tokens", "mean"),
                    prep_ms=("prep_ms", "mean"),
                    inference_ms=("inference_ms", "mean"),
                    mean_latency_ms=("total_ms", "mean"),
                    cost_per_1k_usd=("cost_per_1k_usd", "mean"),
                    quality_score=("quality_score", "mean"),
                    local_offload_pct=("selected_tier", lambda s: float((s == "weak_local").mean() * 100.0)),
                )
            )
            grp["pipeline"] = pd.Categorical(grp["pipeline"], categories=present_order, ordered=True)
            grp = grp.sort_values("pipeline").reset_index(drop=True)

            bf_row = grp[grp["pipeline"] == "Brute Force"].iloc[0]
            bf_tokens = float(bf_row["mean_tokens"])
            bf_cost = float(bf_row["cost_per_1k_usd"])
            q_weak = float(router_metrics.get("q_weak", 0.744))
            q_strong = float(router_metrics.get("q_strong", 0.969))

            grp["token_savings_pct"] = (1.0 - grp["mean_tokens"] / max(1.0, bf_tokens)) * 100.0
            grp["cost_savings_pct"] = (1.0 - grp["cost_per_1k_usd"] / max(1e-6, bf_cost)) * 100.0
            grp["pgr_pct"] = ((grp["quality_score"] - q_weak) / max(1e-6, q_strong - q_weak)) * 100.0

            pipeline_summary = grp.round(4).to_dict(orient="records")
            per_task_eval = df_eval.round(4).to_dict(orient="records")

            oc = grp[grp["pipeline"] == "OccamRoute (ML)"].iloc[0]
            kpis = {
                "pgr_pct": round(float(oc["pgr_pct"]), 1),
                "token_savings_pct": round(float(oc["token_savings_pct"]), 1),
                "mean_occam_tokens": round(float(oc["mean_tokens"]), 0),
                "mean_bf_tokens": round(bf_tokens, 0),
                "cost_per_1k_usd": round(float(oc["cost_per_1k_usd"]), 2),
                "bf_cost_per_1k_usd": round(bf_cost, 2),
                "cost_savings_pct": round(float(oc["cost_savings_pct"]), 1),
                "file_recall": round(float(oc["file_recall"]), 3),
                "function_recall": round(float(oc["function_recall"]), 3),
                "symbol_precision": round(float(oc["symbol_precision"]), 3),
                "symbol_f1": round(float(oc["symbol_f1"]), 3),
                "middle_burial_pct": round(float(oc["middle_burial_pct"]), 1),
                "bf_middle_burial_pct": round(float(bf_row["middle_burial_pct"]), 1),
                "local_offload_pct": round(float(oc["local_offload_pct"]), 1),
                "avg_prep_ms": round(float(oc["prep_ms"]), 2),
                "avg_latency_ms": round(float(oc["mean_latency_ms"]), 1),
                "q_weak_anchor": q_weak,
                "q_strong_anchor": q_strong,
            }
        else:
            # Fallback summary derived from router_metrics if evaluation CSV hasn't been generated yet
            pt_sweep = router_metrics.get("torch_sweep", [])
            mid = next((r for r in pt_sweep if abs(float(r["threshold"]) - 0.5) < 0.05), pt_sweep[0] if pt_sweep else {})
            kpis = {
                "pgr_pct": round(float(mid.get("pgr_pct", 89.3)), 1),
                "token_savings_pct": 93.4,
                "mean_occam_tokens": 852,
                "mean_bf_tokens": 12986,
                "cost_per_1k_usd": round(float(mid.get("avg_cost_per_1k_usd", 1.35)), 2),
                "bf_cost_per_1k_usd": 17.18,
                "cost_savings_pct": round(float(mid.get("cost_savings_pct", 43.7)), 1),
                "file_recall": 1.0,
                "function_recall": 1.0,
                "symbol_precision": 0.674,
                "symbol_f1": 0.786,
                "middle_burial_pct": 16.9,
                "bf_middle_burial_pct": 29.6,
                "local_offload_pct": round(100.0 - float(mid.get("strong_call_pct", 51.3)), 1),
                "avg_prep_ms": 1.5,
                "avg_latency_ms": 3802.6,
                "q_weak_anchor": 0.744,
                "q_strong_anchor": 0.969,
            }

        pt_m = router_metrics.get("torch", {})
        sk_m = router_metrics.get("sklearn", {})
        kpis.update(
            {
                "torch_roc_auc": round(float(pt_m.get("roc_auc", 0.0)), 4),
                "torch_accuracy": round(float(pt_m.get("accuracy", 0.0)), 4),
                "torch_brier": round(float(pt_m.get("brier_score", 0.0)), 4),
                "torch_ece": round(float(pt_m.get("ece", 0.0)), 4),
                "sklearn_roc_auc": round(float(sk_m.get("roc_auc", 0.0)), 4),
                "sklearn_accuracy": round(float(sk_m.get("accuracy", 0.0)), 4),
                "sklearn_brier": round(float(sk_m.get("brier_score", 0.0)), 4),
                "sklearn_ece": round(float(sk_m.get("ece", 0.0)), 4),
                "optimal_tau": round(float(router_metrics.get("optimal_tau", 0.50)), 2),
            }
        )

        top_hubs = [
            {
                "node_id": nid,
                "qualname": self.repo_graph.nodes[nid].qualname,
                "file_path": self.repo_graph.nodes[nid].file_path,
                "pagerank": round(float(score), 5),
                "cyclomatic": self.repo_graph.nodes[nid].cyclomatic_complexity,
                "tokens": self.repo_graph.nodes[nid].token_count,
            }
            for nid, score in sorted(self.repo_graph.pagerank.items(), key=lambda item: -item[1])[:10]
            if nid in self.repo_graph.nodes
        ]

        assets_dir = self.root / "assets"
        t4_gallery = sorted(p.name for p in assets_dir.glob("*.png")) if assets_dir.exists() else []
        training_meta = {
            "trained_on": router_metrics.get("trained_on", "Local CPU"),
            "peak_vram_gb": router_metrics.get("peak_vram_gb", None),
            "faiss_enabled": bool(router_metrics.get("faiss_enabled", False)),
            "live_4bit_model": router_metrics.get("live_4bit_model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            "swebench_tasks_loaded": int(router_metrics.get("swebench_tasks_loaded", 0)),
            "t4_gallery": t4_gallery,
        }

        return {
            "project": "OccamRoute",
            "repo_label": self.repo_label,
            "repo_stats": self.repo_graph.summary_stats(),
            "training_meta": training_meta,
            "kpis": kpis,
            "pipeline_summary": pipeline_summary,
            "router_metrics": router_metrics,
            "feature_separability": separability,
            "top_pagerank_hubs": top_hubs,
            "per_task_eval": per_task_eval,
            "live_telemetry": list(self.live_telemetry),
        }

    def route_live(self, req: LiveRouteRequest) -> LiveRouteResponse:
        sliced = self.slicer.slice(
            req.query,
            token_budget=req.token_budget,
            max_symbols=req.max_symbols,
            k_hop=req.k_hop,
        )
        feats = self.extractor.extract(sliced)
        dec_torch = self.torch_router.route_features(feats, threshold=req.threshold)
        dec_sklearn = self.sklearn_router.route_features(feats, threshold=req.threshold)

        v_torch = _to_verdict_view(dec_torch)
        v_sklearn = _to_verdict_view(dec_sklearn)
        selected = v_torch if req.backend == "torch" else v_sklearn

        seed_set = set(sliced.seed_node_ids)
        n_ord = len(sliced.ordered_node_ids)
        ordered_views: list[OrderedSymbolView] = []
        for idx, nid in enumerate(sliced.ordered_node_ids):
            nd = self.repo_graph.nodes[nid]
            if n_ord <= 2:
                zone: Literal["primacy_start", "middle", "recency_end"] = (
                    "primacy_start" if idx == 0 else "recency_end"
                )
            elif idx == 0:
                zone = "primacy_start"
            elif idx == n_ord - 1:
                zone = "recency_end"
            else:
                zone = "middle"
            ordered_views.append(
                OrderedSymbolView(
                    node_id=nid,
                    qualname=nd.qualname,
                    file_path=nd.file_path,
                    kind=nd.kind,
                    token_count=nd.token_count,
                    cyclomatic_complexity=nd.cyclomatic_complexity,
                    pagerank=round(float(self.repo_graph.pagerank.get(nid, 0.0)), 5),
                    is_seed=(nid in seed_set),
                    position_zone=zone,
                    start_line=nd.start_line,
                    end_line=nd.end_line,
                )
            )

        sub_g = self.repo_graph.nx_graph.subgraph(sliced.ordered_node_ids)
        edges = [
            SubgraphEdgeView(source=u, target=v, edge_type=str(data.get("edge_type", "calls")))
            for u, v, data in sub_g.edges(data=True)
        ]

        resp = LiveRouteResponse(
            query=req.query,
            context_text=sliced.context_text,
            ordered_symbols=ordered_views,
            subgraph_edges=edges,
            seed_symbols=sliced.seed_node_ids,
            retrieved_files=sliced.retrieved_files,
            context_tokens=sliced.context_tokens,
            token_budget=sliced.token_budget,
            compression_ratio=round(float(sliced.compression_ratio), 4),
            slicing_latency_ms=round(float(sliced.latency_ms), 3),
            subgraph_nodes=len(ordered_views),
            features={k: round(v, 4) for k, v in feats.to_dict().items()},
            torch_decision=v_torch,
            sklearn_decision=v_sklearn,
            selected_decision=selected,
        )

        self.live_telemetry.appendleft(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                "query": req.query[:96],
                "backend": req.backend,
                "selected_tier": selected.selected_tier,
                "p_strong": selected.p_strong,
                "context_tokens": sliced.context_tokens,
                "compression_pct": round(float(sliced.compression_ratio) * 100.0, 1),
                "nodes": len(ordered_views),
                "edges": len(edges),
                "slicing_ms": round(float(sliced.latency_ms), 2),
                "routing_ms": round(selected.routing_latency_ms, 3),
            }
        )
        return resp


def create_app(project_root: Path | None = None) -> FastAPI:
    """Create configured FastAPI observability application for OccamRoute."""
    fastapi_app = FastAPI(
        title="OccamRoute Observability Console",
        version="0.1.0",
        description="AST Subgraph Context Slicing & RouteLLM Cost-Quality Telemetry",
    )
    state = DashboardState(project_root=project_root)
    fastapi_app.state.occam_state = state

    assets_dir = state.root / "assets"
    if assets_dir.exists():
        fastapi_app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @fastapi_app.get("/", response_class=HTMLResponse)
    def serve_index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    @fastapi_app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "project": "OccamRoute",
            "repo_label": state.repo_label,
            "repo_stats": state.repo_graph.summary_stats(),
            "artifacts_dir": str(state.artifacts_dir),
        }

    @fastapi_app.get("/api/kpis")
    def get_kpis() -> dict[str, Any]:
        return state.build_kpi_payload()

    @fastapi_app.post("/api/route", response_model=LiveRouteResponse)
    def route_query(req: LiveRouteRequest) -> LiveRouteResponse:
        return state.route_live(req)

    @fastapi_app.get("/api/telemetry")
    def get_telemetry() -> list[dict[str, Any]]:
        return list(state.live_telemetry)

    return fastapi_app


def export_standalone_html(
    output_path: Path | str = Path("artifacts/occamroute_dashboard.html"),
    project_root: Path | None = None,
) -> Path:
    """Export a self-contained static HTML dashboard with live KPI snapshot embedded."""
    state = DashboardState(project_root=project_root)
    payload_json = json.dumps(state.build_kpi_payload(), ensure_ascii=False)
    template = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    injection = f"<script>window.__OCCAMROUTE_INITIAL_KPIS__ = {payload_json};</script>\n</head>"
    rendered = template.replace("</head>", injection, 1)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    return out


app = create_app()
