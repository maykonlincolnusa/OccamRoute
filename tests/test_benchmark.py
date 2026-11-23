"""Automated unit tests for OccamRoute benchmark baselines, evaluator, visualizer, and Colab notebook."""
from __future__ import annotations

import json
from pathlib import Path

from occamroute import OccamRouter, RepoGraphBuilder, build_routing_dataset, SklearnComplexityRouter
from occamroute.benchmark import (
    BruteForceBaseline,
    NaiveRAGBaseline,
    SWEBenchBenchmarkEvaluator,
    build_repo_eval_tasks,
    generate_publication_charts,
)


def _build_local_graph_and_router(tmp_path: Path):
    src_root = Path(__file__).resolve().parents[1] / "src" / "occamroute"
    repo_graph = RepoGraphBuilder().build_from_directory(src_root)
    dataset = build_routing_dataset(repo_graph=repo_graph, n_samples=120, val_ratio=0.25, seed=7)
    sk = SklearnComplexityRouter().fit(dataset.X_train, dataset.y_train)
    model_path = sk.save(tmp_path / "router.joblib")
    router = OccamRouter.from_artifact(model_path, threshold=0.45)
    return repo_graph, router


def test_baselines_chunking_and_retrieval() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "occamroute"
    repo_graph = RepoGraphBuilder().build_from_directory(src_root)

    bf = BruteForceBaseline(repo_graph)
    bf_res = bf.retrieve("Explain SurgicalContextSlicer.slice", max_files=3)
    assert bf_res.pipeline == "Brute Force"
    assert len(bf_res.retrieved_files) == 3
    assert bf_res.context_tokens > 500
    assert len(bf_res.retrieved_symbols) > 0

    rag = NaiveRAGBaseline(repo_graph, chunk_tokens=256, overlap_tokens=32, backend="tfidf")
    assert rag.active_backend == "tfidf"
    assert len(rag.chunks) > 5
    rag_res = rag.retrieve("SurgicalContextSlicer position_aware_order token budget", top_k=3)
    assert rag_res.pipeline == "Naive RAG"
    assert rag_res.num_chunks == 3
    assert rag_res.context_tokens > 0
    assert len(rag_res.retrieved_files) >= 1


def test_evaluator_and_publication_charts(tmp_path: Path) -> None:
    repo_graph, router = _build_local_graph_and_router(tmp_path)
    tasks = build_repo_eval_tasks(repo_graph, max_tasks=6)
    assert len(tasks) >= 4

    evaluator = SWEBenchBenchmarkEvaluator(
        repo_graph=repo_graph,
        router=router,
        rag_backend="tfidf",
        token_budget=1400,
        max_symbols=6,
        k_hop=2,
        top_k_rag=4,
    )
    report = evaluator.evaluate(tasks)
    assert set(report.summary_df["pipeline"]) == {"Brute Force", "Naive RAG", "OccamRoute (ML)"}
    required_cols = {
        "mean_tokens",
        "token_savings_pct",
        "file_recall",
        "function_recall",
        "symbol_f1",
        "middle_burial_pct",
        "quality_score",
        "cost_per_1k_usd",
        "pgr_pct",
    }
    assert required_cols.issubset(set(report.summary_df.columns))

    bf_tokens = float(report.summary_df.loc[report.summary_df["pipeline"] == "Brute Force", "mean_tokens"].iloc[0])
    oc_tokens = float(report.summary_df.loc[report.summary_df["pipeline"] == "OccamRoute (ML)", "mean_tokens"].iloc[0])
    assert oc_tokens < bf_tokens

    saved = report.save(tmp_path / "bench_out")
    assert saved["results_csv"].exists()
    assert saved["summary_json"].exists()

    charts = generate_publication_charts(report, output_dir=tmp_path / "docs_assets")
    for key in ("pareto_frontier", "token_compression"):
        chart_file = charts[key]
        assert chart_file.exists()
        raw = chart_file.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(raw) > 10_000


def test_colab_notebook_structure() -> None:
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "01_occamroute_colab_benchmark.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    assert nb.get("metadata", {}).get("accelerator") == "GPU"
    sources = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    assert "SWEBenchBenchmarkEvaluator" in sources
    assert "generate_publication_charts" in sources
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in sources
