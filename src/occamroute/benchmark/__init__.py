"""Comparative benchmark baselines, SWE-bench evaluator, and publication visualizers."""
from __future__ import annotations

from occamroute.benchmark.baselines import (
    BaselineContextResult,
    BruteForceBaseline,
    NaiveRAGBaseline,
)
from occamroute.benchmark.evaluator import (
    BenchmarkReport,
    BenchmarkTask,
    SWEBenchBenchmarkEvaluator,
    TaskEvalRecord,
    build_repo_eval_tasks,
    tasks_from_swebench_lite,
)
from occamroute.benchmark.visualizer import (
    generate_publication_charts,
    plot_pareto_frontier,
    plot_token_compression,
    render_from_artifacts,
)

__all__ = [
    "BaselineContextResult",
    "BenchmarkReport",
    "BenchmarkTask",
    "BruteForceBaseline",
    "NaiveRAGBaseline",
    "SWEBenchBenchmarkEvaluator",
    "TaskEvalRecord",
    "build_repo_eval_tasks",
    "generate_publication_charts",
    "plot_pareto_frontier",
    "plot_token_compression",
    "render_from_artifacts",
    "tasks_from_swebench_lite",
]
