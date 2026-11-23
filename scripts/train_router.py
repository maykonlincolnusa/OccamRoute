"""Train, calibrate, sweep thresholds (tau in [0.1, 0.9]), and persist Sklearn & PyTorch routers."""
from __future__ import annotations

import json
from pathlib import Path

from occamroute import (
    OccamRouter,
    RepoGraphBuilder,
    SklearnComplexityRouter,
    SubgraphFeatureExtractor,
    SurgicalContextSlicer,
    TorchComplexityRouter,
    build_routing_dataset,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pkg_dir = project_root / "src" / "occamroute"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    repo_graph = RepoGraphBuilder().build_from_directory(pkg_dir)
    dataset = build_routing_dataset(repo_graph=repo_graph, n_samples=420, val_ratio=0.25, seed=42)
    print(
        f"Dataset ready | train={dataset.X_train.shape} | val={dataset.X_val.shape} "
        f"| positive_rate={dataset.y_train.mean():.2f}"
    )

    sklearn_router = SklearnComplexityRouter().fit(dataset.X_train, dataset.y_train)
    torch_router = TorchComplexityRouter().fit(dataset.X_train, dataset.y_train)

    sk_metrics = sklearn_router.evaluate(dataset.X_val, dataset.y_val, threshold=0.50)
    pt_metrics = torch_router.evaluate(dataset.X_val, dataset.y_val, threshold=0.50)

    sk_path = sklearn_router.save(artifacts_dir / "sklearn_router.joblib")
    pt_path = torch_router.save(artifacts_dir / "torch_router.pt")

    print("\n=== Validation Metrics (tau=0.50) ===")
    print(
        f"  Sklearn (Calibrated GBDT) | Acc={sk_metrics['accuracy']:.3f} | ROC-AUC={sk_metrics['roc_auc']:.3f} "
        f"| F1={sk_metrics['f1']:.3f} | Brier={sk_metrics['brier_score']:.3f} | ECE={sk_metrics['ece']:.3f}"
    )
    print(
        f"  PyTorch (Tabular MLP)     | Acc={pt_metrics['accuracy']:.3f} | ROC-AUC={pt_metrics['roc_auc']:.3f} "
        f"| F1={pt_metrics['f1']:.3f} | Brier={pt_metrics['brier_score']:.3f} | ECE={pt_metrics['ece']:.3f}"
    )

    sk_engine = OccamRouter.from_artifact(sk_path, threshold=0.50)
    pt_engine = OccamRouter.from_artifact(pt_path, threshold=0.50)

    sk_sweep = sk_engine.sweep_thresholds(
        dataset.X_val, dataset.y_val, dataset.weak_quality_val, dataset.strong_quality_val
    )
    pt_sweep = pt_engine.sweep_thresholds(
        dataset.X_val, dataset.y_val, dataset.weak_quality_val, dataset.strong_quality_val
    )

    print("\n=== Threshold Sweep (PyTorch MLP Router, tau in [0.1, 0.9]) ===")
    print("  tau  | Strong % | Accuracy |  F1   | Quality |  PGR %  | Cost Savings %")
    for row in pt_sweep:
        print(
            f"  {row['threshold']:.1f}  |  {row['strong_call_pct']:5.1f}%  |  {row['accuracy']:.3f}   | "
            f"{row['f1']:.3f} |  {row['avg_quality']:.3f}  | {row['pgr_pct']:6.1f}% |    {row['cost_savings_pct']:6.1f}%"
        )

    slicer = SurgicalContextSlicer(repo_graph)
    extractor = SubgraphFeatureExtractor(repo_graph)
    demo_queries = [
        "Explain what `position_aware_order` returns.",
        "Explain `count_tokens` fallback logic.",
        "Trace cross-module flow from `RepoGraphBuilder.build_from_directory` through `SurgicalContextSlicer.slice` and `SubgraphFeatureExtractor.extract`.",
        "Refactor `SklearnComplexityRouter` and `TorchComplexityRouter` shared calibration and `OccamRouter.sweep_thresholds`.",
    ]

    print("\n=== Live End-to-End OccamRouter Demo (`src/occamroute`) ===")
    for idx, q in enumerate(demo_queries, start=1):
        sliced, dec = pt_engine.route_query(q, slicer, extractor, token_budget=1500, max_symbols=6)
        print(
            f"  [Q{idx}] tier={dec.selected_tier:<15} | P(strong)={dec.p_strong:.3f} | "
            f"tokens={sliced.context_tokens:<4} | nodes={len(sliced.ordered_node_ids)} | "
            f"route_lat={dec.routing_latency_ms:.2f} ms"
        )

    metrics_payload = {
        "sklearn": sk_metrics,
        "torch": pt_metrics,
        "sklearn_sweep": sk_sweep,
        "torch_sweep": pt_sweep,
        "artifacts": {
            "sklearn": str(sk_path.relative_to(project_root)),
            "torch": str(pt_path.relative_to(project_root)),
        },
    }
    (artifacts_dir / "router_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts saved to {artifacts_dir}/")


if __name__ == "__main__":
    main()
