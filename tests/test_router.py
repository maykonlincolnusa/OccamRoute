"""Unit tests for OccamRoute Phase 2 ML routers (Scikit-learn, PyTorch, and OccamRouter)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from occamroute import (
    FEATURE_NAMES,
    OccamRouter,
    RepoGraphBuilder,
    SklearnComplexityRouter,
    SubgraphFeatureExtractor,
    SurgicalContextSlicer,
    TorchComplexityRouter,
    TorchRouterConfig,
    build_routing_dataset,
)

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src" / "occamroute"


def test_routing_dataset_shapes_and_domains() -> None:
    graph = RepoGraphBuilder().build_from_directory(PACKAGE_SRC)
    ds = build_routing_dataset(repo_graph=graph, n_samples=240, val_ratio=0.25, seed=7)

    assert ds.X_train.shape[1] == len(FEATURE_NAMES)
    assert ds.X_val.shape[1] == len(FEATURE_NAMES)
    assert ds.X_train.dtype == np.float32
    assert set(np.unique(ds.y_train)).issubset({0, 1})
    assert set(np.unique(ds.y_val)).issubset({0, 1})
    assert np.all(np.isfinite(ds.X_train))
    assert np.all(ds.strong_quality_val >= ds.weak_quality_val - 0.1)


def test_sklearn_router_train_save_load(tmp_path: Path) -> None:
    graph = RepoGraphBuilder().build_from_directory(PACKAGE_SRC)
    ds = build_routing_dataset(repo_graph=graph, n_samples=260, val_ratio=0.25, seed=11)

    model = SklearnComplexityRouter().fit(ds.X_train, ds.y_train)
    metrics = model.evaluate(ds.X_val, ds.y_val, threshold=0.5)
    assert metrics["roc_auc"] >= 0.85
    assert metrics["accuracy"] >= 0.80

    ckpt_path = model.save(tmp_path / "sklearn_router.joblib")
    reloaded = SklearnComplexityRouter.load(ckpt_path)

    p_orig = model.predict_proba(ds.X_val)
    p_load = reloaded.predict_proba(ds.X_val)
    assert np.allclose(p_orig, p_load, atol=1e-6)


def test_torch_router_train_save_load(tmp_path: Path) -> None:
    graph = RepoGraphBuilder().build_from_directory(PACKAGE_SRC)
    ds = build_routing_dataset(repo_graph=graph, n_samples=260, val_ratio=0.25, seed=13)

    cfg = TorchRouterConfig(epochs=80, seed=13)
    model = TorchComplexityRouter(config=cfg).fit(ds.X_train, ds.y_train)
    metrics = model.evaluate(ds.X_val, ds.y_val, threshold=0.5)
    assert metrics["roc_auc"] >= 0.85
    assert metrics["accuracy"] >= 0.80

    ckpt_path = model.save(tmp_path / "torch_router.pt")
    reloaded = TorchComplexityRouter.load(ckpt_path)

    p_orig = model.predict_proba(ds.X_val)
    p_load = reloaded.predict_proba(ds.X_val)
    assert np.allclose(p_orig, p_load, atol=1e-6)


def test_unified_occam_router_and_threshold_sweep(tmp_path: Path) -> None:
    graph = RepoGraphBuilder().build_from_directory(PACKAGE_SRC)
    ds = build_routing_dataset(repo_graph=graph, n_samples=240, val_ratio=0.25, seed=21)

    sk_path = SklearnComplexityRouter().fit(ds.X_train, ds.y_train).save(tmp_path / "sk.joblib")
    pt_path = TorchComplexityRouter(TorchRouterConfig(epochs=70)).fit(ds.X_train, ds.y_train).save(tmp_path / "pt.pt")

    sk_engine = OccamRouter.from_artifact(sk_path, threshold=0.50)
    pt_engine = OccamRouter.from_artifact(pt_path, threshold=0.50)
    assert sk_engine.backend_name == "sklearn"
    assert pt_engine.backend_name == "torch"

    slicer = SurgicalContextSlicer(graph)
    extractor = SubgraphFeatureExtractor(graph)
    sliced, dec = pt_engine.route_query(
        "Explain `position_aware_order`.",
        slicer=slicer,
        extractor=extractor,
        token_budget=1000,
    )
    assert 0.0 <= dec.p_strong <= 1.0
    assert dec.selected_tier in ("weak_local", "strong_frontier")
    assert sliced.context_tokens <= 1000

    sweep = sk_engine.sweep_thresholds(
        ds.X_val, ds.y_val, ds.weak_quality_val, ds.strong_quality_val, thresholds=[0.1, 0.5, 0.9]
    )
    assert len(sweep) == 3
    assert sweep[0]["strong_call_pct"] >= sweep[-1]["strong_call_pct"]
