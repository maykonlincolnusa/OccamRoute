"""Unified inference routing interface over Scikit-learn and PyTorch complexity routers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from occamroute.context_slicer import SliceResult, SurgicalContextSlicer
from occamroute.feature_extractor import FEATURE_NAMES, GraphComplexityFeatures, SubgraphFeatureExtractor
from occamroute.router.sklearn_router import SklearnComplexityRouter
from occamroute.router.torch_router import TorchComplexityRouter

RouterBackend = Union[SklearnComplexityRouter, TorchComplexityRouter]
TierName = Literal["weak_local", "strong_frontier"]


@dataclass(frozen=True)
class RoutingDecision:
    """Structured routing verdict returned by OccamRouter."""

    backend: Literal["sklearn", "torch"]
    p_strong: float
    threshold: float
    selected_tier: TierName
    recommended_model: str
    is_strong: bool
    estimated_cost_per_1k_usd: float
    routing_latency_ms: float
    features: dict[str, float] | None = None


class OccamRouter:
    """Unified RouteLLM-style cost-quality router over subgraph topological features."""

    def __init__(
        self,
        model: RouterBackend,
        threshold: float = 0.50,
        weak_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct (4-bit)",
        strong_model_name: str = "gemini-3.8-flash",
        weak_cost_per_1k_usd: float = 0.25,
        strong_cost_per_1k_usd: float = 2.40,
    ) -> None:
        self.model = model
        self.backend_name: Literal["sklearn", "torch"] = (
            "sklearn" if isinstance(model, SklearnComplexityRouter) else "torch"
        )
        self.threshold = threshold
        self.weak_model_name = weak_model_name
        self.strong_model_name = strong_model_name
        self.weak_cost_per_1k_usd = weak_cost_per_1k_usd
        self.strong_cost_per_1k_usd = strong_cost_per_1k_usd

    @classmethod
    def from_artifact(
        cls,
        path: Path | str,
        backend: Literal["sklearn", "torch"] | None = None,
        threshold: float = 0.50,
    ) -> OccamRouter:
        artifact = Path(path)
        inferred = backend
        if inferred is None:
            inferred = "sklearn" if artifact.suffix == ".joblib" else "torch"
        loaded: RouterBackend = (
            SklearnComplexityRouter.load(artifact)
            if inferred == "sklearn"
            else TorchComplexityRouter.load(artifact)
        )
        return cls(model=loaded, threshold=threshold)

    @classmethod
    def from_sklearn_artifacts(
        cls,
        artifacts_dir: Path | str,
        threshold: float = 0.50,
    ) -> OccamRouter:
        """Load sklearn GBDT joblib or refit in memory from router_training_dataset.csv without overwriting disk."""
        art_dir = Path(artifacts_dir)
        joblib_path = art_dir / "sklearn_router.joblib"
        csv_path = art_dir / "router_training_dataset.csv"
        try:
            return cls.from_artifact(joblib_path, backend="sklearn", threshold=threshold)
        except Exception:
            if csv_path.exists():
                import pandas as pd

                df = pd.read_csv(csv_path)
                X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
                y = df["label"].to_numpy(dtype=np.int64)
                model = SklearnComplexityRouter().fit(X, y)
                return cls(model=model, threshold=threshold)
            raise FileNotFoundError(
                f"Neither loadable sklearn_router.joblib nor router_training_dataset.csv found in {art_dir}"
            )

    def route_features(
        self,
        features: GraphComplexityFeatures | np.ndarray,
        threshold: float | None = None,
    ) -> RoutingDecision:
        t0 = time.perf_counter()
        tau = self.threshold if threshold is None else threshold
        if isinstance(features, GraphComplexityFeatures):
            vec = features.to_numpy().reshape(1, -1)
            feat_dict = features.to_dict()
        else:
            vec = np.asarray(features, dtype=np.float32).reshape(1, -1)
            feat_dict = {name: float(vec[0, i]) for i, name in enumerate(FEATURE_NAMES)}

        p_strong = float(self.model.predict_proba(vec)[0])
        is_strong = p_strong >= tau
        tier: TierName = "strong_frontier" if is_strong else "weak_local"
        rec_model = self.strong_model_name if is_strong else self.weak_model_name
        est_cost = self.strong_cost_per_1k_usd if is_strong else self.weak_cost_per_1k_usd
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RoutingDecision(
            backend=self.backend_name,
            p_strong=round(p_strong, 4),
            threshold=tau,
            selected_tier=tier,
            recommended_model=rec_model,
            is_strong=is_strong,
            estimated_cost_per_1k_usd=est_cost,
            routing_latency_ms=latency_ms,
            features=feat_dict,
        )

    def route_slice(
        self,
        slice_result: SliceResult,
        extractor: SubgraphFeatureExtractor,
        threshold: float | None = None,
    ) -> RoutingDecision:
        feats = extractor.extract(slice_result)
        return self.route_features(feats, threshold=threshold)

    def route_query(
        self,
        query: str,
        slicer: SurgicalContextSlicer,
        extractor: SubgraphFeatureExtractor,
        token_budget: int = 1500,
        max_symbols: int = 6,
        k_hop: int = 2,
        threshold: float | None = None,
    ) -> tuple[SliceResult, RoutingDecision]:
        sliced = slicer.slice(query, token_budget=token_budget, max_symbols=max_symbols, k_hop=k_hop)
        decision = self.route_slice(sliced, extractor, threshold=threshold)
        return sliced, decision

    def sweep_thresholds(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        weak_quality: np.ndarray,
        strong_quality: np.ndarray,
        thresholds: list[float] | None = None,
    ) -> list[dict[str, float]]:
        """Evaluate RouteLLM cost-quality trade-offs (PGR & Cost Savings) across tau."""
        taus = [round(0.1 * i, 2) for i in range(1, 10)] if thresholds is None else thresholds
        probs = self.model.predict_proba(X_val)
        q_weak = float(np.mean(weak_quality))
        q_strong = float(np.mean(strong_quality))
        denom = max(1e-6, q_strong - q_weak)

        rows: list[dict[str, float]] = []
        for tau in taus:
            use_strong = probs >= tau
            y_pred = use_strong.astype(np.int64)
            routed_q = np.where(use_strong, strong_quality, weak_quality)
            avg_q = float(np.mean(routed_q))
            pgr = ((avg_q - q_weak) / denom) * 100.0
            strong_pct = float(np.mean(use_strong)) * 100.0
            avg_cost = float(
                np.mean(
                    np.where(use_strong, self.strong_cost_per_1k_usd, self.weak_cost_per_1k_usd)
                )
            )
            cost_savings = (1.0 - (avg_cost / self.strong_cost_per_1k_usd)) * 100.0
            rows.append(
                {
                    "threshold": float(tau),
                    "strong_call_pct": strong_pct,
                    "accuracy": float(accuracy_score(y_val, y_pred)),
                    "f1": float(f1_score(y_val, y_pred, zero_division=0)),
                    "avg_quality": avg_q,
                    "pgr_pct": pgr,
                    "avg_cost_per_1k_usd": avg_cost,
                    "cost_savings_pct": cost_savings,
                }
            )
        return rows
