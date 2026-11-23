"""ML routing package for OccamRoute (Scikit-learn, PyTorch, and unified inference engine)."""
from __future__ import annotations

from occamroute.router.dataset_builder import (
    RouterDataset,
    augment_routing_dataset,
    build_routing_dataset,
    extract_repo_seed_samples,
)
from occamroute.router.engine import OccamRouter, RoutingDecision
from occamroute.router.sklearn_router import SklearnComplexityRouter, SklearnRouterConfig
from occamroute.router.torch_router import TabularRouterMLP, TorchComplexityRouter, TorchRouterConfig

__all__ = [
    "OccamRouter",
    "RouterDataset",
    "RoutingDecision",
    "SklearnComplexityRouter",
    "SklearnRouterConfig",
    "TabularRouterMLP",
    "TorchComplexityRouter",
    "TorchRouterConfig",
    "augment_routing_dataset",
    "build_routing_dataset",
    "extract_repo_seed_samples",
]
