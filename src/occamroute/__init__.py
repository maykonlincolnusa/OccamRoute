"""OccamRoute: AST code-graph context slicing and complexity feature extraction."""
from __future__ import annotations

from occamroute.context_slicer import SliceResult, SurgicalContextSlicer, position_aware_order
from occamroute.dataset_loader import (
    SWEBenchTask,
    count_patch_churn,
    extract_ground_truth_files,
    extract_patch_line_ranges,
    load_swebench_lite,
    tasks_to_dataframe,
)
from occamroute.feature_extractor import (
    FEATURE_NAMES,
    GraphComplexityFeatures,
    SubgraphFeatureExtractor,
)
from occamroute.graph_builder import (
    RepoGraph,
    RepoGraphBuilder,
    SymbolKind,
    SymbolNode,
    count_tokens,
    safe_pagerank,
)
from occamroute.impact_analyzer import DiffImpactReport, GitDiffImpactAnalyzer, load_default_router
from occamroute.mcp_server import OccamRouteMCPServer
from occamroute.router import (
    OccamRouter,
    RouterDataset,
    RoutingDecision,
    SklearnComplexityRouter,
    SklearnRouterConfig,
    TabularRouterMLP,
    TorchComplexityRouter,
    TorchRouterConfig,
    augment_routing_dataset,
    build_routing_dataset,
)

__all__ = [
    "DiffImpactReport",
    "FEATURE_NAMES",
    "GitDiffImpactAnalyzer",
    "GraphComplexityFeatures",
    "OccamRouteMCPServer",
    "OccamRouter",
    "RepoGraph",
    "RepoGraphBuilder",
    "RouterDataset",
    "RoutingDecision",
    "SWEBenchTask",
    "SklearnComplexityRouter",
    "SklearnRouterConfig",
    "SliceResult",
    "SubgraphFeatureExtractor",
    "SurgicalContextSlicer",
    "SymbolKind",
    "SymbolNode",
    "TabularRouterMLP",
    "TorchComplexityRouter",
    "TorchRouterConfig",
    "augment_routing_dataset",
    "build_routing_dataset",
    "count_patch_churn",
    "count_tokens",
    "extract_ground_truth_files",
    "extract_patch_line_ranges",
    "load_swebench_lite",
    "position_aware_order",
    "safe_pagerank",
    "tasks_to_dataframe",
]
