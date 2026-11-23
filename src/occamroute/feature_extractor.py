"""Numeric graph complexity feature extractor for Scikit-learn and PyTorch routers."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np

from occamroute.context_slicer import SliceResult
from occamroute.graph_builder import RepoGraph, count_tokens

FEATURE_NAMES: tuple[str, ...] = (
    "num_nodes",
    "num_edges",
    "edge_density",
    "avg_node_degree",
    "num_files",
    "file_entropy",
    "degree_entropy",
    "max_call_depth",
    "mean_pagerank",
    "max_pagerank",
    "mean_cyclomatic",
    "max_cyclomatic",
    "context_tokens",
    "token_budget_utilization",
    "compression_ratio",
    "lexical_confidence",
    "query_token_count",
)


def _shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


@dataclass(frozen=True)
class GraphComplexityFeatures:
    """Fixed-schema 17-dimensional complexity vector for a sliced subgraph."""

    num_nodes: float
    num_edges: float
    edge_density: float
    avg_node_degree: float
    num_files: float
    file_entropy: float
    degree_entropy: float
    max_call_depth: float
    mean_pagerank: float
    max_pagerank: float
    mean_cyclomatic: float
    max_cyclomatic: float
    context_tokens: float
    token_budget_utilization: float
    compression_ratio: float
    lexical_confidence: float
    query_token_count: float

    @property
    def num_files_touched(self) -> float:
        return self.num_files

    @property
    def graph_density(self) -> float:
        return self.edge_density

    @property
    def total_tokens(self) -> float:
        return self.context_tokens

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def to_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.to_vector(), dtype=np.float32)


class SubgraphFeatureExtractor:
    """Computes topological and lexical complexity features from a SliceResult."""

    def __init__(self, repo_graph: RepoGraph) -> None:
        self.graph = repo_graph

    def extract(self, slice_result: SliceResult) -> GraphComplexityFeatures:
        node_ids = slice_result.ordered_node_ids
        sub_g = self.graph.nx_graph.subgraph(node_ids)
        n_nodes = sub_g.number_of_nodes()
        n_edges = sub_g.number_of_edges()
        edge_density = float(nx.density(sub_g)) if n_nodes > 1 else 0.0

        file_counts: dict[str, int] = defaultdict(int)
        prs: list[float] = []
        cycs: list[int] = []
        for nid in sub_g.nodes:
            sym = self.graph.nodes[nid]
            file_counts[sym.file_path] += 1
            prs.append(self.graph.pagerank.get(nid, 0.0))
            cycs.append(sym.cyclomatic_complexity)

        file_entropy = _shannon_entropy(list(file_counts.values()))

        deg_hist: dict[int, int] = defaultdict(int)
        total_deg = 0
        for _, deg in sub_g.degree():
            deg_hist[deg] += 1
            total_deg += deg
        degree_entropy = _shannon_entropy(list(deg_hist.values()))
        avg_node_degree = (total_deg / n_nodes) if n_nodes > 0 else 0.0

        max_call_depth = 0
        if n_nodes > 1 and n_edges > 0:
            for src in sub_g.nodes:
                lengths = nx.single_source_shortest_path_length(sub_g, src)
                if lengths:
                    max_call_depth = max(max_call_depth, max(lengths.values()))

        mean_pr = (sum(prs) / len(prs)) if prs else 0.0
        max_pr = max(prs) if prs else 0.0
        mean_cyc = (sum(cycs) / len(cycs)) if cycs else 0.0
        max_cyc = float(max(cycs)) if cycs else 0.0
        budget_util = (
            slice_result.context_tokens / slice_result.token_budget
            if slice_result.token_budget > 0
            else 0.0
        )
        seed_ratio = (len(slice_result.seed_node_ids) / n_nodes) if n_nodes > 0 else 0.0
        lexical_confidence = max(0.0, min(1.0, seed_ratio))

        return GraphComplexityFeatures(
            num_nodes=float(n_nodes),
            num_edges=float(n_edges),
            edge_density=edge_density,
            avg_node_degree=float(avg_node_degree),
            num_files=float(len(file_counts)),
            file_entropy=file_entropy,
            degree_entropy=degree_entropy,
            max_call_depth=float(max_call_depth),
            mean_pagerank=mean_pr,
            max_pagerank=max_pr,
            mean_cyclomatic=mean_cyc,
            max_cyclomatic=max_cyc,
            context_tokens=float(slice_result.context_tokens),
            token_budget_utilization=budget_util,
            compression_ratio=float(slice_result.compression_ratio),
            lexical_confidence=float(lexical_confidence),
            query_token_count=float(count_tokens(slice_result.query)),
        )

    def extract_batch(self, slices: list[SliceResult]) -> np.ndarray:
        """Stack extracted feature vectors into an (N, 14) float32 matrix."""
        if not slices:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.vstack([self.extract(sl).to_numpy() for sl in slices]).astype(np.float32)
