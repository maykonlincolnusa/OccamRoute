"""Automated unit tests for OccamRoute AST graph, context slicing, features, and dataset loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from occamroute.context_slicer import SurgicalContextSlicer, position_aware_order
from occamroute.dataset_loader import (
    SWEBenchTask,
    count_patch_churn,
    extract_ground_truth_files,
    extract_patch_line_ranges,
    tasks_to_dataframe,
)
from occamroute.feature_extractor import FEATURE_NAMES, SubgraphFeatureExtractor
from occamroute.graph_builder import RepoGraphBuilder

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src" / "occamroute"


def _build_synthetic_repo(tmp_path: Path) -> Path:
    (tmp_path / "auth.py").write_text(
        '''class BaseAuth:
    def _sign(self, token: str) -> str:
        return f"sig:{token}"

class TokenAuth(BaseAuth):
    def authenticate(self, token: str) -> str:
        if not token:
            raise ValueError("empty token")
        return self._sign(token)
''',
        encoding="utf-8",
    )
    (tmp_path / "client.py").write_text(
        '''from auth import TokenAuth

def helper_normalize(url: str) -> str:
    return url.strip().lower()

def dispatch_request(url: str, token: str) -> str:
    clean = helper_normalize(url)
    auth = TokenAuth()
    return f"{clean}:{auth._sign(token)}"
''',
        encoding="utf-8",
    )
    return tmp_path


def test_repo_graph_builder_ast_and_edges(tmp_path: Path) -> None:
    repo_dir = _build_synthetic_repo(tmp_path)
    graph = RepoGraphBuilder().build_from_directory(repo_dir)

    assert "auth.py::BaseAuth._sign" in graph.nodes
    assert "auth.py::TokenAuth.authenticate" in graph.nodes
    assert "client.py::dispatch_request" in graph.nodes

    # Inheritance edge: TokenAuth -> BaseAuth
    assert graph.nx_graph.has_edge("auth.py::TokenAuth", "auth.py::BaseAuth")
    # Self-inherited method edge: TokenAuth.authenticate -> BaseAuth._sign
    assert graph.nx_graph.has_edge("auth.py::TokenAuth.authenticate", "auth.py::BaseAuth._sign")
    # Cross-file call edge: dispatch_request -> helper_normalize
    assert graph.nx_graph.has_edge("client.py::dispatch_request", "client.py::helper_normalize")

    auth_node = graph.nodes["auth.py::TokenAuth.authenticate"]
    assert auth_node.cyclomatic_complexity >= 2
    assert abs(sum(graph.pagerank.values()) - 1.0) < 1e-4


def test_position_aware_order_u_shape() -> None:
    assert position_aware_order([]) == []
    assert position_aware_order(["a"]) == ["a"]
    assert position_aware_order(["s0", "s1", "s2", "s3", "s4"]) == ["s0", "s2", "s4", "s3", "s1"]
    assert position_aware_order(["s0", "s1", "s2", "s3"]) == ["s0", "s2", "s3", "s1"]


def test_surgical_slicer_and_strict_token_budget() -> None:
    graph = RepoGraphBuilder().build_from_directory(PACKAGE_SRC)
    slicer = SurgicalContextSlicer(graph)

    res = slicer.slice(
        query="How does `SurgicalContextSlicer.slice` enforce `token_budget` with `position_aware_order`?",
        token_budget=1500,
        max_symbols=5,
        k_hop=2,
    )
    assert res.context_tokens <= 1500
    assert "context_slicer.py::SurgicalContextSlicer.slice" in res.ordered_node_ids
    assert res.compression_ratio > 0.0

    # Hard budget clamping even under very tight token budget
    tight = slicer.slice(
        query="Trace `RepoGraphBuilder.build_from_directory` and `SurgicalContextSlicer.slice`.",
        token_budget=120,
        max_symbols=6,
    )
    assert 0 < tight.context_tokens <= 120


def test_feature_extractor_matrix_for_ml() -> None:
    graph = RepoGraphBuilder().build_from_directory(PACKAGE_SRC)
    slicer = SurgicalContextSlicer(graph)
    extractor = SubgraphFeatureExtractor(graph)

    slices = [
        slicer.slice("Explain `position_aware_order`.", token_budget=900, max_symbols=4),
        slicer.slice(
            "Trace `RepoGraphBuilder.build_from_directory` resolving edges and `SurgicalContextSlicer.slice`.",
            token_budget=1600,
            max_symbols=6,
        ),
    ]
    mat = extractor.extract_batch(slices)
    assert mat.shape == (2, len(FEATURE_NAMES))
    assert mat.dtype == np.float32
    assert np.all(np.isfinite(mat))

    feats = extractor.extract(slices[1]).to_dict()
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert 0.0 <= feats["token_budget_utilization"] <= 1.0


def test_dataset_loader_patch_parsing() -> None:
    sample_patch = """diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -120,4 +120,6 @@ class PreparedRequest:
     def prepare_url(self, url, params):
-        pass
+        if not url:
+            raise ValueError("Missing URL")
diff --git a/requests/utils.py b/requests/utils.py
--- a/requests/utils.py
+++ b/requests/utils.py
@@ -42,2 +42,3 @@ def check_header_validity(header):
+    assert header is not None
"""
    gt_files = extract_ground_truth_files(sample_patch)
    assert gt_files == ["requests/models.py", "requests/utils.py"]

    hunks = extract_patch_line_ranges(sample_patch)
    assert hunks["requests/models.py"] == [(120, 123)]
    assert hunks["requests/utils.py"] == [(42, 43)]

    added, removed = count_patch_churn(sample_patch)
    assert (added, removed) == (3, 1)

    df = tasks_to_dataframe(
        [
            SWEBenchTask(
                instance_id="psf__requests-1234",
                repo="psf/requests",
                base_commit="abcdef123456",
                problem_statement="Missing URL check",
                patch=sample_patch,
                ground_truth_files=gt_files,
                ground_truth_hunks=hunks,
                patch_added_lines=added,
                patch_removed_lines=removed,
                num_gt_files=len(gt_files),
            )
        ]
    )
    assert list(df["instance_id"]) == ["psf__requests-1234"]
    assert int(df["num_gt_files"].iloc[0]) == 2
