"""Smoke test running OccamRoute over its own `src/occamroute` codebase."""
from __future__ import annotations

import argparse
from pathlib import Path

from occamroute import (
    FEATURE_NAMES,
    RepoGraphBuilder,
    SubgraphFeatureExtractor,
    SurgicalContextSlicer,
    load_swebench_lite,
    tasks_to_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="OccamRoute Phase 1 smoke test")
    parser.add_argument("--swebench-limit", type=int, default=0, help="Optional live SWE-bench Lite sample size")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1] / "src" / "occamroute"
    graph = RepoGraphBuilder().build_from_directory(repo_root)
    stats = graph.summary_stats()

    print("=== OccamRoute RepoGraph Stats (`src/occamroute`) ===")
    for k, v in stats.items():
        print(f"  {k:<18}: {v}")

    top_hubs = sorted(graph.pagerank.items(), key=lambda item: -item[1])[:5]
    print("\n=== Top 5 PageRank Hubs ===")
    for sid, score in top_hubs:
        print(f"  {sid:<52} PR={score:.4f}")

    slicer = SurgicalContextSlicer(graph)
    extractor = SubgraphFeatureExtractor(graph)

    queries = [
        "Explain how `position_aware_order` mitigates Lost in the Middle.",
        "Trace how `RepoGraphBuilder.build_from_directory` resolves edges and calls `safe_pagerank`.",
        "How do `SurgicalContextSlicer.slice` and `SubgraphFeatureExtractor.extract` enforce `token_budget` and compute complexity features?",
    ]

    slices = []
    print("\n=== Surgical Slicing Results (budget=1200 tokens) ===")
    for idx, q in enumerate(queries, start=1):
        res = slicer.slice(q, token_budget=1200, max_symbols=5, k_hop=2)
        slices.append(res)
        print(
            f"[Q{idx}] tokens={res.context_tokens}/{res.token_budget} "
            f"(compression={res.compression_ratio * 100:.1f}%) | "
            f"nodes={len(res.ordered_node_ids)} | hops={res.hops_used} | {res.latency_ms:.2f} ms"
        )
        print(f"     ordered_nodes={res.ordered_node_ids}")

    X = extractor.extract_batch(slices)
    print(f"\n=== ML Feature Matrix X: shape={X.shape}, dtype={X.dtype} ===")
    for q_idx, row in enumerate(X, start=1):
        feat_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(FEATURE_NAMES, row))
        print(f"  Q{q_idx}: {feat_str}")

    if args.swebench_limit > 0:
        tasks = load_swebench_lite(limit=args.swebench_limit)
        df = tasks_to_dataframe(tasks)
        print(f"\n=== SWE-bench Lite Sample ({len(df)} tasks) ===")
        print(df[["instance_id", "repo", "num_gt_files", "ground_truth_files"]].to_string(index=False))


if __name__ == "__main__":
    main()
