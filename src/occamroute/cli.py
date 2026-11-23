"""Unified OccamRoute CLI (`slice`, `impact`, `map`, `mcp`, `dashboard`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from occamroute.context_slicer import SurgicalContextSlicer
from occamroute.feature_extractor import SubgraphFeatureExtractor
from occamroute.graph_builder import RepoGraphBuilder
from occamroute.impact_analyzer import GitDiffImpactAnalyzer, load_default_router
from occamroute.mcp_server import OccamRouteMCPServer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="occamroute", description="OccamRoute CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_slice = sub.add_parser("slice", help="Slice minimal AST context and route query complexity")
    p_slice.add_argument("query", help="Engineering question or symbol query")
    p_slice.add_argument("--repo", default=".")
    p_slice.add_argument("--budget", type=int, default=1500)
    p_slice.add_argument("--max-symbols", type=int, default=6)
    p_slice.add_argument("--k-hop", type=int, default=2)
    p_slice.add_argument("--threshold", type=float, default=0.30)
    p_slice.add_argument("--json", action="store_true", dest="as_json")

    p_imp = sub.add_parser("impact", help="Analyze git diff blast-radius over the AST call graph")
    p_imp.add_argument("--repo", default=".")
    p_imp.add_argument("--ref", default="HEAD")
    p_imp.add_argument("--patch-file", default=None)
    p_imp.add_argument("--budget", type=int, default=2000)
    p_imp.add_argument("--threshold", type=float, default=0.30)
    p_imp.add_argument("--markdown", action="store_true")

    p_map = sub.add_parser("map", help="Show RepoGraph summary and top PageRank architectural hubs")
    p_map.add_argument("--repo", default=".")
    p_map.add_argument("--top", type=int, default=10)

    p_mcp = sub.add_parser("mcp", help="Start zero-dependency stdio MCP server")
    p_mcp.add_argument("--repo", default=".")
    p_mcp.add_argument("--threshold", type=float, default=0.30)

    p_dash = sub.add_parser("dashboard", help="Launch OccamRoute KPI observability server")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8001)

    args = parser.parse_args(argv)

    if args.command == "map":
        graph = RepoGraphBuilder().build_from_directory(Path(args.repo))
        print(json.dumps(graph.summary_stats(), indent=2))
        print(f"\nTop {args.top} PageRank Hubs:")
        for nid, sc in sorted(graph.pagerank.items(), key=lambda x: -x[1])[: args.top]:
            if nid in graph.nodes:
                nd = graph.nodes[nid]
                print(f"  {nid:<56} PR={sc:.5f} | cyc={nd.cyclomatic_complexity} | tok={nd.token_count}")
        return 0

    if args.command == "slice":
        graph = RepoGraphBuilder().build_from_directory(Path(args.repo))
        slicer = SurgicalContextSlicer(graph)
        extractor = SubgraphFeatureExtractor(graph)
        router = load_default_router(graph, threshold=args.threshold)
        sl, dec = router.route_query(
            args.query,
            slicer=slicer,
            extractor=extractor,
            token_budget=args.budget,
            max_symbols=args.max_symbols,
            k_hop=args.k_hop,
        )
        if args.as_json:
            print(
                json.dumps(
                    {
                        "selected_tier": dec.selected_tier,
                        "recommended_model": dec.recommended_model,
                        "p_strong": dec.p_strong,
                        "context_tokens": sl.context_tokens,
                        "compression_pct": round(sl.compression_ratio * 100.0, 2),
                        "ordered_node_ids": sl.ordered_node_ids,
                        "context_text": sl.context_text,
                    },
                    indent=2,
                )
            )
        else:
            print(
                f"Tier: {dec.selected_tier} (P={dec.p_strong:.3f}, {dec.recommended_model}) | "
                f"Tokens: {sl.context_tokens}/{sl.token_budget} ({sl.compression_ratio * 100:.1f}% compression)"
            )
            print(f"U-Order Symbols: {sl.ordered_node_ids}\n")
            print(sl.context_text)
        return 0

    if args.command == "impact":
        graph = RepoGraphBuilder().build_from_directory(Path(args.repo))
        router = load_default_router(graph, threshold=args.threshold)
        analyzer = GitDiffImpactAnalyzer(graph, router=router)
        if args.patch_file:
            report = analyzer.analyze_patch(Path(args.patch_file).read_text(encoding="utf-8"), token_budget=args.budget)
        else:
            report = analyzer.analyze_git_ref(Path(args.repo), base_ref=args.ref, token_budget=args.budget)
        print(report.to_markdown() if args.markdown else report.to_markdown())
        return 0

    if args.command == "mcp":
        OccamRouteMCPServer(repo_dir=Path(args.repo), threshold=args.threshold).serve_stdio()
        return 0

    if args.command == "dashboard":
        import uvicorn

        uvicorn.run("occamroute.dashboard.app:app", host=args.host, port=args.port, log_level="info")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
