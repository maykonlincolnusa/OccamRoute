"""Zero-dependency JSON-RPC 2.0 Stdio MCP Server for OccamRoute."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from occamroute.context_slicer import SurgicalContextSlicer
from occamroute.feature_extractor import SubgraphFeatureExtractor
from occamroute.graph_builder import RepoGraphBuilder
from occamroute.impact_analyzer import GitDiffImpactAnalyzer, load_default_router


class OccamRouteMCPServer:
    """Exposes OccamRoute AST slicing, diff impact, and repo hubs via MCP stdio."""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, repo_dir: Path | str = ".", threshold: float = 0.30) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.graph = RepoGraphBuilder().build_from_directory(self.repo_dir)
        self.slicer = SurgicalContextSlicer(self.graph)
        self.extractor = SubgraphFeatureExtractor(self.graph)
        self.router = load_default_router(self.graph, threshold=threshold)
        self.impact_analyzer = GitDiffImpactAnalyzer(self.graph, router=self.router)

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "occamroute_slice_context",
                "description": (
                    "Extract a token-budgeted k-hop AST subgraph with U-shaped position-aware ordering "
                    "and ML complexity routing verdict (weak_local vs strong_frontier)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Engineering question, bug, or symbol query"},
                        "token_budget": {"type": "integer", "default": 1500},
                        "max_symbols": {"type": "integer", "default": 6},
                        "k_hop": {"type": "integer", "default": 2},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "occamroute_diff_impact",
                "description": (
                    "Analyze a unified git diff (or current repo HEAD diff) against the AST call graph "
                    "to compute blast-radius callers, risk level, and recommended LLM tier."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string", "description": "Optional unified git diff text"},
                        "base_ref": {"type": "string", "default": "HEAD"},
                        "token_budget": {"type": "integer", "default": 2000},
                    },
                },
            },
            {
                "name": "occamroute_repo_map",
                "description": "Return repository AST graph statistics and top PageRank architectural hubs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "top_k": {"type": "integer", "default": 10},
                    },
                },
            },
        ]

    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "occamroute_slice_context":
            sl, dec = self.router.route_query(
                query=str(args["query"]),
                slicer=self.slicer,
                extractor=self.extractor,
                token_budget=int(args.get("token_budget", 1500)),
                max_symbols=int(args.get("max_symbols", 6)),
                k_hop=int(args.get("k_hop", 2)),
            )
            payload = {
                "selected_tier": dec.selected_tier,
                "recommended_model": dec.recommended_model,
                "p_strong": dec.p_strong,
                "context_tokens": sl.context_tokens,
                "compression_pct": round(sl.compression_ratio * 100.0, 2),
                "ordered_symbols": sl.ordered_node_ids,
                "retrieved_files": sl.retrieved_files,
                "context_text": sl.context_text,
            }
            return json.dumps(payload, indent=2)

        if name == "occamroute_diff_impact":
            patch = args.get("patch")
            budget = int(args.get("token_budget", 2000))
            report = (
                self.impact_analyzer.analyze_patch(str(patch), token_budget=budget)
                if patch
                else self.impact_analyzer.analyze_git_ref(
                    self.repo_dir, base_ref=str(args.get("base_ref", "HEAD")), token_budget=budget
                )
            )
            return report.to_markdown() + "\n\n```python\n" + report.context_slice.context_text + "\n```"

        if name == "occamroute_repo_map":
            top_k = int(args.get("top_k", 10))
            hubs = [
                {
                    "node_id": nid,
                    "pagerank": round(float(score), 5),
                    "cyclomatic": self.graph.nodes[nid].cyclomatic_complexity,
                    "tokens": self.graph.nodes[nid].token_count,
                }
                for nid, score in sorted(self.graph.pagerank.items(), key=lambda x: -x[1])[:top_k]
                if nid in self.graph.nodes
            ]
            return json.dumps({"stats": self.graph.summary_stats(), "top_hubs": hubs}, indent=2)

        raise ValueError(f"Unknown OccamRoute MCP tool: {name}")

    def handle_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params") or {}

        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "occamroute", "version": "0.1.0"},
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": self._tool_definitions()}}
        if method == "tools/call":
            try:
                text = self._call_tool(str(params.get("name", "")), dict(params.get("arguments") or {}))
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": text}]},
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": str(exc)},
                }

        if msg_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_request(msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
