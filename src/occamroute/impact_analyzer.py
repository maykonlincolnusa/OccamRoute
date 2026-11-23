"""Git diff AST blast-radius analyzer and PR complexity router."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx

from occamroute.context_slicer import SliceResult, position_aware_order
from occamroute.dataset_loader import count_patch_churn, extract_patch_line_ranges
from occamroute.feature_extractor import SubgraphFeatureExtractor
from occamroute.graph_builder import RepoGraph, count_tokens
from occamroute.router import OccamRouter, RoutingDecision, TorchComplexityRouter, TorchRouterConfig, build_routing_dataset


def load_default_router(repo_graph: RepoGraph, threshold: float = 0.30) -> OccamRouter:
    """Load trained GBDT (default) or PyTorch router checkpoint, or fit a lightweight fallback."""
    art_dirs = [
        repo_graph.root_dir / "artifacts",
        repo_graph.root_dir.parent / "artifacts",
        repo_graph.root_dir.parent.parent / "artifacts",
        Path("artifacts"),
    ]
    for art_dir in art_dirs:
        if (art_dir / "sklearn_router.joblib").exists() or (art_dir / "router_training_dataset.csv").exists():
            return OccamRouter.from_sklearn_artifacts(art_dir, threshold=threshold)

    for art_dir in art_dirs:
        pt_ckpt = art_dir / "torch_router.pt"
        if pt_ckpt.exists():
            return OccamRouter.from_artifact(pt_ckpt, threshold=threshold)

    ds = build_routing_dataset(repo_graph=repo_graph, n_samples=180, val_ratio=0.2, seed=42)
    pt = TorchComplexityRouter(TorchRouterConfig(epochs=60, seed=42)).fit(ds.X_train, ds.y_train)
    return OccamRouter(pt, threshold=threshold)


@dataclass(frozen=True)
class DiffImpactReport:
    """Structured AST blast-radius and routing verdict for a git diff or patch."""

    changed_files: list[str]
    changed_symbols: list[str]
    impacted_callers: list[str]
    impacted_files: list[str]
    added_lines: int
    removed_lines: int
    context_slice: SliceResult
    routing_decision: RoutingDecision
    risk_level: Literal["LOW", "HIGH"]

    def to_markdown(self) -> str:
        tier_label = (
            "Frontier LLM (`strong_frontier`)"
            if self.routing_decision.is_strong
            else "Local 4-bit GPU (`weak_local`)"
        )
        lines = [
            f"## OccamRoute PR Impact & Routing Report (`{self.risk_level}` RISK)",
            "",
            f"- **Changed Files ({len(self.changed_files)}):** {', '.join(f'`{f}`' for f in self.changed_files) or 'none'}",
            f"- **Modified AST Symbols ({len(self.changed_symbols)}):** {', '.join(f'`{s}`' for s in self.changed_symbols) or 'none'}",
            f"- **Blast-Radius Callers ({len(self.impacted_callers)} across {len(self.impacted_files)} files):** {', '.join(f'`{c}`' for c in self.impacted_callers) or 'none'}",
            f"- **Diff Churn:** `+{self.added_lines}` / `-{self.removed_lines}` lines",
            f"- **Surgical Context:** `{self.context_slice.context_tokens}` tokens (`{self.context_slice.compression_ratio * 100:.1f}%` compression vs raw repo)",
            f"- **Recommended Review Tier:** {tier_label} | `P(strong)={self.routing_decision.p_strong:.3f}` | `tau={self.routing_decision.threshold:.2f}` | `${self.routing_decision.estimated_cost_per_1k_usd:.2f}/1k`",
        ]
        return "\n".join(lines)


class GitDiffImpactAnalyzer:
    """Maps unified git diffs onto RepoGraph AST nodes and traverses incoming callers."""

    def __init__(self, repo_graph: RepoGraph, router: OccamRouter | None = None) -> None:
        self.graph = repo_graph
        self.extractor = SubgraphFeatureExtractor(repo_graph)
        self.router = router or load_default_router(repo_graph)

    def _match_indexed_path(self, patch_path: str) -> str | None:
        if patch_path in self.graph.file_to_nodes:
            return patch_path
        for indexed in self.graph.file_to_nodes:
            if patch_path.endswith("/" + indexed) or indexed.endswith("/" + patch_path):
                return indexed
        return None

    def analyze_patch(
        self,
        patch_text: str,
        token_budget: int = 2000,
        max_callers: int = 6,
        k_hop: int = 2,
    ) -> DiffImpactReport:
        t0 = time.perf_counter()
        raw_hunks = extract_patch_line_ranges(patch_text)
        added, removed = count_patch_churn(patch_text)

        changed_files: list[str] = []
        changed_symbols: list[str] = []
        for raw_path, ranges in raw_hunks.items():
            matched = self._match_indexed_path(raw_path)
            if not matched:
                continue
            if matched not in changed_files:
                changed_files.append(matched)
            for nid in self.graph.file_to_nodes.get(matched, []):
                if nid not in self.graph.leaf_ids:
                    continue
                nd = self.graph.nodes[nid]
                for h_start, h_end in ranges:
                    if max(nd.start_line, h_start) <= min(nd.end_line, h_end):
                        if nid not in changed_symbols:
                            changed_symbols.append(nid)
                        break

        # Reverse call-graph traversal: find downstream callers impacted by changed symbols
        rev_graph = self.graph.nx_graph.reverse(copy=False)
        caller_dist: dict[str, int] = {}
        for sym in changed_symbols:
            if sym not in rev_graph:
                continue
            sp = nx.single_source_shortest_path_length(rev_graph, sym, cutoff=k_hop)
            for nb, dist in sp.items():
                if nb in changed_symbols or nb not in self.graph.leaf_ids:
                    continue
                if nb not in caller_dist or dist < caller_dist[nb]:
                    caller_dist[nb] = dist

        ranked_callers = sorted(
            caller_dist.keys(),
            key=lambda sid: (caller_dist[sid], -self.graph.pagerank.get(sid, 0.0)),
        )[:max_callers]

        ranked_all = changed_symbols + ranked_callers
        budgeted_ids: list[str] = []
        blocks: dict[str, str] = {}
        used_tokens = 0
        for sid in ranked_all:
            nd = self.graph.nodes[sid]
            blk = f"# FILE: {nd.file_path} | SYMBOL: {nd.qualname} (L{nd.start_line}-L{nd.end_line})\n{nd.source}"
            toks = count_tokens(blk)
            sep = 2 if budgeted_ids else 0
            if used_tokens + sep + toks <= token_budget or not budgeted_ids:
                budgeted_ids.append(sid)
                blocks[sid] = blk
                used_tokens += sep + toks

        ordered_ids = position_aware_order(budgeted_ids)
        context_text = "\n\n".join(blocks[sid] for sid in ordered_ids)
        actual_tokens = count_tokens(context_text)
        hops_used = max((caller_dist[c] for c in ranked_callers if c in blocks), default=0)

        slice_res = SliceResult(
            query=f"Analyze git diff impact on {', '.join(changed_files) or 'repository'}",
            context_text=context_text,
            ordered_node_ids=ordered_ids,
            ranked_node_ids=budgeted_ids,
            seed_node_ids=[s for s in changed_symbols if s in blocks],
            retrieved_files=sorted({self.graph.nodes[s].file_path for s in ordered_ids}),
            context_tokens=actual_tokens,
            token_budget=token_budget,
            hops_used=hops_used,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            raw_repo_tokens=self.graph.total_raw_tokens,
        )

        feats = self.extractor.extract(slice_res)
        decision = self.router.route_features(feats)
        impacted_files = sorted({self.graph.nodes[c].file_path for c in ranked_callers})
        risk_level: Literal["LOW", "HIGH"] = (
            "HIGH"
            if (decision.is_strong or len(impacted_files) > 1 or (added + removed) >= 25)
            else "LOW"
        )

        return DiffImpactReport(
            changed_files=sorted(changed_files),
            changed_symbols=changed_symbols,
            impacted_callers=ranked_callers,
            impacted_files=impacted_files,
            added_lines=added,
            removed_lines=removed,
            context_slice=slice_res,
            routing_decision=decision,
            risk_level=risk_level,
        )

    def analyze_git_ref(
        self,
        repo_dir: Path | str,
        base_ref: str = "HEAD",
        token_budget: int = 2000,
    ) -> DiffImpactReport:
        repo_path = Path(repo_dir).resolve()
        proc = subprocess.run(
            ["git", "-C", str(repo_path), "diff", base_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        patch_text = proc.stdout
        if not patch_text.strip() and base_ref == "HEAD":
            proc2 = subprocess.run(
                ["git", "-C", str(repo_path), "diff", "HEAD~1", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            patch_text = proc2.stdout
        return self.analyze_patch(patch_text, token_budget=token_budget)
