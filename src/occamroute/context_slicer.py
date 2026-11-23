"""Surgical k-hop subgraph context slicing with hard token budgets and position-aware ordering."""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from occamroute.graph_builder import RepoGraph, count_tokens, safe_pagerank


def position_aware_order(ranked_ids: list[str]) -> list[str]:
    """
    Apply U-shaped primacy/recency ordering (Lost in the Middle, arXiv:2307.03172).
    Highest-priority items sit at the start and end: [s0, s1, s2, s3, s4] -> [s0, s2, s4, s3, s1].
    """
    left: list[str] = []
    right: list[str] = []
    for idx, node_id in enumerate(ranked_ids):
        if idx % 2 == 0:
            left.append(node_id)
        else:
            right.append(node_id)
    return left + list(reversed(right))


def _split_identifier_tokens(text: str) -> list[str]:
    raw_parts = re.findall(r"[A-Za-z0-9_]+", text)
    tokens: list[str] = []
    for part in raw_parts:
        for snake in part.split("_"):
            if not snake:
                continue
            camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", snake).split()
            tokens.extend(t.lower() for t in camel if len(t) >= 2)
        if "_" in part and len(part) >= 3:
            tokens.append(part.lower())
    return tokens


@dataclass
class SliceResult:
    """Output of SurgicalContextSlicer containing budgeted, position-ordered context."""

    query: str
    context_text: str
    ordered_node_ids: list[str]
    ranked_node_ids: list[str]
    seed_node_ids: list[str]
    retrieved_files: list[str]
    context_tokens: int
    token_budget: int
    hops_used: int
    latency_ms: float
    raw_repo_tokens: int
    compression_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        if self.raw_repo_tokens > 0:
            self.compression_ratio = 1.0 - (self.context_tokens / self.raw_repo_tokens)
        else:
            self.compression_ratio = 0.0


class SurgicalContextSlicer:
    """Extracts token-budgeted k-hop subgraphs ranked by structural relevance + local PageRank."""

    def __init__(self, repo_graph: RepoGraph) -> None:
        self.graph = repo_graph
        self.candidate_ids = sorted(self.graph.leaf_ids)
        self.doc_tokens: dict[str, set[str]] = {}
        self.idf: dict[str, float] = {}
        self._build_structural_index()

    def _build_structural_index(self) -> None:
        df: dict[str, int] = defaultdict(int)
        for sid in self.candidate_ids:
            node = self.graph.nodes[sid]
            meta = f"{node.file_path} {node.qualname} {node.signature} {node.docstring}"
            toks = set(_split_identifier_tokens(meta))
            self.doc_tokens[sid] = toks
            for tok in toks:
                df[tok] += 1
        n_docs = max(1, len(self.candidate_ids))
        self.idf = {tok: math.log((n_docs + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}

    def _extract_query_signals(self, query: str) -> tuple[set[str], set[str], set[str]]:
        backticked = re.findall(r"`([^`]+)`", query)
        qualified_refs: set[str] = set()
        raw_idents: set[str] = set()
        for item in backticked:
            cleaned = item.strip().rstrip("()")
            if "." in cleaned:
                qualified_refs.add(cleaned.lower())
            else:
                raw_idents.add(cleaned.lower())
        for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", query):
            qualified_refs.add(m.lower())
        for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b", query):
            raw_idents.add(m.lower())
        return qualified_refs, raw_idents, set(_split_identifier_tokens(query))

    def _score_seeds(self, query: str) -> tuple[list[tuple[float, str]], set[str], set[str], set[str]]:
        qualified_refs, raw_idents, q_tokens = self._extract_query_signals(query)
        scores: list[tuple[float, str]] = []
        for sid in self.candidate_ids:
            node = self.graph.nodes[sid]
            qname_low = node.qualname.lower()
            name_low = node.name.lower()
            score = 0.0
            if qname_low in qualified_refs:
                score += 50.0
            elif any(
                ref.endswith(f".{name_low}")
                and node.parent_class
                and ref.startswith(node.parent_class.lower())
                for ref in qualified_refs
            ):
                score += 45.0
            if name_low in raw_idents:
                score += 32.0
            elif node.parent_class and node.parent_class.lower() in raw_idents:
                score += 4.0
            score += sum(self.idf.get(tok, 1.0) for tok in (self.doc_tokens[sid] & q_tokens))
            if score > 0.0:
                scores.append((score, sid))
        scores.sort(key=lambda item: (-item[0], self.graph.nodes[item[1]].token_count))
        return scores, qualified_refs, raw_idents, q_tokens

    def _fit_to_budget(self, block: str, remaining_tokens: int) -> str | None:
        if remaining_tokens <= 0:
            return None
        if count_tokens(block) <= remaining_tokens:
            return block
        lines = block.splitlines()
        kept: list[str] = []
        for ln in lines:
            candidate = "\n".join([*kept, ln, "# ... [truncated to budget]"])
            if count_tokens(candidate) <= remaining_tokens:
                kept.append(ln)
            else:
                break
        if not kept:
            return None
        return "\n".join([*kept, "# ... [truncated to budget]"])

    def slice(
        self,
        query: str,
        token_budget: int = 2000,
        max_symbols: int = 8,
        k_hop: int = 2,
        undirected_expansion: bool = False,
    ) -> SliceResult:
        """Slice a token-budgeted k-hop subgraph and serialize with position-aware ordering."""
        t0 = time.perf_counter()
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")

        scores, qualified_refs, raw_idents, q_tokens = self._score_seeds(query)

        seeds: list[str] = []
        if scores:
            top_score = scores[0][0]
            cutoff = max(12.0, top_score * 0.42)
            for sc, sid in scores:
                if len(seeds) >= max_symbols:
                    break
                if sc >= cutoff or (len(seeds) < 2 and sc >= top_score * 0.28):
                    seeds.append(sid)

        ranked_selected = list(seeds)
        hops_used = 0
        if seeds and len(ranked_selected) < max_symbols:
            khop_dist = self.graph.k_hop_neighbors(seeds, k=k_hop, undirected=undirected_expansion)
            sub_nodes = [nid for nid in khop_dist if nid in self.graph.leaf_ids]
            sub_g = self.graph.nx_graph.subgraph(sub_nodes)
            local_pr = safe_pagerank(sub_g) if sub_g.number_of_nodes() > 1 else {}

            seed_out = set().union(*(self.graph.nx_graph.successors(s) for s in seeds if s in self.graph.nx_graph))
            candidates: list[tuple[float, str]] = []
            for nid in sub_nodes:
                if nid in ranked_selected:
                    continue
                node = self.graph.nodes[nid]
                dist = max(1, khop_dist[nid])
                q_overlap = len(self.doc_tokens[nid] & q_tokens)
                sc = 8.0 * local_pr.get(nid, 0.0) + 2.5 / dist + q_overlap * 1.6
                if node.name.lower() in raw_idents or node.qualname.lower() in qualified_refs:
                    sc += 20.0
                is_private_callee = nid in seed_out and node.name.startswith("_")
                if is_private_callee:
                    sc += 3.0
                if (q_overlap >= 1 or is_private_callee or node.name.lower() in raw_idents) and sc >= 4.2:
                    candidates.append((sc, nid))

            candidates.sort(key=lambda item: -item[0])
            for _, nid in candidates:
                if len(ranked_selected) >= max_symbols:
                    break
                ranked_selected.append(nid)
                hops_used = max(hops_used, khop_dist.get(nid, 1))

        # Enforce hard token budget while preserving priority rank
        budgeted_ids: list[str] = []
        blocks_by_id: dict[str, str] = {}
        used_tokens = 0
        for sid in ranked_selected:
            node = self.graph.nodes[sid]
            header = f"# FILE: {node.file_path} | SYMBOL: {node.qualname} (L{node.start_line}-L{node.end_line})"
            raw_block = f"{header}\n{node.source}"
            separator_overhead = 2 if budgeted_ids else 0
            remaining = token_budget - used_tokens - separator_overhead
            block_toks = count_tokens(raw_block)
            if block_toks <= remaining:
                budgeted_ids.append(sid)
                blocks_by_id[sid] = raw_block
                used_tokens += block_toks + separator_overhead
            elif not budgeted_ids:
                fitted = self._fit_to_budget(raw_block, token_budget)
                if fitted is not None:
                    budgeted_ids.append(sid)
                    blocks_by_id[sid] = fitted
                    used_tokens = count_tokens(fitted)
                break

        ordered_ids = position_aware_order(budgeted_ids)
        context_text = "\n\n".join(blocks_by_id[sid] for sid in ordered_ids)
        actual_tokens = count_tokens(context_text)
        # Guard against multi-chunk BPE boundary drift
        while actual_tokens > token_budget and len(ordered_ids) > 1:
            drop_id = budgeted_ids.pop()
            ordered_ids = [sid for sid in ordered_ids if sid != drop_id]
            context_text = "\n\n".join(blocks_by_id[sid] for sid in ordered_ids)
            actual_tokens = count_tokens(context_text)
        if actual_tokens > token_budget and len(ordered_ids) == 1:
            fitted = self._fit_to_budget(blocks_by_id[ordered_ids[0]], token_budget)
            context_text = fitted or ""
            if not context_text:
                ordered_ids = []
                budgeted_ids = []
            actual_tokens = count_tokens(context_text)

        retrieved_files = sorted({self.graph.nodes[sid].file_path for sid in ordered_ids})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SliceResult(
            query=query,
            context_text=context_text,
            ordered_node_ids=ordered_ids,
            ranked_node_ids=budgeted_ids,
            seed_node_ids=[sid for sid in seeds if sid in blocks_by_id],
            retrieved_files=retrieved_files,
            context_tokens=actual_tokens,
            token_budget=token_budget,
            hops_used=hops_used,
            latency_ms=latency_ms,
            raw_repo_tokens=self.graph.total_raw_tokens,
        )
