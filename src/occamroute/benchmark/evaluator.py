"""End-to-end 3-pipeline benchmark evaluator for Brute Force, Naive RAG, and OccamRoute (ML)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from occamroute.benchmark.baselines import BruteForceBaseline, NaiveRAGBaseline, RagBackend
from occamroute.context_slicer import SurgicalContextSlicer
from occamroute.dataset_loader import SWEBenchTask
from occamroute.feature_extractor import SubgraphFeatureExtractor
from occamroute.graph_builder import RepoGraph, count_tokens
from occamroute.router.engine import OccamRouter

LocalLLMCallable = Callable[[str, str], tuple[str, int, int, float]]


@dataclass(frozen=True)
class BenchmarkTask:
    """Normalized benchmark evaluation task spanning a target repository."""

    task_id: str
    query: str
    expected_files: list[str]
    expected_symbols: list[str]
    expected_complexity: Literal["low", "high"]
    target_module_files: list[str] | None = None
    repo: str = "local"


@dataclass(frozen=True)
class TaskEvalRecord:
    """Single task evaluation measurement for one retrieval/routing pipeline."""

    pipeline: str
    task_id: str
    repo: str
    expected_complexity: Literal["low", "high"]
    selected_tier: str
    p_strong: float
    context_tokens: int
    file_recall: float
    function_recall: float
    symbol_precision: float
    symbol_f1: float
    middle_burial_pct: float
    quality_score: float
    prep_ms: float
    inference_ms: float
    total_ms: float
    cost_per_1k_usd: float
    mode: str


@dataclass
class BenchmarkReport:
    """Aggregated benchmark records, per-pipeline summary, and RouteLLM quality bounds."""

    records_df: pd.DataFrame
    summary_df: pd.DataFrame
    q_weak: float
    q_strong: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Path | str) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "benchmark_results.csv"
        json_path = out / "benchmark_summary.json"
        self.records_df.to_csv(csv_path, index=False)
        payload = {
            "q_weak": round(self.q_weak, 4),
            "q_strong": round(self.q_strong, 4),
            "metadata": self.metadata,
            "summary": self.summary_df.to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"results_csv": csv_path, "summary_json": json_path}


def tasks_from_swebench_lite(
    repo_graph: RepoGraph,
    swe_tasks: list[SWEBenchTask],
    max_tasks: int = 20,
) -> list[BenchmarkTask]:
    """Map SWE-bench Lite patch hunks onto leaf AST symbols inside a checked-out RepoGraph."""
    rel_lookup = {k.replace("\\", "/"): k for k in repo_graph.file_to_nodes}
    tasks: list[BenchmarkTask] = []

    for row in swe_tasks:
        gt_files: list[str] = []
        gt_symbols: list[str] = []
        for patch_file, ranges in row.ground_truth_hunks.items():
            norm_p = patch_file.replace("\\", "/")
            if not norm_p.endswith(".py"):
                continue
            matched_key = rel_lookup.get(norm_p)
            if matched_key is None:
                for k in rel_lookup:
                    if k.endswith("/" + norm_p) or norm_p.endswith("/" + k):
                        matched_key = rel_lookup[k]
                        break
            if matched_key is None:
                continue

            leaf_ids = [nid for nid in repo_graph.file_to_nodes.get(matched_key, []) if nid in repo_graph.leaf_ids]
            if not leaf_ids:
                continue
            gt_files.append(matched_key)
            eff_ranges = ranges if ranges else [(1, 10**9)]
            file_hits: list[str] = []
            for nid in leaf_ids:
                nd = repo_graph.nodes[nid]
                if any(nd.end_line >= s_ln - 3 and nd.start_line <= e_ln + 3 for s_ln, e_ln in eff_ranges):
                    file_hits.append(nid)
            if not file_hits:
                file_hits.append(leaf_ids[0])
            for nid in file_hits:
                if nid not in gt_symbols:
                    gt_symbols.append(nid)

        if not gt_symbols:
            continue

        unique_files = sorted(set(gt_files))
        complexity: Literal["low", "high"] = (
            "low" if (len(unique_files) == 1 and len(gt_symbols) <= 2) else "high"
        )
        tasks.append(
            BenchmarkTask(
                task_id=row.instance_id,
                query=row.problem_statement[:1800],
                expected_files=unique_files,
                expected_symbols=gt_symbols,
                expected_complexity=complexity,
                target_module_files=unique_files,
                repo=row.repo,
            )
        )
        if len(tasks) >= max_tasks:
            break
    return tasks


def build_repo_eval_tasks(repo_graph: RepoGraph, max_tasks: int = 8) -> list[BenchmarkTask]:
    """Deterministically build balanced low/high evaluation tasks from any RepoGraph."""
    all_files = sorted(repo_graph.file_to_nodes.keys())
    tasks: list[BenchmarkTask] = []
    half = max(1, max_tasks // 2)

    # Low-complexity single-symbol lookups
    for rel in all_files:
        if len(tasks) >= half:
            break
        leaves = [nid for nid in repo_graph.file_to_nodes[rel] if nid in repo_graph.leaf_ids]
        if not leaves:
            continue
        nd = sorted(leaves, key=lambda s: repo_graph.nodes[s].cyclomatic_complexity)[0]
        sym = repo_graph.nodes[nd]
        tasks.append(
            BenchmarkTask(
                task_id=f"T{len(tasks)+1:02d}",
                query=f"Explain the behavior and contract of `{sym.qualname}` in `{sym.file_path}`.",
                expected_files=[sym.file_path],
                expected_symbols=[nd],
                expected_complexity="low",
                target_module_files=all_files[: min(3, len(all_files))],
            )
        )

    # High-complexity multi-symbol call-chain tasks
    ranked_hubs = sorted(
        repo_graph.leaf_ids,
        key=lambda nid: (repo_graph.nodes[nid].cyclomatic_complexity, repo_graph.pagerank.get(nid, 0.0)),
        reverse=True,
    )
    for hub in ranked_hubs:
        if len(tasks) >= max_tasks:
            break
        dist = repo_graph.k_hop_neighbors([hub], k=2, undirected=True)
        chain = [s for s in dist if s in repo_graph.leaf_ids][:4]
        if len(chain) < 2:
            continue
        exp_files = sorted({repo_graph.nodes[s].file_path for s in chain})
        qnames = " -> ".join(f"`{repo_graph.nodes[s].qualname}`" for s in chain)
        tasks.append(
            BenchmarkTask(
                task_id=f"T{len(tasks)+1:02d}",
                query=f"Trace and refactor {qnames} across {', '.join(exp_files)} while preserving contracts.",
                expected_files=exp_files,
                expected_symbols=chain,
                expected_complexity="high",
                target_module_files=sorted(set(exp_files + all_files[: min(3, len(all_files))])),
            )
        )
    return tasks


class SWEBenchBenchmarkEvaluator:
    """Compare Brute Force, Naive RAG, and OccamRoute (ML) on identical engineering tasks."""

    SYSTEM_PROMPT = "Analyze the codebase context and cite exact file paths and symbol names."

    def __init__(
        self,
        repo_graph: RepoGraph,
        router: OccamRouter,
        rag_backend: RagBackend = "auto",
        token_budget: int = 2000,
        max_symbols: int = 6,
        k_hop: int = 2,
        top_k_rag: int = 5,
        local_llm_fn: LocalLLMCallable | None = None,
    ) -> None:
        self.repo_graph = repo_graph
        self.router = router
        self.token_budget = token_budget
        self.max_symbols = max_symbols
        self.k_hop = k_hop
        self.top_k_rag = top_k_rag
        self.local_llm_fn = local_llm_fn

        self.brute_force = BruteForceBaseline(repo_graph)
        self.naive_rag = NaiveRAGBaseline(repo_graph, chunk_tokens=512, overlap_tokens=64, backend=rag_backend)
        self.slicer = SurgicalContextSlicer(repo_graph)
        self.extractor = SubgraphFeatureExtractor(repo_graph)

    def _middle_burial_rate(self, context_text: str, expected_symbols: list[str]) -> float:
        total = max(1, len(context_text))
        found, buried = 0, 0
        for sid in expected_symbols:
            nd = self.repo_graph.nodes.get(sid)
            if nd is None:
                continue
            first_line = nd.source.splitlines()[0].strip() if nd.source else nd.name
            pos = context_text.find(first_line)
            if pos < 0:
                pos = context_text.find(f"def {nd.name}(")
            if pos >= 0:
                found += 1
                if 0.30 <= (pos / total) <= 0.70:
                    buried += 1
        return (buried / found) if found > 0 else 0.0

    def _score_bundle(
        self,
        context_text: str,
        retrieved_symbols: list[str],
        retrieved_files: list[str],
        context_tokens: int,
        task: BenchmarkTask,
        tier: str,
        live_answer: str | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        ret_s = set(retrieved_symbols)
        exp_s = set(task.expected_symbols)
        tp = len(ret_s & exp_s)
        prec = tp / len(ret_s) if ret_s else 0.0
        fn_rec = tp / len(exp_s) if exp_s else 0.0
        f1 = (2 * prec * fn_rec / (prec + fn_rec)) if (prec + fn_rec) > 0 else 0.0
        exp_f = set(task.expected_files)
        f_rec = len(set(retrieved_files) & exp_f) / len(exp_f) if exp_f else 0.0
        burial = self._middle_burial_rate(context_text, task.expected_symbols)

        loc_sig = 0.60 * fn_rec + 0.25 * f_rec + 0.15 * min(1.0, prec * 1.25)
        stress = min(1.0, max(0.0, (context_tokens - 1200) / 8000.0))
        ret_factor = 1.0 - 0.28 * burial * stress

        if live_answer is not None:
            ans_low = live_answer.lower()
            cited = sum(
                1
                for sid in task.expected_symbols
                if sid in self.repo_graph.nodes and self.repo_graph.nodes[sid].name.lower() in ans_low
            )
            quality = 0.50 * (cited / max(1, len(task.expected_symbols))) + 0.35 * loc_sig + 0.15 * ret_factor
        else:
            tier_factor = 1.00 if tier == "strong_frontier" else (0.94 if task.expected_complexity == "low" else 0.60)
            quality = loc_sig * ret_factor * tier_factor
        return f_rec, fn_rec, prec, f1, burial, round(float(quality), 4)

    @staticmethod
    def _estimate_cost_per_1k(tier: str, in_tokens: int, out_tokens: int, inf_ms: float) -> float:
        if tier == "strong_frontier":
            return 1000.0 * ((in_tokens / 1_000_000.0) * 1.25 + (out_tokens / 1_000_000.0) * 5.00)
        return 1000.0 * ((inf_ms / 1000.0) / 3600.0) * 0.35

    def evaluate(self, tasks: list[BenchmarkTask]) -> BenchmarkReport:
        if not tasks:
            raise ValueError("Cannot evaluate an empty task list.")

        records: list[dict[str, Any]] = []
        weak_qualities: list[float] = []
        strong_qualities: list[float] = []
        base_prompt_tok = count_tokens(self.SYSTEM_PROMPT)

        for task in tasks:
            q_tok = count_tokens(task.query)

            # 1. Baseline A: Brute Force & Baseline B: Naive RAG
            bf_res = self.brute_force.retrieve(task.query, target_files=task.target_module_files)
            rag_res = self.naive_rag.retrieve(task.query, top_k=self.top_k_rag)
            for res in (bf_res, rag_res):
                in_t = base_prompt_tok + q_tok + res.context_tokens
                out_t = 175
                inf_ms = 450.0 + (in_t / 1800.0 + out_t / 85.0) * 1000.0
                fr, fnr, pr, f1, bur, q_val = self._score_bundle(
                    res.context_text,
                    res.retrieved_symbols,
                    res.retrieved_files,
                    res.context_tokens,
                    task,
                    "strong_frontier",
                )
                records.append(
                    TaskEvalRecord(
                        pipeline=res.pipeline,
                        task_id=task.task_id,
                        repo=task.repo,
                        expected_complexity=task.expected_complexity,
                        selected_tier="strong_frontier",
                        p_strong=1.0,
                        context_tokens=res.context_tokens,
                        file_recall=fr,
                        function_recall=fnr,
                        symbol_precision=pr,
                        symbol_f1=f1,
                        middle_burial_pct=bur * 100.0,
                        quality_score=q_val,
                        prep_ms=res.prep_latency_ms,
                        inference_ms=inf_ms,
                        total_ms=res.prep_latency_ms + inf_ms,
                        cost_per_1k_usd=self._estimate_cost_per_1k("strong_frontier", in_t, out_t, inf_ms),
                        mode=f"frontier_{res.retrieval_backend}",
                    ).__dict__
                )

            # 2. OccamRoute (ML)
            sl, dec = self.router.route_query(
                task.query,
                self.slicer,
                self.extractor,
                token_budget=self.token_budget,
                max_symbols=self.max_symbols,
                k_hop=self.k_hop,
            )
            prep_ms = sl.latency_ms + dec.routing_latency_ms
            in_s = base_prompt_tok + q_tok + sl.context_tokens
            out_s = 175
            inf_s = 450.0 + (in_s / 1800.0 + out_s / 85.0) * 1000.0

            live_ans: str | None = None
            if self.local_llm_fn is not None:
                live_ans, in_w, _, inf_w = self.local_llm_fn(task.query, sl.context_text)
            else:
                in_w = in_s
                inf_w = 220.0 + (in_w / 1200.0 + 120.0 / 95.0) * 1000.0

            _, _, _, _, _, qw = self._score_bundle(
                sl.context_text, sl.ordered_node_ids, sl.retrieved_files, sl.context_tokens, task, "weak_local", live_ans
            )
            _, _, _, _, _, qs = self._score_bundle(
                sl.context_text, sl.ordered_node_ids, sl.retrieved_files, sl.context_tokens, task, "strong_frontier", None
            )
            weak_qualities.append(qw)
            strong_qualities.append(qs)

            if dec.is_strong:
                fr, fnr, pr, f1, bur, qo = self._score_bundle(
                    sl.context_text, sl.ordered_node_ids, sl.retrieved_files, sl.context_tokens, task, "strong_frontier", None
                )
                in_o, out_o, inf_o, mode_o = in_s, out_s, inf_s, "frontier_calibrated"
            else:
                fr, fnr, pr, f1, bur, qo = self._score_bundle(
                    sl.context_text, sl.ordered_node_ids, sl.retrieved_files, sl.context_tokens, task, "weak_local", live_ans
                )
                in_o, out_o, inf_o = in_w, 120, inf_w
                mode_o = "live_gpu_4bit" if self.local_llm_fn is not None else "local_calibrated"

            records.append(
                TaskEvalRecord(
                    pipeline="OccamRoute (ML)",
                    task_id=task.task_id,
                    repo=task.repo,
                    expected_complexity=task.expected_complexity,
                    selected_tier=dec.selected_tier,
                    p_strong=dec.p_strong,
                    context_tokens=sl.context_tokens,
                    file_recall=fr,
                    function_recall=fnr,
                    symbol_precision=pr,
                    symbol_f1=f1,
                    middle_burial_pct=bur * 100.0,
                    quality_score=qo,
                    prep_ms=prep_ms,
                    inference_ms=inf_o,
                    total_ms=prep_ms + inf_o,
                    cost_per_1k_usd=self._estimate_cost_per_1k(dec.selected_tier, in_o, out_o, inf_o),
                    mode=mode_o,
                ).__dict__
            )

        records_df = pd.DataFrame(records)
        q_weak = sum(weak_qualities) / max(1, len(weak_qualities))
        q_strong = sum(strong_qualities) / max(1, len(strong_qualities))

        order = ["Brute Force", "Naive RAG", "OccamRoute (ML)"]
        summary_df = (
            records_df.groupby("pipeline", as_index=False)
            .agg(
                file_recall=("file_recall", "mean"),
                function_recall=("function_recall", "mean"),
                symbol_precision=("symbol_precision", "mean"),
                symbol_f1=("symbol_f1", "mean"),
                middle_burial_pct=("middle_burial_pct", "mean"),
                mean_tokens=("context_tokens", "mean"),
                mean_latency_ms=("total_ms", "mean"),
                cost_per_1k_usd=("cost_per_1k_usd", "mean"),
                quality_score=("quality_score", "mean"),
                local_offload_pct=("selected_tier", lambda s: (s == "weak_local").mean() * 100.0),
            )
        )
        summary_df["pipeline"] = pd.Categorical(summary_df["pipeline"], categories=order, ordered=True)
        summary_df = summary_df.sort_values("pipeline").reset_index(drop=True)

        bf_tokens = float(summary_df.loc[summary_df["pipeline"] == "Brute Force", "mean_tokens"].iloc[0])
        summary_df["token_savings_pct"] = (1.0 - summary_df["mean_tokens"] / max(1.0, bf_tokens)) * 100.0
        summary_df["pgr_pct"] = ((summary_df["quality_score"] - q_weak) / max(1e-6, q_strong - q_weak)) * 100.0

        metadata = {
            "num_tasks": len(tasks),
            "rag_backend": self.naive_rag.active_backend,
            "router_backend": self.router.backend_name,
            "router_threshold": self.router.threshold,
            "live_4bit_enabled": self.local_llm_fn is not None,
        }
        return BenchmarkReport(
            records_df=records_df,
            summary_df=summary_df,
            q_weak=q_weak,
            q_strong=q_strong,
            metadata=metadata,
        )
