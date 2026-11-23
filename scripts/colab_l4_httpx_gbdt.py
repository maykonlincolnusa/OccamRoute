"""Remote Google Colab NVIDIA L4 GPU benchmark: Re-run unseen encode/httpx@0.28.1 with v2 GBDT default router."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _pip_install() -> None:
    try:
        import bitsandbytes  # noqa: F401
        import datasets  # noqa: F401
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        import tiktoken  # noqa: F401
        return
    except ImportError:
        pass
    pkgs = [
        "datasets",
        "tiktoken",
        "faiss-cpu",
        "sentence-transformers",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "networkx",
        "scikit-learn",
        "joblib",
        "pandas",
        "matplotlib",
        "seaborn",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for colab_l4_httpx_gbdt.py")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"=== Colab GPU Active: {gpu_name} | Total VRAM: {total_vram_gb:.2f} GB ===", flush=True)

    _pip_install()

    out_dir = Path("/content/colab_out_gbdt")
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path("/content/OccamRoute")
    httpx_dir = Path("/content/httpx")

    if not repo_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/maykonlincolnusa/OccamRoute.git",
                str(repo_dir),
            ],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", str(repo_dir), "pull", "-q", "--ff-only"], check=False)

    if not (httpx_dir / "httpx").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "0.28.1",
                "https://github.com/encode/httpx.git",
                str(httpx_dir),
            ],
            check=True,
        )

    sys.path.insert(0, str((repo_dir / "src").resolve()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from occamroute import (
        OccamRouter,
        RepoGraphBuilder,
        SubgraphFeatureExtractor,
        SurgicalContextSlicer,
        count_tokens,
    )

    metrics = json.loads(Path("/content/router_metrics_v2.json").read_text(encoding="utf-8"))
    sk_sweep = pd.DataFrame(metrics["sklearn_sweep"])
    candidates = sk_sweep[
        (sk_sweep["threshold"] >= 0.30)
        & (sk_sweep["threshold"] <= 0.70)
        & (sk_sweep["strong_call_pct"] >= 20.0)
        & (sk_sweep["strong_call_pct"] <= 70.0)
    ]
    best_row = (candidates if not candidates.empty else sk_sweep).sort_values("pgr_pct", ascending=False).iloc[0]
    tau_sk = float(best_row["threshold"])
    print(f"Selected GBDT optimal threshold tau_sk={tau_sk:.2f}", flush=True)

    try:
        sk_engine = OccamRouter.from_artifact("/content/sklearn_router.joblib", threshold=tau_sk)
    except Exception as exc:
        print(f"ERROR loading /content/sklearn_router.joblib: {exc}", flush=True)
        raise

    httpx_graph = RepoGraphBuilder().build_from_directory(httpx_dir / "httpx")
    slicer = SurgicalContextSlicer(httpx_graph)
    extractor = SubgraphFeatureExtractor(httpx_graph)
    print("RepoGraph (`httpx/`):", httpx_graph.summary_stats(), flush=True)

    @dataclass(frozen=True)
    class EvalTask:
        task_id: str
        expected_complexity: Literal["low", "high"]
        prompt: str
        expected_files: list[str]
        expected_symbols: list[str]
        target_module_files: list[str]

    EVAL_TASKS: list[EvalTask] = [
        EvalTask("T01", "low", "Explique o contrato de `codes.is_redirect` e `codes.get_reason_phrase`.", ["_status_codes.py"], ["_status_codes.py::codes.is_redirect", "_status_codes.py::codes.get_reason_phrase"], ["_status_codes.py", "_models.py"]),
        EvalTask("T02", "low", "Como `BasicAuth._build_auth_header` e `BasicAuth.auth_flow` funcionam?", ["_auth.py"], ["_auth.py::BasicAuth._build_auth_header", "_auth.py::BasicAuth.auth_flow"], ["_auth.py"]),
        EvalTask("T03", "low", "Quais validações `Timeout.__init__` aplica e o que `Timeout.as_dict` retorna?", ["_config.py"], ["_config.py::Timeout.__init__", "_config.py::Timeout.as_dict"], ["_config.py", "_client.py"]),
        EvalTask("T04", "low", "Como `Headers.get_list` e `_normalize_header_key` tratam headers?", ["_models.py"], ["_models.py::Headers.get_list", "_models.py::_normalize_header_key"], ["_models.py"]),
        EvalTask("T05", "low", "Em `Response.raise_for_status`, quando `HTTPStatusError` é levantada e o que `HTTPStatusError.__init__` salva?", ["_models.py", "_exceptions.py"], ["_models.py::Response.raise_for_status", "_exceptions.py::HTTPStatusError.__init__"], ["_models.py", "_exceptions.py"]),
        EvalTask("T06", "low", "Explique `URLPattern.matches` e `URLPattern.priority`.", ["_utils.py"], ["_utils.py::URLPattern.matches", "_utils.py::URLPattern.priority"], ["_utils.py", "_config.py"]),
        EvalTask("T07", "high", "Investigue vazamento em redirects cross-origin: `BaseClient._redirect_headers`, `_same_origin` e `_is_https_redirect`.", ["_client.py"], ["_client.py::BaseClient._redirect_headers", "_client.py::_same_origin", "_client.py::_is_https_redirect"], ["_client.py", "_models.py"]),
        EvalTask("T08", "high", "Rastreie `Client.send` -> `Client._send_handling_auth` -> `Client._send_handling_redirects` -> `Client._send_single_request` -> `HTTPTransport.handle_request`.", ["_client.py", "_transports/default.py"], ["_client.py::Client.send", "_client.py::Client._send_handling_auth", "_client.py::Client._send_handling_redirects", "_client.py::Client._send_single_request", "_transports/default.py::HTTPTransport.handle_request"], ["_client.py", "_transports/default.py"]),
        EvalTask("T09", "high", "Analise o handshake `DigestAuth.auth_flow`, `DigestAuth._parse_challenge`, `DigestAuth._resolve_qop` e `DigestAuth._build_auth_header`.", ["_auth.py"], ["_auth.py::DigestAuth.auth_flow", "_auth.py::DigestAuth._parse_challenge", "_auth.py::DigestAuth._build_auth_header", "_auth.py::DigestAuth._resolve_qop"], ["_auth.py", "_client.py"]),
        EvalTask("T10", "high", "Proponha refatoração de `Client._send_handling_redirects` e `AsyncClient._send_handling_redirects` preservando `BaseClient._build_redirect_request` e `TooManyRedirects`.", ["_client.py", "_exceptions.py"], ["_client.py::Client._send_handling_redirects", "_client.py::AsyncClient._send_handling_redirects", "_client.py::BaseClient._build_redirect_request", "_exceptions.py::TooManyRedirects"], ["_client.py", "_exceptions.py"]),
        EvalTask("T11", "high", "Rastreie descompressão streaming: `Response._get_content_decoder`, `Response.iter_bytes`, `MultiDecoder.decode` e `GZipDecoder.decode`.", ["_models.py", "_decoders.py"], ["_models.py::Response._get_content_decoder", "_models.py::Response.iter_bytes", "_decoders.py::MultiDecoder.decode", "_decoders.py::GZipDecoder.decode"], ["_models.py", "_decoders.py"]),
        EvalTask("T12", "high", "Rastreie mapeamento `httpcore` -> `httpx`: `AsyncHTTPTransport.handle_async_request`, `map_httpcore_exceptions`, `_load_httpcore_exceptions` e `request_context`.", ["_transports/default.py", "_exceptions.py"], ["_transports/default.py::map_httpcore_exceptions", "_transports/default.py::_load_httpcore_exceptions", "_transports/default.py::AsyncHTTPTransport.handle_async_request", "_exceptions.py::request_context"], ["_transports/default.py", "_exceptions.py", "_client.py"]),
    ]

    for et in EVAL_TASKS:
        for sym in et.expected_symbols:
            assert sym in httpx_graph.nodes, f"Missing symbol: {sym}"

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.cuda.reset_peak_memory_stats(0)
    local_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_id)
    local_llm = AutoModelForCausalLM.from_pretrained(local_model_id, quantization_config=bnb, device_map="auto")
    local_llm.eval()

    def run_local_4bit(system_prompt: str, prompt: str, context: str, max_new_tokens: int = 96) -> tuple[str, int, int, float]:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{prompt}"}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=3584).to(local_llm.device)
        in_tok = int(inp["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = local_llm.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        ans = tokenizer.decode(out[0][in_tok:], skip_special_tokens=True).strip()
        return ans, in_tok, int(out.shape[1] - in_tok), ms

    HTTPX_SYSTEM = "You are a Staff Python Engineer analyzing `encode/httpx`. Cite exact file paths and symbol names from the context."

    def middle_burial(ctx: str, expected_syms: list[str]) -> float:
        total = max(1, len(ctx))
        found, buried = 0, 0
        for sid in expected_syms:
            nd = httpx_graph.nodes[sid]
            pos = ctx.find(nd.source.splitlines()[0].strip())
            if pos < 0:
                pos = ctx.find(f"def {nd.name}(")
            if pos >= 0:
                found += 1
                if 0.30 <= (pos / total) <= 0.70:
                    buried += 1
        return (buried / found) if found > 0 else 0.0

    def score_bundle(ctx: str, syms: list[str], files: list[str], ctx_tok: int, task: EvalTask, tier: str, live_ans: str | None) -> tuple[float, float, float, float, float, float]:
        ret_s, exp_s = set(syms), set(task.expected_symbols)
        tp = len(ret_s & exp_s)
        prec = tp / len(ret_s) if ret_s else 0.0
        fn_rec = tp / len(exp_s) if exp_s else 0.0
        f1 = (2 * prec * fn_rec / (prec + fn_rec)) if (prec + fn_rec) > 0 else 0.0
        f_rec = len(set(files) & set(task.expected_files)) / len(set(task.expected_files)) if task.expected_files else 0.0
        bur = middle_burial(ctx, task.expected_symbols)

        loc_sig = 0.60 * fn_rec + 0.25 * f_rec + 0.15 * min(1.0, prec * 1.25)
        stress = min(1.0, max(0.0, (ctx_tok - 1200) / 8000.0))
        ret_factor = 1.0 - 0.28 * bur * stress
        if live_ans is not None:
            ans_low = live_ans.lower()
            cited = sum(1 for s in task.expected_symbols if httpx_graph.nodes[s].name.lower() in ans_low)
            q = 0.50 * (cited / max(1, len(task.expected_symbols))) + 0.35 * loc_sig + 0.15 * ret_factor
        else:
            tf = 1.00 if tier == "strong_frontier" else (0.94 if task.expected_complexity == "low" else 0.60)
            q = loc_sig * ret_factor * tf
        return f_rec, fn_rec, prec, f1, bur, round(q, 4)

    def cost_1k(tier: str, in_t: int, out_t: int, ms: float) -> float:
        if tier == "strong_frontier":
            return 1000.0 * ((in_t / 1_000_000) * 1.25 + (out_t / 1_000_000) * 5.00)
        return 1000.0 * ((ms / 1000.0) / 3600.0) * 0.35

    new_rows = []
    weak_qs, strong_qs = [], []
    for task in EVAL_TASKS:
        sl, dec = sk_engine.route_query(task.prompt, slicer, extractor, token_budget=2200, max_symbols=6, k_hop=2)
        ctx = sl.context_text
        syms = sl.ordered_node_ids
        files = sl.retrieved_files
        ctx_tok = sl.context_tokens
        prep_ms = sl.latency_ms + dec.routing_latency_ms

        live_ans, in_w, out_w, ms_w = run_local_4bit(HTTPX_SYSTEM, task.prompt, ctx, max_new_tokens=160)
        _, _, _, _, _, qw = score_bundle(ctx, syms, files, ctx_tok, task, "weak_local", live_ans)

        in_s = count_tokens(HTTPX_SYSTEM) + count_tokens(task.prompt) + ctx_tok
        out_s = 175
        ms_s = 450.0 + (in_s / 1800.0 + out_s / 85.0) * 1000.0
        _, _, _, _, _, qs = score_bundle(ctx, syms, files, ctx_tok, task, "strong_frontier", None)
        weak_qs.append(qw)
        strong_qs.append(qs)

        if dec.is_strong:
            fr, fnr, pr, f1, bur, qo = score_bundle(ctx, syms, files, ctx_tok, task, "strong_frontier", None)
            in_o, out_o, ms_o, mode_o = in_s, out_s, ms_s, "frontier_calibrated"
        else:
            fr, fnr, pr, f1, bur, qo = score_bundle(ctx, syms, files, ctx_tok, task, "weak_local", live_ans)
            in_o, out_o, ms_o, mode_o = in_w, out_w, ms_w, "live_l4_4bit"

        new_rows.append(
            {
                "pipeline": "OccamRoute (ML)",
                "task_id": task.task_id,
                "expected_complexity": task.expected_complexity,
                "selected_tier": dec.selected_tier,
                "p_strong": dec.p_strong,
                "context_tokens": ctx_tok,
                "file_recall": fr,
                "function_recall": fnr,
                "symbol_precision": pr,
                "symbol_f1": f1,
                "middle_burial_pct": bur * 100.0,
                "quality_score": qo,
                "prep_ms": prep_ms,
                "inference_ms": ms_o,
                "total_ms": prep_ms + ms_o,
                "cost_per_1k_usd": cost_1k(dec.selected_tier, in_o, out_o, ms_o),
                "mode": mode_o,
            }
        )
        print(f"GBDT httpx {task.task_id} ({task.expected_complexity:<4}) | tier={dec.selected_tier:<15} | P={dec.p_strong:.3f} | Q={qo:.3f}", flush=True)

    new_peak = torch.cuda.max_memory_allocated(0) / (1024**3)
    df_old = pd.read_csv("/content/swebench_eval_results_v2.csv")
    old_oc = df_old[df_old["pipeline"] == "OccamRoute (ML)"]
    httpx_torch_run = {
        "local_offload_pct": round(float((old_oc["selected_tier"] == "weak_local").mean() * 100.0), 2),
        "cost_per_1k_usd": round(float(old_oc["cost_per_1k_usd"].mean()), 4),
        "quality_score": round(float(old_oc["quality_score"].mean()), 4),
    }

    df_eval = pd.concat([df_old[df_old["pipeline"] != "OccamRoute (ML)"], pd.DataFrame(new_rows)], ignore_index=True)
    df_eval = df_eval[df_old.columns]
    df_eval.to_csv(out_dir / "swebench_eval_results.csv", index=False)

    Q_WEAK = sum(weak_qs) / len(weak_qs)
    Q_STRONG = sum(strong_qs) / len(strong_qs)

    order = ["Brute Force", "Naive RAG", "Unbounded AST 2-Hop", "OccamRoute (ML)"]
    summary = (
        df_eval.groupby("pipeline", as_index=False)
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
    summary["pipeline"] = pd.Categorical(summary["pipeline"], categories=order, ordered=True)
    summary = summary.sort_values("pipeline").reset_index(drop=True)
    bf_tok = float(summary.loc[summary["pipeline"] == "Brute Force", "mean_tokens"].iloc[0])
    summary["token_savings_pct"] = (1.0 - summary["mean_tokens"] / bf_tok) * 100.0
    summary["pgr_pct"] = ((summary["quality_score"] - Q_WEAK) / max(1e-6, Q_STRONG - Q_WEAK)) * 100.0

    print(f"\n=== COLAB {gpu_name} v2 GBDT SUMMARY (Peak VRAM: {new_peak:.2f} GB) ===", flush=True)
    print(summary.round(3).to_string(index=False), flush=True)

    httpx_offload_pct = float(summary.loc[summary["pipeline"] == "OccamRoute (ML)", "local_offload_pct"].iloc[0])
    metrics["optimal_tau_torch"] = metrics["optimal_tau"]
    metrics["optimal_tau"] = tau_sk
    metrics["default_router"] = "sklearn"
    metrics["httpx_router_backend"] = "sklearn"
    metrics["httpx_local_offload_pct"] = round(httpx_offload_pct, 2)
    metrics["q_weak"] = round(Q_WEAK, 4)
    metrics["q_strong"] = round(Q_STRONG, 4)
    metrics["peak_vram_gb"] = round(max(float(metrics["peak_vram_gb"]), new_peak), 3)
    metrics["httpx_torch_run"] = httpx_torch_run
    (out_dir / "router_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Charts 3 & 4
    BG_FIG, BG_AX, TEXT_MAIN, TEXT_MUTED, GRID_COL = "#0D1117", "#161B22", "#E6EDF3", "#8B949E", "#30363D"
    PALETTE = {"Brute Force": "#F85149", "Naive RAG": "#D29922", "Unbounded AST 2-Hop": "#A371F7", "OccamRoute (ML)": "#2EA043"}
    plt.rcParams.update(
        {
            "figure.facecolor": BG_FIG,
            "axes.facecolor": BG_AX,
            "axes.edgecolor": GRID_COL,
            "axes.labelcolor": TEXT_MAIN,
            "xtick.color": TEXT_MAIN,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT_MAIN,
        }
    )

    # Chart 3: Pareto Frontier (Zero-Shot encode/httpx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=250)
    for _, r in summary.iterrows():
        p = str(r["pipeline"])
        ax1.scatter(
            r["mean_tokens"],
            r["quality_score"],
            color=PALETTE[p],
            s=220 if "OccamRoute" in p else 140,
            marker="*" if "OccamRoute" in p else "o",
            edgecolors="#FFF" if "OccamRoute" in p else "#0D1117",
        )
        ax1.annotate(
            f"{p}\n{r['mean_tokens']:,.0f}t | PGR={r['pgr_pct']:.0f}%",
            (r["mean_tokens"], r["quality_score"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8.5,
            color="#7EE787" if "OccamRoute" in p else TEXT_MAIN,
        )
    ax1.axhline(Q_STRONG, color="#58A6FF", ls="--", alpha=0.6)
    ax1.axhline(Q_WEAK, color=TEXT_MUTED, ls="--", alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_title(f"Zero-Shot Pareto (Colab {gpu_name}: Live FAISS + 4-bit)")
    x = np.arange(len(order))
    w = 0.30
    ax2.bar(x - w / 2, summary["function_recall"], width=w, color="#58A6FF", label="Function Recall")
    ax2.bar(x + w / 2, summary["symbol_f1"], width=w, color="#2EA043", label="Symbol F1")
    for idx, (_, r) in enumerate(summary.iterrows()):
        ax2.text(
            idx,
            1.07,
            f"${r['cost_per_1k_usd']:.2f}/1k\n({r['token_savings_pct']:+.0f}% tok)",
            ha="center",
            fontsize=8.3,
            color="#7EE787" if "OccamRoute" in str(r["pipeline"]) else TEXT_MUTED,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, fontsize=9, rotation=10)
    ax2.set_ylim(0, 1.24)
    ax2.set_title("Localization & Cost/1k (Colab L4 v2)")
    ax2.legend(facecolor=BG_FIG, edgecolor=GRID_COL, loc="upper left")
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID_COL, ls="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "swebench_pareto_frontier.png", dpi=250, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)

    # Chart 4: Task Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=250)
    piv = df_eval.pivot(index="task_id", columns="pipeline", values="context_tokens")[order]
    for p in order:
        ax1.plot(piv.index, piv[p], marker="o", lw=2.2 if "OccamRoute" in p else 1.5, label=p, color=PALETTE[p])
    ax1.set_yscale("log")
    ax1.set_title("Per-Task Token Footprint (Log)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(facecolor=BG_FIG, edgecolor=GRID_COL, fontsize=8.3)
    oc_rows = df_eval[df_eval["pipeline"] == "OccamRoute (ML)"]
    ax2.bar(oc_rows["task_id"], oc_rows["p_strong"], color=["#58A6FF" if c == "low" else "#F78166" for c in oc_rows["expected_complexity"]])
    ax2.axhline(tau_sk, color="#2EA043", ls="--", lw=1.6, label=f"tau={tau_sk:.2f}")
    ax2.set_title("v2 GBDT Router P(strong) on Unseen httpx")
    ax2.set_ylim(0, 1.08)
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(facecolor=BG_FIG, edgecolor=GRID_COL)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID_COL, ls="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "swebench_task_breakdown.png", dpi=250, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)

    print("ALL GBDT HTTPX OUTPUTS WRITTEN", flush=True)


if __name__ == "__main__":
    main()
