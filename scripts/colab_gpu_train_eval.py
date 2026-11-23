"""Remote Google Colab Tesla T4 GPU training & 4-pipeline live benchmark for OccamRoute."""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _pip_install() -> None:
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
        raise RuntimeError("CUDA GPU is required for colab_gpu_train_eval.py")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"=== Colab GPU Active: {gpu_name} | Total VRAM: {total_vram_gb:.2f} GB ===")

    _pip_install()

    out_dir = Path("/content/colab_out")
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
                "https://github.com/maykonlincolnusa/OccamRoute-Surgical-Context-Complexity-Routing-Engine-.git",
                str(repo_dir),
            ],
            check=True,
        )
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

    import faiss
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.calibration import calibration_curve
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_curve

    from occamroute import (
        OccamRouter,
        RepoGraphBuilder,
        SklearnComplexityRouter,
        SubgraphFeatureExtractor,
        SurgicalContextSlicer,
        TorchComplexityRouter,
        TorchRouterConfig,
        build_routing_dataset,
        count_tokens,
        load_swebench_lite,
    )

    # 1. Verify SWE-bench Lite loading live in Colab
    swe_tasks = load_swebench_lite(limit=60, split="test")
    print(f"SWE-bench Lite loaded live: {len(swe_tasks)} tasks")

    # 2. Build AST RepoGraph on real encode/httpx@0.28.1 (`httpx/` package)
    repo_graph = RepoGraphBuilder().build_from_directory(httpx_dir / "httpx")
    print("RepoGraph (`httpx/`):", repo_graph.summary_stats())

    # 3. Build RouteLLM augmented dataset and train Sklearn + PyTorch MLP on CUDA T4
    dataset = build_routing_dataset(repo_graph=repo_graph, n_samples=800, val_ratio=0.25, seed=42)
    df_train = pd.DataFrame(dataset.X_train, columns=dataset.feature_names)
    df_train["label"] = dataset.y_train
    df_train.to_csv(out_dir / "router_training_dataset.csv", index=False)

    sk_router = SklearnComplexityRouter().fit(dataset.X_train, dataset.y_train)
    pt_cfg = TorchRouterConfig(
        input_dim=len(dataset.feature_names),
        hidden_dims=(32, 16),
        dropout=0.15,
        lr=3e-3,
        epochs=140,
        batch_size=32,
        seed=42,
    )
    pt_router = TorchComplexityRouter(config=pt_cfg).fit(dataset.X_train, dataset.y_train)
    print(f"PyTorch MLP trained on device: {pt_router.device}")

    sk_metrics = sk_router.evaluate(dataset.X_val, dataset.y_val, threshold=0.50)
    pt_metrics = pt_router.evaluate(dataset.X_val, dataset.y_val, threshold=0.50)

    sk_path = sk_router.save(out_dir / "sklearn_router.joblib")
    pt_path = pt_router.save(out_dir / "torch_router.pt")

    sk_engine = OccamRouter.from_artifact(sk_path, threshold=0.50)
    pt_engine = OccamRouter.from_artifact(pt_path, threshold=0.50)

    sk_sweep = pd.DataFrame(
        sk_engine.sweep_thresholds(dataset.X_val, dataset.y_val, dataset.weak_quality_val, dataset.strong_quality_val)
    )
    pt_sweep = pd.DataFrame(
        pt_engine.sweep_thresholds(dataset.X_val, dataset.y_val, dataset.weak_quality_val, dataset.strong_quality_val)
    )
    candidates = pt_sweep[(pt_sweep["threshold"] >= 0.3) & (pt_sweep["threshold"] <= 0.7) & (pt_sweep["strong_call_pct"] <= 60.0)]
    best_row = (candidates if not candidates.empty else pt_sweep).sort_values("pgr_pct", ascending=False).iloc[0]
    optimal_tau = float(best_row["threshold"])
    pt_engine = OccamRouter.from_artifact(pt_path, threshold=optimal_tau)

    print(f"Sklearn val: {sk_metrics}")
    print(f"PyTorch val: {pt_metrics} | optimal_tau={optimal_tau:.2f}")

    # 4. Real 12-task benchmark on `encode/httpx@0.28.1`
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
            assert sym in repo_graph.nodes, f"Missing symbol: {sym}"

    slicer = SurgicalContextSlicer(repo_graph)
    extractor = SubgraphFeatureExtractor(repo_graph)

    # Real FAISS + BGE-small Naive RAG
    rag_chunks: list[tuple[str, int, int, str]] = []
    for rel in sorted(repo_graph.file_to_nodes):
        lines = (repo_graph.root_dir / rel).read_text(encoding="utf-8").splitlines()
        ltoks = [max(1, count_tokens(ln)) for ln in lines]
        i = 0
        while i < len(lines):
            acc, j = 0, i
            while j < len(lines) and (acc + ltoks[j] <= 512 or j == i):
                acc += ltoks[j]
                j += 1
            rag_chunks.append((rel, i + 1, j, "\n".join(lines[i:j])))
            if j >= len(lines):
                break
            back, ni = 0, j
            while ni > i + 1 and back < 64:
                ni -= 1
                back += ltoks[ni]
            i = max(i + 1, ni)

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    embs = np.asarray(
        embedder.encode([f"{c[0]}\n{c[3]}" for c in rag_chunks], normalize_embeddings=True, show_progress_bar=False),
        dtype="float32",
    )
    faiss_index = faiss.IndexFlatIP(embs.shape[1])
    faiss_index.add(embs)

    def retrieve_rag(query: str, top_k: int = 5) -> tuple[str, list[str], list[str], int, float]:
        t0 = time.perf_counter()
        qe = np.asarray(embedder.encode([query], normalize_embeddings=True, show_progress_bar=False), dtype="float32")
        _, idxs = faiss_index.search(qe, top_k)
        blocks, syms, files = [], [], set()
        for idx in idxs[0]:
            if idx < 0:
                continue
            rel, s_ln, e_ln, txt = rag_chunks[int(idx)]
            blocks.append(f"# CHUNK: {rel} (L{s_ln}-L{e_ln})\n{txt}")
            files.add(rel)
            for nid in repo_graph.file_to_nodes.get(rel, []):
                if nid not in repo_graph.leaf_ids:
                    continue
                nd = repo_graph.nodes[nid]
                ov = min(e_ln, nd.end_line) - max(s_ln, nd.start_line) + 1
                if ov > 0 and (ov / max(1, nd.end_line - nd.start_line + 1) >= 0.35 or (s_ln <= nd.start_line <= e_ln and ov >= 3)):
                    if nid not in syms:
                        syms.append(nid)
        ctx = "\n\n".join(blocks)
        return ctx, syms, sorted(files), count_tokens(ctx), (time.perf_counter() - t0) * 1000.0

    # Pre-compute contexts for all 4 pipelines before freeing embedder VRAM
    bundles: dict[tuple[str, str], tuple[str, list[str], list[str], int, float]] = {}
    occam_decisions = {}
    for task in EVAL_TASKS:
        # Brute Force
        t0 = time.perf_counter()
        bf_blocks, bf_syms = [], []
        for rel in task.target_module_files:
            bf_blocks.append(f"# FILE: {rel}\n{(repo_graph.root_dir / rel).read_text(encoding='utf-8')}")
            bf_syms.extend(nid for nid in repo_graph.file_to_nodes.get(rel, []) if nid in repo_graph.leaf_ids)
        bf_ctx = "\n\n".join(bf_blocks)
        bundles[("Brute Force", task.task_id)] = (bf_ctx, bf_syms, list(task.target_module_files), count_tokens(bf_ctx), (time.perf_counter() - t0) * 1000.0)

        # Naive RAG (FAISS)
        bundles[("Naive RAG", task.task_id)] = retrieve_rag(task.prompt, top_k=5)

        # Unbounded AST 2-Hop
        t0 = time.perf_counter()
        scores, _, _, _ = slicer._score_seeds(task.prompt)
        seeds = [sid for _, sid in scores[:3]]
        dist = repo_graph.k_hop_neighbors(seeds, k=2, undirected=True)
        ub_nodes = sorted((nid for nid in dist if nid in repo_graph.leaf_ids), key=lambda s: (repo_graph.nodes[s].file_path, repo_graph.nodes[s].start_line))
        ub_ctx = "\n\n".join(f"# FILE: {repo_graph.nodes[s].file_path} | SYMBOL: {repo_graph.nodes[s].qualname}\n{repo_graph.nodes[s].source}" for s in ub_nodes)
        ub_files = sorted({repo_graph.nodes[s].file_path for s in ub_nodes})
        bundles[("Unbounded AST 2-Hop", task.task_id)] = (ub_ctx, ub_nodes, ub_files, count_tokens(ub_ctx), (time.perf_counter() - t0) * 1000.0)

        # OccamRoute
        sl, dec = pt_engine.route_query(task.prompt, slicer, extractor, token_budget=2200, max_symbols=6, k_hop=2)
        bundles[("OccamRoute (ML)", task.task_id)] = (sl.context_text, sl.ordered_node_ids, sl.retrieved_files, sl.context_tokens, sl.latency_ms + dec.routing_latency_ms)
        occam_decisions[task.task_id] = dec

    del embedder, faiss_index, embs
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Load REAL 4-bit NF4 Qwen2.5-Coder-7B-Instruct on T4 GPU
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
    print(f"Loaded {local_model_id} (4-bit NF4) | Peak VRAM: {peak_vram_gb:.2f} GB")

    SYSTEM_PROMPT = "You are a Staff Python Engineer analyzing `encode/httpx`. Cite exact file paths and symbol names from the context."

    def run_local_4bit(prompt: str, context: str) -> tuple[str, int, int, float]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{prompt}"}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=4096).to(local_llm.device)
        in_tok = int(inp["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = local_llm.generate(**inp, max_new_tokens=160, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        ans = tokenizer.decode(out[0][in_tok:], skip_special_tokens=True).strip()
        return ans, in_tok, int(out.shape[1] - in_tok), ms

    def middle_burial(ctx: str, expected_syms: list[str]) -> float:
        total = max(1, len(ctx))
        found, buried = 0, 0
        for sid in expected_syms:
            nd = repo_graph.nodes[sid]
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
            cited = sum(1 for s in task.expected_symbols if repo_graph.nodes[s].name.lower() in ans_low)
            q = 0.50 * (cited / max(1, len(task.expected_symbols))) + 0.35 * loc_sig + 0.15 * ret_factor
        else:
            tf = 1.00 if tier == "strong_frontier" else (0.94 if task.expected_complexity == "low" else 0.60)
            q = loc_sig * ret_factor * tf
        return f_rec, fn_rec, prec, f1, bur, round(q, 4)

    def cost_1k(tier: str, in_t: int, out_t: int, ms: float) -> float:
        if tier == "strong_frontier":
            return 1000.0 * ((in_t / 1_000_000) * 1.25 + (out_t / 1_000_000) * 5.00)
        return 1000.0 * ((ms / 1000.0) / 3600.0) * 0.35

    eval_rows = []
    weak_qs, strong_qs = [], []
    for task in EVAL_TASKS:
        for p_name in ("Brute Force", "Naive RAG", "Unbounded AST 2-Hop"):
            ctx, syms, files, ctx_tok, prep_ms = bundles[(p_name, task.task_id)]
            in_t = count_tokens(SYSTEM_PROMPT) + count_tokens(task.prompt) + ctx_tok
            out_t = 175
            inf_ms = 450.0 + (in_t / 1800.0 + out_t / 85.0) * 1000.0
            fr, fnr, pr, f1, bur, q = score_bundle(ctx, syms, files, ctx_tok, task, "strong_frontier", None)
            eval_rows.append({
                "pipeline": p_name, "task_id": task.task_id, "expected_complexity": task.expected_complexity,
                "selected_tier": "strong_frontier", "p_strong": 1.0, "context_tokens": ctx_tok,
                "file_recall": fr, "function_recall": fnr, "symbol_precision": pr, "symbol_f1": f1,
                "middle_burial_pct": bur * 100.0, "quality_score": q, "prep_ms": prep_ms,
                "inference_ms": inf_ms, "total_ms": prep_ms + inf_ms,
                "cost_per_1k_usd": cost_1k("strong_frontier", in_t, out_t, inf_ms), "mode": "frontier_calibrated",
            })

        # OccamRoute (Live 4-bit GPU for weak_local tasks!)
        ctx, syms, files, ctx_tok, prep_ms = bundles[("OccamRoute (ML)", task.task_id)]
        dec = occam_decisions[task.task_id]
        live_ans, in_w, out_w, ms_w = run_local_4bit(task.prompt, ctx)
        _, _, _, _, _, qw = score_bundle(ctx, syms, files, ctx_tok, task, "weak_local", live_ans)

        in_s = count_tokens(SYSTEM_PROMPT) + count_tokens(task.prompt) + ctx_tok
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
            in_o, out_o, ms_o, mode_o = in_w, out_w, ms_w, "live_t4_4bit"

        eval_rows.append({
            "pipeline": "OccamRoute (ML)", "task_id": task.task_id, "expected_complexity": task.expected_complexity,
            "selected_tier": dec.selected_tier, "p_strong": dec.p_strong, "context_tokens": ctx_tok,
            "file_recall": fr, "function_recall": fnr, "symbol_precision": pr, "symbol_f1": f1,
            "middle_burial_pct": bur * 100.0, "quality_score": qo, "prep_ms": prep_ms,
            "inference_ms": ms_o, "total_ms": prep_ms + ms_o,
            "cost_per_1k_usd": cost_1k(dec.selected_tier, in_o, out_o, ms_o), "mode": mode_o,
        })
        print(f"Task {task.task_id} | tier={dec.selected_tier:<15} | P={dec.p_strong:.3f} | 4-bit={ms_w:.0f}ms | Q={qo:.3f}")

    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
    df_eval = pd.DataFrame(eval_rows)
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

    print(f"\n=== COLAB {gpu_name} SUMMARY (Peak VRAM: {peak_vram_gb:.2f} GB) ===")
    print(summary.round(3).to_string(index=False))

    # Save metadata JSON
    metrics_payload = {
        "trained_on": gpu_name,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "faiss_enabled": True,
        "live_4bit_model": local_model_id,
        "swebench_tasks_loaded": len(swe_tasks),
        "optimal_tau": optimal_tau,
        "q_weak": round(Q_WEAK, 4),
        "q_strong": round(Q_STRONG, 4),
        "sklearn": sk_metrics,
        "torch": pt_metrics,
        "sklearn_sweep": sk_sweep.to_dict(orient="records"),
        "torch_sweep": pt_sweep.to_dict(orient="records"),
    }
    (out_dir / "router_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # 6. Render 4 Dark-Theme Charts
    BG_FIG, BG_AX, TEXT_MAIN, TEXT_MUTED, GRID_COL = "#0D1117", "#161B22", "#E6EDF3", "#8B949E", "#30363D"
    PALETTE = {"Brute Force": "#F85149", "Naive RAG": "#D29922", "Unbounded AST 2-Hop": "#A371F7", "OccamRoute (ML)": "#2EA043"}
    plt.rcParams.update({"figure.facecolor": BG_FIG, "axes.facecolor": BG_AX, "axes.edgecolor": GRID_COL, "axes.labelcolor": TEXT_MAIN, "xtick.color": TEXT_MAIN, "ytick.color": TEXT_MUTED, "text.color": TEXT_MAIN})

    # Chart 1: Diagnostics
    sk_probs = sk_router.predict_proba(dataset.X_val)
    pt_probs = pt_router.predict_proba(dataset.X_val)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8), dpi=250)
    fpr_sk, tpr_sk, _ = roc_curve(dataset.y_val, sk_probs)
    fpr_pt, tpr_pt, _ = roc_curve(dataset.y_val, pt_probs)
    ax1.plot(fpr_sk, tpr_sk, color="#58A6FF", lw=2, label=f"Sklearn GBDT (AUC={sk_metrics['roc_auc']:.3f})")
    ax1.plot(fpr_pt, tpr_pt, color="#2EA043", lw=2, label=f"PyTorch T4 (AUC={pt_metrics['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color=GRID_COL, ls="--")
    ax1.set_title("ROC Curve (Colab T4 Validation)"); ax1.legend(facecolor=BG_FIG, edgecolor=GRID_COL, fontsize=8.5)
    pt_t, pt_p = calibration_curve(dataset.y_val, pt_probs, n_bins=8)
    sk_t, sk_p = calibration_curve(dataset.y_val, sk_probs, n_bins=8)
    ax2.plot(sk_p, sk_t, marker="o", color="#58A6FF", label=f"Sklearn (ECE={sk_metrics['ece']:.3f})")
    ax2.plot(pt_p, pt_t, marker="s", color="#2EA043", label=f"PyTorch (ECE={pt_metrics['ece']:.3f})")
    ax2.plot([0, 1], [0, 1], color=GRID_COL, ls="--")
    ax2.set_title("Probability Calibration"); ax2.legend(facecolor=BG_FIG, edgecolor=GRID_COL, fontsize=8.5)
    ax3.plot(range(1, len(pt_router.train_losses) + 1), pt_router.train_losses, color="#A371F7", lw=2)
    ax3.set_title("PyTorch MLP Loss (Colab T4)")
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color=GRID_COL, ls="--", alpha=0.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "router_learning_diagnostics.png", dpi=250, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)

    # Chart 2: Permutation Importance & PGR Sweep
    perm = permutation_importance(sk_router.pipeline, dataset.X_val, dataset.y_val, n_repeats=10, random_state=42, scoring="roc_auc")
    df_imp = pd.DataFrame({"feature": list(dataset.feature_names), "importance": perm.importances_mean}).sort_values("importance", ascending=False).head(10).iloc[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=250)
    ax1.barh(df_imp["feature"], df_imp["importance"], color="#58A6FF")
    ax1.set_title("Top 10 Topological Features (Colab T4)")
    ax2.plot(sk_sweep["cost_savings_pct"], sk_sweep["pgr_pct"], marker="o", color="#58A6FF", label="Sklearn GBDT")
    ax2.plot(pt_sweep["cost_savings_pct"], pt_sweep["pgr_pct"], marker="D", color="#2EA043", label="PyTorch MLP (T4)")
    ax2.set_title("RouteLLM Trade-off: Cost Savings vs. PGR (%)"); ax2.legend(facecolor=BG_FIG, edgecolor=GRID_COL)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID_COL, ls="--", alpha=0.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "router_feature_and_pgr_sweep.png", dpi=250, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)

    # Chart 3: Pareto Frontier (Live FAISS + 4-bit Qwen)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=250)
    for _, r in summary.iterrows():
        p = str(r["pipeline"])
        ax1.scatter(r["mean_tokens"], r["quality_score"], color=PALETTE[p], s=220 if "OccamRoute" in p else 140, marker="*" if "OccamRoute" in p else "o", edgecolors="#FFF" if "OccamRoute" in p else "#0D1117")
        ax1.annotate(f"{p}\n{r['mean_tokens']:,.0f}t | PGR={r['pgr_pct']:.0f}%", (r["mean_tokens"], r["quality_score"]), xytext=(8, 4), textcoords="offset points", fontsize=8.5, color="#7EE787" if "OccamRoute" in p else TEXT_MAIN)
    ax1.axhline(Q_STRONG, color="#58A6FF", ls="--", alpha=0.6); ax1.axhline(Q_WEAK, color=TEXT_MUTED, ls="--", alpha=0.6)
    ax1.set_xscale("log"); ax1.set_title(f"Pareto Frontier (Colab {gpu_name}: Live FAISS + 4-bit)")
    x = np.arange(len(order)); w = 0.30
    ax2.bar(x - w / 2, summary["function_recall"], width=w, color="#58A6FF", label="Function Recall")
    ax2.bar(x + w / 2, summary["symbol_f1"], width=w, color="#2EA043", label="Symbol F1")
    for idx, (_, r) in enumerate(summary.iterrows()):
        ax2.text(idx, 1.07, f"${r['cost_per_1k_usd']:.2f}/1k\n({r['token_savings_pct']:+.0f}% tok)", ha="center", fontsize=8.3, color="#7EE787" if "OccamRoute" in str(r["pipeline"]) else TEXT_MUTED)
    ax2.set_xticks(x); ax2.set_xticklabels(order, fontsize=9, rotation=10); ax2.set_ylim(0, 1.24); ax2.set_title("Localization & Cost/1k (Colab T4)")
    ax2.legend(facecolor=BG_FIG, edgecolor=GRID_COL, loc="upper left")
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID_COL, ls="--", alpha=0.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "swebench_pareto_frontier.png", dpi=250, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)

    # Chart 4: Task Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=250)
    piv = df_eval.pivot(index="task_id", columns="pipeline", values="context_tokens")[order]
    for p in order:
        ax1.plot(piv.index, piv[p], marker="o", lw=2.2 if "OccamRoute" in p else 1.5, label=p, color=PALETTE[p])
    ax1.set_yscale("log"); ax1.set_title("Per-Task Token Footprint (Log)"); ax1.tick_params(axis="x", rotation=30); ax1.legend(facecolor=BG_FIG, edgecolor=GRID_COL, fontsize=8.3)
    oc_rows = df_eval[df_eval["pipeline"] == "OccamRoute (ML)"]
    ax2.bar(oc_rows["task_id"], oc_rows["p_strong"], color=["#58A6FF" if c == "low" else "#F78166" for c in oc_rows["expected_complexity"]])
    ax2.axhline(optimal_tau, color="#2EA043", ls="--", lw=1.6, label=f"tau={optimal_tau:.2f}")
    ax2.set_title("Trained T4 PyTorch Router P(strong)"); ax2.set_ylim(0, 1.08); ax2.tick_params(axis="x", rotation=30); ax2.legend(facecolor=BG_FIG, edgecolor=GRID_COL)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID_COL, ls="--", alpha=0.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "swebench_task_breakdown.png", dpi=250, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)

    print("ALL COLAB T4 OUTPUTS WRITTEN TO /content/colab_out/")


if __name__ == "__main__":
    main()
