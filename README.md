# OccamRoute — Surgical AST Context Slicing & Cost-Quality LLM Routing Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maykonlincolnusa/OccamRoute/blob/main/notebooks/01_occamroute_colab_benchmark.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Colab NVIDIA L4 Verified](https://img.shields.io/badge/Colab%20NVIDIA%20L4-Peak%20VRAM%205.78%20GB-10b981.svg)](notebooks/01_occamroute_colab_benchmark.ipynb)
[![GBDT LORO AUC](https://img.shields.io/badge/GBDT%20LORO%20AUC-0.941-38bdf8.svg)](artifacts/router_metrics.json)
[![Token Savings](https://img.shields.io/badge/Token%20Savings--94.5%25-10b981.svg)](artifacts/swebench_eval_results.csv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](pyproject.toml)

**OccamRoute** is a deterministic-plus-ML **AST code-graph context compression** and **adaptive LLM complexity routing engine** built for autonomous coding agents, IDE copilots, and large-scale PR triage systems. Instead of saturating long-context windows with brute-force source dumps or severing call hierarchies via flat vector chunking, OccamRoute extracts a **token-budgeted k-hop AST dependency subgraph**, mitigates *Lost in the Middle* degradation through **U-shaped position-aware ordering**, and routes traffic between **4-bit quantized local GPUs** and **frontier reasoning models** via calibrated **Scikit-learn GBDT (default)** and **PyTorch MLP** classifiers.

> **Verified on NVIDIA L4 (`5.78 GB` Peak VRAM, Empirical v2):** **94.5% fewer input tokens** (`709.8` vs `12,981.8` tokens), **1.000 File & Function Recall**, **0.828 Symbol F1** (vs `0.218` FAISS Naive RAG), **`0.941` LORO ROC-AUC** on unseen repositories, and **94.9% lower benchmark cost** (`$0.88` vs `$17.17` per 1,000 runs with GBDT default; `$1.59` / `0.958` Quality / `83.9%` PGR in the PyTorch hybrid run).

---

## System Architecture

```mermaid
flowchart LR
    Q["Developer / Agent Query"] --> AST["Deterministic AST Parser (RepoGraph + PageRank)"]
    AST --> Slice["Surgical k-Hop Context Slicer (Hard Token Budget + U-Ordering)"]
    Slice --> Feat["17-D Topological Feature Extractor (Entropy, Cyclomatic, Centrality)"]
    Feat --> Router{"Calibrated ML Router (GBDT default / PyTorch MLP, P_strong >= tau)"}
    Router -- "P_strong < tau" --> Weak["Local 4-bit GPU (Qwen2.5-Coder-7B NF4, 5.78 GB VRAM)"]
    Router -- "P_strong >= tau" --> Strong["Frontier Reasoning Model (Multi-File / High-Entropy Refactors)"]
```

---

## Engineering Relevance Across Frontier AI & Hyperscale Platforms

As AI coding systems operate over multi-million-line repositories, teams hit two coupled scaling walls: **context rot** (burying definitions inside high-latency token dumps) and **blind frontier routing** (burning frontier API quotas on localized single-symbol queries). OccamRoute addresses core infrastructure bottlenecks shared across frontier labs and hyperscale developer platforms *(illustrative architectural alignment — independent open-source project with no affiliation implied)*:

| Engineering Organization / Class | Core Systems Bottleneck | OccamRoute Mechanism |
| :--- | :--- | :--- |
| **Anthropic & OpenAI** (Agentic Coding Loops) | Iterative tool-use loops compound context tokens quadratically across multi-turn repository exploration. | Hard-budgeted **k-hop PageRank subgraphs** preserve `1.00` call-chain recall inside `~710` avg tokens (`-94.5%`). |
| **Google** (Monorepo Search & Assist) | Multi-million-file monorepos fracture under naive 512-token text chunking (`0.218` Symbol F1). | Deterministic AST edge traversal + **U-shaped ordering** reduces middle-burial from `44.4%` to `12.8%`. |
| **Cursor** (Low-Latency IDE Context) | Real-time inline edits and Composer chats demand sub-5ms structural context selection before generation. | In-memory `RepoGraph` slices subgraphs in `< 2 ms` and computes 17-D routing verdicts in `~0.08 ms`. |
| **Meta** (Quantized Open-Weight Fleets) | Routing trivial lookups to frontier endpoints wastes capacity while self-hosted 4-bit Llama/Qwen clusters sit underutilized. | Calibrated **GBDT (`LORO AUC=0.941`)** generalizes across unseen repositories (`requests`, `flask`) under `5.78 GB` peak VRAM. |
| **DeepSeek** (Cost-Efficiency & Reasoning Tiering) | Balancing sparse local execution against deep multi-file reasoning models requires calibrated uncertainty boundaries. | Empirical L4 outcome labeling + RouteLLM threshold sweeps expose real cross-repo generalization and cost/quality trade-offs. |
| **TikTok / ByteDance** (Microservice FinOps) | High-throughput CI/CD PR triage requires strict per-run token governance across polyglot service graphs. | Cuts measured benchmark spend from **`$17.17` → `$0.88` per 1,000 runs** (GBDT default, `$162.9k / mo` saved at `10M runs/mo`; `$155.8k / mo` with PyTorch hybrid). |

---

## Verified GPU Benchmark Results (`encode/httpx@0.28.1` + Live FAISS + 4-bit NF4 on NVIDIA L4)

Executed on **Google Colab NVIDIA L4 (`22.03 GB` VRAM)** via [colab_l4_train_v2.py](scripts/colab_l4_train_v2.py) and [colab_l4_httpx_gbdt.py](scripts/colab_l4_httpx_gbdt.py) comparing **Brute Force**, **Naive RAG (FAISS + `BAAI/bge-small-en-v1.5`)**, **Unbounded AST 2-Hop**, and **OccamRoute (ML)** using live **4-bit NF4 `Qwen/Qwen2.5-Coder-7B-Instruct`** (`5.78 GB` Peak VRAM):

| Pipeline | Avg Tokens | Token Savings | File Recall | Func Recall | Symbol F1 | Middle-Burial | Quality | Cost / 1k Runs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brute Force** | 12,981.8 | 0.0% | 1.000 | 1.000 | 0.076 | 29.6% | 0.808 | $17.17 |
| **Naive RAG (FAISS + BGE-small)** | 2,462.8 | +81.0% | 0.833 | 0.744 | 0.218 | 22.9% | 0.674 | $4.02 |
| **Unbounded AST 2-Hop** | 4,476.6 | +65.5% | 0.917 | 0.942 | 0.232 | 44.4% | 0.778 | $6.54 |
| **OccamRoute (ML — GBDT Default)** | **709.8** | **+94.5%** | **1.000** | **1.000** | **0.828** | **12.8%** | **0.859** | **$0.88** |

> *PyTorch MLP `httpx` companion run on the same L4 setup:* `41.67%` local offload, `0.958` Quality (`83.9%` PGR), `$1.59 / 1k` runs. Both routers outperform Brute Force (`0.808`), Unbounded AST (`0.778`), and Naive RAG (`0.674`) in both quality and token cost.

### v2 Training Methodology (Empirical, Multi-Repo + LORO)
*Supersedes v1 (Tesla T4 synthetic archetypes, retained only in git history):*
- **178 Real Multi-Repo Tasks:** `58` ground-truth issues from `princeton-nlp/SWE-bench_Lite` + `120` focused single-symbol and multi-hop call-chain tasks mined directly from AST subgraphs.
- **Strict Repository Split (LORO):** Trained on `6` repositories (`pytest-dev/pytest`, `sphinx-doc/sphinx`, `scikit-learn/scikit-learn`, `pylint-dev/pylint`, `pydata/xarray`, `mwaskom/seaborn` — `145` pool tasks, `37` stratified val, `280` intra-class balanced train rows strictly inside train) and evaluated **Leave-One-Repo-Out (LORO)** on `33` tasks from `2` completely unseen repositories (`psf/requests`, `pallets/flask`).
- **Empirical Outcome Labels:** Every task was executed live on NVIDIA L4 with 4-bit NF4 `Qwen/Qwen2.5-Coder-7B-Instruct` (`105` `strong_frontier` / `73` `weak_local`, `59.0%` positive rate).

### ML Router Validation & Dual-Router Diagnosis (Colab L4)

| Router Backend | Val Accuracy | Val ROC-AUC | Val F1 | Val Brier / ECE | Unseen LORO Acc | Unseen LORO AUC | Unseen LORO F1 / ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sklearn Calibrated GBDT (Default)** | **0.946** | **0.991** | **0.957** | `0.046 / 0.119` | **0.818** | **0.941** | `0.833 / 0.155` |
| **PyTorch Tabular MLP (`.pt`)** | 0.838 | 0.906 | 0.857 | `0.133 / 0.164` | 0.576 | 0.548 | `0.632 / 0.402` |

- **Local CPU Reproduction (`scikit-learn 1.9.1`):** Refitting `SklearnComplexityRouter` from [router_training_dataset.csv](artifacts/router_training_dataset.csv) reproduces Colab's unseen LORO generalization (`Acc = 0.818`, `ROC-AUC = 0.941`, `F1 = 0.833`, `Brier = 0.135`, `ECE = 0.155`).
- **Why GBDT is Default & Honest Trade-off:** PyTorch MLP overfit training-repo patterns (`LORO AUC = 0.548`), whereas GBDT transferred cleanly (`LORO AUC = 0.941`), making GBDT the production default. Conversely, on the `12` concise zero-shot `httpx` prompts (`15–32` tokens), GBDT's `69%` split weight on `query_token_count` routed `100%` to `weak_local` (`Q = 0.859`, `$0.88/1k`, `PGR = 0.0%`), whereas the PyTorch MLP's LayerNorm features escalated complex traces (`41.7%` local offload, `Q = 0.958`, `PGR = 83.9%`, `$1.59/1k`).

![OccamRoute Pareto Frontier comparing Brute Force, Naive RAG, Unbounded AST 2-Hop, and OccamRoute GBDT on NVIDIA L4](docs/assets/pareto_frontier.png)
![OccamRoute Per-Task Log Token Compression and Lost-in-the-Middle Burial Reduction](docs/assets/token_compression.png)

> Additional L4 diagnostics & LORO curves: [router_learning_diagnostics.png](assets/router_learning_diagnostics.png) · [router_feature_and_pgr_sweep.png](assets/router_feature_and_pgr_sweep.png) · [swebench_pareto_frontier.png](assets/swebench_pareto_frontier.png) · [swebench_task_breakdown.png](assets/swebench_task_breakdown.png)

---

## Repository Architecture

```text
OccamRoute/
├── pyproject.toml
├── README.md
├── notebooks/
│   └── 01_occamroute_colab_benchmark.ipynb      # 1-Click Colab GPU benchmark (Brute Force vs Naive RAG vs OccamRoute)
├── docs/
│   └── assets/
│       ├── pareto_frontier.png                  # Pareto frontier: context tokens vs quality & cost/1k
│       └── token_compression.png                # Per-task token compression & U-ordering burial reduction
├── src/occamroute/
│   ├── graph_builder.py                         # Deterministic Python AST -> NetworkX DiGraph + PageRank + McCabe CC
│   ├── context_slicer.py                        # Token-budgeted k-hop slicer + U-shaped position-aware ordering
│   ├── feature_extractor.py                     # 17-D topological subgraph feature extractor
│   ├── dataset_loader.py                        # SWE-bench Lite loader & unified patch hunk ground-truth parser
│   ├── impact_analyzer.py                       # Git diff blast-radius analyzer over reverse call graphs
│   ├── cli.py & mcp_server.py                   # Production CLI + FastMCP server for IDEs
│   ├── benchmark/
│   │   ├── baselines.py                         # BruteForceBaseline & NaiveRAGBaseline (512t chunking + FAISS/TF-IDF)
│   │   ├── evaluator.py                         # SWEBenchBenchmarkEvaluator, metrics & RouteLLM PGR calculation
│   │   └── visualizer.py                        # Dark-theme Matplotlib publication chart generator
│   ├── router/
│   │   ├── dataset_builder.py                   # RouteLLM topological dataset builder & augmentation
│   │   ├── sklearn_router.py                    # Calibrated GBDT classifier (.joblib, default)
│   │   ├── torch_router.py                      # TabularRouterMLP (LayerNorm + SiLU + Dropout, .pt)
│   │   └── engine.py                            # Unified OccamRouter, cross-version loader & threshold sweep
│   └── dashboard/                               # FastAPI + Pydantic v2 telemetry & routing playground
├── scripts/                                     # Colab L4 v2 training/httpx scripts & dashboard server
├── artifacts/                                   # L4 v2 checkpoints, 178-row real dataset, telemetry & HTML snapshot
└── tests/                                       # Automated pytest suite (AST, Router, Benchmark, Dashboard, Tools)
```

---

## Quickstart & Programmatic Benchmark API

```bash
# 1. Install dependencies with uv
uv sync --extra dev

# 2. Run the full automated test suite
PYTHONPATH=src uv run --extra dev pytest -q --tb=short

# 3. Launch the interactive KPI Observability Dashboard (http://127.0.0.1:8001)
PYTHONPATH=src .venv/bin/python scripts/run_dashboard.py --host 127.0.0.1 --port 8001
```

### Run the 3-Pipeline Benchmark in Python
```python
from occamroute import OccamRouter, RepoGraphBuilder
from occamroute.benchmark import (
    SWEBenchBenchmarkEvaluator,
    build_repo_eval_tasks,
    generate_publication_charts,
)

repo_graph = RepoGraphBuilder().build_from_directory("src/occamroute")
router = OccamRouter.from_sklearn_artifacts("artifacts", threshold=0.30)
tasks = build_repo_eval_tasks(repo_graph, max_tasks=8)

evaluator = SWEBenchBenchmarkEvaluator(repo_graph=repo_graph, router=router, rag_backend="auto")
report = evaluator.evaluate(tasks)
print(report.summary_df[["pipeline", "mean_tokens", "token_savings_pct", "symbol_f1", "cost_per_1k_usd"]])
generate_publication_charts(report, output_dir="docs/assets")
```

### CLI, IDE MCP Server & GitHub Action
```bash
occamroute map --repo . --top 10
occamroute slice "Trace how SurgicalContextSlicer enforces token_budget" --repo . --budget 1200
occamroute impact --repo . --ref HEAD~1 --markdown
```
- **MCP Integration:** Register `occamroute mcp --repo .` in `mcp_config.json` (Antigravity, Cursor, Claude Code, VS Code) to expose `occamroute_slice_context`, `occamroute_diff_impact`, and `occamroute_repo_map`.
- **GitHub Action:** Ready-to-run workflow in [.github/workflows/occamroute-pr-impact.yml](.github/workflows/occamroute-pr-impact.yml).

---

## References (Peer-Reviewed & Benchmarks)

1. **RepoGraph:** *Enhancing AI Software Engineering with Repository-Level Code Graph* — Ouyang et al. ([arXiv:2410.14684](https://arxiv.org/abs/2410.14684), ICLR 2025)
2. **Lost in the Middle:** *How Language Models Use Long Contexts* — Liu et al. ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172), TACL 2024)
3. **RouteLLM:** *Learning to Route LLMs with Preference Data* — Ong et al. ([arXiv:2406.18665](https://arxiv.org/abs/2406.18665), 2024)
4. **SWE-bench:** *Can Language Models Resolve Real-World GitHub Issues?* — Jimenez et al. ([arXiv:2310.06770](https://arxiv.org/abs/2310.06770), ICLR 2024)

```bibtex
@software{occamroute2026,
  title={OccamRoute: Surgical AST Context Slicing and Adaptive Cost-Quality LLM Routing},
  author={Santos, Maykon Lincoln},
  year={2026},
  url={https://github.com/maykonlincolnusa/OccamRoute}
}
```
