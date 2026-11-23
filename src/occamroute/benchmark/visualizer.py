"""Publication-grade dark-theme Matplotlib charts for OccamRoute benchmarks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from occamroute.benchmark.evaluator import BenchmarkReport

BG_FIG = "#0D1117"
BG_AX = "#161B22"
TEXT_MAIN = "#E6EDF3"
TEXT_MUTED = "#8B949E"
GRID_COL = "#30363D"

PALETTE: dict[str, str] = {
    "Brute Force": "#F85149",
    "Naive RAG": "#D29922",
    "Unbounded AST 2-Hop": "#A371F7",
    "OccamRoute (ML)": "#2EA043",
}


def _apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG_FIG,
            "axes.facecolor": BG_AX,
            "axes.edgecolor": GRID_COL,
            "axes.labelcolor": TEXT_MAIN,
            "xtick.color": TEXT_MAIN,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT_MAIN,
            "font.family": "sans-serif",
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COL, linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pareto_frontier(
    summary_df: pd.DataFrame,
    output_path: Path | str,
    q_weak: float | None = None,
    q_strong: float | None = None,
    title: str | None = None,
) -> Path:
    """Render Pareto Frontier (Mean Tokens vs Quality) and Localization/Cost comparison."""
    _apply_dark_theme()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)

    for _, row in summary_df.iterrows():
        pipeline = str(row["pipeline"])
        is_occam = "OccamRoute" in pipeline
        color = PALETTE.get(pipeline, "#58A6FF")
        ax1.scatter(
            row["mean_tokens"],
            row["quality_score"],
            color=color,
            s=280 if is_occam else 150,
            marker="*" if is_occam else "o",
            edgecolors="#FFFFFF" if is_occam else BG_FIG,
            linewidths=1.4 if is_occam else 0.9,
            zorder=5 if is_occam else 3,
        )
        dx, dy = (-12, -24) if row["mean_tokens"] > 6000 else (10, 6)
        ax1.annotate(
            f"{pipeline}\n{row['mean_tokens']:,.0f} tok | ${row['cost_per_1k_usd']:.2f}/1k",
            (row["mean_tokens"], row["quality_score"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if is_occam else "normal",
            color="#7EE787" if is_occam else TEXT_MAIN,
        )

    if q_strong is not None:
        ax1.axhline(q_strong, color="#58A6FF", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Strong Ceiling (Q={q_strong:.3f})")
    if q_weak is not None:
        ax1.axhline(q_weak, color=TEXT_MUTED, linestyle=":", linewidth=1.2, alpha=0.8, label=f"Local Baseline (Q={q_weak:.3f})")
    if q_strong is not None or q_weak is not None:
        ax1.legend(facecolor=BG_FIG, edgecolor=GRID_COL, fontsize=8.3, loc="lower left")

    ax1.set_xscale("log")
    ax1.set_xlabel("Mean Context Input Tokens (Log Scale)")
    ax1.set_ylabel("Task Localization & Citation Quality Score")
    ax1.set_title(title or "Pareto Frontier: Context Tokens vs. Output Quality")
    _style_axis(ax1)

    pipelines = [str(p) for p in summary_df["pipeline"].tolist()]
    x = np.arange(len(pipelines))
    width = 0.30
    ax2.bar(x - width / 2, summary_df["function_recall"], width=width, color="#58A6FF", label="Function Recall")
    ax2.bar(x + width / 2, summary_df["symbol_f1"], width=width, color="#2EA043", label="Symbol F1")

    for idx, (_, row) in enumerate(summary_df.iterrows()):
        is_occam = "OccamRoute" in str(row["pipeline"])
        ax2.text(
            idx,
            1.06,
            f"${row['cost_per_1k_usd']:.2f}/1k\n({row['token_savings_pct']:+.1f}% tok)",
            ha="center",
            fontsize=8.3,
            fontweight="bold" if is_occam else "normal",
            color="#7EE787" if is_occam else TEXT_MUTED,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(pipelines, fontsize=9.2, rotation=8)
    ax2.set_ylim(0, 1.24)
    ax2.set_ylabel("Retrieval Score (0.0 – 1.0)")
    ax2.set_title("AST Localization Accuracy & Inference Cost / 1k Runs")
    ax2.legend(facecolor=BG_FIG, edgecolor=GRID_COL, loc="upper left", fontsize=8.5)
    _style_axis(ax2)

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    return out


def plot_token_compression(
    records_df: pd.DataFrame,
    output_path: Path | str,
    threshold: float | None = None,
    title: str | None = None,
) -> Path:
    """Render per-task log token compression and Lost-in-the-Middle burial reduction."""
    _apply_dark_theme()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    available = [p for p in ("Brute Force", "Naive RAG", "Unbounded AST 2-Hop", "OccamRoute (ML)") if p in records_df["pipeline"].unique()]
    piv = records_df.pivot(index="task_id", columns="pipeline", values="context_tokens")[available]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)

    for p in available:
        is_occam = "OccamRoute" in p
        ax1.plot(
            piv.index,
            piv[p],
            marker="o",
            linewidth=2.4 if is_occam else 1.6,
            color=PALETTE.get(p, "#58A6FF"),
            label=p,
        )
    if "Brute Force" in piv.columns and "OccamRoute (ML)" in piv.columns:
        ax1.fill_between(
            piv.index,
            piv["OccamRoute (ML)"],
            piv["Brute Force"],
            color="#2EA043",
            alpha=0.10,
            label="Pruned Token Savings",
        )
    ax1.set_yscale("log")
    ax1.set_ylabel("Input Context Tokens (Log Scale)")
    ax1.set_title(title or "Per-Task Context Token Footprint Comparison")
    ax1.tick_params(axis="x", rotation=28)
    ax1.legend(facecolor=BG_FIG, edgecolor=GRID_COL, fontsize=8.2)
    _style_axis(ax1)

    agg = (
        records_df.groupby("pipeline", as_index=False)
        .agg(
            mean_tokens=("context_tokens", "mean"),
            middle_burial_pct=("middle_burial_pct", "mean"),
        )
    )
    agg["pipeline"] = pd.Categorical(agg["pipeline"], categories=available, ordered=True)
    agg = agg.sort_values("pipeline").reset_index(drop=True)

    colors = [PALETTE.get(str(p), "#58A6FF") for p in agg["pipeline"]]
    bars = ax2.bar(agg["pipeline"].astype(str), agg["middle_burial_pct"], color=colors, width=0.52)
    for bar, (_, row) in zip(bars, agg.iterrows()):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.5,
            f"{h:.1f}% burial\n({row['mean_tokens']:,.0f} tok)",
            ha="center",
            fontsize=8.4,
            color=TEXT_MAIN,
        )

    tau_note = f" (tau={threshold:.2f})" if threshold is not None else ""
    ax2.set_ylim(0, max(58.0, float(agg["middle_burial_pct"].max()) + 15.0))
    ax2.set_ylabel("Middle-Burial Rate (% in 30–70% Span)")
    ax2.set_title(f"Lost-in-the-Middle Mitigation via U-Ordering{tau_note}")
    ax2.tick_params(axis="x", rotation=10)
    _style_axis(ax2)

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    return out


def generate_publication_charts(
    report: BenchmarkReport,
    output_dir: Path | str = "docs/assets",
) -> dict[str, Path]:
    """Generate `pareto_frontier.png` and `token_compression.png` from a BenchmarkReport."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pareto_path = plot_pareto_frontier(
        report.summary_df,
        out / "pareto_frontier.png",
        q_weak=report.q_weak,
        q_strong=report.q_strong,
    )
    compression_path = plot_token_compression(
        report.records_df,
        out / "token_compression.png",
        threshold=report.metadata.get("router_threshold"),
    )
    return {"pareto_frontier": pareto_path, "token_compression": compression_path}


def render_from_artifacts(
    eval_csv: Path | str = "artifacts/swebench_eval_results.csv",
    metrics_json: Path | str = "artifacts/router_metrics.json",
    output_dir: Path | str = "docs/assets",
    pipelines: Iterable[str] = ("Brute Force", "Naive RAG", "Unbounded AST 2-Hop", "OccamRoute (ML)"),
) -> dict[str, Path]:
    """Render publication charts directly from verified Colab GPU benchmark artifacts."""
    df_eval = pd.read_csv(eval_csv)
    selected = [p for p in pipelines if p in df_eval["pipeline"].unique()]
    df_eval = df_eval[df_eval["pipeline"].isin(selected)].copy()

    metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    q_weak = float(metrics.get("q_weak", 0.0))
    q_strong = float(metrics.get("q_strong", 1.0))
    tau = metrics.get("optimal_tau")

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
    summary["pipeline"] = pd.Categorical(summary["pipeline"], categories=selected, ordered=True)
    summary = summary.sort_values("pipeline").reset_index(drop=True)
    bf_tok = float(summary.loc[summary["pipeline"] == "Brute Force", "mean_tokens"].iloc[0])
    summary["token_savings_pct"] = (1.0 - summary["mean_tokens"] / max(1.0, bf_tok)) * 100.0
    summary["pgr_pct"] = ((summary["quality_score"] - q_weak) / max(1e-6, q_strong - q_weak)) * 100.0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpu = metrics.get("trained_on", "GPU")
    pareto_path = plot_pareto_frontier(
        summary,
        out / "pareto_frontier.png",
        q_weak=q_weak,
        q_strong=q_strong,
        title=f"Pareto Frontier ({gpu}: Live FAISS + 4-bit Qwen)",
    )
    comp_path = plot_token_compression(
        df_eval,
        out / "token_compression.png",
        threshold=tau,
        title=f"Token Compression & Context Footprint ({gpu})",
    )
    return {"pareto_frontier": pareto_path, "token_compression": comp_path}
