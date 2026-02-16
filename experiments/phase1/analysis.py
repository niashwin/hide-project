"""
Phase 1 Analysis: Aggregation and Figure Generation
=====================================================
Aggregates results across 5 seeds, computes bootstrap CIs,
generates publication-quality figures.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "utils"))
from metrics import bootstrap_ci, aggregate_seeds


def aggregate(
    all_results: Dict[int, Dict],
    results_dir: Path,
    figures_dir: Path,
) -> Dict:
    """Aggregate Phase 1 results across seeds and generate figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Filter out errored seeds
    valid = {s: r for s, r in all_results.items() if "error" not in r}
    if not valid:
        return {"error": "All seeds failed"}

    seeds = sorted(valid.keys())
    task_ids = [1, 2, 3, 4, 5]

    # ── Aggregate accuracy per task and method ──
    methods = ["hide", "no_memory", "full_context", "random_retrieval", "vanilla_rag"]
    method_keys = {
        "hide": "hide_accuracy",
        "no_memory": "no_memory_accuracy",
        "full_context": "full_context_accuracy",
        "random_retrieval": "random_retrieval_accuracy",
        "vanilla_rag": "vanilla_rag_accuracy",
    }

    summary = {"tasks": {}, "methods": {}}

    for task_id in task_ids:
        task_key = str(task_id)
        summary["tasks"][task_key] = {}

        for method, key in method_keys.items():
            values = []
            for seed in seeds:
                task_data = valid[seed].get("tasks", {}).get(task_key, {})
                val = task_data.get(key, 0.0)
                values.append(val)

            values = np.array(values)
            point, ci_lo, ci_hi = bootstrap_ci(values)
            summary["tasks"][task_key][method] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "values": values.tolist(),
            }

    # ── Overall method averages ──
    for method, key in method_keys.items():
        all_values = []
        for seed in seeds:
            accs = []
            for task_id in task_ids:
                task_data = valid[seed].get("tasks", {}).get(str(task_id), {})
                accs.append(task_data.get(key, 0.0))
            all_values.append(np.mean(accs))
        all_values = np.array(all_values)
        point, ci_lo, ci_hi = bootstrap_ci(all_values)
        summary["methods"][method] = {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }

    # ── Validation checks ──
    summary["validation"] = {}

    # HIDE > no_memory on >=4/5 tasks
    hide_wins_no_mem = 0
    for task_id in task_ids:
        tk = str(task_id)
        h = summary["tasks"][tk]["hide"]["mean"]
        n = summary["tasks"][tk]["no_memory"]["mean"]
        if h > n:
            hide_wins_no_mem += 1
    summary["validation"]["hide_beats_no_memory"] = hide_wins_no_mem
    summary["validation"]["hide_beats_no_memory_pass"] = hide_wins_no_mem >= 4

    # HIDE > random on all 5 tasks
    hide_wins_random = 0
    for task_id in task_ids:
        tk = str(task_id)
        h = summary["tasks"][tk]["hide"]["mean"]
        r = summary["tasks"][tk]["random_retrieval"]["mean"]
        if h > r:
            hide_wins_random += 1
    summary["validation"]["hide_beats_random"] = hide_wins_random
    summary["validation"]["hide_beats_random_pass"] = hide_wins_random == 5

    # ── Memory scaling aggregation ──
    scaling = {}
    for seed in seeds:
        seed_scaling = valid[seed].get("memory_scaling", {})
        for n_str, vals in seed_scaling.items():
            n = int(n_str) if isinstance(n_str, str) else n_str
            if n not in scaling:
                scaling[n] = []
            scaling[n].append(vals.get("mean_p_at_5", 0.0))

    summary["memory_scaling"] = {}
    for n in sorted(scaling.keys()):
        values = np.array(scaling[n])
        point, ci_lo, ci_hi = bootstrap_ci(values)
        summary["memory_scaling"][n] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }

    # ── Generate Figures ──
    _generate_figures(summary, task_ids, figures_dir)

    return summary


def _generate_figures(summary: Dict, task_ids: list, figures_dir: Path):
    """Generate Phase 1 publication figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    # ── Figure 2a: Bar chart — HIDE vs baselines per task ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    methods = ["hide", "no_memory", "full_context", "random_retrieval", "vanilla_rag"]
    labels = ["HIDE", "No Memory", "Full Context", "Random", "Vanilla RAG"]
    colors = ["#2196F3", "#9E9E9E", "#4CAF50", "#FF9800", "#E91E63"]

    x = np.arange(len(task_ids))
    width = 0.15

    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means = [summary["tasks"][str(t)][method]["mean"] for t in task_ids]
        stds = [summary["tasks"][str(t)][method]["std"] for t in task_ids]
        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("bAbI Task")
    ax.set_ylabel("Accuracy")
    ax.set_title("Phase 1: HIDE vs Baselines on bAbI Tasks")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"Task {t}" for t in task_ids])
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase1_accuracy_bar.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase1_accuracy_bar.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2b: Memory scaling line chart ──
    scaling = summary.get("memory_scaling", {})
    if scaling:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ns = sorted(scaling.keys(), key=int)
        means = [scaling[n]["mean"] for n in ns]
        ci_lo = [scaling[n]["ci_lower"] for n in ns]
        ci_hi = [scaling[n]["ci_upper"] for n in ns]

        ax.plot(ns, means, "o-", color="#2196F3", linewidth=2, markersize=6, label="HIDE P@5")
        ax.fill_between(ns, ci_lo, ci_hi, alpha=0.2, color="#2196F3")
        ax.set_xlabel("Number of Stored Memories (N)")
        ax.set_ylabel("Precision@5")
        ax.set_title("Memory Scaling: P@5 vs Memory Size")
        ax.set_xscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "phase1_scaling.pdf", bbox_inches="tight")
        fig.savefig(figures_dir / "phase1_scaling.png", bbox_inches="tight")
        plt.close(fig)

    # ── Figure 2c: Retrieval precision heatmap ──
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    heatmap_data = np.zeros((len(task_ids), len(methods)))
    for i, task_id in enumerate(task_ids):
        for j, method in enumerate(methods):
            heatmap_data[i, j] = summary["tasks"][str(task_id)][method]["mean"]

    sns.heatmap(
        heatmap_data, ax=ax,
        xticklabels=labels,
        yticklabels=[f"Task {t}" for t in task_ids],
        annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=1,
    )
    ax.set_title("Accuracy Heatmap: Tasks x Methods")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase1_heatmap.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase1_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Phase 1 figures saved to {figures_dir}")
