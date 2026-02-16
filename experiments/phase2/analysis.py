"""
Phase 2 Analysis: Aggregation and Figure Generation
=====================================================
Aggregates Ebbinghaus results across seeds, generates figures.
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
from metrics import bootstrap_ci


def aggregate(
    all_results: Dict[int, Dict],
    results_dir: Path,
    figures_dir: Path,
) -> Dict:
    """Aggregate Phase 2 results across seeds and generate figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    valid = {s: r for s, r in all_results.items() if "error" not in r}
    if not valid:
        return {"error": "All seeds failed"}

    seeds = sorted(valid.keys())
    summary = {"seeds": seeds, "n_seeds": len(seeds)}

    # ── Ebbinghaus aggregation ──
    r2_values = []
    a_values = []
    b_values = []
    all_ages = []
    all_retentions = []

    for seed in seeds:
        ebb = valid[seed].get("ebbinghaus", {})
        fit = ebb.get("power_law_fit", {})
        if "r_squared" in fit:
            r2_values.append(fit["r_squared"])
            a_values.append(fit["a"])
            b_values.append(fit["b"])
        if "ages_days" in ebb and "retentions" in ebb:
            all_ages.append(ebb["ages_days"])
            all_retentions.append(ebb["retentions"])

    if r2_values:
        r2_arr = np.array(r2_values)
        point, ci_lo, ci_hi = bootstrap_ci(r2_arr)
        summary["ebbinghaus"] = {
            "r_squared_mean": float(np.mean(r2_arr)),
            "r_squared_std": float(np.std(r2_arr)),
            "r_squared_ci_lower": ci_lo,
            "r_squared_ci_upper": ci_hi,
            "a_mean": float(np.mean(a_values)),
            "b_mean": float(np.mean(b_values)),
            "b_std": float(np.std(b_values)),
        }
    else:
        summary["ebbinghaus"] = {"error": "No valid power law fits"}

    # ── TempLAMA aggregation ──
    temporal_accs = []
    no_decay_accs = []
    for seed in seeds:
        tl = valid[seed].get("templama", {})
        temporal_accs.append(tl.get("temporal_accuracy", 0.0))
        no_decay_accs.append(tl.get("no_decay_accuracy", 0.0))

    temporal_arr = np.array(temporal_accs)
    no_decay_arr = np.array(no_decay_accs)
    summary["templama"] = {
        "temporal_mean": float(np.mean(temporal_arr)),
        "temporal_std": float(np.std(temporal_arr)),
        "no_decay_mean": float(np.mean(no_decay_arr)),
        "no_decay_std": float(np.std(no_decay_arr)),
        "temporal_beats_no_decay": int(np.sum(temporal_arr > no_decay_arr)),
    }

    # ── Regression aggregation ──
    reg_accs = []
    reg_deltas = []
    reg_pass = []
    for seed in seeds:
        reg = valid[seed].get("regression", {})
        reg_accs.append(reg.get("mean_accuracy", 0.0))
        reg_deltas.append(reg.get("delta", 1.0))
        reg_pass.append(reg.get("pass", False))

    summary["regression"] = {
        "mean_accuracy": float(np.mean(reg_accs)),
        "mean_delta": float(np.mean(reg_deltas)),
        "all_pass": all(reg_pass),
        "pass_count": sum(reg_pass),
    }

    # ── Validation ──
    summary["validation"] = {
        "ebbinghaus_r2_computed": "r_squared_mean" in summary.get("ebbinghaus", {}),
        "regression_pass": summary["regression"]["all_pass"],
    }

    # ── Generate figures ──
    _generate_figures(summary, all_results, figures_dir)

    return summary


def _generate_figures(summary: Dict, all_results: Dict, figures_dir: Path):
    """Generate Phase 2 publication figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    # ── Figure 3a: Ebbinghaus forgetting curve ──
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Human reference
    ebb_cfg = {"human_times_min": [20, 60, 480, 1440, 2880, 8640, 44640],
               "human_retention": [0.58, 0.44, 0.36, 0.34, 0.28, 0.25, 0.21]}
    human_days = [t / 1440.0 for t in ebb_cfg["human_times_min"]]
    human_ret = ebb_cfg["human_retention"]
    ax.plot(human_days, human_ret, "s-", color="#E91E63", linewidth=2,
            markersize=8, label="Human (Ebbinghaus 1885)", zorder=5)

    # HIDE retention curves (per seed)
    colors_seeds = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    valid = {s: r for s, r in all_results.items() if "error" not in r}
    seeds = sorted(valid.keys())

    for i, seed in enumerate(seeds):
        ebb = valid[seed].get("ebbinghaus", {})
        ages = ebb.get("ages_days", [])
        rets = ebb.get("retentions", [])
        if ages and rets:
            ax.plot(ages, rets, "o--", color=colors_seeds[i % len(colors_seeds)],
                    alpha=0.5, markersize=4, linewidth=1,
                    label=f"HIDE seed {seed}" if i < 3 else None)

    # Mean HIDE curve
    if seeds:
        # Aggregate across seeds at common age bins
        all_ages_flat = []
        all_rets_flat = []
        for seed in seeds:
            ebb = valid[seed].get("ebbinghaus", {})
            if "ages_days" in ebb and "retentions" in ebb:
                all_ages_flat.extend(ebb["ages_days"])
                all_rets_flat.extend(ebb["retentions"])
        if all_ages_flat:
            ax.scatter(all_ages_flat, all_rets_flat, c="#2196F3", alpha=0.15, s=15, zorder=2)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Retention (mean cosine similarity)")
    ax.set_title("Forgetting Curve: HIDE vs Human (Ebbinghaus)")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)

    # Add R² annotation
    ebb_summary = summary.get("ebbinghaus", {})
    if "r_squared_mean" in ebb_summary:
        ax.annotate(
            f"R² = {ebb_summary['r_squared_mean']:.3f} ± {ebb_summary.get('r_squared_std', 0):.3f}\n"
            f"b = {ebb_summary.get('b_mean', 0):.3f} (human ≈ 0.5)",
            xy=(0.95, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray")
        )

    fig.tight_layout()
    fig.savefig(figures_dir / "phase2_forgetting_curve.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase2_forgetting_curve.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3b: Temporal retrieval heatmap (decay comparison) ──
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    methods = ["exponential", "power_law", "logarithmic"]
    method_labels = ["Exponential", "Power Law", "Logarithmic"]

    # Aggregate retention curves per method
    method_retentions = {m: [] for m in methods}
    for seed in seeds:
        ebb = valid[seed].get("ebbinghaus", {})
        per_method = ebb.get("per_method", {})
        for method in methods:
            if method in per_method:
                raw = per_method[method].get("raw_retention_by_bin", {})
                if raw:
                    ages = sorted(raw.keys(), key=float)
                    rets = [raw[a] for a in ages]
                    method_retentions[method].append((ages, rets))

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for method, label, color in zip(methods, method_labels, colors):
        if method_retentions[method]:
            # Average across seeds
            all_curves = method_retentions[method]
            # Plot individual curves with low alpha
            for ages, rets in all_curves:
                ages_float = [float(a) for a in ages]
                ax.plot(ages_float, rets, "o-", color=color, alpha=0.3, markersize=3, linewidth=1)
            # Plot label
            if all_curves:
                ages_float = [float(a) for a in all_curves[0][0]]
                ax.plot([], [], "o-", color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel("Memory Age (days)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Temporal Retrieval: Decay Method Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase2_temporal_heatmap.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase2_temporal_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3c: t-SNE colored by time ──
    # This needs actual embeddings which we don't store across seeds
    # Generate a placeholder with regression comparison
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Regression comparison: Phase 1 vs Phase 2 per-task accuracy
    task_ids = [1, 2, 3, 4, 5]
    phase1_accs = []
    phase2_accs = []

    for seed in seeds:
        reg = valid[seed].get("regression", {})
        task_accs = reg.get("task_accuracies", {})
        for tid in task_ids:
            phase2_accs.append(task_accs.get(str(tid), 0.0))

    # Load Phase 1 reference
    phase1_path = Path("results/phase1/summary.json")
    if phase1_path.exists():
        with open(phase1_path) as f:
            p1 = json.load(f)
        for tid in task_ids:
            p1_task = p1.get("tasks", {}).get(str(tid), {}).get("hide", {})
            phase1_accs.append(p1_task.get("mean", 0.0))

    if phase1_accs and phase2_accs:
        x = np.arange(len(task_ids))
        width = 0.35

        # Average Phase 2 across seeds
        p2_per_task = {}
        for seed in seeds:
            reg = valid[seed].get("regression", {})
            task_accs = reg.get("task_accuracies", {})
            for tid in task_ids:
                p2_per_task.setdefault(tid, []).append(task_accs.get(str(tid), 0.0))

        p2_means = [np.mean(p2_per_task.get(tid, [0])) for tid in task_ids]
        p2_stds = [np.std(p2_per_task.get(tid, [0])) for tid in task_ids]

        ax.bar(x - width/2, phase1_accs[:len(task_ids)], width, label="Phase 1 (HIDE)",
               color="#2196F3", edgecolor="white")
        ax.bar(x + width/2, p2_means, width, yerr=p2_stds, label="Phase 2 (Temporal HIDE)",
               color="#4CAF50", capsize=3, edgecolor="white")

        ax.set_xlabel("bAbI Task")
        ax.set_ylabel("Accuracy")
        ax.set_title("Regression: Phase 1 vs Phase 2 on bAbI")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Task {t}" for t in task_ids])
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

        # Add tolerance band
        tolerance = 0.02
        for i, p1_acc in enumerate(phase1_accs[:len(task_ids)]):
            ax.axhline(y=p1_acc - tolerance, color="red", linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "phase2_regression.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase2_regression.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Phase 2 figures saved to {figures_dir}")
