"""
Phase 3 Analysis: Aggregation and Figure Generation
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
    """Aggregate Phase 3 results across seeds."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    valid = {s: r for s, r in all_results.items() if "error" not in r}
    if not valid:
        return {"error": "All seeds failed"}

    seeds = sorted(valid.keys())
    conditions = ["no_consolidation", "consolidation_only", "replay_only", "full_hide",
                   "naive_pruning", "experience_replay"]

    summary = {"seeds": seeds, "n_seeds": len(seeds)}

    # Aggregate per-condition metrics
    for condition in conditions:
        compressions = []
        backward_transfers = []
        final_accs = []

        for seed in seeds:
            cond_data = valid[seed].get("conditions", {}).get(condition, {})
            compressions.append(cond_data.get("compression_ratio", 1.0))
            backward_transfers.append(cond_data.get("mean_backward_transfer", 0.0))

            # Final accuracy (mean across all tasks at end)
            task_accs = cond_data.get("task_accuracies", {})
            if "9" in task_accs:
                final_task_accs = list(task_accs["9"].values())
                final_accs.append(float(np.mean([float(v) for v in final_task_accs])))

        summary[condition] = {
            "compression_mean": float(np.mean(compressions)),
            "compression_std": float(np.std(compressions)),
            "backward_transfer_mean": float(np.mean(backward_transfers)),
            "backward_transfer_std": float(np.std(backward_transfers)),
            "final_accuracy_mean": float(np.mean(final_accs)) if final_accs else 0.0,
            "final_accuracy_std": float(np.std(final_accs)) if final_accs else 0.0,
        }

    # Validation checks
    full_hide_comp = summary.get("full_hide", {}).get("compression_mean", 1.0)
    no_cons_acc = summary.get("no_consolidation", {}).get("final_accuracy_mean", 0.0)
    full_hide_acc = summary.get("full_hide", {}).get("final_accuracy_mean", 0.0)

    summary["validation"] = {
        "compression_ge_30pct": full_hide_comp <= 0.70,  # ≥30% reduction means ratio ≤0.70
        "accuracy_loss_lt_5pct": abs(full_hide_acc - no_cons_acc) < 0.05 or full_hide_acc >= no_cons_acc,
    }

    # Generate figures
    _generate_figures(summary, all_results, figures_dir)

    return summary


def _generate_figures(summary: Dict, all_results: Dict, figures_dir: Path):
    """Generate Phase 3 figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    valid = {s: r for s, r in all_results.items() if "error" not in r}
    seeds = sorted(valid.keys())

    # ── Figure 4a: Backward transfer heatmap ──
    n_tasks = 10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, condition in enumerate(["no_consolidation", "full_hide"]):
        ax = axes[ax_idx]
        # Average task accuracy matrix across seeds
        acc_matrix = np.zeros((n_tasks, n_tasks))
        counts = np.zeros((n_tasks, n_tasks))

        for seed in seeds:
            cond_data = valid[seed].get("conditions", {}).get(condition, {})
            task_accs = cond_data.get("task_accuracies", {})
            for trained_str, tested_dict in task_accs.items():
                trained = int(trained_str)
                for tested_str, acc in tested_dict.items():
                    tested = int(tested_str)
                    acc_matrix[trained, tested] += float(acc)
                    counts[trained, tested] += 1

        mask = counts > 0
        acc_matrix[mask] /= counts[mask]
        acc_matrix[~mask] = np.nan

        sns.heatmap(acc_matrix, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, mask=~mask,
                    xticklabels=range(n_tasks), yticklabels=range(n_tasks))
        ax.set_xlabel("Test Task")
        ax.set_ylabel("After Training Task")
        ax.set_title(f"{'No Consolidation' if ax_idx == 0 else 'Full HIDE'}")

    fig.suptitle("Backward Transfer Heatmap", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_backward_transfer.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase3_backward_transfer.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4b: Compression vs Accuracy ──
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    conditions = ["no_consolidation", "consolidation_only", "replay_only", "full_hide",
                   "naive_pruning", "experience_replay"]
    labels = ["No Consolidation", "Consolidation Only", "Replay Only", "Full HIDE",
              "Naive Pruning", "Experience Replay"]
    colors = ["#9E9E9E", "#FF9800", "#4CAF50", "#2196F3", "#E91E63", "#9C27B0"]

    for condition, label, color in zip(conditions, labels, colors):
        comp = summary.get(condition, {}).get("compression_mean", 1.0)
        acc = summary.get(condition, {}).get("final_accuracy_mean", 0.0)
        comp_std = summary.get(condition, {}).get("compression_std", 0.0)
        acc_std = summary.get(condition, {}).get("final_accuracy_std", 0.0)
        ax.errorbar(comp, acc, xerr=comp_std, yerr=acc_std, fmt="o", color=color,
                    markersize=10, capsize=5, label=label, linewidth=2)

    ax.set_xlabel("Compression Ratio (lower = more compression)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Compression vs Accuracy Trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_compression_accuracy.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase3_compression_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4c: Interference over time ──
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for condition, label, color in zip(
        ["no_consolidation", "full_hide"],
        ["No Consolidation", "Full HIDE"],
        ["#9E9E9E", "#2196F3"]
    ):
        # Plot accuracy on task 0 as more tasks are added
        task0_accs = []
        for after_task in range(n_tasks):
            accs_per_seed = []
            for seed in seeds:
                cond_data = valid[seed].get("conditions", {}).get(condition, {})
                task_accs = cond_data.get("task_accuracies", {})
                val = task_accs.get(str(after_task), {}).get("0", None)
                if val is not None:
                    accs_per_seed.append(float(val))
            if accs_per_seed:
                task0_accs.append((after_task, np.mean(accs_per_seed), np.std(accs_per_seed)))

        if task0_accs:
            x = [t[0] for t in task0_accs]
            y = [t[1] for t in task0_accs]
            err = [t[2] for t in task0_accs]
            ax.errorbar(x, y, yerr=err, fmt="o-", color=color, label=label,
                        capsize=3, linewidth=2, markersize=5)

    ax.set_xlabel("Number of Tasks Trained")
    ax.set_ylabel("Task 0 Accuracy")
    ax.set_title("Retroactive Interference: Task 0 Accuracy Over Time")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_interference.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase3_interference.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Phase 3 figures saved to {figures_dir}")
