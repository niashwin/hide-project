"""Phase 4 Analysis: Aggregation and Figure Generation"""
import json, numpy as np, matplotlib
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

def aggregate(all_results: Dict[int, Dict], results_dir: Path, figures_dir: Path) -> Dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    valid = {s: r for s, r in all_results.items() if "error" not in r}
    if not valid:
        return {"error": "All seeds failed"}
    seeds = sorted(valid.keys())
    summary = {"seeds": seeds, "n_seeds": len(seeds)}

    metrics = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"]
    for method in ["hide_retrieval", "random_baseline"]:
        method_results = {}
        for m in metrics:
            vals = [valid[s].get(method, {}).get(m, 0) for s in seeds]
            arr = np.array(vals)
            point, ci_lo, ci_hi = bootstrap_ci(arr)
            method_results[m] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                                  "ci_lower": ci_lo, "ci_upper": ci_hi}
        summary[method] = method_results

    # Transfer
    transfer_vals = {"r1": [], "r5": [], "r10": []}
    for s in seeds:
        t = valid[s].get("transfer", {})
        for k in transfer_vals:
            transfer_vals[k].append(t.get(k, 0))
    summary["transfer"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in transfer_vals.items()}

    # Validation
    hide_r1 = summary["hide_retrieval"]["i2t_r1"]["mean"]
    rand_r1 = summary["random_baseline"]["i2t_r1"]["mean"]
    summary["validation"] = {
        "hide_beats_random": hide_r1 > rand_r1,
        "positive_transfer": summary["transfer"]["r1"]["mean"] > 0,
    }

    _generate_figures(summary, figures_dir)
    return summary

def _generate_figures(summary, figures_dir):
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    # Figure 5a: Cross-modal recall curves
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ks = [1, 5, 10]
    for method, label, color in [("hide_retrieval", "HIDE", "#2196F3"), ("random_baseline", "Random Proj", "#9E9E9E")]:
        for direction, style in [("i2t", "-"), ("t2i", "--")]:
            vals = [summary[method][f"{direction}_r{k}"]["mean"] for k in ks]
            errs = [summary[method][f"{direction}_r{k}"]["std"] for k in ks]
            dlabel = "Image→Text" if direction == "i2t" else "Text→Image"
            ax.errorbar(ks, vals, yerr=errs, fmt=f"o{style}", color=color, label=f"{label} {dlabel}",
                       capsize=3, linewidth=2, markersize=6)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Cross-Modal Retrieval: HIDE vs Random")
    ax.legend(fontsize=8)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase4_cross_modal_recall.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase4_cross_modal_recall.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 5b: Transfer bar chart
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    transfer = summary.get("transfer", {})
    x = [1, 5, 10]
    vals = [transfer.get(f"r{k}", {}).get("mean", 0) for k in x]
    stds = [transfer.get(f"r{k}", {}).get("std", 0) for k in x]
    ax.bar(range(len(x)), vals, yerr=stds, color="#4CAF50", capsize=5, edgecolor="white")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([f"R@{k}" for k in x])
    ax.set_ylabel("Hit Rate")
    ax.set_title("Cross-Dataset Transfer: COCO→Flickr30k")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase4_transfer.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase4_transfer.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Phase 4 figures saved to {figures_dir}")
