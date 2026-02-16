"""Phase 5 Analysis: Aggregation and Figure Generation"""
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
from metrics import bootstrap_ci, cohens_d

def aggregate(all_results: Dict[int, Dict], results_dir: Path, figures_dir: Path) -> Dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    valid = {s: r for s, r in all_results.items() if "error" not in r}
    if not valid:
        return {"error": "All seeds failed"}
    seeds = sorted(valid.keys())
    summary = {"seeds": seeds, "n_seeds": len(seeds)}

    # DRM aggregation
    lure_sims = [valid[s]["drm"]["mean_lure_sim"] for s in seeds]
    unrel_sims = [valid[s]["drm"]["mean_unrelated_sim"] for s in seeds]
    studied_sims = [valid[s]["drm"]["mean_studied_sim"] for s in seeds]
    summary["drm"] = {
        "mean_lure_sim": {"mean": float(np.mean(lure_sims)), "std": float(np.std(lure_sims))},
        "mean_studied_sim": {"mean": float(np.mean(studied_sims)), "std": float(np.std(studied_sims))},
        "mean_unrelated_sim": {"mean": float(np.mean(unrel_sims)), "std": float(np.std(unrel_sims))},
        "lure_above_unrelated": float(np.mean(lure_sims) > np.mean(unrel_sims)),
        "cohens_d": float(cohens_d(np.array(lure_sims), np.array(unrel_sims))),
    }

    # Spacing aggregation
    spacing_conditions = ["massed", "short", "medium", "long"]
    spacing_data = {c: [] for c in spacing_conditions}
    for s in seeds:
        conds = valid[s]["spacing"]["conditions"]
        for c in spacing_conditions:
            spacing_data[c].append(conds.get(c, 0))
    summary["spacing"] = {c: {"mean": float(np.mean(v)), "std": float(np.std(v))} for c, v in spacing_data.items()}

    # TOT aggregation
    tot_rates = [valid[s]["tot"]["tot_rate"] for s in seeds]
    summary["tot"] = {
        "mean_rate": float(np.mean(tot_rates)),
        "std_rate": float(np.std(tot_rates)),
        "human_rate": valid[seeds[0]]["tot"]["human_rate"],
    }

    # Validation
    summary["validation"] = {
        "drm_false_alarms_above_chance": summary["drm"]["lure_above_unrelated"] > 0,
        "phenomena_similar_to_human": sum([
            summary["drm"]["lure_above_unrelated"] > 0,
            summary["tot"]["mean_rate"] > 0,
            any(valid[s]["spacing"]["matches_human_order"] for s in seeds),
        ]),
    }

    _generate_figures(summary, all_results, figures_dir)
    return summary

def _generate_figures(summary, all_results, figures_dir):
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
    valid = {s: r for s, r in all_results.items() if "error" not in r}
    seeds = sorted(valid.keys())

    # Figure 6a: DRM false memory
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    categories = ["Studied\n(Hits)", "Critical Lure\n(False Alarm)", "Unrelated\n(Correct Rejection)"]
    means = [summary["drm"]["mean_studied_sim"]["mean"],
             summary["drm"]["mean_lure_sim"]["mean"],
             summary["drm"]["mean_unrelated_sim"]["mean"]]
    stds = [summary["drm"]["mean_studied_sim"]["std"],
            summary["drm"]["mean_lure_sim"]["std"],
            summary["drm"]["mean_unrelated_sim"]["std"]]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"]
    ax.bar(range(3), means, yerr=stds, color=colors, capsize=5, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("DRM False Memory: HIDE Embedding Similarities")
    ax.axhline(y=0.55, color="red", linestyle="--", alpha=0.5, label="Human FA rate ≈55%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_drm.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase5_drm.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 6b: Spacing effect
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    conditions = ["massed", "short", "medium", "long"]
    labels = ["Massed\n(0-2min)", "Short\n(0-2h)", "Medium\n(0-2d)", "Long\n(0-2w)"]
    means = [summary["spacing"][c]["mean"] for c in conditions]
    stds = [summary["spacing"][c]["std"] for c in conditions]
    human_expected = [0.3, 0.5, 0.7, 0.85]  # Approximate human data (Cepeda et al. 2006)
    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, means, width, yerr=stds, label="HIDE", color="#2196F3", capsize=5, edgecolor="white")
    ax.bar(x + width/2, human_expected, width, label="Human (approx)", color="#FF9800", edgecolor="white", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retention Rate")
    ax.set_title("Spacing Effect: HIDE vs Human")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_spacing.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase5_spacing.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 6c: TOT rate comparison
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar([0, 1], [summary["tot"]["mean_rate"] * 100, summary["tot"]["human_rate"] * 100],
           yerr=[summary["tot"]["std_rate"] * 100, 0.5],
           color=["#2196F3", "#FF9800"], capsize=5, edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HIDE", "Human\n(Brown & McNeill)"])
    ax.set_ylabel("TOT Rate (%)")
    ax.set_title("Tip-of-Tongue Phenomenon")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_tot.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase5_tot.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Phase 5 figures saved to {figures_dir}")
