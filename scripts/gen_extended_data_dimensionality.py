#!/usr/bin/env python3
"""
Extended Data Figure: Effective Dimensionality Analysis

4-panel figure showing:
(a) Eigenvalue spectrum (log y-axis) for all 3 embedding models
(b) Cumulative explained variance for all 3 models
(c) Interference exponent b vs effective dimensionality
(d) MiniLM vs BGE-large PCA d=64 interference comparison

Data sources:
- results/spectral/dimensionality_analysis.json
- results/spectral/summary.json
- results/interference/results_seed{42,123,456,789,1024}.json
"""

import os
import sys
import json
import numpy as np

# Add paper directory to path for figure_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (
    set_nature_style, COLORS, FULL_WIDTH, panel_label, save_figure
)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_dimensionality_data():
    """Load dimensionality analysis results."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "spectral", "dimensionality_analysis.json")
    with open(path) as f:
        return json.load(f)


def load_minilm_interference():
    """Load MiniLM interference summary."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "spectral", "summary.json")
    with open(path) as f:
        return json.load(f)


def load_bge_large_interference():
    """Load BGE-large interference results from v3 (all seeds)."""
    seeds = [42, 123, 456, 789, 1024]
    all_data = []
    for seed in seeds:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "interference", f"results_seed{seed}.json")
        with open(path) as f:
            all_data.append(json.load(f))
    return all_data


def compute_mean_eigenvalues(dim_data, model_key, n_components=50):
    """Compute mean eigenvalues across seeds for a model."""
    per_seed = dim_data[model_key]["per_seed"]
    # Truncate to n_components for plotting
    eig_arrays = []
    for seed_data in per_seed:
        eigs = np.array(seed_data["eigenvalues"][:n_components])
        eig_arrays.append(eigs)
    eig_matrix = np.array(eig_arrays)
    mean_eigs = np.mean(eig_matrix, axis=0)
    std_eigs = np.std(eig_matrix, axis=0)
    return mean_eigs, std_eigs


def compute_mean_cumvar(dim_data, model_key, n_components=50):
    """Compute mean cumulative variance across seeds for a model."""
    per_seed = dim_data[model_key]["per_seed"]
    cumvar_arrays = []
    for seed_data in per_seed:
        cumvar = np.array(seed_data["cumulative_variance"][:n_components])
        cumvar_arrays.append(cumvar)
    cumvar_matrix = np.array(cumvar_arrays)
    mean_cumvar = np.mean(cumvar_matrix, axis=0)
    std_cumvar = np.std(cumvar_matrix, axis=0)
    return mean_cumvar, std_cumvar


def compute_bge_max_b_per_dim(bge_data):
    """Compute max interference exponent b per PCA dimension across all near conditions and seeds."""
    dims = ["64", "128", "256", "1024"]
    near_keys = ["0", "5", "10", "20", "50", "100", "200"]

    results = {}
    for dim in dims:
        max_b_per_seed = []
        for seed_data in bge_data:
            seed_max = 0.0
            for near_k in near_keys:
                plf = seed_data["by_dim"][dim]["near"][near_k].get("power_law_fit", {})
                b = plf.get("b", 0.0)
                seed_max = max(seed_max, b)
            max_b_per_seed.append(seed_max)
        results[int(dim)] = {
            "mean": np.mean(max_b_per_seed),
            "std": np.std(max_b_per_seed),
            "values": max_b_per_seed,
        }
    return results


def compute_bge_d64_by_near(bge_data):
    """Compute BGE-large PCA d=64 interference b by near distractor count."""
    near_keys = ["0", "5", "10", "20", "50", "100", "200"]
    results = {}
    for near_k in near_keys:
        bs = []
        total_mem = None
        for seed_data in bge_data:
            entry = seed_data["by_dim"]["64"]["near"][near_k]
            plf = entry.get("power_law_fit", {})
            b = plf.get("b", 0.0)
            bs.append(b)
            total_mem = entry.get("n_total_memories", 0)
        results[near_k] = {
            "total_memories": total_mem,
            "mean_b": np.mean(bs),
            "std_b": np.std(bs),
            "values": bs,
        }
    return results


def main():
    set_nature_style()

    # Load all data
    dim_data = load_dimensionality_data()
    minilm_interf = load_minilm_interference()
    bge_data = load_bge_large_interference()

    # Model info
    models = {
        "MiniLM-L6-v2": {
            "color": COLORS["primary"],
            "label": "MiniLM (d=384)",
            "marker": "o",
        },
        "BGE-base-en-v1.5": {
            "color": COLORS["tertiary"],
            "label": "BGE-base (d=768)",
            "marker": "^",
        },
        "BGE-large-en-v1.5": {
            "color": COLORS["quaternary"],
            "label": "BGE-large (d=1024)",
            "marker": "s",
        },
    }

    n_components = 50  # Show first 50 components for visibility

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.75))

    # ================================================================
    # Panel (a): Eigenvalue spectrum (log y-axis)
    # ================================================================
    ax = axes[0, 0]
    for model_key, info in models.items():
        mean_eigs, std_eigs = compute_mean_eigenvalues(dim_data, model_key, n_components)
        x = np.arange(1, n_components + 1)
        ax.semilogy(x, mean_eigs, color=info["color"], label=info["label"],
                     linewidth=1.2, marker=info["marker"], markersize=2,
                     markevery=5)
        # Shaded error band (clip lower bound to avoid log-scale issues)
        upper = mean_eigs + std_eigs
        lower = np.clip(mean_eigs - std_eigs, 1e-20, None)
        ax.fill_between(x, lower, upper, color=info["color"], alpha=0.1)

    # Add vertical dashed lines at d_eff for each model
    for model_key, info in models.items():
        d_eff = dim_data[model_key]["summary"]["d_eff_mean"]
        ax.axvline(x=d_eff, color=info["color"], linestyle="--", linewidth=0.8,
                   alpha=0.7)

    # Set y-limits to focus on the informative range
    ax.set_ylim(1e-19, 1e-1)

    # Single annotation for d_eff region (they are all ~16)
    ax.annotate(r"$d_\mathrm{eff} \approx 16$",
                xy=(16, 1e-10),
                xytext=(30, 1e-6),
                fontsize=7, fontstyle="italic", color=COLORS["neutral"],
                arrowprops=dict(arrowstyle="->", color=COLORS["neutral"],
                                lw=0.8))

    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_xlim(0, n_components + 1)
    ax.legend(loc="upper right", fontsize=6)
    panel_label(ax, "a")

    # ================================================================
    # Panel (b): Cumulative explained variance
    # ================================================================
    ax = axes[0, 1]
    for model_key, info in models.items():
        mean_cumvar, std_cumvar = compute_mean_cumvar(dim_data, model_key, n_components)
        x = np.arange(1, n_components + 1)
        ax.plot(x, mean_cumvar, color=info["color"], label=info["label"],
                linewidth=1.2, marker=info["marker"], markersize=2,
                markevery=5)
        ax.fill_between(x, mean_cumvar - std_cumvar, mean_cumvar + std_cumvar,
                        color=info["color"], alpha=0.1)

    # Horizontal reference lines at 0.95 and 0.99
    ax.axhline(y=0.95, color=COLORS["neutral"], linestyle=":", linewidth=0.8,
               alpha=0.6)
    ax.axhline(y=0.99, color=COLORS["neutral"], linestyle=":", linewidth=0.8,
               alpha=0.6)
    ax.text(n_components + 0.5, 0.95, "95%", fontsize=6, va="center",
            color=COLORS["neutral"], fontstyle="italic", clip_on=False)
    ax.text(n_components + 0.5, 0.99, "99%", fontsize=6, va="center",
            color=COLORS["neutral"], fontstyle="italic", clip_on=False)

    # Annotate d_95 and d_99 (use MiniLM values as representative)
    d_95_mean = dim_data["MiniLM-L6-v2"]["summary"]["d_95_mean"]
    d_99_mean = dim_data["MiniLM-L6-v2"]["summary"]["d_99_mean"]
    ax.annotate(f"$d_{{95}}={d_95_mean:.0f}$", xy=(d_95_mean, 0.95),
                xytext=(d_95_mean + 8, 0.88), fontsize=7,
                color=COLORS["primary"],
                arrowprops=dict(arrowstyle="->", color=COLORS["primary"],
                                lw=0.8))
    ax.annotate(f"$d_{{99}}={d_99_mean:.0f}$", xy=(d_99_mean, 0.99),
                xytext=(d_99_mean + 8, 0.93), fontsize=7,
                color=COLORS["primary"],
                arrowprops=dict(arrowstyle="->", color=COLORS["primary"],
                                lw=0.8))

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xlim(0, n_components + 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=6)
    panel_label(ax, "b")

    # ================================================================
    # Panel (c): Interference vs effective dimensionality
    # ================================================================
    ax = axes[1, 0]

    # BGE-large at PCA dimensions
    bge_max_b = compute_bge_max_b_per_dim(bge_data)
    pca_dims = [64, 128, 256, 1024]
    pca_mean_b = [bge_max_b[d]["mean"] for d in pca_dims]
    pca_std_b = [bge_max_b[d]["std"] for d in pca_dims]

    ax.errorbar(pca_dims, pca_mean_b, yerr=pca_std_b,
                color=COLORS["quaternary"], marker="s", markersize=5,
                capsize=3, capthick=0.8, linewidth=1.2, linestyle="-",
                label="BGE-large (PCA)", zorder=5)

    # MiniLM at its effective dimensionality
    minilm_d_eff = dim_data["MiniLM-L6-v2"]["summary"]["d_eff_mean"]
    minilm_b_max = 0.678  # From summary: near=0 mean_b
    # Actually use the maximum valid b across near conditions
    minilm_near_data = minilm_interf["near"]
    minilm_valid_bs = []
    for near_k, entry in minilm_near_data.items():
        if entry["n_valid_seeds"] > 0:
            minilm_valid_bs.append(entry["mean_b"])
    minilm_b_max = max(minilm_valid_bs) if minilm_valid_bs else 0.678

    ax.plot(minilm_d_eff, minilm_b_max, marker="o", markersize=7,
            color=COLORS["primary"], zorder=6,
            label=f"MiniLM ($d_{{eff}}$={minilm_d_eff:.1f})")

    # Add error bar for MiniLM (use the b_std from near=10 which has max b)
    # Find which near condition gives max b
    for near_k, entry in minilm_near_data.items():
        if entry["n_valid_seeds"] > 0 and entry["mean_b"] == minilm_b_max:
            minilm_b_std = entry["b_std"]
            break
    ax.errorbar([minilm_d_eff], [minilm_b_max], yerr=[minilm_b_std],
                color=COLORS["primary"], capsize=3, capthick=0.8,
                linewidth=0, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Effective dimensionality")
    ax.set_ylabel("Max interference exponent $b$")
    ax.set_xlim(10, 2000)

    # Fix y limits
    y_max = max(minilm_b_max + minilm_b_std + 0.3, 0.5)
    ax.set_ylim(-0.02, y_max)

    # Biological range shaded region (after ylim is set)
    ax.axvspan(100, 500, color=COLORS["bg_shade"], alpha=0.4, zorder=0,
               label="Biological range")
    ax.text(220, y_max * 0.35, "Biological\nrange",
            fontsize=6, ha="center", va="center",
            color=COLORS["light_neutral"], fontstyle="italic")

    # Add connecting annotation showing the key insight
    ax.annotate("High $b$ despite\nnominal $d$=384",
                xy=(minilm_d_eff + 1, minilm_b_max - 0.1),
                xytext=(55, y_max * 0.85),
                fontsize=6, color=COLORS["primary"], fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color=COLORS["primary"],
                                lw=0.8))

    ax.legend(loc="upper right", fontsize=6)
    panel_label(ax, "c")

    # ================================================================
    # Panel (d): MiniLM vs BGE-large PCA d=64 interference comparison
    # ================================================================
    ax = axes[1, 1]

    # BGE-large PCA d=64 by near distractor count
    bge_d64 = compute_bge_d64_by_near(bge_data)
    near_keys_ordered = ["0", "5", "10", "20", "50", "100", "200"]

    bge_total_mem = []
    bge_mean_bs = []
    bge_std_bs = []
    for nk in near_keys_ordered:
        bge_total_mem.append(bge_d64[nk]["total_memories"])
        bge_mean_bs.append(bge_d64[nk]["mean_b"])
        bge_std_bs.append(bge_d64[nk]["std_b"])

    ax.errorbar(bge_total_mem, bge_mean_bs, yerr=bge_std_bs,
                color=COLORS["quaternary"], marker="s", markersize=4,
                capsize=2.5, capthick=0.8, linewidth=1.2, linestyle="-",
                label="BGE-large PCA $d$=64", zorder=5)

    # MiniLM native (d=384, d_eff~16) by near distractor count
    # Use total_distractors + 200 (targets) for comparable x-axis
    minilm_total_mem = []
    minilm_mean_bs = []
    minilm_std_bs = []
    minilm_ci_lo = []
    minilm_ci_hi = []
    for nk in near_keys_ordered:
        entry = minilm_interf["near"][nk]
        # Only plot conditions with valid data (n_valid_seeds > 0)
        if entry["n_valid_seeds"] > 0:
            total_mem = entry["total_distractors"] + 200  # add 200 targets
            minilm_total_mem.append(total_mem)
            minilm_mean_bs.append(entry["mean_b"])
            minilm_std_bs.append(entry["b_std"])

    ax.errorbar(minilm_total_mem, minilm_mean_bs, yerr=minilm_std_bs,
                color=COLORS["primary"], marker="o", markersize=4,
                capsize=2.5, capthick=0.8, linewidth=1.2, linestyle="-",
                label="MiniLM native $d$=384\n($d_{eff}$=16)", zorder=5)

    # Mark collapsed conditions (n_valid=0) with downward arrows
    collapse_mems = []
    for nk in near_keys_ordered:
        entry = minilm_interf["near"][nk]
        if entry["n_valid_seeds"] == 0 and entry["total_distractors"] > 0:
            total_mem = entry["total_distractors"] + 200
            collapse_mems.append(total_mem)

    ax.set_xscale("log")
    ax.set_xlabel("Total memories")
    ax.set_ylabel("Interference exponent $b$")

    # Set y limits before drawing collapse markers
    all_bs = minilm_mean_bs + bge_mean_bs
    y_max_d = max(all_bs) + max(minilm_std_bs + bge_std_bs) + 0.2
    ax.set_ylim(-0.08, min(y_max_d, 2.5))

    if collapse_mems:
        for i, cm in enumerate(collapse_mems):
            ax.annotate("", xy=(cm, -0.02), xytext=(cm, 0.20),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS["secondary"],
                                        lw=1.0, mutation_scale=8))
            ax.plot(cm, 0.0, marker="x", markersize=5, color=COLORS["secondary"],
                    markeredgewidth=1.5, zorder=6)
        # Single label for all collapse markers
        ax.text(collapse_mems[len(collapse_mems)//2], 0.28,
                "MiniLM collapse\n(retention = 0)",
                fontsize=6, color=COLORS["secondary"], ha="center",
                fontstyle="italic")

    ax.legend(loc="upper left", fontsize=6)
    panel_label(ax, "d")

    # Adjust layout
    fig.tight_layout(w_pad=2.5, h_pad=2.5)

    # Save
    save_figure(fig, "extended_data_dimensionality")
    print("Done.")


if __name__ == "__main__":
    main()
