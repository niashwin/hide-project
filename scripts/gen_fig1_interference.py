#!/usr/bin/env python3
"""
Generate Figure 1: Headline interference figure for HIDE v4 paper.

Three-panel figure showing:
  (a) Forgetting curves fanning out with increasing near-distractor load at d=64
  (b) Dose-response: fitted exponent b vs number of competing memories
  (c) Dimensionality kills interference: max b vs effective dimensionality
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Add paper/ to path so we can import figure_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (
    set_nature_style, COLORS, BLUES_GRADIENT, FULL_WIDTH,
    panel_label, human_reference_line, save_figure,
)

set_nature_style()

# ---------------------------------------------------------------------------
# 1. Load data from all 5 seeds
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'interference')

all_data = {}
for seed in SEEDS:
    fpath = os.path.join(DATA_DIR, f'results_seed{seed}.json')
    with open(fpath) as f:
        all_data[seed] = json.load(f)

# ---------------------------------------------------------------------------
# 2. Helper: power-law model  R(t) = a * t^(-b)
# ---------------------------------------------------------------------------
def power_law(t, a, b):
    return a * np.power(t, -b)

# ---------------------------------------------------------------------------
# 3. Panel (a) — Forgetting curves at d=64 for selected near conditions
# ---------------------------------------------------------------------------
# We select 5 representative near-distractor-per-target counts
SELECTED_NEAR = ['0', '5', '20', '50', '200']
# Corresponding total memory counts (n_targets=200):
#   0 -> 200, 5 -> 1K, 20 -> 4.2K, 50 -> 10.2K, 200 -> 40.2K
NEAR_LABELS = {
    '0':   '0',
    '5':   '1K',
    '20':  '4K',
    '50':  '10K',
    '200': '40K',
}

# Collect per-seed raw (ages, retentions) at d=64.
# Ages differ across seeds, so we store them separately and interpolate to a
# common grid for averaging.
raw_by_cond = {k: [] for k in SELECTED_NEAR}  # list of (ages_arr, rets_arr) per seed

for seed in SEEDS:
    dim_data = all_data[seed]['by_dim']['64']['near']
    for cond in SELECTED_NEAR:
        cond_data = dim_data[cond]
        ages = np.array(cond_data['ages'])
        rets = np.array(cond_data['retentions'])
        raw_by_cond[cond].append((ages, rets))

# Build a common age grid (union of all unique ages, sorted)
all_ages_set = set()
for seed in SEEDS:
    dim_data = all_data[seed]['by_dim']['64']['near']
    ages = dim_data['0']['ages']  # ages are the same across conditions within a seed
    all_ages_set.update(ages)
common_ages = np.sort(np.array(list(all_ages_set)))

# Interpolate each seed's retentions to the common grid; use linear interp
# and clamp to [0,1]. For ages outside a seed's range, extrapolate with
# the boundary value (fill_value uses edge values).
mean_retentions = {}
std_retentions = {}
for cond in SELECTED_NEAR:
    interped = []
    for ages_s, rets_s in raw_by_cond[cond]:
        fn = interp1d(ages_s, rets_s, kind='linear', fill_value='extrapolate')
        interped.append(np.clip(fn(common_ages), 0, 1))
    arr = np.array(interped)  # (5, n_common_ages)
    mean_retentions[cond] = arr.mean(axis=0)
    std_retentions[cond] = arr.std(axis=0)

# Fit power-law to the mean retention curves for smooth plotting
fitted_params = {}
for cond in SELECTED_NEAR:
    mean_r = mean_retentions[cond]
    # Only fit if there is actual variation (not all ~1.0)
    if np.all(mean_r >= 0.999):
        fitted_params[cond] = None  # flat line at 1.0
    else:
        try:
            popt, _ = curve_fit(power_law, common_ages, mean_r,
                                p0=[1.0, 0.05], maxfev=5000,
                                bounds=([0.5, 0.0], [2.0, 2.0]))
            fitted_params[cond] = popt
        except Exception:
            fitted_params[cond] = None

# Human Ebbinghaus data
human_times_days = np.array([0.014, 0.042, 0.333, 1.0, 2.0, 6.0, 31.0])
human_retention = np.array([0.58, 0.44, 0.36, 0.34, 0.28, 0.25, 0.21])

# Fit power-law to human data for a smooth curve
human_popt, _ = curve_fit(power_law, human_times_days, human_retention,
                          p0=[0.5, 0.5], maxfev=5000,
                          bounds=([0.01, 0.01], [2.0, 2.0]))

# ---------------------------------------------------------------------------
# 4. Panel (b) — Dose-response: b vs number of competing memories
# ---------------------------------------------------------------------------
# Collect b values across all near conditions at d=64 and d=128
ALL_NEAR_CONDS = ['0', '5', '10', '20', '50', '100', '200']

def get_b_values(dim_str):
    """Return (n_total_memories list, mean_b list, ci_lo list, ci_hi list)."""
    n_mems = []
    mean_bs = []
    ci_los = []
    ci_his = []
    for cond in ALL_NEAR_CONDS:
        bs = []
        n_mem = None
        for seed in SEEDS:
            cond_data = all_data[seed]['by_dim'][dim_str]['near'][cond]
            if n_mem is None:
                n_mem = cond_data['n_total_memories']
            plf = cond_data.get('power_law_fit', {})
            if plf and 'b' in plf:
                bs.append(plf['b'])
            else:
                bs.append(0.0)  # perfect retention => b=0
        bs = np.array(bs)
        n_mems.append(n_mem)
        mean_bs.append(bs.mean())
        # Bootstrap 95% CI
        rng = np.random.RandomState(42)
        boot_means = [rng.choice(bs, size=len(bs), replace=True).mean()
                      for _ in range(10000)]
        ci_los.append(np.percentile(boot_means, 2.5))
        ci_his.append(np.percentile(boot_means, 97.5))
    return np.array(n_mems), np.array(mean_bs), np.array(ci_los), np.array(ci_his)

n_mems_64, mean_b_64, ci_lo_64, ci_hi_64 = get_b_values('64')
n_mems_128, mean_b_128, ci_lo_128, ci_hi_128 = get_b_values('128')

# ---------------------------------------------------------------------------
# 5. Panel (c) — Max b vs dimensionality
# ---------------------------------------------------------------------------
DIMS = ['64', '128', '256', '1024']
DIM_INTS = [64, 128, 256, 1024]

def get_max_b_per_dim(dim_str):
    """For each seed, find the maximum b across all near conditions. Return array of 5."""
    max_bs = []
    for seed in SEEDS:
        dim_data = all_data[seed]['by_dim'][dim_str]['near']
        seed_bs = []
        for cond in ALL_NEAR_CONDS:
            plf = dim_data[cond].get('power_law_fit', {})
            if plf and 'b' in plf:
                seed_bs.append(plf['b'])
            else:
                seed_bs.append(0.0)
        max_bs.append(max(seed_bs))
    return np.array(max_bs)

max_b_by_dim = {}
for d_str in DIMS:
    max_b_by_dim[d_str] = get_max_b_per_dim(d_str)

# Bootstrap CI for max b
max_b_means = []
max_b_ci_lo = []
max_b_ci_hi = []
for d_str in DIMS:
    vals = max_b_by_dim[d_str]
    max_b_means.append(vals.mean())
    rng = np.random.RandomState(42)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(10000)]
    max_b_ci_lo.append(np.percentile(boots, 2.5))
    max_b_ci_hi.append(np.percentile(boots, 97.5))

max_b_means = np.array(max_b_means)
max_b_ci_lo = np.array(max_b_ci_lo)
max_b_ci_hi = np.array(max_b_ci_hi)

# ---------------------------------------------------------------------------
# 6. Build the 3-panel figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.48))  # ~7.09 x 3.4 in

# GridSpec: left 55% for panel a, right 45% split top/bottom for b and c
gs = gridspec.GridSpec(2, 2, width_ratios=[60, 40], height_ratios=[1, 1],
                       hspace=0.45, wspace=0.35)

ax_a = fig.add_subplot(gs[:, 0])   # panel a spans both rows on left
ax_b = fig.add_subplot(gs[0, 1])   # panel b top right
ax_c = fig.add_subplot(gs[1, 1])   # panel c bottom right

# ---- Panel (a): Forgetting curves ----
t_smooth = np.linspace(common_ages.min() * 0.8, common_ages.max() * 1.2, 300)

for i, cond in enumerate(SELECTED_NEAR):
    color = BLUES_GRADIENT[i]
    label = f'{NEAR_LABELS[cond]} competitors'

    # Plot raw data points (all 5 seeds at their own ages, low alpha)
    for seed_idx in range(len(SEEDS)):
        ages_s, rets_s = raw_by_cond[cond][seed_idx]
        ax_a.scatter(ages_s, rets_s, color=color, s=8, alpha=0.20,
                     zorder=2, edgecolors='none')

    # Plot mean data points on common grid
    ax_a.scatter(common_ages, mean_retentions[cond], color=color, s=18,
                 alpha=0.6, zorder=3, edgecolors='none')

    # Plot fitted curve (or flat line)
    if fitted_params[cond] is not None:
        a_fit, b_fit = fitted_params[cond]
        y_smooth = power_law(t_smooth, a_fit, b_fit)
        y_smooth = np.clip(y_smooth, 0, 1.05)
        ax_a.plot(t_smooth, y_smooth, color=color, linewidth=1.6, label=label,
                  zorder=4)
    else:
        ax_a.axhline(y=1.0, color=color, linewidth=1.0, linestyle='-',
                     label=label, zorder=4, alpha=0.7)

# Human Ebbinghaus curve (dashed red)
t_human_smooth = np.linspace(human_times_days.min() * 0.8,
                             max(human_times_days.max(), common_ages.max()) * 1.2, 300)
y_human_smooth = power_law(t_human_smooth, *human_popt)
ax_a.plot(t_human_smooth, y_human_smooth, color=COLORS['human'], linewidth=2.0,
          linestyle='--', label='Human (Ebbinghaus)', zorder=5)
ax_a.scatter(human_times_days, human_retention, color=COLORS['human'],
             marker='D', s=20, zorder=6, edgecolors='none')

ax_a.set_xscale('log')
ax_a.set_xlabel('Memory age (days)')
ax_a.set_ylabel('Retrieval accuracy')
ax_a.set_ylim(-0.02, 1.08)
ax_a.set_xlim(0.5, 40)
ax_a.legend(loc='lower left', fontsize=6.5, frameon=False, labelspacing=0.35)
panel_label(ax_a, 'a')

# ---- Panel (b): Dose-response b vs competing memories ----
ax_b.plot(n_mems_64, mean_b_64, color=COLORS['primary'], marker='o',
          markersize=4, linewidth=1.2, label='d = 64', zorder=4)
ax_b.fill_between(n_mems_64, ci_lo_64, ci_hi_64,
                  color=COLORS['primary'], alpha=0.18, zorder=2)

ax_b.plot(n_mems_128, mean_b_128, color=COLORS['light_neutral'], marker='s',
          markersize=3.5, linewidth=1.0, label='d = 128', zorder=3)
ax_b.fill_between(n_mems_128, ci_lo_128, ci_hi_128,
                  color=COLORS['light_neutral'], alpha=0.15, zorder=1)

# Human reference line at b=0.5
human_reference_line(ax_b, 0.5, label='Human (b\u22480.5)')

ax_b.set_xscale('log')
ax_b.set_xlabel('Number of competing memories')
ax_b.set_ylabel('Fitted exponent b')
ax_b.set_ylim(-0.02, 0.60)
ax_b.legend(loc='upper left', fontsize=6.5, frameon=False)
panel_label(ax_b, 'b')

# ---- Panel (c): Dimensionality kills interference ----
ax_c.errorbar(DIM_INTS, max_b_means,
              yerr=[max_b_means - max_b_ci_lo, max_b_ci_hi - max_b_means],
              fmt='o', color=COLORS['primary'], markersize=5, capsize=3,
              linewidth=1.2, zorder=4)
ax_c.plot(DIM_INTS, max_b_means, color=COLORS['primary'], linewidth=1.0,
          alpha=0.5, zorder=3)

# Shade "biological range" d=100-500
ax_c.axvspan(100, 500, color=COLORS['bg_shade'], alpha=0.6, zorder=0,
             label='Biological range')

ax_c.set_xscale('log')
ax_c.set_xlabel('Effective dimensionality')
ax_c.set_ylabel('Maximum exponent b')
ax_c.set_xticks(DIM_INTS)
ax_c.set_xticklabels(['64', '128', '256', '1024'])
ax_c.set_ylim(-0.01, max(max_b_means[0] * 1.35, 0.78))

# MiniLM effective dimensionality marker
ax_c.scatter([15.7], [0.678], marker='D', s=40, color=COLORS['highlight'],
             zorder=6, edgecolors='white', linewidths=0.5)
ax_c.annotate('MiniLM\n($d$=384, $d_{eff}$\u224816)', xy=(15.7, 0.678),
              xytext=(30, 0.72), fontsize=6, color=COLORS['highlight'],
              arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=0.8),
              ha='left', va='bottom')
human_reference_line(ax_c, 0.5, label='Human (b\u22480.5)')

ax_c.set_xlim(10, 1500)
ax_c.legend(loc='upper right', fontsize=6.5, frameon=False)
panel_label(ax_c, 'c')

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
save_figure(fig, 'fig1_interference')
print("Figure 1 generation complete.")
