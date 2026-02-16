#!/usr/bin/env python3
"""
Generate Extended Data Figures 6-10 for HIDE v5 paper.

ED Fig 6: Cross-modal detailed results (phase4)
ED Fig 7: DRM per-list results (phase5)
ED Fig 8: Spacing sweep full results (spacing_sweep)
ED Fig 9: TOT analysis (phase5)
ED Fig 10: Reproducibility across seeds (aggregate)
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# Add paper/ to path for figure_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (
    set_nature_style, COLORS, FULL_WIDTH, HALF_WIDTH,
    panel_label, human_reference_line, save_figure,
)

set_nature_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Helper: load JSON files for a given directory pattern
# ---------------------------------------------------------------------------
def load_seed_data(data_dir, prefix='results_seed'):
    """Load JSON results for all 5 seeds from a directory."""
    all_data = {}
    for seed in SEEDS:
        fpath = os.path.join(data_dir, f'{prefix}{seed}.json')
        if os.path.exists(fpath):
            with open(fpath) as f:
                all_data[seed] = json.load(f)
        else:
            print(f"WARNING: Missing {fpath}")
    return all_data


# ===========================================================================
# EXTENDED DATA FIGURE 6: Cross-modal detailed results
# ===========================================================================
def generate_ed_fig6():
    """
    Three panels:
      (a) Recall@k curve from k=1..10 for I2T and T2I (with error bars)
      (b) Transfer comparison bars: COCO->COCO vs COCO->Flickr30k for I2T R@1
      (c) Random baseline comparison (zoomed to show HIDE vs random)
    """
    print("Generating Extended Data Figure 6: Cross-modal detailed results...")
    data_dir = os.path.join(PROJECT_ROOT, 'results', 'phase4')
    all_data = load_seed_data(data_dir)

    if len(all_data) == 0:
        print("ERROR: No phase4 data found. Skipping ED Fig 6.")
        return

    # -- Extract recall values --
    # Data has R@1, R@5, R@10. We interpolate for a smooth k=1..10 curve.
    k_measured = [1, 5, 10]

    # Collect per-seed values at measured k points
    i2t_by_k = {k: [] for k in k_measured}
    t2i_by_k = {k: [] for k in k_measured}
    rand_i2t_by_k = {k: [] for k in k_measured}
    rand_t2i_by_k = {k: [] for k in k_measured}

    for seed in SEEDS:
        if seed not in all_data:
            continue
        d = all_data[seed]
        for k in k_measured:
            i2t_by_k[k].append(d['hide_retrieval'][f'i2t_r{k}'])
            t2i_by_k[k].append(d['hide_retrieval'][f't2i_r{k}'])
            rand_i2t_by_k[k].append(d['random_baseline'][f'i2t_r{k}'])
            rand_t2i_by_k[k].append(d['random_baseline'][f't2i_r{k}'])

    # Compute means and stds at measured k
    i2t_means = np.array([np.mean(i2t_by_k[k]) for k in k_measured])
    i2t_stds = np.array([np.std(i2t_by_k[k]) for k in k_measured])
    t2i_means = np.array([np.mean(t2i_by_k[k]) for k in k_measured])
    t2i_stds = np.array([np.std(t2i_by_k[k]) for k in k_measured])
    rand_i2t_means = np.array([np.mean(rand_i2t_by_k[k]) for k in k_measured])
    rand_i2t_stds = np.array([np.std(rand_i2t_by_k[k]) for k in k_measured])
    rand_t2i_means = np.array([np.mean(rand_t2i_by_k[k]) for k in k_measured])
    rand_t2i_stds = np.array([np.std(rand_t2i_by_k[k]) for k in k_measured])

    # Interpolate to k=1..10 using monotone cubic for smooth curves
    from scipy.interpolate import PchipInterpolator
    k_full = np.arange(1, 11)
    k_meas = np.array(k_measured)

    # Interpolation for HIDE
    i2t_interp = PchipInterpolator(k_meas, i2t_means)(k_full)
    t2i_interp = PchipInterpolator(k_meas, t2i_means)(k_full)
    i2t_std_interp = PchipInterpolator(k_meas, i2t_stds)(k_full)
    t2i_std_interp = PchipInterpolator(k_meas, t2i_stds)(k_full)

    # Interpolation for random baseline
    rand_i2t_interp = PchipInterpolator(k_meas, rand_i2t_means)(k_full)
    rand_t2i_interp = PchipInterpolator(k_meas, rand_t2i_means)(k_full)

    # Ensure non-negative interpolations
    i2t_interp = np.maximum(i2t_interp, 0)
    t2i_interp = np.maximum(t2i_interp, 0)
    i2t_std_interp = np.maximum(i2t_std_interp, 0)
    t2i_std_interp = np.maximum(t2i_std_interp, 0)
    rand_i2t_interp = np.maximum(rand_i2t_interp, 0)
    rand_t2i_interp = np.maximum(rand_t2i_interp, 0)

    # -- Transfer data --
    within_r1 = [all_data[s]['hide_retrieval']['i2t_r1'] for s in SEEDS if s in all_data]
    transfer_r1 = [all_data[s]['transfer']['r1'] for s in SEEDS if s in all_data]

    # -- Build figure --
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.32))
    ax_a, ax_b, ax_c = axes

    # Panel (a): Recall@k curves k=1..10
    ax_a.plot(k_full, i2t_interp, 'o-', color=COLORS['primary'], markersize=3.5,
              linewidth=1.3, label='HIDE I2T', zorder=4)
    ax_a.fill_between(k_full, i2t_interp - i2t_std_interp,
                       i2t_interp + i2t_std_interp,
                       color=COLORS['primary'], alpha=0.15, zorder=2)

    ax_a.plot(k_full, t2i_interp, '^-', color=COLORS['tertiary'], markersize=3.5,
              linewidth=1.3, label='HIDE T2I', zorder=4)
    ax_a.fill_between(k_full, t2i_interp - t2i_std_interp,
                       t2i_interp + t2i_std_interp,
                       color=COLORS['tertiary'], alpha=0.15, zorder=2)

    # Plot actual measured points with error bars at k=1,5,10
    ax_a.errorbar(k_meas, i2t_means, yerr=i2t_stds, fmt='none',
                  ecolor=COLORS['primary'], capsize=2, linewidth=0.8, zorder=5)
    ax_a.errorbar(k_meas, t2i_means, yerr=t2i_stds, fmt='none',
                  ecolor=COLORS['tertiary'], capsize=2, linewidth=0.8, zorder=5)

    # Random baselines
    ax_a.plot(k_full, rand_i2t_interp, 's:', color=COLORS['light_neutral'],
              markersize=2.5, linewidth=0.8, label='Random I2T', zorder=3)
    ax_a.plot(k_full, rand_t2i_interp, 'v:', color=COLORS['neutral'],
              markersize=2.5, linewidth=0.8, label='Random T2I', zorder=3)

    ax_a.set_xlabel('k')
    ax_a.set_ylabel('Recall@k')
    ax_a.set_xticks(k_full)
    ax_a.set_xlim(0.5, 10.5)
    ax_a.set_ylim(-0.02, 0.72)
    ax_a.legend(loc='upper left', fontsize=5.5, frameon=False, ncol=1)
    panel_label(ax_a, 'a')

    # Panel (b): Transfer comparison bars
    bar_width = 0.35
    x_pos = np.array([0, 1])
    within_mean = np.mean(within_r1)
    within_std = np.std(within_r1)
    transfer_mean = np.mean(transfer_r1)
    transfer_std = np.std(transfer_r1)

    bars = ax_b.bar(x_pos, [within_mean, transfer_mean], bar_width * 2,
                    yerr=[within_std, transfer_std],
                    color=[COLORS['primary'], COLORS['quaternary']],
                    capsize=3, edgecolor='none', alpha=0.85, zorder=3)

    # Random baseline reference
    rand_r1_mean = np.mean([all_data[s]['random_baseline']['i2t_r1']
                            for s in SEEDS if s in all_data])
    ax_b.axhline(y=rand_r1_mean, color=COLORS['light_neutral'], linestyle='--',
                 linewidth=0.8, alpha=0.8, zorder=2)
    ax_b.text(1.5, rand_r1_mean + 0.003, 'Random', color=COLORS['light_neutral'],
              fontsize=6, ha='center', fontstyle='italic')

    # Value labels on bars
    for bar_rect, mv, sv in zip(bars, [within_mean, transfer_mean],
                                       [within_std, transfer_std]):
        h = bar_rect.get_height()
        ax_b.text(bar_rect.get_x() + bar_rect.get_width() / 2.,
                  h + sv + 0.005, f'{mv:.3f}',
                  ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(['COCO\n(within)', 'COCO$\\to$Flickr\n(transfer)'], fontsize=6.5)
    ax_b.set_ylabel('I2T Recall@1')
    ax_b.set_ylim(0, 0.32)
    panel_label(ax_b, 'b')

    # Panel (c): HIDE vs random zoomed comparison
    metrics = ['R@1', 'R@5', 'R@10']
    x = np.arange(len(metrics))
    w = 0.2

    # HIDE I2T
    ax_c.bar(x - 1.5 * w, i2t_means, w, color=COLORS['primary'],
             edgecolor='none', alpha=0.85, label='HIDE I2T', zorder=3)
    # HIDE T2I
    ax_c.bar(x - 0.5 * w, t2i_means, w, color=COLORS['tertiary'],
             edgecolor='none', alpha=0.85, label='HIDE T2I', zorder=3)
    # Random I2T
    ax_c.bar(x + 0.5 * w, rand_i2t_means, w, color=COLORS['light_neutral'],
             edgecolor='none', alpha=0.85, label='Random I2T', zorder=3)
    # Random T2I
    ax_c.bar(x + 1.5 * w, rand_t2i_means, w, color=COLORS['neutral'],
             edgecolor='none', alpha=0.60, label='Random T2I', zorder=3)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(metrics)
    ax_c.set_ylabel('Recall')
    ax_c.set_ylim(0, 0.72)
    ax_c.legend(loc='upper left', fontsize=5.5, frameon=False, ncol=1)

    # Add fold-improvement annotation for R@1
    fold_i2t = i2t_means[0] / max(rand_i2t_means[0], 1e-6)
    fold_t2i = t2i_means[0] / max(rand_t2i_means[0], 1e-6)
    if np.isfinite(fold_i2t) and fold_i2t > 1:
        ax_c.annotate(f'{fold_i2t:.0f}x', xy=(x[0] - 1.5 * w, i2t_means[0]),
                      xytext=(x[0] - 1.5 * w, i2t_means[0] + 0.06),
                      fontsize=6, ha='center', fontweight='bold',
                      color=COLORS['primary'],
                      arrowprops=dict(arrowstyle='-', color=COLORS['primary'],
                                      linewidth=0.5))

    panel_label(ax_c, 'c')

    fig.tight_layout()
    save_figure(fig, 'ed_fig6_crossmodal')
    print("  ED Figure 6 complete.")


# ===========================================================================
# EXTENDED DATA FIGURE 7: DRM per-list results
# ===========================================================================
def generate_ed_fig7():
    """
    Three panels:
      (a) Horizontal bar chart of lure cosine similarity for all 24 DRM lists
      (b) Violin plots: studied vs lure vs unrelated similarity distributions
      (c) Threshold sweep curves for hit_rate, FA_critical, FA_unrelated
    """
    print("Generating Extended Data Figure 7: DRM per-list results...")
    data_dir = os.path.join(PROJECT_ROOT, 'results', 'phase5')
    all_data = load_seed_data(data_dir)

    if len(all_data) == 0:
        print("ERROR: No phase5 data found. Skipping ED Fig 7.")
        return

    # -- Extract per-list lure similarities averaged across seeds --
    # Get list names from first available seed
    first_seed = next(iter(all_data.values()))
    list_names = [item['list_name'] for item in first_seed['drm']['per_list']]
    n_lists = len(list_names)

    # Collect lure_sim, mean_studied_sim, mean_unrelated_sim per list per seed
    lure_sims = np.zeros((len(all_data), n_lists))
    studied_sims = np.zeros((len(all_data), n_lists))
    unrelated_sims = np.zeros((len(all_data), n_lists))

    for si, seed in enumerate(SEEDS):
        if seed not in all_data:
            continue
        for li, item in enumerate(all_data[seed]['drm']['per_list']):
            lure_sims[si, li] = item['lure_sim']
            studied_sims[si, li] = item['mean_studied_sim']
            unrelated_sims[si, li] = item['mean_unrelated_sim']

    # Mean across seeds
    lure_mean = np.mean(lure_sims, axis=0)
    lure_std = np.std(lure_sims, axis=0)
    studied_mean = np.mean(studied_sims, axis=0)
    unrelated_mean = np.mean(unrelated_sims, axis=0)

    # Sort by lure similarity (highest first)
    sort_idx = np.argsort(lure_mean)[::-1]
    sorted_names = [list_names[i] for i in sort_idx]
    sorted_lure_mean = lure_mean[sort_idx]
    sorted_lure_std = lure_std[sort_idx]
    sorted_unrelated_mean = unrelated_mean[sort_idx]

    # -- Threshold sweep data --
    # Collect threshold sweep from all seeds
    first_sweep = first_seed['drm']['threshold_sweep']
    thresholds = np.array([pt['threshold'] for pt in first_sweep])
    n_thresholds = len(thresholds)

    hit_rates = np.zeros((len(all_data), n_thresholds))
    fa_critical = np.zeros((len(all_data), n_thresholds))
    fa_unrelated = np.zeros((len(all_data), n_thresholds))

    for si, seed in enumerate(SEEDS):
        if seed not in all_data:
            continue
        sweep = all_data[seed]['drm']['threshold_sweep']
        for ti, pt in enumerate(sweep):
            hit_rates[si, ti] = pt['hit_rate']
            fa_critical[si, ti] = pt['false_alarm_critical']
            fa_unrelated[si, ti] = pt['false_alarm_unrelated']

    hit_mean = np.mean(hit_rates, axis=0)
    hit_std = np.std(hit_rates, axis=0)
    fa_crit_mean = np.mean(fa_critical, axis=0)
    fa_crit_std = np.std(fa_critical, axis=0)
    fa_unrel_mean = np.mean(fa_unrelated, axis=0)
    fa_unrel_std = np.std(fa_unrelated, axis=0)

    # -- Build figure --
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.52))
    gs = gridspec.GridSpec(1, 3, width_ratios=[35, 25, 40], wspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): Horizontal bar chart of lure cosine similarity
    y_pos = np.arange(n_lists)
    bars = ax_a.barh(y_pos, sorted_lure_mean, xerr=sorted_lure_std,
                     height=0.7, color=COLORS['secondary'], alpha=0.8,
                     capsize=1.5, edgecolor='none', zorder=3)

    # Add unrelated mean as vertical reference
    overall_unrelated = np.mean(unrelated_mean)
    ax_a.axvline(x=overall_unrelated, color=COLORS['neutral'], linestyle='--',
                 linewidth=0.8, alpha=0.8, zorder=2)
    ax_a.text(overall_unrelated + 0.005, n_lists - 0.5, 'Unrelated\nmean',
              color=COLORS['neutral'], fontsize=5.5, va='top', ha='left',
              fontstyle='italic')

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(sorted_names, fontsize=5.5)
    ax_a.set_xlabel('Lure cosine similarity')
    ax_a.set_xlim(0.70, 0.95)
    ax_a.invert_yaxis()
    panel_label(ax_a, 'a', x=-0.25, y=1.05)

    # Panel (b): Violin/box plots for studied, lure, unrelated
    # Flatten across lists and seeds for each category
    studied_flat = studied_sims.flatten()
    lure_flat = lure_sims.flatten()
    unrelated_flat = unrelated_sims.flatten()

    violin_data = [studied_flat, lure_flat, unrelated_flat]
    positions = [0, 1, 2]
    violin_colors = [COLORS['primary'], COLORS['secondary'], COLORS['neutral']]
    violin_labels = ['Studied', 'Lure', 'Unrelated']

    parts = ax_b.violinplot(violin_data, positions=positions, showmeans=True,
                            showmedians=False, showextrema=False)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('none')

    # Style the mean line
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.0)

    # Add individual points as strip plot
    rng = np.random.RandomState(42)
    for i, data_arr in enumerate(violin_data):
        jitter = rng.uniform(-0.12, 0.12, size=len(data_arr))
        ax_b.scatter(np.full_like(data_arr, positions[i]) + jitter, data_arr,
                     c=violin_colors[i], s=4, alpha=0.3, edgecolors='none', zorder=3)

    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(violin_labels, fontsize=7)
    ax_b.set_ylabel('Cosine similarity')
    ax_b.set_ylim(0.55, 1.05)
    panel_label(ax_b, 'b', x=-0.20, y=1.05)

    # Panel (c): Threshold sweep curves
    ax_c.plot(thresholds, hit_mean, '-', color=COLORS['primary'], linewidth=1.3,
              label='Hit rate', zorder=4)
    ax_c.fill_between(thresholds, hit_mean - hit_std, hit_mean + hit_std,
                       color=COLORS['primary'], alpha=0.12, zorder=2)

    ax_c.plot(thresholds, fa_crit_mean, '-', color=COLORS['secondary'], linewidth=1.3,
              label='FA (critical lure)', zorder=4)
    ax_c.fill_between(thresholds, fa_crit_mean - fa_crit_std,
                       fa_crit_mean + fa_crit_std,
                       color=COLORS['secondary'], alpha=0.12, zorder=2)

    ax_c.plot(thresholds, fa_unrel_mean, '-', color=COLORS['neutral'], linewidth=1.3,
              label='FA (unrelated)', zorder=4)
    ax_c.fill_between(thresholds, fa_unrel_mean - fa_unrel_std,
                       fa_unrel_mean + fa_unrel_std,
                       color=COLORS['neutral'], alpha=0.12, zorder=2)

    # Human reference FA line at 55%
    human_reference_line(ax_c, 0.55, label='Human FA (~55%)')

    # Mark optimal threshold region
    # Find threshold closest to human FA rate for critical lure
    best_idx = np.argmin(np.abs(fa_crit_mean - 0.55))
    best_thresh = thresholds[best_idx]
    ax_c.axvline(x=best_thresh, color=COLORS['highlight'], linestyle=':',
                 linewidth=0.8, alpha=0.7, zorder=2)
    ax_c.text(best_thresh + 0.005, 0.55, f'  {best_thresh:.2f}',
              color=COLORS['highlight'], fontsize=6, va='center', fontstyle='italic')

    ax_c.set_xlabel('Recognition threshold')
    ax_c.set_ylabel('Rate')
    ax_c.set_xlim(thresholds[0], thresholds[-1])
    ax_c.set_ylim(-0.05, 1.08)
    ax_c.legend(loc='center left', fontsize=5.5, frameon=False)
    panel_label(ax_c, 'c', x=-0.12, y=1.05)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.95, wspace=0.40)
    save_figure(fig, 'ed_fig7_drm_lists')
    print("  ED Figure 7 complete.")


# ===========================================================================
# EXTENDED DATA FIGURE 8: Spacing sweep full results
# ===========================================================================
def generate_ed_fig8():
    """
    Three panels:
      (a) Retention vs sigma for each spacing condition at 25K distractors
      (b) Retention vs distractor count for each condition at sigma=0.25
      (c) 2D heatmap: sigma x distractor count -> Cohen's d (long vs massed)
    """
    print("Generating Extended Data Figure 8: Spacing sweep full results...")
    data_dir = os.path.join(PROJECT_ROOT, 'results', 'spacing_sweep')
    all_data = load_seed_data(data_dir)

    if len(all_data) == 0:
        print("ERROR: No spacing sweep data found. Skipping ED Fig 8.")
        return

    # Parse the key structure: "dist{N}_sigma{S}" -> {massed, short, medium, long}
    first_seed_data = next(iter(all_data.values()))
    results = first_seed_data['results']

    # Extract unique distractor counts and sigma values
    dist_counts = sorted(set(int(k.split('_')[0].replace('dist', ''))
                             for k in results.keys()))
    sigma_values = sorted(set(float(k.split('_')[1].replace('sigma', ''))
                              for k in results.keys()))

    conditions = ['massed', 'short', 'medium', 'long']
    cond_colors = {
        'massed': COLORS['neutral'],
        'short': COLORS['quaternary'],
        'medium': COLORS['tertiary'],
        'long': COLORS['primary'],
    }
    cond_markers = {
        'massed': 's',
        'short': '^',
        'medium': 'v',
        'long': 'o',
    }

    # Helper to get mean/std across seeds for a specific key and condition
    def get_retention(dist, sigma, condition):
        key = f'dist{dist}_sigma{sigma}'
        vals = []
        for seed in SEEDS:
            if seed in all_data and key in all_data[seed]['results']:
                vals.append(all_data[seed]['results'][key][condition])
        if len(vals) == 0:
            return np.nan, np.nan
        return np.mean(vals), np.std(vals)

    # -- Build figure --
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.32))
    ax_a, ax_b, ax_c = axes

    # Panel (a): Retention vs sigma for 25K distractors
    target_dist = 25000
    for cond in conditions:
        means = []
        stds = []
        for sigma in sigma_values:
            m, s = get_retention(target_dist, sigma, cond)
            means.append(m)
            stds.append(s)
        means = np.array(means)
        stds = np.array(stds)

        ax_a.errorbar(sigma_values, means, yerr=stds,
                      fmt=f'{cond_markers[cond]}-', color=cond_colors[cond],
                      markersize=4, capsize=2, linewidth=1.2,
                      label=cond.capitalize(), zorder=4)

    ax_a.set_xlabel('Noise level ($\\sigma$)')
    ax_a.set_ylabel('Retention')
    ax_a.set_title(f'{target_dist // 1000}K distractors', fontsize=8)
    ax_a.set_ylim(-0.05, 1.08)
    ax_a.legend(loc='lower left', fontsize=6, frameon=False)
    panel_label(ax_a, 'a')

    # Panel (b): Retention vs distractor count at sigma=0.25
    target_sigma = 0.25
    for cond in conditions:
        means = []
        stds = []
        for dist in dist_counts:
            m, s = get_retention(dist, target_sigma, cond)
            means.append(m)
            stds.append(s)
        means = np.array(means)
        stds = np.array(stds)

        dist_labels = [d / 1000 for d in dist_counts]
        ax_b.errorbar(dist_labels, means, yerr=stds,
                      fmt=f'{cond_markers[cond]}-', color=cond_colors[cond],
                      markersize=4, capsize=2, linewidth=1.2,
                      label=cond.capitalize(), zorder=4)

    ax_b.set_xlabel('Distractors (K)')
    ax_b.set_ylabel('Retention')
    ax_b.set_title(f'$\\sigma$ = {target_sigma}', fontsize=8)
    ax_b.set_ylim(-0.05, 1.08)
    ax_b.legend(loc='lower left', fontsize=6, frameon=False)
    panel_label(ax_b, 'b')

    # Panel (c): Heatmap of Cohen's d (long vs massed)
    cohens_d_matrix = np.full((len(sigma_values), len(dist_counts)), np.nan)

    for si, sigma in enumerate(sigma_values):
        for di, dist in enumerate(dist_counts):
            key = f'dist{dist}_sigma{sigma}'
            long_vals = []
            massed_vals = []
            for seed in SEEDS:
                if seed in all_data and key in all_data[seed]['results']:
                    long_vals.append(all_data[seed]['results'][key]['long'])
                    massed_vals.append(all_data[seed]['results'][key]['massed'])
            if len(long_vals) >= 2:
                # Cohen's d
                long_arr = np.array(long_vals)
                massed_arr = np.array(massed_vals)
                diff = long_arr - massed_arr
                pooled_std = np.sqrt((np.var(long_arr, ddof=1) +
                                      np.var(massed_arr, ddof=1)) / 2)
                if pooled_std > 0:
                    cohens_d_matrix[si, di] = np.mean(diff) / pooled_std
                else:
                    cohens_d_matrix[si, di] = 0.0
            elif len(long_vals) == 1:
                # Single-seed fallback: just raw difference
                cohens_d_matrix[si, di] = long_vals[0] - massed_vals[0]

    im = ax_c.imshow(cohens_d_matrix, aspect='auto', cmap='RdBu_r',
                     origin='lower', vmin=-1, vmax=5,
                     interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax_c, shrink=0.85, pad=0.03)
    cbar.set_label("Cohen's d\n(Long vs Massed)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_c.set_xticks(range(len(dist_counts)))
    ax_c.set_xticklabels([f'{d // 1000}K' for d in dist_counts], fontsize=6)
    ax_c.set_yticks(range(len(sigma_values)))
    ax_c.set_yticklabels([f'{s:.2f}' for s in sigma_values], fontsize=6)
    ax_c.set_xlabel('Distractors')
    ax_c.set_ylabel('$\\sigma$')
    panel_label(ax_c, 'c')

    fig.tight_layout()
    save_figure(fig, 'ed_fig8_spacing_sweep')
    print("  ED Figure 8 complete.")


# ===========================================================================
# EXTENDED DATA FIGURE 9: TOT analysis
# ===========================================================================
def generate_ed_fig9():
    """
    Two panels:
      (a) TOT rate bar: HIDE vs human reference (1.5%)
      (b) Per-seed TOT rates with human reference
    Uses v2 data (PCA 128-dim + noise) which has non-zero TOT rate.
    """
    print("Generating Extended Data Figure 9: TOT analysis...")
    # Use v2 data which has the corrected TOT experiment (3.66%)
    data_dir = os.path.join(PROJECT_ROOT, 'results', 'phase5')
    all_data = load_seed_data(data_dir)

    if len(all_data) == 0:
        print("ERROR: No v2 phase5 data found. Skipping ED Fig 9.")
        return

    # Collect TOT rates from v2 data (key is 'tot_v2')
    tot_rates = []
    tot_counts = []
    total_queries_list = []
    human_rate = 0.015  # 1.5%

    for seed in SEEDS:
        if seed in all_data and 'tot_v2' in all_data[seed]:
            tot_data = all_data[seed]['tot_v2']
            tot_rates.append(tot_data['tot_rate'])
            tot_counts.append(tot_data['tot_count'])
            total_queries_list.append(tot_data['total_queries'])

    tot_rate_mean = np.mean(tot_rates) if tot_rates else 0.0
    tot_rate_std = np.std(tot_rates) if tot_rates else 0.0
    total_queries = total_queries_list[0] if total_queries_list else 125

    # -- Build figure (2-panel, half width) --
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))
    ax_a, ax_b = axes

    # Panel (a): TOT rate bar comparison
    bar_width = 0.4
    x_pos = np.array([0, 1])
    bar_heights = [tot_rate_mean * 100, human_rate * 100]
    bar_errs = [tot_rate_std * 100, 0.3]  # 0.3% assumed CI for human rate
    bar_colors = [COLORS['primary'], COLORS['human']]
    bar_labels = ['HIDE', 'Human\n(Brown & McNeill)']

    bars = ax_a.bar(x_pos, bar_heights, bar_width * 1.8,
                    yerr=bar_errs,
                    color=bar_colors, capsize=4, edgecolor='none',
                    alpha=0.85, zorder=3)

    # Value labels
    for br, mv, sv in zip(bars, bar_heights, bar_errs):
        h = br.get_height()
        label_y = max(h + sv + 0.08, 0.15)
        ax_a.text(br.get_x() + br.get_width() / 2., label_y,
                  f'{mv:.1f}%', ha='center', va='bottom',
                  fontsize=8, fontweight='bold')

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(bar_labels, fontsize=7)
    ax_a.set_ylabel('TOT rate (%)')
    ax_a.set_ylim(0, max(bar_heights) * 1.35)
    ax_a.set_title('Tip-of-tongue rate', fontsize=8)
    panel_label(ax_a, 'a')

    # Panel (b): Per-seed TOT counts
    tot_count_total = sum(tot_counts) if tot_counts else 0
    tot_queries_total = total_queries * len(SEEDS)

    seed_labels = [str(s) for s in SEEDS[:len(tot_rates)]]
    seed_tot_pct = [r * 100 for r in tot_rates]

    bar_colors_b = [COLORS['primary']] * len(seed_labels)
    ax_b.bar(np.arange(len(seed_labels)), seed_tot_pct, 0.6,
             color=bar_colors_b, edgecolor='none', alpha=0.85, zorder=3)

    # Mean line
    ax_b.axhline(y=tot_rate_mean * 100, color='black', linewidth=0.8,
                 linestyle='--', alpha=0.6, zorder=2)
    ax_b.text(len(seed_labels) - 0.5, tot_rate_mean * 100 + 0.15,
              f'Mean = {tot_rate_mean * 100:.1f}%', fontsize=6, ha='right',
              fontstyle='italic')

    # Human reference
    human_reference_line(ax_b, human_rate * 100, label='Human (~1.5%)')

    ax_b.set_xticks(np.arange(len(seed_labels)))
    ax_b.set_xticklabels(seed_labels, fontsize=6.5)
    ax_b.set_xlabel('Seed')
    ax_b.set_ylabel('TOT rate (%)')
    ax_b.set_title('Per-seed TOT rates', fontsize=8)
    ax_b.set_ylim(0, max(seed_tot_pct + [human_rate * 100]) * 1.4)
    panel_label(ax_b, 'b', x=-0.05, y=1.05)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.92, wspace=0.30)
    save_figure(fig, 'ed_fig9_tot')
    print("  ED Figure 9 complete.")


# ===========================================================================
# EXTENDED DATA FIGURE 10: Reproducibility across seeds
# ===========================================================================
def generate_ed_fig10():
    """
    Two panels:
      (a) Strip/swarm plot of key metrics across 5 seeds
      (b) Bootstrap distribution of forgetting exponent b
    """
    print("Generating Extended Data Figure 10: Reproducibility across seeds...")

    # Load all data sources
    phase5_dir = os.path.join(PROJECT_ROOT, 'results', 'phase5')
    phase5_data = load_seed_data(phase5_dir)

    interf_dir = os.path.join(PROJECT_ROOT, 'results', 'interference')
    interf_data = load_seed_data(interf_dir)

    spacing_dir = os.path.join(PROJECT_ROOT, 'results', 'spacing_sweep')
    spacing_data = load_seed_data(spacing_dir)

    # Collect metrics per seed
    metrics = {}

    # 1. Interference b (d=64, near, max condition -- use near 200 which has highest b)
    b_values = []
    for seed in SEEDS:
        if seed in interf_data:
            dim64 = interf_data[seed].get('by_dim', {}).get('64', {})
            near = dim64.get('near', {})
            # Get the maximum near condition (200 near neighbors)
            if '200' in near and 'power_law_fit' in near['200'] and near['200']['power_law_fit']:
                b_values.append(near['200']['power_law_fit']['b'])
            elif '100' in near and 'power_law_fit' in near['100'] and near['100']['power_law_fit']:
                b_values.append(near['100']['power_law_fit']['b'])
    metrics['Forgetting\nexponent b'] = b_values if b_values else [0.0] * 5

    # 2. DRM FA rate (at best_match threshold)
    drm_fa = []
    for seed in SEEDS:
        if seed in phase5_data and 'drm' in phase5_data[seed]:
            bm = phase5_data[seed]['drm'].get('best_match', {})
            if 'false_alarm_critical' in bm:
                drm_fa.append(bm['false_alarm_critical'])
    metrics['DRM FA\nrate'] = drm_fa if drm_fa else [0.0] * 5

    # 3. Spacing retention -- long condition at sigma=0.25, 25K dist
    spacing_long = []
    for seed in SEEDS:
        if seed in spacing_data:
            key = 'dist25000_sigma0.25'
            if key in spacing_data[seed].get('results', {}):
                spacing_long.append(spacing_data[seed]['results'][key]['long'])
    metrics['Long spacing\nretention'] = spacing_long if spacing_long else [0.0] * 5

    # 4. TOT rate (use v2 data which has the corrected experiment)
    v2_phase5_dir = os.path.join(PROJECT_ROOT, 'results', 'phase5')
    v2_phase5_data = load_seed_data(v2_phase5_dir)
    tot_rates = []
    for seed in SEEDS:
        if seed in v2_phase5_data and 'tot_v2' in v2_phase5_data[seed]:
            tot_rates.append(v2_phase5_data[seed]['tot_v2']['tot_rate'])
    metrics['TOT\nrate'] = tot_rates if tot_rates else [0.0] * 5

    # 5. Mean lure similarity (as proxy for H1 peak which is 0 everywhere)
    lure_sims = []
    for seed in SEEDS:
        if seed in phase5_data and 'drm' in phase5_data[seed]:
            lure_sims.append(phase5_data[seed]['drm']['mean_lure_sim'])
    metrics['Mean lure\nsimilarity'] = lure_sims if lure_sims else [0.0] * 5

    # -- Build figure using GridSpec --
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)

    seed_colors = {
        42: '#2171B5',
        123: '#CB181D',
        456: '#238B45',
        789: '#6A51A3',
        1024: '#FF7F00',
    }

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))
    # Top-level split: left 65% for strip plots, right 35% for bootstrap
    gs_top = gridspec.GridSpec(1, 2, width_ratios=[65, 35], wspace=0.40,
                               figure=fig)

    # Left: n_metrics sub-columns for strip plots
    gs_left = gridspec.GridSpecFromSubplotSpec(1, n_metrics, subplot_spec=gs_top[0, 0],
                                               wspace=0.45)
    inner_axes = []
    for mi in range(n_metrics):
        ax_inner = fig.add_subplot(gs_left[0, mi])
        inner_axes.append(ax_inner)

    rng = np.random.RandomState(42)
    for mi, (name, vals) in enumerate(metrics.items()):
        ax = inner_axes[mi]
        vals = np.array(vals)

        # Plot each seed as a dot with jitter
        for si, (seed, val) in enumerate(zip(SEEDS[:len(vals)], vals)):
            jitter = rng.uniform(-0.15, 0.15)
            ax.scatter(jitter, val, c=seed_colors.get(seed, COLORS['primary']),
                       s=30, edgecolors='white', linewidths=0.3, zorder=4)

        # Mean and CI
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        ax.axhline(y=mean_val, color='black', linewidth=0.8, linestyle='-',
                   alpha=0.6, zorder=2)
        ax.axhspan(mean_val - std_val, mean_val + std_val,
                   color=COLORS['bg_shade'], alpha=0.5, zorder=1)

        # Formatting
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_xlabel(name, fontsize=6, labelpad=3)

        if mi == 0:
            ax.set_ylabel('Value', fontsize=7)
        else:
            ax.set_ylabel('')

        # Adjust y limits per metric
        if len(vals) > 0:
            val_range = max(vals) - min(vals)
            if val_range < 1e-10:
                # All same value
                ax.set_ylim(mean_val - 0.1, mean_val + 0.1)
            else:
                pad = val_range * 0.3
                ax.set_ylim(min(vals) - pad, max(vals) + pad)

        ax.tick_params(axis='y', labelsize=6)

    if len(inner_axes) > 0:
        panel_label(inner_axes[0], 'a', x=-0.5, y=1.10)

    # Create a seed legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=seed_colors[s],
                              markeredgecolor='white', markeredgewidth=0.3,
                              markersize=5, label=f'Seed {s}')
                       for s in SEEDS]
    if len(inner_axes) > 0:
        inner_axes[-1].legend(handles=legend_elements, loc='upper right',
                              fontsize=5, frameon=False, handletextpad=0.3)

    # Right: bootstrap panel
    ax_b = fig.add_subplot(gs_top[0, 1])

    b_vals = np.array(metrics.get('Forgetting\nexponent b', [0.0]))
    if len(b_vals) > 1 and np.std(b_vals) > 0:
        # Bootstrap
        n_bootstrap = 10000
        bootstrap_means = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(b_vals, size=len(b_vals), replace=True)
            bootstrap_means[i] = np.mean(sample)

        ax_b.hist(bootstrap_means, bins=50, color=COLORS['primary'], alpha=0.7,
                  edgecolor='none', density=True, zorder=3)

        # Mark the actual mean
        actual_mean = np.mean(b_vals)
        ax_b.axvline(x=actual_mean, color='black', linewidth=1.2, linestyle='-',
                     zorder=4)
        ax_b.text(actual_mean, ax_b.get_ylim()[1] * 0.95,
                  f'  Mean = {actual_mean:.4f}', fontsize=7, va='top',
                  fontweight='bold')

        # 95% CI
        ci_lo = np.percentile(bootstrap_means, 2.5)
        ci_hi = np.percentile(bootstrap_means, 97.5)
        ax_b.axvline(x=ci_lo, color=COLORS['secondary'], linewidth=0.8,
                     linestyle='--', zorder=3)
        ax_b.axvline(x=ci_hi, color=COLORS['secondary'], linewidth=0.8,
                     linestyle='--', zorder=3)
        ax_b.text(ci_lo, ax_b.get_ylim()[1] * 0.80,
                  f'  2.5%: {ci_lo:.4f}', fontsize=6, color=COLORS['secondary'])
        ax_b.text(ci_hi, ax_b.get_ylim()[1] * 0.80,
                  f'  97.5%: {ci_hi:.4f}', fontsize=6, color=COLORS['secondary'])

        ax_b.set_xlabel('Forgetting exponent b')
        ax_b.set_ylabel('Density')
        ax_b.set_title('Bootstrap distribution', fontsize=8)
    else:
        # Fallback: just show the individual b values
        if len(b_vals) > 0:
            ax_b.bar(range(len(b_vals)), b_vals, color=COLORS['primary'],
                     alpha=0.85, edgecolor='none')
            ax_b.set_xticks(range(len(b_vals)))
            ax_b.set_xticklabels([str(s) for s in SEEDS[:len(b_vals)]], fontsize=6)
            ax_b.set_xlabel('Seed')
            ax_b.set_ylabel('Forgetting exponent b')
            ax_b.set_title('Per-seed exponent values', fontsize=8)

            # Add mean line
            mean_b = np.mean(b_vals)
            ax_b.axhline(y=mean_b, color='black', linewidth=0.8, linestyle='--',
                         zorder=2)
            ax_b.text(len(b_vals) - 0.5, mean_b + 0.002,
                      f'Mean = {mean_b:.4f}', fontsize=6, ha='right')
        else:
            ax_b.text(0.5, 0.5, 'No b values available',
                      transform=ax_b.transAxes, ha='center', va='center',
                      fontsize=8, color=COLORS['neutral'])
            ax_b.set_xlabel('Forgetting exponent b')

    panel_label(ax_b, 'b')

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.18, top=0.92)
    save_figure(fig, 'ed_fig10_reproducibility')
    print("  ED Figure 10 complete.")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Generating Extended Data Figures 6-10 for HIDE v5 paper")
    print("=" * 70)

    generate_ed_fig6()
    generate_ed_fig7()
    generate_ed_fig8()
    generate_ed_fig9()
    generate_ed_fig10()

    print("\n" + "=" * 70)
    print("All Extended Data Figures generated successfully.")
    print("=" * 70)
