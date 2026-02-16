#!/usr/bin/env python3
"""
Generate Extended Data Figures 1-5 for the HIDE v5 paper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figure_style import (
    set_nature_style, COLORS, BLUES_GRADIENT, MARKERS,
    FULL_WIDTH, HALF_WIDTH, panel_label, human_reference_line, save_figure
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json

set_nature_style()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'paper', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(path):
    """Load a JSON file, return None if not found."""
    full = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(full):
        with open(full) as f:
            return json.load(f)
    print(f"WARNING: {full} not found")
    return None


# ========================================================================
# Extended Data Figure 1: Architecture detail
# ========================================================================
def make_ed_fig1():
    fig, ax = plt.subplots(1, 1, figsize=(FULL_WIDTH, FULL_WIDTH * 0.48))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    # --- Input boxes (left) ---
    text_box = FancyBboxPatch((0.3, 2.7), 1.6, 0.9,
                               boxstyle="round,pad=0.1",
                               facecolor='#D6EAF8', edgecolor=COLORS['primary'],
                               linewidth=1.0)
    ax.add_patch(text_box)
    ax.text(1.1, 3.15, 'Text Input', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLORS['primary'])

    img_box = FancyBboxPatch((0.3, 1.2), 1.6, 0.9,
                              boxstyle="round,pad=0.1",
                              facecolor='#D5F5E3', edgecolor=COLORS['tertiary'],
                              linewidth=1.0)
    ax.add_patch(img_box)
    ax.text(1.1, 1.65, 'Image Input', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLORS['tertiary'])

    # Model name labels
    ax.text(1.1, 2.55, 'MiniLM / BGE-large', ha='center', va='center',
            fontsize=5.5, fontstyle='italic', color=COLORS['neutral'])
    ax.text(1.1, 1.05, 'CLIP ViT-B/32', ha='center', va='center',
            fontsize=5.5, fontstyle='italic', color=COLORS['neutral'])

    # --- Embedding Space (middle) ---
    emb_box = FancyBboxPatch((3.0, 0.6), 3.5, 3.3,
                              boxstyle="round,pad=0.2",
                              facecolor=COLORS['bg_shade'], edgecolor=COLORS['neutral'],
                              linewidth=1.2, linestyle='-')
    ax.add_patch(emb_box)
    ax.text(4.75, 3.65, 'HIDE Embedding Space', ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLORS['neutral'])

    # Scatter dots inside embedding space
    rng = np.random.RandomState(42)
    n_dots = 60
    dot_x = rng.uniform(3.3, 6.2, n_dots)
    dot_y = rng.uniform(0.9, 3.4, n_dots)
    # Color by "modality"
    colors_dots = [COLORS['primary']] * 30 + [COLORS['tertiary']] * 20 + [COLORS['quaternary']] * 10
    ax.scatter(dot_x, dot_y, s=8, c=colors_dots[:n_dots], alpha=0.5, zorder=3)

    # Some cluster-like groupings
    for cx, cy, c in [(4.0, 2.5, COLORS['primary']),
                       (5.5, 1.8, COLORS['tertiary']),
                       (4.8, 3.0, COLORS['quaternary'])]:
        circle = plt.Circle((cx, cy), 0.45, fill=False, edgecolor=c,
                            linestyle='--', linewidth=0.6, alpha=0.5)
        ax.add_patch(circle)

    # --- Retrieval + Answer (right) ---
    ret_box = FancyBboxPatch((7.5, 2.2), 2.1, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#FADBD8', edgecolor=COLORS['secondary'],
                              linewidth=1.0)
    ax.add_patch(ret_box)
    ax.text(8.55, 3.15, 'Retrieval', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLORS['secondary'])
    ax.text(8.55, 2.75, 'Answer', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLORS['secondary'])
    ax.text(8.55, 2.35, 'Qwen2.5-7B', ha='center', va='center',
            fontsize=5.5, fontstyle='italic', color=COLORS['neutral'])

    # --- Arrows ---
    arrow_kw = dict(arrowstyle='->', color=COLORS['neutral'],
                    linewidth=1.2, mutation_scale=12)
    # Text -> Embedding
    ax.annotate('', xy=(3.0, 3.0), xytext=(1.9, 3.15),
                arrowprops=arrow_kw)
    # Image -> Embedding
    ax.annotate('', xy=(3.0, 1.8), xytext=(1.9, 1.65),
                arrowprops=arrow_kw)
    # Embedding -> Retrieval
    ax.annotate('', xy=(7.5, 2.95), xytext=(6.5, 2.5),
                arrowprops=arrow_kw)

    # --- Boundary condition boxes (bottom) ---
    bc_labels = ['Temporal decay', 'Interference', 'Noise']
    bc_colors = [COLORS['highlight'], COLORS['secondary'], COLORS['quaternary']]
    for i, (lbl, c) in enumerate(zip(bc_labels, bc_colors)):
        x_start = 2.8 + i * 1.8
        bc = FancyBboxPatch((x_start, 0.0), 1.5, 0.45,
                             boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor=c,
                             linewidth=0.8)
        ax.add_patch(bc)
        ax.text(x_start + 0.75, 0.22, lbl, ha='center', va='center',
                fontsize=6.5, color=c, fontweight='bold')

    fig.tight_layout()
    save_figure(fig, 'ed_fig1_architecture')
    print("ED Fig 1 done.")


# ========================================================================
# Extended Data Figure 2: Phase 1 bAbI full results
# ========================================================================
def make_ed_fig2():
    summary = load_json('results/phase1/summary.json')
    if summary is None:
        print("Skipping ED Fig 2 -- no data")
        return

    tasks = summary['tasks']
    task_ids = sorted(tasks.keys(), key=int)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.33))

    # --- Panel a: Bar chart accuracy by task ---
    ax = axes[0]
    panel_label(ax, 'a')
    methods = ['hide', 'no_memory', 'random_retrieval', 'full_context']
    method_labels = ['HIDE', 'No memory', 'Random retrieval', 'Full context']
    method_colors = [COLORS['primary'], COLORS['light_neutral'],
                     COLORS['neutral'], COLORS['tertiary']]

    x = np.arange(len(task_ids))
    bar_w = 0.18
    for j, (m, lbl, c) in enumerate(zip(methods, method_labels, method_colors)):
        means = [tasks[t][m]['mean'] for t in task_ids]
        stds = [tasks[t][m]['std'] for t in task_ids]
        ax.bar(x + j * bar_w, means, bar_w, yerr=stds,
               label=lbl, color=c, edgecolor='white', linewidth=0.3,
               capsize=2, error_kw={'linewidth': 0.6})

    ax.set_xticks(x + 1.5 * bar_w)
    ax.set_xticklabels([f'Task {t}' for t in task_ids], fontsize=6)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5.5, loc='upper left', ncol=1)
    ax.set_title('Accuracy by bAbI task')

    # --- Panel b: Memory scaling ---
    ax = axes[1]
    panel_label(ax, 'b')
    mem_scaling = summary['memory_scaling']
    ns = sorted(mem_scaling.keys(), key=lambda k: int(k))
    n_vals = [int(n) for n in ns]
    means = [mem_scaling[n]['mean'] for n in ns]
    stds = [mem_scaling[n]['std'] for n in ns]

    ax.errorbar(n_vals, means, yerr=stds, marker='o', color=COLORS['primary'],
                linewidth=1.2, markersize=4, capsize=3, capthick=0.8)
    ax.set_xlabel('Memory store size N')
    ax.set_ylabel('Precision@5')
    ax.set_title('Memory scaling')
    ax.set_xscale('log')

    # --- Panel c: Retrieval precision heatmap ---
    ax = axes[2]
    panel_label(ax, 'c')

    # Build heatmap: rows = methods, cols = tasks
    heatmap_methods = ['hide', 'random_retrieval', 'full_context', 'vanilla_rag']
    heatmap_labels = ['HIDE', 'Random', 'Full context', 'Vanilla RAG']
    data_matrix = np.zeros((len(heatmap_methods), len(task_ids)))
    for i, m in enumerate(heatmap_methods):
        for j, t in enumerate(task_ids):
            data_matrix[i, j] = tasks[t][m]['mean']

    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(task_ids)))
    ax.set_xticklabels([f'T{t}' for t in task_ids], fontsize=6)
    ax.set_yticks(range(len(heatmap_methods)))
    ax.set_yticklabels(heatmap_labels, fontsize=6)
    ax.set_title('Accuracy heatmap')

    # Add text annotations
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            val = data_matrix[i, j]
            txt_color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    save_figure(fig, 'ed_fig2_phase1')
    print("ED Fig 2 done.")


# ========================================================================
# Extended Data Figure 3: Ebbinghaus parameter sweep
# ========================================================================
def make_ed_fig3():
    data = load_json('results/phase2/summary_temporal.json')
    if data is None:
        print("Skipping ED Fig 3 -- no data")
        return

    configs = data['configs']

    # Parse config keys: "sigma{S}_beta{B}"
    sigma_vals = sorted(set(float(k.split('_')[0].replace('sigma', ''))
                            for k in configs.keys()))
    beta_vals = sorted(set(float(k.split('_')[1].replace('beta', ''))
                           for k in configs.keys()))

    # Build 2D arrays
    b_grid = np.full((len(sigma_vals), len(beta_vals)), np.nan)
    r2_grid = np.full((len(sigma_vals), len(beta_vals)), np.nan)

    best_b_diff = float('inf')
    best_ij = (0, 0)

    for i, s in enumerate(sigma_vals):
        for j, b in enumerate(beta_vals):
            key = f"sigma{s}_beta{b}"
            if key not in configs:
                # Try alternate format
                key = f"sigma{s:.1f}_beta{b:.1f}"
            if key in configs:
                b_grid[i, j] = configs[key]['mean_b']
                r2_grid[i, j] = configs[key]['mean_r2']
                diff = abs(configs[key]['mean_b'] - 0.5)
                if diff < best_b_diff:
                    best_b_diff = diff
                    best_ij = (i, j)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.38))

    # --- Panel a: Fitted exponent b ---
    ax = axes[0]
    panel_label(ax, 'a')
    im1 = ax.imshow(b_grid, aspect='auto', cmap='RdYlBu_r', origin='lower',
                    vmin=0.2, vmax=1.2)
    ax.set_xticks(range(len(beta_vals)))
    ax.set_xticklabels([f'{b:.1f}' for b in beta_vals], fontsize=5, rotation=45)
    ax.set_yticks(range(len(sigma_vals)))
    ax.set_yticklabels([f'{s:.1f}' for s in sigma_vals], fontsize=6)
    ax.set_xlabel(r'$\beta$ (decay rate)')
    ax.set_ylabel(r'$\sigma$ (noise)')
    ax.set_title('Fitted exponent $b$')

    # Contour at b=0.5
    if not np.all(np.isnan(b_grid)):
        try:
            cs = ax.contour(b_grid, levels=[0.5], colors=[COLORS['human']],
                           linewidths=1.2, linestyles='--', origin='lower')
            ax.clabel(cs, fmt='$b$=0.5', fontsize=6, colors=[COLORS['human']])
        except Exception:
            pass

    # Mark best with star
    ax.plot(best_ij[1], best_ij[0], marker='*', color=COLORS['highlight'],
            markersize=12, markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_label('Exponent $b$', fontsize=7)

    # --- Panel b: R^2 values ---
    ax = axes[1]
    panel_label(ax, 'b')
    im2 = ax.imshow(r2_grid, aspect='auto', cmap='YlGn', origin='lower',
                    vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(beta_vals)))
    ax.set_xticklabels([f'{b:.1f}' for b in beta_vals], fontsize=5, rotation=45)
    ax.set_yticks(range(len(sigma_vals)))
    ax.set_yticklabels([f'{s:.1f}' for s in sigma_vals], fontsize=6)
    ax.set_xlabel(r'$\beta$ (decay rate)')
    ax.set_ylabel(r'$\sigma$ (noise)')
    ax.set_title('$R^2$ of power-law fit')

    # Mark best with star
    ax.plot(best_ij[1], best_ij[0], marker='*', color=COLORS['highlight'],
            markersize=12, markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label('$R^2$', fontsize=7)

    fig.tight_layout()
    save_figure(fig, 'ed_fig3_ebbinghaus_sweep')
    print("ED Fig 3 done.")


# ========================================================================
# Extended Data Figure 4: Interference full results
# ========================================================================
def make_ed_fig4():
    # Load all seeds and average
    seeds = [42, 123, 456, 789, 1024]
    all_data = []
    for s in seeds:
        d = load_json(f'results/interference/results_seed{s}.json')
        if d is not None:
            all_data.append(d)

    if not all_data:
        print("Skipping ED Fig 4 -- no data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.65))

    # Since seeds have different age bins / lengths, we use per-seed overall_retention
    # and also plot seed 42 curves as representative (with std across seeds for overall)
    ref = all_data[0]  # seed 42

    # Helper: get retention curves from one seed's data, handling variable lengths
    def get_single_seed_curve(data, dim, cond, count):
        entry = data['by_dim'][dim][cond][count]
        return np.array(entry['ages']), np.array(entry['retentions'])

    # For plotting curves, use seed 42 as representative (most common 7 bins)
    ages_ref = np.array(ref['by_dim']['64']['near']['0']['ages'])

    # --- Panel a: d=64 near, all distractor counts ---
    ax = axes[0, 0]
    panel_label(ax, 'a')
    near_counts = sorted(ref['by_dim']['64']['near'].keys(), key=int)

    # Build gradient colors
    n_curves = len(near_counts)
    cmap = plt.cm.Blues
    curve_colors = [cmap(0.3 + 0.7 * i / (n_curves - 1)) for i in range(n_curves)]

    for ci, nc in enumerate(near_counts):
        ages_c, ret_c = get_single_seed_curve(ref, '64', 'near', nc)
        n_total = ref['by_dim']['64']['near'][nc]['n_total_memories']
        label = f'{nc} near ({n_total} total)'
        # Compute overall retention mean/std across seeds
        overs = [d['by_dim']['64']['near'][nc]['overall_retention'] for d in all_data]
        ax.plot(ages_c, ret_c, color=curve_colors[ci], marker='o',
                markersize=3, linewidth=1.0,
                label=f'{nc} near (mean ret={np.mean(overs):.2f})')

    ax.set_xlabel('Memory age (days)')
    ax.set_ylabel('Retention')
    ax.set_title('$d$=64, near distractors')
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=4.5, loc='lower left', ncol=1)

    # --- Panel b: d=128 near, all distractor counts ---
    ax = axes[0, 1]
    panel_label(ax, 'b')
    near_counts_128 = sorted(ref['by_dim']['128']['near'].keys(), key=int)

    for ci, nc in enumerate(near_counts_128):
        ages_c, ret_c = get_single_seed_curve(ref, '128', 'near', nc)
        overs = [d['by_dim']['128']['near'][nc]['overall_retention'] for d in all_data]
        ax.plot(ages_c, ret_c, color=curve_colors[min(ci, n_curves-1)], marker='o',
                markersize=3, linewidth=1.0,
                label=f'{nc} near (mean ret={np.mean(overs):.3f})')

    ax.set_xlabel('Memory age (days)')
    ax.set_ylabel('Retention')
    ax.set_title('$d$=128, near distractors')
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=4.5, loc='lower left', ncol=1)

    # --- Panel c: Near vs Far at d=64, comparable counts ---
    ax = axes[1, 0]
    panel_label(ax, 'c')

    # Use 50 near (total ~10K) and 10000 far (total ~10K)
    near_key = '50'
    far_key = '10000'

    # Near d=64 -- seed 42
    ages_near, ret_near = get_single_seed_curve(ref, '64', 'near', near_key)
    ages_far, ret_far = get_single_seed_curve(ref, '64', 'far', far_key)

    # Overall retention mean/std across seeds
    near_overs = [d['by_dim']['64']['near'][near_key]['overall_retention'] for d in all_data]
    far_overs = [d['by_dim']['64']['far'][far_key]['overall_retention'] for d in all_data]

    ax.plot(ages_near, ret_near, marker='o', color=COLORS['secondary'],
            linewidth=1.2, markersize=4,
            label=f'Near (50/target, ~10K)\nmean={np.mean(near_overs):.3f}$\\pm${np.std(near_overs):.3f}')
    ax.plot(ages_far, ret_far, marker='s', color=COLORS['tertiary'],
            linewidth=1.2, markersize=4,
            label=f'Far (10K total)\nmean={np.mean(far_overs):.3f}$\\pm${np.std(far_overs):.3f}')
    ax.set_xlabel('Memory age (days)')
    ax.set_ylabel('Retention')
    ax.set_title('Near vs Far at $d$=64')
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=5)

    # --- Panel d: Overall retention bar chart d=64 near at 0 vs 200 distractors per seed ---
    ax = axes[1, 1]
    panel_label(ax, 'd')

    # Bar chart of overall retention across seeds for 0 and 200 distractors
    ages_0, ret_0 = get_single_seed_curve(ref, '64', 'near', '0')
    ages_200, ret_200 = get_single_seed_curve(ref, '64', 'near', '200')

    # Use seed 42's age bins; show seed 42 data with bars
    age_labels = [f'{a:.1f}d' for a in ages_0]
    x = np.arange(len(ages_0))
    w = 0.35
    ax.bar(x - w/2, ret_0, w, label='0 distractors',
           color=COLORS['primary'], edgecolor='white', linewidth=0.3)
    ax.bar(x + w/2, ret_200, w, label='200/target (40K total)',
           color=COLORS['secondary'], edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels, fontsize=5.5, rotation=30)
    ax.set_xlabel('Memory age')
    ax.set_ylabel('Retention')
    ax.set_title('$d$=64 near: 0 vs 40K distractors')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=6)

    fig.tight_layout()
    save_figure(fig, 'ed_fig4_interference')
    print("ED Fig 4 done.")


# ========================================================================
# Extended Data Figure 5: Consolidation results
# ========================================================================
def make_ed_fig5():
    # Load v2 consolidation summary (3 conditions) and phase3 summary (6 conditions)
    v2_data = load_json('results/phase3/summary.json')
    p3_data = load_json('results/phase3/summary.json')

    if v2_data is None and p3_data is None:
        print("Skipping ED Fig 5 -- no data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.33))

    # Merge conditions from both sources
    conditions = {}
    # From v2 data (3 conditions)
    if v2_data:
        for key in v2_data:
            if key in ('no_consolidation', 'gentle_consolidation', 'full_hide_v2'):
                conditions[key] = {
                    'bt_mean': v2_data[key]['mean_backward_transfer'],
                    'bt_std': v2_data[key]['std_backward_transfer'],
                    'comp_mean': v2_data[key]['mean_compression'],
                    'comp_std': v2_data[key]['std_compression'],
                }
    # From phase3 summary (more conditions)
    if p3_data:
        for key in p3_data:
            if key in ('seeds', 'n_seeds', 'validation'):
                continue
            if key not in conditions:  # Don't overwrite v2 data
                conditions[key] = {
                    'bt_mean': p3_data[key]['backward_transfer_mean'],
                    'bt_std': p3_data[key]['backward_transfer_std'],
                    'comp_mean': p3_data[key]['compression_mean'],
                    'comp_std': p3_data[key]['compression_std'],
                }

    cond_keys = list(conditions.keys())
    # Shorter labels
    short_name_map = {
        'no_consolidation': 'No consol.',
        'gentle_consolidation': 'Gentle',
        'full_hide_v2': 'Full HIDE\n(v2)',
        'consolidation_only': 'Consol.\nonly',
        'replay_only': 'Replay\nonly',
        'full_hide': 'Full HIDE',
        'naive_pruning': 'Naive\npruning',
        'experience_replay': 'Exp.\nreplay',
    }
    cond_labels = [short_name_map.get(k, k.replace('_', '\n')) for k in cond_keys]
    n_conds = len(cond_keys)

    # Color mapping
    cond_colors = []
    for k in cond_keys:
        if 'no_consol' in k:
            cond_colors.append(COLORS['light_neutral'])
        elif 'gentle' in k:
            cond_colors.append(COLORS['tertiary'])
        elif 'full_hide' in k:
            cond_colors.append(COLORS['primary'])
        elif 'consol' in k:
            cond_colors.append(COLORS['quaternary'])
        elif 'replay' in k:
            cond_colors.append(COLORS['highlight'])
        elif 'naive' in k or 'experience' in k:
            cond_colors.append(COLORS['neutral'])
        else:
            cond_colors.append(COLORS['neutral'])

    x = np.arange(n_conds)

    # --- Panel a: Backward transfer ---
    ax = axes[0]
    panel_label(ax, 'a')
    bt_means = [conditions[k]['bt_mean'] for k in cond_keys]
    bt_stds = [conditions[k]['bt_std'] for k in cond_keys]

    bars = ax.bar(x, bt_means, 0.6, yerr=bt_stds, color=cond_colors,
                  edgecolor='white', linewidth=0.3, capsize=3,
                  error_kw={'linewidth': 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=4.5, rotation=45, ha='right')
    ax.set_ylabel('Backward transfer')
    ax.set_title('Backward transfer by condition')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

    # --- Panel b: Compression ratio ---
    ax = axes[1]
    panel_label(ax, 'b')
    comp_means = [conditions[k]['comp_mean'] for k in cond_keys]
    comp_stds = [conditions[k]['comp_std'] for k in cond_keys]

    ax.bar(x, comp_means, 0.6, yerr=comp_stds, color=cond_colors,
           edgecolor='white', linewidth=0.3, capsize=3,
           error_kw={'linewidth': 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=4.5, rotation=45, ha='right')
    ax.set_ylabel('Compression ratio')
    ax.set_title('Compression by condition')
    ax.set_ylim(0, 1.15)

    # --- Panel c: Scatter: compression vs backward transfer ---
    ax = axes[2]
    panel_label(ax, 'c')
    for i, k in enumerate(cond_keys):
        ax.scatter(conditions[k]['comp_mean'], conditions[k]['bt_mean'],
                   color=cond_colors[i], s=50, marker='o', zorder=3,
                   edgecolors='black', linewidth=0.5)
        # Error bars
        ax.errorbar(conditions[k]['comp_mean'], conditions[k]['bt_mean'],
                    xerr=conditions[k]['comp_std'], yerr=conditions[k]['bt_std'],
                    color=cond_colors[i], linewidth=0.6, capsize=2, fmt='none')
        # Label
        short_label = short_name_map.get(k, k.replace('_', ' ')).replace('\n', ' ')
        # Stagger labels vertically to avoid overlap
        y_offset = 6 if i % 2 == 0 else -10
        ax.annotate(short_label,
                    (conditions[k]['comp_mean'], conditions[k]['bt_mean']),
                    fontsize=4.5, xytext=(5, y_offset), textcoords='offset points',
                    color=cond_colors[i])

    ax.set_xlabel('Compression ratio')
    ax.set_ylabel('Backward transfer')
    ax.set_title('Compression vs backward transfer')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

    fig.tight_layout()
    save_figure(fig, 'ed_fig5_consolidation')
    print("ED Fig 5 done.")


# ========================================================================
# Main
# ========================================================================
if __name__ == '__main__':
    print("Generating Extended Data Figures 1-5...")
    make_ed_fig1()
    make_ed_fig2()
    make_ed_fig3()
    make_ed_fig4()
    make_ed_fig5()
    print("All Extended Data Figures generated.")
