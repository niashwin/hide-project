#!/usr/bin/env python3
"""
Generate Figure 3: Spacing Effect Mechanism
Three-panel figure showing timeline schematic, retention by condition,
and noise sensitivity sweep.

Code generation assisted by Claude (Anthropic).
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add parent dir so we can import figure_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (
    set_nature_style, COLORS, FULL_WIDTH, panel_label,
    human_reference_line, save_figure
)

# ==============================================================================
# Configuration
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'spacing_sweep')
SEEDS = [42, 123, 456, 789, 1024]
CONDITIONS = ['massed', 'short', 'medium', 'long']

# Spacing condition timings (in seconds)
TIMING_SEC = {
    'massed': [0, 60, 120],          # 0, 1min, 2min
    'short':  [0, 3600, 7200],       # 0, 1h, 2h
    'medium': [0, 86400, 172800],    # 0, 1d, 2d
    'long':   [0, 604800, 1209600],  # 0, 1w, 2w
}
TEST_TIME_SEC = 2592000  # 30 days

# Convert to days for panel a
SEC_PER_DAY = 86400
TIMING_DAYS = {k: [t / SEC_PER_DAY for t in v] for k, v in TIMING_SEC.items()}
TEST_DAY = TEST_TIME_SEC / SEC_PER_DAY  # 30.0

# Human reference (Cepeda et al. 2006 inspired values)
HUMAN_REF = {
    'massed': 0.30,
    'short':  0.40,
    'medium': 0.50,
    'long':   0.65,
}

# Condition colors (matching the user spec)
COND_COLORS = {
    'long':   COLORS['primary'],       # Steel blue
    'medium': COLORS['tertiary'],      # Forest green
    'short':  COLORS['quaternary'],    # Purple
    'massed': COLORS['neutral'],       # Dark gray
}

# ==============================================================================
# Load data
# ==============================================================================

def load_seed_data():
    """Load per-seed results."""
    all_data = {}
    for seed in SEEDS:
        fpath = os.path.join(RESULTS_DIR, f'results_seed{seed}.json')
        with open(fpath) as f:
            all_data[seed] = json.load(f)
    return all_data


def load_summary():
    """Load aggregated summary."""
    fpath = os.path.join(RESULTS_DIR, 'summary_v3.json')
    with open(fpath) as f:
        return json.load(f)


# ==============================================================================
# Panel a: Timeline schematic + data hybrid
# ==============================================================================

def draw_panel_a(ax, summary):
    """Timeline schematic with repetition dots and retention values."""
    # Get retention values from dist25000_sigma0.25 (the selected config)
    cfg_key = 'dist25000_sigma0.25'
    cfg = summary['configs'][cfg_key]

    # Lane positions (y-coordinates) — long on top, massed on bottom
    lane_order = ['long', 'medium', 'short', 'massed']
    lane_y = {cond: i for i, cond in enumerate(lane_order)}
    lane_labels = {
        'long':   'Long\n(0, 1w, 2w)',
        'medium': 'Medium\n(0, 1d, 2d)',
        'short':  'Short\n(0, 1h, 2h)',
        'massed': 'Massed\n(0, 1m, 2m)',
    }

    # Draw horizontal lane lines (subtle)
    for cond in lane_order:
        y = lane_y[cond]
        ax.axhline(y=y, color=COLORS['bg_shade'], linewidth=1.0, zorder=0)

    # Draw repetition dots on timeline
    for cond in lane_order:
        y = lane_y[cond]
        color = COND_COLORS[cond]
        times = TIMING_DAYS[cond]

        # Plot repetition markers
        ax.scatter(times, [y] * len(times), color=color, s=70, zorder=5,
                   edgecolors='white', linewidths=0.5)

        # Draw arrow from last repetition to test with fading
        last_rep = times[-1]
        # Create a gradient arrow using multiple line segments
        n_segments = 40
        x_vals = np.linspace(last_rep, TEST_DAY, n_segments + 1)
        for i in range(n_segments):
            alpha = 0.7 * (1 - i / n_segments) + 0.1  # fade from 0.8 to 0.1
            ax.plot([x_vals[i], x_vals[i + 1]], [y, y],
                    color=color, alpha=alpha, linewidth=1.0, zorder=2)

        # Small arrowhead at test line
        ax.annotate('', xy=(TEST_DAY - 0.3, y), xytext=(TEST_DAY - 1.5, y),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.0,
                                     mutation_scale=12),
                     zorder=3)

    # Vertical dashed line at test time
    ax.axvline(x=TEST_DAY, color=COLORS['secondary'], linestyle='--',
               linewidth=1.2, alpha=0.8, zorder=4)
    ax.text(TEST_DAY, len(lane_order) - 0.5, 'Test', color=COLORS['secondary'],
            fontsize=7, ha='center', va='bottom', fontstyle='italic')

    # Retention values to the right of each lane
    retention_vals = {cond: cfg[cond]['mean'] for cond in lane_order}
    for cond in lane_order:
        y = lane_y[cond]
        ret = retention_vals[cond]
        color = COND_COLORS[cond]
        ax.text(TEST_DAY + 1.5, y, f'{ret:.3f}', fontsize=7, ha='left',
                va='center', color=color, fontweight='bold')

    # Retention column header
    ax.text(TEST_DAY + 1.5, len(lane_order) - 0.5, 'Retention', fontsize=7,
            ha='left', va='bottom', color=COLORS['neutral'], fontstyle='italic')

    # Lane labels on left
    for cond in lane_order:
        y = lane_y[cond]
        ax.text(-1.8, y, lane_labels[cond], fontsize=6.5, ha='right',
                va='center', color=COND_COLORS[cond])

    # Axis formatting
    ax.set_xlim(-2.5, TEST_DAY + 6)
    ax.set_ylim(-0.7, len(lane_order) - 0.3)
    ax.set_xlabel('Time (days)')
    ax.set_yticks([])
    ax.set_xticks([0, 2, 7, 14, 21, 30])
    ax.spines['left'].set_visible(False)

    # Light background
    ax.set_facecolor('#FAFAFA')


# ==============================================================================
# Panel b: Retention by condition (HIDE vs Human)
# ==============================================================================

def draw_panel_b(ax, summary):
    """Bar/dot plot: HIDE retention vs human reference at dist25000 sigma=0.25."""
    cfg_key = 'dist25000_sigma0.25'
    cfg = summary['configs'][cfg_key]

    x_pos = np.arange(len(CONDITIONS))
    x_labels = ['Massed', 'Short', 'Medium', 'Long']

    # HIDE data
    hide_means = [cfg[c]['mean'] for c in CONDITIONS]
    hide_ci_lo = [cfg[c]['ci'][0] for c in CONDITIONS]
    hide_ci_hi = [cfg[c]['ci'][1] for c in CONDITIONS]
    hide_err_lo = [m - lo for m, lo in zip(hide_means, hide_ci_lo)]
    hide_err_hi = [hi - m for m, hi in zip(hide_means, hide_ci_hi)]

    # Human reference
    human_vals = [HUMAN_REF[c] for c in CONDITIONS]

    # Plot HIDE dots with error bars and connecting line
    ax.errorbar(x_pos, hide_means, yerr=[hide_err_lo, hide_err_hi],
                fmt='o', color=COLORS['primary'], markersize=6,
                capsize=3, capthick=0.8, linewidth=0.8, zorder=5,
                label='HIDE')
    ax.plot(x_pos, hide_means, '-', color=COLORS['primary'], linewidth=1.0,
            alpha=0.7, zorder=4)

    # Plot human reference diamonds with dashed line
    ax.plot(x_pos, human_vals, 'D', color=COLORS['human'], markersize=5,
            zorder=5, label='Human')
    ax.plot(x_pos, human_vals, '--', color=COLORS['human'], linewidth=0.8,
            alpha=0.7, zorder=4)

    # Axis formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Retention')
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.5, 3.5)
    ax.legend(loc='upper left', frameon=False, fontsize=7)

    # Annotate the monotonic ordering
    ax.annotate('', xy=(3.3, 0.95), xytext=(3.3, 0.20),
                arrowprops=dict(arrowstyle='->', color=COLORS['light_neutral'],
                                lw=1.0))
    ax.text(3.45, 0.57, 'Spacing\nbenefit', fontsize=6,
            color=COLORS['light_neutral'], ha='left', va='center',
            fontstyle='italic')


# ==============================================================================
# Panel c: Noise sensitivity sweep
# ==============================================================================

def draw_panel_c(ax, summary):
    """Retention vs noise level sigma for dist=25000, showing condition separation."""
    # Extract sigma values from dist25000 keys
    sigma_vals = []
    data_by_sigma = {}

    for key, val in summary['configs'].items():
        if key.startswith('dist25000_sigma'):
            sigma = float(key.replace('dist25000_sigma', ''))
            sigma_vals.append(sigma)
            data_by_sigma[sigma] = val

    sigma_vals = sorted(sigma_vals)

    # Plot one line per condition (long first so it's on top in legend)
    plot_order = ['long', 'medium', 'short', 'massed']
    for cond in plot_order:
        color = COND_COLORS[cond]
        means = [data_by_sigma[s][cond]['mean'] for s in sigma_vals]
        ci_lo = [data_by_sigma[s][cond]['ci'][0] for s in sigma_vals]
        ci_hi = [data_by_sigma[s][cond]['ci'][1] for s in sigma_vals]

        ax.plot(sigma_vals, means, 'o-', color=color, markersize=3.5,
                linewidth=1.2, label=cond.capitalize(), zorder=5)
        ax.fill_between(sigma_vals, ci_lo, ci_hi, color=color, alpha=0.12,
                        zorder=2)

    # Vertical dashed line at selected sigma=0.25
    ax.axvline(x=0.25, color=COLORS['highlight'], linestyle='--',
               linewidth=0.8, alpha=0.8, zorder=3)
    ax.text(0.257, 0.55, r'$\sigma$=0.25', fontsize=6.5,
            color=COLORS['highlight'], ha='left', va='center',
            fontstyle='italic', rotation=90)

    # Axis formatting
    ax.set_xlabel(r'Noise level $\sigma$')
    ax.set_ylabel('Retention')
    ax.set_xlim(0.08, 0.52)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks([0.1, 0.2, 0.25, 0.3, 0.4, 0.5])
    ax.legend(loc='upper right', frameon=False, fontsize=6.5, ncol=1)


# ==============================================================================
# Main
# ==============================================================================

def main():
    set_nature_style()

    seed_data = load_seed_data()
    summary = load_summary()

    # Create figure: 180mm wide, ~130mm tall
    fig_height = 130 / 25.4  # ~5.12 inches
    fig = plt.figure(figsize=(FULL_WIDTH, fig_height))

    # GridSpec: 2 rows, 2 cols; top row spans full width
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[1, 1.2],
                  hspace=0.45, wspace=0.35,
                  left=0.10, right=0.92, top=0.94, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, :])   # full width top
    ax_b = fig.add_subplot(gs[1, 0])   # bottom left
    ax_c = fig.add_subplot(gs[1, 1])   # bottom right

    # Draw panels
    draw_panel_a(ax_a, summary)
    draw_panel_b(ax_b, summary)
    draw_panel_c(ax_c, summary)

    # Panel labels
    panel_label(ax_a, 'a', x=-0.06, y=1.12)
    panel_label(ax_b, 'b', x=-0.18, y=1.08)
    panel_label(ax_c, 'c', x=-0.18, y=1.08)

    # Save
    save_figure(fig, 'fig3_spacing')
    print("Figure 3 generated successfully.")


if __name__ == '__main__':
    main()
