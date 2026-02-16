#!/usr/bin/env python3
"""
Generate Figure 6: HIDE vs Human Memory — Summary Comparison
Horizontal paired-dot plot with per-phenomenon mini-axes, each with its own scale.

For each phenomenon row:
  - HIDE value shown as blue dot with 95% CI error bar
  - Human value shown as red diamond
  - Connecting line colored by match quality (green=close, orange=moderate, red=poor)

Code generation assisted by Claude (Anthropic).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent dir so we can import figure_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (
    set_nature_style, COLORS, HALF_WIDTH, MARKERS, panel_label, save_figure
)

# ==============================================================================
# Data: phenomena values
# ==============================================================================

phenomena = [
    {
        'name': 'Forgetting\nexponent $b$',
        'hide': 0.460,
        'hide_ci': (0.354, 0.644),
        'human': 0.500,
        'xrange': (0, 0.8),
    },
    {
        'name': 'DRM false\nalarm rate',
        'hide': 0.583,
        'hide_ci': (0.583, 0.583),
        'human': 0.550,
        'xrange': (0, 1.0),
    },
    {
        'name': 'TOT rate\n(%)',
        'hide': 3.66,
        'hide_ci': (3.55, 3.77),
        'human': 1.50,
        'xrange': (0, 5),
    },
    {
        'name': 'Spacing\nlong retention',
        'hide': 0.994,
        'hide_ci': (0.986, 1.0),
        'human': 0.65,
        'xrange': (0, 1.2),
    },
    {
        'name': 'Spacing\nmassed retention',
        'hide': 0.230,
        'hide_ci': (0.157, 0.303),
        'human': 0.30,
        'xrange': (0, 0.5),
    },
]

# Match quality thresholds (based on relative deviation)
MATCH_GOOD = 0.15       # <15% relative difference = green
MATCH_MODERATE = 0.40   # <40% = orange, >=40% = red

MATCH_COLORS = {
    'good': '#238B45',      # Forest green
    'moderate': '#FF7F00',  # Orange
    'poor': '#CB181D',      # Crimson
}


def get_match_color(hide_val, human_val):
    """Determine connecting line color based on relative deviation."""
    if human_val == 0:
        # Avoid division by zero; use absolute difference
        rel_dev = abs(hide_val - human_val)
    else:
        rel_dev = abs(hide_val - human_val) / abs(human_val)
    if rel_dev <= MATCH_GOOD:
        return MATCH_COLORS['good']
    elif rel_dev <= MATCH_MODERATE:
        return MATCH_COLORS['moderate']
    else:
        return MATCH_COLORS['poor']


# ==============================================================================
# Main figure
# ==============================================================================

def main():
    set_nature_style()

    n_rows = len(phenomena)
    fig_width = HALF_WIDTH  # 89mm = ~3.50 inches
    fig_height = 135 / 25.4  # 135mm = ~5.31 inches (more vertical space)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: n_rows rows, 1 column
    # Leave space on the left for phenomenon labels
    gs = GridSpec(
        n_rows, 1, figure=fig,
        hspace=0.70,
        left=0.32, right=0.92, top=0.93, bottom=0.07,
    )

    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]

    for i, (ax, p) in enumerate(zip(axes, phenomena)):
        hide_val = p['hide']
        human_val = p['human']
        ci_lo, ci_hi = p['hide_ci']
        xmin, xmax = p['xrange']

        # Connecting line between HIDE and Human
        match_color = get_match_color(hide_val, human_val)
        ax.plot(
            [hide_val, human_val], [0, 0],
            color=match_color, linewidth=2.5, alpha=0.7, zorder=2,
            solid_capstyle='round',
        )

        # HIDE dot with CI error bar
        ci_err = np.array([[hide_val - ci_lo], [ci_hi - hide_val]])
        ax.errorbar(
            hide_val, 0, xerr=ci_err,
            fmt=MARKERS['main'], color=COLORS['primary'],
            markersize=9, capsize=3, capthick=0.8, linewidth=0.8,
            zorder=5, markeredgecolor='white', markeredgewidth=0.5,
        )

        # Human diamond
        ax.plot(
            human_val, 0,
            marker=MARKERS['human'], color=COLORS['human'],
            markersize=8, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5,
        )

        # Subtle gridlines on x-axis
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='#E0E0E0', linewidth=0.4, linestyle='-')

        # Axis formatting
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

        # Phenomenon label on the left
        ax.set_ylabel(
            p['name'], rotation=0, fontsize=7.5,
            ha='right', va='center', labelpad=8,
        )

        # Only show x-axis label on bottom row
        if i < n_rows - 1:
            ax.set_xticklabels([])

        # Add subtle background shading
        ax.set_facecolor('#FAFAFA')

        # Annotate the match: show percentage difference next to the phenomenon
        # Place annotation to the right of the rightmost point
        rightmost = max(hide_val, human_val)
        diff = hide_val - human_val
        if human_val != 0:
            rel = abs(diff / human_val) * 100
            sign = '+' if diff > 0 else '-' if diff < 0 else ''
            label_text = f'{sign}{rel:.0f}%'
        else:
            label_text = f'{abs(diff):.2f}'

        # Position label slightly right of rightmost point
        x_offset = (xmax - xmin) * 0.04
        ax.text(
            rightmost + x_offset, 0.0, label_text,
            fontsize=7, ha='left', va='center',
            color=match_color, fontweight='bold',
        )

    # Legend at top: HIDE (blue dot) and Human (red diamond)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=MARKERS['main'], color='w',
               markerfacecolor=COLORS['primary'], markersize=9,
               markeredgecolor='white', markeredgewidth=0.5,
               label='HIDE'),
        Line2D([0], [0], marker=MARKERS['human'], color='w',
               markerfacecolor=COLORS['human'], markersize=8,
               markeredgecolor='white', markeredgewidth=0.5,
               label='Human'),
    ]

    # Add match-quality legend entries
    for quality, color in MATCH_COLORS.items():
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, alpha=0.7,
                   label=f'{quality.capitalize()} match')
        )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=5, fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.62, 0.99),
        columnspacing=1.0,
        handletextpad=0.4,
    )

    # Save
    save_figure(fig, 'fig6_summary')
    print("Figure 6 generated successfully.")


if __name__ == '__main__':
    main()
