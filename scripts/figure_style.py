# paper/figure_style.py — import this in EVERY figure script
"""
Global figure style for HIDE project v4 figures.
Nature-quality formatting, colorblind-safe palette, consistent across all figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def set_nature_style():
    """Call this before every figure."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,

        # Lines and markers
        'lines.linewidth': 1.2,
        'lines.markersize': 4,

        # Axes
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Grid (subtle or none)
        'axes.grid': False,

        # Figure
        'figure.dpi': 150,       # screen preview
        'savefig.dpi': 600,      # publication quality
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # No box around legend
        'legend.frameon': False,
    })


# Nature-quality color palette (colorblind-safe)
COLORS = {
    'primary': '#2171B5',       # Steel blue — main results
    'secondary': '#CB181D',     # Crimson — human reference / critical findings
    'tertiary': '#238B45',      # Forest green — secondary conditions
    'quaternary': '#6A51A3',    # Purple — tertiary conditions
    'neutral': '#525252',       # Dark gray — baselines
    'light_neutral': '#969696', # Light gray — minor baselines
    'highlight': '#FF7F00',     # Orange — attention / annotation
    'bg_shade': '#F0F0F0',      # Light gray background for regions
    'human': '#CB181D',         # Same as secondary — always use for human data
}

# Gradient blues for interference dose-response (light → dark)
BLUES_GRADIENT = ['#C6DBEF', '#9ECAE1', '#6BAED6', '#3182BD', '#08519C']

# Consistent markers
MARKERS = {
    'main': 'o',
    'baseline': 's',
    'human': 'D',
    'condition1': 'o',
    'condition2': '^',
    'condition3': 'v',
    'condition4': 'P',
}

# Nature figure widths in mm → inches
FULL_WIDTH = 180 / 25.4    # ~7.09 inches
HALF_WIDTH = 89 / 25.4     # ~3.50 inches
COLUMN_WIDTH = 120 / 25.4  # ~4.72 inches


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add Nature-style panel label (bold lowercase)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')


def human_reference_line(ax, value, label='Human', orientation='horizontal',
                         position='right'):
    """Add dashed human reference line with label."""
    if orientation == 'horizontal':
        ax.axhline(y=value, color=COLORS['human'], linestyle='--',
                   linewidth=0.8, alpha=0.8, zorder=1)
        xlim = ax.get_xlim()
        if position == 'right':
            ax.text(xlim[1], value, f'  {label}',
                    color=COLORS['human'], fontsize=7, ha='left', va='center',
                    fontstyle='italic', clip_on=False)
        else:
            ax.text(xlim[0], value, f'{label}  ',
                    color=COLORS['human'], fontsize=7, ha='right', va='center',
                    fontstyle='italic', clip_on=False)
    else:
        ax.axvline(x=value, color=COLORS['human'], linestyle='--',
                   linewidth=0.8, alpha=0.8, zorder=1)


def save_figure(fig, name, figures_dir=None):
    """Save as both PDF and PNG at 600 DPI."""
    if figures_dir is None:
        # Default to paper/figures relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        figures_dir = os.path.join(project_root, 'paper', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(figures_dir, f'{name}.png'), format='png')
    plt.close(fig)
    print(f"Saved: {figures_dir}/{name}.pdf and .png")
