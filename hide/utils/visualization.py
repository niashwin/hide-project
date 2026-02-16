"""Figure generation utilities and Nature-quality style settings."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Nature-quality figure parameters
FULL_WIDTH = 180 / 25.4  # 180mm in inches
HALF_WIDTH = 89 / 25.4   # 89mm in inches

COLORS = {
    "primary": "#2171B5",
    "secondary": "#CB181D",
    "tertiary": "#238B45",
    "quaternary": "#6A51A3",
    "human": "#CB181D",
}

BLUES_GRADIENT = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]


def set_nature_style():
    """Set matplotlib parameters for Nature-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
    })


def panel_label(ax, label, x=-0.15, y=1.05):
    """Add a bold panel label (a, b, c, ...) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


def save_figure(fig, path, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
