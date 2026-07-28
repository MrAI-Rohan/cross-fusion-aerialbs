import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# IEEE / Journal plotting style
# ------------------------------------------------------------------

mpl.rcParams.update({

    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",

    # Font sizes
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Figure
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    # Axes
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#D0D0D0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,

    # Lines
    "lines.linewidth": 2.2,
    "lines.markersize": 6,

    # Legend
    "legend.frameon": False,

    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

# Consistent colors across paper
COLORS = {
    "WHU": "#1f77b4",
    "Massachusetts": "#ff7f0e",
    "INRIA": "#2ca02c",
    "Ours": "#d62728",
}

def finish_plot(ax):
    """Apply finishing touches to every plot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")

    plt.tight_layout()