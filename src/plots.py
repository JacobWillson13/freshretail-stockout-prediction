"""Shared matplotlib styling and figure saving."""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.data import ROOT

FIGURES = ROOT / "reports" / "figures"

# categorical slots in fixed order (validated reference palette), plus neutrals
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8985"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
SURFACE = "#fcfcfb"
SEQ_CMAP = "Blues"

CONFIG_COLORS = {"daily": SERIES[0], "weekly": SERIES[1], "category": SERIES[2],
                 "combined": SERIES[3], "today": NEUTRAL}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
        "ytick.color": TEXT_2, "text.color": TEXT, "axes.titlecolor": TEXT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5df", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"],
        "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 110,
    })


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")
