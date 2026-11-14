"""Chart style shared by every figure (report PDF and Flask app use the same PNGs).

Palette: validated categorical order (blue, orange, aqua, yellow, magenta, green); the chosen model
is always blue and the other candidates neutral gray, so the comparison reads at a glance.
Thin marks, recessive grid, one axis per chart, text in ink colors (never in series color).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIG  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CHOSEN = SERIES[0]
OTHER = "#b9b8b2"
BASE = "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110,
})


def save(fig, name: str) -> str:
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def hbar(labels, values, title, xlabel, chosen=None, baseline=None, err=None, fmt="{:.3f}", name="bar",
         lower_better=False):
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 1.1))
    colors = [CHOSEN if l == chosen else (BASE if l in (baseline or []) else OTHER) for l in labels]
    y = range(len(labels))
    ax.barh(y, values, color=colors, height=0.62, xerr=err, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(values):
        ax.text(v, i, "  " + fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_xlabel(xlabel + ("  (lower is better)" if lower_better else "  (higher is better)"))
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.18)
    return save(fig, name)
