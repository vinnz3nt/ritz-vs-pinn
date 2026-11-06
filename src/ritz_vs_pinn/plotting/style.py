"""Shared figure style for the report.

Intended location: src/ritz_vs_pinn/plotting/style.py

Usage (in a script, once, before creating figures):

    from ritz_vs_pinn.plotting import style
    style.apply_style()

    fig, ax = plt.subplots(figsize=style.figsize(0.5))
    ax.plot(epochs, err, label="Ritz", **style.method_style("ritz"))
    style.save(fig, "training_error")          # -> report/figures/training_error.pdf

This module imports only matplotlib and config. No JAX, no models.
Nothing happens at import time; call apply_style() explicitly.

This code was written by Claude Opus 5.5
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

from ritz_vs_pinn.config import PATHS

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------

# article, 11pt, a4paper, no geometry package -> \textwidth = 360pt.
# Verify by putting \the\textwidth in main.tex once and reading the PDF.
TEXTWIDTH_PT = 360.0
PT_PER_INCH = 72.27  # TeX points, not PostScript points
TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_INCH

GOLDEN = (5**0.5 - 1) / 2  # ~0.618, default height/width for line plots


def figsize(fraction: float = 1.0, aspect: float = GOLDEN) -> tuple[float, float]:
    """(width, height) in inches for a figure `fraction` of the text width.

    Include it in LaTeX at its natural size: \\includegraphics{name}
    (or width=<fraction>\\textwidth). Then fonts come out at the size set below.
    """
    width = fraction * TEXTWIDTH_IN
    return width, width * aspect


def panel_figsize(
    ncols: int, nrows: int = 1, fraction: float = 1.0, panel_aspect: float = 1.0
) -> tuple[float, float]:
    """Size for a grid of panels, e.g. pred / exact / error heatmaps.

    panel_aspect=1.0 gives square panels, right for the unit square domain.
    Colorbars eat some width; nudge panel_aspect down (~0.85) if panels look tall.
    """
    width = fraction * TEXTWIDTH_IN
    return width, (width / ncols) * panel_aspect * nrows


# ---------------------------------------------------------------------------
# Colors: identity follows the method, never the plotting order
# ---------------------------------------------------------------------------

INK = "#0b0b0b"            # text, exact/reference curves
INK_SECONDARY = "#52514e"  # axes, ticks
GRID = "#e4e3df"           # recessive grid lines

# Fixed categorical order (slots 1-4). Assign in this order, don't cycle past it.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

METHOD_STYLES: dict[str, dict] = {
    "ritz":  {"color": CATEGORICAL[0], "marker": "o", "linestyle": "-"},
    "pinn":  {"color": CATEGORICAL[1], "marker": "s", "linestyle": "-"},
    "exact": {"color": INK,            "marker": "",  "linestyle": "--"},
}


def method_style(method: str, markers: bool = True) -> dict:
    """Line kwargs for a method, so Ritz is the same color/marker in every figure.

    Markers give a second, non-color cue (print, colorblind readers).
    For dense curves, pass markevery=... at the call site.
    """
    s = dict(METHOD_STYLES[method.lower()])
    if not markers:
        s["marker"] = ""
    return s


# ---------------------------------------------------------------------------
# Colormaps (pass explicitly: ax.pcolormesh(..., cmap=style.CMAP_FIELD))
# ---------------------------------------------------------------------------

# Single-hue sequential ramp (light -> dark) for magnitudes: u itself, |error|.
_SEQUENTIAL_STEPS = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b",
]
CMAP_FIELD = LinearSegmentedColormap.from_list("rvp_field", _SEQUENTIAL_STEPS)
CMAP_ERROR = CMAP_FIELD  # pair with LogNorm for |error| at the call site

# Signed quantities (pred - exact, residual): two hues, neutral midpoint.
# Use with matplotlib.colors.TwoSlopeNorm(vcenter=0) so 0 sits at the middle.
CMAP_SIGNED = mpl.colormaps["RdBu_r"]


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def apply_style(usetex: bool = False) -> None:
    """Set global matplotlib rcParams for report figures.

    Call once at the top of a plotting script. Not run on import.
    usetex=True renders text with the local LaTeX install (exact font match,
    slower, needs TeX). The default uses matplotlib's bundled cmr10 font.
    """
    mpl.rcParams.update({
        # Fonts: Computer Modern, sized for 11pt body text
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,   # cmr10 has no U+2212 glyph
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,           # titles go in captions, but just in case
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # Ink
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,

        # Recessive frame and grid
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # Marks
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "legend.frameon": False,

        # Layout and output
        "figure.constrained_layout.use": True,
        "figure.dpi": 150,             # on-screen preview
        "savefig.dpi": 300,            # affects rasterized parts only (heatmaps)
        "savefig.format": "pdf",
        "pdf.fonttype": 42,            # embed TrueType, keeps text selectable
    })
    if usetex:
        mpl.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        })


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, name: str, *, final: bool = True, ext: str = ".pdf") -> Path:
    """Save and close a figure.

    final=True  -> report/figures/<name>.pdf  (what LaTeX includes)
    final=False -> figures/<name>.pdf         (exploratory, gitignored)

    No bbox_inches="tight": that would change the size set by figsize() and the
    fonts would no longer match the text. constrained_layout handles margins.
    """
    directory = PATHS.report_figures if final else PATHS.figures
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}{ext}"

    kwargs = {}
    if ext == ".pdf":
        # Drop the timestamp so regenerating an unchanged figure gives an
        # identical file (clean git diffs if report/figures is committed).
        kwargs["metadata"] = {"CreationDate": None}

    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path
