"""Plot functions for report figures.

Intended location: src/ritz_vs_pinn/plotting/plots.py

Rules for this module:
  * Inputs are numpy arrays, an Axes (or list of Axes), and label strings.
  * No file I/O, no JAX, no models. Loading data and saving figures is the
    caller's job (scripts/make_figures.py, or an interactive script).
  * Functions draw on the axes they are given and return the mappable or
    nothing. They never create figures, call plt.show(), or set titles
    (the report puts that text in the caption).

Typical use:

    from ritz_vs_pinn.plotting import style, plots
    style.apply_style()

    fig, axes = plt.subplots(1, 3, figsize=style.panel_figsize(3, panel_aspect=0.85))
    plots.plot_solution_comparison(axes, X, Y, U_pred, U_exact)
    style.save(fig, "ritz_solution")

This Code was written by Claude Opus 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from ritz_vs_pinn.plotting import style


# ---------------------------------------------------------------------------
# Data container for line plots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Curve:
    """One line on a plot.

    x, y   : 1D arrays of equal length
    method : key into style.METHOD_STYLES ("ritz", "pinn", "exact"); decides
             color, marker and line style, so a method looks the same everywhere
    label  : legend text, e.g. "Ritz, width 100"
    """
    x: np.ndarray
    y: np.ndarray
    method: str
    label: str


# ---------------------------------------------------------------------------
# 2D fields on the unit square
# ---------------------------------------------------------------------------

def _format_square_axes(ax: Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(False)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def _colorbar(mappable, ax: Axes, label: str | None = None):
    """Colorbar exactly as tall as the (square, aspect-equal) panel.

    fig.colorbar(ax=ax) sizes the bar to the space allotted to the axes,
    which is taller than the square plot itself. An inset axes follows the
    drawn box instead.
    """
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
    cbar = ax.figure.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label)
    return cbar


def plot_field(
    ax: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    cmap=None,
    norm: Normalize | None = None,
    colorbar: bool = True,
    cbar_label: str | None = None,
):
    """Heatmap of a scalar field Z(X, Y) (meshgrid arrays of equal shape).

    Rasterized so the PDF stays small; axes and text remain vector.
    Returns the QuadMesh so callers can share a colorbar between panels.
    """
    mesh = ax.pcolormesh(
        X, Y, Z,
        cmap=cmap if cmap is not None else style.CMAP_FIELD,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    _format_square_axes(ax)
    if colorbar:
        _colorbar(mesh, ax, cbar_label)
    return mesh


def _log_error_norm(err: np.ndarray) -> Normalize:
    """LogNorm for |error|, robust to exact zeros (e.g. on the boundary)."""
    positive = err[err > 0]
    if positive.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmax = float(err.max())
    vmin = max(float(positive.min()), vmax * 1e-8)  # cap the range at 8 decades
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_solution_comparison(
    axes: Sequence[Axes],
    X: np.ndarray,
    Y: np.ndarray,
    U_pred: np.ndarray,
    U_exact: np.ndarray,
    *,
    log_error: bool = True,
    panel_labels: Sequence[str] | None = ("(a)", "(b)", "(c)"),
) -> None:
    """Three panels: prediction, exact solution, pointwise |error|.

    Prediction and exact share one colour scale and one colorbar, so they
    can be compared by eye. The error panel has its own (log) scale.
    Panel letters stand in for titles; say what each is in the caption.
    """
    if len(axes) != 3:
        raise ValueError(f"expected 3 axes, got {len(axes)}")
    ax_pred, ax_exact, ax_err = axes

    shared = Normalize(
        vmin=float(min(U_pred.min(), U_exact.min())),
        vmax=float(max(U_pred.max(), U_exact.max())),
    )
    m = plot_field(ax_pred, X, Y, U_pred, norm=shared, colorbar=False)
    plot_field(ax_exact, X, Y, U_exact, norm=shared, colorbar=False)
    _colorbar(m, ax_exact, "$u$")  # one bar, right of (b), serves (a) and (b)

    err = np.abs(U_pred - U_exact)
    err_norm = _log_error_norm(err) if log_error else None
    plot_field(ax_err, X, Y, err, cmap=style.CMAP_ERROR, norm=err_norm,
               cbar_label=r"$|u_\theta - u|$")

    # Only the leftmost panel needs a y label.
    for ax in (ax_exact, ax_err):
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    if panel_labels:
        for ax, lab in zip(axes, panel_labels):
            add_panel_label(ax, lab)


def plot_signed_error(
    ax: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    U_pred: np.ndarray,
    U_exact: np.ndarray,
    *,
    cbar_label: str = r"$u_\theta - u$",
):
    """Signed error, diverging colormap centred on zero.

    Shows *where* the network over- or undershoots, which |error| hides.
    """
    diff = U_pred - U_exact
    bound = float(np.abs(diff).max()) or 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    return plot_field(ax, X, Y, diff, cmap=style.CMAP_SIGNED, norm=norm,
                      cbar_label=cbar_label)


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------

def _legend_if_needed(ax: Axes, n_series: int) -> None:
    # A single series is named by the axis label / caption; no legend box.
    if n_series >= 2:
        ax.legend()


def plot_curves(
    ax: Axes,
    curves: Sequence[Curve],
    *,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
    logy: bool = False,
    n_markers: int | None = 10,
) -> None:
    """Generic line plot; the building block for the wrappers below.

    n_markers: roughly how many markers to draw per curve (None = every point).
    Markers are the non-colour cue that separates Ritz from PINN in print.
    """
    for c in curves:
        x, y = np.asarray(c.x), np.asarray(c.y)
        if logy and np.any(y <= 0):
            raise ValueError(
                f"curve '{c.label}' has values <= 0 and cannot go on a log axis. "
                "The Ritz loss is an energy and can be negative; plot the error "
                "against the exact solution instead."
            )
        markevery = None if n_markers is None else max(1, len(x) // n_markers)
        ax.plot(x, y, label=c.label, markevery=markevery,
                **style.method_style(c.method))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _legend_if_needed(ax, len(curves))


def plot_training_curves(
    ax: Axes,
    curves: Sequence[Curve],
    *,
    ylabel: str = r"relative $L^2$ error",
    logy: bool = True,
) -> None:
    """Metric against epoch for one or more runs (x = epochs).

    Compare methods on the error, not the loss: Ritz and PINN losses
    measure different things and are not on the same scale.
    """
    plot_curves(ax, curves, xlabel="epoch", ylabel=ylabel, logy=logy)


def plot_sweep(
    ax: Axes,
    curves: Sequence[Curve],
    *,
    xlabel: str,
    ylabel: str = r"relative $L^2$ error",
    logx: bool = False,
    logy: bool = True,
) -> None:
    """Final error against a hyperparameter (width, depth, N, weight, ...).

    Few points per curve, so every point gets a marker.
    """
    plot_curves(ax, curves, xlabel=xlabel, ylabel=ylabel,
                logx=logx, logy=logy, n_markers=None)


def plot_slice(
    ax: Axes,
    s: np.ndarray,
    predictions: Sequence[Curve],
    u_exact: np.ndarray,
    *,
    xlabel: str = "$x$",
    ylabel: str = "$u$",
) -> None:
    """1D cut through the domain, e.g. along y = 0.5.

    s           : coordinate along the cut
    predictions : one Curve per method (x should equal s)
    u_exact     : exact solution along the cut, drawn black dashed
    """
    exact = Curve(x=s, y=u_exact, method="exact", label="exact")
    plot_curves(ax, [exact, *predictions], xlabel=xlabel, ylabel=ylabel)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def add_panel_label(ax: Axes, text: str) -> None:
    """Put '(a)', '(b)', ... just above the top-left corner of a panel."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes,
            ha="left", va="bottom", color=style.INK)
