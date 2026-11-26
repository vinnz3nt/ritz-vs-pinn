"""
Plot the memory log written by src/ritz_vs_pinn/memory_study.py.

Reads checkpoints/memeory_study/memory_log_per_dim.csv (peak memory, from
tracemalloc, of building the second-order finite-difference Laplacian on an
n^D grid) and makes figures/memory_study/fd_laplacian_memory.pdf:

left   peak memory against dimension D, one curve per grid points per
       dimension n
right  the same points against the number of degrees of freedoms N = n^D, which shows
       that the cost is set by N (times the 2D+1 nonzeros per row), not by
       n or D separately

The 'fit' column is ignored. Runs that failed or were killed have no peak
memory and are left out.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ritz_vs_pinn.config import PATHS
from ritz_vs_pinn.plotting import style

LOG = PATHS.checkpoints / "memeory_study" / "memory_log_per_dim.csv"
OUT = "memory_study"


def load_log(path):
    """{n_per_dim: (dims, peak_mb)} sorted by dimension, failed runs dropped."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row["peak_mb"]:
                continue
            n, D = int(row["n_per_dim"]), int(row["dim"])
            rows.setdefault(n, {})[D] = float(row["peak_mb"])
    if not rows:
        raise ValueError(f"No successful runs in {path}")
    return {
        n: (np.array(sorted(by_D)), np.array([by_D[D] for D in sorted(by_D)]))
        for n, by_D in sorted(rows.items())
    }


def n_colors(ns):
    """Sequential ramp: light = coarse grid, dark = fine grid. Skip the palest end."""
    return dict(zip(ns, style.CMAP_FIELD(np.linspace(0.25, 1.0, len(ns)))))


def plot_memory(data):
    colors = n_colors(list(data))
    fig, (ax_D, ax_N) = plt.subplots(
        1, 2, figsize=style.figsize(1.0, aspect=0.45), sharey=True,
    )
    for n, (dims, peak) in data.items():
        kw = dict(color=colors[n], marker="o", markersize=2.5, linewidth=1.0, label=f"$n={n}$")
        ax_D.plot(dims, peak, **kw)
        ax_N.plot(float(n) ** dims, peak, **kw)

    ax_D.set_yscale("log")
    ax_D.set_xlabel("dimension $D$")
    ax_D.set_ylabel("peak memory [MB]")
    ax_D.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_N.set_xscale("log")
    ax_N.set_xlabel("degrees of freedom $N = n^D$")

    handles, labels = ax_D.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", title="points per dim")
    return fig


def main():
    style.apply_style()
    data = load_log(LOG)
    for n, (dims, peak) in data.items():
        print(f"n={n}: D = {dims.min()}..{dims.max()}, peak up to {peak.max():.1f} MB")

    (PATHS.figures / OUT).mkdir(parents=True, exist_ok=True)
    path = style.save(plot_memory(data), f"{OUT}/fd_laplacian_memory", final=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
