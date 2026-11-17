"""
Plot the results of scripts/studies/performance_study.py.

Reads every run in the newest checkpoints/performance_study_<timestamp>/ study
folder and makes two figures in figures/performance_study/:

error_vs_dimension.pdf  final relative L2 and H1 error against D, Ritz vs PINN
error_histories.pdf     relative L2 and H1 error against epoch, one curve per D
"""

import matplotlib.pyplot as plt
import numpy as np

from ritz_vs_pinn.config import PATHS
from ritz_vs_pinn.model_saving import history_unpack, latest_study, study_runs
from ritz_vs_pinn.plotting import plots, style

STUDY = "performance_study"
METHODS = ("ritz", "pinn")
METHOD_LABELS = {"ritz": "Deep Ritz", "pinn": "PINN"}
METRICS = {"L2": r"relative $L^2$ error", "H1": r"relative $H^1$ error"}

# The per-epoch errors are Monte Carlo estimates on freshly sampled points and
# jump around from epoch to epoch, so "final error" is the mean of the last
# TAIL epochs rather than the very last value.
TAIL = 50


def load_runs(study):
    """{method: {D: {"L2": array, "H1": array}}} for every run in the study folder."""
    runs = {m: {} for m in METHODS}
    for run, hp in study_runs(study):
        method, D = hp["loss"], hp["dimension"]
        _, H1, L2 = history_unpack(run)
        if L2 is None or H1 is None:
            print(f"Skipping {run.name}: no error history saved")
            continue
        if D in runs[method]:
            raise ValueError(f"Two {method} runs with D={D} in {study}")
        runs[method][D] = {"L2": L2, "H1": H1}

    for m in METHODS:
        if not runs[m]:
            raise FileNotFoundError(f"No '{m}' runs found in {study}")
    return runs


def plot_error_vs_dimension(runs):
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(1.0, aspect=0.42))
    for ax, (key, ylabel) in zip(axes, METRICS.items()):
        curves = []
        for m in METHODS:
            dims = np.array(sorted(runs[m]))
            final = np.array([runs[m][D][key][-TAIL:].mean() for D in dims])
            curves.append(plots.Curve(x=dims, y=final, method=m, label=METHOD_LABELS[m]))
        plots.plot_sweep(ax, curves, xlabel="dimension $D$", ylabel=ylabel)
        ax.set_xticks(sorted(set().union(*(runs[m] for m in METHODS))))
    axes[1].get_legend().remove()  # one legend is enough
    return fig

def plot_error_histories(runs):
    dims = sorted(set().union(*(runs[m] for m in METHODS)))
    # Sequential ramp: light = low D, dark = high D. Skip the palest end,
    # which is hard to see on white.
    colors = dict(zip(dims, style.CMAP_FIELD(np.linspace(0.25, 1.0, len(dims)))))

    fig, axes = plt.subplots(
        2, 2, figsize=style.figsize(1.0, aspect=0.75), sharex=True, sharey="row"
    )
    for col, m in enumerate(METHODS):
        for row, (key, ylabel) in enumerate(METRICS.items()):
            ax = axes[row, col]
            for D in sorted(runs[m]):
                err = runs[m][D][key]
                ax.plot(np.arange(1, len(err) + 1), err, color=colors[D],
                        linewidth=1.0, label=f"$D={D}$")
            ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("epoch")
        plots.add_panel_label(axes[0, col], METHOD_LABELS[m])


    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    return fig


def main():
    style.apply_style()
    study = latest_study(STUDY)
    print(f"Study: {study.name}")
    runs = load_runs(study)
    for m in METHODS:
        print(f"{METHOD_LABELS[m]}: D = {sorted(runs[m])}")

    out = PATHS.figures / STUDY
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in (
        ("error_vs_dimension", plot_error_vs_dimension(runs)),
        ("error_histories", plot_error_histories(runs)),
    ):
        path = style.save(fig, f"{STUDY}/{name}", final=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()