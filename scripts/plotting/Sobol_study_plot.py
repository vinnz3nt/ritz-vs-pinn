"""
Plot the results of scripts/studies/Sobol_study.py.

Reads the runs in the newest checkpoints/sobol_study_<timestamp>/ folder and
makes figures/sobol_study/error_histories.pdf: a 2x2 grid with Deep Ritz
(left) and PINN (right) as columns and the relative L2 (top) and H1 (bottom)
error as rows. Each panel has one curve for Sobol sampling and one for
uniform sampling.
"""

import matplotlib.pyplot as plt
import numpy as np

from ritz_vs_pinn.config import PATHS
from ritz_vs_pinn.model_saving import history_unpack, latest_study, study_runs
from ritz_vs_pinn.plotting import plots, style

STUDY = "sobol_study"
METHODS = ("ritz", "pinn")
METHOD_LABELS = {"ritz": "Deep Ritz", "pinn": "PINN"}
METRICS = {"L2": r"relative $L^2$ error", "H1": r"relative $H^1$ error"}
SAMPLERS = ("Uniform", "Sobol")
SAMPLER_COLORS = {"Uniform": style.CATEGORICAL[0], "Sobol": style.CATEGORICAL[1]}


def sampler_of(run):
    """The study names runs Key_<Sampler>_<method>_D<dim>; hparams don't record the sampler."""
    for s in SAMPLERS:
        if run.name.startswith(f"Key_{s}_"):
            return s
    return None


def load_runs(study):
    """{method: {sampler: {"L2": array, "H1": array}}} for every run in the study."""
    runs = {m: {} for m in METHODS}
    for run, hp in study_runs(study):
        method, sampler = hp["loss"], sampler_of(run)
        if sampler is None or method not in runs:
            print(f"Skipping {run.name}: unrecognised run name or method")
            continue
        _, H1, L2 = history_unpack(run)
        if L2 is None or H1 is None:
            print(f"Skipping {run.name}: no error history saved")
            continue
        if sampler in runs[method]:
            raise ValueError(f"Two {method} runs with {sampler} sampling in {study}")
        runs[method][sampler] = {"L2": L2, "H1": H1}

    for m in METHODS:
        if not runs[m]:
            raise FileNotFoundError(f"No '{m}' runs found in {study}")
    return runs


def plot_error_histories(runs):
    fig, axes = plt.subplots(
        len(METRICS), len(METHODS), figsize=style.figsize(1.0, aspect=0.75),
        sharex=True, sharey="row", squeeze=False,
    )
    for col, m in enumerate(METHODS):
        for row, (key, ylabel) in enumerate(METRICS.items()):
            ax = axes[row, col]
            for s in SAMPLERS:
                if s not in runs[m]:
                    continue
                err = runs[m][s][key]
                ax.plot(np.arange(1, len(err) + 1), err, color=SAMPLER_COLORS[s],
                        linewidth=0.8, label=s)
            ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(METRICS) - 1:
                ax.set_xlabel("epoch")
        plots.add_panel_label(axes[0, col], METHOD_LABELS[m])

    # One shared legend instead of one per panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(SAMPLERS))
    return fig


def main():
    style.apply_style()
    study = latest_study(STUDY)
    print(f"Study: {study.name}")
    runs = load_runs(study)
    for m in METHODS:
        print(f"{METHOD_LABELS[m]}: {sorted(runs[m])}")

    (PATHS.figures / STUDY).mkdir(parents=True, exist_ok=True)
    path = style.save(plot_error_histories(runs), f"{STUDY}/error_histories", final=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
