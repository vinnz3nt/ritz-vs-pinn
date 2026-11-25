"""
Average the scaling study over its random seeds and plot the results.

Each (method, D) has one run per key in KEYS. For every (method, D) this script:
  - averages the L2 and H1 error histories over the seeds and plots them
  - evaluates each seed's model on the same 10^5 interior + 10^5 boundary points
    and averages the final relative errors over the seeds

Figures go to figures/<STUDY>/:
    histories.pdf     seed-averaged error vs epoch, one curve per D
    final_error.pdf   seed-averaged final error vs D, Ritz vs PINN
"""

import re
from pathlib import Path

import jax.random as jxr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from ritz_vs_pinn.config import PATHS
from ritz_vs_pinn.model_saving import history_unpack, model_unpack, study_runs, latest_run
from ritz_vs_pinn.norm_error import relative_H1_error, relative_L2_error
from ritz_vs_pinn.plotting import plots, style
from ritz_vs_pinn.point_sampler import boundary_sampler, interior_sampler


STUDY = "scaling_study_N1024"
STUDY_DIR = latest_run(STUDY)

DIMS = [2, 5, 10, 20, 50, 100]
METHODS = ("ritz", "pinn")
KEYS = ["Nr1", "Nr2", "Nr3", "Nr4", "Nr5"]
METHOD_LABELS = {"ritz": "Deep Ritz", "pinn": "PINN"}
METRICS = {"L2": r"relative $L^2$ error", "H1": r"relative $H^1$ error"}

N_INTERIOR = 10**5
N_BOUNDARY = 10**5
EVAL_SEED = 0

RUN_NAME = re.compile(r"Key_(Nr\d+)_(ritz|pinn)_D(\d+)")


from ritz_vs_pinn.config import MMS_FUNCTIONS

MMS_NAME = "general_sinus_sum"  # the "mms_functions" entry in your JSON
exact_function = MMS_FUNCTIONS[MMS_NAME]["u"]

# ---------------------------------------------------------------------------
# Evaluation points
# ---------------------------------------------------------------------------

def draw_points(D):
    """Evaluation points for dimension D, shared by every model of that D."""
    k_int, k_bnd = jxr.split(jxr.PRNGKey(EVAL_SEED), 2)
    interior = interior_sampler(N_INTERIOR, D, k_int)
    boundary = boundary_sampler(N_BOUNDARY, D, k_bnd)
    return interior, boundary


# ---------------------------------------------------------------------------
# Loading and evaluating
# ---------------------------------------------------------------------------

def load_runs():
    """{(method, D): [run_path for each key in KEYS]}."""
    found = {}
    for run, _ in study_runs(STUDY_DIR):
        match = RUN_NAME.fullmatch(run.name)
        if match is None:
            print(f"Skipping {run.name}: name does not match Key_..._<method>_D<dim>")
            continue
        key, method, D = match.group(1), match.group(2), int(match.group(3))
        found.setdefault((method, D), {})[key] = run

    missing = [(m, D, k) for m in METHODS for D in DIMS for k in KEYS
               if k not in found.get((m, D), {})]
    if missing:
        raise FileNotFoundError(f"Missing runs (method, D, key): {missing}")

    return {(m, D): [found[(m, D)][k] for k in KEYS] for m in METHODS for D in DIMS}


def load_histories(run_list):
    """Error histories of all seeds stacked: {"L2": (seeds, epochs), "H1": (seeds, epochs)}."""
    stacks = {"L2": [], "H1": []}
    for run in run_list:
        _, H1, L2 = history_unpack(run)
        if H1 is None or L2 is None:
            raise ValueError(f"{run.name}: no error history saved")
        stacks["L2"].append(L2)
        stacks["H1"].append(H1)
    return {k: np.stack(v) for k, v in stacks.items()}  # fails if seeds differ in length


def evaluate_final_errors(run_list, points):
    """Final relative errors of each seed on the same points: {"L2": (seeds,), "H1": (seeds,)}."""
    interior, boundary = points
    errs = {"L2": [], "H1": []}
    for run in run_list:
        model = model_unpack(run)
        errs["L2"].append(float(relative_L2_error(model, exact_function, interior, boundary)))
        errs["H1"].append(float(relative_H1_error(model, exact_function, interior, boundary)))
    return {k: np.array(v) for k, v in errs.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histories(histories):
    # Sequential ramp: light = low D, dark = high D. Skip the palest end, which is hard to see on white.
    colors = dict(zip(DIMS, style.CMAP_FIELD(np.linspace(0.25, 1.0, len(DIMS)))))

    fig, axes = plt.subplots(
        2, 2, figsize=style.figsize(1.0, aspect=0.75), sharex=True, sharey="row"
    )
    for col, m in enumerate(METHODS):
        for row, (key, ylabel) in enumerate(METRICS.items()):
            ax = axes[row, col]
            for D in DIMS:
                mean_err = histories[(m, D)][key].mean(axis=0)
                ax.plot(np.arange(1, len(mean_err) + 1), mean_err, color=colors[D],
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


def plot_final_error(final):
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(1.0, aspect=0.42))
    for ax, (key, ylabel) in zip(axes, METRICS.items()):
        curves = []
        for m in METHODS:
            y = np.array([final[(m, D)][key].mean() for D in DIMS])
            curves.append(plots.Curve(x=np.array(DIMS), y=y, method=m, label=METHOD_LABELS[m]))
        plots.plot_sweep(ax, curves, xlabel="dimension $D$", ylabel=ylabel, logx=True)
        ax.set_xticks(DIMS, labels=[str(D) for D in DIMS])
        ax.xaxis.set_minor_locator(NullLocator())
    axes[1].get_legend().remove()  # one legend is enough
    return fig


# ---------------------------------------------------------------------------

def main():
    style.apply_style()
    print(f"Study: {STUDY_DIR}")
    runs = load_runs()

    histories = {}  # (method, D) -> {"L2": (seeds, epochs), "H1": (seeds, epochs)}
    final = {}      # (method, D) -> {"L2": (seeds,), "H1": (seeds,)}
    for D in DIMS:
        points = draw_points(D)
        for m in METHODS:
            run_list = runs[(m, D)]
            histories[(m, D)] = load_histories(run_list)
            final[(m, D)] = evaluate_final_errors(run_list, points)
            e = final[(m, D)]
            print(f"{METHOD_LABELS[m]:>9}  D={D:<4} "
                  f"L2 {e['L2'].mean():.3e} ± {e['L2'].std(ddof=1):.1e}   "
                  f"H1 {e['H1'].mean():.3e} ± {e['H1'].std(ddof=1):.1e}")

    out = PATHS.figures / STUDY
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in (
        ("histories", plot_histories(histories)),
        ("final_error", plot_final_error(final)),
    ):
        path = style.save(fig, f"{STUDY}/{name}", final=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()