"""
Plot the results of scripts/studies/cont_v_discont.py.

Reads every checkpoints/cont_v_discont_<loss>_<problem>_<D>_<timestamp>/ run
(the newest one if a combination was run more than once), uses the D = 1 runs,
and makes three figures in figures/cont_v_discont/:

  error_histories.pdf  relative L2 and H1 error against epoch, Ritz vs PINN,
                       continuous (left) and discontinuous (right) source
  solutions.pdf        u_theta(x) for both methods against the exact solution
  pointwise_error.pdf  |u_theta(x) - u(x)| on a log axis, shows where the error sits

It also prints a table of the final errors.

The PINN error oscillates by orders of magnitude from epoch to epoch (Adam at
lr 1e-3), so the histories show the raw curve faintly with a rolling median on
top, and the table gives both the last epoch (= the saved model, which is what
the solution plots show) and the median over the last TAIL epochs.

The exact solution of the discontinuous problem is defined in the study script
(it is not in config.MMS_FUNCTIONS), so this script imports it from there.
"""

import importlib.util
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ritz_vs_pinn.config import MMS_FUNCTIONS, PATHS
from ritz_vs_pinn.model_saving import HPARAMS_FILE, history_unpack, model_unpack
from ritz_vs_pinn.plotting import plots, style
from ritz_vs_pinn.plotting.plots import Curve

STUDY = "cont_v_discont"
METHODS = ("ritz", "pinn")
METHOD_LABELS = {"ritz": "Deep Ritz", "pinn": "PINN"}
PROBLEM_LABELS = {"cont": "continuous $f$", "discont": "discontinuous $f$"}
METRICS = {"L2": r"relative $L^2$ error", "H1": r"relative $H^1$ error"}
D = 1

TAIL = 1000     # epochs for the "typical final error" median
SMOOTH = 100    # rolling-median window for the error histories
N_PLOT = 1001  # points for the solution plots


def load_study_module():
    """Import scripts/studies/cont_v_discont.py without running the study.

    Importing it also registers the discontinuous problem in MMS_FUNCTIONS.
    """
    path = PATHS.scripts / "studies" / f"{STUDY}.py"
    spec = importlib.util.spec_from_file_location(f"{STUDY}_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runs(problems):
    """{problem: {method: {"run": Path, "L2": array, "H1": array}}}, newest run each."""
    runs = {p: {} for p in problems}
    for run in sorted(PATHS.checkpoints.glob(f"{STUDY}_*")):  # timestamps sort by date
        with open(run / HPARAMS_FILE) as f:
            hp = json.load(f)
        if hp["dimension"] != D:
            continue
        problem = next((p for p, name in problems.items() if name == hp["mms_functions"]), None)
        if problem is None:
            print(f"Skipping {run.name}: unknown problem '{hp['mms_functions']}'")
            continue
        _, H1, L2 = history_unpack(run)
        if L2 is None or H1 is None:
            print(f"Skipping {run.name}: no error history saved")
            continue
        runs[problem][hp["loss"]] = {"run": run, "L2": L2, "H1": H1}

    missing = [f"{m}/{p}" for p in problems for m in METHODS if m not in runs[p]]
    if missing:
        raise FileNotFoundError(f"No D={D} {STUDY} runs for: {', '.join(missing)}")
    return runs


def print_final_errors(runs):
    print(f"\nFinal errors, D = {D}: last epoch (saved model) | median of last {TAIL} epochs")
    print(f"{'problem':10s}{'method':8s}{'L2 last':>11s}{'L2 median':>11s}{'H1 last':>11s}{'H1 median':>11s}")
    for p in runs:
        for m in METHODS:
            r = runs[p][m]
            print(f"{p:10s}{m:8s}"
                  f"{r['L2'][-1]:11.3e}{np.median(r['L2'][-TAIL:]):11.3e}"
                  f"{r['H1'][-1]:11.3e}{np.median(r['H1'][-TAIL:]):11.3e}")
    print()


def rolling_median(y, window):
    """Median over a centred window; returns (epochs, smoothed), shorter than y."""
    windows = np.lib.stride_tricks.sliding_window_view(y, window)
    epochs = np.arange(len(windows)) + window // 2 + 1
    return epochs, np.median(windows, axis=1)


def plot_error_histories(runs):
    fig, axes = plt.subplots(
        2, 2, figsize=style.figsize(1.0, aspect=0.75), sharex=True, sharey="row"
    )
    for col, p in enumerate(runs):
        for row, (key, ylabel) in enumerate(METRICS.items()):
            ax = axes[row, col]
            curves = []
            for m in METHODS:
                raw = runs[p][m][key]
                # raw history, faint and unlabeled, behind the smoothed line
                ax.plot(np.arange(1, len(raw) + 1), raw, linewidth=0.4, alpha=0.25,
                        zorder=1, **style.method_style(m, markers=False))
                curves.append(Curve(*rolling_median(raw, SMOOTH), m, METHOD_LABELS[m]))
            plots.plot_training_curves(ax, curves, ylabel=ylabel)
            if col == 1:
                ax.set_ylabel("")
            if row == 0:
                ax.set_xlabel("")
        plots.add_panel_label(axes[0, col], PROBLEM_LABELS[p])

    # one legend for the whole figure
    for ax in axes.flat:
        if ax.get_legend():
            ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(METHODS))
    return fig


def evaluate(runs, problems):
    """Network predictions and exact solution on a fine grid, per problem."""
    x = np.linspace(0.0, 1.0, N_PLOT)
    pts = jnp.asarray(x).reshape(-1, 1)
    out = {}
    for p, name in problems.items():
        exact_u = MMS_FUNCTIONS[name]["u"]
        out[p] = {"exact": np.asarray(jax.vmap(exact_u)(pts)).reshape(-1)}
        for m in METHODS:
            net = model_unpack(runs[p][m]["run"])
            out[p][m] = np.asarray(jax.vmap(net)(pts)).reshape(-1)
    return x, out


def _mark_jumps(ax, study):
    """Shade the interval where the discontinuous source is 1."""
    ax.axvspan(study.A, study.B, color=style.GRID, alpha=0.6, linewidth=0, zorder=0)


def plot_solutions(x, values, study):
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(1.0, aspect=0.42))
    for ax, p in zip(axes, values):
        preds = [Curve(x, values[p][m], m, METHOD_LABELS[m]) for m in METHODS]
        plots.plot_slice(ax, x, preds, values[p]["exact"], ylabel=r"$u$")
        lines = ax.get_lines()
        for line in lines:
            line.set_markevery(len(x) // 10)
        lines[0].set_zorder(3)  # exact is drawn first; keep it visible on top
        if p == "discont":
            _mark_jumps(ax, study)
        plots.add_panel_label(ax, PROBLEM_LABELS[p])
    axes[1].get_legend().remove()
    return fig


def plot_pointwise_error(x, values, study):
    fig, axes = plt.subplots(
        1, 2, figsize=style.figsize(1.0, aspect=0.42), sharey=True
    )
    for ax, p in zip(axes, values):
        # The error is exactly 0 nowhere in practice, but guard the log axis anyway.
        curves = [
            Curve(x, np.maximum(np.abs(values[p][m] - values[p]["exact"]), 1e-12), m, METHOD_LABELS[m])
            for m in METHODS
        ]
        plots.plot_curves(ax, curves, xlabel="$x$",
                          ylabel=r"$|u_\theta(x) - u(x)|$", logy=True)
        if p == "discont":
            _mark_jumps(ax, study)
        plots.add_panel_label(ax, PROBLEM_LABELS[p])
    axes[1].set_ylabel("")
    axes[1].get_legend().remove()
    return fig


def main():
    style.apply_style()
    study = load_study_module()
    problems = study.PROBLEMS  # {"cont": "general_sinus_sum", "discont": "discontinuous_indicator"}

    runs = load_runs(problems)
    for p in runs:
        for m in METHODS:
            print(f"{p:8s} {m}: {runs[p][m]['run'].name}")
    print_final_errors(runs)

    x, values = evaluate(runs, problems)

    out = PATHS.figures / STUDY
    out.mkdir(parents=True, exist_ok=True)  # style.save only creates figures/ itself
    for name, fig in (
        ("error_histories", plot_error_histories(runs)),
        ("solutions", plot_solutions(x, values, study)),
        ("pointwise_error", plot_pointwise_error(x, values, study)),
    ):
        path = style.save(fig, f"{STUDY}/{name}", final=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
