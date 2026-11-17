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

STUDY = "discont_known_solution"
METHODS = ("ritz", "pinn")
METHOD_LABELS = {"ritz": "Deep Ritz", "pinn": "PINN"}
PROBLEM_LABELS = {"discont": "discontinuous $f$"}
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


def load_runs(mms_name):
    """{method: {"run": Path, "L2": array, "H1": array}}, newest run each."""
    runs = {}
    for run in sorted(PATHS.checkpoints.glob(f"{STUDY}_*")):  # timestamps sort by date
        with open(run / HPARAMS_FILE) as f:
            hp = json.load(f)
        if hp["dimension"] != D or hp["mms_functions"] != mms_name:
            continue
        _, H1, L2 = history_unpack(run)
        if L2 is None or H1 is None:
            print(f"Skipping {run.name}: no error history saved")
            continue
        runs[hp["loss"]] = {"run": run, "L2": L2, "H1": H1}

    missing = [m for m in METHODS if m not in runs]
    if missing:
        raise FileNotFoundError(f"No D={D} {STUDY} runs for {mms_name}: {', '.join(missing)}")
    return runs


def print_final_errors(runs):
    print(f"\nFinal errors, D = {D}: last epoch (saved model) | median of last {TAIL} epochs")
    print(f"{'method':8s}{'L2 last':>11s}{'L2 median':>11s}{'H1 last':>11s}{'H1 median':>11s}")
    for m in METHODS:
        r = runs[m]
        print(f"{m:8s}"
              f"{r['L2'][-1]:11.3e}{np.median(r['L2'][-TAIL:]):11.3e}"
              f"{r['H1'][-1]:11.3e}{np.median(r['H1'][-TAIL:]):11.3e}")
    print()


def rolling_median(y, window):
    """Median over a centred window; returns (epochs, smoothed), shorter than y."""
    windows = np.lib.stride_tricks.sliding_window_view(y, window)
    epochs = np.arange(len(windows)) + window // 2 + 1
    return epochs, np.median(windows, axis=1)


def plot_error_histories(runs):
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(1.0, aspect=0.42), sharex=True)
    for ax, (key, ylabel) in zip(axes, METRICS.items()):
        curves = []
        for m in METHODS:
            raw = runs[m][key]
            # raw history, faint and unlabeled, behind the smoothed line
            ax.plot(np.arange(1, len(raw) + 1), raw, linewidth=0.4, alpha=0.25,
                    zorder=1, **style.method_style(m, markers=False))
            curves.append(Curve(*rolling_median(raw, SMOOTH), m, METHOD_LABELS[m]))
        plots.plot_training_curves(ax, curves, ylabel=ylabel)
    plots.add_panel_label(axes[0], PROBLEM_LABELS["discont"])

    # one legend for the whole figure
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(METHODS))
    return fig


def evaluate(runs, mms_name):
    """Network predictions and exact solution on a fine grid."""
    x = np.linspace(0.0, 1.0, N_PLOT)
    pts = jnp.asarray(x).reshape(-1, 1)
    exact_u = MMS_FUNCTIONS[mms_name]["u"]
    values = {"exact": np.asarray(jax.vmap(exact_u)(pts)).reshape(-1)}
    for m in METHODS:
        net = model_unpack(runs[m]["run"])
        values[m] = np.asarray(jax.vmap(net)(pts)).reshape(-1)
    return x, values


# def _mark_jumps(ax, study):
#     """Shade the interval where the discontinuous source is 1."""
#     ax.axvspan(study.A, study.B, color=style.GRID, alpha=0.6, linewidth=0, zorder=0)


def plot_solutions(x, values, study):
    fig, ax = plt.subplots(figsize=style.figsize(0.5, aspect=0.84))
    preds = [Curve(x, values[m], m, METHOD_LABELS[m]) for m in METHODS]
    plots.plot_slice(ax, x, preds, values["exact"], ylabel=r"$u$")
    lines = ax.get_lines()
    for line in lines:
        line.set_markevery(len(x) // 10)
    lines[0].set_zorder(3)  # exact is drawn first; keep it visible on top
    # _mark_jumps(ax, study)
    plots.add_panel_label(ax, PROBLEM_LABELS["discont"])
    return fig


def plot_pointwise_error(x, values, study):
    fig, ax = plt.subplots(figsize=style.figsize(0.5, aspect=0.84))
    # The error is exactly 0 nowhere in practice, but guard the log axis anyway.
    curves = [
        Curve(x, np.maximum(np.abs(values[m] - values["exact"]), 1e-12), m, METHOD_LABELS[m])
        for m in METHODS
    ]
    plots.plot_curves(ax, curves, xlabel="$x$",
                      ylabel=r"$|u_\theta(x) - u(x)|$", logy=True)
    # _mark_jumps(ax, study)
    plots.add_panel_label(ax, PROBLEM_LABELS["discont"])
    return fig


def main():
    style.apply_style()
    study = load_study_module()
    mms_name = study.PROBLEMS["discont"]  # "discontinuous_indicator"

    runs = load_runs(mms_name)
    for m in METHODS:
        print(f"{m:8s}: {runs[m]['run'].name}")
    print_final_errors(runs)

    x, values = evaluate(runs, mms_name)

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