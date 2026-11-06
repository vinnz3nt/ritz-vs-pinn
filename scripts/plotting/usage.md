# Plotting: quick usage

Files go in `src/ritz_vs_pinn/plotting/` (`style.py`, `plots.py`).
Plot functions take **numpy arrays + an Axes**; they never load data or save.

## Setup (once per script)

```python
import numpy as np
import matplotlib.pyplot as plt
from ritz_vs_pinn.plotting import style, plots
from ritz_vs_pinn.plotting.plots import Curve

style.apply_style()
```

## Loss curve

`optimize()` returns a list of JAX scalars, so convert it first.

```python
loss = np.asarray(loss_history)
epochs = np.arange(len(loss))

fig, ax = plt.subplots(figsize=style.figsize(0.5))
plots.plot_curves(ax, [Curve(epochs, loss, "pinn", "PINN")],
                  xlabel="epoch", ylabel="loss", logy=True)
style.save(fig, "pinn_loss")
```

**Ritz loss can be negative** (it is an energy), so use `logy=False` for it,
and don't put Ritz and PINN losses on the same axes. They aren't comparable.

## Ritz vs PINN over training (error, not loss)

```python
fig, ax = plt.subplots(figsize=style.figsize(0.5))
plots.plot_training_curves(ax, [
    Curve(epochs, err_ritz, "ritz", "Ritz"),
    Curve(epochs, err_pinn, "pinn", "PINN"),
])
style.save(fig, "training_error")
```

## Solution vs exact (3 panels)

`X, Y` are `np.meshgrid` arrays; `U_pred, U_exact` have the same shape.

```python
fig, axes = plt.subplots(1, 3, figsize=style.panel_figsize(3, panel_aspect=0.85))
plots.plot_solution_comparison(axes, X, Y, U_pred, U_exact)
style.save(fig, "ritz_solution")
```

## Single field / signed error

```python
fig, ax = plt.subplots(figsize=style.figsize(0.5, aspect=0.8))
plots.plot_field(ax, X, Y, U_pred, cbar_label="$u_\\theta$")
# or: plots.plot_signed_error(ax, X, Y, U_pred, U_exact)
```

## 1D slice (e.g. y = 0.5)

```python
i = U_exact.shape[0] // 2           # row index where Y = 0.5
s = X[i]
fig, ax = plt.subplots(figsize=style.figsize(0.5))
plots.plot_slice(ax, s, [
    Curve(s, U_ritz[i], "ritz", "Ritz"),
    Curve(s, U_pinn[i], "pinn", "PINN"),
], U_exact[i])
```

## Hyperparameter sweep

```python
widths = np.array([10, 20, 50, 100])
fig, ax = plt.subplots(figsize=style.figsize(0.5))
plots.plot_sweep(ax, [
    Curve(widths, final_err_ritz, "ritz", "Ritz"),
    Curve(widths, final_err_pinn, "pinn", "PINN"),
], xlabel="width", logx=True)
```

## Saving vs. just looking

| Want | Call |
|---|---|
| Report figure → `report/figures/<name>.pdf` | `style.save(fig, name)` |
| Scratch figure → `figures/<name>.pdf` | `style.save(fig, name, final=False)` |
| Look at it interactively | `plt.show()` (don't call `save` first, it closes the figure) |

## Cheat sheet

- **Sizes:** `style.figsize(fraction, aspect)`, where `fraction` is a share of the
  text width (`0.5` = half page). For heatmap grids, use `style.panel_figsize(ncols, nrows)`.
- **Methods:** `"ritz"` (blue, circles), `"pinn"` (orange, squares), `"exact"` (black dashed).
- **No titles.** Describe the figure in the LaTeX caption; panels get (a), (b), (c).
- **Include in LaTeX** at natural size: `\includegraphics{name}` so fonts match the text.
