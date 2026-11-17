"""
Code that stores and upacks models in checkpoints/

Studies are stored one folder per invocation, with one subfolder per trained model:

    checkpoints/<study>_<timestamp>/
        metrics.csv         optional study-wide metrics, one row per call to metrics_append
        <run>/              model.eqx, hyperparams.json, loss.npy, L2_error.npy, H1_error.npy

Create the study folder with config.study_dir and pass it as `parent` to model_store.
One-off runs that are not part of a study still go straight into
checkpoints/<name>_<timestamp>/ (model_store without `parent`).

This code was written by Claude Opus 5.5
"""

import csv
import json
import os
import re
import numpy as np
from pathlib import Path
import equinox as eqx
from ritz_vs_pinn.config import PATHS, run_dir
from ritz_vs_pinn.models import build_network

MODEL_FILE = "model.eqx"
HPARAMS_FILE = "hyperparams.json"
H1_ERROR_FILE = "H1_error.npy"
L2_ERROR_FILE = "L2_error.npy"
LOSS_FILE = "loss.npy"
METRICS_FILE = "metrics.csv"

# Matches the timestamp config.study_dir appends to a study name
_TIMESTAMP = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"


def _resolve(run) -> Path:
    """`run` can be a path relative to checkpoints/ or a full path."""
    path = Path(run)
    if not path.is_absolute():
        path = PATHS.checkpoints / path
    return path


# ---------------------------------------------------------------------------
# Single runs
# ---------------------------------------------------------------------------

def model_store(name, model, hyper_params, timestamp=True, parent=None) -> Path:
    """Save a model and its hyperparameters.

    With `parent` (a study folder from config.study_dir) the run goes in
    parent/<name>/ without a timestamp, since the study folder already has one.
    Without it, the run goes in checkpoints/<name>[_<timestamp>]/.
    """
    if parent is not None:
        path = Path(parent) / name
        path.mkdir(parents=True, exist_ok=False)  # refuse to overwrite an existing run
    else:
        path = run_dir(name, timestamp)
    eqx.tree_serialise_leaves(path / MODEL_FILE, model)

    with open(path / HPARAMS_FILE, "w") as f:
        json.dump(hyper_params, f, indent=2)

    return path  # handy for printing or logging where the run went


def model_unpack(run):
    """`run` can be a path relative to checkpoints/ or a full path."""
    path = _resolve(run)

    with open(path / HPARAMS_FILE) as f:
        hyper_params = json.load(f)
    model_temp = build_network(hyper_params)

    return eqx.tree_deserialise_leaves(path / MODEL_FILE, model_temp)


def history_store(loss, H1_error, L2_error, run):
    if loss is not None:
        np.save(run / LOSS_FILE, loss)
    if H1_error is not None:
        np.save(run / H1_ERROR_FILE, H1_error)
    if L2_error is not None:
        np.save(run / L2_ERROR_FILE, L2_error)
    print(f"Saved in {run}")


def history_unpack(run):
    """Load the training history saved by `history_store`.

    `run` can be a path relative to checkpoints/ or a full path.
    Returns (loss, H1_error, L2_error); any that weren't saved come back as None.
    """
    path = _resolve(run)

    def _load(filename):
        file = path / filename
        return np.load(file) if file.exists() else None

    loss = _load(LOSS_FILE)
    H1_error = _load(H1_ERROR_FILE)
    L2_error = _load(L2_ERROR_FILE)

    return loss, H1_error, L2_error


def latest_run(name) -> Path:
    """Most recent run folder whose name starts with `name`."""
    runs = sorted(PATHS.checkpoints.glob(f"{name}_*"))
    if not runs:
        raise FileNotFoundError(f"No runs found for '{name}'")
    return runs[-1]  # the timestamp format sorts by date


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def latest_study(name) -> Path:
    """Most recent checkpoints/<name>_<timestamp>/ study folder.

    Matches the name exactly, so "performance_study" does not pick up
    "large_perf_study_..." or "performance_study_ritz_...".
    """
    pattern = re.compile(rf"{re.escape(name)}_{_TIMESTAMP}")
    studies = sorted(
        p for p in PATHS.checkpoints.iterdir()
        if p.is_dir() and pattern.fullmatch(p.name)
    )
    if not studies:
        raise FileNotFoundError(f"No studies named '{name}' in {PATHS.checkpoints}")
    return studies[-1]  # the timestamp format sorts by date


def study_runs(study):
    """(run_path, hyper_params) for every trained model in a study, sorted by folder name.

    `study` can be a path relative to checkpoints/ or a full path.
    """
    for run in sorted(p for p in _resolve(study).iterdir() if (p / HPARAMS_FILE).exists()):
        with open(run / HPARAMS_FILE) as f:
            yield run, json.load(f)


def metrics_append(study, row: dict):
    """Append one row to study/metrics.csv, writing the header on first use.

    For study-specific measurements that don't belong to a single run's
    history. Rows are flushed to disk straight away, so a sweep that crashes
    halfway keeps the rows already written. All rows in a study must have the
    same keys.
    """
    log = _resolve(study) / METRICS_FILE
    new_file = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def metrics_unpack(study):
    """Rows of study/metrics.csv as a list of dicts (values are strings), or [] if absent."""
    log = _resolve(study) / METRICS_FILE
    if not log.exists():
        return []
    with open(log, newline="") as f:
        return list(csv.DictReader(f))
