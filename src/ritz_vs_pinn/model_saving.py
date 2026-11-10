"""
Code that stores and upacks models in checkpoints/ 

This code was written by Claude Opus 5.5
"""

import json
import numpy as np
from pathlib import Path
import equinox as eqx
from ritz_vs_pinn.config import PATHS, run_dir
from ritz_vs_pinn.models import build_network

MODEL_FILE = "model.eqx"
HPARAMS_FILE = "hyperparams.json"
H1_ERROR_FILE = "H1_error.npz"
L2_ERROR_FILE = "L2_error.npz"
LOSS_FILE = "loss.npz"

def model_store(name, model, hyper_params, timestamp=True) -> Path:
    path = run_dir(name, timestamp)
    eqx.tree_serialise_leaves(path / MODEL_FILE, model)
    
    with open(path / HPARAMS_FILE, "w") as f:
        json.dump(hyper_params, f, indent=2)

    return path  # handy for printing or logging where the run went


def model_unpack(run):
    """`run` can be a folder name inside checkpoints/ or a full path."""
    path = Path(run)
    if not path.is_absolute():
        path = PATHS.checkpoints / path
    
    with open(path / HPARAMS_FILE) as f:
        hyper_params = json.load(f)
    model_temp = build_network(hyper_params)

    return eqx.tree_deserialise_leaves(path / MODEL_FILE, model_temp)


def history_store(loss, H1_error, L2_error, run):
    if loss:
        np.save(run / LOSS_FILE, loss)
    if H1_error:
        np.save(run / H1_ERROR_FILE, H1_error)
    if L2_error:
        np.save(run / L2_ERROR_FILE, L2_error)
    print(f"Saved in {run}")


def history_unpack(run):
      """Load the training history saved by `history_store`.

      `run` can be a folder name inside checkpoints/ or a full path.
      Returns (loss, H1_error, L2_error); any that weren't saved come back as None.
      """
      path = Path(run)
      if not path.is_absolute():
          path = PATHS.checkpoints / path

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
