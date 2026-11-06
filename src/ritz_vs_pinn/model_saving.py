# import jax 
# import equinox as eqx
# import json
# from ritz_vs_pinn.config import PATHS, KEYS, checkpoint_path, figure_path
# from ritz_vs_pinn.models import build_network

# def model_store(name, model, hyper_params):
#     eqx.tree_serialise_leaves(checkpoint_path(name), model)
#     with open(checkpoint_path(name,'.json'),'w') as f:
#         json.dump(hyper_params,f)

# def unpack(name):
#     with open(checkpoint_path(name,'.json'),'r') as f:
#         hyper_params = json.load(f)

#     model_temp = build_network(hyper_params)

#     loaded_model = eqx.tree_deserialise_leaves(checkpoint_path(name),model_temp)

#     return loaded_model

import json
from pathlib import Path
import equinox as eqx
from ritz_vs_pinn.config import PATHS, run_dir
from ritz_vs_pinn.models import build_network

MODEL_FILE = "model.eqx"
HPARAMS_FILE = "hyperparams.json"


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

def latest_run(name) -> Path:
    """Most recent run folder whose name starts with `name`."""
    runs = sorted(PATHS.checkpoints.glob(f"{name}_*"))
    if not runs:
        raise FileNotFoundError(f"No runs found for '{name}'")
    return runs[-1]  # the timestamp format sorts by date
