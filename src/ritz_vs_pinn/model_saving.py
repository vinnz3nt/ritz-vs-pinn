import jax 
import equinox as eqx
import json
from ritz_vs_pinn.config import PATHS, KEYS, checkpoint_path, figure_path
from ritz_vs_pinn.models import build_network

def model_store(name, model, hyper_params):
    eqx.tree_serialise_leaves(checkpoint_path(name), model)
    with open(checkpoint_path(name,'.json'),'w') as f:
        json.dump(hyper_params,f)

def unpack(name):
    with open(checkpoint_path(name,'.json'),'r') as f:
        hyper_params = json.load(f)

    model_temp = build_network(hyper_params)

    loaded_model = eqx.tree_deserialise_leaves(checkpoint_path(name),model_temp)

    return loaded_model