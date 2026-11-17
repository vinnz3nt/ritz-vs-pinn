import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from ritz_vs_pinn.config import MMS_FUNCTIONS
from ritz_vs_pinn.model_saving import history_store, model_store
from ritz_vs_pinn.training import optimize

DIMS = [2,5,10,20,50,100]
METHODS = ('ritz','pinn')

lr = 1e-3
epochs = 10000
weight = 100
N_interior = 2000
N_boundary = 2000

hyper_params = {
    "in_size": None,
    "out_size": 'scalar',
    "width": 100,
    "depth": 3,
    "dimension" : None,
    "activation": "sigmoid",
    "keyword": "scaling_study",
    "optimizer": "adam",
    "opt_args": (lr,),
    "n_epochs": epochs,
    "weight": weight,
    "n_interior": N_interior,
    "n_boundary": N_boundary,
    "mms_functions": "general_sinus_sum"
}


def main():
    for dim in DIMS:
        hyper_params['dimension'] = dim
        hyper_params['in_size'] = dim
        for loss_fn in METHODS:
            hyper_params['loss'] = loss_fn
            print(f"Sinus sum: {loss_fn}, D={dim}")
            trained_model, history = optimize(hyper_params)
            run_path = model_store(f"{dim}D_{loss_fn}_sinus_sum", trained_model, hyper_params)
            history_store(
                loss=history["loss"],
                H1_error=history["H1_error"],
                L2_error=history["L2_error"],
                run=run_path
            )

if __name__ == "__main__":
    main()