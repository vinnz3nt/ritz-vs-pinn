import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from ritz_vs_pinn.config import MMS_FUNCTIONS, study_dir
from ritz_vs_pinn.model_saving import history_store, model_store
from ritz_vs_pinn.training import optimize

#start traning with 1024 punkter
STUDY = "scaling_study_N1024"

DIMS = [2,5,10,20,50,100]
METHODS = ('ritz','pinn')
KEYS = ['Nr1','Nr2','Nr3','Nr4','Nr5']

lr = 1e-3
epochs = 10000
weight = 100
N_interior = 1024
N_boundary = 1024

hyper_params = {
    "in_size": None,
    "out_size": 'scalar',
    "width": 100,
    "depth": 3,
    "dimension" : None,
    "activation": "sigmoid",
    "keyword": None,
    "optimizer": "adam",
    "opt_args": (lr,),
    "n_epochs": epochs,
    "weight": weight,
    "n_interior": N_interior,
    "n_boundary": N_boundary,
    "mms_functions": "general_sinus_sum"
}

def main():
    # One folder per invocation: checkpoints/scaling_study_<timestamp>/<method>_D<dim>/
    study = study_dir(STUDY)
    for dim in DIMS:
        hyper_params['dimension'] = dim
        hyper_params['in_size'] = dim
        for loss_fn in METHODS:
            hyper_params['loss'] = loss_fn
            for keyword in KEYS:
                hyper_params['keyword'] = keyword
                print(f"Sinus sum: {loss_fn}, D={dim}, current_key={keyword}")
                trained_model, history = optimize(hyper_params)
                run_path = model_store(f"Key_{keyword}_{loss_fn}_D{dim}", trained_model, hyper_params, parent=study)
                history_store(
                    loss=history["loss"],
                    H1_error=history["H1_error"],
                    L2_error=history["L2_error"],
                    run=run_path
                )

if __name__ == "__main__":
    main()