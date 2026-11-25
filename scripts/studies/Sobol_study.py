import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from ritz_vs_pinn.config import MMS_FUNCTIONS, study_dir
from ritz_vs_pinn.model_saving import history_store, model_store
from ritz_vs_pinn.training import optimize

STUDY = "sobol_study"

DIM = 10
METHODS = ('ritz','pinn')
IF_SOBEL = [False,True]
KEY = 'Sobol'

lr = 1e-3
epochs = 10000
weight = 100
N_interior = 1024
N_boundary = 1024

hyper_params = {
    "in_size": DIM,
    "out_size": 'scalar',
    "width": 100,
    "depth": 3,
    "dimension" : DIM,
    "activation": "sigmoid",
    "keyword": KEY,
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
    for loss_fn in METHODS:
        for sampler in IF_SOBEL:
            if sampler:
                Name = 'Sobol'
            else:
                Name = 'Uniform'
            print(f"Sinus sum:{Name}, {loss_fn}, D={DIM}")
            hyper_params['loss'] = loss_fn
            trained_model, history = optimize(hyper_params, SOBOL = sampler)
            run_path = model_store(f"Key_{Name}_{loss_fn}_D{DIM}", trained_model, hyper_params, parent=study)
            history_store(
                loss=history["loss"],
                H1_error=history["H1_error"],
                L2_error=history["L2_error"],
                run=run_path
            )

if __name__ == "__main__":
    main()