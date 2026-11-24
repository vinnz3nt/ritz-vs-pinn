import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from ritz_vs_pinn.config import MMS_FUNCTIONS, study_dir
from ritz_vs_pinn.model_saving import history_store, model_store
from ritz_vs_pinn.training import optimize

STUDY = "small_d_scaling_study"

DIMS = [1,2,3,4]
METHODS = ('ritz','pinn')
N_runs = 15
N = 8

lr = 1e-3
epochs = 5000
weight = 500
# N_interior = 5000
# N_boundary = 5000

hyper_params = {
    "in_size": None,
    "out_size": 'scalar',
    "width": 64,
    "depth": 3,
    "dimension" : None,
    "activation": "tanh",
    "keyword": "small_d_study",
    "optimizer": "adam",
    "opt_args": (lr,),
    "n_epochs": epochs,
    "weight": weight,
    "n_interior": None,
    "n_boundary": None,
    "mms_functions": "general_sinus_sum"
}


def main():
    # One folder per invocation: checkpoints/small_d_scaling_study_<timestamp>/<method>_D<dim>/
    study = study_dir(STUDY)
    for dim in DIMS:
        hyper_params['dimension'] = dim
        hyper_params['in_size'] = dim

        hyper_params['n_interior'] = N**dim
        hyper_params['n_boundary'] = (2*N) * 2 * dim
        for loss_fn in METHODS:
            hyper_params['loss'] = loss_fn
            print(f"Sinus sum: {loss_fn}, D={dim}")
            histories = {"loss": [], "H1_error": [], "L2_error": []}
            for i in range(N_runs):
                hyper_params["keyword"] = f"small_d_study_n{i}"
                print(f"Run [{i+1}/{N_runs}]\t type: {loss_fn}\t dim: {dim}")
                trained_model, history = optimize(hyper_params)
                for k in histories:
                    histories[k].append(np.asarray(history[k]))
            history_averaged = {k: np.mean(v, axis=0) for k, v in histories.items()}

                
            run_path = model_store(f"{loss_fn}_D{dim}", trained_model, hyper_params, parent=study, save_model=False)
            history_store(
                loss=history_averaged["loss"],
                H1_error=history_averaged["H1_error"],
                L2_error=history_averaged["L2_error"],
                run=run_path
            )

if __name__ == "__main__":
    main()