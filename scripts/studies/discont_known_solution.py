import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from ritz_vs_pinn.config import MMS_FUNCTIONS
from ritz_vs_pinn.model_saving import history_store, model_store
from ritz_vs_pinn.training import optimize

eps = 10**-6
pi = jnp.pi

STUDY = "discont_known_solution"
PROBLEM_NAME = "Approx_dirac"

#approximate the dirac delta function with a gaussian with very small variance
#than the exact solution is u = min(x,1-x)/2
#the BC are just 0
DISCONT_NAME = "known_discont_fn"
DISCONT_FN = {
    "f" : lambda x: (1/(eps*jnp.sqrt(2*pi))) * jnp.exp(-0.5* ((x-0.5)**2) / (eps**2) ),
    "g" : lambda x: 0,
    "u" : lambda x: 0.5*jnp.minimum(x,1-x) 
    }

PROBLEMS = {
    "discont": DISCONT_NAME,
}

MMS_FUNCTIONS[DISCONT_NAME] = DISCONT_FN

lr = 1e-3
epochs = 10000
weight = 100
N_interior = 2000
N_boundary = 2000
DIM = 1
hyper_params = {
    "in_size": DIM,
    "out_size": 'scalar',
    "width": 100,
    "depth": 4,
    "dimension" : DIM,
    "activation": "sigmoid",
    "keyword": "key_2",
    "optimizer": "adam",
    "opt_args": (lr,),
    "n_epochs": epochs,
    "weight": weight,
    "n_interior": N_interior,
    "n_boundary": N_boundary,
    "mms_functions": DISCONT_NAME
}

METHODS = ('ritz','pinn')

mesh = jnp.linspace(0,1,100)

def main():
    for loss_fn in METHODS:
        hyper_params['loss'] = loss_fn
        print(f"{STUDY}: {loss_fn}, {PROBLEM_NAME}, D={1}")
        trained, history = optimize(hyper_params, DIRAC = True)
        run_path = model_store(f"{STUDY}_{loss_fn}_{PROBLEM_NAME}_D{1}", trained, hyper_params)
        history_store(
            loss=history["loss"],
            H1_error=history["H1_error"],
            L2_error=history["L2_error"],
            run=run_path
        )


if __name__ == "__main__":
    main()