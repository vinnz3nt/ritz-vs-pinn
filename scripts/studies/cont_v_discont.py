r"""
This study aims to investigate the performance of ritz vs pinns on two different functions f,
Continuous: general_sinus_sum
Discontinuous: 1 if x \in (0.3, 0.5) otherwise 0

Both problems are -Δu = f on (0, 1)^D with u = g on the boundary, written as a
sum over coordinates so they work for any D (D = 1 gives exactly the problems above):

    continuous:     f(x) = π²/4 Σ_i sin(π x_i / 2),       u(x) = Σ_i sin(π x_i / 2)
    discontinuous:  f(x) = Σ_i 1_{(a, b)}(x_i),           u(x) = Σ_i v(x_i)

where v solves the 1D problem -v'' = 1_{(a, b)}, v(0) = v(1) = 0:

    G(t) = ½ clip(t - a, 0, b - a)² + (b - a) max(t - b, 0)     (G'' = 1_{(a, b)}, G(0) = 0)
    v(t) = G(1) t - G(t)

v is C¹ but v'' jumps at a and b, so the PINN residual (which needs u'') sees the
discontinuity directly, while the Ritz energy only needs u'.

The discontinuous problem is not in config.MMS_FUNCTIONS, so this script adds it at
runtime (training.optimize looks it up in that same dict). Anything that later needs
f, g or u for these runs by name must import this script's DISCONT_FUNCTIONS or
add the entry to config.py.
"""

import jax.numpy as jnp

from ritz_vs_pinn.config import MMS_FUNCTIONS
from ritz_vs_pinn.model_saving import history_store, model_store
from ritz_vs_pinn.training import optimize

STUDY = "cont_v_discont"

# Interval where the discontinuous source is 1
A, B = 0.3, 0.5


def _G(t):
    return 0.5 * jnp.clip(t - A, 0.0, B - A) ** 2 + (B - A) * jnp.maximum(t - B, 0.0)


def _v(t):
    return _G(1.0) * t - _G(t)


DISCONT_NAME = "discontinuous_indicator"
DISCONT_FUNCTIONS = {
    'f': lambda x: jnp.sum(jnp.where((x > A) & (x < B), 1.0, 0.0)),
    'g': lambda x: jnp.sum(_v(x)),
    'u': lambda x: jnp.sum(_v(x)),
}
MMS_FUNCTIONS[DISCONT_NAME] = DISCONT_FUNCTIONS

PROBLEMS = {
    "cont": "general_sinus_sum",
    "discont": DISCONT_NAME,
}
METHODS = ("ritz", "pinn")
DIMENSIONS = (1,)

lr = 1e-3
# Ritz on the discontinuous problem sits at u ≈ 0 for ~5000 epochs at lr 1e-3
# (its exact solution is small, max ≈ 0.043, so the energy gain is tiny), so all
# runs get a budget that lets it escape that plateau.
epochs = 10000
weight = 100
N_interior = 2000
N_boundary = 2000

base_hyper_params = {
    "out_size": 'scalar',
    "width": 100,
    "depth": 3,
    "activation": "sigmoid",
    "keyword": "key",
    "optimizer": "adam",
    "opt_args": (lr,),
    "n_epochs": epochs,
    "weight": weight,
    "n_interior": N_interior,
    "n_boundary": N_boundary,
}


def main():
    for D in DIMENSIONS:
        for problem, mms_name in PROBLEMS.items():
            for method in METHODS:
                hyper_params = dict(base_hyper_params)
                hyper_params["in_size"] = D
                hyper_params["dimension"] = D
                hyper_params["loss"] = method
                hyper_params["mms_functions"] = mms_name

                print(f"{STUDY}: {method}, {problem}, D={D}")
                trained, history = optimize(hyper_params)
                run_path = model_store(f"{STUDY}_{method}_{problem}_{D}", trained, hyper_params)
                history_store(
                    loss=history["loss"],
                    H1_error=history["H1_error"],
                    L2_error=history["L2_error"],
                    run=run_path
                )


if __name__ == "__main__":
    main()
