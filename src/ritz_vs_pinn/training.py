import equinox as eqx
import optax
from tqdm import tqdm
import jax.random as jxr
from ritz_vs_pinn.models import build_network
from ritz_vs_pinn.point_sampler import boundary_sampler, interior_sampler
from ritz_vs_pinn.config import OPTIMIZERS, LOSS, ACTIVATIONS, MMS_FUNCTIONS,KEYS

@eqx.filter_jit
def make_step(network, state, loss_fn, optimizer, mms_functions, weight, N_interior, N_boundary, D, rnd_key):
    k_int, k_bnd = jxr.split(rnd_key)
    interior_points = interior_sampler(N_interior, D, k_int)
    boundary_points = boundary_sampler(N_boundary, D, k_bnd)
    interior_fn, boundary_fn = mms_functions['f'], mms_functions['g']
    loss, grad = eqx.filter_value_and_grad(loss_fn)(network, interior_fn, boundary_fn, interior_points, boundary_points, weight)
    updates, new_state = optimizer.update(grad, state, network)
    new_network = eqx.apply_updates(network, updates)
    return new_network, new_state, loss

def optimize(hyper_params):
    network = build_network(hyper_params)
    optimizer = OPTIMIZERS[hyper_params["optimizer"]](*hyper_params["opt_args"])
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    loss_history = []
    pbar = tqdm(range(hyper_params["n_epochs"]), desc="Training")
    for epoch in pbar:
        rnd_key, step_key = jxr.split(KEYS.get(hyper_params['keyword']))
        network, opt_state, loss = make_step(
            network = network, 
            state = opt_state, 
            loss_fn = LOSS[hyper_params["loss"]], 
            optimizer = optimizer, 
            mms_functions= MMS_FUNCTIONS[hyper_params["mms_functions"]], 
            weight = hyper_params["weight"], 
            N_interior = hyper_params["n_interior"], 
            N_boundary = hyper_params["n_boundary"], 
            D = hyper_params["dimension"], 
            rnd_key=step_key
        )
        pbar.set_postfix(loss=f"{loss:.4f}")
        loss_history.append(loss)
    return network, loss_history