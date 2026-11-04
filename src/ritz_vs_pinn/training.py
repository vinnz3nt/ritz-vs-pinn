import equinox as eqx
import optax
from tqdm import tqdm
import jax.random as jxr
from ritz_vs_pinn.point_sampler import boundary_sampler, interior_sampler

@eqx.filter_jit
def make_step(network, state, loss_fn, optimizer, f, g, weight, N, D, rnd_key):
    k_int, k_bnd = jxr.split(rnd_key)
    interior_points = interior_sampler(N, D, k_int)
    boundary_points = boundary_sampler(N, D, k_bnd)
    loss, grad = eqx.filter_value_and_grad(loss_fn)(network, f, g, interior_points, boundary_points, weight)
    updates, new_state = optimizer.update(grad, state, network)
    new_network = eqx.apply_updates(network, updates)
    return new_network, new_state, loss

def optimize(network, loss_fn, lr, epochs, f, g, weight, N, D, rnd_key):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    loss_history = []
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        rnd_key, step_key = jxr.split(rnd_key)
        network, opt_state, loss = make_step(network, opt_state, loss_fn, optimizer, f, g, weight, N, D, step_key)
        pbar.set_postfix(loss=f"{loss:.4f}")
        loss_history.append(loss)
    return network, loss_history