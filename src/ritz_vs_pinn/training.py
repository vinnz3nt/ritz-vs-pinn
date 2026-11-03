import equinox as eqx
import optax
from tqdm import tqdm

@eqx.filter_jit
def make_step(network, state, loss_fn, optimizer):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(network)
    updates, new_state = optimizer.update(grad, state, network)
    new_network = eqx.apply_updates(network, updates)
    return new_network, new_state, loss

def optimize(network, loss_fn, lr, epochs):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))


    loss_history = []
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        pinn, opt_state, loss = make_step(pinn, opt_state, loss_fn, optimizer)
        pbar.set_postfix(loss=f"{loss:.4f}")
        loss_history.append(loss)
    return loss_history