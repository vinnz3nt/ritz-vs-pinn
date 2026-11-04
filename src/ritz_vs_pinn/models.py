import jax
import equinox as eqx
import jax.random as jr
from ritz_vs_pinn.config import ACTIVATIONS

def build_network(hyper_params, init_key):

    net = eqx.nn.MLP(
        in_size=hyper_params["in_size"],
        out_size=hyper_params["out_size"],
        width_size=hyper_params["width"],
        depth=hyper_params["depth"],
        activation=ACTIVATIONS[hyper_params["activation"]],
        key=init_key
        )
    return net
