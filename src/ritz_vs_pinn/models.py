import jax
import equinox as eqx
import jax.random as jr

def build_network(in_size, width, depth, activation, init_key):
    net = eqx.nn.MLP(
        in_size=in_size,
        out_size="scalar",
        width_size=width,
        depth=depth,
        activation=activation,
        key=init_key
        )
    return net
