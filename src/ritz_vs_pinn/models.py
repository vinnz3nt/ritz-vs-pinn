import jax
import equinox as eqx
import jax.random as jr

key = jr.PRNGKey(42)
key, init_key = jr.split(key)

def build_network(width, depth, activation):
    net = eqx.nn.MLP(
        in_size="vector",
        out_size="scalar",
        width_size=width,
        depth=depth,
        activation=activation,
        key=init_key
        )
    return net

simple_pinn = eqx.nn.MLP(
    in_size="scalar",
    out_size="scalar",
    width_size=10,
    depth=4,
    activation=jax.nn.sigmoid,
    key=init_key,
)