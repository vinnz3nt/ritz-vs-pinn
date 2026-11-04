import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as jnp

def interior_sampler(N, D, rnd_key):
    return jxr.uniform(rnd_key, shape=(N,D),minval=1e-8, maxval=1-1e-8)

def boundary_sampler(N, D, rnd_key):
    k1, k2, k3 = jxr.split(rnd_key, 3)
    samples = jxr.uniform(k1, shape=(N, D))
    side = jxr.randint(k2, shape=(N,), minval=0, maxval=D)
    num = jxr.randint(k3, shape=(N,), minval=0, maxval=2).astype(samples.dtype)
    return samples.at[jnp.arange(N), side].set(num) 