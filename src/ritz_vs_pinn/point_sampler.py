import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

def interior_sampler(N, D, rnd_key, DIRAC = False):
    samples = jxr.uniform(rnd_key, shape=(N,D), minval=1e-8, maxval=1-1e-8)
    if DIRAC:
        samples = samples.at[0].set(0.5)  #if dirac inject a interoir point here
    return samples

def boundary_sampler(N, D, rnd_key):
    k1, k2, k3 = jxr.split(rnd_key, 3)
    samples = jxr.uniform(k1, shape=(N, D))
    side = jxr.randint(k2, shape=(N,), minval=0, maxval=D)
    num = jxr.randint(k3, shape=(N,), minval=0, maxval=2).astype(samples.dtype)
    return samples.at[jnp.arange(N), side].set(num) 

#creates a uniform grid for error calculatation and comparison
def uniform_mesh(n, dim, lower=0.0, upper=1.0):
    axis = jnp.linspace(lower, upper, n)
    grids = jnp.meshgrid(*([axis] * dim), indexing="ij")
    return jnp.stack([g.ravel() for g in grids], axis=-1)

def key_to_seed(key) -> int:
    """Derive a deterministic 32-bit integer seed from a JAX PRNG key."""
    return int(jax.random.bits(key, dtype=jnp.uint32))

def interior_sobol(N,D,seed):
    m = int(np.floor(np.log(N)/np.log(2)))
    samples = qmc.Sobol(d=D, scramble = False, seed=seed).random_base2(m=m)
    return samples

def boundary_sobol(N,D,key):
    seed = key_to_seed(key)
    k1, k2, k3 = jxr.split(key, 3)
    m = int(np.floor(np.log(N)/np.log(2)))
    samples = qmc.Sobol(d=D, scramble = False, seed=seed).random_base2(m=m)
    side = jxr.randint(k2, shape=(N,), minval=0, maxval=D)
    num = jxr.randint(k3, shape=(N,), minval=0, maxval=2).astype(samples.dtype)
    return samples.at[jnp.arange(N), side].set(num) 