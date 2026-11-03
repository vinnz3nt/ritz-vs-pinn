import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as jnp

def interior_sampler(N, D, rnd_key):
    return jxr.uniform(rnd_key, shape=(N,D),minval=1e-8, maxval=1-1e-8)


def boundary_sampler(N, D, rnd_key):
    def suppport_function(D,it):
        side = jxr.randint(rnd_key+it, shape=(1,), minval=0, maxval=D, dtype=jnp.int32)
        num = jxr.randint(rnd_key+it, shape=(1,), minval=0, maxval=2, dtype=jnp.int32)
        sample = jxr.uniform(rnd_key+it, shape = (1,D), minval = 0, maxval = 1)
        
        sample = sample.at[0,side].set(num)
        return sample
    
    samples = suppport_function(D,0)
    for i in range(1,N):
        samples = jnp.vstack((samples,suppport_function(D,i)))

    return samples