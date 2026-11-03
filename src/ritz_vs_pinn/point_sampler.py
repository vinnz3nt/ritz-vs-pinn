import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as jnp

def boundary_sampler(N,D,rnd_key):
    side_nmr = int(2*D)
    def suppport_function(D,it):
        side = jxr.randint(rnd_key+it, shape=(1,), minval=0, maxval=side_nmr, dtype=jnp.int32)
        sample = jxr.uniform(rnd_key+it, shape = (1,D), minval = 0, maxval = 1)
        if side % 2 == 0:
            side = side//2
            sample = sample.at[0,side].set(0)
        else:
            side = side//2
            sample = sample.at[0,side].set(1)
        
        return sample
    
    samples = suppport_function(D,0)
    for i in range(1,N):
        samples = jnp.vstack((samples,suppport_function(D,i)))

    return samples