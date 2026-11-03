import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as np

pringle_key = jxr.PRNGKey(101)

#test func
funcy_wonky = lambda x: np.where(x<0.5, 1, 0)
#1D 
def pinn_loss_fn(network, f, g, N, DIMS, ip, bp, weights):
    f_evaluated = f(ip)
    network_evaluated = network(ip)
    dubble_network_grad = jax.grad(jax.grad(network_evaluated))

    #interior loss
    interior_loss_fn = np.mean(np.square(f_evaluated + dubble_network_grad))

    #boundary loss
    bnd_loos_fn = np.mean(np.square(network(bp) - g(bp)))

    #full loss function
    total_loss = interior_loss_fn + bnd_loos_fn

    return total_loss

def ritz_loss_fn():
    return None


print(np.shape(pinn_loss_fn(2,funcy_wonky,100,1)[0]))