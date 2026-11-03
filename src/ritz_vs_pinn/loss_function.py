import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as jnp

pringle_key = jxr.PRNGKey(101)

#test func
funcy_wonky = lambda x: jnp.where(x<0.5, 1, 0)
#1D 
def pinn_loss_fn(network, f, g, N, DIMS, ip, bp, weights):
    f_evaluated = f(ip)
    network_evaluated = network(ip)
    dubble_network_grad = jax.grad(jax.grad(network_evaluated))

    #interior loss
    interior_loss_fn = jnp.mean(jnp.square(f_evaluated + dubble_network_grad))

    #boundary loss
    bnd_loos_fn = jnp.mean(jnp.square(network(bp) - g(bp)))

    #full loss function
    total_loss = interior_loss_fn + bnd_loos_fn

    return total_loss

def ritz_loss_fn(network, f, g, interior_points, boundary_points, weight):
    grad_u = jax.vmap(jax.grad(network))(interior_points)
    f_int = jax.vmap(f)(interior_points)
    u_int = jax.vmap(network)(interior_points)
    interior_loss = jnp.mean(0.5 * jnp.sum(grad_u**2, axis=-1) - f_int * u_int)
    
    u_bdry = jax.vmap(network)(boundary_points)
    g_bdry = jax.vmap(g)(boundary_points)
    boundary_loss = jnp.mean((u_bdry - g_bdry) ** 2)

    return interior_loss + weight * boundary_loss


# print(np.shape(pinn_loss_fn(2,funcy_wonky,100,1)[0]))