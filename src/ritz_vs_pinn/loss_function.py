import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as jnp

def pinn_loss_fn(network, f, g, interior_points, boundary_points, weight):
    f_evaluated = jax.vmap(f)(interior_points).reshape(-1)
    # dubble_network_grad = jax.vmap(jax.grad(jax.grad(network)))(interior_points)
    laplacian = lambda x: jnp.trace(jax.hessian(network)(x))
    dubble_network_grad = jax.vmap(laplacian)(interior_points).reshape(-1)

    #interior loss
    interior_loss_fn = jnp.mean(jnp.square(f_evaluated + dubble_network_grad))

    #boundary loss
    f_bnd = jax.vmap(network)(boundary_points).reshape(-1)
    g_bnd = jax.vmap(g)(boundary_points).reshape(-1)
    bnd_loss_fn = jnp.mean(jnp.square(f_bnd - g_bnd))

    #full loss function
    total_loss = interior_loss_fn + weight * bnd_loss_fn

    return total_loss

def ritz_loss_fn(network, f, g, interior_points, boundary_points, weight):
    grad_u = jax.vmap(jax.grad(network))(interior_points)
    f_int = jax.vmap(f)(interior_points).reshape(-1)
    u_int = jax.vmap(network)(interior_points).reshape(-1)
    interior_loss = jnp.mean(0.5 * jnp.sum(grad_u**2, axis=-1) - f_int * u_int)
    
    u_bdry = jax.vmap(network)(boundary_points).reshape(-1)
    g_bdry = jax.vmap(g)(boundary_points).reshape(-1)
    boundary_loss = jnp.mean((u_bdry - g_bdry) ** 2)

    return interior_loss + weight * boundary_loss