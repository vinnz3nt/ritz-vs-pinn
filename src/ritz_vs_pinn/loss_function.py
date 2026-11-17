import jax 
import equinox as eqx
import jax.random as jxr
import jax.numpy as jnp
import numpy as np

#monte carlo integration
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

#trapezoidal integration
# _GRID_EPS = 0.02  # width of the Gaussian forcing; the grid is refined around 0.5 to resolve it


# def _build_grid(eps=_GRID_EPS, N_coarse=512, N_fine=401):
#     """Sorted 1D grid on [0, 1], dense around 0.5, including the endpoints."""
#     coarse = np.linspace(0.0, 1.0, N_coarse)
#     fine = np.linspace(0.5 - 6 * eps, 0.5 + 6 * eps, N_fine)
#     return np.unique(np.concatenate([coarse, fine]))


# # Built with NumPy because jnp.unique has a data-dependent output size and fails under jit.
# _GRID = jnp.asarray(_build_grid())     # shape (N,), sorted
# _GRID_PTS = _GRID[:, None]             # shape (N, 1), the input shape the network expects
# _BND_PTS = jnp.array([[0.0], [1.0]])   # boundary points, shape (2, 1)


# def _trapz(values):
#     """Trapezoidal integral of values sampled on the internal grid."""
#     return jnp.trapezoid(values, _GRID)


# def scalar(f):
#     return lambda x: jnp.reshape(f(x), ())


# def pinn_loss_fn(network, f, g, interior_points, boundary_points, weight):
#     # interior_points and boundary_points are ignored; the internal grid is used instead
#     f_evaluated = jax.vmap(f)(_GRID_PTS).reshape(-1)
#     laplacian = lambda x: jnp.trace(jax.hessian(network)(x))
#     dubble_network_grad = jax.vmap(laplacian)(_GRID_PTS).reshape(-1)

#     residual = f_evaluated + dubble_network_grad
#     interior_loss_fn = _trapz(jnp.square(residual))

#     f_bnd = jax.vmap(network)(_BND_PTS).reshape(-1)
#     g_bnd = jax.vmap(g)(_BND_PTS).reshape(-1)
#     bnd_loss_fn = jnp.mean(jnp.square(f_bnd - g_bnd))

#     total_loss = interior_loss_fn + weight * bnd_loss_fn
#     return total_loss


# def ritz_loss_fn(network, f, g, interior_points, boundary_points, weight):
#     # interior_points and boundary_points are ignored; the internal grid is used instead
#     grad_u = jax.vmap(jax.grad(network))(_GRID_PTS)
#     f_int = jax.vmap(f)(_GRID_PTS).reshape(-1)
#     u_int = jax.vmap(network)(_GRID_PTS).reshape(-1)

#     energy_density = 0.5 * jnp.sum(grad_u**2, axis=-1) - f_int * u_int
#     interior_loss = _trapz(energy_density)

#     u_bdry = jax.vmap(network)(_BND_PTS).reshape(-1)
#     g_bdry = jax.vmap(g)(_BND_PTS).reshape(-1)
#     boundary_loss = jnp.mean((u_bdry - g_bdry) ** 2)

#     return interior_loss + weight * boundary_loss