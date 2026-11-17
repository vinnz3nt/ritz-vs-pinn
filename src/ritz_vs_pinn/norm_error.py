import jax
import jax.numpy as jnp
import numpy as np

#helper
def scalar(f):
    return lambda x: jnp.reshape(f(x), ())

#monte carlo integration
def relative_L2_error(model, exact_function, interior_points, boundary_points):
    model_s, exact_s = scalar(model), scalar(exact_function)

    m_int = jax.vmap(model_s)(interior_points)
    e_int = jax.vmap(exact_s)(interior_points)
    m_bnd = jax.vmap(model_s)(boundary_points)
    e_bnd = jax.vmap(exact_s)(boundary_points)

    num = jnp.mean(jnp.square(m_int - e_int)) + jnp.mean(jnp.square(m_bnd - e_bnd))
    den = jnp.mean(jnp.square(e_int)) + jnp.mean(jnp.square(e_bnd))
    return jnp.sqrt(num / den)


def relative_H1_error(model, exact_function, interior_points, boundary_points):
    model_s, exact_s = scalar(model), scalar(exact_function)
    dmodel, dexact = jax.grad(model_s), jax.grad(exact_s)

    def terms(pts):
        N = pts.shape[0]
        m = jax.vmap(model_s)(pts)
        e = jax.vmap(exact_s)(pts)
        dm = jax.vmap(dmodel)(pts).reshape(N, -1)
        de = jax.vmap(dexact)(pts).reshape(N, -1)
        err = jnp.mean(jnp.square(m - e)) + jnp.mean(jnp.sum(jnp.square(dm - de), axis=1))
        norm = jnp.mean(jnp.square(e)) + jnp.mean(jnp.sum(jnp.square(de), axis=1))
        return err, norm

    err_int, norm_int = terms(interior_points)
    err_bnd, norm_bnd = terms(boundary_points)
    return jnp.sqrt((err_int + err_bnd) / (norm_int + norm_bnd))

#with trapezoidal integration
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

# def relative_L2_error(model, exact_function, interior_points, boundary_points):
#     model_s, exact_s = scalar(model), scalar(exact_function)

#     m_int = jax.vmap(model_s)(_GRID_PTS)
#     e_int = jax.vmap(exact_s)(_GRID_PTS)
#     m_bnd = jax.vmap(model_s)(_BND_PTS)
#     e_bnd = jax.vmap(exact_s)(_BND_PTS)

#     num = _trapz(jnp.square(m_int - e_int)) + jnp.mean(jnp.square(m_bnd - e_bnd))
#     den = _trapz(jnp.square(e_int)) + jnp.mean(jnp.square(e_bnd))
#     return jnp.sqrt(num / den)


# def relative_H1_error(model, exact_function, interior_points, boundary_points):
#     model_s, exact_s = scalar(model), scalar(exact_function)
#     dmodel, dexact = jax.grad(model_s), jax.grad(exact_s)

#     def terms(pts):
#         # per-point squared errors and norms, integrated afterwards
#         N = pts.shape[0]
#         m = jax.vmap(model_s)(pts)
#         e = jax.vmap(exact_s)(pts)
#         dm = jax.vmap(dmodel)(pts).reshape(N, -1)
#         de = jax.vmap(dexact)(pts).reshape(N, -1)
#         err = jnp.square(m - e) + jnp.sum(jnp.square(dm - de), axis=1)
#         norm = jnp.square(e) + jnp.sum(jnp.square(de), axis=1)
#         return err, norm

#     err_int, norm_int = terms(_GRID_PTS)
#     err_bnd, norm_bnd = terms(_BND_PTS)

#     num = _trapz(err_int) + jnp.mean(err_bnd)
#     den = _trapz(norm_int) + jnp.mean(norm_bnd)
#     return jnp.sqrt(num / den)

