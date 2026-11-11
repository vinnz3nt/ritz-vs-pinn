import jax
import jax.numpy as jnp

#helper
def scalar(f):
    return lambda x: jnp.reshape(f(x), ())

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
# import jax
# import jax.numpy as jnp

# def relative_L2_error(model,exact_function,interior_points,boundary_points):
#     model_eval = jax.vmap(model)(interior_points)
#     exact_eval = jax.vmap(exact_function)(interior_points)

#     interior_int = jnp.mean(jnp.square(exact_eval-model_eval))

#     model_bnd = jax.vmap(model)(boundary_points)
#     exact_bnd = jax.vmap(exact_function)(boundary_points)

#     boundary_int = jnp.mean(jnp.square(exact_bnd-model_bnd))

#     num = jnp.sqrt(interior_int+boundary_int)

#     den = jnp.sqrt(jnp.mean(jnp.square(exact_eval)) + jnp.mean(jnp.square(exact_bnd)))
#     error = num/den
#     return error

# def relative_H1_error(model, exact_function, interior_points, boundary_points):
#     model_eval = jax.vmap(model)(interior_points)
#     exact_eval = jax.vmap(exact_function)(interior_points)

#     model_bnd = jax.vmap(model)(boundary_points)
#     exact_bnd = jax.vmap(exact_function)(boundary_points)

#     grad_model_int = jax.vmap(jax.grad(model))(interior_points)
#     exact_grad_int = jax.vmap(jax.grad(exact_function))(interior_points)

#     grad_model_bnd = jax.vmap(jax.grad(model))(boundary_points)
#     exact_grad_bnd = jax.vmap(jax.grad(exact_function))(boundary_points)

#     err_val_int = jnp.mean(jnp.square(exact_eval-model_eval)) + jnp.mean(jnp.square(exact_grad_int-grad_model_int))
#     err_val_bnd = jnp.mean(jnp.square(exact_bnd-model_bnd)) + jnp.mean(jnp.square(exact_grad_bnd-grad_model_bnd))

#     num = jnp.sqrt(err_val_int+err_val_bnd)
#     den_int = jnp.mean(jnp.square(exact_eval))+jnp.mean(jnp.square(exact_grad_int))
#     den_bnd = jnp.mean(jnp.square(exact_bnd))+jnp.mean(jnp.square(exact_grad_bnd))
#     den = jnp.sqrt(den_int+den_bnd)

#     error = num/den
#     return error
