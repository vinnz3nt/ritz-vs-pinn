import jax
import jax.numpy as jnp

def relative_L2_error(model,true_function,mesh):
    pred = jax.vmap(model)(mesh)
    true_val = jax.vmap(true_function)(mesh)
    # print(true_val.shape)
    num = jnp.sqrt(jnp.sum(jnp.square(pred-true_val)))
    den = jnp.sqrt(jnp.sum(jnp.square(true_val)))
    error = num/den
    return error

#help func
def scalar(f):
    return lambda x: jnp.reshape(f(x), ())

def relative_H1_error(model, true_function, mesh):
    N = mesh.shape[0]
    model_s = scalar(model)
    true_s = scalar(true_function)

    pred = jax.vmap(model_s)(mesh)
    true_val = jax.vmap(true_s)(mesh)
    pred_grad = jax.vmap(jax.grad(model_s))(mesh).reshape(N, -1)
    true_grad = jax.vmap(jax.grad(true_s))(mesh).reshape(N, -1)

    err_val = jnp.sum(jnp.square(pred - true_val))
    err_grad = jnp.sum(jnp.square(pred_grad - true_grad))
    val_norm = jnp.sum(jnp.square(true_val))
    grad_norm = jnp.sum(jnp.square(true_grad))

    return jnp.sqrt((err_val + err_grad) / (val_norm + grad_norm))