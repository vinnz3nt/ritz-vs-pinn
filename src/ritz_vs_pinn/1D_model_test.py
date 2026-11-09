import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ritz_vs_pinn.config import ACTIVATIONS, KEYS
from ritz_vs_pinn.models import build_network
from ritz_vs_pinn.training import optimize, make_step
from ritz_vs_pinn.loss_function import pinn_loss_fn, ritz_loss_fn
from ritz_vs_pinn.model_saving import model_store, model_unpack
from ritz_vs_pinn.config import MMS_FUNCTIONS

key = KEYS.get('1d_test')

hyper_params = {
    "in_size" : 1,
    "out_size" : 'scalar',
    "width" : 100,
    "depth" : 3,
    "activation" : 'sigmoid',
    "dimension" : 1,
    #"rnd_key" : key,
    "keyword" : '1d_test',
    #"seed " : None,
    "loss" : "ritz",
    "optimizer": "adam",
    "opt_args" : (0.001,),
    "n_epochs" : 1000,
    "weight" : 100,
    "n_interior" : 100,
    "n_boundary" : 2,
    "mms_functions": "1d_sinus_1"
}

u = MMS_FUNCTIONS['1d_sinus_1']['u']

model, hist  = optimize(hyper_params)

model_store('1d_ritz',model,hyper_params)

mesh = jnp.linspace(0,1,100).reshape(100,1)
pred = jax.vmap(model)(mesh)

plt.plot(mesh,u(mesh),label='True function')
plt.plot(mesh,pred,label='Model prediction')
plt.xlabel('Domain')
plt.ylabel('Function value')
plt.grid()
plt.legend()
plt.show()

plt.plot(hist)
plt.show()