"""
This study aims to study how the performance of a fixed nn architecture degrades as the dimension increases
"""

from ritz_vs_pinn.models import build_network
from ritz_vs_pinn.training import optimize
from ritz_vs_pinn.model_saving import model_store, history_store
from ritz_vs_pinn.loss_function import pinn_loss_fn, ritz_loss_fn

from ritz_vs_pinn.config import run_dir, figure_path, results_path, KEYS

rnd_key = KEYS.get("key")
lr = 1e-3
epochs = 1000
weight = 100
N_interior = 2000
N_boundary = 2000
D = 2
hyper_params = {
        "in_size": D,
        "out_size": 'scalar',
        "width": 100,
        "depth": 3,
        "activation": "sigmoid", 
        "dimension": D,
        "keyword": "key",
        "loss": "ritz",
        "optimizer": "adam",
        "opt_args":(lr,),
        "n_epochs": epochs,
        "weight": weight,
        "n_interior": N_interior,
        "n_boundary": N_boundary,
        "mms_functions": "general_sinus_sum"
    }

for D in range(1,10):
    hyper_params["dimension"] = D
    hyper_params["in_size"] = D
    
    hyper_params_ritz = dict(hyper_params)
    hyper_params_ritz["loss"] = "ritz"
    
    trained_ritz, history = optimize(hyper_params_ritz)
    run_path = model_store(f"performance_study_ritz_{D}", trained_ritz, hyper_params_ritz)
    history_store(
        loss=history["loss"], 
        H1_error=history["H1_error"], 
        L2_error=history["L2_error"],
        run=run_path
    )
    
    hyper_params_pinn = dict(hyper_params)
    hyper_params_pinn["loss"] = "pinn"

    trained_pinn, history = optimize(hyper_params_pinn)
    run_path = model_store(f"performance_study_pinn_{D}", trained_pinn, hyper_params_pinn)
    history_store(
        loss=history["loss"], 
        H1_error=history["H1_error"], 
        L2_error=history["L2_error"],
        run=run_path
    )



