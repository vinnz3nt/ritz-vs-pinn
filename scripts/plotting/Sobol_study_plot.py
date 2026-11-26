import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from ritz_vs_pinn.config import MMS_FUNCTIONS, study_dir
from ritz_vs_pinn.model_saving import history_store, model_store
from ritz_vs_pinn.training import optimize, optimize_sobol
from ritz_vs_pinn.config import PATHS
from ritz_vs_pinn.model_saving import history_unpack, model_unpack, study_runs, latest_run
from ritz_vs_pinn.norm_error import relative_H1_error, relative_L2_error
from ritz_vs_pinn.plotting import plots, style
from ritz_vs_pinn.point_sampler import boundary_sampler, interior_sampler

STUDY = "sobol_study"

DIM = 10
METHODS = ('ritz','pinn')
IF_SOBEL = [False,True]
KEY = 'Sobol'
