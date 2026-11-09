import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ritz_vs_pinn.config import checkpoint_path

data = pd.read_csv(checkpoint_path("memory_log_per_dim",".csv"))

print(data)