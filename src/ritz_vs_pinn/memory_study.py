import scipy.sparse as sp
import numpy as np
import csv, os, tracemalloc
from ritz_vs_pinn.config import PATHS, checkpoint_path
from ritz_vs_pinn.finite_difference_solver import laplacian_hypercube

HEADER = ["n_per_dim", "dim", "fit", "peak_mb"]
LOG_NAME = "memory_log_per_dim"


def _append_row(log, row):
    new_file = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _write_marker(marker, n_per_dim, dim):
    with open(marker, "w") as f:
        f.write(f"{n_per_dim},{dim}")
        f.flush()
        os.fsync(f.fileno())


def record_killed_runs():
    """Turn markers left behind by killed processes into fit=False rows."""
    log = checkpoint_path(LOG_NAME, ext=".csv")
    for marker in PATHS.checkpoints.glob(f"{LOG_NAME}_pending_*.txt"):
        n_per_dim, dim = marker.read_text().split(",")
        _append_row(log, [int(n_per_dim), int(dim), False, ""])
        marker.unlink()


def log_memory(n_per_dim, dim):
    record_killed_runs()   # any leftover marker means that earlier run was killed

    log = checkpoint_path(LOG_NAME, ext=".csv")
    marker = checkpoint_path(f"{LOG_NAME}_pending_{n_per_dim}_{dim}", ext=".txt")
    expected = n_per_dim ** dim

    _write_marker(marker, n_per_dim, dim)   # exists only while the build runs
    tracemalloc.start()
    try:
        A = laplacian_hypercube(n_per_dim, dim)
        _, peak = tracemalloc.get_traced_memory()
        if A.shape[0] != expected:
            raise ValueError(f"expected {expected} unknowns for {n_per_dim}^{dim}, got {A.shape[0]}")
    except Exception:
        _append_row(log, [n_per_dim, dim, False, ""])   # Python error: record failure
        marker.unlink()
        raise
    finally:
        tracemalloc.stop()

    _append_row(log, [n_per_dim, dim, True, round(peak / 2**20, 3)])
    marker.unlink()   # build finished, so this run is not a kill
    return A

N_dofs = 2
dims = np.arange(2,101,2)
for i in dims:
    log_memory(N_dofs,i)