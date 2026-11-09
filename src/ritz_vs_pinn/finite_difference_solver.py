import jax
import jax.numpy as jnp
import jax.scipy.sparse as jss
import numpy as np
import scipy.sparse as sp
import csv, os, tracemalloc
from ritz_vs_pinn.config import checkpoint_path

def laplacian_hypercube(dofs, dim):
    if dofs < 1 or dim < 1:
        raise ValueError("dofs and dim must both be >= 1")
 
    n = dofs
    h = 1.0 / (n + 1)
 
    # 1D operator (1/h^2) * tridiag(-1, 2, -1)
    T = sp.diags(
        [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    ) / h**2
 
    # Kronecker sum: A_d = A_{d-1} ⊗ I_n + I_{n^{d-1}} ⊗ T
    A = T
    for _ in range(dim - 1):
        m = A.shape[0]
        A = (
            sp.kron(A, sp.identity(n, format="csr"), format="csr")
            + sp.kron(sp.identity(m, format="csr"), T, format="csr")
        )
 
    return A.tocsr()
