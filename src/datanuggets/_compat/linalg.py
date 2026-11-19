"""R's %*% and eigen() of a symmetric matrix.

%*% (a matrix times a vector, or two matrices) is replayed bit for bit as reference BLAS computes
it. eigen() calls LAPACK's dsyevr with R's arguments, so eigenvalues agree with R's to rounding on
the scale of the largest one, not bit for bit: LAPACK builds differ in their kernels. A second
eigenvalue near zero can be far off relative to itself.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, lapack


def r_matvec(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """a %*% v as R computes it with reference BLAS dgemv: y = 0, then y += a[:, j] * v[j]
    column by column, every product and sum rounded to double."""
    y = np.zeros(a.shape[0])
    for j in range(a.shape[1]):
        y += a[:, j] * v[j]
    return y


def r_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a %*% b as R computes it with reference BLAS dgemm: c = 0, then c += a[:, k] * b[k, :]
    for k in order, every product and sum rounded to double. A sum starts at +0, so no entry
    comes back as -0."""
    c = np.zeros((a.shape[0], b.shape[1]))
    for k in range(a.shape[1]):
        c += np.multiply.outer(a[:, k], b[k, :])
    return c


def r_eigen_sym(s: np.ndarray, vectors: bool = False):
    """eigen(s, symmetric = TRUE): LAPACK dsyevr on the lower triangle with R's arguments
    (range "A", abstol 0, optimal workspace), values decreasing. With vectors=True, returns
    (values, vectors), the vectors as columns in the same order."""
    s = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError("infinite or missing values in 'x'")
    lwork, liwork, info = lapack.dsyevr_lwork(s.shape[0], lower=1)
    if info == 0:
        w, z, _, _, info = lapack.dsyevr(
            s,
            compute_v=int(vectors),
            range="A",
            lower=1,
            abstol=0.0,
            lwork=int(lwork),
            liwork=int(liwork),
        )
    if info != 0:
        raise LinAlgError(f"error code {info} from Lapack routine 'dsyevr'")
    values = w[::-1].copy()
    if not vectors:
        return values
    return values, z[:, ::-1].copy()
