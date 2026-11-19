"""Rfast's distances in plain Python, the reference rfast is held to.

_cross_distances_py does rfastjit's arithmetic in numpy, one feature at a time: Armadillo's two
accumulators over even and odd features, or left to right where dista hands the sum to BLAS
ddot. Only the tests run it, so it lives here and not in the package.
"""

from collections.abc import Callable

import numpy as np

from datanuggets._compat.rfast import _dista_sums_in_order


def _arma_accumulate(term: Callable[[int], np.ndarray], p: int) -> np.ndarray:
    acc1 = acc2 = None
    i, j = 0, 1
    while j < p:
        t_i, t_j = term(i), term(j)
        acc1 = t_i if acc1 is None else np.add(acc1, t_i, out=acc1)
        acc2 = t_j if acc2 is None else np.add(acc2, t_j, out=acc2)
        i += 2
        j += 2
    if i < p:
        t_i = term(i)
        acc1 = t_i if acc1 is None else np.add(acc1, t_i, out=acc1)
    if acc2 is None:
        return acc1
    return np.add(acc1, acc2, out=acc1)


def arma_sum_last(terms: np.ndarray) -> np.ndarray:
    """Sum over the last axis in Armadillo's order."""
    terms = np.asarray(terms, dtype=np.float64)
    return _arma_accumulate(lambda j: terms[..., j].copy(), terms.shape[-1])


def _sequential_accumulate(term: Callable[[int], np.ndarray], p: int) -> np.ndarray:
    acc = term(0)
    for j in range(1, p):
        np.add(acc, term(j), out=acc)
    return acc


def _cross_distances_py(
    a: np.ndarray, b: np.ndarray, metric: str, *, kernel: str = "dist"
) -> np.ndarray:
    """cross_distances in numpy, one feature at a time. The reference the kernels are tested
    against."""
    if metric == "euclidean":

        def term(j: int) -> np.ndarray:
            d = b[None, :, j] - a[:, None, j]
            return d * d

    else:

        def term(j: int) -> np.ndarray:
            return np.abs(b[None, :, j] - a[:, None, j])

    p = a.shape[1]
    if _dista_sums_in_order(p, metric, kernel):
        total = _sequential_accumulate(term, p)
    else:
        total = _arma_accumulate(term, p)
    if metric == "euclidean":
        np.sqrt(total, out=total)
    return total
