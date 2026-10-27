"""Rfast's distances and libstdc++'s std::sort in plain Python, the references rfast is held to.

_cross_distances_py does rfastjit's arithmetic in numpy, one feature at a time: Armadillo's two
accumulators over even and odd features, or left to right where dista hands the sum to BLAS
ddot. gnu_sort_order replays std::sort's whole order, step for step. rfast's gnu_sort_first finds
only its head, and the two share rfast's partition and heap sort. Only the tests run these, so
they live here and not in the package.
"""

from collections.abc import Callable

import numpy as np

from datanuggets._compat.rfast import (
    _S_THRESHOLD,
    _dista_sums_in_order,
    _heap_sort,
    _unguarded_partition_pivot,
)


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


def gnu_sort_order(values, *, depth_limit: int | None = None) -> np.ndarray:
    """Order of indices 0..k-1 after libstdc++ std::sort with comparator v[i] < v[j].

    Reproduces introsort (median-of-three pivot, unguarded partition, heapsort fallback)
    followed by the final insertion sort, step for step, so ties come out as in R.
    """
    v = np.asarray(values, dtype=np.float64).tolist()
    a = list(range(len(v)))
    n = len(a)
    if n > 1:
        depth = 2 * (n.bit_length() - 1) if depth_limit is None else depth_limit
        _introsort_loop(a, v, 0, n, depth)
        _final_insertion_sort(a, v, 0, n)
    return np.asarray(a, dtype=np.intp)


def _introsort_loop(a, v, first, last, depth):
    while last - first > _S_THRESHOLD:
        if depth == 0:
            _heap_sort(a, v, first, last)
            return
        depth -= 1
        cut = _unguarded_partition_pivot(a, v, first, last)
        _introsort_loop(a, v, cut, last, depth)
        last = cut


def _final_insertion_sort(a, v, first, last):
    if last - first > _S_THRESHOLD:
        _insertion_sort(a, v, first, first + _S_THRESHOLD)
        for i in range(first + _S_THRESHOLD, last):
            _unguarded_linear_insert(a, v, i)
    else:
        _insertion_sort(a, v, first, last)


def _insertion_sort(a, v, first, last):
    if first == last:
        return
    for i in range(first + 1, last):
        val = a[i]
        if v[val] < v[a[first]]:
            a[first + 1 : i + 1] = a[first:i]
            a[first] = val
        else:
            _unguarded_linear_insert(a, v, i)


def _unguarded_linear_insert(a, v, last):
    val = a[last]
    nxt = last - 1
    while v[val] < v[a[nxt]]:
        a[last] = a[nxt]
        last = nxt
        nxt -= 1
    a[last] = val
