"""Rfast's arithmetic, reproduced exactly.

Rfast::Dist and Rfast::dista sum per-feature terms with Armadillo's accumulate (two
accumulators over even and odd positions, then added), so a plain left-to-right or
numpy pairwise sum rounds differently. The exception is dista's Euclidean kernel over
more than 32 features: Armadillo hands that sum of squares to BLAS ddot, which R's
reference BLAS adds left to right. An optimized BLAS (OpenBLAS, MKL, Accelerate) adds
in its own order, so fixtures record the BLAS they were made with. Rfast::dista(k = 1,
index = TRUE) picks the nearest center with libstdc++ std::sort, which is not stable:
among exactly tied distances the winner is whatever std::sort leaves first, not the
lowest index.

The distance loops run compiled, in rfastjit, and gnu_sort_first finds only the head of
std::sort's order. The tests hold them to tests/compat/rfast_reference.py: _cross_distances_py
does the same arithmetic in numpy, one feature at a time, and gnu_sort_order replays std::sort's
whole order.
"""

from __future__ import annotations

import numpy as np

from datanuggets._compat import rfastjit as _rfastjit
from datanuggets._compat._compiled import frozen

_S_THRESHOLD = 16  # libstdc++ _S_threshold
_ARMA_DOT_MAX = 32  # Armadillo op_dot::direct_dot: its own loop up to 32 terms, BLAS ddot above


def _dista_sums_in_order(p: int, metric: str, kernel: str) -> bool:
    """dista's Euclidean kernel over more than 32 features adds left to right (BLAS ddot)."""
    return kernel == "dista" and metric == "euclidean" and p > _ARMA_DOT_MAX


def cross_distances(
    a: np.ndarray, b: np.ndarray, metric: str, *, kernel: str = "dist"
) -> np.ndarray:
    """Distances between the rows of a (m x p) and b (k x p), rounded as Rfast::Dist
    (kernel="dist") or Rfast::dista (kernel="dista") rounds them."""
    a = frozen(np.ascontiguousarray(a, dtype=np.float64))
    bt = frozen(np.ascontiguousarray(np.asarray(b, dtype=np.float64).T))
    if bt.shape[0] != a.shape[1]:
        raise ValueError(f"a has {a.shape[1]} columns but b has {bt.shape[0]}")
    if a.shape[1] == 0:  # the kernel would write nothing, and out would keep what memory held
        raise ValueError("a and b have no columns")
    out = np.empty((a.shape[0], bt.shape[1]))
    seq = _dista_sums_in_order(a.shape[1], metric, kernel)
    _rfastjit.cross(a, bt, metric == "manhattan", seq, out)
    return out


def dist_matrix(x: np.ndarray, metric: str) -> np.ndarray:
    """Rfast::Dist(x, method = metric): the full n x n matrix."""
    x = frozen(np.ascontiguousarray(x, dtype=np.float64))
    if x.shape[1] == 0:  # the kernel would write nothing, and out would keep what memory held
        raise ValueError("x has no columns")
    out = np.empty((x.shape[0], x.shape[0]))
    _rfastjit.dist(x, frozen(np.ascontiguousarray(x.T)), metric == "manhattan", out)
    return out


def nearest_index(points: np.ndarray, centers: np.ndarray, metric: str) -> np.ndarray:
    """Rfast::dista(points, centers, type = metric, k = 1, index = TRUE), 0-based."""
    points = frozen(np.ascontiguousarray(points, dtype=np.float64))
    centers = frozen(np.ascontiguousarray(centers, dtype=np.float64))
    if centers.shape[1] != points.shape[1]:
        raise ValueError(
            f"points have {points.shape[1]} columns but centers have {centers.shape[1]}"
        )
    # With no columns the kernel compares unwritten sums, and with no centers it reads past an
    # empty row. Either way best would be garbage.
    if points.shape[1] == 0:
        raise ValueError("points and centers have no columns")
    if centers.shape[0] == 0:
        raise ValueError("there are no centers")
    seq = _dista_sums_in_order(points.shape[1], metric, "dista")
    best = np.empty(points.shape[0], dtype=np.intp)
    ct = frozen(np.ascontiguousarray(centers.T))
    _rfastjit.nearest(points, ct, metric == "manhattan", seq, best)
    tied = np.flatnonzero(best < 0)
    if tied.size:
        d = cross_distances(points[tied], centers, metric, kernel="dista")
        for r, i in enumerate(tied.tolist()):
            best[i] = gnu_sort_first(d[r])
    return best


def gnu_sort_first(values, *, depth_limit: int | None = None) -> int:
    """The index std::sort places first: gnu_sort_order(values)[0], in O(k) instead of O(k log k).

    std::sort loops on the left part of each partition and recurses into the right part, and
    those recursions never touch the left part, so only the partitions along the left edge
    decide the head. The leftmost segment always holds a minimal element (the pivot stays on
    the left), and the final insertion sort is stable, so the head is that segment's first
    minimal element. If the depth limit hits zero on the left edge, the heap sort decides it.
    """
    v = np.asarray(values, dtype=np.float64).tolist()
    n = len(v)
    a = list(range(n))
    depth = 2 * (n.bit_length() - 1) if depth_limit is None else depth_limit
    last = n
    while last > _S_THRESHOLD:
        if depth == 0:
            _heap_sort(a, v, 0, last)
            return a[0]
        depth -= 1
        last = _unguarded_partition_pivot(a, v, 0, last)
    head = min(v[i] for i in a[:last])
    return next(i for i in a[:last] if v[i] == head)


def _move_median_to_first(a, v, result, i, j, k):
    if v[a[i]] < v[a[j]]:
        if v[a[j]] < v[a[k]]:
            t = j
        elif v[a[i]] < v[a[k]]:
            t = k
        else:
            t = i
    elif v[a[i]] < v[a[k]]:
        t = i
    elif v[a[j]] < v[a[k]]:
        t = k
    else:
        t = j
    a[result], a[t] = a[t], a[result]


def _unguarded_partition_pivot(a, v, first, last):
    mid = first + (last - first) // 2
    _move_median_to_first(a, v, first, first + 1, mid, last - 1)
    pivot = v[a[first]]
    lo, hi = first + 1, last
    while True:
        while v[a[lo]] < pivot:
            lo += 1
        hi -= 1
        while pivot < v[a[hi]]:
            hi -= 1
        if not lo < hi:
            return lo
        a[lo], a[hi] = a[hi], a[lo]
        lo += 1


def _heap_sort(a, v, first, last):
    """std::__partial_sort(first, last, last): make_heap, then sort_heap."""
    n = last - first
    if n >= 2:
        parent = (n - 2) // 2
        while True:
            _adjust_heap(a, v, first, parent, n, a[first + parent])
            if parent == 0:
                break
            parent -= 1
    while last - first > 1:
        last -= 1
        val = a[last]
        a[last] = a[first]
        _adjust_heap(a, v, first, 0, last - first, val)


def _adjust_heap(a, v, first, hole, length, val):
    top = hole
    child = hole
    while child < (length - 1) // 2:
        child = 2 * (child + 1)
        if v[a[first + child]] < v[a[first + child - 1]]:
            child -= 1
        a[first + hole] = a[first + child]
        hole = child
    if length % 2 == 0 and child == (length - 2) // 2:
        child = 2 * (child + 1)
        a[first + hole] = a[first + child - 1]
        hole = child - 1
    parent = (hole - 1) // 2
    while hole > top and v[a[first + parent]] < v[val]:
        a[first + hole] = a[first + parent]
        hole = parent
        parent = (hole - 1) // 2
    a[first + hole] = val
