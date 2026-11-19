# Adapted from Rfast 2.1.5.2's Dist and dista (GPL-2 or later), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""Rfast's arithmetic, reproduced exactly.

Rfast::Dist and Rfast::dista sum per-feature terms with Armadillo's accumulate (two
accumulators over even and odd positions, then added), so a plain left-to-right or
numpy pairwise sum rounds differently. The exception is dista's Euclidean kernel over
more than 32 features: Armadillo hands that sum of squares to BLAS ddot, which R's
reference BLAS adds left to right. An optimized BLAS (OpenBLAS, MKL, Accelerate) adds
in its own order, so fixtures record the BLAS they were made with. Among exactly tied
distances nearest_index takes the lowest-numbered center. Rfast::dista(k = 1, index =
TRUE) does too when there are 16 or fewer centers, and with more it can take another of
the tied (docs/compatibility.md).

The distance loops run compiled, in rfastjit. The tests hold them to
tests/compat/rfast_reference.py, where _cross_distances_py does the same arithmetic in
numpy, one feature at a time.
"""

from __future__ import annotations

import numpy as np

from datanuggets._compat import rfastjit as _rfastjit
from datanuggets._compat._compiled import frozen

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
    """Rfast::dista(points, centers, type = metric, k = 1, index = TRUE), 0-based. Among exactly
    tied centers it takes the lowest-numbered one, which is R's pick whenever there are 16 or
    fewer centers."""
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
    return best
