"""WCluster's weighted cluster statistics: wmean, wss, wwcss and distw.

Each one replays R's arithmetic bit for bit. wmean sums in long double, as R's weighted.mean
does. wss goes through R's matrix product, so its sums are double sums taken row by row in
reference BLAS order. wwcss adds the clusters' values in long double. distw is the wss of two
clusters' rows together minus their wwcss, exactly as R computes it, so rounding can leave a
value at zero or below it.
"""

from __future__ import annotations

import numpy as np

from datanuggets._rcompat import _NA_BITS
from datanuggets._rstats import r_matvec, r_sum, r_weighted_mean
from datanuggets._validate import as_labels, as_matrix, as_weights
from datanuggets.result import WwcssResult


def _inputs(x, cl, w):
    x = as_matrix(x, logical=True)
    w = as_weights(w, x.shape[0])
    levels, codes = as_labels(cl, x.shape[0])
    return x, w, levels.size, codes


def wmean(x, cl, w=None) -> np.ndarray:
    """Each cluster's weighted mean, as a (k, p) float64 array.

    Row i is the center of the i-th distinct label in sorted order, so labels 4, 6 and 8 give
    rows 0, 1 and 2. Each coordinate is R's weighted.mean of that cluster's rows, summed in
    long double, so it matches R bit for bit. A cluster whose weights are all zero gets a row of
    NaN. Weights of both signs that cancel to a zero sum give infinities instead, and NaN in a
    column whose weighted sum is zero too. Both are what R gives. R returns the transpose, p by
    k, with one column per cluster. w is one weight per row of x, and None means ones. R's own
    default, rep(1, length(x)), comes after x turns into a data frame, so it has one weight per
    column of x and stops unless x is square.
    """
    x, w, k, codes = _inputs(x, cl, w)
    out = np.empty((k, x.shape[1]))
    for i in range(k):
        rows = codes == i
        xs, ws = x[rows], w[rows]
        for j in range(x.shape[1]):
            out[i, j] = r_weighted_mean(xs[:, j], ws)
    return out


def _wss(x: np.ndarray, w: np.ndarray) -> float:
    """R's wss: the mean is t(x) %*% w over the long double sum(w), and the result is sum() of
    t((x - mean)^2) %*% w. Weights summing to zero give NaN, as in R."""
    with np.errstate(all="ignore"):
        dev = x - r_matvec(x.T, w) / r_sum(w)
        return r_sum(r_matvec((dev * dev).T, w))


def wss(x, w=None) -> float:
    """The weighted sum of squares of the rows of x around their weighted mean, as a float.

    It's R's wss bit for bit: the mean and the squares go through R's matrix product, and the
    columns' values are added in long double. Weights that sum to zero give NaN, as in R. w is
    one weight per row of x, and None means ones, which is R's default here too.
    """
    x = as_matrix(x, logical=True)
    return _wss(x, as_weights(w, x.shape[0]))


def _group_wss(x: np.ndarray, codes: np.ndarray, k: int, w: np.ndarray) -> np.ndarray:
    return np.array([_wss(x[codes == i], w[codes == i]) for i in range(k)])


def wwcss(x, cl, w=None, group_sum=False):
    """The weighted within-cluster sum of squares: the sum over clusters of each one's wss.

    Returns the total as a float, or with group_sum=True a WwcssResult holding each cluster's
    wss in sorted label order, shape (k,), and the total as a 0-d float64. The total is R's
    long double sum of the clusters' values, which can differ from a double sum in the last
    bit. A cluster whose weights sum to zero is NaN, and so is the total, as in R. w is one
    weight per row of x, and None means ones. R's own default, rep(1, length(x)), comes after
    x turns into a data frame, so it has one weight per column of x and stops unless x is
    square.
    """
    x, w, k, codes = _inputs(x, cl, w)
    groups = _group_wss(x, codes, k, w)
    total = r_sum(groups)
    return WwcssResult(per_cluster=groups, total=total) if group_sum else total


def _distwij(x, codes, w, i: int, j: int, alone: np.ndarray) -> float:
    """R's distwij for clusters j < i: wss of their rows together, in row order, minus R's
    long double sum of the two clusters' own wss."""
    rows = (codes == i) | (codes == j)
    return _wss(x[rows], w[rows]) - r_sum([alone[j], alone[i]])


def distw(x, cl, w=None) -> np.ndarray:
    """The weighted distance between every pair of clusters, as a (k, k) float64 array.

    Entry [i, j] with j < i is the wss of clusters i and j pooled, minus the wss of each on its
    own, with clusters in sorted label order. It's R's value bit for bit, so rounding can leave
    it at zero or a little below. The diagonal and the upper triangle hold R's NA, the NaN with
    payload 1954 (7ff00000000007a2), so np.isnan finds them and a bit-level compare matches R.
    A pair that touches a cluster whose weights sum to zero is NaN, as in R. It needs two
    clusters or more and raises ValueError on one, where R stops with "missing value where
    TRUE/FALSE needed". w is one weight per row of x, and None means ones. R's own default,
    rep(1, length(x)), has one weight per cell of x and stops unless x has one column.
    """
    x, w, k, codes = _inputs(x, cl, w)
    if k < 2:
        raise ValueError("distw needs at least two clusters")
    alone = _group_wss(x, codes, k, w)
    out = np.full((k, k), _NA_BITS, dtype=np.uint64).view(np.float64)  # bits, never arithmetic
    for i in range(1, k):
        for j in range(i):
            out[i, j] = _distwij(x, codes, w, i, j, alone)
    return out
