"""WCluster's cluster.predict: the nearest weighted cluster center for new rows."""

from __future__ import annotations

import numpy as np

from datanuggets._compat.linalg import r_matmul, r_matvec
from datanuggets._validate import as_matrix, as_weights
from datanuggets.wcluster.wstats import wmean


def _wdist_mat(x: np.ndarray, centers: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """R's wdist_mat: each row's squared distance to each center, as (|x|^2 + |c|^2) - 2 x.c
    with R's sums in R's order, so a value can be a rounding error below zero, times the row's
    weight. R multiplies by diag(w), so each entry also picks up 0 times every other row of its
    column: it's NaN when another row of its column is NaN or infinite, or when w[i] * z[i, j]
    is NaN. A lone infinity stays infinite. None means unit weights, cluster_predict's case. R's
    sum starts at +0, so no entry comes back as -0."""
    ones = np.ones(x.shape[1])
    with np.errstate(all="ignore"):
        m11 = r_matvec(x * x, ones)
        m22 = r_matvec(centers * centers, ones)
        z = (m11[:, None] + m22[None, :]) - 2 * r_matmul(x, centers.T)
        d = z if w is None else w[:, None] * z + 0.0
    bad = ~np.isfinite(z)
    other_bad = (np.count_nonzero(bad, axis=0)[None, :] - bad) > 0
    return np.where(other_bad | np.isnan(d), np.nan, d)


def cluster_predict(x, cl, newx, *, w=None) -> np.ndarray:
    """The cluster each row of newx is nearest to, as a (m,) intp array of positions.

    The centers are wmean(x, cl, w), and each new row goes to the center at the smallest
    squared distance, computed with R's arithmetic, so the positions match R's cluster.predict.
    A position is 0-based in sorted label order: labels 4, 6 and 8 come back as 0, 1 and 2, and
    a position equals the label only when the labels are 0 to k-1. R returns the 1-based
    position. A tie goes to the first cluster, as R's which.min does. A cluster whose weights
    sum to zero has a center of NaN, or of infinities when weights of both signs cancel, and
    it's never returned while another cluster is at a finite distance. A row can be left with
    every distance NaN. A NaN center gives NaN in its column. R's identity multiply also turns an
    entry NaN when another row of its column is NaN or infinite, which a row of newx that
    overflows causes, and so does a center of infinities once newx has two rows. R then returns
    a list with an empty entry for that row, or integer(0) when no row has an answer, and this
    raises ValueError instead. w is one weight per row of x and is keyword-only, because R's
    order is (x, w, cl, newx) and a call written that way has to fail. None means ones, which is
    R's default too. newx must have the columns of x, and None raises TypeError where R stops
    with "newdata should be a data frame."
    """
    x = as_matrix(x, logical=True, vector=True)
    w = as_weights(w, x.shape[0])
    newx = as_matrix(newx, "newx", logical=True)
    if newx.shape[1] != x.shape[1]:
        raise ValueError(f"newx must have the {x.shape[1]} columns of x, got {newx.shape[1]}")
    z = _wdist_mat(newx, wmean(x, cl, w))
    dead = np.isnan(z)
    if dead.all(axis=1).any():
        raise ValueError(
            "a row of newx has no usable distance to any cluster center: every center is NaN "
            "(its weights sum to zero) or another row of newx overflows"
        )
    pos = np.where(dead, np.inf, z).argmin(axis=1)
    stuck = dead[np.arange(pos.size), pos]  # every live entry is +Inf
    pos[stuck] = (~dead[stuck]).argmax(axis=1)  # which.min takes the first live one
    return pos
