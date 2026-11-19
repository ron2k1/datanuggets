# Adapted from WCluster 1.3.0's cluster.predict (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

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

    Parameters
    ----------
    x : array_like, shape (n, p) or (n,)
        The clustered observations. A 1-D x is one column, and bool data is read as 0 and 1.
    cl : array_like, shape (n,)
        One numeric cluster label per row of ``x``. Labels must be whole numbers smaller than
        1e15 in size and need not be consecutive.
    newx : array_like, shape (m, p)
        The rows to assign, 2-D with the columns of ``x``. Bool data is read as 0 and 1.
    w : array_like, shape (n,), optional
        Keyword-only. One weight per row of ``x``. None means ones.

    Returns
    -------
    numpy.ndarray, shape (m,)
        Each new row's nearest center as an intp position, 0-based in sorted label order.
        ``np.unique(cl)[pos]`` gives the labels.

    Raises
    ------
    TypeError
        If ``x``, ``cl``, ``w`` or ``newx`` is not numeric. A ``newx`` of None counts as not
        numeric. Text and bool labels count as not numeric, and so do bool weights.
    ValueError
        If a row of ``newx`` has no usable distance to any center. Also if ``newx`` is not
        2-D, a 1-D ``newx`` included, does not have the columns of ``x``, or holds NaN or
        infinity. Also, as in ``wmean``, if ``x`` is not 1-D or 2-D, has no columns, or holds
        NaN or infinity, if ``cl`` or ``w`` does not have one entry per row, if ``w`` holds NaN
        or infinity, if ``cl`` is empty, or if a label is not a whole number smaller than 1e15
        in size.

    See Also
    --------
    wmean : The centers this measures against.
    dn_cluster_predict : The same on data nuggets.

    Notes
    -----
    R counterpart: ``WCluster::cluster.predict``, ported from WCluster 1.3.0. Differences
    from R:

    - Positions are 0-based. R returns them 1-based.
    - ``w`` is keyword-only, because R's argument order is (x, w, cl, newx).
    - Where R returns a list with an empty entry for a row with no usable distance, or
      ``integer(0)`` when no row has one, this raises ValueError.
    - A NaN or infinite entry in ``newx`` raises ValueError. R does not stop on a NaN row,
      which empties its whole result.
    - A 1-D ``newx`` raises ValueError. R stops on a vector with "argument is of length
      zero".
    - With integer ``x`` and integer weights, R's integer arithmetic can overflow. For the rows 2e9,
      2e9, 5 and 7 with weights 2, 2, 1 and 1 and labels 1, 1, 2 and 2, R 4.6.1 warns "NAs produced
      by integer overflow" from its call to ``wmean``. That call gives the cluster of label 1 an NA
      center. This reads ``x`` and ``w`` as float64 and computes that center. With a new row of 6,
      R returns 2 and this returns 1, the same cluster. R's ``which.min`` skips the NA distance to
      that center, so R never returns that cluster, while this can.

    References
    ----------
    .. [1] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.
    .. [2] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import RDraws, cluster_predict, wkmeans
    >>> g = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    >>> x = np.vstack([g, g + 5])
    >>> w = np.tile([1, 1, 1, 1, 4], 2)
    >>> cl = wkmeans(x, 2, w=w, num_init=2, seed=RDraws(2)).labels
    >>> cl.tolist()
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> newx = np.array([[0.2, 0.9], [5.5, 4.8], [4.0, 4.0]])
    >>> cluster_predict(x, cl, newx, w=w).tolist()
    [0, 1, 1]
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
