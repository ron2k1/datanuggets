# Adapted from WCluster 1.3.0's wmean, wss, wwcss and distw (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""WCluster's weighted cluster statistics: wmean, wss, wwcss and distw.

Each one replays R's arithmetic bit for bit. wmean sums in long double, as R's weighted.mean
does. wss goes through R's matrix product, so its sums are double sums taken row by row in
reference BLAS order. wwcss adds the clusters' values in long double. distw is the wss of two
clusters' rows together minus their wwcss, exactly as R computes it, so rounding can leave a
value at zero or below it.
"""

from __future__ import annotations

import numpy as np

from datanuggets._compat.arithmetic import r_sum, r_weighted_mean
from datanuggets._compat.linalg import r_matvec
from datanuggets._compat.missing import _NA_BITS
from datanuggets._validate import as_labels, as_matrix, as_weights
from datanuggets.result import WwcssResult


def _inputs(x, cl, w):
    x = as_matrix(x, logical=True, vector=True)
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

    Parameters
    ----------
    x : array_like, shape (n, p) or (n,)
        Numeric observations. A 1-D x is one column, and bool data is read as 0 and 1.
    cl : array_like, shape (n,)
        One numeric cluster label per row. Labels must be whole numbers smaller than 1e15 in
        size and need not be consecutive.
    w : array_like, shape (n,), optional
        One weight per row. None means ones.

    Returns
    -------
    numpy.ndarray, shape (k, p)
        Row i holds the weighted mean of the i-th distinct label in sorted order.

    Raises
    ------
    TypeError
        If ``x``, ``cl`` or ``w`` is not numeric. Text and bool labels count as not numeric,
        and so do bool weights.
    ValueError
        If ``x`` is not 1-D or 2-D, has no columns, or holds NaN or infinity. Also if ``cl``
        or ``w`` does not have one entry per row, ``w`` holds NaN or infinity, ``cl`` is
        empty, or a label is not a whole number smaller than 1e15 in size.

    See Also
    --------
    wwcss, cluster_predict

    Notes
    -----
    R counterpart: ``WCluster::wmean``, ported from WCluster 1.3.0. Differences from R:

    - R returns the p by k transpose, with columns named ``Cluster Center.1`` to
      ``Cluster Center.k``. This returns an unnamed (k, p) array.
    - ``w=None`` means one weight per row. R's default has one per column and stops unless x
      is square.
    - Labels must be numeric. R groups by ``factor(cl)``, so it also takes text.
    - With integer ``x`` and integer weights, R's integer arithmetic can overflow. For the rows 2e9,
      2e9, 5 and 7 with weights 2, 2, 1 and 1 and labels 1, 1, 2 and 2, R 4.6.1 warns "NAs produced
      by integer overflow" and gives the centers NA and 6. This reads ``x`` and ``w`` as float64 and
      returns 2e9 and 6.

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
    >>> from datanuggets import wmean
    >>> x = np.arange(12.0).reshape(6, 2)
    >>> cl = np.array([4, 4, 6, 6, 8, 8])
    >>> w = np.array([1, 3, 1, 3, 2, 2])
    >>> centers = wmean(x, cl, w)
    >>> centers.shape
    (3, 2)
    >>> [round(float(v), 4) for v in centers[:, 0]]
    [1.5, 5.5, 9.0]
    >>> [round(float(v), 4) for v in centers[:, 1]]
    [2.5, 6.5, 10.0]
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

    Parameters
    ----------
    x : array_like, shape (n, p) or (n,)
        Numeric observations. A 1-D x is one column, and bool data is read as 0 and 1.
    w : array_like, shape (n,), optional
        One weight per row. None means ones. Signs are not checked.

    Returns
    -------
    float
        The sum over rows of each row's weight times its squared Euclidean distance from the
        weighted mean.

    Raises
    ------
    TypeError
        If ``x`` or ``w`` is not numeric. Bool weights count as not numeric.
    ValueError
        If ``x`` is not 1-D or 2-D, has no columns, or holds NaN or infinity. Also if ``w``
        does not have one entry per row or holds NaN or infinity.

    See Also
    --------
    wwcss : The sum of wss over clusters.

    Notes
    -----
    R counterpart: ``WCluster::wss``, ported from WCluster 1.3.0. R returns a length-one
    numeric vector, and this returns a Python float. R's manual page for wss cites no
    references.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import wss
    >>> x = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    >>> w = np.array([1, 2, 3, 2])
    >>> round(float(wss(x)), 4)
    8.0
    >>> round(float(wss(x, w)), 4)
    15.5
    """
    x = as_matrix(x, logical=True, vector=True)
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

    Parameters
    ----------
    x : array_like, shape (n, p) or (n,)
        Numeric observations. A 1-D x is one column, and bool data is read as 0 and 1.
    cl : array_like, shape (n,)
        One numeric cluster label per row, with wmean's label rules.
    w : array_like, shape (n,), optional
        One weight per row. None means ones.
    group_sum : bool, default False
        Return each cluster's value as well as the total. Only its truth value is read.

    Returns
    -------
    float or WwcssResult
        The total as a float, or with ``group_sum=True`` a WwcssResult holding
        ``per_cluster``, shape (k,), in sorted label order and ``total`` as a 0-d float64.

    Raises
    ------
    TypeError
        If ``x``, ``cl`` or ``w`` is not numeric. Text and bool labels count as not numeric,
        and so do bool weights.
    ValueError
        If ``x`` is not 1-D or 2-D, has no columns, or holds NaN or infinity. Also if ``cl``
        or ``w`` does not have one entry per row, ``w`` holds NaN or infinity, ``cl`` is
        empty, or a label is not a whole number smaller than 1e15 in size.

    See Also
    --------
    wss, wmean, distw

    Notes
    -----
    R counterpart: ``WCluster::wwcss``, ported from WCluster 1.3.0. Differences from R:

    - R's ``groupSum = TRUE`` list holds ``WWCSS``, named 1 to k, and ``TotalWWCSS``. These
      become the unnamed ``per_cluster`` and ``total``.
    - ``w=None`` means one weight per row. R's default has one per column and stops unless x
      is square.
    - ``group_sum`` is read by its Python truth value, so the string "FALSE" counts as true.
      R's manual asks for a logical value, and its body tests it with ``if (groupSum)``.
      There R 4.6.1 reads "FALSE" as false and "TRUE" as true, and it stops on "abc" with
      "argument is not interpretable as logical".

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
    >>> from datanuggets import wwcss
    >>> sq = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    >>> x = np.vstack([sq, sq + 10])
    >>> cl = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    >>> w = np.array([1, 2, 3, 2, 1, 1, 1, 1])
    >>> res = wwcss(x, cl, w, group_sum=True)
    >>> [round(float(v), 4) for v in res.per_cluster]
    [15.5, 8.0]
    >>> round(float(res.total), 4)
    23.5
    >>> round(float(wwcss(x, cl, w)), 4)
    23.5
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

    Parameters
    ----------
    x : array_like, shape (n, p) or (n,)
        Numeric observations. A 1-D x is one column, and bool data is read as 0 and 1.
    cl : array_like, shape (n,)
        One numeric cluster label per row, with wmean's label rules and at least two distinct
        labels.
    w : array_like, shape (n,), optional
        One weight per row. None means ones.

    Returns
    -------
    numpy.ndarray, shape (k, k)
        The merge cost of each pair of clusters in the strict lower triangle, in sorted label
        order. The diagonal and the upper triangle hold R's NA bit pattern.

    Raises
    ------
    TypeError
        If ``x``, ``cl`` or ``w`` is not numeric. Text and bool labels count as not numeric,
        and so do bool weights.
    ValueError
        If ``cl`` has fewer than two distinct labels. Also if ``x`` is not 1-D or 2-D, has no
        columns, or holds NaN or infinity, if ``cl`` or ``w`` does not have one entry per
        row, if ``w`` holds NaN or infinity, if ``cl`` is empty, or if a label is not a whole
        number smaller than 1e15 in size.

    See Also
    --------
    wwcss, whclust

    Notes
    -----
    R counterpart: ``WCluster::distw``, ported from WCluster 1.3.0. Each entry is Ward's
    merging cost, ``SS(A+B) - SS(A) - SS(B)``, with the sums of squares weighted. Differences
    from R:

    - R names the rows and columns ``Cluster 1`` to ``Cluster k``. This array has no names.
    - One cluster raises ValueError. R stops with "missing value where TRUE/FALSE needed".
    - ``w=None`` means one weight per row. R's default has one per cell of x and stops unless
      x has one column.

    References
    ----------
    .. [1] Ward, J. H., Jr. "Hierarchical Grouping to Optimize an Objective Function." Journal of
       the American Statistical Association 58(301), 236-244, 1963.
       DOI: ``10.1080/01621459.1963.10500845``.
    .. [2] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.
    .. [3] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import distw
    >>> sq = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    >>> x = np.vstack([sq, sq + [10, 0], sq + [0, 20]])
    >>> cl = np.repeat([1, 2, 3], 4)
    >>> w = np.tile([1, 3, 1, 3], 3)
    >>> d = distw(x, cl, w)
    >>> d.shape
    (3, 3)
    >>> [round(float(d[i, j]), 4) for i, j in [(1, 0), (2, 0), (2, 1)]]
    [400.0, 1600.0, 2000.0]
    >>> bool(np.isnan(d[0, 1]))
    True
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
