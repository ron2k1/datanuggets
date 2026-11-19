# Adapted from WCluster 1.3.0's Wpca (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""WCluster's Wpca: principal components of data with a weight per row and per column."""

from __future__ import annotations

import warnings

import numpy as np

from datanuggets._compat.arithmetic import r_sum
from datanuggets._compat.linalg import r_eigen_sym, r_matmul, r_matvec
from datanuggets._validate import as_matrix, as_weights
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import WpcaResult


def _wpca_parts(x: np.ndarray, wrow: np.ndarray, wcol: np.ndarray, corr: bool):
    """The weighted center, the column weights scaled to sum to p, the centered data, the
    scale and the matrix R hands to eigen(), each bit for bit R's."""
    p = x.shape[1]
    total = r_sum(wrow)
    if (wrow < 0).any():
        raise ValueError("wrow must not hold negative weights")
    if total == 0:
        raise ValueError("wrow must not sum to zero")
    if total - 1.0 == 0:
        raise ValueError("wrow must not sum to exactly 1: the covariance divides by sum(wrow) - 1")
    if corr and total - 1.0 < 0:  # every variance is negative, so R's sqrt() gives NaN
        raise ValueError(
            "corr=True needs wrow to sum to more than 1: the covariance divides by sum(wrow) - 1"
        )
    with np.errstate(all="ignore"):
        wcol = (wcol / r_sum(wcol)) * p
    if not np.isfinite(wcol).all() or (wcol < 0).any():
        raise ValueError("wcol must have a nonzero sum and weights of one sign")
    with np.errstate(all="ignore"):  # overflow gives R's Inf and NaN, and eigen then raises
        center = r_matvec(x.T, wrow) / total
        x0 = x - center
        xw = (x0 * np.sqrt(wrow)[:, None]) * np.sqrt(wcol)[None, :]
        s = r_matmul(xw.T, xw) / (total - 1.0)
    scale = np.ones(p)
    if corr:
        with np.errstate(all="ignore"):  # an infinite s times a zero inv is R's NaN
            scale = np.sqrt(np.diag(s))
            inv = 1 / scale
            s = (s * inv[:, None]) * inv[None, :]
        if not np.isfinite(inv).all():
            raise ValueError("corr=True needs a positive weighted variance in every column")
    return center, wcol, x0, scale, s


def wpca(x, wrow=None, wcol=None, corr=False) -> WpcaResult:
    """Weighted principal components, as WCluster's Wpca computes them.

    x is an n by p matrix. wrow holds one nonnegative weight per row and wcol one weight per
    column, and None means ones for either. The column weights are scaled to sum to p. The row
    weights act as counts, so the covariance divides by sum(wrow) - 1, and weights summing to
    exactly 1 or to zero raise a ValueError where R's eigen() stops. With corr=True the
    covariance is scaled to a correlation matrix, and a column with no weighted variance, or
    row weights summing to less than 1, raises.

    Returns a WpcaResult. sdev holds the square roots of the eigenvalues, largest first, and
    rotation the eigenvectors as columns. The sign of each rotation column is LAPACK's and can
    differ from R's. x holds the scores: the centered data times the rotation, not scaled and
    not weighted. An eigenvalue below zero gives a NaN sdev and R's warning, "NaNs produced",
    once. With one column and corr=True the port returns an sdev of 1 to within rounding, where
    R returns the unscaled value or stops, usually with "non-conformable arguments".

    Parameters
    ----------
    x : array_like, shape (n, p) or (n,)
        Numeric observations with at least one row. A 1-D x is one column, and bool data is
        read as 0 and 1.
    wrow : array_like, shape (n,), optional
        One nonnegative weight per row. None means ones.
    wcol : array_like, shape (p,), optional
        One weight per column, all of one sign with a nonzero sum. None means ones.
    corr : bool, default False
        If True, use the weighted correlation matrix instead of the covariance matrix.

    Returns
    -------
    WpcaResult
        ``sdev``, shape (p,), the square roots of the eigenvalues with the largest first.
        ``rotation``, shape (p, p), the eigenvectors as columns. ``x``, shape (n, p), the
        scores. ``center``, shape (p,), the weighted column means. ``scale``, shape (p,), ones
        unless ``corr=True``. With ``corr=True`` it holds the square roots of the covariance
        diagonal, column weights included. ``wrow``, the row weights used. ``wcol``, the column
        weights scaled to sum to p.

    Raises
    ------
    TypeError
        If ``x``, ``wrow`` or ``wcol`` is not numeric, or ``corr`` is not a bool. Bool weights
        count as not numeric.
    ValueError
        If ``x`` has no rows or no columns, is not 1-D or 2-D, or holds NaN or infinity. If
        ``wrow`` or ``wcol`` has the wrong length or holds NaN or infinity. If ``wrow`` has a
        negative weight or sums to zero or to exactly 1, or ``wcol`` sums to zero or mixes
        signs. With ``corr=True``, if ``wrow`` sums to less than 1 or a column has no
        weighted variance. Also with R's message "infinite or missing values in 'x'" when the
        covariance overflows.
    scipy.linalg.LinAlgError
        If LAPACK's dsyevr reports an error.

    Warns
    -----
    DataNuggetWarning
        "NaNs produced", R's sqrt warning, once when an eigenvalue is below zero.

    See Also
    --------
    dn_wpca : The same analysis on data nuggets.

    Notes
    -----
    R counterpart: ``WCluster::Wpca``, ported from WCluster 1.3.0. The eigen decomposition
    calls LAPACK's dsyevr with the arguments R's ``eigen(symmetric = TRUE)`` uses. Differences
    from R:

    - The sign of each rotation column is LAPACK's and can differ from R's.
    - ``scale`` has p entries when ``corr=False``. R returns a single 1.
    - Where R's ``eigen()`` stops on NaN or infinite input, for a negative row weight, a row
      weight sum of zero or 1, column weights of mixed signs or zero sum, or the two
      ``corr=True`` cases above, this raises ValueError naming the cause.
    - With one column and ``corr=True`` the sdev is 1 to within rounding. R's ``diag(1/dd)``
      builds a matrix whose size depends on ``dd`` there, so R returns the unscaled value or
      stops, usually with "non-conformable arguments".
    - ``corr`` takes only a bool. R compares ``corr == T``, so it also takes 1.

    References
    ----------
    .. [1] Amaratunga, D., and Cabrera, J. *Exploration and Analysis of DNA Microarray and Protein
       Array Data*. John Wiley & Sons, 2009 (Vol. 605), as the WCluster manual cites it.
       DOI: ``10.1002/9780470317129``.
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
    >>> from datanuggets import wpca
    >>> x = np.array([[1.0, 1.0], [5.0, 1.0], [3.0, 0.0], [3.0, 2.0]])
    >>> res = wpca(x, wrow=np.array([1.0, 1.0, 2.0, 2.0]))
    >>> [round(float(v), 4) for v in res.sdev]
    [1.2649, 0.8944]
    >>> [round(float(v), 4) for v in res.center]
    [3.0, 1.0]
    >>> res.rotation.shape
    (2, 2)
    >>> res.x.shape
    (4, 2)
    """
    x = as_matrix(x, logical=True, vector=True)
    n, p = x.shape
    if n == 0:
        raise ValueError("x must have at least one row")
    if not isinstance(corr, (bool, np.bool_)):
        raise TypeError("corr must be True or False")
    wrow = as_weights(wrow, n, "wrow")
    wcol = as_weights(wcol, p, "wcol", per="column")
    center, wcol, x0, scale, s = _wpca_parts(x, wrow, wcol, bool(corr))
    values, rotation = r_eigen_sym(s, vectors=True)
    if (values < 0).any():
        warnings.warn("NaNs produced", DataNuggetWarning, stacklevel=2)
    with np.errstate(invalid="ignore"):
        sdev = np.sqrt(values)
    with np.errstate(all="ignore"):  # a row that overflows gets R's infinite score, quietly
        scores = r_matmul(x0, rotation)
    return WpcaResult(
        sdev=sdev,
        rotation=rotation,
        x=scores,
        center=center,
        scale=scale,
        wrow=wrow,
        wcol=wcol,
    )
