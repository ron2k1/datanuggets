"""WCluster's Wpca: principal components of data with a weight per row and per column."""

from __future__ import annotations

import warnings

import numpy as np

from datanuggets._rstats import r_eigen_sym, r_matmul, r_matvec, r_sum
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
    once. With one column and corr=True the port returns an sdev of 1 where R returns the
    unscaled value or stops.
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
