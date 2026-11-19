# Adapted from datanugget 1.5.0's getDN.means (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""Port of R's getDN.means (datanugget 1.5.0)."""

from __future__ import annotations

import numpy as np

from datanuggets._compat.arithmetic import r_mean
from datanuggets._validate import as_matrix, check_assignments
from datanuggets.result import DataNuggets


def get_dn_means(x, dn: DataNuggets) -> np.ndarray:
    """The column means of each nugget's rows of x, as a (k, p) float64 array.

    Row i is the mean of the rows assigned to nugget i, whatever center method built dn. An
    empty nugget gets a row of R's 0 / 0, the NaN with its sign bit set (fff8000000000000)
    that x86 R gives, where numpy's own NaN has it clear. Each mean is R's colMeans, summed in
    long double, so the result matches R bit for bit. R returns a data frame, transposed when x
    has one column. This returns k rows either way. x must have exactly the rows dn assigns,
    each to a nugget id in [0, k). R ignores extra rows and skips a row whose id is out of range
    or fractional. This raises ValueError for both.

    Parameters
    ----------
    x : array_like, shape (n, p)
        The data dn was built from, in the same row order, with n equal to
        ``len(dn.assignments)`` and p the columns of ``dn.centers``. Bool data is read as 0
        and 1.
    dn : DataNuggets
        Nuggets whose ``assignments`` give each row's nugget.

    Returns
    -------
    numpy.ndarray, shape (k, p)
        Row i holds the column means of the rows assigned to nugget i, with k the number of
        nuggets. With ``center_method="mean"`` it equals ``dn.centers`` to rounding, except at
        an empty nugget, where ``dn.centers`` holds the selected center and this holds NaN.

    Raises
    ------
    TypeError
        If ``dn`` is not a DataNuggets or ``x`` is not numeric.
    ValueError
        If ``x`` is not 2-D (a 1-D vector included), has no columns, or holds NaN or infinity.
        Also if its rows or columns do not match dn, or if ``dn.assignments`` holds ids that
        are not whole numbers in [0, k).

    See Also
    --------
    create_dn, refine_dn

    Notes
    -----
    R counterpart: ``datanugget::getDN.means``, ported from datanugget 1.5.0. Differences
    from R:

    - R returns a data frame with columns Center1 to Centerp, transposed into one row when x
      has one column. This returns an unnamed (k, p) array in every case.
    - A row count other than ``len(dn.assignments)`` raises, where R ignores the extra rows.
    - Ids that are fractional or out of range raise, where R skips their rows.

    References
    ----------
    .. [1] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.
    .. [2] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.

    Examples
    --------
    The data are the 300 points of a 10 by 10 by 3 grid, one point per row.

    >>> import numpy as np
    >>> from datanuggets import RDraws, create_dn, get_dn_means
    >>> x = np.indices((10, 10, 3)).reshape(3, -1).T.astype(float)
    >>> dn = create_dn(x, center_method="original", r=100, delete_percent=0.1, dn_num1=60,
    ...                dn_num2=20, seed=RDraws(291102))
    >>> means = get_dn_means(x, dn)
    >>> means.shape
    (20, 3)
    >>> [round(float(v), 4) for v in means[0]]
    [7.375, 2.875, 0.6667]
    >>> [round(float(v), 4) for v in dn.centers[0]]
    [7.0, 3.0, 0.0]
    """
    x = as_matrix(x, logical=True)
    if not isinstance(dn, DataNuggets):
        raise TypeError(
            f"dn must be a DataNuggets from create_dn or refine_dn, not {type(dn).__name__}"
        )
    n, p = x.shape
    assignments = check_assignments(dn, n, p)
    k = dn.centers.shape[0]
    order = np.argsort(assignments, kind="stable")  # R's which(): each nugget's rows in order
    bounds = np.concatenate(([0], np.cumsum(np.bincount(assignments, minlength=k))))
    means = np.empty((k, p))
    for i in range(k):
        means[i] = r_mean(x[order[bounds[i] : bounds[i + 1]]], axis=0)
    return means
