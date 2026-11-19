# Adapted from datanugget 1.5.0's create.DNcenters (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""create.DNcenters: reduce a sample to data nugget centers by deleting close pairs.

select_centers runs R's deletion rounds compiled, in centersjit, inside one distance matrix. The
tests hold it to _select_centers_py in tests/nuggets/centers_reference.py, the rounds as R writes
them.
"""

from __future__ import annotations

import warnings

import numpy as np

from datanuggets._compat._compiled import frozen
from datanuggets._compat.rfast import dist_matrix
from datanuggets._defaults import DELETE_PERCENT, DIST_METRICS
from datanuggets._validate import as_count, as_matrix, check_choice, check_delete_percent
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.nuggets import centersjit as _centersjit
from datanuggets.result import CentersResult

PASSTHROUGH_WARNING = (
    "DN.num is greater than or equal to the number of rows. Returning original data"
)


def select_centers(
    rs: np.ndarray, dn_num: int, delete_percent: float, dist_metric: str
) -> tuple[np.ndarray, bool]:
    """Kept row positions (0-based) and whether the input came back unchanged. Inputs are
    pre-validated."""
    n_obs = rs.shape[0]
    if n_obs <= dn_num:
        return np.arange(n_obs), True
    d = dist_matrix(rs, dist_metric)
    np.fill_diagonal(d, d.max() + 1)
    kept = _centersjit.delete_rounds(frozen(d), dn_num, delete_percent)
    return kept.astype(np.intp, copy=False), False


def create_dn_centers(
    rs, dn_num, *, delete_percent: float = DELETE_PERCENT, dist_metric: str = "euclidean"
) -> CentersResult:
    """Reduce a sample to data nugget centers by deleting one point of each closest pair.

    Port of R's create.DNcenters (datanugget 1.5.0). While more than ``dn_num`` rows remain,
    each round deletes ``max(1, floor(t * delete_percent))`` of the t remaining rows, never
    going below ``dn_num``. A round takes the rows in order of the distance to their nearest
    remaining row and deletes a row unless its nearest neighbour is already gone. If that falls
    short of the round's count, a second pass in the same order deletes rows until it is met.
    Nothing is drawn at random, so the result depends only on the sample and its row order.

    Parameters
    ----------
    rs : array_like, shape (n, p)
        Numeric sample, as a 2-D array or a data frame.
    dn_num : int
        Number of rows to keep, a whole number of at least 1. It has no default.
    delete_percent : float, default 0.1
        Fraction of the remaining rows deleted in each round, within (0, 1). Keyword-only.
    dist_metric : {"euclidean", "manhattan"}, default "euclidean"
        Distance between rows. Keyword-only.

    Returns
    -------
    CentersResult
        ``kept``, shape (m,), holds the kept rows' 0-based positions in ``rs`` in ascending
        order, and ``centers`` equals ``rs[kept]``. m is ``dn_num``, or n when ``dn_num >= n``.

    Raises
    ------
    ValueError
        If ``delete_percent`` is outside (0, 1), ``dist_metric`` is not one of its choices,
        or ``dn_num`` is not a whole number of at least 1. Also if ``rs`` is not 2-D, has no
        columns, or holds NaN or infinity.
    TypeError
        If ``delete_percent`` or ``dn_num`` is a bool or not a real number, or ``rs`` is not
        numeric or is all bool. Also if a call passes ``delete_percent`` positionally, as
        R's argument order would.

    Warns
    -----
    DataNuggetWarning
        R's "DN.num is greater than or equal to the number of rows. Returning original data",
        when ``dn_num >= n``. All rows are then returned.

    See Also
    --------
    create_dn : Runs this selection on each chunk and on the pooled centers.

    Notes
    -----
    R counterpart: ``datanugget::create.DNcenters``, ported from datanugget 1.5.0. R's manual
    says this function is used for reducing a random sample to data nugget centers in the
    create.DN function, and that it is not designed for use outside of create.DN. This port
    validates its own arguments, so a direct call gets the errors listed above. Differences
    from R:

    - ``dn_num`` is the second argument and ``delete_percent`` and ``dist_metric`` are
      keyword-only. R's order is ``(RS, delete.percent, DN.num, ...)``.
    - ``dn_num`` must be a whole number. R checks only that it is numeric.
    - ``make.pbs`` is not ported. There is no progress bar.
    - R returns a data frame of the kept rows with their 1-based positions in its "kept"
      attribute. Here ``centers`` is an unnamed float64 array and ``kept`` is 0-based int64.
    - The pairwise distance matrix needs memory quadratic in n.

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
    >>> import numpy as np
    >>> from datanuggets import create_dn_centers
    >>> rs = np.array([[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [9, 9]], dtype=float)
    >>> res = create_dn_centers(rs, 3, delete_percent=0.1)
    >>> res.kept.tolist()
    [2, 4, 5]
    >>> res.centers.shape
    (3, 2)
    >>> bool(np.array_equal(res.centers, rs[res.kept]))
    True
    """
    delete_percent = check_delete_percent(delete_percent)
    check_choice("dist_metric", dist_metric, DIST_METRICS)
    dn_num = as_count("dn_num", dn_num)
    x = as_matrix(rs, "rs")
    kept, passthrough = select_centers(x, dn_num, delete_percent, dist_metric)
    if passthrough:
        warnings.warn(PASSTHROUGH_WARNING, DataNuggetWarning, stacklevel=2)
    return CentersResult(centers=x[kept], kept=kept)
