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
    """Kept row positions (0-based) and whether the input came back unchanged. Inputs pre-validated."""
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
    """Port of R's create.DNcenters: repeatedly delete one point of each closest pair."""
    delete_percent = check_delete_percent(delete_percent)
    check_choice("dist_metric", dist_metric, DIST_METRICS)
    dn_num = as_count("dn_num", dn_num)
    x = as_matrix(rs, "rs")
    kept, passthrough = select_centers(x, dn_num, delete_percent, dist_metric)
    if passthrough:
        warnings.warn(PASSTHROUGH_WARNING, DataNuggetWarning, stacklevel=2)
    return CentersResult(centers=x[kept], kept=kept)
