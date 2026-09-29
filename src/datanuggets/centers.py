"""create.DNcenters: reduce a sample to data nugget centers by deleting close pairs."""

from __future__ import annotations

import math
import warnings

import numpy as np

from datanuggets._defaults import DELETE_PERCENT, DIST_METRICS
from datanuggets._rfast import dist_matrix
from datanuggets._validate import as_count, as_matrix, check_choice, check_delete_percent
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import CentersResult

PASSTHROUGH_WARNING = (
    "DN.num is greater than or equal to the number of rows. Returning original data"
)


def select_centers(
    rs: np.ndarray, dn_num: int, delete_percent: float, dist_metric: str
) -> tuple[np.ndarray, bool]:
    """Kept row positions (0-based) and whether the input came back unchanged. Inputs pre-validated."""
    n_obs = rs.shape[0]
    keep = np.arange(n_obs)
    if n_obs <= dn_num:
        return keep, True

    d = dist_matrix(rs, dist_metric)
    np.fill_diagonal(d, d.max() + 1)
    tmp_num = n_obs
    while tmp_num > dn_num:
        delete_num = max(1, math.floor(tmp_num * delete_percent))
        delete_num = min(delete_num, tmp_num - dn_num)

        nn_index = d.argmin(axis=1)  # Rfast::rowMins: first index on ties
        nn_dist = d[np.arange(tmp_num), nn_index]
        order = np.argsort(nn_dist, kind="stable").tolist()  # R's order() is stable
        nn = nn_index.tolist()

        marked = [False] * tmp_num
        taken = 0
        for p in order:
            if taken >= delete_num:
                break
            if not marked[nn[p]] and not marked[p]:
                marked[p] = True
                taken += 1
        if taken < delete_num:
            for p in order:
                if taken >= delete_num:
                    break
                if not marked[p]:
                    marked[p] = True
                    taken += 1

        keep_mask = ~np.asarray(marked)
        keep = keep[keep_mask]
        d = d[np.ix_(keep_mask, keep_mask)]
        tmp_num = keep.size
    return keep, False


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
