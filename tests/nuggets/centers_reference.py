"""create.DNcenters' deletion rounds in plain Python, the reference select_centers is held to.

_select_centers_py copies the distance matrix down to the surviving rows every round, as R does.
select_centers runs the same rounds compiled, in centersjit, inside one matrix. Only the tests run
this, so it lives here and not in the package.
"""

import math

import numpy as np

from datanuggets._compat.rfast import dist_matrix


def _select_centers_py(
    rs: np.ndarray, dn_num: int, delete_percent: float, dist_metric: str
) -> tuple[np.ndarray, bool]:
    """select_centers as R writes it, copying the matrix down every round. The reference the
    compiled rounds are tested against."""
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
