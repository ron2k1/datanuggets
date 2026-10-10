"""WCluster's Whclust: weighted Ward hierarchical clustering, merge for merge as R does it.

_whclust_py is R's function statement for statement: its distance matrix, its which.min, its
label bookkeeping and its forder. It's the reference a compiled path will be held to.
"""

from __future__ import annotations

import numpy as np

from datanuggets._compat.arithmetic import r_sum
from datanuggets.result import WhclustResult
from datanuggets.wcluster.wstats import _wss, wwcss

_MISSING = "missing value where TRUE/FALSE needed"
_EMPTY = object()  # R's integer(0), what y0[0] gives


def _at(v, k):
    """R's v[k] for one 1-based k: integer(0) for 0, NA (None) past the end."""
    if k == 0:
        return _EMPTY
    return v[k - 1] if k <= len(v) else None


def _r_which_min(a: np.ndarray) -> int | None:
    """which.min on a matrix: the 1-based column-major index of the first minimum, NaN
    skipped, None where every entry is NaN (R's integer(0))."""
    v = a.ravel(order="F")
    live = np.flatnonzero(~np.isnan(v))
    return None if live.size == 0 else int(live[np.argmin(v[live])]) + 1


def _r_distwij(i, j, y, x, w) -> float:
    """R's distwij(i, j, y, x, w): the wss of clusters i and j pooled, in row order, minus R's
    sum of each one's own wss in sorted label order. Its argument check comes first, and an NA
    or empty label fails it the way R does."""
    if i is _EMPTY or j is _EMPTY:
        raise ValueError("argument is of length zero")
    if i is None or j is None:
        raise ValueError(_MISSING)
    rows = (y == i) | (y == j)
    xs, ws, ys = x[rows], w[rows], y[rows]
    with np.errstate(all="ignore"):
        own = [_wss(xs[ys == g], ws[ys == g]) for g in np.unique(ys)]
        return float(np.float64(_wss(xs, ws)) - np.float64(r_sum(own)))


def _forder(merge) -> list[int]:
    """R's forder: the rows of the last merge, in merge order, 1-based. R's loop runs 2:nrow,
    so one merge row reads a second one that isn't there."""
    if len(merge) < 2:
        raise ValueError("subscript out of bounds")
    u = [[-merge[0][0], -merge[0][1]]]
    for a, b in merge[1:]:
        if a is None or b is None:
            raise ValueError(_MISSING)
        u.append((u[a - 1] if a > 0 else [-a]) + (u[b - 1] if b > 0 else [-b]))
    return u[-1]


def _whclust_py(x, w, *, counts=None):
    """WCluster's Whclust, statement for statement, raising ValueError with R's message where
    R stops. Returns R's merge rows as label pairs (None for NA), the heights and the 1-based
    order. counts, a dict, gets inf_pick (which.min landed on +Inf), self_merge (R merged a
    cluster with itself) and self_merge_mid (one before the last step)."""
    n = x.shape[0]
    if counts is not None:
        counts.update(inf_pick=0, self_merge=0, self_merge_mid=0)
    if n < 2:
        raise ValueError(_MISSING)  # distw's loop over 2:1 calls distwij(NA, ...)
    y = -np.arange(1, n + 1, dtype=np.int64)
    y0 = list(np.unique(y))
    m = n
    dd = np.full((n, n), np.nan)  # distw: NA on and above the diagonal
    for i in range(1, n):
        for j in range(i):
            dd[i, j] = _r_distwij(y0[i], y0[j], y, x, w)
    merge, height = [], []
    for ii in range(1, n):
        k = _r_which_min(dd)
        if k is None:
            raise ValueError("argument is of length zero")  # if (k0 == 0) on integer(0)
        size = dd.shape[0]
        if counts is not None and dd.ravel(order="F")[k - 1] == np.inf:
            counts["inf_pick"] += 1
        k0 = k % m  # m is R's counter, which a self-merge leaves smaller than the matrix
        k1 = k // m + (0 if k0 == 0 else 1)
        u, v = _at(y0, k0), _at(y0, k1)
        if k0 == 0:
            u, k0 = y0[m - 1], m
        if counts is not None and k0 == k1:
            counts["self_merge"] += 1
            counts["self_merge_mid"] += ii < n - 1
        merge.append((None if u is None else int(u), None if v is None else int(v)))
        if v is not None:  # y[y == u] = y[y == v] = ii runs the right one first, and NA is a no-op
            y[y == v] = ii
        if u is not None:
            y[y == u] = ii
        y0 = list(np.unique(y))
        m -= 1
        if ii < n - 1:
            keep = [t for t in range(size) if t + 1 not in (k0, k1)]  # past the end is ignored
            new = np.full((len(keep) + 1, len(keep) + 1), np.inf)  # cbind(rbind(dd, 0), Inf)
            new[:-1, :-1] = dd[np.ix_(keep, keep)]
            new[-1, :-1] = 0.0
            vals = [_r_distwij(_at(y0, t), _at(y0, m), y, x, w) for t in range(1, m)]
            if m > new.shape[0]:
                raise ValueError("subscript out of bounds")
            new[m - 1, : m - 1] = vals
            dd = new
        height.append(wwcss(x, y, w))
    return merge, height, _forder(merge)


def _to_result(n: int, merge, height, order) -> WhclustResult:
    """R's merge labels in scipy's numbering, and order 0-based."""
    pairs = [[-a - 1 if a < 0 else n + a - 1 for a in row] for row in merge]
    return WhclustResult(
        merge=np.array(pairs, dtype=np.intp).reshape(-1, 2),
        height=np.array(height, dtype=np.float64),
        order=np.array(order, dtype=np.intp) - 1,
    )
