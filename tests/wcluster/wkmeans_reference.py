"""Wkmeans in plain Python, the reference the tests hold the compiled wkmeans to.

_wkmeans_py is WCluster's Wkmeans and Wkmeans_single statement for statement on the Python
primitives, with the package's argument checks, init and result. Given counts, a dict from
_counts(), it tallies how often it meets each of R's quirks in QUIRKS. Only the tests run it, so
it lives here and not in the package.
"""

import math

import numpy as np

from datanuggets._random.seeds import as_draws
from datanuggets.wcluster.predict import _wdist_mat
from datanuggets.wcluster.wkmeans import (
    _DEAD_ALL_STOP,
    _DEAD_SOME_STOP,
    _TB_STOP,
    _init,
    _prepare,
    _r_seq,
    _r_stop,
    _result,
    _test,
)
from datanuggets.wcluster.wstats import wmean, wwcss

QUIRKS = ("redraw", "revert", "fixed", "stale_decides", "recreate", "gap", "tb_na", "nan_test")


def _which_min_rows(z: np.ndarray) -> np.ndarray:
    """apply(z, 1, which.min), 1-based, 0 for a row of NaN."""
    out = np.zeros(z.shape[0], np.int64)
    for i, r in enumerate(z):
        live = np.flatnonzero(~np.isnan(r))
        if live.size:
            out[i] = live[np.argmin(r[live])] + 1
    return out


def _table(y: np.ndarray) -> np.ndarray:
    """table(y): the count of each label present, in label order. R reads it by position."""
    return np.unique(y, return_counts=True)[1]


def _lloyd_labels(yy: np.ndarray) -> np.ndarray:
    """R's stop when apply(z, 1, which.min) leaves a row without a label: wwcss gets a list,
    or integer(0) when no row has one."""
    if (yy == 0).all():
        raise _r_stop(*_DEAD_ALL_STOP)
    if (yy == 0).any():
        raise _r_stop(*_DEAD_SOME_STOP)
    return yy


def _single_py(x, w, y, K, counts=None):
    """WCluster's Wkmeans_single from init's labels y, statement for statement. Returns R's
    (uu0, yy) and the passes R made, one "Iteration l" message each. counts gains revert, fixed
    (a Lloyd step that changed nothing before step 20, the kernel's shortcut), stale_decides,
    recreate, tb_na, nan_test and gap."""
    n = x.shape[0]
    k = int(y.max())
    tb = _table(y)
    yy = y.copy()
    yy0 = yy
    uu0 = math.nan
    passes = 0
    _r_seq("max_iter", K)
    for _ in range(K):
        passes += 1
        uu0 = uu00 = wwcss(x, yy, w)
        for ll in range(20):
            yy0 = yy
            yy = _lloyd_labels(_which_min_rows(_wdist_mat(x, wmean(x, yy0, w), w)))
            uu0 = wwcss(x, yy, w)
            _test(uu0, uu00, counts)
            if uu0 > uu00:
                if counts is not None:
                    counts["revert"] += 1
                yy = yy0
            else:
                if counts is not None and ll < 19 and np.array_equal(yy, yy0):
                    counts["fixed"] += 1
                uu00 = uu0
        uu0 = uu00 = wwcss(x, yy, w)
        for i in range(n):
            for j in range(1, k + 1):
                a = int(yy[i])
                if a == j:  # R's FALSE & NA is FALSE, so tb isn't read
                    continue
                if a > tb.size:
                    if counts is not None:
                        counts["tb_na"] += 1
                    raise _r_stop(*_TB_STOP)
                if counts is not None and (tb[a - 1] > 1) != (np.count_nonzero(yy == a) > 1):
                    counts["stale_decides"] += 1
                if tb[a - 1] > 1:
                    y2 = yy.copy()
                    y2[i] = j
                    uu1 = wwcss(x, y2, w)
                    _test(uu1, uu0, counts)
                    if uu1 <= uu0:
                        if counts is not None and not (yy == j).any():
                            counts["recreate"] += 1
                        uu0 = uu1
                        yy = y2
                        tb = _table(yy)
        if np.array_equal(yy0, yy):
            break
    if counts is not None and np.unique(yy).size < yy.max():
        counts["gap"] += 1
    return uu0, yy, passes


def _counts() -> dict:
    return dict.fromkeys(QUIRKS, 0)


def _wkmeans_py(
    x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=291102, counts=None, trace=None
):
    """WCluster's Wkmeans on the Python primitives, the test reference. trace, a list, gets
    (start labels, wwcss, labels, passes) for each start that finishes."""
    x, w, k, centers, num_init, max_iter = _prepare(x, k, w, centers, num_init, max_iter)
    draws = as_draws(seed)

    def assign(c):
        return _which_min_rows(_wdist_mat(x, c, w))

    def wmean_of(y):
        return wmean(x, y, w)

    best = None
    for _ in range(num_init):
        start = _init(x.shape[0], k, centers, draws, assign, wmean_of, counts)
        u, labels, passes = _single_py(x, w, start, max_iter, counts)
        if trace is not None:
            trace.append((start, u, labels, passes))
        if best is None:
            best = (u, labels)
        else:
            _test(best[0], u)
            if best[0] > u:
                best = (u, labels)
    return _result(x, w, best[1])
