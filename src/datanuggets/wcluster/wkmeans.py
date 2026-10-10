"""WCluster's Wkmeans: weighted k-means with R's random starts, Lloyd steps and single-point
moves, bit for bit.

wkmeans draws each start's labels here and refines them in the compiled wkmeansjit.
_wkmeans_py is R's Wkmeans, Wkmeans_single and init statement for statement on the Python
primitives, the reference the tests hold the compiled path to. R's quirks are kept: the table
of cluster sizes is refreshed only when a move is taken and is read by position, the Lloyd step
renumbers the labels left, and a move can empty a label or fill an empty one.
"""

from __future__ import annotations

import logging
import math
import numbers

import numpy as np
from joblib import Parallel, delayed

from datanuggets._compat._compiled import frozen
from datanuggets._random.seeds import as_draws
from datanuggets._validate import as_matrix, as_weights, check_n_jobs
from datanuggets.result import WkmeansResult
from datanuggets.wcluster import wkmeansjit, wstatsjit
from datanuggets.wcluster.predict import _wdist_mat
from datanuggets.wcluster.wstats import wmean, wwcss

log = logging.getLogger("datanuggets")

_MISSING = "missing value where TRUE/FALSE needed"
_LIST = "unimplemented type 'list' in 'orderVector1'"
_MAX_LIST = "invalid 'type' (list) of argument"
_LENGTH = "The length of cluster assignments should be the same as the number of observations. "
_USER_EMPTY = (
    "For the initialization, no data assigned to one(or multiple) of the clusters based on the "
    "weighted\n         distance to the initial cluster centers chosen by users."
)
# Where Wkmeans_single stops: R's message, and why. The reference and the kernel share these.
_NAN_STOP = (_MISSING, "a WWCSS is NaN, which a cluster whose weights sum to zero gives")
_TB_STOP = (_MISSING, "R's stale table of cluster sizes is shorter than the label")
_DEAD_SOME_STOP = (_LIST, "a row's distances to every center are NaN")
_DEAD_ALL_STOP = (_LENGTH, "every row's distances are NaN")
_STATUS = {
    wkmeansjit.NAN_TEST: _NAN_STOP,
    wkmeansjit.TB_NA: _TB_STOP,
    wkmeansjit.DEAD_SOME: _DEAD_SOME_STOP,
    wkmeansjit.DEAD_ALL: _DEAD_ALL_STOP,
}
MAX_STARTS = 10_000  # init draws this many starts before saying R would never stop
QUIRKS = ("redraw", "revert", "fixed", "stale_decides", "recreate", "gap", "tb_na", "nan_test")


def _r_stop(message: str, why: str) -> ValueError:
    return ValueError(f"{why} (R stops: {message})")


def _numeric(name: str, v) -> None:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise TypeError(f'{name} must be of class "numeric", got {type(v).__name__}')


def _prepare(x, k, w, centers, num_init, max_iter):
    """R's argument checks in R's order, then R's floors, then the inputs on which R would draw
    starts forever. Returns float64 x and w, int k, num_init and max_iter, and centers as a
    (k, p) float64 array or None."""
    _numeric("k", k)
    x = as_matrix(x, logical=True, vector=True)
    n, p = x.shape
    if centers is not None:
        centers = as_matrix(centers, "centers")
        if centers.shape[0] != k or centers.shape[1] != p:
            raise ValueError(
                f"centers must be a k by ncol(x) matrix, ({k}, {p}), got {centers.shape}"
            )
    w = as_weights(w, n)
    _numeric("num_init", num_init)
    _numeric("max_iter", max_iter)
    k, num_init, max_iter = math.floor(k), math.floor(num_init), math.floor(max_iter)
    if centers is not None:
        num_init = 1
    if k < 1:
        raise ValueError(f"k must be at least 1 after R's floor, got {k}")
    for name, v in (("num_init", num_init), ("max_iter", max_iter)):
        if v < 1:
            raise ValueError(
                f"{name} must be at least 1 after R's floor, got {v}: R's 1:{v} would run twice"
            )
    if centers is None:
        if n < 2 * k:
            raise ValueError(
                f"wkmeans needs at least 2k = {2 * k} rows to draw a start, got {n}: R draws "
                "forever, since every label must get two rows"
            )
        if np.unique(np.column_stack([x, w]), axis=0).shape[0] < k:
            raise ValueError(
                f"x and w have fewer than k = {k} distinct (row, weight) pairs, so no start "
                "reaches k clusters: R draws forever"
            )
    elif n == 1 and not 1 <= w[0] < 2:
        raise _r_stop(
            "invalid 'nrow' value (< 0)" if w[0] < 0 else "non-conformable arguments",
            "R's diag(w) for one row is an identity matrix of size w",
        )
    return x, w, k, centers, num_init, max_iter


def _draw_labels(n: int, k: int, draws) -> np.ndarray:
    """R's ff(k, n): sample(c(1:k, sample(k, n - k, replace = TRUE))), 1-based."""
    pool = np.concatenate([np.arange(1, k + 1), draws.sample_replace(k, n - k) + 1])
    return pool[draws.permutation(n)].astype(np.int64)


def _init(n, k, centers, draws, assign, wmean_of, counts=None) -> np.ndarray:
    """WCluster's init for n rows: draw labels with every label at least twice, take their
    weighted means (or the user's centers), give each row its nearest, and draw again until k
    labels are used. assign(c) returns 1-based labels, 0 for a row whose distances are all NaN,
    and wmean_of(y) the (k, p) centers of labels y. Returns R's 1-based labels."""
    for _ in range(MAX_STARTS):
        c = centers
        if c is None:
            y = _draw_labels(n, k, draws)
            while np.bincount(y, minlength=k + 1)[1:].min() < 2:
                if counts is not None:
                    counts["redraw"] += 1
                y = _draw_labels(n, k, draws)
            c = wmean_of(y)
        zz = assign(c)
        dead = zz == 0
        # R's apply gives a list with integer(0) for a dead row, which unique() counts once
        distinct = np.unique(zz[~dead]).size + (1 if dead.any() and not dead.all() else 0)
        if distinct >= k:
            if dead.any():
                raise _r_stop(_MAX_LIST, "a row's distances to every start center are NaN")
            return zz
        if centers is not None:
            raise _r_stop(_USER_EMPTY, "a cluster gets no rows from the centers given")
        if counts is not None:
            counts["redraw"] += 1
    raise ValueError(
        f"no start among {MAX_STARTS} reached k = {k} clusters (every center was NaN or a row's "
        "distances all were): R would keep drawing forever"
    )


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


def _test(a: float, b: float, counts=None) -> None:
    """R's if (a > b) or if (a <= b), which stops on NaN."""
    if math.isnan(a) or math.isnan(b):
        if counts is not None:
            counts["nan_test"] += 1
        raise _r_stop(*_NAN_STOP)


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
    (uu0, yy) and the passes R made, one "Iteration l" message each. counts gains revert, fixed (a Lloyd step that changed nothing before step 20,
    the kernel's shortcut), stale_decides, recreate, tb_na, nan_test and gap."""
    n = x.shape[0]
    k = int(y.max())
    tb = _table(y)
    yy = y.copy()
    yy0 = yy
    uu0 = math.nan
    passes = 0
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


def _result(x, w, labels) -> WkmeansResult:
    res = wwcss(x, labels, w, group_sum=True)
    return WkmeansResult(
        labels=labels.astype(np.intp) - 1,
        centers=wmean(x, labels, w),
        wwcss=res.per_cluster,
        total_wwcss=res.total,
    )


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


def wkmeans(
    x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=291102, n_jobs=1
) -> WkmeansResult:
    """Weighted k-means, as WCluster's Wkmeans computes it.

    Each start draws labels at random (every label on at least two rows), takes their weighted
    means as centers and gives every row its nearest center, then runs up to max_iter rounds of
    20 Lloyd steps followed by single-point moves, keeping a move that doesn't raise the
    weighted within-cluster sum of squares (WWCSS). The start with the smallest WWCSS wins, the
    first on ties. To get R's clusters, pass seed=RDraws(seed), R's set.seed(seed) draws. An int
    or a numpy Generator draws from numpy instead, which is reproducible but not R's.

    centers (k, p) replaces the random starts, and then num_init is 1. k, num_init and max_iter
    are floored like R's, and num_init or max_iter below 1 raises where R's 1:0 would run twice.
    labels keeps R's quirks: R's table of cluster sizes goes stale between moves and is read by
    position, so a cluster can empty and come back, and labels can have gaps. Where R stops,
    this raises ValueError with R's message. Where R would draw forever (fewer than 2k rows,
    fewer than k distinct (row, weight) pairs, or MAX_STARTS starts that never reach k
    clusters), this raises ValueError. n_jobs refines the starts on threads and never changes
    the result. Arguments after k are keyword-only, since R's order puts centers before the
    weights.
    """
    return _wkmeans(
        x, k, w=w, centers=centers, num_init=num_init, max_iter=max_iter, seed=seed, n_jobs=n_jobs
    )


def _wkmeans(x, k, *, w, centers, num_init, max_iter, seed, n_jobs, trace=None) -> WkmeansResult:
    """wkmeans, with trace, a list, given (start labels, wwcss, labels, passes) for each start
    refined."""
    x, w, k, centers, num_init, max_iter = _prepare(x, k, w, centers, num_init, max_iter)
    check_n_jobs(n_jobs)
    draws = as_draws(seed)
    n = x.shape[0]
    xf, wf = frozen(np.ascontiguousarray(x)), frozen(w)
    m11 = frozen(wstatsjit.row_norms(xf))
    zbuf = np.empty((n, k))

    def assign(c):
        out = np.empty(n, np.int64)
        wstatsjit.assign(xf, wf, frozen(np.ascontiguousarray(c)), c.shape[0], m11, zbuf, out)
        return out

    def wmean_of(y):
        return wkmeansjit.centers_of(xf, wf, y, k)

    def refine(y):
        y = y.copy()
        status, u, it = wkmeansjit.refine(xf, wf, y, max_iter, m11)
        return status, u, it, y

    # One start at a time is R's order, which replayed draws need: a start that stops R leaves
    # later starts undrawn. With threads every start is drawn first, then refined, and the first
    # error in start order is raised, which is R's, since in R a start's own init comes after
    # every earlier start's refinement.
    results, init_error = [], None
    if n_jobs == 1 or num_init == 1:
        for _ in range(num_init):
            s = _init(n, k, centers, draws, assign, wmean_of)
            results.append((s, *refine(s)))
            if results[-1][1] != wkmeansjit.OK:
                break
    else:
        starts = []
        try:
            for _ in range(num_init):
                starts.append(_init(n, k, centers, draws, assign, wmean_of))
        except ValueError as e:
            init_error = e
        refined = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(refine)(s) for s in starts)
        results = [(s, *r) for s, r in zip(starts, refined, strict=True)]
    best = None
    for i, (s, status, u, it, labels) in enumerate(results):
        if trace is not None:
            trace.append((s, u, labels, it))
        if status != wkmeansjit.OK:
            raise _r_stop(*_STATUS[status])
        log.debug("Wkmeans start %d: %d iterations, WWCSS %r", i + 1, it, u)
        if best is None:
            best = (u, labels)
        else:
            _test(best[0], u)
            if best[0] > u:
                best = (u, labels)
    if init_error is not None:
        raise init_error
    return _result(x, w, best[1])
