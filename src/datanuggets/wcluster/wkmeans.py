"""WCluster's Wkmeans: weighted k-means with R's random starts, Lloyd steps and single-point
moves, bit for bit.

wkmeans draws each start's labels here and refines them in the compiled wkmeansjit. The tests
hold the compiled path to _wkmeans_py in tests/wcluster/wkmeans_reference.py, R's Wkmeans and
Wkmeans_single statement for statement on the Python primitives, with this module's init. R's
quirks are kept: the table of cluster sizes is refreshed only when a move is taken and is read
by position, the Lloyd step renumbers the labels left, and a move can empty a label or fill an
empty one.
"""

from __future__ import annotations

import logging
import math
import numbers

import numpy as np
from joblib import Parallel, delayed

from datanuggets._compat._compiled import frozen
from datanuggets._defaults import SEED
from datanuggets._random.replay import ReplayMismatch
from datanuggets._random.seeds import as_draws
from datanuggets._validate import as_matrix, as_weights, check_n_jobs
from datanuggets.result import WkmeansResult
from datanuggets.wcluster import wkmeansjit, wstatsjit
from datanuggets.wcluster.wstats import wmean, wwcss

log = logging.getLogger("datanuggets")

_MISSING = "missing value where TRUE/FALSE needed"
_NAN_SEQ = "NA/NaN argument"
_LONG_SEQ = "result would be too long a vector"
_SIZE = "invalid 'size' argument"
_ZERO = "argument is of length zero"
_CENTERS = "cl.centers must be a ncol(dataset) by k numeric matrix"
_LIST = "unimplemented type 'list' in 'orderVector1'"
_MAX_LIST = "invalid 'type' (list) of argument"
_LENGTH = "The length of cluster assignments should be the same as the number of observations. "
_USER_EMPTY = (
    "For the initialization, no data assigned to one(or multiple) of the clusters based on the "
    "weighted\n         distance to the initial cluster centers chosen by users."
)
# Where Wkmeans_single stops: R's message, and why. The kernel and the reference in
# tests/wcluster/wkmeans_reference.py share these.
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


def _r_stop(message: str, why: str) -> ValueError:
    return ValueError(f"{why} (R stops: {message})")


def _numeric(name: str, v) -> None:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise TypeError(f'{name} must be of class "numeric", got {type(v).__name__}')


def _r_seq(name: str, v) -> None:
    """R's 1:v, which stops on a v that isn't finite."""
    if math.isnan(v):
        raise _r_stop(_NAN_SEQ, f"{name} is NaN")
    if math.isinf(v):
        raise _r_stop(_LONG_SEQ, f"{name} is infinite")


def _prepare(x, k, w, centers, num_init, max_iter):
    """R's argument checks in R's order, then R's floors, then the arguments R stops on before a
    start finishes, in the order R reaches them, then the inputs on which R would draw starts
    forever. Returns float64 x and w, int k and num_init, centers as a (k, p) float64 array or
    None, and max_iter, an int unless it's the NaN or infinity R's 1:K stops on once the first
    start's init is done, which refine and the test reference's _single_py raise."""
    _numeric("k", k)
    x = as_matrix(x, logical=True, vector=True)
    n, p = x.shape
    if centers is not None:
        if np.ndim(centers) == 1:
            raise _r_stop(_ZERO, "centers is a vector, whose nrow R takes as NULL")
        centers = as_matrix(centers, "centers")
        if centers.shape[1] == p and math.isnan(k):  # R's FALSE | NA
            raise _r_stop(_MISSING, "k is NaN, so R's ncol(cl.centers) != k is NA")
        if centers.shape[0] != k or centers.shape[1] != p:
            raise _r_stop(
                _CENTERS, f"centers must be a k by ncol(x) matrix, ({k}, {p}), got {centers.shape}"
            )
    w = as_weights(w, n)
    _numeric("num_init", num_init)
    _numeric("max_iter", max_iter)
    k, num_init, max_iter = (
        math.floor(v) if math.isfinite(v) else v for v in (k, num_init, max_iter)
    )
    if centers is not None:
        num_init = 1
    _r_seq("num_init", num_init)
    if math.isnan(k):  # init's while (length(unique(zz)) < k)
        raise _r_stop(_MISSING, "k is NaN")
    if k == math.inf:  # init's 1:k
        raise _r_stop(_LONG_SEQ, "k is infinite")
    if k < 1:  # init gives no labels, then Wkmeans_single's 1:K runs and wwcss stops
        _r_seq("max_iter", max_iter)
        raise _r_stop(_LENGTH, f"k is {k} after R's floor, so init gives no labels")
    if centers is None:
        if n < k:
            raise _r_stop(
                _SIZE,
                f"x has {n} rows, fewer than k = {k}, so R's sample(k, n - k) gets a negative size",
            )
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
    for name, v in (("num_init", num_init), ("max_iter", max_iter)):
        if math.isfinite(v) and v < 1:
            raise ValueError(
                f"{name} must be at least 1 after R's floor, got {v}: R's 1:{v} would run twice"
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


def _test(a: float, b: float, counts=None) -> None:
    """R's if (a > b) or if (a <= b), which stops on NaN."""
    if math.isnan(a) or math.isnan(b):
        if counts is not None:
            counts["nan_test"] += 1
        raise _r_stop(*_NAN_STOP)


def _result(x, w, labels) -> WkmeansResult:
    res = wwcss(x, labels, w, group_sum=True)
    return WkmeansResult(
        labels=labels.astype(np.intp) - 1,
        centers=wmean(x, labels, w),
        wwcss=res.per_cluster,
        total_wwcss=res.total,
    )


def wkmeans(
    x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=SEED, n_jobs=1
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
    this raises ValueError with R's message. Where R would draw forever (at least k but fewer
    than 2k rows, fewer than k distinct (row, weight) pairs, or MAX_STARTS starts that never
    reach k clusters), this raises ValueError. n_jobs refines the starts on threads and never
    changes the result. With n_jobs above 1 every start is drawn before any is refined, so when
    a start stops, a numpy Generator given as seed has drawn the later starts too. Arguments
    after k are keyword-only, since R's order puts centers before the weights.
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
        _r_seq("max_iter", max_iter)  # Wkmeans_single's 1:K
        y = y.copy()
        status, u, it = wkmeansjit.refine(xf, wf, y, max_iter, m11)
        return status, u, it, y

    # One start at a time is R's order, which replayed draws need: a start that stops R leaves
    # later starts undrawn. With threads every start is drawn first, then refined, and the first
    # error in start order is raised, which is R's, since in R a start's own init comes after
    # every earlier start's refinement. So an init error waits, R's or a ReplayMismatch from
    # draws R never made.
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
        except (ValueError, ReplayMismatch) as e:
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
