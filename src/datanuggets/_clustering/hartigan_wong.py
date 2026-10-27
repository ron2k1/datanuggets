# Adapted from R's kmns.f: (C) Royal Statistical Society 1979.
# Upstream modifications: B. D. Ripley, 1998-06-17; M. Maechler, 2001-04-12.
# Python adaptation by Ronil Basu, recorded 2026-10-21: zero-based indexing,
# Python interfaces and compiled-kernel dispatch. GPL-2.0-only; no warranty.
# See LICENSE and THIRD_PARTY.md for the license and source notices.
"""Hartigan-Wong k-means as R's stats::kmeans runs it.

kmns() runs R's kmns.f (Hartigan and Wong 1979, algorithm AS 136, with R's changes) compiled,
in kmnsjit. Every floating-point operation happens in the Fortran's order, in double, so from
the same starting centers the clusters, centers and within-cluster sums of squares match R bit
for bit. The reference it's tested against, a line-by-line port in plain Python, is _HW in
tests/clustering/kmns_reference.py. kmeans() adds what stats::kmeans does around it: the random
starts, the nstart restarts, and R's errors and warnings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from datanuggets._clustering import kmnsjit as _kmnsjit
from datanuggets._compat._compiled import frozen
from datanuggets._compat.arithmetic import r_sum
from datanuggets._compat.indexing import unique_first
from datanuggets._random.interfaces import Draws

_INT_MAX = 2**31 - 1
_QTRAN_STEPS_PER_ROW = 50  # kmeans(): isteps.Qtran = min(.Machine$integer.max, 50 * m)


class KMeansError(ValueError):
    """stats::kmeans stopped with an error. warnings holds what it warned before stopping."""

    def __init__(self, message: str, warnings: tuple[str, ...] = ()):
        super().__init__(message)
        self.warnings = warnings


@dataclass(frozen=True)
class Kmns:
    """kmns.f's outputs IC1 (0-based), C, NC, WSS, ITER and IFAULT."""

    cluster: list[int]
    centers: np.ndarray
    size: list[int]
    withinss: list[float]
    iter: int
    ifault: int


@dataclass(frozen=True, eq=False)
class KMeansResult:
    """What stats::kmeans returns (cluster 0-based), plus the warnings it raised, in order."""

    cluster: np.ndarray
    centers: np.ndarray
    withinss: np.ndarray
    size: np.ndarray
    tot_withinss: float
    iter: int
    ifault: int
    warnings: tuple[str, ...]


def kmns(a: np.ndarray, c: np.ndarray, iter_max: int, max_qtran: int) -> Kmns:
    """R's C_kmns: Hartigan-Wong on the rows of a (m x p) from the starting centers c (k x p)."""
    a = frozen(np.ascontiguousarray(a, dtype=np.float64))
    c = np.array(c, dtype=np.float64, order="C")  # a copy: the kernel overwrites the centers
    if c.shape[1] != a.shape[1]:
        raise ValueError(f"a has {a.shape[1]} columns but c has {c.shape[1]}")
    ic1 = np.zeros(a.shape[0], dtype=np.int64)
    nc = np.zeros(c.shape[0], dtype=np.int64)
    wss = np.zeros(c.shape[0])
    it, ifault = _kmnsjit.hartigan_wong(a, c, iter_max, max_qtran, ic1, nc, wss)
    return Kmns(ic1.tolist(), c, nc.tolist(), wss.tolist(), it, ifault)


def kmeans(x, k: int, draws: Draws, *, nstart: float = 1, iter_max: int = 10) -> KMeansResult:
    """stats::kmeans(x, centers = k, iter.max, nstart) with the Hartigan-Wong algorithm.

    With nstart == 1 the starts are k random rows. If those repeat a row, or nstart >= 2, the
    starts are drawn from R's unique(x) instead, and floor(nstart) - 1 more starts follow. The
    run with the smallest total within-cluster sum of squares wins, the earliest on ties.
    Raises KMeansError where R stops, and returns R's warnings instead of emitting them.
    """
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[0]
    if k < 2:
        raise ValueError("k must be at least 2; R switches to MacQueen's algorithm for k = 1")
    if not (nstart == 1 or nstart >= 2):
        raise KMeansError("argument is of length zero")  # R: if (ncol(x) != ncol(<a number>))
    starts = cn = None
    if nstart == 1:
        if k > m:
            raise KMeansError(
                "cannot take a sample larger than the population when 'replace = FALSE'"
            )
        starts = x[draws.sample(m, k)]
    if nstart >= 2 or unique_first(starts).size < k:
        cn = x[unique_first(x)]
        if cn.shape[0] < k:
            raise KMeansError("more cluster centers than distinct data points.")
        starts = cn[draws.sample(cn.shape[0], k)]
    warns: list[str] = []
    best = _one_run(x, starts, iter_max, warns)
    best_wss = r_sum(best.withinss)
    for _ in range(2, math.floor(nstart) + 1):
        run = _one_run(x, cn[draws.sample(cn.shape[0], k)], iter_max, warns)
        wss = r_sum(run.withinss)
        if wss < best_wss:
            best, best_wss = run, wss
    return KMeansResult(
        cluster=np.array(best.cluster, dtype=np.intp),
        centers=best.centers,
        withinss=np.array(best.withinss),
        size=np.array(best.size, dtype=np.int64),
        tot_withinss=best_wss,
        iter=best.iter,
        ifault=best.ifault,
        warnings=tuple(warns),
    )


def _one_run(x: np.ndarray, starts: np.ndarray, iter_max: int, warns: list[str]) -> Kmns:
    """kmeans()'s do_one(): kmns, then R's handling of its fault codes."""
    max_qtran = min(_INT_MAX, _QTRAN_STEPS_PER_ROW * x.shape[0])
    z = kmns(x, starts, iter_max, max_qtran)
    if z.ifault == 1:
        raise KMeansError("empty cluster: try a better set of initial centers", tuple(warns))
    if z.ifault == 3:
        raise KMeansError("number of cluster centres must lie between 1 and nrow(x)", tuple(warns))
    if z.ifault == 4:
        warns.append(f"Quick-TRANSfer stage steps exceeded maximum (= {max_qtran})")
    if z.ifault == 2:
        z = replace(z, iter=max(z.iter, iter_max + 1))
    if z.iter > iter_max:
        plural = "" if iter_max == 1 else "s"
        warns.append(f"did not converge in {iter_max} iteration{plural}")
    return z
