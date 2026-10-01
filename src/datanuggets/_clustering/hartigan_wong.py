"""Hartigan-Wong k-means as R's stats::kmeans runs it.

kmns() is a line-by-line port of R's kmns.f (Hartigan and Wong 1979, algorithm AS 136, with
R's changes). Every floating-point operation happens in the Fortran's order, in double, so from
the same starting centers the clusters, centers and within-cluster sums of squares match R bit
for bit. kmeans() adds what stats::kmeans does around it: the random starts, the nstart
restarts, and R's errors and warnings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from datanuggets._compat.rcompat import unique_first
from datanuggets._compat.rstats import r_sum
from datanuggets._draws import Draws

BIG = float(np.float32(1e30))  # kmns.f: DATA BIG /1.E30/, a REAL constant widened to double
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
    hw = _HW(np.asarray(a, dtype=np.float64), np.asarray(c, dtype=np.float64))
    return hw.run(iter_max, max_qtran)


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


class _HW:
    """One kmns.f call. Names follow the Fortran; cluster and row indices are 0-based."""

    def __init__(self, a: np.ndarray, c: np.ndarray):
        self.m, self.n = a.shape
        self.k = c.shape[0]
        self.a = a.tolist()
        self.c = c.tolist()
        m, k = self.m, self.k
        self.ic1 = [0] * m
        self.ic2 = [0] * m
        self.d = [0.0] * m
        self.nc = [0] * k
        self.an1 = [0.0] * k
        self.an2 = [0.0] * k
        self.ncp = [0] * k
        self.itran = [0] * k
        self.live = [0] * k
        self.wss = [0.0] * k

    def run(self, iter_max: int, max_qtran: int) -> Kmns:
        m, k = self.m, self.k
        if k <= 1 or k >= m:
            return self._result(iter_max, 3)
        self._initial_assignment()
        if not self._initial_centers():
            return self._result(iter_max, 1)
        indx = 0
        ifault, ij = 2, iter_max + 1  # a Fortran DO variable after its loop ran out
        for it in range(1, iter_max + 1):
            indx = self._optra(indx)
            if indx == m:  # a full pass without a transfer: converged
                ifault, ij = 0, it
                break
            indx, max_qtran = self._qtran(indx, max_qtran)
            if max_qtran < 0:
                ifault, ij = 4, it
                break
            if k == 2:  # with two clusters the optimal-transfer stage has nothing left to do
                ifault, ij = 0, it
                break
            for g in range(k):
                self.ncp[g] = 0
        self._final_centers_and_wss()
        return self._result(ij, ifault)

    def _result(self, iter_: int, ifault: int) -> Kmns:
        return Kmns(
            cluster=list(self.ic1),
            centers=np.array(self.c),
            size=list(self.nc),
            withinss=list(self.wss),
            iter=iter_,
            ifault=ifault,
        )

    def _initial_assignment(self) -> None:
        """IC1 and IC2: each point's closest and second closest starting center."""
        a, c, rn = self.a, self.c, range(self.n)
        for i in range(self.m):
            ai = a[i]
            dt = [0.0, 0.0]
            for il in (0, 1):
                s = 0.0
                cg = c[il]
                for j in rn:
                    da = ai[j] - cg[j]
                    s += da * da
                dt[il] = s
            c1, c2 = 0, 1
            if dt[0] > dt[1]:
                c1, c2 = 1, 0
                dt[0], dt[1] = dt[1], dt[0]
            for g in range(2, self.k):
                db = 0.0
                cg = c[g]
                for j in rn:
                    dc = ai[j] - cg[j]
                    db += dc * dc
                    if db >= dt[1]:
                        break
                else:
                    if db >= dt[0]:
                        dt[1] = db
                        c2 = g
                    else:
                        dt[1] = dt[0]
                        c2 = c1
                        dt[0] = db
                        c1 = g
            self.ic1[i] = c1
            self.ic2[i] = c2

    def _initial_centers(self) -> bool:
        """Centers become the means of the initial clusters. False on an empty cluster."""
        rn = range(self.n)
        c = [[0.0] * self.n for _ in range(self.k)]
        self.c = c
        for i in range(self.m):
            g = self.ic1[i]
            self.nc[g] += 1
            cg = c[g]
            ai = self.a[i]
            for j in rn:
                cg[j] += ai[j]
        for g in range(self.k):
            if self.nc[g] == 0:
                return False
            aa = float(self.nc[g])
            cg = c[g]
            for j in rn:
                cg[j] /= aa
            self.an2[g] = aa / (aa + 1.0)
            self.an1[g] = aa / (aa - 1.0) if aa > 1.0 else BIG
            self.itran[g] = 1
            self.ncp[g] = -1
        return True

    def _optra(self, indx: int) -> int:
        """Optimal-transfer stage: each point may move to the cluster that most reduces WSS."""
        a, c, ic1, ic2, nc = self.a, self.c, self.ic1, self.ic2, self.nc
        an1, an2, ncp, d, itran, live = self.an1, self.an2, self.ncp, self.d, self.itran, self.live
        m, k, rn = self.m, self.k, range(self.n)
        for g in range(k):
            if itran[g] == 1:
                live[g] = m + 1
        for i0 in range(m):
            i = i0 + 1  # the Fortran's 1-based row number, which LIVE and NCP store
            indx += 1
            l1 = ic1[i0]
            l2 = ic2[i0]
            ll = l2
            if nc[l1] != 1:
                ai = a[i0]
                if ncp[l1] != 0:
                    de = 0.0
                    cg = c[l1]
                    for j in rn:
                        df = ai[j] - cg[j]
                        de += df * df
                    d[i0] = de * an1[l1]
                da = 0.0
                cg = c[l2]
                for j in rn:
                    db = ai[j] - cg[j]
                    da += db * db
                r2 = da * an2[l2]
                for g in range(k):
                    if (i >= live[l1] and i >= live[g]) or g == l1 or g == ll:
                        continue
                    rr = r2 / an2[g]
                    dc = 0.0
                    cg = c[g]
                    for j in rn:
                        dd = ai[j] - cg[j]
                        dc += dd * dd
                        if dc >= rr:
                            break
                    else:
                        r2 = dc * an2[g]
                        l2 = g
                if r2 >= d[i0]:
                    ic2[i0] = l2
                else:
                    indx = 0
                    live[l1] = m + i
                    live[l2] = m + i
                    ncp[l1] = i
                    ncp[l2] = i
                    self._transfer(ai, l1, l2)
                    ic1[i0] = l2
                    ic2[i0] = l1
            if indx == m:
                return indx
        for g in range(k):
            itran[g] = 0
            live[g] -= m
        return indx

    def _qtran(self, indx: int, max_qtran: int) -> tuple[int, int]:
        """Quick-transfer stage: swap points between their two closest clusters until a full
        pass moves nothing. Returns max_qtran = -1 when the step budget runs out."""
        a, c, ic1, ic2, nc = self.a, self.c, self.ic1, self.ic2, self.nc
        an1, an2, ncp, d, itran = self.an1, self.an2, self.ncp, self.d, self.itran
        m, rn = self.m, range(self.n)
        icoun = 0
        istep = 0
        while True:
            for i0 in range(m):
                icoun += 1
                istep += 1
                if istep >= max_qtran:
                    return indx, -1
                l1 = ic1[i0]
                l2 = ic2[i0]
                if nc[l1] != 1:
                    ai = a[i0]
                    if istep <= ncp[l1]:
                        da = 0.0
                        cg = c[l1]
                        for j in rn:
                            db = ai[j] - cg[j]
                            da += db * db
                        d[i0] = da * an1[l1]
                    if istep < ncp[l1] or istep < ncp[l2]:
                        r2 = d[i0] / an2[l2]
                        dd = 0.0
                        cg = c[l2]
                        for j in rn:
                            de = ai[j] - cg[j]
                            dd += de * de
                            if dd >= r2:
                                break
                        else:
                            icoun = 0
                            indx = 0
                            itran[l1] = 1
                            itran[l2] = 1
                            ncp[l1] = istep + m
                            ncp[l2] = istep + m
                            self._transfer(ai, l1, l2)
                            ic1[i0] = l2
                            ic2[i0] = l1
                if icoun == m:
                    return indx, max_qtran

    def _transfer(self, ai: list[float], l1: int, l2: int) -> None:
        """Move point ai from cluster l1 to l2: both centers and their AN1/AN2 factors."""
        c, nc, an1, an2 = self.c, self.nc, self.an1, self.an2
        al1 = float(nc[l1])
        alw = al1 - 1.0
        al2 = float(nc[l2])
        alt = al2 + 1.0
        c1, c2 = c[l1], c[l2]
        for j in range(self.n):
            c1[j] = (c1[j] * al1 - ai[j]) / alw
            c2[j] = (c2[j] * al2 + ai[j]) / alt
        nc[l1] -= 1
        nc[l2] += 1
        an2[l1] = alw / al1
        an1[l1] = alw / (alw - 1.0) if alw > 1.0 else BIG
        an1[l2] = alt / al2
        an2[l2] = alt / (alt + 1.0)

    def _final_centers_and_wss(self) -> None:
        """Recompute the centers from the final clusters, then each cluster's WSS."""
        a, ic1, nc, rn, k = self.a, self.ic1, self.nc, range(self.n), self.k
        c = [[0.0] * self.n for _ in range(k)]
        wss = [0.0] * k
        for i in range(self.m):
            cg = c[ic1[i]]
            ai = a[i]
            for j in rn:
                cg[j] += ai[j]
        for j in rn:
            for g in range(k):
                c[g][j] /= float(nc[g])
            for i in range(self.m):
                ii = ic1[i]
                da = a[i][j] - c[ii][j]
                wss[ii] += da * da
        self.c = c
        self.wss = wss
