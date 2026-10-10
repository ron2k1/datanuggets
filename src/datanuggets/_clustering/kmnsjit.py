"""Hartigan-Wong k-means (R's kmns.f), compiled.

The same operations in the same order as hartigan_wong._HW, which ports kmns.f line by line, in
double with no fused multiply-add, so the two agree bit for bit. Cluster and row indices are
0-based. LIVE and NCP hold the Fortran's 1-based row numbers, as in _HW.
"""

import numpy as np

from datanuggets._compat._compiled import compiled

BIG = float(np.float32(1e30))  # kmns.f: DATA BIG /1.E30/, a REAL constant widened to double


@compiled(nogil=True)
def hartigan_wong(a, c, iter_max, max_qtran, ic1, nc, wss):
    """kmns.f on the rows of a (m x p) from the centers in c (k x p), which it overwrites. Writes
    IC1 into ic1, NC into nc and WSS into wss, all zero on entry. Returns (ITER, IFAULT)."""
    m = a.shape[0]
    k = c.shape[0]
    if k <= 1 or k >= m:
        return iter_max, 3
    ic2 = np.zeros(m, dtype=np.int64)
    d = np.zeros(m)
    an1 = np.zeros(k)
    an2 = np.zeros(k)
    ncp = np.zeros(k, dtype=np.int64)
    itran = np.zeros(k, dtype=np.int64)
    live = np.zeros(k, dtype=np.int64)
    _initial_assignment(a, c, ic1, ic2)
    if not _initial_centers(a, c, ic1, nc, an1, an2, itran, ncp):
        return iter_max, 1
    indx = 0
    ifault, ij = 2, iter_max + 1  # a Fortran DO variable after its loop ran out
    for it in range(1, iter_max + 1):
        indx = _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx)
        if indx == m:  # a full pass without a transfer: converged
            ifault, ij = 0, it
            break
        indx, max_qtran = _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, max_qtran)
        if max_qtran < 0:
            ifault, ij = 4, it
            break
        if k == 2:  # with two clusters the optimal-transfer stage has nothing left to do
            ifault, ij = 0, it
            break
        ncp[:] = 0
    _final_centers_and_wss(a, c, ic1, nc, wss)
    return ij, ifault


@compiled(nogil=True)
def _initial_assignment(a, c, ic1, ic2):
    """IC1 and IC2: each point's closest and second closest starting center."""
    m, n = a.shape
    k = c.shape[0]
    for i in range(m):
        dt0 = 0.0
        for j in range(n):
            da = a[i, j] - c[0, j]
            dt0 += da * da
        dt1 = 0.0
        for j in range(n):
            da = a[i, j] - c[1, j]
            dt1 += da * da
        c1, c2 = 0, 1
        if dt0 > dt1:
            c1, c2 = 1, 0
            dt0, dt1 = dt1, dt0
        for g in range(2, k):
            db = 0.0
            done = True
            for j in range(n):
                dc = a[i, j] - c[g, j]
                db += dc * dc
                if db >= dt1:
                    done = False
                    break
            if done:
                if db >= dt0:
                    dt1 = db
                    c2 = g
                else:
                    dt1 = dt0
                    c2 = c1
                    dt0 = db
                    c1 = g
        ic1[i] = c1
        ic2[i] = c2


@compiled(nogil=True)
def _initial_centers(a, c, ic1, nc, an1, an2, itran, ncp):
    """Centers become the means of the initial clusters. False on an empty cluster."""
    m, n = a.shape
    k = c.shape[0]
    c[:, :] = 0.0
    for i in range(m):
        g = ic1[i]
        nc[g] += 1
        for j in range(n):
            c[g, j] += a[i, j]
    for g in range(k):
        if nc[g] == 0:
            return False
        aa = float(nc[g])
        for j in range(n):
            c[g, j] /= aa
        an2[g] = aa / (aa + 1.0)
        an1[g] = aa / (aa - 1.0) if aa > 1.0 else BIG
        itran[g] = 1
        ncp[g] = -1
    return True


@compiled(nogil=True)
def _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx):
    """Optimal-transfer stage: each point may move to the cluster that most reduces WSS."""
    m, n = a.shape
    k = c.shape[0]
    for g in range(k):
        if itran[g] == 1:
            live[g] = m + 1
    for i0 in range(m):
        i = i0 + 1
        indx += 1
        l1 = ic1[i0]
        l2 = ic2[i0]
        ll = l2
        if nc[l1] != 1:
            if ncp[l1] != 0:
                de = 0.0
                for j in range(n):
                    df = a[i0, j] - c[l1, j]
                    de += df * df
                d[i0] = de * an1[l1]
            da = 0.0
            for j in range(n):
                db = a[i0, j] - c[l2, j]
                da += db * db
            r2 = da * an2[l2]
            for g in range(k):
                if (i >= live[l1] and i >= live[g]) or g == l1 or g == ll:
                    continue
                rr = r2 / an2[g]
                dc = 0.0
                done = True
                for j in range(n):
                    dd = a[i0, j] - c[g, j]
                    dc += dd * dd
                    if dc >= rr:
                        done = False
                        break
                if done:
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
                _transfer(a, i0, c, nc, an1, an2, l1, l2)
                ic1[i0] = l2
                ic2[i0] = l1
        if indx == m:
            return indx
    for g in range(k):
        itran[g] = 0
        live[g] -= m
    return indx


@compiled(nogil=True)
def _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, max_qtran):
    """Quick-transfer stage: swap points between their two closest clusters until a full pass
    moves nothing. Returns max_qtran = -1 when the step budget runs out."""
    m, n = a.shape
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
                if istep <= ncp[l1]:
                    da = 0.0
                    for j in range(n):
                        db = a[i0, j] - c[l1, j]
                        da += db * db
                    d[i0] = da * an1[l1]
                if istep < ncp[l1] or istep < ncp[l2]:
                    r2 = d[i0] / an2[l2]
                    dd = 0.0
                    done = True
                    for j in range(n):
                        de = a[i0, j] - c[l2, j]
                        dd += de * de
                        if dd >= r2:
                            done = False
                            break
                    if done:
                        icoun = 0
                        indx = 0
                        itran[l1] = 1
                        itran[l2] = 1
                        ncp[l1] = istep + m
                        ncp[l2] = istep + m
                        _transfer(a, i0, c, nc, an1, an2, l1, l2)
                        ic1[i0] = l2
                        ic2[i0] = l1
            if icoun == m:
                return indx, max_qtran


@compiled(nogil=True)
def _transfer(a, i0, c, nc, an1, an2, l1, l2):
    """Move point i0 from cluster l1 to l2: both centers and their AN1/AN2 factors."""
    al1 = float(nc[l1])
    alw = al1 - 1.0
    al2 = float(nc[l2])
    alt = al2 + 1.0
    for j in range(a.shape[1]):
        c[l1, j] = (c[l1, j] * al1 - a[i0, j]) / alw
        c[l2, j] = (c[l2, j] * al2 + a[i0, j]) / alt
    nc[l1] -= 1
    nc[l2] += 1
    an2[l1] = alw / al1
    an1[l1] = alw / (alw - 1.0) if alw > 1.0 else BIG
    an1[l2] = alt / al2
    an2[l2] = alt / (alt + 1.0)


@compiled(nogil=True)
def _final_centers_and_wss(a, c, ic1, nc, wss):
    """Recompute the centers from the final clusters, then each cluster's WSS."""
    m, n = a.shape
    k = c.shape[0]
    c[:, :] = 0.0
    for i in range(m):
        g = ic1[i]
        for j in range(n):
            c[g, j] += a[i, j]
    for j in range(n):
        for g in range(k):
            c[g, j] /= float(nc[g])
        for i in range(m):
            ii = ic1[i]
            da = a[i, j] - c[ii, j]
            wss[ii] += da * da
