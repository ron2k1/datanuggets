"""WCluster's Wkmeans_single, compiled: R's Lloyd steps, then its single-point moves.

R recomputes the whole WWCSS for every trial move, O(n p) each. Here each label's wss and its
rows in original order are kept, so a trial computes only the two clusters that change, each on
its rows in original order, and adds the cached values of the rest in label order: the same bits
R gets. R's stale table of cluster sizes is kept as R keeps it: refreshed only when a move is
accepted, read by position, with NA past its end.

A Lloyd step's result depends only on the labels it starts from, and the WWCSS it's compared
with is always the WWCSS of those labels. So a step that changes nothing, or reverts, repeats
for the rest of the 20, and the loop stops there with yy0 set to those labels, which is R's state
at the start of step 20.
"""

import math

import numpy as np

from datanuggets._compat._compiled import compiled
from datanuggets.wcluster import wstatsjit

OK = 0
NAN_TEST = 1  # an if() on a NaN WWCSS
TB_NA = 2  # tb[label] past the end of R's stale table
DEAD_SOME = 3  # a Lloyd step's row with every distance NaN: R's labels become a list
DEAD_ALL = 4  # every row so: R's labels become integer(0)


@compiled(nogil=True, error_model="numpy")
def _group(y, k, order, starts, counts):
    """Counting sort of the rows by label 1..k, each label's rows in original order."""
    for g in range(k + 2):
        counts[g] = 0
    for i in range(y.size):
        counts[y[i]] += 1
    s = 0
    for g in range(k + 1):
        starts[g] = s
        s += counts[g]
    fill = starts.copy()
    for i in range(y.size):
        order[fill[y[i]]] = i
        fill[y[i]] += 1


@compiled(nogil=True, error_model="numpy")
def _table(k, counts, tb):
    """R's table(y) from counts: the present labels' sizes by position. Returns its length."""
    tbn = 0
    for g in range(1, k + 1):
        if counts[g] > 0:
            tb[tbn] = counts[g]
            tbn += 1
    return tbn


@compiled(nogil=True, error_model="numpy")
def _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf):
    """wwcss(x, y, w): each present label's wss into own[label], and their r_sum in label
    order."""
    _group(y, k, order, starts, counts)
    q = 0
    for g in range(1, k + 1):
        if counts[g] > 0:
            own[g] = wstatsjit.wss_rows(x, w, order[starts[g] :], counts[g], acc, mean, wbuf)
            vals[q] = own[g]
            q += 1
    return wstatsjit.rsum(vals, q)


@compiled(nogil=True, error_model="numpy")
def centers_of(x, w, y, k):
    """wmean(x, y, w) for labels 1..k, all present: (k, p)."""
    n, p = x.shape
    order = np.empty(n, np.int64)
    starts = np.empty(k + 2, np.int64)
    counts = np.empty(k + 2, np.int64)
    _group(y, k, order, starts, counts)
    out = np.empty((k, p))
    buf = np.empty(n)
    wbuf = np.empty(n)
    for g in range(1, k + 1):
        wstatsjit.wmean_rows(x, w, order[starts[g] :], counts[g], out[g - 1], buf, wbuf)
    return out


@compiled(nogil=True, error_model="numpy")
def _same(a, b):
    for i in range(a.size):
        if a[i] != b[i]:
            return False
    return True


@compiled(nogil=True, error_model="numpy")
def refine(x, w, y, K, m11):
    """Wkmeans_single from init's labels y (1-based, changed in place to R's final labels).
    Returns (status, uu0, outer iterations run)."""
    n, p = x.shape
    k = 0
    for i in range(n):
        k = max(k, y[i])
    order = np.empty(n, np.int64)
    starts = np.empty(k + 2, np.int64)
    counts = np.empty(k + 2, np.int64)
    own = np.empty(k + 1)
    vals = np.empty(k + 1)
    acc = np.empty(p)
    mean = np.empty(p)
    wbuf = np.empty(n + 1)
    buf = np.empty(n + 1)
    rows = np.empty(n + 1, np.int64)
    centers = np.empty((k, p))
    zbuf = np.empty((n, k))
    yy0 = y.copy()
    tb = np.empty(k, np.int64)
    _group(y, k, order, starts, counts)
    tbn = _table(k, counts, tb)
    uu0 = np.nan
    it = 0
    for _ in range(K):
        it += 1
        uu00 = _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf)
        for _ll in range(20):
            yy0[:] = y
            _group(yy0, k, order, starts, counts)
            c = 0
            for g in range(1, k + 1):
                if counts[g] > 0:
                    wstatsjit.wmean_rows(x, w, order[starts[g] :], counts[g], centers[c], buf, wbuf)
                    c += 1
            dead = wstatsjit.assign(x, w, centers, c, m11, zbuf, y)
            if dead == n:
                return DEAD_ALL, np.nan, it
            if dead > 0:
                return DEAD_SOME, np.nan, it
            uu0 = _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf)
            if math.isnan(uu0) or math.isnan(uu00):
                return NAN_TEST, np.nan, it
            if uu0 > uu00:
                y[:] = yy0
                break
            uu00 = uu0
            if _same(y, yy0):
                break
        uu0 = _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf)
        for i in range(n):
            have_wo = False
            wo = 0.0
            for j in range(1, k + 1):
                a = y[i]
                if a == j:
                    continue
                if a > tbn:
                    return TB_NA, np.nan, it
                if tb[a - 1] <= 1:
                    continue
                if not have_wo and counts[a] > 1:
                    q = 0
                    for t in range(counts[a]):
                        r = order[starts[a] + t]
                        if r != i:
                            rows[q] = r
                            q += 1
                    wo = wstatsjit.wss_rows(x, w, rows, q, acc, mean, wbuf)
                have_wo = True
                q = 0
                placed = False
                for t in range(counts[j]):
                    r = order[starts[j] + t]
                    if not placed and i < r:
                        rows[q] = i
                        q += 1
                        placed = True
                    rows[q] = r
                    q += 1
                if not placed:
                    rows[q] = i
                    q += 1
                wj = wstatsjit.wss_rows(x, w, rows, q, acc, mean, wbuf)
                q = 0
                for g in range(1, k + 1):
                    if g == a:
                        if counts[a] > 1:
                            vals[q] = wo
                            q += 1
                    elif g == j:
                        vals[q] = wj
                        q += 1
                    elif counts[g] > 0:
                        vals[q] = own[g]
                        q += 1
                uu1 = wstatsjit.rsum(vals, q)
                if math.isnan(uu1) or math.isnan(uu0):
                    return NAN_TEST, np.nan, it
                if uu1 <= uu0:
                    uu0 = uu1
                    y[i] = j
                    own[a] = wo
                    own[j] = wj
                    _group(y, k, order, starts, counts)
                    tbn = _table(k, counts, tb)
                    have_wo = False
        if _same(y, yy0):
            break
    return OK, uu0, it
