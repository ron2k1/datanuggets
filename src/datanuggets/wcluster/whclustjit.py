"""WCluster's Whclust, compiled.

R keeps the distances between the clusters left in an m by m matrix, copies it smaller after
every merge, and computes every cluster's wss again for each height. Every value R computes here
is a value of wss on the same rows in the same order, so computing it once gives the same bits:
a cluster's own wss when it forms, and its distance to each other cluster when it forms. The
distances live in one n by n array, and a merged cluster takes over its column parent's slot.
byrank lists the live slots in R's position order, which is the order of R's sorted labels:
singletons first in reverse row order (R labels them -1 to -n), then merged clusters in the
order they formed.

R's which.min reads its matrix column by column. Each column here keeps its first minimum below
the diagonal, so the pick is the first column, in position order, holding the smallest one. A
column is searched again only when the row holding its minimum is merged away, and a new
cluster, the last row of every column, takes a column's minimum only by being strictly smaller.
Above the diagonal R holds NA in a singleton's column and Inf in a merged cluster's, and those
Inf entries come into play only when no distance below +Inf is left.
"""

import math

import numpy as np

from datanuggets._compat._compiled import compiled
from datanuggets.wcluster import wstatsjit

OK = 0
SELF_MERGE = 1  # R merges a cluster with itself: the caller replays R in the reference
NO_CANDIDATE = 2  # every distance left is NaN and no merged cluster is left to pick: R stops


@compiled(nogil=True, error_model="numpy")
def _pooled(x, w, pool, sa, na, sb, nb, rows, acc, mean, wbuf):
    """wss of two clusters' rows together, merged into original row order."""
    i = 0
    j = 0
    t = 0
    while i < na and j < nb:
        if pool[sa + i] < pool[sb + j]:
            rows[t] = pool[sa + i]
            i += 1
        else:
            rows[t] = pool[sb + j]
            j += 1
        t += 1
    while i < na:
        rows[t] = pool[sa + i]
        i += 1
        t += 1
    while j < nb:
        rows[t] = pool[sb + j]
        j += 1
        t += 1
    return wstatsjit.wss_rows(x, w, rows, t, acc, mean, wbuf)


@compiled(nogil=True, error_model="numpy")
def _scan(d, byrank, m, idx, colmin, colarg):
    """The column at position idx: its first minimum over the rows after it, NaN skipped."""
    c = byrank[idx]
    best = np.nan
    arg = -1
    for t in range(idx + 1, m):
        r = byrank[t]
        v = d[r, c]
        if not math.isnan(v) and (arg < 0 or v < best):
            best = v
            arg = r
    colmin[c] = best
    colarg[c] = arg


@compiled(nogil=True, error_model="numpy")
def _compact(pool, start, size, byrank, m, tmp):
    """Moves the live clusters' rows to the front of pool, returns the first free index."""
    t = 0
    for idx in range(m):
        c = byrank[idx]
        for i in range(size[c]):
            tmp[t + i] = pool[start[c] + i]
        start[c] = t
        t += size[c]
    for i in range(t):
        pool[i] = tmp[i]
    return t


@compiled(nogil=True, error_model="numpy")
def whclust(x, w, merge, height):
    """Fills merge (n-1, 2) with R's labels and height (n-1) with R's heights, n >= 3.
    Returns OK, SELF_MERGE or NO_CANDIDATE, leaving merge and height unfinished for the last
    two."""
    n, p = x.shape
    d = np.empty((n, n))
    pool = np.empty(2 * n, np.int64)  # each live cluster's rows, sorted, at start[slot]
    start = np.empty(n, np.int64)
    size = np.empty(n, np.int64)
    label = np.empty(n, np.int64)
    merged = np.zeros(n, np.bool_)
    own = np.empty(n)
    colmin = np.empty(n)
    colarg = np.empty(n, np.int64)
    byrank = np.empty(n, np.int64)
    rows = np.empty(n, np.int64)
    wbuf = np.empty(n)
    acc = np.empty(p)
    mean = np.empty(p)
    two = np.empty(2)
    tot = np.empty(n)
    for r in range(n):
        pool[r] = r
        start[r] = r
        size[r] = 1
        label[r] = -(r + 1)
        byrank[n - 1 - r] = r
        rows[0] = r
        own[r] = wstatsjit.wss_rows(x, w, rows, 1, acc, mean, wbuf)
    for a in range(n):
        for b in range(a + 1, n):
            two[0] = own[a]
            two[1] = own[b]
            v = _pooled(x, w, pool, a, 1, b, 1, rows, acc, mean, wbuf) - wstatsjit.rsum(two, 2)
            d[a, b] = v
            d[b, a] = v
    m = n
    for idx in range(m):
        _scan(d, byrank, m, idx, colmin, colarg)
    top = n
    for step in range(1, n):
        found = False
        best = 0.0
        bc = -1
        for idx in range(m):
            c = byrank[idx]
            if colarg[c] >= 0 and (not found or colmin[c] < best):
                found = True
                best = colmin[c]
                bc = c
        ru = -1
        cv = -1
        if found and best < np.inf:
            ru = colarg[bc]
            cv = bc
        else:  # which.min lands on a +Inf: a merged column's row 1, or a singleton's first +Inf
            for idx in range(m):
                c = byrank[idx]
                if merged[c]:
                    if idx == 0:
                        return SELF_MERGE
                    ru = byrank[0]
                    cv = c
                    break
                if colarg[c] >= 0:
                    ru = colarg[c]
                    cv = c
                    break
            if ru < 0:
                return NO_CANDIDATE
        merge[step - 1, 0] = label[ru]
        merge[step - 1, 1] = label[cv]
        na = size[ru]
        nb = size[cv]
        if top + na + nb > 2 * n:
            top = _compact(pool, start, size, byrank, m, rows)
        sa = start[ru]
        sb = start[cv]
        s = top
        i = 0
        j = 0
        while i < na and j < nb:
            if pool[sa + i] < pool[sb + j]:
                pool[s] = pool[sa + i]
                i += 1
            else:
                pool[s] = pool[sb + j]
                j += 1
            s += 1
        while i < na:
            pool[s] = pool[sa + i]
            i += 1
            s += 1
        while j < nb:
            pool[s] = pool[sb + j]
            j += 1
            s += 1
        new = cv
        start[new] = top
        size[new] = na + nb
        top += na + nb
        own[new] = wstatsjit.wss_rows(x, w, pool[start[new] :], size[new], acc, mean, wbuf)
        label[new] = step
        merged[new] = True
        q = 0
        for idx in range(m):
            c = byrank[idx]
            if c != ru and c != cv:
                byrank[q] = c
                q += 1
        byrank[q] = new
        m = q + 1
        for idx in range(m):
            tot[idx] = own[byrank[idx]]
        height[step - 1] = wstatsjit.rsum(tot, m)
        if step < n - 1:
            for idx in range(m - 1):
                t = byrank[idx]
                two[0] = own[t]
                two[1] = own[new]
                v = _pooled(
                    x, w, pool, start[t], size[t], start[new], size[new], rows, acc, mean, wbuf
                ) - wstatsjit.rsum(two, 2)
                d[t, new] = v
                d[new, t] = v
            colarg[new] = -1
            for idx in range(m - 1):
                c = byrank[idx]
                a = colarg[c]
                if a == ru or a == cv:
                    _scan(d, byrank, m, idx, colmin, colarg)
                else:
                    v = d[new, c]
                    if not math.isnan(v) and (a < 0 or v < colmin[c]):
                        colmin[c] = v
                        colarg[c] = new
    return OK
