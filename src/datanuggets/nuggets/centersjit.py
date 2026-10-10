"""create.DNcenters' deletion rounds, compiled.

R copies the distance matrix down to the surviving rows every round. Here the matrix stays put and
the survivors are a list of row numbers in their original order, so "first index on ties" means
the same point it means in R's smaller copy. A row's nearest neighbour only changes when that
neighbour is deleted: the old minimum is still there, and nothing earlier can tie it that wasn't
there before. So after the first round only those rows are searched again.
"""

import math

import numpy as np

from datanuggets._compat._compiled import compiled


@compiled(nogil=True)
def _nearest(d, act, t, a):
    """(first position in act[:t] with the smallest d[a, .], that distance): Rfast::rowMins.

    Four running minima over positions 0, 1, 2, 3 mod 4 don't wait on each other's compares. Each
    keeps its lane's first minimal position, so merging by value, then by position, gives the
    first minimal position overall.
    """
    row = d[a]
    b = 0
    v = row[act[0]]
    rest = 1
    if t >= 8:
        b0, b1, b2, b3 = 0, 1, 2, 3
        v0, v1, v2, v3 = row[act[0]], row[act[1]], row[act[2]], row[act[3]]
        jj = 4
        while jj + 3 < t:
            w = row[act[jj]]
            if w < v0:
                v0, b0 = w, jj
            w = row[act[jj + 1]]
            if w < v1:
                v1, b1 = w, jj + 1
            w = row[act[jj + 2]]
            if w < v2:
                v2, b2 = w, jj + 2
            w = row[act[jj + 3]]
            if w < v3:
                v3, b3 = w, jj + 3
            jj += 4
        rest = jj
        b, v = b0, v0
        if v1 < v or (v1 == v and b1 < b):
            b, v = b1, v1
        if v2 < v or (v2 == v and b2 < b):
            b, v = b2, v2
        if v3 < v or (v3 == v and b3 < b):
            b, v = b3, v3
    for jj in range(rest, t):  # positions past every lane's: only a smaller value wins
        w = row[act[jj]]
        if w < v:
            v, b = w, jj
    return b, v


@compiled(nogil=True)
def delete_rounds(d, dn_num, delete_percent):
    """Row numbers create.DNcenters keeps from d (n x n, diagonal above every distance): each round
    deletes one point of each closest pair, in order of distance, as R does."""
    n = d.shape[0]
    act = np.arange(n)
    nn = np.empty(n, dtype=np.int64)  # nearest active row, by row number
    nn_dist = np.empty(n)
    gone = np.zeros(n, dtype=np.bool_)  # deleted, in this round or an earlier one
    key = np.empty(n)
    t = n
    first = True
    while t > dn_num:
        delete_num = max(1, math.floor(t * delete_percent))
        delete_num = min(delete_num, t - dn_num)
        for i in range(t):
            a = act[i]
            if first or gone[nn[a]]:
                b, v = _nearest(d, act, t, a)
                nn[a] = act[b]
                nn_dist[a] = v
            key[i] = nn_dist[a]
        first = False
        order = np.argsort(key[:t], kind="mergesort")  # stable, like R's order()
        taken = 0
        for p in order:
            if taken >= delete_num:
                break
            a = act[p]
            if not gone[nn[a]] and not gone[a]:
                gone[a] = True
                taken += 1
        if taken < delete_num:
            for p in order:
                if taken >= delete_num:
                    break
                a = act[p]
                if not gone[a]:
                    gone[a] = True
                    taken += 1
        w = 0
        for i in range(t):
            if not gone[act[i]]:
                act[w] = act[i]
                w += 1
        t = w
    return act[:t].copy()
