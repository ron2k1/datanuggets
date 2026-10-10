"""Rfast's distance loops, compiled: Dist, dista, and dista's nearest center.

Every distance adds its per-feature terms in Rfast's order, in double, with no fused multiply-add:
Armadillo's two accumulators over even and odd features, added at the end, or left to right for
dista's Euclidean kernel over more than 32 features. So the bits match rfast's numpy replay. Each
kernel fills one output row at a time, looping over centers innermost, which keeps a row's
accumulators in cache and lets LLVM vectorize across centers without reordering any sum. The
inputs are finite, which as_matrix guarantees upstream.
"""

import math

import numpy as np

from datanuggets._compat._compiled import compiled


@compiled(nogil=True)
def _sums(a, bt, manhattan, sequential, out, acc2, k):
    """out[:k] = the distances from a (p,) to the first k columns of bt (p x k)."""
    p = a.shape[0]
    for j in range(p):
        x = a[j]
        into = out if sequential or j % 2 == 0 else acc2
        if j == 0 or (j == 1 and not sequential):
            if manhattan:
                for c in range(k):
                    into[c] = abs(bt[j, c] - x)
            else:
                for c in range(k):
                    d = bt[j, c] - x
                    into[c] = d * d
        elif manhattan:
            for c in range(k):
                into[c] += abs(bt[j, c] - x)
        else:
            for c in range(k):
                d = bt[j, c] - x
                into[c] += d * d
    if p > 1 and not sequential:
        for c in range(k):
            out[c] += acc2[c]
    if not manhattan:
        for c in range(k):
            out[c] = math.sqrt(out[c])


@compiled(nogil=True)
def cross(a, bt, manhattan, sequential, out):
    """out (m x k) = the distances between the rows of a (m x p) and the columns of bt (p x k)."""
    k = bt.shape[1]
    acc2 = np.empty(k)
    for r in range(a.shape[0]):
        _sums(a[r], bt, manhattan, sequential, out[r], acc2, k)


@compiled(nogil=True)
def dist(x, xt, manhattan, out):
    """out (n x n) = Rfast::Dist(x). Row r computes columns 0..r and mirrors them into column r,
    which is exact: b - a is -(a - b) in IEEE arithmetic, and both metrics drop the sign."""
    n = x.shape[0]
    acc2 = np.empty(n)
    for r in range(n):
        row = out[r]
        _sums(x[r], xt, manhattan, False, row, acc2, r + 1)
        for c in range(r):
            out[c, r] = row[c]


@compiled(nogil=True)
def nearest(points, ct, manhattan, sequential, best):
    """best[i] = the first center at the smallest distance from points[i], or -1 when a later center
    ties it. The caller settles those rows as std::sort does."""
    k = ct.shape[1]
    row = np.empty(k)
    acc2 = np.empty(k)
    for i in range(points.shape[0]):
        _sums(points[i], ct, manhattan, sequential, row, acc2, k)
        b = 0
        for c in range(1, k):
            if row[c] < row[b]:
                b = c
        best[i] = b
        for c in range(b + 1, k):
            if row[c] == row[b]:
                best[i] = -1
                break
