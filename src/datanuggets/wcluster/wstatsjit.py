"""WCluster's statistics, compiled, for the whclust and wkmeans kernels.

Each kernel replays a Python primitive bit for bit: rsum is r_sum, wss_rows is _wss on some rows
in their original order, wmean_rows is wmean's r_weighted_mean, and assign is the row-wise
which.min over _wdist_mat. Sums run in the primitive's order, double sums row by row from +0 as
reference BLAS adds them, long double sums through ldjit. The one difference is which NaN: R's
long double sum keeps a particular NaN's bits, and rsum gives numpy's NaN, so a NaN here matches
R's as a NaN, not bit for bit.
"""

import math

import numpy as np

from datanuggets._compat import ldjit
from datanuggets._compat._compiled import compiled

_DBL_MAX_M = np.uint64(0xFFFFFFFFFFFFF800)  # DBL_MAX == _DBL_MAX_M * 2**960 as an ldjit triple


@compiled(nogil=True, error_model="numpy")
def rsum(v, n):
    """r_sum(v[:n]): a long double running sum rounded to double once, +-inf when it's past
    +-DBL_MAX. With Inf or NaN in v[:n] only those decide: NaN if there's a NaN or both
    infinities, else the infinity. v is C-contiguous."""
    nan = False
    pos = False
    neg = False
    for i in range(n):
        a = v[i]
        if math.isnan(a):
            nan = True
        elif a == np.inf:
            pos = True
        elif a == -np.inf:
            neg = True
    if nan or (pos and neg):
        return np.nan
    if pos:
        return np.inf
    if neg:
        return -np.inf
    bits = v.view(np.uint64)
    s, m, e = 0, ldjit.Z64, 0
    for i in range(n):
        bs, bm, be = ldjit.from_bits(bits[i])
        s, m, e = ldjit.add(s, m, e, bs, bm, be)
    if m != ldjit.Z64 and (e > ldjit.MAX_EXP or (e == ldjit.MAX_EXP and m > _DBL_MAX_M)):
        return -np.inf if s == 1 else np.inf
    return ldjit.to_double(s, m, e)


@compiled(nogil=True, error_model="numpy")
def wss_rows(x, w, rows, m, acc, mean, wbuf):
    """_wss(x[rows[:m]], w[rows[:m]]): each column's mean is its double sum of x * w in row
    order over r_sum of the weights, then the result is r_sum over the columns of each column's
    double sum of (dev * dev) * w in row order. acc and mean are scratch of length p, wbuf of
    length at least m."""
    p = x.shape[1]
    for t in range(m):
        wbuf[t] = w[rows[t]]
    sw = rsum(wbuf, m)
    for c in range(p):
        acc[c] = 0.0
    for t in range(m):
        r = rows[t]
        wr = w[r]
        for c in range(p):
            acc[c] += x[r, c] * wr
    for c in range(p):
        mean[c] = acc[c] / sw
        acc[c] = 0.0
    for t in range(m):
        r = rows[t]
        wr = w[r]
        for c in range(p):
            d = x[r, c] - mean[c]
            acc[c] += d * d * wr
    return rsum(acc, p)


@compiled(nogil=True, error_model="numpy")
def wmean_rows(x, w, rows, m, out, buf, wbuf):
    """wmean of x[rows[:m]] into out: per column R's weighted.mean, r_sum of x * w over the rows
    whose weight isn't zero, in row order, over r_sum of the weights. buf and wbuf are scratch
    of length at least m."""
    p = x.shape[1]
    for t in range(m):
        wbuf[t] = w[rows[t]]
    den = rsum(wbuf, m)
    for c in range(p):
        q = 0
        for t in range(m):
            r = rows[t]
            if w[r] != 0.0:
                buf[q] = x[r, c] * w[r]
                q += 1
        out[c] = rsum(buf, q) / den


@compiled(nogil=True, error_model="numpy")
def row_norms(x):
    """m11 in R's wdist_mat: each row's double sum of squares from +0, column by column."""
    n, p = x.shape
    out = np.empty(n)
    for i in range(n):
        s = 0.0
        for c in range(p):
            s += x[i, c] * x[i, c]
        out[i] = s
    return out


@compiled(nogil=True, error_model="numpy")
def assign(x, w, centers, c, m11, zbuf, out):
    """Each row's 1-based which.min over R's wdist_mat(x, centers[:c], w) into out, 0 for a row
    whose entries are all NaN, and returns how many rows that is. z[i, j] is (m11[i] + m22[j])
    - 2 * m12[i, j] with R's double sums in order. R's diag(w) %*% z gives w[i] * z[i, j],
    except NaN where another row of column j of z isn't finite or where the product is NaN.
    which.min takes the first minimum and skips NaN. zbuf is n by at least c."""
    n, p = x.shape
    m22 = np.empty(c)
    nbad = np.zeros(c, np.int64)
    badrow = np.full(c, -1, np.int64)
    for j in range(c):
        s = 0.0
        for t in range(p):
            s += centers[j, t] * centers[j, t]
        m22[j] = s
    for i in range(n):
        for j in range(c):
            s = 0.0
            for t in range(p):
                s += x[i, t] * centers[j, t]
            z = (m11[i] + m22[j]) - 2 * s
            zbuf[i, j] = z
            if not np.isfinite(z):
                nbad[j] += 1
                badrow[j] = i
    dead = 0
    for i in range(n):
        best = 0.0
        arg = 0
        for j in range(c):
            if nbad[j] > 1 or (nbad[j] == 1 and badrow[j] != i):
                continue
            v = w[i] * zbuf[i, j]
            if math.isnan(v):
                continue
            if arg == 0 or v < best:
                best = v
                arg = j + 1
        out[i] = arg
        if arg == 0:
            dead += 1
    return dead
