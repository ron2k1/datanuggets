"""R's long double sums and the statistics built on them, bit for bit.

sum, mean, colMeans and rowMeans, median, mad, type-7 quantile, cov and weighted.mean are replayed
as R computes them. R accumulates sums in long double (x87 extended precision on x86-64).
colMeans, rowMeans and cov run that arithmetic compiled, in ldjit. sum and mean replay it with
Python integers through ldouble, which is also the reference the compiled code is tested against.
The tests also hold colMeans and rowMeans to _r_mean_py in tests/compat/rmean_reference.py.
"""

from __future__ import annotations

import math

import numpy as np

from datanuggets._compat import ldjit as _ldjit
from datanuggets._compat import ldouble as ld
from datanuggets._compat.missing import _R_NAN, _non_finite_sum
from datanuggets._defaults import MAD_CONSTANT


def r_mean(a: np.ndarray, axis: int) -> np.ndarray:
    """R's colMeans (axis=0) or rowMeans (axis=1) of a 2-D double matrix, bit for bit.

    R accumulates each line in long double, which on x86-64 is x87 extended precision with a
    64-bit significand, divides in long double, and only then rounds to double. numpy's
    pairwise double mean differs by several ulps, and on columns that cancel to nearly zero
    (standardized data) even a correctly rounded mean differs, because R's own running sum
    rounds. Finite lines run that arithmetic compiled, in ldjit. A line with Inf or NaN gets
    the infinity or the very NaN R's long double sum gives, from _non_finite_sum. A double sum
    can differ there: it overflows [1.7e308, 1.7e308, -inf] to inf - inf = NaN where R gets -Inf.
    Platforms whose long double is plain double (arm64 macOS, MSVC builds) give R different
    means. Fixture manifests record longdouble_digits for that reason.
    """
    lines = np.moveaxis(np.asarray(a, dtype=np.float64), axis, -1)
    shape, width = lines.shape[:-1], lines.shape[-1]
    if width == 0:
        return np.full(shape, _R_NAN)  # R's 0 / 0 in long double
    flat = np.ascontiguousarray(lines).reshape(-1, width)
    finite = np.isfinite(flat).all(axis=1)
    out = np.empty(flat.shape[0])
    done = flat if finite.all() else flat[finite]
    out[finite] = _ldjit.line_means(_ldjit.as_bits(done))
    for i in np.flatnonzero(~finite).tolist():
        out[i] = _non_finite_sum(flat[i])
    return out.reshape(shape)


def _ld_sum(v: list[float]) -> tuple[int, int]:
    s = ld.ZERO
    for x in v:
        s = ld.add(s, ld.from_float(x))
    return s


_DBL_MAX = (2**53 - 1) << 971  # DBL_MAX as an exact integer


def r_sum(v) -> float:
    """sum() of a double vector, as R's rsum: a long double running sum that's +-Inf once it's
    past +-DBL_MAX, even where rounding would give DBL_MAX back, and otherwise rounds to double
    once. With Inf or NaN in v only those decide."""
    v = [float(x) for x in v]
    if not all(map(math.isfinite, v)):
        return _non_finite_sum(v)
    m, e = _ld_sum(v)
    if abs(m) << max(e, 0) > _DBL_MAX << max(-e, 0):  # |m| * 2**e > DBL_MAX, exactly
        return math.copysign(math.inf, m)
    return ld.to_float((m, e))


def r_vec_mean(v) -> float:
    """mean() of a double vector, as R 4.6.1's real_mean: the long double sum divided by n,
    plus the mean of the residuals from a second pass. If the sum overflows double, R sums
    x / n instead and divides each residual by n before adding it, so neither pass overflows."""
    v = [float(x) for x in v]
    n = len(v)
    if not all(map(math.isfinite, v)):
        return _non_finite_sum(v, quiet=True)  # R's sum of x / n, and no second pass
    s = _ld_sum(v)
    finite_s = math.isfinite(ld.to_float(s))
    if finite_s:
        s = ld.div_int(s, n)
    else:
        s = _ld_sum([x / n for x in v])  # each x / n is rounded to double first, as in R
    if math.isfinite(ld.to_float(s)):
        t = ld.ZERO
        for x in v:
            r = ld.sub(ld.from_float(x), s)
            t = ld.add(t, r if finite_s else ld.div_int(r, n))
        s = ld.add(s, ld.div_int(t, n) if finite_s else t)
    return ld.to_float(s)


def r_median(v) -> float:
    """median(): the middle value, or R's mean() of the two middle values."""
    s = sorted(float(x) for x in v)
    half = (len(s) + 1) // 2
    if len(s) % 2:
        return s[half - 1]
    return r_vec_mean(s[half - 1 : half + 1])


def r_mad(v) -> float:
    """mad() with R's default constant, 1.4826 exactly."""
    c = r_median(v)
    return MAD_CONSTANT * r_median([abs(float(x) - c) for x in v])


def r_quantile7(v, prob: float) -> float:
    """quantile(v, prob) with R's default type 7."""
    s = sorted(float(x) for x in v)
    index = 1 + max(len(s) - 1, 0) * prob
    lo, hi = math.floor(index), math.ceil(index)
    qs = s[lo - 1]
    if index > lo and s[hi - 1] != qs:
        h = index - lo
        qs = (1 - h) * qs + h * s[hi - 1]
    return qs


def r_cov(x: np.ndarray) -> np.ndarray:
    """cov(x): each column mean in two long double passes, stored as a double, then long double
    sums of the products of deviations divided by n - 1. Compiled for finite data. Inf or NaN
    goes to the Python replay, which raises OverflowError or ValueError."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n, p = x.shape
    if n < 2:
        raise ValueError("cov needs at least two rows")
    out = np.empty((p, p))
    rows = np.arange(n, dtype=np.int64)
    if np.isfinite(x).all() and _ldjit.cov_rows(_ldjit.as_bits(x), rows, out):
        return out
    return _r_cov_py(x)


def _r_cov_py(x: np.ndarray) -> np.ndarray:
    """cov(x) replayed with Python integers: r_cov's path for data with Inf or NaN, which it
    rejects as from_float does, and the reference the compiled path is tested against."""
    n, p = x.shape
    if n < 2:
        raise ValueError("cov needs at least two rows")
    dev = []
    for j in range(p):
        col = [ld.from_float(v) for v in x[:, j].tolist()]
        s = ld.ZERO
        for v in col:
            s = ld.add(s, v)
        mean = ld.div_int(s, n)
        if math.isfinite(ld.to_float(mean)):
            s = ld.ZERO
            for v in col:
                s = ld.add(s, ld.sub(v, mean))
            mean = ld.add(mean, ld.div_int(s, n))
        mean = ld.from_float(ld.to_float(mean))  # R keeps the column means as doubles
        dev.append([ld.sub(v, mean) for v in col])
    out = np.empty((p, p))
    for i in range(p):
        for j in range(i + 1):
            s = ld.ZERO
            for a, b in zip(dev[i], dev[j], strict=True):
                s = ld.add(s, ld.mul(a, b))
            out[i, j] = out[j, i] = ld.to_float(ld.div_int(s, n - 1))
    return out


def r_weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    """weighted.mean(v, w): sum((v * w)[w != 0]) / sum(w), both sums R's long double sum and
    the division in double. A zero total gives NaN or an infinity, as in R. A NaN made by the
    division is the x86 default NaN R gets, whatever this machine's own division gives."""
    with np.errstate(all="ignore"):
        num = r_sum((v * w)[w != 0])
        den = r_sum(w)
        q = float(np.float64(num) / np.float64(den))
    if math.isnan(q) and not (math.isnan(num) or math.isnan(den)):
        return _R_NAN
    return q
