"""Exact re-implementations of base-R behavior the ported code depends on."""

from __future__ import annotations

import math
import struct

import numpy as np

from datanuggets import _ldjit


def _r_seq_length_out(start: float, stop: float, n: int) -> np.ndarray:
    """R's seq(start, stop, length.out = n): start + i * by, with the last value set to stop."""
    if n == 1:
        return np.array([start])
    by = (stop - start) / (n - 1)
    out = start + np.arange(n, dtype=np.float64) * by
    out[-1] = stop
    return out


def split_indices(nx: int, ncl: int) -> list[np.ndarray]:
    """R's parallel::splitIndices(nx, ncl), 0-based. Empty chunks are kept, as in R."""
    if ncl == 0:
        return []
    i = np.arange(1, nx + 1)
    if ncl == 1 or nx == 1:
        return [i - 1]
    fuzz = min((nx - 1) / 1000, 0.4 * nx / ncl)
    breaks = _r_seq_length_out(1 - fuzz, nx + fuzz, ncl + 1)
    bins = np.searchsorted(breaks, i, side="left") - 1  # cut(): right-closed (b_j, b_j+1]
    return [np.flatnonzero(bins == j) for j in range(ncl)]


_LDOUBLE_DIGITS = 64  # R's LDOUBLE on x86-64: x87 extended precision, 64-bit significand


def _round_half_even(a: int, shift: int, sticky: bool = False) -> int:
    """a >> shift rounded to nearest, ties to even; sticky marks nonzero bits below a."""
    q = a >> shift
    r = a & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if r > half or (r == half and (sticky or q & 1)):
        q += 1
    return q


_QUIET = 1 << 51  # the top fraction bit, set in a quiet NaN
_FRACTION = (1 << 52) - 1
_INDEFINITE = 0xFFF8000000000000  # the x87's default NaN, which Inf - Inf gives


def _non_finite_sum(v, *, quiet: bool = False) -> float:
    """R's long double running sum of doubles, at least one of them Inf or NaN, down to which
    NaN it is. The finite ones can't overflow a long double, so only the rest decide, one x87
    add of a double at a time. Emulated on the bits, since another FPU picks other NaNs:

    - a finite or infinite sum plus a NaN is that NaN, quieted
    - a sum plus an infinity is that infinity, or the default NaN if the signs differ
    - a NaN sum stays put, except a quiet NaN with a larger significand replaces it (the
      positive one on a tie). A signaling NaN never does, and R's NA is one.

    quiet=True quiets every NaN first, as mean()'s x / n does in SSE. Dividing by a count keeps
    the result, so without it this is colMeans and rowMeans too."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    acc = None  # the running sum's bits once it's non-finite
    for b in v[~np.isfinite(v)].view(np.uint64).tolist():
        if b & _FRACTION:  # a NaN
            b |= _QUIET if quiet else 0
            if acc is None or not acc & _FRACTION:
                acc = b | _QUIET
            elif b & _QUIET and (b & _FRACTION, not b >> 63) > (acc & _FRACTION, not acc >> 63):
                acc = b
        elif acc is None:
            acc = b
        elif not acc & _FRACTION and acc != b:
            acc = _INDEFINITE
    if acc is None:
        raise ValueError("no Inf or NaN to sum")
    return struct.unpack("<d", struct.pack("<Q", acc))[0]


def _ldouble_mean(v: list[float]) -> float:
    """sum(v) / len(v) the way R computes it: every partial sum and the quotient are rounded to
    a 64-bit significand (ties to even), then the quotient is rounded to double. No values give
    NaN, R's 0 / 0."""
    if not v:
        return math.nan
    m = e = 0  # the running sum is exactly m * 2**e
    for x in v:
        xm, d = x.as_integer_ratio()
        xe = 1 - d.bit_length()  # d == 2**-xe
        if xe < e:
            m <<= e - xe
            e = xe
        else:
            xm <<= xe - e
        m += xm
        shift = abs(m).bit_length() - _LDOUBLE_DIGITS
        if shift > 0:
            m = _round_half_even(abs(m), shift) * (1 if m > 0 else -1)
            e += shift
    if m == 0:
        return 0.0
    a, n = abs(m), len(v)
    k = max(0, _LDOUBLE_DIGITS + 2 + n.bit_length() - a.bit_length())  # quotient keeps 66+ bits
    quo, rem = divmod(a << k, n)
    shift = quo.bit_length() - _LDOUBLE_DIGITS
    q = _round_half_even(quo, shift, sticky=rem != 0)
    exp = e - k + shift
    mean = float(q << exp) if exp >= 0 else q / (1 << -exp)  # int to float rounds once, to even
    return mean if m > 0 else -mean


def r_mean(a: np.ndarray, axis: int) -> np.ndarray:
    """R's colMeans (axis=0) or rowMeans (axis=1) of a 2-D double matrix, bit for bit.

    R accumulates each line in long double, which on x86-64 is x87 extended precision with a
    64-bit significand, divides in long double, and only then rounds to double. numpy's
    pairwise double mean differs by several ulps, and on columns that cancel to nearly zero
    (standardized data) even a correctly rounded mean differs, because R's own running sum
    rounds. Finite lines run that arithmetic compiled, in _ldjit. A line with Inf or NaN gets
    the infinity or the very NaN R's long double sum gives, from _non_finite_sum. A double sum
    can differ there: it overflows [1.7e308, 1.7e308, -inf] to inf - inf = NaN where R gets -Inf.
    Platforms whose long double is plain double (arm64 macOS, MSVC builds) give R different
    means. Fixture manifests record longdouble_digits for that reason.
    """
    lines = np.moveaxis(np.asarray(a, dtype=np.float64), axis, -1)
    shape, width = lines.shape[:-1], lines.shape[-1]
    if width == 0:
        return np.full(shape, np.nan)  # R's 0 / 0
    flat = np.ascontiguousarray(lines).reshape(-1, width)
    finite = np.isfinite(flat).all(axis=1)
    out = np.empty(flat.shape[0])
    done = flat if finite.all() else flat[finite]
    out[finite] = _ldjit.line_means(_ldjit.as_bits(done))
    for i in np.flatnonzero(~finite).tolist():
        out[i] = _non_finite_sum(flat[i])
    return out.reshape(shape)


def _r_mean_py(a: np.ndarray, axis: int) -> np.ndarray:
    """r_mean replayed with Python integers, about 0.4 s per million values. It's the
    reference the compiled path is tested against."""
    lines = np.moveaxis(np.asarray(a, dtype=np.float64), axis, -1)
    out = np.empty(lines.shape[:-1])
    for idx in np.ndindex(out.shape):
        v = lines[idx].tolist()
        try:
            out[idx] = _ldouble_mean(v)
        except (OverflowError, ValueError):  # Inf or NaN in the line
            out[idx] = _non_finite_sum(v)
    return out


def unique_first(x: np.ndarray) -> np.ndarray:
    """0-based positions of the rows R's unique(x) keeps from a numeric matrix: the first
    occurrence of each distinct row, in row order. R 4.x compares the rows exactly, with
    0 equal to -0, and so do Python float tuples. Finite data only."""
    first: dict[tuple[float, ...], int] = {}
    for i, row in enumerate(np.asarray(x, dtype=np.float64).tolist()):
        first.setdefault(tuple(row), i)
    return np.fromiter(first.values(), dtype=np.intp, count=len(first))
