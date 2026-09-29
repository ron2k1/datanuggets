"""Exact re-implementations of base-R behavior the ported code depends on."""

from __future__ import annotations

import numpy as np


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


def _ldouble_mean(v: list[float]) -> float:
    """sum(v) / len(v) the way R computes it: every partial sum and the quotient are rounded to
    a 64-bit significand (ties to even), then the quotient is rounded to double."""
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
    rounds. This replays R's arithmetic exactly with Python integers, about 0.4 s per million
    values. Platforms whose long double is plain double (arm64 macOS, MSVC builds) give R
    different means; fixture manifests record longdouble_digits for that reason.
    """
    lines = np.moveaxis(np.asarray(a, dtype=np.float64), axis, -1)
    out = np.empty(lines.shape[:-1])
    for idx in np.ndindex(out.shape):
        v = lines[idx].tolist()
        try:
            out[idx] = _ldouble_mean(v)
        except (OverflowError, ValueError):  # Inf or NaN in the line: R propagates it silently
            with np.errstate(invalid="ignore"):
                out[idx] = np.mean(v)
    return out


def unique_first(x: np.ndarray) -> np.ndarray:
    """0-based positions of the rows R's unique(x) keeps from a numeric matrix: the first
    occurrence of each distinct row, in row order. R 4.x compares the rows exactly, with
    0 equal to -0, and so do Python float tuples. Finite data only."""
    first: dict[tuple[float, ...], int] = {}
    for i, row in enumerate(np.asarray(x, dtype=np.float64).tolist()):
        first.setdefault(tuple(row), i)
    return np.fromiter(first.values(), dtype=np.intp, count=len(first))
