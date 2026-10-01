"""Long double arithmetic as R does it on x86-64, replayed exactly with Python integers.

R accumulates sums, means and covariances in LDOUBLE, which on x86-64 is x87 extended precision:
a 64-bit significand, rounded to nearest with ties to even after every operation. A value here is
a pair (m, e) standing for exactly m * 2**e. Every rounded result has |m| < 2**64. from_float
doesn't round. It keeps a double's integer ratio, so a double of 2**64 or more gives a wider m that
ends in zeros, and add passes such a value through when the other operand is zero. Exponent range
limits are ignored: long double's range covers every sum and product of finite doubles that R
forms here.

The compiled kernels in _ldjit do the same arithmetic fast. This module is the reference their
tests compare against.
"""

from __future__ import annotations

import math

from datanuggets._compat.rcompat import _LDOUBLE_DIGITS as DIGITS

ZERO = (0, 0)


def _rounded(m: int, e: int) -> tuple[int, int]:
    """m * 2**e rounded to DIGITS significant bits, ties to even."""
    a = -m if m < 0 else m
    shift = a.bit_length() - DIGITS
    if shift <= 0:
        return m, e
    q = a >> shift
    r = a & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if r > half or (r == half and q & 1):
        q += 1
        if q >> DIGITS:  # rounded up to 2**DIGITS
            q >>= 1
            shift += 1
    return (-q if m < 0 else q), e + shift


def from_float(x: float) -> tuple[int, int]:
    """A finite double, exactly (53 significant bits fit in 64)."""
    n, d = x.as_integer_ratio()
    return n, 1 - d.bit_length()


def add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    am, ae = a
    bm, be = b
    if not am:
        return b
    if not bm:
        return a
    if ae > be:
        return _rounded((am << (ae - be)) + bm, be)
    return _rounded(am + (bm << (be - ae)), ae)


def sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return add(a, (-b[0], b[1]))


def mul(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _rounded(a[0] * b[0], a[1] + b[1])


def div_int(a: tuple[int, int], n: int) -> tuple[int, int]:
    """a / n for a positive integer n (R divides long double sums by counts)."""
    am, ae = a
    if not am:
        return a
    sign = -1 if am < 0 else 1
    am *= sign
    k = max(0, DIGITS + 2 + n.bit_length() - am.bit_length())  # quotient keeps DIGITS + 1 bits
    quo, rem = divmod(am << k, n)
    shift = quo.bit_length() - DIGITS
    q = quo >> shift
    r = quo & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if r > half or (r == half and (rem or q & 1)):
        q += 1
        if q >> DIGITS:
            q >>= 1
            shift += 1
    return sign * q, ae - k + shift


def to_float(a: tuple[int, int]) -> float:
    """Round to double like R's (double) cast; beyond DBL_MAX gives +-inf."""
    m, e = a
    try:
        return float(m << e) if e >= 0 else m / (1 << -e)  # both round once, ties to even
    except OverflowError:
        return math.copysign(math.inf, m)
