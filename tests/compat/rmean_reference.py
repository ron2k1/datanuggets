"""R's colMeans and rowMeans in plain Python, the reference r_mean is tested against.

_r_mean_py adds each line with Python integers, rounding every partial sum and the quotient to a
64-bit significand as R's x87 long double does, and only then rounds to double. r_mean runs the
same arithmetic compiled, in ldjit, and the tests hold it to this bit for bit. Only the tests run
it, so it lives here and not in the package.
"""

import numpy as np

from datanuggets._compat.ldouble import DIGITS as _LDOUBLE_DIGITS
from datanuggets._compat.missing import _R_NAN, _non_finite_sum


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
    a 64-bit significand (ties to even), then the quotient is rounded to double. No values give
    NaN, R's 0 / 0."""
    if not v:
        return _R_NAN
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
