"""R's x87 long double arithmetic compiled with numba, the fast path behind _rstats and _rcompat.

R sums in long double, which on x86-64 is x87 extended precision: a 64-bit significand, rounded to
nearest with ties to even after every operation. _ldouble replays that with Python integers and
stays as the reference these kernels are tested against. Here a long double is a triple (s, m, e)
standing for (-1)**s * m * 2**e, where m is a uint64 with its top bit set. Zero is (0, 0, 0).

numba has hazards Python integers don't, so every kernel follows four rules:

- A uint64 mixed with an int64 becomes a float64, so every unsigned constant is a np.uint64.
- A shift of 64 bits or more is undefined, so every shift amount stays in 0..63.
- There's no 128-bit integer. A 64 x 64 bit product is built from 32-bit halves, and a division
  by a count runs one bit at a time.
- When two exponents differ by more than 63 bits, the bits that fall off the smaller operand
  collapse into one sticky bit. That makes the truncated value odd, and every rounding boundary
  at that width is even, so it can't change the rounding direction.

Kernels take finite doubles as their bit patterns (a float64 array viewed as uint64). The first
call compiles them, and cache=True keeps the machine code on disk for later runs.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

Z64 = np.uint64(0)
ONE = np.uint64(1)
TOP = np.uint64(1 << 63)
M32 = np.uint64((1 << 32) - 1)
S32 = np.uint64(32)
S63 = np.uint64(63)
S52 = np.uint64(52)
S11 = np.uint64(11)
M11 = np.uint64(0x7FF)
M52 = np.uint64((1 << 52) - 1)
B52 = np.uint64(1 << 52)
C32 = np.uint64(1 << 32)
C16 = np.uint64(1 << 16)
C8 = np.uint64(1 << 8)
C4 = np.uint64(16)
C2 = np.uint64(4)
C1 = np.uint64(2)
SH16 = np.uint64(16)
SH8 = np.uint64(8)
SH4 = np.uint64(4)
SH2 = np.uint64(2)
MAX_EXP = 960  # a triple with a larger e is at least 2**1024, past DBL_MAX


@njit(cache=True)
def bitlen(x):
    """Number of significant bits in a uint64 (0 for 0)."""
    n = 0
    y = x
    if y >= C32:
        n += 32
        y = y >> S32
    if y >= C16:
        n += 16
        y = y >> SH16
    if y >= C8:
        n += 8
        y = y >> SH8
    if y >= C4:
        n += 4
        y = y >> SH4
    if y >= C2:
        n += 2
        y = y >> SH2
    if y >= C1:
        n += 1
        y = y >> ONE
    if y >= ONE:
        n += 1
    return n


@njit(cache=True)
def mul64(a, b):
    """a * b as a 128-bit (hi, lo), from 32-bit halves."""
    a0 = a & M32
    a1 = a >> S32
    b0 = b & M32
    b1 = b >> S32
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    mid = (p00 >> S32) + (p01 & M32) + (p10 & M32)
    lo = (mid << S32) | (p00 & M32)
    hi = p11 + (p01 >> S32) + (p10 >> S32) + (mid >> S32)
    return hi, lo


@njit(cache=True)
def round128(hi, lo, e):
    """(hi * 2**64 + lo) * 2**e, nonzero, rounded to 64 bits with ties to even."""
    if hi == Z64:
        s = 64 - bitlen(lo)
        return lo << np.uint64(s), e - s
    L = bitlen(hi)
    shift = L
    if L == 64:
        q = hi
        r = lo
        half = TOP
    else:
        q = (hi << np.uint64(64 - L)) | (lo >> np.uint64(L))
        r = lo & ((ONE << np.uint64(L)) - ONE)
        half = ONE << np.uint64(L - 1)
    if r > half or (r == half and (q & ONE) == ONE):
        q = q + ONE
        if q == Z64:  # carried out of 64 bits
            q = TOP
            shift += 1
    return q, e + shift


@njit(cache=True)
def place(m, s):
    """m * 2**s as a 128-bit (hi, lo), for s in 0..63."""
    if s == 0:
        return Z64, m
    return m >> np.uint64(64 - s), m << np.uint64(s)


@njit(cache=True)
def add(sa, ma, ea, sb, mb, eb):
    """a + b, rounded once."""
    if ma == Z64:
        return sb, mb, eb
    if mb == Z64:
        return sa, ma, ea
    if eb > ea or (eb == ea and mb > ma):
        sa, ma, ea, sb, mb, eb = sb, mb, eb, sa, ma, ea
    d = ea - eb
    ah, al = place(ma, 63)
    if d <= 63:
        bh, bl = place(mb, 63 - d)
    else:
        bh = Z64
        t = d - 63
        if t <= 63:
            bl = mb >> np.uint64(t)
            if (mb << np.uint64(64 - t)) != Z64:
                bl = bl | ONE  # sticky
        else:
            bl = ONE
    if sa == sb:
        lo = al + bl
        hi = ah + bh + (ONE if lo < al else Z64)
    else:
        lo = al - bl
        hi = ah - bh - (ONE if al < bl else Z64)
        if hi == Z64 and lo == Z64:
            return 0, Z64, 0
    q, e = round128(hi, lo, ea - 63)
    return sa, q, e


@njit(cache=True)
def sub(sa, ma, ea, sb, mb, eb):
    """a - b, rounded once."""
    return add(sa, ma, ea, (1 - sb) if mb != Z64 else 0, mb, eb)


@njit(cache=True)
def mul(sa, ma, ea, sb, mb, eb):
    """a * b, rounded once."""
    if ma == Z64 or mb == Z64:
        return 0, Z64, 0
    hi, lo = mul64(ma, mb)
    q, e = round128(hi, lo, ea + eb)
    return sa ^ sb, q, e


@njit(cache=True)
def from_bits(bits):
    """A finite double, from its bit pattern, exactly."""
    sign = np.int64(bits >> S63)
    ex = np.int64((bits >> S52) & M11)
    frac = bits & M52
    if ex == 0:
        if frac == Z64:
            return 0, Z64, 0
        s = 64 - bitlen(frac)
        return sign, frac << np.uint64(s), -1074 - s
    return sign, (frac | B52) << S11, ex - 1075 - 11


@njit(cache=True)
def div_count(s, m, e, n):
    """a / n for a count 1 <= n < 2**63, rounded once. Long division one bit at a time: 64
    quotient bits, a guard bit, and the remainder as the sticky bit."""
    if m == Z64:
        return 0, Z64, 0
    d = np.uint64(n)
    r = Z64
    q = Z64
    got = 0  # quotient bits kept in q
    i = 0  # numerator bits used: m's 64, then zeros
    guard = Z64
    while True:
        bit = (m >> np.uint64(63 - i)) & ONE if i < 64 else Z64
        r = (r << ONE) | bit  # r < n < 2**63 before the shift, so this fits
        qb = Z64
        if r >= d:
            r = r - d
            qb = ONE
        i += 1
        if got == 0 and qb == Z64:
            continue
        if got < 64:
            q = (q << ONE) | qb
            got += 1
        else:
            guard = qb
            break
    # m / n * 2**(i - 64) = 2q + guard + r / n, so a / n = (q + (guard + r / n) / 2) * 2**x
    x = e + 64 - i + 1
    if guard == ONE and (r != Z64 or (q & ONE) == ONE):
        q = q + ONE
        if q == Z64:
            q = TOP
            x += 1
    return s, q, x


@njit(cache=True)
def round53(s, m, e):
    """The double nearest a as (s, q, x) standing for q * 2**x, q <= 2**53: 53 bits for a normal
    double, fewer below 2**-1022 where the grid is 2**-1074, ties to even."""
    if m == Z64:
        return s, Z64, 0
    shift = max(-1074 - e, 11)
    if shift > 64:  # below half the smallest subnormal
        return s, Z64, 0
    if shift == 64:
        q = Z64
        r = m
        half = TOP
    else:
        q = m >> np.uint64(shift)
        r = m & ((ONE << np.uint64(shift)) - ONE)
        half = ONE << np.uint64(shift - 1)
    if r > half or (r == half and (q & ONE) == ONE):
        q = q + ONE
    return s, q, e + shift


@njit(cache=True)
def to_double(s, m, e):
    """a rounded to double once, like R's (double) cast. Past DBL_MAX gives +-inf."""
    s, q, x = round53(s, m, e)
    v = math.ldexp(np.float64(q), x)  # q * 2**x is a double, so this is exact (or inf)
    return -v if s == 1 else v


@njit(cache=True)
def as_double(s, m, e):
    """a rounded to double and read back: R storing a long double in a double variable. The
    result has e > MAX_EXP exactly when that double is +-inf."""
    s, q, x = round53(s, m, e)
    if q == Z64:
        return 0, Z64, 0
    k = 64 - bitlen(q)
    return s, q << np.uint64(k), x - k
