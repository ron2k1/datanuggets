"""The compiled long double kernels in ldjit against the Python-int reference in ldouble, bit for
bit, on a fixed-seed fuzz built to hit exact ties, sticky bits and exponent-gap edges."""

import math

import numpy as np
import pytest
from numba import njit

from datanuggets._compat import ldjit as J
from datanuggets._compat import ldouble as ld

U = np.uint64
ONE, TOP, ALL = U(1), U(1 << 63), U((1 << 64) - 1)
FAST = 200_000
SLOW_CHUNKS, SLOW_CHUNK = 20, 500_000  # 10,000,000 cases per operation
EDGE = np.array([s * k for k in (62, 63, 64, 65, 66, 67, 126, 127, 128, 129) for s in (-1, 1)])


def canonical(v: tuple[int, int]) -> tuple[int, int, int]:
    """_ldouble's (m, e) as the kernels' (s, m, e), m shifted to exactly 64 bits."""
    m, e = v
    if not m:
        return 0, 0, 0
    a = -m if m < 0 else m
    sh = 64 - a.bit_length()
    if sh < 0:  # from_float keeps a double's integer ratio, so 2**64 and up carry trailing zeros
        assert not a & ((1 << -sh) - 1)
        return (1 if m < 0 else 0), a >> -sh, e - sh
    return (1 if m < 0 else 0), a << sh, e - sh


def mags(rng, n):
    """Magnitudes with the top bit set: full random, double-like, random trailing zeros, sparse."""
    raw = rng.integers(0, 2**64, n, dtype=U, endpoint=False) | TOP
    tz = rng.integers(0, 64, n).astype(U)
    bit = ONE << rng.integers(0, 63, n).astype(U)
    k = rng.integers(0, 5, n)
    special = np.select(
        [k == 0, k == 1, k == 2, k == 3],
        [np.full(n, TOP), np.full(n, ALL), np.full(n, TOP | ONE), TOP | bit],
        ALL ^ bit,
    )
    kind = rng.integers(0, 4, n)
    return np.select(
        [kind == 0, kind == 1, kind == 2],
        [raw, raw & ~U(0x7FF), raw & ~((ONE << tz) - ONE)],
        special,
    )


def odd_of_len(rng, n, length):
    """Random odd values with exactly `length` bits (1 to 64)."""
    raw = rng.integers(0, 2**64, n, dtype=U, endpoint=False)
    return (raw >> (64 - length).astype(U)) | (ONE << (length - 1).astype(U)) | ONE


def operands(rng, n, op):
    """Pairs for add, sub and mul, 15% of them exact ties, and for add and sub 5% near
    cancellation."""
    sa, sb = rng.integers(0, 2, n), rng.integers(0, 2, n)
    ma, mb = mags(rng, n), mags(rng, n)
    ea = rng.integers(-1100, 1101, n)
    if op == "mul":
        eb = rng.integers(-1100, 1101, n)
        # odd parts whose product has 65 bits sit exactly halfway
        tie = rng.random(n) < 0.15
        la = rng.integers(1, 65, n)
        lb = np.clip(65 - la + rng.integers(0, 2, n), 1, 64)
        ma = np.where(tie, odd_of_len(rng, n, la) << (64 - la).astype(U), ma)
        mb = np.where(tie, odd_of_len(rng, n, lb) << (64 - lb).astype(U), mb)
        return sa, ma, ea, sb, mb, eb
    u = rng.random(n)
    delta = np.select(
        [u < 0.5, u < 0.75, u < 0.9],
        [rng.integers(-70, 71, n), rng.integers(-140, 141, n), rng.integers(-2000, 2001, n)],
        rng.choice(EDGE, n),
    )
    eb = ea + delta
    # b is an odd number of half-ulps of a, so the exact result sits halfway (either sign)
    tie = rng.random(n) < 0.15
    lc = rng.integers(1, 41, n)
    mb = np.where(tie, odd_of_len(rng, n, lc) << (64 - lc).astype(U), mb)
    eb = np.where(tie, ea - 1 - (64 - lc), eb)
    # b is -a (a for sub) with its low bits changed
    can = ~tie & (rng.random(n) < 0.05)
    low = rng.integers(0, 2**64, n, dtype=U, endpoint=False) >> rng.integers(1, 64, n).astype(U)
    mb = np.where(can, ma ^ low, mb) | TOP
    eb = np.where(can, ea, eb)
    sb = np.where(can, (1 - sa) if op == "add" else sa, sb)
    return sa, ma, ea, sb, mb, eb


def binary_loop(kernel):
    @njit
    def run(sa, ma, ea, sb, mb, eb):
        n = sa.size
        so, mo, eo = np.empty(n, np.int64), np.empty(n, np.uint64), np.empty(n, np.int64)
        for i in range(n):
            so[i], mo[i], eo[i] = kernel(sa[i], ma[i], ea[i], sb[i], mb[i], eb[i])
        return so, mo, eo

    return run


BINARY = {
    "add": (binary_loop(J.add), ld.add),
    "sub": (binary_loop(J.sub), ld.sub),
    "mul": (binary_loop(J.mul), ld.mul),
}


def binary_mismatches(op, seed, n):
    run, ref = BINARY[op]
    sa, ma, ea, sb, mb, eb = operands(np.random.default_rng(seed), n, op)
    got = zip(*(v.tolist() for v in run(sa, ma, ea, sb, mb, eb)), strict=True)
    bad = []
    for a_s, a_m, a_e, b_s, b_m, b_e, g in zip(
        *(v.tolist() for v in (sa, ma, ea, sb, mb, eb)), got, strict=True
    ):
        want = canonical(ref(((-a_m if a_s else a_m), a_e), ((-b_m if b_s else b_m), b_e)))
        if want != g:
            bad.append(((a_s, a_m, a_e), (b_s, b_m, b_e), want, g))
    return bad


@njit
def from_bits_loop(bits):
    n = bits.size
    so, mo, eo = np.empty(n, np.int64), np.empty(n, np.uint64), np.empty(n, np.int64)
    for i in range(n):
        so[i], mo[i], eo[i] = J.from_bits(bits[i])
    return so, mo, eo


def from_bits_mismatches(seed, n):
    """Bit patterns of finite doubles, a tenth of them zero or subnormal."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2**64, n, dtype=U, endpoint=False)
    ex = rng.integers(0, 2047, n).astype(U)  # never 2047, which is Inf and NaN
    ex[: n // 10] = 0
    bits = (bits & ~(U(0x7FF) << U(52))) | (ex << U(52))
    got = zip(*(v.tolist() for v in from_bits_loop(bits)), strict=True)
    vals = bits.view(np.float64).tolist()
    return [(v, g) for v, g in zip(vals, got, strict=True) if canonical(ld.from_float(v)) != g]


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_binary_kernels_match_the_reference(op):
    assert binary_mismatches(op, {"add": 1, "sub": 2, "mul": 3}[op], FAST) == []


def test_from_bits_matches_the_reference():
    assert from_bits_mismatches(4, FAST) == []


def counts(rng, n):
    """Counts to divide by: small, powers of two, powers of two +-1, 31-bit and 63-bit."""
    k = rng.integers(0, 6, n)
    small = rng.integers(1, 11, n)
    p2 = np.int64(1) << rng.integers(0, 63, n)
    near = np.clip((np.int64(1) << rng.integers(1, 63, n)) + rng.integers(-1, 2, n), 1, None)
    mid = rng.integers(1, 2**31, n)
    big = rng.integers(1, 2**63 - 1, n, dtype=np.int64)
    return np.select([k == 0, k == 1, k == 2, k == 3, k == 4], [small, p2, near, mid, big], big)


@njit
def div_loop(sa, ma, ea, n):
    k = sa.size
    so, mo, eo = np.empty(k, np.int64), np.empty(k, np.uint64), np.empty(k, np.int64)
    for i in range(k):
        so[i], mo[i], eo[i] = J.div_count(sa[i], ma[i], ea[i], n[i])
    return so, mo, eo


def div_mismatches(seed, n):
    rng = np.random.default_rng(seed)
    sa, ma, ea = rng.integers(0, 2, n), mags(rng, n), rng.integers(-1200, 1201, n)
    cnt = counts(rng, n)
    got = zip(*(v.tolist() for v in div_loop(sa, ma, ea, cnt)), strict=True)
    bad = []
    for a_s, a_m, a_e, c, g in zip(
        sa.tolist(), ma.tolist(), ea.tolist(), cnt.tolist(), got, strict=True
    ):
        if canonical(ld.div_int(((-a_m if a_s else a_m), a_e), c)) != g:
            bad.append(((a_s, a_m, a_e), c, g))
    return bad


def near_double_edges(rng, n):
    """Long doubles around double's range: normal, subnormal, underflow and overflow, a fifth of
    them exactly halfway between two doubles (or two subnormals)."""
    s, m = rng.integers(0, 2, n), mags(rng, n)
    u = rng.random(n)
    lead = np.select(  # exponent of the leading bit
        [u < 0.3, u < 0.6, u < 0.8],
        [rng.integers(-1022, 1024, n), rng.integers(-1080, -1018, n), rng.integers(1018, 1030, n)],
        rng.integers(-1200, 1200, n),
    )
    tie = rng.random(n) < 0.2
    length = rng.integers(2, 65, n)
    m = np.where(tie, odd_of_len(rng, n, length) << (64 - length).astype(U), m)
    return s, m, lead - 63


@njit
def to_double_loop(sa, ma, ea):
    out = np.empty(sa.size)
    for i in range(sa.size):
        out[i] = J.to_double(sa[i], ma[i], ea[i])
    return out


@njit
def as_double_loop(sa, ma, ea):
    k = sa.size
    so, mo, eo = np.empty(k, np.int64), np.empty(k, np.uint64), np.empty(k, np.int64)
    for i in range(k):
        so[i], mo[i], eo[i] = J.as_double(sa[i], ma[i], ea[i])
    return so, mo, eo


def to_double_mismatches(seed, n):
    sa, ma, ea = near_double_edges(np.random.default_rng(seed), n)
    got = to_double_loop(sa, ma, ea)
    want = np.array(
        [
            ld.to_float(((-m if s else m), e))
            for s, m, e in zip(sa.tolist(), ma.tolist(), ea.tolist(), strict=True)
        ]
    )
    return np.flatnonzero(got.view(U) != want.view(U)).tolist()  # bits, so -0.0 != 0.0


def as_double_mismatches(seed, n):
    sa, ma, ea = near_double_edges(np.random.default_rng(seed), n)
    got = zip(*(v.tolist() for v in as_double_loop(sa, ma, ea)), strict=True)
    bad = []
    for s, m, e, g in zip(sa.tolist(), ma.tolist(), ea.tolist(), got, strict=True):
        d = ld.to_float(((-m if s else m), e))
        if math.isinf(d):
            ok = g[2] > J.MAX_EXP
        else:
            ok = g == canonical(ld.from_float(d)) and g[2] <= J.MAX_EXP
        if not ok:
            bad.append(((s, m, e), g))
    return bad


def test_div_count_matches_the_reference():
    assert div_mismatches(5, FAST) == []


def test_to_double_matches_the_reference():
    assert to_double_mismatches(6, FAST) == []


def test_as_double_matches_the_reference():
    assert as_double_mismatches(7, FAST) == []


def test_to_double_keeps_the_sign_of_zero_and_infinity():
    big = 1100 - 63
    zero = J.to_double(0, U(0), 0)
    assert zero == 0.0 and math.copysign(1, zero) == 1
    tiny = J.to_double(1, TOP, -1200)  # far below the smallest subnormal
    assert tiny == 0.0 and math.copysign(1, tiny) == -1
    assert J.to_double(0, TOP, big) == math.inf and J.to_double(1, TOP, big) == -math.inf
    # DBL_MAX plus half an ulp is a tie, and ties go to even: 2**1024, which is inf
    assert J.to_double(0, U(((1 << 54) - 1) << 10), 1024 - 64) == math.inf
    # half the smallest subnormal is a tie too, and goes to even: zero
    assert J.to_double(0, TOP, -1075 - 63) == 0.0
    assert J.to_double(0, TOP | ONE, -1075 - 63) == 5e-324


LONG_FUZZ = {
    "add": lambda seed, n: binary_mismatches("add", seed, n),
    "sub": lambda seed, n: binary_mismatches("sub", seed, n),
    "mul": lambda seed, n: binary_mismatches("mul", seed, n),
    "from_bits": from_bits_mismatches,
    "div_count": div_mismatches,
    "to_double": to_double_mismatches,
    "as_double": as_double_mismatches,
}


@pytest.mark.slow
@pytest.mark.parametrize("chunk", range(SLOW_CHUNKS))
@pytest.mark.parametrize("op", list(LONG_FUZZ))
def test_long_fuzz(op, chunk):
    seed = 1000 + 100 * chunk + list(LONG_FUZZ).index(op)
    assert LONG_FUZZ[op](seed, SLOW_CHUNK) == []
