# Speed: Compiled Long Double Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python-integer replay of R's x87 long double in `r_cov` and `r_mean` with numba kernels that give the same bits. Compute `refine_dn`'s per-nugget centers and covariances in one compiled call, threaded when `n_jobs` isn't 1. Then check spec section 9.5's stopping rule on Covertype 100k, and once it holds, benchmark on Covertype.

**Architecture:** A new `src/datanuggets/_ldjit.py` holds the kernels. A long double is a triple (sign, uint64 magnitude, exponent). The module has add, sub, mul, divide by a count, from double (bit pattern) and to double, plus whole compiled loops for `cov`, row means and per-group moments. `_rstats.r_cov` and `_rcompat.r_mean` keep their signatures and call the kernels. Today's bodies stay as `_r_cov_py` and `_r_mean_py`, which with `_ldouble.py` are the references the tests compare against. A new `_rstats.group_moments` feeds `_Nuggets.refresh` in `refine.py`. `benchmarks/profile_stop.py` checks the stopping rule.

**Tech Stack:** Python 3.12+ (dev on 3.14.3), numpy >= 2.0, scipy >= 1.13, numba >= 0.67 (0.67.0 locked), pytest + pytest-xdist, uv, ruff 0.16. Tests run on the Mac mini (M4, arm64, 10 cores) through `mini-run.sh`. Published timings come from the laptop.

**Spec:** `docs/specs/2026-09-29-datanuggets-python-port-design.md` section 9 (9.1 measurements, 9.2 compiled long double, 9.3 parallelism, 9.5 when to stop). Approved by RON2K on 2026-09-29 at commit 0196d18.

**Execution:** RON2K: "merge PR #3 and approve the spec, go ahead". PR #3 merged as d08d7b1. The execution method is chosen at the plan handoff. Tasks 1 to 5 build and test. Task 6 profiles and is a decision gate. Task 7 runs once the stopping rule holds. The dry run (Evidence) says that takes a follow-up plan (see Task 6).

**Evidence behind the code in this plan** (throwaway scripts in the session scratchpad, not in the repo):

- Spike (spec 9.2 table): `add`, `mul` and `from_bits` matched `_ldouble` on 10,000,000 cases per operation on the mini and 1,000,000 on the laptop. They also matched on all 109,670 of `r_cov`'s product sums over Covertype 100k's nuggets.
- Plan-time draft of this plan's `div_count` and `to_double` on the laptop: 200,000 fuzz cases each, 0 mismatches. That includes subnormal results, negative underflow to -0.0, and ties at DBL_MAX and at half the smallest subnormal.
- Plan-time draft of this plan's `line_means`, `cov_rows` and group kernels: 0 mismatches against `_r_mean_py`/`_r_cov_py` on every fixture matrix, both axes. The same holds across 68 fixture assignments (2,270 nonempty nugget groups), and 1 thread against 4 gave identical bits. On Covertype 100k's 2,000 nuggets, every center and covariance took 0.354 s on one thread, 0.098 s on 4 and 0.061 s on 20. The Python replay took 0.77 s for the first 200 nuggets alone. First compile of those kernels took 6.35 s on the laptop.
- The fuzz tests below ran green against the draft: 8 tests in 3.7 s on 8 xdist workers.
- Dry run of this plan as written (2026-09-30): every code block in Tasks 1 to 7 pasted into a throwaway worktree off 0196d18 by a script, with each edit's anchor checked. ruff check and ruff format were clean, the mini ran the suite with 1069 passed in 9.72 s, and each file's collected count matched its task's Expected line. `benchmarks/covertype.py 10000 cores` gave `same_result: true` for all three steps, and R parsed the new `covertype.R`.
- The dry run's profile on the mini, which Task 6 repeats: at 100,000 rows `create_dn` took 15.42 s and `refine_dn` 0.91 s, both matching R on every output. The stopping rule failed on both. In `create_dn` the largest Python costs were `_rfast.py`'s `term` (39.6% of wall), `centers.select_centers` (33.4%) and `_rfast._arma_accumulate` (11.9%). In `refine_dn`, `compiled_group_moments_serial` took 19.7%, and Hartigan-Wong's `_qtran` 17.2%, `_optra` 13.6%, `_final_centers_and_wss` 12.2% and `_initial_assignment` 12.1%. So Task 6's gate is expected to fail, and Task 7 to wait for the follow-up plan it calls for.

## Global Constraints

- Branch `feat/speed`, tracking `origin/feat/speed`. It's based on f792b51, which `main` (d08d7b1) already contains. `main` advances only by PR, and merging is RON2K's call.
- Parity: every existing fixture test passes unchanged. Compiled results equal the Python replay bit for bit. Compare with `.view(np.uint64)`, so -0.0 and 0.0 count as different.
- Kernels follow spec 9.2's four rules: every unsigned constant is a `np.uint64`, every shift amount is in 0..63, there's no 128-bit type, and exponent gaps above 63 collapse into one sticky bit. Each compiled loop runs its operations in the same order as the Python code it replaces.
- `_rstats` and `_rcompat` keep their function signatures. Inf or NaN never reaches a kernel: the Python replay handles those lines and raises or propagates exactly as it does today.
- `_ldouble.py`, `_r_cov_py` and `_r_mean_py` stay in the package as test references and don't run in normal use.
- `numba >= 0.67` is a runtime dependency with no fallback. Every kernel uses `cache=True`.
- Callers reach kernels through the module, `from datanuggets import _ldjit` then `_ldjit.name(...)`, never `from datanuggets._ldjit import name`. Task 6's profiler swaps that module reference to book kernel time as compiled.
- `n_jobs == 1` runs the serial kernel, so numba's threading layer is never touched. Any other `n_jobs` uses numba threads, `min(n_jobs, NUMBA_NUM_THREADS)` or all of them for -1, under a process-wide lock. The thread count is restored afterwards.
- Real data only: every timing written into docs comes from a run made in this plan. No invented numbers.
- Fixtures are recorded only on the laptop's x87 R, never on the mini. This plan records none.
- Shell: Git Bash on Windows 11 from the repo root. `UV=/c/Users/thegr/AppData/Local/hermes/bin/uv.exe`, `RS="$LOCALAPPDATA/Programs/R/R-4.6.1/bin/Rscript.exe"`, `MR="bash $HOME/.claude/tools/mini-run.sh"` and `SCRATCH=/c/Users/thegr/AppData/Local/Temp/claude/C--Users-thegr/25ee381d-b280-4d5f-83df-1350491cb8e9/scratchpad` (the session scratchpad, for commit message files and profile output, never the repo). `$MR` alone runs the full suite on the mini on 8 workers. `$MR uv run pytest -q -n 8 -p no:cacheprovider <path>` runs part of it. The laptop's `$UV run pytest` works too when RAM allows.
- Commits: one logical change each, imperative subject <= 70 chars plus a why/what body. Write the message to a file in the scratchpad and commit with `git commit -F`. No `Claude-Session:`, `Co-Authored-By` or "Generated with" lines, and no mention of AI tools. Stage files by name, never `git add -A`. `git checkout --` any `docs/validation/*.md` that shows modified with no content change. Push `feat/speed` after each task's green commits.
- Repo root holds only toolchain files. Code in `src/datanuggets/`, tests in `tests/`, benchmarks in `benchmarks/`, docs in `docs/`.
- Session trio: update the tasklist, notes and log (`python scratchpad/f3/bk.py <tasklist|notes|log> append ...`, timestamp from `date`) on every task status change.
- Published prose (README, docstrings, report headers) follows the humanizer rules: no em or en dashes, no semicolons, contractions, no bold or italic in prose, no comma-pause chains, no test counts.

## Review Focus

1. Two Python threads calling `refine_dn(..., n_jobs=4)` at once, say from a threaded web server or joblib's threading backend. A person expects both calls to finish with the serial result, not a numba abort over concurrent parallel launches. Test `test_group_moments_from_two_threads_at_once` (Task 5).
2. `x` that isn't C-contiguous float64 (Fortran order, a strided slice, int columns) handed to `r_cov` or `r_mean`. Kernels read bit patterns, so a wrong view would silently read garbage. A person expects the answer for the contiguous float64 copy. Tests `test_r_cov_takes_any_layout_and_dtype` (Task 3) and `test_r_mean_takes_any_layout_and_dtype` (Task 4).
3. Values at double's edges: subnormals, signed zeros, sums and products past DBL_MAX, a large offset with a small spread. A person expects R's bits. Tests `test_r_cov_edge_columns_match_the_replay` (Task 3) and `test_r_mean_edge_lines_match_the_replay` (Task 4).
4. Wide data with many nuggets, where a (k, p, p) block of covariances wouldn't fit in memory. A person expects `refine_dn` to run in bounded memory with the same answer. Test `test_group_moments_chunking_gives_the_same_result` (Task 5).
5. `n_jobs` above the core count, or -1 while `NUMBA_NUM_THREADS` is limited. A person expects the thread count capped, no error, and numba's thread setting left as it was. Test `test_group_moments_caps_and_restores_threads` (Task 5).

---

### Task 1: numba dependency and the add, sub, mul and from-double kernels

**Files:**
- Modify: `pyproject.toml` (numba dependency through `uv add`, the `slow` marker)
- Modify: `uv.lock` (written by `uv add`)
- Modify: `README.md:25` (Install requirements line)
- Create: `src/datanuggets/_ldjit.py`
- Create: `tests/test_ldjit.py`
- Modify: `src/datanuggets/_ldouble.py:1-7` (module docstring, deferred minor M13)

**Interfaces:**
- Consumes: `_ldouble.add`, `_ldouble.sub`, `_ldouble.mul`, `_ldouble.from_float` (references, tests only).
- Produces: in `datanuggets._ldjit`, constants `Z64`, `ONE`, `TOP` (np.uint64) and `MAX_EXP = 960`, and `@njit(cache=True)` kernels `bitlen(x) -> int`, `mul64(a, b) -> (hi, lo)`, `round128(hi, lo, e) -> (m, e)`, `place(m, s) -> (hi, lo)`, `add(sa, ma, ea, sb, mb, eb) -> (s, m, e)`, `sub(...) -> (s, m, e)`, `mul(...) -> (s, m, e)`, `from_bits(bits) -> (s, m, e)`. Signs and exponents are int64, magnitudes uint64, and zero is `(0, Z64, 0)`. In `tests/test_ldjit.py`: helpers `canonical(v)`, `mags(rng, n)`, `odd_of_len(rng, n, length)`, `operands(rng, n, op)`, the dict `LONG_FUZZ` (op name to `mismatches(seed, n) -> list`) and constants `FAST`, `SLOW_CHUNKS`, `SLOW_CHUNK`. Task 2 adds to both files.

- [ ] **Step 1: Add numba as a runtime dependency**

Run: `$UV add "numba>=0.67"`
Expected: `pyproject.toml` `dependencies` gains `"numba>=0.67"` and `uv.lock` resolves `numba==0.67.0` plus `llvmlite`. Then `$UV run python -c "import numba; print(numba.__version__)"` prints `0.67.0`.

- [ ] **Step 2: Register the `slow` marker and leave it out of default runs**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` table with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -m 'not slow'"
markers = ["slow: the long long-double fuzz, 10,000,000 cases per operation (pytest -m slow)"]
```

A later `-m slow` on the command line overrides the default, since pytest keeps the last `-m`.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_ldjit.py`:

```python
"""The compiled long double kernels in _ldjit against the Python-int reference in _ldouble, bit for
bit, on a fixed-seed fuzz built to hit exact ties, sticky bits and exponent-gap edges."""

import numpy as np
import pytest
from numba import njit

from datanuggets import _ldjit as J
from datanuggets import _ldouble as ld

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


LONG_FUZZ = {
    "add": lambda seed, n: binary_mismatches("add", seed, n),
    "sub": lambda seed, n: binary_mismatches("sub", seed, n),
    "mul": lambda seed, n: binary_mismatches("mul", seed, n),
    "from_bits": from_bits_mismatches,
}


@pytest.mark.slow
@pytest.mark.parametrize("chunk", range(SLOW_CHUNKS))
@pytest.mark.parametrize("op", list(LONG_FUZZ))
def test_long_fuzz(op, chunk):
    seed = 1000 + 100 * chunk + list(LONG_FUZZ).index(op)
    assert LONG_FUZZ[op](seed, SLOW_CHUNK) == []
```

- [ ] **Step 4: Run the tests to see them fail**

Run: `$UV run pytest tests/test_ldjit.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'datanuggets._ldjit'`.

- [ ] **Step 5: Write the kernels**

Create `src/datanuggets/_ldjit.py`:

```python
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
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `$MR uv run pytest -q -n 8 -p no:cacheprovider tests/test_ldjit.py`
Expected: `4 passed`, 80 deselected (the slow chunks).

- [ ] **Step 7: Run the long fuzz on the mini**

Run: `$MR uv run pytest -q -n 8 -p no:cacheprovider -m slow tests/test_ldjit.py`
Expected: `80 passed` (4 operations x 20 chunks of 500,000). Log the wall time in the session log.

- [ ] **Step 8: Correct `_ldouble`'s docstring (deferred minor M13)**

Replace the module docstring of `src/datanuggets/_ldouble.py` with:

```python
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
```

- [ ] **Step 9: Name numba in the README's requirements**

In `README.md`, replace the line `Requires Python 3.12 or newer, \`numpy >= 2.0\`, \`scipy >= 1.13\` and \`joblib >= 1.4\`.` with:

```markdown
Requires Python 3.12 or newer, `numpy >= 2.0`, `scipy >= 1.13`, `joblib >= 1.4` and `numba >= 0.67`.
```

- [ ] **Step 10: Run the full suite on the mini**

Run: `$MR`
Expected: every test passes (the 883 from before plus the 4 new ones), no failures.

- [ ] **Step 11: Commit and push**

Message file 1 (`add-kernels.txt`):

```text
Add numba long double add, multiply and from-double kernels

R sums in x87 long double and the port replays that with Python ints,
which is exact but slow. _ldjit does the same arithmetic on (sign,
uint64 magnitude, exponent) triples in compiled code. The tests check
add, sub, mul and from_bits against _ldouble bit for bit on a
fixed-seed fuzz with constructed ties, sticky-bit exponent gaps and
near cancellation. A slow marker holds the 10M-case run for the mini.
numba becomes a runtime dependency.
```

```bash
git add pyproject.toml uv.lock README.md src/datanuggets/_ldjit.py tests/test_ldjit.py
git commit -F "$SCRATCH/speed/add-kernels.txt"
```

Message file 2 (`m13.txt`):

```text
Say where _ldouble's m can pass 64 bits

The docstring promised |m| < 2**64 for every value. Rounded results
keep that, but from_float returns a double's integer ratio unrounded,
so doubles from 2**64 up come back wider, and add passes them through
when the other side is zero. The spike's fuzz tripped on it.
```

```bash
git add src/datanuggets/_ldouble.py
git commit -F "$SCRATCH/speed/m13.txt"
git push
```

### Task 2: divide by a count, and round to double

**Files:**
- Modify: `src/datanuggets/_ldjit.py` (append four kernels, add `import math`)
- Modify: `tests/test_ldjit.py` (add `import math`, the fuzz for the new kernels, the sign test, three `LONG_FUZZ` entries)

**Interfaces:**
- Consumes: from Task 1, `_ldjit.Z64`, `ONE`, `TOP`, `MAX_EXP`, `bitlen`, and in the tests `canonical`, `mags`, `odd_of_len`, `FAST`, `LONG_FUZZ`. Reference: `_ldouble.div_int(v, n)`, `_ldouble.to_float(v)`, `_ldouble.from_float(d)`.
- Produces: `@njit(cache=True)` kernels `div_count(s, m, e, n) -> (s, m, e)` for a count 1 <= n < 2**63; `round53(s, m, e) -> (s, q, x)`, the nearest double as q * 2**x with q <= 2**53; `to_double(s, m, e) -> float`, which gives -0.0 on negative underflow and +-inf past DBL_MAX like `_ldouble.to_float`; `as_double(s, m, e) -> (s, m, e)`, the long double equal to that double, whose e is above `MAX_EXP` exactly when the double is +-inf. Tasks 3 to 5 build on all four.

- [ ] **Step 1: Write the failing tests**

In `tests/test_ldjit.py`, add `import math` above `import numpy as np`. Then insert this block directly above the line `LONG_FUZZ = {`:

```python
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
```

Then replace the `LONG_FUZZ` dict with:

```python
LONG_FUZZ = {
    "add": lambda seed, n: binary_mismatches("add", seed, n),
    "sub": lambda seed, n: binary_mismatches("sub", seed, n),
    "mul": lambda seed, n: binary_mismatches("mul", seed, n),
    "from_bits": from_bits_mismatches,
    "div_count": div_mismatches,
    "to_double": to_double_mismatches,
    "as_double": as_double_mismatches,
}
```

The chunk seeds are `1000 + 100 * chunk + index` with index < 100, so the seven operations never share one.

- [ ] **Step 2: Run the tests to see them fail**

Run: `$UV run pytest tests/test_ldjit.py -q`
Expected: 4 failures. The three fuzz tests fail with numba's `TypingError: ... Unknown attribute 'div_count' of type Module(...)` (or `to_double`, `as_double`) when it compiles the loop. The sign test fails with `AttributeError: module 'datanuggets._ldjit' has no attribute 'to_double'`. The Task 1 tests still pass.

- [ ] **Step 3: Write the kernels**

In `src/datanuggets/_ldjit.py`, add `import math` above `import numpy as np`, then append:

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `$MR uv run pytest -q -n 8 -p no:cacheprovider tests/test_ldjit.py`
Expected: `8 passed`, 140 deselected.

- [ ] **Step 5: Run the long fuzz for the new kernels on the mini**

Run: `$MR uv run pytest -q -n 8 -p no:cacheprovider -m slow tests/test_ldjit.py -k "div_count or to_double or as_double"`
Expected: `60 passed` (3 operations x 20 chunks). Log the wall time.

- [ ] **Step 6: Run the full suite on the mini**

Run: `$MR`
Expected: every test passes.

- [ ] **Step 7: Commit and push**

Message file (`div-round.txt`):

```text
Add long double division by a count and rounding to double

Means and covariances divide a long double sum by a row count, then
round to double. div_count does the division one bit at a time, since
numba has no 128-bit integer, and keeps a guard bit and the remainder
as sticky for ties to even. to_double rounds once, down to subnormals,
and keeps -0.0 and +-inf where _ldouble.to_float does. as_double reads
the double back as a long double, which cov needs for its column means.
```

```bash
git add src/datanuggets/_ldjit.py tests/test_ldjit.py
git commit -F "$SCRATCH/speed/div-round.txt"
git push
```

### Task 3: compiled cov()

**Files:**
- Modify: `src/datanuggets/_ldjit.py` (append `cov_rows`)
- Modify: `src/datanuggets/_rstats.py:1-8` (module docstring), `:17` (import), `:83-110` (`r_cov` becomes `_r_cov_py`, new `r_cov`)
- Modify: `tests/rfixtures.py` (add `same_bits`, `fixture_data`)
- Modify: `tests/test_rstats.py` (imports, new tests at the end)

**Interfaces:**
- Consumes: from Tasks 1 and 2, `_ldjit.from_bits`, `add`, `sub`, `mul`, `div_count`, `to_double`, `as_double`, `Z64`, `MAX_EXP`.
- Produces: `_ldjit.cov_rows(bits, rows, out) -> bool`. `bits` is a (n, p) uint64 view of finite float64 data, `rows` a 1-D int64 array of 2 or more row positions, and `out` a (p, p) float64 array that receives `cov(x[rows])`. It returns False, with `out` unfinished, only where a column mean rounds past DBL_MAX. In `_rstats`: `r_cov(x)` with an unchanged signature, and `_r_cov_py(x)`, today's body. In `tests/rfixtures.py`: `same_bits(a, b) -> bool` and `fixture_data() -> dict[str, tuple[np.ndarray, list[np.ndarray]]]`, which maps "group/case" to (x, [0-based assignments]). Tasks 4 and 5 use both helpers.

- [ ] **Step 1: Add the fixture helpers**

In `tests/rfixtures.py`, add `from functools import cache` to the imports and append:

```python
def same_bits(a, b) -> bool:
    """Equal as float64 bit patterns: -0.0 isn't 0.0, and a NaN equals only the same NaN."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


@cache
def fixture_data() -> dict[str, tuple[np.ndarray, list[np.ndarray]]]:
    """Each distinct finite x in the create_dn, refine_dn, rcompat_means and rcompat_stats
    fixtures, keyed "group/case" by the first case holding it, with the distinct 0-based nugget
    assignments recorded for it. The compiled long double tests compare against the Python
    replay, not R, so any real matrix serves."""
    first: dict[tuple, str] = {}
    out: dict[str, tuple[np.ndarray, dict[bytes, np.ndarray]]] = {}
    for group in ("create_dn", "refine_dn", "rcompat_means", "rcompat_stats"):
        for case in load_group(group):
            if not case.has("x"):
                continue
            x = np.ascontiguousarray(case.array("x"), dtype=np.float64)
            if x.ndim != 2 or not np.isfinite(x).all():
                continue
            key = first.setdefault((x.shape, x.tobytes()), f"{group}/{case.id}")
            sets = out.setdefault(key, (x, {}))[1]
            for name in ("assignments", "in_assignments"):
                if case.has(name):
                    a = case.array(name).astype(np.int64) - 1
                    sets.setdefault(a.tobytes(), a)
    return {key: (x, list(sets.values())) for key, (x, sets) in out.items()}
```

At plan time this gives 17 matrices and 59 assignment sets, 2,261 nonempty nuggets in all. The largest matrix is `create_dn/volcano_integer_original` (87 x 61).

- [ ] **Step 2: Write the failing tests**

In `tests/test_rstats.py`, change the imports to:

```python
import numpy as np
import pytest

import datanuggets._rstats as rstats
from datanuggets._rcompat import unique_first
from datanuggets._rstats import (
    _r_cov_py,
    r_cov,
    r_eigen_sym,
    r_mad,
    r_matvec,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
from rfixtures import ATOL, RTOL, fixture_data, load_group, same_bits
```

and append:

```python
DATA = fixture_data()
BIGGEST = max((k for k in DATA if DATA[k][1]), key=lambda k: DATA[k][0].size)


def _nuggets(a):
    """Each nugget's rows in ascending order, for 0-based assignments a."""
    order = np.argsort(a, kind="stable")
    return np.split(order, np.cumsum(np.bincount(a))[:-1])


@pytest.mark.parametrize("key", sorted(DATA))
def test_r_cov_matches_the_replay_on_every_fixture_nugget(key):
    x, sets = DATA[key]
    for rows in [np.arange(x.shape[0])] + [g for a in sets for g in _nuggets(a)]:
        if rows.size >= 2:
            assert same_bits(r_cov(x[rows]), _r_cov_py(x[rows])), rows[:5]


def test_r_cov_uses_the_compiled_path_for_finite_data(monkeypatch):
    def replay(x):
        raise AssertionError("r_cov fell back to the Python replay")

    monkeypatch.setattr(rstats, "_r_cov_py", replay)
    x = DATA[BIGGEST][0]
    assert r_cov(x).shape == (x.shape[1], x.shape[1])


EDGE_COLUMNS = {
    "subnormal": [5e-324, -5e-324, 1e-310, 0.0, 2.5e-320, -1e-315],
    "signed_zero": [0.0, -0.0, -0.0, 0.0, -0.0, 0.0],
    "constant": [3.7] * 6,
    "big_offset": [1e15 + 0.125 * i for i in range(6)],
    "near_one": [1.0, 1.0 + 2**-52, 1.0, 1.0 - 2**-53, 1.0, 1.0 + 2**-51],
    "huge": [1e300, -1e300, 1e300, -1e300, 1e300, 1.7e308],
}


def test_r_cov_edge_columns_match_the_replay():
    # every pair of columns meets in one cov() call, and "huge" gives covariances past DBL_MAX
    x = np.column_stack(list(EDGE_COLUMNS.values()))
    got = r_cov(x)
    assert same_bits(got, _r_cov_py(x))
    assert np.isinf(got[-1, -1])


def test_r_cov_takes_any_layout_and_dtype():
    x = DATA[BIGGEST][0]
    want = _r_cov_py(x)
    for view in (np.asfortranarray(x), np.repeat(x, 2, axis=0)[::2]):
        assert same_bits(r_cov(view), want)
    ints = np.rint(x * 10).astype(np.int64)
    assert same_bits(r_cov(ints), _r_cov_py(ints.astype(np.float64)))


@pytest.mark.parametrize(("bad", "error"), [(np.inf, OverflowError), (np.nan, ValueError)])
def test_r_cov_raises_on_inf_and_nan_like_the_replay(bad, error):
    x = np.array([[1.0, 2.0], [bad, 3.0], [4.0, 5.0]])
    with pytest.raises(error, match="cannot convert"):
        r_cov(x)
```

The last test already holds today. It pins the errors `r_cov` has to keep once it routes non-finite data to the replay.

- [ ] **Step 3: Run the tests to see them fail**

Run: `$UV run pytest tests/test_rstats.py -q`
Expected: collection error, `ImportError: cannot import name '_r_cov_py' from 'datanuggets._rstats'`.

- [ ] **Step 4: Write the kernel**

Append to `src/datanuggets/_ldjit.py`:

```python
@njit(cache=True)
def cov_rows(bits, rows, out):
    """cov(x[rows]) into out, for 2 or more rows: each column mean in two long double passes,
    stored as a double, then long double sums of the products of deviations divided by n - 1.
    Returns False, leaving out unfinished, where a column mean is past DBL_MAX."""
    n = rows.size
    p = bits.shape[1]
    ds = np.empty((p, n), np.int64)
    dm = np.empty((p, n), np.uint64)
    de = np.empty((p, n), np.int64)
    for j in range(p):
        s, m, e = 0, Z64, 0
        for i in range(n):
            vs, vm, ve = from_bits(bits[rows[i], j])
            s, m, e = add(s, m, e, vs, vm, ve)
        ms, mm, me = div_count(s, m, e, n)
        if as_double(ms, mm, me)[2] <= MAX_EXP:  # R: if (R_FINITE((double) tmp))
            s, m, e = 0, Z64, 0
            for i in range(n):
                vs, vm, ve = from_bits(bits[rows[i], j])
                ts, tm, te = sub(vs, vm, ve, ms, mm, me)
                s, m, e = add(s, m, e, ts, tm, te)
            cs, cm, ce = div_count(s, m, e, n)
            ms, mm, me = add(ms, mm, me, cs, cm, ce)
        ms, mm, me = as_double(ms, mm, me)  # R keeps the column means as doubles
        if me > MAX_EXP:
            return False
        for i in range(n):
            vs, vm, ve = from_bits(bits[rows[i], j])
            ds[j, i], dm[j, i], de[j, i] = sub(vs, vm, ve, ms, mm, me)
    for a in range(p):
        for b in range(a + 1):
            s, m, e = 0, Z64, 0
            for i in range(n):
                ps, pm, pe = mul(ds[a, i], dm[a, i], de[a, i], ds[b, i], dm[b, i], de[b, i])
                s, m, e = add(s, m, e, ps, pm, pe)
            qs, qm, qe = div_count(s, m, e, n - 1)
            out[a, b] = to_double(qs, qm, qe)
            out[b, a] = out[a, b]
    return True
```

- [ ] **Step 5: Route `r_cov` through the kernel**

In `src/datanuggets/_rstats.py`, replace the module docstring with:

```python
"""R's summary statistics as refine.DN uses them.

sum, mean, median, mad, type-7 quantile, cov and %*% are replayed bit for bit. R accumulates
sums in long double (x87 extended precision on x86-64). cov runs that arithmetic compiled, in
_ldjit. sum and mean replay it with Python integers through _ldouble, which is also the reference
the compiled code is tested against. eigen() calls LAPACK's dsyevr with R's arguments, so
eigenvalues agree with R's to rounding on the scale of the largest one, not bit for bit: LAPACK
builds differ in their kernels. A second eigenvalue near zero can be far off relative to itself.
"""
```

Change the import line `from datanuggets import _ldouble as ld` to:

```python
from datanuggets import _ldjit
from datanuggets import _ldouble as ld
```

Rename today's `def r_cov(x: np.ndarray) -> np.ndarray:` to `def _r_cov_py(x: np.ndarray) -> np.ndarray:` and give it this docstring, leaving the body as it is:

```python
    """cov(x) replayed with Python integers: r_cov's path for data with Inf or NaN, which it
    rejects as from_float does, and the reference the compiled path is tested against."""
```

Then add, directly above `_r_cov_py`:

```python
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
    if np.isfinite(x).all() and _ldjit.cov_rows(x.view(np.uint64), rows, out):
        return out
    return _r_cov_py(x)
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `$MR uv run pytest -q -n 8 -p no:cacheprovider tests/test_rstats.py tests/test_refine.py`
Expected: all pass, including 22 new tests: 17 fixture matrices, the compiled-path check, the edge columns, the layout test and 2 error tests.

- [ ] **Step 7: Run the full suite on the mini**

Run: `$MR`
Expected: every test passes.

- [ ] **Step 8: Commit and push**

Message file (`cov.txt`):

```text
Compute cov() with compiled long double arithmetic

refine_dn calls cov() once per nugget and once per split, and the
Python-int replay made it the slowest step. cov_rows runs the same two
mean passes, the double-rounded column means and the product sums in
numba. Finite data takes the kernel and anything else takes the old
replay, now _r_cov_py, so Inf and NaN still raise the same errors. The
tests check every fixture nugget, double's edges, and odd layouts and
dtypes against the replay bit for bit.
```

```bash
git add src/datanuggets/_ldjit.py src/datanuggets/_rstats.py tests/rfixtures.py tests/test_rstats.py
git commit -F "$SCRATCH/speed/cov.txt"
git push
```

### Task 4: compiled colMeans and rowMeans

**Files:**
- Modify: `src/datanuggets/_ldjit.py` (append `line_means`)
- Modify: `src/datanuggets/_rcompat.py:1-8` (import), `:78-98` (`r_mean` becomes `_r_mean_py`, new `r_mean`)
- Modify: `tests/test_rcompat.py` (imports, new tests at the end)

**Interfaces:**
- Consumes: from Tasks 1 and 2, `_ldjit.from_bits`, `add`, `div_count`, `to_double`, `Z64`. From Task 3, `rfixtures.fixture_data` and `rfixtures.same_bits`.
- Produces: `_ldjit.line_means(bits) -> np.ndarray`. `bits` is a (k, n) uint64 view of finite float64 lines, and the result is each line's mean (NaN for n == 0). In `_rcompat`: `r_mean(a, axis)` with an unchanged signature, and `_r_mean_py(a, axis)`, today's body.

- [ ] **Step 1: Write the failing tests**

In `tests/test_rcompat.py`, change the imports to:

```python
import math
from fractions import Fraction

import numpy as np
import pytest

import datanuggets._rcompat as rcompat
from datanuggets._rcompat import _r_mean_py, r_mean, split_indices
from rfixtures import fixture_data, load_group, same_bits
```

and append:

```python
DATA = fixture_data()
BIGGEST = max(DATA, key=lambda k: DATA[k][0].size)


@pytest.mark.parametrize("key", sorted(DATA))
def test_r_mean_matches_the_replay_on_every_fixture_matrix(key):
    x = DATA[key][0]
    for axis in (0, 1):
        assert same_bits(r_mean(x, axis), _r_mean_py(x, axis)), axis


def test_r_mean_uses_the_compiled_path_for_finite_lines(monkeypatch):
    def replay(v):
        raise AssertionError("r_mean fell back to the Python replay")

    monkeypatch.setattr(rcompat, "_ldouble_mean", replay)
    x = DATA[BIGGEST][0]
    assert r_mean(x, 0).shape == (x.shape[1],)


EDGE_LINES = [
    [5e-324, -5e-324, 1e-310, 2.5e-320],
    [-0.0, -0.0, -0.0],
    [0.0, -0.0],
    [-1.5e-323, 5e-324],
    [1e308, 1e308, -1e308],
    [1.7e308, 1.7e308, 1.7e308, 1.7e308],
    [1e16, 1.0, -1e16],
    [1.0, 2.0**-60, 2.0**-60, 2.0**-60],
]


def test_r_mean_edge_lines_match_the_replay():
    for line in EDGE_LINES:
        a = np.array([line])
        assert same_bits(r_mean(a, 1), _r_mean_py(a, 1)), line


def test_r_mean_takes_any_layout_and_dtype():
    x = DATA[BIGGEST][0]
    for axis in (0, 1):
        want = _r_mean_py(x, axis)
        for view in (np.asfortranarray(x), np.repeat(x, 2, axis=0)[::2]):
            assert same_bits(r_mean(view, axis), want), axis
    ints = np.rint(x * 10).astype(np.int64)
    assert same_bits(r_mean(ints, 0), _r_mean_py(ints.astype(np.float64), 0))
    cube = np.stack([x, -x])
    assert same_bits(r_mean(cube, 1), _r_mean_py(cube, 1))


def test_r_mean_mixes_finite_and_non_finite_lines():
    x = np.array(
        [
            [1.0, 2.0, 4.0],
            [math.inf, 1.0, 2.0],
            [0.1, 0.2, 0.3],
            [math.nan, 1.0, 1.0],
            [math.inf, -math.inf, 0.0],
        ]
    )
    got, want = r_mean(x, 1), _r_mean_py(x, 1)
    np.testing.assert_array_equal(got, want)  # NaN matches NaN here
    assert same_bits(got[[0, 2]], want[[0, 2]])
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `$UV run pytest tests/test_rcompat.py -q`
Expected: collection error, `ImportError: cannot import name '_r_mean_py' from 'datanuggets._rcompat'`.

- [ ] **Step 3: Write the kernel**

Append to `src/datanuggets/_ldjit.py`:

```python
@njit(cache=True)
def line_means(bits):
    """rowMeans of a finite matrix given as bit patterns: each row summed in long double from the
    left, divided by its length, then rounded to double once."""
    k, n = bits.shape
    out = np.empty(k)
    if n == 0:
        out[:] = np.nan
        return out
    for r in range(k):
        s, m, e = 0, Z64, 0
        for i in range(n):
            vs, vm, ve = from_bits(bits[r, i])
            s, m, e = add(s, m, e, vs, vm, ve)
        qs, qm, qe = div_count(s, m, e, n)
        out[r] = to_double(qs, qm, qe)
    return out
```

- [ ] **Step 4: Route `r_mean` through the kernel**

In `src/datanuggets/_rcompat.py`, add below `import numpy as np` (after a blank line):

```python
from datanuggets import _ldjit
```

Rename today's `def r_mean(a: np.ndarray, axis: int) -> np.ndarray:` to `def _r_mean_py(a: np.ndarray, axis: int) -> np.ndarray:` and replace its docstring with the following, leaving the body as it is:

```python
    """r_mean replayed with Python integers, about 0.4 s per million values. It's the
    reference the compiled path is tested against."""
```

Then add, directly above `_r_mean_py`:

```python
def r_mean(a: np.ndarray, axis: int) -> np.ndarray:
    """R's colMeans (axis=0) or rowMeans (axis=1) of a 2-D double matrix, bit for bit.

    R accumulates each line in long double, which on x86-64 is x87 extended precision with a
    64-bit significand, divides in long double, and only then rounds to double. numpy's
    pairwise double mean differs by several ulps, and on columns that cancel to nearly zero
    (standardized data) even a correctly rounded mean differs, because R's own running sum
    rounds. Finite lines run that arithmetic compiled, in _ldjit. A line with Inf or NaN gets
    numpy's mean: R propagates those silently, and the answer doesn't depend on precision.
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
    out[finite] = _ldjit.line_means(done.view(np.uint64))
    with np.errstate(invalid="ignore"):
        for i in np.flatnonzero(~finite).tolist():
            out[i] = np.mean(flat[i])
    return out.reshape(shape)
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `$MR uv run pytest -q -n 8 -p no:cacheprovider tests/test_rcompat.py tests/test_create.py tests/test_means.py`
Expected: all pass, including 21 new tests: 17 fixture matrices, the compiled-path check, the edge lines, the layout test and the mixed lines.

- [ ] **Step 6: Run the full suite on the mini**

Run: `$MR`
Expected: every test passes.

- [ ] **Step 7: Commit and push**

Message file (`mean.txt`):

```text
Compute colMeans and rowMeans with compiled long double

create_dn takes colMeans of all of x and get_dn_means one per nugget,
both through the Python-int replay. line_means sums each line in
numba's long double and rounds once. Lines with Inf or NaN keep numpy's
mean as before, and the replay stays as _r_mean_py for the tests,
which check every fixture matrix on both axes and double's edges bit
for bit.
```

```bash
git add src/datanuggets/_ldjit.py src/datanuggets/_rcompat.py tests/test_rcompat.py
git commit -F "$SCRATCH/speed/mean.txt"
git push
```

### Task 5: every nugget's center and covariance in one compiled pass

**Files:**
- Modify: `src/datanuggets/_ldjit.py` (append `EXP_BITS`, `_group`, `group_moments_serial`, `group_moments_parallel`, and import `prange`)
- Modify: `src/datanuggets/_rstats.py` (imports, add `_MOMENTS_CHUNK`, `_THREADS`, `group_moments`, `_run_threaded`)
- Modify: `src/datanuggets/refine.py:32-42` (imports), `:76-78` (refine_dn docstring), `:116` (`_Nuggets` call), `:184-197` (delete `nugget_features`), `:260-296` (`_Nuggets.__init__` and `refresh`)
- Modify: `tests/test_rstats.py` (imports, group tests at the end)
- Modify: `tests/test_refine.py:14` (imports), `:133-157` (two tests)

**Interfaces:**
- Consumes: from Task 3, `_ldjit.cov_rows`, `_rstats.r_cov`, `_rstats._r_cov_py`, `rfixtures.fixture_data`, `rfixtures.same_bits`, and in `tests/test_rstats.py` the names `DATA`, `BIGGEST`, `_nuggets`. From Task 4, `_rcompat.r_mean` and `_rcompat._r_mean_py`. `refine.r_eigen_sym`, `refine.r_sum`.
- Produces: `_ldjit.group_moments_serial(bits, order, starts, counts, centers, covs, ok)` and `_ldjit.group_moments_parallel(...)` with the same arguments. Group g's rows are `order[starts[g]:starts[g] + counts[g]]`. The kernels write `centers[g]` (NaN for no rows) and, for 2 or more rows, `covs[g]`, and set `ok[g]` False where a value is Inf or NaN or `cov_rows` returns False. In `_rstats`: `group_moments(x, groups, n_jobs=1)`, a generator of `(center, cov)` per group in order, where `cov` is None below 2 rows. The values are the bits of `r_mean(x[g], 0)` and `r_cov(x[g])`. Also `_MOMENTS_CHUNK = 1 << 22`, `_THREADS` (a `threading.Lock`) and `_run_threaded(kernel, args, n_jobs)`. In `refine`: `_Nuggets(x, assignments, k, n_jobs=1)`, and `nugget_features` is gone. Task 6 profiles `group_moments` through `_ldjit`.

- [ ] **Step 1: Write the failing group tests**

In `tests/test_rstats.py`, replace the imports (everything from `import numpy as np` through the `from rfixtures import ...` line, below the module docstring) with:

```python
import threading
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import pytest

import datanuggets._rstats as rstats
from datanuggets._rcompat import _r_mean_py, unique_first
from datanuggets._rstats import (
    _r_cov_py,
    group_moments,
    r_cov,
    r_eigen_sym,
    r_mad,
    r_matvec,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
from rfixtures import ATOL, RTOL, fixture_data, load_group, same_bits
```

Then append:

```python
MOMENT_SETS = [(k, i) for k in sorted(DATA) for i in range(len(DATA[k][1]))]


def _same_moments(got, want) -> bool:
    return len(got) == len(want) and all(
        same_bits(c, wc) and (v is None if wv is None else same_bits(v, wv))
        for (c, v), (wc, wv) in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize(("key", "index"), MOMENT_SETS, ids=[f"{k}#{i}" for k, i in MOMENT_SETS])
def test_group_moments_match_the_replay_on_every_fixture_nugget(key, index, n_jobs):
    x, sets = DATA[key]
    groups = _nuggets(sets[index])
    got = list(group_moments(x, groups, n_jobs))
    want = [(_r_mean_py(x[g], 0), _r_cov_py(x[g]) if g.size >= 2 else None) for g in groups]
    assert _same_moments(got, want)


def test_group_moments_chunking_gives_the_same_result(monkeypatch):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    p = x.shape[1]
    for chunk in (1, 3 * p * p):  # one group per call, then three
        monkeypatch.setattr(rstats, "_MOMENTS_CHUNK", chunk)
        assert _same_moments(list(group_moments(x, groups)), want), chunk
        assert _same_moments(list(group_moments(x, groups, 4)), want), chunk


@pytest.mark.parametrize("n_jobs", [-1, 10**6])
def test_group_moments_caps_and_restores_threads(monkeypatch, n_jobs):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    before = numba.get_num_threads()
    asked = []
    real = numba.set_num_threads

    def spy(k):
        asked.append(k)
        real(k)

    monkeypatch.setattr(numba, "set_num_threads", spy)
    assert _same_moments(list(group_moments(x, groups, n_jobs)), want)
    assert asked == [numba.config.NUMBA_NUM_THREADS, before]
    assert numba.get_num_threads() == before


def test_group_moments_from_two_threads_at_once():
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    start = threading.Barrier(2)

    def run():
        start.wait()
        return list(group_moments(x, groups, 4))

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results = [f.result() for f in futures]
    assert all(_same_moments(got, want) for got in results)


def test_group_moments_leaves_non_finite_groups_to_the_replay():
    x = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0], [6.0, 7.0]])
    groups = [np.array([1]), np.array([2, 3]), np.array([], dtype=np.int64)]
    (c0, v0), (c1, v1), (c2, v2) = group_moments(x, groups)
    assert same_bits(c0, [np.inf, 3.0]) and v0 is None
    assert same_bits(c1, _r_mean_py(x[[2, 3]], 0)) and same_bits(v1, _r_cov_py(x[[2, 3]]))
    assert np.isnan(c2).all() and v2 is None
    with pytest.raises(OverflowError):
        list(group_moments(x, [np.array([0, 1])]))
```

- [ ] **Step 2: Rewrite the refine tests that used `nugget_features`**

In `tests/test_refine.py`, replace the line `from datanuggets.refine import _split_nugget, nugget_features` with:

```python
from datanuggets._validate import check_assignments
from datanuggets.refine import _Nuggets, _split_nugget
```

Replace `test_initial_nugget_features_match_r` with:

```python
@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("case", _first_case_per_input(), ids=lambda c: c.id)
def test_initial_nugget_features_match_r(case, n_jobs):
    x, dn, _ = refine_inputs(case)
    state = _Nuggets(x, check_assignments(dn, *x.shape), dn.centers.shape[0], n_jobs)
    state.refresh(range(state.size), state.members())
    np.testing.assert_array_equal(state.centers, case.array("init_centers"))
    np.testing.assert_array_equal(state.scales, case.array("init_scale"))
    if x.shape[1] == 1:  # EV1 = EV2 = the variance, which is exact
        np.testing.assert_array_equal(state.ev1, case.array("init_ev1"))
        np.testing.assert_array_equal(state.ev2, case.array("init_ev2"))
    else:
        np.testing.assert_allclose(state.ev1, case.array("init_ev1"), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(state.ev2, case.array("init_ev2"), rtol=RTOL, atol=ATOL)
```

In `test_n_jobs_does_not_change_the_result`, replace the `two = ...` line and the final assert with:

```python
        two = refine_dn(x, dn, seed=make_seed(), n_jobs=2, **kwargs)
        four = refine_dn(x, dn, seed=make_seed(), n_jobs=4, **kwargs)
    assert one.n_nuggets > np.count_nonzero(dn.weights)  # something was split
    assert one == two == four
```

(the existing `assert one.n_nuggets ...` line moves below `four`, unchanged).

- [ ] **Step 3: Run the tests to see them fail**

Run: `$UV run pytest tests/test_rstats.py tests/test_refine.py -q`
Expected: `tests/test_rstats.py` fails at collection with `ImportError: cannot import name 'group_moments' from 'datanuggets._rstats'`. In `tests/test_refine.py`, every `test_initial_nugget_features_match_r` case fails with `TypeError: _Nuggets.__init__() takes 4 positional arguments but 5 were given`. The n_jobs test still passes (joblib already gives the same result at 4).

- [ ] **Step 4: Write the group kernels**

In `src/datanuggets/_ldjit.py`, change `from numba import njit` to `from numba import njit, prange`, add below `MAX_EXP`:

```python
EXP_BITS = np.uint64(0x7FF << 52)  # a double's exponent field, all ones for Inf and NaN
```

and append:

```python
@njit(cache=True)
def _group(bits, rows, center, cov):
    """One group's colMeans into center (NaN for no rows) and, for 2 or more rows, its cov()
    into cov. False, leaving both unfinished, where a value is Inf or NaN or cov_rows gives
    False: the caller redoes that group in Python."""
    n = rows.size
    p = bits.shape[1]
    if n == 0:
        center[:] = np.nan
        return True
    for i in range(n):
        for j in range(p):
            if (bits[rows[i], j] & EXP_BITS) == EXP_BITS:
                return False
    for j in range(p):
        s, m, e = 0, Z64, 0
        for i in range(n):
            vs, vm, ve = from_bits(bits[rows[i], j])
            s, m, e = add(s, m, e, vs, vm, ve)
        qs, qm, qe = div_count(s, m, e, n)
        center[j] = to_double(qs, qm, qe)
    if n < 2:
        return True
    return cov_rows(bits, rows, cov)


@njit(cache=True)
def group_moments_serial(bits, order, starts, counts, centers, covs, ok):
    """_group for each group in turn. Group g's rows are order[starts[g]:starts[g] + counts[g]]."""
    for g in range(starts.size):
        rows = order[starts[g] : starts[g] + counts[g]]
        ok[g] = _group(bits, rows, centers[g], covs[g])


@njit(cache=True, parallel=True)
def group_moments_parallel(bits, order, starts, counts, centers, covs, ok):
    """group_moments_serial with the groups spread over numba's threads. Each group runs the
    same serial arithmetic, so the result doesn't depend on the thread count."""
    for g in prange(starts.size):
        rows = order[starts[g] : starts[g] + counts[g]]
        ok[g] = _group(bits, rows, centers[g], covs[g])
```

- [ ] **Step 5: Add `group_moments` to `_rstats`**

In `src/datanuggets/_rstats.py`, make the imports:

```python
import math
import threading
from collections.abc import Iterator

import numba
import numpy as np
from scipy.linalg import LinAlgError, lapack

from datanuggets import _ldjit
from datanuggets import _ldouble as ld
from datanuggets._defaults import MAD_CONSTANT
from datanuggets._rcompat import r_mean
```

Replace the module docstring with:

```python
"""R's summary statistics as refine.DN uses them.

sum, mean, median, mad, type-7 quantile, cov and %*% are replayed bit for bit. R accumulates
sums in long double (x87 extended precision on x86-64). cov and group_moments, which gives
every nugget's colMeans and cov in one call, run that arithmetic compiled, in _ldjit. sum and
mean replay it with Python integers through _ldouble, which is also the reference the compiled
code is tested against. eigen() calls LAPACK's dsyevr with R's arguments, so eigenvalues agree
with R's to rounding on the scale of the largest one, not bit for bit: LAPACK builds differ in
their kernels. A second eigenvalue near zero can be far off relative to itself.
"""
```

Append below `_r_cov_py`:

```python
_MOMENTS_CHUNK = 1 << 22  # covariance entries group_moments holds at once (32 MB)
_THREADS = threading.Lock()


def group_moments(
    x: np.ndarray, groups: list[np.ndarray], n_jobs: int = 1
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """(colMeans, cov) of x[g] for each g in groups, in order, bit for bit r_mean(x[g], 0) and
    r_cov(x[g]), with cov None below 2 rows. A group with no rows has a NaN center. Groups run
    compiled, a chunk at a time, and n_jobs other than 1 spreads each chunk over numba threads
    (all of them for -1). A group with Inf or NaN is redone through r_mean and r_cov."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    bits = x.view(np.uint64)
    p = x.shape[1]
    per = max(1, _MOMENTS_CHUNK // (p * p))
    for lo in range(0, len(groups), per):
        chunk = groups[lo : lo + per]
        counts = np.array([g.size for g in chunk], dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        order = np.concatenate(chunk).astype(np.int64)
        centers = np.empty((len(chunk), p))
        covs = np.empty((len(chunk), p, p))
        ok = np.empty(len(chunk), dtype=np.bool_)
        args = (bits, order, starts, counts, centers, covs, ok)
        if n_jobs == 1:
            _ldjit.group_moments_serial(*args)
        else:
            _run_threaded(_ldjit.group_moments_parallel, args, n_jobs)
        for i, g in enumerate(chunk):
            if ok[i]:
                yield centers[i], (covs[i] if g.size >= 2 else None)
            else:
                yield r_mean(x[g], 0), (r_cov(x[g]) if g.size >= 2 else None)


def _run_threaded(kernel, args, n_jobs: int) -> None:
    """kernel(*args) on min(n_jobs, NUMBA_NUM_THREADS) numba threads, or all of them for -1.
    One parallel launch at a time: numba's workqueue threading layer aborts the process on
    concurrent launches. The caller's thread count is restored afterwards."""
    cap = numba.config.NUMBA_NUM_THREADS
    threads = cap if n_jobs == -1 else min(n_jobs, cap)
    with _THREADS:
        before = numba.get_num_threads()
        numba.set_num_threads(threads)
        try:
            kernel(*args)
        finally:
            numba.set_num_threads(before)
```

Each group's deviation arrays inside `cov_rows` take 24 bytes per value of that group. They're freed when the group finishes, so memory in flight is at most 24 bytes per row per column for the groups running at once, on top of the chunk's `covs`.

- [ ] **Step 6: Make `refine_dn` use it**

In `src/datanuggets/refine.py`:

Delete the line `from datanuggets._rcompat import r_mean` and add `group_moments` to the `datanuggets._rstats` import list (alphabetically first):

```python
from datanuggets._rstats import (
    group_moments,
    r_cov,
    r_eigen_sym,
    r_mad,
    r_matvec,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
```

In `refine_dn`'s docstring, replace the last paragraph (the three lines starting `R re-seeds before every split`) with:

```text
    R re-seeds before every split, so each split draws from a fresh copy of seed's stream
    (Draws.fork) and the result is the same for any n_jobs. n_jobs other than 1 runs each
    round's splits in joblib worker processes and computes the nuggets' centers and
    covariances on that many numba threads (all cores for -1). progress=True shows a tqdm bar
    per round.
```

Replace `state = _Nuggets(x, check_assignments(dn, n, p), dn.centers.shape[0])` with:

```python
    state = _Nuggets(x, check_assignments(dn, n, p), dn.centers.shape[0], n_jobs)
```

Delete the function `nugget_features` and the two blank lines after it.

In `_Nuggets`, replace `__init__` and `refresh` with:

```python
    def __init__(self, x: np.ndarray, assignments: np.ndarray, k: int, n_jobs: int = 1):
        self.x = x
        self.n_jobs = n_jobs
        self.assignments = assignments.copy()
        self.centers = np.full((k, x.shape[1]), np.nan)
        self.weights = np.zeros(k, dtype=np.int64)
        self.scales = np.zeros(k)
        self.ev1 = np.zeros(k)
        self.ev2 = np.zeros(k)
```

```python
    def refresh(self, positions, members: list[np.ndarray]) -> None:
        """refresh_nuggets_info: recompute these positions' center (colMeans), weight, scale
        (the trace of the sample covariance over p) and two largest eigenvalues from their rows.
        An empty nugget has a NaN center, a single row has scale and eigenvalues 0, and with one
        column EV2 = EV1."""
        positions = list(positions)
        p = self.x.shape[1]
        moments = group_moments(self.x, [members[c] for c in positions], self.n_jobs)
        for c, (center, cov) in zip(positions, moments, strict=True):
            self.centers[c] = center
            self.weights[c] = members[c].size
            if cov is None:
                self.scales[c] = self.ev1[c] = self.ev2[c] = 0.0
                continue
            self.scales[c] = r_sum(np.diag(cov)) / p
            ev = cov.ravel() if p == 1 else r_eigen_sym(cov)
            self.ev1[c] = float(ev[0])
            self.ev2[c] = float(ev[1] if ev.size > 1 else ev[0])
```

- [ ] **Step 7: Run the tests to see them pass**

Run: `$MR uv run pytest -q -n 8 -p no:cacheprovider tests/test_rstats.py tests/test_refine.py`
Expected: all pass. The new ones are 118 fixture-set cases (59 sets x n_jobs 1 and 4), the chunking test, 2 thread-cap cases, the two-thread test, the non-finite test, and the init-features test now run at n_jobs 1 and 4.

Run: `$MR uv run python -c "import numba, numpy as np; from datanuggets import _ldjit; b = np.zeros((3, 1), np.uint64); _ldjit.group_moments_parallel(b, np.arange(3), np.array([0]), np.array([3]), np.empty((1, 1)), np.empty((1, 1, 1)), np.empty(1, np.bool_)); print(numba.threading_layer(), numba.config.NUMBA_NUM_THREADS)"`
Expected: the mini's threading layer and thread count (for example `workqueue 10`). Write both in the session notes. The lock in `_run_threaded` is what keeps a workqueue layer safe.

- [ ] **Step 8: Run the full suite on the mini**

Run: `$MR`
Expected: every test passes.

- [ ] **Step 9: Commit and push**

Message file (`moments.txt`):

```text
Compute every nugget's center and covariance in one compiled pass

refine_dn's first pass called colMeans and cov() once per nugget from
Python, 50 s against R's 0.9 s on full Covertype. group_moments hands
a whole chunk of nuggets to one numba call, threaded over the nuggets
when n_jobs isn't 1. Each nugget's arithmetic is the same serial loop,
so the bits don't depend on the thread count. A lock keeps parallel
launches one at a time, since numba's workqueue layer aborts on
concurrent ones, and the thread count is put back afterwards.
nugget_features goes, and _Nuggets.refresh takes the moments from
group_moments.
```

```bash
git add src/datanuggets/_ldjit.py src/datanuggets/_rstats.py src/datanuggets/refine.py tests/test_rstats.py tests/test_refine.py
git commit -F "$SCRATCH/speed/moments.txt"
git push
```

### Task 6: profile against the stopping rule (gate)

**Files:**
- Modify: `benchmarks/covertype.py` (add `warm_up`)
- Create: `benchmarks/profile_stop.py`

**Interfaces:**
- Consumes: `benchmarks/covertype.py`'s `load`, `compare`, `run`, `stage_lines`; `_rcompat.r_mean`, `_rstats.r_cov`, `_rstats.group_moments` (Tasks 3 to 5); `_rstats._ldjit` and `_rcompat._ldjit`, the module references the Global Constraints require.
- Produces: `covertype.warm_up(x) -> None`, which loads or compiles every kernel on 50 rows of x. Task 7 calls it too. `benchmarks/profile_stop.py [size]` prints each run's wall time, whether its outputs match R, the top Python functions in `src/datanuggets` by cProfile tottime, and the compiled kernels' time. It exits 1 when the rule fails or an output differs.

This task has no unit test. It's a measuring script, and its checks run inside it: R parity on every output, and a fail-closed check that kernel time was actually booked.

- [ ] **Step 1: Add `warm_up` to `benchmarks/covertype.py`**

Change the `from datanuggets import ...` line to:

```python
from datanuggets import DataNuggets, DataNuggetWarning, _rcompat, _rstats, create_dn, refine_dn
```

and add above `def run(`:

```python
def warm_up(x: np.ndarray) -> None:
    """Load (or on a cold cache compile) every numba kernel, so no timing includes it."""
    head = np.ascontiguousarray(x[:50], dtype=np.float64)
    _rcompat.r_mean(head, 0)
    _rstats.r_cov(head)
    list(_rstats.group_moments(head, [np.arange(50)]))
    list(_rstats.group_moments(head, [np.arange(50)], n_jobs=2))
```

Call it in `main` right after `x = cr.array("x")`:

```python
    warm_up(x)
```

- [ ] **Step 2: Write the profiler**

Create `benchmarks/profile_stop.py`:

```python
"""Checks spec section 9.5's stopping rule on Covertype: no function written in Python in
src/datanuggets may have a cProfile tottime above 10% of the run's wall time, for create_dn and
for refine_dn (shape splits off), both with n_jobs=1 on R's replayed draws.

cProfile can't see into numba, so a kernel's time would land in the Python function that called
it. For the run, every kernel is wrapped in a Python function named compiled_<kernel>, which
books the time under that name instead. Run from the repo root after benchmarks/covertype.R
with the same size:

    uv run python benchmarks/profile_stop.py [10000 | 100000]

Exits 1 if the rule fails, no kernel time was booked, or an output stops matching R's.
"""

import cProfile
import os
import pstats
import sys
from pathlib import Path

from numba.core.dispatcher import Dispatcher

from covertype import compare, load, run, stage_lines, warm_up
from datanuggets import DataNuggets, _ldjit, _rcompat, _rstats, create_dn, refine_dn

SRC = os.path.normcase(str(Path(_ldjit.__file__).resolve().parent))
LIMIT = 0.10
TOP = 8


def booked(name, kernel):
    """A Python function that calls kernel, named compiled_<name> so cProfile books its time."""

    def call(*args):
        return kernel(*args)

    label = f"compiled_{name}"
    call.__code__ = call.__code__.replace(co_name=label, co_qualname=label)
    return call


class Booked:
    """_ldjit's attributes, with every kernel swapped for its booked wrapper."""

    def __init__(self, module):
        for name, value in vars(module).items():
            if not name.startswith("__"):
                setattr(self, name, booked(name, value) if isinstance(value, Dispatcher) else value)


def profiled(fn):
    prof = cProfile.Profile()
    prof.enable()
    try:
        result = fn()
    finally:
        prof.disable()
    return result, pstats.Stats(prof)


def report(label: str, stats: pstats.Stats, secs: float, checks: dict) -> bool:
    ours, compiled = [], []
    for (path, line, name), (_, _, tottime, _, _) in stats.stats.items():
        if name.startswith("compiled_"):
            compiled.append((tottime, name))
        elif os.path.normcase(os.path.abspath(path)).startswith(SRC):
            ours.append((tottime, f"{Path(path).name}:{line} {name}"))
    ours.sort(reverse=True)
    compiled.sort(reverse=True)
    parity = all(v is True for v in checks.values())
    worst = ours[0][0] / secs if ours else 0.0
    print(f"{label}: {secs:.2f} s wall, outputs {'match R' if parity else f'DIFFER {checks}'}")
    print(f"  Python in src/datanuggets by own time (limit {LIMIT:.0%} of wall):")
    for tottime, where in ours[:TOP]:
        print(f"    {tottime / secs:6.1%} {tottime:8.2f} s  {where}")
    print("  compiled kernels:")
    for tottime, name in compiled:
        print(f"    {tottime / secs:6.1%} {tottime:8.2f} s  {name}")
    if not compiled:
        print("  no kernel time was booked: a caller holds a kernel by name, not via _ldjit")
    passed = worst <= LIMIT
    print(f"  stopping rule: {'PASS' if passed else 'FAIL'} (largest {worst:.1%})")
    return parity and passed and bool(compiled)


def main(size: str) -> int:
    cid = f"covertype_{size}"
    cr = load(cid + "_create")
    x = cr.array("x")
    warm_up(x)
    _rstats._ldjit = _rcompat._ldjit = Booked(_ldjit)
    try:
        (dn, secs, extra), stats = profiled(
            lambda: run(lambda d: create_dn(x, seed=d), cr, stage_lines(cr.manifest["messages"]))
        )
        good = report(f"create_dn {cid}", stats, secs, {**compare(dn, cr), **extra})
        r_dn = DataNuggets(
            centers=cr.array("centers"),
            weights=cr.array("weights"),
            scales=cr.array("scales"),
            assignments=cr.array("assignments") - 1,
        )
        rc = load(cid + "_refine")
        (got, secs, extra), stats = profiled(lambda: run(lambda d: refine_dn(x, r_dn, seed=d), rc))
        good &= report(f"refine_dn {cid}", stats, secs, {**compare(got, rc), **extra})
    finally:
        _rstats._ldjit = _rcompat._ldjit = _ldjit
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "100000"))
```

Run: `$UV run ruff check benchmarks && $UV run ruff format --check benchmarks`
Expected: no errors. If the format check fails, run `$UV run ruff format benchmarks` and look at the diff before keeping it.

- [ ] **Step 3: Put R's Covertype outputs on the mini**

`mini-run.sh` keeps ignored files between runs, so this is a one-time copy. Check first: they were copied with the Covertype cache on 2026-09-29, and if `ssh mini "ls ~/datanuggets/tmp/covertype/out"` already lists the four folders below, skip to Step 4. Otherwise:

```bash
tar -C tmp/covertype/out -cf - covertype_10000_create covertype_10000_refine covertype_100000_create covertype_100000_refine \
  | ssh mini "mkdir -p ~/datanuggets/tmp/covertype/out && tar -C ~/datanuggets/tmp/covertype/out -xf -"
```

Expected: no output. `ssh mini "ls ~/datanuggets/tmp/covertype/out"` lists the four folders.

- [ ] **Step 4: Profile at 10,000 rows**

Run: `$MR uv run python benchmarks/profile_stop.py 10000`
Expected: two reports, each saying `outputs match R`, each with a `compiled_*` row (`compiled_line_means` under create_dn, `compiled_group_moments_serial` under refine_dn, which calls `cov_rows` inside the kernel, so `cov_rows` gets no row of its own), and a PASS or FAIL line. At this size the script's job is to prove it works. The rule is judged at 100,000. If parity fails on the mini, rerun on the laptop (`$UV run python benchmarks/profile_stop.py 10000`) before reading anything into it: the mini's LAPACK can move an eigenvalue by rounding, and that's a different question from this task's.

- [ ] **Step 5: Profile at 100,000 rows**

Run: `$MR uv run python benchmarks/profile_stop.py 100000 | tee "$SCRATCH/speed/profile_100000.txt"`
Expected: both outputs match R. Write the two reports into the session notes as they are, with the machine name.

- [ ] **Step 6: Commit and push**

Message file (`profile.txt`). Paste the two `stopping rule:` lines and each run's top three Python rows from Step 5 as the last paragraph:

```text
Add a profiler for spec 9.5's stopping rule

The rule is that no Python function in src/datanuggets has more than
10% of create_dn's or refine_dn's wall time on Covertype. cProfile
can't see into numba, so a kernel's time lands on its Python caller
and would read as a Python hot spot. profile_stop.py swaps _ldjit for
wrappers named compiled_<kernel> during the run, fails if none of
them books time, and checks every output against R's.

<the Step 5 lines>
```

```bash
git add benchmarks/covertype.py benchmarks/profile_stop.py
git commit -F "$SCRATCH/speed/profile.txt"
git push
```

- [ ] **Step 7: Gate**

The whole-branch review of Tasks 1 to 6 runs here, before anything builds on the kernels: an Opus code-reviewer on `git diff f792b51..HEAD` with this plan, spec section 9 and the Review Focus list. Fix its Critical and Important findings test-first, then rerun `$MR` and Step 5.

Then:

- If both runs at 100,000 print `stopping rule: PASS`, the rule holds. Go to Task 7.
- If either prints FAIL, stop building. Spec 9.5 says to compile or restructure the largest remaining Python cost next, and the report names it. Write a follow-up plan, `docs/plans/2026-09-30-speed-next-targets.md`, for those functions only. The dry run's profile (plan header, Evidence) expects three. In `create_dn`, the first is the Rfast distance replay in `_rfast.py`: `cross_distances` builds and adds one m x k temporary per column through `term` and `_arma_accumulate`. The second is `centers.select_centers`, spec 9.4's submatrix copy and row minimum. In `refine_dn`, the third is Hartigan-Wong's loops, spec 9.5's expected second target. `r_sum` and `r_vec_mean` (spec 9.2) stayed under 4% of wall. Where Step 5's report differs from this, the report wins. Hand that plan to RON2K for review before writing any of its code. Task 7 runs once a later profile passes.

### Task 7: Covertype benchmarks and the README (after the rule holds)

Runs only on the laptop: its R has the x87 long double the Covertype outputs came from, and spec 9.5 publishes one machine's R and Python side by side.

**Files:**
- Modify: `benchmarks/measure.py` (summed per-process peak)
- Modify: `benchmarks/covertype.R` (header comment, `cores` mode)
- Modify: `benchmarks/covertype.py` (docstring, `cores` mode, dispatch)
- Modify: `README.md` (Install note, line 131, the Covertype section)

**Interfaces:**
- Consumes: `covertype.warm_up` (Task 6), `covertype.load`, the R manifests' `secs`.
- Produces: `Rscript benchmarks/covertype.R <size> cores [noshape]`, which prints R's timings on its default `no.cores` and records nothing. `python benchmarks/covertype.py <size> cores`, which prints JSON with each step's all-core and single-core Python time and `same_result`. `measure.py`'s `MEASURE` line gains `peak_wset_sum` and `procs`.

- [ ] **Step 1: Check the laptop has room**

Run: `pwsh -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)"`
Expected: at least 5120 (MB free). If it's lower, run `bash ~/.claude/tools/notify.sh "datanuggets speed: the Covertype benchmarks need 5 GB free RAM, the laptop has <n> MB. Free some and say go."` and stop until RON2K answers. Repeat this check before each 100,000-row and full-size run below.

- [ ] **Step 2: Sum the peaks across the process tree**

In `benchmarks/measure.py`, replace the docstring with the first block below, and the four lines from `peak = max(...)` through the `print(...)` call's closing parenthesis with the second:

```python
"""Runs a command and reports its wall time and peak working set across its child processes
(Rscript.exe hands the work to a child Rterm). peak_wset is the largest single process's peak,
and peak_wset_sum adds every process's own peak, which bounds the whole tree's when work spreads
over worker processes. Windows only, since it reads peak_wset.

    uv run --with psutil python benchmarks/measure.py <cmd> [args]
"""
```

```python
peak = max(peaks.values(), default=0)
total = sum(peaks.values())
print(
    f"MEASURE exit={proc.returncode} wall={secs:.1f}s peak_wset={peak / 2**20:.0f} MiB "
    f"peak_wset_sum={total / 2**20:.0f} MiB procs={len(peaks)}",
    flush=True,
)
```

Run: `$UV run --with psutil python benchmarks/measure.py '.venv\Scripts\python.exe' -c "import numpy"` (backslashes: Windows won't start a program from a relative path written with forward slashes, and measure.py hands the path straight to it)
Expected: one `MEASURE exit=0 ... peak_wset_sum=... procs=1` line.

- [ ] **Step 3: Give covertype.R a timing-only all-core mode**

In `benchmarks/covertype.R`, change the usage line of the header comment to:

```r
#   Rscript benchmarks/covertype.R 10000 | 100000 | all [noshape] [cores]
# With cores, it times create.DN and refine.DN on R's default no.cores (detectCores() - 1
# PSOCK workers) and records nothing: the workers' draws can't be traced.
```

and insert directly after the line `id <- paste0("covertype_", size)`:

```r
if ("cores" %in% commandArgs(trailingOnly = TRUE)) {
  timed <- function(expr) {
    invisible(gc())
    t0 <- proc.time()[["elapsed"]]
    value <- suppressMessages(suppressWarnings(expr))
    list(value = value, secs = proc.time()[["elapsed"]] - t0)
  }
  workers <- parallel::detectCores() - 1
  cr <- timed(create.DN(X, make.pbs = FALSE))
  cat(sprintf("%s create.DN  %8.1f s  no.cores %d\n", id, cr$secs, workers))
  for (shape in shapes) {
    rr <- timed(refine.DN(X, cr$value, shape.split = shape, make.pbs = FALSE))
    cat(sprintf("%s refine.DN  %8.1f s  shape=%s  no.cores %d\n", id, rr$secs, shape, workers))
  }
  quit(save = "no")
}
```

Run: `"$RS" benchmarks/covertype.R 10000 cores`
Expected: three lines, create.DN then refine.DN with shape FALSE and TRUE, each ending `no.cores 19`. Nothing under `tmp/covertype/out/` changes (`ls -l tmp/covertype/out` shows the same times as before).

- [ ] **Step 4: Give covertype.py an all-core mode**

In `benchmarks/covertype.py`, change the usage block of the docstring to:

```python
"""Replays R's recorded draws on the Covertype benchmark and compares every output bit for bit.
With cores, times Python on every core instead and checks it against itself on one core.

Run from the repo root after benchmarks/covertype.R with the same size:
    uv run python benchmarks/covertype.py 10000 | 100000 | all [cores]
"""
```

Append above `if __name__ == "__main__":`:

```python
SEED = 20260929


def cores(size: str) -> None:
    """create_dn and refine_dn with n_jobs=-1 on a numpy seed, timed, then again with n_jobs=1
    to check the result doesn't depend on it. R's all-core run draws in PSOCK workers that
    nothing can trace, so this compares Python with itself, not with R."""
    cid = f"covertype_{size}"
    x = load(cid + "_create").array("x")
    warm_up(x)
    shapes = [
        s
        for suffix, s in (("_refine", False), ("_refine_shape", True))
        if (OUT / (cid + suffix)).is_dir()
    ]
    runs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        for n_jobs in (-1, 1):
            t0 = time.perf_counter()
            dn = create_dn(x, seed=SEED, n_jobs=n_jobs)
            runs["create", n_jobs] = (time.perf_counter() - t0, dn)
            for shape in shapes:
                t0 = time.perf_counter()
                got = refine_dn(x, dn, shape_split=shape, seed=SEED, n_jobs=n_jobs)
                step = "refine_shape" if shape else "refine"
                runs[step, n_jobs] = (time.perf_counter() - t0, got)
    report = {"n": int(x.shape[0]), "p": int(x.shape[1])}
    for step in dict.fromkeys(step for step, _ in runs):
        (all_s, all_dn), (one_s, one_dn) = runs[step, -1], runs[step, 1]
        report[step] = {
            "python_all_s": round(all_s, 2),
            "python_1_s": round(one_s, 2),
            "same_result": all_dn == one_dn,
        }
    print(json.dumps(report, indent=2))
```

and make the dispatch:

```python
if __name__ == "__main__":
    (cores if "cores" in sys.argv[2:] else main)(sys.argv[1])
```

Run: `$UV run ruff check benchmarks && $UV run ruff format --check benchmarks`
Expected: no errors.

Run: `$UV run python benchmarks/covertype.py 10000 cores`
Expected: JSON with `create`, `refine` and `refine_shape`, each with `"same_result": true`.

- [ ] **Step 5: Run the benchmark matrix**

Each command runs alone, with nothing else heavy open. Everything goes to one log (`LOG=tmp/covertype/runs-$(date +%F).txt`, `M="$UV run --with psutil python benchmarks/measure.py"`, `PY='.venv\Scripts\python.exe'`, with backslashes as in Step 2):

```bash
$M $PY benchmarks/covertype.py 10000 2>&1 | tee -a $LOG
$M $PY benchmarks/covertype.py 10000 cores 2>&1 | tee -a $LOG
$M "$RS" benchmarks/covertype.R 10000 cores 2>&1 | tee -a $LOG
# RAM check (Step 1), then:
$M $PY benchmarks/covertype.py 100000 2>&1 | tee -a $LOG
$M $PY benchmarks/covertype.py 100000 cores 2>&1 | tee -a $LOG
$M "$RS" benchmarks/covertype.R 100000 cores 2>&1 | tee -a $LOG
# RAM check (Step 1), then:
$M $PY benchmarks/covertype.py all 2>&1 | tee -a $LOG
$M $PY benchmarks/covertype.py all cores 2>&1 | tee -a $LOG
$M "$RS" benchmarks/covertype.R all noshape cores 2>&1 | tee -a $LOG
```

Expected: every `MEASURE` line has `exit=0`. Every replay report has `assignments`, `weights`, `centers`, `scales`, `messages` and `warnings` all true. Every cores report has `same_result: true`. A false anywhere stops the task. It's a parity bug to debug (superpowers:systematic-debugging) before any number gets published. Copy the log into the session notes.

- [ ] **Step 6: Measure the first-call compile**

numba keeps its cache next to the module, in `src/datanuggets/__pycache__/` (`*.nbi`, `*.nbc`, gitignored). Clear it, then time the warm-up twice:

```bash
rm -f src/datanuggets/__pycache__/*.nbi src/datanuggets/__pycache__/*.nbc
for run in cold warm; do
  $PY -c "import sys, time; sys.path.insert(0, 'benchmarks'); from covertype import load, warm_up; x = load('covertype_10000_create').array('x'); t = time.perf_counter(); warm_up(x); print('$run', round(time.perf_counter() - t, 2), 's')"
done
```

Expected: a `cold` line (compiling, several seconds) and a `warm` line (loading from the cache, well under a second). Note both in the session notes.

- [ ] **Step 7: Update the README**

All numbers come from Steps 5 and 6. Wall time is the script's own timing around the function call. Peak memory is `measure.py`'s.

After the Install requirements line, add:

```markdown
The first call compiles the long double kernels with numba, which took <cold> seconds on my laptop. numba caches them on disk, so later runs load them in <warm> seconds.
```

In the parity paragraph, replace `R's \`colMeans\` and \`rowMeans\` add in 80-bit x87 precision before rounding to double, and the port replays that with Python integers.` with:

```markdown
R's `colMeans` and `rowMeans` add in 80-bit x87 precision before rounding to double. The port repeats that arithmetic exactly in 64-bit integer code that numba compiles.
```

In the Covertype section, add after the paragraph that ends `the shape-split run replayed 383,475 draws.`:

```markdown
The all-core runs use each package's default. That's `no.cores = detectCores() - 1` in R, 19 worker processes here, and `n_jobs=-1` in Python. R's workers draw where nothing can trace them, so those runs can't be checked against R. Python's all-core output was checked against its own single-core output on the same seed instead, and matched at every size.
```

Replace the table and its lead-in with:

```markdown
Wall time in seconds, around the function call only, one run each:

| Rows | Step | R, 1 core | Python, 1 core | R, all cores | Python, all cores |
| --- | --- | --- | --- | --- | --- |
| 10,000 | create | … | … | … | … |
| 10,000 | refine | … | … | … | … |
| 10,000 | refine, shape splits | … | … | … | … |
| 100,000 | create | … | … | … | … |
| 100,000 | refine | … | … | … | … |
| 100,000 | refine, shape splits | … | … | … | … |
| 581,012 | create | … | … | … | … |
| 581,012 | refine | … | … | … | … |
```

Each cell gets the matching number from the Step 5 log: `r_s` and `python_s` from the replay reports for the one-core columns, the R `cores` lines and `python_all_s` from the cores reports for the all-core ones. Round to one decimal, and write whole seconds above 1,000 as the current table does (1801).

Replace the paragraph starting `R's shape-split run took 30 minutes` with:

```markdown
R's shape-split run took 30 minutes on one core at 100,000 rows, so the full-size run skips shape splits.
```

Replace the memory paragraph's first sentence with the new single-core peaks (`peak_wset` from the replay runs' and R's recorded runs' `MEASURE` lines, as before) and add:

```markdown
On all cores the figure is the sum of every process's own peak: <R at the three sizes> MiB for R and <Python at the three sizes> MiB for Python.
```

In the machine line, change `Python was 3.14.3 with numpy 2.5.3 and scipy 1.18.1.` to `Python was 3.14.3 with numpy 2.5.3, scipy 1.18.1 and numba 0.67.0.`

R's single-core peaks come from the runs that recorded the manifests and are already in the README. Keep them unless a rerun replaces them.

Run: `git diff -U0 README.md | grep -nP '^\+.*(\x{2014}|\x{2013}|;)'`
Expected: no output. The added lines keep the humanizer rules: no em or en dashes, no semicolons.

- [ ] **Step 8: Commit, review and open the PR**

Three commits, each with a message file:

```text
Report the summed peak memory of a process tree

peak_wset showed only the largest process, which undercounts a run
spread over worker processes. measure.py now adds every process's own
peak as well, so the all-core runs have a memory figure too.
```

```text
Add all-core runs to the Covertype benchmark

R's default no.cores spreads create.DN and refine.DN over PSOCK
workers whose draws can't be traced, so the R side only times them.
The Python side runs n_jobs=-1 on a numpy seed and checks the result
against n_jobs=1, since it can't be checked against R.
```

```text
Publish the compiled long double timings in the README

<one line per size with the old and new Python refine time, from the
Step 5 log>
```

```bash
git add benchmarks/measure.py && git commit -F "$SCRATCH/speed/measure.txt"
git add benchmarks/covertype.R benchmarks/covertype.py && git commit -F "$SCRATCH/speed/cores.txt"
git add README.md && git commit -F "$SCRATCH/speed/readme.txt"
git push
```

Review: an Opus code-reviewer on `git diff <Task 6 gate commit>..HEAD` with Step 5's log. Fix its Critical and Important findings, then rerun `$MR`.

Then open the PR against `main`, and don't merge it (RON2K's call):

```bash
gh pr create --base main --head feat/speed --title "Compile the long double arithmetic with numba" --body-file "$SCRATCH/speed/pr.md"
```

`pr.md` says what changed (kernels, compiled cov and means, threaded first pass, the profiler and the benchmark modes), the stopping-rule result from Task 6 and the README table. Humanizer rules, no test counts, no AI attribution.
