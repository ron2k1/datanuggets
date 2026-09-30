# Speed: Next Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile the three costs that fail spec section 9.5's stopping rule on Covertype 100k, Rfast's distance replay, create.DNcenters' deletion rounds and Hartigan-Wong, with every output still bit-identical to R. Then profile again.

**Architecture:** Three new kernel modules, each compiled with numba and reached through its module the way `_ldjit` is. `_rfastjit.py` computes Rfast's `Dist` and `dista` one output row at a time, with the centers innermost, adding each feature's term in Armadillo's order. `_centersjit.py` runs all of create.DNcenters' deletion rounds over one distance matrix. `_kmnsjit.py` is `_HW`'s kmns.f port, statement for statement. `_rfast`, `centers` and `_hartigan_wong` keep their function signatures (minus the unused `block_rows`) and call the kernels. Today's bodies stay as `_cross_distances_py`, `_select_centers_py` and `_kmns_py`, the references the tests compare against. `benchmarks/profile_stop.py` books every kernel module, not only `_ldjit`.

**Tech Stack:** Python 3.12+ (dev on 3.14.3), numpy >= 2.0, numba >= 0.67 (0.67.0 locked), pytest + pytest-xdist, uv, ruff 0.16. Tests run on the Mac mini (M4, arm64) through `mini-run.sh`.

**Spec:** `docs/specs/2026-09-29-datanuggets-python-port-design.md` section 9 (9.3 parallelism, 9.4 other planned optimizations, 9.5 when to stop). This plan is the follow-up that `docs/plans/2026-09-30-speed-compiled-long-double.md` Task 6 Step 7 calls for when the 100k profile fails, and it covers the functions that profile named, nothing else.

**Execution:** RON2K reviews this plan before any of its code is written. It runs after the Task 6 gate's fix pass on plan 1 is committed. Tasks 1 to 3 build and test. Task 4 profiles and is a decision gate. When both runs pass, plan 1 resumes at Task 7.

**Evidence behind the code in this plan** (a throwaway worktree off 58cefb1 and its own clone on the mini, not in the repo):

- The gate profile this plan answers, plan 1 Task 6 Step 5 on the mini at 100,000 rows, commit 58cefb1. `create_dn` took 15.49 s, and the largest Python costs were `_rfast.py`'s `term` (39.2% of wall, 6.07 s), `centers.select_centers` (34.5%, 5.35 s) and `_rfast._arma_accumulate` (11.6%, 1.79 s). `refine_dn` took 0.91 s, with Hartigan-Wong's `_qtran` at 17.4%, `_optra` 13.6%, `_final_centers_and_wss` 12.3%, `_initial_assignment` 12.1%, `_initial_centers` 4.5% and `_transfer` 3.5%. cProfile books numpy work done through operators (`-`, `*`, fancy indexing) to the Python function that runs it, so `term` is really numpy streaming an m x k temporary per feature, and `select_centers` is mostly `d[np.ix_(keep, keep)]` copying the matrix every round.
- numba 0.67.0 doesn't fuse `a * a + c` into one FMA rounding without fastmath, on the laptop (AMD64) or the mini (arm64), in a scalar function, a loop or a prange loop (probe with a = 1 + 2^-30, c = -1). So a compiled `s += d * d` rounds the product, then the sum, as numpy and R's C do.
- The prototype is this plan's code. On the mini the full suite passed, 1225 tests. `tests/test_rfast.py` went from 45 collected to 131, `tests/test_centers.py` from 141 to 175, `tests/test_hartigan_wong.py` from 31 to 40, and `tests/test_create.py` stayed at 96.
- Its profile on the mini at 100,000 rows, run twice with the same times: `create_dn` 2.23 s wall and `refine_dn` 0.32 s, both matching R on every output, and both passing the stopping rule. The largest Python function was 0.2% of wall in `create_dn` (`create.py` `stage1`) and 2.9% in `refine_dn` (`_rstats.r_eigen_sym`). Kernel time in `create_dn`: `compiled_dist` 0.95 s, `compiled_nearest` 0.77 s, `compiled_delete_rounds` 0.40 s, `compiled_line_means` 0.01 s. In `refine_dn`: `compiled_group_moments_serial` 0.17 s, `compiled_hartigan_wong` 0.01 s.
- The deletion rounds' row minimum, on one 5,000-row Covertype chunk on the mini: a plain scan took 22.7 ms for the first round and 41.0 ms for all rounds down to 500 rows. The four-lane scan in Task 2 took 7.0 ms and 16.3 ms. numpy's `argmin(axis=1)` over the same 5,000 x 5,000 matrix took 12.6 ms, and computing the matrix took 39.5 ms. With the plain scan, the whole 100k profile had `create_dn` at 2.85 s.
- Memory: under tracemalloc, R's copy-every-round version peaked at 2.59 times one 1,500 x 1,500 matrix. The compiled rounds stay under 1.1 times.
- The kmns fuzz in Task 3 ran 1,800 random problems against `_HW` with 0 mismatches. Each of its six cases reaches all five fault codes. For example, 1-D grid data gave 83 converged runs, 174 empty clusters, 9 out of iterations, 10 bad k and 24 out of quick-transfer steps.
- First-call compile of the new kernels on the laptop (Python 3.12.13, cold cache): `cross` 0.84 s, `nearest` 0.24 s, `dist` with `delete_rounds` 2.27 s, `hartigan_wong` 2.54 s. Loading them from the cache, imports included, took 0.88 s.

**Where this plan goes past the spec's text** (RON2K rules on these with the plan):

- Spec 9.4 says the greedy marking loop "gets compiled only if the profile flags it". The profile flagged `select_centers` as a whole, and cProfile can't split the marking loop from the copy inside the same function. So Task 2 compiles each deletion round whole, marking included.
- Spec 9.4 says the remaining rows are tracked "with a mask". The kernel keeps the survivors' row numbers in their original order, which is the same information in a form a loop can scan. It also searches a row again only when its nearest neighbour was deleted. That's exact: the old minimum is still there, and nothing earlier can tie it that wasn't there before. The fuzz compares it with R's copy-every-round version on tie-heavy data.
- Spec 9.4 says the row minimum over the active set is "vectorized". A first-index argmin doesn't vectorize without reordering compares, so the kernel keeps four independent running minima and merges them, which gives the same index faster.
- Spec 9.3 says stage 1 chunks run "under joblib (loky)". The code has used `prefer="threads"` since create_dn landed, and the kernels here release the GIL (`nogil=True`), so with `n_jobs > 1` stage 1's thread workers now run their distance and deletion kernels at the same time. Each kernel stays serial, as 9.3 requires. The spec's "(loky)" should say "(threads)".
- `DIST_BLOCK_ROWS` and `ASSIGN_BLOCK_ROWS` go away with the `block_rows` arguments. They bounded numpy temporaries, and the kernels make none.

Out of scope: numba threads inside `nearest_index` or `dist_matrix` (spec 9.3 doesn't ask for them), and `r_sum`/`r_vec_mean` (under 2% of wall each in the prototype's profile).

## Global Constraints

- Branch `feat/speed`, tracking `origin/feat/speed`. `main` advances only by PR, and merging is RON2K's call.
- Parity: every existing fixture test passes unchanged. Compiled results equal the kept Python references bit for bit. Compare floats with `same_bits` (`.view(np.uint64)`), so -0.0 and 0.0 count as different.
- Every kernel is `@compiled(nogil=True)` from `datanuggets._compiled`, with no `fastmath` and no `parallel`. `compiled` is numba's `njit(cache=True)`, falling back to plain `njit` where numba has nowhere to write its cache. `tests/test_compiled.py` finds every `*jit` module by itself and checks both, so a kernel written with `njit` directly fails there. Each loop runs its floating-point operations in the same order as the Python code it replaces. Loops may be reordered only where no sum or compare changes order, like looping over centers innermost.
- Kernels see only finite float64 data, which `as_matrix` guarantees for every public entry point. The Python wrappers make every array C-contiguous float64 before a kernel reads it, because numba indexes raw memory without bounds checks. For the same reason a wrapper whose two inputs must have the same number of columns checks that and raises `ValueError`. Every array a kernel only reads goes through `_compiled.frozen`, a read-only view, because numba compiles a separate kernel for read-only arrays and one kernel should serve every caller.
- `_cross_distances_py`, `arma_sum_last`, `_select_centers_py` and `_kmns_py` (with `_HW`) stay in the package as test references and don't run in normal use.
- Callers reach kernels through the module, `from datanuggets import _rfastjit` then `_rfastjit.name(...)`, never `from datanuggets._rfastjit import name`. Kernel names are unique across the `*jit` modules, because the profiler books each one as `compiled_<name>`.
- Real data only: every timing written into docs or commits comes from a run made in this plan or the plan-time evidence above. No invented numbers.
- Fixtures are recorded only on the laptop's x87 R. This plan records none.
- Shell: Git Bash on Windows 11 from the repo root. `UV=/c/Users/thegr/AppData/Local/hermes/bin/uv.exe`, `MR="bash $HOME/.claude/tools/mini-run.sh"` and `SCRATCH=/c/Users/thegr/AppData/Local/Temp/claude/C--Users-thegr/25ee381d-b280-4d5f-83df-1350491cb8e9/scratchpad` (the session scratchpad, for commit message files and profile output, never the repo). `$MR` alone runs the full suite on the mini on 8 workers. `$MR uv run pytest -q -p no:cacheprovider <path>` runs part of it.
- Commits: one logical change each, imperative subject <= 70 chars plus a why/what body. Write the message to a file in the scratchpad and commit with `git commit -F`. No `Claude-Session:`, `Co-Authored-By` or "Generated with" lines, and no mention of AI tools. Stage files by name, never `git add -A`. `git checkout --` any `docs/validation/*.md` that shows modified with no content change. Push `feat/speed` after each task's green commits.
- Repo root holds only toolchain files. Code in `src/datanuggets/`, tests in `tests/`, benchmarks in `benchmarks/`.
- Session trio: update the tasklist, notes and log (`$UV run python $SCRATCH/f3/bk.py <tasklist|notes|log> append ...`, timestamp from `date`) on every task status change.
- Prose in docstrings and commit messages follows the humanizer rules: no em or en dashes, no semicolons, contractions, no bold or italic in prose, no comma-pause chains.

## Review Focus

1. Integer-valued data, where many distances tie exactly. Every Covertype column is an integer. A person expects R's picks among ties: `Rfast::rowMins`' first index, `order()`'s stable order and std::sort's head for `dista`. Tests `test_nearest_index_matches_the_numpy_replay` (Task 1), and `test_row_min_is_numpys_first_argmin_over_the_active_rows` and `test_select_centers_matches_the_python_rounds` (Task 2).
2. Rows far enough apart that a squared difference overflows to inf, or close enough that it's subnormal. A person expects R's bits, inf where R has inf. Tests `test_cross_distances_match_the_numpy_replay` and `test_dist_matrix_matches_the_numpy_replay` (Task 1).
3. Arrays that aren't C-contiguous float64 reaching a kernel (Fortran order, strided slices, int columns, read-only arrays), or two inputs with different column counts. A kernel reads raw memory, so a wrong layout or width would silently read the wrong numbers, and a read-only array would compile a second kernel. A person expects the answer for the contiguous float64 copy, one compile, and a `ValueError` for mismatched widths. Tests `test_distances_take_any_layout_and_dtype`, `test_distances_reject_mismatched_columns` and `test_read_only_inputs_run_the_same_compiled_kernels` (Task 1), and `test_kmns_takes_any_layout_and_dtype`, `test_kmns_rejects_mismatched_columns` and `test_kmns_read_only_data_runs_the_same_compiled_kernels` (Task 3).
4. `create_dn` with `n_jobs` above 1. Stage 1's thread workers now run kernels outside the GIL at the same time. A person expects the `n_jobs=1` result. Test `test_n_jobs_does_not_change_result`, extended to `n_jobs=4` (Task 2).
5. Stage 2 at the default `dn_num1 = 10,000`, an 800 MB distance matrix. A person on a laptop expects one matrix in memory, not a second copy each round. Test `test_select_centers_holds_one_distance_matrix` (Task 2).

---

### Task 1: compiled Rfast distances

**Files:**
- Create: `src/datanuggets/_rfastjit.py`
- Modify: `src/datanuggets/_rfast.py:11` (module docstring), `:20` (import), `:56-107` (`cross_distances` through `nearest_index`)
- Modify: `src/datanuggets/_defaults.py:15-17` (drop `DIST_BLOCK_ROWS`, `ASSIGN_BLOCK_ROWS`)
- Test: `tests/test_rfast.py`

**Interfaces:**
- Consumes: `_rfast._arma_accumulate`, `_rfast._sequential_accumulate` and `_rfast.gnu_sort_first`, all unchanged. `rfixtures.same_bits(a, b) -> bool`. `_compiled.compiled(**options)` and `_compiled.frozen(a)` (a read-only view), already on the branch.
- Produces: in `datanuggets._rfastjit`, `@compiled(nogil=True)` kernels `cross(a, bt, manhattan, sequential, out)`, `dist(x, xt, manhattan, out)` and `nearest(points, ct, manhattan, sequential, best)`, plus the helper `_sums`. `a`, `x` and `points` are (rows x p), `bt`, `xt` and `ct` are the other side transposed (p x columns), and `out`/`best` are written in place. In `datanuggets._rfast`: `cross_distances(a, b, metric, *, kernel="dist")` (unchanged signature), `_cross_distances_py(a, b, metric, *, kernel="dist")` (the reference), `_dista_sums_in_order(p, metric, kernel) -> bool`, `dist_matrix(x, metric)` and `nearest_index(points, centers, metric)`. The last two lose `block_rows`. `cross_distances` and `nearest_index` raise `ValueError` when the two sides' column counts differ. Task 2 calls `dist_matrix(rs, metric)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_rfast.py`, replace the lines from `import numpy as np` through `    arma_sum_last,` with:

```python
import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from datanuggets import _rfastjit
from datanuggets._rfast import (
    _cross_distances_py,
    arma_sum_last,
```

Replace `from rfixtures import load_group` with:

```python
from rfixtures import load_group, same_bits
```

In `test_dist_matrix_is_bitwise_equal_to_rfast`, drop the `block_rows=7` argument:

```python
    got = dist_matrix(case.array("x"), case.args["metric"])
```

Delete `test_nearest_index_block_size_invariant` whole. Blocks are gone, and the replay fuzz below covers every row count.

Append:

```python
def _fuzz_rows(rng, kind, m, p):
    """Rows built to stress the kernels: ties, huge spreads that overflow to inf, subnormals."""
    if kind == "grid":
        return rng.integers(0, 3, size=(m, p)).astype(float)
    if kind == "huge":
        return rng.standard_normal((m, p)) * 1e154
    if kind == "tiny":
        return rng.standard_normal((m, p)) * 1e-310
    return rng.standard_normal((m, p)) * 10.0 ** rng.integers(-3, 4, size=(m, 1))


KINDS = ["grid", "huge", "tiny", "mixed"]
WIDTHS = [1, 2, 3, 7, 32, 33, 54]


def _replay(a, b, metric, kernel="dist"):
    with np.errstate(over="ignore"):  # "huge" rows overflow to inf, in R too
        return _cross_distances_py(a, b, metric, kernel=kernel)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("p", WIDTHS)
def test_cross_distances_match_the_numpy_replay(p, kind):
    rng = np.random.default_rng(p * 10 + KINDS.index(kind))
    a, b = _fuzz_rows(rng, kind, 23, p), _fuzz_rows(rng, kind, 31, p)
    for metric in ("euclidean", "manhattan"):
        for kernel in ("dist", "dista"):
            want = _replay(a, b, metric, kernel)
            got = cross_distances(a, b, metric, kernel=kernel)
            assert same_bits(got, want), (metric, kernel)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("p", WIDTHS)
def test_dist_matrix_matches_the_numpy_replay(p, kind):
    x = _fuzz_rows(np.random.default_rng(100 + p * 10 + KINDS.index(kind)), kind, 41, p)
    for metric in ("euclidean", "manhattan"):
        assert same_bits(dist_matrix(x, metric), _replay(x, x, metric)), metric


def _nearest_index_py(points, centers, metric):
    d = _replay(points, centers, metric, "dista")
    best = d.argmin(axis=1)
    mins = d[np.arange(d.shape[0]), best]
    for r in np.flatnonzero(np.count_nonzero(d == mins[:, None], axis=1) > 1):
        best[r] = gnu_sort_first(d[r])
    return best


@pytest.mark.parametrize("k", [1, 2, 5, 16, 17, 40, 200])
@pytest.mark.parametrize("kind", KINDS)
def test_nearest_index_matches_the_numpy_replay(kind, k):
    rng = np.random.default_rng(1000 + k * 10 + KINDS.index(kind))
    for p in (2, 33):
        points, centers = _fuzz_rows(rng, kind, 150, p), _fuzz_rows(rng, kind, k, p)
        centers[k // 2 :] = centers[: k - k // 2]  # duplicated centers: exact ties
        for metric in ("euclidean", "manhattan"):
            want = _nearest_index_py(points, centers, metric)
            np.testing.assert_array_equal(nearest_index(points, centers, metric), want)


def test_distances_take_any_layout_and_dtype():
    rng = np.random.default_rng(7)
    a = rng.integers(-50, 50, size=(30, 12))
    b = np.asfortranarray(rng.standard_normal((40, 24)))[::2, ::2]
    a64, b64 = a.astype(float), np.ascontiguousarray(b)
    for metric in ("euclidean", "manhattan"):
        assert same_bits(cross_distances(a, b, metric), _replay(a64, b64, metric))
        assert same_bits(dist_matrix(b, metric), dist_matrix(b64, metric))
        np.testing.assert_array_equal(nearest_index(a, b, metric), nearest_index(a64, b64, metric))


def test_distances_reject_mismatched_columns():
    for a, b in ((np.ones((3, 2)), np.ones((4, 3))), (np.ones((3, 3)), np.ones((4, 2)))):
        with pytest.raises(ValueError, match="columns"):
            cross_distances(a, b, "euclidean")
        with pytest.raises(ValueError, match="columns"):
            nearest_index(a, b, "euclidean")


def _all_distances(a, b):
    out = []
    for metric in ("euclidean", "manhattan"):
        out += [cross_distances(a, b, metric, kernel=k) for k in ("dist", "dista")]
        out += [dist_matrix(b, metric), nearest_index(a, b, metric)]
    return out


def test_read_only_inputs_run_the_same_compiled_kernels():
    rng = np.random.default_rng(11)
    a, b = rng.standard_normal((30, 40)), rng.standard_normal((20, 40))  # 40 > 32: dista in order
    want = _all_distances(a, b)
    kernels = [k for k in vars(_rfastjit).values() if isinstance(k, CPUDispatcher)]
    compiled = [len(k.signatures) for k in kernels]
    a, b = a.copy(), b.copy()
    a.setflags(write=False)
    b.setflags(write=False)
    got = _all_distances(a, b)
    assert all(same_bits(g, w) for g, w in zip(got, want, strict=True))
    assert [len(k.signatures) for k in kernels] == compiled  # no second signature to compile
```

- [ ] **Step 2: Run them to see them fail**

Run: `$MR uv run pytest -q -p no:cacheprovider tests/test_rfast.py`
Expected: `1 error`, collecting `tests/test_rfast.py`: `ImportError: cannot import name '_rfastjit' from 'datanuggets'`.

- [ ] **Step 3: Write the kernels**

Create `src/datanuggets/_rfastjit.py`:

```python
"""Rfast's distance loops, compiled: Dist, dista, and dista's nearest center.

Every distance adds its per-feature terms in Rfast's order, in double, with no fused multiply-add:
Armadillo's two accumulators over even and odd features, added at the end, or left to right for
dista's Euclidean kernel over more than 32 features. So the bits match _rfast's numpy replay. Each
kernel fills one output row at a time, looping over centers innermost, which keeps a row's
accumulators in cache and lets LLVM vectorize across centers without reordering any sum. The
inputs are finite, which as_matrix guarantees upstream.
"""

import math

import numpy as np

from datanuggets._compiled import compiled


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
```

- [ ] **Step 4: Call them from `_rfast.py`**

At the end of the module docstring, after `lowest index.`, add a paragraph:

```text
The distance loops run compiled, in _rfastjit. _cross_distances_py does the same arithmetic in
numpy, one feature at a time, and is the reference the kernels are tested against.
```

Replace `from datanuggets._defaults import ASSIGN_BLOCK_ROWS, DIST_BLOCK_ROWS` with:

```python
from datanuggets import _rfastjit
from datanuggets._compiled import frozen
```

Replace everything from `def cross_distances(` up to (not including) `def gnu_sort_first(` with the block below, keeping two blank lines before `def gnu_sort_first(`. It adds `_dista_sums_in_order` above `cross_distances`, and the old `cross_distances` body moves into `_cross_distances_py`.

```python
def _dista_sums_in_order(p: int, metric: str, kernel: str) -> bool:
    """dista's Euclidean kernel over more than 32 features adds left to right (BLAS ddot)."""
    return kernel == "dista" and metric == "euclidean" and p > _ARMA_DOT_MAX


def cross_distances(
    a: np.ndarray, b: np.ndarray, metric: str, *, kernel: str = "dist"
) -> np.ndarray:
    """Distances between the rows of a (m x p) and b (k x p), rounded as Rfast::Dist
    (kernel="dist") or Rfast::dista (kernel="dista") rounds them."""
    a = frozen(np.ascontiguousarray(a, dtype=np.float64))
    bt = frozen(np.ascontiguousarray(np.asarray(b, dtype=np.float64).T))
    if bt.shape[0] != a.shape[1]:
        raise ValueError(f"a has {a.shape[1]} columns but b has {bt.shape[0]}")
    out = np.empty((a.shape[0], bt.shape[1]))
    seq = _dista_sums_in_order(a.shape[1], metric, kernel)
    _rfastjit.cross(a, bt, metric == "manhattan", seq, out)
    return out


def _cross_distances_py(
    a: np.ndarray, b: np.ndarray, metric: str, *, kernel: str = "dist"
) -> np.ndarray:
    """cross_distances in numpy, one feature at a time. The reference the kernels are tested
    against."""
    if metric == "euclidean":

        def term(j: int) -> np.ndarray:
            d = b[None, :, j] - a[:, None, j]
            return d * d

    else:

        def term(j: int) -> np.ndarray:
            return np.abs(b[None, :, j] - a[:, None, j])

    p = a.shape[1]
    if _dista_sums_in_order(p, metric, kernel):
        total = _sequential_accumulate(term, p)
    else:
        total = _arma_accumulate(term, p)
    if metric == "euclidean":
        np.sqrt(total, out=total)
    return total


def dist_matrix(x: np.ndarray, metric: str) -> np.ndarray:
    """Rfast::Dist(x, method = metric): the full n x n matrix."""
    x = frozen(np.ascontiguousarray(x, dtype=np.float64))
    out = np.empty((x.shape[0], x.shape[0]))
    _rfastjit.dist(x, frozen(np.ascontiguousarray(x.T)), metric == "manhattan", out)
    return out


def nearest_index(points: np.ndarray, centers: np.ndarray, metric: str) -> np.ndarray:
    """Rfast::dista(points, centers, type = metric, k = 1, index = TRUE), 0-based."""
    points = frozen(np.ascontiguousarray(points, dtype=np.float64))
    centers = frozen(np.ascontiguousarray(centers, dtype=np.float64))
    if centers.shape[1] != points.shape[1]:
        raise ValueError(
            f"points have {points.shape[1]} columns but centers have {centers.shape[1]}"
        )
    seq = _dista_sums_in_order(points.shape[1], metric, "dista")
    best = np.empty(points.shape[0], dtype=np.intp)
    ct = frozen(np.ascontiguousarray(centers.T))
    _rfastjit.nearest(points, ct, metric == "manhattan", seq, best)
    tied = np.flatnonzero(best < 0)
    if tied.size:
        d = cross_distances(points[tied], centers, metric, kernel="dista")
        for r, i in enumerate(tied.tolist()):
            best[i] = gnu_sort_first(d[r])
    return best
```

In `src/datanuggets/_defaults.py`, delete these three lines and the blank line after them:

```python
# Rows per block when computing distances, so no n x k or n x n temporary exceeds memory.
DIST_BLOCK_ROWS = 1024
ASSIGN_BLOCK_ROWS = 1024  # 1024 x 2000 centers = 16 MB per temporary
```

Run: `grep -rn "BLOCK_ROWS\|block_rows" src tests benchmarks`
Expected: no output.

- [ ] **Step 5: Run the tests**

Run: `$UV run ruff check src tests && $UV run ruff format --check src tests`
Expected: no errors. If the format check fails, run `$UV run ruff format src tests` and read the diff before keeping it.

Run: `$MR uv run pytest -q -p no:cacheprovider tests/test_rfast.py`
Expected: `131 passed`. That's the 45 from before, less the deleted block test, plus 28 cross-distance, 28 dist-matrix and 28 nearest-index fuzz cases, the layout test, the column-count test and the read-only test. The first run compiles the kernels, which takes a few seconds.

Run: `$MR`
Expected: `1182 passed` (1096 before this plan, plus 86).

- [ ] **Step 6: Commit and push**

Message file `$SCRATCH/speed/next-1.txt`:

```text
Compile Rfast's distance loops

cross_distances built an m x k temporary per feature and added them
in numpy. On Covertype 100k that was half of create_dn's wall time,
booked to term and _arma_accumulate. The loops now run in numba, one
output row at a time with the centers innermost. Each feature's term
is added in Armadillo's even/odd order, or left to right for dista's
Euclidean kernel over 32 features, and nothing is fused, so the bits
don't move. Dist computes half the matrix and mirrors it. That's exact
because b - a is -(a - b) in IEEE arithmetic.

nearest_index finds each row's minimum and flags ties in the same
pass, so only tied rows go through std::sort's replay. The numpy path
stays as _cross_distances_py, the reference a fuzz compares against.
The block_rows arguments go: there are no temporaries left to bound.

The wrappers hand the kernels read-only views, so a caller's read-only
array runs the kernel already compiled instead of compiling a second
one. And a b whose column count differs from a's is a ValueError now,
where the kernel would have read past the end of a row.
```

```bash
git add src/datanuggets/_rfastjit.py src/datanuggets/_rfast.py src/datanuggets/_defaults.py tests/test_rfast.py
git commit -F "$SCRATCH/speed/next-1.txt"
git push
```

---

### Task 2: create.DNcenters' deletion rounds in one compiled loop

**Files:**
- Create: `src/datanuggets/_centersjit.py`
- Modify: `src/datanuggets/centers.py:10` (import), `:21-32` (`select_centers` head, the old body kept as `_select_centers_py`)
- Test: `tests/test_centers.py`, `tests/test_create.py:78-83`

**Interfaces:**
- Consumes: `_rfast.dist_matrix(x, metric)` from Task 1. `_compiled.frozen(a)`.
- Produces: in `datanuggets._centersjit`, `@compiled(nogil=True)` kernels `delete_rounds(d, dn_num, delete_percent) -> int64 array` (the kept row numbers, ascending) and `_nearest(d, act, t, a) -> (position, distance)`. In `datanuggets.centers`: `select_centers(rs, dn_num, delete_percent, dist_metric) -> (kept, passthrough)` (unchanged signature) and `_select_centers_py` with the same signature, the reference.

- [ ] **Step 1: Write the failing tests**

In `tests/test_centers.py`, replace `import warnings` with:

```python
import tracemalloc
import warnings
```

and replace `from datanuggets import DataNuggetWarning, create_dn_centers` with:

```python
from datanuggets import DataNuggetWarning, _centersjit, create_dn_centers
from datanuggets.centers import _select_centers_py, select_centers
```

Append:

```python
@pytest.mark.parametrize("delete_percent", [0.001, 0.1, 0.5, 0.99])
@pytest.mark.parametrize("kind", ["grid", "dups", "normal"])
def test_select_centers_matches_the_python_rounds(kind, delete_percent):
    """Ties everywhere (a small grid, duplicated rows) are where a row's nearest neighbour and the
    order of equal distances decide which point goes."""
    rng = np.random.default_rng(int(delete_percent * 1000) + len(kind))
    for n in (2, 3, 17, 120, 301):
        if kind == "grid":
            rs = rng.integers(0, 3, size=(n, 2)).astype(float)
        elif kind == "dups":
            rs = rng.standard_normal((n, 3))
            rs[n // 2 :] = rs[: n - n // 2]
        else:
            rs = rng.standard_normal((n, 4))
        for dn_num in sorted({1, max(1, n // 10), max(1, n // 2), max(1, n - 1)}):
            for metric in ("euclidean", "manhattan"):
                got = select_centers(rs, dn_num, delete_percent, metric)
                want = _select_centers_py(rs, dn_num, delete_percent, metric)
                np.testing.assert_array_equal(got[0], want[0], err_msg=f"{n} {dn_num} {metric}")
                assert got[1] == want[1]


@pytest.mark.parametrize("t", [*range(1, 20), 37, 301])
def test_row_min_is_numpys_first_argmin_over_the_active_rows(t):
    """The four running minima against numpy's argmin, on rows full of ties, for every remainder
    past the last group of four and a scattered set of active rows."""
    rng = np.random.default_rng(t)
    n = t + 25
    d = rng.integers(0, 3, size=(n, n)).astype(float)
    act = np.sort(rng.choice(n, size=t, replace=False))
    for a in range(n):
        want = int(d[a, act].argmin())
        assert _centersjit._nearest(d, act, t, a) == (want, d[a, act[want]])


def test_select_centers_holds_one_distance_matrix():
    """Stage 2 builds a dn_num1 x dn_num1 matrix, 800 MB at the default 10,000. The rounds must
    work inside it, not copy it down each round as R does."""
    rs = np.random.default_rng(0).standard_normal((1500, 3))
    matrix = rs.shape[0] ** 2 * 8
    tracemalloc.start()
    try:
        select_centers(rs, 150, 0.1, "euclidean")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.1 * matrix
```

In `tests/test_create.py`, replace `test_n_jobs_does_not_change_result` with:

```python
def test_n_jobs_does_not_change_result():
    x = QUAKES.array("x")
    a = create_dn(x, **SMALL, seed=11, n_jobs=1)
    b = create_dn(x, **SMALL, seed=11, n_jobs=2)
    c = create_dn(x, **SMALL, seed=11, n_jobs=4)  # stage 1's kernels run in 4 threads at once
    for field in ("centers", "weights", "scales", "assignments"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        np.testing.assert_array_equal(getattr(a, field), getattr(c, field))
```

This one passes before the change too. It guards stage 1 once its thread workers run kernels without the GIL.

- [ ] **Step 2: Run them to see them fail**

Run: `$MR uv run pytest -q -p no:cacheprovider tests/test_centers.py`
Expected: `1 error`, collecting `tests/test_centers.py`: `ImportError: cannot import name '_centersjit' from 'datanuggets'`.

Run: `$MR uv run pytest -q -p no:cacheprovider tests/test_create.py::test_n_jobs_does_not_change_result`
Expected: `1 passed`. It runs on its own because pytest stops at the first collection error.

- [ ] **Step 3: Write the kernel**

Create `src/datanuggets/_centersjit.py`:

```python
"""create.DNcenters' deletion rounds, compiled.

R copies the distance matrix down to the surviving rows every round. Here the matrix stays put and
the survivors are a list of row numbers in their original order, so "first index on ties" means
the same point it means in R's smaller copy. A row's nearest neighbour only changes when that
neighbour is deleted: the old minimum is still there, and nothing earlier can tie it that wasn't
there before. So after the first round only those rows are searched again.
"""

import math

import numpy as np

from datanuggets._compiled import compiled


@compiled(nogil=True)
def _nearest(d, act, t, a):
    """(first position in act[:t] with the smallest d[a, .], that distance): Rfast::rowMins.

    Four running minima over positions 0, 1, 2, 3 mod 4 don't wait on each other's compares. Each
    keeps its lane's first minimal position, so merging by value, then by position, gives the
    first minimal position overall.
    """
    row = d[a]
    b = 0
    v = row[act[0]]
    rest = 1
    if t >= 8:
        b0, b1, b2, b3 = 0, 1, 2, 3
        v0, v1, v2, v3 = row[act[0]], row[act[1]], row[act[2]], row[act[3]]
        jj = 4
        while jj + 3 < t:
            w = row[act[jj]]
            if w < v0:
                v0, b0 = w, jj
            w = row[act[jj + 1]]
            if w < v1:
                v1, b1 = w, jj + 1
            w = row[act[jj + 2]]
            if w < v2:
                v2, b2 = w, jj + 2
            w = row[act[jj + 3]]
            if w < v3:
                v3, b3 = w, jj + 3
            jj += 4
        rest = jj
        b, v = b0, v0
        if v1 < v or (v1 == v and b1 < b):
            b, v = b1, v1
        if v2 < v or (v2 == v and b2 < b):
            b, v = b2, v2
        if v3 < v or (v3 == v and b3 < b):
            b, v = b3, v3
    for jj in range(rest, t):  # positions past every lane's: only a smaller value wins
        w = row[act[jj]]
        if w < v:
            v, b = w, jj
    return b, v


@compiled(nogil=True)
def delete_rounds(d, dn_num, delete_percent):
    """Row numbers create.DNcenters keeps from d (n x n, diagonal above every distance): each round
    deletes one point of each closest pair, in order of distance, as R does."""
    n = d.shape[0]
    act = np.arange(n)
    nn = np.empty(n, dtype=np.int64)  # nearest active row, by row number
    nn_dist = np.empty(n)
    gone = np.zeros(n, dtype=np.bool_)  # deleted, in this round or an earlier one
    key = np.empty(n)
    t = n
    first = True
    while t > dn_num:
        delete_num = max(1, math.floor(t * delete_percent))
        delete_num = min(delete_num, t - dn_num)
        for i in range(t):
            a = act[i]
            if first or gone[nn[a]]:
                b, v = _nearest(d, act, t, a)
                nn[a] = act[b]
                nn_dist[a] = v
            key[i] = nn_dist[a]
        first = False
        order = np.argsort(key[:t], kind="mergesort")  # stable, like R's order()
        taken = 0
        for p in order:
            if taken >= delete_num:
                break
            a = act[p]
            if not gone[nn[a]] and not gone[a]:
                gone[a] = True
                taken += 1
        if taken < delete_num:
            for p in order:
                if taken >= delete_num:
                    break
                a = act[p]
                if not gone[a]:
                    gone[a] = True
                    taken += 1
        w = 0
        for i in range(t):
            if not gone[act[i]]:
                act[w] = act[i]
                w += 1
        t = w
    return act[:t].copy()
```

- [ ] **Step 4: Call it from `centers.py`**

Replace `from datanuggets._defaults import DELETE_PERCENT, DIST_METRICS` with:

```python
from datanuggets import _centersjit
from datanuggets._compiled import frozen
from datanuggets._defaults import DELETE_PERCENT, DIST_METRICS
```

Replace the head of `select_centers`, from `def select_centers(` through the line `    tmp_num = n_obs`, with the block below. The rest of the old body (the `while tmp_num > dn_num:` loop and `return keep, False`) stays where it is and becomes `_select_centers_py`'s.

```python
def select_centers(
    rs: np.ndarray, dn_num: int, delete_percent: float, dist_metric: str
) -> tuple[np.ndarray, bool]:
    """Kept row positions (0-based) and whether the input came back unchanged. Inputs pre-validated."""
    n_obs = rs.shape[0]
    if n_obs <= dn_num:
        return np.arange(n_obs), True
    d = dist_matrix(rs, dist_metric)
    np.fill_diagonal(d, d.max() + 1)
    kept = _centersjit.delete_rounds(frozen(d), dn_num, delete_percent)
    return kept.astype(np.intp, copy=False), False


def _select_centers_py(
    rs: np.ndarray, dn_num: int, delete_percent: float, dist_metric: str
) -> tuple[np.ndarray, bool]:
    """select_centers as R writes it, copying the matrix down every round. The reference the
    compiled rounds are tested against."""
    n_obs = rs.shape[0]
    keep = np.arange(n_obs)
    if n_obs <= dn_num:
        return keep, True

    d = dist_matrix(rs, dist_metric)
    np.fill_diagonal(d, d.max() + 1)
    tmp_num = n_obs
```

- [ ] **Step 5: Run the tests**

Run: `$UV run ruff check src tests && $UV run ruff format --check src tests`
Expected: no errors.

Run: `$MR uv run pytest -q -p no:cacheprovider tests/test_centers.py tests/test_create.py::test_n_jobs_does_not_change_result`
Expected: `176 passed`: 141 from before, 12 round fuzz cases, 21 row-minimum cases, the memory test, and the `n_jobs` test.

Run: `$MR`
Expected: `1216 passed` (1182 plus 34).

- [ ] **Step 6: Commit and push**

Message file `$SCRATCH/speed/next-2.txt`:

```text
Run create.DNcenters' deletion rounds in one compiled loop

select_centers copied the distance matrix down to the survivors every
round, as R does. On Covertype 100k that was 34.5% of create_dn's wall
time, and it held a second matrix in memory. The rounds now run in
numba over the one matrix. Survivors are a list of row numbers in
their original order, so first index on ties means what it meant in
R's smaller copy.

A row's nearest neighbour only changes when that neighbour is
deleted, so after the first round only those rows are searched again.
The search keeps four running minima that don't wait on each other's
compares, merged by value, then position. R's version stays as
_select_centers_py, the reference a tie-heavy fuzz compares against.
```

```bash
git add src/datanuggets/_centersjit.py src/datanuggets/centers.py tests/test_centers.py tests/test_create.py
git commit -F "$SCRATCH/speed/next-2.txt"
git push
```

---

### Task 3: compiled Hartigan-Wong

**Files:**
- Create: `src/datanuggets/_kmnsjit.py`
- Modify: `src/datanuggets/_hartigan_wong.py:1-8` (docstring), `:17-21` (import, `BIG`), `:60-63` (`kmns`, and `_kmns_py` after it)
- Test: `tests/test_hartigan_wong.py`

**Interfaces:**
- Consumes: `_hartigan_wong._HW` and `Kmns`, unchanged. `_compiled.frozen(a)`.
- Produces: in `datanuggets._kmnsjit`, the constant `BIG` and `@compiled(nogil=True)` kernels `hartigan_wong(a, c, iter_max, max_qtran, ic1, nc, wss) -> (iter, ifault)` (overwrites `c`, fills `ic1`, `nc`, `wss`) and its stages `_initial_assignment`, `_initial_centers`, `_optra`, `_qtran`, `_transfer`, `_final_centers_and_wss`. In `datanuggets._hartigan_wong`: `kmns(a, c, iter_max, max_qtran) -> Kmns` (unchanged signature and result types: `cluster`, `size` and `withinss` are lists) and `_kmns_py`, same signature, the reference. `kmns` raises `ValueError` when `c`'s column count differs from `a`'s.

- [ ] **Step 1: Write the failing tests**

In `tests/test_hartigan_wong.py`, replace the two imports:

```python
from datanuggets._hartigan_wong import KMeansError, _kmns_py, kmeans, kmns
```

```python
from rfixtures import load_group, same_bits
```

Add `from numba.core.registry import CPUDispatcher` after `import pytest`, and `from datanuggets import _kmnsjit` above `from datanuggets._draws import ReplayDraws`.

Append:

```python
def _same_kmns(got, want):
    assert (got.ifault, got.iter) == (want.ifault, want.iter)
    assert got.cluster == want.cluster
    assert got.size == want.size
    assert same_bits(got.centers, want.centers)
    assert same_bits(got.withinss, want.withinss)


@pytest.mark.parametrize("p", [1, 2, 5])
@pytest.mark.parametrize("kind", ["grid", "normal"])
def test_kmns_kernel_matches_the_python_port(kind, p):
    """Starts that repeat a row or a value (empty clusters), tiny step budgets (ifault 2 and 4),
    and k from 2 up to m and past it (ifault 3), on tied and untied data."""
    rng = np.random.default_rng(p * 7 + len(kind))
    faults = set()
    for _ in range(300):
        m = int(rng.integers(3, 60))
        if kind == "grid":
            a = rng.integers(0, 10, size=(m, p)).astype(float)
        else:
            a = rng.standard_normal((m, p))
        k = int(rng.integers(2, min(8, m + 1) + 1))
        repeats = k > m or rng.random() < 0.2
        c = a[rng.integers(0, m, size=k) if repeats else rng.choice(m, size=k, replace=False)]
        iter_max = int(rng.choice([1, 2, 10, 1000]))
        max_qtran = int(rng.choice([1, 3, 50 * m], p=[0.15, 0.15, 0.7]))
        want = _kmns_py(a, c, iter_max, max_qtran)
        _same_kmns(kmns(a, c, iter_max, max_qtran), want)
        faults.add(want.ifault)
    assert faults == {0, 1, 2, 3, 4}


def test_kmns_takes_any_layout_and_dtype():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 9, size=(40, 6))
    c = np.asfortranarray(a[:4].astype(float))
    want = _kmns_py(a[:, ::2].astype(float), np.ascontiguousarray(c[:, ::2]), 50, 2000)
    _same_kmns(kmns(a[:, ::2], c[:, ::2], 50, 2000), want)


def test_kmns_rejects_mismatched_columns():
    for a, c in ((np.ones((5, 3)), np.ones((2, 4))), (np.ones((5, 4)), np.ones((2, 3)))):
        with pytest.raises(ValueError, match="columns"):
            kmns(a, c, 10, 100)


def test_kmns_read_only_data_runs_the_same_compiled_kernels():
    case = max(RUNS, key=lambda c: c.array("x").size)
    want = _kmns(case)
    kernels = [k for k in vars(_kmnsjit).values() if isinstance(k, CPUDispatcher)]
    compiled = [len(k.signatures) for k in kernels]
    x, c0 = case.array("x"), case.array("c0")
    x.setflags(write=False)
    c0.setflags(write=False)
    _same_kmns(_kmns(case, x, c0), want)
    assert [len(k.signatures) for k in kernels] == compiled  # no second signature to compile
```

- [ ] **Step 2: Run them to see them fail**

Run: `$MR uv run pytest -q -p no:cacheprovider tests/test_hartigan_wong.py`
Expected: `1 error`, collecting `tests/test_hartigan_wong.py`: `ImportError: cannot import name '_kmnsjit' from 'datanuggets'`.

- [ ] **Step 3: Write the kernel**

Create `src/datanuggets/_kmnsjit.py`:

```python
"""Hartigan-Wong k-means (R's kmns.f), compiled.

The same operations in the same order as _hartigan_wong._HW, which ports kmns.f line by line, in
double with no fused multiply-add, so the two agree bit for bit. Cluster and row indices are
0-based. LIVE and NCP hold the Fortran's 1-based row numbers, as in _HW.
"""

import numpy as np

from datanuggets._compiled import compiled

BIG = float(np.float32(1e30))  # kmns.f: DATA BIG /1.E30/, a REAL constant widened to double


@compiled(nogil=True)
def hartigan_wong(a, c, iter_max, max_qtran, ic1, nc, wss):
    """kmns.f on the rows of a (m x p) from the centers in c (k x p), which it overwrites. Writes
    IC1 into ic1, NC into nc and WSS into wss, all zero on entry. Returns (ITER, IFAULT)."""
    m = a.shape[0]
    k = c.shape[0]
    if k <= 1 or k >= m:
        return iter_max, 3
    ic2 = np.zeros(m, dtype=np.int64)
    d = np.zeros(m)
    an1 = np.zeros(k)
    an2 = np.zeros(k)
    ncp = np.zeros(k, dtype=np.int64)
    itran = np.zeros(k, dtype=np.int64)
    live = np.zeros(k, dtype=np.int64)
    _initial_assignment(a, c, ic1, ic2)
    if not _initial_centers(a, c, ic1, nc, an1, an2, itran, ncp):
        return iter_max, 1
    indx = 0
    ifault, ij = 2, iter_max + 1  # a Fortran DO variable after its loop ran out
    for it in range(1, iter_max + 1):
        indx = _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx)
        if indx == m:  # a full pass without a transfer: converged
            ifault, ij = 0, it
            break
        indx, max_qtran = _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, max_qtran)
        if max_qtran < 0:
            ifault, ij = 4, it
            break
        if k == 2:  # with two clusters the optimal-transfer stage has nothing left to do
            ifault, ij = 0, it
            break
        ncp[:] = 0
    _final_centers_and_wss(a, c, ic1, nc, wss)
    return ij, ifault


@compiled(nogil=True)
def _initial_assignment(a, c, ic1, ic2):
    """IC1 and IC2: each point's closest and second closest starting center."""
    m, n = a.shape
    k = c.shape[0]
    for i in range(m):
        dt0 = 0.0
        for j in range(n):
            da = a[i, j] - c[0, j]
            dt0 += da * da
        dt1 = 0.0
        for j in range(n):
            da = a[i, j] - c[1, j]
            dt1 += da * da
        c1, c2 = 0, 1
        if dt0 > dt1:
            c1, c2 = 1, 0
            dt0, dt1 = dt1, dt0
        for g in range(2, k):
            db = 0.0
            done = True
            for j in range(n):
                dc = a[i, j] - c[g, j]
                db += dc * dc
                if db >= dt1:
                    done = False
                    break
            if done:
                if db >= dt0:
                    dt1 = db
                    c2 = g
                else:
                    dt1 = dt0
                    c2 = c1
                    dt0 = db
                    c1 = g
        ic1[i] = c1
        ic2[i] = c2


@compiled(nogil=True)
def _initial_centers(a, c, ic1, nc, an1, an2, itran, ncp):
    """Centers become the means of the initial clusters. False on an empty cluster."""
    m, n = a.shape
    k = c.shape[0]
    c[:, :] = 0.0
    for i in range(m):
        g = ic1[i]
        nc[g] += 1
        for j in range(n):
            c[g, j] += a[i, j]
    for g in range(k):
        if nc[g] == 0:
            return False
        aa = float(nc[g])
        for j in range(n):
            c[g, j] /= aa
        an2[g] = aa / (aa + 1.0)
        an1[g] = aa / (aa - 1.0) if aa > 1.0 else BIG
        itran[g] = 1
        ncp[g] = -1
    return True


@compiled(nogil=True)
def _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx):
    """Optimal-transfer stage: each point may move to the cluster that most reduces WSS."""
    m, n = a.shape
    k = c.shape[0]
    for g in range(k):
        if itran[g] == 1:
            live[g] = m + 1
    for i0 in range(m):
        i = i0 + 1
        indx += 1
        l1 = ic1[i0]
        l2 = ic2[i0]
        ll = l2
        if nc[l1] != 1:
            if ncp[l1] != 0:
                de = 0.0
                for j in range(n):
                    df = a[i0, j] - c[l1, j]
                    de += df * df
                d[i0] = de * an1[l1]
            da = 0.0
            for j in range(n):
                db = a[i0, j] - c[l2, j]
                da += db * db
            r2 = da * an2[l2]
            for g in range(k):
                if (i >= live[l1] and i >= live[g]) or g == l1 or g == ll:
                    continue
                rr = r2 / an2[g]
                dc = 0.0
                done = True
                for j in range(n):
                    dd = a[i0, j] - c[g, j]
                    dc += dd * dd
                    if dc >= rr:
                        done = False
                        break
                if done:
                    r2 = dc * an2[g]
                    l2 = g
            if r2 >= d[i0]:
                ic2[i0] = l2
            else:
                indx = 0
                live[l1] = m + i
                live[l2] = m + i
                ncp[l1] = i
                ncp[l2] = i
                _transfer(a, i0, c, nc, an1, an2, l1, l2)
                ic1[i0] = l2
                ic2[i0] = l1
        if indx == m:
            return indx
    for g in range(k):
        itran[g] = 0
        live[g] -= m
    return indx


@compiled(nogil=True)
def _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, max_qtran):
    """Quick-transfer stage: swap points between their two closest clusters until a full pass
    moves nothing. Returns max_qtran = -1 when the step budget runs out."""
    m, n = a.shape
    icoun = 0
    istep = 0
    while True:
        for i0 in range(m):
            icoun += 1
            istep += 1
            if istep >= max_qtran:
                return indx, -1
            l1 = ic1[i0]
            l2 = ic2[i0]
            if nc[l1] != 1:
                if istep <= ncp[l1]:
                    da = 0.0
                    for j in range(n):
                        db = a[i0, j] - c[l1, j]
                        da += db * db
                    d[i0] = da * an1[l1]
                if istep < ncp[l1] or istep < ncp[l2]:
                    r2 = d[i0] / an2[l2]
                    dd = 0.0
                    done = True
                    for j in range(n):
                        de = a[i0, j] - c[l2, j]
                        dd += de * de
                        if dd >= r2:
                            done = False
                            break
                    if done:
                        icoun = 0
                        indx = 0
                        itran[l1] = 1
                        itran[l2] = 1
                        ncp[l1] = istep + m
                        ncp[l2] = istep + m
                        _transfer(a, i0, c, nc, an1, an2, l1, l2)
                        ic1[i0] = l2
                        ic2[i0] = l1
            if icoun == m:
                return indx, max_qtran


@compiled(nogil=True)
def _transfer(a, i0, c, nc, an1, an2, l1, l2):
    """Move point i0 from cluster l1 to l2: both centers and their AN1/AN2 factors."""
    al1 = float(nc[l1])
    alw = al1 - 1.0
    al2 = float(nc[l2])
    alt = al2 + 1.0
    for j in range(a.shape[1]):
        c[l1, j] = (c[l1, j] * al1 - a[i0, j]) / alw
        c[l2, j] = (c[l2, j] * al2 + a[i0, j]) / alt
    nc[l1] -= 1
    nc[l2] += 1
    an2[l1] = alw / al1
    an1[l1] = alw / (alw - 1.0) if alw > 1.0 else BIG
    an1[l2] = alt / al2
    an2[l2] = alt / (alt + 1.0)


@compiled(nogil=True)
def _final_centers_and_wss(a, c, ic1, nc, wss):
    """Recompute the centers from the final clusters, then each cluster's WSS."""
    m, n = a.shape
    k = c.shape[0]
    c[:, :] = 0.0
    for i in range(m):
        g = ic1[i]
        for j in range(n):
            c[g, j] += a[i, j]
    for j in range(n):
        for g in range(k):
            c[g, j] /= float(nc[g])
        for i in range(m):
            ii = ic1[i]
            da = a[i, j] - c[ii, j]
            wss[ii] += da * da
```

- [ ] **Step 4: Call it from `_hartigan_wong.py`**

Replace the module docstring's second paragraph, from `kmns() is a line-by-line port` through `restarts, and R's errors and warnings.`, with:

```text
_HW is a line-by-line port of R's kmns.f (Hartigan and Wong 1979, algorithm AS 136, with R's
changes). Every floating-point operation happens in the Fortran's order, in double, so from the
same starting centers the clusters, centers and within-cluster sums of squares match R bit for
bit. kmns() runs the same operations compiled, in _kmnsjit, and _HW is the reference it's
tested against. kmeans() adds what stats::kmeans does around it: the random starts, the nstart
restarts, and R's errors and warnings.
```

Add `from datanuggets import _kmnsjit` and `from datanuggets._compiled import frozen` as the first two lines of the `datanuggets` imports, and replace the `BIG = ...` line with:

```python
BIG = _kmnsjit.BIG  # kmns.f: DATA BIG /1.E30/, a REAL constant widened to double
```

Replace `kmns` (its body is two lines after the docstring) with `kmns` and `_kmns_py`:

```python
def kmns(a: np.ndarray, c: np.ndarray, iter_max: int, max_qtran: int) -> Kmns:
    """R's C_kmns: Hartigan-Wong on the rows of a (m x p) from the starting centers c (k x p)."""
    a = frozen(np.ascontiguousarray(a, dtype=np.float64))
    c = np.array(c, dtype=np.float64, order="C")  # a copy: the kernel overwrites the centers
    if c.shape[1] != a.shape[1]:
        raise ValueError(f"a has {a.shape[1]} columns but c has {c.shape[1]}")
    ic1 = np.zeros(a.shape[0], dtype=np.int64)
    nc = np.zeros(c.shape[0], dtype=np.int64)
    wss = np.zeros(c.shape[0])
    it, ifault = _kmnsjit.hartigan_wong(a, c, iter_max, max_qtran, ic1, nc, wss)
    return Kmns(ic1.tolist(), c, nc.tolist(), wss.tolist(), it, ifault)


def _kmns_py(a: np.ndarray, c: np.ndarray, iter_max: int, max_qtran: int) -> Kmns:
    """kmns in pure Python, the reference the kernel is tested against."""
    hw = _HW(np.asarray(a, dtype=np.float64), np.asarray(c, dtype=np.float64))
    return hw.run(iter_max, max_qtran)
```

- [ ] **Step 5: Run the tests**

Run: `$UV run ruff check src tests && $UV run ruff format --check src tests`
Expected: no errors.

Run: `$MR uv run pytest -q -p no:cacheprovider tests/test_hartigan_wong.py`
Expected: `40 passed`: 31 from before, 6 fuzz cases, the layout test, the column-count test and the read-only test. Every R fixture in the file now runs the kernel.

Run: `$MR`
Expected: `1225 passed` (1216 plus 9).

- [ ] **Step 6: Commit and push**

Message file `$SCRATCH/speed/next-3.txt`:

```text
Compile Hartigan-Wong

refine_dn's 2-means splits spent 63% of its wall time on Covertype
100k in _HW's Python loops. kmns now runs the same kmns.f port in
numba, statement for statement, in double with no fused multiply-add.
So clusters, centers and within-cluster sums still match R bit for
bit. _HW stays as the reference, and a fuzz that reaches every fault
code compares the two. kmns hands the kernel a read-only view of the
data, and a c whose column count differs from a's is a ValueError.
```

```bash
git add src/datanuggets/_kmnsjit.py src/datanuggets/_hartigan_wong.py tests/test_hartigan_wong.py
git commit -F "$SCRATCH/speed/next-3.txt"
git push
```

---

### Task 4: profile against the stopping rule again (gate)

**Files:**
- Modify: `benchmarks/covertype.py:19` (import), `:60-66` (`warm_up`)
- Modify: `benchmarks/profile_stop.py` (book every kernel module)

**Interfaces:**
- Consumes: `_rfast.cross_distances`, `_rfast.nearest_index` (Task 1), `centers.select_centers` (Task 2), `_hartigan_wong.kmns` (Task 3), and the module references `_rfast._rfastjit`, `centers._centersjit`, `_hartigan_wong._kmnsjit`, `_rstats._ldjit` and `_rcompat._ldjit`.
- Produces: `covertype.warm_up(x)` now loads every kernel, which plan 1's Task 7 relies on. `profile_stop.kernel_refs() -> list[(module, name, kernel module)]`.

This task has no unit test. It's a measuring script, and its checks run inside it: R parity on every output, and a fail-closed check that kernel time was booked.

- [ ] **Step 1: Warm up the new kernels**

In `benchmarks/covertype.py`, replace the `from datanuggets import ...` line with:

```python
from datanuggets import (
    DataNuggets,
    DataNuggetWarning,
    _hartigan_wong,
    _rcompat,
    _rfast,
    _rstats,
    centers,
    create_dn,
    refine_dn,
)
```

and add at the end of `warm_up`:

```python
    _rfast.cross_distances(head, head[:5], "euclidean")
    _rfast.nearest_index(head, head[:5], "euclidean")
    centers.select_centers(head, 10, 0.1, "euclidean")
    _hartigan_wong.kmns(head[:, :1], head[:3, :1], 10, 2500)
```

- [ ] **Step 2: Book every kernel module**

Replace `benchmarks/profile_stop.py` with:

```python
"""Checks spec section 9.5's stopping rule on Covertype: no function written in Python in
src/datanuggets may have a cProfile tottime above 10% of the run's wall time, for create_dn and
for refine_dn (shape splits off), both with n_jobs=1 on R's replayed draws.

cProfile can't see into numba, so a kernel's time would land in the Python function that called
it. Callers reach kernels through their module (_ldjit, _rfastjit, _centersjit, _kmnsjit). For
the run, each of those references is swapped for a copy whose kernels are wrapped in Python
functions named compiled_<kernel>, which book the time under that name instead. Run from the repo root after benchmarks/covertype.R
with the same size:

    uv run python benchmarks/profile_stop.py [10000 | 100000]

Exits 1 if the rule fails, no kernel time was booked, or an output stops matching R's.
"""

import cProfile
import os
import pstats
import sys
import types
from pathlib import Path

from numba.core.dispatcher import Dispatcher

import datanuggets
from covertype import compare, load, run, stage_lines, warm_up
from datanuggets import DataNuggets, create_dn, refine_dn

SRC = os.path.normcase(str(Path(datanuggets.__file__).resolve().parent))
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
    """A kernel module's attributes, with every kernel swapped for its booked wrapper."""

    def __init__(self, module):
        for name, value in vars(module).items():
            if not name.startswith("__"):
                setattr(self, name, booked(name, value) if isinstance(value, Dispatcher) else value)


def kernel_refs() -> list[tuple[types.ModuleType, str, types.ModuleType]]:
    """(module, name, kernel module) for every reference a datanuggets module holds to a module
    of kernels, a datanuggets module whose name ends in jit."""
    refs = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("datanuggets."):
            for name, value in list(vars(mod).items()):
                if (
                    isinstance(value, types.ModuleType)
                    and value.__name__.startswith("datanuggets.")
                    and value.__name__.endswith("jit")
                ):
                    refs.append((mod, name, value))
    return refs


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
        print("  no kernel time was booked: a caller holds a kernel by name, not its module")
    passed = worst <= LIMIT
    print(f"  stopping rule: {'PASS' if passed else 'FAIL'} (largest {worst:.1%})")
    return parity and passed and bool(compiled)


def main(size: str) -> int:
    cid = f"covertype_{size}"
    cr = load(cid + "_create")
    x = cr.array("x")
    warm_up(x)
    refs = kernel_refs()
    for mod, name, jit in refs:
        setattr(mod, name, Booked(jit))
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
        for mod, name, jit in refs:
            setattr(mod, name, jit)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "100000"))
```

Run: `$UV run ruff check benchmarks && $UV run ruff format --check benchmarks`
Expected: no errors.

- [ ] **Step 3: Profile at 10,000 rows**

Run: `$MR uv run python benchmarks/profile_stop.py 10000`
Expected: two reports, each saying `outputs match R`. `create_dn` lists `compiled_dist`, `compiled_nearest`, `compiled_delete_rounds` and `compiled_line_means` (and `compiled_cross` if any row tied). `refine_dn` lists `compiled_group_moments_serial` and `compiled_hartigan_wong`. At this size the script's job is to prove the booking works. The rule is judged at 100,000.

- [ ] **Step 4: Profile at 100,000 rows**

Run: `$MR uv run python benchmarks/profile_stop.py 100000 | tee "$SCRATCH/speed/profile_100000_next.txt"`
Expected: both outputs match R, and both runs print `stopping rule: PASS`. The prototype gave `create_dn` 2.23 s and `refine_dn` 0.32 s, with the largest Python function at 0.2% and 2.9% of wall. Write both reports into the session notes as they are, with the machine name.

- [ ] **Step 5: Commit and push**

Message file `$SCRATCH/speed/next-4.txt`. Paste the two `stopping rule:` lines and each run's wall time from Step 4 as the last paragraph:

```text
Book every kernel module in the stopping-rule profiler

profile_stop.py swapped only _ldjit for booked wrappers, so time in
the distance, deletion-round and k-means kernels would have landed on
their Python callers and read as Python hot spots. It now swaps every
reference a datanuggets module holds to a *jit module. warm_up loads
the new kernels too, so no timing includes a compile.

<the Step 4 lines>
```

```bash
git add benchmarks/covertype.py benchmarks/profile_stop.py
git commit -F "$SCRATCH/speed/next-4.txt"
git push
```

- [ ] **Step 6: Gate**

The whole-branch review of this plan's commits runs here: an Opus code-reviewer on `git diff <this plan's BASE>..HEAD` with this plan, spec section 9 and the Review Focus list. Fix its Critical and Important findings test-first, then rerun `$MR` and Step 4.

Then:

- If both runs at 100,000 print `stopping rule: PASS`, the rule holds. Resume `docs/plans/2026-09-30-speed-compiled-long-double.md` at Task 7. Two things there change because of this plan. Its Step 6 compile timing now covers every kernel, since it calls `warm_up`. And its README sentence says "the long double kernels", which should say "its kernels". Ledger both as rulings in plan 1's ledger when Task 7 runs.
- If either prints FAIL, stop building. The report names the largest remaining Python cost. Write the next follow-up plan for that function only, and hand it to RON2K for review before writing any of its code.
