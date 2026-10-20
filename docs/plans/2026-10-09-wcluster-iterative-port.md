# WCluster Iterative Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port WCluster 1.3.0's `Wkmeans`, `Whclust`, `DN.Wkmeans`, `DN.Whclust`, `DN.Wpca` and `DNcluster.predict` as standalone Python and numba, bit-identical to R 4.6.1 and faster than R by the spec's gates.

**Architecture:** Each R function gets a plain-Python reference that transliterates R step for step on the existing exact primitives (`_wss`, `wwcss`, `wmean`, `_wdist_mat`, `r_sum`), and a compiled kernel that computes each value R computes once, on the same rows in the same order, so it has the same bits. Shared compiled pieces (`rsum`, `wss_rows`, `wmean_rows`, `assign`) live in `wcluster/wstatsjit.py`, the Whclust kernel in `wcluster/whclustjit.py` and the Wkmeans refinement in `wcluster/wkmeansjit.py`. The public functions run the kernels and fall back to the reference only for Whclust's self-merge. Wkmeans' starts are drawn in Python through the `Draws` protocol, which gains `sample_replace`.

**Tech Stack:** Python 3.12+ (dev on 3.14 and 3.15), numpy >= 2.0, numba 0.68.0 with llvmlite 0.50.0 (locked), scipy (`to_linkage` checks), pytest + pytest-xdist, uv, ruff. R 4.6.1 on the laptop records fixtures only.

**Spec:** `docs/specs/2026-10-09-wcluster-iterative-port-design.md` (approved 2026-10-09). The plan argues from it. Section numbers below are the spec's.

**Execution:** RON2K approved the spec and asked for the plan and an end-to-end build in one go ("approved, write the plan then code it e2e"), so this plan runs natively right after it's committed, task by task with TDD, then one Opus review of the whole branch.

**Evidence behind the code in this plan** (R probes run 2026-10-09 on the laptop's R 4.6.1, scripts in the session scratchpad):

- `apply(z, 1, which.min)` on rows that are all NaN gives `integer(0)` when every row is, and a list with `integer(0)` entries when some are. `length(unique(list(1L, integer(0), 1L, 2L)))` is 3. So `init`'s distinct count is the distinct labels plus one when some but not all rows are dead.
- With a list of labels, `max()` stops with "invalid 'type' (list) of argument" and `wwcss` with "unimplemented type 'list' in 'orderVector1'". With `integer(0)`, `wwcss` stops with "The length of cluster assignments should be the same as the number of observations. ".
- With one row, `wdist_mat`'s `diag(w)` is a 1 by 1 identity for 1 <= w < 2, "non-conformable arguments" for 0 <= w < 1 or w >= 2, and "invalid 'nrow' value (< 0)" for w < 0.
- `if (3 != 1 & tb[3] > 1)` past the end of the table, and `if (NaN > 1)`, both stop with "missing value where TRUE/FALSE needed".
- `sample.int`'s formals are `n, size, replace, prob, useHash`, so the exit trace can read `replace`.
- After a self-merge R's matrix indexing: `y[y == NA] <- ii` is a no-op, a negative subscript past the end is ignored, a row past the end in an assignment stops with "subscript out of bounds", `distwij(NA, ...)` stops with "missing value where TRUE/FALSE needed", `distwij(integer(0), ...)` with "argument is of length zero", and `forder` on a merge row holding NA with "missing value where TRUE/FALSE needed".
- `Whclust(matrix(c(0, 0.1, 5, 5.1)), c(1, 1, -1, -1))` returns `merge` rows (-3, -4), (-1, -2), (1, 1), `order` c(3, 4, 3, 4): a self-merge at the last step, returned without an error.
- R's `%*%` under matprod "default" switches to `simple_matprod` when an input holds NaN or Inf, a double running sum in the same order as reference BLAS. So `r_matvec` and `r_matmul`'s order holds in both cases, and the kernels sum the same way.
- Timings (spec section 10): Whclust n = 50, 100, 200, 400 took 5.0, 20.9, 79.7 and 327.4 s, Wkmeans k = 4 at n = 100, 250, 500, 1000 took 1.7, 5.9, 3.9 and 8.4 s.

**Where this plan goes past the spec's text** (RON2K rules on these with the final report):

- `init` can loop forever in more ways than the spec's two hang guards: when every drawn start gives a non-finite center (all weights zero, say), every row's distances are NaN and R redraws forever. The port keeps both guards and adds a cap: after 10,000 starts without k labels, `wkmeans` raises `ValueError` saying R would keep drawing. A case where R would succeed on start 10,001 would differ, and none is known.
- R's `DN.Wkmeans` stops on a one-variable datanugget ("argument is of length zero"), because `[, 2:2]` drops the table to a vector. So do `DN.Wpca` without `wcol` ("invalid 'times' argument") and `DNcluster.predict` with a one-variable datanugget as `newx`. The port raises in the same places, per section 7's rule.
- `DNcluster.predict` with a datanugget `newx` takes `newx`'s table columns 2 to `ncol(datanugget) - 2`, sized by the first datanugget, so a second datanugget with fewer variables passes its Weight and Scale columns as data. The port does the same and stops where R stops ("undefined columns selected").
- `whclust`'s kernel stores its distances in an n by n array, 800 MB at n = 10,000, which is R's own footprint and the spec's design. It isn't chunked.

Out of scope: `optimal.DN` (next spec), the closed-form Ward update, R's `hclust` metadata fields, and threads inside `whclust`.

## Global Constraints

- Branch `feat/wcluster-iterative`, tracking `origin/feat/wcluster-iterative`. `main` advances only by PR, and merging is RON2K's call.
- Parity: labels, `merge` and `order` equal R's, every float bit-identical, compared with `assert_same_values` from `tests/rfixtures.py` (same float64 bits, -0 isn't 0, any NaN matches any NaN). `dn_wpca` keeps `wpca`'s eigen tolerance.
- Kernels only in modules named `*jit`, each `@compiled(nogil=True, error_model="numpy")` from `datanuggets._compat._compiled`, no `fastmath`, no `parallel`. `error_model="numpy"` because a cluster whose weights sum to zero divides by zero, and R's answer is Inf or NaN, not an exception. Python callers reach a kernel through its module (`whclustjit.whclust(...)`). Every array a kernel only reads goes through `frozen`. Kernel names are unique across the `*jit` modules. `tests/compat/test_compiled.py` finds the new modules by itself.
- Floating point: every sum in a kernel runs in the order of the Python primitive it replays. No FMA (numba doesn't contract without fastmath, as the speed-next plan measured).
- Inputs: `as_matrix` and `as_weights` reject NaN and Inf up front. Kernels get C-contiguous float64 copies.
- References `_whclust_py` and `_wkmeans_py` stay in the package. They run in the tests, and `_whclust_py` also runs for the self-merge.
- Fixtures are recorded once, on the laptop's x87 R 4.6.1 with reference BLAS, and never regenerated to make a test pass. R runs at dev time only.
- Seeds: an int or `Generator` gives `NumpyDraws`, and `RDraws(seed)` gives R's draws. Docstrings say to pass `RDraws(seed)` to match R.
- Shell: Git Bash from the repo root, quoted (the path has spaces), no `cd` out of the repo. `UV=/c/Users/thegr/AppData/Local/hermes/bin/uv.exe`, `RS="$LOCALAPPDATA/Programs/R/R-4.6.1/bin/Rscript.exe"`, `SCRATCH=/c/Users/thegr/AppData/Local/Temp/claude/C--Users-thegr/51b44fcf-5f32-450e-bfa0-039ae9e9024d/scratchpad`, `NUMBA_CACHE_DIR=$SCRATCH/numba` for every Python run (wiped when `wstatsjit.py` changes, since numba's cache doesn't track a callee in another module). R runs on script files only, bare, from the repo root.
- Commits: one logical change each, imperative subject <= 70 chars plus a why/what body, written to `$SCRATCH/msg.txt` and committed with `git commit -F`. No `Co-Authored-By`, `Claude-Session:` or "Generated with" lines and no claude.ai URL. Stage files by name. Push after each task's green commits.
- Prose in docstrings, docs and commit messages: no em or en dashes, no semicolons, contractions, no bold or italic in prose.
- Repo root holds only toolchain files. Benchmark output goes to the gitignored `tmp/wcluster/`.
- Session trio (`OneDrive/Desktop/{TASKLIST,NOTES,LOGS} ALL CC SESSIONS/2026-10-09_datanuggets-wcluster-port/`) updated on every task status change, timestamps from `date`.

## Review Focus

1. Tie-heavy integer data, where many distances and many WWCSS values are exactly equal. A person expects R's first minimum: the first column in position order, then the first row, for `whclust`, and the first center for `wkmeans`, and R's `<=` acceptance for moves. Tests: the `grid_dup` and `iris_unit` fixtures (Task 6), and `test_kernel_matches_reference_on_tie_heavy_data` for both functions (Tasks 6 and 8).
2. Zero and negative weights, which make NaN distances, Inf picks, self-merges, emptied clusters and R's NaN stops. A person expects R's output or R's error at R's step. Tests: the `zero_w*`, `selfmerge*`, `recreate`, `tb_na` and `nan_stop` fixtures, and the fuzz's quirk-coverage check (Tasks 6 and 8).
3. Inputs that aren't C-contiguous float64: Fortran order, strided slices, int columns, read-only arrays, data frames, and `w` as a list. A person expects the answer for the float64 copy. Tests: `test_whclust_takes_any_layout_and_dtype` (Task 6) and `test_wkmeans_takes_any_layout_and_dtype` (Task 8).
4. Seeds and threads: an int seed gives a reproducible `NumpyDraws` run, `RDraws(seed)` gives R's labels, and `n_jobs` never changes the result. Tests: `test_int_seed_is_reproducible`, `test_rdraws_seed_matches_r` and `test_n_jobs_does_not_change_the_result` (Task 8).
5. Values big enough to overflow a squared distance or a sum to Inf. A person expects R's Inf and R's picks. Tests: the `huge` fuzz family, data scaled by 1e154, in both kernel-against-reference fuzzes (Tasks 6 and 8), and `test_rsum_matches_r_sum` on overflowing input (Task 4).

---

## File map

| File | Task | Responsibility |
|---|---|---|
| `src/datanuggets/_random/{interfaces,numpy_draws,r_draws,replay}.py` | 1 | `sample_replace`, `DrawRecord.replace` |
| `r/fixture_io.R`, `r/fixtures_rsample.R`, `tests/rfixtures.py` | 1 | record and read `replace` |
| `r/wcluster_counted.R` | 2 | counted copies of R's functions, for quirk proofs |
| `r/fixtures_whclust.R`, `r/fixtures_wkmeans.R`, `r/fixtures_dn_wcluster.R` | 2 | the three fixture groups |
| `src/datanuggets/result.py` | 3 | `WkmeansResult`, `DnWkmeansResult`, `WhclustResult` |
| `src/datanuggets/wcluster/wstatsjit.py` | 4 | `rsum`, `wss_rows`, `wmean_rows`, `assign` |
| `src/datanuggets/wcluster/predict.py` | 4 | `_wdist_mat` takes `w` |
| `src/datanuggets/wcluster/whclust.py` | 5, 6 | `whclust`, `_whclust_py`, merge renumbering, `forder` |
| `src/datanuggets/wcluster/whclustjit.py` | 6 | the Whclust kernel |
| `src/datanuggets/wcluster/wkmeans.py` | 7, 8 | `wkmeans`, argument checks, `init`, `_wkmeans_py` |
| `src/datanuggets/wcluster/wkmeansjit.py` | 8 | the refinement kernel |
| `src/datanuggets/wcluster/dn.py` | 9 | the four DN wrappers |
| `src/datanuggets/{__init__,wcluster/__init__}.py`, `docs/architecture.md` | 10 | exports and the module map |
| `benchmarks/wcluster.R`, `benchmarks/wcluster.py` | 11 | speed gates |
| `tests/wcluster/test_{whclust,wkmeans,dn,wstatsjit}.py`, `tests/randomness/*` | each | tests |

---

### Task 1: Replace draws

**Files:**
- Modify: `src/datanuggets/_random/interfaces.py`, `numpy_draws.py`, `r_draws.py`, `replay.py`
- Modify: `tests/rfixtures.py` (`Case.draws`)
- Modify: `r/fixture_io.R` (`capture_draws`, `draw_arrays`, fail-closed check), `r/fixtures_rsample.R` (replace cases)
- Test: `tests/randomness/test_draws.py`, `tests/randomness/test_rdraws.py`

**Interfaces:**
- Produces: `Draws.sample_replace(n: int, size: int) -> np.ndarray` (intp, 0-based), `DrawRecord(n, size, value, replace=False)`, fixture array `draw_replace` (int 0/1 per draw), rsample cases `replace_*` with args `n`, `size`, `seed` and arrays `state0`, `first`, `state1`, `then`, `state2`.

- [ ] **Step 1: Failing tests.** In `tests/randomness/test_draws.py`:

```python
def test_replay_sample_replace_returns_zero_based_values_and_checks_the_kind():
    draws = ReplayDraws([DrawRecord(3, 4, np.array([1, 3, 3, 2]), replace=True)])
    assert draws.sample_replace(3, 4).tolist() == [0, 2, 2, 1]
    draws.assert_exhausted()
    with pytest.raises(ReplayMismatch, match=r"replace = TRUE"):
        ReplayDraws([DrawRecord(3, 4, np.array([1, 3, 3, 2]), replace=True)]).sample(3, 4)
    with pytest.raises(ReplayMismatch, match=r"replace = TRUE"):
        ReplayDraws([DrawRecord(3, 3, np.array([2, 3, 1]))]).sample_replace(3, 3)


def test_numpy_sample_replace_draws_values_in_range():
    v = NumpyDraws(7).sample_replace(3, 1000)
    assert v.dtype == np.intp and set(v.tolist()) == {0, 1, 2}
    assert NumpyDraws(7).sample_replace(3, 1000).tolist() == v.tolist()


def test_fixtures_without_draw_replace_read_as_plain_draws():
    case = load_group("kmeans")[0]
    assert not case.has("draw_replace")
    assert case.draws() and not any(r.replace for r in case.draws())
```

and add `sample_replace` to the methods `test_both_implementations_satisfy_the_protocol` checks. In `tests/randomness/test_rdraws.py`, add `REPLACE = [c for c in GROUP.values() if c.id.startswith("replace_")]`, extend `test_every_case_is_recorded` with the six ids from Step 5, and:

```python
@cases(REPLACE)
def test_sample_replace_matches_r(case):
    draws = RDraws(case.args["seed"])
    got = draws.sample_replace(case.args["n"], case.args["size"]) + 1
    assert got.tolist() == ints(case, "first")
    assert draws.random_seed().tolist() == ints(case, "state1")
    assert (draws.sample(10, 10) + 1).tolist() == ints(case, "then")
    assert draws.random_seed().tolist() == ints(case, "state2")


def test_sample_replace_rejects_what_r_rejects():
    with pytest.raises(ValueError, match="invalid first argument"):
        RDraws(1).sample_replace(0, 2)
    with pytest.raises(ValueError, match="invalid 'size' argument"):
        RDraws(1).sample_replace(3, -1)
```

(`cases` is the file's parametrize helper, whatever its name is there.)

- [ ] **Step 2: Run, expect failures.** `NUMBA_CACHE_DIR=$SCRATCH/numba $UV run pytest tests/randomness -q`. Expected: `AttributeError: 'ReplayDraws' object has no attribute 'sample_replace'` and a `TypeError` on `DrawRecord(..., replace=True)`.

- [ ] **Step 3: Implement.**

`interfaces.py`, after `sample`:

```python
    def sample_replace(self, n: int, size: int) -> np.ndarray:
        """R's sample.int(n, size, replace = TRUE): size values from range(n), 0-based."""
        ...
```

`numpy_draws.py`:

```python
    def sample_replace(self, n: int, size: int) -> np.ndarray:
        return self._rng.integers(n, size=size).astype(np.intp)
```

`r_draws.py`, after `sample`:

```python
    def sample_replace(self, n: int, size: int) -> np.ndarray:
        """sample.int(n, size, replace = TRUE): do_sample's replace branch, one R_unif_index(n)
        per value. R never hashes a replace draw."""
        n, size = int(n), int(size)
        if n < 0 or n > 4.5e15 or (size > 0 and n == 0):
            raise ValueError("invalid first argument")
        if size < 0:
            raise ValueError("invalid 'size' argument")
        return np.fromiter((self._unif_index(n) for _ in range(size)), dtype=np.intp, count=size)
```

`replay.py`:

```python
@dataclass(frozen=True)
class DrawRecord:
    n: int
    size: int
    value: np.ndarray  # 1-based, as R returned it
    replace: bool = False


def _call(n: int, size: int, replace: bool) -> str:
    return f"sample.int({n}, {size}{', replace = TRUE' if replace else ''})"
```

`_next(self, n, size, replace=False)` compares `rec.replace != replace` along with `n` and `size` and words both messages with `_call`, and `sample_replace` is `return self._next(n, size, True).astype(np.intp)`. Text for plain draws stays `sample.int(n, size)`, so existing message matches still hold.

`tests/rfixtures.py`, `Case.draws`:

```python
        flags = (
            self.array("draw_replace").astype(bool).tolist()
            if self.has("draw_replace")
            else [False] * sizes.size
        )
        records, pos = [], 0
        for n, size, rep in zip(self.array("draw_n").tolist(), sizes.tolist(), flags, strict=True):
            records.append(DrawRecord(n, size, values[pos : pos + size], rep))
            pos += size
```

`r/fixture_io.R`: the exit hook records `list(n = n, size = size, replace = replace, value = returnValue())`, `draw_arrays` adds `draw_replace = as.integer(vapply(draws, function(d) isTRUE(d$replace), logical(1)))`, and the fail-closed block gains:

```r
  chk <- capture_draws({ set.seed(1); sample(3, 5, replace = TRUE) })
  stopifnot(length(chk$draws) == 1L, isTRUE(chk$draws[[1]]$replace),
            identical(as.integer(chk$draws[[1]]$value), as.integer(chk$value)))
```

`r/fixtures_rsample.R`, appended:

```r
# sample.int(n, size, replace = TRUE): do_sample's replace branch, one R_unif_index(n) per value.
# Wkmeans' init draws sample(k, n - k, replace = TRUE).
replace_case <- function(id, n, size, seed) {
  set.seed(seed)
  s0 <- seed_now()
  first <- sample.int(n, size, replace = TRUE)
  s1 <- seed_now()
  then <- sample.int(10, 10)
  write_case("rsample", paste0("replace_", id), list(n = n, size = size, seed = seed),
             list(state0 = s0, first = first, state1 = s1, then = then, state2 = seed_now()),
             extra = list(rng = rng))
}
replace_case("k3_size147", 3, 147, 291102)  # iris k = 3
replace_case("k4_size6", 4, 6, 55)
replace_case("k1_size5", 1, 5, 2)           # rbits(0) still reads one uniform per value
replace_case("k7_size1", 7, 1, 291102)
replace_case("intmax_size3", 2147483647, 3, 291102)  # 31 bits, two uniforms per value
replace_case("k5_size0", 5, 0, 291102)
```

- [ ] **Step 4: Record and check.** `"$RS" r/fixtures_rsample.R`. Expected: the existing cases rewrite byte for byte (`git status` shows only the six new folders, plus manifests whose `versions` didn't change; `git diff --stat tests/fixtures/rsample` must be empty for existing cases), and six `replace_*` folders. Then `$UV run pytest tests/randomness -q`. Expected: all pass.

- [ ] **Step 5: Gates and commits.** `ruff check` and `ruff format --check` on the touched Python files. Commits: `Add replace draws to the Draws protocol` (the four `_random` files, `tests/rfixtures.py`, both test files), `Record replace draws in the R capture` (`r/fixture_io.R`, `r/fixtures_rsample.R`, the six fixture folders). Push.

---

### Task 2: R fixtures for whclust, wkmeans and the DN wrappers

**Files:**
- Create: `r/wcluster_counted.R`, `r/fixtures_whclust.R`, `r/fixtures_wkmeans.R`, `r/fixtures_dn_wcluster.R`
- Create: `tests/fixtures/whclust/*`, `tests/fixtures/wkmeans/*`, `tests/fixtures/dn_wcluster/*`
- Modify: `r/README.md` (the three scripts and the counted copies)

**Interfaces:**
- Produces, per group (`Case.args`, arrays, manifest extras):
  - `whclust/<id>`: arrays `x`, `w`, and unless R stopped `merge` (int, (n-1, 2), R's labels), `height`, `order` (int, 1-based). Extras `error` ("" when R returned) and `counts` (the counted copy's `inf_pick` and `self_merge`).
  - `wkmeans/<id>`: args `k`, `num_init`, `max_iter`, `seed`. Arrays `x`, `w`, optional `centers_in` ((k, p), the transpose of R's `cl.centers`), draws (`draw_n`, `draw_size`, `draw_values`, `draw_replace`), traced `init_labels` ((num_init, n), each start), `single_labels` ((num_init, n)) and `single_wwcss` ((num_init,)), and unless R stopped `labels`, `centers` ((k', p)), `wwcss`, `total`. Extras `error` and `counts`.
  - `dn_wcluster/<id>`: the R datanugget table as `dn_centers`, `dn_weights`, `dn_scales`, `dn_assignments` (1-based), plus per call the outputs above, `pca_*` (`wpca`'s seven fields) and `predict_frame`, `predict_dn` (1-based) with `newx`, `newx_dn_*`.

- [ ] **Step 1: `r/wcluster_counted.R`.** Copies of `WCluster:::init`, `WCluster:::Wkmeans_single`, `WCluster::Wkmeans` and `WCluster::Whclust`, statement for statement from the 1.3.0 source, calling the namespace's own `wmean`, `wwcss`, `wdist_mat`, `distw`, `distwij` and `forder`, with counters in an environment. The copies only prove a case reaches its quirk. A case always records WCluster's own output, and `prove()` stops unless the copy returned exactly that.

```r
# WCluster 1.3.0's iterative functions, statement for statement, plus counters. Used only to
# prove that a fixture case reaches the quirk it's named for: prove() checks the copy returns
# exactly what WCluster returns, and the case records WCluster's own output.
W <- asNamespace("WCluster")

new_counts <- function() {
  cnt <- new.env()
  for (nm in c("redraw", "revert", "stale_decides", "recreate", "gap", "tb_na", "nan_test",
               "inf_pick", "self_merge")) assign(nm, 0L, cnt)
  cnt
}
bump <- function(cnt, nm) assign(nm, get(nm, cnt) + 1L, cnt)

counted_init <- function(x, w = rep(1, nrow(x)), k, u = NULL, cnt) {
  clcenter <- u
  z = NULL
  zz = NULL
  n <- nrow(x)
  ff <- function(k, n) sample(c(1:k, sample(k, n - k, replace = TRUE)))
  while (length(unique(zz)) < k) {
    if (!is.null(zz)) bump(cnt, "redraw")
    if (is.null(u)) {
      y <- ff(k, n)
      while (min(table(y)) < 2) { bump(cnt, "redraw"); y <- ff(k, n) }
      clcenter <- W$wmean(x, cl = y, w)
    }
    z = W$wdist_mat(x, centers = clcenter, w = w)
    zz = apply(z, 1, which.min)
    zz = setNames(zz, row.names(x))
    if (length(unique(zz)) < k & !is.null(u)) {
      stop("For the initialization, no data assigned to one(or multiple) of the clusters based on the weighted\n         distance to the initial cluster centers chosen by users.")
    }
  }
  return(zz)
}

counted_single <- function(x, y, u = NULL, w = rep(1, length(y)), k, K = 30, cnt) {
  if (missing(y)) y = counted_init(x, w, k, u, cnt)
  n = length(y)
  k <- max(y)
  tb <- table(y)
  yy = y2 = y
  for (l in 1:K) {
    uu0 <- uu00 <- W$wwcss(x, cl = yy, w)
    for (ll in 1:20) {
      yy0 <- yy
      z = NULL
      kcl_centers <- W$wmean(x, cl = yy0, w = w)
      z = W$wdist_mat(x, centers = kcl_centers, w = w)
      yy = apply(z, 1, which.min)
      uu0 = W$wwcss(x, cl = yy, w)
      if (is.na(uu0 > uu00)) bump(cnt, "nan_test")
      if (uu0 > uu00) {
        bump(cnt, "revert")
        yy = yy0
      }
      else if (uu0 <= uu00)
        uu00 = uu0
    }
    uu0 = uu00 = W$wwcss(x, cl = yy, w)
    for (i in 1:n) {
      for (j in 1:k) {
        y2 = yy
        if (y2[i] != j) {
          if (is.na(tb[y2[i]])) bump(cnt, "tb_na")
          else if ((tb[y2[i]] > 1) != (sum(yy == y2[i]) > 1)) bump(cnt, "stale_decides")
        }
        if (y2[i] != j & tb[y2[i]] > 1) {
          y2[i] = j
          uu1 = W$wwcss(x, y2, w)
          if (is.na(uu1 <= uu0)) bump(cnt, "nan_test")
          if (uu1 <= uu0) {
            if (!(j %in% yy)) bump(cnt, "recreate")
            uu0 = uu1
            yy = y2
            tb = table(yy)
          }
        }
      }
    }
    if (all(yy0 == yy))
      break
  }
  if (length(unique(yy)) < max(yy)) bump(cnt, "gap")
  yy = setNames(yy, row.names(x))
  list(uu0, yy)
}

# Wkmeans' loop over starts, without its argument checks, prints and messages.
counted_wkmeans <- function(dataset, k, cl.centers = NULL, obs.weights = rep(1, nrow(dataset)),
                            num.init = 1, max.iterations = 10, seed = 291102, cnt) {
  dataset <- as.matrix(dataset)
  set.seed(seed)
  k = floor(k)
  num.init = floor(num.init)
  max.iterations = floor(max.iterations)
  if (!is.null(cl.centers)) num.init = 1
  obs.weights = as.vector(obs.weights)
  best.clusters = NULL
  for (U in 1:num.init) {
    result = counted_single(x = dataset, u = cl.centers, w = obs.weights, k = k,
                            K = max.iterations, cnt = cnt)
    if (is.null(best.clusters) || best.WWCSS.results > result[[1]]) {
      best.clusters = result[[2]]
      best.WWCSS.results = result[[1]]
    }
  }
  list(best.clusters, W$wwcss(x = dataset, cl = best.clusters, w = obs.weights, groupSum = TRUE))
}

counted_whclust <- function(x, w = rep(1, nrow(x)), cnt) {
  x = as.matrix(x)
  y = -(1:(n <- nrow(x)))
  kk = NULL
  y0 = sort(unique(y))
  m = n
  ws2 = rep(NA, n - 1)
  dd = W$distw(x, y, w)
  for (ii in 1:(n - 1)) {
    k = which.min(dd)
    r0 = (k - 1) %% nrow(dd) + 1
    if (r0 <= (k - 1) %/% nrow(dd) + 1) bump(cnt, "inf_pick")
    u = y0[k0 <- k %% m]
    v = y0[k1 <- k %/% m + if (k0 == 0) 0 else 1]
    if (k0 == 0) {
      u = y0[m]
      k0 <- m
    }
    if (k0 == k1) bump(cnt, "self_merge")
    kk = rbind(kk, c(u, v))
    y[y == u] = y[y == v] = ii
    y0 = sort(unique(y))
    m = m - 1
    if (ii < n - 1) {
      dd = dd[-c(k0, k1), -c(k0, k1)]
      newdd = cbind(rbind(dd, 0), Inf)
      newdd[m, 1:(m - 1)] = sapply(1:(m - 1), function(t) {
        W$distwij(y0[t], y0[m], y, x, w)
      })
      dd = newdd
    }
    ws2[ii] = W$wwcss(x, y, w)
  }
  list(merge = kk, height = ws2, order = W$forder(kk))
}

# Runs f (a counted copy) and g (WCluster's own) on the same arguments, stops unless both give
# the same result or the same error, and returns the counts and g's result.
prove <- function(id, quirks, f, g, args) {
  cnt <- new_counts()
  run <- function(fn, extra) tryCatch(
    suppressMessages(withCallingHandlers(do.call(fn, c(args, extra)),
                                         warning = function(w) invokeRestart("muffleWarning"))),
    error = function(e) structure(conditionMessage(e), class = "r_error"))
  mine <- run(f, list(cnt = cnt))
  theirs <- run(g, list())
  if (!identical(unname(unclass(mine)), unname(unclass(theirs)))) stop(id, ": the counted copy disagrees with WCluster")
  counts <- as.list(cnt)
  for (q in quirks) if (counts[[q]] == 0L) stop(id, ": never reaches ", q)
  list(counts = counts, value = theirs)
}
```

`g` for Wkmeans is a wrapper that calls `WCluster::Wkmeans` inside `capture.output` and returns `list(res[[1]], res[[3]])`, and for Whclust `function(x, w) unclass(WCluster::Whclust(x, w))[c("merge", "height", "order")]`, so both sides have the same shape.

- [ ] **Step 2: `r/fixtures_whclust.R`.** Sources `r/fixture_io.R` and `r/wcluster_counted.R`. One function records a case:

```r
add_whclust <- function(id, x, w, quirks = character()) {
  x <- as_dbl(x)
  pr <- prove(id, quirks, counted_whclust,
              function(x, w) unclass(WCluster::Whclust(x, w))[c("merge", "height", "order")],
              list(x = x, w = w))
  arrays <- list(x = x, w = as.double(w))
  err <- if (inherits(pr$value, "r_error")) as.character(pr$value) else ""
  if (!nzchar(err)) arrays <- c(arrays, list(merge = pr$value$merge, height = pr$value$height,
                                             order = pr$value$order))
  write_case("whclust", id, no_args, arrays, extra = list(error = err, counts = pr$counts))
  cat(sprintf("whclust %-16s n=%d error=%s counts=%s\n", id, nrow(x), err,
              paste(names(pr$counts), unlist(pr$counts), sep = "=", collapse = " ")))
}
```

Cases (spec section 8):

| id | x | w | quirks proved |
|---|---|---|---|
| `iris_unit` | `iris[, 1:4]` (rows 102 and 143 are equal) | 1 | |
| `iris_w1372` | `iris[, 1:4]` | `rep(c(1, 3, 7, 2), length.out = 150)` | |
| `quakes200_w1372` | `quakes[1:200, ]` | same pattern | |
| `grid_dup` | `expand.grid(1:4, 1:3)` stacked twice (24 rows) | 1 | |
| `zero_w1` | `iris[1:30, 1:4]` | 1, with `w[7] = 0` | `inf_pick` |
| `zero_w2` | `iris[1:30, 1:4]` | 1, with `w[c(7, 19)] = 0` | `inf_pick` |
| `selfmerge_last` | `c(0, 0.1, 5, 5.1)` as one column | `c(1, 1, -1, -1)` | `self_merge` |
| `selfmerge_mid` | searched: integer columns, `n` in 5 to 9, weights from `c(-1, 1)` | | `self_merge` before the last step |
| `n3` | `iris[c(1, 51, 101), 1:4]` | 1 | |
| `one_col` | `faithful[1:40, 1, drop = FALSE]` | 1 | |
| `n2` | `iris[1:2, 1:4]` | 1 | R stops |
| `n1` | `iris[1, 1:4]` | 1 | R stops |
| `no_candidate` | `iris[1:5, 1:4]` | 0 | R stops |

`selfmerge_mid` searches `set.seed(s)` for `s` in 1 to 2,000 over random small inputs and keeps the first whose counted run has a self-merge with `ii < n - 1`, recording whatever WCluster returns or stops with. If none turns up, the case is dropped and printed as dropped (spec section 8's rule), and the fuzz carries it.

- [ ] **Step 3: `r/fixtures_wkmeans.R`.** One function records a case, with `init` and `Wkmeans_single` traced in the namespace:

```r
add_wkmeans <- function(id, x, k, w = rep(1, nrow(x)), centers = NULL, num.init = 1,
                        max.iterations = 10, seed = 291102, quirks = character()) {
  x <- as_dbl(x)
  args <- list(dataset = x, k = k, cl.centers = centers, obs.weights = w, num.init = num.init,
               max.iterations = max.iterations, seed = seed)
  pr <- prove(id, quirks, counted_wkmeans, function(...) {
    out <- NULL
    capture.output(out <- WCluster::Wkmeans(...))
    list(out[[1]], out[[3]])
  }, args)
  .capture$inits <- list()
  .capture$singles <- list()
  trace("init", where = W, print = FALSE,
        exit = quote(.capture$inits[[length(.capture$inits) + 1L]] <- returnValue()))
  trace("Wkmeans_single", where = W, print = FALSE,
        exit = quote(.capture$singles[[length(.capture$singles) + 1L]] <- returnValue()))
  on.exit(suppressMessages(for (f in c("init", "Wkmeans_single")) untrace(f, where = W)))
  set.seed(seed)
  res <- capture_draws(tryCatch(suppressMessages({
    out <- NULL
    capture.output(out <- WCluster::Wkmeans(x, k, centers, w, num.init, max.iterations, seed))
    out
  }), error = function(e) structure(conditionMessage(e), class = "r_error")))
  err <- if (inherits(res$value, "r_error")) as.character(res$value) else ""
  arrays <- c(list(x = x, w = as.double(w)), draw_arrays(res$draws))
  if (!is.null(centers)) arrays$centers_in <- t(centers)
  if (length(.capture$inits)) arrays$init_labels <- do.call(rbind, lapply(.capture$inits, as.integer))
  if (length(.capture$singles)) {
    arrays$single_labels <- do.call(rbind, lapply(.capture$singles, function(s) as.integer(s[[2]])))
    arrays$single_wwcss <- vapply(.capture$singles, function(s) as.double(s[[1]]), 0)
  }
  if (!nzchar(err)) {
    out <- res$value
    arrays <- c(arrays, list(labels = as.integer(out[[1]]), centers = unname(out[[2]]),
                             wwcss = unname(out[[3]]$WWCSS), total = out[[3]]$TotalWWCSS))
  }
  write_case("wkmeans", id, list(k = k, num_init = num.init, max_iter = max.iterations, seed = seed),
             arrays, extra = list(error = err, counts = pr$counts))
  cat(sprintf("wkmeans %-16s draws=%d error=%s\n", id, length(res$draws), err))
}
```

`capture_draws` is what records `set.seed`'s following draws, so the `set.seed(seed)` before it only matters for the trace order. A start that errors leaves its `init` traced and its `Wkmeans_single` not, which is how the test knows where R stopped. Cases (spec section 8), each searching seeds 1 to 500 where it names a quirk, keeping the first seed that proves it:

| id | data | k and options | quirks proved |
|---|---|---|---|
| `iris_k3` | `iris[, 1:4]` | k = 3 | |
| `iris_k3_init5` | `iris[, 1:4]` | k = 3, `num.init = 5` | |
| `quakes500_k4_w` | `quakes[1:500, ]`, weights 1, 3, 7, 2 repeating | k = 4 | |
| `ucenters` | `iris[, 1:4]` | k = 3, centers rows 1, 51, 101 | |
| `ucenters_empty` | `iris[, 1:4]` | k = 3, centers rows 1, 1, 101 | R stops |
| `small_k4` | `iris[c(1:3, 51:53, 101:103), 1:4]` (9 rows) | k = 4 | `redraw` |
| `k1` | `iris[, 1:4]` | k = 1 | |
| `maxiter1` | `iris[, 1:4]` | k = 3, `max.iterations = 1` | |
| `one_col` | `faithful[, 1, drop = FALSE]` | k = 2 | |
| `revert` | `quakes[1:60, 1:2]`, weights from `c(-1, 1, 2)` by seed | k = 3 | `revert` |
| `stale_tb` | `iris[1:40, 1:4]` | k = 3 | `stale_decides` |
| `recreate` | integer grid with zero weights | k = 4 | `recreate`, `gap` |
| `tb_na` | integer grid with zero weights | k = 4 | `tb_na`, R stops |
| `nan_stop` | small data, weights from `c(-1, 1)` | k = 2 | `nan_test`, R stops |

The searched cases loop over seeds for the draws and, where the table says "by seed", draw the weights from a second stream (`set.seed(10000 + s)` before `sample`) so the case is reproducible from its recorded arrays alone. A case whose search finds nothing within its budget is dropped and printed as dropped.

- [ ] **Step 4: `r/fixtures_dn_wcluster.R`.** `create.DN` on `quakes` with the arguments `r/fixtures_create_dn.R` uses for its quakes case, sized for about 150 nuggets (`DN.num1`, `DN.num2` chosen from that script's run, checked with `stopifnot(nrow(dn$"Data Nuggets") >= 120, ...)`), seed 291102, single core. A second datanugget from `quakes[1:400, ]` with about 60 nuggets serves as `newx`. Then `datanugget::DN.Wkmeans(dn, 3)`, `datanugget::DN.Whclust(dn)`, `WCluster::DN.Wpca(dn)` and `WCluster::DN.Wpca(dn, corr = TRUE)`, `WCluster::DNcluster.predict(dn, cl, quakes[901:1000, ])` and `WCluster::DNcluster.predict(dn, cl, dn2)` with `cl` the DN.Wkmeans labels. A check that `datanugget::DN.Wkmeans` and `WCluster::DN.Wkmeans` give identical results on this datanugget, and that the two `DN.Whclust` bodies are equal once `WCluster::` is removed from the text, keeps the fixtures valid for both packages.

- [ ] **Step 5: Record.** `"$RS" r/fixtures_whclust.R`, `"$RS" r/fixtures_wkmeans.R`, `"$RS" r/fixtures_dn_wcluster.R`, each output kept in `$SCRATCH/port/fixtures_*.txt`. Expected: every case prints, dropped cases are named, no proof fails. Run each script a second time into a copy and compare the `.bin` files byte for byte (the recording is deterministic). Report the content size of the three folders. Budget: under 8 MB together.

- [ ] **Step 6: Commit.** `Record R fixtures for Whclust, Wkmeans and the DN wrappers` (the four R scripts, `r/README.md`, the three fixture folders). Push.

---

### Task 3: Result types

**Files:**
- Modify: `src/datanuggets/result.py`
- Test: `tests/test_result.py`

**Interfaces:**
- Produces: `WkmeansResult(labels, centers, wwcss, total_wwcss)`, `DnWkmeansResult(labels, centers, wwcss, total_wwcss, row_labels)`, `WhclustResult(merge, height, order)` with `to_linkage() -> np.ndarray` ((n-1, 4) float64).

- [ ] **Step 1: Failing tests.**

```python
def test_whclust_result_to_linkage_counts_cluster_sizes():
    r = WhclustResult(merge=np.array([[2, 3], [0, 1], [5, 4]]), height=np.array([0.5, 1.0, 4.0]),
                      order=np.array([0, 1, 2, 3]))
    z = r.to_linkage()
    assert z.tolist() == [[2, 3, 0.5, 2], [0, 1, 1.0, 2], [5, 4, 4.0, 4]]
    assert scipy.cluster.hierarchy.is_valid_linkage(z)


def test_to_linkage_refuses_a_self_merge():
    r = WhclustResult(merge=np.array([[2, 3], [0, 1], [4, 4]]), height=np.zeros(3),
                      order=np.array([2, 3, 2, 3]))
    with pytest.raises(ValueError, match="itself"):
        r.to_linkage()


def test_new_results_are_frozen_and_compare_by_value():
    a = WkmeansResult(np.array([0, 1]), np.ones((2, 2)), np.ones(2), np.float64(2))
    assert a == WkmeansResult(np.array([0, 1]), np.ones((2, 2)), np.ones(2), np.float64(2))
    with pytest.raises(ValueError):
        a.labels[0] = 1
    d = DnWkmeansResult(a.labels, a.centers, a.wwcss, a.total_wwcss, np.array([1, 0, 0]))
    assert d.row_labels.tolist() == [1, 0, 0]
```

- [ ] **Step 2: Run, expect `ImportError`.**

- [ ] **Step 3: Implement** in `result.py`, after `WpcaResult`:

```python
@dataclass(frozen=True, eq=False)
class WkmeansResult(_ArrayFields):
    """wkmeans output. labels is each row's cluster, R's label minus one, so a label R left
    empty is a gap here too. centers and wwcss have one row or value per label present, in label
    order, and total_wwcss is their sum as R adds them, a 0-d float64."""

    labels: np.ndarray
    centers: np.ndarray
    wwcss: np.ndarray
    total_wwcss: np.ndarray


@dataclass(frozen=True, eq=False)
class DnWkmeansResult(_ArrayFields):
    """dn_wkmeans output: wkmeans' four fields for the nuggets, plus row_labels, each original
    row's cluster, which is the label of its nugget."""

    labels: np.ndarray
    centers: np.ndarray
    wwcss: np.ndarray
    total_wwcss: np.ndarray
    row_labels: np.ndarray


@dataclass(frozen=True, eq=False)
class WhclustResult(_ArrayFields):
    """whclust output in scipy's numbering. Row s of merge joins two clusters: a row r is
    cluster r, and the cluster made at row s is n + s. Each row keeps R's column order. height
    is R's total WWCSS after each merge, which only decreases if weights are negative, and order
    is the rows in dendrogram order, 0-based. When weights cancel, R can merge a cluster with
    itself, and then order repeats rows, as R's does."""

    merge: np.ndarray
    height: np.ndarray
    order: np.ndarray

    def to_linkage(self) -> np.ndarray:
        """scipy's (n-1, 4) linkage: merge, height, and the size of each new cluster, for
        scipy.cluster.hierarchy.dendrogram. Raises ValueError for a self-merge, which scipy
        can't represent."""
        n = self.merge.shape[0] + 1
        if np.any(self.merge[:, 0] == self.merge[:, 1]):
            raise ValueError("merge joins a cluster with itself, which a scipy linkage can't hold")
        size = np.ones(2 * n - 1)
        z = np.empty((n - 1, 4))
        for s, (a, b) in enumerate(self.merge.tolist()):
            size[n + s] = size[a] + size[b]
            z[s] = (a, b, self.height[s], size[n + s])
        return z
```

- [ ] **Step 4: Pass, gates, commit** `Add the wkmeans and whclust result types`. Push.

---

### Task 4: Shared compiled statistics

**Files:**
- Create: `src/datanuggets/wcluster/wstatsjit.py`
- Modify: `src/datanuggets/wcluster/predict.py` (`_wdist_mat(x, centers, w=None)`)
- Test: `tests/wcluster/test_wstatsjit.py`, `tests/wcluster/test_predict.py` (unchanged, must still pass)

**Interfaces:**
- Consumes: `ldjit.from_bits`, `ldjit.add`, `ldjit.to_double`, `ldjit.Z64`, `ldjit.MAX_EXP`.
- Produces (all `@compiled(nogil=True, error_model="numpy")`):
  - `rsum(v, n) -> float`: `r_sum(v[:n])`, any NaN where R's is NaN.
  - `wss_rows(x, w, rows, m, acc, mean, wbuf) -> float`: `_wss(x[rows[:m]], w[rows[:m]])`.
  - `wmean_rows(x, w, rows, m, out, buf, wbuf) -> None`: `wmean` of those rows into `out` (p).
  - `assign(x, w, centers, c, m11, zbuf, out) -> int`: each row's 1-based `which.min` over R's `wdist_mat(x, centers[:c], w)` into `out`, 0 for a row whose entries are all NaN. Returns the number of such rows.
  - `row_norms(x) -> np.ndarray`: `m11`, each row's sum of squares in order.
  - `_wdist_mat(x, centers, w=None) -> np.ndarray` in `predict.py`.

- [ ] **Step 1: Failing tests.** `tests/wcluster/test_wstatsjit.py` compares every kernel with its Python primitive on tie-heavy and overflowing data:

```python
"""wstatsjit's kernels against the Python primitives they replay, bit for bit."""

import numpy as np
import pytest

from datanuggets._compat._compiled import frozen
from datanuggets._compat.arithmetic import r_sum
from datanuggets.wcluster import wstatsjit
from datanuggets.wcluster.predict import _wdist_mat
from datanuggets.wcluster.wstats import _wss, wmean
from rfixtures import assert_same_values

RNG = np.random.default_rng(20261009)


def data(kind, n=40, p=3):
    if kind == "grid":
        return RNG.integers(0, 4, (n, p)).astype(float), RNG.integers(-1, 4, n).astype(float)
    if kind == "huge":
        return RNG.normal(size=(n, p)) * 1e154, RNG.integers(1, 4, n).astype(float)
    return RNG.normal(size=(n, p)), RNG.uniform(0.5, 3, n)


@pytest.mark.parametrize("v", [
    [1.0, 1e-30, -1.0], [1.7976931348623157e308, 1.7976931348623157e308, -1.7976931348623157e308],
    [1.7976931348623157e308, 1e292], [np.inf, 1.0], [np.inf, -np.inf], [np.nan, 1.0], [],
    list(RNG.normal(size=1000) * 10.0 ** RNG.integers(-20, 20, 1000)),
])
def test_rsum_matches_r_sum(v):
    a = np.array(v, dtype=float)
    assert_same_values(np.float64(wstatsjit.rsum(np.ascontiguousarray(a), a.size)), np.float64(r_sum(a)))


@pytest.mark.parametrize("kind", ["grid", "huge", "normal"])
def test_wss_rows_and_wmean_rows_match_the_python_primitives(kind):
    x, w = data(kind)
    p = x.shape[1]
    acc, mean, out, buf, wbuf = np.empty(p), np.empty(p), np.empty(p), np.empty(40), np.empty(40)
    for _ in range(50):
        rows = np.sort(RNG.choice(40, RNG.integers(1, 40), replace=False)).astype(np.int64)
        got = wstatsjit.wss_rows(frozen(x), frozen(w), rows, rows.size, acc, mean, wbuf)
        with np.errstate(all="ignore"):
            assert_same_values(np.float64(got), np.float64(_wss(x[rows], w[rows])))
        wstatsjit.wmean_rows(frozen(x), frozen(w), rows, rows.size, out, buf, wbuf)
        assert_same_values(out, wmean(x[rows], np.zeros(rows.size), w[rows])[0])


@pytest.mark.parametrize("kind", ["grid", "huge", "normal"])
def test_assign_is_which_min_over_wdist_mat(kind):
    x, w = data(kind)
    for k in (1, 2, 4):
        centers = x[RNG.choice(40, k, replace=False)] + RNG.integers(0, 2, (k, x.shape[1]))
        if kind == "grid":
            centers[0] = np.nan if k > 1 else centers[0]  # a NaN center, as a zero-weight cluster gives
        z = _wdist_mat(x, centers, w)
        want = np.array([np.flatnonzero(r == np.nanmin(r))[0] + 1 if not np.isnan(r).all() else 0
                         for r in z])
        out = np.empty(40, np.int64)
        dead = wstatsjit.assign(frozen(x), frozen(w), frozen(np.ascontiguousarray(centers)), k,
                                wstatsjit.row_norms(frozen(x)), np.empty((40, k)), out)
        assert out.tolist() == want.tolist() and dead == int((want == 0).sum())
```

and in `tests/wcluster/test_predict.py` (new test):

```python
def test_wdist_mat_with_weights_is_the_weighted_identity_product():
    x = np.array([[0.0, 1.0], [2.0, 2.0], [1e200, 0.0]])
    c = np.array([[0.0, 0.0], [1.0, 1.0]])
    w = np.array([2.0, -1.0, 0.0])
    got = _wdist_mat(x, c, w)
    assert np.isnan(got[:2]).all()  # row 2's z is Inf in both columns, and 0 * Inf is NaN
    assert np.isnan(got[2]).all()  # w[2] * Inf is 0 * Inf
    small = _wdist_mat(x[:2], c, w[:2])
    assert_same_values(small, w[:2, None] * _wdist_mat(x[:2], c) + 0.0)
```

- [ ] **Step 2: Run, expect `ImportError`.**

- [ ] **Step 3: Implement `_wdist_mat`** in `predict.py`:

```python
def _wdist_mat(x: np.ndarray, centers: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """R's wdist_mat: each row's squared distance to each center, as (|x|^2 + |c|^2) - 2 x.c
    with R's sums in R's order, so a value can be a rounding error below zero, times the row's
    weight. R multiplies by diag(w), so each entry also picks up 0 times every other row of its
    column: it's NaN when another row of its column is NaN or infinite, or when w[i] * z[i, j]
    is NaN. A lone infinity stays infinite. None means unit weights, cluster_predict's case. R's
    sum starts at +0, so no entry comes back as -0."""
    ones = np.ones(x.shape[1])
    with np.errstate(all="ignore"):
        m11 = r_matvec(x * x, ones)
        m22 = r_matvec(centers * centers, ones)
        z = (m11[:, None] + m22[None, :]) - 2 * r_matmul(x, centers.T)
        d = z if w is None else w[:, None] * z + 0.0
    bad = ~np.isfinite(z)
    other_bad = (np.count_nonzero(bad, axis=0)[None, :] - bad) > 0
    return np.where(other_bad | np.isnan(d), np.nan, d)
```

`cluster_predict` keeps calling `_wdist_mat(newx, wmean(x, cl, w))`.

- [ ] **Step 4: Implement `wstatsjit.py`:**

```python
"""WCluster's statistics, compiled, for the whclust and wkmeans kernels.

Each kernel replays a Python primitive bit for bit: rsum is r_sum, wss_rows is _wss on some rows
in their original order, wmean_rows is wmean's r_weighted_mean, and assign is the row-wise
which.min over _wdist_mat. Sums run in the primitive's order, double sums row by row from +0 as
reference BLAS adds them, long double sums through ldjit. The one difference is which NaN: R's
long double sum keeps a particular NaN's bits, and rsum gives numpy's NaN, so a NaN here matches
R's as a NaN, not bit for bit.
"""

import numpy as np

from datanuggets._compat import ldjit
from datanuggets._compat._compiled import compiled

_DBL_MAX_M = np.uint64(0xFFFFFFFFFFFFF800)  # DBL_MAX == _DBL_MAX_M * 2**960 as an ldjit triple


@compiled(nogil=True, error_model="numpy")
def rsum(v, n):
    """r_sum(v[:n]): a long double running sum rounded to double once, +-inf when it's past
    +-DBL_MAX. With Inf or NaN in v[:n] only those decide: NaN if there's a NaN or both
    infinities, else the infinity."""
    nan = False
    pos = False
    neg = False
    for i in range(n):
        a = v[i]
        if a != a:
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
    whose weight isn't zero, in row order, over r_sum of the weights."""
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
            if v != v:
                continue
            if arg == 0 or v < best:
                best = v
                arg = j + 1
        out[i] = arg
        if arg == 0:
            dead += 1
    return dead
```

- [ ] **Step 5: Run.** `$UV run pytest tests/wcluster/test_wstatsjit.py tests/wcluster/test_predict.py tests/compat/test_compiled.py -q`. Expected: all pass, and `test_compiled` lists `datanuggets.wcluster.wstatsjit` among the jit modules. If numba rejects the cross-module call `ldjit.add`, import the functions by name inside the module (`from datanuggets._compat.ldjit import add`) and note it in the docstring.

- [ ] **Step 6: Gates, commits.** `Compile the WCluster statistics the iterative kernels share` (`wstatsjit.py`, its test), `Let wdist_mat take row weights` (`predict.py`, its test). Push.

---

### Task 5: The Whclust reference

**Files:**
- Create: `src/datanuggets/wcluster/whclust.py` (reference, `forder`, renumbering)
- Test: `tests/wcluster/test_whclust.py`

**Interfaces:**
- Consumes: `_wss`, `wwcss` from `wstats`, `r_sum`, `as_matrix`, `as_weights`, `WhclustResult`.
- Produces: `_whclust_py(x, w, *, counts=None) -> tuple[list[tuple], list[float], list[int]]` (R's merge labels with `None` for NA, heights, 1-based order) raising `ValueError` with R's message where R stops. `counts` (a dict) gains `inf_pick` and `self_merge`. `_to_result(n, merge_labels, height, order) -> WhclustResult`. `_forder(merge) -> list[int]`.

- [ ] **Step 1: Failing tests.**

```python
"""whclust against R's Whclust: the reference and the kernel on every fixture, and the kernel
against the reference on tie-heavy data."""

import numpy as np
import pytest

from datanuggets.wcluster import whclust as wh
from rfixtures import assert_same_values, load_group

CASES = load_group("whclust")
RETURNED = [c for c in CASES if not c.manifest["error"]]
STOPPED = [c for c in CASES if c.manifest["error"]]
SMALL = [c for c in RETURNED if c.array("x").shape[0] <= 40]


def expected(case):
    n = case.array("x").shape[0]
    merge = case.array("merge").astype(np.int64)
    merge = np.where(merge < 0, -merge - 1, n + merge - 1)
    return merge, case.array("height"), case.array("order").astype(np.int64) - 1


def check_reference(case):
    counts = {}
    n = case.array("x").shape[0]
    got = wh._to_result(n, *wh._whclust_py(case.array("x"), case.array("w"), counts=counts))
    merge, height, order = expected(case)
    assert got.merge.tolist() == merge.tolist()
    assert_same_values(got.height, height)
    assert got.order.tolist() == order.tolist()
    for q, v in case.manifest["counts"].items():
        if q in counts:
            assert counts[q] == v, q


@pytest.mark.parametrize("case", [c for c in RETURNED if c in SMALL], ids=lambda c: c.id)
def test_reference_matches_r(case):
    check_reference(case)


@pytest.mark.slow
@pytest.mark.parametrize("case", [c for c in RETURNED if c not in SMALL], ids=lambda c: c.id)
def test_reference_matches_r_on_the_large_fixtures(case):
    check_reference(case)


@pytest.mark.parametrize("case", STOPPED, ids=[c.id for c in STOPPED])
def test_reference_stops_where_r_stops(case):
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        wh._whclust_py(case.array("x"), case.array("w"))
```

- [ ] **Step 2: Run, expect `ImportError` (no `whclust` module).**

- [ ] **Step 3: Implement the reference.** It follows R's Whclust literally, with R's 1-based `y0[k]` (NA past the end), the counter `m` decoding R's index whatever the matrix size, and R's errors:

```python
_MISSING = "missing value where TRUE/FALSE needed"
_EMPTY = object()  # R's integer(0), from y0[0]


def _at(v, k):
    """R's v[k] for one 1-based k: integer(0) for 0, NA (None) past the end."""
    if k == 0:
        return _EMPTY
    return v[k - 1] if k <= len(v) else None


def _r_which_min(a: np.ndarray) -> int | None:
    """which.min on a matrix: the 1-based column-major index of the first minimum, NaN
    skipped, None where every entry is NaN."""
    v = a.ravel(order="F")
    live = np.flatnonzero(~np.isnan(v))
    return None if live.size == 0 else int(live[np.argmin(v[live])]) + 1


def _r_distwij(i, j, y, x, w) -> float:
    """R's distwij(i, j, y, x, w), its argument check included."""
    if i is _EMPTY or j is _EMPTY:
        raise ValueError("argument is of length zero")
    if i is None or j is None:
        raise ValueError(_MISSING)
    rows = (y == i) | (y == j)
    xs, ws, ys = x[rows], w[rows], y[rows]
    with np.errstate(all="ignore"):
        own = [_wss(xs[ys == g], ws[ys == g]) for g in np.unique(ys)]
        return float(np.float64(_wss(xs, ws)) - np.float64(r_sum(own)))


def _forder(merge) -> list[int]:
    """R's forder: the members of the last merge, in merge order, 1-based."""
    if len(merge) < 2:
        raise ValueError("subscript out of bounds")
    u = [[-merge[0][0], -merge[0][1]]]
    for a, b in merge[1:]:
        if a is None or b is None:
            raise ValueError(_MISSING)
        u.append((u[a - 1] if a > 0 else [-a]) + (u[b - 1] if b > 0 else [-b]))
    return u[-1]


def _whclust_py(x, w, *, counts=None):
    """WCluster's Whclust, statement for statement, as the test reference and the self-merge
    fallback. Returns R's merge rows (labels, None for NA), heights and 1-based order."""
    n = x.shape[0]
    if n < 2:
        raise ValueError("subscript out of bounds")  # distw's dd[2, 1]
    if counts is not None:
        counts.update(inf_pick=0, self_merge=0)
    y = -np.arange(1, n + 1, dtype=np.int64)
    y0 = list(np.unique(y))
    m = n
    dd = np.full((n, n), np.nan)
    for i in range(1, n):
        for j in range(i):
            dd[i, j] = _r_distwij(y0[i], y0[j], y, x, w)
    merge, height = [], []
    for ii in range(1, n):
        k = _r_which_min(dd)
        if k is None:
            raise ValueError("argument is of length zero")
        size = dd.shape[0]
        if counts is not None and (k - 1) % size <= (k - 1) // size:
            counts["inf_pick"] += 1
        k0 = k % m
        k1 = k // m + (0 if k0 == 0 else 1)
        u, v = _at(y0, k0), _at(y0, k1)
        if k0 == 0:
            u, k0 = y0[m - 1], m
        if counts is not None and k0 == k1:
            counts["self_merge"] += 1
        merge.append((None if u is None else int(u), None if v is None else int(v)))
        if v is not None:
            y[y == v] = ii
        if u is not None:
            y[y == u] = ii
        y0 = list(np.unique(y))
        m -= 1
        if ii < n - 1:
            keep = [t for t in range(size) if t + 1 not in (k0, k1)]
            dd = dd[np.ix_(keep, keep)]
            new = np.full((len(keep) + 1, len(keep) + 1), np.inf)
            new[:-1, :-1] = dd
            new[-1, :-1] = 0.0
            vals = [_r_distwij(_at(y0, t), _at(y0, m), y, x, w) for t in range(1, m)]
            if m > new.shape[0]:
                raise ValueError("subscript out of bounds")
            new[m - 1, : m - 1] = vals
            dd = new
        height.append(wwcss(x, y, w))
    return merge, height, _forder(merge)


def _to_result(n, merge, height, order) -> WhclustResult:
    """R's merge labels in scipy's numbering, order 0-based."""
    pairs = [[-a - 1 if a < 0 else n + a - 1 for a in row] for row in merge]
    return WhclustResult(
        merge=np.array(pairs, dtype=np.intp).reshape(-1, 2),
        height=np.array(height, dtype=np.float64),
        order=np.array(order, dtype=np.intp) - 1,
    )
```

The `n2` fixture stops in `_forder` and `n1` in the `n < 2` check, both with "subscript out of bounds". `no_candidate` stops at `_r_which_min` with "argument is of length zero".

- [ ] **Step 4: Run.** `$UV run pytest tests/wcluster/test_whclust.py -q` then with `-m slow` for the iris and quakes fixtures. Expected: all pass. A mismatch on `selfmerge_mid` is debugged against R directly with a scratch script running the counted copy step by step, never by editing the fixture.

- [ ] **Step 5: Commit** `Port Whclust as a plain-Python reference`. Push.

---

### Task 6: The Whclust kernel and `whclust`

**Files:**
- Create: `src/datanuggets/wcluster/whclustjit.py`
- Modify: `src/datanuggets/wcluster/whclust.py` (`whclust`)
- Test: `tests/wcluster/test_whclust.py`

**Interfaces:**
- Consumes: `wstatsjit.rsum`, `wstatsjit.wss_rows`, `_whclust_py`, `_forder`, `_to_result`.
- Produces: `whclustjit.whclust(x, w, merge, height) -> int` (status `OK`, `SELF_MERGE`, `NO_CANDIDATE`), `whclust(x, w=None) -> WhclustResult`.

- [ ] **Step 1: Failing tests** (added to `test_whclust.py`):

```python
@pytest.mark.parametrize("case", RETURNED, ids=[c.id for c in RETURNED])
def test_whclust_matches_r(case):
    got = whclust(case.array("x"), case.array("w"))
    merge, height, order = expected(case)
    assert got.merge.tolist() == merge.tolist()
    assert_same_values(got.height, height)
    assert got.order.tolist() == order.tolist()


@pytest.mark.parametrize("case", STOPPED, ids=[c.id for c in STOPPED])
def test_whclust_stops_where_r_stops(case):
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        whclust(case.array("x"), case.array("w"))


def fuzz_inputs(seed, family):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 14))
    p = int(rng.integers(1, 4))
    x = rng.integers(0, 3, (n, p)).astype(float)
    if family == "dup":
        x[rng.integers(0, n, n // 2)] = x[0]
    if family == "huge":
        x = x * 1e154 + rng.normal(size=(n, p)) * 1e153
    w = {"unit": np.ones(n), "int": rng.integers(1, 4, n), "zero": rng.integers(0, 3, n),
         "signed": rng.choice([-1, 1], n), "dup": rng.integers(1, 3, n),
         "huge": rng.integers(1, 3, n)}[family].astype(float)
    return x, w


FAMILIES = ["unit", "int", "zero", "signed", "dup", "huge"]


def run_both(x, w, counts):
    try:
        ref = wh._to_result(x.shape[0], *wh._whclust_py(x, w, counts=counts))
    except ValueError as e:
        with pytest.raises(ValueError, match=re.escape(str(e))):
            whclust(x, w)
        return
    got = whclust(x, w)
    assert got.merge.tolist() == ref.merge.tolist()
    assert_same_values(got.height, ref.height)
    assert got.order.tolist() == ref.order.tolist()


@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_tie_heavy_data(family):
    for seed in range(40):
        run_both(*fuzz_inputs(seed, family), {})


def test_fuzz_reaches_the_inf_pick_and_the_self_merge():
    total = {"inf_pick": 0, "self_merge": 0}
    for family in ("zero", "signed"):
        for seed in range(40):
            c = {}
            run_both(*fuzz_inputs(seed, family), c)
            for q in total:
                total[q] += c.get(q, 0)
    assert all(v > 0 for v in total.values()), total


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_a_large_fuzz(family):
    for seed in range(40, 1040):
        run_both(*fuzz_inputs(seed, family), {})


def test_whclust_takes_any_layout_and_dtype():
    case = next(c for c in RETURNED if c.id == "zero_w1")
    x, w = case.array("x"), case.array("w")
    want = whclust(x, w)
    for xv in (np.asfortranarray(x), np.repeat(x, 2, axis=1)[:, ::2], frozen(x.copy())):
        assert whclust(xv, list(w)) == want
    grid = next(c for c in RETURNED if c.id == "grid_dup").array("x")
    assert whclust(grid.astype(np.int64)) == whclust(grid)


def test_whclust_needs_three_rows():
    with pytest.raises(ValueError, match="subscript out of bounds"):
        whclust(np.zeros((2, 2)))


def test_to_linkage_drives_scipy_dendrogram_in_r_order():
    case = next(c for c in RETURNED if c.id == "iris_w1372")
    got = whclust(case.array("x"), case.array("w"))
    leaves = scipy.cluster.hierarchy.dendrogram(got.to_linkage(), no_plot=True)["leaves"]
    assert leaves == got.order.tolist()
```

- [ ] **Step 2: Run, expect `ImportError` for `whclust`.**

- [ ] **Step 3: Implement `whclustjit.py`:**

```python
"""WCluster's Whclust, compiled.

R keeps the distances between the clusters left in an m by m matrix, copies it smaller after
every merge, and computes every cluster's wss again for each height. Every value R computes here
is a value of wss on the same rows in the same order, so computing it once gives the same bits:
a cluster's own wss when it forms, and its distance to each other cluster when it forms. The
distances live in one n by n array, a merged cluster takes over its column parent's slot, and
rank[slot] is the cluster's position in R's sorted labels: singletons first in reverse row order
(R labels them -1 to -n), then merged clusters in the order they formed.

R's which.min reads its matrix column by column. Each column here keeps its first minimum below
the diagonal, so the pick is the first column, in position order, holding the smallest one. A
column is searched again only when the row holding its minimum is merged away, and a new
cluster, the last row of every column, takes a column's minimum only by being strictly smaller.
"""

import numpy as np

from datanuggets._compat._compiled import compiled
from datanuggets.wcluster import wstatsjit

OK = 0
SELF_MERGE = 1  # R merges a cluster with itself: the caller replays R in the reference
NO_CANDIDATE = 2  # every distance left is NaN and no merged cluster is left to pick: R stops


@compiled(nogil=True, error_model="numpy")
def _pooled(x, w, pool, sa, na, sb, nb, rows, acc, mean, wbuf):
    """wss of two clusters' rows together, merged into original row order."""
    i = 0
    j = 0
    t = 0
    while i < na and j < nb:
        if pool[sa + i] < pool[sb + j]:
            rows[t] = pool[sa + i]
            i += 1
        else:
            rows[t] = pool[sb + j]
            j += 1
        t += 1
    while i < na:
        rows[t] = pool[sa + i]
        i += 1
        t += 1
    while j < nb:
        rows[t] = pool[sb + j]
        j += 1
        t += 1
    return wstatsjit.wss_rows(x, w, rows, t, acc, mean, wbuf)


@compiled(nogil=True, error_model="numpy")
def _scan(d, byrank, m, idx, colmin, colarg):
    """The column at position idx: its first minimum over the rows after it, NaN skipped."""
    c = byrank[idx]
    best = np.nan
    arg = -1
    for t in range(idx + 1, m):
        r = byrank[t]
        v = d[r, c]
        if v == v and (arg < 0 or v < best):
            best = v
            arg = r
    colmin[c] = best
    colarg[c] = arg


@compiled(nogil=True, error_model="numpy")
def _compact(pool, start, size, byrank, m, tmp):
    """Moves the live clusters' rows to the front of pool, returns the first free index."""
    t = 0
    for idx in range(m):
        c = byrank[idx]
        for i in range(size[c]):
            tmp[t + i] = pool[start[c] + i]
        start[c] = t
        t += size[c]
    for i in range(t):
        pool[i] = tmp[i]
    return t


@compiled(nogil=True, error_model="numpy")
def whclust(x, w, merge, height):
    """Fills merge (n-1, 2) with R's labels and height (n-1) with R's heights, n >= 3.
    Returns OK, SELF_MERGE or NO_CANDIDATE."""
    n, p = x.shape
    d = np.empty((n, n))
    pool = np.empty(2 * n, np.int64)
    start = np.empty(n, np.int64)
    size = np.empty(n, np.int64)
    rank = np.empty(n, np.int64)
    label = np.empty(n, np.int64)
    merged = np.zeros(n, np.bool_)
    own = np.empty(n)
    colmin = np.empty(n)
    colarg = np.empty(n, np.int64)
    byrank = np.empty(n, np.int64)
    rows = np.empty(n, np.int64)
    wbuf = np.empty(n)
    acc = np.empty(p)
    mean = np.empty(p)
    two = np.empty(2)
    tot = np.empty(n)
    for r in range(n):
        pool[r] = r
        start[r] = r
        size[r] = 1
        rank[r] = n - 1 - r
        label[r] = -(r + 1)
        byrank[n - 1 - r] = r
        rows[0] = r
        own[r] = wstatsjit.wss_rows(x, w, rows, 1, acc, mean, wbuf)
    for a in range(n):
        for b in range(a + 1, n):
            two[0] = own[a]
            two[1] = own[b]
            v = _pooled(x, w, pool, a, 1, b, 1, rows, acc, mean, wbuf) - wstatsjit.rsum(two, 2)
            d[a, b] = v
            d[b, a] = v
    m = n
    for idx in range(m):
        _scan(d, byrank, m, idx, colmin, colarg)
    top = n
    for step in range(1, n):
        found = False
        best = 0.0
        bc = -1
        for idx in range(m):
            c = byrank[idx]
            if colarg[c] >= 0 and (not found or colmin[c] < best):
                found = True
                best = colmin[c]
                bc = c
        ru = -1
        cv = -1
        if found and best < np.inf:
            ru = colarg[bc]
            cv = bc
        else:  # R's which.min lands on a +Inf: a merged column's row 1, or a singleton's first +Inf
            for idx in range(m):
                c = byrank[idx]
                if merged[c]:
                    if idx == 0:
                        return SELF_MERGE
                    ru = byrank[0]
                    cv = c
                    break
                if colarg[c] >= 0:
                    ru = colarg[c]
                    cv = c
                    break
            if ru < 0:
                return NO_CANDIDATE
        merge[step - 1, 0] = label[ru]
        merge[step - 1, 1] = label[cv]
        na = size[ru]
        nb = size[cv]
        if top + na + nb > 2 * n:
            top = _compact(pool, start, size, byrank, m, rows)
        sa = start[ru]
        sb = start[cv]
        s = top
        i = 0
        j = 0
        while i < na and j < nb:
            if pool[sa + i] < pool[sb + j]:
                pool[s] = pool[sa + i]
                i += 1
            else:
                pool[s] = pool[sb + j]
                j += 1
            s += 1
        while i < na:
            pool[s] = pool[sa + i]
            i += 1
            s += 1
        while j < nb:
            pool[s] = pool[sb + j]
            j += 1
            s += 1
        new = cv
        start[new] = top
        size[new] = na + nb
        top += na + nb
        own[new] = wstatsjit.wss_rows(x, w, pool[start[new]:], size[new], acc, mean, wbuf)
        label[new] = step
        merged[new] = True
        rank[new] = n + step
        q = 0
        for idx in range(m):
            c = byrank[idx]
            if c != ru and c != cv:
                byrank[q] = c
                q += 1
        byrank[q] = new
        m = q + 1
        for idx in range(m):
            tot[idx] = own[byrank[idx]]
        height[step - 1] = wstatsjit.rsum(tot, m)
        if step < n - 1:
            for idx in range(m - 1):
                t = byrank[idx]
                two[0] = own[t]
                two[1] = own[new]
                v = _pooled(x, w, pool, start[t], size[t], start[new], size[new], rows, acc,
                            mean, wbuf) - wstatsjit.rsum(two, 2)
                d[t, new] = v
                d[new, t] = v
            colarg[new] = -1
            for idx in range(m - 1):
                c = byrank[idx]
                a = colarg[c]
                if a == ru or a == cv:
                    _scan(d, byrank, m, idx, colmin, colarg)
                else:
                    v = d[new, c]
                    if v == v and (a < 0 or v < colmin[c]):
                        colmin[c] = v
                        colarg[c] = new
    return OK
```

- [ ] **Step 4: Implement `whclust`** in `whclust.py`:

```python
def whclust(x, w=None) -> WhclustResult:
    """Weighted Ward hierarchical clustering, as WCluster's Whclust computes it.

    Every row starts as its own cluster, and each step merges the pair whose merge adds the
    least weighted within-cluster sum of squares, R's distwij, with R's first pick on ties. merge
    is in scipy's numbering (WhclustResult), height is R's total WWCSS after each merge, and
    order is R's forder, so the output matches R's merge, height and order exactly. w is one
    weight per row and None means ones. A row with weight zero has NaN distances, so it merges
    last, the way R's which.min picks an Inf. Weights that cancel can make R merge a cluster
    with itself, which this replays through R's own steps. R stops below three rows, and so
    does this. Needs n by n float64 working memory.
    """
    x = as_matrix(x, logical=True, vector=True)
    w = as_weights(w, x.shape[0])
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"whclust needs at least 3 rows, got {n}: R stops with \"subscript out of bounds\"")
    merge = np.empty((n - 1, 2), np.int64)
    height = np.empty(n - 1)
    status = whclustjit.whclust(frozen(np.ascontiguousarray(x)), frozen(w), merge, height)
    if status == whclustjit.SELF_MERGE:
        return _to_result(n, *_whclust_py(x, w))
    if status == whclustjit.NO_CANDIDATE:
        raise ValueError("every distance left is NaN, so R's which.min finds nothing to merge "
                         "and R stops with \"argument is of length zero\"")
    rows = [tuple(r) for r in merge.tolist()]
    return _to_result(n, rows, height, _forder(rows))
```

The error tests match on R's quoted message, so each message quotes it.

- [ ] **Step 5: Run.** `NUMBA_CACHE_DIR=$SCRATCH/numba $UV run pytest tests/wcluster/test_whclust.py -q -n 8` then `-m slow`. Expected: all pass, including the quakes 200 fixture in the default run (the kernel takes well under a second there).

- [ ] **Step 6: Commit** `Compile Whclust with cached cluster sums` (`whclustjit.py`, `whclust.py`, tests). Push.

---

### Task 7: The Wkmeans reference and init

**Files:**
- Create: `src/datanuggets/wcluster/wkmeans.py` (`_prepare`, `_init`, `_single_py`, `_wkmeans_py`)
- Test: `tests/wcluster/test_wkmeans.py`

**Interfaces:**
- Consumes: `wmean`, `wwcss`, `_wdist_mat`, `as_draws`, `ReplayDraws`, `Draws.sample_replace`.
- Produces:
  - `_prepare(x, k, w, centers, num_init, max_iter) -> tuple` (`x`, `w`, `k`, `centers` or None, `num_init`, `max_iter`), with R's checks in R's order.
  - `_init(x, w, k, centers, draws, assign, wmean_of, counts=None) -> np.ndarray` (int64, R's 1-based labels).
  - `_single_py(x, w, y, K, counts=None) -> tuple[float, np.ndarray]`.
  - `_wkmeans_py(x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=291102, counts=None, trace=None) -> WkmeansResult`. `trace`, a list, gets `(start, wwcss, labels)` per start.
  - Messages: `_MISSING`, `_LIST`, `_LENGTH`, `_HANG_*`, `_USER_EMPTY`.

- [ ] **Step 1: Failing tests.**

```python
"""wkmeans against R's Wkmeans: draws, starts, per-start results and the answer, for the
reference and the kernel, and the kernel against the reference."""

import re

import numpy as np
import pytest

from datanuggets._random.replay import ReplayDraws
from datanuggets.wcluster import wkmeans as wk
from rfixtures import assert_same_values, load_group

CASES = load_group("wkmeans")
RETURNED = [c for c in CASES if not c.manifest["error"]]
STOPPED = [c for c in CASES if c.manifest["error"]]
SMALL = [c for c in CASES if c.array("x").shape[0] <= 160]


def call(case, fn, **extra):
    a = case.args
    centers = case.array("centers_in") if case.has("centers_in") else None
    return fn(case.array("x"), a["k"], w=case.array("w"), centers=centers,
              num_init=a["num_init"], max_iter=a["max_iter"], seed=ReplayDraws(case.draws()),
              **extra)


def check(case, got, trace):
    starts = case.array("init_labels") if case.has("init_labels") else np.zeros((0, 0))
    assert [t[0].tolist() for t in trace] == starts.astype(np.int64).tolist()[: len(trace)]
    if case.has("single_labels"):
        singles = case.array("single_labels").astype(np.int64)
        for i, (_, u, labels) in enumerate(trace[: singles.shape[0]]):
            assert labels.tolist() == singles[i].tolist()
            assert_same_values(np.float64(u), np.float64(case.array("single_wwcss")[i]))
    assert got.labels.tolist() == (case.array("labels").astype(np.int64) - 1).tolist()
    assert_same_values(got.centers, case.array("centers"))
    assert_same_values(got.wwcss, case.array("wwcss"))
    assert_same_values(got.total_wwcss, case.array("total").reshape(()))


@pytest.mark.parametrize("case", [c for c in RETURNED if c in SMALL], ids=lambda c: c.id)
def test_reference_matches_r_and_uses_r_draws(case):
    trace = []
    draws = ReplayDraws(case.draws())
    got = wk._wkmeans_py(case.array("x"), case.args["k"], w=case.array("w"),
                         centers=case.array("centers_in") if case.has("centers_in") else None,
                         num_init=case.args["num_init"], max_iter=case.args["max_iter"],
                         seed=draws, trace=trace)
    draws.assert_exhausted()
    check(case, got, trace)


@pytest.mark.parametrize("case", STOPPED, ids=[c.id for c in STOPPED])
def test_reference_stops_where_r_stops(case):
    draws = ReplayDraws(case.draws())
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        wk._wkmeans_py(case.array("x"), case.args["k"], w=case.array("w"),
                       centers=case.array("centers_in") if case.has("centers_in") else None,
                       num_init=case.args["num_init"], max_iter=case.args["max_iter"], seed=draws)
    draws.assert_exhausted()


@pytest.mark.parametrize("bad, exc, msg", [
    ({"k": "3"}, TypeError, 'k must be of class "numeric"'),
    ({"k": True}, TypeError, 'k must be of class "numeric"'),
    ({"k": 0.5}, ValueError, "k must be"),
    ({"num_init": 0.9}, ValueError, "num_init"),
    ({"max_iter": 0}, ValueError, "max_iter"),
    ({"centers": np.zeros((2, 4))}, ValueError, "centers must be"),
    ({"w": np.ones(3)}, ValueError, "w"),
])
def test_arguments_are_checked_as_r_checks_them(bad, exc, msg):
    args = {"k": 3, "w": None, "centers": None, "num_init": 1, "max_iter": 10} | bad
    x = np.arange(40.0).reshape(10, 4)
    with pytest.raises(exc, match=re.escape(msg)):
        wk._prepare(x, args["k"], args["w"], args["centers"], args["num_init"], args["max_iter"])


def test_hang_guards_raise_at_their_boundary():
    x = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]])
    wk._wkmeans_py(x, 3, seed=1)  # exactly k distinct pairs and n == 2k: runs
    with pytest.raises(ValueError, match="2k"):
        wk._wkmeans_py(x[:5], 3, seed=1)
    two_values = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    with pytest.raises(ValueError, match="distinct"):
        wk._wkmeans_py(two_values, 3, seed=1)
    # the same rows with a third weight make three distinct (row, weight) pairs, so it runs
    wk._wkmeans_py(two_values, 3, w=np.array([1.0, 1, 2, 1, 1, 1]), seed=1)


def test_init_gives_up_where_r_draws_forever():
    with pytest.raises(ValueError, match="keep drawing"):
        wk._wkmeans_py(np.arange(12.0).reshape(6, 2), 2, w=np.zeros(6), seed=1)


def test_one_row_follows_rs_diag():
    x = np.array([[1.0, 2.0]])
    c = np.zeros((1, 2))
    assert wk._wkmeans_py(x, 1, w=np.array([1.5]), centers=c).labels.tolist() == [0]
    with pytest.raises(ValueError, match="non-conformable arguments"):
        wk._wkmeans_py(x, 1, w=np.array([2.0]), centers=c)
    with pytest.raises(ValueError, match=re.escape("invalid 'nrow' value (< 0)")):
        wk._wkmeans_py(x, 1, w=np.array([-1.0]), centers=c)
```

- [ ] **Step 2: Run, expect `ImportError`.**

- [ ] **Step 3: Implement `wkmeans.py`'s reference half:**

```python
"""WCluster's Wkmeans: weighted k-means with R's random starts, Lloyd steps and single-point
moves, bit for bit."""

from __future__ import annotations

import logging
import math
import numbers

import numpy as np

from datanuggets._random.seeds import as_draws
from datanuggets._validate import as_matrix, as_weights
from datanuggets.result import WkmeansResult
from datanuggets.wcluster.predict import _wdist_mat
from datanuggets.wcluster.wstats import wmean, wwcss

log = logging.getLogger("datanuggets")

_MISSING = "missing value where TRUE/FALSE needed"
_LIST = "unimplemented type 'list' in 'orderVector1'"
_LENGTH = "The length of cluster assignments should be the same as the number of observations. "
_USER_EMPTY = ("For the initialization, no data assigned to one(or multiple) of the clusters "
               "based on the weighted\n         distance to the initial cluster centers chosen "
               "by users.")
MAX_STARTS = 10_000  # init draws this many starts before saying R would never stop


def _r_stop(message: str, why: str) -> ValueError:
    return ValueError(f"{why} (R stops: {message})")


def _prepare(x, k, w, centers, num_init, max_iter):
    """R's argument checks in R's order, then R's floors. Returns float64 x and w, int k,
    num_init and max_iter, and centers as a (k, p) float64 array or None."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Real):
        raise TypeError('k must be of class "numeric", got ' + type(k).__name__)
    x = as_matrix(x, logical=True, vector=True)
    n, p = x.shape
    if centers is not None:
        centers = as_matrix(centers, "centers")
        if centers.shape[0] != k or centers.shape[1] != p:
            raise ValueError(f"centers must be a k by ncol(x) matrix, ({k}, {p}), got {centers.shape}")
    w = as_weights(w, n)
    for name, v in (("num_init", num_init), ("max_iter", max_iter)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
            raise TypeError(f'{name} must be of class "numeric", got {type(v).__name__}')
    k, num_init, max_iter = math.floor(k), math.floor(num_init), math.floor(max_iter)
    if centers is not None:
        num_init = 1
    if k < 1:
        raise ValueError(f"k must be at least 1 after R's floor, got {k}")
    for name, v in (("num_init", num_init), ("max_iter", max_iter)):
        if v < 1:
            raise ValueError(f"{name} must be at least 1 after R's floor, got {v}: R's 1:{v} would run it twice")
    if centers is None:
        if n < 2 * k:
            raise ValueError(f"wkmeans needs at least 2k = {2 * k} rows to draw a start, got {n}: R draws forever")
        if np.unique(np.column_stack([x, w]), axis=0).shape[0] < k:
            raise ValueError(f"x and w have fewer than k = {k} distinct (row, weight) pairs, so no start reaches k clusters: R draws forever")
    elif n == 1 and not 1 <= w[0] < 2:
        raise _r_stop("invalid 'nrow' value (< 0)" if w[0] < 0 else "non-conformable arguments",
                      "R's diag(w) for one row is an identity matrix of size w")
    return x, w, k, centers, num_init, max_iter


def _draw_labels(n, k, draws):
    """R's ff(k, n): sample(c(1:k, sample(k, n - k, replace = TRUE))), 1-based."""
    pool = np.concatenate([np.arange(1, k + 1), draws.sample_replace(k, n - k) + 1])
    return pool[draws.permutation(n)].astype(np.int64)


def _init(x, w, k, centers, draws, assign, wmean_of, counts=None):
    """WCluster's init: draw labels with every label at least twice, take their weighted means
    (or the user's centers), assign each row its nearest, and repeat until k labels are used.
    assign(c) returns 1-based labels with 0 for a row whose distances are all NaN."""
    n = x.shape[0]
    for _ in range(MAX_STARTS):
        c = centers
        if c is None:
            y = _draw_labels(n, k, draws)
            while np.bincount(y, minlength=k + 1)[1:].min() < 2:
                if counts is not None:
                    counts["redraw"] += 1
                y = _draw_labels(n, k, draws)
            c = wmean_of(y)
        zz = assign(c)
        dead = zz == 0
        distinct = np.unique(zz[~dead]).size + (1 if dead.any() and not dead.all() else 0)
        if distinct >= k:
            if dead.any():
                raise _r_stop("invalid 'type' (list) of argument",
                              "a row's distances to every start center are NaN")
            return zz
        if centers is not None:
            raise _r_stop(_USER_EMPTY, "a cluster gets no rows from the centers given")
        if counts is not None:
            counts["redraw"] += 1
    raise ValueError(f"no start among {MAX_STARTS} reached k = {k} clusters (every center was "
                     "NaN or a row's distances all were): R would keep drawing forever")


def _which_min_rows(z):
    """apply(z, 1, which.min), 1-based, 0 for a row of NaN."""
    out = np.zeros(z.shape[0], np.int64)
    for i, r in enumerate(z):
        live = np.flatnonzero(~np.isnan(r))
        if live.size:
            out[i] = live[np.argmin(r[live])] + 1
    return out


def _table(y):
    """table(y): the count of each present label, in label order."""
    return np.unique(y, return_counts=True)[1]


def _test(a, b):
    if math.isnan(a) or math.isnan(b):
        raise _r_stop(_MISSING, "a WWCSS is NaN, which a cluster whose weights sum to zero gives")


def _single_py(x, w, y, K, counts=None):
    """WCluster's Wkmeans_single from init's labels y, statement for statement. Returns R's
    (uu0, yy). counts gains revert, fixed (a Lloyd step that changed nothing before step 20,
    the kernel's shortcut), stale_decides, recreate, tb_na and gap."""
    n = x.shape[0]
    k = int(y.max())
    tb = _table(y)
    yy = y.copy()
    yy0 = yy
    uu0 = math.nan
    for _ in range(K):
        uu0 = uu00 = wwcss(x, yy, w)
        for ll in range(20):
            yy0 = yy
            z = _wdist_mat(x, wmean(x, yy0, w), w)
            yy = _which_min_rows(z)
            if (yy == 0).all():
                raise _r_stop(_LENGTH, "every row's distances are NaN")
            if (yy == 0).any():
                raise _r_stop(_LIST, "a row's distances to every center are NaN")
            uu0 = wwcss(x, yy, w)
            _test(uu0, uu00)
            if uu0 > uu00:
                if counts is not None:
                    counts["revert"] += 1
                yy = yy0
            else:
                if counts is not None and ll < 19 and np.array_equal(yy, yy0):
                    counts["fixed"] += 1
                uu00 = uu0
        uu0 = uu00 = wwcss(x, yy, w)
        for i in range(n):
            for j in range(1, k + 1):
                a = int(yy[i])
                if a == j:
                    continue
                if a > tb.size:
                    if counts is not None:
                        counts["tb_na"] += 1
                    raise _r_stop(_MISSING, "R's stale table of cluster sizes is shorter than the label")
                if counts is not None and (tb[a - 1] > 1) != (np.count_nonzero(yy == a) > 1):
                    counts["stale_decides"] += 1
                if tb[a - 1] > 1:
                    y2 = yy.copy()
                    y2[i] = j
                    uu1 = wwcss(x, y2, w)
                    _test(uu1, uu0)
                    if uu1 <= uu0:
                        if counts is not None and j not in yy:
                            counts["recreate"] += 1
                        uu0 = uu1
                        yy = y2
                        tb = _table(yy)
        if np.array_equal(yy0, yy):
            break
    if counts is not None and np.unique(yy).size < yy.max():
        counts["gap"] += 1
    return uu0, yy


def _counts():
    return dict.fromkeys(("redraw", "revert", "fixed", "stale_decides", "recreate", "tb_na", "gap"), 0)


def _wkmeans_py(x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=291102,
                counts=None, trace=None):
    """WCluster's Wkmeans on the Python primitives, the test reference."""
    x, w, k, centers, num_init, max_iter = _prepare(x, k, w, centers, num_init, max_iter)
    draws = as_draws(seed)

    def assign(c):
        return _which_min_rows(_wdist_mat(x, c, w))

    best = None
    for _ in range(num_init):
        start = _init(x, w, k, centers, draws, assign, lambda y: wmean(x, y, w), counts)
        u, labels = _single_py(x, w, start, max_iter, counts)
        if trace is not None:
            trace.append((start, u, labels))
        if best is None:
            best = (u, labels)
        else:
            _test(best[0], u)
            if best[0] > u:
                best = (u, labels)
    return _result(x, w, best[1])


def _result(x, w, labels) -> WkmeansResult:
    res = wwcss(x, labels, w, group_sum=True)
    return WkmeansResult(labels=labels.astype(np.intp) - 1, centers=wmean(x, labels, w),
                         wwcss=res.per_cluster, total_wwcss=res.total)
```

The `fixed` counter counts only steps that reproduced their start without a revert. A revert also leaves the state fixed, and `revert` counts those, so between them the kernel's shortcut is covered.

- [ ] **Step 4: Run.** `$UV run pytest tests/wcluster/test_wkmeans.py -q -n 8`. Expected: all pass. The reference on `quakes500_k4_w` is slow (Python long double sums in every trial) and sits under `-m slow`.

- [ ] **Step 5: Commit** `Port Wkmeans as a plain-Python reference`. Push.

---

### Task 8: The Wkmeans kernel and `wkmeans`

**Files:**
- Create: `src/datanuggets/wcluster/wkmeansjit.py`
- Modify: `src/datanuggets/wcluster/wkmeans.py` (`wkmeans`)
- Test: `tests/wcluster/test_wkmeans.py`

**Interfaces:**
- Consumes: `wstatsjit.*`, `_prepare`, `_init`, `_result`, `_r_stop` and the messages.
- Produces: `wkmeansjit.refine(x, w, y, K, m11) -> tuple[int, float, int]` (status, uu0, outer iterations run), statuses `OK`, `NAN_TEST`, `TB_NA`, `DEAD_SOME`, `DEAD_ALL`. `wkmeansjit.centers_of(x, w, y, k) -> np.ndarray` ((k, p), `wmean` of labels 1..k, all present). `wkmeans(x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=291102, n_jobs=1) -> WkmeansResult`.

- [ ] **Step 1: Failing tests** (added to `test_wkmeans.py`):

```python
@pytest.mark.parametrize("case", RETURNED, ids=[c.id for c in RETURNED])
def test_wkmeans_matches_r_and_uses_r_draws(case):
    trace = []
    draws = ReplayDraws(case.draws())
    got = wk._wkmeans(case.array("x"), case.args["k"], w=case.array("w"),
                      centers=case.array("centers_in") if case.has("centers_in") else None,
                      num_init=case.args["num_init"], max_iter=case.args["max_iter"],
                      seed=draws, n_jobs=1, trace=trace)
    draws.assert_exhausted()
    check(case, got, trace)


@pytest.mark.parametrize("case", STOPPED, ids=[c.id for c in STOPPED])
def test_wkmeans_stops_where_r_stops(case):
    draws = ReplayDraws(case.draws())
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        wkmeans(case.array("x"), case.args["k"], w=case.array("w"),
                centers=case.array("centers_in") if case.has("centers_in") else None,
                num_init=case.args["num_init"], max_iter=case.args["max_iter"], seed=draws)
    draws.assert_exhausted()


def fuzz_inputs(seed, family):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 30))
    p = int(rng.integers(1, 4))
    k = int(rng.integers(1, max(2, n // 3)))
    x = rng.integers(0, 4, (n, p)).astype(float)
    if family == "huge":
        x = x * 1e154
    w = {"unit": np.ones(n), "int": rng.integers(1, 4, n), "zero": rng.integers(0, 3, n),
         "signed": rng.choice([-1, 1, 2], n), "huge": rng.integers(1, 3, n)}[family].astype(float)
    return x, k, w


def run_both(x, k, w, seed, counts):
    def go(fn, **kw):
        try:
            return fn(x, k, w=w, num_init=3, max_iter=4, seed=RDraws(seed), **kw), None
        except ValueError as e:
            return None, str(e)
    ref, ref_err = go(wk._wkmeans_py, counts=counts)
    got, got_err = go(wkmeans)
    assert got_err == ref_err
    if ref is not None:
        assert got.labels.tolist() == ref.labels.tolist()
        assert_same_values(got.centers, ref.centers)
        assert_same_values(got.wwcss, ref.wwcss)
        assert_same_values(got.total_wwcss, ref.total_wwcss)


FAMILIES = ["unit", "int", "zero", "signed", "huge"]


@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_tie_heavy_data(family):
    for seed in range(25):
        run_both(*fuzz_inputs(seed, family), seed, wk._counts())


def test_fuzz_reaches_every_quirk():
    total = wk._counts()
    for family in ("zero", "signed", "int"):
        for seed in range(60):
            c = wk._counts()
            run_both(*fuzz_inputs(seed, family), seed, c)
            for q in total:
                total[q] += c[q]
    assert all(total[q] > 0 for q in ("redraw", "revert", "fixed", "stale_decides", "recreate",
                                       "tb_na", "gap")), total


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_a_large_fuzz(family):
    for seed in range(25, 625):
        run_both(*fuzz_inputs(seed, family), seed, wk._counts())


def test_int_seed_is_reproducible():
    x = load_group("wkmeans")[0].array("x")
    assert wkmeans(x, 3, seed=5) == wkmeans(x, 3, seed=5)


def test_rdraws_seed_matches_r():
    case = next(c for c in RETURNED if c.id == "iris_k3_init5")
    got = wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102))
    assert got.labels.tolist() == (case.array("labels").astype(np.int64) - 1).tolist()


def test_n_jobs_does_not_change_the_result():
    case = next(c for c in RETURNED if c.id == "iris_k3_init5")
    one = wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102), n_jobs=1)
    assert wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102), n_jobs=4) == one
    assert wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102), n_jobs=-1) == one


def test_wkmeans_takes_any_layout_and_dtype():
    case = next(c for c in RETURNED if c.id == "quakes500_k4_w")
    x, w = case.array("x"), case.array("w")
    want = wkmeans(x, 4, w=w, seed=RDraws(291102))
    for xv in (np.asfortranarray(x), np.repeat(x, 2, axis=1)[:, ::2], frozen(x.copy())):
        assert wkmeans(xv, 4, w=list(w), seed=RDraws(291102)) == want


def test_r_positional_order_fails():
    with pytest.raises(TypeError):
        wkmeans(np.zeros((4, 2)), 2, None, np.ones(4))
```

- [ ] **Step 2: Run, expect `ImportError` (no `wkmeans`, no `wkmeansjit`).**

- [ ] **Step 3: Implement `wkmeansjit.py`:**

```python
"""WCluster's Wkmeans_single, compiled: R's Lloyd steps, then its single-point moves.

R recomputes the whole WWCSS for every trial move, O(n p) each. Here each label's wss and its
rows in original order are kept, so a trial computes only the two clusters that change, each on
its rows in original order, and adds the cached values of the rest in label order: the same bits
R gets. R's stale table of cluster sizes is kept as R keeps it: refreshed only when a move is
accepted, read by position, with NA past its end.

A Lloyd step's result depends only on the labels it starts from, so a step that changes nothing,
or reverts, repeats for the rest of the 20, and the loop stops there with yy0 set to those labels,
which is R's state at the start of step 20.
"""

import numpy as np

from datanuggets._compat._compiled import compiled
from datanuggets.wcluster import wstatsjit

OK = 0
NAN_TEST = 1  # an if() on a NaN WWCSS
TB_NA = 2  # tb[label] past the end of R's stale table
DEAD_SOME = 3  # a Lloyd step's row with every distance NaN: R's labels become a list
DEAD_ALL = 4  # every row so: R's labels become integer(0)


@compiled(nogil=True, error_model="numpy")
def _group(y, k, order, starts, counts):
    """Counting sort of the rows by label 1..k, each label's rows in original order."""
    for g in range(k + 2):
        counts[g] = 0
    for i in range(y.size):
        counts[y[i]] += 1
    s = 0
    for g in range(k + 1):
        starts[g] = s
        s += counts[g]
    fill = starts.copy()
    for i in range(y.size):
        order[fill[y[i]]] = i
        fill[y[i]] += 1


@compiled(nogil=True, error_model="numpy")
def _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf):
    """wwcss(x, y, w): each present label's wss into own[label], and their r_sum in label
    order."""
    _group(y, k, order, starts, counts)
    q = 0
    for g in range(1, k + 1):
        if counts[g] > 0:
            own[g] = wstatsjit.wss_rows(x, w, order[starts[g]:], counts[g], acc, mean, wbuf)
            vals[q] = own[g]
            q += 1
    return wstatsjit.rsum(vals, q)


@compiled(nogil=True, error_model="numpy")
def centers_of(x, w, y, k):
    """wmean(x, y, w) for labels 1..k, all present: (k, p)."""
    n, p = x.shape
    order = np.empty(n, np.int64)
    starts = np.empty(k + 2, np.int64)
    counts = np.empty(k + 2, np.int64)
    _group(y, k, order, starts, counts)
    out = np.empty((k, p))
    buf = np.empty(n)
    wbuf = np.empty(n)
    for g in range(1, k + 1):
        wstatsjit.wmean_rows(x, w, order[starts[g]:], counts[g], out[g - 1], buf, wbuf)
    return out


@compiled(nogil=True, error_model="numpy")
def refine(x, w, y, K, m11):
    """Wkmeans_single from init's labels y (1-based, changed in place to R's final labels).
    Returns (status, uu0, outer iterations run)."""
    n, p = x.shape
    k = 0
    for i in range(n):
        k = max(k, y[i])
    order = np.empty(n, np.int64)
    starts = np.empty(k + 2, np.int64)
    counts = np.empty(k + 2, np.int64)
    own = np.empty(k + 1)
    vals = np.empty(k + 1)
    acc = np.empty(p)
    mean = np.empty(p)
    wbuf = np.empty(n + 1)
    buf = np.empty(n + 1)
    rows = np.empty(n + 1, np.int64)
    centers = np.empty((k, p))
    zbuf = np.empty((n, k))
    yy0 = y.copy()
    tb = np.empty(k, np.int64)
    _group(y, k, order, starts, counts)
    tbn = 0
    for g in range(1, k + 1):
        if counts[g] > 0:
            tb[tbn] = counts[g]
            tbn += 1
    uu0 = np.nan
    it = 0
    for _ in range(K):
        it += 1
        uu00 = _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf)
        for _ll in range(20):
            yy0[:] = y
            _group(yy0, k, order, starts, counts)
            c = 0
            for g in range(1, k + 1):
                if counts[g] > 0:
                    wstatsjit.wmean_rows(x, w, order[starts[g]:], counts[g], centers[c], buf, wbuf)
                    c += 1
            dead = wstatsjit.assign(x, w, centers, c, m11, zbuf, y)
            if dead == n:
                return DEAD_ALL, np.nan, it
            if dead > 0:
                return DEAD_SOME, np.nan, it
            uu0 = _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf)
            if uu0 != uu0 or uu00 != uu00:
                return NAN_TEST, np.nan, it
            same = True
            if uu0 > uu00:
                y[:] = yy0
            else:
                uu00 = uu0
                for i in range(n):
                    if y[i] != yy0[i]:
                        same = False
                        break
            if same:
                break
        uu0 = _total(x, w, y, k, order, starts, counts, own, vals, acc, mean, wbuf)
        for i in range(n):
            have_wo = False
            wo = 0.0
            for j in range(1, k + 1):
                a = y[i]
                if a == j:
                    continue
                if a > tbn:
                    return TB_NA, np.nan, it
                if tb[a - 1] <= 1:
                    continue
                if not have_wo and counts[a] > 1:
                    q = 0
                    for t in range(counts[a]):
                        r = order[starts[a] + t]
                        if r != i:
                            rows[q] = r
                            q += 1
                    wo = wstatsjit.wss_rows(x, w, rows, q, acc, mean, wbuf)
                have_wo = True
                q = 0
                placed = False
                for t in range(counts[j]):
                    r = order[starts[j] + t]
                    if not placed and i < r:
                        rows[q] = i
                        q += 1
                        placed = True
                    rows[q] = r
                    q += 1
                if not placed:
                    rows[q] = i
                    q += 1
                wj = wstatsjit.wss_rows(x, w, rows, q, acc, mean, wbuf)
                q = 0
                for g in range(1, k + 1):
                    if g == a:
                        if counts[a] > 1:
                            vals[q] = wo
                            q += 1
                    elif g == j:
                        vals[q] = wj
                        q += 1
                    elif counts[g] > 0:
                        vals[q] = own[g]
                        q += 1
                uu1 = wstatsjit.rsum(vals, q)
                if uu1 != uu1 or uu0 != uu0:
                    return NAN_TEST, np.nan, it
                if uu1 <= uu0:
                    uu0 = uu1
                    y[i] = j
                    own[a] = wo
                    own[j] = wj
                    _group(y, k, order, starts, counts)
                    tbn = 0
                    for g in range(1, k + 1):
                        if counts[g] > 0:
                            tb[tbn] = counts[g]
                            tbn += 1
                    have_wo = False
        done = True
        for i in range(n):
            if y[i] != yy0[i]:
                done = False
                break
        if done:
            break
    return OK, uu0, it
```

`refine` doesn't regroup after the Lloyd loop: `_total` regroups `y` itself. After an accept, `own[a]` is `wo` only while `a` is still present, and an emptied label is skipped by `counts[a] == 0` everywhere, so its stale `own` is never read.

- [ ] **Step 4: Implement `wkmeans`** in `wkmeans.py`:

```python
_STATUS = {
    wkmeansjit.NAN_TEST: (_MISSING, "a WWCSS is NaN, which a cluster whose weights sum to zero gives"),
    wkmeansjit.TB_NA: (_MISSING, "R's stale table of cluster sizes is shorter than the label"),
    wkmeansjit.DEAD_SOME: (_LIST, "a row's distances to every center are NaN"),
    wkmeansjit.DEAD_ALL: (_LENGTH, "every row's distances are NaN"),
}


def wkmeans(x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=291102, n_jobs=1):
    """Weighted k-means, as WCluster's Wkmeans computes it.

    Each start draws labels at random (every label on at least two rows), takes their weighted
    means as centers and assigns every row its nearest center, then runs up to max_iter rounds
    of 20 Lloyd steps followed by single-point moves, keeping a move that doesn't raise the
    weighted within-cluster sum of squares (WWCSS). The start with the smallest WWCSS wins, the
    first on ties. To get R's clusters, pass seed=RDraws(seed), R's set.seed(seed) draws. An int
    or a numpy Generator draws from numpy instead, which is reproducible but not R's.

    centers (k, p) replaces the random starts, and then num_init is 1. k, num_init and max_iter
    are floored like R's, and num_init or max_iter below 1 raises where R's 1:0 would run twice.
    labels keeps R's quirks: R's table of cluster sizes goes stale between moves and is read by
    position, so a cluster can empty and come back, and labels can have gaps. Where R stops,
    this raises ValueError with R's message. Where R would draw forever (fewer than 2k rows,
    fewer than k distinct (row, weight) pairs, or MAX_STARTS starts that never reach k clusters),
    this raises ValueError. n_jobs runs the starts' refinements on threads and never changes the
    result. Positional arguments stop after k, since R's order puts centers before the weights.
    """
    return _wkmeans(x, k, w=w, centers=centers, num_init=num_init, max_iter=max_iter, seed=seed,
                    n_jobs=n_jobs)


def _wkmeans(x, k, *, w, centers, num_init, max_iter, seed, n_jobs, trace=None):
    x, w, k, centers, num_init, max_iter = _prepare(x, k, w, centers, num_init, max_iter)
    jobs = check_n_jobs(n_jobs)
    draws = as_draws(seed)
    xf, wf = frozen(np.ascontiguousarray(x)), frozen(w)
    m11 = frozen(wstatsjit.row_norms(xf))
    zbuf = np.empty((x.shape[0], k))

    def assign(c):
        out = np.empty(x.shape[0], np.int64)
        wstatsjit.assign(xf, wf, frozen(np.ascontiguousarray(c)), c.shape[0], m11, zbuf, out)
        return out

    def start():
        return _init(x, w, k, centers, draws, assign, lambda y: wkmeansjit.centers_of(xf, wf, y, k))

    def refine(y):
        y = y.copy()
        status, u, it = wkmeansjit.refine(xf, wf, y, max_iter, m11)
        return status, u, it, y

    workers = min(num_init, (os.cpu_count() or 1) if jobs == -1 else jobs)
    results = []
    if workers <= 1:
        for _ in range(num_init):
            s = start()
            results.append((s, *refine(s)))
            if results[-1][1] != wkmeansjit.OK:
                break
    else:
        starts, init_error = [], None
        try:
            for _ in range(num_init):
                starts.append(start())
        except ValueError as e:
            init_error = e
        with ThreadPoolExecutor(workers) as pool:
            results = [(s, *r) for s, r in zip(starts, pool.map(refine, starts))]
    best = None
    for i, (s, status, u, it, labels) in enumerate(results):
        if trace is not None:
            trace.append((s, u, labels))
        if status != wkmeansjit.OK:
            raise _r_stop(*_STATUS[status])
        log.debug("Wkmeans start %d: %d iterations, WWCSS %r", i + 1, it, u)
        if best is None:
            best = (u, labels)
        else:
            _test(best[0], u)
            if best[0] > u:
                best = (u, labels)
    if workers > 1 and init_error is not None:
        raise init_error
    return _result(x, w, best[1])
```

The sequential branch is R's order exactly, which `ReplayDraws` needs: a start that stops R leaves later starts undrawn. The threaded branch draws every start first, then refines, then reports the first error in start order, R's error, since a start's own init error comes after every earlier start's refinement in R too. The trace records a stopped start's `(start, nan, labels)`, and `check` compares only the starts R traced.

- [ ] **Step 5: Run.** `NUMBA_CACHE_DIR=$SCRATCH/numba $UV run pytest tests/wcluster/test_wkmeans.py -q -n 8` then `-m slow`. Expected: all pass, and the quirk coverage test passes. If a quirk never shows up in the default fuzz, widen that family's seed range in `test_fuzz_reaches_every_quirk` until it does, never the check.

- [ ] **Step 6: Commit** `Compile Wkmeans with cached cluster sums and threaded starts`. Push.

---

### Task 9: The DN wrappers

**Files:**
- Create: `src/datanuggets/wcluster/dn.py`
- Test: `tests/wcluster/test_dn.py`

**Interfaces:**
- Consumes: `wkmeans`, `whclust`, `wpca`, `cluster_predict`, `DataNuggets`, `DnWkmeansResult`.
- Produces: `dn_wkmeans`, `dn_whclust`, `dn_wpca`, `dn_cluster_predict` with spec section 3's signatures.

- [ ] **Step 1: Failing tests.** `tests/wcluster/test_dn.py` builds the `DataNuggets` from R's table (`assignments` minus one) and checks each wrapper against the `dn_wcluster` fixture: `dn_wkmeans(dn, 3, seed=ReplayDraws(...))` labels, centers, wwcss, total and `row_labels` against R's "Cluster Assignments for original dataset" minus one, `dn_whclust(dn)` against R's merge, height and order, `dn_wpca(dn)` and `dn_wpca(dn, corr=True)` with `test_wpca.py`'s comparison helper (bit-for-bit fields plus the eigen rule), and `dn_cluster_predict` for the data frame and the datanugget `newx` (positions plus one equal R's). Then the errors:

```python
def test_dn_functions_need_a_datanuggets():
    for fn in (lambda: dn_whclust(np.zeros((4, 2))), lambda: dn_wpca(None),
               lambda: dn_wkmeans({}, 2), lambda: dn_cluster_predict([], [0], np.zeros((1, 2)))):
        with pytest.raises(TypeError, match='class "datanugget"'):
            fn()


def test_cluster_labels_must_cover_the_nuggets():
    with pytest.raises(ValueError, match="number of data nuggets"):
        dn_cluster_predict(DN, np.zeros(DN.n_nuggets - 1), DN.centers[:3])


def test_one_variable_nuggets_stop_where_r_stops():
    one = DataNuggets(DN.centers[:, :1], DN.weights, DN.scales, DN.assignments)
    with pytest.raises(ValueError, match="argument is of length zero"):
        dn_wkmeans(one, 2)
    with pytest.raises(ValueError, match="invalid 'times' argument"):
        dn_wpca(one)
    assert dn_wpca(one, wcol=[1.0]).sdev.shape == (1,)
    assert dn_whclust(one).merge.shape == (DN.n_nuggets - 1, 2)
    with pytest.raises(ValueError, match="argument is of length zero"):
        dn_cluster_predict(one, np.zeros(DN.n_nuggets), one)
```

- [ ] **Step 2: Run, expect `ImportError`.**

- [ ] **Step 3: Implement `dn.py`:**

```python
"""WCluster's DN.Wkmeans, DN.Whclust, DN.Wpca and DNcluster.predict: the WCluster functions on a
datanugget, its centers as the data and its weights as the row weights."""

from __future__ import annotations

import numpy as np

from datanuggets._validate import as_matrix
from datanuggets.result import DataNuggets, DnWkmeansResult, WhclustResult, WpcaResult
from datanuggets.wcluster.predict import cluster_predict
from datanuggets.wcluster.whclust import whclust
from datanuggets.wcluster.wkmeans import wkmeans
from datanuggets.wcluster.wpca import wpca

_ZERO = "argument is of length zero"


def _check(dn, name="dn") -> DataNuggets:
    if not isinstance(dn, DataNuggets):
        raise TypeError(f'{name} must be a DataNuggets, R\'s datanugget must be of class "datanugget", got {type(dn).__name__}')
    return dn


def _one_variable(dn, what):
    if dn.centers.shape[1] == 1:
        raise ValueError(f"{what} on a datanugget with one variable: R's [, 2:2] drops its table "
                         f"to a vector and R stops with \"{_ZERO}\"")


def dn_wkmeans(dn, k, *, centers=None, num_init=1, max_iter=10, seed=291102, n_jobs=1) -> DnWkmeansResult:
    """DN.Wkmeans: wkmeans on the nuggets, plus each original row's cluster, the one its nugget
    gets. Pass seed=RDraws(seed) to match R. R stops on a one-variable datanugget, and so does
    this."""
    _check(dn)
    _one_variable(dn, "DN.Wkmeans")
    res = wkmeans(dn.centers, k, w=dn.weights, centers=centers, num_init=num_init,
                  max_iter=max_iter, seed=seed, n_jobs=n_jobs)
    return DnWkmeansResult(res.labels, res.centers, res.wwcss, res.total_wwcss,
                           res.labels[dn.assignments])


def dn_whclust(dn) -> WhclustResult:
    """DN.Whclust: whclust on the nuggets with their weights."""
    _check(dn)
    return whclust(dn.centers, dn.weights)


def dn_wpca(dn, wcol=None, corr=False) -> WpcaResult:
    """DN.Wpca: wpca of the nuggets with their weights as row weights and wcol (ones by
    default) as column weights. R's default wcol stops on a one-variable datanugget."""
    _check(dn)
    if wcol is None:
        if dn.centers.shape[1] == 1:
            raise ValueError("DN.Wpca's default wcol on a datanugget with one variable: R's "
                             "rep(1, ncol(vector)) stops with \"invalid 'times' argument\"")
        wcol = np.ones(dn.centers.shape[1])
    return wpca(dn.centers, wrow=dn.weights, wcol=wcol, corr=corr)


def dn_cluster_predict(dn, cl, newx) -> np.ndarray:
    """DNcluster.predict: the cluster each row of newx is nearest to, with the centers the
    weighted means of the nuggets in each cluster of cl, as 0-based positions in sorted label
    order (cluster_predict). newx is a matrix, a data frame or a DataNuggets. R takes a
    datanugget newx's table columns sized by dn, so one with fewer variables passes its Weight
    and Scale columns as data, and this does the same."""
    _check(dn)
    if np.size(cl) != dn.n_nuggets:
        raise ValueError("The length of cluster assignments should be the same as the number of "
                         f"data nuggets. Got {np.size(cl)} for {dn.n_nuggets}")
    if isinstance(newx, DataNuggets):
        p = dn.centers.shape[1]
        table = np.column_stack([newx.centers, newx.weights, newx.scales])
        if p > table.shape[1]:
            raise ValueError("newx has fewer columns than dn's variables: R stops with "
                             "\"undefined columns selected\"")
        if p == 1:
            raise ValueError(f"a one-variable datanugget as newx drops to a vector, and R stops with \"{_ZERO}\"")
        newx = table[:, :p]
    return cluster_predict(dn.centers, cl, newx, w=dn.weights)
```

- [ ] **Step 4: Run, gates, commit** `Add the datanugget wrappers for the WCluster functions`. Push.

---

### Task 10: Exports and the architecture doc

**Files:**
- Modify: `src/datanuggets/wcluster/__init__.py`, `src/datanuggets/__init__.py`, `docs/architecture.md`, `r/README.md` if Task 2 didn't cover it
- Test: `tests/test_result.py` or the existing export test, whichever lists `__all__`

- [ ] **Step 1: Failing test.** The existing public-API test (grep `__all__` in `tests/`) gains `wkmeans`, `whclust`, `dn_wkmeans`, `dn_whclust`, `dn_wpca`, `dn_cluster_predict`, `WkmeansResult`, `DnWkmeansResult`, `WhclustResult`. If no such test exists, add `tests/test_exports.py` asserting each name is importable from `datanuggets` and listed in `__all__`.

- [ ] **Step 2: Implement.** Both `__init__` files import and list the six functions, and the package's `__init__` the three result types. `docs/architecture.md`: the module map gains `wcluster/{whclust,whclustjit,wkmeans,wkmeansjit,wstatsjit,dn}.py` with one line each, the dependency table lets `wcluster` import `_random`, the sentence "WCluster draws nothing" becomes "Wkmeans draws its starts through `_random`, the other WCluster functions draw nothing", and the tests map lists the three new test files.

- [ ] **Step 3: Run the full default suite** `NUMBA_CACHE_DIR=$SCRATCH/numba $UV run pytest -q -n 8`. Expected: all pass.

- [ ] **Step 4: Commit** `Export the WCluster iterative functions and map their modules`. Push.

---

### Task 11: Speed gates

**Files:**
- Create: `benchmarks/wcluster.R`, `benchmarks/wcluster.py`

- [ ] **Step 1: `benchmarks/wcluster.R`.** Usage `Rscript benchmarks/wcluster.R whclust200 | whclust400 | whclust800 | wkmeans`. Each workload writes `tmp/wcluster/out/<id>/` with the inputs, R's outputs, draws (for wkmeans) and `elapsed` in the manifest, using `fixture_io.R`'s `write_array`. `quakes[1:n, ]` with the weights 1, 3, 7, 2 repeating for Whclust at n = 200, 400, 800, and `quakes[1:n, ]` with unit weights for Wkmeans k = 4, seed 291102, at n = 100, 250, 500, 1000 (the session's `rtime.R` workloads, which gave the spec's R column). R runs single-threaded, under `benchmarks/pin.py`.

- [ ] **Step 2: `benchmarks/wcluster.py`.** For each R output folder: one untimed warm-up call on 10 rows, reported as first-call compile time on its own line, then the Python call timed (best of 3 for runs under a second, one run otherwise), its outputs compared bit for bit with R's (`same_values`), and a line `workload  R s  Python s  speedup  bits`. Then the workloads R doesn't run: `whclust` on the first 2,000 Covertype rows (10 columns, read from `tmp/covertype/covtype.data.gz` like `covertype.R`), `wkmeans` k = 4 on the first 10,000, and `dn_whclust` on the nugget table R's create.DN recorded in `tmp/covertype/out/covertype_100000_create`. It ends with a gate table, PASS or FAIL per row of spec section 10, and exits nonzero on any FAIL.

- [ ] **Step 3: Run in one pinned sitting.** `uv run --with psutil python benchmarks/pin.py "$RS" benchmarks/wcluster.R whclust200`, then `whclust400`, `wkmeans`, and `whclust800` once (about 22 minutes, in the background), then `uv run --with psutil python benchmarks/pin.py $UV run python benchmarks/wcluster.py`. Save the output in `$SCRATCH/port/bench.txt`.

- [ ] **Step 4: Gate.** Every row PASS and every comparison "yes". A FAIL is reported with a profile (`python -m cProfile` or numba timing of `whclustjit.whclust` and `wkmeansjit.refine`) and its target isn't loosened without RON2K. The obvious next step for `whclust` at n = 2,000 is the spec's non-goal, threads over the new cluster's distances, and that needs RON2K's yes.

- [ ] **Step 5: Commit** `Benchmark the WCluster ports against R` (both scripts) with the gate table in the body. Push.

---

### Task 12: Verify, review, ship

- [ ] **Step 1: Full suites.** On the laptop, `pytest -q -n 8` and `pytest -q -n 8 -m slow` on 3.14 and on 3.15 (`benchmarks/venv315` or `uv run --python 3.15`). On the Mac mini, `bash $HOME/.claude/tools/mini-run.sh` and the slow marker. Expected: all pass everywhere.

- [ ] **Step 2: Mutants (spec section 9.6).** Each in a scratch copy, run the wcluster tests, expect failures, revert:
  1. `whclustjit._scan` and the new-row update take `<=` instead of `<` (the last minimum).
  2. `wkmeansjit.refine` refreshes `tb` after the Lloyd loop too (a fresh table).
  3. The Lloyd shortcut breaks without leaving `yy0` at the fixed point (set `yy0[:] = -1` before `break`).
  4. `whclustjit.whclust` writes `merge` with its columns swapped.
  Record which tests caught each in the session log.

- [ ] **Step 3: Gates.** `ruff check src tests benchmarks` and `ruff format --check` on the changed files. Repo-root audit: nothing new at the root, `tmp/` untracked.

- [ ] **Step 4: Review.** One Opus code-reviewer pass over `git diff main...feat/wcluster-iterative`, with the spec and this plan as context and the Review Focus as its checklist. Each finding evaluated on merit (superpowers:receiving-code-review), fixes committed one per finding, the suite rerun.

- [ ] **Step 5: PR.** `gh pr create` against `main`, title `Port Wkmeans, Whclust and the DN wrappers`, body: what's ported, the parity evidence (fixture counts, fuzz sizes, quirk coverage), the gate table, the departures listed at the top of this plan for RON2K's ruling, and the test plan. Read the body before creating it: no attribution lines, no session URL. Don't merge.

- [ ] **Step 6: Session trio.** Tasklist done, notes with what surprised, log with gates and mutant results.
