# Feature 4: the six WCluster leaves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `wmean`, `wss`, `wwcss`, `distw`, `Wpca` and `cluster.predict` from R WCluster 1.3.0 so that the five cluster statistics match R bit for bit on every fixture, and `wpca` matches R bit for bit up to the matrix it hands to `eigen()` and to LAPACK rounding after it.

**Architecture:** Same oracle setup as Features 1 to 3: R scripts in `r/` write frozen fixtures and `pytest` never needs R. Two new replays in `_rstats.py` carry the arithmetic: `r_matmul` is `%*%` as reference BLAS dgemm sums it, and `r_weighted_mean` is `weighted.mean` with R's long double sums. `wcluster/wstats.py` builds `wmean`, `wss`, `wwcss` and `distw` on them, `wcluster/predict.py` replays `wdist_mat` and `which.min`, and `wcluster/wpca.py` replays `Wpca` up to `eigen()`, which goes through the existing `r_eigen_sym`. Three generators record R: `r/fixtures_rwstats.R` (the two helpers and `sum` boundary vectors), `r/fixtures_wstats.R` (one case feeds all five statistics) and `r/fixtures_wpca.R` (outputs plus the traced internals of `Wpca`).

**Tech Stack:** Python 3.12+ (dev on 3.14), numpy >= 2.0, scipy (LAPACK `dsyevr`), pytest, uv, ruff 0.16. R 4.6.1 with datanugget 1.5.0 and WCluster 1.3.0 (dev only).

**Spec:** `docs/specs/2026-09-29-datanuggets-python-port-design.md` (section 2 rows `wmean` to `cluster.predict`, section 4 conventions, section 5 harness and tiers, section 10 deviations table, section 12 advisor questions). The spec's section 8 build order is superseded by feature-by-feature execution. This is the rest of phase 2.

**Execution:** RON2K: "restarted what is next? and can we finish and test next feature how much remains till ful package is ported". Subagent-driven: one foundation task, then the statistics chain and the PCA chain side by side on disjoint files, each followed by a verifier that tries to break it. One multi-lens whole-branch review, one fix pass with a failing test first for each fix, push `feat/wcluster-leaves`, open a PR against `feat/speed`, do not merge.

## Global Constraints

- Branch `feat/wcluster-leaves` from `feat/speed` (4de33ae). The PR's base is `feat/speed`. `main` advances only by PR, and merging is RON2K's call.
- Public signatures, all exported from `datanuggets` and from `datanuggets.wcluster`:
  - `wmean(x, cl, w=None) -> np.ndarray`, shape `(k, p)` float64. Row `i` is the weighted mean of cluster `i`. R returns the transpose, `p` by `k`.
  - `wss(x, w=None) -> float`.
  - `wwcss(x, cl, w=None, group_sum=False) -> float | WwcssResult`. `WwcssResult(per_cluster, total)`: `per_cluster` is `(k,)` float64 in sorted label order, `total` is a 0-d float64.
  - `distw(x, cl, w=None) -> np.ndarray`, shape `(k, k)` float64. The strict lower triangle holds R's values. The diagonal and the upper triangle are NaN, where R has NA.
  - `cluster_predict(x, cl, newx, *, w=None) -> np.ndarray`, shape `(m,)` intp: each new row's 0-based position among the sorted distinct labels. `w` is keyword-only, because R's order is `(x, w, cl, newx)` and a call in that order has to fail loudly.
  - `wpca(x, wrow=None, wcol=None, corr=False) -> WpcaResult`. `WpcaResult(sdev, rotation, x, center, scale, wrow, wcol)`, the field names and order of spec section 4. `x` holds the scores. `scale` is `(p,)` float64: R's `dd` when `corr=True`, ones when `corr=False` (R returns the single number 1). `wrow` is the weights as float64. `wcol` is normalised to sum to `p`, as R returns it.
- Inputs: `x` and `newx` go through `as_matrix` (2-D, numeric, finite, converted to C-contiguous float64). A weight vector is 1-D numeric of the right length with no NaN or infinity. `None` means ones. Negative and zero weights pass in the four statistics and `cluster_predict`, as in R. `wpca` rejects a negative `wrow`. Labels are 1-D numeric whole numbers smaller than 1e15 in size, in an integer or float dtype. Text, bool, fractional, NaN and infinite labels raise. Clusters are ordered by `np.unique`, which is R's order for every label vector this rule accepts.
- Integer input is converted to float64 before any arithmetic. In R an integer `x` times an integer `w` is integer arithmetic, and what overflow does depends on the function: `wmean` and `cluster.predict` get an NA center with a warning, `Wpca` turns that NA into 0 in its column sum and returns a wrong center with a warning, and `wss`, `wwcss` and `distw` never multiply integers (`%*%` converts to double first). Python never overflows there. Listed as a deviation.
- A cluster whose weights sum to zero keeps R's result: a NaN center, a NaN `wss`, a NaN `wwcss` total, NaN in every `distw` pair that touches it. `cluster_predict` never predicts it. No numpy `RuntimeWarning` reaches the caller: every division or square root that can produce NaN or infinity runs inside `np.errstate`.
- R's `warning()`s become `DataNuggetWarning` with R's text. The only one here is `"NaNs produced"` from `wpca`, raised once when an eigenvalue is negative.
- Arithmetic rules for everything under `src/`: no `@`, `.sum()`, `np.dot`, `einsum`, `np.average`, `np.cumsum`, `math.fsum` or `**` on these paths. Squares are `a * a`. Sums that R takes with `sum()` or `weighted.mean` go through `r_sum`. Sums that R takes with `%*%` go through `r_matvec` or `r_matmul`. No compiled kernels in this feature (spec section 9: optimise after parity is green).
- Parity tiers:
  - Exact for the six public functions means `np.testing.assert_array_equal` (NaN equals NaN). It applies to `wmean`, `wss`, `wwcss`, `distw`, `cluster_predict` (the positions and the distance matrix `z`), and to `wpca`'s `center`, `scale`, `wcol`, `wrow`, centered data and eigen input matrix.
  - Exact for `r_matmul` and `r_weighted_mean` means `same_bits` from `tests/rfixtures.py`, as for every other `_rstats` replay.
  - Eigen tier for `wpca`'s `sdev`, `rotation` and `x`: the residual and subspace rule in Task 5, with floor 1e-10, C = 256, K = 1024 and tie threshold 1e-12. No eigen output is ever compared bit for bit, and rotation signs are unspecified.
- Real data only: every fixture comes from R run on R's built-in datasets or on R's own nugget outputs read back from `tests/fixtures/create_dn`. Built points (tie midpoints, ulp nudges, overflowing rows) and built weights (powers of two) exist only in the group `wstats_constructed` and the helper group `rcompat_wstats`, which are tested and never reported.
- No vacuous passes: a parametrized test over an empty case list is a skip, not a failure. Each test module has a coverage test that asserts every case list it parametrizes over is non-empty and that the fixtures hold the property each behaviour test relies on (a tie, a dead cluster, an asymmetric eigen input). A behaviour test that relies on a fixture property asserts that property first.
- Tests never reach the module `datanuggets.wcluster.wpca` through attribute access or `import ... as`: once Task 6 exports the function `wpca` from `datanuggets.wcluster`, that attribute is the function and shadows the module. A test that needs the module uses `importlib.import_module("datanuggets.wcluster.wpca")`. The same holds for `wkmeans` and `whclust` in Phase 3.
- Fixtures: raw `.bin` plus `manifest.json` via `r/fixture_io.R`, recorded only on the laptop's R 4.6.1 (x87 long double, reference BLAS, `matprod` "default"). Never on the Mac mini. Total under 25 MB allocated (12 MB before this feature, 6.1 MB of content). This feature may add up to 8 MB allocated.
- Machines: R and the targeted tests run on the laptop. The full suite runs on the Mac mini through `mini-run.sh`, one run at a time. A bitwise test of a NaN result takes its NaN from a constant, never from the machine's own division.
- Never run `r/setup.R` during this feature: it reinstalls the pinned packages from CRAN and could replace them. Its edit is checked by parsing it and by running its assertions against the library in place.
- Commits: one logical change each, imperative subject <= 70 chars plus a why/what body, written to a message file in the scratchpad and committed with `git commit -F`. No `Claude-Session:`, `Co-Authored-By` or "Generated with" lines, and no mention of AI tools. Stage files by name, never `git add -A`. `git checkout --` any `docs/validation/*.md` that shows modified with no content change.
- Executors never run `git add` or `git commit`. The two chains share one working tree, so a commit by either would sweep in the other's staged files. Each task ends with its file list and the commit subjects it names, and the lead commits them after the task's verifier passes.
- Gates name only the task's own files (`ruff check`, `ruff format --check`, `pytest`, `git status --short`), for the same reason. The whole-repo gates run once, in Task 6.
- Repo root holds only toolchain files. Code in `src/datanuggets/`, tests flat in `tests/`, R in `r/`, scripts in `scripts/`, docs in `docs/`.
- Shell: the repo root, whose path has spaces, so quote it. `UV=/c/Users/thegr/AppData/Local/hermes/bin/uv.exe`, `RS="$LOCALAPPDATA/Programs/R/R-4.6.1/bin/Rscript.exe"`. Run R on script files only, never `-e`, and bare (no `R_LIBS_USER`): `r/fixture_io.R` finds the library itself. Set `NUMBA_CACHE_DIR` to a scratch folder for any Python run.
- Published prose (README, docstrings, report headers) follows the humanizer rules: no em or en dashes, no semicolons, contractions, no bold or italic in prose, no comma-pause chains, no test counts.

## Review Focus

1. A cluster whose weights sum to zero, first (`mtcars_gear_am`), in the middle (`mtcars_carb_am`) or last (`mtcars_cyl_vs`): a person expects NaN for that cluster and correct numbers for the others, with no warning and no `ZeroDivisionError`, and expects `cluster_predict` never to return it. Tests `test_zero_weight_cluster_is_nan_and_quiet` (Task 3), `test_dead_cluster_is_never_predicted` and `test_every_cluster_dead_raises` (Task 4).
2. Labels that aren't `0..k-1`: 4/6/8, negative labels, whole numbers stored as floats. A person expects clusters in sorted order and positions back from `cluster_predict`. Text, bool, fractional, NaN and huge labels must raise instead of being grouped some other way than R groups them. Tests `test_labels_need_not_start_at_zero`, `test_label_rules` (Task 1 and Task 3).
3. A call written in R's argument order or with the wrong shapes: `cluster_predict(x, w, cl, newx)`, a weight per column instead of per row, `newx` missing or with another column count, 1-D `x`, a new row whose squares overflow. A person expects an error that names the argument or the cause. Tests `test_r_argument_order_fails`, `test_weight_rules`, `test_newx_rules`, `test_newx_must_have_x_columns`, `test_overflow_rows_match_r` (Tasks 1, 3, 4).
4. `wpca` weights that make R's covariance meaningless: a sum of exactly 1 (`iris$Petal.Width[1:5]` sums to exactly 1), a sum below 1, a negative row weight, a zero total, a column with no variance under `corr=True`, one row. A person expects an error that names the cause, or R's NaN `sdev` with R's one warning where R returns. Tests `test_weights_summing_to_one_raise`, `test_weights_summing_below_one_warn_once`, `test_weight_rules`, `test_constant_column_with_corr_raises`, `test_one_row` (Task 5).
5. Eigen output on another machine or LAPACK build: a person expects `wpca` to pass on the Mac mini and to fail when the rotation and the scores disagree. The rule's own tests feed it wrong outputs and require a failure. Tests `test_eigen_rule_rejects_wrong_outputs`, `test_sdev_comes_from_the_vectors_call` (Task 5).

## File Structure

| Path | Task | Responsibility |
|---|---|---|
| `r/setup.R`, `r/fixture_io.R` | 1 | pin WCluster 1.3.0, record its version and `matprod` in every new manifest |
| `r/fixtures_rwstats.R`, `tests/fixtures/rcompat_wstats/` | 1 | R's `%*%` of two matrices, `weighted.mean`, and `sum` boundary vectors |
| `src/datanuggets/_rstats.py` | 1 | `r_matmul`, `r_weighted_mean` |
| `src/datanuggets/_validate.py` | 1 | `as_weights`, `as_labels` |
| `src/datanuggets/result.py` | 1 | `WwcssResult`, `WpcaResult` |
| `src/datanuggets/wcluster/__init__.py` | 1, 6 | package marker in Task 1, exports in Task 6 |
| `tests/test_rstats.py`, `tests/test_validate.py`, `tests/test_result.py` | 1 | helper, validator and result tests |
| `r/fixtures_wstats.R`, `tests/fixtures/wstats/`, `tests/fixtures/wstats_constructed/` | 2 | the statistics oracle |
| `src/datanuggets/wcluster/wstats.py`, `tests/test_wstats.py` | 3 | `wmean`, `wss`, `wwcss`, `distw` |
| `src/datanuggets/wcluster/predict.py`, `tests/test_predict.py` | 4 | `cluster_predict`, `_wdist_mat` |
| `r/fixtures_wpca.R`, `tests/fixtures/wpca/` | 5 | the PCA oracle |
| `src/datanuggets/wcluster/wpca.py`, `tests/test_wpca.py`, `tests/rfixtures.py` | 5 | `wpca`, and the eigen rule next to `RTOL` and `ATOL` |
| `src/datanuggets/__init__.py`, `scripts/parity_report.py`, `docs/validation/*.md` | 6 | exports and six parity reports |
| `README.md`, `r/README.md`, the spec | 7 | docs, tiers, deviations, advisor questions |

Tasks 2 to 4 (statistics) and Task 5 (PCA) touch disjoint files after Task 1 and can run side by side. `tests/rfixtures.py` belongs to Task 5 alone. Tasks 3 and 4 keep their helpers inside their own test modules.

## R's stops, and what each becomes

| R stop | Kind | Python |
|---|---|---|
| weights length (`wmean`, `wss`, `wwcss`, `distw`, `distwij`, `cluster.predict`, `Wpca` row and column) | argument check | `ValueError` from `as_weights`, one test per function and argument |
| assignments length | argument check | `ValueError` from `as_labels` |
| `wrow`/`wcol` must be of class "numeric" or "table" | argument check | `TypeError` from `as_weights` for a non-numeric dtype |
| `newdata should be a data frame` (a NULL `newx`), `newdata should have the same variables` | argument check | `as_matrix(newx, "newx")` raises `TypeError` on `None`, and a `ValueError` on another column count |
| NA label (`NA/NaN argument`), one cluster in `distw` (`missing value where TRUE/FALSE needed`), factor labels in `distw` | accidental crash | explicit `ValueError`/`TypeError` naming the cause, R's behaviour in a test comment |
| `eigen`: `infinite or missing values in 'x'` (weights summing to 1, zero total, negative `wrow`, zero variance under `corr`) | accidental crash | explicit `ValueError` naming the cause, before `eigen` |
| `distwij`: "The cluster index i or j should be smaller than the number of clusters." | unreachable from `distw` | not ported: `_distwij` is private and gets valid positions |
| default `w` of the wrong length (`wmean`, `wwcss`, `distw`), `w = NULL` | upstream bug | `w=None` means ones |
| `Wpca` with one column and `corr = TRUE` outside `0.5 < dd <= 1` | upstream bug (`diag()` of one number builds an identity) | the general formula, so `sdev` is 1 to rounding |

Before writing the tests that mirror a stop, print the R function (`WCluster::distw`, `WCluster:::distwij`, `WCluster::cluster.predict`) from a script file and confirm the check is there.

---

### Task 1: harness, replays, validators, results

**Files:**
- Modify: `r/setup.R` (add `"WCluster"` to the install vector and `stopifnot(packageVersion("WCluster", lib.loc = lib) == "1.3.0")`)
- Modify: `r/fixture_io.R` (assert WCluster 1.3.0 without attaching it, add `WCluster` and `matprod` to `versions()`)
- Create: `r/fixtures_rwstats.R`
- Create: `tests/fixtures/rcompat_wstats/` (generated)
- Modify: `src/datanuggets/_rstats.py` (`r_matmul`, `r_weighted_mean`, module docstring)
- Modify: `src/datanuggets/_validate.py` (`as_weights`, `as_labels`)
- Modify: `src/datanuggets/result.py` (`WwcssResult`, `WpcaResult`)
- Create: `src/datanuggets/wcluster/__init__.py` (docstring only)
- Modify: `tests/test_rstats.py`, `tests/test_result.py`
- Create: `tests/test_validate.py`

**Interfaces:**
- Consumes: `r_sum(v) -> float`, `r_matvec(a, v) -> np.ndarray` (`_rstats`), `_INDEFINITE` (`_rcompat`), `_ArrayFields` (`result`), `same_bits`, `hex_bits`, `load_group` (`tests/rfixtures.py`), `write_case`, `as_dbl`, `from_bits`, `no_args` (`r/fixture_io.R`).
- Produces: `r_matmul(a, b) -> np.ndarray`, `r_weighted_mean(v, w) -> float`, `as_weights(w, n, name="w", per="row") -> np.ndarray`, `as_labels(cl, n, name="cl") -> tuple[np.ndarray, np.ndarray]` (sorted levels, intp codes), `WwcssResult`, `WpcaResult`, fixture group `rcompat_wstats` with `args.kind` in `matmul`, `wmean`, `sum`, and a `versions` block that carries `WCluster` and `matprod`.

- [ ] **Step 1: Baseline.** The suite at 4de33ae on the Mac mini, before any change. Expected: all pass. Record the count in the ledger.

- [ ] **Step 2: `r/setup.R` and `r/fixture_io.R`.** In `r/fixture_io.R`, after the datanugget assertion add `stopifnot(packageVersion("WCluster") == "1.3.0")`. Don't attach WCluster: the generators call `WCluster::` and `WCluster:::`. In `versions()` add `WCluster = as.character(packageVersion("WCluster"))` and `matprod = getOption("matprod")`. Existing fixture groups keep their manifests until someone regenerates them. Check: a scratch script that sources `r/fixture_io.R` from the repo root and prints `str(versions())`. Expected: nine keys, `WCluster` "1.3.0", `matprod` "default". Check `r/setup.R` with `parse()` from a scratch script, never by running it.

- [ ] **Step 3: `r/fixtures_rwstats.R`.** Three kinds of case in group `rcompat_wstats`:
  - `matmul_*` (`args = list(kind = "matmul")`, arrays `a`, `b`, `prod`): `t(x) %*% x` for `iris[, 1:4]` and `scale(volcano)`. `x %*% centers` for quakes with R's `wmean` centers by depth band. A row vector times a matrix and a matrix times a column, from faithful. `mtcars %*% wmean(mtcars, gear, am)`, whose first column is NaN. One built case where every product in a column is `-0` (a column of `-0` times positive numbers), which R sums to `+0`.
  - `wmean_*` (`kind = "wmean"`, arrays `v`, `w`, `mean`): `weighted.mean(v, w)` for iris Sepal.Length by Petal.Width, mtcars mpg by vs (zeros inside), mtcars mpg of the gear 3 rows by am (every weight zero, so NaN), quakes depth by stations, a nugget center column by nugget weights, and one built pair of weights `c(1, -1)` that sum to zero with a nonzero numerator (R gives an infinity).
  - `sum_*` (`kind = "sum"`, arrays `v`, `sum`): the boundary vectors the research recorded for R's long double `sum`, which pin its order dependence and the mapping at `DBL_MAX`.
  Run from the repo root: `"$RS" r/fixtures_rwstats.R`. Expected: the case count printed, and `git status --short tests/fixtures` shows only `rcompat_wstats/`.

- [ ] **Step 4: Failing tests in `tests/test_rstats.py`.** The module gains `WCASES = load_group("rcompat_wstats")` beside its existing group, and each new test filters `WCASES` by `args["kind"]`. `test_rcompat_wstats_covers_each_kind` (each of `matmul`, `wmean`, `sum` has cases, and the NaN matmul case, the `-0` matmul case and the zero-total weighted mean case are present by id). `test_r_matmul_matches_r` (every `matmul` case, the inputs in C order and in Fortran order, `same_bits` on the finite cases and NaN-aware equality on the NaN case), `test_r_matmul_never_returns_negative_zero`, `test_r_weighted_mean_matches_r` (`same_bits`, including the NaN case, whose bits are the x86 default NaN), `test_r_sum_boundaries` (`same_bits`). Update line 1 of the module docstring. Run: `$UV run pytest tests/test_rstats.py -q`. Expected: the new tests fail with `ImportError`.

- [ ] **Step 5: The two replays in `_rstats.py`.**

```python
_R_NAN = float(np.array([_INDEFINITE], dtype=np.uint64).view(np.float64)[0])


def r_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a %*% b as R computes it with reference BLAS dgemm: c = 0, then c += a[:, k] * b[k, :]
    for k in order, every product and sum rounded to double. A sum starts at +0, so no entry
    comes back as -0."""
    c = np.zeros((a.shape[0], b.shape[1]))
    for k in range(a.shape[1]):
        c += np.multiply.outer(a[:, k], b[k, :])
    return c


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
```

  Import `_INDEFINITE` from `_rcompat`. Add `%*%` of two matrices and `weighted.mean` to the module docstring's list. Run: `$UV run pytest tests/test_rstats.py -q`. Expected: all pass.

- [ ] **Step 6: Failing tests in `tests/test_validate.py`.** `test_weight_rules`: `None` gives ones of length n, a list and an integer array come back float64, a wrong length raises `ValueError` naming the argument and the expected count, 2-D raises `ValueError`, text and bool raise `TypeError`, NaN and infinity raise `ValueError`, negative and zero weights pass, `per="column"` changes the message. `test_label_rules`: levels are sorted and codes are 0-based positions for `[8, 4, 6, 4]`, for negative labels and for whole floats. `-0.0` and `0.0` are one label. Text and bool raise `TypeError`. A fractional, NaN or infinite label raises `ValueError`. A label of `1e15` raises `ValueError`. A wrong length raises `ValueError`. No labels at all raises `ValueError`. Each test comments what R does with the same input.

- [ ] **Step 7: The validators in `_validate.py`.**

```python
_LABEL_LIMIT = 1e15  # R's factor() tells labels apart by 15 significant digits


def as_weights(w, n: int, name: str = "w", per: str = "row") -> np.ndarray:
    """One weight per row (or column) as float64. None means ones: R's own default,
    rep(1, length(x)), is one weight per column in wmean and wwcss, which stops unless x is
    square, and one per cell in distw, which stops unless x has one column."""
    if w is None:
        return np.ones(n)
    arr = np.asarray(w)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric")
    if arr.shape != (n,):
        raise ValueError(f"{name} must have one weight per {per} ({n}), got shape {arr.shape}")
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_labels(cl, n: int, name: str = "cl") -> tuple[np.ndarray, np.ndarray]:
    """The sorted distinct labels and each row's 0-based position among them. Labels are whole
    numbers smaller than 1e15 in size: the labels R's factor() and sort(unique()) group and
    order alike, which is how wwcss and distw read them."""
    arr = np.asarray(cl)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold numeric cluster labels")
    if arr.shape != (n,):
        raise ValueError(f"{name} must have one label per row ({n}), got shape {arr.shape}")
    if n == 0:
        raise ValueError(f"{name} is empty: there are no rows to cluster")
    size = np.abs(arr.astype(np.float64))
    if not np.isfinite(size).all() or not np.array_equal(size, np.floor(size)):
        raise ValueError(f"{name} must hold whole numbers")
    if size.max() >= _LABEL_LIMIT:
        raise ValueError(f"{name} must hold labels smaller than 1e15 in size")
    levels, codes = np.unique(arr, return_inverse=True)
    return levels, codes.reshape(-1).astype(np.intp)
```

  Run: `$UV run pytest tests/test_validate.py -q`. Expected: all pass.

- [ ] **Step 8: Results.** In `result.py`, two `_ArrayFields` dataclasses: `WwcssResult` with `per_cluster` and `total`, and `WpcaResult` with `sdev`, `rotation`, `x`, `center`, `scale`, `wrow`, `wcol` in that order. Docstrings say what each holds and that `WpcaResult.x` is the scores, named as R names it. Tests in `tests/test_result.py`: fields are read-only copies, `==` compares by value with NaN equal to NaN, the field order is the spec's.

- [ ] **Step 9: Gates and commits.** `$UV run pytest tests/test_rstats.py tests/test_validate.py tests/test_result.py -q`, then `ruff check` and `ruff format --check` on exactly the Python files this task touched. Task 1 runs alone, so `git status --short` must list only this task's files. The lead makes three commits: `Pin WCluster 1.3.0 and record matprod in fixture manifests` (the two R files), `Replay R's matrix product and weighted.mean bit for bit` (generator, fixtures, `_rstats.py`, `tests/test_rstats.py`), `Add weight and label checks and the wwcss and wpca results` (`_validate.py`, `result.py`, `wcluster/__init__.py`, their tests).

### Task 2: the statistics oracle

**Files:**
- Create: `r/fixtures_wstats.R`
- Create: `tests/fixtures/wstats/`, `tests/fixtures/wstats_constructed/` (generated)

**Interfaces:**
- Consumes: `write_case`, `as_dbl`, `read_array`, `no_args`, `versions` (`r/fixture_io.R`), WCluster's `wmean`, `wss`, `wwcss`, `distw`, `cluster.predict` and the internal `wdist_mat`.
- Produces: per case the arrays `x` `(n, p)`, `cl` `(n,)`, `w` `(n,)`, `wss` (one number), `wwcss_group` `(k,)`, `wwcss_total` (one number), `wmean` `(p, k)` as R returns it, `distw` `(k, k)` when `k >= 2`, and, where the case predicts, `newx` `(m, p)`, `z` `(m, k)` and `pred` `(m,)` 1-based.

- [ ] **Step 1: The generator.** One function records a case: it calls the five R functions on the same `x`, `cl`, `w`, `newx`, gets `z` from `WCluster:::wdist_mat(newx, wmean(x, cl, w), rep(1, nrow(newx)))`, asserts that `apply(z, 1, which.min)` equals what `cluster.predict` returned, and writes the arrays. Every `cl` is written on purpose with `as.integer()` (dtype `<i4`), except the cases marked double, which use `as.double()`. R gets the same vector the fixture stores, never a factor. Data is stored with `as_dbl`, except the two integer-typed cases, which store `<i4` and pass R an integer matrix and integer weights. A case goes in `wstats` only when its data, labels, weights and `newx` are all real. Anything with built points or built weights goes in `wstats_constructed`. The helpers the research used, copied verbatim (`f4v/ss/e06_dump.R:5`, `f4v/centers/e6_dump.R:57-59`):

```r
ward <- function(m, k) cutree(hclust(dist(m), "ward.D2"), k)
bands <- cut(quakes$depth, c(0, 150, 300, 450, 700), labels = FALSE)
```

  Nugget inputs are read back from the repo's own `tests/fixtures/create_dn` cases with `read_array`, as `f4v/ss/e06_dump.R:6-8` does with its `read_fix`. When `cluster.predict` returns a list instead of an integer vector (a row with no finite distance), the case records `z`, leaves out `pred` and sets `args$pred = "list"`.

- [ ] **Step 2: The `wstats` cases.** `newx` is `x` unless the row says otherwise.

| Case | Definition | Pins |
|---|---|---|
| `iris_species_ones` | iris 1:4, Species, unit weights | baseline |
| `iris_species_petalwidth` | iris 1:4, Species, `w` = Petal.Width | non-integer weights, the long double denominator |
| `mtcars_cyl_carb` | mtcars, `cl` = cyl as double, `w` = carb | labels 4/6/8, float labels, p = 11 |
| `mtcars_gear_vs` | mtcars, gear, `w` = vs | zero weights inside clusters |
| `mtcars_gear_am` | mtcars, gear, `w` = am | a dead cluster first: NaN center, NaN group, NaN pairs, never predicted |
| `mtcars_carb_am` | mtcars, carb, `w` = am | a dead cluster in the middle (the third of six) |
| `mtcars_cyl_vs` | mtcars, cyl, `w` = vs | a dead cluster last |
| `chick_diettime_time` | ChickWeight weight and Time, `cl` = the codes of `interaction(Diet, Time)`, `w` = Time | k = 48, four dead clusters, tie rows |
| `quakes_depth_stations` | quakes, depth bands at 0/150/300/450/700, `w` = stations | n = 1000, k = 4, count weights |
| `faithful_waiting` | `faithful, (faithful$waiting > 70) + 1, rep(1, 272)` | k = 2, unit weights |
| `faithful1d` | `faithful[, 1, drop = FALSE], (faithful$waiting > 70) + 1, rep(1, 272)` | p = 1 |
| `faithful1d_by_waiting`, `infert_stratum_parity` | as the research recorded them | `which.min(distw)` moves under wrong variants |
| `nug_quakes` | nugget centers and weights from `create_dn/quakes`, Ward labels, `newx` = raw quakes | the DNcluster.predict path |
| `nug_iris_passthrough` | nuggets from `create_dn/iris_all_passthrough` | a zero-weight nugget |
| `nug_volcano_scaled` | nuggets of `scale(volcano)`, `newx` = raw rows | p = 61 |
| `volcano_ward4` | `volcano, ward(volcano, 4), rep(1, 87)` | whole-number data, p = 61 |
| `iris50_singletons` | `iris[1:50, 1:4]`, `cl` = 1:50, `w` = Petal.Width | singleton `wss` isn't zero, negative `z`, the long double total |
| `iris150_singletons_pw`, `iris_singletons_neg_pw` | every iris row its own cluster, labels `1:150` and `-(1:150)` | duplicate rows tie, negative labels |
| `iris3_singletons_pw`, `iris12_singletons_pl`, `iris1_singletons_pw`, `iris124_singletons_pl` | singleton cases from the research | zero and negative `distw` entries |
| `quakes200_singletons_st` | 200 quakes rows as singletons, `w` = stations | `x * x` against a C library `pow` |
| `mtcars_singletons_vs`, `nug_iris_passthrough_singletons` | singletons with zero weights | NaN pairs, zero-weight rows kept |
| `iris_one_cluster`, `mtcars_one_carb`, `volcano_one_ones`, `volcano_scaled_one_ones` | one cluster | k = 1, `m22` summed in R's order, cancellation in the centers |
| `mtcars_cyl_carb_row15`, `iris_species_pw_row150`, `mtcars_gear_am_row01` | one new row | `m11` on a single row, a NaN center on one row |
| `mtcars_cyl1d_gear` | mtcars cyl as one column, `newx` = gear | real exact ties between centers 4 and 6 |
| `iris_species_petallength_k3`, `mtcars_cyl_qsec_k3`, `usarrests_regions123_pop_k3`, `plantgrowth_group_k3` | k = 3 | the long double `wwcss` total differs from a double sum by one ulp |
| `eustock_dax_low88_k2` | EuStockMarkets, `w` = DAX, cluster 1 = the 88 rows with the smallest DAX | a long double sum of two terms inside `distw` |
| `mtcars_carb_wt` | mtcars, carb, `w` = wt | k = 6, catches the most wrong variants |
| `faithful_ward10_erupt`, `rock_perm_shape`, `lcs_ward5_dpi` | Ward or real labels, non-integer weights | the long double denominator |
| `quakes_ward100_stations`, `volcano_ward30_col1`, `usarrests_scaled_ward2_col1` | many clusters, standardized data | k = 100, k = 30 with p = 61 |
| `quakes_int2_mag5`, `quakes_int_by_mag` | integer-typed quakes depth and stations, integer weights | integer storage |

  The exact R expression for each case the research recorded is in the research scripts the evidence index lists: `f4v/ss/e06_dump.R` (the note's cases, the k = 2 and k = 3 cases), `f4v/ss/cases.R` (singletons, negative labels, `mtcars_gear_am` at line 56), `f4v/centers/e6_dump.R` and `e8_dump2.R` (one-row cases through the loops at `e6_dump.R:115-116`, `mtcars_cyl1d_gear` at line 90, Ward and nugget cases), and `f4v/centers/e12_sqrt_search.R:70` for the scaled Ward cases, where `x = scale(D)`, `cl = ward(scale(D), K)` and `w` is column 1 of `D` or ones. Copy the expression, not the numbers. A case whose expression can't be found is left out and named in the task's report, never invented.

- [ ] **Step 3: The `wstats_constructed` cases.** `ties_iris`, `ties_mtcars`, `ties_quakes`, `ties_volcano`, `ties_faithful`: the midpoints of every pair of R's weighted centers, plus each midpoint nudged by one and two ulp, recording `wmean`, `z` and `pred`. `swiss_scaled_ward4_col1`, `mtcars_scaled_ward6_col1`, `lcs_scaled_ward2_ones`: midpoints and nudges on standardized data. `hv_quakes_by_mag_w2pow`, `hv_chick_by_chick_w2pow`, `hv_quakes_by_stations_w2pow`, `hv_eustock_year_wdax_1e296`, `hv_faithful_by_waiting_wsubnormal` (`f4v/ss/e14_heavy.R:9-35`): real data with built weights whose ratios reach 5e36. These record the four statistics only, no `newx`. Three overflow cases from `f4lead/B/r1/ovf.R`, recording `wmean`, `z` and `pred` (or `args$pred = "list"`):
  - `ovf_one_row`: `x = rbind(c(1, 2), c(3, 4), c(1e200, 0))`, `cl = c(1, 1, 2)`, unit weights, `newx = rbind(c(1, 2))`. R's `z` is `[2, Inf]` and it predicts 1: a lone Inf survives R's identity multiply.
  - `ovf_two_rows`: the same `x`, `newx = rbind(c(1, 2), c(2, 2))`. The Inf column turns NaN through `0 * Inf` from the other row.
  - `ovf_newx_row`: `x = rbind(c(1, 2), c(3, 4), c(5, 6))`, `cl = c(1, 2, 2)`, unit weights, `newx = rbind(c(1, 2), c(1e200, 0), c(4, 4))`. R's `z` is `[[NaN, NaN], [Inf, Inf], [NaN, NaN]]` and `cluster.predict` returns a list.
  - one case whose `z` row is NaN in its first column and `Inf` in the others, where `which.min` returns the first live column, not the first column. Build it from `ovf.R`'s pattern (a dead first cluster plus an overflowing new row) and check R's answer when recording it.

- [ ] **Step 4: Record and check.** `"$RS" r/fixtures_wstats.R` from the repo root. Expected: both groups written, every manifest's `versions` carries `WCluster` 1.3.0 and `matprod` default. Stage nothing. Copy both group folders to a scratch folder, run the generator a second time, and compare every `.bin` byte for byte with the copy. Expected: identical (the recording is deterministic). Print the allocated and the content size of `tests/fixtures/wstats` and `tests/fixtures/wstats_constructed`. The joint budget check across all new groups happens in Task 6.

- [ ] **Step 5: Commit (lead).** `Record WCluster's cluster statistics on real data` (the generator and both groups).

### Task 3: wmean, wss, wwcss, distw

**Files:**
- Create: `src/datanuggets/wcluster/wstats.py`
- Create: `tests/test_wstats.py`

**Interfaces:**
- Consumes: `r_sum`, `r_matvec`, `r_weighted_mean` (`_rstats`), `as_matrix`, `as_weights`, `as_labels` (`_validate`), `WwcssResult` (`result`), fixture groups `wstats` and `wstats_constructed`.
- Produces: `wmean`, `wss`, `wwcss`, `distw`, and the private `_wss(x, w) -> float`, `_group_wss(x, codes, k, w) -> np.ndarray`, `_distwij(x, codes, w, i, j, alone) -> float`.

- [ ] **Step 1: Failing parity tests.** `test_wmean_matches_r` (asserts R's shape `(p, k)`, then compares with the transpose), `test_wss_matches_r`, `test_wwcss_matches_r` (the float, and `WwcssResult` with `group_sum=True`), `test_distw_matches_r` (every case with `k >= 2`, lower triangle Exact, NaN on the diagonal and above). Each is parametrized over every case of both groups that has the array, by case id, and runs inside `warnings.catch_warnings()` with `simplefilter("error")` so a stray numpy warning fails. Each passes `x` once C-ordered and once Fortran-ordered. `test_fixtures_cover_the_edge_cases`: every parametrized list is non-empty, and across `wstats` there is a case with a `<i4` `x`, a `<f8` `cl`, k = 1, p = 1, one new row, a zero-weight cluster first, in the middle and last, negative labels, a negative entry in R's `z` and an exact tie in R's `z`. Tests import from `datanuggets.wcluster.wstats` (Task 6 adds the package exports). Run: `$UV run pytest tests/test_wstats.py -q`. Expected: `ImportError`.

- [ ] **Step 2: Failing behaviour tests.** `test_zero_weight_cluster_is_nan_and_quiet` (on `mtcars_gear_am`, `mtcars_carb_am` and `mtcars_cyl_vs`: no warning, no exception, NaN exactly where R has it, and first asserts that R's `wmean` has a NaN row in that position). `test_labels_need_not_start_at_zero` (labels 4/6/8 as ints and as floats give the same arrays as labels 0/1/2). `test_default_weights_are_ones` (`w=None` equals `np.ones(n)`. R's default `rep(1, length(x))` stops unless `x` is square in `wmean` and `wwcss`, and unless it has one column in `distw`. `wss` defaults to `rep(1, nrow(x))`, which works). `test_distw_needs_two_clusters` (`ValueError`. R stops with "missing value where TRUE/FALSE needed"). `test_weight_and_label_errors` (one case per rule, through each public function). `test_rejects_1d_and_non_finite_x`. `test_accepts_a_dataframe` (skipped without pandas). `test_wwcss_total_is_the_long_double_sum` (on the k = 3 cases: first asserts `want_total != float(np.float64(g[0]) + g[1] + g[2])` with `g` R's per-cluster values, then that the port's total equals R's).

- [ ] **Step 3: `wstats.py`.**

```python
"""WCluster's weighted cluster statistics: wmean, wss, wwcss and distw.

Each one replays R's arithmetic bit for bit. wmean sums in long double, as R's weighted.mean
does. wss goes through R's matrix product, so its sums are double sums taken row by row in
reference BLAS order. wwcss adds the clusters' values in long double. distw is the wss of two
clusters' rows together minus their wwcss, exactly as R computes it, so rounding can leave a
value at zero or below it.
"""

from __future__ import annotations

import numpy as np

from datanuggets._rstats import r_matvec, r_sum, r_weighted_mean
from datanuggets._validate import as_labels, as_matrix, as_weights
from datanuggets.result import WwcssResult


def _inputs(x, cl, w):
    x = as_matrix(x)
    w = as_weights(w, x.shape[0])
    levels, codes = as_labels(cl, x.shape[0])
    return x, w, levels.size, codes


def wmean(x, cl, w=None) -> np.ndarray:
    x, w, k, codes = _inputs(x, cl, w)
    out = np.empty((k, x.shape[1]))
    for i in range(k):
        rows = codes == i
        xs, ws = x[rows], w[rows]
        for j in range(x.shape[1]):
            out[i, j] = r_weighted_mean(xs[:, j], ws)
    return out


def _wss(x: np.ndarray, w: np.ndarray) -> float:
    """R's wss: the mean is t(x) %*% w over the long double sum(w), and the result is sum() of
    t((x - mean)^2) %*% w. Weights summing to zero give NaN, as in R."""
    with np.errstate(all="ignore"):
        dev = x - r_matvec(x.T, w) / r_sum(w)
        return r_sum(r_matvec((dev * dev).T, w))


def wss(x, w=None) -> float:
    x = as_matrix(x)
    return _wss(x, as_weights(w, x.shape[0]))


def _group_wss(x: np.ndarray, codes: np.ndarray, k: int, w: np.ndarray) -> np.ndarray:
    return np.array([_wss(x[codes == i], w[codes == i]) for i in range(k)])


def wwcss(x, cl, w=None, group_sum=False):
    x, w, k, codes = _inputs(x, cl, w)
    groups = _group_wss(x, codes, k, w)
    total = r_sum(groups)
    return WwcssResult(per_cluster=groups, total=total) if group_sum else total


def _distwij(x, codes, w, i: int, j: int, alone: np.ndarray) -> float:
    """R's distwij for clusters j < i: wss of their rows together, in row order, minus R's
    long double sum of the two clusters' own wss."""
    rows = (codes == i) | (codes == j)
    return _wss(x[rows], w[rows]) - r_sum([alone[j], alone[i]])


def distw(x, cl, w=None) -> np.ndarray:
    x, w, k, codes = _inputs(x, cl, w)
    if k < 2:
        raise ValueError("distw needs at least two clusters")
    alone = _group_wss(x, codes, k, w)
    out = np.full((k, k), np.nan)
    for i in range(1, k):
        for j in range(i):
            out[i, j] = _distwij(x, codes, w, i, j, alone)
    return out
```

  Each public function gets a docstring in the repo's style: what it returns and its shape, the cluster order, what a zero total weight gives, and where Python departs from R (the layout of `wmean`, the NaN that stands for R's NA in `distw`, the default weights).
  Run: `$UV run pytest tests/test_wstats.py -q`. Expected: all pass. If a parity case fails, the code is wrong or the plan is: find which R operation differs before changing anything, and never loosen a comparison.

- [ ] **Step 4: Gates and commit.** `$UV run pytest tests/test_wstats.py -q`, `$UV run ruff check src/datanuggets/wcluster/wstats.py tests/test_wstats.py`, `$UV run ruff format --check` on the same two files. The lead commits `Port wmean, wss, wwcss and distw with R's arithmetic`.

### Task 4: cluster_predict

**Files:**
- Create: `src/datanuggets/wcluster/predict.py`
- Create: `tests/test_predict.py`

**Interfaces:**
- Consumes: `wmean` (Task 3), `r_matvec`, `r_matmul`, `as_matrix`, `as_weights`, the arrays `newx`, `z`, `pred` of both fixture groups.
- Produces: `cluster_predict(x, cl, newx, *, w=None) -> np.ndarray`, and the private `_wdist_mat(newx, centers) -> np.ndarray`.

- [ ] **Step 1: Failing tests.** `test_wdist_mat_matches_r` (feeds R's recorded `wmean`, transposed, and compares with R's `z`, Exact, NaN equal to NaN, both layouts). `test_cluster_predict_matches_r` (`pred - 1`, every case that has `pred`, under `simplefilter("error")`). `test_dead_cluster_is_never_predicted` (`mtcars_gear_am`, `mtcars_gear_am_row01` and `mtcars_carb_am`: first asserts that `np.argmin` on R's `z` lands on the dead column for at least one row, then that the port never returns it). `test_every_cluster_dead_raises` (all weights zero: `ValueError`. R returns `integer(0)`). `test_overflow_rows_match_r` (the three `ovf_*` cases and the NaN-then-Inf case: `_wdist_mat` equals R's `z` with NaN equal to NaN and Inf kept, the positions equal R's where R returns a vector, and `ValueError` where `args["pred"] == "list"`). `test_first_index_wins_a_tie` (on `mtcars_cyl1d_gear`: first asserts that some row of R's `z` has an exact tie at its minimum, then that the port returns the first index). `test_negative_squared_distance_is_kept` (`iris50_singletons`: first asserts that R's `z` has a negative entry, then that the positions match R). `test_returns_positions_not_labels` (labels 4/6/8 give 0/1/2). `test_r_argument_order_fails` (`cluster_predict(x, w, cl, newx)` raises `TypeError`). `test_newx_must_have_x_columns`. `test_newx_rules` (`None` raises `TypeError`, as R stops on a NULL `newx` with "newdata should be a data frame.". 1-D, NaN and text raise as `as_matrix` does). Tests import from `datanuggets.wcluster.predict`.

- [ ] **Step 2: `predict.py`.**

```python
"""WCluster's cluster.predict: the nearest weighted cluster center for new rows."""

from __future__ import annotations

import numpy as np

from datanuggets._rstats import r_matmul, r_matvec
from datanuggets._validate import as_matrix, as_weights
from datanuggets.wcluster.wstats import wmean


def _wdist_mat(newx: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """R's wdist_mat with unit weights: each row's squared distance to each center, as
    (|x|^2 + |c|^2) - 2 x.c with R's sums in R's order, so a value can be a rounding error
    below zero. R then multiplies by an identity matrix, so each entry picks up 0 times every
    other row of its column: an entry becomes NaN when another row of its column is NaN or
    infinite, or when it is NaN itself. A lone infinity stays infinite."""
    ones = np.ones(newx.shape[1])
    with np.errstate(all="ignore"):
        m11 = r_matvec(newx * newx, ones)
        m22 = r_matvec(centers * centers, ones)
        z = (m11[:, None] + m22[None, :]) - 2 * r_matmul(newx, centers.T)
    bad = ~np.isfinite(z)
    other_bad = (bad.sum(axis=0)[None, :] - bad) > 0
    return np.where(other_bad | np.isnan(z), np.nan, z)


def cluster_predict(x, cl, newx, *, w=None) -> np.ndarray:
    x = as_matrix(x)
    w = as_weights(w, x.shape[0])
    newx = as_matrix(newx, "newx")
    if newx.shape[1] != x.shape[1]:
        raise ValueError(f"newx must have the {x.shape[1]} columns of x, got {newx.shape[1]}")
    z = _wdist_mat(newx, wmean(x, cl, w))
    dead = np.isnan(z)
    if dead.all(axis=1).any():
        raise ValueError(
            "a row of newx has no usable distance to any cluster center: every center is NaN "
            "(its weights sum to zero) or the squared distances overflow"
        )
    pos = np.where(dead, np.inf, z).argmin(axis=1)
    stuck = dead[np.arange(pos.size), pos]  # every live entry is +Inf
    pos[stuck] = (~dead[stuck]).argmax(axis=1)  # which.min takes the first live one
    return pos
```

  The docstring says the result is a position in sorted label order, 0-based, that it equals the label only when the labels are `0..k-1`, that a cluster with zero total weight is never returned, and that a row R would answer with an empty entry raises `ValueError`.
  Run: `$UV run pytest tests/test_predict.py -q`. Expected: all pass.

- [ ] **Step 3: Gates and commit.** `$UV run pytest tests/test_predict.py tests/test_wstats.py -q`, `$UV run ruff check src/datanuggets/wcluster/predict.py tests/test_predict.py`, `$UV run ruff format --check` on the same two files. The lead commits `Port cluster.predict with R's distance arithmetic and which.min`.

### Task 5: wpca

**Files:**
- Create: `r/fixtures_wpca.R`
- Create: `tests/fixtures/wpca/` (generated)
- Modify: `tests/rfixtures.py` (the eigen rule)
- Create: `src/datanuggets/wcluster/wpca.py`
- Create: `tests/test_wpca.py`

**Interfaces:**
- Consumes: `r_sum`, `r_matvec`, `r_matmul`, `r_eigen_sym(s, vectors=True)` (`_rstats`), `as_matrix`, `as_weights`, `WpcaResult`, `DataNuggetWarning`.
- Produces: `wpca`, the private `_wpca_parts(x, wrow, wcol, corr) -> (center, wcol, x0, scale, s)`, `rfixtures.eigen_mismatches(...) -> list[str]`, and fixture group `wpca` with arrays `x`, `wrow`, `wcol_in`, `sdev`, `rotation`, `scores`, `center`, `scale`, `wcol`, `values` (R's eigenvalues before the square root), `cap_x_0`, `cap_s` (the matrix R gave `eigen`), and `cap_cov` for `corr = TRUE` cases (the covariance before scaling), `args = list(corr = ...)`, and `warnings`.

- [ ] **Step 1: The generator.** Trace `Wpca` in the WCluster namespace on exit and read its locals `x_0`, `x_wcov` and `x_e$values`. Assert each capture is non-NULL. For a `corr = TRUE` case, `cap_cov` is `x_wcov` from a second run with `corr = FALSE`. Warnings are caught with the `withCallingHandlers` pattern of `r/fixtures_create_dn.R` and written as `extra = list(warnings = I(warns))`.

| Case | Definition | Pins |
|---|---|---|
| `iris`, `iris_corr` | iris 1:4, defaults | baseline, the `corr` scaling order and the lower triangle |
| `quakes_stations`, `quakes_stations_corr` | quakes, `wrow` = stations | count weights, n = 1000 |
| `w_iris_petalwidth`, `w_mtcars_wt` | non-integer row weights | the long double `sum(wrow)` and the double subtraction of 1 |
| `mtcars_carb_invsd` | mtcars, `wrow` = carb, `wcol` = 1/sd | column weight normalisation, p = 11 |
| `nug_quakes`, `nug_quakes_corr`, `nug_iris_passthrough` | nugget centers and weights | the DN.Wpca input, a zero row weight |
| `faithful1d` | one column | p = 1 |
| `petalwidth_1d_corr` | Petal.Width alone, `corr = TRUE` | R's one-column quirk, where Python departs |
| `volcano_scaled` | `scale(volcano)` | p = 61, close eigenvalues |
| `nug_volcano_scaled` | 25 nuggets, p = 61 | rank deficiency, NaN `sdev` |
| `mtcars_cyl8_constant` | the cyl 8 rows of mtcars | a constant column with `corr = FALSE` |
| `wcol_volcano49_default`, `wcol_volcano49_corr` | 49 volcano columns, default `wcol` | `(1 / p) * p` isn't 1 at p = 49 |
| `den_mtcars_first18_wt`, `den_trees_first2_girth` | real weights where `sum(wrow) - 1` in double is one ulp from a long double `sum(c(wrow, -1))` (`f4v/wpca/v9_extra.R:31-33`). Trees has an exact zero eigenvalue in R | the double subtraction of 1, no warning |
| `den_iris_first2_petalwidth`, `den_iris_first3_petalwidth`, `den_iris_first4_petalwidth` | `iris[1:k, 1:4]` with `wrow = Petal.Width[1:k]`, sums 0.4, 0.6 and 0.8 | weights summing below 1: clearly negative eigenvalues, NaN `sdev`, exactly one warning |
| `one_iris_first5_petalwidth` | `iris[1:5, 1:4]`, `wrow = Petal.Width[1:5]`, which sums to exactly 1 | R stops in `eigen` with "infinite or missing values in 'x'". Recorded as inputs only, with `extra = list(error = conditionMessage(e))` |
| `tie_npk` | the npk design columns | tied eigenvalues |

  Run `"$RS" r/fixtures_wpca.R`, copy `tests/fixtures/wpca` to a scratch folder, run it again and compare every `.bin` byte for byte. Expected: identical. Print the group's content and allocated size.

- [ ] **Step 2: The eigen rule in `tests/rfixtures.py`.**

```python
EIG_FLOOR, EIG_C, EIG_K, EIG_TIE = 1e-10, 256.0, 1024.0, 1e-12


def eigen_mismatches(s, values_r, vectors_r, scores_r, x0, sdev, vectors, scores) -> list[str]:
    """Where a port's eigen output differs from R's by more than two correct LAPACK builds can.
    Eigenvalues within EIG_TIE of each other form a cluster, and only a cluster's subspace is
    determined, so vectors are compared through their projectors. R1 orthonormal vectors. R2
    each vector's residual against R's eigenvalue. R3 sdev squared against R's eigenvalue (a
    NaN sdev needs a value at most the band above zero). R4 each cluster's projector against
    R's, within max(EIG_FLOOR, EIG_C eps / gap). R5 the same for the scores, row by row."""
    eps = float(np.finfo(float).eps)
    p = values_r.size
    for name, a in (("sdev", sdev), ("rotation", vectors), ("scores", scores)):
        if a.shape[-1] != p:
            raise ValueError(f"{name} has {a.shape[-1]} columns, R has {p}")
    norm = max(float(np.max(np.abs(values_r))), 1e-300)
    band = EIG_K * eps * norm
    lower = np.tril(s) + np.tril(s, -1).T
    out = []
    orth = float(np.max(np.abs(vectors.T @ vectors - np.eye(p))))
    if orth > EIG_FLOOR:
        out.append(f"R1 rotation is not orthonormal: {orth:.3e}")
    res = np.linalg.norm(lower @ vectors - vectors * values_r, axis=0)
    bad = np.flatnonzero(res > band)
    out += [f"R2 column {i}: residual {res[i]:.3e} > {band:.3e}" for i in bad]
    for i in range(p):
        if np.isnan(sdev[i]):
            if values_r[i] > band:
                out.append(f"R3 sdev {i} is NaN but R's eigenvalue is {values_r[i]:.3e}")
        elif abs(sdev[i] ** 2 - values_r[i]) > band:
            out.append(f"R3 sdev {i}: {sdev[i] ** 2:.17g} against {values_r[i]:.17g}")
    cid = np.concatenate(([0], np.cumsum(np.abs(np.diff(values_r)) > EIG_TIE * norm)))
    rows = np.linalg.norm(x0, axis=1)
    for c in range(int(cid[-1]) + 1):
        idx, rest = np.flatnonzero(cid == c), np.flatnonzero(cid != c)
        gap = np.inf
        if rest.size:
            gap = float(np.min(np.abs(values_r[idx][:, None] - values_r[rest][None, :]))) / norm
        tol = max(EIG_FLOOR, EIG_C * eps / gap)
        v, vr = vectors[:, idx], vectors_r[:, idx]
        sin = float(np.linalg.norm(v @ v.T - vr @ vr.T, 2))
        if sin > tol:
            out.append(f"R4 columns {idx[0]}..{idx[-1]}: subspace off by {sin:.3e} > {tol:.3e}")
        d = np.linalg.norm(scores[:, idx] @ v.T - scores_r[:, idx] @ vr.T, axis=1)
        off = d > (tol + 8 * p * eps) * rows
        if off.any():
            out.append(f"R5 columns {idx[0]}..{idx[-1]}: scores off in {int(off.sum())} rows")
    return out
```

  (`@` is fine in tests. The ban covers `src/`.) The rule needs `sdev`, `vectors` and `scores` with R's column count. A call with another length raises `ValueError` before any comparison, so a wrong-length mutant can't be truncated into a pass. Write the rule into `tests/rfixtures.py` in a single edit, because the statistics chain imports that module while this task runs.

- [ ] **Step 3: Failing tests in `tests/test_wpca.py`.** "Every case" below means every case with outputs: `one_iris_first5_petalwidth` holds inputs only and is used by `test_weights_summing_to_one_raise`. Tests import `wpca` and `_wpca_parts` with `from datanuggets.wcluster.wpca import ...`, which reads the module from `sys.modules` and keeps working after Task 6.
  - `test_fixtures_cover_the_edge_cases`: every parametrized list is non-empty, and the group holds a `corr = TRUE` case, a p = 1 case, a case with a zero row weight, a case whose R `values` has an entry below `-band` with a recorded warning, a case where every R eigenvalue is above `band` with none, a tie (two R eigenvalues within `EIG_TIE`), a `corr = TRUE` case whose `cap_s` isn't symmetric, and the sum-to-one stop case with its recorded error.
  - `test_parts_match_r`: `_wpca_parts` against `center`, `wcol`, `cap_x_0`, `cap_s` and `scale`, Exact, every case but `petalwidth_1d_corr`, both layouts. For `corr = FALSE` cases R's `scale` is the one number 1 and the port's is all ones. For `corr = TRUE` cases also check that `_wpca_parts(..., corr=False)` gives `cap_cov`.
  - `test_scores_product_matches_r`: `r_matmul(cap_x_0, rotation)` on R's own arrays equals R's `scores`, Exact. This pins the score product apart from LAPACK.
  - `test_wpca_matches_r`: every case but `petalwidth_1d_corr` (which `test_one_column_corr_departs_from_r` covers). `center`, `scale`, `wcol`, `wrow` Exact, and `eigen_mismatches(cap_s, values, rotation, scores, cap_x_0, got.sdev, got.rotation, got.x) == []`.
  - `test_warnings_match_r`: where R's smallest eigenvalue is below `-band` (the three `den_iris_*` cases), exactly one `DataNuggetWarning("NaNs produced")`, equal to the manifest's `warnings`. Where every R eigenvalue is above `+band`, no warning and none in the manifest. On every case, a warning exactly when the port's `sdev` has a NaN. Values inside the band (the exact zero in `den_trees_first2_girth`, rank-deficient nuggets) may take either sign on another LAPACK, so nothing more is required there.
  - `test_eigen_rule_rejects_wrong_outputs`: the rule must report a mismatch for each of: the first two columns of rotation and scores swapped (`iris`, `quakes_stations`, `volcano_scaled`), the sign of one rotation column flipped with the scores left alone (same three), scores computed from the weighted data `x_w` (`quakes_stations`, `mtcars_carb_invsd`, `w_iris_petalwidth` only, because with unit weights and `(1 / p) * p == 1` the weighted data equals the centered data bit for bit), and `sdev` from another case with the same p (`iris` takes `iris_corr`'s, `quakes_stations` takes `quakes_stations_corr`'s, `volcano_scaled` takes `nug_volcano_scaled`'s). Each mutant first asserts that it differs from the right output somewhere. And the rule must report none when a tied pair of columns is rotated among themselves (`tie_npk`).
  - `test_sdev_comes_from_the_vectors_call`: `mod = importlib.import_module("datanuggets.wcluster.wpca")`, then `monkeypatch.setattr(mod, "r_eigen_sym", spy)` with a spy that records its `vectors` argument and calls the real one. Assert one call with `vectors=True`. R's `Wpca` asks `eigen` for vectors, and LAPACK's values-only route returns other bits. Never patch by the dotted string path: after Task 6 `datanuggets.wcluster.wpca` is the function.
  - `test_eigen_reads_the_lower_triangle` (in `tests/test_wpca.py`): for each `corr = TRUE` case, first assert that `cap_s` isn't symmetric, then that `r_eigen_sym(cap_s, vectors=True)` equals bit for bit what it returns for the lower triangle mirrored up.
  - `test_one_column_corr_departs_from_r`: on `petalwidth_1d_corr`, R's `sdev` equals R's `scale` (its quirk), the port's `scale` and `center` equal R's, and the port's `sdev` is 1 within 4 eps.
  - `test_weights_summing_to_one_raise` (on `one_iris_first5_petalwidth`: first asserts the manifest records R's error, then `ValueError` matching "sum to exactly 1"), `test_weights_summing_below_one_warn_once`, `test_weight_rules` (negative `wrow`, a `wcol` that sums to zero or mixes signs, wrong lengths, text, bool, NaN), `test_constant_column_with_corr_raises`, `test_one_row` (`wrow = [1]` raises the sum-to-one error, `[0]` the zero-total error, and `[0.5]` or `[2]` returns an all-zero `sdev` with no warning, as R does for `Wpca(matrix(c(1, 2, 3), 1), wrow = 0.5)`), `test_corr_must_be_a_bool`, `test_rejects_1d_and_non_finite_x`, `test_default_weights_are_ones`.

- [ ] **Step 4: `wpca.py`.**

```python
"""WCluster's Wpca: principal components of data with a weight per row and per column."""

from __future__ import annotations

import warnings

import numpy as np

from datanuggets._rstats import r_eigen_sym, r_matmul, r_matvec, r_sum
from datanuggets._validate import as_matrix, as_weights
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import WpcaResult


def _wpca_parts(x: np.ndarray, wrow: np.ndarray, wcol: np.ndarray, corr: bool):
    """The weighted center, the column weights scaled to sum to p, the centered data, the
    scale and the matrix R hands to eigen(), each bit for bit R's."""
    p = x.shape[1]
    total = r_sum(wrow)
    if (wrow < 0).any():
        raise ValueError("wrow must not hold negative weights")
    if total == 0:
        raise ValueError("wrow must not sum to zero")
    if total - 1.0 == 0:
        raise ValueError("wrow must not sum to exactly 1: the covariance divides by sum(wrow) - 1")
    with np.errstate(all="ignore"):
        wcol = (wcol / r_sum(wcol)) * p
    if not np.isfinite(wcol).all() or (wcol < 0).any():
        raise ValueError("wcol must have a nonzero sum and weights of one sign")
    with np.errstate(all="ignore"):  # overflow gives R's Inf and NaN, and eigen then raises
        center = r_matvec(x.T, wrow) / total
        x0 = x - center
        xw = (x0 * np.sqrt(wrow)[:, None]) * np.sqrt(wcol)[None, :]
        s = r_matmul(xw.T, xw) / (total - 1.0)
    scale = np.ones(p)
    if corr:
        with np.errstate(all="ignore"):
            scale = np.sqrt(np.diag(s))
            inv = 1 / scale
        if not np.isfinite(inv).all():
            raise ValueError("corr=True needs a positive weighted variance in every column")
        s = (s * inv[:, None]) * inv[None, :]
    return center, wcol, x0, scale, s


def wpca(x, wrow=None, wcol=None, corr=False) -> WpcaResult:
    x = as_matrix(x)
    n, p = x.shape
    if n == 0:
        raise ValueError("x must have at least one row")
    if not isinstance(corr, (bool, np.bool_)):
        raise TypeError("corr must be True or False")
    wrow = as_weights(wrow, n, "wrow")
    wcol = as_weights(wcol, p, "wcol", per="column")
    center, wcol, x0, scale, s = _wpca_parts(x, wrow, wcol, bool(corr))
    values, rotation = r_eigen_sym(s, vectors=True)
    if (values < 0).any():
        warnings.warn("NaNs produced", DataNuggetWarning, stacklevel=2)
    with np.errstate(invalid="ignore"):
        sdev = np.sqrt(values)
    return WpcaResult(
        sdev=sdev,
        rotation=rotation,
        x=r_matmul(x0, rotation),
        center=center,
        scale=scale,
        wrow=wrow,
        wcol=wcol,
    )
```

  The docstring says: the weights act as counts, so the covariance divides by `sum(wrow) - 1`. The scores are the centered data times the rotation, not scaled and not weighted. The sign of each rotation column is LAPACK's and can differ from R's. An eigenvalue below zero gives a NaN `sdev` and R's warning. With one column and `corr=True` the port returns an `sdev` of 1 where R returns the unscaled value or stops.
  Run: `$UV run pytest tests/test_wpca.py -q`. Expected: all pass.

- [ ] **Step 5: Gates and commits.** `$UV run pytest tests/test_wpca.py -q`, `$UV run ruff check src/datanuggets/wcluster/wpca.py tests/test_wpca.py tests/rfixtures.py`, `$UV run ruff format --check` on the same three files. The lead makes two commits: `Record WCluster's Wpca with its traced internals` (generator and group), `Port Wpca up to eigen bit for bit, and compare eigen output by subspace` (`wpca.py`, `tests/rfixtures.py`, `tests/test_wpca.py`).

### Task 6: exports and parity reports

**Files:**
- Modify: `src/datanuggets/wcluster/__init__.py`, `src/datanuggets/__init__.py`
- Modify: `scripts/parity_report.py`
- Create: `docs/validation/wmean.md`, `wss.md`, `wwcss.md`, `distw.md`, `cluster_predict.md`, `wpca.md` (generated)

- [ ] **Step 1: Exports.** `wcluster/__init__.py` imports the six functions and lists them in a sorted `__all__`. The top-level `__init__.py` re-exports the six plus `WpcaResult` and `WwcssResult`, keeping `__all__` sorted. Test: `from datanuggets import wmean` and `from datanuggets.wcluster import wmean` are the same object.
- [ ] **Step 2: Reports.** Six report functions in the script's existing style, reusing its `verdict()`. The five statistics load `wstats` only (the constructed group is tested and not reported) and report per case from the same comparison the tests use, with R's shapes handled: transpose `wmean`, compare R's length-one arrays through `[0]` (`verdict` of a 0-d value against a shape `(1,)` array says NO), and skip a case that lacks the array (`distw` needs k >= 2, `pred` needs a predicting case). `wpca.md` reports per case whether the Exact fields match and whether the eigen rule passes, and says in its header what the rule is, in words. Every header names the fixture folder, the R, WCluster, BLAS and LAPACK versions from the manifests, and `matprod`.
- [ ] **Step 3: Run.** `$UV run python scripts/parity_report.py`, then `git status --short docs/validation`. Expected: six new files, every row `yes`, and the four existing reports with no content change (`git checkout --` them if they show as modified). Then the joint fixture budget: print the content and allocated size of `tests/fixtures` as a whole and of each new group. Expected: the new groups add at most 8 MB allocated and the total stays under 25 MB.
- [ ] **Step 4: Whole-repo gates, full suite on the Mac mini, commit.** `$UV run ruff check .` and `$UV run ruff format --check .` (the first whole-repo lint of the feature). The lead runs the full suite on the Mac mini. Commit `Export the WCluster leaves and report their parity with R`.

### Task 7: README, r/README and spec

- [ ] **Step 1: README.** Status lists the six as ported. The section on differences from R gains the rows of the table below, and the sentence "Defaults match R except for `n_jobs`" is corrected for the default weights. The validation section gets one paragraph per function whose numbers come only from the generated reports, and says the tie and heavy-weight cases are constructed. The credit line names both packages, `datanugget` and `WCluster`, with WCluster's authors taken from its installed DESCRIPTION (`packageDescription("WCluster")$Author`), never typed from memory. Update README lines 11, 21 (which names these six as next), 133 (the versions parity is against gains WCluster 1.3.0) and 196-198, and the sentence at line 94 that says every function rejects NaN and infinite values, if a new function makes it false. Fix the two sentences the speed work left stale, if they are still stale.
- [ ] **Step 2: `r/README.md`.** Step 2 names WCluster 1.3.0. The generator list gains the three new scripts.
- [ ] **Step 3: Spec.** Section 2: the five statistics become `Exact`, and `Wpca` becomes `Exact (center, scale, wcol, eigen input) + Eigen (sdev, rotation, scores)`. Section 5: the Exact row says NaN equals NaN, the Numeric row drops WWCSS and eigenvalues, the Up-to-sign row is replaced by a new `Eigen` row that describes the residual and subspace rule with its four constants, and the fixture dataset list names every R built-in the fixtures use (iris, faithful, quakes, mtcars, ChickWeight, volcano, EuStockMarkets, USArrests with state.x77, PlantGrowth, rock, LifeCycleSavings, swiss, infert, trees, npk). Section 9.4: the closed-form Ward cost changes 6 of the 149 merge steps on iris with unit weights, so by the section's own condition it doesn't ship. Section 10: correct the default-weights row (R's default stops unless the data is square for `wmean` and `wwcss`, or has one column for `distw`), and append:

| Item | R behavior | Python |
|---|---|---|
| `wmean` layout | `p` rows by `k` columns | `(k, p)`, like every other center array in the package |
| `distw` cells R never fills | `NA`, distinct from a computed `NaN` | `NaN` in both places |
| `distw` with one cluster | stops ("missing value where TRUE/FALSE needed") | `ValueError` naming the cause |
| cluster labels | `wwcss` groups by `factor()` (15 significant digits), `distw` by `sort(unique())`, text by locale collation | whole numbers smaller than 1e15 in size, sorted. Text, bool, fractional and NaN labels raise |
| `cluster.predict` result | 1-based position in sorted label order, not the label | 0-based position, documented |
| `cluster.predict` argument order | `(x, w, cl, newx)`, and a swapped `w` and `cl` runs silently | `(x, cl, newx, *, w=None)` |
| `cluster.predict` for a row with no finite distance (every center NaN, or the squares overflow) | `integer(0)` when no row has one, a list with empty entries when only some rows do | `ValueError` |
| `cluster.predict` distances after R's identity multiply | `0 * Inf` from one overflowing row turns its whole column NaN, and a lone `Inf` stays | the same, computed without the multiply |
| 1-D `x` in the six WCluster leaves | accepted as one column | `ValueError`, as everywhere else |
| integer `x` with integer weights | `wmean` and `cluster.predict`: an NA center on overflow with a warning. `Wpca`: the NA becomes 0 in its column sum, a wrong center with a warning. `wss`, `wwcss`, `distw`: no integer products | float64 arithmetic, no overflow |
| NaN or infinite weights, text or bool weights | passed through or stopped late in `eigen` | `ValueError` or `TypeError` up front |
| `Wpca` weights summing to 1, a zero total, a negative row weight, zero variance under `corr` | stop inside `eigen` ("infinite or missing values in 'x'") | `ValueError` naming the cause |
| `Wpca` with one column and `corr = TRUE` | `diag()` of one number builds an identity: the covariance passes through unscaled for `0.5 < dd <= 1`, and R stops otherwise | the general formula, `sdev` 1 to rounding. A deviation that fixes an upstream bug |
| `Wpca` `scale` with `corr = FALSE` | the single number 1 | ones, one per column |
| `Wpca` `corr` | `TRUE`, `1` and `"TRUE"` all pass, other values fail obscurely | a bool, else `TypeError` |
| `w = NULL` | stops, though the help pages say it means unit weights | `None` means unit weights |

  Section 12 gains three questions: whether `Wpca`'s one-column `corr` behaviour is worth an upstream fix, whether `distw` values at or below zero (rounding on identical rows in different clusters) should be clamped upstream, and whether `wwcss` and `distw` grouping labels differently is intended.
- [ ] **Step 4: Prose gate and commit.** Grep the published prose for em dashes, en dashes and semicolons. No test runs the README's Python blocks, so run each block the README gains in a scratch script and check its printed output by hand. The lead runs the full suite on the Mac mini. Commit `Document the WCluster leaves and where they depart from R`.

## Execution Handoff

Tasks in order 1, then 2 to 4 beside 5, then 6 and 7. Executors don't commit: each task ends with its gates green and its file list, and the lead commits after that task's verifier passes (see Global Constraints). After Task 7: one whole-branch review with the Review Focus word for word and a second reviewer that must show facts and specific test cases, one fix pass limited to Critical and Important findings with each fix written test first, the full suite on the Mac mini, push, open a PR against `feat/speed`, do not merge.

## Self-Review

1. Spec coverage. Section 2's six rows each have a task. Section 4's rules are in Global Constraints. Every R `stop()` is in the table above with its kind and its Python form. Section 5's fixture rules hold, with the constructed cases kept out of the reports.
2. Placeholders. The source for `src/` is in the plan. The generators and tests are given as contracts (arrays, case tables, test names and what each asserts), because their case definitions are R expressions recorded during the research. This departs from the Feature 2 plan, which inlined every file.
3. Type consistency. `as_labels` returns `(levels, codes)` everywhere. Centers are `(k, p)` everywhere in Python, and only the fixture array `wmean` is `(p, k)`. The fixture array `scores` is compared with the result field `x`.
4. Review Focus. Each of the five lines names its tests and the task that owns them.

Rulings carried into execution (ledger them at pre-flight):

- Zero-weight clusters keep R's NaN instead of raising. Cost if wrong: one check and one error test to add.
- Labels are whole numbers only. Cost if wrong: a caller with text labels has to encode them first.
- `cluster_predict` returns positions, not labels. Cost if wrong: one indexing line and a documentation change.
- `wpca` with one column and `corr=True` follows the general formula, not R's quirk. Cost if wrong: a three-line branch.
- `scale` is ones when `corr=False`. Cost if wrong: a return type change.
- `wpca` raises a named `ValueError` where R stops inside `eigen`. Cost if wrong: message text only.
- Eigen output is compared by residual and subspace, not entry by entry. Cost if wrong: a looser or tighter constant. The rule's own tests guard against a vacuous pass.
- The closed-form Ward `distw` doesn't ship. Cost if wrong: none for parity, speed only.
- `_wdist_mat` has no weight argument yet. `Wkmeans` needs one in Phase 3, where the multiply by `w` comes before the argmin.
- A new row with no finite distance raises `ValueError` instead of returning R's empty entry. Cost if wrong: a sentinel position and a documentation change.
- `_wdist_mat` reproduces the NaN pattern of R's identity multiply instead of skipping it, so `z` matches R even for overflowing rows. Cost if wrong: none for real data, where `z` is finite.
- The spec's module names stay (`wcluster/wpca.py` beside the function `wpca`). Tests reach the module through `importlib`. Cost if wrong: a module rename, here and for `wkmeans` and `whclust`.

Plan attack (two Opus reviewers, before execution) changed: `_wdist_mat` and the argmin (a lone Inf survives R's identity multiply, and `which.min(c(NaN, Inf, Inf))` is 2), the `den_*` cases (only the iris ones sum below 1), the scores-from-weighted-data mutant (identical to the right output under unit weights), the spy test (the function shadows its module after Task 6), gates and commits for two chains in one working tree, coverage guards against vacuous passes, the one-row and sum-to-one tests, the integer-overflow deviation, the spec's tier table and dataset list, and ruff formatting of two code blocks.
