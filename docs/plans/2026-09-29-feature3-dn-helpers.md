# Feature 3: get_dn_means and arithmetic_complexity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `getDN.means` and `AC` from R datanugget 1.5.0 so that `get_dn_means` matches R bit for bit on every nugget set in the create and refine fixtures, and `arithmetic_complexity` matches R's loop counts exactly and its value to 1e-8 on a grid of real dataset shapes.

**Architecture:** Same oracle setup as Features 1 and 2: R scripts in `r/` write frozen fixtures and `pytest` never needs R. `r/fixtures_get_dn_means.R` reads the committed create_dn and refine_dn fixtures back (a new `read_array` in `r/fixture_io.R`), runs R's `getDN.means` on each nugget set and records the means. `r/fixtures_ac.R` traces `AC`'s local loop state (G, R, A, DN.num1, B) on exit over a grid of real dataset shapes, and records the calls that never return under a one-second limit. `get_dn_means` groups rows with a stable argsort and averages each nugget with the existing long-double `r_mean`. `arithmetic_complexity` replays AC's two while loops in the same double arithmetic, with Brent's cycle check where R loops forever.

**Tech Stack:** Python 3.12+ (dev on 3.14), numpy >= 2.0, pytest, uv, ruff 0.16; R 4.6.1 with datanugget 1.5.0 (dev only).

**Spec:** `docs/specs/2026-09-29-datanuggets-python-port-design.md` (section 2 rows `getDN.means` and `AC`, section 4 conventions, section 5 harness, section 10 deviations table). Phase 2 in section 8, which comes before `optimal_dn` (phase 4).

**Execution:** RON2K: "start working on the next now do not invent numbers make sure code is proper and equivalent to R package". Native execution (superpowers:executing-plans), as in Feature 2: build every task with TDD, one Opus whole-branch review, one fix pass, push `feat/dn-helpers`, open a PR against `main`, do not merge.

## Global Constraints

- Branch `feat/dn-helpers` from `main` (57fc88d). `main` advances only by PR, and merging is RON2K's call.
- Public signatures: `get_dn_means(x, dn: DataNuggets) -> np.ndarray` (shape `(k, p)`, float64) and `arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2) -> float` with no defaults (R's `AC` has none).
- Spec section 4: R dots become underscores (`DN` -> `dn`, `R` -> `r`, `delete.percent` -> `delete_percent`, `DN.num1` -> `dn_num1`). 0-based nugget ids. Every R `stop()` condition becomes a `ValueError`/`TypeError` with a test. Non-numeric `x` raises `TypeError`, 1-D `x` raises `ValueError`.
- R's `warning()`s become `DataNuggetWarning` with R's text (`"NaNs produced"` from `log10`).
- Parity tiers: Exact = `np.testing.assert_array_equal` (NaN equals NaN). Numeric = `rtol=1e-8, atol=1e-10` (`RTOL`/`ATOL` in `tests/rfixtures.py`). `get_dn_means`: Exact. `arithmetic_complexity`: loop counts Exact, value Numeric (it goes through the platform's `log10`).
- Real data only: every fixture comes from R run on R's built-in datasets or on R's own nugget outputs.
- Fixtures: raw `.bin` + `manifest.json` via `r/fixture_io.R`. Total under 25 MB (11 MB before this feature).
- Commits: one logical change each, imperative subject <= 70 chars plus a why/what body, written to a message file in the scratchpad and committed with `git commit -F`. No `Claude-Session:`, `Co-Authored-By` or "Generated with" lines, and no mention of AI tools. Stage files by name, never `git add -A`. `git checkout --` any `docs/validation/*.md` that shows modified with no content change (the report script writes CRLF on Windows).
- Repo root holds only toolchain files. Code in `src/datanuggets/`, tests in `tests/`, R in `r/`, scripts in `scripts/`, docs in `docs/`.
- Shell: Git Bash on Windows 11 from the repo root. `UV=/c/Users/thegr/AppData/Local/hermes/bin/uv.exe`, `RS="$LOCALAPPDATA/Programs/R/R-4.6.1/bin/Rscript.exe"`. Run R bare (no `R_LIBS_USER`): `r/fixture_io.R` finds the library itself.
- Published prose (README, docstrings, report headers) follows the humanizer rules: no em or en dashes, no semicolons, contractions, no bold or italic in prose, no comma-pause chains, no test counts.

## Review Focus

1. A `DataNuggets` that doesn't describe `x` (assignment length, ids outside `[0, k)`, fractional ids, another column count): a person expects a `ValueError` like `refine_dn` gives, not means of the wrong rows. R silently drops extra rows. Tests `test_x_must_have_the_rows_dn_assigns`, `test_assignments_must_be_nugget_ids`, `test_x_must_have_dn_columns` (Task 1).
2. Arguments that make R's `AC` loop forever, including `dn_num1 = 2**60` where `x - 1 == x`: a person expects an error in milliseconds, never a hang. Every such row in R's grid (`hung = 1`) must raise. Test `test_raises_where_r_never_returns` (Task 2).
3. numpy scalars (`np.int64`, `np.float64`, `np.int32`) as `arithmetic_complexity` arguments: same value as the Python numbers. Test `test_numpy_scalars_act_like_python_numbers` (Task 2).
4. A pandas DataFrame `x`, including non-numeric columns for `arithmetic_complexity` (R's `AC` reads only `nrow`/`ncol`) and mixed int/float columns for `get_dn_means`. Tests `test_reads_only_the_shape_of_any_2d_input` (Task 2) and `test_accepts_a_dataframe` (Task 1).
5. One-column data: R's `getDN.means` returns its means as one transposed row. A person expects `k` rows. Pinned by the `faithful_1d*` sets in `test_get_dn_means_matches_r`, which also assert R's `(1, k)` shape (Task 1).

---

### Task 1: get_dn_means

**Files:**
- Modify: `r/fixture_io.R` (add `read_array` and its round-trip check)
- Create: `r/fixtures_get_dn_means.R`
- Create: `tests/fixtures/get_dn_means/{create_dn,refine_dn_in,refine_dn_out}/` (generated)
- Modify: `src/datanuggets/_rcompat.py` (`_ldouble_mean` of no values is NaN)
- Modify: `src/datanuggets/_validate.py` (gains `check_assignments`, moved from `refine.py`)
- Modify: `src/datanuggets/refine.py` (use `check_assignments`)
- Create: `src/datanuggets/means.py`
- Modify: `src/datanuggets/__init__.py`
- Modify: `tests/rfixtures.py` (add `means_inputs`)
- Modify: `tests/test_rcompat.py`
- Create: `tests/test_means.py`

**Interfaces:**
- Consumes: `r_mean(a, axis) -> np.ndarray` (`_rcompat`), `as_matrix(x) -> np.ndarray` (`_validate`), `DataNuggets` (`result`), `Case`, `load_group`, `FIXTURE_ROOT` (`tests/rfixtures.py`).
- Produces: `datanuggets.get_dn_means(x, dn) -> np.ndarray`; `datanuggets._validate.check_assignments(dn, n: int, p: int) -> np.ndarray` (intp ids); `rfixtures.means_inputs(means: Case, name: str) -> tuple[np.ndarray, DataNuggets]`; fixture group `get_dn_means` with cases `create_dn`, `refine_dn_in`, `refine_dn_out`, each with `args = {"source": <group>, "prefix": "" | "in_"}` and one `(k, p)` array per source case id (`(1, k)` when R transposed it).

- [ ] **Step 1: Add `read_array` to `r/fixture_io.R`** (append after `write_case`)

```r
# Reads back an array write_array wrote: row-major little-endian bytes, in R's own layout.
read_array <- function(dir, name) {
  meta <- read_json(file.path(dir, "manifest.json"))$arrays[[name]]
  if (is.null(meta)) stop("no array ", name, " in ", dir)
  shape <- as.integer(unlist(meta$shape))
  int <- meta$dtype == "<i4"
  con <- file(file.path(dir, meta$file), "rb")
  on.exit(close(con))
  v <- readBin(con, if (int) "integer" else "double", n = prod(shape),
               size = if (int) 4L else 8L, endian = "little")
  stopifnot(length(v) == prod(shape))
  if (length(shape) == 1L) v else t(matrix(v, nrow = shape[2], ncol = shape[1]))
}

# Fail closed if read_array does not invert write_array.
local({
  d <- tempfile()
  dir.create(d)
  m <- matrix(c(1.5, -2, 3, 4, 5, 6), 2)
  meta <- list(arrays = list(m = write_array(d, "m", m), i = write_array(d, "i", 3:1)))
  write_json(meta, file.path(d, "manifest.json"), auto_unbox = TRUE, digits = NA)
  stopifnot(identical(read_array(d, "m"), m), identical(read_array(d, "i"), 3:1))
  unlink(d, recursive = TRUE)
})
```

- [ ] **Step 2: Write `r/fixtures_get_dn_means.R`**

```r
# Fixtures for getDN.means: R's nugget means for every nugget set the create_dn and refine_dn
# fixtures hold (create.DN outputs, refine.DN inputs and refine.DN outputs), read back from their
# .bin files. Run after fixtures_create_dn.R and fixtures_refine_dn.R.
source("r/fixture_io.R")

# A "datanugget" object laid out as create.DN returns it. getDN.means reads only the table's
# shape, the center column names and the assignments.
as_dn <- function(centers, weights, scales, assignments) {
  tab <- data.frame(seq_len(nrow(centers)), centers, weights, scales)
  names(tab) <- c("Data Nugget", paste0("Center", seq_len(ncol(centers))), "Weight", "Scale")
  structure(list(`Data Nuggets` = tab, `Data Nugget Assignments` = assignments),
            class = "datanugget")
}

targets <- list(create_dn = c("create_dn", ""), refine_dn_in = c("refine_dn", "in_"),
                refine_dn_out = c("refine_dn", ""))
for (target in names(targets)) {
  group <- targets[[target]][1]
  prefix <- targets[[target]][2]
  dirs <- list.dirs(file.path("tests", "fixtures", group), recursive = FALSE)
  means <- lapply(dirs, function(dir) {
    part <- function(name) read_array(dir, paste0(prefix, name))
    dn <- as_dn(part("centers"), part("weights"), part("scales"), part("assignments"))
    unname(as.matrix(getDN.means(read_array(dir, "x"), dn)))
  })
  names(means) <- basename(dirs)
  write_case("get_dn_means", target, list(source = group, prefix = prefix), means)
  cat(target, length(means), "nugget sets\n")
}
```

- [ ] **Step 3: Generate the fixtures**

Run: `"$RS" r/fixtures_get_dn_means.R`
Expected: three lines, `create_dn 37 nugget sets`, `refine_dn_in 128 nugget sets`, `refine_dn_out 128 nugget sets` (one per case directory: 37 create_dn, 128 refine_dn), exit 0.

- [ ] **Step 4: Commit the generator and fixtures**

```bash
git add r/fixture_io.R r/fixtures_get_dn_means.R tests/fixtures/get_dn_means
git commit -F "$SCRATCH/f3-c1.txt"
```
Message: `Record R's getDN.means on every create and refine nugget set` + body (reads the committed fixtures back instead of rerunning create.DN, so the means belong to exactly R's own nugget sets; includes the empty nugget, one-column and all-singleton sets).

- [ ] **Step 5: Write the failing r_mean test** (append to `tests/test_rcompat.py`)

```python
def test_r_mean_of_no_values_is_nan_like_r():
    # R's colMeans of a 0-row matrix divides 0 by 0
    got = r_mean(np.empty((0, 3)), axis=0)
    assert got.shape == (3,) and np.isnan(got).all()
    assert np.isnan(r_mean(np.empty((2, 0)), axis=1)).all()
```

- [ ] **Step 6: Run it**

Run: `$UV run pytest tests/test_rcompat.py -q`
Expected: 1 failed (`test_r_mean_of_no_values_is_nan_like_r`: `_ldouble_mean([])` returns 0.0).

- [ ] **Step 7: Fix `_ldouble_mean`** (in `src/datanuggets/_rcompat.py` add `import math`, and make the docstring and first line)

```python
def _ldouble_mean(v: list[float]) -> float:
    """sum(v) / len(v) the way R computes it: every partial sum and the quotient are rounded to
    a 64-bit significand (ties to even), then the quotient is rounded to double. No values give
    NaN, R's 0 / 0."""
    if not v:
        return math.nan
```

- [ ] **Step 8: Run it**

Run: `$UV run pytest tests/test_rcompat.py tests/test_create.py tests/test_refine.py -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/datanuggets/_rcompat.py tests/test_rcompat.py
git commit -F "$SCRATCH/f3-c2.txt"
```
Message: `Give r_mean R's NaN for a line with no values` + body (colMeans of zero rows is 0/0 in R; the replay returned 0.0 because an empty sum and a zero sum looked the same; getDN.means relies on it for empty nuggets).

- [ ] **Step 10: Move the assignment check** from `refine.py` to `_validate.py` as `check_assignments`, unchanged except the name, its docstring and an `a.size` guard (so zero rows give no ids instead of numpy's reduction error):

```python
def check_assignments(dn, n: int, p: int) -> np.ndarray:
    """dn's assignments as 0-based nugget ids (intp), after checking that dn describes n rows
    with p columns."""
    if dn.centers.ndim != 2 or dn.centers.shape[1] != p:
        raise ValueError(f"dn has centers of shape {dn.centers.shape}, but x has {p} columns")
    k = dn.centers.shape[0]
    a = dn.assignments
    if a.shape != (n,):
        raise ValueError(
            f"dn.assignments must have one entry per row of x ({n}), got shape {a.shape}"
        )
    if a.dtype.kind not in "iu" and (a.dtype.kind != "f" or not np.array_equal(a, np.floor(a))):
        raise ValueError("dn.assignments must be whole numbers (nugget ids)")
    if a.size and (a.min() < 0 or a.max() >= k):
        raise ValueError(
            f"dn.assignments must be nugget ids in [0, {k}), got values from {a.min()} to {a.max()}"
        )
    return a.astype(np.intp)
```

In `refine.py`: delete `_checked_assignments`, import `check_assignments` from `datanuggets._validate`, and call `check_assignments(dn, n, p)` where `_checked_assignments(dn, n, p)` was.

Run: `$UV run pytest tests/test_refine.py -q`
Expected: all pass (the refine tests pin the messages).

- [ ] **Step 11: Add `means_inputs` to `tests/rfixtures.py`**

```python
def means_inputs(means: Case, name: str) -> tuple[np.ndarray, DataNuggets]:
    """x and the nugget set a get_dn_means fixture holds R's means for under `name`, read from
    the create_dn or refine_dn case it came from (assignments made 0-based)."""
    path = FIXTURE_ROOT / means.args["source"] / name
    case = Case(path, json.loads((path / "manifest.json").read_text(encoding="utf-8")))
    pre = means.args["prefix"]
    dn = DataNuggets(
        centers=case.array(pre + "centers"),
        weights=case.array(pre + "weights"),
        scales=case.array(pre + "scales"),
        assignments=case.array(pre + "assignments") - 1,
    )
    return case.array("x"), dn
```

- [ ] **Step 12: Write the failing tests** (`tests/test_means.py`)

```python
"""get_dn_means against R's getDN.means on every nugget set in the create and refine fixtures."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from datanuggets import get_dn_means
from rfixtures import load_group, means_inputs

MEANS = load_group("get_dn_means")
PAIRS = [(means, name) for means in MEANS for name in means.manifest["arrays"]]
X, DN = means_inputs(
    next(m for m in MEANS if m.id == "create_dn"), "faithful_euclidean_mean"
)


@pytest.mark.parametrize(("means", "name"), PAIRS, ids=[f"{m.id}-{n}" for m, n in PAIRS])
def test_get_dn_means_matches_r(means, name):
    x, dn = means_inputs(means, name)
    want = means.array(name)
    k, p = dn.centers.shape
    if p == 1 and k > 1:
        # R's t(sapply(...)) turns one column's means into a single row
        assert want.shape == (1, k)
        want = want.T
    np.testing.assert_array_equal(get_dn_means(x, dn), want)


def test_fixtures_cover_empty_nuggets_and_one_column():
    sets = [means_inputs(m, n)[1] for m, n in PAIRS]
    assert any((dn.weights == 0).any() for dn in sets)
    assert any(dn.centers.shape[1] == 1 and dn.n_nuggets > 1 for dn in sets)


def test_dn_must_be_datanuggets():
    with pytest.raises(TypeError, match="dn must be a DataNuggets"):
        get_dn_means(X, {"centers": DN.centers, "assignments": DN.assignments})


def test_x_must_be_numeric_and_2d():
    with pytest.raises(ValueError, match="2-D"):
        get_dn_means(X[:, 0], DN)
    with pytest.raises(TypeError, match="numeric"):
        get_dn_means(X.astype(str), DN)


@pytest.mark.parametrize("rows", [np.arange(100), np.r_[0:272, 0:10]], ids=["fewer", "more"])
def test_x_must_have_the_rows_dn_assigns(rows):
    # R stops on fewer rows (subscript out of bounds) and silently drops extra ones
    with pytest.raises(ValueError, match="one entry per row"):
        get_dn_means(X[rows], DN)


def test_x_must_have_dn_columns():
    # R stops too: the means get dn's column names, and the lengths differ
    with pytest.raises(ValueError, match="columns"):
        get_dn_means(np.hstack([X, X[:, :1]]), DN)


def test_assignments_must_be_nugget_ids():
    for bad in (-1, DN.n_nuggets, 0.5):
        a = DN.assignments.astype(np.float64)
        a[5] = bad
        with pytest.raises(ValueError, match="nugget ids|whole numbers"):
            get_dn_means(X, replace(DN, assignments=a))


def test_x_must_be_finite():
    # R's colMeans passes NA through to that nugget's mean. The package rejects NaN everywhere.
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        get_dn_means(bad, DN)


def test_accepts_a_dataframe():
    frame = pd.DataFrame({"eruptions": X[:, 0], "waiting": X[:, 1].astype(np.int64)})
    np.testing.assert_array_equal(get_dn_means(frame, DN), get_dn_means(X, DN))
```

- [ ] **Step 13: Run them**

Run: `$UV run pytest tests/test_means.py -q`
Expected: collection error, `ImportError: cannot import name 'get_dn_means'`.

- [ ] **Step 14: Write `src/datanuggets/means.py`**

```python
"""Port of R's getDN.means (datanugget 1.5.0)."""

from __future__ import annotations

import numpy as np

from datanuggets._rcompat import r_mean
from datanuggets._validate import as_matrix, check_assignments
from datanuggets.result import DataNuggets


def get_dn_means(x, dn: DataNuggets) -> np.ndarray:
    """The column means of each nugget's rows of x, as a (k, p) float64 array.

    Row i is the mean of the rows assigned to nugget i, whatever center method built dn. An
    empty nugget gets a row of NaN, as in R. Each mean is R's colMeans, summed in long double,
    so the result matches R bit for bit. R returns a data frame, transposed when x has one
    column. This returns k rows either way.
    """
    x = as_matrix(x)
    if not isinstance(dn, DataNuggets):
        raise TypeError(
            f"dn must be a DataNuggets from create_dn or refine_dn, not {type(dn).__name__}"
        )
    n, p = x.shape
    assignments = check_assignments(dn, n, p)
    k = dn.centers.shape[0]
    order = np.argsort(assignments, kind="stable")  # R's which(): each nugget's rows in order
    bounds = np.concatenate(([0], np.cumsum(np.bincount(assignments, minlength=k))))
    means = np.empty((k, p))
    for i in range(k):
        means[i] = r_mean(x[order[bounds[i] : bounds[i + 1]]], axis=0)
    return means
```

In `src/datanuggets/__init__.py` add `from datanuggets.means import get_dn_means` and `"get_dn_means"` to `__all__` (kept sorted).

- [ ] **Step 15: Run them**

Run: `$UV run pytest tests/test_means.py -q`
Expected: all pass.

- [ ] **Step 16: Full suite and lint, then commit**

Run: `$UV run pytest -q -n auto` then `$UV run ruff check . && $UV run ruff format --check .`
Expected: all pass, ruff clean.

```bash
git add src/datanuggets/means.py src/datanuggets/__init__.py src/datanuggets/_validate.py \
  src/datanuggets/refine.py tests/rfixtures.py tests/test_means.py
git commit -F "$SCRATCH/f3-c3.txt"
```
Message: `Port getDN.means as get_dn_means, exact against R` + body (stable argsort matches which() row order, so the long-double sums add in R's order; the assignment check moves to _validate so refine_dn and get_dn_means share one rule; deviations: (k, 1) for one column, ValueError where R drops extra rows).

### Task 2: arithmetic_complexity

**Files:**
- Create: `r/fixtures_ac.R`
- Create: `tests/fixtures/ac/grid/` (generated)
- Create: `src/datanuggets/complexity.py`
- Modify: `src/datanuggets/__init__.py`
- Modify: `tests/rfixtures.py` (add `exact_ac_count`)
- Create: `tests/test_complexity.py`

**Interfaces:**
- Consumes: `as_real`, `check_delete_percent` (`_validate`), `R_MIN`, `R_MAX` (`_defaults`), `DataNuggetWarning`, `load_group`, `RTOL`, `ATOL`.
- Produces: `datanuggets.arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2) -> float`; `datanuggets.complexity._ac_state(n: int, r: float, delete_percent: float, dn_num1: float, dn_num2: float) -> tuple[float, float, int, float, int, float]` returning `(G, R, A, DN.num1, B, DN.num2)` as R's AC holds them when it returns; `rfixtures.exact_ac_count(value, target, delete_percent) -> int`; fixture `ac/grid` with arrays `n, p, r, delete_percent, dn_num1, dn_num2, hung, warned, g, r_end, a, dn_num1_end, b, value` (one entry per call) and `shapes` (dataset name per call) in the manifest.

- [ ] **Step 1: Write `r/fixtures_ac.R`**

```r
# Fixtures for AC: its value and the loop state it returns with (G, R, A, DN.num1, B), over the
# row and column counts of real R datasets crossed with R, delete.percent and DN.num1/DN.num2.
# AC reads only nrow(x) and ncol(x). Some arguments send AC's loops into a cycle, so every call
# runs under a one-second limit, and a call that hits it is recorded as never returning.
source("r/fixture_io.R")

# R 4.6.1's AC accepts only a plain data.frame: class(x) %in% ... has length 2 for a matrix
frame <- function(d) {
  d <- as.data.frame(d)
  stopifnot(identical(class(d), "data.frame"))
  d
}
shapes <- lapply(list(faithful = faithful, iris = iris[, 1:4], quakes = quakes,
                      volcano = volcano, EuStockMarkets = EuStockMarkets,
                      airquality = airquality, women = women, trees = trees), frame)

pairs <- function(...) do.call(rbind, lapply(list(...), function(v) data.frame(dn1 = v[1], dn2 = v[2])))
main <- merge(expand.grid(shape = names(shapes), R = c(100, 150, 250.7, 1000, 5000, 10000),
                          dp = c(0.01, 0.1, 0.3, 0.5, 0.9, 0.99), stringsAsFactors = FALSE),
              pairs(c(10^4, 2000), c(100, 20), c(100.9, 20.2), c(60, 60), c(1000, 1), c(3, 0),
                    c(0, 0), c(0, -1), c(5000, 4999), c(7, 3), c(2, 1), c(1, 1)))
# Cycles: a DN.num2 of -2 or less traps loop 2 between 0 and -1, a target DN.num1/G below -1 or
# strictly between -1 and 0 traps loop 1 the same way, and at 2^60 a cut that floors to 0
# leaves DN.num1 - 1 == DN.num1
hangs <- merge(expand.grid(shape = c("faithful", "women"), R = c(100, 5000), dp = c(0.1, 0.5),
                           stringsAsFactors = FALSE),
               pairs(c(1, -3), c(-5, -10), c(-0.5, -1)))
big <- data.frame(shape = "faithful", R = 5000, dp = 1e-20, dn1 = 2^60, dn2 = 0)
grid <- rbind(main, hangs, big)

ac_state <- NULL
trace("AC", where = asNamespace("datanugget"), print = FALSE, exit = quote(
  if (exists("my.AC", inherits = FALSE))
    ac_state <<- c(G = G, R = R, A = A, DN.num1 = DN.num1, B = B, value = my.AC)))

call_ac <- function(x, R, dp, dn1, dn2) {
  ac_state <<- NULL
  warned <- FALSE
  setTimeLimit(elapsed = 1, transient = TRUE)
  on.exit(setTimeLimit(elapsed = Inf))
  out <- tryCatch(withCallingHandlers(datanugget::AC(x, R, dp, dn1, dn2),
    warning = function(w) {
      if (conditionMessage(w) != "NaNs produced") stop("unexpected warning: ", conditionMessage(w))
      warned <<- TRUE
      invokeRestart("muffleWarning")
    }), error = function(e) {
      if (!grepl("reached elapsed time limit", conditionMessage(e))) stop(e)
      NULL
    })
  if (is.null(out)) {
    return(c(hung = 1, warned = 0, G = NA, R = NA, A = NA, DN.num1 = NA, B = NA, value = NA))
  }
  stopifnot(identical(out, unname(ac_state[["value"]])))
  c(hung = 0, warned = as.numeric(warned), ac_state)
}

rows <- t(vapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]
  call_ac(shapes[[g$shape]], g$R, g$dp, g$dn1, g$dn2)
}, numeric(8)))
untrace("AC", where = asNamespace("datanugget"))

write_case("ac", "grid", no_args, list(
  n = vapply(grid$shape, function(s) nrow(shapes[[s]]), integer(1), USE.NAMES = FALSE),
  p = vapply(grid$shape, function(s) ncol(shapes[[s]]), integer(1), USE.NAMES = FALSE),
  r = grid$R, delete_percent = grid$dp, dn_num1 = grid$dn1, dn_num2 = grid$dn2,
  hung = as.integer(rows[, "hung"]), warned = as.integer(rows[, "warned"]),
  g = rows[, "G"], r_end = rows[, "R"], a = rows[, "A"], dn_num1_end = rows[, "DN.num1"],
  b = rows[, "B"], value = rows[, "value"]), extra = list(shapes = grid$shape))
cat(nrow(grid), "calls,", sum(rows[, "hung"]), "never returned,", sum(rows[, "warned"]),
    "warned\n")
```

- [ ] **Step 2: Generate the fixture**

Run: `"$RS" r/fixtures_ac.R`
Expected: `3481 calls, 19 never returned, <w> warned` with `<w>` > 0, exit 0, in under a minute. (19 = 8 rows of `(1, -3)`, 8 of `(-5, -10)`, the 2 faithful `R = 100` rows of `(-0.5, -1)` whose target is -1/3, and the `2^60` row.) If the count differs, stop and read which rows hung before going on.

- [ ] **Step 3: Commit the generator and fixture**

```bash
git add r/fixtures_ac.R tests/fixtures/ac
git commit -F "$SCRATCH/f3-c4.txt"
```
Message: `Record R's AC and its loop state over real dataset shapes` + body (trace on exit captures A and B, which the log10 value alone can hide; the time limit records where R never returns, so Python's cycle check can be held to exactly those calls).

- [ ] **Step 4: Add `exact_ac_count` to `tests/rfixtures.py`** (add `import math` and `from fractions import Fraction`)

```python
def exact_ac_count(value, target, delete_percent) -> int:
    """One of AC's while loops run in exact rational arithmetic, returning R's counter. It's
    what the code says without double rounding. Where it disagrees with R's count, only a port
    doing R's double operations in R's order can match."""
    value, target, dp = Fraction(value), Fraction(target), Fraction(delete_percent)
    count = 1
    while value != target:
        if count > 100_000:
            raise AssertionError("the exact loop did not end")
        cut = math.floor(dp * value)
        if value - cut < target:
            value = target
        elif cut == 0:
            value -= 1
        else:
            value -= cut
        count += 1
    return count
```

- [ ] **Step 5: Write the failing tests** (`tests/test_complexity.py`)

```python
"""arithmetic_complexity against R's AC over the row and column counts of real R datasets."""

import math
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from datanuggets import DataNuggetWarning, arithmetic_complexity
from datanuggets.complexity import _ac_state
from rfixtures import ATOL, RTOL, exact_ac_count, load_group

(GRID,) = load_group("ac")
ARGS = ("r", "delete_percent", "dn_num1", "dn_num2")
GOOD = {"r": 5000, "delete_percent": 0.1, "dn_num1": 10_000, "dn_num2": 2000}


def _rows(hung: bool) -> list[dict]:
    names = ("n", "p", *ARGS, "warned", "g", "r_end", "a", "dn_num1_end", "b", "value")
    cols = {name: GRID.array(name).tolist() for name in names}
    return [
        {name: col[i] for name, col in cols.items()}
        for i in np.flatnonzero(GRID.array("hung") == int(hung)).tolist()
    ]


def _call(row: dict) -> float:
    x = np.broadcast_to(0.0, (row["n"], row["p"]))
    return arithmetic_complexity(x, *(row[a] for a in ARGS))


def test_counts_value_and_warning_match_r():
    rows, bad = _rows(hung=False), []
    for row in rows:
        state = _ac_state(row["n"], *(row[a] for a in ARGS))[:5]
        want = (row["g"], row["r_end"], row["a"], row["dn_num1_end"], row["b"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = _call(row)
        warned = sum(issubclass(w.category, DataNuggetWarning) for w in caught)
        close = np.isclose(value, row["value"], rtol=RTOL, atol=ATOL, equal_nan=True)
        if state != want or not close or warned != row["warned"]:
            bad.append((row, state, value, warned))
    assert not bad, f"{len(bad)} of {len(rows)} calls differ from R, first: {bad[:2]}"
    assert any(row["warned"] for row in rows) and any(row["value"] == -math.inf for row in rows)


def test_raises_where_r_never_returns():
    rows = _rows(hung=True)
    assert rows
    for row in rows:
        with pytest.raises(ValueError, match="never returns"):
            _call(row)


def test_grid_needs_r_double_rounding():
    """In some rows double rounding changes R's count A, so a port that ran the loop in exact
    arithmetic would fail test_counts_value_and_warning_match_r."""
    differs = 0
    for row in _rows(hung=False):
        r, dn1 = math.floor(row["r"]), math.floor(row["dn_num1"])
        target = Fraction(dn1, -(-row["n"] // r))
        differs += exact_ac_count(r, target, row["delete_percent"]) != row["a"]
    assert differs > 0


def test_reads_only_the_shape_of_any_2d_input():
    # R 4.6.1's AC stops on a matrix or a data.table and takes any plain data frame, text too
    (want,) = [
        row["value"]
        for row in _rows(hung=False)
        if (row["n"], row["p"]) == (272, 2) and all(row[a] == GOOD[a] for a in ARGS)
    ]
    frame = pd.DataFrame({"label": ["a"] * 272, "count": range(272)})
    for x in (np.zeros((272, 2)), frame, [[0.0, 0.0]] * 272):
        assert arithmetic_complexity(x, **GOOD) == pytest.approx(want, rel=RTOL, abs=ATOL)


def test_numpy_scalars_act_like_python_numbers():
    x = np.zeros((272, 2))
    got = arithmetic_complexity(x, np.int64(5000), np.float64(0.1), np.int64(10_000), np.int32(2000))
    assert got == arithmetic_complexity(x, **GOOD)


BAD_ARGS = [
    ({"r": "5000"}, TypeError, "r must be a number"),
    ({"r": True}, TypeError, "r must be a number"),
    ({"r": 99.5}, ValueError, "r must be within"),
    ({"r": 10_000.5}, ValueError, "r must be within"),
    ({"r": math.nan}, ValueError, "r must be within"),
    ({"delete_percent": "0.1"}, TypeError, "delete_percent must be a number"),
    ({"delete_percent": 0}, ValueError, "delete_percent must be within"),
    ({"delete_percent": 1}, ValueError, "delete_percent must be within"),
    ({"dn_num1": "1e4"}, TypeError, "dn_num1 must be a number"),
    ({"dn_num2": None}, TypeError, "dn_num2 must be a number"),
    # R stops on NaN, stops or loops on most infinite values and returns Inf when both are Inf
    ({"dn_num1": math.nan}, ValueError, "dn_num1 must be finite"),
    ({"dn_num1": math.inf, "dn_num2": math.inf}, ValueError, "dn_num1 must be finite"),
    ({"dn_num2": -math.inf}, ValueError, "dn_num2 must be finite"),
    # R compares before flooring, so 100.7 > 100.5 stops though both floor to 100
    ({"dn_num1": 100.5, "dn_num2": 100.7}, ValueError, "dn_num2 must not exceed dn_num1"),
]


@pytest.mark.parametrize(("kwargs", "exc", "match"), BAD_ARGS)
def test_rejects_what_r_rejects(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        arithmetic_complexity(np.zeros((272, 2)), **{**GOOD, **kwargs})


def test_x_must_be_2d_with_rows():
    with pytest.raises(ValueError, match="2-D"):
        arithmetic_complexity(np.zeros(272), **GOOD)
    # R returns NaN for a data frame with no rows
    with pytest.raises(ValueError, match="at least one row"):
        arithmetic_complexity(np.zeros((0, 2)), **GOOD)
    # and -Inf, log10(0), for one with no columns, which is kept
    assert arithmetic_complexity(np.zeros((272, 0)), **GOOD) == -math.inf
```

- [ ] **Step 6: Run them**

Run: `$UV run pytest tests/test_complexity.py -q`
Expected: collection error, `ImportError: cannot import name 'arithmetic_complexity'`.

- [ ] **Step 7: Write `src/datanuggets/complexity.py`**

```python
"""Port of R's AC (datanugget 1.5.0): the arithmetic complexity of create_dn."""

from __future__ import annotations

import math
import warnings

import numpy as np

from datanuggets._defaults import R_MAX, R_MIN
from datanuggets._validate import as_real, check_delete_percent
from datanuggets.exceptions import DataNuggetWarning


def arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2) -> float:
    """R's AC: the arithmetic complexity of create_dn(x, r=r, ...) on a log10 scale.

    Only x's shape is read, and there are no defaults, as in R. AC counts G = ceiling(n / r)
    chunks thinned from r rows to dn_num1 / G centers in A passes and dn_num1 thinned to
    dn_num2 in B passes, each pass costing a pairwise distance matrix, plus n + 1 rows measured
    against dn_num2 centers, all times the column count. A and B come from R's two while loops,
    run here in the same double arithmetic, so they match R's counts exactly. Some arguments
    make those loops cycle and R never returns, a dn_num2 of -2 or less for one. This raises
    ValueError for them.
    """
    shape = np.shape(x)
    if len(shape) != 2:
        raise ValueError(f"x must be 2-D (a matrix or data frame), got {len(shape)}-D")
    n, p = shape
    if n == 0:
        raise ValueError("x must have at least one row")
    r = as_real("r", r)
    if not R_MIN <= r <= R_MAX:
        raise ValueError(f"r must be within [{R_MIN}, {R_MAX}], got {r!r}")
    delete_percent = check_delete_percent(delete_percent)
    dn_num1 = as_real("dn_num1", dn_num1)
    dn_num2 = as_real("dn_num2", dn_num2)
    for name, value in (("dn_num1", dn_num1), ("dn_num2", dn_num2)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if dn_num2 > dn_num1:
        raise ValueError("dn_num2 must not exceed dn_num1")
    g, r, a, dn_num1, b, dn_num2 = _ac_state(n, r, delete_percent, dn_num1, dn_num2)
    count = p * (g * ((r * (r - 1) / 2) * a) + (dn_num1 * (dn_num1 - 1) / 2) * b + (n + 1) * dn_num2)
    return _r_log10(count)


def _ac_state(
    n: int, r: float, delete_percent: float, dn_num1: float, dn_num2: float
) -> tuple[float, float, int, float, int, float]:
    """AC after its argument checks: floor R, DN.num1 and DN.num2, set G and run both loops.
    Returns (G, R, A, DN.num1, B, DN.num2) as R holds them when it returns."""
    r, dn_num1, dn_num2 = (float(math.floor(v)) for v in (r, dn_num1, dn_num2))
    g = float(math.ceil(n / r))
    r, a = _shrink("r", r, dn_num1 / g, delete_percent)
    dn_num1, b = _shrink("dn_num1", dn_num1, dn_num2, delete_percent)
    return g, r, a, dn_num1, b, dn_num2


def _shrink(name: str, value: float, target: float, delete_percent: float) -> tuple[float, int]:
    """One of AC's while loops: cut value by floor(delete_percent * value) until it equals
    target, jumping to target when a cut would pass it and stepping by 1 when the cut floors to
    0. Returns the final value and R's counter, 1 plus the number of passes. Every operation is
    R's double arithmetic in R's order, because the rounding of delete_percent * value and of
    value - (value - target) changes the count on real inputs.

    The loop is deterministic, so a value that comes back means R never returns. Brent's cycle
    check compares each value with one saved at power-of-two intervals, in constant memory
    however long the loop runs.
    """
    count = 1
    saved, horizon, steps = value, 1, 0
    while value != target:
        cut = float(math.floor(delete_percent * value))
        if value - cut < target:
            value = value - (value - target)
        elif cut == 0:
            value = value - 1
        else:
            value = value - cut
        count += 1
        if value == saved:
            raise ValueError(
                f"R's AC never returns for these arguments: {name} cycles through {value!r}"
                f" without reaching {target!r}"
            )
        steps += 1
        if steps == horizon:
            saved, horizon, steps = value, 2 * horizon, 0
    return value, count


def _r_log10(v: float) -> float:
    """R's log10: -Inf at 0, and NaN with R's warning below 0."""
    if v < 0:
        warnings.warn("NaNs produced", DataNuggetWarning, stacklevel=3)
        return math.nan
    if v == 0:
        return -math.inf
    return math.log10(v)  # NaN and Inf pass through, as in R
```

In `src/datanuggets/__init__.py` add `from datanuggets.complexity import arithmetic_complexity` and `"arithmetic_complexity"` to `__all__` (kept sorted).

- [ ] **Step 8: Run them**

Run: `$UV run pytest tests/test_complexity.py -q`
Expected: all pass.

- [ ] **Step 9: Full suite and lint, then commit**

Run: `$UV run pytest -q -n auto` then `$UV run ruff check . && $UV run ruff format --check .`
Expected: all pass, ruff clean.

```bash
git add src/datanuggets/complexity.py src/datanuggets/__init__.py tests/rfixtures.py \
  tests/test_complexity.py
git commit -F "$SCRATCH/f3-c5.txt"
```
Message: `Port AC as arithmetic_complexity with R's loop arithmetic` + body (double rounding changes A on real grid rows, so the loops copy R's operations in order; Brent's check raises exactly where R never returns; deviations: any 2-D x, finite dn arguments, rows required).

### Task 3: Validation report, README and spec

**Files:**
- Modify: `scripts/parity_report.py`
- Create: `docs/validation/get_dn_means.md`, `docs/validation/arithmetic_complexity.md` (generated)
- Modify: `README.md`
- Modify: `docs/specs/2026-09-29-datanuggets-python-port-design.md` (section 2 tiers, section 10 rows)

**Interfaces:**
- Consumes: `get_dn_means`, `arithmetic_complexity`, `_ac_state`, `means_inputs`, `exact_ac_count`, `load_group`.
- Produces: two generated reports whose numbers the README quotes.

- [ ] **Step 1: Extend `scripts/parity_report.py`**

`verdict` compares with `equal_nan=True` (empty nuggets have NaN means, and the create and refine outputs hold no NaN, so their reports don't change). Add:

```python
def means_report() -> str:
    rows = []
    groups = load_group("get_dn_means")
    for means in groups:
        for name in means.manifest["arrays"]:
            x, dn = means_inputs(means, name)
            want = means.array(name)
            k, p = dn.centers.shape
            if p == 1 and k > 1:
                want = want.T
            empty = int((dn.weights == 0).sum())
            rows.append(
                f"| {means.id} | {name} | {x.shape[0]} x {p} | {k} | {empty} |"
                f" {verdict(get_dn_means(x, dn), want)} |"
            )
    v = groups[0].manifest["versions"]
    header = (
        "# get_dn_means parity with R\n\n"
        "Generated by `scripts/parity_report.py` from the fixtures in"
        f" `tests/fixtures/get_dn_means/` ({v['R']}, datanugget {v['datanugget']},"
        f" {v['platform']}, long double: {v['longdouble_digits']}-bit significand). Each row is"
        " one nugget set from the create_dn or refine_dn fixtures: create.DN's output, the"
        " nuggets refine.DN was given, or refine.DN's output. R's getDN.means returns one"
        ' column\'s means as a single row, and those are compared transposed. "yes" means'
        " bit-identical (`np.array_equal`, with NaN equal to NaN for empty nuggets).\n\n"
        "| set | case | data | nuggets | empty | means |\n"
        "|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"


def ac_report() -> str:
    (grid,) = load_group("ac")
    cols = {
        name: grid.array(name).tolist()
        for name in ("n", "p", "r", "delete_percent", "dn_num1", "dn_num2", "hung", "g", "r_end",
                     "a", "dn_num1_end", "b", "value")
    }
    stats: dict[str, list[int]] = {}
    worst: dict[str, float] = {}
    for i, shape in enumerate(grid.manifest["shapes"]):
        row = {name: col[i] for name, col in cols.items()}
        args = (row["r"], row["delete_percent"], row["dn_num1"], row["dn_num2"])
        s = stats.setdefault(shape, [0] * 6)
        s[0] += 1
        x = np.broadcast_to(0.0, (row["n"], row["p"]))
        if row["hung"]:
            s[4] += 1
            try:
                arithmetic_complexity(x, *args)
            except ValueError:
                s[5] += 1
            continue
        state = _ac_state(row["n"], *args)[:5]
        s[1] += state == (row["g"], row["r_end"], row["a"], row["dn_num1_end"], row["b"])
        r, dn1 = math.floor(row["r"]), math.floor(row["dn_num1"])
        exact = exact_ac_count(r, Fraction(dn1, -(-row["n"] // r)), row["delete_percent"])
        s[2] += exact != row["a"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = arithmetic_complexity(x, *args)
        want = row["value"]
        same = got == want or (math.isnan(got) and math.isnan(want))
        s[3] += same
        if not same:
            worst[shape] = max(worst.get(shape, 0.0), abs(got - want) / abs(want))
    rows = [
        f"| {shape} | {cols['n'][grid.manifest['shapes'].index(shape)]} x"
        f" {cols['p'][grid.manifest['shapes'].index(shape)]} | {s[0]} | {s[0] - s[4]} | {s[1]} |"
        f" {s[2]} | {s[3]} | {f'{worst[shape]:.1e}' if shape in worst else '0'} | {s[4]} |"
        f" {s[5]} |"
        for shape, s in stats.items()
    ]
    v = grid.manifest["versions"]
    header = (
        "# arithmetic_complexity parity with R\n\n"
        "Generated by `scripts/parity_report.py` from `tests/fixtures/ac/grid/`"
        f" ({v['R']}, datanugget {v['datanugget']}, {v['platform']}). R's AC reads only the row"
        " and column counts of x, so the grid takes them from R datasets and crosses them with"
        " R, delete.percent and DN.num1/DN.num2. \"same counts\" means G, R, A, DN.num1 and B"
        " equal R's traced values exactly. \"exact arithmetic differs\" counts the calls where"
        " running AC's first loop in exact rational arithmetic gives another A than R, so only"
        " R's double operations in R's order can match there. The value goes through the"
        " platform's log10 and the tests hold it to 1e-8 relative. R never returned from some"
        " calls within a second, and arithmetic_complexity must raise ValueError for exactly"
        " those.\n\n"
        "| data | shape | calls | R returned | same counts | exact arithmetic differs |"
        " value bit-identical | max rel. diff | R never returned | Python raised |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"
```

Imports gain `math`, `Fraction`, `get_dn_means`, `arithmetic_complexity`, `_ac_state`, `means_inputs`, `exact_ac_count`. `main()` loops over `create_dn`, `refine_dn`, `get_dn_means`, `arithmetic_complexity`.

- [ ] **Step 2: Generate the reports**

Run: `$UV run python scripts/parity_report.py` then `git status --short docs/validation` and `git diff --stat docs/validation`
Expected: two new files. `create_dn.md` and `refine_dn.md` show no content diff (restore them with `git checkout --` if they show as modified). Every means row says `yes`, and for each dataset "same counts" equals "R returned" and "Python raised" equals "R never returned".

- [ ] **Step 3: Commit**

```bash
git add scripts/parity_report.py docs/validation/get_dn_means.md docs/validation/arithmetic_complexity.md
git commit -F "$SCRATCH/f3-c6.txt"
```
Message: `Report get_dn_means and arithmetic_complexity parity with R`.

- [ ] **Step 4: README.** Status lists five functions and puts the phase 2 helpers next, in the spec's order. "Differences from R" gains the argument mappings, a paragraph per function with each deviation, and one line that every function rejects NaN or infinite `x`, where R passes NA through (`create.DN` gives NA centers, `getDN.means` NA means). "Parity with R" gains one paragraph per function whose numbers come only from the generated reports. Humanizer lint over the changed lines: no dashes, no semicolons, no bold or italic in prose.

- [ ] **Step 5: Spec.** In section 2, the `getDN.means` tier becomes `Exact` and the `AC` tier becomes `Exact (loop counts) + Numeric (value)`. Append rows to the section 10 table:

| Item | R behavior | Python |
|---|---|---|
| `getDN.means` with one column | `t(sapply(...))` returns a 1 x k data frame | `(k, 1)` array, documented |
| `getDN.means` with other rows than the assignments | fewer rows stop (subscript out of bounds), extra rows are silently ignored | `ValueError` on any mismatch, the check `refine_dn` uses |
| NA or NaN in `x` | passed through (`create.DN` gives NA centers, `getDN.means` NA means) | rejected with `ValueError` by every function |
| `AC` on a matrix or data.table | `class(x) %in% ...` has length 2, so R 4.6.1 stops | any 2-D input, listed as a deviation that fixes an upstream bug |
| `AC` loops that cycle (`DN.num2 <= -2`, a target `DN.num1/G` below -1 or strictly between -1 and 0, `2^60` with a cut that floors to 0) | never returns | `ValueError` from a cycle check |
| `AC` with a non-finite `DN.num1`/`DN.num2` or an `x` with no rows | stops, loops, returns Inf (both Inf) or NaN (no rows) | `ValueError` |

- [ ] **Step 6: Gates and commit**

Run: `$UV run pytest -q -n auto`, `$UV run ruff check .`, `$UV run ruff format --check .` (it checks the README's Python blocks too).
Expected: all pass, ruff clean.

```bash
git add README.md docs/specs/2026-09-29-datanuggets-python-port-design.md
git commit -F "$SCRATCH/f3-c7.txt"
```
Message: `Document get_dn_means and arithmetic_complexity and their R deviations`.
