# Feature 2: refine_dn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `refine.DN` from R datanugget 1.5.0, with the Hartigan-Wong k-means it depends on, so that with R's random draws replayed every kmeans call, assignment, weight, center and scale matches R exactly.

**Architecture:** Same oracle setup as Feature 1: R scripts in `r/` write frozen fixtures, recording every `sample.int` draw and now every `stats::kmeans` call through `trace()`. Three new private layers carry R's arithmetic: `_ldouble` (x87 long double replayed with Python integers), `_rstats` (R's `sum`, `mean`, `median`, `mad`, type-7 `quantile`, `cov`, `%*%` bit for bit, plus R's `dsyevr` call for `eigen`), and `_hartigan_wong` (a line-by-line port of `kmns.f` and the `stats::kmeans` start logic). `refine_dn` is assembled from them. Eigenvalues come from scipy's LAPACK, so they match R to about 1e-12 relative rather than bit for bit; everything that reaches the output (memberships, colMeans centers, long-double scales) is exact.

**Tech Stack:** Python 3.12+ (dev on 3.14), numpy >= 2.0, scipy >= 1.13 (new runtime dep, spec section 11), joblib, optional tqdm/pandas, pytest, uv, ruff; R 4.6.1 with datanugget 1.5.0.

**Spec:** `docs/specs/2026-09-29-datanuggets-python-port-design.md` (sections 4, 5 and 6 "refine_dn" + "Hartigan-Wong 2-means" govern this plan; feature 2 = refine_dn, per the feature-by-feature build order used for Feature 1).

**Execution:** RON2K asked to "finish it e2e while i check feature 1". That selects native execution (superpowers:executing-plans) with no review stop between plan and build: plan, build every task with TDD, one Opus whole-branch review, one fix pass, push `feat/refine-dn`, open a PR against `main`, do not merge.

## Global Constraints

- Branch `feat/refine-dn` from `main` (a92ad2a). `main` advances only by PR; merging this PR is RON2K's call.
- Public signature (spec section 4): `refine_dn(x, dn, ev_tol=0.9, max_splits=5, min_nugget_size=2, shape_split=False, min_shape_size=10, delta=2, nstart=25, seed=291102, n_jobs=1, progress=False) -> DataNuggets`.
- Algorithm defaults equal R's exactly (refine.DN 1.5.0): EV.tol 0.9, max.splits 5, min.nugget.size 2, shape.split FALSE, min.shape.size 10, delta 2, nstart 25, seed 291102; split kmeans uses `centers = 2, iter.max = 1000`; `mad` constant 1.4826.
- Python indices are 0-based everywhere; tests add 1 when comparing with R. Nugget ids are positions.
- Parity tiers: Exact = `np.testing.assert_array_equal`; Numeric = `assert_allclose(rtol=1e-8, atol=1e-10)`. Exact: assignments, weights, centers, scales, kmeans clusters, kmeans inputs for EV splits, every `_rstats` function except eigen. Numeric: eigenvalues, the first eigenvector (up to sign), kmeans inputs for shape splits (projections on that eigenvector).
- Messages go to `logging.getLogger("datanuggets")` at INFO with R's exact `message()` text. R's `warning()`s become `DataNuggetWarning` with R's text. Rejected splits log one DEBUG line with the reason.
- `n_jobs > 1` must give exactly the `n_jobs=1` result. `ReplayDraws` needs `n_jobs=1`.
- Runtime deps after this feature: numpy >= 2.0, scipy >= 1.13, joblib >= 1.4. Extras unchanged.
- Fixtures: raw `.bin` + `manifest.json` via `r/fixture_io.R`; `pytest` never needs R. Total fixtures stay under 25 MB.
- Commits: one logical change each, imperative subject <= 70 chars plus a why/what body, written to a message file and committed with `git commit -F`. Never add `Claude-Session:`, `Co-Authored-By`, or "Generated with" lines (`git log --all --grep=Claude-Session` must stay 0). Stage files by name, never `git add -A`. Do not stage `docs/validation/create_dn.md` unless its content changed.
- Repo root holds only toolchain files. New code in `src/datanuggets/`, tests in `tests/`, R in `r/`, scripts in `scripts/`, docs in `docs/`.
- Shell: Git Bash on Windows 11, run from the repo root. `uv` is `/c/Users/thegr/AppData/Local/hermes/bin/uv.exe` (alias `UV` below); Rscript is `"$LOCALAPPDATA/Programs/R/R-4.6.1/bin/Rscript.exe"` (alias `RS` below).
- Code blocks tagged `file=<path>` are written to disk verbatim by `extract.py` (scratchpad helper); `mode=append` blocks are appended.

## Review Focus

1. **Duplicate rows inside a nugget** (discretized data like iris petals): kmeans draws its starts from `unique(x)`, so the number of draws depends on exact row equality with 0 == -0. A user expects the same split R makes. Pinned by the `petal_*` refine fixtures, `petal_k2_n1_dup`/`petal_k2_n25` kmeans fixtures and `unique_*` rstats fixtures (Tasks 2 to 4).
2. **`n_jobs > 1`** with an int seed and with a numpy Generator: must equal the `n_jobs=1` result exactly. Test `test_n_jobs_does_not_change_the_result` (Task 4).
3. **A DataNuggets from different data**: wrong assignment length, ids out of range, or a different column count must raise `ValueError` instead of refining garbage. Tests `test_assignments_must_cover_every_row`, `test_assignments_must_be_nugget_ids`, `test_dn_from_data_with_other_columns_is_rejected` (Task 4).
4. **Refining an already refined result**: must behave like R on its own output. Fixture `faithful_chain` (Task 4).
5. **p = 1 data**: shape splits are skipped and EV2 = EV1 = the variance, exactly. Fixtures `faithful_1d_*` (Task 4) and `cov_one_col` (Task 2).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/datanuggets/result.py` (rewrite) | Frozen results store read-only copies; value equality; unhashable |
| `src/datanuggets/_draws.py` (rewrite) | Draws protocol gains `sample(n, size)` and `fork()` |
| `src/datanuggets/_ldouble.py` (new) | x87 long double arithmetic on Python integers |
| `src/datanuggets/_rstats.py` (new) | R's sum/mean/median/mad/quantile/cov/%*% bit for bit, eigen via LAPACK dsyevr |
| `src/datanuggets/_rcompat.py` (append) | `unique_first`: R's `unique()` on matrix rows |
| `src/datanuggets/_hartigan_wong.py` (new) | `kmns` (kmns.f port) and `kmeans` (stats::kmeans start logic) |
| `src/datanuggets/refine.py` (new) | `refine_dn`, `nugget_features` |
| `src/datanuggets/_defaults.py` (append) | refine.DN defaults, kmeans iter.max, mad constant |
| `src/datanuggets/_validate.py` (append) | `check_n_jobs` shared by create and refine |
| `src/datanuggets/create.py` (edit) | use `check_n_jobs` |
| `src/datanuggets/__init__.py` (edit) | export `refine_dn` |
| `r/fixtures_rstats.R` (new) | group `rcompat_stats` |
| `r/fixtures_kmeans.R` (new) | groups `kmns`, `kmeans` |
| `r/fixtures_refine_dn.R` (new) | group `refine_dn` |
| `tests/test_result.py` (new), `tests/test_draws.py` (append), `tests/test_rstats.py`, `tests/test_hartigan_wong.py`, `tests/test_refine.py` (new) | parity and unit tests |
| `scripts/parity_report.py` (edit) | also writes `docs/validation/refine_dn.md` |
| `README.md`, `r/README.md` (edit) | status, differences, fixture scripts |

---

### Task 1: Results and draws for refine

**Files:**
- Modify: `src/datanuggets/result.py` (whole file)
- Modify: `src/datanuggets/_draws.py` (whole file)
- Create: `tests/test_result.py`
- Modify: `tests/test_draws.py` (append tests, extend imports)

**Interfaces:**
- Consumes: nothing new.
- Produces: `DataNuggets`/`CentersResult` with value `==` (NaN == NaN), `__hash__ = None`, fields stored as read-only copies. `Draws.sample(n: int, size: int) -> np.ndarray` (R `sample.int(n, size)`, 0-based) and `Draws.fork() -> Draws` on `NumpyDraws` and `ReplayDraws`.

- [ ] **Step 1: Write the failing result tests**

```python file=tests/test_result.py
import numpy as np
import pytest

from datanuggets import CentersResult, DataNuggets


def _dn(**over):
    fields = dict(
        centers=np.array([[0.0, 1.0], [np.nan, np.nan]]),
        weights=np.array([3, 0]),
        scales=np.array([0.5, 0.0]),
        assignments=np.array([0, 0, 0]),
    )
    fields.update(over)
    return DataNuggets(**fields)


def test_result_copies_the_callers_arrays():
    weights = np.array([3, 0])
    dn = _dn(weights=weights)
    weights[0] = 99
    assert dn.weights[0] == 3
    assert weights.flags.writeable


def test_result_arrays_are_read_only():
    with pytest.raises(ValueError):
        _dn().centers[0, 0] = 5.0


def test_lists_become_arrays():
    assert isinstance(_dn(weights=[3, 0]).weights, np.ndarray)


def test_equal_by_value_with_nan_centers():
    assert _dn() == _dn()
    assert _dn() != _dn(weights=np.array([2, 1]))
    assert _dn() != _dn(assignments=np.array([0, 0]))


def test_not_equal_to_other_types():
    assert _dn() != "nuggets"
    assert CentersResult(centers=np.zeros((1, 2)), kept=np.array([0])) != _dn()


def test_results_are_unhashable():
    with pytest.raises(TypeError):
        hash(_dn())
    with pytest.raises(TypeError):
        hash(CentersResult(centers=np.zeros((1, 2)), kept=np.array([0])))


def test_centers_result_equality():
    a = CentersResult(centers=np.eye(2), kept=np.array([0, 1]))
    assert a == CentersResult(centers=np.eye(2), kept=np.array([0, 1]))
    assert a != CentersResult(centers=np.eye(2), kept=np.array([1, 0]))
```

- [ ] **Step 2: Append the failing draws tests**

Add `Draws` to the `from datanuggets._draws import (...)` list at the top of `tests/test_draws.py`, then append:

```python file=tests/test_draws.py mode=append
def test_numpy_sample_draws_distinct_values_in_range():
    got = NumpyDraws(1).sample(10, 4)
    assert got.shape == (4,)
    assert len(set(got.tolist())) == 4
    assert got.min() >= 0 and got.max() < 10


def test_numpy_fork_with_int_seed_restarts_the_stream():
    draws = NumpyDraws(5)
    first = draws.fork().sample(1000, 5)
    np.testing.assert_array_equal(draws.fork().sample(1000, 5), first)
    np.testing.assert_array_equal(NumpyDraws(5).sample(1000, 5), first)


def test_numpy_fork_with_generator_spawns_reproducible_children():
    a = NumpyDraws(np.random.default_rng(3))
    b = NumpyDraws(np.random.default_rng(3))
    a1, a2 = a.fork().sample(10**6, 3), a.fork().sample(10**6, 3)
    assert not np.array_equal(a1, a2)
    np.testing.assert_array_equal(b.fork().sample(10**6, 3), a1)


def test_replay_sample_returns_zero_based_values_and_checks_the_call():
    draws = ReplayDraws([DrawRecord(10, 2, np.array([3, 7]))])
    np.testing.assert_array_equal(draws.sample(10, 2), [2, 6])
    draws.assert_exhausted()
    with pytest.raises(ReplayMismatch):
        ReplayDraws([DrawRecord(10, 2, np.array([3, 7]))]).sample(9, 2)


def test_replay_fork_shares_the_recorded_stream():
    draws = ReplayDraws([])
    assert draws.fork() is draws


def test_both_implementations_satisfy_the_protocol():
    assert isinstance(NumpyDraws(1), Draws)
    assert isinstance(ReplayDraws([]), Draws)
```

- [ ] **Step 3: Run the new tests and watch them fail**

Run: `UV run pytest tests/test_result.py tests/test_draws.py -q`
Expected: FAIL. `test_result_copies_the_callers_arrays` (ValueError: the caller's array was frozen in place), `test_equal_by_value_with_nan_centers` and `test_centers_result_equality` (ValueError: truth value of an array is ambiguous), every new draws test (AttributeError: no `sample`/`fork`). `test_results_are_unhashable` may already pass by accident (arrays are unhashable); it pins the behavior by design after Step 4.

- [ ] **Step 4: Rewrite result.py**

```python file=src/datanuggets/result.py
"""Result containers."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


def _freeze(value) -> np.ndarray:
    """A read-only copy: a result never shares memory with, or locks, the caller's array."""
    arr = np.array(value, copy=True)
    arr.setflags(write=False)
    return arr


class _ArrayFields:
    """Base for the frozen result dataclasses: every field is stored as a read-only copy, and ==
    compares the arrays by value. Dataclass equality would compare them with ==, whose truth
    value is ambiguous. NaN equals NaN, since empty nuggets have NaN centers. Like numpy arrays,
    results are unhashable."""

    __hash__ = None

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _freeze(getattr(self, f.name)))

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name), equal_nan=True)
            for f in fields(self)
        )


@dataclass(frozen=True, eq=False)
class CentersResult(_ArrayFields):
    """create_dn_centers output: the kept rows and their 0-based positions in the input."""

    centers: np.ndarray
    kept: np.ndarray


@dataclass(frozen=True, eq=False)
class DataNuggets(_ArrayFields):
    """Data nuggets: one center, weight and scale per nugget, plus each row's nugget (0-based)."""

    centers: np.ndarray
    weights: np.ndarray
    scales: np.ndarray
    assignments: np.ndarray

    @property
    def n_nuggets(self) -> int:
        return self.centers.shape[0]

    def to_frame(self):
        """R's "Data Nuggets" table: Data Nugget, Center1..Centerp, Weight, Scale (ids 0-based)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame() needs pandas: pip install 'datanuggets[frame]'") from exc
        p = self.centers.shape[1]
        df = pd.DataFrame(self.centers, columns=[f"Center{j + 1}" for j in range(p)])
        df.insert(0, "Data Nugget", np.arange(self.n_nuggets))
        df["Weight"] = self.weights
        df["Scale"] = self.scales
        return df
```

- [ ] **Step 5: Rewrite _draws.py**

```python file=src/datanuggets/_draws.py
"""Source of randomness. Users get numpy; parity tests replay the draws R recorded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class ReplayMismatch(AssertionError):
    """Python requested a random draw that R did not make, or in a different order."""


@runtime_checkable
class Draws(Protocol):
    def permutation(self, n: int) -> np.ndarray:
        """R's sample(n, n), 0-based."""
        ...

    def index(self, m: int) -> int:
        """R's sample.int(m, 1), 0-based."""
        ...

    def sample(self, n: int, size: int) -> np.ndarray:
        """R's sample.int(n, size): size distinct values from range(n), 0-based."""
        ...

    def fork(self) -> Draws:
        """The stream for one refine_dn split, where R calls set.seed(seed) again."""
        ...


class NumpyDraws:
    def __init__(self, seed: int | np.integer | np.random.Generator):
        if isinstance(seed, np.random.Generator):
            self._seed, self._rng = None, seed
        else:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

    def permutation(self, n: int) -> np.ndarray:
        return self._rng.permutation(n)

    def index(self, m: int) -> int:
        return int(self._rng.integers(m))

    def sample(self, n: int, size: int) -> np.ndarray:
        return self._rng.choice(n, size, replace=False)

    def fork(self) -> NumpyDraws:
        """An int seed restarts the same stream on every call, like R's set.seed(seed). A
        Generator hands out a new child stream per call (Generator.spawn), so forks made in the
        same order give the same streams."""
        if self._seed is not None:
            return NumpyDraws(self._seed)
        return NumpyDraws(self._rng.spawn(1)[0])


@dataclass(frozen=True)
class DrawRecord:
    n: int
    size: int
    value: np.ndarray  # 1-based, as R returned it


class ReplayDraws:
    def __init__(self, records: list[DrawRecord]):
        self._records = list(records)
        self._pos = 0

    def _next(self, n: int, size: int) -> np.ndarray:
        if self._pos >= len(self._records):
            raise ReplayMismatch(
                f"draw #{self._pos + 1} sample.int({n}, {size}) requested, "
                f"but R made only {len(self._records)} draws"
            )
        rec = self._records[self._pos]
        if rec.n != n or rec.size != size:
            raise ReplayMismatch(
                f"draw #{self._pos + 1}: Python asked for sample.int({n}, {size}), "
                f"R did sample.int({rec.n}, {rec.size})"
            )
        self._pos += 1
        return rec.value - 1

    def permutation(self, n: int) -> np.ndarray:
        return self._next(n, n).astype(np.intp)

    def index(self, m: int) -> int:
        return int(self._next(m, 1)[0])

    def sample(self, n: int, size: int) -> np.ndarray:
        return self._next(n, size).astype(np.intp)

    def fork(self) -> ReplayDraws:
        """R's re-seeding is already in the recording, so forks share the one stream."""
        return self

    def assert_exhausted(self) -> None:
        if self._pos != len(self._records):
            raise ReplayMismatch(f"R made {len(self._records)} draws, Python consumed {self._pos}")


def as_draws(seed) -> Draws:
    if isinstance(seed, Draws):
        return seed
    if isinstance(seed, np.random.Generator) or (
        isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
    ):
        return NumpyDraws(seed)
    raise TypeError(
        f"seed must be an int, a numpy Generator, or a Draws object, not {type(seed).__name__}"
    )
```

- [ ] **Step 6: Run the full suite**

Run: `UV run pytest -q`
Expected: PASS, every Feature 1 test plus the 13 new ones.

- [ ] **Step 7: Commit**

```bash
git add src/datanuggets/result.py src/datanuggets/_draws.py tests/test_result.py tests/test_draws.py
git commit -F <msg>   # "Copy result arrays, compare by value, add Draws.sample and fork"
```

---

### Task 2: R's long-double statistics, unique() and eigen

**Files:**
- Create: `src/datanuggets/_ldouble.py`, `src/datanuggets/_rstats.py`, `r/fixtures_rstats.R`, `tests/test_rstats.py`
- Modify: `src/datanuggets/_rcompat.py` (append `unique_first`), `src/datanuggets/_defaults.py` (append refine defaults), `pyproject.toml` + `uv.lock` (scipy)
- Fixtures: `tests/fixtures/rcompat_stats/`

**Interfaces:**
- Consumes: `_rcompat._LDOUBLE_DIGITS`.
- Produces: `_ldouble.{ZERO, from_float, add, sub, mul, div_int, to_float}` on `(m, e)` pairs; `_rstats.r_sum(v) -> float`, `r_vec_mean(v) -> float`, `r_median(v) -> float`, `r_mad(v) -> float`, `r_quantile7(v, prob) -> float`, `r_cov(x) -> ndarray (p, p)`, `r_matvec(a, v) -> ndarray (m,)`, `r_eigen_sym(s, vectors=False) -> values | (values, vectors)` (descending); `_rcompat.unique_first(x) -> ndarray[intp]` (row positions of R's `unique(x)`); defaults `EV_TOL, MAX_SPLITS, MIN_NUGGET_SIZE, MIN_SHAPE_SIZE, DELTA, NSTART, KMEANS_ITER_MAX, MAD_CONSTANT`.

- [ ] **Step 1: Add scipy**

Run: `UV add "scipy>=1.13"`
Expected: `pyproject.toml` dependencies gain `"scipy>=1.13"`, `uv.lock` updates, exit 0.

- [ ] **Step 2: Write the R fixture script**

```r file=r/fixtures_rstats.R
# Fixtures for the R statistics refine.DN depends on: cov, sum, mean, median, mad and type-7
# quantile (long double accumulation), unique() on matrix rows, %*% and eigen(symmetric = TRUE).
source("r/fixture_io.R")

cov_sets <- list(
  iris = as_dbl(iris[, 1:4]),
  iris_setosa = as_dbl(iris[1:50, 1:4]),
  quakes = as_dbl(quakes),
  faithful = as_dbl(faithful),
  mtcars = as_dbl(mtcars),
  volcano5 = as_dbl(scale(volcano)[, 1:5]),
  big_offset = cbind(1e8 + faithful[, 1], faithful[, 2]),  # the mean's correction pass matters
  one_col = as_dbl(faithful[, 1, drop = FALSE]),
  two_rows = as_dbl(iris[1:2, 1:4])
)
for (nm in names(cov_sets)) {
  x <- cov_sets[[nm]]
  S <- cov(x)
  arrays <- list(x = x, cov = unname(S), scale = sum(diag(S)) / ncol(x))
  if (ncol(x) > 1L) {
    ev <- eigen(S, symmetric = TRUE)
    arrays <- c(arrays, list(
      values = eigen(S, symmetric = TRUE, only.values = TRUE)$values,
      values_v = ev$values,
      vectors = ev$vectors,
      proj = as.numeric(x %*% ev$vectors[, 1, drop = FALSE])))
  }
  write_case("rcompat_stats", paste0("cov_", nm), list(kind = "cov"), arrays)
}

set.seed(3)
vec_sets <- list(
  ev = eigen(cov(quakes))$values,
  scales = c(0.1, 0.2, 0.3, 1/3, 2/3, 0.7, 1e-17, 5),
  even = c(0.1 + 0.2, 0.3, 1/3, 1e-3),
  odd = rnorm(7),
  iris_col = iris[, 1],
  cancel = c(1e16, 1, -1e16, 3.5),
  ratios = c(1.9999999999999998, 2.0000000000000004),
  rn = rnorm(100) * 1e3,
  huge = c(1e308, 1e308, -1e307)  # the long double sum exceeds DBL_MAX: mean() re-sums x / n
)
probs <- c(0.5, 0.9, 0.25, 0.1, 0.999, 0.3)
for (nm in names(vec_sets)) {
  v <- as.double(vec_sets[[nm]])
  write_case("rcompat_stats", paste0("vec_", nm), list(kind = "vec", probs = probs), list(
    v = v, sum = sum(v), mean = mean(v), median = median(v), mad = mad(v),
    q = quantile(v, probs, names = FALSE)))
}

mean_sets <- list(inf = c(Inf, 2), inf_minus_inf = c(Inf, -Inf), neg_inf = c(-Inf, 1),
                  nan = c(NaN, 1), pair = c(1.5, 2.5000000000000004))
for (nm in names(mean_sets)) {
  v <- mean_sets[[nm]]
  write_case("rcompat_stats", paste0("mean_", nm), list(kind = "mean"), list(v = v, mean = mean(v)))
}

# R 4.x compares matrix rows exactly (0 == -0), not by a 15-digit paste
unique_sets <- list(
  petal = as_dbl(iris[, 3:4]),
  signed_zero = rbind(c(0.1 + 0.2, 1), c(0.3, 1), c(-0, 2), c(0, 2), c(0.3, 1)),
  near_dup = rbind(c(1, 1), c(1 + 2^-52, 1), c(1, 1)),
  one_col = matrix(c(3, 1, 3, 2, 1, -0, 0), ncol = 1)
)
for (nm in names(unique_sets)) {
  x <- unique_sets[[nm]]
  write_case("rcompat_stats", paste0("unique_", nm), list(kind = "unique"),
             list(x = x, first = which(!duplicated(x))))
}
cat(length(cov_sets) + length(vec_sets) + length(mean_sets) + length(unique_sets), "cases\n")
```

- [ ] **Step 3: Generate the fixtures**

Run: `RS r/fixtures_rstats.R`
Expected: prints `27 cases`; `tests/fixtures/rcompat_stats/` holds 27 case folders.

- [ ] **Step 4: Write the failing tests**

```python file=tests/test_rstats.py
"""R's long-double statistics, unique() and %*% replayed bit for bit; eigen() to Numeric tier."""

import numpy as np
import pytest

from datanuggets._rcompat import unique_first
from datanuggets._rstats import (
    r_cov,
    r_eigen_sym,
    r_mad,
    r_matvec,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
from rfixtures import ATOL, RTOL, load_group

CASES = load_group("rcompat_stats")


def _kind(kind):
    return [c for c in CASES if c.args["kind"] == kind]


COV = _kind("cov")
EIGEN = [c for c in COV if c.has("values")]


@pytest.mark.parametrize("case", COV, ids=lambda c: c.id)
def test_cov_and_scale_match_r_bitwise(case):
    x = case.array("x")
    s = r_cov(x)
    np.testing.assert_array_equal(s, case.array("cov"))
    assert r_sum(np.diag(s)) / x.shape[1] == case.array("scale")[0]


@pytest.mark.parametrize("case", EIGEN, ids=lambda c: c.id)
def test_eigen_matches_r_numerically(case):
    s = case.array("cov")
    np.testing.assert_allclose(r_eigen_sym(s), case.array("values"), rtol=RTOL, atol=ATOL)
    values, vectors = r_eigen_sym(s, vectors=True)
    np.testing.assert_allclose(values, case.array("values_v"), rtol=RTOL, atol=ATOL)
    got, want = vectors[:, 0], case.array("vectors")[:, 0]
    np.testing.assert_allclose(got * np.sign(got @ want), want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", EIGEN, ids=lambda c: c.id)
def test_matvec_matches_r_bitwise(case):
    got = r_matvec(case.array("x"), case.array("vectors")[:, 0])
    np.testing.assert_array_equal(got, case.array("proj"))


@pytest.mark.parametrize("case", _kind("vec"), ids=lambda c: c.id)
def test_vector_stats_match_r_bitwise(case):
    v = case.array("v").tolist()
    assert r_sum(v) == case.array("sum")[0]
    assert r_vec_mean(v) == case.array("mean")[0]
    assert r_median(v) == case.array("median")[0]
    assert r_mad(v) == case.array("mad")[0]
    got = [r_quantile7(v, q) for q in case.args["probs"]]
    np.testing.assert_array_equal(got, case.array("q"))


@pytest.mark.parametrize("case", _kind("mean"), ids=lambda c: c.id)
def test_mean_of_non_finite_values_matches_r(case):
    np.testing.assert_array_equal(r_vec_mean(case.array("v").tolist()), case.array("mean")[0])


@pytest.mark.parametrize("case", _kind("unique"), ids=lambda c: c.id)
def test_unique_first_matches_r(case):
    np.testing.assert_array_equal(unique_first(case.array("x")) + 1, case.array("first"))
```

- [ ] **Step 5: Run the tests and watch them fail**

Run: `UV run pytest tests/test_rstats.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'datanuggets._rstats'`.

- [ ] **Step 6: Write _ldouble.py**

```python file=src/datanuggets/_ldouble.py
"""Long double arithmetic as R does it on x86-64, replayed exactly with Python integers.

R accumulates sums, means and covariances in LDOUBLE, which on x86-64 is x87 extended precision:
a 64-bit significand, rounded to nearest with ties to even after every operation. A value here is
a pair (m, e) standing for exactly m * 2**e with |m| < 2**64. Exponent range limits are ignored:
long double's range covers every sum and product of finite doubles that R forms here.
"""

from __future__ import annotations

import math

from datanuggets._rcompat import _LDOUBLE_DIGITS as DIGITS

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
```

- [ ] **Step 7: Write _rstats.py**

```python file=src/datanuggets/_rstats.py
"""R's summary statistics as refine.DN uses them.

sum, mean, median, mad, type-7 quantile, cov and %*% are replayed bit for bit. R accumulates
sums in long double (x87 extended precision on x86-64), which _ldouble reproduces with Python
integers. eigen() calls LAPACK's dsyevr with R's arguments, so eigenvalues agree with R's to
rounding (about 1e-12 relative) but not bit for bit: LAPACK builds differ in their kernels.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import LinAlgError, lapack

from datanuggets import _ldouble as ld
from datanuggets._defaults import MAD_CONSTANT


def _ld_sum(v: list[float]) -> tuple[int, int]:
    s = ld.ZERO
    for x in v:
        s = ld.add(s, ld.from_float(x))
    return s


def r_sum(v) -> float:
    """sum() of a double vector: a long double running sum, rounded to double once."""
    v = [float(x) for x in v]
    if not all(map(math.isfinite, v)):
        return float(sum(v))  # Inf and NaN come out the same at any precision
    return ld.to_float(_ld_sum(v))


def r_vec_mean(v) -> float:
    """mean() of a double vector (R's real_mean): the long double sum divided by n, plus the
    mean of the residuals from a second pass. If the sum overflows double, R sums x / n."""
    v = [float(x) for x in v]
    n = len(v)
    if not all(map(math.isfinite, v)):
        return float(sum(x / n for x in v))  # R's rescaled sum, and no second pass
    s = _ld_sum(v)
    if math.isfinite(ld.to_float(s)):
        s = ld.div_int(s, n)
    else:
        s = _ld_sum([x / n for x in v])  # each x / n is rounded to double first, as in R
    if math.isfinite(ld.to_float(s)):
        t = ld.ZERO
        for x in v:
            t = ld.add(t, ld.sub(ld.from_float(x), s))
        s = ld.add(s, ld.div_int(t, n))
    return ld.to_float(s)


def r_median(v) -> float:
    """median(): the middle value, or R's mean() of the two middle values."""
    s = sorted(float(x) for x in v)
    half = (len(s) + 1) // 2
    if len(s) % 2:
        return s[half - 1]
    return r_vec_mean(s[half - 1 : half + 1])


def r_mad(v) -> float:
    """mad() with R's default constant, 1.4826 exactly."""
    c = r_median(v)
    return MAD_CONSTANT * r_median([abs(float(x) - c) for x in v])


def r_quantile7(v, prob: float) -> float:
    """quantile(v, prob) with R's default type 7."""
    s = sorted(float(x) for x in v)
    index = 1 + max(len(s) - 1, 0) * prob
    lo, hi = math.floor(index), math.ceil(index)
    qs = s[lo - 1]
    if index > lo and s[hi - 1] != qs:
        h = index - lo
        qs = (1 - h) * qs + h * s[hi - 1]
    return qs


def r_cov(x: np.ndarray) -> np.ndarray:
    """cov(x): each column mean in two long double passes, stored as a double, then long double
    sums of the products of deviations divided by n - 1."""
    n, p = x.shape
    if n < 2:
        raise ValueError("cov needs at least two rows")
    dev = []
    for j in range(p):
        col = [ld.from_float(v) for v in x[:, j].tolist()]
        s = ld.ZERO
        for v in col:
            s = ld.add(s, v)
        mean = ld.div_int(s, n)
        if math.isfinite(ld.to_float(mean)):
            s = ld.ZERO
            for v in col:
                s = ld.add(s, ld.sub(v, mean))
            mean = ld.add(mean, ld.div_int(s, n))
        mean = ld.from_float(ld.to_float(mean))  # R keeps the column means as doubles
        dev.append([ld.sub(v, mean) for v in col])
    out = np.empty((p, p))
    for i in range(p):
        for j in range(i + 1):
            s = ld.ZERO
            for a, b in zip(dev[i], dev[j], strict=True):
                s = ld.add(s, ld.mul(a, b))
            out[i, j] = out[j, i] = ld.to_float(ld.div_int(s, n - 1))
    return out


def r_matvec(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """a %*% v as R computes it with reference BLAS dgemv: y = 0, then y += a[:, j] * v[j]
    column by column, every product and sum rounded to double."""
    y = np.zeros(a.shape[0])
    for j in range(a.shape[1]):
        y += a[:, j] * v[j]
    return y


def r_eigen_sym(s: np.ndarray, vectors: bool = False):
    """eigen(s, symmetric = TRUE): LAPACK dsyevr on the lower triangle with R's arguments
    (range "A", abstol 0, optimal workspace), values decreasing. With vectors=True, returns
    (values, vectors), the vectors as columns in the same order."""
    s = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError("infinite or missing values in 'x'")
    lwork, liwork, info = lapack.dsyevr_lwork(s.shape[0], lower=1)
    if info == 0:
        w, z, _, _, info = lapack.dsyevr(
            s,
            compute_v=int(vectors),
            range="A",
            lower=1,
            abstol=0.0,
            lwork=int(lwork),
            liwork=int(liwork),
        )
    if info != 0:
        raise LinAlgError(f"error code {info} from Lapack routine 'dsyevr'")
    values = w[::-1].copy()
    if not vectors:
        return values
    return values, z[:, ::-1].copy()
```

- [ ] **Step 8: Append unique_first to _rcompat.py and the refine defaults to _defaults.py**

```python file=src/datanuggets/_rcompat.py mode=append
def unique_first(x: np.ndarray) -> np.ndarray:
    """0-based positions of the rows R's unique(x) keeps from a numeric matrix: the first
    occurrence of each distinct row, in row order. R 4.x compares the rows exactly, with
    0 equal to -0, and so do Python float tuples. Finite data only."""
    first: dict[tuple[float, ...], int] = {}
    for i, row in enumerate(np.asarray(x, dtype=np.float64).tolist()):
        first.setdefault(tuple(row), i)
    return np.fromiter(first.values(), dtype=np.intp, count=len(first))
```

```python file=src/datanuggets/_defaults.py mode=append
# refine.DN's defaults, and what it passes to stats::kmeans
EV_TOL = 0.9
MAX_SPLITS = 5
MIN_NUGGET_SIZE = 2
MIN_SHAPE_SIZE = 10
DELTA = 2
NSTART = 25
KMEANS_ITER_MAX = 1000
MAD_CONSTANT = 1.4826  # stats::mad's default, not scipy's 1.482602...
```

- [ ] **Step 9: Run the stats tests**

Run: `UV run pytest tests/test_rstats.py -q`
Expected: PASS (9 cov, 8 x 2 eigen/matvec, 9 vec, 5 mean, 4 unique).

- [ ] **Step 10: Run the whole suite and the linters**

Run: `UV run pytest -q && UV run ruff check . && UV run ruff format --check .`
Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml uv.lock src/datanuggets/_ldouble.py src/datanuggets/_rstats.py \
  src/datanuggets/_rcompat.py src/datanuggets/_defaults.py r/fixtures_rstats.R \
  tests/test_rstats.py tests/fixtures/rcompat_stats
git commit -F <msg>   # "Replay R's long-double stats, unique() and eigen for refine"
```

---

### Task 3: Hartigan-Wong k-means

**Files:**
- Create: `src/datanuggets/_hartigan_wong.py`, `r/fixtures_kmeans.R`, `tests/test_hartigan_wong.py`
- Modify: `r/fixture_io.R` (append `draw_arrays`)
- Fixtures: `tests/fixtures/kmns/`, `tests/fixtures/kmeans/`

**Interfaces:**
- Consumes: `Draws.sample` (Task 1), `unique_first`, `r_sum` (Task 2).
- Produces: `kmns(a, c, iter_max: int, max_qtran: int) -> Kmns` (fields `cluster: list[int]` 0-based, `centers`, `size`, `withinss`, `iter`, `ifault`); `kmeans(x, k: int, draws: Draws, *, nstart: float = 1, iter_max: int = 10) -> KMeansResult` (fields `cluster` ndarray 0-based, `centers`, `withinss`, `size`, `tot_withinss`, `iter`, `ifault`, `warnings: tuple[str, ...]`); `KMeansError(ValueError)` with `.warnings`. Error texts are R's: `"argument is of length zero"` (nstart not 1 and not >= 2), `"more cluster centers than distinct data points."`, `"empty cluster: try a better set of initial centers"`, `"number of cluster centres must lie between 1 and nrow(x)"`. Warning texts: `"Quick-TRANSfer stage steps exceeded maximum (= N)"`, `"did not converge in N iteration(s)"`.
- R: `draw_arrays(draws)` in `r/fixture_io.R` returns `list(draw_n, draw_size, draw_values)`.

- [ ] **Step 1: Add draw_arrays to the R helpers**

```r file=r/fixture_io.R mode=append
# The draws capture_draws recorded, as the three arrays tests/rfixtures.py reads back.
draw_arrays <- function(draws) list(
  draw_n = as.integer(vapply(draws, function(d) as.numeric(d$n), numeric(1))),
  draw_size = as.integer(vapply(draws, function(d) as.numeric(d$size), numeric(1))),
  draw_values = as.integer(unlist(lapply(draws, function(d) d$value)))
)
```

- [ ] **Step 2: Write the R fixture script**

```r file=r/fixtures_kmeans.R
# Fixtures for Hartigan-Wong k-means: R's own Fortran (stats:::C_kmns) from fixed starting
# centers (group kmns), and stats::kmeans end to end with its random draws (group kmeans).
source("r/fixture_io.R")

int_max <- .Machine$integer.max

# Calls C_kmns exactly as kmeans()'s do_one() does. Fails closed on an unexpected fault code.
add_direct <- function(id, x, rows, iter.max = 1000L, isteps = min(int_max, 50 * nrow(x)),
                       ifault = 0L) {
  x <- as_dbl(x)
  c0 <- x[rows, , drop = FALSE]
  m <- nrow(x)
  p <- ncol(x)
  k <- nrow(c0)
  Z <- .Fortran(stats:::C_kmns, x, as.integer(m), as.integer(p), centers = c0, as.integer(k),
                c1 = integer(m), c2 = integer(m), nc = integer(k), double(k), double(k),
                ncp = integer(k), D = double(m), iTran = c(as.integer(isteps), integer(k)),
                live = integer(k), iter = as.integer(iter.max), wss = double(k), ifault = 0L)
  if (Z$ifault != ifault) stop(id, ": expected ifault ", ifault, ", C_kmns gave ", Z$ifault)
  write_case("kmns", id,
             list(iter.max = as.integer(iter.max), isteps = as.integer(isteps), ifault = ifault),
             list(x = x, c0 = c0, c1 = Z$c1, centers = matrix(Z$centers, k), nc = Z$nc,
                  wss = Z$wss, iter = Z$iter, ifault = Z$ifault))
  cat(sprintf("kmns   %-20s ifault=%d iter=%d\n", id, Z$ifault, Z$iter))
}

iris4 <- iris[, 1:4]
f1 <- faithful[, 1, drop = FALSE]
add_direct("iris_k2", iris4, c(1, 51))
add_direct("iris_k3", iris4, c(1, 51, 101))
add_direct("iris_k5", iris4, c(3, 60, 77, 120, 140))
add_direct("iris_k3_bad_start", iris4, c(1, 2, 3))
add_direct("iris_k4_iter1", iris4, c(1, 2, 3, 4), iter.max = 1L, ifault = 2L)
add_direct("faithful_k2", faithful, c(10, 20))
add_direct("faithful_k4", faithful, c(1, 2, 3, 4))
add_direct("quakes_k3", quakes, c(5, 500, 900))
add_direct("quakes_k6", quakes, c(1, 100, 200, 300, 400, 500))
add_direct("faithful1d_k2", f1, c(1, 2))
add_direct("faithful1d_k3", f1, c(1, 2, 3))
add_direct("quakes_k4_isteps50", quakes, c(1, 2, 3, 4), isteps = 50L, ifault = 4L)
add_direct("iris_dup_start", iris4, c(1, 1), ifault = 1L)  # kmeans() stops: empty cluster
add_direct("iris_k1", iris4, 1, ifault = 3L)               # kmns.f needs 1 < k < m
add_direct("volcano_k3", scale(volcano), c(1, 40, 80))

add_kmeans <- function(id, x, k, nstart, seed = 291102, iter.max = 10L) {
  x <- as_dbl(x)
  warns <- character()
  err <- ""
  set.seed(seed)
  res <- capture_draws(tryCatch(
    withCallingHandlers(kmeans(x, k, iter.max = iter.max, nstart = nstart),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) {
      err <<- conditionMessage(e)
      NULL
    }))
  km <- res$value
  arrays <- c(list(x = x), draw_arrays(res$draws))
  if (!is.null(km)) {
    arrays <- c(arrays, list(cluster = km$cluster, centers = unname(km$centers),
                             withinss = km$withinss, size = km$size, iter = km$iter,
                             tot_withinss = km$tot.withinss))
  }
  write_case("kmeans", id, list(k = k, nstart = nstart, seed = seed, iter.max = iter.max),
             arrays, extra = list(error = err, warnings = I(warns)))
  cat(sprintf("kmeans %-20s draws=%d warnings=%d error=%s\n", id, length(res$draws),
              length(warns), err))
  invisible(list(draws = res$draws, error = err, warnings = warns))
}

petal <- as_dbl(iris[, 3:4])
add_kmeans("faithful_k2_n1", faithful, 2, 1)
add_kmeans("faithful_k2_n25", faithful, 2, 25)
add_kmeans("faithful_k2_n25.7", faithful, 2, 25.7)  # 2:25.7 runs 25 starts in all
add_kmeans("petal_k2_n25", petal, 2, 25)            # starts come from unique(x)
# nstart = 1 redraws from unique(x) when its two random rows are equal: find a seed that does
dup_seed <- Find(function(s) {
  set.seed(s)
  i <- sample.int(nrow(petal), 2L)
  all(petal[i[1], ] == petal[i[2], ])
}, seq_len(10000))
r <- add_kmeans("petal_k2_n1_dup", petal, 2, 1, seed = dup_seed)
stopifnot(length(r$draws) == 2L)
add_kmeans("faithful1d_k2_n25", f1, 2, 25)
add_kmeans("iris_k3_n5", iris4, 3, 5)
add_kmeans("quakes_k4_n3", quakes, 4, 3)
r <- add_kmeans("iris_k4_iter1", iris4, 4, 1, iter.max = 1L)
stopifnot(length(r$warnings) == 1L)
r <- add_kmeans("faithful_k2_n0", faithful, 2, 0)  # refine.DN with nstart = 0 hits this
stopifnot(nzchar(r$error), length(r$draws) == 0L)
r <- add_kmeans("petal_k149", petal, 149, 1)
stopifnot(r$error == "more cluster centers than distinct data points.", length(r$draws) == 1L)
```

- [ ] **Step 3: Generate the fixtures**

Run: `RS r/fixtures_kmeans.R`
Expected: 15 `kmns` lines (ifault 2 for `iris_k4_iter1`, 4 for `quakes_k4_isteps50`, 1 for `iris_dup_start`, 3 for `iris_k1`, 0 otherwise) and 11 `kmeans` lines; no `stop()`.

- [ ] **Step 4: Write the failing tests**

```python file=tests/test_hartigan_wong.py
"""Hartigan-Wong k-means against R's own C_kmns and stats::kmeans, bit for bit."""

import numpy as np
import pytest

from datanuggets._draws import ReplayDraws
from datanuggets._hartigan_wong import KMeansError, kmeans, kmns
from rfixtures import load_group

KMNS = load_group("kmns")
KMEANS = load_group("kmeans")
RUNS = [c for c in KMNS if c.args["ifault"] in (0, 2, 4)]
STOPS = [c for c in KMNS if c.args["ifault"] in (1, 3)]


def _kmns(case, x=None, c0=None):
    a = case.args
    x = case.array("x") if x is None else x
    c0 = case.array("c0") if c0 is None else c0
    return kmns(x, c0, a["iter.max"], a["isteps"])


def test_fixtures_cover_every_fault_code():
    assert {c.args["ifault"] for c in KMNS} == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("case", RUNS, ids=lambda c: c.id)
def test_kmns_matches_r(case):
    z = _kmns(case)
    assert z.ifault == case.args["ifault"]
    np.testing.assert_array_equal(np.array(z.cluster) + 1, case.array("c1"))
    np.testing.assert_array_equal(z.centers, case.array("centers"))
    np.testing.assert_array_equal(z.size, case.array("nc"))
    np.testing.assert_array_equal(z.withinss, case.array("wss"))
    assert z.iter == case.array("iter")[0]


@pytest.mark.parametrize("case", STOPS, ids=lambda c: c.id)
def test_kmns_reports_the_faults_kmeans_stops_on(case):
    assert _kmns(case).ifault == case.args["ifault"]


def test_kmns_is_symmetric_under_negation():
    """refine_dn clusters projections onto an eigenvector whose sign LAPACK picks. Negating
    the data and starts must give the same clusters and exactly negated centers."""
    case = next(c for c in KMNS if c.id == "faithful1d_k2")
    z = _kmns(case)
    neg = _kmns(case, -case.array("x"), -case.array("c0"))
    assert neg.cluster == z.cluster
    np.testing.assert_array_equal(neg.centers, -z.centers)
    np.testing.assert_array_equal(neg.withinss, z.withinss)


def _run(case, draws):
    a = case.args
    return kmeans(case.array("x"), a["k"], draws, nstart=a["nstart"], iter_max=a["iter.max"])


@pytest.mark.parametrize("case", [c for c in KMEANS if not c.manifest["error"]], ids=lambda c: c.id)
def test_kmeans_matches_r(case):
    draws = ReplayDraws(case.draws())
    got = _run(case, draws)
    draws.assert_exhausted()
    np.testing.assert_array_equal(got.cluster + 1, case.array("cluster"))
    np.testing.assert_array_equal(got.centers, case.array("centers"))
    np.testing.assert_array_equal(got.withinss, case.array("withinss"))
    np.testing.assert_array_equal(got.size, case.array("size"))
    assert got.tot_withinss == case.array("tot_withinss")[0]
    assert got.iter == case.array("iter")[0]
    assert list(got.warnings) == case.manifest["warnings"]


@pytest.mark.parametrize("case", [c for c in KMEANS if c.manifest["error"]], ids=lambda c: c.id)
def test_kmeans_stops_where_r_stops(case):
    draws = ReplayDraws(case.draws())
    with pytest.raises(KMeansError) as err:
        _run(case, draws)
    draws.assert_exhausted()
    assert str(err.value) == case.manifest["error"]
```

- [ ] **Step 5: Run the tests and watch them fail**

Run: `UV run pytest tests/test_hartigan_wong.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'datanuggets._hartigan_wong'`.

- [ ] **Step 6: Write _hartigan_wong.py**

```python file=src/datanuggets/_hartigan_wong.py
"""Hartigan-Wong k-means as R's stats::kmeans runs it.

kmns() is a line-by-line port of R's kmns.f (Hartigan and Wong 1979, algorithm AS 136, with
R's changes). Every floating-point operation happens in the Fortran's order, in double, so from
the same starting centers the clusters, centers and within-cluster sums of squares match R bit
for bit. kmeans() adds what stats::kmeans does around it: the random starts, the nstart
restarts, and R's errors and warnings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from datanuggets._draws import Draws
from datanuggets._rcompat import unique_first
from datanuggets._rstats import r_sum

BIG = float(np.float32(1e30))  # kmns.f: DATA BIG /1.E30/, a REAL constant widened to double
_INT_MAX = 2**31 - 1
_QTRAN_STEPS_PER_ROW = 50  # kmeans(): isteps.Qtran = min(.Machine$integer.max, 50 * m)


class KMeansError(ValueError):
    """stats::kmeans stopped with an error. warnings holds what it warned before stopping."""

    def __init__(self, message: str, warnings: tuple[str, ...] = ()):
        super().__init__(message)
        self.warnings = warnings


@dataclass(frozen=True)
class Kmns:
    """kmns.f's outputs IC1 (0-based), C, NC, WSS, ITER and IFAULT."""

    cluster: list[int]
    centers: np.ndarray
    size: list[int]
    withinss: list[float]
    iter: int
    ifault: int


@dataclass(frozen=True, eq=False)
class KMeansResult:
    """What stats::kmeans returns (cluster 0-based), plus the warnings it raised, in order."""

    cluster: np.ndarray
    centers: np.ndarray
    withinss: np.ndarray
    size: np.ndarray
    tot_withinss: float
    iter: int
    ifault: int
    warnings: tuple[str, ...]


def kmns(a: np.ndarray, c: np.ndarray, iter_max: int, max_qtran: int) -> Kmns:
    """R's C_kmns: Hartigan-Wong on the rows of a (m x p) from the starting centers c (k x p)."""
    hw = _HW(np.asarray(a, dtype=np.float64), np.asarray(c, dtype=np.float64))
    return hw.run(iter_max, max_qtran)


def kmeans(x, k: int, draws: Draws, *, nstart: float = 1, iter_max: int = 10) -> KMeansResult:
    """stats::kmeans(x, centers = k, iter.max, nstart) with the Hartigan-Wong algorithm.

    With nstart == 1 the starts are k random rows. If those repeat a row, or nstart >= 2, the
    starts are drawn from R's unique(x) instead, and floor(nstart) - 1 more starts follow. The
    run with the smallest total within-cluster sum of squares wins, the earliest on ties.
    Raises KMeansError where R stops, and returns R's warnings instead of emitting them.
    """
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[0]
    if k < 2:
        raise ValueError("k must be at least 2; R switches to MacQueen's algorithm for k = 1")
    if not (nstart == 1 or nstart >= 2):
        raise KMeansError("argument is of length zero")  # R: if (ncol(x) != ncol(<a number>))
    starts = cn = None
    if nstart == 1:
        if k > m:
            raise KMeansError(
                "cannot take a sample larger than the population when 'replace = FALSE'"
            )
        starts = x[draws.sample(m, k)]
    if nstart >= 2 or unique_first(starts).size < k:
        cn = x[unique_first(x)]
        if cn.shape[0] < k:
            raise KMeansError("more cluster centers than distinct data points.")
        starts = cn[draws.sample(cn.shape[0], k)]
    warns: list[str] = []
    best = _one_run(x, starts, iter_max, warns)
    best_wss = r_sum(best.withinss)
    for _ in range(2, math.floor(nstart) + 1):
        run = _one_run(x, cn[draws.sample(cn.shape[0], k)], iter_max, warns)
        wss = r_sum(run.withinss)
        if wss < best_wss:
            best, best_wss = run, wss
    return KMeansResult(
        cluster=np.array(best.cluster, dtype=np.intp),
        centers=best.centers,
        withinss=np.array(best.withinss),
        size=np.array(best.size, dtype=np.int64),
        tot_withinss=best_wss,
        iter=best.iter,
        ifault=best.ifault,
        warnings=tuple(warns),
    )


def _one_run(x: np.ndarray, starts: np.ndarray, iter_max: int, warns: list[str]) -> Kmns:
    """kmeans()'s do_one(): kmns, then R's handling of its fault codes."""
    max_qtran = min(_INT_MAX, _QTRAN_STEPS_PER_ROW * x.shape[0])
    z = kmns(x, starts, iter_max, max_qtran)
    if z.ifault == 1:
        raise KMeansError("empty cluster: try a better set of initial centers", tuple(warns))
    if z.ifault == 3:
        raise KMeansError(
            "number of cluster centres must lie between 1 and nrow(x)", tuple(warns)
        )
    if z.ifault == 4:
        warns.append(f"Quick-TRANSfer stage steps exceeded maximum (= {max_qtran})")
    if z.ifault == 2:
        z = replace(z, iter=max(z.iter, iter_max + 1))
    if z.iter > iter_max:
        plural = "" if iter_max == 1 else "s"
        warns.append(f"did not converge in {iter_max} iteration{plural}")
    return z


class _HW:
    """One kmns.f call. Names follow the Fortran; cluster and row indices are 0-based."""

    def __init__(self, a: np.ndarray, c: np.ndarray):
        self.m, self.n = a.shape
        self.k = c.shape[0]
        self.a = a.tolist()
        self.c = c.tolist()
        m, k = self.m, self.k
        self.ic1 = [0] * m
        self.ic2 = [0] * m
        self.d = [0.0] * m
        self.nc = [0] * k
        self.an1 = [0.0] * k
        self.an2 = [0.0] * k
        self.ncp = [0] * k
        self.itran = [0] * k
        self.live = [0] * k
        self.wss = [0.0] * k

    def run(self, iter_max: int, max_qtran: int) -> Kmns:
        m, k = self.m, self.k
        if k <= 1 or k >= m:
            return self._result(iter_max, 3)
        self._initial_assignment()
        if not self._initial_centers():
            return self._result(iter_max, 1)
        indx = 0
        ifault, ij = 2, iter_max + 1  # a Fortran DO variable after its loop ran out
        for it in range(1, iter_max + 1):
            indx = self._optra(indx)
            if indx == m:  # a full pass without a transfer: converged
                ifault, ij = 0, it
                break
            indx, max_qtran = self._qtran(indx, max_qtran)
            if max_qtran < 0:
                ifault, ij = 4, it
                break
            if k == 2:  # with two clusters the optimal-transfer stage has nothing left to do
                ifault, ij = 0, it
                break
            for g in range(k):
                self.ncp[g] = 0
        self._final_centers_and_wss()
        return self._result(ij, ifault)

    def _result(self, iter_: int, ifault: int) -> Kmns:
        return Kmns(
            cluster=list(self.ic1),
            centers=np.array(self.c),
            size=list(self.nc),
            withinss=list(self.wss),
            iter=iter_,
            ifault=ifault,
        )

    def _initial_assignment(self) -> None:
        """IC1 and IC2: each point's closest and second closest starting center."""
        a, c, rn = self.a, self.c, range(self.n)
        for i in range(self.m):
            ai = a[i]
            dt = [0.0, 0.0]
            for il in (0, 1):
                s = 0.0
                cg = c[il]
                for j in rn:
                    da = ai[j] - cg[j]
                    s += da * da
                dt[il] = s
            c1, c2 = 0, 1
            if dt[0] > dt[1]:
                c1, c2 = 1, 0
                dt[0], dt[1] = dt[1], dt[0]
            for g in range(2, self.k):
                db = 0.0
                cg = c[g]
                for j in rn:
                    dc = ai[j] - cg[j]
                    db += dc * dc
                    if db >= dt[1]:
                        break
                else:
                    if db >= dt[0]:
                        dt[1] = db
                        c2 = g
                    else:
                        dt[1] = dt[0]
                        c2 = c1
                        dt[0] = db
                        c1 = g
            self.ic1[i] = c1
            self.ic2[i] = c2

    def _initial_centers(self) -> bool:
        """Centers become the means of the initial clusters. False on an empty cluster."""
        rn = range(self.n)
        c = [[0.0] * self.n for _ in range(self.k)]
        self.c = c
        for i in range(self.m):
            g = self.ic1[i]
            self.nc[g] += 1
            cg = c[g]
            ai = self.a[i]
            for j in rn:
                cg[j] += ai[j]
        for g in range(self.k):
            if self.nc[g] == 0:
                return False
            aa = float(self.nc[g])
            cg = c[g]
            for j in rn:
                cg[j] /= aa
            self.an2[g] = aa / (aa + 1.0)
            self.an1[g] = aa / (aa - 1.0) if aa > 1.0 else BIG
            self.itran[g] = 1
            self.ncp[g] = -1
        return True

    def _optra(self, indx: int) -> int:
        """Optimal-transfer stage: each point may move to the cluster that most reduces WSS."""
        a, c, ic1, ic2, nc = self.a, self.c, self.ic1, self.ic2, self.nc
        an1, an2, ncp, d, itran, live = self.an1, self.an2, self.ncp, self.d, self.itran, self.live
        m, k, rn = self.m, self.k, range(self.n)
        for g in range(k):
            if itran[g] == 1:
                live[g] = m + 1
        for i0 in range(m):
            i = i0 + 1  # the Fortran's 1-based row number, which LIVE and NCP store
            indx += 1
            l1 = ic1[i0]
            l2 = ic2[i0]
            ll = l2
            if nc[l1] != 1:
                ai = a[i0]
                if ncp[l1] != 0:
                    de = 0.0
                    cg = c[l1]
                    for j in rn:
                        df = ai[j] - cg[j]
                        de += df * df
                    d[i0] = de * an1[l1]
                da = 0.0
                cg = c[l2]
                for j in rn:
                    db = ai[j] - cg[j]
                    da += db * db
                r2 = da * an2[l2]
                for g in range(k):
                    if (i >= live[l1] and i >= live[g]) or g == l1 or g == ll:
                        continue
                    rr = r2 / an2[g]
                    dc = 0.0
                    cg = c[g]
                    for j in rn:
                        dd = ai[j] - cg[j]
                        dc += dd * dd
                        if dc >= rr:
                            break
                    else:
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
                    self._transfer(ai, l1, l2)
                    ic1[i0] = l2
                    ic2[i0] = l1
            if indx == m:
                return indx
        for g in range(k):
            itran[g] = 0
            live[g] -= m
        return indx

    def _qtran(self, indx: int, max_qtran: int) -> tuple[int, int]:
        """Quick-transfer stage: swap points between their two closest clusters until a full
        pass moves nothing. Returns max_qtran = -1 when the step budget runs out."""
        a, c, ic1, ic2, nc = self.a, self.c, self.ic1, self.ic2, self.nc
        an1, an2, ncp, d, itran = self.an1, self.an2, self.ncp, self.d, self.itran
        m, rn = self.m, range(self.n)
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
                    ai = a[i0]
                    if istep <= ncp[l1]:
                        da = 0.0
                        cg = c[l1]
                        for j in rn:
                            db = ai[j] - cg[j]
                            da += db * db
                        d[i0] = da * an1[l1]
                    if istep < ncp[l1] or istep < ncp[l2]:
                        r2 = d[i0] / an2[l2]
                        dd = 0.0
                        cg = c[l2]
                        for j in rn:
                            de = ai[j] - cg[j]
                            dd += de * de
                            if dd >= r2:
                                break
                        else:
                            icoun = 0
                            indx = 0
                            itran[l1] = 1
                            itran[l2] = 1
                            ncp[l1] = istep + m
                            ncp[l2] = istep + m
                            self._transfer(ai, l1, l2)
                            ic1[i0] = l2
                            ic2[i0] = l1
                if icoun == m:
                    return indx, max_qtran

    def _transfer(self, ai: list[float], l1: int, l2: int) -> None:
        """Move point ai from cluster l1 to l2: both centers and their AN1/AN2 factors."""
        c, nc, an1, an2 = self.c, self.nc, self.an1, self.an2
        al1 = float(nc[l1])
        alw = al1 - 1.0
        al2 = float(nc[l2])
        alt = al2 + 1.0
        c1, c2 = c[l1], c[l2]
        for j in range(self.n):
            c1[j] = (c1[j] * al1 - ai[j]) / alw
            c2[j] = (c2[j] * al2 + ai[j]) / alt
        nc[l1] -= 1
        nc[l2] += 1
        an2[l1] = alw / al1
        an1[l1] = alw / (alw - 1.0) if alw > 1.0 else BIG
        an1[l2] = alt / al2
        an2[l2] = alt / (alt + 1.0)

    def _final_centers_and_wss(self) -> None:
        """Recompute the centers from the final clusters, then each cluster's WSS."""
        a, ic1, nc, rn, k = self.a, self.ic1, self.nc, range(self.n), self.k
        c = [[0.0] * self.n for _ in range(k)]
        wss = [0.0] * k
        for i in range(self.m):
            cg = c[ic1[i]]
            ai = a[i]
            for j in rn:
                cg[j] += ai[j]
        for j in rn:
            for g in range(k):
                c[g][j] /= float(nc[g])
            for i in range(self.m):
                ii = ic1[i]
                da = a[i][j] - c[ii][j]
                wss[ii] += da * da
        self.c = c
        self.wss = wss
```

- [ ] **Step 7: Run the tests**

Run: `UV run pytest tests/test_hartigan_wong.py -q`
Expected: PASS (1 coverage, 13 kmns runs, 2 kmns stops, 1 negation, 9 kmeans, 2 kmeans stops).

- [ ] **Step 8: Run the whole suite and the linters**

Run: `UV run pytest -q && UV run ruff check . && UV run ruff format --check .`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/datanuggets/_hartigan_wong.py r/fixture_io.R r/fixtures_kmeans.R \
  tests/test_hartigan_wong.py tests/fixtures/kmns tests/fixtures/kmeans
git commit -F <msg>   # "Port Hartigan-Wong k-means from R's kmns.f"
```

---

### Task 4: refine_dn

**Files:**
- Create: `src/datanuggets/refine.py`, `r/fixtures_refine_dn.R`, `tests/test_refine.py`
- Modify: `src/datanuggets/_validate.py` (append `check_n_jobs`), `src/datanuggets/create.py` (use it), `src/datanuggets/__init__.py` (export `refine_dn`), `tests/rfixtures.py` (append `refine_inputs`, one import)
- Fixtures: `tests/fixtures/refine_dn/`

**Interfaces:**
- Consumes: `DataNuggets` value equality and `Draws.fork` (Task 1); `r_cov`, `r_sum`, `r_vec_mean`, `r_median`, `r_mad`, `r_quantile7`, `r_matvec`, `r_eigen_sym`, the refine defaults (Task 2); `kmeans`, `KMeansError`, `KMeansResult.cluster`/`.warnings` (Task 3); `_rcompat.r_mean`, `create._progress_bar` (Feature 1); R `draw_arrays` (Task 3).
- Produces: `refine_dn(x, dn, ev_tol=0.9, max_splits=5, min_nugget_size=2, shape_split=False, min_shape_size=10, delta=2, nstart=25, seed=291102, n_jobs=1, progress=False) -> DataNuggets`; `nugget_features(obs) -> (center ndarray (p,), weight int, scale float, ev1 float, ev2 float)`; `_validate.check_n_jobs(n_jobs) -> int`; `rfixtures.refine_inputs(case) -> (x, DataNuggets, kwargs)`. INFO messages are R's `message()` texts; rejected splits log DEBUG `"nugget %d not split: %s"` with the reasons `"kmeans error: <R's error>"`, `"a child is smaller than min_nugget_size"`, `"mean EV1/EV2 of the children >= delta"`.

- [ ] **Step 1: Share the n_jobs check**

```python file=src/datanuggets/_validate.py mode=append
def check_n_jobs(n_jobs) -> int:
    """joblib's n_jobs: a positive int, or -1 for every core."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be a positive int or -1, got {n_jobs!r}")
    return n_jobs
```

In `src/datanuggets/create.py`, add `check_n_jobs,` to the `from datanuggets._validate import (...)` list after `check_delete_percent,`, and replace

```python
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be a positive int or -1, got {n_jobs!r}")
```

with

```python
    check_n_jobs(n_jobs)
```

- [ ] **Step 2: Run the create tests**

Run: `UV run pytest tests/test_create.py -q`
Expected: PASS, unchanged (the n_jobs cases in the bad-argument test still raise the same message).

- [ ] **Step 3: Commit**

```bash
git add src/datanuggets/_validate.py src/datanuggets/create.py
git commit -F <msg>   # "Share the n_jobs check between create_dn and refine_dn"
```

- [ ] **Step 4: Write the R fixture script**

```r file=r/fixtures_refine_dn.R
# Fixtures for refine.DN: the input nuggets, every random draw, every stats::kmeans call (the data
# it clustered and the clusters it returned), the messages and warnings, and the refined nuggets.
source("r/fixture_io.R")

grid <- as_dbl(expand.grid(a = 0:5, b = 0:5))
datasets <- list(
  faithful = as_dbl(faithful),
  iris = as_dbl(iris[, 1:4]),
  quakes = as_dbl(quakes),
  petal = as_dbl(iris[, 3:4]),                                     # heavy duplicates
  faithful_1d = as_dbl(faithful[, "eruptions", drop = FALSE]),
  volcano_scaled = as_dbl(scale(volcano)),                         # p = 61
  grid_ties = grid[rep(seq_len(nrow(grid)), 4), , drop = FALSE]    # every point 4 times
)
# create.DN's R, DN.num1 and DN.num2 for the input nuggets: about 10 to 20 rows each
dn_sizes <- list(faithful = c(100, 90, 20), iris = c(100, 60, 12), quakes = c(300, 400, 50),
                 petal = c(100, 100, 12), faithful_1d = c(100, 90, 20),
                 volcano_scaled = c(100, 40, 8), grid_ties = c(100, 60, 12))

make_dn <- function(ds, center.method = "mean") {
  s <- dn_sizes[[ds]]
  suppressWarnings(suppressMessages(create.DN(datasets[[ds]], center.method = center.method,
    R = s[1], DN.num1 = s[2], DN.num2 = s[3], no.cores = 1, make.pbs = FALSE)))
}

refine_args <- function(EV.tol = 0.9, max.splits = 5, min.nugget.size = 2, shape.split = FALSE,
                        min.shape.size = 10, delta = 2, nstart = 25, seed = 291102) {
  list(EV.tol = EV.tol, max.splits = max.splits, min.nugget.size = min.nugget.size,
       shape.split = shape.split, min.shape.size = min.shape.size, delta = delta,
       nstart = nstart, seed = seed)
}

# refine.DN reaches stats::kmeans through `::`, which resolves the namespace binding. Trace it
# and the attached copy, recording the data each call clusters and the clusters it returns
# (NULL when kmeans stops with an error, which refine.DN catches).
capture_kmeans <- function(expr) {
  .capture$km <- list()
  hook <- quote(.capture$km[[length(.capture$km) + 1L]] <- list(
    x = as.matrix(x), cluster = returnValue(NULL)$cluster))
  envs <- list(asNamespace("stats"), as.environment("package:stats"))
  for (e in envs) trace("kmeans", exit = hook, print = FALSE, where = e)
  on.exit(for (e in envs) suppressWarnings(untrace("kmeans", where = e)), add = TRUE)
  value <- expr
  list(value = value, calls = .capture$km)
}

run_refine <- function(x, dn, args) {
  msgs <- character()
  warns <- character()
  res <- capture_kmeans(capture_draws(withCallingHandlers(
    do.call(refine.DN, c(list(x = x, DN = dn), args, list(no.cores = 1, make.pbs = FALSE))),
    message = function(m) {
      msgs <<- c(msgs, sub("\n$", "", conditionMessage(m)))
      invokeRestart("muffleMessage")
    },
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })))
  list(dn = res$value$value, draws = res$value$draws, km = res$calls, messages = msgs,
       warnings = warns)
}

# refine.DN's get_nugget_features on the input nuggets, before any split
nugget_stats <- function(x, a, k) {
  p <- ncol(x)
  out <- vapply(seq_len(k), function(i) {
    obs <- x[a == i, , drop = FALSE]
    m <- nrow(obs)
    if (m == 0L) return(c(rep(NA_real_, p), 0, 0, 0))
    ctr <- colMeans(obs)
    if (m == 1L) return(c(ctr, 0, 0, 0))
    S <- cov(obs)
    ev <- if (p == 1L) as.numeric(S) else eigen(S, symmetric = TRUE, only.values = TRUE)$values
    c(ctr, sum(diag(S)) / p, ev[1], if (length(ev) > 1L) ev[2] else ev[1])
  }, numeric(p + 3L))
  list(init_centers = t(out[seq_len(p), , drop = FALSE]), init_scale = out[p + 1L, ],
       init_ev1 = out[p + 2L, ], init_ev2 = out[p + 3L, ])
}

km_arrays <- function(km) list(
  km_nrow = as.integer(vapply(km, function(k) as.numeric(nrow(k$x)), numeric(1))),
  km_ncol = as.integer(vapply(km, function(k) as.numeric(ncol(k$x)), numeric(1))),
  km_ok = as.integer(vapply(km, function(k) !is.null(k$cluster), logical(1))),
  km_x = as.double(unlist(lapply(km, function(k) t(k$x)))),   # row-major, calls back to back
  km_cluster = as.integer(unlist(lapply(km, function(k) k$cluster)))
)

all_paths <- list()
add_refine <- function(id, ds, dn, input, args = refine_args()) {
  x <- datasets[[ds]]
  p <- ncol(x)
  r <- run_refine(x, dn, args)
  n_rounds <- sum(grepl("^split #", r$messages))
  if (length(r$km) < n_rounds) {
    stop(id, ": ", n_rounds, " split rounds but only ", length(r$km), " kmeans calls recorded")
  }
  ok <- vapply(r$km, function(k) !is.null(k$cluster), logical(1))
  mns <- floor(args$min.nugget.size)
  paths <- list(
    kmeans_error = any(!ok),
    child_too_small = any(vapply(r$km[ok], function(k) min(tabulate(k$cluster, 2L)) < mns,
                                 logical(1))))
  in_tab <- dn[["Data Nuggets"]]
  in_a <- as.integer(dn[["Data Nugget Assignments"]])
  tab <- r$dn[["Data Nuggets"]]
  arrays <- c(
    list(x = x,
         in_centers = unname(as.matrix(in_tab[, 1 + seq_len(p)])),
         in_weights = as.integer(in_tab[["Weight"]]),
         in_scales = as.double(in_tab[["Scale"]]),
         in_assignments = in_a),
    nugget_stats(x, in_a, nrow(in_tab)),
    draw_arrays(r$draws),
    km_arrays(r$km),
    list(centers = unname(as.matrix(tab[, 1 + seq_len(p)])),
         weights = as.integer(tab[["Weight"]]),
         scales = as.double(tab[["Scale"]]),
         assignments = as.integer(r$dn[["Data Nugget Assignments"]])))
  write_case("refine_dn", id, args, arrays, extra = list(
    dataset = ds, input = input, messages = I(r$messages), warnings = I(r$warnings),
    paths = paths))
  all_paths[[id]] <<- paths
  cat(sprintf("%-40s nuggets %3d -> %3d  kmeans=%3d draws=%4d%s%s\n", id, nrow(in_tab),
              nrow(tab), length(r$km), length(r$draws),
              if (paths$kmeans_error) "  kmeans-error" else "",
              if (paths$child_too_small) "  small-child" else ""))
  invisible(r)
}

# The spec's grid on faithful: ev_tol x max_splits x min_nugget_size x shape_split x delta x nstart
dn_f <- make_dn("faithful")
for (ev in c(0.5, 0.9)) for (ms in c(0, 1, 5)) for (mns in c(2, 5)) for (shape in c(FALSE, TRUE))
  for (d in c(1.5, 2)) for (ns in c(1, 25)) {
    id <- sprintf("faithful_ev%s_ms%d_mns%d_%s_d%s_ns%d", ev, ms, mns,
                  if (shape) "shape" else "noshape", d, ns)
    add_refine(id, "faithful", dn_f, "mean", refine_args(
      EV.tol = ev, max.splits = ms, min.nugget.size = mns, shape.split = shape, delta = d,
      nstart = ns))
  }

for (ds in setdiff(names(datasets), "faithful")) {
  dn <- make_dn(ds)
  add_refine(paste0(ds, "_default"), ds, dn, "mean")
  add_refine(paste0(ds, "_shape"), ds, dn, "mean", refine_args(shape.split = TRUE))
  add_refine(paste0(ds, "_shape_loose"), ds, dn, "mean",
             refine_args(EV.tol = 0.5, shape.split = TRUE, delta = 1.5))
  add_refine(paste0(ds, "_nstart1"), ds, dn, "mean", refine_args(nstart = 1))
}

# kmeans(nstart = 0) stops with "argument is of length zero", so every split fails
add_refine("faithful_nstart0", "faithful", dn_f, "mean", refine_args(nstart = 0))
# refine.DN recomputes every center, so the input's center method must not matter
add_refine("faithful_input_random", "faithful", make_dn("faithful", "random"), "random")
add_refine("faithful_input_original", "faithful", make_dn("faithful", "original"), "original")
# max.splits and min.nugget.size are floored, min.shape.size is not, nstart runs floor(nstart)
add_refine("faithful_floored_args", "faithful", dn_f, "mean", refine_args(
  max.splits = 2.7, min.nugget.size = 2.5, shape.split = TRUE, min.shape.size = 10.5,
  delta = 1.5, nstart = 25.7))
# refine.DN applied to its own output
chain_in <- run_refine(datasets$faithful, dn_f, refine_args(shape.split = TRUE))$dn
add_refine("faithful_chain", "faithful", chain_in, "refined", refine_args(shape.split = TRUE))
# An empty nugget in the middle (create.DN keeps empty nuggets): refine drops it and renumbers
k_f <- nrow(dn_f[["Data Nuggets"]])
tab_e <- dn_f[["Data Nuggets"]][c(1:4, 4, 5:k_f), ]
tab_e[5, "Weight"] <- 0
tab_e[5, "Scale"] <- 0
tab_e[["Data Nugget"]] <- seq_len(k_f + 1L)
rownames(tab_e) <- NULL
a_f <- as.integer(dn_f[["Data Nugget Assignments"]])
dn_e <- structure(list(`Data Nuggets` = tab_e,
                       `Data Nugget Assignments` = ifelse(a_f >= 5L, a_f + 1L, a_f)),
                  class = "datanugget")
add_refine("faithful_empty_middle", "faithful", dn_e, "empty_middle")
# Every row its own nugget: no positive first eigenvalue, so no EV split round at all
n_f <- nrow(datasets$faithful)
dn_s <- structure(list(
  `Data Nuggets` = data.frame(`Data Nugget` = seq_len(n_f), Center1 = datasets$faithful[, 1],
                              Center2 = datasets$faithful[, 2], Weight = 1L, Scale = 0,
                              check.names = FALSE),
  `Data Nugget Assignments` = seq_len(n_f)), class = "datanugget")
add_refine("faithful_singletons", "faithful", dn_s, "singletons", refine_args(shape.split = TRUE))
# nstart = 1 on duplicate rows: find a seed where some split's two random starts are equal rows,
# so kmeans redraws from unique(x) and R makes more draws than kmeans calls
dn_p <- make_dn("petal")
dup_seed <- Find(function(s) {
  r <- run_refine(datasets$petal, dn_p, refine_args(nstart = 1, seed = s))
  length(r$draws) > length(r$km)
}, seq_len(200))
if (is.null(dup_seed)) stop("no seed in 1:200 makes a petal split redraw its starts")
add_refine("petal_nstart1_dups", "petal", dn_p, "mean", refine_args(nstart = 1, seed = dup_seed))

flags <- function(name) vapply(all_paths, function(p) p[[name]], logical(1))
if (!any(flags("kmeans_error"))) stop("no case has a kmeans call that stopped with an error")
if (!any(flags("child_too_small"))) stop("no case has a split child below min.nugget.size")
cat(length(all_paths), "cases\n")
```

- [ ] **Step 5: Generate the fixtures**

Run: `RS r/fixtures_refine_dn.R`
Expected: one line per case, `128 cases` at the end, no `stop()`. `faithful_nstart0` shows `kmeans-error`; at least one `mns5` case shows `small-child`.

- [ ] **Step 6: Add refine_inputs to the fixture loader**

In `tests/rfixtures.py`, add `from datanuggets.result import DataNuggets` on the line after `from datanuggets._draws import DrawRecord`, then append:

```python file=tests/rfixtures.py mode=append
# refine.DN's argument names in the fixture manifests, and refine_dn's
REFINE_ARGS = {
    "EV.tol": "ev_tol",
    "max.splits": "max_splits",
    "min.nugget.size": "min_nugget_size",
    "shape.split": "shape_split",
    "min.shape.size": "min_shape_size",
    "delta": "delta",
    "nstart": "nstart",
}


def refine_inputs(case: Case) -> tuple[np.ndarray, DataNuggets, dict]:
    """x, the input nuggets (assignments made 0-based) and refine_dn's keyword arguments."""
    dn = DataNuggets(
        centers=case.array("in_centers"),
        weights=case.array("in_weights"),
        scales=case.array("in_scales"),
        assignments=case.array("in_assignments") - 1,
    )
    return case.array("x"), dn, {py: case.args[r] for r, py in REFINE_ARGS.items()}
```

- [ ] **Step 7: Write the failing tests**

```python file=tests/test_refine.py
"""refine_dn against R's refine.DN: every kmeans call, message, warning and output."""

import logging
import warnings
from dataclasses import replace

import numpy as np
import pytest

import datanuggets.refine as refine_module
from datanuggets import DataNuggetWarning, DataNuggets, refine_dn
from datanuggets._draws import ReplayDraws
from datanuggets.refine import nugget_features
from rfixtures import ATOL, RTOL, load_group, refine_inputs

CASES = load_group("refine_dn")
# The rejection reasons refine_dn logs at DEBUG, keyed by the flag R's fixture records
REASONS = {
    "kmeans_error": "kmeans error",
    "child_too_small": "a child is smaller than min_nugget_size",
}
SHAPE_REJECTED = "mean EV1/EV2 of the children >= delta"


def _case(case_id):
    return next(c for c in CASES if c.id == case_id)


@pytest.fixture
def kmeans_calls(monkeypatch):
    """The data and 0-based clusters (None on an error) of every kmeans call refine_dn makes."""
    calls = []
    real = refine_module.kmeans

    def spy(x, k, draws, **kwargs):
        call = [np.array(x), None]
        calls.append(call)
        result = real(x, k, draws, **kwargs)
        call[1] = result.cluster
        return result

    monkeypatch.setattr(refine_module, "kmeans", spy)
    return calls


def _r_kmeans_calls(case):
    """R's kmeans calls from the fixture: (data, 1-based clusters or None on an error)."""
    shapes = zip(
        case.array("km_nrow").tolist(),
        case.array("km_ncol").tolist(),
        case.array("km_ok").tolist(),
        strict=True,
    )
    xs, clusters = case.array("km_x"), case.array("km_cluster")
    calls, i, j = [], 0, 0
    for rows, cols, ok in shapes:
        x = xs[i : i + rows * cols].reshape(rows, cols)
        i += rows * cols
        cluster = clusters[j : j + rows] if ok else None
        j += rows if ok else 0
        calls.append((x, cluster))
    assert (i, j) == (xs.size, clusters.size)
    return calls


def _run(case, caplog):
    x, dn, kwargs = refine_inputs(case)
    draws = ReplayDraws(case.draws())
    caplog.set_level(logging.DEBUG, logger="datanuggets")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = refine_dn(x, dn, seed=draws, **kwargs)
    draws.assert_exhausted()
    return got, caught


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_refine_matches_r(case, kmeans_calls, caplog):
    got, caught = _run(case, caplog)
    p = case.array("x").shape[1]
    want = _r_kmeans_calls(case)
    assert len(kmeans_calls) == len(want)
    for (gx, gc), (wx, wc) in zip(kmeans_calls, want, strict=True):
        if wx.shape[1] == p:
            np.testing.assert_array_equal(gx, wx)
        else:  # a shape split: the projection on an eigenvector whose sign LAPACK picks
            sign = 1.0 if gx[:, 0] @ wx[:, 0] >= 0 else -1.0
            np.testing.assert_allclose(gx * sign, wx, rtol=RTOL, atol=ATOL)
        if wc is None:
            assert gc is None
        else:
            np.testing.assert_array_equal(gc + 1, wc)
    np.testing.assert_array_equal(got.assignments + 1, case.array("assignments"))
    np.testing.assert_array_equal(got.weights, case.array("weights"))
    np.testing.assert_array_equal(got.centers, case.array("centers"))
    np.testing.assert_array_equal(got.scales, case.array("scales"))
    records = [r for r in caplog.records if r.name == "datanuggets"]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert infos == case.manifest["messages"]
    texts = [str(w.message) for w in caught if w.category is DataNuggetWarning]
    assert texts == case.manifest["warnings"]
    debug = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
    for flag, reason in REASONS.items():
        assert any(reason in m for m in debug) == case.manifest["paths"][flag], flag


def test_every_rejection_path_is_exercised(caplog):
    """R's calls show kmeans errors and too-small children (the parity test ties each case's
    Python reasons to them). A shape split rejected on its children's EV1/EV2 leaves no trace in
    R's calls, so find one on the Python side: exact output parity on that case means R rejected
    the same split."""
    for flag in REASONS:
        assert any(c.manifest["paths"][flag] for c in CASES), flag
    for case in CASES:
        if not case.args["shape.split"] or case.array("x").shape[1] < 2:
            continue
        caplog.clear()
        _run(case, caplog)
        if any(SHAPE_REJECTED in r.getMessage() for r in caplog.records):
            return
    pytest.fail("no fixture rejects a shape split on its children's EV1/EV2")


def _first_case_per_input():
    first = {}
    for c in CASES:
        first.setdefault((c.manifest["dataset"], c.manifest["input"]), c)
    return list(first.values())


@pytest.mark.parametrize("case", _first_case_per_input(), ids=lambda c: c.id)
def test_initial_nugget_features_match_r(case):
    x, dn, _ = refine_inputs(case)
    feats = [nugget_features(x[dn.assignments == i]) for i in range(dn.n_nuggets)]
    np.testing.assert_array_equal(np.array([f[0] for f in feats]), case.array("init_centers"))
    np.testing.assert_array_equal([f[2] for f in feats], case.array("init_scale"))
    ev1, ev2 = [f[3] for f in feats], [f[4] for f in feats]
    if x.shape[1] == 1:  # EV1 = EV2 = the variance, which is exact
        np.testing.assert_array_equal(ev1, case.array("init_ev1"))
        np.testing.assert_array_equal(ev2, case.array("init_ev2"))
    else:
        np.testing.assert_allclose(ev1, case.array("init_ev1"), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ev2, case.array("init_ev2"), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "make_seed", [lambda: 11, lambda: np.random.default_rng(11)], ids=["int", "generator"]
)
def test_n_jobs_does_not_change_the_result(make_seed):
    x, dn, kwargs = refine_inputs(_case("faithful_ev0.9_ms5_mns2_shape_d2_ns25"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        one = refine_dn(x, dn, seed=make_seed(), n_jobs=1, **kwargs)
        two = refine_dn(x, dn, seed=make_seed(), n_jobs=2, **kwargs)
    assert one.n_nuggets > np.count_nonzero(dn.weights)  # something was split
    assert one == two


def _tiny():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    dn = DataNuggets(
        centers=np.zeros((2, 2)),
        weights=np.array([3, 1]),
        scales=np.zeros(2),
        assignments=np.array([0, 0, 0, 1]),
    )
    return x, dn


BAD_ARGS = [
    ({"ev_tol": 0}, ValueError, r"ev_tol must be within \(0, 1\)"),
    ({"ev_tol": 1}, ValueError, r"ev_tol must be within \(0, 1\)"),
    ({"ev_tol": "0.5"}, TypeError, "ev_tol must be a number"),
    ({"max_splits": -1}, ValueError, "max_splits must be a finite number >= 0"),
    ({"max_splits": float("inf")}, ValueError, "max_splits must be a finite number >= 0"),
    ({"min_nugget_size": 1.5}, ValueError, "min_nugget_size must be at least 2"),
    ({"min_nugget_size": float("nan")}, ValueError, "min_nugget_size must be at least 2"),
    ({"shape_split": 1}, TypeError, "shape_split must be True or False"),
    ({"min_shape_size": 9}, ValueError, "min_shape_size must be at least 10"),
    ({"delta": 1}, ValueError, "delta must be greater than 1"),
    ({"nstart": float("nan")}, ValueError, "nstart must be finite"),
    ({"n_jobs": 0}, ValueError, "n_jobs must be a positive int or -1"),
    ({"seed": "abc"}, TypeError, "seed must be an int"),
]


@pytest.mark.parametrize("kwargs, exc, match", BAD_ARGS)
def test_bad_arguments_are_rejected(kwargs, exc, match):
    x, dn = _tiny()
    with pytest.raises(exc, match=match):
        refine_dn(x, dn, **kwargs)


def test_dn_must_be_datanuggets():
    x, dn = _tiny()
    with pytest.raises(TypeError, match="dn must be a DataNuggets"):
        refine_dn(x, {"assignments": dn.assignments})


def test_x_needs_a_row():
    _, dn = _tiny()
    with pytest.raises(ValueError, match="at least one observation"):
        refine_dn(np.zeros((0, 2)), replace(dn, assignments=np.array([], dtype=int)))


def test_assignments_must_cover_every_row():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="one entry per row of x"):
        refine_dn(x[:3], dn)


@pytest.mark.parametrize("bad", [[0, 0, 0, 2], [-1, 0, 0, 1]])
def test_assignments_must_be_nugget_ids(bad):
    x, dn = _tiny()
    with pytest.raises(ValueError, match=r"nugget ids in \[0, 2\)"):
        refine_dn(x, replace(dn, assignments=np.array(bad)))


def test_assignments_must_be_whole_numbers():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="whole numbers"):
        refine_dn(x, replace(dn, assignments=np.array([0, 0.5, 0, 1])))
    whole = refine_dn(x, replace(dn, assignments=np.array([0.0, 0.0, 0.0, 1.0])))
    assert whole == refine_dn(x, dn)


def test_dn_from_data_with_other_columns_is_rejected():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="x has 3 columns"):
        refine_dn(np.hstack([x, x[:, :1]]), dn)


def test_replayed_draws_need_n_jobs_1():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="n_jobs=1"):
        refine_dn(x, dn, seed=ReplayDraws([]), n_jobs=2)
```

- [ ] **Step 8: Run the tests and watch them fail**

Run: `UV run pytest tests/test_refine.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'datanuggets.refine'`.

- [ ] **Step 9: Write refine.py**

```python file=src/datanuggets/refine.py
"""refine.DN: split data nuggets that are spread out or elongated.

Everything that reaches the result (memberships, colMeans centers, long double scales) matches
R bit for bit. Split decisions compare eigenvalues, which come from the same LAPACK routine R
calls (dsyevr) but from scipy's build of it, so they agree with R's to rounding. A decision can
only differ from R's when two eigenvalues, or an eigenvalue and a threshold, agree to about 12
significant digits.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from datanuggets._defaults import (
    DELTA,
    EV_TOL,
    KMEANS_ITER_MAX,
    MAX_SPLITS,
    MIN_NUGGET_SIZE,
    MIN_SHAPE_SIZE,
    NSTART,
    SEED,
)
from datanuggets._draws import Draws, NumpyDraws, as_draws
from datanuggets._hartigan_wong import KMeansError, kmeans
from datanuggets._rcompat import r_mean
from datanuggets._rstats import (
    r_cov,
    r_eigen_sym,
    r_mad,
    r_matvec,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
from datanuggets._validate import as_matrix, as_real, check_n_jobs
from datanuggets.create import _progress_bar
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import DataNuggets

log = logging.getLogger("datanuggets")


def refine_dn(
    x,
    dn: DataNuggets,
    ev_tol: float = EV_TOL,
    max_splits: float = MAX_SPLITS,
    min_nugget_size: float = MIN_NUGGET_SIZE,
    shape_split: bool = False,
    min_shape_size: float = MIN_SHAPE_SIZE,
    delta: float = DELTA,
    nstart: float = NSTART,
    seed=SEED,
    n_jobs: int = 1,
    progress: bool = False,
) -> DataNuggets:
    """Port of R's refine.DN (datanugget 1.5.0).

    Only dn's assignments are used: every nugget's center, weight, scale and two largest
    eigenvalues are recomputed from its rows of x. For up to max_splits rounds, nuggets whose
    first eigenvalue is above the ev_tol quantile of the positive ones (taken once, before the
    first round) are split in two by Hartigan-Wong k-means. With shape_split=True and two or
    more columns, up to max_splits more rounds split elongated nuggets (EV1/EV2 above delta, or
    EV1 above the median plus twice the mad of the scales) along their first eigenvector,
    keeping a split only if its children are rounder. Empty nuggets are dropped and the rest
    renumbered from 0.

    R re-seeds before every split, so each split draws from a fresh copy of seed's stream
    (Draws.fork) and the result is the same for any n_jobs. n_jobs other than 1 runs each
    round's splits in joblib worker processes. progress=True shows a tqdm bar per round.
    """
    x = as_matrix(x)
    if not isinstance(dn, DataNuggets):
        raise TypeError(
            f"dn must be a DataNuggets from create_dn or refine_dn, not {type(dn).__name__}"
        )
    ev_tol = as_real("ev_tol", ev_tol)
    if not 0 < ev_tol < 1:
        raise ValueError(f"ev_tol must be within (0, 1), got {ev_tol!r}")
    max_splits = as_real("max_splits", max_splits)
    if not 0 <= max_splits < math.inf:
        raise ValueError(f"max_splits must be a finite number >= 0, got {max_splits!r}")
    min_nugget_size = as_real("min_nugget_size", min_nugget_size)
    if not min_nugget_size >= 2:
        raise ValueError(f"min_nugget_size must be at least 2, got {min_nugget_size!r}")
    if not isinstance(shape_split, (bool, np.bool_)):
        raise TypeError(f"shape_split must be True or False, got {shape_split!r}")
    min_shape_size = as_real("min_shape_size", min_shape_size)
    if not min_shape_size >= 10:
        raise ValueError(f"min_shape_size must be at least 10, got {min_shape_size!r}")
    delta = as_real("delta", delta)
    if not delta > 1:
        raise ValueError(f"delta must be greater than 1, got {delta!r}")
    nstart = as_real("nstart", nstart)
    if not math.isfinite(nstart):
        raise ValueError(f"nstart must be finite, got {nstart!r}")
    check_n_jobs(n_jobs)
    draws = as_draws(seed)
    if n_jobs != 1 and not isinstance(draws, NumpyDraws):
        raise ValueError(
            "replayed or custom Draws need n_jobs=1: their stream can't be copied into worker "
            "processes. Pass an int seed or a numpy Generator to split in parallel."
        )
    bar = _progress_bar() if progress else None
    n, p = x.shape
    if n == 0:
        raise ValueError("x must contain at least one observation")
    state = _Nuggets(x, _checked_assignments(dn, n, p), dn.centers.shape[0])
    state.refresh(range(state.size), state.members())
    max_splits = math.floor(max_splits)
    mns = math.floor(min_nugget_size) if math.isfinite(min_nugget_size) else min_nugget_size

    def run_round(cands: np.ndarray, project: bool, desc: str) -> bool:
        """One round of refine.DN's run_splitting and apply_splits. False if nothing split."""
        members = state.members()
        jobs = [
            delayed(_split_nugget)(x[members[c]], draws.fork(), mns, nstart, project, delta)
            for c in cands.tolist()
        ]
        if n_jobs == 1:
            results = (f(*args, **kwargs) for f, args, kwargs in jobs)
        else:
            results = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
        if bar is not None:
            results = bar(results, total=len(jobs), desc=desc)
        splits = []
        for c, out in zip(cands.tolist(), results, strict=True):
            for message in out.warned:
                warnings.warn(message, DataNuggetWarning, stacklevel=3)
            if out.labels is None:
                log.debug("nugget %d not split: %s", c, out.reason)
            else:
                splits.append((c, members[c], out.labels))
        if not splits:
            return False
        state.refresh(state.apply(splits), state.members())
        return True

    if max_splits > 0:
        positive = [v for v in state.ev1.tolist() if math.isfinite(v) and v > 0]
        if not positive:
            log.info("No data nugget has a positive first eigen value")
        else:
            threshold = r_quantile7(positive, ev_tol)
            log.info("Splitting data nuggets according to max EV...")
            for attempt in range(1, max_splits + 1):
                ev1 = state.ev1
                cands = np.flatnonzero(
                    np.isfinite(ev1) & (ev1 > threshold) & (state.weights >= 2 * mns)
                )
                if cands.size == 0:
                    break
                log.info("split #%d", attempt)
                if not run_round(cands, project=False, desc=f"EV split #{attempt}"):
                    break
            log.info("complete!")

    if shape_split and p >= 2 and max_splits > 0:
        log.info("Splitting data nuggets according to shape...")
        min_size = max(min_shape_size, 2 * mns)
        for attempt in range(1, max_splits + 1):
            cands = _shape_candidates(state, min_size, delta)
            if cands.size == 0:
                break
            log.info("split #%d", attempt)
            if not run_round(cands, project=True, desc=f"shape split #{attempt}"):
                break
        log.info("complete")

    log.info("Renumbering the data nuggets...")
    result = state.finalize()
    log.info("completed!")
    return result


def nugget_features(obs: np.ndarray) -> tuple[np.ndarray, int, float, float, float]:
    """refine.DN's get_nugget_features for one nugget's rows: center (colMeans), weight, scale
    (the trace of the sample covariance over p) and the two largest eigenvalues. An empty nugget
    has a NaN center, a single row has scale and eigenvalues 0, and with one column EV2 = EV1."""
    m, p = obs.shape
    if m == 0:
        return np.full(p, np.nan), 0, 0.0, 0.0, 0.0
    center = r_mean(obs, axis=0)
    if m == 1:
        return center, 1, 0.0, 0.0, 0.0
    s = r_cov(obs)
    scale = r_sum(np.diag(s)) / p
    ev = s.ravel() if p == 1 else r_eigen_sym(s)
    return center, m, scale, float(ev[0]), float(ev[1] if ev.size > 1 else ev[0])


@dataclass(frozen=True)
class _Outcome:
    """One split attempt's answer to the main process: each row's child (0 or 1), or None and
    the reason the nugget stays whole, plus the warnings kmeans raised."""

    labels: np.ndarray | None
    reason: str | None
    warned: tuple[str, ...] = ()


def _split_nugget(
    obs: np.ndarray,
    draws: Draws,
    min_nugget_size: float,
    nstart: float,
    project: bool,
    delta: float,
) -> _Outcome:
    """refine.DN's split_nugget, plus the shape check on the children when project is True."""
    if obs.shape[0] < 2 * min_nugget_size:
        return _Outcome(None, "fewer than 2 * min_nugget_size rows")
    data = obs
    if project:
        _, vectors = r_eigen_sym(r_cov(obs), vectors=True)
        data = r_matvec(obs, vectors[:, 0])[:, None]
    try:
        km = kmeans(data, 2, draws, nstart=nstart, iter_max=KMEANS_ITER_MAX)
    except KMeansError as err:
        return _Outcome(None, f"kmeans error: {err}", err.warnings)
    if np.bincount(km.cluster, minlength=2).min() < min_nugget_size:
        return _Outcome(None, "a child is smaller than min_nugget_size", km.warnings)
    if project:
        ratios = [_shape_ratio(obs[km.cluster == g]) for g in (0, 1)]
        if any(math.isnan(r) for r in ratios) or r_vec_mean(ratios) >= delta:
            return _Outcome(None, "mean EV1/EV2 of the children >= delta", km.warnings)
    return _Outcome(km.cluster, None, km.warnings)


def _shape_ratio(obs: np.ndarray) -> float:
    """EV1/EV2 of one child of a shape split: NaN below 2 rows, Inf when EV2 <= 0."""
    if obs.shape[0] < 2:
        return math.nan
    ev = r_eigen_sym(r_cov(obs))
    if ev.size < 2 or ev[1] <= 0:
        return math.inf
    return float(ev[0] / ev[1])


def _shape_candidates(state: _Nuggets, min_size: float, delta: float) -> np.ndarray:
    """Positions big enough to split that are elongated (EV1/EV2 > delta) or spread out (EV1
    above the median plus twice the mad of the scales of nuggets with 2 or more rows)."""
    w, ev1, ev2 = state.weights, state.ev1, state.ev2
    multi = state.scales[np.isfinite(state.scales) & (w > 1)].tolist()
    s_threshold = r_median(multi) + 2 * r_mad(multi) if multi else math.inf
    ratio = np.full(ev1.shape, math.inf)
    np.divide(ev1, ev2, out=ratio, where=np.isfinite(ev2) & (ev2 > 0))
    big = (w >= min_size) & np.isfinite(ev1) & (ev1 > 0)
    return np.flatnonzero(big & ((ratio > delta) | (ev1 > s_threshold)))


def _checked_assignments(dn: DataNuggets, n: int, p: int) -> np.ndarray:
    """dn's assignments as 0-based positions, after checking that dn describes these rows."""
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
    if a.min() < 0 or a.max() >= k:
        raise ValueError(
            f"dn.assignments must be nugget ids in [0, {k}), got values from {a.min()} to {a.max()}"
        )
    return a.astype(np.intp)


class _Nuggets:
    """refine.DN's working state, one entry per position (R's id minus 1). A split appends its
    new nugget at the end, so positions stay stable until finalize() renumbers them."""

    def __init__(self, x: np.ndarray, assignments: np.ndarray, k: int):
        self.x = x
        self.assignments = assignments.copy()
        self.centers = np.full((k, x.shape[1]), np.nan)
        self.weights = np.zeros(k, dtype=np.int64)
        self.scales = np.zeros(k)
        self.ev1 = np.zeros(k)
        self.ev2 = np.zeros(k)

    @property
    def size(self) -> int:
        return self.weights.size

    def members(self) -> list[np.ndarray]:
        """R's split(seq_len(n), factor(assignments, levels = ids)): each position's rows in
        ascending order."""
        order = np.argsort(self.assignments, kind="stable")
        counts = np.bincount(self.assignments, minlength=self.size)
        return np.split(order, np.cumsum(counts)[:-1])

    def refresh(self, positions, members: list[np.ndarray]) -> None:
        """refresh_nuggets_info: recompute these positions' stats from their rows."""
        for c in positions:
            center, weight, scale, ev1, ev2 = nugget_features(self.x[members[c]])
            self.centers[c] = center
            self.weights[c] = weight
            self.scales[c] = scale
            self.ev1[c] = ev1
            self.ev2[c] = ev2

    def apply(self, splits: list[tuple[int, np.ndarray, np.ndarray]]) -> list[int]:
        """apply_splits: each split's rows in child 1 (R's cluster 2) move to a new position,
        appended in split order. Returns the parent and child positions to refresh."""
        k, new = self.size, len(splits)
        self.centers = np.vstack([self.centers, np.full((new, self.x.shape[1]), np.nan)])
        self.weights = np.concatenate([self.weights, np.zeros(new, dtype=np.int64)])
        self.scales = np.concatenate([self.scales, np.zeros(new)])
        self.ev1 = np.concatenate([self.ev1, np.zeros(new)])
        self.ev2 = np.concatenate([self.ev2, np.zeros(new)])
        touched = []
        for i, (c, rows, labels) in enumerate(splits):
            self.assignments[rows[labels == 1]] = k + i
            touched += [c, k + i]
        return touched

    def finalize(self) -> DataNuggets:
        """Drop empty nuggets, renumber the rest in position order, zero the scale of
        single-row nuggets and check R's stopifnot() invariants."""
        keep = self.weights > 0
        assignments = (np.cumsum(keep) - 1)[self.assignments]
        weights = self.weights[keep]
        scales = self.scales[keep]
        scales[weights < 2] = 0.0
        if not np.array_equal(np.bincount(assignments, minlength=weights.size), weights):
            raise AssertionError("refine_dn: the weights no longer match the assignments")
        return DataNuggets(
            centers=self.centers[keep], weights=weights, scales=scales, assignments=assignments
        )
```

- [ ] **Step 10: Export refine_dn**

In `src/datanuggets/__init__.py`, add `from datanuggets.refine import refine_dn` after `from datanuggets.create import create_dn`, and change `__all__` to:

```python
__all__ = [
    "CentersResult",
    "DataNuggetWarning",
    "DataNuggets",
    "create_dn",
    "create_dn_centers",
    "refine_dn",
]
```

- [ ] **Step 11: Run the refine tests**

Run: `UV run pytest tests/test_refine.py -q`
Expected: PASS (128 parity cases, the rejection-path test, 12 initial-feature cases, 2 n_jobs, 13 bad arguments, 8 dn checks).

- [ ] **Step 12: Run the whole suite and the linters**

Run: `UV run pytest -q && UV run ruff check . && UV run ruff format --check .`
Expected: all pass.

- [ ] **Step 13: Commit**

```bash
git add src/datanuggets/refine.py src/datanuggets/__init__.py r/fixtures_refine_dn.R \
  tests/rfixtures.py tests/test_refine.py tests/fixtures/refine_dn
git commit -F <msg>   # "Port refine_dn with exact parity against R's refine.DN"
```

---

### Task 5: Parity report and docs

**Files:**
- Modify: `scripts/parity_report.py` (whole file), `README.md`, `r/README.md`
- Create: `docs/validation/refine_dn.md` (generated)

**Interfaces:**
- Consumes: `refine_dn`, `rfixtures.refine_inputs` (Task 4); the `create_dn` report (Feature 1), whose output must not change.
- Produces: `uv run python scripts/parity_report.py` writes both `docs/validation/create_dn.md` and `docs/validation/refine_dn.md`.

- [ ] **Step 1: Rewrite the parity report**

```python file=scripts/parity_report.py
"""Write docs/validation/create_dn.md and refine_dn.md: Python vs R on every fixture."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from datanuggets import create_dn, refine_dn
from datanuggets._draws import ReplayDraws
from rfixtures import load_group, refine_inputs


def verdict(a: np.ndarray, b: np.ndarray) -> str:
    """'yes' when bit-identical, else the largest relative difference."""
    if np.array_equal(a, b):
        return "yes"
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    return f"**NO** (max rel. {np.max(rel):.1e})"


def _replayed(case, fn, *args, **kwargs):
    """fn with R's draws replayed and warnings silenced; checks every draw was consumed."""
    draws = ReplayDraws(case.draws())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(*args, seed=draws, **kwargs)
    draws.assert_exhausted()
    return out


def _outputs(case, dn) -> str:
    return (
        f"{verdict(dn.assignments + 1, case.array('assignments'))} | "
        f"{verdict(dn.weights, case.array('weights'))} | "
        f"{verdict(dn.centers, case.array('centers'))} | "
        f"{verdict(dn.scales, case.array('scales'))} |"
    )


def create_report() -> str:
    rows = []
    cases = load_group("create_dn")
    for case in cases:
        a = case.args
        dn = _replayed(
            case,
            create_dn,
            case.array("x"),
            center_method=a["center.method"],
            r=a["R"],
            delete_percent=a["delete.percent"],
            dn_num1=a["DN.num1"],
            dn_num2=a["DN.num2"],
            dist_metric=a["dist.metric"],
        )
        n, p = case.array("x").shape
        rows.append(f"| {case.id} | {n} x {p} | {dn.n_nuggets} | {_outputs(case, dn)}")
    v = cases[0].manifest["versions"]
    header = (
        "# create_dn parity with R\n\n"
        "Generated by `scripts/parity_report.py` from the fixtures in `tests/fixtures/create_dn/`"
        f" ({v['R']}, datanugget {v['datanugget']}, Rfast {v['Rfast']}, {v['platform']},"
        f" BLAS: {v['blas']}, long double: {v['longdouble_digits']}-bit significand)."
        " R's random draws are replayed, so any difference would be arithmetic, not randomness."
        ' "yes" means bit-identical (`np.array_equal`).\n\n'
        "| case | data | nuggets | assignments | weights | centers | scales |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"


def refine_report() -> str:
    rows = []
    cases = load_group("refine_dn")
    for case in cases:
        x, dn_in, kwargs = refine_inputs(case)
        dn = _replayed(case, refine_dn, x, dn_in, **kwargs)
        n, p = x.shape
        rows.append(
            f"| {case.id} | {n} x {p} | {dn_in.n_nuggets} | {dn.n_nuggets} | {_outputs(case, dn)}"
        )
    v = cases[0].manifest["versions"]
    header = (
        "# refine_dn parity with R\n\n"
        "Generated by `scripts/parity_report.py` from the fixtures in `tests/fixtures/refine_dn/`"
        f" ({v['R']}, datanugget {v['datanugget']}, {v['platform']}, LAPACK {v['lapack']},"
        f" BLAS: {v['blas']}, long double: {v['longdouble_digits']}-bit significand)."
        " R's random draws are replayed, so any difference would be arithmetic, not randomness."
        ' "yes" means bit-identical (`np.array_equal`). Eigenvalues only steer which nuggets'
        " split. They come from LAPACK's dsyevr through SciPy, the routine R's eigen() calls,"
        " and the tests hold them to R's within a relative 1e-8.\n\n"
        "| case | data | nuggets in | nuggets out | assignments | weights | centers | scales |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    for name, report in (("create_dn", create_report), ("refine_dn", refine_report)):
        out = ROOT / "docs" / "validation" / f"{name}.md"
        out.write_text(report(), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate both reports**

Run: `UV run python scripts/parity_report.py && git diff --ignore-all-space --stat -- docs/validation/create_dn.md && grep -c "NO" docs/validation/refine_dn.md`
Expected: two paths printed; the create diff is empty apart from line endings (its content is unchanged); `0` rows contain `NO` in the refine report.

- [ ] **Step 3: Update the READMEs**

In `README.md`, replace the Status section body (from "Version `0.1.0.dev0` is in development." through "...data-nugget wrappers.") with:

```markdown
Version `0.1.0.dev0` is in development. Three functions are ported:

- `create_dn`, from R's `create.DN`
- `create_dn_centers`, from R's `create.DNcenters`
- `refine_dn`, from R's `refine.DN`

`refine_dn` splits nuggets that are spread out or elongated with 2-means, through a port of the Hartigan-Wong k-means in R's `stats::kmeans`. Next come `optimal_dn` and the remaining `datanugget` helpers, followed by `WCluster`'s weighted statistics, clustering methods and data-nugget wrappers.
```

Replace "Requires Python 3.12 or newer, `numpy >= 2.0` and `joblib >= 1.4`." with "Requires Python 3.12 or newer, `numpy >= 2.0`, `scipy >= 1.13` and `joblib >= 1.4`."

At the end of the Differences from R section, after the paragraph that starts "Nugget IDs and assignments are 0-based", append:

```markdown
`refine_dn` maps its arguments the same way:

| R | Python |
| --- | --- |
| `x` | `x` |
| `DN` | `dn` |
| `EV.tol` | `ev_tol` |
| `max.splits` | `max_splits` |
| `min.nugget.size` | `min_nugget_size` |
| `shape.split` | `shape_split` |
| `min.shape.size` | `min_shape_size` |
| `delta` | `delta` |
| `nstart` | `nstart` |
| `seed` | `seed` |
| `no.cores` | `n_jobs` |
| `make.pbs` | `progress` |

`refine_dn` reads only the assignments from `dn`. Like R, it recomputes every center, weight and scale from the rows. It also drops empty nuggets and gives single-row nuggets a scale of 0, where `create_dn` keeps empty nuggets and floors single-row scales. Both come from R. R re-seeds before every split, so each split starts from a fresh copy of the seed's stream and the result doesn't depend on `n_jobs`. With `n_jobs` other than 1 the splits of a round run in worker processes. Unlike R, `refine_dn` rejects an `nstart` or `max_splits` that isn't finite and a `shape_split` that isn't a bool.
```

In the Parity with R section, after the paragraph that ends with the link to `docs/validation/create_dn.md`, add:

```markdown
`refine_dn` is checked the same way on its own grid. faithful runs every combination of `ev_tol`, `max_splits`, `min_nugget_size`, `shape_split`, `delta` and `nstart`. iris, quakes, the petal columns of iris, a one-column dataset, the scaled volcano data and a tied grid run four settings each. More cases cover refining a refined result, an empty input nugget and splits that fail. Every kmeans call clusters the same rows as R's and returns the same clusters, and the outputs match bit for bit. Shape splits cluster the rows projected on the first eigenvector, and those projections match R's within a relative 1e-8, like the eigenvalues. The table is in [docs/validation/refine_dn.md](docs/validation/refine_dn.md).
```

In the same section, change "and the port replays that with Python integers." to "and the port replays that with Python integers. `refine_dn` needs R's `cov`, `mean`, `median`, `mad` and `quantile` too, which also add in long double."

In the Development section, change "Rebuild the parity table with:" to "Rebuild the parity tables with:".

In `r/README.md`, replace step 3 with:

```markdown
3. `Rscript r/fixtures_rcompat.R`, `Rscript r/fixtures_create_dn.R`, `Rscript r/fixtures_rstats.R`, `Rscript r/fixtures_kmeans.R` and `Rscript r/fixtures_refine_dn.R`.
```

and replace the final paragraph with:

```markdown
Arrays are raw little-endian float64 (`<f8`) or int32 (`<i4`), row-major. `manifest.json` gives each file's shape. Random draws are recorded with `trace(sample.int)` and internal calls with `trace(create.DNcenters)` and `trace(kmeans)`, so the packages run unmodified.
```

- [ ] **Step 4: Check the prose**

Run: `grep -nP "\x{2014}|\x{2013}|;" README.md r/README.md docs/validation/refine_dn.md`
Expected: no matches in the lines this task added (the humanizer voice bans dashes and semicolons in published prose).

- [ ] **Step 5: Run the gates**

Run: `UV run pytest -q && UV run ruff check . && UV run ruff format --check .`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/parity_report.py docs/validation/refine_dn.md README.md r/README.md
git commit -F <msg>   # "Report refine_dn parity with R and document refine_dn"
```

Stage `docs/validation/create_dn.md` only if Step 2 showed a content change, which would be a regression to investigate first.

---

## Execution Handoff

RON2K chose native execution with no plan-review stop ("finish it e2e while i check feature 1"). Execute with superpowers:executing-plans: `sdd-workspace` for the ledger, `task-start`/`task-done` per task, TDD on every step, then one whole-branch review by an Opus subagent with this plan's Review Focus verbatim, one fix pass (Critical and Important only, each RED then GREEN), push `feat/refine-dn` and open a PR against `main`. Do not merge.

## Self-Review

1. Spec coverage. Section 6 refine_dn items 1 to 7 map to Task 4 (`refine_dn`, `nugget_features`, `_shape_candidates`, `_split_nugget`, `_Nuggets.apply`/`finalize`). Hartigan-Wong maps to Task 3 (`kmns`, `kmeans`, fault codes and warnings). Draws.fork and the DataNuggets fixes map to Task 1. The section 5 grid (ev_tol, max_splits, min_nugget_size, shape_split, delta, nstart) is the 96-case faithful factorial. Its failure paths: kmeans error (`faithful_nstart0`, fail-closed flag), child below min_nugget_size (fail-closed flag), rejected shape split (`test_every_rejection_path_is_exercised`). The empty-cluster kmeans error can't happen inside refine (2-means on at least 4 rows always fills both clusters from distinct starts), so it's pinned on the direct `kmns` fixture `iris_dup_start`.
2. Placeholders. None. Every code step carries its code, and every run step carries its command and expected output.
3. Type consistency. `Draws.sample`/`fork` (Task 1) are used by `kmeans` (Task 3) and `run_round` (Task 4). `KMeansResult.cluster` is 0-based `intp`, `.warnings` is a tuple, and `KMeansError.warnings` matches. `_Outcome.warned` is read only in `run_round`. `nugget_features` returns `(center, weight, scale, ev1, ev2)` everywhere. `refine_inputs` returns `(x, DataNuggets, kwargs)` in tests and the report.
4. Review Focus. Each of the five lines has a test in its owning task (listed in the section).

Rulings carried into execution (ledger them at pre-flight):
- `Draws.sample(n, size)` replaces the spec's `kmeans_starts`. How many draws kmeans makes depends on duplicate detection inside kmeans, so a start-drawing helper can't know it up front.
- Eigenvalues use scipy's `lapack.dsyevr` with R's arguments, not `numpy.linalg.eigh` (spec text). It's the routine R's `eigen` calls, which keeps them as close to R's as a different LAPACK build allows. Numeric tier.
- scipy becomes a runtime dependency (spec section 11 allowed it).
- Non-finite `nstart` and `max_splits` raise. R would loop forever or silently skip every split. Inf stays allowed for `min_nugget_size`, `min_shape_size` and `delta`, as in R.
- `shape_split` must be a bool.
- A `dn` whose centers have a different column count, or whose assignments don't cover x's rows with valid ids, raises ValueError. R would read the wrong columns or fail later.
- Replayed or custom Draws with `n_jobs != 1` raise. `n_jobs != 1` uses loky processes. Warnings and DEBUG reasons are collected per split and re-emitted in the main process in candidate order.
- kmeans returns its warnings instead of emitting them. refine re-emits them as DataNuggetWarning.
- Projected kmeans inputs and eigenvalues are Numeric tier. Everything else is Exact.
- Fixture design: separate `fixtures_rstats.R` and `fixtures_kmeans.R`, a `draw_arrays` helper in `fixture_io.R`, a `refine_inputs` helper in `rfixtures.py`, a hand-built empty nugget instead of hunting for one in `create.DN` output.
- Speed is out of scope. Long double replay and the kmns port are pure Python, fine for the fixture grid but slow on very large nuggets. Benchmarks come in a later milestone.
