# datanuggets: native Python port of datanugget + WCluster

- **Status:** design approved in brainstorming (2026-09-29), awaiting written-spec review
- **Author:** Ronil Basu (Rutgers), supervised by an original author of the R packages
- **Upstream:** CRAN `datanugget` 1.5.0 (2026-08-21) and `WCluster` 1.3.0 (2025-03-17), both GPL-2
- **Method reference:** Beavers et al. (2024), *Journal of Computational and Graphical Statistics*

## 1. Goal

Ship a pip-installable Python package, `datanuggets`, that reimplements the R packages `datanugget` and `WCluster` natively (numpy/scipy, no R at runtime). Every deterministic output must match R to about 1e-8, and the match is proven by a test suite that anyone can rerun. The work is aimed at a JOSS submission (and possibly JCGS), so the validation evidence is a first-class deliverable.

**Priority:** `create_dn` and `refine_dn` are the two most important functions. They are on the critical path, get the densest fixture grid, and anchor the benchmarks. The first milestone (M1) is those two at full parity.

### Success criteria

1. All 18 exported R functions have a Python equivalent with the parity tier listed in section 2.
2. `pytest` passes on Linux, Windows and macOS for Python 3.12 to 3.14 with no R installed.
3. The validation report (function x tier x max abs/rel error) is generated from the parity suite, not written by hand.
4. Benchmarks against R on one real large dataset report wall time and peak memory.
5. A JOSS `paper.md`, docs site and `CITATION.cff` are ready for the advisor to review.

### Non-goals

- Bit-exact reproduction of R's random number generator. The same integer seed gives different nuggets in R and in Python, and the docs say so. Parity is proven by replaying R's recorded draws (section 5).
- scikit-learn estimator classes (`fit`/`transform`). They can be added later as thin wrappers.
- Porting PPbigdata or MosaiClusteR.
- Reproducing `mgcv` to 1e-8 (see section 7).

## 2. Scope: the 18 functions

| R function | Python | Module | Phase | Parity tier |
|---|---|---|---|---|
| `create.DNcenters` | `create_dn_centers` | `centers.py` | 1 | Exact (kept indices, centers) |
| **`create.DN`** | **`create_dn`** | `create.py` | 1 | Exact (assignments, weights, centers, scales) |
| **`refine.DN`** | **`refine_dn`** | `refine.py` | 1 | Exact (assignments, weights) + Numeric |
| (internal) `stats::kmeans` Hartigan-Wong, k=2 | `_hartigan_wong.py` | | 1 | Exact labels + Numeric |
| `getDN.means` | `get_dn_means` | `means.py` | 2 | Numeric |
| `AC` | `arithmetic_complexity` | `complexity.py` | 2 | Numeric |
| `wmean` | `wmean` | `wcluster/wstats.py` | 2 | Numeric |
| `wss` | `wss` | `wcluster/wstats.py` | 2 | Numeric |
| `wwcss` | `wwcss` | `wcluster/wstats.py` | 2 | Numeric |
| `distw` | `distw` | `wcluster/wstats.py` | 2 | Numeric |
| `Wpca` | `wpca` | `wcluster/wpca.py` | 2 | Numeric (sdev), Up-to-sign (rotation, scores) |
| `cluster.predict` | `cluster_predict` | `wcluster/predict.py` | 2 | Exact |
| `Wkmeans` | `wkmeans` | `wcluster/wkmeans.py` | 3 | Exact labels + Numeric |
| `Whclust` | `whclust` | `wcluster/whclust.py` | 3 | Exact merge/order + Numeric height |
| `DN.Wkmeans` | `dn_wkmeans` | `wcluster/dn.py` | 4 | Exact labels + Numeric |
| `DN.Whclust` | `dn_whclust` | `wcluster/dn.py` | 4 | Exact merge/order + Numeric height |
| `DN.Wpca` | `dn_wpca` | `wcluster/dn.py` | 4 | as `wpca` |
| `DNcluster.predict` | `dn_cluster_predict` | `wcluster/dn.py` | 4 | Exact |
| `optimal.DN` | `optimal_dn` | `optimal.py` | 4 | Distributional (section 7) |

## 3. Package shape

```
datanuggets/                      repo root (Rutgers Research\datanuggets\)
  pyproject.toml  uv.lock  README.md  LICENSE  CITATION.cff  .gitattributes  .gitignore
  src/datanuggets/
    __init__.py        public API re-exports
    _defaults.py       every R default in one place (seed 291102, R=5000, DN.num1=1e4, ...)
    _draws.py          Draws interface, NumpyDraws, ReplayDraws
    _rcompat.py        R semantics helpers: split_indices, order, which_min_colmajor, mad, quantile7, unique_first
    result.py          DataNuggets dataclass
    centers.py  create.py  refine.py  _hartigan_wong.py  optimal.py  means.py  complexity.py
    wcluster/
      __init__.py  wstats.py  wpca.py  predict.py  wkmeans.py  whclust.py  dn.py
  r/                   fixture generator (dev only)
    make_fixtures.R  cases.R  capture.R
  tests/
    fixtures/<case-id>/manifest.json + *.bin
    conftest.py  fixtures.py (loader)
    test_centers.py  test_create.py  test_refine.py  ...   (mirrors src/)
  docs/  specs/  (mkdocs site in phase 5)
  paper/  paper.md  paper.bib
  benchmarks/
```

`.gitattributes` must mark `*.bin binary` so no git line-ending setting on a collaborator's machine can alter fixture bytes.

## 4. Public API conventions

- **Naming:** R dots become underscores and names are lowercased (`create.DN` becomes `create_dn`, `DN.num1` becomes `dn_num1`, `R` becomes `r`). The docs include the full R-to-Python mapping table.
- **Indexing:** 0-based everywhere in Python (nugget ids, cluster labels, merge rows). The parity layer adds 1 when comparing to R.
- **Inputs:** anything `np.asarray` accepts, including a pandas DataFrame. Converted to C-contiguous float64. Input must be 2-D (R requires a matrix or data frame); 1-D input raises `ValueError`. Non-numeric input raises `TypeError`, matching R's `stop()`.
- **Argument checks:** every R `stop()` condition is ported as a `ValueError`/`TypeError` with the same condition, and each one has a test.
- **Algorithm parameters** keep R's defaults exactly.
- **Resource parameters** follow Python conventions: `n_jobs=1` (joblib) replaces `no.cores = detectCores() - 1`, `progress=False` (optional tqdm bar) replaces `make.pbs`, and messages go through `logging` (logger `datanuggets`) instead of `message()`/`cat()`/`print()`.
- **Randomness:** `seed: int | np.random.Generator | Draws = 291102`.

### DataNuggets result

```python
@dataclass(frozen=True)
class DataNuggets:
    centers: np.ndarray      # (k, p) float64
    weights: np.ndarray      # (k,)   int64, may contain 0 after create_dn (empty nuggets), never after refine_dn
    scales: np.ndarray       # (k,)   float64
    assignments: np.ndarray  # (n,)   int64, values in [0, k)
    def to_frame(self) -> "pd.DataFrame": ...   # R's "Data Nuggets" table layout
```

### Core signatures

```python
def create_dn(x, center_method="mean", r=5000, delete_percent=0.1, dn_num1=10_000, dn_num2=2000,
              dist_metric="euclidean", seed=291102, n_jobs=1, progress=False) -> DataNuggets

def refine_dn(x, dn, ev_tol=0.9, max_splits=5, min_nugget_size=2, shape_split=False,
              min_shape_size=10, delta=2, nstart=25, seed=291102, n_jobs=1, progress=False) -> DataNuggets
```

`whclust` returns a result carrying R's `merge`, `height`, `order` (0-based) plus `.to_linkage()` for `scipy.cluster.hierarchy.dendrogram`. `wkmeans` returns labels, centers, and WWCSS (total + per cluster). `wpca` returns `sdev`, `rotation`, `x`, `center`, `scale`, `wrow`, `wcol`.

## 5. Parity harness

### Fixture generator (`r/`, dev only)

- Pins R 4.6.x, `datanugget` 1.5.0 and `WCluster` 1.3.0 (`remotes::install_version`).
- Runs every case in `cases.R` and writes one folder per case: `manifest.json` (function, arguments, array names/shapes/dtypes, R + package + BLAS versions, dataset SHA-256) and one raw little-endian `.bin` per array via base-R `writeBin` (float64 as `double`, integers as int32).
- Raw binary is required. A 15-digit text dump rounds the input data, and in `create.DNcenters` a rounded input can flip which of two nearly equal distances is deleted first.
- **Draw capture:** `trace(sample.int, exit = ...)` records every draw R makes, in order, without modifying either package (`sample()` calls `sample.int` internally). `trace(WCluster:::init, exit = ...)` records Wkmeans' starting assignment.
- **Intermediate capture** for the priority functions (so a failure points at a stage, not just "output differs"):
  - `create.DN`: permutation `RS.loc`, chunk boundaries, per-chunk `m.each`, stage-1 centers, stage-2 kept indices, assignments, raw scales before the floor, final table.
  - `refine.DN`: the initial refreshed stats, then per split round the candidate positions, each kmeans' starting centers and cluster vector, touched positions, and EV1/EV2/scale arrays; then the final table.

### Draw replay in Python

All randomness goes through `Draws`:

```python
class Draws(Protocol):
    def permutation(self, n: int) -> np.ndarray: ...              # R: sample(n, n)
    def index(self, m: int) -> int: ...                           # R: sample.int(m, 1)
    def kmeans_starts(self, m: int, k: int, nstart: int) -> list[np.ndarray]: ...
    def wkmeans_init(self, n: int, k: int) -> np.ndarray: ...
    def fork(self) -> "Draws": ...                                 # fresh stream, R's set.seed(seed) inside a loop
```

- `NumpyDraws(seed)` is the production implementation. With an int seed, `fork()` returns a fresh `NumpyDraws(seed)`, which mirrors R's `set.seed(seed)` before every split: each split starts from the same stream. With a user `Generator`, `fork()` uses `Generator.spawn(1)`, and forks are created in the main process in candidate order before any parallel dispatch, so results still do not depend on `n_jobs`.
- `ReplayDraws(fixture)` returns R's recorded draws (converted to 0-based) and raises `ReplayMismatch` if Python requests a draw R did not make or requests them in a different order. This checks that the call sequence matches R, not just the final numbers.

### Tolerance tiers

| Tier | Applies to | Assertion |
|---|---|---|
| Exact | assignments, weights, kept indices, cluster labels, merge matrix, order; `create_dn` centers and scales | `np.array_equal` |
| Numeric | other centers and scales, WWCSS, eigenvalues, heights, PSI math | `assert_allclose(rtol=1e-8, atol=1e-10)` |
| Up-to-sign | PCA rotation and scores | flip each column to match R's sign, then Numeric |
| Distributional | `optimal_dn` propensity fits only | section 7 |

### Fixture datasets

Parity fixtures use real data:
- R built-ins: `iris` (150 x 4), `faithful` (272 x 2), `quakes` (1000 x 5)
- a pinned real subsample of UCI Covertype (20,000 x 10 continuous columns)
- the Beavers et al. datasets, if the advisor provides them

Small synthetic cases exist only to reach edge cases real data rarely produces: exact distance ties and duplicate rows (which produce empty nuggets), `p = 1`, `n <= DN.num` (the warning path), and uneven chunks. They are correctness checks, never a reported metric.

Total fixture size target: under 25 MB.

### Fixture grid for the priority functions

- **create_dn:** every dataset x `center_method` {mean, random, original} x `dist_metric` {euclidean, manhattan} x `delete_percent` {0.1, 0.3}, with `r` and `dn_num1`/`dn_num2` chosen so that single-chunk, multi-chunk, uneven-chunk, and `n <= dn_num` paths are each covered at least once.
- **refine_dn:** run on create_dn outputs, crossing `ev_tol` {0.5, 0.9}, `max_splits` {0, 1, 5}, `min_nugget_size` {2, 5}, `shape_split` {False, True}, `delta` {1.5, 2}, `nstart` {1, 25}. Cases that trip the split failure paths (child below `min_nugget_size`, kmeans empty-cluster error, rejected shape split) must each appear.

## 6. Behavior spec for the priority functions

These notes come from reading the R 1.5.0 source. Each item is a parity requirement. Items marked (verify) must be confirmed against fixtures before relying on them.

### create_dn

1. Validate arguments exactly as R does (`r` in [100, 10000], `delete_percent` in (0, 1), `dn_num2 <= dn_num1`, metric in {euclidean, manhattan}, center method in {mean, random, original}, `n >= 2`).
2. One permutation `RS.loc = sample(n, n)`. This is the only draw unless `center_method == "random"`.
3. Chunking uses R's `parallel::splitIndices(n, ceiling(n / r))`, which cuts by fuzzed breakpoints. `_rcompat.split_indices` reproduces it exactly, and fixtures check the chunk boundaries.
4. `m.each`: `dn_num1 // chunks` per chunk, remainder added to the first chunks, clamped to [1, chunk length].
5. Stage 1: `create_dn_centers` on each permuted chunk, results stacked in chunk order. Stage 2: `create_dn_centers` on the stacked centers down to `dn_num2`.
6. `create_dn_centers`: full pairwise distance matrix (Rfast `Dist`, whose per-feature sums use Armadillo's even/odd two-accumulator order; `_rfast.dist_matrix` reproduces it bit for bit), diagonal set to `max + 1`. Loop while more than `DN.num` rows remain: `delete.num = min(max(1, floor(tmp.num * delete_percent)), tmp.num - DN.num)`; row minimum value and index (Rfast `rowMins`, first index on ties, pinned by the `rcompat_rowmins` fixture); visit rows in `order(nn.dist)` (stable sort, so `np.argsort(kind="stable")`); mark a row if neither it nor its nearest neighbor is marked; if not enough are marked, a second pass marks any unmarked rows in the same order; drop marked rows and columns. If `n <= DN.num` at the start, warn and return the input unchanged.
7. Assignment: nearest final center for every observation via `Rfast::dista(k = 1, index = TRUE)`. dista's per-feature sums round differently from `Dist`'s at more than 32 features: its Euclidean kernel is a sum of squares that Armadillo's `op_dot::direct_dot` sums with its own even/odd loop up to 32 terms and hands to BLAS `ddot` above that, and R's reference BLAS adds left to right. Manhattan uses even/odd at every `p`. `_rfast.cross_distances(kernel="dista")` reproduces both, pinned by the `perm32`/`perm33`/`perm40` fixtures (coordinate permutations of one vector, which are mathematically equidistant from constant rows, so only the summation order decides the winner). An R linked to an optimized BLAS (OpenBLAS, MKL, Accelerate) sums in its own order, so exact parity above 32 features holds against the reference BLAS that every fixture manifest records. Rfast sorts the whole distance row with libstdc++ `std::sort` (not stable), so among exactly tied distances the winner is whichever index `std::sort` leaves first: the first index when there are at most 16 centers, otherwise an index decided by introsort's partitions. `_rfast.nearest_index` uses argmin when the minimum is unique and replays `std::sort` (`gnu_sort_first`, O(k)) only for tied rows. Verified against R's complete sort order in `rcompat_dista`.
8. Weights: counts per nugget. Empty nuggets are kept with weight 0.
9. Scales: computed on data centered by the global column mean (`colMeans`), with R's one-pass formula (`ss = S2 - n * mean^2`, negatives clamped to 0, divided by `max(n - 1, 1)`, averaged over columns with `rowMeans`). The port uses the same formula, not `np.var`, but the formula alone does not make round-off match: `colMeans` and `rowMeans` sum in long double (x87 extended precision, 64-bit significand, on x86-64) and divide before rounding to double. `_rcompat.r_mean` replays that arithmetic exactly with Python integers. A correctly rounded mean is not enough, because on columns that cancel to nearly zero (standardized data) R's own running sum rounds. `rowsum` is a sequential double sum in row order (`np.add.at`). Pinned by `rcompat_means`, which includes `scale(volcano)`. With all three, centers and scales match R bit for bit on every fixture. Nuggets with weight < 2 get the floor `min(scale of multi-obs nuggets) / 100`, or machine epsilon if there are none.
10. Centers: `mean` gives the per-nugget mean (empty nuggets fall back to the stage-2 center); `original` gives the stage-2 centers; `random` draws one member per non-empty nugget in nugget order (`sample.int(len, 1)`), falling back to the stage-2 center for empty nuggets.

### refine_dn

1. Validate as R does (`ev_tol` in (0, 1), `max_splits >= 0`, `min_nugget_size >= 2`, `min_shape_size >= 10`, `delta > 1`, assignments length = n).
2. Recompute every nugget's stats from its members first. Centers become member means even if create_dn used `original` or `random`. Scale = trace(sample cov) / p. EV1, EV2 = two largest eigenvalues (`eigh`, reversed to descending; for `p = 1`, EV2 = EV1). Empty nugget: weight 0, NaN center. Singleton: scale 0, EV 0.
3. **EV splits** (if `max_splits > 0`): the threshold is `quantile(EV1[finite & > 0], ev_tol)` with R's type-7 quantile, computed once before the loop. Each round, candidates = positions with EV1 > threshold and weight >= 2 x `min_nugget_size`, in ascending position order. Stop when there are no candidates or no split succeeds.
4. **Split rule:** skip if fewer than 2 x `min_nugget_size` rows. The random stream is re-seeded before every kmeans call (R calls `set.seed(seed)` inside the split), so `Draws.fork()` is used per split. Results are therefore identical for any `n_jobs`. Run Hartigan-Wong 2-means with `iter.max = 1000` and `nstart`. A kmeans error (e.g. an empty cluster) means no split. A child smaller than `min_nugget_size` means no split.
5. **Applying splits:** rows labeled 2 move to a new id (`next_id`, incrementing, in candidate order); stats are refreshed for the parent and child only.
6. **Shape splits** (if `shape_split`, `p >= 2`, `max_splits > 0`): each round, `s.threshold = median(scale) + 2 * mad(scale)` over nuggets with finite scale and weight > 1, using R's `mad` constant 1.4826 exactly (not scipy's 1.482602...). Candidates: weight >= max(`min_shape_size`, 2 x `min_nugget_size`), EV1 > 0, and (EV1/EV2 > `delta` or EV1 > `s.threshold`). The comparison of EV1 against a scale-based threshold is R's behavior and is kept as-is. Project the members onto the first eigenvector, run the split on the 1-D projection, and accept only if both children have >= 2 rows and the mean of their EV1/EV2 ratios is < `delta`.
7. **Finalize:** drop weight-0 nuggets (keeping position order), renumber ids and assignments to 0..k-1, set scale 0 where weight < 2, then assert R's invariants (weights sum to n, no missing assignments, all weights > 0, ids contiguous).

### Hartigan-Wong 2-means

- Port R's `kmns.f` (Hartigan and Wong 1979, AS 136, as modified in R) for general k but only exercised at k = 2. R is GPL-2+, so this is license-compatible.
- Starting centers follow R's `kmeans` rule (verify): with `nstart >= 2` they are drawn from `unique(x)` in first-occurrence order (`_rcompat.unique_first`, not `np.unique`, which sorts). The best start is kept by strict `<` on total within-SS.
- Error and warning conditions (empty cluster, Quick-TRANSfer step limit, non-convergence) map to the same outcomes R's caller sees.

## 7. The exception: optimal_dn's propensity model

R fits `mgcv::bam(DN_flag ~ s(x1) + ... + s(xp), family = binomial, discrete = TRUE, weights = w)` on nugget centers (weight = nugget weight, flag 1) stacked on the raw data (weight 1, flag 0). No Python library reproduces mgcv's thin-plate spline + fREML fit to 1e-8.

- `optimal_dn(..., propensity_model=None)` accepts any callable `(X, y, w) -> p_hat`. The default is pyGAM `LogisticGAM` with one `s()` term per column and 10 basis functions (mgcv's default `k` for `s()`), with smoothing chosen by pyGAM's grid search.
- **Parity for the fit:** same `opt.dn.no` as R on every fixture, and Spearman correlation of the PSI curve >= 0.95.
- **Parity for everything around the fit:** fixtures save mgcv's fitted propensities, and a test feeds those into the PSI index, early-stop rule, second-difference `deltas` and `which.max` selection. That part is held to Exact/Numeric.
- The elbow and difference plots are matplotlib (optional extra `datanuggets[plot]`); the function returns the data frames, and plots are built only when requested.

## 8. Build order and milestones

| Phase | Work | Exit gate |
|---|---|---|
| 0. Harness | `winget install RProject.R`; pin packages; scaffold (`uv`, `pyproject.toml`, `.gitattributes`, CI matrix); `r/make_fixtures.R` with `create.DN`/`create.DNcenters`/`refine.DN` cases **first**, then the rest; fixture loader; `Draws`/`ReplayDraws`; `_rcompat` helpers with their own tests | priority fixtures committed; CI green |
| 1. **Critical path (M1)** | `create_dn_centers` then `create_dn` (all center methods, both metrics); `_hartigan_wong` then `refine_dn` (EV splits, then shape splits) | full parity on the section 5 grid; short parity note sent to the advisor |
| 2. Leaves | `wmean`, `wss`, `wwcss`, `distw`, `get_dn_means`, `arithmetic_complexity`, `wpca`, `cluster_predict` | listed tiers pass |
| 3. Iterative | `wkmeans`, `whclust` | exact labels and merge order |
| 4. Wrappers | `dn_*` wrappers, `optimal_dn` | wrappers exact; optimal_dn per section 7 |
| 5. Paper | docs site (mkdocs-material + mkdocstrings) + mapping table; generated validation report; benchmarks; `paper.md`; `CITATION.cff` | draft to advisor |

Within phases 2 to 4 the functions are independent and can be ported in parallel.

Implementation plans are written per milestone so each stays reviewable: plan 1 covers phases 0 and 1 (M1: harness + create_dn + refine_dn), plan 2 covers phases 2 to 4, plan 3 covers phase 5.

## 9. Performance

The first version copies R's algorithms, including their memory cost. Optimizations land only after parity is green, and each one must keep every fixture passing.

- `create_dn_centers`: stage 2 builds a `dn_num1` x `dn_num1` float64 matrix (800 MB at 1e4), as R does. Later optimization: track remaining rows with a mask instead of copying the submatrix every pass, and vectorize the row minimum over the active set. The greedy marking loop is inherently sequential and is fine in Python at these sizes; numba is an option only if profiling shows it matters.
- `create_dn`'s global column mean replays R's long-double `colMeans` in pure Python, about 0.4 s per million values (about 4 s at n = 1e6, p = 10). A later fast path can use `np.add.accumulate` over `np.longdouble` where numpy's long double is x87 extended (Linux x86-64), keeping the integer replay as the reference everywhere else.
- `create_dn` stage 1 chunks run under joblib (loky) when `n_jobs > 1`. Chunk results are stacked in chunk order, so output does not depend on `n_jobs`.
- `refine_dn` splits within a round run under joblib; per-split reseeding keeps results identical for any `n_jobs`.
- `whclust`: R recomputes WSS from raw rows for every pair, O(n^3) with a large constant. The closed-form weighted Ward cost `W_A W_B / (W_A + W_B) * ||mu_A - mu_B||^2` is algebraically equal to R's `wss(A+B) - wss(A) - wss(B)` and ships only if merge order matches R on every fixture. Ties are broken column-major, like R's `which.min` on a matrix.
- **Benchmarks:** `create_dn` and `refine_dn` first, on real UCI Covertype at n in {1e4, 1e5, 581k}, R vs Python, single-core and all-core, reporting wall time and peak RSS.

## 10. R behaviors kept, fixed, or documented

| Item | R behavior | Python |
|---|---|---|
| `wmean`/`wwcss`/`distw` default `w = rep(1, length(x))` | wrong length (columns, not rows), so the default always errors | default `np.ones(n)`; listed as a deviation that fixes an upstream bug |
| create vs refine scale for single-obs nuggets | create floors to min/100, refine sets 0 | kept as-is, documented |
| create keeps empty nuggets, refine drops them | | kept as-is, documented |
| `Whclust` height | cumulative total WWCSS, not merge cost | kept (it is the R result); `.to_linkage()` notes it |
| Wkmeans distance weighting | `w_i * d_i` does not change the argmin | kept for parity, documented |
| Shape-split threshold compares EV1 with a scale-based cutoff | | kept, raised with advisor |
| Printing (`print(wwcss)`, `cat`) | | `logging` at INFO/DEBUG |

## 11. Repo, tooling, license

- Location `OneDrive\Desktop\Rutgers Research\datanuggets\`; private GitHub repo `ron2k1/datanuggets` until the advisor approves making it public.
- `uv` for environments and builds (hatchling backend); `ruff` for lint and format; `pytest` with `pytest-xdist`; GitHub Actions matrix (ubuntu, windows, macos) x (3.12, 3.13, 3.14). No R in CI.
- Runtime deps: numpy >= 2.0, scipy >= 1.13, joblib, pygam. Extras: `plot` (matplotlib), `frame` (pandas), `progress` (tqdm).
- License: GPL-2.0-only, same as upstream, with credit to the original authors in `LICENSE`, `README`, and `CITATION.cff`. Relicensing (e.g. BSD-3) is possible only with the authors' consent and is an advisor question.

## 12. Open questions for the advisor

1. License: keep GPL-2 or relicense with the authors' consent?
2. Can the Beavers et al. (2024) datasets be shared for fixtures and benchmarks?
3. Is comparing EV1 against the scale-based shape threshold intentional?
4. Authorship and venue: JOSS first, JCGS software note, or both?
5. Is the `wmean`/`wwcss`/`distw` default-weight bug worth an upstream fix on CRAN?

## 13. Risks

| Risk | Mitigation |
|---|---|
| Rfast tie-breaking or distance round-off differs from numpy | intermediates captured per stage; `_rcompat` isolates each R semantic with its own test |
| R's kmeans start rule differs from the (verify) notes | the replay fixtures record the actual draws; Hartigan-Wong tested against `stats::kmeans` directly on dedicated cases |
| mgcv vs pyGAM chooses a different optimal nugget count | pluggable model; decision parity measured and reported honestly; mgcv values still exercise the downstream math |
| 800 MB distance matrix on laptops | documented; mask-based optimization after parity |
| OneDrive syncing the repo | `.gitattributes`; if sync corrupts anything, move the repo to a non-synced path and leave a pointer |
