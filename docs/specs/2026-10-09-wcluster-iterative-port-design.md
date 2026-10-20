# WCluster iterative port: wkmeans, whclust and the DN wrappers

- **Status:** design approved in brainstorming (2026-10-09), awaiting written-spec review
- **Author:** Ronil Basu (Rutgers)
- **Upstream:** CRAN `WCluster` 1.3.0 and `datanugget` 1.5.0. `datanugget`'s `DN.Wkmeans` and `DN.Whclust` are the same code as `WCluster`'s and call `WCluster::Wkmeans` and `WCluster::Whclust`.
- **Parent spec:** [the port design](2026-09-29-datanuggets-python-port-design.md). This spec replaces its sketch of these six functions (sections 2, 4, 5 and 9.4, phases 3 and 4). `optimal.DN` gets a spec of its own, because its `mgcv::bam` fit can't be matched exactly.

## 1. Goal

Port `Wkmeans`, `Whclust`, `DN.Wkmeans`, `DN.Whclust`, `DN.Wpca` and `DNcluster.predict` as standalone Python and numba code. Nothing calls R at runtime: R is used only on the laptop, by the scripts in `r/`, to record answers the tests compare against.

The output must be R 4.6.1's output, and the code must be faster than R by the gates in section 10.

### Success criteria

1. On every fixture, labels, `merge` and `order` are equal to R's, and every float output (heights, WWCSS, centers) is bit-identical to R's. `dn_wpca` keeps `wpca`'s eigen tolerance.
2. Every seeded `Wkmeans` fixture replays R's draws through `ReplayDraws` and consumes exactly R's draws in R's order.
3. The compiled kernels and the plain-Python reference agree bit for bit on the fuzz, including on every quirk path listed in section 9.
4. The suite passes on Python 3.14 and 3.15, on the laptop and on the Mac mini, with no R installed.
5. Every speed gate in section 10 passes in one pinned sitting.

### Non-goals

- `optimal.DN` (next spec).
- The closed-form Ward update. It rounds differently from R's subtraction of sums of squares and changes 6 of iris's 149 merges (parent spec 9.4).
- R's metadata fields on the `hclust` object (`labels`, `method`, `call`, `dist.method`), which are constants.
- Threads inside `whclust`. Its steps are sequential, and section 10's targets don't need them.

## 2. Scope

| R function | Python | Module | Parity |
|---|---|---|---|
| `Wkmeans` | `wkmeans` | `wcluster/wkmeans.py` | exact labels, bit-identical centers and WWCSS |
| `Whclust` | `whclust` | `wcluster/whclust.py` | exact `merge` and `order`, bit-identical `height` |
| `DN.Wkmeans` | `dn_wkmeans` | `wcluster/dn.py` | as `wkmeans`, plus exact row labels |
| `DN.Whclust` | `dn_whclust` | `wcluster/dn.py` | as `whclust` |
| `DN.Wpca` | `dn_wpca` | `wcluster/dn.py` | as `wpca` |
| `DNcluster.predict` | `dn_cluster_predict` | `wcluster/dn.py` | exact |

The kernels live in `wcluster/wkmeansjit.py` and `wcluster/whclustjit.py`, since the repo keeps numba kernels only in modules named `*jit`.

## 3. Public API

```python
def wkmeans(x, k, *, w=None, centers=None, num_init=1, max_iter=10, seed=291102,
            n_jobs=1) -> WkmeansResult
def whclust(x, w=None) -> WhclustResult

def dn_wkmeans(dn, k, *, centers=None, num_init=1, max_iter=10, seed=291102,
               n_jobs=1) -> DnWkmeansResult
def dn_whclust(dn) -> WhclustResult
def dn_wpca(dn, wcol=None, corr=False) -> WpcaResult
def dn_cluster_predict(dn, cl, newx) -> np.ndarray
```

- `wkmeans` takes R's `Wkmeans(dataset, k, cl.centers, obs.weights, num.init, max.iterations, seed)` arguments with R's defaults. Everything after `k` is keyword-only, because R's positional order puts `cl.centers` before the weights and a call written in R's order has to fail, as `cluster_predict` already does for its `w`.
- `centers` is (k, p), one row per cluster. R takes p by k. The wrapper transposes.
- `n_jobs` follows `check_n_jobs` (a positive int, or -1 for every core). It only spreads the `num_init` refinements over threads (section 5.3). The result doesn't depend on it.
- `seed` follows the package rule: an int or a numpy `Generator` gives `NumpyDraws`, and `RDraws(seed)` reproduces R's `set.seed(seed)` draws exactly. When breaking change B1 lands, every function switches together.
- The `dn_*` functions take a `DataNuggets` and read it the way R reads the datanugget table: the centers are the data and the weights are the row weights.
  - `dn_wkmeans` adds each original row's cluster as `labels[dn.assignments]`, R's `sapply` lookup.
  - `dn_wpca` calls `wpca(dn.centers, wrow=dn.weights, wcol=wcol, corr=corr)`, with `wcol` defaulting to ones.
  - `dn_cluster_predict` calls `cluster_predict(dn.centers, cl, newx, w=dn.weights)`. `newx` is a matrix, a data frame or another `DataNuggets`, whose centers are used.
- R's printed WWCSS per init and its `message("Iteration l")` go to `logging.debug` on the `datanuggets` logger.

### Result types

In `result.py`, as frozen `_ArrayFields` dataclasses like the others. `_ArrayFields` stores every field as an array, so the WWCSS pair is two fields, not a nested `WwcssResult`.

| Type | Field | Shape | Meaning |
|---|---|---|---|
| `WkmeansResult` | `labels` | (n,) intp | R's label minus one, gaps included |
| | `centers` | (k', p) | one row per present label, in label order |
| | `wwcss` | (k',) | each present cluster's WWCSS, in label order |
| | `total_wwcss` | 0-d | R's `TotalWWCSS` |
| `DnWkmeansResult` | the four above, plus `row_labels` | (n_rows,) intp | each original row's cluster |
| `WhclustResult` | `merge` | (n-1, 2) intp | scipy numbering (below) |
| | `height` | (n-1,) | R's cumulative total WWCSS after each merge |
| | `order` | (n,) intp | 0-based rows in dendrogram order |

k' is k except when a cluster has emptied, which R allows (section 4.2), and then labels have a gap and `centers` and `wwcss` have fewer rows, exactly as in R.

`merge` renumbers R's encoding one to one: R's singleton `-r` becomes `r - 1`, and R's cluster `s` (made at step s) becomes `n + s - 1`. Each row keeps R's column order. `WhclustResult.to_linkage()` returns scipy's (n-1, 4) linkage, `merge` plus `height` plus each cluster's size, for `scipy.cluster.hierarchy.dendrogram`. It raises when a row merges a cluster with itself (section 4.4), since scipy can't represent that.

## 4. The R algorithms, as ported

These are read from WCluster 1.3.0's source and are what both Python implementations follow. Labels are R's 1-based labels here.

### 4.1 Shared pieces

- **`wss(x, w)`:** `mean = r_matvec(x.T, w) / r_sum(w)`, then `dev = x - mean`, then `r_sum(r_matvec((dev * dev).T, w))`. This is the existing `_wss`. It depends only on the cluster's rows in their original order and their weights.
- **`wwcss(x, cl, w)`:** `wss` of each present label's rows, in sorted label order (`factor(cl)`), and the total is `r_sum` of those values in that order.
- **`wmean(x, cl, w)`:** `r_weighted_mean` per column for each present label, in sorted label order. This is the existing `wmean`.
- **`wdist_mat(x, centers, w)`:** `D = (m11 + m22) - 2 * m12`, with `m11 = r_matvec(x * x, ones)`, `m22` the column sums of `centers * centers` in order, and `m12 = r_matmul(x, centers.T)`. R then computes `diag(w) %*% D`, which with reference BLAS gives entry (i, j) as `w[i] * D[i, j]`, except that it's NaN when another row of column j of D is NaN or infinite, or when `w[i] * D[i, j]` is NaN. The existing `_wdist_mat` in `predict.py` does this for unit weights and is generalized to `w`.
- **`which.min`:** the first minimum, skipping NaN. On a matrix, it scans column-major.

### 4.2 Wkmeans

`Wkmeans(x, k, cl.centers, w, num.init, max.iterations, seed)`:

1. Check the arguments, then `set.seed(seed)`. Floor `k`, `num.init` and `max.iterations`. With user centers, `num.init` becomes 1.
2. For each init in order: run `Wkmeans_single`, log its WWCSS, and keep it if it's the first or if the best so far is strictly greater (`>`).
3. Return the best labels as they are, `t(wmean(x, best, w))` as the centers, and `wwcss(x, best, w, groupSum = TRUE)`.

`init(x, w, k, u)` repeats until the assignment has k distinct labels:

1. Without user centers, draw `y = sample(c(1:k, sample(k, n - k, replace = TRUE)))`. That's two draws in this order: `sample.int(k, n - k, replace = TRUE)`, then the permutation `sample.int(n, n)`. Redraw while some label appears fewer than twice. The centers are `wmean(x, y, w)`. With user centers, they're used as given.
2. `zz` is the row-wise `which.min` of `wdist_mat(x, centers, w)`.
3. With user centers and fewer than k distinct labels, R stops.

`Wkmeans_single(x, u, w, k, K)`:

1. `y = init(...)`, then `k = max(y)` and `tb = table(y)`, the count of each present label in label order.
2. For each outer iteration `l` in 1 to K:
   1. `uu0 = uu00 = wwcss(yy)`.
   2. Run 20 Lloyd steps. Each sets `yy0 = yy`, computes the centers `wmean(yy0)` (one per present label), sets `yy` to the row-wise `which.min` of `wdist_mat`, which renumbers the labels to column positions, and computes `uu0 = wwcss(yy)`. If `uu0 > uu00`, `yy` goes back to `yy0`, else `uu00 = uu0`.
   3. `uu0 = uu00 = wwcss(yy)`.
   4. The move loop. For each point i and each label j in 1 to k: if `yy[i] != j` and `tb[yy[i]] > 1`, try `yy[i] = j`. If the new `wwcss` is `<= uu0`, keep it, set `uu0` to it and refresh `tb = table(yy)`.
   5. Stop if `yy0 == yy` everywhere, where `yy0` is the state at the start of Lloyd step 20.
3. Return `uu0` and `yy`.

The quirks both Python implementations keep:

- `tb` is computed from `init` and refreshed only when a move is kept, so Lloyd steps leave it stale.
- `tb[yy[i]]` indexes the table by position. Once a label is missing, positions and labels disagree, and a position past the end gives NA. `yy[i] != j & NA` is NA when `yy[i] != j`, and R stops.
- `k` stays `max(init's labels)`, so a move can recreate a label that a Lloyd step left empty, and labels can have gaps.
- A move that leaves WWCSS unchanged is kept (`<=`).
- A NaN WWCSS in any of these `if` tests makes R stop.

### 4.3 Whclust

`Whclust(x, w)`:

1. `y = -(1:n)`, so every row starts as its own cluster, and `y0 = sort(unique(y))` lists the clusters in position order. `dd = distw(x, y, w)`: below the diagonal, entry (i, j) is `distwij` of the clusters at positions i and j. The diagonal and everything above it are NA.
2. For `ii` in 1 to n - 1:
   1. `k = which.min(dd)`. Decode the row `k0 = k %% m` (m when that's 0) and the column `k1 = k %/% m + (k0 != 0)`, where m is the current number of clusters.
   2. `u = y0[k0]`, `v = y0[k1]`, and the merge row is `c(u, v)`. Both clusters are relabeled `ii`, and `y0` is sorted again, so the new cluster is last.
   3. Unless this is the last step, drop rows and columns `k0` and `k1`, then `cbind(rbind(dd, 0), Inf)`: a new last row and a new last column of Inf. Fill the new row's entries below the diagonal with `distwij(y0[t], y0[m], ...)`. So a merged cluster's column holds Inf above and on its diagonal, while a singleton's column holds NA there.
   4. The height is `wwcss(x, y, w)`, the total over all clusters.
3. `order = forder(merge)`: the concatenated members of the last merge.

`distwij(i, j)` is the pooled `wss` of both clusters' rows, in original order, minus `r_sum` of the two clusters' own `wss` values, lower label first.

`which.min` picks one of the Inf cells only when no entry below the diagonal is finite. Two ways that happens:

- A row with weight zero has a NaN `wss` alone, so all of its distances are NaN. Every other row merges first, and the last pick is an Inf cell above the diagonal: the merge row comes out with the lower position first.
- When no singletons are left and the merged cluster at position 1 has its diagonal Inf as the first candidate, R merges that cluster with itself (`k0 == k1`), and its later steps work on a matrix of the wrong size. R returns this without stopping.

R stops below 3 rows: `distw` subscripts out of bounds at 1 row, and `forder` at 2.

## 5. The fast path

### 5.1 The invariant

A cluster's `wss` depends only on its rows in original order and their weights. Every value R recomputes is a value of `wss` on the same rows in the same order, so computing it once and reusing it gives the same bits. Totals are `r_sum` of the cached values in sorted label order, using `_compat/ldjit.py`'s compiled long double, as R adds them.

### 5.2 whclust

- **Distances:** an n by n float64 array of the values R keeps (32 MB at 2,000 rows, the same as R's matrix). A merged cluster reuses one of its two parents' slots. A creation counter gives each cluster its position rank, so position order is the counter order without moving data.
- **Own wss:** cached per cluster when it forms. Pooled `wss` is computed only for pairs involving the new cluster, on the merged rows in original order, by merging the two clusters' sorted row lists.
- **Height:** `r_sum` of the cached own values in position order, which is sorted label order.
- **Picking the pair:** each column keeps its minimum below the diagonal and the first row holding it. Rows and columns are visited in position order, and only a strictly smaller value replaces the minimum.
  - The new cluster is the last row of every surviving column, so it takes over a column only by being strictly smaller.
  - A column whose minimum was in a deleted row is scanned again.
  - The step's pick is the first column, in position order, whose minimum is the smallest. That's R's column-major first minimum, ties included.
  - NaN never wins, as `which.min` skips it.
- **The Inf cells:** when no entry below the diagonal is finite, the kernel applies R's rule directly. It scans the columns in position order. A merged column's first candidate is its row 1 (an Inf above the diagonal), or its own diagonal when it's at position 1. A singleton column's candidate is its first entry equal to +Inf. The first candidate wins. If there's none, R's `which.min` is empty and R stops.
- **Self-merge:** if the winning cell is a diagonal, the kernel returns a status, and `whclust` reruns the whole call through the reference (section 5.4), which follows R's matrix operations literally. This needs every remaining distance NaN and no singletons left, so the slow path is only for weights that cancel.

### 5.3 wkmeans

- **Draws first:** every random draw happens inside `init`, before any refinement. So all `num_init` starting assignments are drawn in order on the main thread. The refinements then run as nogil kernels, on threads when `n_jobs` allows, and the best is picked in init order with R's strict `>`.
- **Lloyd steps:** `wmean` and `wdist_mat` are O(n·k·p). R's `diag(w) %*% D` is O(n²·k), and `w[i] * D[i, j]` with the NaN rule gives the same values.
- **Lloyd shortcut:** a Lloyd step's result depends only on the assignment and `uu00`. If a step reverts, or produces the assignment it started from, every remaining step repeats it, so the kernel stops and sets `yy0` to that assignment, which is what R holds at the start of step 20.
- **Move loop:**
  - Each label's `wss` and its sorted row list are cached.
  - A trial move of point i from label a to label j recomputes only `a` without i and `j` with i, each on its rows in original order.
  - `a` without i is computed once per point and kept until a move is accepted.
  - The trial total is `r_sum` of the cached values for the present labels, in label order, with those two replaced. A label left empty drops out, as `factor` drops it.
  - `tb` is kept as R keeps it: an array of counts in label order, refreshed only when a move is accepted, and indexed by position with NA past its end.
- **Cost:** a trial costs O((|a| + |j|)·p) where R's costs O(n·p) plus interpreter overhead. There's no exact shortcut below that, because adding a point moves the mean and every deviation with it.

### 5.4 The reference implementations

`_wkmeans_py` and `_whclust_py` are plain Python transliterations of section 4, step for step, built on the existing exact primitives (`_wss`, `wwcss`, `wmean`, `r_sum`, `r_matvec`, `r_matmul`), with no caching. They sit next to the kernels, following `_select_centers_py`.

- The tests compare the kernels against them bit for bit.
- `whclust` falls back to `_whclust_py` for the self-merge case.
- With `count=True`, each reference returns how often each quirk path ran, which the fuzz uses for its coverage check (section 9).

## 6. Draws

`init` makes two kinds of draw. The capture today records `n`, `size` and the values, not `replace`, so a replace draw would replay as a no-replace one with the same `n` and `size`.

- **Protocol:** `Draws` gains `sample_replace(n, size)`, R's `sample.int(n, size, replace = TRUE)`, 0-based.
  - `RDraws` returns one `_unif_index(n)` per element, R's `do_sample` replace branch.
  - `NumpyDraws` returns `rng.integers(n, size=size)`.
  - `ReplayDraws` requires the recorded draw to be a replace draw with the same `n` and `size`, and raises `ReplayMismatch` otherwise. Its existing methods require a no-replace record.
- **Recording:** `DrawRecord` gains `replace: bool = False`. `capture_draws` in `r/fixture_io.R` also records `replace`, written as a new `draw_replace` array, and its fail-closed check covers a replace call. `tests/rfixtures.py` reads a fixture without `draw_replace` as all no-replace, which is what those calls were, so no existing fixture is regenerated.
- **Seeding:** `wkmeans` calls `as_draws(seed)` once at the top, which forks an `RDraws` from its seed as R's `set.seed(seed)` does. With user centers, no draw is made.

## 7. Errors

Four kinds of R behavior, each with its rule.

**R stops:** raise at the same condition with R's message, `ValueError` for values and `TypeError` for types:

- argument checks: `k` not a number, a wrong `centers` shape, a weights length that doesn't match, `dn` not a `DataNuggets` (R: "datanugget must be of class \"datanugget\""), and `cl` not one label per nugget
- user centers that leave a cluster empty in `init`
- a NaN WWCSS in a Lloyd test, a move test or the strict `>` between inits (R: "missing value where TRUE/FALSE needed")
- the stale `tb` lookup past the end of the table
- a row of `wdist_mat` with every entry NaN
- `whclust` with fewer than 3 rows, or with no candidate at all at a step

**R hangs:** raise, since R never returns.

- `n < 2k` without user centers: `init` needs every label at least twice.
- Fewer than k distinct (row, weight) pairs without user centers. Identical pairs give identical rows of `wdist_mat`, so they always get the same label, and `init` can never reach k labels.

Otherwise `init` redraws exactly as R does.

**R returns something odd:** match it. This covers label gaps, the self-merge, and height as cumulative WWCSS.

**R's `1:0` loops:** `num_init` or `max_iter` below 1 makes R's `for (U in 1:0)` run twice. Raise `ValueError` instead, and say so in the docstring.

- `k`, `num_init` and `max_iter` are floored like R, following the kmeans port's `n25.7` fixture, and `k` below 1 raises.
- x and w go through `as_matrix` and `as_weights`, so NaN or Inf input is rejected up front, as everywhere in the package.
- Kernels report an error as a status code, and the wrapper raises the message. The reference raises the same error at the same point.

## 8. Fixtures

Recorded once on the laptop's x87 R 4.6.1 with reference BLAS by `r/fixtures_wkmeans.R`, `r/fixtures_whclust.R` and `r/fixtures_dn_wcluster.R`, into `tests/fixtures/{wkmeans,whclust,dn_wcluster}/`. Fixtures are never regenerated to make a test pass.

- **Intermediate capture for Wkmeans:** the script traces `WCluster:::init` (each starting assignment) and `WCluster:::Wkmeans_single` (each init's final assignment and WWCSS) in the namespace, so a failure names its stage.
- **Quirk proof:** a case named for a quirk must hit it. The R script holds an instrumented copy of the R function (the same code plus counters) and uses it only to show the case hits its quirk, stopping when the count is zero. It checks that the copy's output equals the package's, then records the package's own output. To find such inputs, it searches seeds and small tie-heavy subsets in R. If the search finds no input for a case within its budget, the case is dropped from the table below and the fuzz's coverage check (section 9) carries that quirk alone. A case is never recorded without its proof.

| Function | Cases |
|---|---|
| `Whclust` | iris unit weights (a duplicate row, so zero distances and ties) • iris with weights 1, 3, 7, 2 repeating • quakes rows 1 to 200 with the same weights • a duplicated integer grid where many distances tie • one zero-weight row and two zero-weight rows (NaN distances, the Inf pick) • ±1 weights that make R merge a cluster with itself • 3 rows • one column |
| `Wkmeans` | iris k = 3 • iris k = 3, `num.init = 5` • quakes rows 1 to 500, k = 4, weighted • user centers • user centers that leave a cluster empty (R stops) • small n with k = 4 (forces `init` redraws) • k = 1 • `max.iterations = 1` • one column • a Lloyd revert • a stale `tb` that disagrees with the labels • a cluster emptied and recreated (a label gap) • a stale `tb` past the end (R stops) • a zero-weight cluster (R stops on NaN) |
| DN wrappers | one R `create.DN` on quakes with about 150 nuggets, so R's Whclust stays fast, then `DN.Wkmeans`, `DN.Whclust`, `DN.Wpca`, and `DNcluster.predict` with a data frame and with a second datanugget as `newx` |
| `sample.int` | `fixtures_rsample.R` gains `sample.int(k, size, replace = TRUE)` cases with `.Random.seed` before and after |

The Python DN tests build `DataNuggets` from R's recorded table, so they don't depend on `create_dn`.

## 9. Testing

`tests/wcluster/test_wkmeans.py`, `test_whclust.py` and `test_dn.py`, plus additions to `tests/randomness/`.

1. **R parity:** the kernel path and the reference each run every fixture. Labels, `merge` and `order` must be equal, floats bit-identical, and errors must match where R stopped. `dn_wpca` uses `wpca`'s tolerance.
2. **Draw order:** every seeded `Wkmeans` fixture runs on `ReplayDraws` and ends with `assert_exhausted()`. The traced starting assignments and per-init WWCSS must match.
3. **Kernel against reference fuzz:** seeded small cases on tie-heavy data (integer grids, duplicate rows, integer weights including zero and negative values). Both must give the same bits, or the same error at the same point. The test fails unless the reference's counters show every quirk path ran: Lloyd revert, Lloyd shortcut, stale `tb` disagreeing with labels, a label emptied and recreated, the Inf pick and the self-merge. The default suite runs a fast slice, and a larger run goes under the existing `slow` marker.
4. **Errors:** a test for each case in section 7. The two hang guards are tested at their boundary: one pair fewer raises, exactly k runs.
5. **Draws:** `RDraws.sample_replace` against the new `sample.int` fixtures, and `ReplayDraws` raising on a kind mismatch.
6. **Mechanism checks at verify time:** hand-made mutants must each fail the suite: the last minimum instead of the first, a fresh `tb` instead of the stale one, the Lloyd shortcut without the `yy0` reset, and a merge row with its columns swapped.
7. **Platforms:** Python 3.14 and 3.15 on the laptop and on the Mac mini (arm64 has no x87, and the compiled long double makes the bits match there too, as the current suite shows).

## 10. Speed gates

Timed through `benchmarks/pin.py` (one thread per P-core, EcoQoS off), with R and Python in one sitting. `benchmarks/wcluster.R` times R and writes its inputs and outputs to `tmp/wcluster/out`. `benchmarks/wcluster.py` times Python on the same inputs and checks its outputs bit for bit against R's, as `covertype.py` does. First-call compile time is reported on its own line.

| Workload | R 4.6.1 | Gate |
|---|---|---|
| `whclust`, quakes n = 200 and 400 | 80 s and 327 s | under 1 s at both |
| `whclust`, n = 800 | about 22 min, run once for the record | under 1 s, and equal to R |
| `whclust`, n = 2,000 Covertype rows | about 3 h by extrapolation, not run | under 10 s |
| `wkmeans` k = 4, quakes n = 100, 250, 500, 1,000 | 1.7, 5.9, 3.9 and 8.4 s | at least 50x faster at each |
| `wkmeans` k = 4, n = 10,000 Covertype rows | not run: R's `diag(w)` alone is 800 MB | under 30 s |
| `dn_whclust` on 2,000 nuggets (`create_dn`'s default) | as `whclust` at 2,000 | under 10 s |

The Covertype rows are the first n rows of the matrix the existing Covertype benchmark uses. A gate that misses is reported with a profile, and its target isn't loosened without RON2K's agreement.

## 11. Departures from R

| R | Python | Why |
|---|---|---|
| labels are 1-based | 0-based, R's label minus one | the package convention |
| `merge` uses negative singletons and 1-based steps | scipy numbering, one to one | the package convention and `to_linkage()` |
| `cl.centers` is p by k | `centers` is k by p | numpy's row-per-cluster convention |
| prints and messages | `logging.debug` | a library doesn't write to stdout |
| hangs when `n < 2k` or with fewer than k distinct (row, weight) pairs | `ValueError` | R never returns |
| `num.init` or `max.iterations` below 1 run `1:0` twice | `ValueError` | R's `1:0` counts down, which isn't intended behavior |
| NaN or Inf in x or w reach the arithmetic | rejected up front | the package rule in `as_matrix` and `as_weights` |
