# API reference

This manual documents the public Python API for **datanuggets 0.1.0a3**. For installation and a first analysis, start with the [installation guide](installation.md) and [quickstart](quickstart.md). This page gives function-level arguments, return values, examples, and compatibility notes.

The 18 core functions are exported from `datanuggets`, and eight plotting helpers live in `datanuggets.plot`. Three helper modules need no plotting library: `datanuggets.colors`, `datanuggets.filters` and `datanuggets.projection`.

The [math reference](https://github.com/ron2k1/datanuggets/blob/main/docs/math/README.md) states what each of the 18 functions computes, as equations. It is a draft that the authors of the R packages have not reviewed.

| Task | Functions |
| --- | --- |
| Create and refine nuggets | [`create_dn`](#create_dn), [`create_dn_centers`](#create_dn_centers), [`refine_dn`](#refine_dn), [`get_dn_means`](#get_dn_means) |
| Estimate arithmetic work | [`arithmetic_complexity`](#arithmetic_complexity) |
| Calculate weighted summaries | [`wmean`](#wmean), [`wss`](#wss), [`wwcss`](#wwcss), [`distw`](#distw) |
| Cluster and predict | [`wkmeans`](#wkmeans), [`whclust`](#whclust), [`cut_tree`](#cut_tree), [`cluster_predict`](#cluster_predict) |
| Run weighted PCA | [`wpca`](#wpca) |
| Analyze nuggets | [`dn_wkmeans`](#dn_wkmeans), [`dn_whclust`](#dn_whclust), [`dn_wpca`](#dn_wpca), [`dn_cluster_predict`](#dn_cluster_predict) |
| Colour clusters, filter rows, place rows in a PCA view, fit t-SNEs | [Helper modules](#helper-modules) |
| Draw and export figures | [Plotting](#plotting) |
| Inspect returned objects | [Result objects](#result-objects) |

## Conventions

- **Shapes:** `n` means input rows, `p` features, `m` nuggets, and `k` clusters. Prepare numeric features before calling the package; it does not select columns, fill missing values, or standardize features.
- **Inputs:** core functions that read numeric input data reject NaN and infinity. Nugget creation and refinement require a two-dimensional numeric array or data frame. They read Boolean columns as 0/1 in a data frame that also has numeric columns, and raise `TypeError` for all-Boolean data. `get_dn_means` and the weighted analysis functions also read all-Boolean data as 0/1, and the weighted analysis functions accept a vector as one feature. Prediction data must be two-dimensional.
- **Weights:** `None` means ones. Supply one finite numeric weight per row. Positive weights are the usual choice. Several WCluster ports accept zero or signed weights for R compatibility; these can produce undefined means, NaNs, or degenerate clusters. PCA has stricter rules.
- **Labels:** generated nugget and cluster IDs are **0-based**. Input `cl` labels must be finite numeric whole numbers smaller than 1e15 in absolute value, one per row. They need not be consecutive. Cluster summaries follow sorted distinct label order. Plotting requires labels with integer dtype.
- **Randomness:** `seed` accepts an integer, a NumPy `Generator`, or an `RDraws` R-compatible random-number generator. Integer seeds use NumPy: the same seed number in R and Python does not imply the same random draws. `create_dn`, `refine_dn`, `wkmeans` and `dn_wkmeans` take R's default seed, `291102`, as an integer. `seed=RDraws(291102)` gives the draws of R's `create.DN`, `refine.DN`, `Wkmeans` and `DN.Wkmeans` at that default. Import it with `from datanuggets import RDraws`. `RDraws(seed)` takes an int or a float and truncates it toward zero, as R's `set.seed` does. A bool, or a seed that is not a `numbers.Real`, raises `TypeError`. A NaN seed, or one that does not truncate to an integer strictly between -2^31 and 2^31, raises `ValueError` "supplied seed is not a valid integer". See [reproducibility](reproducibility.md) and [R-compatible randomness](compatibility.md#rdraws).
- **Parallelism:** `n_jobs=1` runs serially. Other positive Python ints select worker counts and `-1` uses all cores. A float, a bool, a NumPy integer or any other value raises `ValueError`. `progress=True` requires the `progress` extra. Warnings use [`DataNuggetWarning`](#datanuggetwarning); informational messages use the `datanuggets` logger.
- **Results:** core functions return arrays, scalars, or named result objects, rather than R lists. Result-array fields are read-only copies. See [result objects](#result-objects).

### R-to-Python function map

The entries below follow the Description, Usage, Arguments, Details, Value, Examples, and See also structure. Python names use `snake_case`.

Each entry's Usage gives the R call after the Python one, and its Arguments table names the R argument beside each Python argument.

| R | R package | Python |
| --- | --- | --- |
| `create.DN` | `datanugget` 1.5.0 | [`create_dn`](#create_dn) |
| `create.DNcenters` | `datanugget` 1.5.0 | [`create_dn_centers`](#create_dn_centers) |
| `refine.DN` | `datanugget` 1.5.0 | [`refine_dn`](#refine_dn) |
| `getDN.means` | `datanugget` 1.5.0 | [`get_dn_means`](#get_dn_means) |
| `AC` | `datanugget` 1.5.0 | [`arithmetic_complexity`](#arithmetic_complexity) |
| `wmean` | `WCluster` 1.3.0 | [`wmean`](#wmean) |
| `wss` | `WCluster` 1.3.0 | [`wss`](#wss) |
| `wwcss` | `WCluster` 1.3.0 | [`wwcss`](#wwcss) |
| `distw` | `WCluster` 1.3.0 | [`distw`](#distw) |
| `Wkmeans` | `WCluster` 1.3.0 | [`wkmeans`](#wkmeans) |
| `Whclust` | `WCluster` 1.3.0 | [`whclust`](#whclust) |
| `cutree` | `stats` | [`cut_tree`](#cut_tree) |
| `cluster.predict` | `WCluster` 1.3.0 | [`cluster_predict`](#cluster_predict) |
| `Wpca` | `WCluster` 1.3.0 | [`wpca`](#wpca) |
| `DN.Wkmeans` | `WCluster` 1.3.0 | [`dn_wkmeans`](#dn_wkmeans) |
| `DN.Whclust` | `WCluster` 1.3.0 | [`dn_whclust`](#dn_whclust) |
| `DN.Wpca` | `WCluster` 1.3.0 | [`dn_wpca`](#dn_wpca) |
| `DNcluster.predict` | `WCluster` 1.3.0 | [`dn_cluster_predict`](#dn_cluster_predict) |

**`optimal.DN` is not ported.** The Python plotting helpers and helper modules are additional APIs, not direct translations of every R plotting method.

The upstream references are the [datanugget 1.5.0 HTML manual](https://cran.r-project.org/web/packages/datanugget/refman/datanugget.html) and [WCluster 1.3.0 manual](https://cran.r-project.org/web/packages/WCluster/WCluster.pdf). This page describes the Python implementation; consult [compatibility](compatibility.md) for intentional differences and edge cases.

## Example data

The core examples share this setup. From a source checkout, run it at the repository root. It loads the measured R `quakes` data already included in the parity fixtures; fixture files are not required when using your own matrix `X`.

```python
from pathlib import Path
import json

import numpy as np
import datanuggets as dnlib

fixture = Path("tests/fixtures/create_dn/quakes_seed7_random")
spec = json.loads((fixture / "manifest.json").read_text())["arrays"]["x"]
raw = np.fromfile(fixture / spec["file"], dtype=spec["dtype"]).reshape(spec["shape"])

# Put the five features on comparable scales for this distance-based example.
X = (raw - raw.mean(axis=0)) / raw.std(axis=0)

dn = dnlib.create_dn(X, r=300, dn_num1=200, dn_num2=40, seed=42)
tree = dnlib.dn_whclust(dn)
nugget_labels = dnlib.cut_tree(tree, k=3)
pca = dnlib.dn_wpca(dn)
```

For your own data, replace the fixture-loading block with your numeric `X` and choose preprocessing for its features. Check for constant columns before dividing by standard deviations. These examples demonstrate API use; they do not select an optimal nugget count or establish the quality of three clusters.

## Creating and refining nuggets

### `create_dn`

**Description**

Reduce a numeric data set to weighted representative points while retaining each original row's nugget assignment.

**Usage**

```text
create_dn(x, center_method="mean", r=5000, delete_percent=0.1,
          dn_num1=10000, dn_num2=2000, dist_metric="euclidean",
          seed=291102, n_jobs=1, progress=False)
```

```r
# datanugget 1.5.0
create.DN(x, center.method = "mean", R = 5000, delete.percent = .1, DN.num1 = 10^4, DN.num2 = 2000, dist.metric = "euclidean", seed = 291102, no.cores = (parallel::detectCores() - 1), make.pbs = FALSE)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Numeric `(n, p)` data with at least two rows. R takes a matrix, data frame or data.table; Python takes any 2-D numeric array-like, pandas frames included. |
| `center_method` | `center.method` | `"mean"` uses the mean of assigned rows; `"original"` retains the selected input center; `"random"` chooses an assigned input row. |
| `r` | `R` | Initial chunk-size control, within `[100, 10000]`. A fraction is accepted. The algorithm uses `ceil(n / r)` chunks. |
| `delete_percent` | `delete.percent` | Fraction removed per thinning round; strictly between 0 and 1. |
| `dn_num1` | `DN.num1`, default `10^4` | Target for centers pooled from the first stage. Python requires a whole number of at least 1; R checks only that it is numeric. |
| `dn_num2` | `DN.num2` | Target for the final stage, with the same whole-number rule; it cannot exceed `dn_num1`. |
| `dist_metric` | `dist.metric` | `"euclidean"` or `"manhattan"`, used for center selection and assignment. On an exact distance tie a row goes to the lowest-numbered center ([exact ties](compatibility.md#exact-distance-ties-in-create_dn)). |
| `seed` | `seed` | Draws the chunk shuffle and the `"random"` centers ([randomness](#conventions)). A float or bool seed raises `TypeError`. |
| `n_jobs` | `no.cores`, default `(parallel::detectCores() - 1)` | Threads for the first-stage chunks; default serial. Only stage 1 runs in parallel, and the result does not depend on `n_jobs`. R's `no.cores` also parallelizes the assignment step. |
| `progress` | `make.pbs` | Draw a `tqdm` bar over the first-stage chunks when `True`; without `tqdm` it raises `ImportError`. R's `make.pbs` also draws a bar over the stage-2 deletions. |

**Details**

The algorithm shuffles rows into chunks, thins each chunk, then thins the pooled centers. It assigns the original rows to the retained centers before applying `center_method`. Changing centers to means or sampled rows does not trigger another assignment step. With `"original"` or `"random"` every center is a row of `x`. With `"mean"` or `"random"` an empty nugget keeps its stage-2 center.

The requested counts are targets. Stage 1 gives each chunk `dn_num1 // chunks` centers, gives the remainder to the first chunks one each, and clamps every share to `[1, chunk size]`, so the pooled count can exceed or fall short of `dn_num1`: `dn_num1=2` over 6 chunks pools 6 centers. The final count `m` is at most `dn_num2`. A selection stage that already has no more than its requested count returns its input with a `DataNuggetWarning` carrying R's text, "DN.num is greater than or equal to the number of rows. Returning original data". Empty nuggets are retained.

Scale is the average sample variance across features. A nugget with fewer than two rows gets a floor instead, and so does a nugget whose scale is not finite. The floor is the smallest finite scale among nuggets with two or more rows divided by 100. It is float64 eps (`2.22e-16`) when there is no such scale or that value is not positive.

Center selection builds pairwise distance matrices. Stage 2 holds a square float64 matrix over the pooled centers, 800 MB at the default `dn_num1=10000`. Each stage-1 chunk of `c` rows needs a `c` by `c` matrix, 200 MB for a 5,000-row chunk, and each `n_jobs` thread holds one. See [limitations](limitations.md).

It raises `ValueError` for `r` outside `[100, 10000]`, `delete_percent` outside `(0, 1)`, a `dn_num1` or `dn_num2` that is below 1, fractional, NaN or infinite, `dn_num2 > dn_num1`, an unknown `center_method` or `dist_metric`, and an `n_jobs` that is not a Python int or is 0 or below -1. It also raises `ValueError` for an `x` that is not 2-D, has no columns, holds NaN or infinity or has fewer than two rows. It raises `TypeError` for an `r`, `delete_percent`, `dn_num1` or `dn_num2` that is a bool or not a real number, for non-numeric or all-bool `x`, and for a seed that is not an int, `Generator` or `Draws`. Fewer than two rows gives R's message, "x must contain at least two observations". `dn_num2 > dn_num1` gives "dn_num2 must be less than dn_num1", where R says "DN.num2 must be less than DN.num1".

**Value**

A [`DataNuggets`](#datanuggets) object with `centers (m, p)`, `weights (m,)`, `scales (m,)`, and `assignments (n,)`. `centers` and `scales` are float64, `weights` and `assignments` int64. Weights count assigned rows; assignments are 0-based nugget IDs. R's list of class `datanugget` holds the `"Data Nuggets"` table, whose `Center1` to `Centerp`, `Weight` and `Scale` columns become `centers`, `weights` and `scales` (`to_frame()` rebuilds the table with 0-based `Data Nugget` ids), and the 1-based `"Data Nugget Assignments"`, which becomes `assignments`.

**Examples**

Using `X` from [example data](#example-data):

```python
created = dnlib.create_dn(
    X, center_method="mean", r=300,
    dn_num1=200, dn_num2=40, seed=42,
)
assert created.weights.sum() == len(X)
row_representatives = created.centers[created.assignments]
```

**See also:** [`refine_dn`](#refine_dn), [`create_dn_centers`](#create_dn_centers), [quickstart](quickstart.md).

**References:** [Beavers et al. (2025)](#references), [Cherasia et al. (2023)](#references).

### `create_dn_centers`

**Description**

Select representative rows from a sample by repeatedly removing members of close pairs.

**Usage**

```text
create_dn_centers(rs, dn_num, *, delete_percent=0.1,
                  dist_metric="euclidean")
```

```r
# datanugget 1.5.0
create.DNcenters(RS, delete.percent = 0.1, DN.num, dist.metric = "euclidean", make.pbs = FALSE)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `rs` | `RS` | Numeric `(n, p)` sample. All-bool data raises `TypeError`. Input that is not 2-D, has no columns or holds NaN or infinity raises `ValueError`. |
| `dn_num` | `DN.num` | Target number of retained rows, with no default on either side. Python requires a whole number of at least 1 (`ValueError` otherwise, `TypeError` for a bool or a non-number). R checks only that it is numeric. It is R's third argument and Python's second. |
| `delete_percent` | `delete.percent` | Fraction removed per round, strictly between 0 and 1. Keyword-only, where it is R's second argument. |
| `dist_metric` | `dist.metric` | `"euclidean"` or `"manhattan"`; anything else raises `ValueError`. Keyword-only. |

Not ported: `make.pbs`. There is no progress bar, and `make_pbs=` raises `TypeError`. A call in R's positional order, `create_dn_centers(rs, 0.1, 50)`, also raises `TypeError` instead of taking 0.1 as the count.

**Details**

R's help page says the function reduces a random sample to data nugget centers in the `create.DN` function, and adds: "NOTE THAT THIS FUNCTION IS NOT DESIGNED FOR USE OUTSIDE OF THE create.DN FUNCTION." The Python function checks its own inputs. It checks `delete_percent`, `dist_metric` and `dn_num` in that order and then reads `rs` as a float64 matrix. The first check that fails raises. When `rs` has more than `dn_num` rows, it computes the distance between every pair of rows under `dist_metric`. Each round then deletes `max(1, floor(t * delete_percent))` of the `t` remaining rows, but never so many that fewer than `dn_num` remain. The rounds end with exactly `dn_num` rows. `rs` itself is not changed.

Selection draws nothing, so it is deterministic for a fixed sample and row order and takes no seed. It returns selected input rows, without computing weights, scales, or assignments. If `dn_num >= n`, all rows are returned with a `DataNuggetWarning` carrying R's text, "DN.num is greater than or equal to the number of rows. Returning original data". Pairwise distances require quadratic memory in sample size.

**Value**

A [`CentersResult`](#centersresult): `centers` has shape `(m, p)` and `kept` has shape `(m,)`. Each entry in `kept` is a **0-based row position in `rs`**, in ascending order, and `centers` equals `rs[kept]`. R returns the kept rows as a data frame with row names 1 to m, which becomes the unnamed float64 `centers`, and their 1-based positions in its `"kept"` attribute, which become the int64 `kept`.

**Examples**

```python
sample = X[:100]
selected = dnlib.create_dn_centers(sample, dn_num=20)
np.testing.assert_array_equal(selected.centers, sample[selected.kept])
```

**See also:** [`create_dn`](#create_dn).

**References:** [Beavers et al. (2025)](#references), [Cherasia et al. (2023)](#references).

### `refine_dn`

**Description**

Split overly dispersed nuggets, optionally followed by a pass that attempts to improve elongated nuggets.

**Usage**

```text
refine_dn(x, dn, ev_tol=0.9, max_splits=5, min_nugget_size=2,
          shape_split=False, min_shape_size=10, delta=2,
          nstart=25, seed=291102, n_jobs=1, progress=False)
```

```r
# datanugget 1.5.0
refine.DN(x, DN, EV.tol = 0.9, max.splits = 5, min.nugget.size = 2, shape.split = FALSE, min.shape.size = 10, delta = 2, nstart = 25, seed = 291102, no.cores = (parallel::detectCores() - 1), make.pbs = FALSE)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Original numeric data, in exactly the row order used to create `dn`. Its row count must equal `len(dn.assignments)` and its column count that of `dn.centers`; R checks only the rows. |
| `dn` | `DN` | A `DataNuggets` object; anything else raises `TypeError`. Only `dn.assignments` and the shape of `dn.centers` are read. Every center, weight, scale and eigenvalue is recomputed from `x`. Ids must be whole numbers in `[0, m)`. |
| `ev_tol` | `EV.tol` | Quantile in `(0, 1)` used to set the first-eigenvalue splitting threshold before the first round. |
| `max_splits` | `max.splits` | Maximum rounds in the eigenvalue pass, and again in the optional shape pass. Floored. A negative, NaN or infinite value raises `ValueError`. |
| `min_nugget_size` | `min.nugget.size` | Minimum child size for splitting; at least 2, floored. |
| `shape_split` | `shape.split` | Enable the shape pass. It has no effect with one feature or with `max_splits=0`. Python takes only a bool and raises `TypeError` otherwise; R does no type check. |
| `min_shape_size` | `min.shape.size` | Minimum nugget size considered for shape splitting; at least 10, not floored. |
| `delta` | `delta` | Eigenvalue-ratio threshold for the shape pass; greater than 1. |
| `nstart` | `nstart` | Random starts for each two-means split. A fraction of 2 or more is floored. A value other than 1 that is below 2 makes every two-means call fail, so no nugget splits, as in R. NaN or infinity raises `ValueError`. |
| `seed` | `seed` | With an int seed or an `RDraws`, each split starts from a fresh copy of the seed's stream, as R's `set.seed(seed)` before every split does. With a NumPy `Generator`, each split draws from its own child stream. Each round spawns its children from the Generator in candidate order before any of its splits run, so the result still does not depend on `n_jobs`. A Generator made fresh from the same seed repeats the result. A second call that reuses one Generator object spawns new children and can split differently. |
| `n_jobs` | `no.cores`, default `(parallel::detectCores() - 1)` | Each round's splits run in this many worker processes, and the nuggets' centers and covariances are computed on up to this many numba threads. The result does not depend on `n_jobs`. With `n_jobs` other than 1, a replayed or custom `Draws` raises `ValueError`; an int, a `Generator` and `RDraws` are allowed. |
| `progress` | `make.pbs` | Draw a `tqdm` bar per split round, as R does; without `tqdm` it raises `ImportError`. |

**Details**

The eigenvalue threshold is the type-7 quantile at `ev_tol` of the nuggets' positive finite first eigenvalues (EV1, the largest eigenvalue of a nugget's sample covariance), calculated once from the initial nuggets. In each round of the eigenvalue pass, every nugget with a finite EV1 above the threshold and at least `2 * floor(min_nugget_size)` rows is split by two-means. A split is kept only if both children have at least `floor(min_nugget_size)` rows, and a round with no kept split ends the pass.

The shape pass takes nuggets with at least `max(min_shape_size, 2 * floor(min_nugget_size))` rows and a positive finite EV1. Of those it splits a nugget whose EV1/EV2, the ratio of its two largest eigenvalues, exceeds `delta`, or whose EV1 exceeds the median plus twice the MAD of the scales of nuggets with two or more rows. The split is 2-means on the projection onto the nugget's first eigenvector. It is kept only when both children have at least `floor(min_nugget_size)` rows and the mean of their EV1/EV2 is below `delta`. In both passes a kept split's second child becomes a new nugget after the existing ones.

Refinement uses `dn.assignments` to recompute centers, weights, and scales from `x`. The centers are always column means, whatever `center_method` built `dn`. It removes empty nuggets and renumbers survivors in position order. Scale is the trace of the sample covariance divided by `p`, and single-row nuggets receive scale 0. Even `max_splits=0` performs this resummarization.

A small numerical change near a splitting threshold can change later splits. See [refinement compatibility](compatibility.md#refine_dn) before expecting an identical hierarchy across numerical libraries.

It raises `TypeError` when `dn` is not a `DataNuggets`, when `x` is not numeric or is all bool, for a `shape_split` that is not a bool, and for a bad seed type. It also raises `TypeError` for an `ev_tol`, `max_splits`, `min_nugget_size`, `min_shape_size`, `delta` or `nstart` that is a bool or not a real number. It raises `ValueError` for out-of-range arguments, a non-finite `max_splits` or `nstart`, an `n_jobs` that is not a Python int or is 0 or below -1, and ids that are not whole or fall outside `[0, m)`. It also raises `ValueError` for an `x` that is not 2-D, has no rows or columns, holds NaN or infinity or does not match the rows and columns of `dn`. With two or more columns, a nugget whose sample covariance overflows to infinity raises `ValueError` with R's `eigen()` message, "infinite or missing values in 'x'". Two-means warnings, including those from worker processes, arrive as `DataNuggetWarning`.

**Value**

A new [`DataNuggets`](#datanuggets) with assignments for all original rows. R's `refine.DN` returns the same `datanugget` list as `create.DN`, renumbered 1 to m, and its components map to the fields as under [`create_dn`](#create_dn).

**Examples**

```python
refined = dnlib.refine_dn(X, dn, max_splits=1, nstart=5, seed=42)
assert refined.weights.sum() == len(X)
assert refined.assignments.shape == (len(X),)
```

**See also:** [`create_dn`](#create_dn), [`get_dn_means`](#get_dn_means).

**References:** [Beavers et al. (2025)](#references), [Cherasia et al. (2023)](#references).

### `get_dn_means`

**Description**

Calculate the mean of the original rows assigned to each nugget, regardless of the center method used to create it.

**Usage**

```text
get_dn_means(x, dn)
```

```r
# datanugget 1.5.0
getDN.means(x, DN)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Original `(n, p)` data, with the same row order and feature count as `dn`. Bool data is read as 0/1; a 1-D vector raises `ValueError`. A row count other than `len(dn.assignments)` raises `ValueError`, where R ignores rows past the assigned ones. |
| `dn` | `DN` | A `DataNuggets` object whose assignments identify each row's nugget. Ids that are not whole numbers in `[0, m)` raise `ValueError`, where R skips their rows. |

**Details**

Empty nuggets produce a row of NaN with the sign bit set (`fff8000000000000`), R's `-NaN`. The Python output remains two-dimensional for one feature, where R transposes it into one row. With `center_method="mean"` the result equals `dn.centers` only to rounding, and at an empty nugget `dn.centers` holds the stage-2 center while the mean is NaN.

It raises `TypeError` when `dn` is not a `DataNuggets` or `x` is not numeric. It raises `ValueError` for an `x` that is not 2-D or has no columns, for NaN or infinity, for a row-count or column-count mismatch, and for bad ids.

**Value**

A float64 array of shape `(m, p)`. Row `i` contains the means for nugget `i`. R returns a data frame with columns `Center1` to `Centerp`, which becomes this unnamed array.

**Examples**

```python
means = dnlib.get_dn_means(X, dn)
np.testing.assert_allclose(means, dn.centers)  # This setup used center_method="mean".
```

**See also:** [`create_dn`](#create_dn), [`refine_dn`](#refine_dn).

**References:** [Beavers et al. (2025)](#references), [Cherasia et al. (2023)](#references).

### `arithmetic_complexity`

**Description**

Evaluate the port of R's `AC` arithmetic-work estimate for nugget creation.

**Usage**

```text
arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2)
```

```r
# datanugget 1.5.0
AC(x, R, delete.percent, DN.num1, DN.num2)
```

**Arguments**

No argument has a default, in R or in Python.

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Two-dimensional data with at least one row; only its dimensions are used, so its values can be anything. R's `AC` accepts only a plain data frame. |
| `r` | `R` | Chunk-size control within `[100, 10000]`, floored before counting. |
| `delete_percent` | `delete.percent` | Thinning fraction, strictly between 0 and 1. |
| `dn_num1` | `DN.num1` | First-stage center target, floored before counting. |
| `dn_num2` | `DN.num2` | Second-stage center target, floored before counting. It must not exceed `dn_num1`, compared before flooring. |

**Details**

The estimate follows the R counting formula and its rounding rules:

```text
log10(p * (G * A * R1 * (R1 - 1) / 2  +  B * D * (D - 1) / 2  +  (n + 1) * D))
```

`G = ceil(n / floor(r))` is the chunk count, `R1 = floor(dn_num1) / G` the per-chunk target and `D = floor(dn_num2)`. `A` and `B` are 1 plus the passes of R's two loops, which take `floor(r)` to `R1` and `floor(dn_num1)` to `D`. A pass sets the value to the target when removing `floor(delete_percent * value)` would take it below the target. Otherwise it removes that amount, or 1 when the amount is 0. A step of 1 can pass a fractional target, and the next pass moves back up to it. With `n=1000`, `r=150`, `delete_percent=0.005` and `dn_num1=701`, `R1` is 100.142857: the first loop steps from 150 down to 100 one at a time and then up to `R1`, so `A` is 52. A target above the start value is reached by the first pass, which raises the value: `n=1000`, `r=300` and `dn_num1=2000` give `R1 = 500`, and `A` is 2. The loops run in R's double arithmetic, so `A` and `B` match R's counts. Inputs that would make R's loops never return raise `ValueError`; [compatibility](compatibility.md#arithmetic_complexity) gives the conditions.

It raises `TypeError` for an `r`, `delete_percent`, `dn_num1` or `dn_num2` that is a bool or not a real number. It raises `ValueError` for `r` outside `[100, 10000]`, `delete_percent` outside `(0, 1)`, a non-finite `dn_num1` or `dn_num2`, an `x` that is not 2-D or has no rows, and `dn_num2 > dn_num1`. The last gives "dn_num2 must not exceed dn_num1", where R says "DN.num2 must be less than DN.num1".

**Value**

A Python float: the base-10 logarithm of the estimated arithmetic-operation count. A count of 0 gives `-inf`, and a negative count gives `nan` with `DataNuggetWarning` "NaNs produced", as R's `log10` does. R's `AC` returns the same number as a length-one numeric vector.

**Examples**

```python
log10_work = dnlib.arithmetic_complexity(
    X, r=300, delete_percent=0.1, dn_num1=200, dn_num2=40,
)
```

**See also:** [`create_dn`](#create_dn), [limitations](limitations.md).

**References:** [Beavers et al. (2025)](#references), [Cherasia et al. (2023)](#references).

## Weighted summaries

Each function replays R's arithmetic in R's order. On the recorded R cases the tests compare, finite and infinite results match R's bit for bit, and NaN results are NaN where R's are. The NA cells on `distw`'s diagonal and upper triangle also match R's bit pattern. Integer `x` with integer weights is an exception: R's `wmean` gives an NA center when the integer products overflow, where the port computes in float64 ([integer input](compatibility.md#the-wcluster-statistics-and-wpca)). For the examples, `dn.centers` supplies observations and `dn.weights` their counts.

### `wmean`

**Description**

Calculate a weighted feature mean for each cluster.

**Usage**

```text
wmean(x, cl, w=None)
```

```r
# WCluster 1.3.0
wmean(x,cl,w)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Numeric observations. |
| `cl` | `cl` | One numeric whole-number cluster label per row, smaller than 1e15 in magnitude. Text and bool labels raise `TypeError`; fractional and NaN labels raise `ValueError`. R groups by `factor(cl)`, so it also takes text. |
| `w` | `w`, default `rep(1,length(x))` | One row weight per observation; `None` uses ones. R's usage line shows no default. Its argument text says `NULL` means unit weights, but R's code stops on `NULL` at its length check. The default in R's code is evaluated after `x` becomes a data frame, so it has `ncol(x)` entries and stops unless `x` is square. Bool or text weights raise `TypeError`. |

**Details**

Clusters are ordered by sorted distinct labels. Each coordinate is R's `weighted.mean` of the cluster's rows, summed in long double. A cluster whose weights are all zero gets a row of NaN. Weights of both signs that cancel to a zero sum give infinities instead, with NaN in any column whose weighted sum is also zero. Both are R's results. Python returns one row per cluster; R's `wmean` uses the opposite orientation.

**Value**

A float64 array of shape `(k, p)` in sorted label order. R returns the `p` by `k` transpose, with columns named `Cluster Center.1` to `Cluster Center.k` and rows named after the columns of `x`; the Python array carries no names.

**Examples**

```python
cluster_means = dnlib.wmean(dn.centers, nugget_labels, w=dn.weights)
cluster_ids = np.unique(nugget_labels)  # cluster_ids[i] names cluster_means[i].
```

**See also:** [`wss`](#wss), [`wwcss`](#wwcss), [`cluster_predict`](#cluster_predict).

**References:** [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

### `wss`

**Description**

Measure weighted squared deviation from the overall weighted feature mean.

**Usage**

```text
wss(x, w=None)
```

```r
# WCluster 1.3.0
wss(x,w = rep(1,nrow(x)))
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Numeric data. |
| `w` | `w`, default `rep(1,nrow(x))` | One weight per row, or `None` for ones, the weights R's default gives. R's argument text says `NULL` means unit weights, but R's code stops on `NULL` at its length check. Signs are not checked. Bool or text weights raise `TypeError`. |

**Details**

The function sums squared Euclidean distances from the weighted mean, multiplied by row weights. The mean and the squares go through R's matrix product. A zero total weight, including weights of both signs that cancel, gives NaN, as in R.

**Value**

A Python `float`, where R returns a length-one numeric vector. `wwcss(..., group_sum=True).total` is a 0-d array instead.

**Examples**

```python
total_variation = dnlib.wss(dn.centers, w=dn.weights)
```

**See also:** [`wmean`](#wmean), [`wwcss`](#wwcss).

**References:** R's help page for `wss` lists no references.

### `wwcss`

**Description**

Measure weighted within-cluster variation around each cluster's weighted mean.

**Usage**

```text
wwcss(x, cl, w=None, group_sum=False)
```

```r
# WCluster 1.3.0
wwcss(x, cl, w = rep(1,length(x)), groupSum = FALSE)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Numeric observations. |
| `cl` | `cl` | One cluster label per row, with [`wmean`](#wmean)'s label rules. R groups by `factor(cl)`. |
| `w` | `w`, default `rep(1,length(x))` | Row weights, or `None` for ones. R's default is evaluated after `x` becomes a data frame, so it has `ncol(x)` entries and stops unless `x` is square. R's argument text says `NULL` means unit weights, but R's code stops on `NULL` at its length check. |
| `group_sum` | `groupSum` | Return individual cluster sums as well as their total when `True`. Python tests only truthiness, so the string `"FALSE"` also returns a `WwcssResult`, where R reads it as false. |

**Details**

Per-cluster values follow sorted distinct label order. A cluster whose weights sum to zero, all zero or cancelling, contributes NaN, making the total NaN too. The total is R's long-double sum of the per-cluster values, so it can differ from `per_cluster.sum()` in the last bit.

**Value**

With `group_sum=False`, a Python `float` total. With `True`, a [`WwcssResult`](#wwcssresult) containing `per_cluster (k,)` and `total ()`, a 0-d float64 array. R's `groupSum = TRUE` list holds `WWCSS`, named 1 to k, which becomes the unnamed `per_cluster`, and `TotalWWCSS`, which becomes `total`.

**Examples**

```python
within = dnlib.wwcss(
    dn.centers, nugget_labels, w=dn.weights, group_sum=True,
)
np.testing.assert_allclose(within.total, within.per_cluster.sum())
```

**See also:** [`wss`](#wss), [`wkmeans`](#wkmeans), [`distw`](#distw).

**References:** [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

### `distw`

**Description**

Calculate the change in weighted within-cluster variation from merging each pair of clusters.

**Usage**

```text
distw(x, cl, w=None)
```

```r
# WCluster 1.3.0
distw(x,cl,w)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Numeric data. |
| `cl` | `cl` | One label per row, with at least two distinct values and [`wmean`](#wmean)'s label rules. One cluster raises `ValueError`, where R stops with "missing value where TRUE/FALSE needed". |
| `w` | `w`, default `rep(1,length(x))` | Row weights, or `None` for ones. R's usage line shows no default. Its argument text says `NULL` means unit weights, but R's code stops on `NULL` at its length check. The default in R's code is evaluated after `as.matrix(x)`, so it has `n * p` entries and stops unless `x` has one column. |

**Details**

Each populated entry is the merged cluster's weighted sum of squares minus R's long-double sum of the two separate sums. This is a merge cost, not a complete symmetric distance matrix. Rounding can leave an entry at exactly zero or slightly below it, and those values are retained. Every entry in the row or column of a cluster whose weights sum to zero is NaN.

**Value**

A `(k, k)` float64 array in sorted label order. Values occupy the **strict lower triangle**. The diagonal and upper triangle hold R's `NA` bit pattern (`7ff00000000007a2`), which `np.isnan` detects and a bit-level comparison with R matches. R's matrix also names its rows and columns `Cluster 1` to `Cluster k`; the Python array carries no names.

**Examples**

```python
merge_costs = dnlib.distw(dn.centers, nugget_labels, w=dn.weights)
pair_costs = merge_costs[np.tril_indices(len(merge_costs), k=-1)]
```

**See also:** [`wwcss`](#wwcss), [`whclust`](#whclust).

**References:** [Ward (1963)](#references), [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

## Clustering and prediction

### `wkmeans`

**Description**

Partition rows with WCluster's weighted k-means algorithm.

**Usage**

```text
wkmeans(x, k, *, w=None, centers=None, num_init=1, max_iter=10,
        seed=291102, n_jobs=1)
```

```r
# WCluster 1.3.0
Wkmeans(dataset, k, cl.centers = NULL, obs.weights = rep(1, nrow(dataset)), num.init = 1, max.iterations = 10, seed = 291102)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `dataset` | Numeric `(n, p)` data, or a vector for one feature. R stops on a vector `dataset`. |
| `k` | `k` | Requested cluster count, floored as R floors it, so `k=2.7` acts as 2 without `centers`. With `centers`, the number of center rows must equal `k` before flooring, as R's check requires. So `k=2.7` with two center rows raises `ValueError`. A bool or non-number raises `TypeError`; NaN, infinity or a value below 1 after flooring raises `ValueError`. |
| `w` | `obs.weights`, default `rep(1, nrow(dataset))` | Row weights, or `None` for ones. It is R's fourth positional argument and keyword-only here, so `wkmeans(x, k, w)` raises `TypeError`. |
| `centers` | `cl.centers` | Optional `(k, p)` initial centers, where R takes `p` by `k`; pass R's matrix as `.T`. A `(p, k)` array raises `ValueError` unless `p` equals `k`, when it is read as `(k, p)` without an error. Supplying centers uses one initialization. |
| `num_init` | `num.init` | Number of random initializations when centers are omitted. Floored. A finite value below 1 raises `ValueError` only without `centers`, where R runs 2 - v starts for the floored value v and returns a result. With `centers`, `num_init` is set to 1 after its type check and before that range check. |
| `max_iter` | `max.iterations` | Maximum outer optimization rounds. Floored. A finite value below 1 raises `ValueError`, where R runs at most 2 - v rounds for the floored value v and returns a result. |
| `seed` | `seed` | Draws the random starts ([randomness](#conventions)). |
| `n_jobs` | Python only | Threads that refine independent initializations. The result does not depend on `n_jobs`. With `n_jobs` other than 1, including -1 for every core, every start is drawn before any is refined. |

**Details**

Each of the `max_iter` rounds runs 20 Lloyd assignment steps and then single-point moves. The run with the smallest total weighted within-cluster sum of squares wins, and the first start wins ties.

Automatic initialization requires at least `2 * k` rows and at least `k` distinct row/weight combinations. These conditions do not guarantee a successful initialization. Each start may draw at most `MAX_DRAWS` random partitions (`from datanuggets.wcluster.wkmeans import MAX_DRAWS`), and `MAX_DRAWS` is 1,000. Every partition drawn counts, including the redraws for a label on fewer than two rows and the redraws whose centers leave a cluster with no row. When a start uses up its draws, `wkmeans` raises `ValueError`, where R's loops keep drawing. Near `n = 2 * k` a start can need more draws than that. With `x = np.arange(16.0)`, `k=8` and `seed=RDraws(1)`, the first start needs 2,179 draws and the call raises. The cap only stops a start early, so a start found within it is the start an uncapped run finds. Explicit centers must produce occupied initial clusters.

The `ValueError` message ends with R's own message as `(R stops: ...)` when R stops inside the algorithm or on the value of `k`, `centers`, `num_init` or `max_iter`. An argument of the wrong class raises `TypeError` without the `(R stops: ...)` suffix. Weights of the wrong length raise `ValueError` with the port's own message. So do data, weights or centers that are not finite.

With one row and `centers`, R multiplies the distances by `diag(w)`, an identity matrix of size `as.integer(w)`. So R runs only for `1 <= w < 2`. For a finite weight of size 2^31 or more, `as.integer(w)` is NA. R warns "NAs introduced by coercion to integer range" and then stops with "invalid 'nrow' value (too large or NA)". Below that size R stops with "invalid 'nrow' value (< 0)" for `w <= -1`. From 67108865 (2^26 + 1) up to 2^31, `diag(w)` would have more than 2^52 entries, and R stops with "too many elements specified". For the other weights R stops with "non-conformable arguments" once it has built `diag(w)`. A weight too large for R to allocate `diag(w)` makes R stop on memory before that. `wkmeans` raises with R's message in each case and gives no warning. Where R stops on memory, the message is "non-conformable arguments".

Final labels can contain gaps if a cluster becomes empty. Returned center rows follow sorted **present** labels, so do not assume `result.centers[result.labels]` is always valid.

**Value**

A [`WkmeansResult`](#clustering-results) with row `labels`, occupied-cluster `centers`, per-cluster `wwcss`, and scalar `total_wwcss`. R's 1-based `"Cluster Assignments"` become `labels`, R's minus one; `"Cluster Centers"` become `centers`; and `WWCSS$WWCSS` and `WWCSS$TotalWWCSS` become `wwcss` and `total_wwcss`.

**Examples**

```python
km = dnlib.wkmeans(
    dn.centers, 3, w=dn.weights, num_init=3, seed=42,
)
occupied_ids = np.unique(km.labels)
center_for_each_nugget = km.centers[np.searchsorted(occupied_ids, km.labels)]
```

**See also:** [`dn_wkmeans`](#dn_wkmeans), [`wwcss`](#wwcss), [`cluster_predict`](#cluster_predict).

**References:** [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

### `whclust`

**Description**

Build a weighted Ward hierarchy by repeatedly merging the pair with the smallest increase in within-cluster variation.

**Usage**

```text
whclust(x, w=None)
```

```r
# WCluster 1.3.0
Whclust(x,w)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Numeric data with at least three rows. Fewer raise `ValueError` quoting R's stop: "subscript out of bounds" for two rows whose merge cost is not NaN, "missing value where TRUE/FALSE needed" for one, "invalid 'times' argument" for none. With two rows whose merge cost is NaN, as a zero weight gives, the `ValueError` quotes "argument is of length zero". |
| `w` | `w`, default `rep(1,nrow(x))` | One weight per row; `None` means ones. R's usage line shows no default. Its argument text says `NULL` means unit weights, but R's code stops on `NULL` at its length check. The default in R's code gives one weight per row. |

**Details**

Stored heights are the **total weighted within-cluster sum of squares after each merge**. They are not SciPy's usual Ward distances. Memory grows quadratically with the number of input rows, which is one reason to cluster nugget representatives. A tie goes to the pair R's `which.min` finds first. R reads its matrix of merge costs column by column, with single rows first in reverse row order and merged clusters after them in the order they formed.

A zero-weight row has NaN merge costs, so every height is NaN until that row merges. Signed weights can yield degenerate trees, including self-merges. When every remaining merge cost is NaN, it raises `ValueError`, where R stops with "argument is of length zero".

**Value**

A [`WhclustResult`](#clustering-results) containing merge IDs, heights, and leaf order. R's `hclust` `merge`, with negative row numbers for single rows and 1-based steps for merged clusters, becomes scipy's numbering; `height` keeps its values; `order` becomes 0-based. R's `labels`, `method`, `call` and `dist.method` are not ported, and `to_linkage()` gives a scipy linkage.

**Examples**

```python
weighted_tree = dnlib.whclust(dn.centers, w=dn.weights)
labels_from_tree = dnlib.cut_tree(weighted_tree, k=3)
```

**See also:** [`cut_tree`](#cut_tree), [`dn_whclust`](#dn_whclust), [`dendrogram`](#dendrogram).

**References:** [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

### `cut_tree`

**Description**

Turn a weighted hierarchy into cluster labels using a cluster count or a height threshold.

**Usage**

```text
cut_tree(tree, k=None, h=None)
```

```r
# stats
cutree(tree, k = NULL, h = NULL)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `tree` | `tree` | A `WhclustResult`. Only `.merge`, in scipy numbering, and `.height` are read, so any object with those works. R's `tree$labels` has no equivalent. |
| `k` | `k` | A single cluster count in `[1, n]`, truncated toward zero as R's `as.integer` does, so 3.9 acts as 3. An array raises `ValueError`, where R returns one column per value. |
| `h` | `h` | A single cut height, using the cumulative-variation units stored in `tree.height`. Used only when `k` is `None`. An array raises `ValueError`. |

**Details**

Supply `k` or `h`; `k` takes precedence if both are given. Cutting by height requires nondecreasing heights. With `h`, the cluster count is `n + 1` minus the 1-based position of the first height above `h`, counting a last height of infinity. So `h` below the first height gives `n` clusters, a finite `h` at or above the last height gives 1, and `h` of plus or minus infinity gives `n`, as in R. Generated labels are numbered by first appearance in the original row order, not dendrogram order.

These cases raise `ValueError` with R's message:

- neither `k` nor `h`: "either 'k' or 'h' must be specified"
- a `k` of magnitude below 2^31 that is outside `[1, n]` after truncation: "elements of 'k' must be between 1 and 20" for 20 leaves
- a NaN `k`, or a `k` of magnitude 2^31 or more, infinities included: "missing value where TRUE/FALSE needed". Before it stops on such a large `k`, R warns "NAs introduced by coercion to integer range". The port gives no warning.
- a NaN `h`: "length of 'dimnames' [2] not equal to array extent"
- a cut by `h` on NaN heights: "missing value where TRUE/FALSE needed"
- a cut by `h` on unsorted heights: "the 'height' component of 'tree' is not sorted (increasingly)"
- an empty `merge`: "invalid 'tree' ('merge' component)"

The port's own checks give other messages. An array `k` or `h` raises `ValueError` with the port's own message, where R returns one column per value. A string `k` or `h` that `float()` can't parse also raises `ValueError`, and any other value that `float()` rejects raises `TypeError`.

**Value**

An int64 array of shape `(n,)`. For a nugget tree, `n` here is the number of nuggets. For a scalar `k` or `h`, R returns a vector of 1-based group memberships, which becomes this array of R's numbers minus one. R's `cutree` names that vector after `tree$labels`. `Whclust` sets `labels` to `NULL` even when the rows of `x` have names, so R's result for its trees has no names, and the port's array carries none either.

**Examples**

```python
labels_at_three = dnlib.cut_tree(tree, k=3)
original_row_labels = labels_at_three[dn.assignments]
assert original_row_labels.shape == (len(X),)
```

**See also:** [`whclust`](#whclust), [`dn_whclust`](#dn_whclust).

**References:** [Becker, Chambers and Wilks (1988)](#references), as R's `stats::cutree` help page cites it.

### `cluster_predict`

**Description**

Assign new observations to the nearest weighted mean of an existing set of clusters.

**Usage**

```text
cluster_predict(x, cl, newx, *, w=None)
```

```r
# WCluster 1.3.0
cluster.predict(x,w = rep(1,nrow(x)),cl,newx)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Training observations. |
| `cl` | `cl` | One cluster label per training row, with [`wmean`](#wmean)'s label rules. |
| `newx` | `newx` | Two-dimensional `(n_new, p)` data, with the same features and preprocessing as `x`. `None` raises `TypeError`; a 1-D array, another column count, NaN or infinity raises `ValueError`. R stops on `NULL`, on a column mismatch, and with "argument is of length zero" on a vector `newx`, whose `ncol` is `NULL`. It does not stop on a NaN row, which empties its whole result. It takes a matrix although its help page asks for a data frame. |
| `w` | `w`, default `rep(1,nrow(x))` | Weights for training rows, or `None` for ones. It is R's second positional argument and keyword-only here, so a call in R's order, `cluster_predict(x, w, cl, newx)`, raises `TypeError`. |

**Details**

The cluster centers are [`wmean`](#wmean)'s weighted means, so `w` affects only the centers. Distances from new rows are unweighted squared Euclidean distances, computed as `|x|^2 + |c|^2 - 2 x.c` with R's sums in R's order, so rounding can decide a near-tie. Ties choose the first mean in sorted label order. Clusters with undefined distances are skipped; if every distance for a new row is NaN, the call raises `ValueError`. When one new row's squared distances overflow to infinity, R's identity multiply makes every other new row's distances NaN. The port gives the same NaNs, so the call raises.

**Value**

An integer array of shape `(n_new,)` containing **positions in sorted cluster order, not original label values**. Recover original labels with `np.unique(cl)[positions]`. R's `cluster.predict` returns the same positions 1-based, as `which.min` gives them.

**Examples**

This assigns a few observed rows to the fitted clusters; it is not a held-out evaluation.

```python
named_labels = 10 + 10 * nugget_labels
positions = dnlib.cluster_predict(
    dn.centers, named_labels, X[:5], w=dn.weights,
)
predicted_labels = np.unique(named_labels)[positions]
```

**See also:** [`wmean`](#wmean), [`dn_cluster_predict`](#dn_cluster_predict).

**References:** [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

## Weighted PCA

### `wpca`

**Description**

Compute WCluster's weighted principal components using row and optional feature weights.

**Usage**

```text
wpca(x, wrow=None, wcol=None, corr=False)
```

```r
# WCluster 1.3.0
Wpca(x, wrow = rep(1, nrow(x)), wcol = rep(1, ncol(x)), corr = FALSE)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `x` | `x` | Numeric `(n, p)` data, or a vector treated as one feature. Zero rows raise `ValueError`. |
| `wrow` | `wrow`, default `rep(1, nrow(x))` | Nonnegative row weights. `None` means ones. These act as counts: covariance divides by `sum(wrow) - 1`. |
| `wcol` | `wcol`, default `rep(1, ncol(x))` | One feature weight per column. `None` means ones. Weights are normalized to sum to `p`. |
| `corr` | `corr` | Use a weighted correlation matrix for eigendecomposition when `True`. Python takes only a bool and raises `TypeError` for `1`; R also takes `1` and `"TRUE"` ([details](compatibility.md#the-wcluster-statistics-and-wpca)). |

**Details**

Row weights summing to 0 or exactly 1, and a negative row weight, raise `ValueError`. For ordinary covariance analysis, use count-like weights whose sum exceeds 1. Probability weights normalized to 1 are unsuitable without rescaling. With `corr=False`, a sum strictly between 0 and 1 does not raise. The covariance is divided by a negative number, so each direction with positive weighted variance gets a negative eigenvalue and a NaN `sdev`, and `DataNuggetWarning` "NaNs produced" is issued once. A direction with no weighted variance has an eigenvalue of 0 up to rounding, so its `sdev` is 0, a small positive number or NaN. A single row with weight 0.5 gives `sdev` 0 in every column and no warning.

Feature weights must share one sign and have a nonzero total; mixed signs and a zero total raise `ValueError`. All-negative weights are accepted because the normalization flips them, so `[-1, -3]` becomes `[0.5, 1.5]`. For `corr=True`, the row-weight sum must exceed 1 and every feature must have positive weighted variance, so a zero `wcol` entry raises `ValueError`. Bool or text weights raise `TypeError`, as R's class check stops them.

`result.x` equals `(x - result.center) @ result.rotation` up to rounding, **even when `corr=True`**. Scores are not divided by `result.scale` or multiplied by weights. Project new observations using this same expression.

Component signs can differ between eigensolvers. Repeated eigenvalues allow different bases for the same subspace. Rank-deficient inputs can produce small negative eigenvalues, NaN standard deviations, and `DataNuggetWarning`; the implementation does not clip those eigenvalues to zero. See [numerical limitations](limitations.md). With one column and `corr=True` the port returns an `sdev` of 1 to within rounding, where R returns the unscaled value or stops.

**Value**

A [`WpcaResult`](#wpcaresult), including eigenvalue standard deviations `sdev`, directions `rotation`, scores `x`, and the centering and weighting information. R's list has the same seven names, `sdev`, `rotation`, `x`, `center`, `scale`, `wrow` and `wcol`, and each becomes the field of that name. `scale` with `corr=True` is the square root of the diagonal of the column-weighted covariance, so it includes `sqrt(wcol)`; without `corr` it is ones, one per column, where R returns the single number 1.

**Examples**

```python
weighted_pca = dnlib.wpca(dn.centers, wrow=dn.weights)
projected_rows = (X - weighted_pca.center) @ weighted_pca.rotation
np.testing.assert_allclose(
    weighted_pca.x,
    (dn.centers - weighted_pca.center) @ weighted_pca.rotation,
)
```

**See also:** [`dn_wpca`](#dn_wpca), [`scores`](#scores), [`loadings`](#loadings), [`scree`](#scree).

**References:** [Amaratunga and Cabrera (2009)](#references), [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

## Nugget analysis

These wrappers use `dn.centers` as observations and `dn.weights` as row weights, where R reads the datanugget table's center columns and its `Weight` column. Each requires a `DataNuggets` object and raises `TypeError` for anything else. Empty nuggets, which `create_dn` keeps, enter as rows of weight 0.

### `dn_wkmeans`

**Description**

Cluster nugget centers with weighted k-means and map the resulting labels back to original rows.

**Usage**

```text
dn_wkmeans(dn, k, *, centers=None, num_init=1, max_iter=10,
           seed=291102, n_jobs=1)
```

```r
# WCluster 1.3.0
DN.Wkmeans(datanugget, k, cl.centers = NULL, num.init = 1, max.iterations = 10, seed = 291102)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `dn` | `datanugget` | The nugget set. |
| `k` | `k` | As in [`wkmeans`](#wkmeans). |
| `centers` | `cl.centers` | Optional `(k, p)` initial centers, where R takes `p` by `k`; pass R's matrix as `.T`. As in [`wkmeans`](#wkmeans), a `(p, k)` array is read as `(k, p)` without an error when `p` equals `k`. |
| `num_init` | `num.init` | As in [`wkmeans`](#wkmeans). |
| `max_iter` | `max.iterations` | As in [`wkmeans`](#wkmeans). |
| `seed` | `seed` | As in [`wkmeans`](#wkmeans). |
| `n_jobs` | Python only | As in [`wkmeans`](#wkmeans); it does not change the result. |

**Details**

This wrapper rejects one-feature nuggets with `ValueError`, preserving the R wrapper's behavior: R's `[, 2:2]` drops the table to a vector, and R stops with "argument is of length zero". For one feature, call `wkmeans(dn.centers, k, w=dn.weights, ...)` and map the returned labels through `dn.assignments`. Empty nuggets are clustered as zero-weight rows.

**Value**

A [`DnWkmeansResult`](#clustering-results): k-means fields for the nuggets, plus `row_labels (n,)` for the original rows. R's `"Cluster Assignments for data nuggets"` and `"Cluster Assignments for original dataset"` become `labels` and `row_labels`, both R's minus one, and the other components map as under [`wkmeans`](#wkmeans).

**Examples**

```python
nugget_km = dnlib.dn_wkmeans(dn, 3, num_init=3, seed=42)
np.testing.assert_array_equal(
    nugget_km.row_labels, nugget_km.labels[dn.assignments],
)
```

**See also:** [`wkmeans`](#wkmeans), [`dn_cluster_predict`](#dn_cluster_predict).

**References:** [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

### `dn_whclust`

**Description**

Build a weighted Ward hierarchy from nugget centers and their counts.

**Usage**

```text
dn_whclust(dn)
```

```r
# WCluster 1.3.0
DN.Whclust(datanugget)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `dn` | `datanugget` | A `DataNuggets` object with at least three nuggets; fewer raise `ValueError`. |

**Details**

It runs `whclust(dn.centers, w=dn.weights)`. One-feature nuggets are supported, as in R. Heights and weight-related limitations are the same as for [`whclust`](#whclust). A nugget of weight 0 makes every height NaN until it merges, and [`dendrogram`](#dendrogram) then raises `ValueError`.

**Value**

A [`WhclustResult`](#clustering-results) whose leaves are nuggets. R's `hclust` components map as under [`whclust`](#whclust).

**Examples**

```python
nugget_tree = dnlib.dn_whclust(dn)
labels_for_nuggets = dnlib.cut_tree(nugget_tree, k=3)
labels_for_rows = labels_for_nuggets[dn.assignments]
```

**See also:** [`whclust`](#whclust), [`cut_tree`](#cut_tree), [`dendrogram`](#dendrogram).

**References:** [Ward (1963)](#references), [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

### `dn_wpca`

**Description**

Compute weighted PCA of nugget centers using nugget counts as row weights.

**Usage**

```text
dn_wpca(dn, wcol=None, corr=False)
```

```r
# WCluster 1.3.0
DN.Wpca(datanugget,wcol = NULL, corr = FALSE)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `dn` | `datanugget` | The nugget set: `dn.centers` are the rows and `dn.weights` the row weights. |
| `wcol` | `wcol` | As in [`wpca`](#wpca). `None` means ones, as R's `NULL` does. |
| `corr` | `corr` | As in [`wpca`](#wpca); Python requires a bool. |

**Details**

For one-feature nuggets, the default `wcol=None` raises `ValueError`, where R's `rep(1, ncol(vector))` stops with "invalid 'times' argument". Pass `wcol=[1]` explicitly or call `wpca(dn.centers, wrow=dn.weights)`. Because nugget weights act as counts, the covariance divides by `sum(dn.weights) - 1`, which is `n - 1` for nuggets from `create_dn` or `refine_dn`. Empty nuggets are allowed, since row weights only have to be nonnegative.

**Value**

A [`WpcaResult`](#wpcaresult) with one score row per nugget. R's `DN.Wpca` returns `Wpca`'s list, which maps as under [`wpca`](#wpca).

**Examples**

```python
nugget_pca = dnlib.dn_wpca(dn)
all_row_scores = (X - nugget_pca.center) @ nugget_pca.rotation
```

**See also:** [`wpca`](#wpca), [`scores`](#scores).

**References:** [Amaratunga and Cabrera (2009)](#references), [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

### `dn_cluster_predict`

**Description**

Assign new rows or nugget centers to clusters fitted on a nugget set.

**Usage**

```text
dn_cluster_predict(dn, cl, newx)
```

```r
# WCluster 1.3.0
DNcluster.predict(datanugget, cl, newx)
```

**Arguments**

| Python | R | Meaning |
| --- | --- | --- |
| `dn` | `datanugget` | Training nuggets: `dn.centers` with `w=dn.weights`. |
| `cl` | `cl` | One cluster label per training nugget, with [`wmean`](#wmean)'s label rules. Another length raises `ValueError` with R's message and the two counts. |
| `newx` | `newx` | Numeric `(n_new, p)` matrix or data frame, or another `DataNuggets` object. R stops on a matrix ("newdata should be either a data frame or a datanugget object."); Python takes one, as `cluster_predict` does. A `DataNuggets` `newx`'s weights are not used as weights. |

**Details**

Prefer passing `newx.centers` explicitly for another nugget set, with matching features and preprocessing. The `DataNuggets` input path preserves an R table-slicing quirk: it takes the first p columns of `newx`'s `[centers, weights, scales]` table, with p the number of variables in `dn`. That path raises when `dn` has one variable, or when `newx` has fewer than p - 2 variables. A `newx` with fewer variables than `dn` otherwise runs and reads its weights (and scales) as features. An explicit `newx.centers` matrix avoids both.

**Value**

One integer per new row or new nugget: its position in sorted training-label order, as in [`cluster_predict`](#cluster_predict). R's `DNcluster.predict` returns the same positions 1-based.

**Examples**

```python
predicted_positions = dnlib.dn_cluster_predict(dn, nugget_labels, X[:5])
predicted_clusters = np.unique(nugget_labels)[predicted_positions]
```

**See also:** [`cluster_predict`](#cluster_predict), [R compatibility](compatibility.md).

**References:** [Cherasia et al. (2023)](#references), [Beavers et al. (2025)](#references).

## Result objects

Core result containers are frozen dataclasses. Array fields are independent, read-only copies; use `.copy()` before editing them. Scalar totals are zero-dimensional NumPy arrays, convertible with `float(result.total)` or `float(result.total_wwcss)`.

### `DataNuggets`

Returned by nugget creation and refinement.

| Field or method | Shape / result | Meaning |
| --- | --- | --- |
| `centers` | `(m, p)` | One representative per nugget. |
| `weights` | `(m,)` | Number of original rows assigned to each nugget. |
| `scales` | `(m,)` | Average within-nugget sample variance across features, with the creation/refinement edge cases described above. |
| `assignments` | `(n,)` | Original row to 0-based nugget ID. |
| `n_nuggets` | `int` | Number of nuggets. |
| `to_frame()` | pandas DataFrame | Columns `Data Nugget`, `Center1` through `Centerp`, `Weight`, `Scale`. Requires the `frame` extra. Nugget IDs start at 0. |

### `CentersResult`

| Field | Shape | Meaning |
| --- | --- | --- |
| `centers` | `(m, p)` | Selected input rows. |
| `kept` | `(m,)` | Their 0-based positions in the original sample. |

### `WwcssResult`

| Field | Shape | Meaning |
| --- | --- | --- |
| `per_cluster` | `(k,)` | Within-cluster sums in sorted label order. |
| `total` | `()` | Their total; NaN if a cluster has undefined variation. |

### `WpcaResult`

| Field | Shape | Meaning |
| --- | --- | --- |
| `sdev` | `(p,)` | Square roots of eigenvalues, largest first. |
| `rotation` | `(p, p)` | Principal component directions, one per column. |
| `x` | `(n, p)` | Input-row scores using the WCluster convention under [`wpca`](#wpca). |
| `center` | `(p,)` | Weighted feature means. |
| `scale` | `(p,)` | Covariance standard deviations when `corr=True`; ones otherwise. |
| `wrow` | `(n,)` | Input row weights. |
| `wcol` | `(p,)` | Column weights normalized to sum to `p`. |

For `dn_wpca`, replace `n` with the number of nuggets.

### Clustering results

**`WkmeansResult`**

| Field | Shape | Meaning |
| --- | --- | --- |
| `labels` | `(n,)` | Cluster label per input row; gaps are possible. |
| `centers` | `(q, p)` | Weighted means of the `q` labels present, in sorted label order. |
| `wwcss` | `(q,)` | Within-cluster sums in that same order. |
| `total_wwcss` | `()` | Total within-cluster sum of squares. |

**`DnWkmeansResult`** has the same fields for the nuggets, plus `row_labels (n,)` for the original observations.

**`WhclustResult`**

| Field or method | Shape | Meaning |
| --- | --- | --- |
| `merge` | `(n - 1, 2)` | SciPy-style IDs: leaves `0` through `n - 1`; merge step `s` creates node `n + s`. |
| `height` | `(n - 1,)` | Total within-cluster sum of squares after each merge. |
| `order` | `(n,)` | Leaf order for a dendrogram. |
| `to_linkage()` | `(n - 1, 4)` | Merge IDs, unchanged heights, and cluster sizes in SciPy linkage layout. |

Linkage cluster sizes count input rows or nuggets, not the total observation weights. `to_linkage()` rejects self-merges; negative and NaN heights remain unchanged and may be unsuitable for downstream SciPy functions. Signed-weight degeneracies can also make `order` repeat or omit leaves.

## Warnings

### `DataNuggetWarning`

`DataNuggetWarning` is a subclass of `UserWarning`, exported from `datanuggets`. The package issues it with `warnings.warn` in these cases:

- `create_dn_centers` issues "DN.num is greater than or equal to the number of rows. Returning original data" when `dn_num >= n`.
- `create_dn` issues the same message once for each stage-1 chunk with no more rows than its share of `dn_num1`. It issues it again when the pooled centers number no more than `dn_num2`.
- `refine_dn` issues "Quick-TRANSfer stage steps exceeded maximum (= N)" when the two-means run of a split uses up the N steps of its quick-transfer stage, 50 per row being split. It comes from every split attempted, kept or not. A run with two clusters stops after its first quick-transfer stage, so R's warning "did not converge in 1000 iterations" cannot occur.
- `arithmetic_complexity` issues "NaNs produced" when the count it takes the base-10 logarithm of is negative.
- `wpca` and `dn_wpca` issue "NaNs produced" once per call when an eigenvalue is below zero. `datanuggets.projection.pca_view` calls `wpca`, so it can issue it too. With row weights that sum to less than 1, `pca_view` can also issue NumPy's `RuntimeWarning` "invalid value encountered in divide", which a filter on `DataNuggetWarning` doesn't catch.
- `RDraws(seed)` issues "NAs introduced by coercion to integer range" when a seed that is not NaN does not truncate to an integer strictly between -2^31 and 2^31. Its `ValueError`, "supplied seed is not a valid integer", follows.

`refine_dn` collects the messages of worker processes and issues them in the main process. Python's warnings filters therefore apply to every case at any `n_jobs`. `warnings.simplefilter("error", DataNuggetWarning)` turns each one into an exception, and `warnings.simplefilter("ignore", DataNuggetWarning)` silences them.

## Helper modules

These modules need no matplotlib or pandas to import. t-SNE needs scikit-learn, which the `tsne` extra installs.

### `datanuggets.colors`

```text
TAB10                                   # matplotlib's ten default colours, lowercase "#rrggbb"
PARENT_K                                # 8
cluster_colors(k, parents=None) -> list[str]
block_parents(k) -> list[int]
tree_parents(tree, k) -> list[int]
```

`cluster_colors(k)` gives cluster `c` (from 0) its colour. Up to `PARENT_K` clusters it is `TAB10[:k]`, and `parents` is ignored. With more, `parents` gives each cluster's family, 0 to 7, and `None` means `block_parents(k)`. A `parents` of another length raises `ValueError("parents must give one family per cluster, ...")`, and a family outside 0 to 7 raises `ValueError("parents must be family numbers from 0 to 7")`. A family's first cluster takes its `TAB10` colour, and its others take colours near that one, picked one at a time from a pool. The pool holds the sRGB colours whose channels each take one of 24 evenly spaced 8-bit values. While the pool and the eight family colours together hold fewer than k colours, the number of levels doubles, to at most 256. The pool keeps only colours with WCAG contrast at least 2.5 against both a white and a dark (`#111318`) background, and it leaves out the eight family colours. Each pool colour belongs to the region of the family whose colour, among `TAB10[:PARENT_K]`, is nearest by CIEDE2000. Each pick takes the free colour in the family's region that is farthest by CIEDE2000 from the colours already given, its siblings' counting at 5 and other families' at 3 (it maximises the smaller of the two distances divided by those). A family whose region has no free colour left takes the free colour farthest from every colour already given. So all k colours differ, siblings stay apart, and the same `parents` give the same colours on every platform. At k = 100, on the nuggets of R's `quakes` data and on the California housing data, the tests check that siblings stay at least 5 apart and any two clusters at least 3 (CIEDE2000). Each pick scores every colour in the pool, so the time grows faster than k. The pool also grows in steps of about 8 times: 6,636 colours up to 6,644 clusters, 53,595 up to 53,603, and more beyond that. One laptop took 0.1 s for 100 clusters, 1 s for 1,000, 3.5 s for 3,000 and 11 s for 6,644, then 2 minutes for 6,645. `block_parents(k)` makes families of neighbouring cluster numbers, `c * 8 // k`; `tree_parents(tree, k)` takes each cluster of `cut_tree(tree, k=k)` to the cluster of the same tree cut into 8 that its members fall in, so related clusters share a hue. Both give `range(k)` for k up to 8.

```python
from datanuggets.colors import cluster_colors, tree_parents

colors = cluster_colors(12, tree_parents(tree, 12))
```

### `datanuggets.filters`

```text
RowFilter(column, op, value)
parse_filter(text) -> RowFilter
format_filter(f) -> str
filter_mask(get_column, filters, n) -> np.ndarray
RowFilter.numeric -> bool                 # every value is a number
OPS = ("<", "<=", ">", ">=", "==", "!=", "between", "in")
NUMERIC_OPS = ("<", "<=", ">", ">=", "between")   # take numbers only
TEXT_FORBIDDEN = (",", "[", "]")          # characters a text value can't hold
Value = float | str | tuple[float, float] | tuple[float | str, ...]
```

`parse_filter` reads `COLUMN OP VALUE` as [row filters](#row-filters) describes, and raises `ValueError` naming the problem. `RowFilter` is frozen and checks its own fields, so one built directly is valid too. It keeps a number as a float. For `==`, `!=` and `in` it also keeps text that reads as a finite number as that number, the way a spec is read. `RowFilter("x", "==", "3")` therefore equals `RowFilter("x", "==", 3.0)`. The operators in `NUMERIC_OPS` take numbers only. A text value raises `ValueError` for them even when it reads as a number. `format_filter` writes the canonical spec, the column bracketed when it holds whitespace, starts with `[` or is an operator such as `in`, and `parse_filter(format_filter(f)) == f` for every `RowFilter`; a bracketed column ends at the last `]`, since a value can't hold one. `filter_mask(get_column, filters, n)` returns the n rows every filter keeps as booleans; `get_column(name)` gives a column's n cells, as numbers or text. A number filter compares as float64 and raises `ValueError` naming the column and 0-based data row of a cell that isn't a finite number; text compares exactly. An `in` filter that holds both numbers and text matches a cell by its exact text, or by its value when the cell reads as a finite number.

```python
import numpy as np
from datanuggets.filters import filter_mask, parse_filter

table = {"species": np.array(["setosa", "virginica"]), "petal_length": np.array([1.4, 5.1])}
keep = filter_mask(table.__getitem__, [parse_filter("petal_length between 1,1.5")], 2)
```

#### Row filters

Write a filter as `COLUMN OP VALUE`, with a space on each side of the operator:

| Operator | Value | Keeps rows whose cell is |
| --- | --- | --- |
| `<` `<=` `>` `>=` | a number | less than, at most, greater than, or at least the number |
| `==` `!=` | a number or text | equal to, or different from, the value |
| `between` | `lo,hi` with lo <= hi | from lo to hi, both ends included |
| `in` | one or more values separated by commas | equal to any of them |

A value that reads as a finite number is compared as a number. Any other value is compared as exact text, and only `==`, `!=` and `in` take text. Text can't contain `,`, `[` or `]`. A bare column name is everything before the first operator that has whitespace or an end of the filter on each side, stripped. Put a column name in square brackets when it is an operator, starts or ends with an operator next to whitespace, holds an operator with whitespace on each side, starts with `[`, or starts or ends with whitespace: `[size in cm] between 1,1.5`.

### `datanuggets.projection`

```text
PcaView(center, rotation, scores, sdev, variance_ratio)
pca_view(centers, weights, *, fix_signs) -> PcaView
view_of(result, *, fix_signs) -> PcaView
component_signs(rotation) -> np.ndarray
project(x, view, n_components=None) -> np.ndarray
tsne_2d(centers, *, perplexity=30.0, seed=0) -> np.ndarray
tsne_3d(centers, *, perplexity=30.0, seed=0) -> np.ndarray | None
TSNE_MISSING                              # the ImportError message without scikit-learn
```

`pca_view` is `wpca(centers, wrow=weights)` as a `PcaView`. With `fix_signs=True` each component is flipped so its largest absolute loading is positive, as `plot.scores` draws it; `fix_signs=False` keeps LAPACK's signs, as `wpca` returns them. `variance_ratio` is each component's share of the variance, a NaN `sdev` counted as 0. `project(x, view, m)` is `(x - center) @ rotation[:, :m]`, every component by default, for rows on the scale the centers were; the centers project onto `view.scores`. `tsne_2d` is a 2-D t-SNE of the centers, its perplexity capped at (n - 1) / 3; without scikit-learn it raises `ImportError('t-SNE needs scikit-learn: pip install "datanuggets[tsne]"')`. `tsne_3d` is a 3-D t-SNE of the same centers with the same arguments, perplexity cap and seed. It is a fit of its own, not `tsne_2d` plus a third axis. It returns `None` for centers of fewer than three columns or rows, whose PCA start can't give three components, and raises the same `ImportError` without scikit-learn.

## Plotting

Install the `plot` extra and import `datanuggets.plot`. Drawing functions accept an optional Matplotlib `ax` and return the axes; without `ax`, they use the current axes. They do not automatically display or save a figure.

`dims` always selects two distinct **0-based** feature or component positions. Plot labels must have integer dtype. `colors` accepts a nonempty sequence of Matplotlib colors. The module also exports palette constants `OKABE_ITO` and `R_PALETTE`, plus the style dictionary `PAPER_RC`.

Run this import setup before the plotting examples, using `dn`, `tree`, `nugget_labels`, and `pca` from [example data](#example-data):

```python
import matplotlib.pyplot as plt
from datanuggets import plot
```

### `clusters`

**Description**

Draw observations in two selected features, colored by cluster membership.

**Usage**

```text
clusters(x, labels, *, w=None, centers=None, dims=(0, 1),
         size="area", colors=None, ax=None)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Numeric `(n, p)` data. |
| `labels` | Integer cluster labels, one per row. |
| `w` | Optional point weights. For `"area"` and `"sqrt"` they must be finite and nonnegative, with at least one positive. |
| `centers` | Optional cluster centers, one row per sorted distinct label. |
| `dims` | Two feature positions to display. |
| `size` | `"area"` scales marker area by weight; `"sqrt"` by the square root of weight, so heavy points dominate less; `"log"` uses the R-style logarithmic size rule. |
| `colors` | Optional sequence of cluster colors. |
| `ax` | Optional Matplotlib axes. |

**Details**

Without weights, `size="area"` and `size="sqrt"` use fixed-size markers. `size="log"` requires weights and makes points with weight at most 1 invisible. Supplied centers are shown as stars.

**Value**

The Matplotlib axes containing the scatter plot.

**Examples**

```python
fig, ax = plt.subplots()
plot.clusters(
    dn.centers, nugget_labels, w=dn.weights,
    centers=dnlib.wmean(dn.centers, nugget_labels, w=dn.weights),
    ax=ax,
)
```

**See also:** [`nuggets`](#nuggets), [`wkmeans`](#wkmeans).

### `nuggets`

**Description**

Draw nugget centers in two selected features, using weight shading or cluster colors.

**Usage**

```text
nuggets(dn, *, labels=None, dims=(0, 1), colors=None, ax=None)
```

**Arguments**

`dn` is the nugget set. Optional integer `labels` provide one cluster per nugget. `dims` chooses two features, `colors` supplies cluster colors, and `ax` selects the axes.

**Details**

Markers have a fixed size. Without labels, green intensity represents weight, with heavier nuggets shown in lighter shades. With labels, colors represent clusters.

**Value**

The Matplotlib axes.

**Examples**

```python
fig, ax = plt.subplots()
plot.nuggets(dn, labels=nugget_labels, ax=ax)
```

**See also:** [`clusters`](#clusters), [`create_dn`](#create_dn).

### `dendrogram`

**Description**

Draw a weighted hierarchy using its stored cumulative within-cluster-variation heights.

**Usage**

```text
dendrogram(tree, *, labels=None, hang=0.1, ax=None)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `tree` | A `WhclustResult`. |
| `labels` | Optional leaf labels; `None` uses 0-based row IDs, and `False` hides labels. |
| `hang` | How far leaves hang below their merges, as a fraction of the last height minus the first. When `hang` times that difference is negative, every leaf sits at 0. |
| `ax` | Optional Matplotlib axes. |

**Details**

Invalid trees, self-merges, and nonfinite heights cannot be drawn. Interpret the vertical axis as the height stored by `whclust`, not as a standard SciPy Ward distance.

**Value**

The Matplotlib axes.

**Examples**

```python
fig, ax = plt.subplots()
plot.dendrogram(tree, labels=False, ax=ax)
```

**See also:** [`whclust`](#whclust), [`cut_tree`](#cut_tree).

### `scores`

**Description**

Draw weighted-PCA score rows in a pair of components.

**Usage**

```text
scores(wpca, *, w=None, labels=None, rows=None, dims=(0, 1),
       colors=None, ax=None)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `wpca` | A `WpcaResult`. |
| `w` | Optional finite nonnegative weights for score-row marker areas, with at least one positive value. |
| `labels` | Optional integer cluster labels for the score rows. |
| `rows` | Optional original-feature data to project as a gray background. |
| `dims` | Two component positions. |
| `colors`, `ax` | Optional palette and Matplotlib axes. |

**Details**

Background rows must match the PCA's feature count and preprocessing. At most 50,000 background rows are displayed, selected at regular strides. Component signs are normalized for display so the largest absolute loading is positive; the PCA result itself is unchanged.

**Value**

The Matplotlib axes.

**Examples**

```python
fig, ax = plt.subplots()
plot.scores(pca, w=dn.weights, labels=nugget_labels, rows=X, ax=ax)
```

**See also:** [`wpca`](#wpca), [`dn_wpca`](#dn_wpca), [`loadings`](#loadings).

### `scree`

**Description**

Draw the percentage of variance associated with each PCA component.

**Usage**

```text
scree(wpca, *, ax=None)
```

**Arguments**

`wpca` is a `WpcaResult`; `ax` optionally supplies the plotting axes.

**Details**

Percentages are calculated from squared `sdev` values. NaN standard deviations are treated as zero for the display. Inspect PCA warnings and numerical results before interpreting a plot; a chart does not resolve undefined or zero total variance.

**Value**

The Matplotlib axes.

**Examples**

```python
fig, ax = plt.subplots()
plot.scree(pca, ax=ax)
```

**See also:** [`wpca`](#wpca), [`scores`](#scores).

### `loadings`

**Description**

Draw feature-loading arrows for two principal components.

**Usage**

```text
loadings(wpca, *, names=None, dims=(0, 1), ax=None)
```

**Arguments**

`wpca` is a `WpcaResult`. `names` optionally supplies one label per feature; the default is its 0-based column number. `dims` chooses two components, and `ax` supplies axes.

**Details**

Component signs are normalized using the same display convention as `scores`. The underlying rotation matrix is unchanged.

**Value**

The Matplotlib axes.

**Examples**

```python
fig, ax = plt.subplots()
plot.loadings(pca, ax=ax)
```

**See also:** [`wpca`](#wpca), [`scores`](#scores).

### `style`

**Description**

Temporarily apply the package's figure style.

**Usage**

```text
style()
```

**Arguments**

None.

**Details**

Use a `with` block. The context applies `PAPER_RC` and restores the previous Matplotlib settings when it exits. The PDF font type and the dpi of `savefig` are read when a figure is saved. Save inside the `with` block to embed TrueType fonts. A figure saved after it gets Matplotlib's own font type, Type 3 by default, even through `save`.

**Value**

A Matplotlib style context manager.

**Examples**

```python
with plot.style():
    styled_fig, styled_ax = plt.subplots()
    plot.scores(pca, w=dn.weights, ax=styled_ax)
```

**See also:** [`save`](#save).

### `save`

**Description**

Save a figure at a specified publication width.

**Usage**

```text
save(fig, path, *, width="single", height=None)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `fig` | Matplotlib figure to export. |
| `path` | Output path including a supported image extension, such as `.png` or `.pdf`. |
| `width` | `"single"` for 3.5 inches, `"double"` for 7 inches, or a positive numeric width in inches. |
| `height` | Positive height in inches; defaults to `0.75 * width`. |

**Details**

The extension selects the output format. Export uses 300 dpi and preserves the chosen figure dimensions without a tight bounding-box crop. The helper changes the figure's size, does not create parent directories, and overwrites an existing file at the supplied path.

**Value**

A `pathlib.Path` for the saved file.

**Examples**

This example exports into a temporary directory. Replace that directory with your output folder to retain the figure.

```python
from tempfile import TemporaryDirectory

with TemporaryDirectory() as output_dir:
    saved = plot.save(styled_fig, Path(output_dir) / "quakes-pca.png")
    assert saved.is_file()
```

**See also:** [`style`](#style), [installation](installation.md).

## References

These are the works that the R help pages of the ported functions cite. Each core entry above names the ones its R help page cites.

Amaratunga, D., and Cabrera, J. *Exploration and Analysis of DNA Microarray and Protein Array Data*. John Wiley & Sons, 2009 (Vol. 605), as the WCluster manual cites it. DOI: `10.1002/9780470317129`.

Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`.

Becker, R. A., Chambers, J. M., and Wilks, A. R. *The New S Language*. Chapman and Hall/CRC, London, 1988. ISBN 053409192X.

Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023. Published online in 2022. DOI: `10.1007/978-3-031-22687-8_20`.

Ward, J. H., Jr. "Hierarchical Grouping to Optimize an Objective Function." Journal of the American Statistical Association 58(301), 236-244, 1963. DOI: `10.1080/01621459.1963.10500845`.
