# API reference

This manual documents the public Python API for **datanuggets 0.1.0a1**. For installation and a first analysis, start with the [installation guide](installation.md) and [quickstart](quickstart.md). This page gives function-level arguments, return values, examples, and compatibility notes.

The 18 core functions are exported from `datanuggets`. Ten flow helpers live in `datanuggets.flow`, the `datanuggets-flow` pipeline's library in four of its submodules, and nine plotting helpers in `datanuggets.plot`.

| Task | Functions |
| --- | --- |
| Create and refine nuggets | [`create_dn`](#create_dn), [`create_dn_centers`](#create_dn_centers), [`refine_dn`](#refine_dn), [`get_dn_means`](#get_dn_means) |
| Estimate arithmetic work | [`arithmetic_complexity`](#arithmetic_complexity) |
| Calculate weighted summaries | [`wmean`](#wmean), [`wss`](#wss), [`wwcss`](#wwcss), [`distw`](#distw) |
| Cluster and predict | [`wkmeans`](#wkmeans), [`whclust`](#whclust), [`cut_tree`](#cut_tree), [`cluster_predict`](#cluster_predict) |
| Run weighted PCA | [`wpca`](#wpca) |
| Analyze nuggets | [`dn_wkmeans`](#dn_wkmeans), [`dn_whclust`](#dn_whclust), [`dn_wpca`](#dn_wpca), [`dn_cluster_predict`](#dn_cluster_predict) |
| Work with flow populations | [Flow utilities](#flow-utilities) |
| Gate and cluster FCS files from Python | [The flow pipeline library](#the-flow-pipeline-library) |
| Draw and export figures | [Plotting](#plotting) |
| Inspect returned objects | [Result objects](#result-objects) |

## Conventions

- **Shapes:** `n` means input rows, `p` features, `m` nuggets, and `k` clusters. Prepare numeric features before calling the package; it does not select columns, fill missing values, or standardize features.
- **Inputs:** core functions that read numeric input data reject NaN and infinity. Nugget creation and refinement require a two-dimensional numeric array or data frame. Weighted analysis functions also accept a vector as one feature and Boolean data as 0/1. Prediction data must be two-dimensional.
- **Weights:** `None` means ones. Supply one finite numeric weight per row. Positive weights are the usual choice. Several WCluster ports accept zero or signed weights for R compatibility; these can produce undefined means, NaNs, or degenerate clusters. PCA has stricter rules.
- **Labels:** generated nugget and cluster IDs are **0-based**. Input `cl` labels must be finite numeric whole numbers, one per row; they need not be consecutive. Cluster summaries follow sorted distinct label order. Plotting requires labels with integer dtype. Flow grid coordinates use a separate **1-based** convention.
- **Randomness:** `seed` accepts an integer, a NumPy `Generator`, or an `RDraws` R-compatible random-number generator. Integer seeds use NumPy: the same seed number in R and Python does not imply the same random draws. See [reproducibility](reproducibility.md) and [R-compatible randomness](compatibility.md#rdraws).
- **Parallelism:** `n_jobs=1` runs serially; positive integers select worker counts and `-1` uses all cores. `progress=True` requires the `progress` extra. Warnings use `DataNuggetWarning`; informational messages use the `datanuggets` logger.
- **Results:** core functions return arrays, scalars, or named result objects, rather than R lists. Result-array fields are read-only copies. See [result objects](#result-objects).

### R-to-Python function map

The entries below follow the familiar Description, Usage, Arguments, Details, Value, Examples, and See also structure. Python names use `snake_case`, and Python row labels and indices start at zero.

| Python | R counterpart |
| --- | --- |
| `create_dn`, `create_dn_centers` | `create.DN`, `create.DNcenters` |
| `refine_dn`, `get_dn_means`, `arithmetic_complexity` | `refine.DN`, `getDN.means`, `AC` |
| `wmean`, `wss`, `wwcss`, `distw` | `wmean`, `wss`, `wwcss`, `distw` |
| `wkmeans`, `whclust`, `wpca` | `Wkmeans`, `Whclust`, `Wpca` |
| `cluster_predict` | `cluster.predict` |
| `dn_wkmeans`, `dn_whclust`, `dn_wpca`, `dn_cluster_predict` | `DN.Wkmeans`, `DN.Whclust`, `DN.Wpca`, `DNcluster.predict` |
| `cut_tree` | Python helper for cutting a hierarchy, serving the role of `stats::cutree` |

**`optimal.DN` is not ported.** Flow helpers and Python plotting helpers are additional APIs, not direct translations of every R plotting method.

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

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Numeric `(n, p)` data with at least two rows. |
| `center_method` | `"mean"` uses the mean of assigned rows; `"original"` retains the selected input center; `"random"` chooses an assigned input row. |
| `r` | Initial chunk-size control, within `[100, 10000]`. The algorithm uses `ceil(n / r)` chunks. |
| `delete_percent` | Fraction removed per thinning round; strictly between 0 and 1. |
| `dn_num1` | Positive integer target for centers pooled from the first stage. |
| `dn_num2` | Positive integer target for the final stage; cannot exceed `dn_num1`. |
| `dist_metric` | `"euclidean"` or `"manhattan"`, used for center selection and assignment. |
| `seed` | Random seed or generator; see [conventions](#conventions). |
| `n_jobs` | Workers for first-stage chunks; default serial. |
| `progress` | Show first-stage progress when `True`. Requires the `progress` extra. |

**Details**

The algorithm shuffles rows into chunks, thins each chunk, then thins the pooled centers. It assigns the original rows to the retained centers before applying `center_method`. Changing centers to means or sampled rows does not trigger another assignment step.

The requested counts are targets: chunk allocation and small inputs can change the available count. A selection stage that already has no more than its requested count returns its input and warns. Empty nuggets are retained. Scale is the average sample variance across features, with a small positive floor for single-row and empty nuggets.

Center selection builds pairwise distance matrices. Large intermediate center counts can use substantial memory; see [limitations](limitations.md).

**Value**

A [`DataNuggets`](#datanuggets) object with `centers (m, p)`, `weights (m,)`, `scales (m,)`, and `assignments (n,)`. Weights count assigned rows; assignments are 0-based nugget IDs.

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

### `create_dn_centers`

**Description**

Select representative rows from a sample by repeatedly removing members of close pairs.

**Usage**

```text
create_dn_centers(rs, dn_num, *, delete_percent=0.1,
                  dist_metric="euclidean")
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `rs` | Numeric `(n, p)` sample. |
| `dn_num` | Positive integer target number of retained rows. |
| `delete_percent` | Fraction removed per round, strictly between 0 and 1. |
| `dist_metric` | `"euclidean"` or `"manhattan"`. |

**Details**

Selection is deterministic for a fixed sample and row order. It returns selected input rows, without computing weights, scales, or assignments. If `dn_num >= n`, all rows are returned with a warning. Pairwise distances require quadratic memory in sample size.

**Value**

A [`CentersResult`](#centersresult): `centers` has shape `(m, p)` and `kept` has shape `(m,)`. Each entry in `kept` is a **0-based row position in `rs`**.

**Examples**

```python
sample = X[:100]
selected = dnlib.create_dn_centers(sample, dn_num=20)
np.testing.assert_array_equal(selected.centers, sample[selected.kept])
```

**See also:** [`create_dn`](#create_dn).

### `refine_dn`

**Description**

Split overly dispersed nuggets, optionally followed by a pass that attempts to improve elongated nuggets.

**Usage**

```text
refine_dn(x, dn, ev_tol=0.9, max_splits=5, min_nugget_size=2,
          shape_split=False, min_shape_size=10, delta=2,
          nstart=25, seed=291102, n_jobs=1, progress=False)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Original numeric data, in exactly the row order used to create `dn`. |
| `dn` | A `DataNuggets` object matching the rows and features of `x`. |
| `ev_tol` | Quantile in `(0, 1)` used to set the first-eigenvalue splitting threshold before the first round. |
| `max_splits` | Maximum rounds in the eigenvalue pass, and again in the optional shape pass. Use a nonnegative integer. |
| `min_nugget_size` | Minimum child size for splitting; use an integer of at least 2. |
| `shape_split` | Enable the shape pass. It has no effect with one feature. |
| `min_shape_size` | Minimum nugget size considered for shape splitting; at least 10. |
| `delta` | Eigenvalue-ratio threshold for the shape pass; greater than 1. |
| `nstart` | Random starts for each two-means split; use a positive integer. |
| `seed` | Random seed or generator. |
| `n_jobs` | Worker count. Use serial execution with custom or replayed random draws. |
| `progress` | Display refinement progress; requires the `progress` extra. |

**Details**

The eigenvalue threshold is calculated once from the initial nuggets. Eligible nuggets are split by two-means; the optional shape pass uses eigenvalue ratios and accepts splits that improve the shape criterion.

Refinement uses `dn.assignments` to recompute centers, weights, and scales from `x`. It removes empty nuggets and renumbers survivors. Single-row nuggets receive scale 0. Even `max_splits=0` performs this resummarization.

A small numerical change near a splitting threshold can change later splits. See [refinement compatibility](compatibility.md#refine_dn) before expecting an identical hierarchy across numerical libraries.

**Value**

A new [`DataNuggets`](#datanuggets) with assignments for all original rows. The input object is unchanged.

**Examples**

```python
refined = dnlib.refine_dn(X, dn, max_splits=1, nstart=5, seed=42)
assert refined.weights.sum() == len(X)
assert refined.assignments.shape == (len(X),)
```

**See also:** [`create_dn`](#create_dn), [`get_dn_means`](#get_dn_means).

### `get_dn_means`

**Description**

Calculate the mean of the original rows assigned to each nugget, regardless of the center method used to create it.

**Usage**

```text
get_dn_means(x, dn)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Original `(n, p)` data, with the same row order and feature count as `dn`. |
| `dn` | A `DataNuggets` object whose assignments identify each row's nugget. |

**Details**

This is useful when centers were retained input rows or sampled representatives. Empty nuggets produce a row of NaNs. The Python output remains two-dimensional for one feature.

**Value**

A float array of shape `(m, p)`. Row `i` contains the means for nugget `i`.

**Examples**

```python
means = dnlib.get_dn_means(X, dn)
np.testing.assert_allclose(means, dn.centers)  # This setup used center_method="mean".
```

**See also:** [`create_dn`](#create_dn), [`refine_dn`](#refine_dn).

### `arithmetic_complexity`

**Description**

Evaluate the port of R's `AC` arithmetic-work estimate for nugget creation.

**Usage**

```text
arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Two-dimensional data with at least one row; only its dimensions are used. |
| `r` | Chunk-size control within `[100, 10000]`. |
| `delete_percent` | Thinning fraction, strictly between 0 and 1. |
| `dn_num1`, `dn_num2` | First- and second-stage center targets, with `dn_num2 <= dn_num1`. Use positive integers. |

**Details**

The estimate follows the R counting formula and its rounding rules. It is not a runtime prediction, a memory estimate, or a measured benchmark. Inputs that would make the original counting loop fail to terminate raise `ValueError`.

**Value**

A scalar: the base-10 logarithm of the estimated arithmetic-operation count.

**Examples**

```python
log10_work = dnlib.arithmetic_complexity(
    X, r=300, delete_percent=0.1, dn_num1=200, dn_num2=40,
)
```

**See also:** [`create_dn`](#create_dn), [limitations](limitations.md).

## Weighted summaries

These functions accept an `(n, p)` matrix or a vector treated as one feature. For the examples, `dn.centers` supplies observations and `dn.weights` their counts.

### `wmean`

**Description**

Calculate a weighted feature mean for each cluster.

**Usage**

```text
wmean(x, cl, w=None)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Numeric observations. |
| `cl` | One numeric whole-number cluster label per row. |
| `w` | One row weight per observation; `None` uses ones. |

**Details**

Clusters are ordered by sorted distinct labels. A cluster whose weights sum to zero has an undefined mean; zero or cancelling weights can produce NaNs or infinities. Python returns one row per cluster; R's `wmean` uses the opposite orientation.

**Value**

An array of shape `(k, p)` in sorted label order.

**Examples**

```python
cluster_means = dnlib.wmean(dn.centers, nugget_labels, w=dn.weights)
cluster_ids = np.unique(nugget_labels)  # cluster_ids[i] names cluster_means[i].
```

**See also:** [`wss`](#wss), [`wwcss`](#wwcss), [`cluster_predict`](#cluster_predict).

### `wss`

**Description**

Measure weighted squared deviation from the overall weighted feature mean.

**Usage**

```text
wss(x, w=None)
```

**Arguments**

`x` is the numeric data; `w` contains one weight per row, or `None` for equal weights.

**Details**

The function sums squared Euclidean distances from the weighted mean, multiplied by row weights. A zero total weight gives NaN. With ordinary positive weights, the result measures the total variation represented by the supplied points.

**Value**

A floating-point scalar.

**Examples**

```python
total_variation = dnlib.wss(dn.centers, w=dn.weights)
```

**See also:** [`wmean`](#wmean), [`wwcss`](#wwcss).

### `wwcss`

**Description**

Measure weighted within-cluster variation around each cluster's weighted mean.

**Usage**

```text
wwcss(x, cl, w=None, group_sum=False)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Numeric observations. |
| `cl` | One cluster label per row. |
| `w` | Row weights, or `None` for ones. |
| `group_sum` | Return individual cluster sums as well as their total when `True`. |

**Details**

Per-cluster values follow sorted distinct label order. A cluster whose weights sum to zero contributes NaN, making the total NaN too.

**Value**

With `group_sum=False`, a floating-point total. With `True`, a [`WwcssResult`](#wwcssresult) containing `per_cluster (k,)` and `total ()`.

**Examples**

```python
within = dnlib.wwcss(
    dn.centers, nugget_labels, w=dn.weights, group_sum=True,
)
np.testing.assert_allclose(within.total, within.per_cluster.sum())
```

**See also:** [`wss`](#wss), [`wkmeans`](#wkmeans), [`distw`](#distw).

### `distw`

**Description**

Calculate the change in weighted within-cluster variation from merging each pair of clusters.

**Usage**

```text
distw(x, cl, w=None)
```

**Arguments**

`x` is the numeric data, `cl` supplies its labels, and `w` supplies row weights or `None` for ones. At least two distinct clusters are required.

**Details**

Each populated entry is the merged cluster's weighted sum of squares minus the two separate sums. This is a merge cost, not a complete symmetric distance matrix. Tiny negative values from rounding are retained.

**Value**

A `(k, k)` array in sorted label order. Values occupy the **strict lower triangle**; the diagonal and upper triangle are NaN.

**Examples**

```python
merge_costs = dnlib.distw(dn.centers, nugget_labels, w=dn.weights)
pair_costs = merge_costs[np.tril_indices(len(merge_costs), k=-1)]
```

**See also:** [`wwcss`](#wwcss), [`whclust`](#whclust).

## Clustering and prediction

### `wkmeans`

**Description**

Partition rows with WCluster's weighted k-means algorithm.

**Usage**

```text
wkmeans(x, k, *, w=None, centers=None, num_init=1, max_iter=10,
        seed=291102, n_jobs=1)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Numeric `(n, p)` data, or a vector for one feature. |
| `k` | Requested cluster count; use a positive integer. |
| `w` | Row weights, or `None` for ones. |
| `centers` | Optional `(k, p)` initial centers. Supplying centers uses one initialization. |
| `num_init` | Number of random initializations when centers are omitted; use a positive integer. |
| `max_iter` | Maximum outer optimization rounds; use a positive integer. |
| `seed` | Random seed or generator. |
| `n_jobs` | Workers for independent initializations. |

**Details**

The algorithm combines Lloyd assignment steps with single-point moves and selects the run with the smallest total within-cluster sum of squares. It is not an alias for scikit-learn k-means.

Automatic initialization requires at least `2 * k` rows and at least `k` distinct row/weight combinations. These conditions do not guarantee a successful initialization. Explicit centers must produce occupied initial clusters.

Final labels can contain gaps if a cluster becomes empty. Returned center rows follow sorted **present** labels, so do not assume `result.centers[result.labels]` is always valid. Use positive weights for ordinary analysis.

**Value**

A [`WkmeansResult`](#clustering-results) with row `labels`, occupied-cluster `centers`, per-cluster `wwcss`, and scalar `total_wwcss`.

**Examples**

```python
km = dnlib.wkmeans(
    dn.centers, 3, w=dn.weights, num_init=3, seed=42,
)
occupied_ids = np.unique(km.labels)
center_for_each_nugget = km.centers[np.searchsorted(occupied_ids, km.labels)]
```

**See also:** [`dn_wkmeans`](#dn_wkmeans), [`wwcss`](#wwcss), [`cluster_predict`](#cluster_predict).

### `whclust`

**Description**

Build a weighted Ward hierarchy by repeatedly merging the pair with the smallest increase in within-cluster variation.

**Usage**

```text
whclust(x, w=None)
```

**Arguments**

`x` is numeric data with at least three rows. `w` supplies one weight per row; `None` means ones.

**Details**

Stored heights are the **total weighted within-cluster sum of squares after each merge**. They are not SciPy's usual Ward distances. Memory grows quadratically with the number of input rows, which is one reason to cluster nugget representatives.

Positive weights are recommended. Zero weights can produce NaN heights; signed weights can yield degenerate trees, including self-merges.

**Value**

A [`WhclustResult`](#clustering-results) containing merge IDs, heights, and leaf order.

**Examples**

```python
weighted_tree = dnlib.whclust(dn.centers, w=dn.weights)
labels_from_tree = dnlib.cut_tree(weighted_tree, k=3)
```

**See also:** [`cut_tree`](#cut_tree), [`dn_whclust`](#dn_whclust), [`dendrogram`](#dendrogram).

### `cut_tree`

**Description**

Turn a weighted hierarchy into cluster labels using a cluster count or a height threshold.

**Usage**

```text
cut_tree(tree, k=None, h=None)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `tree` | A `WhclustResult`. |
| `k` | A single cluster count in `[1, n]`. Use an integer. |
| `h` | A single cut height, using the cumulative-variation units stored in `tree.height`. |

**Details**

Supply `k` or `h`; `k` takes precedence if both are given. Cutting by height requires nondecreasing heights. Generated labels are numbered by first appearance in the original row order, not dendrogram order.

**Value**

An integer array of shape `(n,)`. For a nugget tree, `n` here is the number of nuggets.

**Examples**

```python
labels_at_three = dnlib.cut_tree(tree, k=3)
original_row_labels = labels_at_three[dn.assignments]
assert original_row_labels.shape == (len(X),)
```

**See also:** [`whclust`](#whclust), [`dn_whclust`](#dn_whclust).

### `cluster_predict`

**Description**

Assign new observations to the nearest weighted mean of an existing set of clusters.

**Usage**

```text
cluster_predict(x, cl, newx, *, w=None)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Training observations. |
| `cl` | One cluster label per training row. |
| `newx` | Two-dimensional `(n_new, p)` data, with the same features and preprocessing as `x`. |
| `w` | Weights for training rows, or `None` for ones. |

**Details**

The function compares squared Euclidean distances to weighted cluster means. Ties choose the first mean in sorted label order. Clusters with undefined distances are skipped; if every distance for a new row is NaN, the call raises.

**Value**

An integer array of shape `(n_new,)` containing **positions in sorted cluster order, not original label values**. Recover original labels with `np.unique(cl)[positions]`.

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

## Weighted PCA

### `wpca`

**Description**

Compute WCluster's weighted principal components using row and optional feature weights.

**Usage**

```text
wpca(x, wrow=None, wcol=None, corr=False)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x` | Numeric `(n, p)` data, or a vector treated as one feature. |
| `wrow` | Nonnegative row weights. `None` means ones. These act as counts: covariance divides by `sum(wrow) - 1`. |
| `wcol` | One feature weight per column. `None` means ones. Weights are normalized to sum to `p`. |
| `corr` | Use a weighted correlation matrix for eigendecomposition when `True`. |

**Details**

Row weights cannot sum to 0 or exactly 1. For ordinary covariance analysis, use count-like weights whose sum exceeds 1. Probability weights normalized to 1 are unsuitable without rescaling.

For feature weights, use nonnegative values with a positive total. Mixed signs and a zero total are rejected. For `corr=True`, the row-weight sum must exceed 1 and every feature must have positive weighted variance.

The score convention is important: `result.x` equals `(x - result.center) @ result.rotation`, **even when `corr=True`**. Scores are not divided by `result.scale` or multiplied by weights. Project new observations using this same expression.

Component signs can differ between eigensolvers. Repeated eigenvalues allow different bases for the same subspace. Rank-deficient inputs can produce small negative eigenvalues, NaN standard deviations, and `DataNuggetWarning`; the implementation does not clip those eigenvalues to zero. See [numerical limitations](limitations.md).

**Value**

A [`WpcaResult`](#wpcaresult), including eigenvalue standard deviations `sdev`, directions `rotation`, scores `x`, and the centering and weighting information.

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

## Nugget analysis

These wrappers use `dn.centers` as observations and `dn.weights` as row weights. Each requires a `DataNuggets` object. Their clustering and PCA results describe the representatives; compression does not preserve all information in the original observations.

### `dn_wkmeans`

**Description**

Cluster nugget centers with weighted k-means and map the resulting labels back to original rows.

**Usage**

```text
dn_wkmeans(dn, k, *, centers=None, num_init=1, max_iter=10,
           seed=291102, n_jobs=1)
```

**Arguments**

`dn` is the nugget set. `k`, `centers`, `num_init`, `max_iter`, `seed`, and `n_jobs` have the meanings given under [`wkmeans`](#wkmeans).

**Details**

This wrapper rejects one-feature nuggets, preserving the R wrapper's behavior. For one feature, call `wkmeans(dn.centers, k, w=dn.weights, ...)` and map the returned labels through `dn.assignments`.

**Value**

A [`DnWkmeansResult`](#clustering-results): k-means fields for the nuggets, plus `row_labels (n,)` for the original rows.

**Examples**

```python
nugget_km = dnlib.dn_wkmeans(dn, 3, num_init=3, seed=42)
np.testing.assert_array_equal(
    nugget_km.row_labels, nugget_km.labels[dn.assignments],
)
```

**See also:** [`wkmeans`](#wkmeans), [`dn_cluster_predict`](#dn_cluster_predict).

### `dn_whclust`

**Description**

Build a weighted Ward hierarchy from nugget centers and their counts.

**Usage**

```text
dn_whclust(dn)
```

**Arguments**

`dn` is a `DataNuggets` object with at least three nuggets.

**Details**

One-feature nuggets are supported. Heights and weight-related limitations are the same as for [`whclust`](#whclust). Empty nuggets can therefore cause problems through their zero weights.

**Value**

A [`WhclustResult`](#clustering-results) whose leaves are nuggets.

**Examples**

```python
nugget_tree = dnlib.dn_whclust(dn)
labels_for_nuggets = dnlib.cut_tree(nugget_tree, k=3)
labels_for_rows = labels_for_nuggets[dn.assignments]
```

**See also:** [`whclust`](#whclust), [`cut_tree`](#cut_tree), [`dendrogram`](#dendrogram).

### `dn_wpca`

**Description**

Compute weighted PCA of nugget centers using nugget counts as row weights.

**Usage**

```text
dn_wpca(dn, wcol=None, corr=False)
```

**Arguments**

`dn` is the nugget set. `wcol` and `corr` follow [`wpca`](#wpca).

**Details**

For one-feature nuggets, the default `wcol=None` raises. Pass `wcol=[1]` explicitly or call `wpca(dn.centers, wrow=dn.weights)`. The WCluster score convention and eigensolver caveats still apply.

**Value**

A [`WpcaResult`](#wpcaresult) with one score row per nugget.

**Examples**

```python
nugget_pca = dnlib.dn_wpca(dn)
all_row_scores = (X - nugget_pca.center) @ nugget_pca.rotation
```

**See also:** [`wpca`](#wpca), [`scores`](#scores).

### `dn_cluster_predict`

**Description**

Assign new rows or nugget centers to clusters fitted on a nugget set.

**Usage**

```text
dn_cluster_predict(dn, cl, newx)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `dn` | Training nuggets. |
| `cl` | One cluster label per training nugget. |
| `newx` | Numeric `(n_new, p)` matrix, or another `DataNuggets` object. |

**Details**

Prefer passing `newx.centers` explicitly for another nugget set, with matching features and preprocessing. The `DataNuggets` input path preserves an R table-slicing quirk: different feature counts can silently select unintended columns. That path also rejects one-feature input; an explicit one-column matrix works.

**Value**

One integer per new row or new nugget: its position in sorted training-label order, as in [`cluster_predict`](#cluster_predict).

**Examples**

```python
predicted_positions = dnlib.dn_cluster_predict(dn, nugget_labels, X[:5])
predicted_clusters = np.unique(nugget_labels)[predicted_positions]
```

**See also:** [`cluster_predict`](#cluster_predict), [R compatibility](compatibility.md).

## Result objects

Core result containers are frozen dataclasses. Array fields are independent, read-only copies; use `.copy()` before editing them. Scalar totals are zero-dimensional NumPy arrays, convertible with `float(result.total)` or `float(result.total_wwcss)`.

### `DataNuggets`

Returned by nugget creation and refinement.

| Field or method | Shape / result | Meaning |
| --- | --- | --- |
| `centers` | `(m, p)` | One representative per nugget. |
| `weights` | `(m,)` | Number of original rows assigned to each nugget. |
| `scales` | `(m,)` | Average within-nugget sample variance across features, with the creation/refinement edge cases described above. |
| `assignments` | `(n,)` | Original row → 0-based nugget ID. |
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

## Flow utilities

Import these functions from `datanuggets.flow`. They implement Cabrera's gate-finding workflow separately from the R package ports. See [methodology](methodology.md) for the approach and [CLI guide](cli.md) for processing FCS files.

### Flow example data

These examples use the measured `202` scatter-channel fixture. Run from the repository root after the [core setup](#example-data), or import `numpy as np` yourself.

```python
from datanuggets import flow

with np.load("tests/fixtures/flow/202.npz") as recorded:
    fsc = recorded["fsc"]
    ssc = recorded["ssc"]

# Match the finder's scatter-channel transformation and grid orientation.
shift = min(fsc.min(), ssc.min())
grid = flow.f_plot(
    np.sqrt(ssc - shift + 3), np.sqrt(fsc - shift + 3),
    nr=100, nc=100, scale="l",
)
density = grid.z.T
candidates = flow.local_maxima(density)
peaks = flow.refine_peaks(density, candidates)
peak_row, peak_col = peaks[0]  # This measured fixture has surviving peaks.
```

For arbitrary data, check `len(peaks)` before indexing: detecting no populations is a valid outcome.

### `lymphmacro_finder`

**Description**

Find candidate populations using forward- and side-scatter channels, a density grid, and convex hull gates.

**Usage**

```text
lymphmacro_finder(fsc_a, ssc_a, *, n_row=11, n_col=11,
                 t11=22, t21=0.25, t12=20, t22=0.5, area=find_area)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `fsc_a`, `ssc_a` | Equal-length, finite, nonconstant one-dimensional FSC-A and SSC-A channels. |
| `n_row`, `n_col` | Row and column widths of the local-maximum neighborhood. |
| `t11`, `t21` | Radius and relative density threshold for the first retained peak. |
| `t12`, `t22` | Radius and relative density threshold for subsequent peaks. |
| `area` | Area-selection function. Defaults to `find_area`; `mirrored_area` reproduces the historical alternative. |

**Details**

The finder transforms the channels, constructs a 100 × 100 density grid, refines detected peaks, and builds hulls in the original channel coordinates. The library default uses `find_area`; the CLI default uses `mirrored_area`. Use the same choice when comparing them.

Gates may overlap, and some events may be outside every gate. Population columns are candidates for inspection, not validated biological cell-type labels.

**Value**

A Boolean array of shape `(n_events, n_populations)`. No surviving peaks produces zero columns.

**Examples**

```python
masks = flow.lymphmacro_finder(fsc, ssc)
events_in_any_gate = masks.any(axis=1)
events_per_gate = masks.sum(axis=0)  # Overlapping events can contribute to multiple gates.
```

**See also:** [`find_area`](#find_area), [`mirrored_area`](#mirrored_area), [`finder_peaks`](#finder_peaks), [`gates`](#gates), [CLI guide](cli.md).

### `finder_peaks`

**Description**

Return the density grid `lymphmacro_finder` gates on, its local maxima, and the peaks it keeps, for inspection or drawing with [`plot.peaks`](#peaks).

**Usage**

```text
finder_peaks(fsc_a, ssc_a, *, n_row=11, n_col=11)
```

**Arguments**

`fsc_a` and `ssc_a` are the channels given to `lymphmacro_finder`. `n_row` and `n_col` are its local-maximum neighborhood.

**Details**

The grid, candidates and peaks are computed by the same steps as the finder's, so peak *k* (counting from 1) is the peak that seeds the finder's population column *k* − 1, "Pop *k*" in `gates`. A candidate missing from `peaks` was removed by `refine_peaks`: too near the grid's edge, or below the 0.9 density quantile.

**Value**

A [`FinderPeaks`](#finderpeaks).

**Examples**

```python
fp = flow.finder_peaks(fsc, ssc)
assert len(fp.peaks) == masks.shape[1]
```

**See also:** [`local_maxima`](#local_maxima), [`refine_peaks`](#refine_peaks), [`peaks`](#peaks).

### `gate_peaks`

**Description**

Return `lymphmacro_finder`'s masks and `finder_peaks`' view together, from one density grid.

**Usage**

```text
gate_peaks(fsc_a, ssc_a, *, n_row=11, n_col=11,
           t11=22, t21=0.25, t12=20, t22=0.5, area=find_area)
```

**Arguments**

The same as [`lymphmacro_finder`](#lymphmacro_finder)'s.

**Details**

A file that is gated and drawn needs both results, and the two functions build the same grid, local maxima and peaks. `gate_peaks` builds them once. `datanuggets-flow` gates every file with it.

**Value**

A tuple `(masks, peaks)`: the array `lymphmacro_finder` returns with the same arguments, and the [`FinderPeaks`](#finderpeaks) `finder_peaks` returns with the same `n_row` and `n_col`.

**Examples**

```python
masks, fp = flow.gate_peaks(fsc, ssc, area=flow.mirrored_area)
assert masks.shape[1] == len(fp.peaks)
```

**See also:** [`lymphmacro_finder`](#lymphmacro_finder), [`finder_peaks`](#finder_peaks), [The flow pipeline library](#the-flow-pipeline-library).

### Grid and gate helpers

Flow peak and cell coordinates are **1-based**. `DensityGrid.zz` has `(x bin, y bin)` columns, while `DensityGrid.z` has shape `(nc, nr)`. Use `grid.z.T` for the peak pipeline, as in the setup, so the coordinates match `zz`.

### `f_plot`

**Description**

Bin two channel vectors into a density grid. Despite its name, this function does not draw a figure.

**Usage**

```text
f_plot(x, y, *, nr=20, nc=20, scale="raw", alpha=0.1)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `x`, `y` | Equal-length finite, nonconstant numeric vectors. |
| `nr`, `nc` | Positive integer numbers of bins along `x` and `y`. |
| `scale` | `"raw"` for counts, `"l"` for `log1p(count)`, or `"d"` for a second `log1p` transformation. |
| `alpha` | After transformation, cells below `alpha * maximum_density` become zero. |

**Details**

Use the same channel transformation and bin counts across the steps of a gating workflow. The finder uses 100 × 100 bins; the helper's default is 20 × 20.

**Value**

A [`DensityGrid`](#densitygrid) with density `z`, grid coordinates `x` and `y`, and per-event bin positions `zz`.

**Examples**

```python
raw_grid = flow.f_plot(ssc, fsc, nr=100, nc=100)
assert raw_grid.zz.shape == (len(fsc), 2)
assert raw_grid.z.shape == (100, 100)
```

**See also:** [`local_maxima`](#local_maxima), [`lymphmacro_finder`](#lymphmacro_finder).

### `local_maxima`

**Description**

Find cells that are maxima in a surrounding window and exceed a global density threshold.

**Usage**

```text
local_maxima(z, *, n_row=11, n_col=11, quantile=0.85)
```

**Arguments**

`z` is a two-dimensional density matrix. `n_row` and `n_col` specify the neighborhood size. `quantile` selects the density quantile that maxima must strictly exceed.

**Details**

The maximum filter reflects at matrix boundaries. A flat maximum can yield more than one candidate cell; use `refine_peaks` for the workflow's additional filtering.

**Value**

An integer `(q, 2)` array of 1-based row/column coordinates.

**Examples**

```python
candidate_cells = flow.local_maxima(density, quantile=0.85)
```

**See also:** [`refine_peaks`](#refine_peaks), [`f_plot`](#f_plot).

### `search_ind`

**Description**

Move a candidate coordinate to a maximum in its local search window.

**Usage**

```text
search_ind(x, idx, idy, threshold=2)
```

**Arguments**

`x` is a density matrix. `idx` and `idy` are the 1-based row and column of the starting point. `threshold` sets the window's half-width in cells.

**Details**

Input coordinates are rounded upward; the search window is clipped to matrix bounds. Ties choose the first maximum in row order.

**Value**

A length-two integer array with the selected 1-based row and column.

**Examples**

```python
nearby_peak = flow.search_ind(density, peak_row, peak_col, threshold=2)
```

**See also:** [`local_maxima`](#local_maxima), [`refine_peaks`](#refine_peaks).

### `refine_peaks`

**Description**

Move candidate peaks to local maxima, remove duplicates, and apply the reference workflow's density and boundary filters.

**Usage**

```text
refine_peaks(x, candidates, *, threshold=2)
```

**Arguments**

`x` is the density matrix, `candidates` is a `(q, 2)` array of 1-based coordinates, and `threshold` is passed to `search_ind`.

**Details**

The fixed boundary checks require row coordinates strictly between 10 and 90 and column coordinates below 90. Retained cells must exceed the matrix's 90th density percentile. These rules assume the reference **100 × 100** grid; they are not adjusted automatically for another size.

**Value**

A `(q_kept, 2)` integer coordinate array, ordered by row position. No retained peaks gives shape `(0, 2)`.

**Examples**

```python
refined_peaks = flow.refine_peaks(density, candidate_cells)
```

**See also:** [`search_ind`](#search_ind), [`find_area`](#find_area).

### `find_area`

**Description**

Select dense grid cells within a disc centered on a peak.

**Usage**

```text
find_area(x, idx, idy, t1=35, t2=0.85)
```

**Arguments**

`x` is the density matrix; `idx` and `idy` are the peak's 1-based row and column. `t1` is the disc radius in grid cells, and `t2` is a density threshold relative to the peak.

**Details**

Cells must lie strictly within radius `t1` and have density strictly greater than `t2` times the peak density. Returned coordinates follow the helper's column-by-column traversal.

**Value**

A `(q, 2)` integer array of 1-based row/column coordinates.

**Examples**

```python
population_cells = flow.find_area(
    density, peak_row, peak_col, t1=22, t2=0.25,
)
```

**See also:** [`hull_gate`](#hull_gate), [`mirrored_area`](#mirrored_area).

### `mirrored_area`

**Description**

Select dense cells using the historical mirrored-disc calculation.

**Usage**

```text
mirrored_area(x, idx, idy, t1=35, t2=0.85)
```

**Arguments**

The density matrix, 1-based peak coordinates, radius `t1`, and relative density threshold `t2` have the same roles as in `find_area`.

**Details**

The disc calculation mirrors the peak coordinates, preserving the original workflow's behavior rather than centering the disc as `find_area` does. Use it to reproduce that workflow on its square grid. It is the CLI's default area method.

**Value**

A `(q, 2)` array of 1-based selected cell coordinates.

**Examples**

```python
historical_cells = flow.mirrored_area(
    density, peak_row, peak_col, t1=22, t2=0.25,
)
```

**See also:** [`find_area`](#find_area), [`lymphmacro_finder`](#lymphmacro_finder), [methodology](methodology.md).

### `hull_gate`

**Description**

Convert a set of population grid cells into an event gate in the original scatter channels.

**Usage**

```text
hull_gate(fsc_a, ssc_a, zz, cells)
```

**Arguments**

| Argument | Meaning |
| --- | --- |
| `fsc_a`, `ssc_a` | Original scatter-channel vectors. |
| `zz` | `(n_events, 2)` per-event 1-based bin coordinates from the matching `DensityGrid`. |
| `cells` | `(q, 2)` 1-based population cell coordinates, aligned with `zz`. |

**Details**

The helper first selects events through a hull of grid cells, then forms a hull in raw channel coordinates and tests all events against it. Too few points or a degenerate hull produces an all-false mask. Always preserve event order between the original channels and `zz`.

**Value**

A Boolean vector of shape `(n_events,)`.

**Examples**

```python
single_gate = flow.hull_gate(fsc, ssc, grid.zz, population_cells)
assert single_gate.shape == fsc.shape
```

**See also:** [`find_area`](#find_area), [`gates`](#gates).

### `DensityGrid`

The result of `f_plot`, exported from `datanuggets.flow`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `z` | `(nc, nr)` | Transformed and thresholded cell densities. |
| `x` | `(nr,)` | Grid positions along the first channel. |
| `y` | `(nc,)` | Grid positions along the second channel. |
| `zz` | `(n_events, 2)` | Per-event 1-based `(x bin, y bin)` coordinates. |

The container is frozen, but its arrays do not have the read-only-copy guarantee of the core result classes.

### `FinderPeaks`

The result of `finder_peaks`, a named tuple exported from `datanuggets.flow`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `density` | `(100, 100)` | The finder's `log1p` density grid, `grid.z.T`: SSC-A bins as rows, FSC-A bins as columns. |
| `candidates` | `(q, 2)` | 1-based `(row, col)` of every local maximum. |
| `peaks` | `(p, 2)` | 1-based `(row, col)` of the refined peaks, in the finder's population order. |
| `fsc` | `(100,)` | Raw FSC-A value at each column. |
| `ssc` | `(100,)` | Raw SSC-A value at each row. |

## The flow pipeline library

`datanuggets-flow` prints, asks and handles its output folders around four submodules of `datanuggets.flow`, which a script or another front end can call directly. They never print or ask: progress comes back as `StageEvent`s and refusals as `FlowError`s. Import them by their module; `datanuggets.flow` itself doesn't import them. They need the `flow` extra where they read or write: flowio to read FCS files, pandas for the tables, matplotlib for the figures and scikit-learn for the t-SNE, each imported only where it is used, except in `figures`.

| Module | Public names |
| --- | --- |
| `datanuggets.flow.settings` | `GateSettings`, `ClusterSettings`, `check_channels`, `cli_flag` |
| `datanuggets.flow.pipeline` | `gate_stage` and `cluster_stage`; the steps they run, `read_fcs`, `gate_files`, `population_stats`, `population_table`, `parse_pops`, `selection_choices`, `parse_selection`, `pops_spec`, `pick_events`, `resolve_channels`, `selection_table`, `make_nuggets`, `cluster_nuggets`, `tsne_embedding`, `result_tables` and `write_tables`; the types `StageEvent`, `Sample`, `PopStats`, `Kept`, `Pool`, `Tables`, `RunResult`, `Reader` and `OnEvent`; `MIN_KEPT_EVENTS` (2) and `MIN_OCCUPIED_NUGGETS` (3) |
| `datanuggets.flow.state` | `save_gates`, `load_gates`, `GateFolder`, `check_identity`, `file_sha256`, `check_new`, `begin`, `finish` and `write_run`; the record's constants `GATES_SCHEMA`, `RUN_SCHEMA`, `MARKER`, `MARKER_TEXT`, `GATE_FILES`, `NPZ_KEYS`, `FIXED`, `READER` and `PACKAGES` |
| `datanuggets.flow.figures` | `local_maxima_figure`, `gates_figure`, `result_figures`, `png_figure` and `save_figures` |
| `datanuggets.exceptions` | `FlowError` |

### Flow settings

`GateSettings` and `ClusterSettings` are frozen, keyword-only dataclasses whose fields are the command's flags, with the same defaults, taken from `datanuggets._defaults`:

```text
GateSettings(fsc="FSC-A", ssc="SSC-A", disc="his", t11=(22,), t21=(0.25,),
             t12=20, t22=0.5, n_row=11, n_col=11)
ClusterSettings(channels=None, exclude_channels=(), cofactor=6000.0, dn_num1=10000,
                dn_num2=1000, seed=291102, n_jobs=-1, k=8,
                tsne=True, tsne_perplexity=30.0, tsne_seed=0)
```

`t11` and `t21` hold one value for every file or one per file. `channels=None` takes the default channel rule, every `-A` channel of the first file but FSC and SSC, less `exclude_channels`. `check()` returns the settings when every value is usable and raises `FlowError` otherwise. `GateSettings.check(n_files)` also checks that `t11` and `t21` give one value or one per file, and `for_file(i)` gives [`gate_peaks`](#gate_peaks)' keyword arguments for file `i`, counting from 0. `to_dict()` gives the settings as JSON writes them, and `from_dict()` reads them back and refuses a key that isn't a setting. `check_channels(channels)` raises `FlowError` for a channel list that doesn't hold two or more distinct channels, or that repeats one: `ClusterSettings.check()` runs it on `channels`, and `resolve_channels` on the list it settles on. `cli_flag(name)` is the flag that sets `name`, such as `--dn-num2` for `dn_num2` and `--jobs` for `n_jobs`.

### `FlowError` and `StageEvent`

`FlowError(message, param=None)` is a `ValueError` for a bad setting or input. A message names its setting as `{param}`: `str(e)` writes the setting's name and `e.cli(cli_flag)` its flag, so one error reads `n_jobs is 0; ...` from Python and `--jobs is 0; ...` from the command. `e.param` is the setting's name, and `"from_gates"` for a gate folder `load_gates` refuses.

`StageEvent(stage, file, n_files, message, data)` is what a stage passes to `on_event` as it goes. `stage` is `"gate"` for each file, `"select"`, `"nuggets"` for each kept file, or `"cluster"`. `file` is the file's number, from 1, or `None` for the whole run. `message` holds the lines the command prints, and `data` the same facts as values. With `on_event=None` nothing is reported.

### Stages

```text
gate_stage(paths, out, settings, *, read=read_fcs, on_event=None, figure=None)
load_gates(folder)
cluster_stage(samples, chosen, out, settings, *, read=read_fcs, on_event=None, figure=None)
```

`gate_stage` gates the files at `paths` in order, with a `GateSettings`, into the existing folder `out`: it draws `local_maxima.png` and `gates.png`, saves `gates.npz` and then `gates.json`, and returns one `Sample` per file. `load_gates` checks a gate folder as `--from-gates` does, raising `FlowError` on the first problem, and returns a `GateFolder(path, settings, samples)`. `cluster_stage` clusters the populations `chosen` gives for each sample, numbered from 1 with `[]` leaving a file out, with a `ClusterSettings`, into the existing folder `out`. It reads each kept file again, writes the tables and the clustering's figures, and returns a `RunResult` with the channels, the tables, the pooled nuggets and their clusters, the t-SNE coordinates and the files written. Both stages check their settings before they read a file, `gate_stage` with `check(len(paths))`, and raise `FlowError` for no files or samples, and for a `chosen` that doesn't give one choice per sample.

`read` takes a path and returns the events and each channel's `$PnN` and `$PnS` names, as `read_fcs` does, so stored arrays can stand in for files. `figure` makes each figure. Its default, `matplotlib.figure.Figure`, leaves nothing in pyplot's registry, so a server can draw run after run; the command passes `plt.figure` to show them.

The command adds the rest of a run folder: `begin(out)` makes the folder with its `INCOMPLETE.txt` and `finish(out)` removes the marker, `write_run` writes `run.json`, and a `--from-gates` run copies the gate figures into its folder. `--gates-only` runs `gate_stage` itself. The one-shot run calls `gate_files` instead, because it asks for the populations after drawing `local_maxima.png` and before drawing `gates.png`.

**Examples**

```python
from pathlib import Path

from datanuggets.flow.pipeline import cluster_stage, gate_stage, parse_selection
from datanuggets.flow.settings import ClusterSettings, GateSettings
from datanuggets.flow.state import load_gates

gates, run = Path("outputs/gates-001"), Path("outputs/run-001")
gates.mkdir(parents=True)
gate_stage(sorted(Path("path/to/fcs_folder").glob("*.fcs")), gates, GateSettings())

folder = load_gates(gates)
chosen = parse_selection("1,2", [len(s["stats"]) for s in folder.samples])
run.mkdir()
result = cluster_stage(folder.samples, chosen, run, ClusterSettings(k=12))
print(result.tables.summary)
```

`parse_selection` reads a `--pops` string against each file's population count, so `"1,2"` keeps Pop 1 and Pop 2 of every file, and refuses a file that has fewer. [Staged runs](cli.md#staged-runs) describes the gate folder, and [reproducibility](reproducibility.md#flow-run-records) its records.

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
| `w` | Optional nonnegative point weights, with at least one positive value. |
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
| `hang` | Fraction of the height span used to hang leaves below their merges; negative values place leaves at zero. |
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
| `w` | Optional nonnegative weights for score-row marker areas, with at least one positive value. |
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

### `gates`

**Description**

Draw raw scatter-channel events with candidate populations overlaid.

**Usage**

```text
gates(fsc_a, ssc_a, masks, *, title=None, ax=None)
```

**Arguments**

`fsc_a` and `ssc_a` are the original event channels. `masks` is a Boolean `(n_events, n_populations)` array. `title` optionally labels the plot, and `ax` supplies axes.

**Details**

All events appear in gray; nonempty population masks are colored on top. Display names such as “Pop 1” number the drawn population columns for readability, without assigning biological cell types.

**Value**

The Matplotlib axes.

**Examples**

Use the measured channels and `masks` from the [flow example](#lymphmacro_finder):

```python
fig, ax = plt.subplots()
plot.gates(fsc, ssc, masks, title="Candidate populations", ax=ax)
```

**See also:** [`lymphmacro_finder`](#lymphmacro_finder), [`hull_gate`](#hull_gate), [`peaks`](#peaks).

### `peaks`

**Description**

Draw the gate finder's density grid with its local maxima and the peaks that seed each population.

**Usage**

```text
peaks(fp, *, title=None, ax=None)
```

**Arguments**

`fp` is a [`FinderPeaks`](#finderpeaks) from `finder_peaks`. `title` optionally labels the plot, and `ax` supplies axes.

**Details**

The grid is drawn FSC-A across and SSC-A up, as in `gates`, on the finder's square-root scale with tick labels in raw units; empty bins are blank. Every local maximum is an open circle and each kept peak a red cross labelled with the Pop number it seeds in `gates`. A colorbar is added to the axes' figure.

**Value**

The Matplotlib axes.

**Examples**

```python
fig, ax = plt.subplots()
plot.peaks(flow.finder_peaks(fsc, ssc), title="Density peaks", ax=ax)
```

**See also:** [`finder_peaks`](#finder_peaks), [`gates`](#gates).

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

Use a `with` block. The context applies `PAPER_RC` and restores the previous Matplotlib settings when it exits.

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
