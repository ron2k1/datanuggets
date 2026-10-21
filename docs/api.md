# API reference

Reference for `datanuggets` **0.1.0.dev0**. Start with the [quickstart](quickstart.md) for a complete workflow; use this page to look up arguments and results. The API is still in development.

All 18 core functions are available directly from `datanuggets`. Flow utilities and plotting helpers live in `datanuggets.flow` and `datanuggets.plot`.

| I want to… | Functions |
| --- | --- |
| Build and refine nuggets | [`create_dn`](#create_dn), [`refine_dn`](#refine_dn) |
| Inspect or select centers | [`get_dn_means`](#get_dn_means), [`create_dn_centers`](#create_dn_centers) |
| Estimate arithmetic work | [`arithmetic_complexity`](#arithmetic_complexity) |
| Compute weighted summaries | [`wmean`](#wmean), [`wss`](#wss), [`wwcss`](#wwcss), [`distw`](#distw) |
| Cluster rows | [`wkmeans`](#wkmeans), [`whclust`](#whclust), [`cut_tree`](#cut_tree) |
| Assign new rows | [`cluster_predict`](#cluster_predict) |
| Run weighted PCA | [`wpca`](#wpca) |
| Analyze nuggets directly | [The four `dn_` wrappers](#nugget-analysis) |
| Find flow populations or draw results | [Flow utilities](#flow-utilities), [plotting](#plotting) |

## Conventions

- **Shape:** `n` is the number of input rows, `p` the number of features, `m` the number of nuggets, and `k` the number of clusters. Prepare numeric features before calling the package; it does not choose columns, impute missing values, or standardize features for you.
- **Input:** value-reading core functions reject NaN and infinity. Nugget creation and refinement require a two-dimensional numeric array or data frame. Weighted analysis functions also accept a vector as one feature and Boolean data as 0/1. Prediction data must be two-dimensional.
- **Weights:** `None` means equal weights. Supply one finite numeric weight per row. Positive weights are the usual choice; several WCluster ports accept zero or signed weights to preserve R behavior, which can produce undefined means, NaNs, or unusual clusters. PCA has stricter rules below.
- **Labels:** nugget IDs and generated cluster labels are **0-based**. Input `cl` labels must be finite numeric whole numbers, one per row; they need not be consecutive. Cluster summaries follow sorted distinct label order. Flow grid coordinates are a separate **1-based** convention.
- **Randomness:** `seed` accepts an integer, a NumPy `Generator`, or `RDraws`. Integer seeds use NumPy, so matching an R seed number alone does not reproduce R's draws. See [reproducibility](reproducibility.md) and [R compatibility](compatibility.md#rdraws).
- **Parallelism:** `n_jobs=1` runs serially; a positive integer selects workers and `-1` uses all cores. `progress=True` requires the `progress` extra. Package warnings use `DataNuggetWarning`; informational messages use the `datanuggets` logger.

## Creating and refining nuggets

### `create_dn`

```text
create_dn(x, center_method="mean", r=5000, delete_percent=0.1,
          dn_num1=10000, dn_num2=2000, dist_metric="euclidean",
          seed=291102, n_jobs=1, progress=False)
```

Reduce `x` to weighted representatives and record which nugget contains each original row. Returns a [`DataNuggets`](#datanuggets) object.

| Argument | Meaning |
| --- | --- |
| `x` | Numeric `(n, p)` data, with at least two rows. |
| `center_method` | `"mean"`: average the rows assigned to each nugget. `"original"`: keep the selected input center. `"random"`: choose an assigned input row. |
| `r` | Controls the initial chunk size; must be within `[100, 10000]`. |
| `delete_percent` | Fraction of centers removed per thinning pass; strictly between 0 and 1. |
| `dn_num1` | Positive integer target for centers pooled from the first stage. |
| `dn_num2` | Positive integer target for the final stage; cannot exceed `dn_num1`. |
| `dist_metric` | `"euclidean"` or `"manhattan"`, used for thinning and assignment. |
| `seed`, `n_jobs`, `progress` | See [conventions](#conventions). |

These counts are targets: small inputs or chunk allocation can change the number of available centers. A thinning stage that already has no more than its requested count returns its input rows and warns. Empty nuggets are retained. Nugget scale is the average sample variance across features; creation applies a small positive floor to single-row and empty nuggets.

**Memory:** center selection builds pairwise distance matrices. Increasing `dn_num1` can quickly increase memory use; see [limitations](limitations.md).

### `create_dn_centers`

```text
create_dn_centers(rs, dn_num, *, delete_percent=0.1,
                  dist_metric="euclidean")
```

Select representative rows from numeric `(n, p)` sample `rs` using deterministic close-pair thinning. `dn_num` is a positive integer target; the other arguments have the same meaning as in `create_dn`.

Returns `CentersResult` with `centers` of shape `(m, p)` and `kept` of shape `(m,)`, the selected **0-based positions in `rs`**. It does not calculate weights or assignments. If `dn_num >= n`, it returns every row and warns.

### `refine_dn`

```text
refine_dn(x, dn, ev_tol=0.9, max_splits=5, min_nugget_size=2,
          shape_split=False, min_shape_size=10, delta=2,
          nstart=25, seed=291102, n_jobs=1, progress=False)
```

Split nuggets that are too spread out, optionally followed by a pass for elongated nuggets. Returns a new `DataNuggets`.

| Argument | Meaning |
| --- | --- |
| `x`, `dn` | The original numeric data and its nuggets. Preserve the same rows, row order, and number of features. |
| `ev_tol` | Quantile in `(0, 1)` used to set the first-eigenvalue splitting threshold before the first round. |
| `max_splits` | Maximum splitting rounds for the eigenvalue pass, and again for the optional shape pass. Use a nonnegative integer. |
| `min_nugget_size` | Minimum allowed child size; use an integer of at least 2. |
| `shape_split` | Whether to run the shape pass; requires at least two features to have an effect. |
| `min_shape_size` | Minimum nugget size eligible for the shape pass; at least 10. |
| `delta` | Eigenvalue-ratio threshold for the shape pass; greater than 1. |
| `nstart` | Number of random starts for each two-means split; use a positive integer. |
| `seed`, `n_jobs`, `progress` | See [conventions](#conventions). |

Refinement reads the assignments from `dn` and recomputes every center, weight, and scale from `x`. It removes empty nuggets, renumbers the survivors, and gives single-row nuggets scale 0. Numerical differences in eigenvalues can affect borderline splits; consult [R compatibility](compatibility.md#refine_dn).

### `get_dn_means`

```text
get_dn_means(x, dn)
```

Return an `(m, p)` array containing the mean of the rows assigned to each nugget, regardless of how its center was originally chosen. `x` must have the same row count and feature count as `dn`, in the original row order. An empty nugget produces a row of NaNs.

### `arithmetic_complexity`

```text
arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2)
```

Return the base-10 logarithm of R's arithmetic-operation estimate for nugget creation. Only the two-dimensional shape of `x` is read. Pass the creation settings explicitly; this function has no defaults.

This is an operation-count model, **not a runtime or memory estimate**. It preserves R's formula, with errors for arguments that would make R's calculation loop indefinitely.

## Weighted summaries

These functions operate on `x` with shape `(n, p)` and optional row weights `w` with shape `(n,)`. Where present, `cl` assigns one numeric integer label to each row. Let `k` be the number of distinct labels.

### `wmean`

```text
wmean(x, cl, w=None)
```

Return the weighted mean of each cluster as a `(k, p)` array, in sorted label order. A cluster with zero total weight has an undefined mean; zero or canceling signed weights can produce NaN or infinity.

### `wss`

```text
wss(x, w=None)
```

Return a scalar weighted sum of squared distances from the overall weighted mean. A zero total weight produces NaN.

### `wwcss`

```text
wwcss(x, cl, w=None, group_sum=False)
```

Return the total weighted within-cluster sum of squares. With `group_sum=True`, return `WwcssResult`: `per_cluster` has shape `(k,)` in sorted label order, and `total` is a zero-dimensional array. Use `float(result.total)` when a Python scalar is needed.

### `distw`

```text
distw(x, cl, w=None)
```

Return a `(k, k)` matrix of the increase in weighted within-cluster sum of squares from merging each pair of clusters. Requires at least two clusters.

Only the **strict lower triangle** contains distances; the diagonal and upper triangle are NaN. This output is not a full symmetric distance matrix.

## Clustering and prediction

### `wkmeans`

```text
wkmeans(x, k, *, w=None, centers=None, num_init=1, max_iter=10,
        seed=291102, n_jobs=1)
```

Run weighted k-means using WCluster's behavior. Returns [`WkmeansResult`](#clustering-results).

| Argument | Meaning |
| --- | --- |
| `x`, `w` | `(n, p)` data and optional `(n,)` row weights. |
| `k` | Requested cluster count; use a positive integer. |
| `centers` | Optional `(k, p)` initial centers. Supplying them forces a single start. |
| `num_init` | Number of random starts; at least 1. The best within-cluster sum of squares wins. |
| `max_iter` | Maximum iterations per start; at least 1. |
| `seed`, `n_jobs` | Random source and parallel starts; see [conventions](#conventions). |

Automatic initialization needs at least `2 * k` rows and at least `k` distinct `(row, weight)` pairs. It can fail when it cannot find a usable initialization. Supplying centers bypasses automatic initialization.

R's behavior can leave empty labels: `labels` may contain gaps, and `centers` and `wwcss` contain only labels that remain present, in sorted order. Do not assume `result.centers[label]` always indexes the matching center.

### `whclust`

```text
whclust(x, w=None)
```

Perform weighted Ward hierarchical clustering on `(n, p)` data, with at least three rows. Returns [`WhclustResult`](#clustering-results).

The algorithm merges the pair with the smallest increase in weighted within-cluster sum of squares. Its stored `height` is the **total within-cluster sum of squares after each merge**, so it is not the usual SciPy Ward distance. Memory grows quadratically with the number of input rows. Prefer positive weights: zero or signed weights can produce NaN heights or degenerate trees.

### `cut_tree`

```text
cut_tree(tree, k=None, h=None)
```

Cut a `WhclustResult` by cluster count `k` or height `h`. Returns one 0-based label per input row. Labels are numbered by first appearance in the original row order.

Supply a single count in `[1, n]` or a single height; `k` takes precedence if both are supplied. Cutting by height requires nondecreasing heights. For a nugget tree, map the resulting nugget labels to original rows with `labels[dn.assignments]`.

### `cluster_predict`

```text
cluster_predict(x, cl, newx, *, w=None)
```

Calculate the weighted means of the training clusters in `x` and assign each row of `newx` to its nearest mean. Training labels `cl` and optional weights `w` have shape `(n,)`; `newx` must be a two-dimensional matrix with the same features in the same order.

Returns an integer vector with one entry per new row. **These are positions in sorted cluster order, not the original label values.** Equal distances choose the first position. To recover the original labels:

```python
import numpy as np
from datanuggets import cluster_predict

positions = cluster_predict(x, cl, newx, w=w)
predicted_labels = np.unique(cl)[positions]
```

Clusters with undefined distances are skipped. The call raises if every distance for a new row is NaN; this can result from zero-weight clusters or arithmetic overflow.

## Weighted PCA

### `wpca`

```text
wpca(x, wrow=None, wcol=None, corr=False)
```

Run WCluster's weighted principal component analysis on `(n, p)` data. Returns [`WpcaResult`](#wpcaresult).

| Argument | Meaning |
| --- | --- |
| `wrow` | Nonnegative weights, one per row. `None` means ones. Weights act as counts: covariance divides by `sum(wrow) - 1`. |
| `wcol` | Weights, one per feature. `None` means ones. They are normalized to sum to `p`; mixed signs and a zero total are rejected. Use nonnegative weights with a positive total for ordinary analysis. |
| `corr` | If `True`, standardize the weighted covariance to a correlation matrix before eigendecomposition. |

Row weights cannot sum to 0 or exactly 1. With `corr=True`, their sum must exceed 1 and every feature must have positive weighted variance. Probability weights normalized to sum to 1 are therefore unsuitable without rescaling.

**Score convention:** `result.x` is `(x - result.center) @ result.rotation`, even with `corr=True`. It is not divided by `result.scale` or multiplied by row or column weights. Use this convention when projecting more rows or comparing with another PCA library.

**Numerical limits:** component signs can differ from R's. Repeated eigenvalues allow different bases, and rank-deficient inputs can produce small negative eigenvalues, NaN standard deviations, and `DataNuggetWarning`. Exact eigensolver parity with R is not guaranteed; see [limitations](limitations.md).

## Nugget analysis

The wrappers below use `dn.centers` as the data and `dn.weights` as row weights. `dn` must be a `DataNuggets` object.

### `dn_wkmeans`

```text
dn_wkmeans(dn, k, *, centers=None, num_init=1, max_iter=10,
           seed=291102, n_jobs=1)
```

Weighted k-means on nuggets. Arguments have the same meaning as in [`wkmeans`](#wkmeans). Returns `DnWkmeansResult`, including `labels` for the nuggets and `row_labels` for the original observations.

This wrapper rejects one-feature nuggets, preserving the R wrapper's behavior. For one feature, call `wkmeans(dn.centers, k, w=dn.weights, ...)` and map its labels through `dn.assignments`.

### `dn_whclust`

```text
dn_whclust(dn)
```

Weighted Ward clustering on nugget centers. Returns `WhclustResult` and requires at least three nuggets. One-feature nuggets are supported. Use `cut_tree` to obtain nugget labels, then index them by `dn.assignments` for original-row labels.

### `dn_wpca`

```text
dn_wpca(dn, wcol=None, corr=False)
```

Weighted PCA on nugget centers. Returns `WpcaResult` with one score row per nugget; `wcol` and `corr` follow [`wpca`](#wpca).

For one-feature nuggets, the default `wcol=None` raises. Pass `wcol=[1]` explicitly or use `wpca(dn.centers, wrow=dn.weights)`.

### `dn_cluster_predict`

```text
dn_cluster_predict(dn, cl, newx)
```

Predict from clustered nuggets. `cl` contains one cluster label per nugget. `newx` may be a numeric matrix or another `DataNuggets`; the output has one position per new row or new nugget, using the same sorted-label convention as [`cluster_predict`](#cluster_predict).

Prefer passing `newx.centers` explicitly when predicting another nugget set, and ensure feature counts and order match. The `DataNuggets` input path preserves an R table-slicing quirk: mismatched feature counts can select unintended columns. It also rejects one-feature nugget input; an explicit one-column matrix works.

## Result objects

Core result containers are frozen dataclasses. Their array fields are independent, read-only copies; call `.copy()` before editing an array. Scalar totals are stored as zero-dimensional NumPy arrays.

### `DataNuggets`

| Field or method | Shape / result | Meaning |
| --- | --- | --- |
| `centers` | `(m, p)` | One representative per nugget. |
| `weights` | `(m,)` | Number of original rows assigned to each nugget. |
| `scales` | `(m,)` | Average within-nugget sample variance across features, with the creation/refinement edge cases described above. |
| `assignments` | `(n,)` | Original row → 0-based nugget ID. |
| `n_nuggets` | `int` | Number of nuggets. |
| `to_frame()` | pandas DataFrame | `Data Nugget`, `Center1` through `Centerp`, `Weight`, `Scale`. Requires the `frame` extra. |

`CentersResult` contains only `centers` and the selected input positions `kept`. `WwcssResult` contains `per_cluster` and `total`, as described under [`wwcss`](#wwcss).

### `WpcaResult`

| Field | Shape | Meaning |
| --- | --- | --- |
| `sdev` | `(p,)` | Square roots of eigenvalues, largest first. |
| `rotation` | `(p, p)` | Principal component directions, one per column. |
| `x` | `(n, p)` | Scores of the input rows using the WCluster convention above. |
| `center` | `(p,)` | Weighted feature means. |
| `scale` | `(p,)` | Covariance standard deviations for `corr=True`; ones otherwise. |
| `wrow` | `(n,)` | Input row weights. |
| `wcol` | `(p,)` | Column weights normalized to sum to `p`. |

For `dn_wpca`, replace `n` with the number of nuggets.

### Clustering results

`WkmeansResult` contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `labels` | `(n,)` | Cluster label for each input row; gaps are possible. |
| `centers` | `(q, p)` | Weighted mean for each of the `q` labels present, in sorted label order. |
| `wwcss` | `(q,)` | Within-cluster sum of squares in that same order. |
| `total_wwcss` | `()` | Total within-cluster sum of squares. |

`DnWkmeansResult` has the same fields for the nuggets, plus `row_labels` with shape `(n,)` for the original data.

`WhclustResult` contains:

| Field or method | Shape | Meaning |
| --- | --- | --- |
| `merge` | `(n - 1, 2)` | SciPy-style IDs: leaves `0` through `n - 1`; merge step `s` creates node `n + s`. |
| `height` | `(n - 1,)` | Total within-cluster sum of squares after each merge. |
| `order` | `(n,)` | Leaf order for the dendrogram. |
| `to_linkage()` | `(n - 1, 4)` | Merge IDs, unchanged heights, and cluster sizes in SciPy linkage layout. |

`to_linkage()` rejects self-merges. A linkage layout does not make unusual heights valid for every SciPy consumer: negative or NaN heights remain unchanged. Signed-weight degeneracies can also make `order` repeat or omit leaves.

## Flow utilities

Import from `datanuggets.flow`. These functions implement Cabrera's gate-finding workflow; they are separate from the R package ports. See [methodology](methodology.md) for the algorithm and [CLI guide](cli.md) for FCS-file processing.

### `lymphmacro_finder`

```text
lymphmacro_finder(fsc_a, ssc_a, *, n_row=11, n_col=11,
                 t11=22, t21=0.25, t12=20, t22=0.5, area=find_area)
```

Find candidate populations from equal-length, finite, nonconstant one-dimensional FSC-A and SSC-A channels. The finder transforms the channels, builds a 100 × 100 density grid, detects peaks, and constructs hull gates.

`n_row` and `n_col` control the local-maximum window. `t11` and `t21` set the radius and relative density threshold for the first peak; `t12` and `t22` do so for the others. `area=find_area` uses a disc centered on the peak; `area=mirrored_area` preserves the reference code's mirrored-disc behavior. The CLI's default is the mirrored variant, so specify the same choice when comparing results.

Returns a Boolean `(n_events, n_populations)` array. Gates may overlap, some events may be outside every gate, and no surviving peaks gives zero columns. Column positions are candidate populations, not validated biological cell-type labels.

### Grid and gate helpers

```text
f_plot(x, y, *, nr=20, nc=20, scale="raw", alpha=0.1)
local_maxima(z, *, n_row=11, n_col=11, quantile=0.85)
search_ind(x, idx, idy, threshold=2)
refine_peaks(x, candidates, *, threshold=2)
find_area(x, idx, idy, t1=35, t2=0.85)
mirrored_area(x, idx, idy, t1=35, t2=0.85)
hull_gate(fsc_a, ssc_a, zz, cells)
```

| Function | Inputs and return value |
| --- | --- |
| `f_plot` | Bin finite, nonconstant channel vectors `x` and `y`. `nr` and `nc` are bin counts; `scale` is `"raw"`, `"l"` for `log1p`, or `"d"` for a second `log1p`. Cells below `alpha * max` become zero. Returns `DensityGrid`. |
| `local_maxima` | Find window maxima above the specified density quantile in a matrix. Returns `(q, 2)` peak coordinates. |
| `search_ind` | Search around `(idx, idy)` within `threshold` cells. Returns the coordinate pair of a nearby maximum. |
| `refine_peaks` | Refine candidate coordinates and apply the reference density and boundary filters. Returns `(q, 2)` coordinates; its fixed boundary rules assume the reference 100 × 100 grid. |
| `find_area` | Return `(q, 2)` grid cells within radius `t1` of the peak and above `t2` times its density. |
| `mirrored_area` | Same inputs and output as `find_area`, using the historical mirrored-disc calculation. |
| `hull_gate` | Use per-event bins `zz` and population `cells` to build hull gates in the original channels. Returns a Boolean event mask; insufficient or degenerate hull points produce an all-false mask. |

**Coordinate convention:** flow cell and peak coordinates are 1-based. `DensityGrid.zz` is `(n_events, 2)` with `(x bin, y bin)` columns. Its `z` matrix has shape `(nc, nr)`, and `x` and `y` hold the grid positions. The peak pipeline uses `grid.z.T` so that peak coordinates align with `zz`; do not silently mix these orientations.

## Plotting

Install the `plot` extra, then `from datanuggets import plot`. Drawing functions accept an optional Matplotlib `ax` and return the axes; without `ax`, they use the current axes. `dims` selects two **0-based** feature or component positions.

```text
clusters(x, labels, *, w=None, centers=None, dims=(0, 1),
         size="area", colors=None, ax=None)
nuggets(dn, *, labels=None, dims=(0, 1), colors=None, ax=None)
dendrogram(tree, *, labels=None, hang=0.1, ax=None)
scores(wpca, *, w=None, labels=None, rows=None, dims=(0, 1),
       colors=None, ax=None)
scree(wpca, *, ax=None)
loadings(wpca, *, names=None, dims=(0, 1), ax=None)
gates(fsc_a, ssc_a, masks, *, title=None, ax=None)
style()
save(fig, path, *, width="single", height=None)
```

| Function | Purpose and key arguments |
| --- | --- |
| `plot.clusters` | Scatter the selected features, colored by `labels`. Optional `w` controls point area; `size="log"` uses R's log-weight sizing. Optional `centers` supplies cluster means in sorted label order. |
| `plot.nuggets` | Scatter nugget centers. Without `labels`, green intensity represents weight; higher weights are lighter. |
| `plot.dendrogram` | Draw a `WhclustResult`. `labels` may be custom leaf labels or `False` to hide them. `hang` controls leaf height; a negative value puts leaves at zero. |
| `plot.scores` | Scatter PCA score rows, with optional weights and labels. Optional original-data `rows` are projected as a background overlay, limited to 50,000 displayed points. |
| `plot.scree` | Plot the percentage of variance explained by each component. |
| `plot.loadings` | Draw feature loading arrows; `names` supplies feature labels. |
| `plot.gates` | Draw raw FSC-A/SSC-A events and the Boolean population masks. |
| `plot.style` | Context manager for the package's paper-figure style; restores settings on exit. |
| `plot.save` | Save a figure and return its `Path`. `width` is `"single"` (3.5 inches), `"double"` (7 inches), or a numeric width in inches. Default height is 0.75 × width; output uses 300 dpi and the filename extension selects the format. |

Plot label arrays must have integer dtype, and `dims` must select two distinct columns. `colors` accepts a nonempty sequence of Matplotlib colors. The package also exports `OKABE_ITO`, `R_PALETTE`, and `PAPER_RC`.

PCA score and loading plots normalize component signs for display without changing the underlying result. Variance-percentage plots treat NaN standard deviations as zero; inspect the PCA result and its warnings when assessing numerical problems.

For install options, see [installation](installation.md). For deliberate R quirks and unsupported cases, keep [compatibility](compatibility.md) and [limitations](limitations.md) alongside this reference.
