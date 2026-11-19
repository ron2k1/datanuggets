# Getting started

This walkthrough takes a numeric table from preprocessing to nuggets, clusters, plots, and predictions. For exact signatures and every option, use the [function reference](api.md).

## 1. Install and check

Use Python 3.12 or newer and [activate an environment](installation.md#choose-a-python-version). Install the alpha from PyPI with the `frame` (pandas) and `plot` (matplotlib) extras this guide uses, then check the version:

```sh
python -m pip install "datanuggets[frame,plot]==0.1.0a3"
python -c "import datanuggets; print(datanuggets.__version__)"
```

For the current `main` instead: `python -m pip install "datanuggets[frame,plot] @ git+https://github.com/ron2k1/datanuggets.git@main"`.

The first alpha on PyPI is `0.1.0a3`. The first numerical call can take longer while Numba compiles its kernels. See [installation](installation.md) for other options.

**Try the included data first:** after a [local checkout and install](installation.md#from-a-local-checkout), run `python examples/first_analysis.py`. It uses the recorded R `quakes` data and saves a plot and tables under the checkout's `outputs/` folder. The rest of this guide shows the individual steps using your own CSV files.

## 2. Choose and prepare your features

Rows are observations; columns are numeric features. Replace the two example names below with the columns you want to analyze. Exclude identifiers and known outcome labels.

```python
import numpy as np
import pandas as pd

features = ["feature_a", "feature_b"]
df = pd.read_csv("my_data.csv")
raw = df[features].to_numpy(dtype=float)

if len(raw) < 10 or not np.isfinite(raw).all():
    raise ValueError("This walkthrough needs at least ten rows of finite values.")

location = raw.mean(axis=0)
scale = raw.std(axis=0)
scale[scale == 0] = 1  # Constant columns become zero after centering.
x = (raw - location) / scale
```

Distances depend on feature units. Standardizing is useful when features should have comparable influence, but choose transformations that fit your data. Constant features contribute no distance; remove them if they are not meaningful.

Resolve missing or infinite values before continuing. This example does not drop rows, so the original row order stays intact. Keep `features`, `location`, and `scale` to apply the same transformation to new observations later.

## 3. Make a smaller, weighted representation

```python
from datanuggets import create_dn

dn = create_dn(x, dn_num1=2_000, dn_num2=200, seed=42)
print(f"{len(x):,} observations -> {dn.n_nuggets:,} nuggets")
print(dn.to_frame().head())
```

The two counts are targets: `dn_num1` controls the intermediate reduction and `dn_num2` the final reduction. If a target reaches or exceeds the available rows, the function can return those rows with a warning instead of reducing them further. Inspect the actual result.

`create_dn` is `datanugget`'s `create.DN`, with R's arguments in snake_case: `DN.num1` is `dn_num1`, `R` is `r`, and `no.cores` is `n_jobs`, which defaults to 1. `seed=42` draws from NumPy, so it does not give the nuggets `create.DN(..., seed = 42)` gives in R. `seed=RDraws(42)`, imported with `from datanuggets import RDraws`, draws what R draws.

| Field | Shape | Meaning |
| --- | --- | --- |
| `dn.centers` | `(m, p)` | One center per nugget, in the coordinates you supplied |
| `dn.weights` | `(m,)` | Number of original rows assigned to each nugget |
| `dn.scales` | `(m,)` | Average sample variance across features; creation uses a small floor for single-row and empty nuggets |
| `dn.assignments` | `(n,)` | Zero-based nugget index for each original row |

Here `n` is the original row count, `p` the feature count, and `m` the returned nugget count. The arrays are read-only. Keep the weights when analyzing the centers: treating every center as equally frequent discards information.

These settings are a starting point, not an automatic choice of resolution. A 2,000-by-2,000 float64 matrix, which the second stage builds over 2,000 pooled centers, takes about 32 MB. The first stage builds a distance matrix over each chunk that has more rows than its share of `dn_num1`. A chunk holds about 5,000 rows or fewer at the default `r`, and a 5,000-by-5,000 matrix takes 200 MB. The complete algorithm uses additional memory. The defaults use a larger intermediate target. See [memory limits](limitations.md#memory-and-speed).

### Optional: refine the nuggets

```python
from datanuggets import refine_dn

dn = refine_dn(x, dn, max_splits=2, seed=42)
```

Refinement can split broad nuggets, increase the count, and remove empty nuggets. Pass the exact same preprocessed data in the same row order used to create `dn`. `refine_dn` is `refine.DN`, with `EV.tol` as `ev_tol` and `max.splits` as `max_splits`.

## 4. Cluster and recover row labels

```python
from datanuggets import cut_tree, dn_whclust

k = 5
if dn.n_nuggets < max(k, 3) or np.any(dn.weights == 0):
    raise ValueError("Use enough occupied nuggets; review duplicates and nugget targets.")

tree = dn_whclust(dn)
nugget_labels = cut_tree(tree, k=k)
row_labels = nugget_labels[dn.assignments]
clustered = df.assign(cluster=row_labels)
print(clustered["cluster"].value_counts().sort_index())
```

`nugget_labels` has one entry per nugget. Indexing it by `dn.assignments` gives one label per original row. `dn_whclust` and `cut_tree` are `WCluster`'s `DN.Whclust` and R's `stats::cutree`. The labels are R's minus one, numbered in the order the clusters first appear along the nuggets. The numbers are only labels.

The hierarchy needs at least three nuggets and uses memory proportional to the square of their count. It clusters the weighted centers; it does not rebuild a hierarchy of all original observations.

### Alternatively: weighted k-means

```python
from datanuggets import dn_wkmeans

km = dn_wkmeans(dn, k=5, num_init=10, max_iter=100, seed=42)
kmeans_rows = df.assign(cluster=km.row_labels)
```

`km.row_labels` already maps labels to the original rows. `dn_wkmeans` is `DN.Wkmeans`, with `num.init` as `num_init` and `max.iterations` as `max_iter`, both keyword-only. `km.labels` and `km.row_labels` are R's `Cluster Assignments for data nuggets` and `Cluster Assignments for original dataset`, minus one. This wrapper requires at least two features; automatic initialization also needs enough rows and distinct inputs. Multiple starts can reduce sensitivity to initialization. Keep this result separate from the hierarchical labels used below.

## 5. Plot the clusters and explore PCA

```python
from pathlib import Path

import matplotlib.pyplot as plt
from datanuggets import plot

fig, ax = plt.subplots(figsize=(7, 5))
plot.clusters(dn.centers, nugget_labels, w=dn.weights, ax=ax)
ax.set(
    xlabel=f"{features[0]} (standardized)",
    ylabel=f"{features[1]} (standardized)",
    title="Nugget clusters · larger points represent more observations",
)
fig.tight_layout()
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / "nugget_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
```

This plot shows the first two features. When you select more columns, clustering still uses all of them. The image goes to `outputs/nugget_clusters.png` under the folder you run from, and rerunning replaces it; use a different filename to keep both.

```python
from datanuggets import dn_wpca

pca = dn_wpca(dn)
print(pca.sdev)     # Component standard deviations
print(pca.x.shape)  # One score row per nugget
```

`dn_wpca` is `DN.Wpca`. This is PCA of weighted nugget centers. It does not reconstruct the full within-nugget covariance of the original data. Eigenvector signs can differ from R's, and degenerate inputs have additional [compatibility limits](compatibility.md).

## 6. Assign new observations

Use the same columns, in the same order, and the transformation fitted in step 2.

```python
from datanuggets import dn_cluster_predict

new_df = pd.read_csv("new_data.csv")
new_raw = new_df[features].to_numpy(dtype=float)
if len(new_raw) == 0 or not np.isfinite(new_raw).all():
    raise ValueError("Provide at least one new row of finite values.")
new_x = (new_raw - location) / scale

positions = dn_cluster_predict(dn, nugget_labels, new_x)
levels = np.unique(nugget_labels)
predicted = new_df.assign(cluster=levels[positions])
print(predicted.head())
```

Prediction chooses the nearest weighted cluster mean. It does not rerun hierarchical clustering with the new rows. The function returns positions in the sorted training labels, so `levels[positions]` also works if your labels are values such as `10`, `20`, and `30`. `dn_cluster_predict` is `DNcluster.predict`, which in R stops unless `newx` is a data frame or a datanugget and returns 1-based positions; here `newx` can be an array and the positions are 0-based.
