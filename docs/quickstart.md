# Getting started

This walkthrough takes a numeric table from preprocessing to nuggets, clusters, plots, and predictions. For exact signatures and every option, use the [function reference](api.md). For FCS files, jump to [the flow command](#a-first-datanuggets-flow-run).

## 1. Install and check

Use Python 3.12 or newer. Python 3.14 and 3.15 run the same package; choose one and [activate an environment](installation.md#choose-a-python-version). Install from `main` with Git and repository access, or use the [pinned PyPI command](installation.md#install-the-alpha-from-pypi) once the alpha is published:

```sh
python -m pip install "datanuggets[frame,plot] @ git+https://github.com/ron2k1/datanuggets.git@main"
python -c "import datanuggets; print(datanuggets.__version__)"
```

The first alpha version is `0.1.0a1`. The first numerical call can take longer while Numba compiles its kernels. See [installation](installation.md) for other options.

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

| Field | Shape | Meaning |
| --- | --- | --- |
| `dn.centers` | `(m, p)` | One center per nugget, in the coordinates you supplied |
| `dn.weights` | `(m,)` | Number of original rows assigned to each nugget |
| `dn.scales` | `(m,)` | Average sample variance across features; creation uses a small floor for single-row and empty nuggets |
| `dn.assignments` | `(n,)` | Zero-based nugget index for each original row |

Here `n` is the original row count, `p` the feature count, and `m` the returned nugget count. The arrays are read-only. Keep the weights when analyzing the centers: treating every center as equally frequent discards information.

These settings are a starting point, not an automatic choice of resolution. A 2,000-by-2,000 float64 matrix takes about 32 MB; the complete algorithm uses additional memory. The defaults use a larger intermediate target. See [memory limits](limitations.md#memory-and-speed).

### Optional: refine the nuggets

```python
from datanuggets import refine_dn

dn = refine_dn(x, dn, max_splits=2, seed=42)
```

Refinement can split broad nuggets, increase the count, and remove empty nuggets. Pass the exact same preprocessed data in the same row order used to create `dn`. It does not guarantee a better answer for every analysis.

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

`nugget_labels` has one entry per nugget. Indexing it by `dn.assignments` gives one label per original row. Python labels start at zero; their numbers have no inherent ordering or biological meaning.

The hierarchy needs at least three nuggets and uses memory proportional to the square of their count. It clusters the weighted centers; it does not rebuild a hierarchy of all original observations.

### Alternatively: weighted k-means

```python
from datanuggets import dn_wkmeans

km = dn_wkmeans(dn, k=5, num_init=10, max_iter=100, seed=42)
kmeans_rows = df.assign(cluster=km.row_labels)
```

`km.row_labels` already maps labels to the original rows. This wrapper requires at least two features; automatic initialization also needs enough rows and distinct inputs. Multiple starts can reduce sensitivity to initialization. Keep this result separate from the hierarchical labels used below.

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

This plot shows the first two features. When you select more columns, clustering still uses all of them. Running from the repository root saves the image in its ignored `outputs/` folder. Rerunning this example replaces `nugget_clusters.png`; use a different filename to keep both. `create_dn` itself returns data and does not write images.

```python
from datanuggets import dn_wpca

pca = dn_wpca(dn)
print(pca.sdev)     # Component standard deviations
print(pca.x.shape)  # One score row per nugget
```

This is PCA of weighted nugget centers. It does not reconstruct the full within-nugget covariance of the original data. Eigenvector signs can differ from R's, and degenerate inputs have additional [compatibility limits](compatibility.md).

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

Prediction chooses the nearest weighted cluster mean. It does not rerun hierarchical clustering with the new rows. The function returns positions in the sorted training labels, so `levels[positions]` also works if your labels are values such as `10`, `20`, and `30`.

## A first datanuggets-flow run

The flow workflow is experimental. Install its extra, then run it on a file or folder:

```sh
python -m pip install "datanuggets[flow] @ git+https://github.com/ron2k1/datanuggets.git@main"
datanuggets-flow path/to/fcs_folder
```

1. **Run from the repository root.** Each run creates a new `outputs/flow-<timestamp>/` folder for its figures and tables. An optional `--out` must name a new folder; existing paths are refused. A failed or cancelled run leaves an `INCOMPLETE.txt` marker beside any partial results.
2. **Inspect the gates.** The tool displays candidate populations and their counts. Choose populations for each file; the same population number need not identify the same cells across files.
3. **Review the clustered results.** The selected events are reduced per file, and their nuggets are clustered together. Examine the tables and figures before interpreting a cluster as a cell type.

Choose at least two distinct fluorescence channels; the CLI checks this before asking for populations. Clustering also needs at least three occupied nuggets across retained files. Channel flags use the FCS `$PnN` names, not display marker names. By default, it selects fluorescence-like names ending in `-A` and applies `arcsinh(x / 6000)`. Check both choices for your panel. It does not apply compensation.

The CLI's default `--disc his` and the low-level array finder's default area rule differ and can change population membership. Gates can overlap; selecting several keeps their union, with each event counted once.

After inspecting each file's populations, a scripted run can use:

```sh
datanuggets-flow path/to/fcs_folder --pops "1,2" --no-show
```

This gets its own output folder too. `--pops "1,2"` applies those population numbers to every file, so do not assume this selects equivalent biological populations without checking.

The [CLI reference](cli.md) lists flags, outputs, and errors. The [methodology](methodology.md) explains the implemented pipeline, and [reproducibility](reproducibility.md) records the public datasets used for validation.
