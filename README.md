# datanuggets

**Summarize large datasets. Keep the weights. Work in Python.**

Data nuggets represent a dataset with a smaller set of points. Each nugget stores a center, the number of observations it represents, and a measure of spread. You can cluster these weighted points, explore them with PCA, and map cluster labels back to the original rows.

This is a native Python port of methods from R's `datanugget` 1.5.0 and `WCluster` 1.3.0. R is not required. See the [R-to-Python function map](https://github.com/ron2k1/datanuggets/blob/main/docs/api.md#r-to-python-function-map) for coverage; automatic nugget-count selection (`optimal.DN`) is not implemented.

[CSV command line](https://github.com/ron2k1/datanuggets/blob/main/docs/cli.md#analyze-a-csv) · [Python walkthrough](https://github.com/ron2k1/datanuggets/blob/main/docs/quickstart.md) · [Function reference](https://github.com/ron2k1/datanuggets/blob/main/docs/api.md) · [Limitations](https://github.com/ron2k1/datanuggets/blob/main/docs/limitations.md)

> **Research alpha — `0.1.0a1`.** The API may change. The port is tested against recorded R cases, with [known differences](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md). See the [release notes](https://github.com/ron2k1/datanuggets/blob/main/docs/releases.md) for scope and limitations.

Publication is on hold while the provenance of the GNU sorting helpers is reviewed. See the [third-party notices](https://github.com/ron2k1/datanuggets/blob/main/THIRD_PARTY.md#publication-hold-gnu-sorting-helpers).

## Get started

### 1. Install

Use Python 3.12 or newer. Create and activate a [virtual environment](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#choose-a-python-version). Once the alpha is published, install its exact version from PyPI:

```sh
python -m pip install "datanuggets[cli]==0.1.0a1"
```

For the current source, including before the first publication, install from `main` with Git and repository access:

```sh
python -m pip install "datanuggets[cli] @ git+https://github.com/ron2k1/datanuggets.git@main"
```

The `cli` extra adds pandas and matplotlib for CSV analysis, tables, and plots. On a MacBook, you can also [install a supplied wheel once](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#macbook-install-once-from-a-wheel) without Git or uv.

**Python 3.14 and 3.15 run the same package**; they are alternative interpreter versions, not separate editions of datanuggets. You only need one. Contributors testing both use separate environments, as explained in [installation](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#choose-a-python-version).

Prefer a ready-to-run example? [Clone the repository](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#from-a-local-checkout) and run `python examples/first_analysis.py`. It uses the included R `quakes` data and saves a plot and tables under `outputs/`.

### Analyze your own CSV from Terminal

Keep your dataset wherever you normally store it. In your installed environment, replace the path and column names below with yours:

```sh
datanuggets analyze "path/to/study.csv" --columns measurement_a measurement_b --transform standardize --nuggets 200 --analysis kmeans pca --clusters 3
```

This creates nuggets, runs weighted k-means and PCA, and writes tables and a `plots/` folder under a new `outputs/analysis-.../` directory in your current working folder. `assignments.csv` maps every input row to its nugget and cluster; `cluster_summary.csv` reports cluster sizes and feature means in the original units. The Terminal prints a summary and the result location.

Choose numeric measurement columns, leaving out IDs and outcome labels. Missing or nonfinite values stop the run. Standardization and the requested counts are choices to review for your data; the command does not select them automatically. See the [CSV guide](https://github.com/ron2k1/datanuggets/blob/main/docs/cli.md#analyze-a-csv) for transformations, refinement, hierarchical clustering, and output details. `python -m datanuggets` also works in place of `datanuggets`.

### 2. Create nuggets in Python

Choose the columns you want to analyze; leave out IDs and known outcome labels. Replace `feature_a` and `feature_b` with your CSV's column names.

```python
import numpy as np
import pandas as pd
from datanuggets import create_dn

df = pd.read_csv("my_data.csv")
x = df[["feature_a", "feature_b"]].to_numpy(dtype=float)
if len(x) < 2 or not np.isfinite(x).all():
    raise ValueError("Provide at least two rows of finite numeric values.")

# Standardize features; constant columns become zero.
scale = x.std(axis=0)
scale[scale == 0] = 1
x = (x - x.mean(axis=0)) / scale

dn = create_dn(x, dn_num1=2_000, dn_num2=200, seed=42)
print(dn.to_frame().head())
```

`dn_num1` controls the intermediate reduction; `dn_num2` is the final target. Start small, then check whether the reduction preserves the structure you care about. Standardization is one preprocessing choice, not a requirement for every dataset.

### 3. Analyze and visualize

| You want to… | Use |
| --- | --- |
| Refine the nuggets | `refine_dn` |
| Build and cut a weighted hierarchy | `dn_whclust` + `cut_tree` |
| Run weighted k-means | `dn_wkmeans` |
| Explore weighted principal components | `dn_wpca` |
| Assign new observations to clusters | `dn_cluster_predict` |

The [walkthrough](https://github.com/ron2k1/datanuggets/blob/main/docs/quickstart.md) connects these steps, plots the results, and shows how `dn.assignments` maps nugget labels back to your rows. The [function reference](https://github.com/ron2k1/datanuggets/blob/main/docs/api.md) covers arguments, defaults, outputs, and conventions.

## Working with FCS files

The experimental `datanuggets-flow` command is a separate downstream workflow using the package. It finds candidate gates, lets you choose populations, and clusters their events across files. Once the alpha is published, install the `flow` extra at the same version:

```sh
python -m pip install "datanuggets[flow]==0.1.0a1"
datanuggets-flow path/to/fcs_folder
```

Before publication, use the [source install](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#install-from-github) with `[flow]` in place of `[cli]`.

Figures and tables go into a new `outputs/flow-<timestamp>/` folder for each run. Run from the repository root to keep them together there; `outputs/` is ignored by Git. You can choose another new folder with `--out`; existing paths are refused.

A run can also be split in two. `--gates-only` gates the files and saves the gates in a folder, and `--from-gates` chooses populations from that folder and clusters, as often as needed, without gating again. The [staged runs](https://github.com/ron2k1/datanuggets/blob/main/docs/cli.md#staged-runs) section covers both.

Inspect the gates, choose at least two fluorescence channels, and check your compensation and preprocessing: the tool does not apply compensation. See the [FCS guide](https://github.com/ron2k1/datanuggets/blob/main/docs/cli.md#fcs-pipeline) before using it on a study.

## Know the limits

- **Memory still matters.** Hierarchical clustering uses a square matrix: 10,000 pooled nuggets need about 800 MB for that matrix alone.
- **Reduction changes the analysis.** Compare nugget counts and preprocessing choices before interpreting clusters. Cluster numbers are not validated biological identities.
- **R results are not universally identical.** Random streams, eigen calculations, and some edge cases differ. Read the [compatibility notes](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md).

See [limitations and open problems](https://github.com/ron2k1/datanuggets/blob/main/docs/limitations.md), [methodology](https://github.com/ron2k1/datanuggets/blob/main/docs/methodology.md), [reproducibility](https://github.com/ron2k1/datanuggets/blob/main/docs/reproducibility.md), or the [contributor guide](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) for details.

## Citation and credits

Beavers, T. E., et al. *Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure.* Journal of Computational and Graphical Statistics **34**(1), 330–342 (2025; online 2024). DOI: `10.1080/10618600.2024.2341896`.

The intended package license is [GPL-2.0-only](https://github.com/ron2k1/datanuggets/blob/main/LICENSE), subject to the unresolved sorting-source review above. The R algorithms are the work of Rituparna Dey, Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi, and Mariusz Lubomirski (`datanugget`); and Rituparna Dey, Yajie Duan, Javier Cabrera, and Ge Cheng (`WCluster`). The flow gate finder follows Prof. Javier Cabrera's Python translation of his R code. See [third-party notices](https://github.com/ron2k1/datanuggets/blob/main/THIRD_PARTY.md) for upstream credits, permission records, and the separate CC-BY-4.0 license for flow test data in the source archive.
