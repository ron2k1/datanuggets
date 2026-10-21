# datanuggets

**Summarize large datasets. Keep the weights. Work in Python.**

Data nuggets represent a dataset with a smaller set of points. Each nugget stores a center, the number of observations it represents, and a measure of spread. You can cluster these weighted points, explore them with PCA, and map cluster labels back to the original rows.

This is a native Python port of methods from R's `datanugget` 1.5.0 and `WCluster` 1.3.0. R is not required. See the [R-to-Python function map](https://github.com/ron2k1/datanuggets/blob/main/docs/api.md#r-to-python-function-map) for coverage; automatic nugget-count selection (`optimal.DN`) is not implemented.

[Getting started](https://github.com/ron2k1/datanuggets/blob/main/docs/quickstart.md) · [Function reference](https://github.com/ron2k1/datanuggets/blob/main/docs/api.md) · [Flow cytometry](https://github.com/ron2k1/datanuggets/blob/main/docs/cli.md) · [Limitations](https://github.com/ron2k1/datanuggets/blob/main/docs/limitations.md)

> **Research alpha — `0.1.0.dev0`.** The API may change. The repository currently requires access and the package is not on PyPI. The port is tested against recorded R cases, with [known differences](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md).

## Get started

### 1. Install

Use Python 3.12 or newer and Git. Create and activate a [virtual environment](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#choose-a-python-version), then install from `main`:

```sh
python -m pip install "datanuggets[frame,plot] @ git+https://github.com/ron2k1/datanuggets.git@main"
```

The extras add pandas and matplotlib. **Python 3.14 and 3.15 run the same package**; they are alternative interpreter versions, not separate editions of datanuggets. You only need one. Contributors testing both use separate environments, as explained in [installation](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#choose-a-python-version).

Prefer a ready-to-run example? [Clone the repository](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#from-a-local-checkout) and run `python examples/first_analysis.py`. It uses the included R `quakes` data and saves a plot and tables under `outputs/`.

### 2. Create nuggets

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

The experimental `datanuggets-flow` command finds candidate gates, lets you choose populations, and clusters their events across files. Install the `flow` extra to use it:

```sh
python -m pip install "datanuggets[flow] @ git+https://github.com/ron2k1/datanuggets.git@main"
datanuggets-flow path/to/fcs_folder
```

Figures and tables go into a new `outputs/flow-<timestamp>/` folder for each run. Run from the repository root to keep them together there; `outputs/` is ignored by Git. You can choose another new folder with `--out`; existing paths are refused.

Inspect the gates, choose at least two fluorescence channels, and check your compensation and preprocessing: the tool does not apply compensation. See the [CLI guide](https://github.com/ron2k1/datanuggets/blob/main/docs/cli.md) before using it on a study.

## Know the limits

- **Memory still matters.** Hierarchical clustering uses a square matrix: 10,000 pooled nuggets need about 800 MB for that matrix alone.
- **Reduction changes the analysis.** Compare nugget counts and preprocessing choices before interpreting clusters. Cluster numbers are not validated biological identities.
- **R results are not universally identical.** Random streams, eigen calculations, and some edge cases differ. Read the [compatibility notes](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md).

See [limitations and open problems](https://github.com/ron2k1/datanuggets/blob/main/docs/limitations.md), [methodology](https://github.com/ron2k1/datanuggets/blob/main/docs/methodology.md), [reproducibility](https://github.com/ron2k1/datanuggets/blob/main/docs/reproducibility.md), or the [contributor guide](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) for details.

## Citation and credits

Beavers, T. E., et al. *Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure.* Journal of Computational and Graphical Statistics **34**(1), 330–342 (2025; online 2024). DOI: `10.1080/10618600.2024.2341896`.

Licensed under [GPL-2.0-only](https://github.com/ron2k1/datanuggets/blob/main/LICENSE). The R algorithms are the work of Rituparna Dey, Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi, and Mariusz Lubomirski (`datanugget`); and Rituparna Dey, Yajie Duan, Javier Cabrera, and Ge Cheng (`WCluster`). The flow gate finder follows Prof. Javier Cabrera's Python translation of his R code.
