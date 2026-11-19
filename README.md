# datanuggets

Data nuggets bring a large dataset down to a size you can explore with weighted clustering and PCA. This package implements the method in Python, following the R packages [`datanugget`](https://CRAN.R-project.org/package=datanugget) and [`WCluster`](https://CRAN.R-project.org/package=WCluster).

A million observations can be hard to work with even when there are only a handful of variables. Hierarchical clustering needs pairwise distances and quickly runs into memory limits. Taking a random sample is easier, but a small sample can miss a rare group or the edges of a distribution.

Data nuggets summarize all the rows as a much smaller collection of weighted points. Each nugget has:

- a center, which is its location
- a weight, which counts the original observations that belong to it
- a scale, which describes the variation among those observations

The package keeps track of which nugget each original row belongs to, so you can cluster the nuggets and carry the cluster labels back to the full dataset. Reduction isn't lossless, though. You still need to check whether the structures you care about survive at the nugget count you choose.

This is a native Python port of 17 functions from R's `datanugget` 1.5.0 and `WCluster` 1.3.0, plus `stats::cutree`. You don't need R to use it.

> **Research alpha: `0.1.0a3`.** The API may change. Results are checked against recorded R reference cases, but there are [documented differences](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md). Use the [release notes](https://github.com/ron2k1/datanuggets/blob/main/docs/releases.md) if you need the exact scope of this version.

[Python walkthrough](https://github.com/ron2k1/datanuggets/blob/main/docs/quickstart.md) · [API reference](https://github.com/ron2k1/datanuggets/blob/main/docs/api.md) · [Methodology](https://github.com/ron2k1/datanuggets/blob/main/docs/methodology.md) · [Limitations](https://github.com/ron2k1/datanuggets/blob/main/docs/limitations.md)

## Quick start

### Install

Use Python 3.12 or newer. The package installs on Windows x64, Apple silicon Macs running macOS 12 or newer, and Linux x86-64 or ARM64 with glibc 2.27 or newer. Numba doesn't publish wheels for Intel Macs or Alpine Linux, so the normal install doesn't support those systems. See the [installation guide](https://github.com/ron2k1/datanuggets/blob/main/docs/installation.md#supported-platforms) for details.

Create a virtual environment, then install the published alpha:

```sh
python -m pip install "datanuggets[frame,plot]==0.1.0a3"
```

The `frame` and `plot` extras install pandas and matplotlib. The version is pinned on purpose. `0.1.0a3` is a pre-release, and naming it tells pip to install that release.

If you want the latest code on `main`, which may differ from the PyPI release:

```sh
python -m pip install "datanuggets[frame,plot] @ git+https://github.com/ron2k1/datanuggets.git@main"
```

There's also a complete example in [`examples/first_analysis.py`](https://github.com/ron2k1/datanuggets/blob/main/examples/first_analysis.py). It runs from a clone of the repository, uses the recorded R `quakes` data and saves a plot and two tables under `outputs/`.

### Make nuggets from a CSV

Pick numeric columns that describe the observations. Leave out IDs and known outcome labels. Here's a two-feature example. Replace the column names with your own.

```python
import numpy as np
import pandas as pd
from datanuggets import create_dn

df = pd.read_csv("my_data.csv")
features = ["feature_a", "feature_b"]
x = df[features].to_numpy(dtype=float)

if len(x) < 2 or not np.isfinite(x).all():
    raise ValueError("Expected at least two rows of finite numeric data.")

# Put these features on comparable scales before using distances.
mean = x.mean(axis=0)
scale = x.std(axis=0)
scale[scale == 0] = 1
x = (x - mean) / scale

dn = create_dn(x, dn_num1=2_000, dn_num2=200, seed=42)
print(f"{len(x):,} rows -> {dn.n_nuggets:,} nuggets")
print(dn.to_frame().head())
```

`dn_num1` sets the intermediate target and `dn_num2` the final target. They're starting values for this example, and they won't suit every dataset. If a target is at least as large as the data available at that stage, the function may warn and skip that reduction. Check the number of nuggets it returns.

The result contains `dn.centers`, `dn.weights`, `dn.scales` and `dn.assignments`. The last array has one nugget index for every input row. Keep your original row order if you plan to attach cluster labels to the DataFrame later.

### Cluster the nuggets

Weighted k-means clusters the nugget centers and counts each one by its weight:

```python
from datanuggets import dn_wkmeans

km = dn_wkmeans(dn, k=5, num_init=10, max_iter=100, seed=42)
df["cluster"] = km.row_labels
print(df["cluster"].value_counts().sort_index())
```

`km.labels` labels the nuggets. `km.row_labels` gives the labels for the original rows, using their nugget assignments.

For a weighted Ward hierarchy, build a tree over the nuggets and choose where to cut it:

```python
from datanuggets import cut_tree, dn_whclust

if dn.n_nuggets < 5 or np.any(dn.weights == 0):
    raise ValueError("Ward clustering needs at least five occupied nuggets here.")

tree = dn_whclust(dn)
nugget_labels = cut_tree(tree, k=5)
ward_row_labels = nugget_labels[dn.assignments]
```

Or run weighted PCA on the nugget centers:

```python
from datanuggets import dn_wpca

pca = dn_wpca(dn)
print(pca.sdev)     # Principal-component standard deviations
print(pca.x.shape)  # One score row per nugget
```

You can also call `refine_dn(x, dn, ...)` before these analyses to split nuggets that are unusually spread out. Refinement can raise the nugget count. The [walkthrough](https://github.com/ron2k1/datanuggets/blob/main/docs/quickstart.md) covers refinement, plots and assigning new observations to existing clusters.

## How the reduction works

The method is from Beavers and colleagues. It selects a smaller, well-spaced set of starting points. Those points aren't k-means centers of the full dataset. On data with more rows than the chunk size, the method never computes the distance between every pair of original observations.

`create_dn` shuffles the rows and divides them into chunks of nearly equal size. The chunk size `r` is 5,000 by default, and data with no more rows than `r` stays in one chunk. Within each chunk, it repeatedly removes observations that have very close neighbors. It then pools the survivors, reduces them again and assigns every original row to its nearest surviving center. With the default `center_method="mean"`, each final center is replaced by the mean of its assigned observations.

This tends to remove more points from dense parts of the data and keep points in sparse regions, including the tails. That's why the method can keep details that a random sample may miss. It's still a heuristic. The result depends on the random seed and the reduction settings, and it doesn't guarantee that every rare group is recovered.

A nugget's scale is the trace of the sample covariance of its rows divided by the number of features. Nuggets with fewer than two rows are handled separately. `refine_dn` looks at covariance eigenvalues and splits selected nuggets using two-means. It can also split elongated nuggets when `shape_split=True`.

Weighted k-means, Ward clustering and PCA use the nugget centers and weights. The scales describe the variation inside each nugget. The weighted methods don't use them, so they don't reconstruct all the variation lost in compression.

For the exact chunking, deletion, tie-breaking and refinement rules, see the [methodology guide](https://github.com/ron2k1/datanuggets/blob/main/docs/methodology.md#creating-nuggets). The port follows the R implementation where it differs from the published algorithm.

## What comes from the research

The original [data nuggets paper](https://doi.org/10.1080/10618600.2024.2341896) develops the reduction method and runs PCA and k-means clustering for weighted observations on the nuggets. One application starts with 1,048,575 flow cytometry observations of nine variables, creates 2,000 nuggets and refines them to 3,135. In the paper's comparison, the refined nuggets recovered the structure of the first three principal components, and a random sample of 3,000 rows didn't recover it well. This package doesn't reproduce that comparison.

A [second paper](https://doi.org/10.1007/978-3-031-22687-8_20) uses data nuggets for linear regression, and weighted least squares on the nugget summaries is one of its estimators. The original paper also names projection pursuit as a method that could be applied to data nuggets. This port implements neither nugget regression nor projection pursuit.

For what this package implements, see the function map below and the [API reference](https://github.com/ron2k1/datanuggets/blob/main/docs/api.md).

## R and Python names

R function names become snake_case in Python. The Python results are objects with named fields rather than R lists. Indices and labels are zero-based rather than one-based.

| R | Python |
| --- | --- |
| `datanugget::create.DN` | `create_dn` |
| `datanugget::refine.DN` | `refine_dn` |
| `datanugget::create.DNcenters` | `create_dn_centers` |
| `datanugget::getDN.means` | `get_dn_means` |
| `datanugget::AC` | `arithmetic_complexity` |
| `WCluster::Wkmeans` | `wkmeans` |
| `WCluster::Whclust` | `whclust` |
| `WCluster::Wpca` | `wpca` |
| `WCluster::cluster.predict` | `cluster_predict` |
| `WCluster::wmean` | `wmean` |
| `WCluster::wss` | `wss` |
| `WCluster::wwcss` | `wwcss` |
| `WCluster::distw` | `distw` |
| `WCluster::DN.Wkmeans` | `dn_wkmeans` |
| `WCluster::DN.Whclust` | `dn_whclust` |
| `WCluster::DN.Wpca` | `dn_wpca` |
| `WCluster::DNcluster.predict` | `dn_cluster_predict` |
| `stats::cutree` | `cut_tree` |

`datanugget::optimal.DN` isn't ported. It selects a nugget count by running the method at candidate sizes and comparing propensity score indices.

A few call signatures differ from R's on purpose. `create_dn_centers` and `cluster_predict` take their arguments in a different order, and `wkmeans` and `dn_wkmeans` take every option after `k` by keyword. Some arguments are renamed too, such as `no.cores` to `n_jobs`. Read the [compatibility notes](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md#differences-from-r) before translating an R script line by line. They also show [the same pipeline in R and in Python](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md#the-same-pipeline-in-r-and-python).

### Reproducing R's random draws

An integer seed draws from NumPy, so it doesn't give the draws R makes after `set.seed()`. To get R's draws, pass `RDraws`:

```python
from datanuggets import RDraws, create_dn

dn = create_dn(x, dn_num1=2_000, dn_num2=200, seed=RDraws(291102))
```

That helps when comparing the two implementations, but it doesn't make every result identical. Eigen calculations, some exact distance ties and some edge cases can still differ. The [compatibility notes](https://github.com/ron2k1/datanuggets/blob/main/docs/compatibility.md) say what was checked and what still differs.

## Things to check before trusting a result

- **Try more than one seed and nugget count.** Nugget creation is a heuristic. Weighted k-means also depends on its random starts, and Ward clustering never revisits a merge.
- **Look for lost structure.** Compare the reduced data with the original where you can. Small or distant groups are a reason to try data nuggets, and the method can still miss them.
- **Keep the weights.** Ignoring them changes what the clustering and the PCA mean. A cluster number is an arbitrary label, and it doesn't identify a known class.
- **Watch memory use.** When the data has more rows than the chunk size, reduction avoids a full pairwise matrix over the original rows. It still builds a distance matrix within each chunk and one over the intermediate centers. Ward clustering is still quadratic in the number of nuggets. A dense 10,000 by 10,000 `float64` matrix alone takes about 800 MB.
- **Record your environment.** Keep the package version, input preprocessing, seed and nugget settings with the analysis. This is a research alpha, and exact R parity isn't guaranteed on every machine or dataset.

See [limitations and open problems](https://github.com/ron2k1/datanuggets/blob/main/docs/limitations.md), [reproducibility](https://github.com/ron2k1/datanuggets/blob/main/docs/reproducibility.md) and the [contributor guide](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) for more detail.

## Papers and credits

If you use the data nuggets method, please cite the original research:

Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. (2025). Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure. *Journal of Computational and Graphical Statistics*, 34(1), 330-342. Published online in 2024. [doi:10.1080/10618600.2024.2341896](https://doi.org/10.1080/10618600.2024.2341896).

Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. (2023). Data Nuggets in Supervised Learning. In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate Statistical Methods: Festschrift in Honor of David E. Tyler* (pp. 429-449). Springer. Published online in 2022. [doi:10.1007/978-3-031-22687-8_20](https://doi.org/10.1007/978-3-031-22687-8_20).

The methods come from the R packages `datanugget` and `WCluster`. Those packages are the work of Rituparna Dey, Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi and Mariusz Lubomirski (`datanugget`), and of Rituparna Dey, Yajie Duan, Javier Cabrera and Ge Cheng (`WCluster`). Rituparna Dey maintains both R packages. The [third-party notices](https://github.com/ron2k1/datanuggets/blob/main/THIRD_PARTY.md) give the upstream credits and the release approval. This Python port is distributed separately under [GPL-2.0-only](https://github.com/ron2k1/datanuggets/blob/main/LICENSE) and claims no authorship of the original algorithms.
