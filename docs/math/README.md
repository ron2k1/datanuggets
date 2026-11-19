# Mathematical reference

Draft. This page has not yet been reviewed by the authors of the R packages.

These pages state what each function in `datanuggets.__all__` computes, as equations, and trace each step to the Python code and to its source. They do not cover the modules `datanuggets.plot`, `datanuggets.projection`, `datanuggets.colors` and `datanuggets.filters`. The sources are the R packages datanugget 1.5.0 and WCluster 1.3.0 running on R 4.6.1, and Beavers et al. (2025). Where the paper and an R package differ, the port follows the R package.

## Pages

| Page | What it covers |
| --- | --- |
| [create-dn.md](create-dn.md) | `create_dn` and `create_dn_centers`: the chunks, the two deletion stages, the assignment, and each nugget's center, weight and scale. |
| [refine-dn.md](refine-dn.md) | `refine_dn`: the EV threshold, the EV pass that splits nuggets with a large first eigenvalue in two, and the shape pass that splits elongated or spread-out nuggets. |
| [hartigan-wong.md](hartigan-wong.md) | The two-cluster k-means that `refine_dn` runs for each split, as R's `stats::kmeans` runs it. |
| [nugget-means-and-complexity.md](nugget-means-and-complexity.md) | `get_dn_means` and `arithmetic_complexity`: the mean of the rows of each nugget and a cost model for `create_dn` on the $`\log_{10}`$ scale. |
| [weighted-statistics.md](weighted-statistics.md) | `wmean`, `wss`, `wwcss` and `distw`: the weighted mean and sum of squares, the WWCSS and the merge cost of Ward's method. |
| [weighted-kmeans.md](weighted-kmeans.md) | `wkmeans`, `dn_wkmeans`, `cluster_predict` and `dn_cluster_predict`. |
| [weighted-ward.md](weighted-ward.md) | `whclust`, `dn_whclust`, `cut_tree` and `WhclustResult.to_linkage`. |
| [weighted-pca.md](weighted-pca.md) | `wpca` and `dn_wpca`. |
| [randomness.md](randomness.md) | Integer seeds, NumPy generators and `RDraws`, which replays R's `set.seed` and `sample.int`. |
| [numerical-parity.md](numerical-parity.md) | The R arithmetic the port copies: long double sums, means, `cov`, quantiles, matrix products, `eigen` and Rfast distances. |

## Notation

These symbols are shared by the pages. Each page's Notation section adds its own symbols, and a page that uses one of these letters in another sense says so there.

| Symbol | Meaning |
| --- | --- |
| $`n`$ | the number of rows of the input |
| $`p`$ | the number of columns (variables) |
| $`m`$ | the number of nuggets |
| $`k`$ | the number of clusters |
| $`i`$, $`j`$, $`l`$, $`c`$ | a row, a column, a nugget and a cluster |
| $`X`$, $`\mathbf{x}_i`$, $`x_{ij}`$ | the $`n \times p`$ data, its row $`i`$ and its entry in row $`i`$ and column $`j`$ |
| $`w_i`$ | the weight of row $`i`$ |
| $`w_l`$ | the weight of nugget $`l`$ |
| $`\lvert S \rvert`$ | the number of elements of a set $`S`$ |
| $`\mathrm{fl}(y)`$ | $`y`$ rounded to a double |

Indices. The math counts from 1, as R does. The port's nugget ids, row assignments, cluster labels and cluster positions count from 0. So row $`i`$ is in nugget $`l`$ when `assignments[i - 1] == l - 1`. The `merge` matrix of `whclust` uses SciPy's numbering, which [weighted-ward.md](weighted-ward.md) gives.

Weights. `w=None` means a weight of 1 for every row. In `wpca`, `wrow=None` and `wcol=None` mean a weight of 1 for every row and for every column. A nugget's weight $`w_l`$ is the number of rows assigned to it. `create_dn` and `refine_dn` return it as an integer, and the weights sum to $`n`$. The `dn_` functions use the nugget centers as rows and the nugget weights as their row weights.

Random draws. An integer `seed` or a NumPy `Generator` gives NumPy's draws, not R's. `seed=RDraws(s)` gives the draws of R's `set.seed(s)`. The default seed of `create_dn`, `refine_dn`, `wkmeans` and `dn_wkmeans` is the integer 291102. The R functions have the same default. [randomness.md](randomness.md) gives the details.

Floating point. An equation without $`\mathrm{fl}`$ is exact real arithmetic. Where the bits of a result depend on the order of a sum or on long double, the page says so. [numerical-parity.md](numerical-parity.md) describes that arithmetic. A double has machine epsilon $`\varepsilon = 2^{-52}`$.

## How a page is laid out

Each page has these sections in this order: Scope, Notation, Definitions, Algorithm, Numerical notes, Tests and References. A last section, "Not covered by a test", lists statements that no test in `tests/` checks.

Code is cited by its path from the repository root and the function name. R code is cited by package, version and function name, and R's C and Fortran sources by file and routine. The paper is cited by section, algorithm and step.

The Tests section of each page names the tests that check it. The tests that compare the port with R read R's recorded results from `tests/fixtures/`. The fixtures were recorded with R 4.6.1, datanugget 1.5.0, WCluster 1.3.0 and Rfast 2.1.5.2 on x86_64-w64-mingw32. The parity statements describe which cases those fixtures cover.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`.
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0.
- Dey, R., Duan, Y., Cabrera, J., and Cheng, G. (2025). WCluster: Clustering and PCA with Weights, and Data Nuggets Clustering. R package version 1.3.0.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria.
