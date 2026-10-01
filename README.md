# datanuggets

A native Python port of CRAN's `datanugget` 1.5.0 and `WCluster` 1.3.0. It runs without R. Its tests check it against 482 cases recorded from R itself and a few values checked by hand in R, and the port matches R bit for bit on them except for the eigen solver's output and the differences documented below.

Data nuggets reduce a dataset to groups described by a center, a row count and a scale. Scale measures spread as the average variance across columns. Centers are chosen by repeatedly removing one point from the closest pair, which spreads them across the data instead of concentrating them in dense regions. Each row joins its nearest center.

Beavers et al. (cited below) run weighted PCA and weighted k-means on the nuggets instead of the full dataset. Their flow cytometry example has over one million observations.

## Status

Version `0.1.0.dev0` is in development. Eleven functions are ported:

| Python | R | R package |
| --- | --- | --- |
| `create_dn` | `create.DN` | `datanugget` |
| `create_dn_centers` | `create.DNcenters` | `datanugget` |
| `refine_dn` | `refine.DN` | `datanugget` |
| `get_dn_means` | `getDN.means` | `datanugget` |
| `arithmetic_complexity` | `AC` | `datanugget` |
| `wmean` | `wmean` | `WCluster` |
| `wss` | `wss` | `WCluster` |
| `wwcss` | `wwcss` | `WCluster` |
| `distw` | `distw` | `WCluster` |
| `wpca` | `Wpca` | `WCluster` |
| `cluster_predict` | `cluster.predict` | `WCluster` |

`refine_dn` splits nuggets that are spread out or elongated with 2-means, through a port of the Hartigan-Wong k-means in R's `stats::kmeans`. `get_dn_means` averages the rows in each nugget. `arithmetic_complexity` gives the arithmetic cost of a `create_dn` call on a log10 scale, from the shape of the data alone.

The `WCluster` functions treat each row as a point with a weight, such as a nugget with its row count. `wmean` gives each cluster's weighted mean and `wss` the weighted sum of squares around the weighted mean. `wwcss` adds up each cluster's `wss`, and `distw` gives how much merging two clusters adds to it. `cluster_predict` sends new rows to the nearest weighted cluster mean. `wpca` is principal components with a weight per row and per column.

`RDraws` reproduces R's `set.seed` and `sample.int`. Passed as `seed=RDraws(291102)`, it makes `create_dn` and `refine_dn` draw what R's `create.DN` and `refine.DN` draw with that seed. It isn't exported from `datanuggets` yet, so import it from `datanuggets._random.r_draws`.

Still to port: `WCluster`'s `Wkmeans` and `Whclust`, then `datanugget`'s `DN.Wkmeans`, `DN.Whclust` and `optimal.DN`.

## Install

Requires Python 3.12 or newer, `numpy >= 2.0`, `scipy >= 1.13`, `joblib >= 1.4` and `numba >= 0.67`.

It isn't on PyPI. `pip install datanugget`, without the s, gets an unrelated package of code snippets.

```sh
pip install "git+https://github.com/ron2k1/datanuggets.git"
```

Or from a clone:

```sh
git clone https://github.com/ron2k1/datanuggets.git
cd datanuggets
uv sync
```

The `frame` extra adds pandas for `DataNuggets.to_frame()`. The `progress` extra adds tqdm for `progress=True`.

## Example

The table output needs the `frame` extra.

```python
import numpy as np
from datanuggets import create_dn

rng = np.random.default_rng(0)
x = rng.normal(size=(100_000, 3))

dn = create_dn(x, dn_num1=2000, dn_num2=200)
print(dn.n_nuggets)  # 200
print(dn.weights.sum())  # 100000
print(dn.to_frame().head())
```

Output of `to_frame().head()`:

```text
   Data Nugget   Center1   Center2   Center3  Weight     Scale
0            0 -3.230698  2.430919 -1.423236       1  0.000331
1            1  3.265895  0.927756  1.551690       6  0.041483
2            2  4.144727 -1.874403 -0.815790       1  0.000331
3            3  1.614533  3.945550  1.704140       1  0.000331
4            4  3.074095 -1.813385  0.944272       3  0.033129
```

Those weight-1 nuggets are expected. Closest-pair deletion keeps points far out in the tails as centers. They share the scale 0.000331 because a nugget with fewer than 2 rows gets the smallest scale among nuggets with 2 or more rows, divided by 100. R does the same. The call took 14.6 seconds on my laptop (i7-13650HX).

The `WCluster` functions take the nuggets' centers as the data and their weights as row weights. Continuing the example, this runs weighted PCA and splits the nuggets by the sign of their first coordinate:

```python
from datanuggets import cluster_predict, wmean, wpca, wwcss

pca = wpca(dn.centers, wrow=dn.weights)
print(pca.sdev)

cl = (dn.centers[:, 0] > 0).astype(int)
print(wmean(dn.centers, cl, dn.weights))
print(wwcss(dn.centers, cl, dn.weights))
print(cluster_predict(dn.centers, cl, x[:5], w=dn.weights))
```

Output:

```text
[0.95152184 0.94524503 0.94031221]
[[-0.7682159  -0.02598546  0.21580461]
 [ 0.71713231  0.01851018 -0.19804108]]
208903.8865164328
[0 1 1 0 0]
```

`cluster_predict` returns each new row's 0-based position among the sorted labels. The labels here are 0 and 1, so the positions equal the labels.

## What was done

The work landed in the order below. Most steps started from a written plan, linked in the table. Steps 4 and 5 also had an independent review, and its findings were fixed or recorded. `RDraws` grew out of a spike and has no plan document, and the architecture document was written after the moves it describes.

| Step | Record | What it added |
| --- | --- | --- |
| 1 | PR #1, [plan](docs/plans/2026-09-29-feature1-create-dn.md) | `create_dn` and `create_dn_centers`. Rfast's distances are replayed with Armadillo's two running sums over even and odd columns, and R's `colMeans` and `rowMeans` in their 80-bit long double |
| 2 | PR #2, [plan](docs/plans/2026-09-29-feature2-refine-dn.md) | `refine_dn`, with a statement-by-statement port of the Hartigan-Wong `kmns` Fortran behind `stats::kmeans`, and R's `cov`, `mean`, `median`, `mad` and `quantile` with long double sums |
| 3 | PR #3, [plan](docs/plans/2026-09-29-feature3-dn-helpers.md) | `get_dn_means`, and `arithmetic_complexity`, which repeats `AC`'s double arithmetic step by step |
| 4 | [plan](docs/plans/2026-09-30-speed-compiled-long-double.md), [review](docs/reviews/2026-09-30-speed-fixpass-review.md) | The long double replay moved from Python integers into numba kernels that give the same bits. `refine_dn` gets every nugget's center and covariance from one compiled call, threaded when `n_jobs` isn't 1 |
| 5 | PR #4, [plan](docs/plans/2026-09-30-feature4-wcluster-leaves.md), [review](docs/reviews/2026-09-30-wcluster-leaves-review.md) | The six `WCluster` functions, with the reference BLAS order of R's `%*%`, `weighted.mean` and `sum` in long double, and a tolerance rule for the output of `eigen()` |
| 6 | PR #5 | `RDraws`: R's Mersenne-Twister seeding and `sample.int`, including the hash sampler R uses when it draws at most half of more than 1e7 values |
| 7 | [plan](docs/plans/2026-09-30-r-compat.md) | A decision record for every known difference from R, backed by 46 new recorded cases. The first four fixes: R's -NaN for an empty nugget, logical data read as 0 and 1, a vector `x` read as one column, and R's NA bits in `distw` |
| 8 | [plan](docs/plans/2026-09-30-eigen-reference-solver.md) | A measured spike with no code shipped: R's own LAPACK reproduces R's eigen output on every input tried, where scipy's doesn't. It sets out the options for a reference solver |
| 9 | [architecture](docs/architecture.md) | The package split into `nuggets/`, `wcluster/`, `_compat/`, `_random/` and `_clustering/`, with the tests in matching folders. The five old public module paths and the private `datanuggets._draws` still import. The other moved private modules are reachable only at their new paths |

PRs #4 and #5 were opened against the speed branch, and both reached `main` together through the merge of this branch.

## Parity with R

Every case in the `create_dn` parity table matches R bit for bit in all four outputs: assignments, weights, centers and scales. The cases run iris, faithful and quakes under both metrics and all three center methods. Edge cases add a one-column dataset, tied grids, duplicate rows and a 61-column scaled dataset. Tests replay the exact random draws recorded from R and compare with no tolerance. The full table is in [docs/validation/create_dn.md](docs/validation/create_dn.md).

`refine_dn` is checked the same way on its own grid. On faithful it runs every combination of `ev_tol`, `max_splits`, `min_nugget_size`, `shape_split`, `delta` and `nstart`. Six more datasets run four settings each: iris, quakes, the petal columns of iris, a one-column dataset, the scaled volcano data and a tied grid. More cases cover refining a refined result, an empty input nugget and splits that fail. Every kmeans call clusters the same rows as R's and returns the same clusters, and the outputs match bit for bit.

Shape splits cluster the rows projected on the first eigenvector. Those projections and the eigenvalues that decide which nuggets get a split attempt come from LAPACK. They're compared with NumPy's `assert_allclose` using `rtol=1e-8` and `atol=1e-10`. The eigenvalues differ by rounding on the scale of the largest eigenvalue. A second eigenvalue near zero can have a large relative error while staying within `1e-10` absolute. No split decision in the fixture grid changes because of that rounding. The table is in [docs/validation/refine_dn.md](docs/validation/refine_dn.md).

`get_dn_means` runs on every nugget set in those two fixture grids: each `create.DN` output, and each input and output of `refine.DN`. That's 293 sets, two of them with an empty nugget and nine with one column. All 293 match R bit for bit. An empty nugget's row holds the NaN that R's long double 0/0 gives, with its sign bit set (`fff8000000000000`), and the tests compare those bits. Reversing each nugget's rows changes none of their means, so one constructed case, recorded from R, pins the order of R's long-double sums. The table is in [docs/validation/get_dn_means.md](docs/validation/get_dn_means.md).

`arithmetic_complexity` runs on the row and column counts of eight R datasets, crossed with values of `r`, `delete_percent`, `dn_num1` and `dn_num2`. That makes 3481 calls. Whenever R's `AC` returns, the pass counts traced inside it equal the port's, and the value matches bit for bit. The tests still allow a relative difference of `1e-8` and an absolute one of `1e-10` in the value, since it goes through the platform's `log10`. Running the first loop in exact arithmetic would get 289 of its counts wrong, so the port repeats R's double arithmetic step by step. R never returned from 19 of the calls, and the port raises on exactly those 19. The table is in [docs/validation/arithmetic_complexity.md](docs/validation/arithmetic_complexity.md).

`wmean`, `wss`, `wwcss`, `distw` and `cluster_predict` share one set of cases recorded from R. The data comes from iris, faithful, quakes, mtcars, volcano, ChickWeight, EuStockMarkets, USArrests, PlantGrowth, rock, LifeCycleSavings and infert, and from nuggets R's `create.DN` made of iris, quakes and scaled volcano. Weights are ones or a real column, such as the nugget weights. Labels come from factors, cuts of a real column, Ward clusterings and one cluster per row. Some clusters have weights that sum to zero, and some labels are negative. `wmean` matches R bit for bit in all 49 cases, which run from 25 to 1860 rows, up to 61 columns and up to 200 clusters. The table is in [docs/validation/wmean.md](docs/validation/wmean.md).

`wss` matches R bit for bit on the same 49 cases. R returns a vector of length one, so its only entry is compared. The table is in [docs/validation/wss.md](docs/validation/wss.md).

`wwcss` matches R bit for bit in all 49 cases. That holds for the per-cluster values and the total that `group_sum=True` returns, and for the plain total. The table is in [docs/validation/wwcss.md](docs/validation/wwcss.md).

`distw` needs two clusters, so it runs on the 45 of those cases that have more than one. The whole matrix matches R in all 45. Every entry R computes matches bit for bit, including entries at zero or just below it. Each cell R leaves NA holds R's NA in the port too, the NaN with payload 1954 (`7ff00000000007a2`), so `np.isnan` finds it and a bit-level comparison matches R. The table is in [docs/validation/distw.md](docs/validation/distw.md).

`cluster_predict` runs on the 39 cases recorded with new rows to place, 6,422 rows in all. In every case the squared distances from R's centers match R's, each number bit for bit and each NaN in the same place. The positions match too, compared with R's 1-based ones minus one. Ties go to the first cluster in sorted label order, as in R. The table is in [docs/validation/cluster_predict.md](docs/validation/cluster_predict.md).

The tie and heavy-weight cases are constructed, so they're tested and not reported. R recorded them in `tests/fixtures/wstats_constructed/` like the rest. The ties are midpoints between two cluster centers and one- and two-ulp nudges of them. The heavy weights are powers of two of a real column, or a real column scaled by 1e296 or into the subnormal range. A few more cases hold rows whose squares overflow.

Logical data and vectors follow R too. In the `WCluster` functions and `get_dn_means`, TRUE and FALSE count as 1 and 0. `create_dn` and `refine_dn` raise R's "x must contain only numeric columns" on an all-logical matrix. `create_dn_centers` raises "rs must contain only numeric columns" there, where R's message says "RS", the name of its argument. A data frame mixing logical and numeric columns gives the same nuggets as the numeric matrix, as `as.matrix()` makes it in R. A vector `x` is one column in the six `WCluster` functions, and the recorded faithful cases show R returning on every one of them.

`wpca` has 29 recorded cases, from iris, mtcars, faithful, quakes, volcano, trees and npk, from the same nuggets, and five constructed weight vectors that pin R's long double `sum(wrow)`, two of which make R stop. The table reports 25 of them, 4 with `corr=True`. Everything up to the matrix R hands to `eigen()` matches R bit for bit in all 25: the center, the scale, the column and row weights, the centered data and the eigen input. The eigen output isn't compared bit for bit, because LAPACK builds round it differently. A rule holds it to R's instead, and passes in all 25. The rotation must be orthonormal, and each column must solve the eigen input for R's eigenvalue within 1024 machine epsilons times the largest eigenvalue. Eigenvalues closer than 1e-12 times the largest form a cluster whose subspace has to match R's, since the vectors inside it aren't determined. The scores must be R's matrix product of the centered data and the port's rotation. Four cases aren't reported. R stops in `eigen()` on three of them, and the port raises there too. The fourth is the one-column `corr=True` case, where the port departs from R on purpose. The table is in [docs/validation/wpca.md](docs/validation/wpca.md).

`RDraws` is checked on 21 recorded cases of `set.seed` and `sample.int`, with `.Random.seed` read before and after each call, so the state it leaves and the draws that follow are compared too. The cases cover both sides of R's switch to the hash sampler, at 1e7 and 1e7 + 1 values. It also reproduces every draw recorded in the `create_dn` fixtures, and `refine_dn` with `RDraws` returns R's nuggets on all 128 recorded `refine.DN` cases. Two workers give the same result as one.

Getting the last bit right meant copying how R adds things up. Rfast keeps two running sums for distance terms, one over even columns and one over odd. R's `colMeans` and `rowMeans` add in 80-bit x87 precision before rounding to double. The port repeats that arithmetic exactly in 64-bit integer code that numba compiles. `refine_dn` needs R's `cov`, `mean`, `median`, `mad` and `quantile` too, which also add in long double. The `WCluster` functions add through R's `%*%`, which the reference BLAS sums in its own order, and through `weighted.mean` and `sum`, which add in long double.

Parity is against R 4.6.1 on x86-64 Windows with `datanugget` 1.5.0, `WCluster` 1.3.0, Rfast 2.1.5.2, R's reference BLAS and R's internal LAPACK 3.12.1. The port doesn't use the host's long double, so it gives those x86-64 results on arm64 macOS as well, where the same tests pass. Two things still come from the host. The eigen output comes from the installed scipy, and a NaN that `distw` computes in its lower triangle carries another sign bit on arm64, so those cells are compared with `same_values`. R itself is another matter. OpenBLAS, MKL and Accelerate can round the last bits differently, and so can R on arm64 macOS, where `long double` is plain double. On tied data, those differences can change assignments. The fixture manifests record the BLAS and `long double` width.

## Regression testing

### What R recorded

Every R result the tests compare with comes from R itself. A few were checked by hand in an R session and written into the tests, such as the cancelling-weight cases in `tests/wcluster/test_wstats.py` and `arithmetic_complexity` on an `x` with no rows or no columns. The long double fuzz below checks against a Python reference instead of R. Everything else comes from the scripts in `r/`. They run R 4.6.1 with the pinned packages, trace the functions under test without modifying them, and write the inputs, the outputs, every random draw and the internal values that matter into `tests/fixtures/`. Each case carries a manifest with the R version, the package versions, the BLAS and the long double width. Running the tests doesn't need R. Fixtures are recorded on one machine only, the laptop with the pinned R. There are 482 cases in 18 groups:

| Group | Script in `r/` | Cases | What R recorded |
| --- | --- | --- | --- |
| `create_dn` | `fixtures_create_dn.R` | 37 | `create.DN` inputs, every random draw, every `create.DNcenters` call, and the outputs |
| `refine_dn` | `fixtures_refine_dn.R` | 128 | the input nuggets, every draw, every `kmeans` call with the rows it clustered, the messages and warnings, and the refined nuggets |
| `get_dn_means` | `fixtures_get_dn_means.R` | 4 | `getDN.means` on every nugget set the two groups above hold, and on one constructed set that pins the order of its sums |
| `ac` | `fixtures_ac.R` | 1 | 3481 `AC` calls with the loop state each returns with, and the 19 that never return |
| `kmeans`, `kmns` | `fixtures_kmeans.R` | 13, 15 | `stats::kmeans` end to end with its draws, and R's own `kmns` Fortran from fixed starting centers |
| `rcompat_dist`, `rcompat_dista`, `rcompat_rowmins`, `rcompat_split_indices`, `rcompat_means` | `fixtures_rcompat.R` | 10, 12, 1, 1, 8 | the Rfast and base R operations `create.DN` depends on |
| `rcompat_stats` | `fixtures_rstats.R` | 54 | `cov`, `sum`, `mean`, `median`, `mad`, type-7 `quantile`, `unique()` on rows, `%*%` and `eigen(symmetric = TRUE)` |
| `rcompat_wstats` | `fixtures_rwstats.R` | 36 | `%*%` under the reference BLAS, `weighted.mean`, and vectors that pin the order of R's long double `sum()` and where it overflows |
| `wstats`, `wstats_constructed` | `fixtures_wstats.R` | 49, 17 | the five `WCluster` statistics on real data, and on built ties, heavy weights and overflowing rows |
| `wpca` | `fixtures_wpca.R` | 29 | `Wpca`'s outputs and the locals it hands to `eigen()` |
| `rsample` | `fixtures_rsample.R` | 21 | `set.seed` and `sample.int`, with `.Random.seed` before and after |
| `compat` | `fixtures_compat.R` | 46 | the evidence behind the R-compatibility decisions: labels, logical and vector data, predictions with no answer, `corr` values, `nstart` and fractional nugget counts |

### How the tests compare

Exact comparisons come in three strengths. `same_bits` compares every float64 bit, NaN payloads included. `same_values` compares the bits too, so -0.0 and 0.0 differ, but lets any NaN match any NaN. NumPy's `assert_array_equal` treats -0.0 and 0.0 as equal.

The `WCluster`, compat, R primitive and `get_dn_means` tests use `same_bits` and `same_values`, with `same_bits` wherever R's NaN is part of the answer: an empty nugget's -NaN, `distw`'s NA cells, and R's long double primitives. The `create_dn`, `create_dn_centers`, `refine_dn`, `kmeans` and Rfast tests, and the `RDraws` replays of them, use `assert_array_equal`. For `create_dn` and `refine_dn` the bit-for-bit claim rests on the parity tables, which `scripts/parity_report.py` builds with `same_values`.

Tolerances appear only where LAPACK or the platform's `log10` is involved. `wpca`'s eigen output is held by the rule above, which fails closed, so a NaN or an infinity in the port's output fails it. `refine_dn`'s eigenvalues and projections, the eigen routine's own tests and `AC`'s value use `rtol=1e-8` and `atol=1e-10`.

Parity tests feed R's recorded draws back through `ReplayDraws`. It raises if the port asks for a draw R didn't make, asks with different arguments, or leaves one unread, so the tests check the order of the calls as well as the numbers.

The compiled long double kernels are checked against a slower reference written with Python integers, on a fixed-seed fuzz aimed at exact ties, sticky bits and exponent edges. The default run tries 200,000 cases per operation. The slow run, `pytest -m slow`, tries 10 million per operation for each of the seven operations. Another test finds every compiled module by itself, checks that each kernel is made through the package's shared `compiled` wrapper, and checks that the kernels cache where numba can and still compile where it can't.

Several tests target a specific wrong implementation and were seen to fail on it: `math.fsum` in place of R's long double `sum(wrow)` in `wpca`, numpy's +NaN in place of R's -NaN for an empty nugget, `RDraws` taking the wrong side of R's hash switch, and a matrix product that returns -0.0 where R's starts from +0. Reviews also ran mutants against the suite and turned most survivors into tests. The [WCluster review](docs/reviews/2026-09-30-wcluster-leaves-review.md) lists the ones it ran, and the survivors are under Open problems.

The package layout has its own tests: a bare `import datanuggets` in a fresh interpreter still binds the old module paths, and a pickle naming an old path still loads.

### Results at the merge

The suite ran on both machines at the commit that merged into `main`, with no failures:

| Machine | Environment | Run |
| --- | --- | --- |
| Laptop, i7-13650HX | Windows 11, Python 3.14.3, numpy 2.5.3, scipy 1.18.1, numba 0.67.0 | the default suite, and the slow fuzz separately. Two tests skip because Windows has no `os.fork` |
| Mac mini, Apple M4 | macOS 26.6.2 arm64, Python 3.14.3, numpy 2.5.3, scipy 1.18.1, numba 0.67.0 | the default suite on 8 workers, the two fork tests included |

The package layout change was also checked from a built wheel. The wheel holds the same files as `src/`, and the suite passed in a clean virtual environment from outside the repo.

The Covertype run below is the large-data regression check. Every output matched R's at all three sizes with the merged code.

## Performance

The fixtures are small, so I also ran both packages on a real large dataset: UCI Covertype, 581,012 rows. I used its first 10 columns, the continuous ones. They're all whole numbers, so there are plenty of ties. The 10,000 and 100,000 row sizes are random subsets drawn once in R with a fixed seed.

Each size runs `create.DN` with R's defaults, then `refine.DN` on its result with shape splits off and on. The only change from R's defaults is `no.cores = 1`, and `n_jobs=1` in Python, so R's random draws can be recorded in order. Python replays them the way the tests do. Assignments, weights, centers, scales, progress messages and warnings matched exactly in every run. At 100,000 rows the shape-split run replayed 383,475 draws.

Wall time in seconds, around the function call only, one run each. The middle column is the port before the compiled long double kernels. The last column is the merged code, with the numba cache warm:

| Rows | Step | R | Python, before | Python, now |
| --- | --- | --- | --- | --- |
| 10,000 | create | 5.7 | 6.7 | 6.8 |
| 10,000 | refine | 0.2 | 0.9 | 0.1 |
| 10,000 | refine, shape splits | 5.2 | 3.3 | 0.9 |
| 100,000 | create | 36.2 | 48.1 | 43.2 |
| 100,000 | refine | 0.5 | 8.9 | 1.3 |
| 100,000 | refine, shape splits | 1801 | 110 | 28.0 |
| 581,012 | create | 202.1 | 239.6 | 220.5 |
| 581,012 | refine | 0.9 | 50.0 | 8.7 |

R's shape-split run took 30 minutes at 100,000 rows, so the full-size run skips shape splits. The compiled kernels cut `refine_dn`'s time several times over. `create_dn` barely moved, because its time goes into Rfast's distances and the closest-pair deletion rounds, which the kernels don't touch yet. The open problems below describe the plan for those.

Peak memory was 2072, 2092 and 2183 MiB in R at the three sizes, and 1501, 1510 and 1560 MiB in Python now. That's the peak working set of the whole process, data load included. It hardly grows with the row count. The biggest object is the 10,000 by 10,000 distance matrix between the first-stage centers, which is the same size every time.

They ran on my laptop (i7-13650HX, 16 GB, Windows 11). R was 4.6.1 with `datanugget` 1.5.0 and its reference BLAS. Python was 3.14.3 with numpy 2.5.3, scipy 1.18.1 and numba 0.67.0.

Covertype data: Blackard, J. (1998). [Covertype](https://doi.org/10.24432/C50K5N) [Dataset]. UCI Machine Learning Repository. It's licensed under CC BY 4.0.

## Differences from R

Defaults match R except for `n_jobs` and the default weights of `wmean`, `wwcss` and `distw`, which are described below. `create_dn` and `refine_dn` default to `n_jobs=1`, while R's `create.DN` and `refine.DN` default to `no.cores = detectCores() - 1`. The argument names map as follows:

| R | Python |
| --- | --- |
| `x` | `x` |
| `center.method` | `center_method` |
| `R` | `r` |
| `delete.percent` | `delete_percent` |
| `DN.num1` | `dn_num1` |
| `DN.num2` | `dn_num2` |
| `dist.metric` | `dist_metric` |
| `seed` | `seed` |
| `no.cores` | `n_jobs` |
| `make.pbs` | `progress` |

`center_method` accepts `"mean"`, `"random"` or `"original"`. `dist_metric` accepts `"euclidean"` or `"manhattan"`.

Nugget IDs and assignments are 0-based instead of R's 1-based values. `seed` accepts an integer, a NumPy `Generator` or an `RDraws`. An integer or a `Generator` draws from NumPy, so the same seed doesn't give the same nuggets as R. `RDraws(seed)` draws what R draws.

Every function that reads the values in the data rejects NaN and infinite values. What R does with them isn't recorded for the `datanugget` functions. In `cluster.predict` one NaN row in `newx` empties R's whole result. Probes outside the fixtures show `wmean`, `wss`, `wwcss` and `distw` passing NaN through their arithmetic, and `Wpca` stops in `eigen()` once the matrix it hands over isn't finite, as the recorded weight cases show. Open problem 4 has the plan.

A few differences are kept on purpose. `n_jobs=-1` means all cores. Warnings come as `DataNuggetWarning`, and those raised in worker processes are sent back and raised again in the caller's process. Progress messages go to the `datanuggets` logger in place of R's `message()`. The port draws from its own generator, so it doesn't advance the caller's global random state the way R's functions advance `.Random.seed`. `create_dn`, `create_dn_centers` and `refine_dn` raise on a vector `x`, since R's behavior there isn't recorded.

`refine_dn` maps its arguments the same way:

| R | Python |
| --- | --- |
| `x` | `x` |
| `DN` | `dn` |
| `EV.tol` | `ev_tol` |
| `max.splits` | `max_splits` |
| `min.nugget.size` | `min_nugget_size` |
| `shape.split` | `shape_split` |
| `min.shape.size` | `min_shape_size` |
| `delta` | `delta` |
| `nstart` | `nstart` |
| `seed` | `seed` |
| `no.cores` | `n_jobs` |
| `make.pbs` | `progress` |

`refine_dn` reads only the assignments from `dn`. Like R, it recomputes every center, weight and scale from the rows. It also drops empty nuggets and gives single-row nuggets a scale of 0, where `create_dn` keeps empty nuggets and floors single-row scales. Both come from R. R re-seeds before every split, so each split starts from a fresh copy of the seed's stream and the result doesn't depend on `n_jobs`. With `n_jobs` other than 1 the splits of a round run in worker processes. Unlike R, `refine_dn` rejects an `nstart` or `max_splits` that isn't finite and a `shape_split` that isn't a bool.

`get_dn_means(x, dn)` takes R's `x` and `DN` and returns a NumPy array with one row per nugget, where R returns a data frame. An empty nugget gets a row of R's -NaN. With one column R transposes its result into a single row, but `get_dn_means` still returns one row per nugget. R ignores any rows of `x` past the ones `dn` assigns. It also skips a row whose nugget id isn't a whole number from 1 to the nugget count. `get_dn_means` raises `ValueError` in both cases.

`arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2)` names its arguments like `create_dn` and has no defaults, like R's `AC`. It reads only the shape of `x`. R 4.6.1's class check in `AC` stops on a matrix or a data.table, so R only takes a plain data frame. The port takes any 2-D input. Some arguments send R's loops around a cycle forever. A `DN.num2` below -1 does it unless `DN.num1` floors to the same integer. The port raises `ValueError` on these at once, even where R would count down for a long time before the cycle starts. It also raises on a `dn_num1` or `dn_num2` that isn't finite and on an `x` with no rows, where R stops, loops, or returns `Inf` or `NaN`.

In the `WCluster` functions `Wpca` becomes `wpca`, `cluster.predict` becomes `cluster_predict` and the `groupSum` argument of `wwcss` becomes `group_sum`. The other names are R's. A weight argument left as `None` means one weight per row, or per column for `wpca`'s `wcol`. R's `wmean`, `wwcss` and `distw` default to `rep(1, length(x))` instead, which has the wrong length. It stops unless the data is square (`wmean`, `wwcss`) or has one column (`distw`). A cluster whose weights sum to zero gets a NaN `wwcss` and NaN in every `distw` entry that touches it, as in R. Its `wmean` row is NaN when every weight is zero. When weights of both signs cancel, the row is infinite instead, with NaN in any column whose weighted sum is zero too, again as in R. `cluster_predict` doesn't pick such a cluster while another is at a finite distance. The other differences are these:

| Item | R | Python |
| --- | --- | --- |
| `wmean` layout | `p` rows by `k` columns | `(k, p)`, like every other center array in the package |
| `distw` with one cluster | stops ("missing value where TRUE/FALSE needed") | `ValueError` naming the cause |
| cluster labels | `wwcss` groups by `factor()` (15 significant digits), `distw` by `sort(unique())`, text by locale collation | whole numbers smaller than 1e15 in size, sorted. Text, bool, fractional and NaN labels raise |
| `cluster.predict` result | 1-based position in sorted label order, not the label | 0-based position |
| `cluster.predict` argument order | `(x, w, cl, newx)`, and a swapped `w` and `cl` runs silently | `(x, cl, newx, *, w=None)` |
| `cluster.predict` for a row where every distance is NaN (NaN centers, another new row that overflows, or centers of infinities once there are two new rows) | `integer(0)` when every row is like that, a list with empty entries when only some are | `ValueError` |
| `cluster.predict` distances after R's identity multiply | `0 * Inf` from one overflowing row turns its whole column NaN, and a lone `Inf` stays | the same, computed without the multiply |
| a vector `newx` in `cluster.predict`, a vector `x` in `getDN.means` | stops | `ValueError` |
| integer `x` with integer weights | `wmean` and `cluster.predict`: an NA center on overflow with a warning. `Wpca`: the NA becomes 0 in its column sum, a wrong center with a warning. `wss`, `wwcss`, `distw`: no integer products | float64 arithmetic, no overflow |
| NaN or infinite weights, text or bool weights | passed through or stopped late in `eigen` | `ValueError` or `TypeError` up front |
| `Wpca` row weights summing to 1, a zero total, a negative row weight, zero variance under `corr`, row weights summing below 1 under `corr`, column weights with a zero sum or mixed signs | stop inside `eigen` ("infinite or missing values in 'x'"), some after a "NaNs produced" warning | `ValueError` naming the cause |
| `Wpca` with one column and `corr = TRUE` | `diag()` of one number builds an identity: the covariance passes through unscaled for `0.5 < dd <= 1`, and R stops otherwise | the general formula, `sdev` 1 to rounding. It fixes an upstream bug |
| `Wpca` `scale` with `corr = FALSE` | the single number 1 | ones, one per column |
| `Wpca` `corr` | `TRUE`, `1`, `1L` and `"TRUE"` mean corr. `FALSE`, `0`, `0L`, `"FALSE"`, `"F"`, `"false"` and `"False"` mean no corr. `"T"`, `"true"`, `"True"`, `2` and `-1` stop with "object 'dd' not found" in a clean session and use the caller's `dd` otherwise. `NA` stops with "missing value where TRUE/FALSE needed" | a bool, else `TypeError` |
| `w = NULL` | stops, though the help pages say it means unit weights | `None` means unit weights |

`wpca`'s `rotation` columns, and the scores with them, can have the opposite sign from R's, since LAPACK picks each column's sign.

## Limitations

- Parity means one reference environment: R 4.6.1 on x86-64 with the reference BLAS, R's internal LAPACK and the x87 long double. Results from an R built another way can differ in the last bits, and on tied data that can move rows between nuggets.
- The eigen output in `wpca` and `refine_dn` comes from scipy's LAPACK, so it's held to R's by a rule or a tolerance instead of matched bit for bit. In `refine_dn` the first eigenvalue decides which nuggets every split round tries, with or without shape splits. The eigen spike found two kinds of input where the difference goes past rounding. With a repeated top eigenvalue, `refine_dn`'s nuggets can differ from R's, and scipy matched R on 1 of 5 such refine inputs. On rank-deficient `wpca` inputs, the NaN positions in `sdev` and R's warning can differ. scipy reproduced R's `sdev` bits, NaN positions, warning and rotation bits together on 9 of 420. A rotation column can also come back with the opposite sign from R's.
- An integer seed or a NumPy `Generator` draws from NumPy, so the nuggets differ from R's for the same seed. Only `RDraws` gives R's draws, and it isn't exported from `datanuggets` yet.
- NaN and infinite values in the data or the weights raise everywhere. R's behavior with them is recorded only for `cluster.predict` so far, so the rest keep raising until it is.
- Labels must be whole numbers smaller than 1e15. Fractional, bool and text labels raise.
- `create_dn` builds a `dn_num1` by `dn_num1` distance matrix in its second stage. At the default 10,000 that's 800 MB of float64, whatever the row count, and the process peaked near 1.5 GB on Covertype.
- `create_dn` is slower than R at every Covertype size, and `refine_dn` without shape splits is slower than R at 100,000 rows and above.
- numba compiles the kernels on first use and caches them. On my laptop, importing the package and making a first small call to each main function took 3.6 seconds with an empty cache and 0.65 seconds with a warm one. Where numba can't write its cache, every new process compiles again.
- Windows has no `os.fork`, so the two tests of forked children run only on macOS and Linux.
- It isn't on PyPI, and it needs Python 3.12 or newer with numba.

## Open problems

These are known and unsolved. Items 1 to 5 are tracked in the [R-compatibility record](docs/plans/2026-09-30-r-compat.md), which holds or links the evidence for each. Item 6 has its own plan, item 7 is the rest of the port, and items 8 and 9 came out of the reviews and the wheel check.

1. Nine breaking changes toward R's behavior are approved in principle and held until they're confirmed as a set. B1 makes R's draws the default for an integer seed. B2 returns `wmean` as `p` by `k`. B3 makes `wpca`'s scale a single 1.0. B4 and B5 adopt R's argument order for `cluster_predict` and `create_dn_centers`. B6 makes `progress` take bools only. B7 gives `to_frame()` R's 1-based ids and types. B8 makes `cluster_predict` return R's empty results instead of raising. B9 orders categorical labels by category.
2. Eight compatibility tasks are pending. T4 accepts fractional and bool labels under `factor()`'s 15-digit rule and adds `cluster_levels`. Two recorded cases, `labels_faithful_float_merge` and `labels_faithful_big`, show a gap in that design. In both, `distw` builds three axes from the exact values where `factor()` makes two levels, so one list of levels can't describe both. T6 floors `dn_num1`, `dn_num2` and `dn_num` the way R does. T7 makes `refine_dn` accept any real `nstart` and return R's result, which leaves the nuggets unsplit when `kmeans` can't run. Today `refine_dn` raises `ValueError` on a NaN or infinite `nstart`, and the internal `kmeans` raises an `OverflowError` on an infinite one. T8 accepts R's other `corr` values. T9 exports `RDraws` and `cluster_levels`. T10 adds independent checks for the one-column `"random"` case of `create_dn`, where R's own table is malformed. T11 adds tests for the documented exceptions. T12 writes `docs/compatibility.md` and corrects the remaining docs.
3. One-column `wpca` with `corr=True` waits on a decision. R's result there is a side effect of `diag()`, and both independent checks give the port's 1. Keeping the port's answer is the recommendation.
4. R's behavior with NaN, NA and infinite values isn't recorded for `create_dn`, `refine_dn`, `create_dn_centers`, `arithmetic_complexity` or `get_dn_means`, so they keep raising until it is. The `WCluster` statistics need NA and NaN recorded separately before they can pass them through with R's payloads.
5. A reference eigen solver needs approval. In the spike, R's own `Rlapack` reproduced R on all 420 rank-deficient `wpca` inputs, where scipy matched 9, on 100 of 100 refine cases against scipy's 63, and on all 43 `wpca` rotation and score cases against 16. The recommended option builds R's LAPACK sources with R's flags for Windows x64 first. It needs the R source and Rtools downloads, platform wheels in place of the pure-Python one, and a license review. Nothing was measured on Linux, macOS arm64 or another R version.
6. The speed stopping rule fails. The design spec stops optimizing once no Python function takes more than 10% of wall time on Covertype at 100,000 rows. Under the profiler on the Mac mini, `create_dn` took 15.49 seconds, with 39.2% in Rfast's distance terms, 34.5% in the deletion rounds of `create_dn_centers` and 11.6% in the even and odd sums. `refine_dn` took 0.91 seconds, with four Hartigan-Wong functions at 12% to 17% each. The [follow-up plan](docs/plans/2026-09-30-speed-next-targets.md) compiles those three paths. A prototype of it, kept out of the repo, ran `create_dn` in 2.23 seconds and `refine_dn` in 0.32 seconds on the same machine with every output still matching R. The plan waits on review, and its file paths need updating to the new layout first.
7. `Wkmeans`, `Whclust`, `DN.Wkmeans`, `DN.Whclust` and `optimal.DN` aren't ported.
8. Some behavior isn't covered by a recorded case or a test:
   - R hasn't recorded zero-column inputs or centers with no rows.
   - A process that launches numba threads under GNU OpenMP and then forks, before it has ever imported the moments module, leaves the child without the hook that makes it run serially. Neither test machine has GNU OpenMP.
   - Some exact checks use `same_values`, which lets any NaN match any NaN, so a wrong NaN payload there wouldn't fail.
   - The `create_dn`, `create_dn_centers`, `refine_dn`, `kmeans` and Rfast tests compare with `assert_array_equal`, which can't tell -0.0 from 0.0. A wrong sign of zero there would pass the suite. For `create_dn` and `refine_dn` it would show when the parity tables are rebuilt.
   - Three mutants from the WCluster review aren't killed. One in `r_weighted_mean` survives on the laptop, where its NaN and R's have the same bits, and it wasn't run on arm64, where it should fail. One moves `wpca`'s warning threshold below -1e-300, where no recorded input has an eigenvalue. One drops the orthonormality check from the eigen rule, which another check covers, though that isn't proven for tied eigenvalues.
   - `RDraws` isn't checked against R for a sample of more than 2^31 - 1 values from the hash sampler, which only a population of more than four billion would reach.
9. A source distribution built from a working tree that holds the local `.superpowers/sdd/` scratch folder includes it. The folder ignores itself through its own nested `.gitignore`, which hatchling doesn't read. A fresh clone has no such folder, and the wheel isn't affected.

## Development

```sh
uv sync
uv run pytest
uv run pytest -m slow
```

The first runs the default suite. The second runs only the long fuzz of the compiled long double kernels.

[docs/architecture.md](docs/architecture.md) shows where each part of the code, tests and fixtures lives, and where new work goes.

Tests use fixtures in `tests/fixtures/` generated by running the R packages. Running tests doesn't need R. Regenerating fixtures needs R 4.6.1. The generators are in `r/`, with instructions in [r/README.md](r/README.md).

Rebuild the parity tables with:

```sh
uv run python scripts/parity_report.py
```

To rerun the Covertype benchmark, save the [data file](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz) as `tmp/covertype/covtype.data.gz`. The script checks its `sha256`. Set up the private R library described in [r/README.md](r/README.md), then run from the repo root:

```sh
Rscript benchmarks/covertype.R 10000
uv run python benchmarks/covertype.py 10000
```

Use `10000`, `100000` or `all` for the size. Add `noshape` to the `Rscript` line to skip shape splits. The R script writes to `tmp/covertype/out/`. The Python script reads those outputs and prints a JSON report.

To report wall time and peak memory, prefix either command with `uv run --with psutil python benchmarks/measure.py`. It reads the peak working set, so it only works on Windows.

## Citing the method

Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 2025. Published online in 2024.

DOI: `10.1080/10618600.2024.2341896`. arXiv: `2403.03099`.

## License and credit

GPL-2.0-only, the same license as the R packages `datanugget` and `WCluster`. The algorithms are the R authors' work.

The `datanugget` authors are Rituparna Dey (maintainer), Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi and Mariusz Lubomirski.

The `WCluster` authors are Rituparna Dey (maintainer), Yajie Duan, Javier Cabrera and Ge Cheng.
