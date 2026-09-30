# datanuggets

A native Python port of CRAN's `datanugget` 1.5.0 and `WCluster` 1.3.0. It runs without R.

Data nuggets reduce a dataset to groups described by a center, a row count and a scale. Scale measures spread as the average variance across columns. Centers are chosen by repeatedly removing one point from the closest pair, which spreads them across the data instead of concentrating them in dense regions. Each row joins its nearest center.

Beavers et al. (cited below) run weighted PCA and weighted k-means on the nuggets instead of the full dataset. Their flow cytometry example has over one million observations.

## Status

Version `0.1.0.dev0` is in development. Five functions are ported:

- `create_dn`, from R's `create.DN`
- `create_dn_centers`, from R's `create.DNcenters`
- `refine_dn`, from R's `refine.DN`
- `get_dn_means`, from R's `getDN.means`
- `arithmetic_complexity`, from R's `AC`

`refine_dn` splits nuggets that are spread out or elongated with 2-means, through a port of the Hartigan-Wong k-means in R's `stats::kmeans`. `get_dn_means` averages the rows in each nugget. `arithmetic_complexity` gives the arithmetic cost of a `create_dn` call on a log10 scale, from the shape of the data alone.

Next come the rest of `WCluster`'s building blocks: `wmean`, `wss`, `wwcss`, `distw`, `wpca` and `cluster_predict`. Its weighted k-means and hierarchical clustering follow. The data-nugget wrappers and `optimal_dn` come last.

## Install

Requires Python 3.12 or newer, `numpy >= 2.0`, `scipy >= 1.13` and `joblib >= 1.4`.

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

Those weight-1 nuggets are expected. Closest-pair deletion keeps points far out in the tails as centers. They share the scale 0.000331 because a nugget with fewer than 2 rows gets the smallest scale among nuggets with 2 or more rows, divided by 100. R does the same. The call took 28.9 seconds on my laptop (i7-13650HX).

## Differences from R

Defaults match R except for `n_jobs`. `create_dn` and `refine_dn` default to `n_jobs=1`, while R's `create.DN` and `refine.DN` default to `no.cores = detectCores() - 1`. The argument names map as follows:

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

Nugget IDs and assignments are 0-based instead of R's 1-based values. `seed` accepts an integer or a NumPy `Generator`. The same seed doesn't give the same nuggets as R because the random number generators differ.

Every function that reads the values in the data rejects NaN and infinite values. R lets NA through, so `create.DN` returns NA centers and `getDN.means` returns NA means for the nuggets it lands in.

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

`get_dn_means(x, dn)` takes R's `x` and `DN` and returns a NumPy array with one row per nugget, where R returns a data frame. An empty nugget gets a row of NaN, as in R. With one column R transposes its result into a single row, but `get_dn_means` still returns one row per nugget. R ignores any rows of `x` past the ones `dn` assigns. `get_dn_means` raises `ValueError` unless `x` has exactly those rows.

`arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2)` names its arguments like `create_dn` and has no defaults, like R's `AC`. It reads only the shape of `x`. R 4.6.1's class check in `AC` stops on a matrix or a data.table, so R only takes a plain data frame. The port takes any 2-D input. Some arguments send R's loops around a cycle forever, a `DN.num2` of -2 or less among them. The port spots the cycle and raises `ValueError`. It also raises on a `dn_num1` or `dn_num2` that isn't finite and on an `x` with no rows, where R stops, loops, or returns `Inf` or `NaN`.

## Parity with R

Every case in the `create_dn` parity table matches R bit for bit in all four outputs: assignments, weights, centers and scales. The cases run iris, faithful and quakes under both metrics and all three center methods. Edge cases add a one-column dataset, tied grids, duplicate rows and a 61-column scaled dataset. Tests replay the exact random draws recorded from R and compare with `np.array_equal`, with no tolerance. The full table is in [docs/validation/create_dn.md](docs/validation/create_dn.md).

`refine_dn` is checked the same way on its own grid. On faithful it runs every combination of `ev_tol`, `max_splits`, `min_nugget_size`, `shape_split`, `delta` and `nstart`. Six more datasets run four settings each: iris, quakes, the petal columns of iris, a one-column dataset, the scaled volcano data and a tied grid. More cases cover refining a refined result, an empty input nugget and splits that fail. Every kmeans call clusters the same rows as R's and returns the same clusters, and the outputs match bit for bit.

Shape splits cluster the rows projected on the first eigenvector. Those projections and the eigenvalues that decide which nuggets get a split attempt come from LAPACK. They're compared with NumPy's `assert_allclose` using `rtol=1e-8` and `atol=1e-10`. The eigenvalues differ by rounding on the scale of the largest eigenvalue. A second eigenvalue near zero can have a large relative error while staying within `1e-10` absolute. No split decision in the fixture grid changes because of that rounding. The table is in [docs/validation/refine_dn.md](docs/validation/refine_dn.md).

`get_dn_means` runs on every nugget set in those two fixture grids: each `create.DN` output, and each input and output of `refine.DN`. That's 293 sets, two of them with an empty nugget and nine with one column. All 293 match R bit for bit. They're too small for the order of R's long-double sums to change a mean, so one constructed case, recorded from R, pins that order. The table is in [docs/validation/get_dn_means.md](docs/validation/get_dn_means.md).

`arithmetic_complexity` runs on the row and column counts of eight R datasets, crossed with values of `r`, `delete_percent`, `dn_num1` and `dn_num2`. That makes 3481 calls. Whenever R's `AC` returns, the pass counts traced inside it equal the port's, and the value matches bit for bit. The tests still allow a relative difference of `1e-8` in the value, since it goes through the platform's `log10`. Running those loops in exact arithmetic would get 289 of the counts wrong, so the port repeats R's double arithmetic step by step. R never returned from 19 of the calls, and the port raises on exactly those 19. The table is in [docs/validation/arithmetic_complexity.md](docs/validation/arithmetic_complexity.md).

Getting the last bit right meant copying how R adds things up. Rfast keeps two running sums for distance terms, one over even columns and one over odd. R's `colMeans` and `rowMeans` add in 80-bit x87 precision before rounding to double, and the port replays that with Python integers. `refine_dn` needs R's `cov`, `mean`, `median`, `mad` and `quantile` too, which also add in long double.

Parity is against R 4.6.1 on x86-64 Windows with `datanugget` 1.5.0, Rfast 2.1.5.2 and R's reference BLAS. OpenBLAS, MKL and Accelerate can round the last bits differently. So can arm64 macOS, where `long double` is plain double. On tied data, those differences can change assignments. The fixture manifests record the BLAS and `long double` width.

## Covertype benchmark

The fixtures are small, so I also ran both packages on a real large dataset: UCI Covertype, 581,012 rows. I used its first 10 columns, the continuous ones. They're all whole numbers, so there are plenty of ties. The 10,000 and 100,000 row sizes are random subsets drawn once in R with a fixed seed.

Each size runs `create.DN` with R's defaults, then `refine.DN` on its result with shape splits off and on. The only change from R's defaults is `no.cores = 1`, and `n_jobs=1` in Python, so R's random draws can be recorded in order. Python replays them the way the tests do. Assignments, weights, centers, scales, progress messages and warnings matched exactly in every run. At 100,000 rows the shape-split run replayed 383,475 draws.

Wall time in seconds, around the function call only, one run each:

| Rows | Step | R | Python |
| --- | --- | --- | --- |
| 10,000 | create | 5.7 | 6.7 |
| 10,000 | refine | 0.2 | 0.9 |
| 10,000 | refine, shape splits | 5.2 | 3.3 |
| 100,000 | create | 36.2 | 48.1 |
| 100,000 | refine | 0.5 | 8.9 |
| 100,000 | refine, shape splits | 1801 | 110 |
| 581,012 | create | 202.1 | 239.6 |
| 581,012 | refine | 0.9 | 50.0 |

R's shape-split run took 30 minutes at 100,000 rows, so the full-size run skips shape splits. Python's 50 seconds for `refine_dn` at full size go into its first pass. That pass computes every nugget's covariance with R's long-double sums, replayed in pure Python one nugget at a time. Speeding it up is future work.

Peak memory was 2072, 2092 and 2183 MiB in R at the three sizes and 1444, 1454 and 1501 MiB in Python. That's the peak working set of the whole process, data load included. It hardly grows with the row count. The biggest object is the 10,000 by 10,000 distance matrix between the first-stage centers, which is the same size every time.

They ran on my laptop (i7-13650HX, 16 GB, Windows 11). R was 4.6.1 with `datanugget` 1.5.0 and its reference BLAS. Python was 3.14.3 with numpy 2.5.3 and scipy 1.18.1.

Covertype data: Blackard, J. (1998). [Covertype](https://doi.org/10.24432/C50K5N) [Dataset]. UCI Machine Learning Repository. It's licensed under CC BY 4.0.

## Development

```sh
uv sync
uv run pytest
```

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

GPL-2.0-only, the same license as the R packages. The algorithms are the R authors' work.

The `datanugget` authors are Rituparna Dey (maintainer), Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi and Mariusz Lubomirski.
