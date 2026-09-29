# datanuggets

A native Python port of CRAN's `datanugget` 1.5.0 and `WCluster` 1.3.0. It runs without R.

Data nuggets reduce a dataset to groups described by a center, a row count and a scale. Scale measures spread as the average variance across columns. Centers are chosen by repeatedly removing one point from the closest pair, which spreads them across the data instead of concentrating them in dense regions. Each row joins its nearest center.

Beavers et al. (cited below) run weighted PCA and weighted k-means on the nuggets instead of the full dataset. Their flow cytometry example has over one million observations.

## Status

Version `0.1.0.dev0` is in development. Three functions are ported:

- `create_dn`, from R's `create.DN`
- `create_dn_centers`, from R's `create.DNcenters`
- `refine_dn`, from R's `refine.DN`

`refine_dn` splits nuggets that are spread out or elongated with 2-means, through a port of the Hartigan-Wong k-means in R's `stats::kmeans`. Next come `optimal_dn` and the remaining `datanugget` helpers, followed by `WCluster`'s weighted statistics, clustering methods and data-nugget wrappers.

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

Defaults match R. The argument names map as follows:

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

## Parity with R

Every case in the `create_dn` parity table matches R bit for bit in all four outputs: assignments, weights, centers and scales. The cases run iris, faithful and quakes under both metrics and all three center methods. Edge cases add a one-column dataset, tied grids, duplicate rows and a 61-column scaled dataset. Tests replay the exact random draws recorded from R and compare with `np.array_equal`, with no tolerance. The full table is in [docs/validation/create_dn.md](docs/validation/create_dn.md).

`refine_dn` is checked the same way on its own grid. On faithful it runs every combination of `ev_tol`, `max_splits`, `min_nugget_size`, `shape_split`, `delta` and `nstart`. Six more datasets run four settings each: iris, quakes, the petal columns of iris, a one-column dataset, the scaled volcano data and a tied grid. More cases cover refining a refined result, an empty input nugget and splits that fail. Every kmeans call clusters the same rows as R's and returns the same clusters, and the outputs match bit for bit.

Shape splits cluster the rows projected on the first eigenvector. Those projections and the eigenvalues that decide which nuggets get a split attempt come from LAPACK. They're compared with NumPy's `assert_allclose` using `rtol=1e-8` and `atol=1e-10`. The eigenvalues differ by rounding on the scale of the largest eigenvalue. A second eigenvalue near zero can have a large relative error while staying within `1e-10` absolute. No split decision in the fixture grid changes because of that rounding. The table is in [docs/validation/refine_dn.md](docs/validation/refine_dn.md).

Getting the last bit right meant copying how R adds things up. Rfast keeps two running sums for distance terms, one over even columns and one over odd. R's `colMeans` and `rowMeans` add in 80-bit x87 precision before rounding to double, and the port replays that with Python integers. `refine_dn` needs R's `cov`, `mean`, `median`, `mad` and `quantile` too, which also add in long double.

Parity is against R 4.6.1 on x86-64 Windows with `datanugget` 1.5.0, Rfast 2.1.5.2 and R's reference BLAS. OpenBLAS, MKL and Accelerate can round the last bits differently. So can arm64 macOS, where `long double` is plain double. On tied data, those differences can change assignments. The fixture manifests record the BLAS and `long double` width.

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

## Citing the method

Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 2025. Published online in 2024.

DOI: `10.1080/10618600.2024.2341896`. arXiv: `2403.03099`.

## License and credit

GPL-2.0-only, the same license as the R packages. The algorithms are the R authors' work.

The `datanugget` authors are Rituparna Dey (maintainer), Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi and Mariusz Lubomirski.
