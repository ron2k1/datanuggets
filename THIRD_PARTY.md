# Third-party notices

The package is licensed under GPL-2.0-only. The full text is in `LICENSE`.
The data described below keep their own licenses.

## Sorting helpers removed

Earlier development versions settled exactly tied distances in `nearest_index`
(`src/datanuggets/_compat/rfast.py`) with Python helpers that followed the internals of
GCC libstdc++'s `std::sort`, so that the port picked the center Rfast's unstable sort
picks. The license provenance of those helpers could not be shown to fit GPL-2.0-only,
so on 2026-11-01 they were removed from the package and from its test reference. They were
removed before the first upload to PyPI, so no version on PyPI contains them.

On an exact distance tie the port now takes the lowest center index. That matches R
whenever there are 16 or fewer centers. With more, R can take another of the tied centers.
[docs/compatibility.md](docs/compatibility.md#exact-distance-ties-in-create_dn)
records where the results differ.

## R packages

This project ports **datanugget 1.5.0** and **WCluster 1.3.0**, both published under GPL-2.

- datanugget authors: Rituparna Dey, Yajie Duan, Traymon Beavers, Javier Cabrera,
  Ge Cheng, Kunting Qi and Mariusz Lubomirski.
  [Package and source](https://CRAN.R-project.org/package=datanugget).
- WCluster authors: Rituparna Dey, Yajie Duan, Javier Cabrera and Ge Cheng.
  [Package and source](https://CRAN.R-project.org/package=WCluster).

Rituparna Dey is the maintainer of both packages.

Ronil Basu adapted the implementations to Python, with Python interfaces, zero-based
indices, compiled numerical kernels and the documented compatibility differences.
This notice records the adaptations included in the first alpha as of 2026-11-01, the date
the change notice at the top of each ported module carries.
The Python port is independently maintained; upstream authors do not endorse this release.

## Rfast

`datanugget` computes its distances with the R package
[Rfast 2.1.5.2](https://CRAN.R-project.org/package=Rfast) by Manos Papadakis, Michail Tsagris
and others, published under GPL (>= 2.0). `src/datanuggets/_compat/rfast.py` and `rfastjit.py`
adapt the computations of `Rfast::Dist` and `Rfast::dista`, and
`src/datanuggets/nuggets/centersjit.py` those of `Rfast::rowMins`, so that the port's
distances round as R's do. These adaptations use GPL version 2.

## Armadillo's summation order

Rfast sums each distance's per-feature terms with [Armadillo](https://arma.sourceforge.net/),
which is licensed under Apache-2.0. The port repeats the order of those additions. `accumulate`
keeps two running sums, over the even and the odd positions, and adds them at the end, and
`op_dot::direct_dot` uses its own loop for up to 32 terms and BLAS `ddot` above that.
`src/datanuggets/_compat/rfastjit.py` and `_arma_accumulate` in
`tests/compat/rfast_reference.py` follow that order. The repository and its archives contain
no Armadillo source code. Only the order of the arithmetic is reproduced.

## R's Hartigan–Wong implementation

`src/datanuggets/_clustering/hartigan_wong.py` and `kmnsjit.py`, and the test reference
`tests/clustering/kmns_reference.py`, adapt
[R's `src/library/stats/src/kmns.f`](https://svn.r-project.org/R/trunk/src/library/stats/src/kmns.f).
The test reference is included in the source distribution, not the wheel.

> (C) Royal Statistical Society 1979

The source credits B. D. Ripley's modifications dated 1998-06-17 and M. Maechler's
modifications dated 2001-04-12. R distributes this source under the GNU General Public
License; this port uses version 2. See [R's copyright notices](https://svn.r-project.org/R/trunk/doc/COPYRIGHTS).
The Python adaptations preserve operation order while adding zero-based indices,
Python result types and Numba compilation. Changed-file notices also appear in all
three modules. The software is provided without warranty, as stated in `LICENSE`.

## Other R source adaptations

The following R sources are published under GPL-2.0-or-later. This project's
adaptations use GPL version 2. The copyright holders' notices remain applicable:

| Python module | R source and copyright notices |
| --- | --- |
| `plot/tree.py` | [`stats/R/hclust.R`](https://svn.r-project.org/R/trunk/src/library/stats/R/hclust.R): Copyright (C) 1995–2020 The R Core Team. [`graphics/src/plot.c`](https://svn.r-project.org/R/trunk/src/library/graphics/src/plot.c): Copyright (C) 1997–2026 The R Core Team; 2002–2009 The R Foundation; 1995, 1996 Robert Gentleman and Ross Ihaka. |
| `wcluster/cutree.py` | [`stats/src/hclust-utils.c`](https://svn.r-project.org/R/trunk/src/library/stats/src/hclust-utils.c): Copyright (C) 1999–2025 The R Core Team. [`stats/R/cutree.R`](https://svn.r-project.org/R/trunk/src/library/stats/R/cutree.R): Copyright (C) 1995–2012 The R Core Team. |
| `_random/r_draws.py` | [`main/RNG.c`](https://svn.r-project.org/R/trunk/src/main/RNG.c): Copyright (C) 1997–2026 The R Core Team; 1995, 1996 Robert Gentleman and Ross Ihaka. [`main/random.c`](https://svn.r-project.org/R/trunk/src/main/random.c): Copyright (C) 1997–2025 The R Core Team; 1995, 1996 Robert Gentleman and Ross Ihaka; 2003–2018 The R Foundation. [`main/unique.c`](https://svn.r-project.org/R/trunk/src/main/unique.c): Copyright (C) 1997–2025 The R Core Team; 1995, 1996 Robert Gentleman and Ross Ihaka. |
| `_compat/indexing.py` | [`parallel/R/clusterApply.R`](https://svn.r-project.org/R/trunk/src/library/parallel/R/clusterApply.R): Copyright (C) 1995–2025 The R Core Team; derived from snow 0.3-6 by Luke Tierney. |

Ronil Basu's Python adaptations, recorded here as of 2026-10-21, add NumPy arrays,
zero-based indices, package result types and matplotlib drawing while preserving
the documented R behavior. The random-draw adapter uses NumPy's MT19937 engine.
These sources and adaptations are supplied without warranty; see `LICENSE`.

## Release approval

On 2026-11-01, Prof. Javier Cabrera approved releasing the port publicly under
GPL-2.0-only, as the `datanuggets` package on PyPI.

## R datasets in the test fixtures

The fixtures under `tests/fixtures/` store values from datasets that ship with R in its
`datasets` package: iris, faithful, quakes, mtcars, volcano, ChickWeight, EuStockMarkets,
USArrests, PlantGrowth, rock, LifeCycleSavings, infert, trees, npk, airquality, women, swiss,
state.x77 and state.region. The example `examples/first_analysis.py` reads the quakes values from them. The
`datasets` package is part of R, which
is distributed under the GNU General Public License, version 2 or 3. The source distribution
includes these fixtures and the example, and the wheel includes neither.
[reproducibility.md](docs/reproducibility.md)
describes how the fixtures were recorded.

## California housing test data

`tests/fixtures/housing/california_housing.csv` is public domain. The values are 1990 US Census
block-group aggregates, a work of the US government, which 17 U.S.C. 105 leaves without
copyright. They were compiled by Pace, R. K. and Barry, R. (1997), Sparse Spatial
Autoregressions, *Statistics and Probability Letters* 33, 291-297, and are redistributed by
scikit-learn (`fetch_california_housing`), whose copy this is. Two of the colour tests read it.
It is in the repository only: neither the source distribution nor the wheel includes it.
