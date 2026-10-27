# Third-party notices

The project's intended package license is GPL-2.0-only; the full text is in `LICENSE`.
Public redistribution of this candidate is on hold pending the sorting-source review
below. The dataset described below retains its separate CC BY 4.0 license.

## Publication hold: GNU sorting helpers

`src/datanuggets/_compat/rfast.py` holds `gnu_sort_first` and the median-of-three
partition and heap-sort helpers it calls. The test reference
`tests/compat/rfast_reference.py` holds `gnu_sort_order` and its introsort and
insertion-sort helpers, which call the same partition and heap-sort helpers.
`rfast.py` names these helpers after libstdc++'s std::sort internals, such as
`_unguarded_partition_pivot` and `_adjust_heap`, and cites libstdc++'s `_S_threshold`
and `std::__partial_sort`. The test reference describes `gnu_sort_order`, which calls
them, as replaying std::sort step for step. The wheel includes `rfast.py`, and the
source distribution includes both files. Their exact upstream revision and license
provenance have not been established.

Current GCC [stl_algo.h](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/stl_algo.h)
and [stl_heap.h](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/stl_heap.h)
carry GPL-3.0-or-later notices with the
[GCC Runtime Library Exception 3.1](https://github.com/gcc-mirror/gcc/blob/master/COPYING.RUNTIME),
alongside older HP/SGI notices. Those headers alone do not establish permission to
redistribute these Python translations under GPL-2.0-only. Similarity to an older
algorithm does not establish which licensed source was used.

Before publication, establish and record compatible provenance and all required
notices, obtain compatible permission, or replace the affected helpers with a
compatibly licensed implementation and recheck R tie-order parity. This is an open
provenance question, not a determination of infringement. The release workflow
allows check-only runs but blocks uploads to both package indexes until resolved.

## R packages

This project ports **datanugget 1.5.0** and **WCluster 1.3.0**, both published under GPL-2.

- datanugget authors: Rituparna Dey, Yajie Duan, Traymon Beavers, Javier Cabrera,
  Ge Cheng, Kunting Qi and Mariusz Lubomirski.
  [Package and source](https://CRAN.R-project.org/package=datanugget).
- WCluster authors: Rituparna Dey, Yajie Duan, Javier Cabrera and Ge Cheng.
  [Package and source](https://CRAN.R-project.org/package=WCluster).

Ronil Basu adapted the implementations to Python, with Python interfaces, zero-based
indices, compiled numerical kernels and the documented compatibility differences.
This notice records the adaptations included in the first alpha as of 2026-10-21.
The Python port is independently maintained; upstream authors do not endorse this release.

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

## Supplied flow-analysis code

The flow-analysis implementation and its reference benchmarks adapt Python code supplied
by Prof. Javier Cabrera on 2026-10-12. The adaptations add package interfaces, validation,
CLI input/output handling and the documented gate correction.

On 2026-10-21, maintainer Ronil Basu confirmed that Prof. Cabrera's permission covers
redistribution of the supplied code and these adaptations under GPL-2.0-only.

## Flow-cytometry test data

The flow fixtures derive from:

Stefanie Kreutmair and Susanne Unger (2026). *Flow cytometry data of PBMC of critically
ill, mechanically ventilated patients* (V1). Zenodo.
[doi:10.5281/zenodo.22079206](https://doi.org/10.5281/zenodo.22079206).

The original data are licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The derived data retain that license. Use of the data does not imply endorsement.

`tests/fixtures/flow/154.npz`, `202.npz` and `223.npz` select channels from three FCS
files, retain every sixth, fifth and first event respectively, cap each selection at
50,000 events, and convert it to NPZ. They also contain computed reference results.
`202_fluor.npz` contains three fluorescence channels for the same selected events as
`202.npz`. These fixtures are included in the source distribution, not the wheel.
