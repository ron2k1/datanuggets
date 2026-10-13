# Architecture

Where the code, tests, fixtures and tools of datanuggets live, which way the imports run, and where new work goes. The last section maps the old paths that the plans, specs and reviews in `docs/` still use.

## Repository

| Folder | Holds |
|---|---|
| `src/datanuggets/` | the package, the only folder that ships in the wheel |
| `tests/` | the pytest suite, `rfixtures.py` and the R fixtures it reads |
| `r/` | the R scripts that record the fixtures, with setup in `r/README.md` |
| `scripts/` | `parity_report.py`, which rebuilds the tables in `docs/validation/` |
| `benchmarks/` | the Covertype benchmark in R and Python, the stopping-rule profiler and the memory meter, and in `benchmarks/flow/` Prof. Cabrera's flow script with the comparison against `datanuggets.flow` |
| `docs/specs/` | the design specs, dated |
| `docs/plans/` | implementation plans and decision records, dated |
| `docs/reviews/` | review reports, dated and kept as written |
| `docs/validation/` | one parity table per public function, written by `scripts/parity_report.py` |

The package imports nothing from `tests/`, `r/`, `scripts/` or `benchmarks/`, and none of them ship.

Some folders a fuller layout would have aren't here yet, because nothing exists to put in them: `scripts/` holds one script, the fixture generators are R and stay in `r/` beside their setup notes, the README carries the only example, and `docs/compatibility.md` arrives with the R-compatibility work (task T12 in `docs/plans/2026-09-30-r-compat.md`). Add each folder with its first file.

## The package

```
src/datanuggets/
  __init__.py          the public API: imports, __all__ and the old module paths, nothing else
  exceptions.py        DataNuggetWarning
  result.py            DataNuggets, CentersResult, WpcaResult, WwcssResult, WkmeansResult,
                       DnWkmeansResult, WhclustResult
  _validate.py         input coercion and argument checks, mirroring R's stop() conditions
  _defaults.py         R's default argument values and the internal sizes
  nuggets/             the datanugget ports
    create.py          create_dn (create.DN)
    centers.py         create_dn_centers (create.DNcenters)
    centersjit.py      create.DNcenters' deletion rounds compiled, over one distance matrix
    refine.py          refine_dn (refine.DN)
    means.py           get_dn_means (getDN.means)
    complexity.py      arithmetic_complexity (AC)
  wcluster/            the WCluster ports
    wstats.py          wmean, wss, wwcss and distw, which share their input and sum helpers
    predict.py         cluster_predict (cluster.predict)
    wpca.py            wpca (Wpca)
    wstatsjit.py       the sums, means and nearest centers the iterative kernels share, compiled
    whclust.py         whclust (Whclust), and the plain-Python reference its kernel is held to
    whclustjit.py      Whclust's merges compiled, each wss computed once
    wkmeans.py         wkmeans (Wkmeans): its random starts, and the plain-Python reference
    wkmeansjit.py      Wkmeans_single compiled, each trial move over two clusters' rows
    dn.py              dn_wkmeans, dn_whclust, dn_wpca and dn_cluster_predict (the DN wrappers)
    cutree.py          cut_tree (stats::cutree), and R's merge numbering from the port's
  flow/                Prof. Cabrera's flow gate finder, never imported by the package itself
    __init__.py        the public names
    grid.py            f_plot and DensityGrid, his density grid counted with one bincount
    peaks.py           local_maxima, search_ind, refine_peaks and find_area
    gate.py            hull_gate and lymphmacro_finder
  plot/                the matplotlib views, never imported by the package itself
    __init__.py        the matplotlib check and the public names
    tree.py            dendrogram (plot.hclust's lines)
    flow.py            gates, his figure of the finder's populations
    points.py          clusters and nuggets, the help examples' weight colors, the size rules
    pca.py             scores, scree and loadings, with the sign rule
    style.py           style, save, PAPER_RC and the palettes
  _compat/             R's arithmetic and numerics, bit for bit
    arithmetic.py      long double sums, mean, cov, median, mad, quantile type 7, weighted.mean
    missing.py         R's NA and NaN bits, and sums that meet a non-finite value
    indexing.py        parallel::splitIndices and the rows unique() keeps
    linalg.py          %*% and eigen() of a symmetric matrix
    moments.py         every nugget's colMeans and cov in one compiled call, with its threading
    rfast.py           Rfast's Dist and dista, Armadillo's accumulation order, libstdc++'s sort
    rfastjit.py        Dist, dista and dista's nearest center compiled, in rfast's order
    ldouble.py         x87 long double replayed with Python integers, the reference
    ldjit.py           the same long double arithmetic compiled with numba
    _compiled.py       the numba decorator every kernel goes through, and read-only input views
  _random/             sources of randomness
    interfaces.py      the Draws protocol: the R sampling calls the ports make
    numpy_draws.py     NumpyDraws, what an int or Generator seed gives
    r_draws.py         RDraws, R's set.seed and sample.int
    replay.py          ReplayDraws, the draws R recorded for a fixture
    seeds.py           as_draws, what seed= becomes
  _clustering/         algorithms the workflows call internally
    hartigan_wong.py   R's kmeans with the Hartigan-Wong algorithm
    kmnsjit.py         kmns.f compiled, statement for statement with hartigan_wong's port
  create.py  refine.py  centers.py  means.py  complexity.py  _draws.py
                       old import paths that only forward (see Retained import paths)
```

## Public and private

The public API is `datanuggets.__all__`, imported from `datanuggets` as the README does. `datanuggets.nuggets` and `datanuggets.wcluster` re-export the same functions from the modules that define them. `result` and `exceptions` hold the public classes. `datanuggets.plot` is public too, with its own `__all__`, but it's imported only when asked for by name, since it needs matplotlib from the `plot` extra. Without matplotlib, importing it raises an `ImportError` that names the extra. `datanuggets.flow` is public in the same way, with its own `__all__`, and is imported only when asked for by name. Its own modules import numpy, scipy and `_validate`, though importing it runs `datanuggets/__init__.py`, numba included, as any submodule does.

A module or package whose name starts with an underscore is private. It can change without notice, and code outside this repo shouldn't import it. `datanuggets._draws` is the one private path kept working on purpose, for the reasons below.

`datanuggets/__init__.py` imports the two workflow packages, the result and warning classes and the six old module paths, and does nothing else. Importing an old path binds it as an attribute of the package, as the old `__init__` did when it imported those modules directly, so `datanuggets.create.create_dn` works after a bare `import datanuggets`. Importing the package compiles no kernel, starts no thread or process, draws no random number and doesn't import matplotlib.

## Dependency direction

Imports run from the workflows down to the R-compatibility layer and never back up.

| Layer | Imports from the package |
|---|---|
| `_validate`, `result`, `exceptions`, `_defaults` | nothing |
| `_compat` | `_compat`, `_defaults` |
| `_random` | `_random`, `exceptions` |
| `_clustering` | `_compat`, `_random` |
| `wcluster` | `wcluster`, `_compat`, `_random`, `_validate`, `result`, `exceptions` |
| `nuggets` | `nuggets`, `_clustering`, `_compat`, `_random`, `_defaults`, `_validate`, `result`, `exceptions` |
| `flow` | `flow`, `_validate` |
| `plot` | `plot`, `wcluster`, `_compat` |
| the old paths: `create`, `refine`, `centers`, `means`, `complexity`, `_draws` | `nuggets`, `_random` |
| `datanuggets/__init__.py` | `nuggets`, `wcluster`, the old paths, `result`, `exceptions` |

So R's arithmetic and the draw sources never import a workflow, and new code in them can't create a cycle. Wkmeans draws its starts through `_random`, the other WCluster functions draw nothing, and `wcluster` never imports `nuggets`: the DN wrappers take a `DataNuggets` from `result`. `plot` sits on top: it imports `wcluster` for R's merge numbering and `_compat` for the sum and quantile in R's nugget colors, and nothing in the package imports it. `flow` sits beside the workflows and imports only `_validate`. `plot.gates` takes the finder's masks as an array, so `plot` doesn't import `flow` either. The table is the whole import graph of the package by layer, imports inside a layer and inside functions included.

## Where new code goes

| Adding | Goes in | Also |
|---|---|---|
| a datanugget function | `nuggets/<name>.py` | export it from `nuggets/__init__.py` and `datanuggets/__init__.py` |
| a WCluster function | `wcluster/` | the same two exports. A new cluster statistic joins `wstats.py` when it shares its helpers |
| a step of the flow gate finder | `flow/`: the grid in `grid`, peaks and areas in `peaks`, gating in `gate` | export it from `flow/__init__.py` only. `benchmarks/flow/compare.py` gets a row that times it against his piece and checks the output |
| a plot | the `plot/` module for what it draws: trees in `tree`, points in `points`, PCA results in `pca`, the flow figure in `flow`, settings and saving in `style` | export it from `plot/__init__.py` only. It takes `ax=`, returns the Axes, and checks its input before it draws anything |
| a piece of R arithmetic | the `_compat` module named for what it reproduces: sums and statistics in `arithmetic`, NA and NaN bits in `missing`, matrix products and eigen in `linalg`, index helpers in `indexing`, Rfast's code in `rfast` | keep it out of `_validate`, which only checks and converts arguments |
| a compiled kernel | a module whose name ends in `jit`, made with `compiled` from `_compat/_compiled.py` | `tests/compat/test_compiled.py` finds the `*jit` modules at any depth and fails a numba decorator anywhere else |
| a sampling routine or seed rule | `_random/`. A new source implements `Draws` from `interfaces.py`, and `seeds.as_draws` decides what `seed=` becomes | |
| an algorithm a workflow calls | `_clustering/` | |
| a default argument value or internal size | `_defaults.py` | |
| a report or audit tool | `scripts/` | |
| a timing or memory measurement | `benchmarks/` | |

## Tests

| Folder | Covers |
|---|---|
| `tests/nuggets/` | `nuggets/` |
| `tests/wcluster/` | `wcluster/`, the iterative functions in `test_whclust.py`, `test_wkmeans.py` and `test_wstatsjit.py`, the DN wrappers in `test_dn.py`, and `cut_tree` in `test_cutree.py` |
| `tests/flow/` | `flow/`, against fixtures recorded from his own Python. `test_flow_import.py` runs the finder with matplotlib blocked |
| `tests/plot/` | `plot/`, on matplotlib's Agg backend. Its `conftest.py` closes every figure and turns matplotlib's `UserWarning`s into failures |
| `tests/compat/` | `_compat/` |
| `tests/randomness/` | `_random/` |
| `tests/clustering/` | `_clustering/` |
| `tests/` | `test_result.py` (the result classes and the import paths), `test_validate.py` (`_validate`), `test_compat_inputs.py` (R's handling of inputs across the public functions, from the `compat` fixture group), `conftest.py`, `rfixtures.py`, and `flowcases.py`, which loads the flow fixtures and holds the flow tests' independent area rules and edge check |

A test file holds the R parity cases and the unit cases for the module it covers. Each file mixes both against the same fixtures, so the suite is grouped by area and not into unit and parity folders. The long fuzz cases carry the `slow` mark and run with `pytest -m slow`.

- Test folders have no `__init__.py`, and pytest imports each test file by its file name, so a file name must be unique across all the folders.
- In its default import mode, prepend, pytest puts the folder of each `conftest.py` on `sys.path` when it loads that file, before it imports any test module. `tests/conftest.py` is there for that, and it's how a test in any folder imports `rfixtures`. So the suite runs in the default mode and not with `--noconftest`, and `--import-mode=importlib` doesn't work with it, before the reorganization or after. Tests don't change `sys.path` themselves.
- A test that patches a name patches the module that looks it up when the code runs. `group_moments` looks up `r_mean` and `r_cov` in `moments`, for example, so a test patches `moments.r_cov`, not `arithmetic.r_cov`.

## Fixtures

- Each case lives in `tests/fixtures/<group>/<case>/`: a `manifest.json` and its arrays as `.bin` files.
- Tests read them through `rfixtures.py`, and `rfixtures.FIXTURE_ROOT` is the one place that names the location.
- The `r/fixtures_*.R` scripts record the groups, each case through `write_case` in `r/fixture_io.R`, in the pinned R environment `r/README.md` describes. Most groups come from the script of the same name, and nine don't. `fixtures_rcompat.R` writes `rcompat_split_indices`, `rcompat_dist`, `rcompat_dista`, `rcompat_rowmins` and `rcompat_means`. `fixtures_rstats.R` writes `rcompat_stats` and `fixtures_rwstats.R` writes `rcompat_wstats`. `fixtures_kmeans.R` also writes `kmns`, and `fixtures_wstats.R` also writes `wstats_constructed`. To find the script for any group, search `r/` for its name. A script that reads another group's cases back names that group too, so `create_dn`, `kmeans` and `refine_dn` turn up in more than one script, and the one that records the group is the one whose cases go to `write_case` under that name.
- The flow fixtures are the exception. `tests/fixtures/flow/` holds three `.npz` files recorded by `benchmarks/flow/record.py` from Prof. Cabrera's Python on real PBMC files, not from R, with their source and keys in its `README.md`.
- Fixtures are evidence of what R did. They're never regenerated to make a test pass, and the reorganization left every fixture byte where it was.

## Retained import paths

| Old path | Forwards to | Kept because | Tested by |
|---|---|---|---|
| `datanuggets.create`, `.refine`, `.centers`, `.means`, `.complexity` | the function of the same name in `datanuggets.nuggets` | they were the module paths of the public functions | `tests/test_result.py::test_nugget_functions_are_exported_and_their_old_paths_forward` |
| `datanuggets._draws` | the eight public names now in `datanuggets._random` | scripts outside the repo import it, and a pickle of a draw source made before the split names it | `tests/randomness/test_draws.py`, the last two tests |

A forwarding module imports the names and lists them in `__all__`, so each name keeps one implementation. Only those names forward. A private helper is reachable only at its new path.

`datanuggets/__init__.py` imports all six old paths, so after a bare `import datanuggets` each is an attribute of the package, as it was before. `tests/test_result.py::test_a_bare_import_binds_the_old_module_paths` checks that in a fresh interpreter. None of the six is in `__all__`, so `from datanuggets import *` gives the same names as before.

The other moved private modules (`_rcompat`, `_rstats`, `_ldjit`, `_ldouble`, `_rfast`, `_compiled`, `_hartigan_wong`) don't forward. Nothing outside the repo imports them, and none defines a class that gets pickled.

## Old paths in older documents

The plans, specs and reviews in `docs/` record their day and keep the paths they were written with. Line numbers in them refer to the files as they were then. This section translates their paths.

| Old file | Now |
|---|---|
| `src/datanuggets/create.py`, `refine.py`, `centers.py`, `means.py`, `complexity.py` | `src/datanuggets/nuggets/`, same file names. The old files now only forward |
| `src/datanuggets/_hartigan_wong.py` | `src/datanuggets/_clustering/hartigan_wong.py` |
| `src/datanuggets/_compiled.py` | `src/datanuggets/_compat/_compiled.py` |
| `src/datanuggets/_ldjit.py` | `src/datanuggets/_compat/ldjit.py` |
| `src/datanuggets/_ldouble.py` | `src/datanuggets/_compat/ldouble.py` |
| `src/datanuggets/_rfast.py` | `src/datanuggets/_compat/rfast.py` |
| `src/datanuggets/_rcompat.py`, `_rstats.py` | split across `src/datanuggets/_compat/`, by name below |
| `src/datanuggets/_draws.py` | split across `src/datanuggets/_random/`, by name below. The old file now only forwards |
| `tests/test_centers.py`, `test_complexity.py`, `test_create.py`, `test_means.py`, `test_refine.py` | `tests/nuggets/` |
| `tests/test_predict.py`, `test_wpca.py`, `test_wpca_sums.py`, `test_wstats.py` | `tests/wcluster/` |
| `tests/test_compiled.py`, `test_ldjit.py`, `test_rcompat.py`, `test_rfast.py`, `test_rstats.py` | `tests/compat/` |
| `tests/test_draws.py`, `test_rdraws.py` | `tests/randomness/` |
| `tests/test_hartigan_wong.py` | `tests/clustering/` |

| Was in | Names | Now in `_compat/` or `_random/` |
|---|---|---|
| `_rcompat` | `split_indices`, `unique_first`, `_r_seq_length_out` | `_compat/indexing.py` |
| `_rcompat` | `r_mean`, `_r_mean_py`, `_ldouble_mean`, `_round_half_even` | `_compat/arithmetic.py` |
| `_rcompat` | `_R_NAN`, `_NA_BITS`, `_QUIET`, `_FRACTION`, `_INDEFINITE`, `_non_finite_sum` | `_compat/missing.py` |
| `_rcompat` | `_LDOUBLE_DIGITS` | `_compat/ldouble.py` as `DIGITS`, which `arithmetic.py` imports under the old name |
| `_rstats` | `r_sum`, `r_vec_mean`, `r_median`, `r_mad`, `r_quantile7`, `r_cov`, `_r_cov_py`, `r_weighted_mean`, `_ld_sum`, `_DBL_MAX` | `_compat/arithmetic.py` |
| `_rstats` | `group_moments`, `_run_threaded`, `_layer_forbids_fork`, `_after_fork_in_child`, `_THREADS`, `_FORKED_UNSAFE`, and the `os.register_at_fork` call | `_compat/moments.py` |
| `_rstats` | `r_matvec`, `r_matmul`, `r_eigen_sym` | `_compat/linalg.py` |
| `_draws` | `Draws` | `_random/interfaces.py` |
| `_draws` | `NumpyDraws` | `_random/numpy_draws.py` |
| `_draws` | `RDraws`, `r_use_hash`, `_r_seed`, `_M32`, `_I2_32M1`, `_TWO_M32`, `_MT_KINDS`, `_HASH_TRIES` | `_random/r_draws.py` |
| `_draws` | `DrawRecord`, `ReplayDraws`, `ReplayMismatch` | `_random/replay.py` |
| `_draws` | `as_draws` | `_random/seeds.py` |
