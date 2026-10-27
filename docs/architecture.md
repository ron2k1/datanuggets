# Architecture

Where the code, tests, fixtures and tools of datanuggets live, which way the imports run, and where new work goes. The last section maps module paths used before the package reorganization.

## Repository

| Folder | Holds |
|---|---|
| `src/datanuggets/` | the package, the only folder that ships in the wheel |
| `tests/` | the pytest suite, `rfixtures.py` and the R fixtures it reads, and `flowcases.py` and the flow fixtures |
| `scripts/` | `parity_report.py`, which rebuilds the tables in `docs/validation/`, and `check_dist.py`, which checks a built sdist and wheel before they ship |
| `benchmarks/` | Python comparison and profiling tools; the Covertype and WCluster comparisons require local R baseline records that are not distributed. `benchmarks/flow/` contains the Python flow reference and comparisons |
| `.github/workflows/` | `tests.yml`, `matrix.yml`, `package.yml` and `release.yml`, which [development.md](development.md#what-ci-runs) describes |
| `docs/` | the pages for users and maintainers: installation, quickstart, the `datanuggets-flow` command, methodology, compatibility, reproducibility, development, limitations, releasing and this one |
| `examples/` | runnable examples, starting with `first_analysis.py` |
| `docs/validation/` | one parity table per public function, written by `scripts/parity_report.py` |

The package imports nothing from `tests/`, `scripts/` or `benchmarks/`. The wheel contains the Python package and its metadata. The source archive adds the tests and fixtures (except `tests/scripts/`), user guides, `examples/first_analysis.py`, README, license, `pyproject.toml` and `.gitignore`. Maintainer scripts, benchmarks, internal notes and R source files do not ship.

Internal specs, plans and reviews are local files in ignored folders under `docs/`. R source files, the local `r/` tooling and its benchmark workflow are ignored too. Recorded reference fixtures remain tracked so ordinary tests need only Python.

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
  flow/                Prof. Cabrera's flow gate finder and the datanuggets-flow command, never
                       imported by the package itself
    __init__.py        the finder's public names
    grid.py            f_plot and DensityGrid, Cabrera's density grid counted with one bincount
    peaks.py           local_maxima, search_ind, refine_peaks, find_area, and mirrored_area, the
                       copy of cfc.py's find_area that --disc his gates with
    gate.py            hull_gate and lymphmacro_finder, which takes the area rule as area=
    pick.py            the datanuggets-flow command (main), one function per step: FCS files read
                       with flowio, gated, picked and clustered
  plot/                the matplotlib views, imported in the package only by flow/pick.py
    __init__.py        the matplotlib check and the public names
    tree.py            dendrogram (plot.hclust's lines)
    flow.py            gates, Cabrera's figure of the finder's populations
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
    kmnsjit.py         kmns.f compiled, statement for statement with the plain-Python port in
                       tests/clustering/kmns_reference.py
  create.py  refine.py  centers.py  means.py  complexity.py  _draws.py
                       old import paths that only forward (see Retained import paths)
```

## Public and private

The public API is `datanuggets.__all__`, imported from `datanuggets` as the README does. `datanuggets.nuggets` and `datanuggets.wcluster` re-export the same functions from the modules that define them. `result` and `exceptions` hold the public classes. `datanuggets.plot` is public too, with its own `__all__`, but it's imported only when asked for by name, since it needs matplotlib from the `plot` extra. Without matplotlib, importing it raises an `ImportError` that names the extra. `datanuggets.flow` is public in the same way, with its own `__all__`, and is imported only when asked for by name. The finder's modules import numpy, scipy and `_validate`, though importing `datanuggets.flow` runs `datanuggets/__init__.py`, numba included, as any submodule does. `datanuggets.flow.pick` is the `datanuggets-flow` command, the entry point `datanuggets.flow.pick:main` in `pyproject.toml`, and `datanuggets.flow` doesn't import it. At module level it imports numpy, the standard library and `_defaults`, for `create_dn`'s first-stage count, which caps `--dn-num2`. Its steps import matplotlib and pandas, the workflows, `plot` and the finder from `datanuggets.flow` only when they run, and `read_fcs` imports flowio. The `flow` extra installs matplotlib, pandas and flowio.

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
| `flow`, the finder: `__init__.py`, `grid`, `peaks`, `gate` | `flow`, `_validate` |
| `plot` | `plot`, `wcluster`, `_compat` |
| `flow/pick.py`, the `datanuggets-flow` command | `datanuggets` itself (for `create_dn`, `whclust`, `cut_tree`, `wmean` and `wpca`), `plot`, `flow`, `_defaults`, and `result` for type hints only |
| the old paths: `create`, `refine`, `centers`, `means`, `complexity`, `_draws` | `nuggets`, `_random` |
| `datanuggets/__init__.py` | `nuggets`, `wcluster`, the old paths, `result`, `exceptions` |

So R's arithmetic and the draw sources never import a workflow, and new code in them can't create a cycle. Wkmeans draws its starts through `_random`, the other WCluster functions draw nothing, and `wcluster` never imports `nuggets`: the DN wrappers take a `DataNuggets` from `result`. `plot` sits above the workflows: it imports `wcluster` for R's merge numbering and `_compat` for the sum and quantile in R's nugget colors, and in the package only `flow/pick.py` imports it. The finder's modules in `flow` sit beside the workflows and import only `_validate`. `plot.gates` takes the finder's masks as an array, so `plot` doesn't import `flow` either. `flow/pick.py` is an application layer above `plot`: it reaches `nuggets` and `wcluster` through `datanuggets/__init__.py`, draws with `plot` and gates with `lymphmacro_finder` and an area rule from `flow`, and nothing in the package imports it, `datanuggets.flow` included. The table is the whole import graph of the package by layer, imports inside a layer and inside functions included.

## Where new code goes

| Adding | Goes in | Also |
|---|---|---|
| a datanugget function | `nuggets/<name>.py` | export it from `nuggets/__init__.py` and `datanuggets/__init__.py` |
| a WCluster function | `wcluster/` | the same two exports. A new cluster statistic joins `wstats.py` when it shares its helpers |
| a step of the flow gate finder | `flow/`: the grid in `grid`, peaks and areas in `peaks`, gating in `gate` | export it from `flow/__init__.py` only. `benchmarks/flow/compare.py` gets a row that times it against the reference code's piece and checks the output |
| an option or output of `datanuggets-flow` | `flow/pick.py` | document it in `docs/cli.md`, and add its cases to `tests/flow/test_flow_pick.py`. A changed output changes the table digests its end-to-end tests pin, which are recorded again only for a change made on purpose |
| a plot | the `plot/` module for what it draws: trees in `tree`, points in `points`, PCA results in `pca`, the flow figure in `flow`, settings and saving in `style` | export it from `plot/__init__.py` only. It takes `ax=`, returns the Axes, and checks its input before it draws anything |
| a piece of R arithmetic | the `_compat` module named for what it reproduces: sums and statistics in `arithmetic`, NA and NaN bits in `missing`, matrix products and eigen in `linalg`, index helpers in `indexing`, Rfast's code in `rfast` | keep it out of `_validate`, which only checks and converts arguments |
| a compiled kernel | a module whose name ends in `jit`, made with `compiled` from `_compat/_compiled.py` | `tests/compat/test_compiled.py` finds the `*jit` modules at any depth and fails a numba decorator anywhere else |
| a sampling routine or seed rule | `_random/`. A new source implements `Draws` from `interfaces.py`, and `seeds.as_draws` decides what `seed=` becomes | |
| an algorithm a workflow calls | `_clustering/` | |
| a default argument value or internal size | `_defaults.py` | |
| a report, audit or release check | `scripts/` | its tests, if it has any, in `tests/scripts/`, which the sdist leaves out |
| a timing or memory measurement | `benchmarks/` | |

## Tests

| Folder | Covers |
|---|---|
| `tests/nuggets/` | `nuggets/`, one file per module: `test_create.py`, `test_centers.py`, `test_refine.py`, `test_means.py` and `test_complexity.py` |
| `tests/wcluster/` | `wcluster/`: the statistics in `test_wstats.py`, `cluster_predict` in `test_predict.py`, `wpca` in `test_wpca.py` and its row-weight total in `test_wpca_sums.py`, the iterative functions in `test_whclust.py` and `test_wkmeans.py` and the kernels they share in `test_wstatsjit.py`, the DN wrappers in `test_dn.py`, and `cut_tree` in `test_cutree.py` |
| `tests/flow/` | `flow/`. `test_flow_grid.py`, `test_flow_peaks.py`, `test_flow_area.py` and `test_flow_gate.py` test the finder's steps against fixtures recorded from Prof. Cabrera's own Python. `test_flow_import.py` runs the finder with matplotlib blocked and checks that `import datanuggets` doesn't load `flow`. `test_flow_pick.py` unit-tests `datanuggets-flow`'s threshold and population parsing, the union of picks, the file listing, the prompt and the checks that stop a run before any file is read, and runs the command end to end on the events of `202.npz` and `202_fluor.npz`, one file asked or given `--pops` and three files with their own thresholds, checking its output files, their columns and their tables' digests |
| `tests/plot/` | `plot/`, on matplotlib's Agg backend, one file per module: `test_plot_tree.py`, `test_plot_points.py`, `test_plot_pca.py`, `test_plot_flow.py` and `test_plot_style.py`, and `test_plot_import.py` for the matplotlib check. Its `conftest.py` closes every figure and turns matplotlib's `UserWarning`s into failures |
| `tests/compat/` | `_compat/`: R's means and `splitIndices` in `test_rcompat.py`, the long double statistics, `unique()`, `%*%`, `weighted.mean` and `eigen()` in `test_rstats.py`, Rfast in `test_rfast.py`, the compiled long double kernels in `test_ldjit.py`, and the numba decorator rule and cache in `test_compiled.py` |
| `tests/randomness/` | `_random/`: the draw sources and `as_draws` in `test_draws.py`, and `RDraws` against R in `test_rdraws.py` |
| `tests/clustering/` | `_clustering/`, in `test_hartigan_wong.py`, which holds the compiled kmns.f to the plain-Python port in `kmns_reference.py` |
| `tests/scripts/` | `scripts/check_dist.py`, in `test_check_dist.py`, on small sdists and wheels built by the tests. The sdist leaves this folder out, so these tests run only from a clone |
| `tests/` | `test_result.py` (the result classes and the import paths), `test_validate.py` (`_validate`), `test_compat_inputs.py` (R's handling of inputs across the public functions, from the `compat` fixture group), `test_version.py` (the version, written once in `pyproject.toml` and read from the package metadata), `conftest.py`, `rfixtures.py`, and `flowcases.py`, which loads the flow fixtures and holds the flow tests' independent area rules and edge check |

A test file holds the R parity cases and the unit cases for the module it covers. Each file mixes both against the same fixtures, so the suite is grouped by area and not into unit and parity folders. The long fuzz cases carry the `slow` mark and run with `pytest -m slow`.

- Test folders have no `__init__.py`, and pytest imports each test file by its file name, so a file name must be unique across all the folders. That includes a helper beside the tests, such as `kmns_reference.py`: pytest puts a test file's folder on `sys.path` before it imports the file, and the test imports the helper by its bare name.
- In its default import mode, prepend, pytest puts the folder of each `conftest.py` on `sys.path` when it loads that file, before it imports any test module. `tests/conftest.py` is there for that, and it's how a test in any folder imports `rfixtures`. So the suite runs in the default mode and not with `--noconftest`, and `--import-mode=importlib` doesn't work with it, before the reorganization or after. Tests don't change `sys.path` themselves.
- A test that patches a name patches the module that looks it up when the code runs. `group_moments` looks up `r_mean` and `r_cov` in `moments`, for example, so a test patches `moments.r_cov`, not `arithmetic.r_cov`.

## Fixtures

- Each case lives in `tests/fixtures/<group>/<case>/`: a `manifest.json` and its arrays as `.bin` files.
- Tests read them through `rfixtures.py`, and `rfixtures.FIXTURE_ROOT` is the one place that names the location.
- R fixture generators recorded the inputs, draws and outputs in the reference environment described in [reproducibility.md](reproducibility.md#what-r-recorded). Those generators are local maintainer tools and are not distributed; the recorded data is included.
- The flow fixtures are the exception. `tests/fixtures/flow/` holds three `.npz` files recorded by `benchmarks/flow/record.py` from Prof. Cabrera's Python on real PBMC files, not from R, and `202_fluor.npz`, recorded by `benchmarks/flow/record_fluor.py`: three fluorescence channels of the same `202.fcs` events, for the command's end-to-end tests. Their source and keys are in its `README.md`.
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

Older references may use paths from before the package reorganization. These tables map them to the current modules.

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
