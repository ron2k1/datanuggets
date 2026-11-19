# The package's layout

How `src/datanuggets` is laid out. This page covers how it is divided by feature, what is public, what ships and which way its imports run. It also gives the rules its imports keep, where new code goes, its tests and fixtures, and the one old import path it keeps. [ARCHITECTURE.md](../ARCHITECTURE.md) places the package among the repository's other folders.

## Domain map

| Folder or module | Responsibility |
|---|---|
| `__init__.py` | the public API: imports, `__version__` and `__all__`, nothing else |
| `nuggets/` | the datanugget ports: creating, refining and summarizing nuggets |
| `wcluster/` | the WCluster ports: weighted statistics, prediction, PCA, Whclust, Wkmeans, the DN wrappers and `cut_tree` |
| `plot/` | the matplotlib views, which only the modules that draw import |
| `colors.py`, `filters.py`, `projection.py` | cluster colours, row filters, and PCA and t-SNE as numbers |
| `result.py`, `exceptions.py` | the public result classes and the warning |
| `_validate.py`, `_defaults.py` | argument checks, and R's numeric defaults for the datanugget functions, the default seed, the helper modules' settings and some internal sizes |
| `_compat/`, `_random/`, `_clustering/` | R's arithmetic bit for bit, the sources of randomness, and R's Hartigan-Wong k-means, which nugget refinement calls |
| `_draws.py` | the draw sources' old import path, kept on purpose |

A folder or module whose name starts with an underscore is private, and its layout can change without notice. To find where a function is defined, search the source. This page names folders and jobs, not functions.

## Public and private

The public API is `datanuggets.__all__`, imported from `datanuggets` as the README does. `datanuggets.nuggets` and `datanuggets.wcluster` re-export the same functions from the modules that define them, and `result` and `exceptions` hold the public classes. The other public modules are imported only when asked for by name, and [api.md](../api.md) documents each one:

- `datanuggets.plot` needs matplotlib from the `plot` extra. Without it, importing `datanuggets.plot` raises an `ImportError` that names the extra.
- `datanuggets.colors`, `.filters` and `.projection` give cluster colours as hex strings, row filters, and PCA and t-SNE coordinates as arrays. They import without matplotlib or pandas. t-SNE needs scikit-learn from the `tsne` extra.

A module or package whose name starts with an underscore is private. It can change without notice, and code outside this repo shouldn't import it. `datanuggets._draws` is the one private path kept working on purpose, for the reasons under [Retained import paths](#retained-import-paths).

`datanuggets/__init__.py` imports `nuggets`, `wcluster`, `RDraws` and the result and warning classes, and reads `__version__` from the installed metadata. It does nothing else. Importing the package compiles no kernel, starts no thread or process, draws no random number and doesn't import matplotlib.

## What ships

The wheel contains the Python package and its metadata. The source archive adds the tests and fixtures (except `tests/scripts/` and `tests/fixtures/housing/`), the user guides, `examples/first_analysis.py`, the README, `LICENSE`, `THIRD_PARTY.md`, `pyproject.toml` and `.gitignore`. Its list is `only-include` under `[tool.hatch.build.targets.sdist]` in `pyproject.toml`, and `scripts/check_dist.py` checks a built archive against it. Maintainer scripts, benchmarks, the maintainer docs (this page among them) and R source files do not ship.

Internal specs, plans and reviews are local files in ignored folders under `docs/`. R source files, the local `r/` tooling and its benchmark workflow are ignored too. Recorded reference fixtures remain tracked, so ordinary tests need only Python.

## Dependency direction

Imports run down to R's arithmetic and never back up. Each layer imports only from the layers listed for it, and from inside itself.

| Layer | Imports from the package |
|---|---|
| `result`, `exceptions`, `_validate`, `_defaults` | nothing |
| `_compat`, `_random` | the first layer |
| `_clustering` | `_compat`, `_random` |
| `wcluster` | `_compat`, `_random`, and the first layer |
| `nuggets` | the same, and `_clustering` |
| `datanuggets/__init__.py` | `nuggets`, `wcluster`, `result`, `exceptions`, and `_random` for `RDraws` |
| `colors`, `filters`, `projection` | `wcluster`, and the first layer |
| `plot` | `wcluster`, `_compat`, `projection` |
| the old path `_draws` | `_random` |

R's arithmetic and the draw sources never import `nuggets`, `wcluster` or any module that imports them, so new code in them can't create a cycle. `wcluster` never imports `nuggets`: the DN wrappers take a `DataNuggets` from `result`. Only the `plot` package reaches matplotlib (rule 2 below). `tests/test_import_graph.py` fails on any cycle and on any module outside `plot` that reaches drawing. The table is the intended layering. That test checks for cycles and for drawing, not each row of the table.

## Rules

1. **No import cycles.** The rule counts every import, including those inside functions and under `TYPE_CHECKING`, because a lazy import is one edit from a load-time one. `tests/test_import_graph.py` reads the graph from the source and fails on any cycle.
2. **The core never imports plotting.** Only the modules that draw may reach matplotlib or `datanuggets.plot`, whether directly or through another module. The `plot` package is the only one, and `DRAWS` in `tests/test_import_graph.py`, which would name any other, is empty. Adding a module there is a design change, made in review, not a way to make the test pass.
3. **Optional packages load where they are used.** pandas, tqdm and scikit-learn are imported inside the function that needs them. Only the `plot` package imports matplotlib, and `import datanuggets` works without it (`tests/plot/test_plot_import.py`).
4. **Inside the package, import from the defining module.** A module imports a name from the part that defines it, not from a module that re-exports it, so it loads only what it uses.
5. **Patch where the name is looked up.** A test that stands in for a function patches the module whose code calls it when it runs.
6. **One job per module.** A module is split when it takes on a second job, not when it passes a line count. Moving code is its own commit, with the bodies unchanged, and new behaviour comes in a later commit.

## Where new code goes

| Adding | Goes in | Also |
|---|---|---|
| a datanugget function | `nuggets/<name>.py` | export it from `nuggets/__init__.py` and `datanuggets/__init__.py` |
| a WCluster function | `wcluster/` | the same two exports. A new cluster statistic joins `wstats.py` when it shares its helpers |
| a public helper outside the R ports | `colors.py`, `filters.py` or `projection.py` when it fits their job, or a new module named for its job | list it in `docs/api.md`, which puts it under `test_public_paths.py` |
| a plot | the `plot/` module for what it draws: trees in `tree`, points in `points`, PCA results in `pca`, settings and saving in `style` | export it from `plot/__init__.py` only. It takes `ax=`, returns the Axes, and checks its input before it draws anything |
| a piece of R arithmetic | the `_compat` module named for what it reproduces: sums and statistics in `arithmetic`, NA and NaN bits in `missing`, matrix products and eigen in `linalg`, index helpers in `indexing`, Rfast's code in `rfast` | keep it out of `_validate`, which only checks and converts arguments |
| a compiled kernel | a module whose name ends in `jit`, made with `compiled` from `_compat/_compiled.py` | `tests/compat/test_compiled.py` finds the `*jit` modules at any depth and fails a numba decorator anywhere else |
| a sampling routine or seed rule | `_random/`. A new source implements `Draws` from `interfaces.py`, and `seeds.as_draws` decides what `seed=` becomes | |
| an algorithm the R ports call internally | `_clustering/` | |
| a default argument value or internal size | `_defaults.py` | |
| a report, audit or release check | `scripts/` | its tests, if it has any, in `tests/scripts/`, which the sdist leaves out |
| a timing or memory measurement | `benchmarks/` | |

## Left whole for now

These modules hold more than one job but are not split yet. Each split needs a decision of its own.

- `_compat/ldjit.py`: the long double primitives and the moment kernels built on them, compiled code held bit for bit to R.

## Tests

The suite is grouped by area, and its folders follow the package's: `tests/nuggets/`, `tests/wcluster/`, `tests/plot/`, `tests/compat/` for `_compat`, `tests/randomness/` for `_random`, `tests/clustering/` for `_clustering`, and `tests/scripts/` for `scripts/check_dist.py`, `scripts/parity_report.py` and the version in `CITATION.cff` (the sdist leaves that folder out, so those tests run only from a clone). A module's tests are named for it, and a module that has none yet is found by searching `tests/` for its name. The files at the top of `tests/` cover the modules at the top of the package: the result classes, `_validate`, R's handling of inputs across the public functions, the version, `colors`, `filters` and `projection`. `test_docstring_examples.py` runs the Examples section in the docstring of each function in `datanuggets.__all__`. The two boundary tests sit there too: `test_import_graph.py` and `test_public_paths.py`.

A test file holds the R parity cases and the unit cases for the module it covers. Each file mixes both against the same fixtures, so the suite is not split into unit and parity folders. A compiled kernel is held to a plain-Python reference beside its tests, such as `tests/clustering/kmns_reference.py`. The long fuzz cases carry the `slow` mark and run with `pytest -m slow`.

- Test folders have no `__init__.py`, and pytest imports each test file by its file name, so a file name must be unique across all the folders. That includes a helper beside the tests, such as `kmns_reference.py`: pytest puts a test file's folder on `sys.path` before it imports the file, and the test imports the helper by its bare name.
- In its default import mode, prepend, pytest puts the folder of each `conftest.py` on `sys.path` when it loads that file, before it imports any test module. `tests/conftest.py` is there for that, and it's how a test in any folder imports `rfixtures`. So the suite runs in the default mode and not with `--noconftest`, and `--import-mode=importlib` doesn't work with it. Tests don't change `sys.path` themselves.
- A test that patches a name patches the module that looks it up when the code runs (rule 5). `group_moments` looks up `r_mean` and `r_cov` in `moments`, for example, so a test patches `moments.r_cov`, not `arithmetic.r_cov`.

## Fixtures

- Each R case lives in `tests/fixtures/<group>/<case>/`: a `manifest.json` and its arrays as `.bin` files.
- Tests read them through `rfixtures.py`, and `rfixtures.FIXTURE_ROOT` is the one place that names the location.
- R fixture generators recorded the inputs, draws and outputs in the reference environment described in [reproducibility.md](../reproducibility.md#what-r-recorded). Those generators are local maintainer tools and are not distributed; the recorded data is included.
- `tests/fixtures/housing/california_housing.csv` is the California housing data as scikit-learn distributes it. Two of the colour tests cluster it at k = 100. The sdist leaves it out, and those two tests skip there.
- Fixtures are evidence of what R did. They're never regenerated to make a test pass.

## Retained import paths

| Old path | Forwards to | Kept because | Tested by |
|---|---|---|---|
| `datanuggets._draws` | the eight public names now in `datanuggets._random` | scripts outside the repo import it, and a pickle of a draw source made before the split names it | `tests/randomness/test_draws.py`, the last two tests |

`_draws` imports the names and lists them in `__all__`, so each name keeps one implementation. Only those names forward. A private helper is reachable only at its new path.

`datanuggets.create`, `.refine`, `.centers`, `.means` and `.complexity` forwarded to `datanuggets.nuggets` the same way until they were removed. Each function had two module paths, and no guide imported from the old ones. [releases.md](../releases.md#removed-import-paths) lists what replaces them, and `tests/test_result.py::test_the_old_nugget_module_paths_are_gone` checks that they stay removed.

`datanuggets/__init__.py` doesn't import `_draws`, so a bare `import datanuggets` doesn't make it an attribute of the package. Scripts outside the repo import it by name, as in `from datanuggets._draws import DrawRecord, ReplayDraws`.

The other moved private modules (`_rcompat`, `_rstats`, `_ldjit`, `_ldouble`, `_rfast`, `_compiled`, `_hartigan_wong`) don't forward. Nothing outside the repo imports them, and none defines a class that gets pickled.
