# Release notes

## 0.1.0a3: library only

This version is the Python library alone: the port of 17 functions from R's `datanugget` 1.5.0 and `WCluster` 1.3.0 and of R's `cutree`, their result objects, `RDraws` for R's random draws, the plots in `datanuggets.plot`, and the helpers in `datanuggets.colors`, `datanuggets.filters` and `datanuggets.projection`. The command-line tools, the flow cytometry tools and the web app that earlier development versions held are not part of it. Neither 0.1.0a2 nor 0.1.0a1 was uploaded to PyPI.

### Added

- Numpydoc docstrings for the 18 functions that `datanuggets` exports. Each lists its parameters, return value and errors. The result classes and `RDraws` have them too.
- An Examples section in each of those 18 docstrings, as R's help pages have. A test runs every example with doctest.
- A [math reference](https://github.com/ron2k1/datanuggets/blob/main/docs/math/README.md) in `docs/math/`. It states what each of the 18 functions computes, as equations, and names the tests that hold each statement. It is a draft that the authors of the R packages have not reviewed.
- Parity pages in `docs/validation/` for `wkmeans`, `whclust`, `dn_wkmeans`, `dn_whclust` and `cut_tree`. Each names the tests and recorded R cases behind it, what must match R, and what was not recorded from R.

### Changed

- `wkmeans` and `dn_wkmeans` give up on a start after `MAX_DRAWS`, 1,000 random partitions counted across both of `WCluster`'s redraw loops, and raise a `ValueError` naming `k`. Before, a start near `n = 2k` could run for a very long time. A start found within the cap is R's start, so the cap changes no result it lets through ([compatibility.md](compatibility.md#wkmeans-whclust-and-the-dn_-functions)).
- `RDraws` is exported from `datanuggets`, so `seed=RDraws(n)` needs no private import ([compatibility.md](compatibility.md#rdraws)).
- The license is `GPL-2.0-only`. 0.1.0a2's metadata added `CC-BY-4.0` for test data the source archive no longer holds.
- The extras are `frame`, `progress`, `plot` and the new `tsne`, which installs scikit-learn for `tsne_2d` and `tsne_3d` ([extras](installation.md#extras)).

### Removed

- The `datanuggets.flow`, `datanuggets.analysis` and `datanuggets.plot.flow` modules, `datanuggets.exceptions.FlowError`, `datanuggets.colors.population_colors`, and `datanuggets.plot.gates` and `datanuggets.plot.peaks`.
- The `datanuggets` and `datanuggets-flow` commands, the `datanuggets.cli` module, `python -m datanuggets`, and the `cli` and `flow` extras.

## 0.1.0a2: not uploaded

This version was prepared but never uploaded to PyPI. Its library changes, listed here, are part of 0.1.0a3, and so is the [0.1.0a1 candidate](#010a1-test-candidate-not-uploaded) below. Its changes outside the library are not listed, since those parts are not in this package.

### Added

- `datanuggets.colors`, `datanuggets.filters` and `datanuggets.projection` ([helper modules](api.md#helper-modules)).
- `CITATION.cff`, so GitHub's "Cite this repository" gives the data nuggets article.

### Changed

- `create_dn` gives a row that is exactly as near to several centers to the lowest-numbered of them. Earlier development versions copied the order R's Rfast leaves tied centers in. The two agree whenever there are 16 or fewer centers. With more, a tied row can now join a different nugget than it does in R or did in those versions ([compatibility.md](compatibility.md#exact-distance-ties-in-create_dn)).
- The package metadata names Windows, macOS and Linux in place of "OS Independent" ([supported platforms](installation.md#supported-platforms)). The sorting helpers whose license could not be confirmed are gone, so the package can be published ([third-party notices](https://github.com/ron2k1/datanuggets/blob/main/THIRD_PARTY.md#sorting-helpers-removed)).
- The wheel's core-only smoke test also checks that scikit-learn isn't installed.

### Removed import paths

The package reorganization moved each datanugget function into a module of the same name in `datanuggets.nuggets` and left a module at the package root that only re-exported it. Each function had two module paths, and no guide imported from the root ones, so those five modules are gone. Import each function from `datanuggets`, as the README does:

| Removed | Use instead | Defined in |
|---|---|---|
| `datanuggets.create.create_dn` | `datanuggets.create_dn` | `datanuggets.nuggets.create` |
| `datanuggets.refine.refine_dn` | `datanuggets.refine_dn` | `datanuggets.nuggets.refine` |
| `datanuggets.centers.create_dn_centers` | `datanuggets.create_dn_centers` | `datanuggets.nuggets.centers` |
| `datanuggets.complexity.arithmetic_complexity` | `datanuggets.arithmetic_complexity` | `datanuggets.nuggets.complexity` |
| `datanuggets.means.get_dn_means` | `datanuggets.get_dn_means` | `datanuggets.nuggets.means` |

So `from datanuggets.create import create_dn` becomes `from datanuggets import create_dn`.

The private `datanuggets._draws` stays, for scripts that import `DrawRecord` and `ReplayDraws` from it, but a bare `import datanuggets` no longer makes it an attribute of the package. Import it by name, as `from datanuggets._draws import DrawRecord, ReplayDraws` does.

## 0.1.0a1: test candidate, not uploaded

This candidate was built and tested but never uploaded to PyPI. Its library is part of 0.1.0a3. Like 0.1.0a2, it was intended for research evaluation.

### Included

- Native Python implementations of 17 functions from R's `datanugget` 1.5.0 and `WCluster` 1.3.0 and of R's `cutree`, with no R runtime dependency.
- Nugget creation and refinement, weighted clustering, weighted PCA, prediction, and optional plotting and pandas support.
- Installation and analysis walkthroughs, a Mac setup without Git or uv, function documentation, and an example using the recorded `quakes` data. The source archive includes the guides, example, and recorded test fixtures.
- CI coverage for Python 3.12–3.15 on Linux, Windows, and macOS, plus installed-wheel and distribution-content checks.

### Before using it for research

The API may change before a stable release. Pin the package version and record preprocessing choices, seeds, and source revisions. `optimal.DN` is not implemented. Hierarchical clustering uses quadratic memory, and numerical results are not universally identical to R.

Read the [compatibility notes](compatibility.md) and [limitations](limitations.md) before interpreting results. Python 3.14 and 3.15 are alternative interpreters for this same package.
