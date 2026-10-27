# Release notes

## Unreleased

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

## 0.1.0a1 — first public alpha candidate

This version is intended for research evaluation. Preparing these files does not publish the package; check the [PyPI project page](https://pypi.org/project/datanuggets/) for availability.

**Publication hold:** the GNU sorting helpers need a compatible source-provenance record or replacement before this candidate can be redistributed publicly. See [third-party notices](../THIRD_PARTY.md#publication-hold-gnu-sorting-helpers). Both TestPyPI and PyPI uploads are blocked in the release workflow; check-only runs remain available.

### Included

- Native Python implementations of 18 functions from R's `datanugget` 1.5.0 and `WCluster` 1.3.0, with no R runtime dependency.
- Nugget creation and refinement, weighted clustering, weighted PCA, prediction, and optional plotting and pandas support.
- A `datanuggets analyze` command for your own CSV files: explicit feature selection, optional transformations and refinement, weighted analyses, row assignments, summaries, and plots in a fresh output folder. Install the `cli` extra; `python -m datanuggets` is also supported.
- An experimental FCS workflow with new output folders under `outputs/`, protection against overwriting existing paths, and markers for incomplete runs.
- Installation and analysis walkthroughs, MacBook wheel setup without Git or uv, function documentation, and an example using the recorded `quakes` data. The source archive includes the guides, example, and recorded test fixtures.
- CI coverage for Python 3.12–3.15 on Linux, Windows, and macOS, plus installed-wheel and distribution-content checks.

### Before using it in a study

The API may change before a stable release. Pin the package version and record preprocessing choices, seeds, and source revisions. `optimal.DN` is not implemented. Hierarchical clustering uses quadratic memory, and numerical results are not universally identical to R. The flow command does not apply compensation or establish biological cell identities.

The CSV command does not infer appropriate transformations or optimal nugget and cluster counts. It rejects missing or nonfinite selected values instead of dropping rows. If you received an earlier `0.1.0a1` wheel, reinstall the newly supplied build to get this command; the version number alone does not distinguish unpublished builds.

Read the [CSV guide](cli.md#analyze-a-csv), [compatibility notes](compatibility.md), and [limitations](limitations.md) before interpreting results; use the [FCS guide](cli.md#fcs-pipeline) for that separate pipeline. Python 3.14 and 3.15 are alternative interpreters for this same package.
