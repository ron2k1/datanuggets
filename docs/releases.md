# Release notes

## 0.1.0a1 — first public alpha candidate

This version is intended for research evaluation. Preparing these files does not publish the package; check the [PyPI project page](https://pypi.org/project/datanuggets/) for availability.

**Publication hold:** the GNU sorting helpers need a compatible source-provenance record or replacement before this candidate can be redistributed publicly. See [third-party notices](../THIRD_PARTY.md#publication-hold-gnu-sorting-helpers). Both TestPyPI and PyPI uploads are blocked in the release workflow; check-only runs remain available.

### Included

- Native Python implementations of 18 functions from R's `datanugget` 1.5.0 and `WCluster` 1.3.0, with no R runtime dependency.
- Nugget creation and refinement, weighted clustering, weighted PCA, prediction, and optional plotting and pandas support.
- An experimental FCS workflow with new output folders under `outputs/`, protection against overwriting existing paths, and markers for incomplete runs.
- Installation and analysis walkthroughs, function documentation, and an example using the recorded `quakes` data. The source archive includes the guides, example, and recorded test fixtures.
- CI coverage for Python 3.12–3.15 on Linux, Windows, and macOS, plus installed-wheel and distribution-content checks.

### Before using it in a study

The API may change before a stable release. Pin the package version and record preprocessing choices, seeds, and source revisions. `optimal.DN` is not implemented. Hierarchical clustering uses quadratic memory, and numerical results are not universally identical to R. The flow command does not apply compensation or establish biological cell identities.

Read the [compatibility notes](compatibility.md), [limitations](limitations.md), and [flow guide](cli.md) before interpreting results. Python 3.14 and 3.15 are alternative interpreters for this same package.
