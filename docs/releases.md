# Release notes

## Unreleased

### Added

- `datanuggets-flow` draws `local_maxima.png`, each file's density grid, local maxima and numbered peaks, and asks for the populations over it, before it draws `gates.png`.
- `clusters_tsne.png`, a t-SNE of the clustered nuggets, colored as `clusters_pca.png`. Its `n_components`, `init`, `learning_rate` and `max_iter` are pinned, so a change of scikit-learn's defaults can't move them. The `flow` extra now installs scikit-learn.
- `datanuggets-flow` runs in two stages as well as at once. `--gates-only` gates the files and saves the gates, figures and a `gates.json` record in a gate folder. `--from-gates` asks over that folder's figures and clusters without gating again, as often as needed, and writes `run.json`. For the same files, settings and populations, the two stages write the one-shot run's tables. [Staged runs](cli.md#staged-runs) describes both.
- New `datanuggets-flow` flags: `--t12`, `--t22`, `--n-row` and `--n-col` for the finder, `--dn-num1`, `--seed` and `--jobs` for `create_dn`, `--exclude-channels` to leave channels such as a viability dye out of the default set, and `--no-tsne`, `--tsne-perplexity` and `--tsne-seed` for the t-SNE figure.
- An asked run prints the `--pops` that repeats its choice.
- The pipeline's library in `datanuggets.flow.settings`, `.pipeline`, `.state` and `.figures`, with `FlowError` in `datanuggets.exceptions`, so a script can gate and cluster without the command ([API](api.md#the-flow-pipeline-library)). `datanuggets.flow.gate_peaks` gives the finder's masks and peaks from one density grid.

### Changed

- `datanuggets-flow` checks every setting before it reads a file, and `--cofactor` refuses a negative, NaN or infinite value.
- The finder's tunable defaults are in `datanuggets._defaults`, and the constants it keeps for parity with Cabrera's code are named in their modules ([methodology](methodology.md#the-finders-parameters)). Its results are unchanged.
- The wheel's core-only smoke test also checks that scikit-learn isn't installed.

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
