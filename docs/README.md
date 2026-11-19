# datanuggets documentation

These guides document the datanuggets Python library. The function reference follows R's help-page structure (description, usage, arguments, details, value and examples) and pairs each Python function with its R usage. Its signatures and behavior describe the Python implementation.

| Guide | What you will find |
| --- | --- |
| [Installation](installation.md) | The supported platforms (Windows x64, macOS 12+ on Apple silicon, Linux with glibc 2.27+), the pinned `0.1.0a3` from PyPI, `main` from GitHub, the extras, a local checkout or built wheel, and a Mac setup without Git or uv |
| [Getting started](quickstart.md) | One CSV taken through `create_dn`, `refine_dn`, Ward or k-means clustering, a plot, weighted PCA and prediction of new rows, with the R function each step ports |
| [Function reference](api.md) | Arguments, return values and examples for the 18 core functions, their R usage, the result objects, the plotting functions, and the modules for cluster colours, row filters and the PCA and t-SNE projections |
| [Limitations](limitations.md) | Memory at a given nugget count, where results can differ from R, planned API changes and open problems |
| [R compatibility](compatibility.md) | The [R-to-Python function map](compatibility.md#ported-functions), `RDraws` for R's random draws, and each function's differences from R in arguments, defaults, results and plots |
| [Methodology](methodology.md) | The steps of `create_dn`, `refine_dn` and the weighted analyses, where the defaults come from, and what was tested against R |
| [Reproducibility](reproducibility.md) | The R and Python environments, seeds, datasets, what R recorded for the fixtures, and the experiments behind the documented numbers |
| [Math reference](https://github.com/ron2k1/datanuggets/blob/main/docs/math/README.md) | What each ported function computes, as equations, with the floating-point steps the port replays to match R and the tests that hold each statement. A draft that the authors of the R packages have not reviewed |
| [Development](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) | Setup, tests, lint, build checks, the parity tables and the fixtures |
| [Architecture](https://github.com/ron2k1/datanuggets/blob/main/docs/ARCHITECTURE.md) | What each folder of the repository holds, which way the package's imports run, the two tests that keep its boundaries, and where new work goes |
| [Release notes](releases.md) | What `0.1.0a3` contains and leaves out, and the `0.1.0a2` and `0.1.0a1` versions that were not uploaded |
| [Releasing](https://github.com/ron2k1/datanuggets/blob/main/docs/releasing.md) | The release workflows, one-time setup, testing a release candidate and publishing the tested commit |

For a first run without preparing a CSV, follow the [checkout example](installation.md#from-a-local-checkout).
