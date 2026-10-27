# datanuggets documentation

To analyze your own dataset from Terminal, start with installation and the CSV guide. For Python code, use the walkthrough and function reference. The reference follows the familiar R help-page structure: description, usage, arguments, details, return value, and examples. Its signatures and behavior describe the Python implementation.

| Guide | What you will find |
| --- | --- |
| [Installation](installation.md) | Install a supplied wheel, the alpha, or `main`; MacBook setup without Git or uv |
| [CSV CLI](cli.md#analyze-a-csv) | Select columns, transform data, create nuggets, and export weighted analyses |
| [Getting started](quickstart.md) | Prepare data, create and refine nuggets, cluster, plot, and predict |
| [Function reference](api.md) | Python functions, result types, and examples |
| [Flow CLI](cli.md#fcs-pipeline) | The separate FCS pipeline: population selection, flags, and output files |
| [Limitations](limitations.md) | Memory, statistical assumptions, missing features, and alpha status |
| [R compatibility](compatibility.md) | The [R-to-Python function map](compatibility.md#ported-functions), recorded comparisons, and intentional differences from R |
| [Methodology](methodology.md) | How the implemented analysis fits together |
| [Reproducibility](reproducibility.md) | Recorded datasets, fixtures, and validation procedures |
| [Development](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) | Run tests, lint, and distribution checks |
| [Release notes](releases.md) | What is included in the first alpha |
| [Releasing](https://github.com/ron2k1/datanuggets/blob/main/docs/releasing.md) | Check, test, and publish a release |

For a first run without preparing a CSV, follow the [checkout example](installation.md#from-a-local-checkout). Python 3.14 and 3.15 run the same library; ordinary use does not require both.
