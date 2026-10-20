# datanuggets

A flow cytometry sample holds hundreds of thousands to millions of events, and a study holds many samples. Weighted hierarchical clustering and the other methods of the R package `WCluster` can't run on that many rows directly. Data nuggets (Beavers et al., cited below) reduce the data to a few thousand weighted points first. Each nugget has a center, a row count and a scale, and the analysis runs on the nuggets with their row counts as weights.

This package is a native Python port of CRAN's `datanugget` 1.5.0 and `WCluster` 1.3.0, so that work runs without R. Its tests check it against 544 cases recorded from R. It matches R bit for bit on them, except for the eigen solver's output and the differences listed in [docs/compatibility.md](docs/compatibility.md). It also includes Prof. Cabrera's flow cytometry gate finder, ported from his Python. A command line tool, `datanuggets-flow`, gates FCS files, asks which cell populations to keep, and clusters the kept cells.

Version `0.1.0.dev0` is in development and isn't on PyPI. `pip install datanugget`, without the s, installs an unrelated package.

## Installation

Python 3.12 or newer. Until the open pull requests merge into `main`, the plots and `datanuggets-flow` are only on the `feat/flow-pick` branch:

```sh
pip install "datanuggets[flow] @ git+https://github.com/ron2k1/datanuggets.git@feat/flow-pick"
```

The `flow` extra adds flowio, matplotlib and pandas. [docs/installation.md](docs/installation.md) covers the other extras, installing from a wheel, and an editable install for development.

## Quickstart

### Your own table of numbers

Any numeric table works: rows are observations and columns are variables. This clusters the rows of a CSV file into 5 groups through 500 nuggets:

```python
import pandas as pd
from datanuggets import create_dn, cut_tree, dn_whclust

df = pd.read_csv("my_data.csv").dropna()
x = df.select_dtypes("number").to_numpy()
x = (x - x.mean(axis=0)) / x.std(axis=0)  # put the columns on one scale

dn = create_dn(x, dn_num2=500)
tree = dn_whclust(dn)
nugget_cluster = cut_tree(tree, k=5)
df["cluster"] = nugget_cluster[dn.assignments]
print(df["cluster"].value_counts().sort_index())
```

`create_dn` measures Euclidean distance on the columns as given, so a column in large units would otherwise decide the nuggets alone. It rejects NaN and infinite values, hence the `dropna()`. `dn.assignments` gives each row its nugget, which is how the nugget clusters reach the rows. On the first 50,000 rows and 10 continuous columns of UCI Covertype this ran in 11 seconds on a laptop (i7-13650HX, Windows 11). `pd.read_csv` needs pandas, which the `flow` extra installs.

### Your own FCS files

Point `datanuggets-flow` at a folder of FCS files:

```sh
datanuggets-flow path/to/fcs_folder
```

It gates each file on FSC-A against SSC-A and shows every file's numbered populations in one figure, with their event counts and median FSC-A and SSC-A. Then it asks, file by file, which populations to keep. The kept cells are clustered on their fluorescence channels, `arcsinh(x / 6000)` by default, with nuggets per file and weighted Ward clustering of all the files' nuggets together, so a cluster means the same cells in every file. Tables and figures go to `datanuggets-out/`.

Files from other instruments may need these flags:

```sh
datanuggets-flow path/to/fcs_folder --fsc FSC-H --ssc SSC-H        # other scatter channel names
datanuggets-flow path/to/fcs_folder --channels "CD3-A,CD4-A,CD8-A"  # cluster only these channels
datanuggets-flow path/to/fcs_folder --cofactor 150 --k 10           # another arcsinh cofactor, 10 clusters
datanuggets-flow path/to/fcs_folder --pops "1,2" --no-show          # no questions: Pop 1 and 2 of every file
```

Channels are named as the file's `$PnN` keywords name them, such as `FJComp-FITC-A`, not by marker. A file without one of the named channels stops the run with a list of the channels it has. The gate finder's thresholds were tuned on Prof. Cabrera's study, so check `gates.png` before trusting the populations on other data, and adjust `--t11` and `--t21` if they look wrong. [docs/cli.md](docs/cli.md) documents every flag, the prompt, the outputs and the errors.

To try it without data of your own, [docs/quickstart.md](docs/quickstart.md) runs it on a public file from [Zenodo record 22079206](https://zenodo.org/records/22079206).

## Documentation

- [docs/installation.md](docs/installation.md): requirements, extras, wheel and editable installs.
- [docs/quickstart.md](docs/quickstart.md): the Python API, plots, the gate finder and a first `datanuggets-flow` run.
- [docs/cli.md](docs/cli.md): `datanuggets-flow` flags, the population prompt, inputs, outputs and errors.
- [docs/methodology.md](docs/methodology.md): the data nuggets method, the flow pipeline as implemented, how it relates to the papers, and how the port is checked against R.
- [docs/compatibility.md](docs/compatibility.md): the ported R functions, argument names and every documented difference from R.
- [docs/reproducibility.md](docs/reproducibility.md): environments, seeds, datasets, what R recorded, and the benchmarks with their results.
- [docs/development.md](docs/development.md): running the tests, lint, the build check and the history of the port.
- [docs/limitations.md](docs/limitations.md): known limitations and open problems.
- [docs/architecture.md](docs/architecture.md): where the code, tests and fixtures live.

## Citing the method

Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 2025. Published online in 2024.

DOI: `10.1080/10618600.2024.2341896`. arXiv: `2403.03099`.

## License and credit

GPL-2.0-only, the same license as the R packages `datanugget` and `WCluster`. The algorithms are the R authors' work.

The `datanugget` authors are Rituparna Dey (maintainer), Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi and Mariusz Lubomirski.

The `WCluster` authors are Rituparna Dey (maintainer), Yajie Duan, Javier Cabrera and Ge Cheng.

The flow gate finder is Prof. Javier Cabrera's, from his Python port of his R code.
