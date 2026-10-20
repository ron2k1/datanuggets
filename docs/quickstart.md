# Quickstart

These examples run in a clone after `uv sync`, which installs the package with pandas, matplotlib and flowio from the `dev` group. Installed with pip, the table needs the `frame` extra, the plots the `plot` extra and the command line the `flow` extra; [installation.md](installation.md#extras) lists them. The Python blocks continue one another, so run them in one session started from the clone's root, for example with `uv run python`.

Every output on this page comes from a run on the development laptop (i7-13650HX, Windows 11, Python 3.14.3, numpy 2.5.3, scipy 1.18.1, numba 0.68.0) on 2026-10-20.

## Your own data

### A table of numbers

Rows are observations and columns are numeric variables. This reads a CSV file, clusters its rows into 5 groups through 500 nuggets, and writes each row's cluster back out:

```python
import pandas as pd
from datanuggets import create_dn, cut_tree, dn_whclust

df = pd.read_csv("my_data.csv").dropna()
x = df.select_dtypes("number").to_numpy()
x = (x - x.mean(axis=0)) / x.std(axis=0)  # put the columns on one scale

dn = create_dn(x, dn_num2=500)
nugget_cluster = cut_tree(dn_whclust(dn), k=5)
df["cluster"] = nugget_cluster[dn.assignments]
df.to_csv("my_data_clustered.csv", index=False)
```

Three things decide whether this works on your data:

- `create_dn` rejects NaN and infinite values, so drop or fill them first.
- Distances are Euclidean on the columns as given, so a column in large units would decide the nuggets alone. Standardizing is one choice; a transform that suits the measurement, such as arcsinh for fluorescence, is another.
- `dn_num1`, 10,000 by default, is the number of first-stage centers, and the second stage builds a `dn_num1` by `dn_num1` distance matrix, 800 MB at the default. Lower it if memory is short. With fewer rows than `dn_num1`, the first stage is skipped and `create_dn` warns `DN.num is greater than or equal to the number of rows. Returning original data`, as R's `create.DN` does, then still makes `dn_num2` nuggets.

`dn.assignments` holds each row's nugget, which is how the nugget clusters reach the rows. On the first 50,000 rows and 10 continuous columns of UCI Covertype this took 11 seconds. [methodology.md](methodology.md) explains the two stages and what the weights and scales mean.

### A folder of FCS files

`datanuggets-flow` takes the files, or folders of them:

```sh
datanuggets-flow path/to/fcs_folder
```

It gates each file on FSC-A against SSC-A, prints and draws each file's numbered populations, and asks file by file which ones to keep. Before the first run on files from a new instrument, list their channels. `--fsc`, `--ssc` and `--channels` take the `$PnN` names in the first column; the second is the `$PnS` marker name, which the outputs add to the column labels:

```python
import flowio

fd = flowio.FlowData("path/to/fcs_folder/sample1.fcs")
for k in range(1, fd.channel_count + 1):
    print(fd.text[f"p{k}n"], fd.text.get(f"p{k}s", ""))
```

On `202.fcs` from the public record below, the first lines are `FSC-A`, `FSC-H`, `SSC-A` and `SSC-B-A` with no marker, and further down come lines such as `FJComp-FITC-A CD57`. By default every channel ending in `-A` that doesn't start with FSC or SSC is clustered. On `202.fcs` that is 32 channels, the viability dye `FJComp-LIVE DEAD Blue-A` among them, so check the list and name the channels you want with `--channels` when it holds ones you don't. Then run with the names your files use:

```sh
datanuggets-flow path/to/fcs_folder --fsc FSC-H --ssc SSC-H --channels "FJComp-FITC-A,FJComp-PE-Cy7-A"
```

The gate finder's thresholds were set on Prof. Cabrera's data. Look at `gates.png` before trusting the populations on other data, and change `--t11` and `--t21` if the first population's area looks wrong. [cli.md](cli.md) documents every flag. The rest of this page runs on data anyone can download.

## Data nuggets from a matrix

```python
import numpy as np
from datanuggets import create_dn

rng = np.random.default_rng(0)
x = rng.normal(size=(100_000, 3))

dn = create_dn(x, dn_num1=2000, dn_num2=200)
print(dn.n_nuggets)  # 200
print(dn.weights.sum())  # 100000
print(dn.to_frame().head())
```

Output of `to_frame().head()`:

```text
   Data Nugget   Center1   Center2   Center3  Weight     Scale
0            0 -3.230698  2.430919 -1.423236       1  0.000331
1            1  3.265895  0.927756  1.551690       6  0.041483
2            2  4.144727 -1.874403 -0.815790       1  0.000331
3            3  1.614533  3.945550  1.704140       1  0.000331
4            4  3.074095 -1.813385  0.944272       3  0.033129
```

Those weight-1 nuggets are expected. Closest-pair deletion keeps points far out in the tails as centers. They share the scale 0.000331 because a nugget with fewer than 2 rows gets the smallest scale among nuggets with 2 or more rows, divided by 100. R does the same. The call took 3.0 seconds on the development laptop with the numba cache warm. The first call in a new environment also compiles, as [installation.md](installation.md#the-first-call-compiles) explains.

## Weighted statistics on the nuggets

The `WCluster` functions take the nuggets' centers as the data and their weights as row weights. Continuing the example, this runs weighted PCA and splits the nuggets by the sign of their first coordinate:

```python
from datanuggets import cluster_predict, wmean, wpca, wwcss

pca = wpca(dn.centers, wrow=dn.weights)
print(pca.sdev)

cl = (dn.centers[:, 0] > 0).astype(int)
print(wmean(dn.centers, cl, dn.weights))
print(wwcss(dn.centers, cl, dn.weights))
print(cluster_predict(dn.centers, cl, x[:5], w=dn.weights))
```

Output:

```text
[0.95152184 0.94524503 0.94031221]
[[-0.7682159  -0.02598546  0.21580461]
 [ 0.71713231  0.01851018 -0.19804108]]
208903.8865164328
[0 1 1 0 0]
```

`cluster_predict` returns each new row's 0-based position among the sorted labels. The labels here are 0 and 1, so the positions equal the labels.

## Clustering the nuggets

`dn_wkmeans` and `dn_whclust` cluster the nuggets, weighted by their row counts. With `seed=RDraws(291102)`, `dn_wkmeans` draws its starts as R's `DN.Wkmeans` does with seed 291102, so the same nuggets give R's clusters. `RDraws` isn't exported from `datanuggets` yet, so it comes from its private module; [compatibility.md](compatibility.md#rdraws) describes it.

```python
from datanuggets import dn_whclust, dn_wkmeans
from datanuggets._random.r_draws import RDraws

km = dn_wkmeans(dn, 3, seed=RDraws(291102))
print(km.wwcss, km.total_wwcss)
print(np.bincount(km.row_labels))

hc = dn_whclust(dn)
print(hc.merge[-1], hc.height[-1])
```

Output:

```text
[52989.7953179  54819.45581478 57960.64673812] 165769.8978708033
[31348 32229 36423]
[397 396] 268304.21872574475
```

`row_labels` puts each of the 100,000 rows in its nugget's cluster, so the counts add up to 100,000. `merge` is in scipy's numbering, so the last merge joins clusters 396 and 397, the ones made at 0-based steps 196 and 197, and `height` ends at the total WWCSS of all 200 nuggets as one cluster. `hc.to_linkage()` gives the linkage that `scipy.cluster.hierarchy.dendrogram` takes.

## Plots

`datanuggets.plot` draws with matplotlib. `import datanuggets` doesn't import it, so the rest of the package runs without matplotlib. Continuing the example, this cuts the nuggets' tree into three clusters and draws four views of them in the paper style, saved as a two-column PDF:

```python
import matplotlib.pyplot as plt
from datanuggets import cut_tree, dn_whclust, dn_wpca, wmean
from datanuggets import plot

hc = dn_whclust(dn)
labels = cut_tree(hc, k=3)
print(np.bincount(labels))
pca = dn_wpca(dn)
centers = wmean(dn.centers, labels, dn.weights)

with plot.style():
    fig, axs = plt.subplots(2, 2)
    plot.dendrogram(hc, labels=False, ax=axs[0, 0])
    plot.nuggets(dn, ax=axs[0, 1])
    plot.clusters(dn.centers, labels, w=dn.weights, centers=centers, ax=axs[1, 0])
    plot.scores(pca, w=dn.weights, labels=labels, rows=x, ax=axs[1, 1])
    plot.save(fig, "nuggets.pdf", width="double", height=5.5)
```

Output:

```text
[107  58  35]
```

That's how many of the 200 nuggets each cluster holds. Every plot function takes `ax=`, draws on `plt.gca()` without it, and returns the Axes it drew on. [compatibility.md](compatibility.md#plots) gives each plot function's details and how it relates to R's drawing.

## Flow gates

`datanuggets.flow.lymphmacro_finder` is Prof. Cabrera's gate finder for the lymphocyte and macrophage populations on FSC-A against SSC-A. It needs only numpy and scipy. The fixtures in `tests/fixtures/flow/` hold the FSC-A and SSC-A of three files of a public PBMC record, thinned to 50,000 events each, so this runs in a clone with nothing to download:

```python
import numpy as np
from datanuggets.flow import lymphmacro_finder

fixture = np.load("tests/fixtures/flow/202.npz")
fsc, ssc = fixture["fsc"], fixture["ssc"]

masks = lymphmacro_finder(fsc, ssc)
print(masks.shape)
print(masks.sum(axis=0))
print((masks.sum(axis=1) > 1).sum(), (masks.sum(axis=1) == 0).sum())
```

Output:

```text
(50000, 6)
[38318 33642 11696 11044  8166  6016]
41306 4154
```

`masks` has one row per event and one column per population, in the reference finder's order. The first population's area takes `t11` and `t21`, the others `t12` and `t22`, with the reference defaults. An event can be in more than one population, as in the reference code: here 41,306 of the 50,000 events are, and 4,154 are in none. These masks equal the fixture's `fixed_masks`, the reference gate run on the cells of the disc centred on the peak, except for one event on the hull's edge that ours keeps and the reference gate doesn't. [methodology.md](methodology.md#where-the-finder-departs-from-cabreras-code) explains both departures. `plot.gates(fsc, ssc, masks, title=..., ax=...)` draws the reference code's figure from the masks.

### From an FCS file

flowio, from the `flow` extra, reads the two channels from an FCS file the way FlowKit's raw events hold them. `202.fcs` is one of the 44 files of [Zenodo record 22079206](https://zenodo.org/records/22079206), licensed CC BY 4.0, and the source of the `202.npz` fixture:

```python
import flowio
from datanuggets.flow import lymphmacro_finder

fd = flowio.FlowData("202.fcs")
events = fd.as_array(preprocess=True)
names = [fd.text[f"p{k}n"] for k in range(1, fd.channel_count + 1)]
fsc = events[:, names.index("FSC-A")]
ssc = events[:, names.index("SSC-A")]

masks = lymphmacro_finder(fsc, ssc)
print(masks.shape)
print(masks.sum(axis=0))
print((masks.sum(axis=1) > 1).sum(), (masks.sum(axis=1) == 0).sum())
```

Output:

```text
(282270, 3)
[214085  66149  37742]
57797 29065
```

The whole file has three populations, where the thinned fixture has six. [reproducibility.md](reproducibility.md) lists the record and the fixtures.

## A first datanuggets-flow run

`datanuggets-flow` gates FCS files, asks which populations hold the cells you want, and clusters those cells' fluorescence channels. It reads FCS files only. The fixtures above are `.npz` arrays of two channels, so pointed at `tests/fixtures/flow` it stops with `no .fcs files in tests/fixtures/flow`. This run uses `223.fcs`, a 9 MB file of the same record:

```sh
mkdir fcs
curl -L -o fcs/223.fcs "https://zenodo.org/records/22079206/files/223.fcs?download=1"
uv run datanuggets-flow fcs --pops all --no-show --out flow-out
```

`--pops all` keeps every population of every file instead of asking, and `--no-show` saves the figures without opening them. Output, with the absolute path on the last line shortened:

```text
File 1, 223.fcs: 57,943 events
  Pop 1:    51,967 events (89.7%), median FSC-A 1,010,996, SSC-A   569,303
  Pop 2:    14,314 events (24.7%), median FSC-A 1,146,372, SSC-A   628,614
  Pop 3:     2,481 events ( 4.3%), median FSC-A 1,459,248, SSC-A 1,488,002
  Pop 4:     2,687 events ( 4.6%), median FSC-A 1,282,087, SSC-A   744,224
  Pop 5:       113 events ( 0.2%), median FSC-A 1,393,077, SSC-A   655,550
  Pop 6:       148 events ( 0.3%), median FSC-A 1,462,489, SSC-A   824,276
  Pop 7:       193 events ( 0.3%), median FSC-A 1,615,336, SSC-A 1,135,636

The populations are drawn in flow-out/gates.png

 file_number    file   populations  events  kept   percent
           1 223.fcs 1,2,3,4,5,6,7   57943 54426 93.930242
File 1, 223.fcs: 54,426 cells, 1,000 nuggets

Percent of each file's kept cells in each cluster:
(columns are file numbers; selection.csv names the files)
file_number     1
cluster          
1            22.3
2            23.3
3            16.3
4             3.5
5             5.2
6             8.4
7            11.8
8             9.3

Outputs are in .../flow-out
```

The populations are the ones Cabrera's script draws, since `--disc his` is the default. The 54,426 kept events went into 1,000 data nuggets, and weighted Ward clustering of the nuggets was cut at the default 8 clusters. `flow-out` now holds:

```text
cluster_centers.csv  cluster_centers.png  cluster_summary.csv  clusters_pca.png
dendrogram.png       events_1.csv         gates.png            nuggets_1.csv
selection.csv
```

Without `--pops`, the command opens the gate figure and asks file by file. [cli.md](cli.md) documents every flag, the prompt, the processing steps and each output's columns.
