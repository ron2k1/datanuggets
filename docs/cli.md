# The datanuggets-flow command

`datanuggets-flow` gates FCS files, lets you choose candidate populations, and clusters their events. It creates nuggets for each file, then fits one weighted Ward tree to the pooled nuggets. Cluster numbers therefore refer to the same fitted clustering across files; they are not independently validated cell-type labels. [methodology.md](methodology.md#the-datanuggets-flow-pipeline) explains the method and its design choices.

The command needs the `flow` extra, which adds flowio, matplotlib and pandas ([installation.md](installation.md#extras)). In a clone, `uv sync` installs them and `uv run datanuggets-flow` runs it. The [quickstart](quickstart.md#a-first-datanuggets-flow-run) walks through a first run.

> **Experimental workflow.** Each run creates a fresh folder under `outputs/` by default. Existing output paths are refused, and unfinished runs retain an `INCOMPLETE.txt` marker. Choose at least two distinct fluorescence channels and enough retained cells to make at least three occupied nuggets.

Inspect the gates and confirm your compensation and preprocessing before interpreting results. The command does not apply compensation. With many input files, also check the **total** nugget count: 10,000 pooled nuggets require about 800 MB for the hierarchy's matrix alone.

## Synopsis

```text
usage: datanuggets-flow [-h] [--pops POPS] [--out OUT] [--k K]
                        [--cofactor COFACTOR] [--dn-num2 DN_NUM2]
                        [--channels CHANNELS] [--fsc FSC] [--ssc SSC]
                        [--disc {his,peak}] [--t11 T11] [--t21 T21]
                        [--no-show]
                        paths [paths ...]
```

## Inputs

Each path is an FCS file or a folder of them. A folder gives its files whose suffix is `.fcs` in any case, so `.FCS` counts too, sorted by name. Names that start with a dot are skipped, such as the `._` copies macOS writes on some drives. Subfolders aren't searched. A file named directly is read whatever its suffix.

Files and folders can be mixed. The files are numbered from 1 in the order of the arguments, with each folder's files in name order. Name order is string order, so `10.fcs` comes before `9.fcs`. That number is what per-file thresholds, `--pops` entries and the `N` in `nuggets_N.csv` and `events_N.csv` follow. The run prints `File N, name` for every file, and `selection.csv` records the numbering.

flowio reads each file's events with `as_array(preprocess=True)`, as the file stores them. Channels are found by their `$PnN` names. Every file must have the `--fsc` and `--ssc` channels. The channels to cluster are taken from the first file, and every file must have them too.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--pops POPS` | ask at the prompt | Populations to keep, in the prompt's grammar: one entry for every file, or one per file separated by `;` |
| `--out OUT` | `outputs/flow-<timestamp>` | New output folder, relative to the current directory. Existing paths are refused, including empty folders |
| `--k K` | `8` | Clusters cut from the tree, 1 or more. Lowered to the number of nuggets when there are fewer |
| `--cofactor COFACTOR` | `6000` | Fluorescence is clustered as `arcsinh(x / cofactor)`. `0` keeps it linear |
| `--dn-num2 DN_NUM2` | `1000` | Nuggets per file, 2 to 9999. Lowered to a file's kept events when there are fewer. Clustering requires at least 3 occupied nuggets across all retained files |
| `--channels CHANNELS` | every `-A` channel but FSC and SSC | Comma-separated `$PnN` names. Select at least two for the current plotting step. The default takes names in the first file ending in `-A` that do not start with `FSC` or `SSC` |
| `--fsc FSC` | `FSC-A` | The channel gated as FSC |
| `--ssc SSC` | `SSC-A` | The channel gated as SSC |
| `--disc {his,peak}` | `his` | The area rule the gates use, described below |
| `--t11 T11` | `22` | The first peak's distance threshold, one value or one per file |
| `--t21 T21` | `0.25` | The first peak's density threshold, one value or one per file |
| `--no-show` | off | Save the figures without opening them, on matplotlib's Agg backend |

On the files of the Zenodo record the default channels are the 32 unmixed `FJComp-...-A` channels. `SSC-B-A` starts with `SSC`, so it's left out. A channel name with a space needs quotes, as in `--channels "FJComp-BV510-A,FJComp-Spark Blue-A"`.

### Per-file thresholds

`--t11` and `--t21` take one number for every file, or a comma-separated list with one number per file in file order. Any other count stops the run. The peaks after the first use the reference finder's `t12` of 20 and `t22` of 0.5, and the finder's 100 by 100 grid and the 11 by 11 window of its focal maxima are fixed too. None of these has a flag.

Cabrera's `cfc.py` sets the first peak's thresholds file by file. For a folder of four files:

```sh
datanuggets-flow path/to/fcs_folder --t11 22,20,18,18 --t21 0.25,0.5,0.5,0.5 --out outputs/threshold-run
```

`cfc.py` pairs these values with the samples in the order FlowKit returns them, by their `$FIL` keyword, which may not be the order of the workbook it reads ([methodology.md](methodology.md#against-cabreras-script)). Here they follow file order, so check the `File N` lines against the values you meant.

### --disc

- `his`, the default, gates with `datanuggets.flow.mirrored_area`, a copy of `cfc.py`'s own `find_area`, whose distance disc sits on the mirrored cell. The populations and their numbers are the ones `cfc.py`'s figures show.
- `peak` gates with `datanuggets.flow.find_area`, which centres the disc on the peak, as `lymphmacro_finder` does.

On `202.fcs`, `his` gives populations of 225,770, 58,245 and 27,590 events and `peak` gives 214,085, 66,149 and 37,742. Which rule is right is open problem 10 in [limitations.md](limitations.md#open-problems). [methodology.md](methodology.md#the-disc---disc-his-and---disc-peak) explains the difference.

## Choosing populations

### At the prompt

Without `--pops`, the run gates every file, prints each file's populations and asks. Each population line gives its event count, its share of the file's events, and its median on the `--fsc` and `--ssc` channels, FSC-A and SSC-A by default. `local_maxima.png` draws each file in its own panel: the density grid, every local maximum, and each kept peak labelled with its Pop number, so you choose from the peaks. `gates.png`, the populations' full gates, is drawn only after the choice. From a run on three files of the Zenodo record, after the first answer:

```text
Which populations hold the cells you want? Each file's Pop numbers label the peaks in its own panel of the local maxima figure, and the same number can be a different peak in another file. Type numbers like 1 or 1,2 or 1-3, "all", or "none" to leave the file out; Enter repeats the previous file's choice, Ctrl+C stops. Click this terminal before typing.

File 2, 202.fcs:
  Pop 1:   225,770 events (80.0%), median FSC-A 1,015,754, SSC-A   569,799
  Pop 2:    58,245 events (20.6%), median FSC-A 1,171,461, SSC-A   626,855
  Pop 3:    27,590 events ( 9.8%), median FSC-A 1,404,158, SSC-A 1,355,836
Keep which, Pop 1 to 3 [1,2]:
```

The brackets hold the previous file's answer. Each answer is one of:

| Answer | Keeps |
| --- | --- |
| `1` | Pop 1 |
| `1,2` or `1 2` | Pop 1 and Pop 2 |
| `1-3` | Pop 1 to Pop 3. `3-1` means the same |
| `1-2,5` | Pop 1, Pop 2 and Pop 5 |
| `all` | every population of the file |
| `none` | nothing, so the file is left out |
| Enter | the previous file's answer |

Case doesn't matter, and a number given twice counts once. An answer that names a population the file doesn't have, or that isn't numbers, ranges, `all` or `none`, prints the reason and asks again. So does Enter at the first file, which has no previous answer. A file in which the finder found no population is left out without a question. Ctrl+C or the end of input stops the run with `cancelled; nothing was clustered`.

The prompt needs a terminal. When standard input is not one, such as a pipe, the run stops after gating with `no terminal to ask in; pass --pops, e.g. --pops "1,2"`. On Windows, input redirected from `NUL` counts as a terminal, so the run reaches the first question and stops with `cancelled; nothing was clustered` instead.

Without `--no-show`, the gate figure opens before the first question and stays responsive while the prompt waits, so it can be zoomed and panned. Click the terminal before typing. On Windows the wait ends at the first key press.

### With --pops

`--pops` takes the same answers, for scripted runs. One entry applies to every file, and one entry per file, separated by `;`, applies file by file. Quote it, since `;` ends a command in the shell:

```sh
datanuggets-flow a.fcs b.fcs --pops "1,2;1" --out outputs/selection-run
```

Enter has no meaning here. A file in which the finder found no population takes `all` or `none`. An entry that doesn't parse or names a missing population stops the run, as the failure table below shows. `--pops ""` counts as no `--pops`, so the run asks.

An event in two kept populations counts once: the kept events are the union of the chosen populations.

## What a run does

1. Checks that matplotlib and pandas import, checks `--k` and `--dn-num2`, finds the files, reads the thresholds, and checks the number of `--pops` entries. Creates a new output folder and an `INCOMPLETE.txt` marker; an existing output path stops the run without changing its contents.
2. Gates each file: reads its events, runs the finder on `--fsc` and `--ssc` with the file's `--t11` and `--t21` and the `--disc` rule, prints the population lines and the number of local maxima and kept peaks, and draws the file's panel of `local_maxima.png`.
3. Checks the clustering channels, requiring at least two distinct channels present in every file, then chooses each file's populations at the prompt or from `--pops`, and draws `gates.png`.
4. For each file, keeps the events in any chosen population. A file with fewer than 2 kept events is left out of the clustering but stays in `selection.csv`. The file is read again, so only one file's events are in memory at a time, and the kept events' clustered channels are transformed as `arcsinh(x / cofactor)`, or left as they are with `--cofactor 0`. Writes `selection.csv`.
5. Makes data nuggets of each file's kept events separately with `create_dn`, `dn_num2` set to `--dn-num2` or to the number of kept events if that is smaller, every other argument at its default and `n_jobs=-1`, which uses every core. When a file has no more kept events than `--dn-num2`, `create_dn` warns `DN.num is greater than or equal to the number of rows. Returning original data`, and each event becomes its own nugget.
6. Pools every file's nugget centers and weights in file order, drops nuggets with weight 0, checks that at least three occupied nuggets remain, builds one weighted Ward tree with `whclust` weighted by each nugget's event count, and cuts it at `--k` clusters with `cut_tree`.
7. Writes the outputs below and prints each file's share of kept cells in each cluster. Removes `INCOMPLETE.txt` when all tables and figures are saved. Without `--no-show`, it then opens the figures and returns when they are closed.

`create_dn` runs with its fixed default seed. Two runs on the same three files with the same flags, on the development laptop, wrote identical CSV files.

## Outputs

Figures and tables share one run folder: `outputs/flow-<timestamp>/` by default, or a new folder chosen with `--out`. Paths are relative to the current directory. Run from the repository root to keep results inside its ignored `outputs/` directory.

| File | Contents |
| --- | --- |
| `local_maxima.png` | One panel per file, two to a row, titled `File N: name`: the finder's 100 × 100 density grid, every local maximum as an open circle and each kept peak as a cross labelled with the Pop number it seeds. Written before the prompt; populations are chosen from it. 110 dpi |
| `gates.png` | One panel per file, laid out as `local_maxima.png`: every event in gray and each population in color. Written after the populations are chosen. 110 dpi |
| `selection.csv` | One row per file: what was kept |
| `nuggets_N.csv` | One row per nugget of file N |
| `events_N.csv` | One row per kept event of file N |
| `cluster_summary.csv` | One row per clustered file and cluster |
| `cluster_centers.csv` | One row per cluster: its center |
| `dendrogram.png` | The Ward tree of the pooled nuggets, leaves at 0, titled with the nugget count. 150 dpi |
| `clusters_pca.png` | The nuggets' weighted PCA scores, colored by cluster and sized by weight. 150 dpi |
| `cluster_centers.png` | `cluster_centers.csv` as a heat map, one column per cluster and one row per channel, in the transformed units. 150 dpi |

The headers from the three-file run, with the 32 channel columns between the first and the last cut:

```text
selection.csv        file_number,file,populations,events,kept,percent
nuggets_1.csv        nugget0,CD38 (FJComp-APC-Cy5.5-A),...,CD161 (FJComp-eFluor 450-A),weight,scale,cluster
events_1.csv         event0,nugget0,cluster
cluster_summary.csv  file_number,file,cluster,cells,percent
cluster_centers.csv  cluster,CD38 (FJComp-APC-Cy5.5-A),...,CD161 (FJComp-eFluor 450-A)
```

The columns:

- `file_number` is the file's number and `file` its name. In `selection.csv`, `populations` is the kept populations, such as `1,2`, or `none`, `events` the file's event count, `kept` the kept events, and `percent` the kept share of the events.
- Channel columns are named `$PnS ($PnN)`, such as `CD3 (FJComp-BV510-A)`, or by `$PnN` alone when the channel has no `$PnS`. Their values are in the transformed units.
- In `nuggets_N.csv`, `nugget0` is the nugget's row, numbered from 0, `weight` its event count and `scale` its `create_dn` scale. `cluster` is numbered from 1, in the order the clusters first appear among the pooled nuggets. A nugget with weight 0 has cluster 0.
- In `events_N.csv`, `event0` is the event's row in its FCS file, numbered from 0, so the rows of events that weren't kept are missing. `nugget0` is its nugget's row in `nuggets_N.csv` and `cluster` its nugget's cluster.
- In `cluster_summary.csv`, `cells` is the file's kept events in the cluster and `percent` their share of the file's kept events. Every cluster has a row for every clustered file, with 0 cells where it has none. The table the run prints is this file's `percent` rounded to one decimal, one column per file number.
- `cluster_centers.csv` holds each cluster's weighted mean of its nuggets' centers, with the nugget weights as weights.

A file left out of the clustering has no `nuggets_N.csv` or `events_N.csv`, so the numbers can skip.

### Output safety

**Existing output paths are never reused.** The command refuses a file, directory, or symbolic link at the requested path, including an empty directory. This keeps a new run from deleting or overwriting earlier results. Let the default choose a fresh run folder, or give `--out` an unused name.

If a run fails or is cancelled after creating its folder, partial figures and tables stay there with `INCOMPLETE.txt`. The error also prints their location. Rerun in a new folder; the command does not resume unfinished runs. The marker is removed only after every result file has been saved.

## Spillover

Events are clustered as stored, uncompensated. When a file's `$SPILLOVER` or `$SPILL` keyword holds a matrix with any entry more than 0.001 from the identity, the run prints `warning: NAME has a compensation matrix, but its events are clustered as stored, uncompensated` and goes on. It prints each time the file is read, so twice for a file whose events are clustered. Spectral files are unmixed already. The four files of the Zenodo record tried here, `154`, `199`, `202` and `223`, carry no spillover keyword, so this warning hasn't been seen on real data.

## Failure conditions

The checks run in the order of the steps above, so the first condition a run meets is the one it reports. These messages go to standard error and the exit status is 1:

| Message | Cause |
| --- | --- |
| `PATH: no such file or folder` | A path that doesn't exist |
| `no .fcs files in PATHS` | The paths hold no FCS file, such as `tests/fixtures/flow`, whose fixtures are `.npz` arrays |
| `--k is K; it takes 1 or more clusters` | `--k` below 1 |
| `--dn-num2 is N; it takes 2 to 9999 nuggets per file` | `--dn-num2` outside 2 to 9999. `create_dn` needs it below its `dn_num1` of 10,000 |
| `--t11 takes numbers like 22 or 22,20,18,18, got 'a'` | A threshold that isn't a number. `--t21` gives the same message |
| `--t11 gives 2 values for 3 files; give one, or one per file` | A threshold list of the wrong length |
| `--pops gives 2 choices for 3 files; give one for all of them or one per file, separated by ;` | A `--pops` list of the wrong length |
| `output path already exists: PATH; choose a new --out folder` | The selected output path already exists |
| `choose at least two distinct fluorescence channels for clustering and PCA` | Fewer than two distinct clustering channels |
| `--channels repeats a channel; list each channel only once` | Duplicate entries in a list with at least two distinct channels |
| `223.fcs has no channel FOO; its channels: FSC-A, FSC-H, ...` | A file without the `--fsc` or `--ssc` channel |
| `--pops: there is no Pop 9; this file has Pop 1 to Pop 7` | A `--pops` entry naming a population the file doesn't have |
| `--pops: 'x' isn't a population number or a range like 1-3` | A `--pops` entry that doesn't parse |
| `no terminal to ask in; pass --pops, e.g. --pops "1,2"` | No `--pops`, and standard input isn't a terminal |
| `cancelled; nothing was clustered` | Ctrl+C or the end of input at the prompt |
| `223.fcs has no channel FOO; its channels: FSC-A, FSC-H, ...` | A file without one of the channels to cluster |
| `no file has cells kept, so there is nothing to cluster` | Every file was left out or kept fewer than 2 events |
| `the kept cells make fewer than 3 occupied nuggets, too few for weighted Ward clustering; keep more cells or increase --dn-num2` | Fewer than 3 nuggets with weight above 0 |
| `datanuggets-flow needs the flow extra: pip install "datanuggets[flow]"` | matplotlib or pandas isn't installed |

A bad flag value, such as `--disc mirror` or a non-integer `--k`, gets argparse's usage message and exit status 2. Other errors can end in a Python traceback with exit status 1: missing flowio, an unreadable FCS file, or invalid scatter values. A failed run retains its incomplete marker and any partial outputs.

## Examples

Each output directory below must be unused. Replace `path/to/fcs_folder` and the channel names with your inputs.

```sh
datanuggets-flow path/to/fcs_folder --out outputs/run-001
datanuggets-flow path/to/fcs_folder --pops "1,2" --no-show --out outputs/run-002
datanuggets-flow a.fcs b.fcs --pops "1,2;1" --out outputs/run-003
datanuggets-flow path/to/fcs_folder --t11 22,20,18,18 --t21 0.25,0.5,0.5,0.5 --out outputs/run-004
datanuggets-flow path/to/fcs_folder --pops "1,2" --disc peak --k 12 --out outputs/run-005
datanuggets-flow a.fcs --pops 1 --channels "FJComp-BV510-A,FJComp-Spark Blue-A,FJComp-Alexa Fluor 700-A" --cofactor 0 --dn-num2 200 --k 3 --out outputs/run-006
```

The last command uses three untransformed channels from Pop 1, reduces the retained events to up to 200 nuggets in total, and groups those nuggets into three clusters.
