# Flow gate: `lymphmacro_finder`, faster than Cabrera's Python

- Status: designed in brainstorming on 2026-10-12. RON2K set the reference and the test ("we just need to get our own big dataset transform it and test it on the mac mini comparative to ours, and then we will go piece by piece to see if it matches output with faster speed and graphs on python") and approved the design ("we need the flowgate done").
- Author: Ronil Basu (Rutgers)
- Upstream: the Python script Prof. Cabrera sent on 2026-10-12, his own port of his R code. It reads FCS files with FlowKit and finds the lymphocyte and macrophage populations on FSC-A against SSC-A with `f_plot`, `search_ind`, `find_area` and `lymphmacro_finder`. The R original isn't used.
- Branch: `feat/flow-gate`, stacked on `feat/plotting`, since the graph goes in `datanuggets.plot`.

## 1. Goal

Cabrera's script gates each sample's scatter plot into populations: it bins FSC-A and SSC-A into a 100 by 100 density grid, finds the grid's local maxima, grows an area around each peak, and keeps the events inside the convex hull of each area. RON2K wants the same thing in the package, faster, on a large real dataset, checked piece by piece against Cabrera's script on the Mac mini, with the graphs in Python.

### Success criteria

1. Pieces 1 to 3 (section 4) give arrays equal to his (`np.array_equal`) on all 44 files of the dataset.
2. Piece 4 equals the corrected rule on every recorded case. Against his `find_area` it differs only by where the distance disc is centred, and the comparison reports the differing cells per file.
3. Piece 5, run on his own piece-4 cells, gives his event masks event for event on all 44 files. Any event that differs is listed with its distance to the hull edge, and none may be farther than 1e-9 of the data's range.
4. Ours is faster than his end to end on every file, and on each piece that takes him more than 1 ms. All timings are taken on the Mac mini, from the same arrays, in the same Python.
5. RON2K can run the whole comparison himself with one command on the laptop and one on the mini (section 6), and Cabrera can run the same command on his own study files.
6. The suite passes on Python 3.14 and 3.15 on the laptop and the mini, with ruff and pyright clean.

### Non-goals

- Parity with Cabrera's R code. RON2K ruled that his Python is the reference.
- Reading FCS files or compensating inside the package. The package takes FSC-A and SSC-A arrays. FlowKit is used only in the benchmark, the way his script uses it.
- Retuning his cutoffs (the 0.85 and 0.9 quantiles, the 10 and 90 bin limits, `t11` to `t22`) for this dataset.
- Checking that the gates are biologically right. The dataset has no reference gates to compare with.

## 2. The data

[Zenodo record 22079206](https://zenodo.org/records/22079206), "Flow cytometry data of PBMC of critically ill, mechanically ventilated patients", CC-BY-4.0: 44 FCS 3.1 files from a Cytek Aurora, 1.6 GB, with a metadata sheet. File 202 has 282,270 events and 39 channels, among them FSC-A and SSC-A, and the file sizes put the 44 files near 10 million events. The files were compensated in FlowJo before export (`FJComp-` channels, no spill keyword), so his script takes its `raw` branch. A debris gate was applied before export (FSC-A starts at 369,044 in file 202), and the lymphocyte blob and the monocyte cloud are both there.

"Transform it" means putting the dataset in his study layout: `benchmarks/flow/prepare.py` writes a metadata workbook whose second sheet has the `Filename` column his script reads, next to the folder of FCS files. His script then runs with only its two path lines changed. The sqrt transform of FSC-A and SSC-A stays inside the finder, as he wrote it.

The mini downloads the files itself from Zenodo and checks each one's md5.

## 3. Scope and API

```python
# datanuggets.flow
def f_plot(x, y, *, nr=20, nc=20, scale="raw", alpha=0.1) -> DensityGrid
def local_maxima(z, *, n_row=11, n_col=11, quantile=0.85) -> np.ndarray
def search_ind(x, idx, idy, threshold=2) -> np.ndarray
def find_area(x, idx, idy, t1=35, t2=0.85) -> np.ndarray
def hull_gate(fsc_a, ssc_a, zz, cells) -> np.ndarray
def lymphmacro_finder(fsc_a, ssc_a, *, n_row=11, n_col=11, t11=22, t21=0.25, t12=20, t22=0.5) -> np.ndarray

# datanuggets.plot
def gates(fsc_a, ssc_a, masks, *, title=None, ax=None) -> Axes
```

The pieces keep his names so the two can be read side by side. `DensityGrid` holds his `z`, `x`, `y` and `zz` (his padded `zx` and `zy` are an artifact of R's `table` and aren't kept). `lymphmacro_finder` returns his `final_indices`: a bool array with one row per event and one column per population, populations in his order.

| Module | Holds |
|---|---|
| `flow/__init__.py` | the public names |
| `flow/grid.py` | `f_plot`, `DensityGrid` |
| `flow/peaks.py` | `local_maxima`, `search_ind`, the peak filters, `find_area` |
| `flow/gate.py` | `hull_gate`, `lymphmacro_finder` |
| `plot/flow.py` | `gates` |

No new runtime dependency: numpy and scipy are already required, and `gates` uses the `plot` extra's matplotlib. A new dependency group `flow` holds FlowKit, for Python below 3.15 only, since flowutils has no 3.15 wheel.

## 4. The pieces

Each piece is compared on the same inputs, his function against ours.

**1. Density grid (`f_plot`).** His version pads the bin indices so R's `table` sees every level, counts with `np.histogram2d` and subtracts the padding. Ours computes the bin indices with his exact expression, `floor(nr * (x - min) / (max - min)) + 1` clipped to `[1, nr]`, so `zz` is the same, and counts with one `np.bincount` on the flat index. The counts are the same integers, so `z` is equal after his `log1p` scale and his alpha cut.

**2. Local maxima.** His `generic_filter` calls a Python function for each of the 10,000 cells. Ours takes `scipy.ndimage.maximum_filter` with the same 11 by 11 window and the same `reflect` edges, then his rule: the window maximum where it is above the 0.85 quantile of `z`, else -1. A maximum is exact, so the candidates `x == local_max` are equal.

**3. Peak refinement.** `search_ind` per candidate, then his `np.unique`, his filters (`row < 90`, `col < 90`, `row > 10`, density above the 0.9 quantile) and his `np.argsort` on the row. Ours makes the same calls in the same order, since `np.unique`'s sort and `argmax`'s first-in-row-order tie rule decide which peak comes first and gets `t11` and `t21`.

**4. Area (`find_area`).** His version has a transpose bug. It builds the grid row by row and flattens the density column by column, then swaps the output columns. The density test lands on the right cell, but the distance disc is centred on the mirrored peak `(idy, idx)`. A probe matched that description in 200 of 200 random grids, and on a single blob at row 20, column 70 his version keeps 0 cells where the disc at the peak keeps 121. Ours centres the disc on the peak: a cell `(i, j)` is kept when its distance to `(idx, idy)` is below `t1` and `x[i, j]` is above `t2 * x[idx, idy]`, and cells come back as `(i, j)`, the axes of `zz`. On a peak on the diagonal the two agree.

**5. Hull gate.** For each area his code takes the convex hull of its cells, keeps the events whose bin `zz` is inside it, and then keeps every event inside the convex hull of those events, testing with `Delaunay(...).find_simplex(...) >= 0`. The second step triangulates every kept event, which is where his time goes. Ours:

- Tests each distinct occupied bin once instead of each event, with the same `Delaunay(hull vertices).find_simplex` call, and looks the answer up per event. Events in one bin have the same coordinates, so the answer can't change.
- Takes `ConvexHull` of the kept events and tests every event with `Delaunay(hull vertices).find_simplex`, a triangulation of the hull's vertices (tens of points) in place of all the kept events (hundreds of thousands). Both triangulations cover the same polygon. They can disagree only on an event within rounding of the hull's edge, which criterion 3 counts.

The piece runs twice in the comparison: on his piece-4 cells, which tests the gate alone, and end to end.

**6. Graph (`gates`).** His figure: 6 by 6 inches, every event in gray (`s=0.5`, alpha 0.3), each population over it (`s=5`, alpha 0.8, matplotlib's default colors), his title, axis labels and legend. Ours draws the same artists, the scatter layers rasterized so a PDF stays small. The test checks the drawn offsets and colors, as the other plot tests do.

## 5. Departures from his script

1. `find_area` centres the disc on the peak (piece 4). This is the one change in output, and RON2K decides at that piece whether it stays.
2. No peak left after the filters: his code raises `IndexError` at `tt[0, 0]`, ours returns a bool array with zero columns.
3. An area whose cells are all on one line: his `ConvexHull` raises `QhullError`, ours leaves that population empty. His guard of more than 3 cells is kept.
4. NaN or infinite FSC-A or SSC-A, or a channel with one value: his code bins garbage or divides by zero, ours raises `ValueError`, as the rest of the package does.
5. His unused `plot=False` argument is dropped.

Outside the package, his `sample.apply_compensation(spill)` returns `None` (FlowKit documents `:return: None`), so his `dt2_compensated` holds `None` for each sample with a spill matrix. Nothing reads it, and the scatter channels aren't compensated anyway. The benchmark notes it and reads raw events as his loop does.

## 6. The comparison RON2K runs

```
uv run --group flow python benchmarks/flow/compare.py <fcs file | folder> [--metadata X.xlsx --dir D] [--repeat 5] [--out tmp/flow]
```

On the laptop `.\benchmarks\flow\compare.ps1 <path>` wraps it and opens the picture. On the mini it goes through `mini-run.sh`. Cabrera can run the same command on his Mac with `--metadata` and `--dir` pointed at his 20250618 study files, and `--t0 22,20,18,18 --t00 0.25,0.5,0.5,0.5` for his per-sample settings.

For each file it reads FSC-A and SSC-A with FlowKit as his script does, runs each piece his way and ours on the same arrays, and prints one table:

| Piece | His s | Ours s | Speedup | Match |
|---|---|---|---|---|
| 1 grid | | | | equal / first difference |
| 2 maxima | | | | |
| 3 peaks | | | | |
| 4 area | | | | equal / cells differing |
| 5 gate on his cells | | | | equal / events differing |
| 5 gate end to end | | | | equal / events differing |
| 6 graph | | | | |

Times are medians of `--repeat` runs. It writes `<file>_compare.png` with his graph, ours and the events gated differently, and appends a row per file to `summary.csv`.

`benchmarks/flow/cabrera_reference.py` holds his functions word for word. `benchmarks/flow/run_cabrera.py` is his whole script, with only the two paths taken from the command line and matplotlib's Agg backend, so it runs unattended. The comparison checks once that the whole script's masks equal the masks of his pieces run one by one, which shows the reference copy is his code.

## 7. Fixtures and tests

`benchmarks/flow/record.py` runs his pieces on three files, each thinned to every k-th event for 50,000 events, and saves the inputs and each piece's output to `tests/fixtures/flow/<file>.npz`: FSC-A and SSC-A as float32 (the FCS values are float32), `zz` as uint8, masks bit-packed. That's about 2 MB. `tests/fixtures/flow/README.md` credits the Zenodo record under CC-BY-4.0. The fixtures are recorded on the mini and checked on the laptop as well, since nothing here depends on the x87 long double.

`tests/flow/` checks each piece against the fixtures with the criteria of section 1, `find_area` against an independent re-statement of the corrected rule and the blob case, the departures of section 5, and `gates`' artists.

## 8. The speed report

`compare.py` on the mini over all 44 files at `--repeat 5`. Then a stress run on all 44 files joined into one array of about 10 million events, with his version capped at 30 minutes. The README gets a section with the per-piece medians, the spread over the 44 files and the stress result. Times exclude reading the FCS files, which both sides do the same way.

## 9. Risks

- His cutoffs were tuned on his study. On this instrument's scale the finder may find no peak, or one, in some files. The comparison counts those files. If most files find nothing, the build stops and RON2K decides whether to change the thresholds.
- Events on the hull's edge in piece 5, counted by criterion 3.
- FlowKit's dependencies (bokeh, pyarrow, lxml) make the `flow` group heavy. It stays out of the default install.

## 10. Build order

1. Reference copy, `prepare.py`, `record.py`, fixtures from the mini.
2. Pieces 1 to 5, each test first against its fixture.
3. `lymphmacro_finder` end to end.
4. `gates`.
5. `compare.py` and `compare.ps1`.
6. The 44-file run and the stress run on the mini, then the README section and `docs/architecture.md`.
7. Review, fixes, PR into `feat/plotting`.
