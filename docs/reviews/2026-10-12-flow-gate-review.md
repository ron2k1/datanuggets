# Flow gate review, 2026-10-12

The range under review is the [flow gate plan](../plans/2026-10-12-flow-gate.md) on `feat/flow-gate`, 84599ab..7608fe5, with the README and architecture.md sections still uncommitted then: `datanuggets.flow` (`f_plot`, `local_maxima`, `refine_peaks`, `find_area`, `hull_gate`, `lymphmacro_finder`), `plot.gates` and the comparison scripts in `benchmarks/flow`, against the [design](../specs/2026-10-12-flow-gate-design.md). Its findings are fixed in 4373352..6638074 and b4fcc1b, and the docs commit that adds this record.

One reviewer ran: Opus, read-only, on the laptop, Windows with Python 3.14.3. It ran `tests/flow` and `tests/plot/test_plot_flow.py` (97 passed), drove `compare.ps1`'s parameter binding through stubs with the same param block, probed Qhull on thin slivers of events and tried mutants in memory without editing the source. It didn't run the mini or the full suite, and it found no FCS file on the laptop. Its verdict was changes needed: no blockers, one major finding, six minors and a list of nits. The major one broke the Windows wrapper in the forms the README showed. On the package itself it found one uncaught Qhull error his code shares, and tests that couldn't fail.

## Findings

| Finding | Severity | Fixed in | Evidence it was real |
|---|---|---|---|
| 1. `compare.ps1` breaks an unquoted comma list and can't run `--metadata` mode | major | 4373352 | see below |
| 2. `hull_gate` lets a `QhullError` out of `Delaunay` on a sliver | minor | d63556c | see below |
| 3. `compare.py` loses every file's rows when one file raises | minor | a0e324f, 5b3d3cf | `summary.csv` was written once, after the last file |
| 4. `stress.py` leaves his masks behind, drops his traceback and always exits 0 | minor | b973872 | an 847 KB `stress_his_masks.npy` left in `tmp/flow` |
| 5. no test pins his guards or his peak limits | minor | 80299e3 | eight mutants survived, see below |
| 6. the float32 test can't fail, and `bin_mask` and pandas columns aren't tested | minor | b53b41c, 6638074 | see below |
| 7. two README sentences are wrong | minor | b4fcc1b | see below |

### compare.ps1's arguments

PowerShell reads an unquoted `--t0 22,20,18,18` as an array, and binding it to `[string[]]$Rest` hands the script the one string `22 20 18 18`. `numbers()` split on commas only, so the form the README showed stopped with "could not convert string to float". And with the file or folder as a mandatory positional parameter, `compare.ps1 --metadata x.xlsx --dir D` bound `--metadata` as the file, and argparse exited 2. Both reproduced in the reviewer's stubs.

`numbers()` now splits on commas and whitespace, and the script has no positional parameter, so argparse alone reads the arguments. The script refuses `--out`, since it looks for the picture in its own folder, and opens a picture only if this run wrote it. It was checked on two copies of `202.fcs` with `--t0 22,20 --t00 0.25,0.5`.

### The sliver

`hull_gate` caught `QhullError` from `ConvexHull` but tested the events against `Delaunay` of the hull's vertices outside the try. The two run Qhull with different options. The reviewer's probe, 2,000 events with FSC-A from 4e5 to 1e6 and SSC-A 3e5 plus 1e-9 times normal noise, passes `ConvexHull` and raises "QH6154 Initial simplex is flat" from `Delaunay`, where README departure 4 promises an empty population. It raised in 20 runs of 20. His own `Delaunay` raises there too, and float32 FCS values near 3e5 sit about 0.03 apart, so no real file reaches it. The test now runs inside the same try.

`bin_mask` keeps its shape. Its cells are integer grid indices, which are either exactly collinear, caught by `ConvexHull`, or span a triangle of area at least 0.5.

### The guards and limits

The guard test passed 3 collinear cells, which come back empty with or without the guard, and the filter test put its peaks at rows 5 and 95 and column 95, far from his limits. The reviewer moved each constant by one in memory: `MIN_POINTS` 3 to 2 and 4, `ROW_MIN` 10 to 9 and 11, `ROW_MAX` and `COL_MAX` 90 to 89 and 91. All eight survived, while `ROW_MIN` at 30 and `MIN_POINTS` at 1000 were caught.

The new tests sit on each side of each limit: rows 10 and 11, 89 and 90, columns 89 and 90, and column 1, since he sets no lower column limit. The guards get 3 cells against 4 spanning the same triangle, and 3 kept events against 4. Each of the eight mutants now fails a test, and each `MIN_POINTS` mutant fails both guard tests.

### The float32 test, bin_mask and pandas

The float32 half of the plan's Review Focus 1 fed the fixtures' columns. Their values are exact in float32, and float32 arithmetic bins every one of their events where float64 does, so the test would pass with `channel`'s float64 cast dropped. 6638074 adds an event on a float32 bin edge. With FSC-A from 369,044, `202.fcs`'s floor, to 2^22, 4,003,041 sits on bin 20's lower edge. In float32, 20 times 3,633,997 rounds half to even down to 72,679,936, so float32 arithmetic puts the event in bin 19. The test asserts that first, so the case can't quietly stop being one, then that `f_plot` puts it in bin 20. With `channel` returning `np.asarray(v)` in memory, the new test fails and the old one still passes.

His code keeps float32 input in float32, so on such an event his bin and ours would differ. Nothing hands him float32. The benchmark gives both sides flowio's `as_array(preprocess=True)`, which is float64 (flowio 1.4.0 on `202.fcs`, on the mini) and equal to FlowKit's raw events on the three files the mini compared.

`bin_mask` was checked only through `hull_gate`, behind a second hull and its allowance for events on the hull's edge. His code triangulates the area hull's vertices for the bins too, so `bin_mask` must equal his `jj_type` exactly, and b53b41c asserts that on each fixture's populations. It also gives `lymphmacro_finder` pandas Series with a reversed index, the way FlowKit's DataFrame columns would arrive, and the masks equal the plain arrays'.

### The README sentences

The README gave a peak's "rows and columns between 10 and 90". The filter is a row above 10 and below 90 and a column below 90, with no lower limit on the column, as `peaks.py` and his code have it. And the README and architecture.md said flow needs "numpy and scipy only", but `import datanuggets.flow` runs `datanuggets/__init__.py`, which loads numba and joblib. Both now say its own modules import numpy, scipy and `_validate`, and that importing it loads the package.

## Nits

| Nit | Fixed in |
|---|---|
| `hull_gate` and `bin_mask` don't check `zz`: a 0 bin read the grid's last cell with no error, and a float `zz` raised `IndexError` from inside numpy | 2eafe52 |
| `plot.gates` takes its colors from rcParams rather than the axes' cycle, and fixes the edge at 1 point where his follows `patch.linewidth`. On axes given their own cycle, his first population was black where ours was blue | bfcbd82 |
| `compare.ps1` passes a `--out` through but looks for the picture in `tmp\flow` | 4373352 |
| `scp` without `-p` stamps every copied file with the copy time, so a picture from an earlier run could open as new | 55d2d26 |
| a `'` in an argument breaks `compare.ps1`'s quoting for the mini | 55d2d26 |
| `compare.py` quietly drops `--metadata` when it's given a target too | 55d2d26 |
| `fcs_files` matches `*.fcs` case-sensitively on macOS, so `.FCS` files are missed there | 2c6dd68 |
| `run_cabrera.py`'s usage line has its continuation flattened into a run of spaces | 2c6dd68 |
| the README says `run_cabrera.py` changes "only its two paths", leaving out the Agg backend | b4fcc1b |
| our gate's timer starts after `cells` is built, while his filtering is timed | not changed |

The timing one isn't changed. Building `cells` takes well under a millisecond against his seconds, and the reviewer judged it doesn't change a result.

## Suspicions and questions

- His study's row count. His loop reads `range(4)`, and `compare.py --metadata` needs one `--t0` value per file in the workbook, saying so when the count differs. Taking the first `len(t0)` rows instead would run fewer files than the workbook names without a word, the quiet drop 55d2d26 refused elsewhere. Not changed.
- No cap on differing events. Over the 44 files of the Zenodo record, the gate on his cells and the finder given his `find_area` differ from his by at most one event per file, on 4 files, each event on the hull's edge at distance 0, and `compare.py` prints the count. A cap would be a guess at a number no file has come near, so none is set.
- Float32 input at scale. Ours bins float32 by its float64 values, now pinned by finding 6's test. His code fed float32 would bin in float32, but neither his FlowKit pipeline nor the benchmark feeds him float32, so it wasn't run at 10 million events.
- The README's figures from `202.fcs`. The reviewer had no FCS file to check them. They come from runs on the mini: his populations of 225,770, 58,245 and 27,590 events, the shipped ones of 214,085, 66,149 and 37,742, with 57,797 events in two or more and 29,065 in none, and 107,909 in a different set of populations. The reviewer did trace the README's claim that one kept event differed end to end: population 1 on the corrected area, event 43179, ours in and his out, and the event is a hull vertex.

## Found while fixing

- 74efddb carries finding 3's change to `compare.py` under finding 6's message, and fails ruff's BLE001. A heredoc's closing line ended the `&&` chain, so the commit ran after the lint failed, with the last message on disk, and it was pushed. a0e324f reverts it and 5b3d3cf brings the change back under its own message, with a `noqa` for BLE001 on the catch that's blind on purpose, rather than rewriting history that was already pushed. Read 74efddb's diff as 5b3d3cf's.
- A run stopped at `stress.py --cap` exits 0, since the time it reached is the measurement. A failed child or a mask that differs exits 1.

## Checked and found correct

The reviewer traced each of these:

- Pieces 1 to 3 match his: the same bin expression and order of operations, `maximum_filter` with reflect equal to his `generic_filter`, a strict `>` against the same quantiles, `np.unique` and the unstable `argsort` reproduced by the identical call on identical int64 arrays, and 1-based indices.
- `bin_mask` is his `find_simplex(zz)`. The hull's vertices and their order are his, the coordinates are the same integers and `zz`'s maximum is always `nr`. Only the query order differs, and that can't move an integer grid point across an integer hull.
- `Delaunay` and `ConvexHull` both raise on collinear points and on two repeated points, so departure 4 held apart from finding 2.
- `find_area`'s disc on the peak and its column-major order are right, and int or float thresholds give the same areas.
- `compare.py`'s exit status matches its docstring, and `gate_match` fails closed on `QhullError`. Zero peaks, a one-file folder and the comma lists work when it's called directly.
- The `~` path through Git Bash and `mini-run.sh`. The single quotes keep `~` literal, `printf %q` sends it as `\~` and Python's `expanduser` expands it on the mini. The exit code comes back through bash, ssh and caffeinate.
- `plot.gates`' checks, its skipping of an empty population's color, its sizes and its alphas match his `scatter`.
- architecture.md's module rows and import table match the code.

## Test runs

At 6638074, after the fixes, with numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2 and numba 0.68.0:

- Laptop, Windows: 2928 passed and 2 skipped (no `os.fork` on Windows) on Python 3.14.3 and on 3.15.0, with the 157 slow tests deselected as by default. ruff is clean, and pyright finds no errors in `datanuggets.flow` and `datanuggets.plot`.
- Mac mini, arm64: 2930 passed on 3.14.3 and on 3.15.0.
- Mutants: each mutant the reviewer described fails the tests after its fix, the eight constants of finding 5 and `channel` without its cast.

The comparison over all 44 files and the stress run on their joined 10,183,290 events are in the README's Speed section.
