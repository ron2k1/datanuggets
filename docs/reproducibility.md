# Reproducibility

This page is for reproducing a number that datanuggets reports. It starts with what every experiment shares: the machines and environments, how seeds behave, and the datasets with their licenses. Then come what R recorded for the tests and one section per experiment, each with what it measures, its command, what it prints and its recorded result. Installation is in [installation.md](installation.md), running the tests in [development.md](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md), and the differences from R in [compatibility.md](compatibility.md).

The benchmark commands require a full Git clone, including the development tools and `uv.lock`; these are not included in the source archive. Run every command on this page from the repository root. The R scripts stop anywhere else, and the flow scripts write to `tmp/flow` under the current folder.

## Environments

### Machines

| Machine | Hardware | System | Used for |
| --- | --- | --- | --- |
| Laptop | Intel i7-13650HX, 16 GB | Windows 11 | every R fixture, the Covertype and `WCluster` benchmarks, the suite at the merge |
| Mac mini | Apple M4, 16 GB | macOS 26.6.2, arm64 | the suite at the merge, the `WCluster` benchmark, the flow fixtures and every flow comparison |

### The R reference

Every R result in the tests and benchmarks comes from R 4.6.1 on the laptop. `r/fixture_io.R` writes the environment into the `versions` field of each output's `manifest.json`, so a case recorded on another build would show it. All 544 fixture manifests record the same values:

| Field | Value |
| --- | --- |
| `R` | `R version 4.6.1 (2026-06-24 ucrt)` |
| `platform` | `x86_64-w64-mingw32` |
| `datanugget` | `1.5.0` |
| `Rfast` | `2.1.5.2` |
| `lapack` | `3.12.1`, R's internal LAPACK |
| `blas` | `Rblas.dll md5 5fe8258e86d8a1a52a6c3db05faa0225`, R's reference BLAS |
| `longdouble_digits` | `64`, the x87 80-bit long double |

Some manifests also record `WCluster` 1.3.0 and R's `matprod` option, `default`.

R's last bits depend on that build. Rfast's `dista` sums Euclidean terms through the BLAS for more than 32 columns, `colMeans` and `rowMeans` add in long double, and `%*%` follows the reference BLAS's order. The port repeats that arithmetic in its own code instead of using the host's. So an R built with OpenBLAS, MKL or Accelerate, or R on arm64 macOS, where `long double` is plain double, can round differently from the fixtures, and on tied data that can change assignments. [compatibility.md](compatibility.md) and [limitations.md](limitations.md) have the details.

To set up the same R, follow [r/README.md](../r/README.md): install R 4.6.1, then run `Rscript r/setup.R`, which installs the packages into a private library whose location `DATANUGGETS_R_LIB` overrides. `setup.R` stops unless it installed `datanugget` 1.5.0 and `WCluster` 1.3.0, and every script that sources `r/fixture_io.R` checks both again. Rfast comes from CRAN as a dependency and isn't pinned. `setup.R` prints its version without checking it, so compare that line with 2.1.5.2 before recording anything.

### Python

Python 3.14 and 3.15 are two interpreters running the same package. Choose one for normal use; separate environments are needed only to compare or test both. The numbers below record the interpreter used in each experiment.

`uv.lock` pins numpy 2.5.3, scipy 1.18.1 and numba 0.68.0, with llvmlite 0.50.0 and joblib 1.6.0. Its dev group adds what the plots, the flow scripts and the tests use, among them matplotlib 3.11.2, pandas 3.0.6, flowio 1.4.0, openpyxl 3.1.5 and pytest 9.1.1. The suite, Covertype and `WCluster` runs below give the Python they used, 3.14.3 or 3.15.0. The flow runs don't record theirs. To get the locked environment on a given Python:

```sh
uv sync --locked --python 3.14
```

`--locked` stops if `uv.lock` and `pyproject.toml` disagree, instead of resolving again. Keeping a second Python's environment next to the first is covered in [development.md](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md).

Two outputs still depend on the host. The eigen output in `wpca` and `refine_dn` comes from the installed scipy's LAPACK, so the tests hold it to R's by a rule or a tolerance. A NaN that `distw` computes in its lower triangle carries another sign bit on arm64. Everything else gives the x86-64 R results on arm64 macOS too, where the same tests pass.

numba compiles each kernel on first use and caches it per Python version, so the first run on a new Python is slower. The Covertype timings below were taken with the cache warm, and `wcluster.py` reports its first calls on a line of their own.

### Pinning and memory on Windows

`benchmarks/pin.py` runs a command, and every process it starts, on chosen CPUs with Windows power throttling off. `benchmarks/measure.py` runs a command and reports its wall time and the peak working set across its process tree, since `Rscript` hands its work to a child `Rterm`. Both need psutil, which `uv run --with psutil` supplies, and both read Windows-only process information, so they run only on Windows. Put either in front of a benchmark command:

```sh
uv run --with psutil python benchmarks/pin.py Rscript benchmarks/covertype.R 10000
uv run --with psutil python benchmarks/pin.py uv run python benchmarks/covertype.py 10000
uv run --with psutil python benchmarks/measure.py uv run python benchmarks/covertype.py 10000
```

`PIN_MASK`, a hex affinity mask, picks the CPUs. Its default, `0x555`, puts one thread on each of the laptop's six performance cores, which sit on logical CPUs 0 to 11. Check which CPUs are which before using it on another machine. Unpinned, Windows sometimes put a run on an efficiency core, where the same kernel took 2.8 times as long, and one run could take twice as long as the next.

`measure.py` prints one line, `MEASURE exit=<status> wall=<seconds>s peak_wset=<MiB> MiB`, and exits with the command's status. Its wall time includes starting the process and loading the data, unlike the timings the benchmark scripts print.

## Seeds

R's `create.DN`, `refine.DN`, `Wkmeans` and `DN.Wkmeans` default to `seed = 291102` and call `set.seed` with it themselves, so every call starts from the same stream. The port keeps 291102 as the default seed of `create_dn`, `refine_dn`, `wkmeans` and `dn_wkmeans` (`SEED` in `src/datanuggets/_defaults.py`). What it draws depends on the seed's type:

| `seed=` | Draws | The same call twice |
| --- | --- | --- |
| an int, the default 291102 included | numpy's `default_rng(seed)`, restarted on every call | the same result, but not R's |
| a numpy `Generator` | a new child stream per call, from `Generator.spawn` | different results. Calls made in the same order from a Generator with the same seed repeat |
| `RDraws(seed)` | R's Mersenne-Twister `set.seed(seed)` and `sample.int`, restarted on every call | R's result for that seed |

`RDraws` isn't exported from `datanuggets` yet, so import it from `datanuggets._random.r_draws`. The same default seed gives different nuggets through numpy and through R's draws:

```python
import numpy as np
from datanuggets import create_dn
from datanuggets._random.r_draws import RDraws

x = np.random.default_rng(0).normal(size=(20_000, 3))
a = create_dn(x, dn_num1=1000, dn_num2=100)  # numpy's draws from seed 291102
b = create_dn(x, dn_num1=1000, dn_num2=100, seed=RDraws(291102))  # R's draws from seed 291102
print(np.array_equal(a.assignments, b.assignments))  # False
```

`n_jobs` doesn't change a result. Tests check that for `create_dn` with an int seed, `refine_dn` with an int and a `Generator`, `refine_dn` with `RDraws` on one and two workers, and `wkmeans` with `RDraws` on 1, 4 and all threads. Replaying R's recorded draws is different: `ReplayDraws` hands them out in R's order, so the benchmarks call R with `no.cores = 1` to record them in order and the port with `n_jobs=1` to read them back. `refine_dn` refuses a replay with any other `n_jobs`.

291102 appears in these places:

- `tests/randomness/test_rdraws.py` runs all 128 recorded `refine.DN` cases with `RDraws` of the case's seed, 291102 when the case gives none, and checks two workers against one.
- `tests/wcluster/test_wkmeans.py` runs iris with k = 3 and five starts under `RDraws(291102)` against R's labels, and the 28 runs of the `wkmeans_args` group.
- `tests/randomness/test_draws.py` checks that two `NumpyDraws(291102)` give the same stream.
- `benchmarks/wcluster.py` runs `wkmeans` on 10,000 Covertype rows with `RDraws(291102)`. `benchmarks/wcluster.R` and `benchmarks/covertype.R` rely on the R functions' default of 291102.
- In `r/`, `fixtures_create_dn.R`, `fixtures_refine_dn.R`, `fixtures_kmeans.R`, `fixtures_wkmeans.R`, `wcluster_counted.R`, `fixtures_dn_wcluster.R`, `fixtures_compat.R` and `fixtures_rsample.R` pass it to the functions they record.

Other seeds build inputs. `covertype.R` draws its 10,000 and 100,000 row subsets after `set.seed(20260929)`. The generators in `r/` use other seeds to make some test inputs, and `fixtures_rsample.R` records `set.seed` itself on 12 seeds, among them negative, fractional and near-overflow values. Every fixture stores its inputs, so the tests never draw them again.

The flow gate finder draws nothing. The `datanuggets-flow` command calls `create_dn` with the default int seed; [cli.md](cli.md) covers its options.

## Datasets

### R's built-in datasets

The fixtures use datasets that ship with R in its `datasets` package: iris, faithful, quakes, mtcars, volcano, ChickWeight, EuStockMarkets, USArrests, PlantGrowth, rock, LifeCycleSavings, infert, trees, npk, airquality, women, swiss and state.x77. Other inputs are built in the scripts, such as whole-number grids, tie midpoints and heavy weights, or are nuggets R's `create.DN` made of iris, quakes and scaled volcano. Each fixture stores its inputs, so the tests read them from `tests/fixtures/` and need neither R nor its datasets.

### UCI Covertype

Blackard, J. (1998). [Covertype](https://doi.org/10.24432/C50K5N) [Dataset]. UCI Machine Learning Repository. It's licensed under CC BY 4.0.

The benchmarks read the [data file](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz) saved as `tmp/covertype/covtype.data.gz`: 581,012 rows, of which they use the first 10 columns, the continuous ones. They're all whole numbers, so there are plenty of ties. `benchmarks/covertype.R` stops unless the file's SHA-256 is this one:

```text
614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771
```

`benchmarks/wcluster.py` reads the same file without checking it. `sha256sum tmp/covertype/covtype.data.gz` prints the hash in Git Bash or on Linux, and `Get-FileHash tmp/covertype/covtype.data.gz` prints it in PowerShell, in capitals.

### Zenodo record 22079206

The flow data is [Zenodo record 22079206](https://zenodo.org/records/22079206), "Flow cytometry data of PBMC of critically ill, mechanically ventilated patients". It's licensed under CC BY 4.0. It holds 44 FCS 3.1 files from a Cytek Aurora, 10,183,290 events in all and 20,844 to 516,810 a file. The flow comparisons below take a folder holding them.

The fixtures in `tests/fixtures/flow/` are three of its files, `154`, `202` and `223`, each thinned to every k-th event and cut at 50,000 events. `benchmarks/flow/record.py` recorded them on the Mac mini on 2026-10-12 from Prof. Cabrera's own Python. Their [README](../tests/fixtures/flow/README.md) gives each file's k, its populations and every key the files hold.

## What R recorded

Every R result the tests compare with comes from R itself. A few were checked by hand in an R session and written into the tests, such as the cancelling-weight cases in `tests/wcluster/test_wstats.py` and `arithmetic_complexity` on an `x` with no rows or no columns. The long double fuzz checks against a Python reference instead of R. Everything else comes from the scripts in `r/`. They run R 4.6.1 with the pinned packages, trace the functions under test without modifying them, and write the inputs, the outputs, every random draw and the internal values that matter into `tests/fixtures/`. Each case carries a manifest with the R version, the package versions, the BLAS and the long double width. Running the tests doesn't need R. Fixtures are recorded on one machine only, the laptop with the pinned R. There are 544 cases in 23 groups:

| Group | Script in `r/` | Cases | What R recorded |
| --- | --- | --- | --- |
| `create_dn` | `fixtures_create_dn.R` | 37 | `create.DN` inputs, every random draw, every `create.DNcenters` call, and the outputs |
| `refine_dn` | `fixtures_refine_dn.R` | 128 | the input nuggets, every draw, every `kmeans` call with the rows it clustered, the messages and warnings, and the refined nuggets |
| `get_dn_means` | `fixtures_get_dn_means.R` | 4 | `getDN.means` on every nugget set the two groups above hold, and on one constructed set that pins the order of its sums |
| `ac` | `fixtures_ac.R` | 1 | 3481 `AC` calls with the loop state each returns with, and the 19 that never return |
| `kmeans`, `kmns` | `fixtures_kmeans.R` | 13, 15 | `stats::kmeans` end to end with its draws, and R's own `kmns` Fortran from fixed starting centers |
| `rcompat_dist`, `rcompat_dista`, `rcompat_rowmins`, `rcompat_split_indices`, `rcompat_means` | `fixtures_rcompat.R` | 10, 12, 1, 1, 8 | the Rfast and base R operations `create.DN` depends on |
| `rcompat_stats` | `fixtures_rstats.R` | 54 | `cov`, `sum`, `mean`, `median`, `mad`, type-7 `quantile`, `unique()` on rows, `%*%` and `eigen(symmetric = TRUE)` |
| `rcompat_wstats` | `fixtures_rwstats.R` | 36 | `%*%` under the reference BLAS, `weighted.mean`, and vectors that pin the order of R's long double `sum()` and where it overflows |
| `wstats`, `wstats_constructed` | `fixtures_wstats.R` | 49, 17 | the five `WCluster` statistics on real data, and on built ties, heavy weights and overflowing rows |
| `wpca` | `fixtures_wpca.R` | 29 | `Wpca`'s outputs and the locals it hands to `eigen()` |
| `rsample` | `fixtures_rsample.R` | 27 | `set.seed` and `sample.int`, with and without `replace = TRUE`, with `.Random.seed` before and after |
| `compat` | `fixtures_compat.R` | 46 | the evidence behind the R-compatibility decisions: labels, logical and vector data, predictions with no answer, `corr` values, `nstart` and fractional nugget counts |
| `whclust` | `fixtures_whclust.R` | 14 | `Whclust`'s `merge`, `height` and `order` or R's message where it stops, and how often each quirk fired in a copy of `Whclust` with counters |
| `wkmeans` | `fixtures_wkmeans.R` | 15 | `Wkmeans`' every draw, each start's labels from `init`, each start's refined labels and WWCSS, the "Iteration" messages R prints, the outputs or R's message, and the quirk counts |
| `wkmeans_args` | `fixtures_wkmeans.R` | 1 | 28 `Wkmeans` runs on arguments R stops on, with R's message, or its labels where it returns |
| `dn_wcluster` | `fixtures_dn_wcluster.R` | 8 | two quakes datanuggets and a data frame of new rows, `DN.Wkmeans`, `DN.Whclust`, `DN.Wpca` and `DNcluster.predict` on them, and R's messages and outputs on a datanugget with one variable and on a matrix `newx` |
| `plot` | `fixtures_plot.R` | 18 | `cutree` at every `k` and at every height, the midpoints and both infinities on the 13 trees of the `whclust` and `dn_wcluster` groups, the lines `plot.hclust` drew for each tree it draws, read back from its PDF, and for one of them again at two other hangs and with its heights out of order, the disc `pch = 16` draws at `cex` 1 and 0.5, the help examples' nugget colors, and R's stops in all three |

Arrays are stored as raw little-endian float64 or int32, row-major, and `manifest.json` gives each one's shape. The flow fixtures aren't among the 544: they come from Prof. Cabrera's Python, not from R. Regenerating fixtures is covered in [development.md](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) and [r/README.md](../r/README.md).

## Experiments

### The test suite at the merge into main

The suite compares the port with the fixtures above. The first command runs the default suite, the second only the long fuzz of the compiled long double kernels:

```sh
uv run pytest
uv run pytest -m slow
```

The mini ran the default suite on 8 workers, which pytest-xdist from the dev group gives with `uv run pytest -n 8`. [development.md](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) covers the rest of the test setup.

The suite ran on both machines at the commit that merged into `main`, with no failures:

| Machine | Environment | Run |
| --- | --- | --- |
| Laptop, i7-13650HX | Windows 11, Python 3.14.3 and 3.15.0, numpy 2.5.3, scipy 1.18.1, numba 0.68.0 | the default suite and the slow fuzz on each Python. Two tests skip because Windows has no `os.fork` |
| Mac mini, Apple M4 | macOS 26.6.2 arm64, Python 3.14.3 and 3.15.0, numpy 2.5.3, scipy 1.18.1, numba 0.68.0 | the default suite on 8 workers on each Python, the two fork tests included |

That commit is `ad662ba`, the merge of PR #7 on 2026-10-09. It holds steps 1 to 10 of the history in [development.md](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md). `wkmeans`, `whclust`, the `dn_` functions, the plots and the flow finder came after it and weren't part of this run.

The package layout change was also checked from a built wheel. The wheel holds the same files as `src/`, and the suite passed in a clean virtual environment from outside the repo.

### Covertype: create_dn and refine_dn against R

This is the large-data regression check. The fixtures are small, so both packages also run on Covertype: the full 581,012 rows, and random subsets of 10,000 and 100,000 rows that `covertype.R` draws once in R after `set.seed(20260929)`. Each size runs `create.DN` with R's defaults, then `refine.DN` on its result with shape splits off and on. The only change from R's defaults is `no.cores = 1`, and `n_jobs=1` in Python, so R's random draws can be recorded in order. Python takes the rows from R's output and replays R's draws the way the tests do. `refine_dn` starts from R's `create.DN` output, so a mismatch in `create_dn` can't hide one in `refine_dn`.

The R script needs the data file and the private R library from [r/README.md](../r/README.md). Run it first, then the Python script with the same size:

```sh
Rscript benchmarks/covertype.R 10000
uv run python benchmarks/covertype.py 10000
```

The size is `10000`, `100000` or `all`, and `covertype.py` stops without one. Add `noshape` to the `Rscript` line to skip shape splits, as the full-size runs did:

```sh
Rscript benchmarks/covertype.R all noshape
uv run python benchmarks/covertype.py all
```

`covertype.R` writes each call's inputs, draws, outputs, seconds, messages and warnings to `tmp/covertype/out/covertype_<size>_create`, `_refine` and `_refine_shape`, and prints one line per call with its seconds, its nugget count and how many draws it made. `covertype.py` prints a JSON report: `n` and `p`, then for `create`, `refine` and `refine_shape` the Python seconds `python_s`, R's recorded seconds `r_s`, the nugget counts, and one true or false for each of `assignments`, `weights`, `centers`, `scales`, `messages` and `warnings`. When the assignments differ it adds how many. It skips a refine step whose R output is missing. It exits nonzero only on an error, such as the port asking for a draw R didn't make or leaving one unread, so read the checks in the report.

Assignments, weights, centers, scales, progress messages and warnings matched exactly in every run, and every output matched R's at all three sizes with the code merged at `ad662ba`. At 100,000 rows the shape-split run replayed 383,475 draws.

Wall time in seconds, around the function call only, one run each, all in one sitting. "Python, before" is `main` before step 10, commit `e3343b6`, which compiled the long double sums but not Rfast's distances, the deletion rounds or Hartigan-Wong. The last two columns are the code at step 10 on two Pythons, with the numba cache warm:

| Rows | Step | R | Python, before | Python 3.14 | Python 3.15 |
| --- | --- | --- | --- | --- | --- |
| 10,000 | create | 5.86 | 6.72 | 0.65 | 0.65 |
| 10,000 | refine | 0.17 | 0.16 | 0.09 | 0.09 |
| 10,000 | refine, shape splits | 5.61 | 1.05 | 0.45 | 0.41 |
| 100,000 | create | 37.3 | 45.8 | 4.19 | 4.35 |
| 100,000 | refine | 0.53 | 1.50 | 0.36 | 0.38 |
| 100,000 | refine, shape splits | 1801 | 32.9 | 8.28 | 8.22 |
| 581,012 | create | 184.1 | 230.6 | 21.7 | 21.6 |
| 581,012 | refine | 0.89 | 9.50 | 1.52 | 1.43 |

R's shape-split time at 100,000 rows comes from an earlier sitting. It took 30 minutes, so it wasn't run again, and the full-size runs skip shape splits. `create_dn` is about 10 times faster than before and 8.5 to 9 times faster than R. `refine_dn` beats R at 10,000 and 100,000 rows. On the full data it takes 1.5 seconds to R's 0.9, and [limitations.md](limitations.md) says why. Python 3.14 and 3.15 are within noise of each other, since nearly all the time is spent in compiled kernels, not in the interpreter.

Every run was pinned to one thread on each of the laptop's six performance cores, with Windows power throttling off, by `benchmarks/pin.py`.

Peak memory was 2072, 2092 and 2183 MiB in R at the three sizes, measured in an earlier sitting. In Python it was 1501, 1511 and 1559 MiB before and 887, 896 and 944 MiB at step 10, with 3.15 within 2 MiB of 3.14. That's the peak working set of the whole process, data load included, from `measure.py`. It hardly grows with the row count. The biggest object is the 10,000 by 10,000 distance matrix between the first-stage centers, which is the same size every time. The compiled deletion rounds work on it in place, where the Python code before copied the surviving rows and columns every round.

They ran on the laptop (i7-13650HX, 16 GB, Windows 11). R was 4.6.1 with `datanugget` 1.5.0 and its reference BLAS. Python was 3.14.3 and 3.15.0, each with numpy 2.5.3, scipy 1.18.1 and numba 0.68.0. The timings were committed on 2026-10-09 and R's peak memory on 2026-09-29.

### Covertype: where the time goes

`benchmarks/profile_stop.py` checks the speed work's stopping rule: no function written in Python in `src/datanuggets` may take more than 10% of the run's wall time by cProfile's own time, for `create_dn` and for `refine_dn` without shape splits, on R's replayed draws. cProfile can't see into numba, so the script wraps each kernel in a Python function named `compiled_<kernel>` that books the kernel's time under that name. It reads the outputs of `covertype.R` for the size it's given, which takes the same values as `covertype.py`'s, and runs at `100000` without one:

```sh
uv run python benchmarks/profile_stop.py 100000
```

For each function it prints the wall time and whether the outputs match R, the Python functions and the compiled kernels by their own time, and `stopping rule: PASS` or `FAIL` with the largest share. It exits with 1 if the rule fails, no kernel time was booked, or an output stops matching R's.

The rule passed on the laptop at 100,000 rows, 5.8% for `create_dn` and 3.6% for `refine_dn`, and at full size, 7.0% and 1.1%. Those figures were committed on 2026-10-09.

### WCluster: whclust and wkmeans against R

`whclust` and `wkmeans` ran on the first n rows of R's `quakes` with the weights 1, 3, 7, 2 repeating, `wkmeans` with k = 4, R's defaults otherwise (one start, 10 iterations, seed 291102) and R's draws replayed. These are the workloads behind the speed gates in the [design spec](https://github.com/ron2k1/datanuggets/blob/main/docs/specs/2026-10-09-wcluster-iterative-port-design.md), section 10. The port is checked against R bit for bit: `merge`, `height` and `order` for `whclust`, and the labels, centers, each cluster's WWCSS and the total for `wkmeans`. The port alone also runs `whclust` on the first 2,000 Covertype rows, `wkmeans` with k = 4 and `RDraws(291102)` on the first 10,000, and `dn_whclust` on the 2,000 nuggets R's `create.DN` made of the 100,000-row subset.

R's `Whclust` is timed once, since it takes minutes. R's `Wkmeans` is the best of 3 untraced runs, and one more traced run records its draws. Python takes the best of 3 when a run takes under a second, and one run otherwise. The scripts say to run them under `pin.py`, in one sitting:

```sh
Rscript benchmarks/wcluster.R whclust200
Rscript benchmarks/wcluster.R whclust400
Rscript benchmarks/wcluster.R whclust800
Rscript benchmarks/wcluster.R wkmeans
uv run python benchmarks/wcluster.py
```

`whclust800` is the long one in R: 11 minutes on the M4 mini, and 49 on the laptop when it was short of memory. `wcluster.R` writes each workload to `tmp/wcluster/out/` and prints its seconds. `wcluster.py` also needs the Covertype file and, for its `dn_whclust` row, the `covertype.R 100000` outputs. It prints the first calls' time, a table of R seconds, Python seconds, speedup and whether every output equals R's, then one PASS or FAIL line per gate, and exits with 1 on a FAIL. A missing R output gets a line saying which script to run, and the gates that need it fail. The six gates are `whclust` on quakes under 1 s at 200 and 400 rows, and at 800 rows while equal to R, `wkmeans` on quakes at least 50 times faster than R at each n, `whclust` on 2,000 Covertype rows under 10 s, `wkmeans` on 10,000 Covertype rows under 30 s, and `dn_whclust` on 2,000 nuggets under 10 s. The quakes gates also require every output to equal R's.

On the laptop on 2026-10-20, unpinned and against the R outputs behind the laptop's R column below, it printed this. Its Python times are an unpinned run's, not the recorded ones:

```text
first calls on 10 rows (compile or cache load): 0.25 s

workload                         R s   Python s   speedup  bits
whclust200                    102.21     0.0137    7,441x  yes
whclust400                    534.25     0.0576    9,268x  yes
whclust800                   2950.95     0.2096   14,081x  yes
wkmeans100                      2.42     0.0018    1,338x  yes
wkmeans250                      8.25     0.0064    1,297x  yes
wkmeans500                      4.70     0.0092      510x  yes
wkmeans1000                    10.06     0.0277      363x  yes
whclust covertype 2000       not run     2.6498
wkmeans covertype 10000      not run     5.7025
dn_whclust 2000 nuggets      not run     1.7785

PASS  whclust quakes 200 and 400 under 1 s
PASS  whclust quakes 800 under 1 s, equal to R
PASS  whclust 2,000 Covertype rows under 10 s
PASS  wkmeans quakes 50x faster at each n
PASS  wkmeans 10,000 Covertype rows under 30 s
PASS  dn_whclust 2,000 nuggets under 10 s
```

The recorded seconds, on the laptop and on the Mac mini (Apple M4, macOS arm64, R 4.6.1, Python 3.14.3), committed on 2026-10-10:

| Workload | R, laptop | Python, laptop | R, mini | Python, mini |
| --- | --- | --- | --- | --- |
| `whclust`, 200 rows | 102.2 | 0.0107 | 40.0 | 0.0070 |
| `whclust`, 400 rows | 534.2 | 0.0454 | 166.2 | 0.0278 |
| `whclust`, 800 rows | 2951 | 0.186 | 682.9 | 0.111 |
| `wkmeans`, 100 rows | 2.42 | 0.0017 | 0.82 | 0.0011 |
| `wkmeans`, 250 rows | 8.25 | 0.0057 | 3.00 | 0.0046 |
| `wkmeans`, 500 rows | 4.70 | 0.0081 | 2.03 | 0.0068 |
| `wkmeans`, 1,000 rows | 10.06 | 0.0218 | 4.33 | 0.0189 |
| `whclust`, 2,000 Covertype rows | not run | 1.34 | not run | 0.97 |
| `wkmeans`, 10,000 Covertype rows | not run | 5.00 | not run | 5.38 |
| `dn_whclust`, 2,000 nuggets of 100,000 Covertype rows | not run | 1.49 | not run | 1.03 |

The laptop's R times run high. The laptop had between 0.2 and 1.5 GB free while R ran, so R spent part of the time waiting on memory, and its `Whclust` at 800 rows took 49 minutes where the mini took 11. So the laptop's ratios, 9,500 to 15,800 times for `whclust` and 460 to 1,450 for `wkmeans`, are higher than a quiet machine gives. The mini's, 5,700 to 6,200 and 230 to 730, are the fairer measure. On both machines the times clear every gate's limit: `whclust` under a second at 800 rows and `wkmeans` at least 50 times faster at each n.

On the laptop every output was bit for bit R's. On the mini, R's merges, orders and labels were the port's, but some heights and centers differ in the last bits, under 1e-15 relative. R on arm64 macOS has no 80-bit long double, so its sums round to double at every step, while the port reproduces x86 R's 80-bit sums everywhere. The gates that also ask for R's exact bits fail there for that reason only. R wasn't run on the Covertype rows, whose gates are plain time limits. By its growth from 400 to 800 rows, R's `Whclust` on 2,000 rows would take about 75 minutes on the mini, an estimate that wasn't run.

### Flow: Cabrera's script against its pieces

The flow comparisons run Prof. Cabrera's gate finder, in its Python version ported from the original R code, against `datanuggets.flow`. `benchmarks/flow/` holds the reference functions word for word in `cabrera_reference.py`, the whole reference script in `run_cabrera.py` with its two paths read from the command line and matplotlib drawing on Agg so it runs unattended, nothing else changed, and the reference finder cut at its steps in `cabrera_pieces.py`. [methodology.md](methodology.md) describes the finder and where the port departs from it.

The reference script reads a workbook whose second sheet has a `Filename` column, and the folder those files are in. `prepare.py` writes one for a folder of FCS files, in name order, to `<folder>/study_metadata.xlsx` unless `--out` says otherwise. `--source` joins the columns of the record's metadata sheet on its `file_name` column, and `--limit N` keeps the first N files:

```sh
uv run python benchmarks/flow/prepare.py <folder> --limit 4
```

`check_reference.py` runs the whole reference script with FlowKit and checks that its masks equal the masks of `cabrera_pieces.py` on the arrays the comparison reads, flowio's. The script reads the first four samples in FlowKit's order with the thresholds `t0 = [22, 20, 18, 18]` and `t00 = [0.25, 0.5, 0.5, 0.5]`, so the workbook needs at least four files. FlowKit needs pandas below 3 and the lock has pandas 3, so it runs in a throwaway environment:

```sh
uv run --no-project --with flowkit --with openpyxl --with matplotlib --with flowio \
    python benchmarks/flow/check_reference.py <metadata.xlsx> <folder>
```

For each sample it prints whether flowio's arrays equal FlowKit's and whether the script's masks equal those of the pieces, then how many samples match. It exits with 1 unless all of them do. On the mini it matched on all four of the files it was given, masks and arrays both, on 2026-10-12.

### Flow: compare.py, piece by piece

`compare.py` runs the reference code and ours piece by piece on real FCS files, each piece on the same arrays on both sides, timed as the median of `--repeat` runs. It runs in the repo's environment, whose dev group has flowio and matplotlib:

```sh
uv run python benchmarks/flow/compare.py <file.fcs | folder> --repeat 5
uv run python benchmarks/flow/compare.py --metadata <folder>/study_metadata.xlsx --dir <folder> --t0 22,20,18,18 --t00 0.25,0.5,0.5,0.5
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `target` | none | an FCS file, or a folder whose `*.fcs` files run in name order |
| `--metadata`, `--dir` | none | the reference script's workbook and its folder, in place of `target`: the files of the second sheet's `Filename` column, in the sheet's order |
| `--t0` | 22 for every file | `t11` per file, as `22,20` or `22 20` |
| `--t00` | 0.25 for every file | `t21` per file |
| `--repeat` | 3 | runs per piece, timed as their median |
| `--out` | `tmp/flow` | where the pictures and `summary.csv` go |
| `--no-png` | off | skips the pictures |

It stops with a usage error if it gets both `target` and `--metadata` or `--dir`, or neither a `target` nor both of them, a `--repeat` below 1, or a number of thresholds that isn't one per file. `--t0` and `--t00` pair with the files in the order they run, so with `--metadata` they follow the workbook's rows. The reference script itself pairs its thresholds with the samples in FlowKit's `$FIL` order instead, which [methodology.md](methodology.md) explains.

On Windows, `.\benchmarks\flow\compare.ps1` takes the same arguments, `--metadata` and an unquoted `22,20,18,18` included. It runs `compare.py` in the repo's `.venv`, writes to `tmp\flow`, refuses `--out`, and opens the picture the run wrote. `-Mini` runs it on the Mac mini over SSH and copies the pictures and CSV files back to `tmp\flow\mini`; it depends on the maintainer's SSH setup for that machine.

```powershell
.\benchmarks\flow\compare.ps1 --metadata <folder>/study_metadata.xlsx --dir <folder> --t0 22,20,18,18 --t00 0.25,0.5,0.5,0.5
```

For each file it prints a table like this one, from the mini on `202.fcs`:

```text
202.fcs: 282,270 events, t11=22 t21=0.25, median of 3
  piece             his s     ours s   speedup  match
  1 grid           0.0169     0.0017     10.0x  equal
  2 maxima         0.0071     0.0001     50.5x  equal
  3 peaks          0.0001     0.0001      1.2x  equal, 3 peaks
  4 area           0.0004     0.0001      3.6x  his keeps 586 cells ours doesn't, ours 1485 his doesn't (the disc)
  5 gate           3.1627     0.0436     72.6x  on his cells: equal
  finder his       3.2071     0.0577     55.5x  his find_area: equal
  finder fixed     3.2071     0.0599     53.5x  his 3 populations, ours 3, 107,909 events gated differently (the disc)
  6 graph          0.4674     0.1451      3.2x  his masks both, mean pixel difference 0.05 of 255
```

`5 gate` runs the reference gate and ours on the reference area's cells. `finder his` runs the port's whole finder with the reference `find_area` in place of its own, and `finder fixed` runs it as shipped, both against the reference finder end to end. `6 graph` draws both figures from the reference masks, so the difference is the drawing alone, and times each to the finished pixels.

The picture, `<file>_compare.png`, shows the reference figure, ours from the shipped finder, and the events whose populations differ in red. Each piece's row goes into `summary.csv`, with the columns `file`, `events`, `piece`, `his_s`, `ours_s`, `speedup` and `match`, as each file ends, so a stopped run keeps what it did. A file that fails to run is reported and skipped. Over several files it also prints each piece's median, smallest and largest speedup. It exits with 1 if a file fails to run, or if the grid, the maxima, the peaks, the gate on the reference cells or `finder his` differ, beyond events within 1e-9 of the data's range from the hull's edge.

### Flow: speed over the 44 files

This is `compare.py` on a folder holding all 44 files of the Zenodo record, with `--repeat 5`:

```sh
uv run python benchmarks/flow/compare.py <folder> --repeat 5
```

On the mini, an Apple M4 with 16 GB, over all 44 files: 10,183,290 events, 20,844 to 516,810 a file, each piece timed as the median of 5 runs. Committed on 2026-10-12.

| Piece | Reference, median s | Ours, median s | Speedup, median | Smallest | Largest |
| --- | --- | --- | --- | --- | --- |
| `1 grid` | 0.0163 | 0.0014 | 11.4x | 7.9x | 12.0x |
| `2 maxima` | 0.0077 | 0.0001 | 54.0x | 49.7x | 58.6x |
| `3 peaks` | 0.0001 | 0.0001 | 1.1x | 0.9x | 1.2x |
| `4 area` | 0.0003 | 0.0001 | 3.7x | 3.3x | 3.9x |
| `5 gate` | 2.1350 | 0.0348 | 64.0x | 21.0x | 126.2x |
| `finder his` | 2.1524 | 0.0458 | 49.7x | 18.1x | 93.0x |
| `finder fixed` | 2.1524 | 0.0461 | 47.8x | 20.2x | 84.1x |
| `6 graph` | 0.3797 | 0.1209 | 3.1x | 1.8x | 4.0x |

The grid, the maxima and the peaks equal the reference output on all 44 files. The gate on the reference cells and `finder his` equal the reference output on 40, and on the other 4, `186`, `190`, `278` and `345`, one event each differs, lying exactly on the hull's edge. The reference finder took 100.2 s over the 44 files and ours 1.85 s. `3 peaks` takes a tenth of a millisecond on both sides, so its ratio is noise. The shipped finder, with the disc on the peak, gates 1.4% to 38.7% of a file's events differently from the reference finder, 8.0% at the median. `202.fcs`, at 38.2%, is second only to `193.fcs`.

### Flow: one joined sample

`stress.py` joins every file's events into one sample and runs both finders once on it. Ours runs in the script's own process, as shipped and given the reference `find_area`. The reference finder runs in a child process stopped after `--cap` seconds, since its Delaunay triangulation of every kept event grows with the sample:

```sh
uv run python benchmarks/flow/stress.py <folder>
```

`--cap` defaults to 1800 seconds, 30 minutes, and `--out` to `tmp/flow`. It prints the event count, each run's seconds and populations, how ours given the reference `find_area` compares with the reference run, how many events the shipped finder moves, and the speedup. It appends a row to `stress.csv` in `--out` and deletes the events and masks it saved for the child. It exits with 1 if the reference run fails, or if ours given the reference `find_area` differs from it beyond the hull's edge. A run that hits `--cap` exits 0, since the cap is the measurement.

On the mini, `stress.py` joined the 44 files into one sample of 10,183,290 events. Ours took 1.44 s and the reference finder 1190.0 s, about 20 minutes, so 827x. Ours given the reference `find_area` gave the reference masks exactly, every event, and the shipped finder gated 499,311 of them, 4.9%, differently. The speedup grows with the sample because the reference gate triangulates every kept event, 4,977,743 and 4,880,276 for the two populations here, where ours triangulates the 56 vertices of their hull, and tests each of the 10,000 grid cells once where the reference gate tests every event's bin. Committed on 2026-10-12.
