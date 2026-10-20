# WCluster iterative port review, 2026-10-10

The range under review is the [WCluster iterative port plan](../plans/2026-10-09-wcluster-iterative-port.md) on `feat/wcluster-iterative` up to 5122e29: `wkmeans`, `whclust`, `dn_wkmeans`, `dn_whclust`, `dn_wpca` and `dn_cluster_predict`, against the [design](../specs/2026-10-09-wcluster-iterative-port-design.md). An addendum covered a0b0bb9 and 903b185, which landed while it ran. Its findings are fixed in d053ba1..aa70ca8.

One reviewer ran: Opus, read-only, on the laptop, Windows with Python 3.14.3. It ran the whclust, wkmeans and DN tests (99 passed, 17 slow ones deselected) and fuzzed the compiled kernels against the Python references, with no mismatches over 400 whclust seeds and 150 wkmeans seeds, wkmeans at `n_jobs` 1 and 3. It didn't run the benchmarks. Its verdict was approve with minors, with no blockers, and the addendum kept it. The findings are about error paths, docs and claims no fixture backed, not about numerical output.

## Findings

| Finding | Severity | Fixed in | Evidence it was real |
|---|---|---|---|
| with `ReplayDraws`, the error `wkmeans` raises depends on `n_jobs` | minor | 2fda14e | see below |
| a numpy `Generator` seed is left in a different state by `n_jobs` when a start stops | nit | 2fda14e, docstring | the reviewer's next draw from seed 0 was 595191111 after `n_jobs=1` and 1002921588 after `n_jobs=3` |
| `to_linkage` says it's for scipy's `dendrogram`, which rejects the negative heights negative weights give | minor | 8546cef, docstring | see the ruling below |
| `math.floor` raises `OverflowError` for an infinite `k` or `num_init` | nit | d053ba1 | "cannot convert float infinity to integer", where the spec promised `ValueError` |
| `n < k` without centers says R draws forever, where R stops | minor, suspected | d053ba1 | see below |
| the centers check uses `k` before R's floor | question | d053ba1 | see below |
| the one-variable datanugget stops and the matrix `newx` stop had no recorded R case | minor, suspected | cbb3c4e | see below |
| the README's 22 minutes for `whclust800` in R is an extrapolation | nit | the benchmark commit | the plan timed R only up to 400 rows |
| R's wkmeans time is one traced run, Python's the best of 3 | nit | the benchmark commit | see below |
| `Draws` gains `sample_replace`, so `as_draws` rejects a hand-written `Draws` without it | nit | README, step 11 | `Draws` is a `runtime_checkable` Protocol, checked for every function |
| `dn_cluster_predict` doesn't check the sign of a hand-built nugget's assignments | nit | not changed | see the ruling below |
| the `_single_py` docstring has a 146-column line | nit, addendum | 2c6a113 | ruff passes it only because E501 isn't selected |
| the pass count is checked against the reference, never against R | nit, addendum | aa70ca8 | see below |

### Replayed draws and threads

With threads, `wkmeans` draws every start before it refines any, then raises the first error in start order. That's R's error, since in R a start's init comes after every earlier start's refinement. But the init loop caught only `ValueError`. `ReplayMismatch` is an `AssertionError`, so when R's replayed draws ran out in start 2 it escaped before start 1's refinement could raise R's error. The reviewer built a case with R's draws for a run that stops in start 1: `n_jobs=1` raised R's "a WWCSS is NaN" and `n_jobs=3` raised "draw #3 sample.int(2, 10, replace = TRUE) requested, but R made only 2 draws". `RDraws` and the numpy draws never run out, so only replays hit it.

The init loop now holds back `ReplayMismatch` too. `test_replayed_draws_give_rs_error_for_any_n_jobs` replays R's recorded `nan_stop` draws with three starts and expects R's message at `n_jobs` 1 and 3. It failed at 3 before the fix.

### Arguments R stops on

Three findings were about arguments, and all three turned on what R does, which the reviewer couldn't run. A probe in R 4.6.1 settled them, and `fixtures_wkmeans.R` now records 28 `Wkmeans` runs as the `wkmeans_args` case, with R's message where it stops and its labels where it returns. What they showed:

- R checks `cl.centers`' shape against `k` before it floors `k`. So `k = 2.5` with two centers stops, and the port's order was already R's. A vector of centers stops with "argument is of length zero", since R's `nrow()` of a vector is `NULL`.
- `n < k` stops in `sample(k, n - k, replace = TRUE)` with "invalid 'size' argument". Only `k <= n < 2k` draws forever. The spec's line that every `n < 2k` hangs was wrong.
- An infinite `num.init` or `max.iterations` stops in `1:v` with "result would be too long a vector", a NaN one with "NA/NaN argument". Without centers, an infinite `k` stops the same way in `init`, and a NaN `k` with "missing value where TRUE/FALSE needed".
- A `k` below 1 after the floor makes `init` return `NULL`, and R stops later in `wwcss` with "The length of cluster assignments should be the same as the number of observations.", the message the port already raised when every row's distances are NaN. So `k < 1` is R's own stop, no longer a departure.

`_prepare` now raises R's message in R's order for each of these. `test_arguments_stop_where_r_stops` runs all 28 rows through the Python reference and the compiled path. The port's own "k must be at least 1" is gone, with its test row. The two departures that remain, `num_init` or `max_iter` below 1 and the draw-forever cases, are checked only after R's own stops.

The probes also turned up a bug the reviewer hadn't listed. Given centers, R sets `num.init` to 1 before the `1:num.init` loop, so a NaN or infinite `num_init` with centers runs. The port had raised on them. It now matches.

### One variable and a matrix newx

The README said the `dn_` functions stop where R's do on a datanugget with one variable, and that R stops on a matrix `newx`, but those rested on a reading of R's source. The new `one_variable` case in `fixtures_dn_wcluster.R` records them: "argument is of length zero" from `DN.Wkmeans` and from `DNcluster.predict` with a datanugget `newx`, "invalid 'times' argument" from `DN.Wpca` without `wcol`, and "newdata should be either a data frame or a datanugget object." for a matrix `newx`. It also records what R returns where it runs: `DN.Whclust`, `DN.Wpca` with `wcol = 1`, and `DNcluster.predict` on a one-column data frame. The port already did all of it, so these tests passed on arrival. They pin it now.

### Benchmark timing

`wcluster.R` took R's wkmeans time from one run with `sample.int` traced, while `wcluster.py` takes the best of 3 for a run under a second. `Wkmeans` sets the seed itself, so every run is the same. R's time is now the best of 3 untraced runs, and a fourth, untimed run traces the draws, with `identical()` checking it returned what the timed runs did. An unpinned smoke run still put R's wkmeans 366 to 1,034 times behind the port, against a gate of 50.

### Pass counts against R

a0b0bb9 made the tests compare each start's pass count between the kernel and the reference, which killed the one mutant that had survived. The addendum found the check sound: on the test's own fuzz the count ran from 1 to the cap of 4, so a mutant that adds or drops a pass changes what it compares. But the reference copies R's loop, so it agrees with R by construction, and no fixture held R's count.

R prints the count. `Wkmeans_single` calls `message("Iteration l")` at the top of every pass, and the port logs each start's count at debug level in its place. `fixtures_wkmeans.R` now records those messages for the 15 cases, through a `withCallingHandlers` that sees each one before `run_quiet`'s `suppressMessages` drops it. Fourteen cases print some, from 1 to 4 passes a start. `ucenters_empty` stops before the loop. The tests split the messages into starts and compare them with the trace on both paths, and with the debug log on the public one. The `lloyd_break_drops_yy0` mutant now fails seven R parity cases as well as the fuzz, including `k1` and `one_col`, where R took one pass and the mutant adds one that changes nothing.

### Ruling on negative heights in `to_linkage`

The reviewer offered a docstring or a `ValueError` like the self-merge one. The docstring was taken. A self-merge is a row scipy's format can't hold, but a negative height fits it fine, and it's R's own number. Raising would hide it from a caller who reads the linkage without plotting it. A probe checked the docstring's claims: weights 1, 1, 1, 1, -3 on five rows gave heights 0.5, 1, 51 and -408, which `dendrogram`, `fcluster` and `is_valid_linkage` each rejected. Weights with a zero gave NaN heights, which all three took without a word, since scipy only checks `Z[:, 2] < 0`.

### Ruling on assignment signs

Not changed. `create_dn` never gives a negative assignment, and `DataNuggets` doesn't validate assignments for any function, so a check here would guard one function out of all that read them. If it's wanted, it belongs in `DataNuggets` itself.

## Checked and found correct

The reviewer traced each of these against the code it replaces:

- The whclust kernel takes the column-major first minimum with strict `<`, skipping NaN. A new cluster takes over a column only when strictly smaller, and a column is searched again when its row minimum was one of the two merged clusters.
- The infinite pick happens only when no finite candidate is left. It takes a merged column's row 1 or a singleton column's first `+Inf`, detects a self-merge only when the merged column at position 0 is picked, and then falls back to `_whclust_py`. No candidate gives R's "argument is of length zero".
- whclust's merge rows are `(label[ru], label[cv])` as in R. Positions run singletons in reverse row order, then merged clusters in the order they formed. Heights are the long double sums in rank order.
- wkmeans' Lloyd loop breaks on a revert or no change, with `yy0` equal to R's state at the start of step 20. Dead clusters are checked before NaN, where R stops in `wwcss`.
- The move loop reads the stale `tb` by position, raises on R's `FALSE & NA` after the `a == j` check, skips `tb[a] <= 1`, keeps a move with `<=`, and drops an emptied label and includes a recreated one in label order.
- The replace draw comes before the permutation, and `ReplayDraws` checks the kind. `MAX_STARTS` raises its error, and a user-centers start with an empty cluster stops. The best start is picked with strict `>` in start order.
- `refine` allocates its own buffers and copies `y`, and the shared arrays are read-only, so the result doesn't depend on `n_jobs`, which the fuzz confirmed.
- `wcluster/` doesn't import `nuggets`, `_clustering` or `_defaults`.
- The README's case counts, fuzz claims and quirk counts matched the tests and manifests, and the benchmark gates match the design's section 10.
- In the addendum: the kernel, the reference and R's counted copy all count a pass at the top of the loop, and the kernel returns the count on every exit, so an early break and a full run count alike.
- The README example's scipy numbering matches `WhclustResult`: with 200 nuggets, the new clusters are 200 to 398. Its `to_linkage` and `dendrogram` sentence holds there, since nugget weights are positive counts. The reviewer didn't rerun the example's output, which needs the 100,000-row `create_dn` run.

## Test runs

At aa70ca8, after the fixes, with numpy 2.5.3, scipy 1.18.1 and numba 0.68.0:

- Laptop, Windows: 2661 passed and 2 skipped (no `os.fork` on Windows) on Python 3.14.3 and on 3.15.0. The slow tier, 157 tests, passed on both.
- Mac mini, arm64: 2663 passed on 3.14 and on 3.15, and the slow tier passed on both.
- Mutants: all five of the plan's mutants were killed at a0b0bb9. At aa70ca8, `lloyd_break_drops_yy0` fails seven R parity cases as well as the fuzz.
- The benchmarks run in their own pinned sitting. The gate table is in the commit that adds them.
