# wkmeans parity with R

This page describes `tests/wcluster/test_wkmeans.py` and the fixtures in `tests/fixtures/wkmeans/` and `tests/fixtures/wkmeans_args/`. The fixtures were recorded under R version 4.6.1 (2026-06-24 ucrt) and WCluster 1.3.0 on x86_64-w64-mingw32 (BLAS: Rblas.dll md5 5fe8258e86d8a1a52a6c3db05faa0225, LAPACK 3.12.1, matprod "default", long double: 64-bit significand). `scripts/parity_report.py` reads neither folder, so it does not generate this page.

Each case in `tests/fixtures/wkmeans/` holds one call of R's `Wkmeans` with its arguments and the random draws R made. It also holds what R returned or the message R stopped with. The tests replay R's draws (`ReplayDraws`) and fail unless every recorded draw is used. Where R returns, they compare three things for each start: its initial labels, its labels and WWCSS after refinement, and the number of passes it made. R sends one "Iteration l" message per pass. Then they compare the returned labels, centers, per-cluster WWCSS and total WWCSS. Labels and pass counts must be equal integers. The returned labels are R's minus one. Centers and WWCSS values must have the same float64 bits, so -0 isn't 0, except that any NaN matches any NaN (`same_values` in `tests/rfixtures.py`). Where R stops, the port must raise ValueError with R's message inside its own, and the draws must again be used up.

Two implementations are held to R this way. One is `wkmeans`, whose refinement runs in compiled code. The other is `_wkmeans_py` in `tests/wcluster/wkmeans_reference.py`, R's `Wkmeans` and `Wkmeans_single` statement for statement in plain Python. It shares the package's argument checks, init and result (`_prepare`, `_init` and `_result` in `src/datanuggets/wcluster/wkmeans.py`). `wkmeans` is checked on every case by default. The reference is checked on the cases of at most 160 rows by default. On the two larger cases, `one_col` and `quakes500_k4_w`, it runs only in the slow suite (`pytest -m slow`). `pyproject.toml` deselects that suite by default. The reference also counts how often each run meets each of R's quirks, and every count must equal the one in the case's manifest.

The quirks column lists the counters above zero in each manifest. `redraw` counts the extra partitions init draws after the first. `revert` counts Lloyd steps that raised the WWCSS and were undone. `fixed` counts Lloyd steps before the 20th that changed no label. `stale_decides` counts checks where R's stale table of cluster sizes and the current labels disagree on whether a row's cluster has more than one row. `recreate` counts moves that refilled an empty label. `gap` means the final labels skip a label. `tb_na` and `nan_test` count two of the places where R stops in `Wkmeans_single`: a label past the end of the stale table, and a comparison with a NaN WWCSS.

The 12 x 2 cases are the integer grid with x from 0 to 3 and y from 0 to 2. The 150 x 4 cases are the four measurements of iris. `small_k4` is iris rows 1 to 3, 51 to 53 and 101 to 103. `one_col` is the 272 eruption times of faithful. `quakes500_k4_w` is the first 500 rows of quakes, and `revert` is the first two columns of its first 60 rows. `ucenters` and `ucenters_empty` pass three centers and draw nothing.

`tests/fixtures/wkmeans_args/table/` records R on 28 argument combinations. Each runs on the first n rows of the same grid, with n of 1, 3, 11 or 12. The combinations set k, num_init or max_iter to values such as 0, 0.5, NaN, Inf and -Inf. Some give fewer rows than k. Five of them pass centers as a matrix and one as a vector. R stops on 26 combinations. It returns labels on the other 2, which pass a centers matrix with num_init NaN or Inf. `test_arguments_stop_where_r_stops` runs every combination through both `wkmeans` and the reference with `RDraws(291102)`. It requires the message to contain `(R stops: ` followed by R's message, or the labels to equal R's.

`RDraws` makes R's draws from a seed instead of replaying recorded ones. Its labels are compared with R's on `iris_k3_init5` (`test_rdraws_seed_matches_r`) and on `gap` and `iris_k3` (`test_clusters_on_a_wkmeans_result` in `tests/plot/test_plot_points.py`), each at its recorded seed.

An int or a numpy Generator as seed draws from numpy. Those draws are not R's, and `test_int_seed_is_reproducible` checks only that one int seed gives the same result twice.

The compiled path is held to the reference, not to R, on fuzzed data. `test_kernel_matches_reference_on_tie_heavy_data` runs 25 seeds in each of six families: unit, integer, zero-containing, signed and mostly negative weights, and data multiplied by 1e153. The data are small integer matrices, the draws come from `RDraws`, and each run makes three starts with `n_jobs` 1 and 3. Labels, pass counts and error messages must be equal, and centers and WWCSS must pass `same_values`. Both paths run with the draw cap lowered to 100 there. The slow suite adds 600 seeds per family (`test_kernel_matches_reference_on_a_large_fuzz`). `test_fuzz_reaches_every_quirk` checks that the signed and mostly negative runs for seeds 0 to 9, together with three grid inputs chosen to return a gap, meet every quirk.

Where R would keep drawing forever, R has no result to record and the port raises ValueError. Without centers, this covers at least k but fewer than 2k rows, fewer than k distinct (row, weight) pairs, and a start whose 1,000 random partitions (`MAX_DRAWS` in `src/datanuggets/wcluster/wkmeans.py`) all fail. These are tested on constructed inputs only, in `test_hang_guards_raise_at_their_boundary`, `test_init_gives_up_where_r_draws_forever` and the tests of the draw cap. On 16 one-column rows with k = 8 and `RDraws(1)`, the first start needs 2,179 draws when uncapped. The capped port raises there (`test_the_cap_allows_exactly_max_draws`).

A string or bool k raises TypeError. Those messages are typed into `test_arguments_are_checked_as_r_checks_them` and not recorded from R. `test_one_row_follows_rs_diag` runs the reference alone on a single row with user centers, where R's `wdist_mat` calls `diag(w)`. Its expected labels and messages are typed into the test and are not stored as a fixture. The messages are those of R 4.6.1 with WCluster 1.3.0. The threaded path is compared with `n_jobs=1` and not with R's records: `test_n_jobs_does_not_change_the_result` runs `iris_k3_init5` with `n_jobs` 1, 4 and -1. `test_replayed_draws_give_rs_error_for_any_n_jobs` reruns `nan_stop` with three starts for `n_jobs` 1 and 3 and expects R's error. No recorded case has a zero weight, and only `iris_k3_init5` makes more than one start.

| case | data | weights | k | num_init | max_iter | seed | quirks met | R's result |
|---|---|---|---|---|---|---|---|---|
| gap | 12 x 2 | -0.73, 1.19, 2.41 | 3 | 1 | 4 | 195 | revert, gap | returned, label 2 empty |
| iris_k3 | 150 x 4 | ones | 3 | 1 | 10 | 291102 | fixed | returned |
| iris_k3_init5 | 150 x 4 | ones | 3 | 5 | 10 | 291102 | redraw, fixed | returned |
| k1 | 150 x 4 | ones | 1 | 1 | 10 | 291102 | fixed | returned |
| maxiter1 | 150 x 4 | ones | 3 | 1 | 1 | 291102 | fixed | returned |
| nan_stop | 12 x 2 | -1, 1 | 2 | 1 | 4 | 1 | revert, nan_test | stops: "missing value where TRUE/FALSE needed" |
| one_col | 272 x 1 | ones | 2 | 1 | 10 | 291102 | fixed | returned |
| quakes500_k4_w | 500 x 5 | 1, 3, 7, 2 repeated | 4 | 1 | 10 | 291102 | redraw, fixed | returned |
| recreate | 12 x 2 | -0.73, 1.19, 2.41 | 3 | 1 | 4 | 4 | revert, fixed, recreate | returned |
| revert | 60 x 2 | -0.73, 1.19, 2.41 | 3 | 1 | 4 | 1 | revert | returned |
| small_k4 | 9 x 4 | ones | 4 | 1 | 10 | 291102 | redraw, fixed | returned |
| stale_tb | 12 x 2 | -0.73, 1.19, 2.41 | 3 | 1 | 4 | 15 | revert, stale_decides | returned |
| tb_na | 12 x 2 | -0.73, 1.19, 2.41 | 3 | 1 | 4 | 65 | revert, stale_decides, tb_na | stops: "missing value where TRUE/FALSE needed" |
| ucenters | 150 x 4 | ones | 3 | 1 | 10 | 291102 | fixed | returned |
| ucenters_empty | 150 x 4 | ones | 3 | 1 | 10 | 291102 | none | stops: "For the initialization, no data assigned to one(or multiple) of the clusters ..." |
