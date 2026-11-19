# dn_wkmeans parity with R

This page describes the `dn_wkmeans` tests in `tests/wcluster/test_dn.py` and their fixtures in `tests/fixtures/dn_wcluster/`. The fixtures were recorded under R version 4.6.1 (2026-06-24 ucrt) and WCluster 1.3.0 on x86_64-w64-mingw32 (BLAS: Rblas.dll md5 5fe8258e86d8a1a52a6c3db05faa0225, LAPACK 3.12.1, matprod "default", long double: 64-bit significand). `scripts/parity_report.py` does not read that folder, so it does not generate this page.

The case `dn` holds the datanugget table that R's DN functions were given. It is a datanugget 1.5.0 create.DN result for the 1000 rows of quakes, with 150 nuggets of 5 variables whose weights count the rows in each nugget. The tests rebuild it as a `DataNuggets` with its assignments made 0-based.

The case `wkmeans` is R's `DN.Wkmeans` on that table with k = 3, num_init = 1, max_iter = 10 and seed 291102. `test_dn_wkmeans_matches_r` replays R's draws (`ReplayDraws`) and fails unless every recorded draw is used. It compares the labels of the 150 nuggets and the labels of the 1000 original rows (`row_labels`) as integers, the port's being R's minus one. It compares the centers, the per-cluster WWCSS and the total WWCSS bit for bit. Bit for bit means the same float64 bits, so -0 isn't 0, except that any NaN matches any NaN (`same_values` in `tests/rfixtures.py`).

The case `one_variable` holds R's results on the same table cut to its first variable. R's `DN.Wkmeans` stops there with "argument is of length zero". `test_one_variable_nuggets_stop_where_r_stops` requires `dn_wkmeans` to raise ValueError with that message inside its own.

The recorded call that returns makes one start with random centers. User centers and more than one start are not recorded for `DN.Wkmeans`, and `n_jobs` has no R counterpart. `dn_wkmeans` calls `wkmeans` on the nugget centers with the nugget weights and passes those arguments through (`dn_wkmeans` in `src/datanuggets/wcluster/dn.py`), so their checks against R are the ones on the [wkmeans](wkmeans.md) page. No test runs `dn_wkmeans` with `RDraws`. The case records no per-start labels, so only the final result is compared. Every nugget in the table has a weight of at least 1, so a nugget of weight zero is not recorded.

Three checks are not against R. A dn that isn't a `DataNuggets` must raise TypeError whose message contains `class "datanugget"` (`test_dn_functions_need_a_datanuggets`). A call in R's positional order must raise TypeError (`test_r_positional_order_fails`). `test_default_seed_is_the_one_in_defaults` in `tests/wcluster/test_wkmeans.py` checks that the default seed is `SEED` from `src/datanuggets/_defaults.py`.

| case | nuggets | variables | original rows | R's arguments | R's result |
|---|---|---|---|---|---|
| dn | 150 | 5 | 1000 | the input table | none |
| wkmeans | 150 | 5 | 1000 | k = 3, num_init = 1, max_iter = 10, seed = 291102 | returned, 3 clusters |
| one_variable | 150 | 1 | 1000 | not recorded | stops: "argument is of length zero" |
