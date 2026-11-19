# dn_whclust parity with R

This page describes the `dn_whclust` tests in `tests/wcluster/test_dn.py` and their fixtures in `tests/fixtures/dn_wcluster/`. The fixtures were recorded under R version 4.6.1 (2026-06-24 ucrt) and WCluster 1.3.0 on x86_64-w64-mingw32 (BLAS: Rblas.dll md5 5fe8258e86d8a1a52a6c3db05faa0225, LAPACK 3.12.1, matprod "default", long double: 64-bit significand). `scripts/parity_report.py` does not read that folder, so it does not generate this page.

The case `dn` holds the datanugget table that R's DN functions were given. It is a datanugget 1.5.0 create.DN result for the 1000 rows of quakes, with 150 nuggets of 5 variables. Its weights count the rows in each nugget and run from 1 to 59. The tests rebuild it as a `DataNuggets` with its assignments made 0-based.

The case `whclust` holds the merge, height and order of R's `DN.Whclust` on that table. The case `one_variable` holds them for the same table cut to its first variable, where R returns a tree as well. `test_dn_whclust_matches_r` and `test_one_variable_nuggets_stop_where_r_stops` turn R's merge into the port's numbering and R's order into 0-based positions. They require merge and order to be equal integers. They require heights to have the same float64 bits, so -0 isn't 0, except that any NaN matches any NaN (`same_values` in `tests/rfixtures.py`). No height in either tree is NaN.

`dn_whclust` calls `whclust` on the nugget centers with the nugget weights (`dn_whclust` in `src/datanuggets/wcluster/dn.py`). Both trees are also stored as `tree_dn` and `tree_dn_one` in `tests/fixtures/plot/`, where `tests/wcluster/test_cutree.py` cuts them (see [cut_tree](cut_tree.md)).

Only these two tables are recorded for `DN.Whclust`. A nugget of weight zero, signed weights and fewer than three nuggets do not reach R through `DN.Whclust` in the fixtures. `whclust` meets each of them against R on plain data in `tests/fixtures/whclust/` (see [whclust](whclust.md)). A dn that isn't a `DataNuggets` must raise TypeError whose message contains `class "datanugget"` (`test_dn_functions_need_a_datanuggets`). That message is typed into the test, not recorded from R.

| case | nuggets | variables | merges | R's result |
|---|---|---|---|---|
| whclust | 150 | 5 | 149 | returned |
| one_variable | 150 | 1 | 149 | returned |
