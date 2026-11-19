# whclust parity with R

This page describes `tests/wcluster/test_whclust.py` and the fixtures in `tests/fixtures/whclust/`. The fixtures were recorded under R version 4.6.1 (2026-06-24 ucrt) and WCluster 1.3.0 on x86_64-w64-mingw32 (BLAS: Rblas.dll md5 5fe8258e86d8a1a52a6c3db05faa0225, LAPACK 3.12.1, matprod "default", long double: 64-bit significand). `scripts/parity_report.py` does not read that folder, so it does not generate this page.

Each case holds the data and weights given to R's `Whclust` and the merge, height and order R returned, or the message R stopped with. The tests turn R's merge into the port's numbering, where R's -i becomes row i - 1 and R's step s becomes n + s - 1. They turn R's 1-based order into 0-based positions. Merge and order must then be equal integers. Heights must have the same float64 bits, so -0 isn't 0, except that any NaN matches any NaN (`same_values` in `tests/rfixtures.py`). Where R stops, the port must raise ValueError with R's message inside its own.

Two implementations are held to R this way. One is `whclust`, which runs a compiled kernel. The other is `_whclust_py` in `src/datanuggets/wcluster/whclust.py`, R's function statement for statement. `whclust` is checked on every case by default. The reference is checked on the cases of at most 40 rows by default. On the three larger cases it runs only in the slow suite (`pytest -m slow`), which `pyproject.toml` deselects by default. The reference also counts the steps where R's which.min picks an infinite distance (`inf_pick`) and where R merges a cluster with itself (`self_merge`). `self_merge_mid` counts the self-merges before the last step. Every count must equal the one in the case's manifest.

Signed weights (`selfmerge_last`) or values near plus and minus 1e160 (`selfmerge_mid`) make R merge a cluster with itself. The kernel hands such a run to `_whclust_py` (the `SELF_MERGE` branch of `whclust` in `src/datanuggets/wcluster/whclust.py`), so on these two cases both checks run the same Python code. R's order then repeats rows or drops them. R's order for `selfmerge_last` is 3, 4, 3, 4 for 4 rows. R's order for `selfmerge_mid` has 4 entries for 6 rows. The port returns the same. On `zero_w1` and `zero_w2` a zero weight gives NaN distances, and 28 of R's 29 heights are NaN. `same_values` lets any NaN match any NaN there, so NaN payload bits are not compared. Both heights of `inf_single` are Inf, one height of `selfmerge_mid` is Inf, and one height of `selfmerge_last` is negative.

The 4-column cases are rows of iris. `iris_unit` and `iris_w1372` take all 150 rows. `n1`, `n2` and `no_candidate` take the first 1, 2 and 5 rows. The zero-weight cases take the first 30. `n3` takes rows 1, 51 and 101. `grid_dup` is the integer grid with x from 1 to 4 and y from 1 to 3, every point twice. `one_col` is the first 40 eruption times of faithful, and `quakes200_w1372` is the first 200 rows of quakes. `inf_single` holds 0, 1e160 and -1e160. `selfmerge_last` holds 0, 0.1, 5 and 5.1. `selfmerge_mid` holds 0, 1, two values near 1e160 and two near -1e160.

The kernel is held to the reference, not to R, on fuzzed data. `test_kernel_matches_reference_on_tie_heavy_data` runs 40 seeds in each of six families: unit, integer, zero-containing and signed weights, duplicated rows, and data multiplied by 1e154 with added noise. Each input has 3 to 13 rows of 1 to 3 columns. Merge and order must be equal, and heights must pass `same_values`. Where the reference raises, `whclust` must raise a ValueError whose message contains the reference's. The slow suite adds 1,000 seeds per family (`test_kernel_matches_reference_on_a_large_fuzz`). `test_fuzz_reaches_the_inf_pick_and_the_self_merge` checks that the zero, signed and scaled runs meet both quirks.

The recorded weights are ones, zeros on one or two rows, and all zeros. Two cases repeat the weights 1, 3, 7 and 2. `selfmerge_last` has 1, 1, -1 and -1. Other weights appear only in the fuzz. No recorded case has more than 200 rows. `test_whclust_takes_any_layout_and_dtype` compares the port with itself across memory layouts and dtypes. `test_to_linkage_drives_scipy_dendrogram_in_r_order` checks scipy's dendrogram of `to_linkage()` against the port's order on `iris_w1372`, which the R comparison above already holds to R's order.

| case | data | weights | merges | quirks met | R's result |
|---|---|---|---|---|---|
| grid_dup | 24 x 2 | ones | 23 | none | returned |
| inf_single | 3 x 1 | ones | 2 | inf_pick (2) | returned, both heights Inf |
| iris_unit | 150 x 4 | ones | 149 | none | returned |
| iris_w1372 | 150 x 4 | 1, 3, 7, 2 repeated | 149 | none | returned |
| n1 | 1 x 4 | ones | none | none | stops: "missing value where TRUE/FALSE needed" |
| n2 | 2 x 4 | ones | none | none | stops: "subscript out of bounds" |
| n3 | 3 x 4 | ones | 2 | none | returned |
| no_candidate | 5 x 4 | all zero | none | none | stops: "argument is of length zero" |
| one_col | 40 x 1 | ones | 39 | none | returned |
| quakes200_w1372 | 200 x 5 | 1, 3, 7, 2 repeated | 199 | none | returned |
| selfmerge_last | 4 x 1 | 1, 1, -1, -1 | 3 | inf_pick, self_merge | returned, last step a self-merge |
| selfmerge_mid | 6 x 1 | ones | 5 | inf_pick, self_merge, self_merge_mid | returned, order of 4 entries |
| zero_w1 | 30 x 4 | row 7 zero | 29 | inf_pick | returned, 28 heights NaN |
| zero_w2 | 30 x 4 | rows 7 and 19 zero | 29 | inf_pick (2) | returned, 28 heights NaN |
