# cut_tree parity with R

This page describes `tests/wcluster/test_cutree.py` and the fixtures in `tests/fixtures/plot/`. The fixtures were recorded under R version 4.6.1 (2026-06-24 ucrt) and WCluster 1.3.0 on x86_64-w64-mingw32 (BLAS: Rblas.dll md5 5fe8258e86d8a1a52a6c3db05faa0225, LAPACK 3.12.1, matprod "default", long double: 64-bit significand). cut_tree has no fixture folder of its own. R's `cutree` results are stored with the plotting fixtures. `scripts/parity_report.py` does not read that folder, so it does not generate this page. The R function is `stats::cutree` as R 4.6.1 runs it. The trees it cut are WCluster 1.3.0 `Whclust` trees.

Each `tree_` case holds one tree's merge, height and order with R's cutree labels for it. Their merge, height and order are the same arrays as in the returned cases of `tests/fixtures/whclust/` and in the `whclust` and `one_variable` cases of `tests/fixtures/dn_wcluster/`. The tests build each tree from these recorded arrays, so the cuts are checked on R's own trees and don't depend on `whclust`. `tree_hang` holds no cuts and is left out.

`test_every_k_matches_r` cuts every tree at every k from 1 to n and compares with R's labels for that k. `test_every_h_matches_r` cuts at every recorded height and compares with R's labels for that height. The recorded heights are each distinct merge height, the midpoint of each pair of neighbouring distinct heights, 1 below the lowest and 1 above the highest, and plus and minus infinity. That makes 940 cuts at k and 1,729 at h over the 13 trees. Labels must be equal integers, the port's being R's minus one. On `tree_zero_w1` and `tree_zero_w2` 28 of the 29 heights are NaN, and R stops when cutting by height with "missing value where TRUE/FALSE needed". The test cuts both at h = 1 and requires exactly that message.

The trees include self-merges (`tree_selfmerge_last` and `tree_selfmerge_mid`) and infinite heights (`tree_inf_single` and `tree_selfmerge_mid`). `tree_selfmerge_last` has a negative height. Heights repeat in `tree_grid_dup`, `tree_inf_single`, `tree_one_col` and both self-merge trees.

`tests/fixtures/plot/stops/` holds the 3-leaf tree that `tree_n3` also holds and R's messages for the calls where R stops, plotting calls included. `test_stops_match_r` requires each cutree message exactly. The calls give no k or h, a k of 0, 4, NaN, Inf or 3e9, or an h of NaN. Three more cut a tree with a NaN height, a tree with unsorted heights and a tree with an empty merge. The test builds the NaN-height and unsorted trees from the recorded one and the empty tree from empty arrays. The manifest stores R's message for each of these three but not the tree R was given. `test_k_truncates_like_as_integer` cuts the recorded tree at k = 2.9 and compares with R's labels for that call.

`test_k_wins_over_h` compares k = 1 with h = -1 against labels typed into the test, not recorded from R. cut_tree takes one k or one h, and an array raises ValueError (`test_one_cut_per_call`). No vector k or h was recorded from R. `test_numpy_loop_matches_c_on_fuzzed_trees` holds the numpy loop `_cutree` to `c_cutree`, a line-by-line Python copy of R's C routine kept in the test file, and not to R. It draws 300 random trees of 2 to 39 leaves without self-merges and 300 that allow them, and compares every k. `test_r_merge_inverts_the_port_numbering` checks on `tree_dn` that `r_merge` gives back R's merge.

| case | tree from | leaves | k cut | merge heights | heights cut | R's result by height |
|---|---|---|---|---|---|---|
| tree_dn | dn_wcluster/whclust | 150 | 1 to 150 | 149, all distinct | 301 | labels |
| tree_dn_one | dn_wcluster/one_variable | 150 | 1 to 150 | 149, all distinct | 301 | labels |
| tree_grid_dup | whclust/grid_dup | 24 | 1 to 24 | 23, 12 distinct | 27 | labels |
| tree_inf_single | whclust/inf_single | 3 | 1 to 3 | 2, both Inf | 5 | labels |
| tree_iris_unit | whclust/iris_unit | 150 | 1 to 150 | 149, all distinct | 301 | labels |
| tree_iris_w1372 | whclust/iris_w1372 | 150 | 1 to 150 | 149, all distinct | 301 | labels |
| tree_n3 | whclust/n3 | 3 | 1 to 3 | 2, all distinct | 7 | labels |
| tree_one_col | whclust/one_col | 40 | 1 to 40 | 39, 32 distinct | 67 | labels |
| tree_quakes200_w1372 | whclust/quakes200_w1372 | 200 | 1 to 200 | 199, all distinct | 401 | labels |
| tree_selfmerge_last | whclust/selfmerge_last | 4 | 1 to 4 | 3, 2 distinct, one negative | 7 | labels |
| tree_selfmerge_mid | whclust/selfmerge_mid | 6 | 1 to 6 | 5, 4 distinct, one Inf | 11 | labels |
| tree_zero_w1 | whclust/zero_w1 | 30 | 1 to 30 | 29, 28 NaN | none | stops: "missing value where TRUE/FALSE needed" |
| tree_zero_w2 | whclust/zero_w2 | 30 | 1 to 30 | 29, 28 NaN | none | stops: "missing value where TRUE/FALSE needed" |
