# Weighted Ward clustering

Draft. This page has not yet been reviewed by the authors of the R packages.

WCluster's `Whclust` is agglomerative clustering with Ward's criterion and row weights. Every row starts as its own cluster. Each step merges the two clusters whose merge adds the least to the weighted within-cluster sum of squares (WWCSS). `DN.Whclust` runs it on the centers and weights of data nuggets. The port returns R's merges, heights and order, including R's pick among tied costs. Every height that is not NaN equals R's bit for bit. Beavers et al. (2025) do not describe weighted Ward clustering. Their Section 2 notes that hierarchical clustering needs a distance matrix that cannot be formed for very large data. The WCluster 1.3.0 help pages of `distw` and `DN.Whclust` cite Ward (1963) for the criterion.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `whclust(x, w=None)` | `src/datanuggets/wcluster/whclust.py`, `whclust` and `_whclust_py`; `src/datanuggets/wcluster/whclustjit.py`, `whclust` | WCluster 1.3.0, `Whclust` and `forder` |
| `dn_whclust(dn)` | `src/datanuggets/wcluster/dn.py`, `dn_whclust` | WCluster 1.3.0, `DN.Whclust` |
| `cut_tree(tree, k=None, h=None)` | `src/datanuggets/wcluster/cutree.py`, `cut_tree` and `_cutree` | R 4.6.1, `stats::cutree`, which calls `src/library/stats/src/hclust-utils.c`, `cutree` |
| `WhclustResult.to_linkage()` | `src/datanuggets/result.py`, `WhclustResult.to_linkage` | none: it builds a SciPy linkage matrix |

`cut_tree` takes one `k` or one `h` per call. R's `cutree` also takes a vector and returns one column of labels per value.

## Notation

The symbols these pages share are in [README.md](README.md#notation). $`\mathrm{SS}`$, $`W_S`$, the merge cost $`d(A, B)`$ and $`\mathrm{ldsum}`$ are defined in [weighted-statistics.md](weighted-statistics.md). This page adds the following.

- $`\mathcal{C}_s`$ is the clustering after merge $`s`$, for $`s = 0, \dots, n - 1`$. $`\mathcal{C}_0`$ has one cluster per row.
- $`d_s`$ is the cost of merge $`s`$, and $`h_s`$ is its height.
- A hat marks a computed value, as in [weighted-statistics.md](weighted-statistics.md). $`\hat{h}_s`$ is R's computed height, `ws2[s]` of Step 7.
- R labels row $`i`$ as $`-i`$ and the cluster formed by merge $`s`$ as $`s`$.
- A cluster's position is the rank of its label among the labels left, in increasing order. Singletons come first in decreasing row order, then merged clusters in the order they formed.
- R's matrix `dd` holds the merge costs between positions. It is $`r \times r`$ when $`r`$ clusters are left.

## Definitions

D1. Merge sequence. $`\hat{d}(A, B)`$ is the cost of two clusters $`A`$ and $`B`$ as Step 7 of [weighted-statistics.md](weighted-statistics.md) computes it. When every computed cost is finite and no earlier merge has joined a cluster with itself, $`\mathcal{C}_s`$ comes from $`\mathcal{C}_{s-1}`$ by merging the two distinct clusters $`A`$ and $`B`$ with the smallest $`\hat{d}(A, B)`$. Step 4 breaks ties. When merge $`s`$ joins two distinct clusters $`A`$ and $`B`$, $`d_s = d(A, B)`$ is its exact cost. Rounding can make the merged pair differ from the pair with the smallest exact cost.

D2. Height. $`h_s = \mathrm{WWCSS}(\mathcal{C}_s)`$, the total after merge $`s`$. It is not $`d_s`$.

D3. Merge rows. Row $`s`$ of R's `merge` holds the labels $`(u_s, v_s)`$ of the two clusters that merge $`s`$ joins.

D4. Order. R's `forder` builds one list of rows per merge. A label $`-i`$ contributes the row $`i`$, and a label $`t \gt 0`$ contributes the list $`L_t`$. $`L_s`$ is the contribution of $`u_s`$ followed by the contribution of $`v_s`$, and the order is $`L_{n-1}`$.

D5. Cut into $`k`$ clusters. R's `cutree` applies merges 1 to $`n - k`$ and numbers the clusters it then has from 1, in the order of their smallest row. A row still alone is a cluster of its own. A merge that joins a cluster with itself leaves the clusters unchanged. So the cut has $`k`$ clusters plus one for each such merge among merges 1 to $`n - k`$ (`selfmerge_last` under Numerical notes).

D6. Cut at a height $`h`$. Let $`q`$ be the smallest index with $`\hat{h}_q \gt h`$ in $`(\hat{h}_1, \dots, \hat{h}_{n-1}, +\infty)`$, or $`q = 1`$ when no entry exceeds $`h`$. The cut at $`h`$ is the cut into $`k = n + 1 - q`$ clusters.

P1. Heights add the merge costs. If no merge joins a cluster with itself and every cluster that appears has nonzero total weight, then $`h_s = d_1 + \dots + d_s`$. Proof. A row $`i`$ with $`w_i \neq 0`$ is its own weighted mean, so its $`\mathrm{SS}`$ is 0 and $`\mathrm{WWCSS}(\mathcal{C}_0) = 0`$. P2 of [weighted-statistics.md](weighted-statistics.md) adds $`d_t`$ at merge $`t`$.

P2. With positive weights and no merge that joins a cluster with itself, the heights never decrease. Each $`d_t \ge 0`$ by D6 of [weighted-statistics.md](weighted-statistics.md), so P1 gives $`h_1 \le h_2 \le \dots \le h_{n-1}`$. P1 and P2 hold in exact arithmetic. The code adds the computed values $`\widehat{\mathrm{SS}}`$ of the clusters, which can carry rounding errors. A singleton's computed $`\mathrm{SS}`$ need not be 0 ([weighted-statistics.md](weighted-statistics.md), Numerical notes). So the computed heights need not be monotone, even with positive weights.

P3. Order. If no merge joins a cluster with itself, $`L_s`$ lists each row of the cluster formed by merge $`s`$ once. Proof by induction on $`s`$: the two clusters merge $`s`$ joins are disjoint, and their lists are concatenated. So the order lists every row once, and the rows of every cluster in every $`\mathcal{C}_s`$ are adjacent in it.

## Algorithm

Step 1. Inputs. R calls `as.matrix(x)` and stops unless `w` has one entry per row. The port reads `x` with `as_matrix` and `w` with `as_weights`, as in [weighted-statistics.md](weighted-statistics.md) Step 1. R stops below three rows. With no rows it stops at `ws2 = rep(NA, n - 1)`, before `distw` runs, with "invalid 'times' argument". With one row the loop in `distw` runs over `2:1`, calls `distwij` with an `NA` label and stops with "missing value where TRUE/FALSE needed". With two rows R reaches `forder` only when the one cost is not NaN. The loop in `forder` then runs over `2:1`, reads a second merge row and stops with "subscript out of bounds". When the cost is NaN, as with a zero weight or the weights 1 and -1, `which.min` returns `integer(0)` and R stops earlier with "argument is of length zero". For these the port runs `_whclust_py` and raises `ValueError` with R's message in it.

Step 2. Labels. R sets `y = -(1:n)` and `y0 = sort(unique(y))`, so position 1 holds row $`n`$.

Step 3. First matrix. `dd = distw(x, y, w)`. Entry $`(i, j)`$ with $`i \gt j`$ is the cost of merging positions $`i`$ and $`j`$, computed as in Step 7 of [weighted-statistics.md](weighted-statistics.md). The diagonal and the entries above it are `NA`.

Step 4. Pick. `k = which.min(dd)` is the first smallest entry in column-major order, with `NA` and NaN skipped. R splits `k` into a row $`k_0`$ and a column $`k_1`$ with R's counter `m`, and takes $`u = y_0[k_0]`$ and $`v = y_0[k_1]`$. A pick below the diagonal has its row after its column, so $`u \gt v`$. Column-major order sends a tie to the first column in position order that holds the smallest cost. Within that column the tie goes to the first row.

Step 5. Relabel. R appends $`(u, v)`$ to `merge`, sets the labels $`u`$ and $`v`$ in `y` to $`s`$ and decrements `m`. The new label $`s`$ is the largest, so the new cluster takes the last position.

Step 6. Shrink. If $`s \lt n - 1`$, R drops rows and columns $`k_0`$ and $`k_1`$ of `dd`. It appends a row with the cost from the new cluster to each other cluster, computed by `distwij`, and a column of `Inf`. Above the diagonal a singleton's column therefore holds `NA` and a merged cluster's column holds `Inf`.

Step 7. Height. `ws2[s] = wwcss(x, y, w)`, the $`\mathrm{ldsum}`$ of the clusters' $`\mathrm{SS}`$ in position order.

Step 8. Result. R returns `merge`, `height = ws2`, `order = forder(merge)`, `labels = NULL`, `method = "ward.D"`, `call` and `dist.method = "euclidean"` with class `"hclust"`. The port's `WhclustResult` holds `merge`, `height` and `order`.

Picks at `Inf`. A zero weight makes a singleton's $`\mathrm{SS}`$ NaN, so every cost with that singleton is NaN. A pair whose computed total weight is 0 also has a NaN cost. Overflow gives `Inf` costs, and `Inf - Inf` is NaN. When no entry below `Inf` is left, `which.min` lands on the first `Inf` in column-major order. That entry is in the first column in position order that is either a merged cluster's column or a singleton's column with an `Inf` below the diagonal. A merged cluster's column starts with `Inf`, so the pick joins the cluster at position 1 to it. If that column is at position 1, the pick is its diagonal entry and R merges the cluster with itself, so $`u = v`$. After such a self-merge R's counter `m` runs one below the size of `dd`, and the later steps follow from R's code as written. If every entry is `NA` or NaN, `which.min` returns `integer(0)` and R stops with "argument is of length zero".

Step K. The compiled kernel. `whclust` runs `src/datanuggets/wcluster/whclustjit.py`, `whclust`. R recomputes each cluster's $`\mathrm{SS}`$ for every cost and every height. Each of those values is `wss` of the same rows in the same order, so the kernel computes each cluster's own $`\mathrm{SS}`$ once when the cluster forms and each cost once. A two-term $`\mathrm{ldsum}`$ is the same in either order. The costs live in one symmetric $`n \times n`$ array, and a merged cluster takes the slot of its column cluster $`v`$. Each column keeps its first smallest cost below the diagonal, with NaN skipped. The pick is the first column in position order that holds the smallest of these, compared with strict `<`. After a merge a column is searched again only if its smallest cost was in a row that left. The new cluster is the last row of every column, so it replaces a column's smallest cost only when it is strictly smaller. A column that kept no cost, since all its costs were NaN or it had none, takes the new cost unless that cost is NaN. When no cost below `Inf` is left, the kernel takes the `Inf` pick above. On a self-merge it stops, and `whclust` reruns the input through `_whclust_py`, a statement for statement replay of R's function. When no candidate is left, `whclust` raises `ValueError` naming R's message.

Numbering. The port uses SciPy's numbering. Row $`i`$ (R's $`-i`$) is $`i - 1`$, and the cluster formed by merge $`s`$ (R's $`s`$) is $`n + s - 1`$. Each merge row keeps R's column order, and `order` is R's order less one. `src/datanuggets/wcluster/cutree.py`, `r_merge` maps the port's merge rows back to R's.

Linkage. `to_linkage` returns SciPy's $`(n - 1) \times 4`$ linkage. Row $`s - 1`$ holds the two merged clusters, $`\hat{h}_s`$ and the size of the new cluster. The size counts rows, not weight. It raises `ValueError` when a merge joins a cluster with itself.

Nuggets. R's `DN.Whclust` takes columns 2 to `ncol - 2` of the "Data Nuggets" table as the data and its "Weight" column as the weights, then calls `Whclust`. With one variable the column slice drops to a vector, and `Whclust` turns it back into one column. `dn_whclust` calls `whclust(dn.centers, dn.weights)` and raises `TypeError` unless `dn` is a `DataNuggets`.

Cutting. `cut_tree` follows R's `cutree`.

- C1. An empty `merge` stops with "invalid 'tree' ('merge' component)". Neither `k` nor `h` stops with "either 'k' or 'h' must be specified". When both are given, `k` wins.
- C2. R applies `as.integer` to `k`, which truncates toward zero. If $`k`$ is NaN, $`k \ge 2^{31}`$ or $`k \le -2^{31}`$, `as.integer` gives `NA` and R stops with "missing value where TRUE/FALSE needed". For $`k \ge 2^{31}`$ or $`k \le -2^{31}`$, R first warns "NAs introduced by coercion to integer range". The port raises the same stop with no warning. Outside the range 1 to $`n`$ R stops with "elements of 'k' must be between 1 and n", with $`n`$ written out.
- C3. With `h` alone, NaN heights give the `NA` stop. Decreasing heights stop with "the 'height' component of 'tree' is not sorted (increasingly)". `h = NaN` stops with "length of 'dimnames' [2] not equal to array extent". Otherwise $`k`$ comes from D6.
- C4. R's C `cutree` applies merges 1 to $`n - k`$. It keeps, for each row, whether the row is still alone and the last merge that took it in. It then numbers rows 1 to $`n`$. A lone row gets the next number, and a row in a cluster gets the number its cluster received at its first row. For $`k = n`$ the labels are 1 to $`n`$. `_cutree` keeps the same two arrays and numbers clusters by first appearance. The port returns R's labels less one.

Example. The fixture `tests/fixtures/whclust/n3` has the rows $`(5.1, 3.5, 1.4, 0.2)`$, $`(7.0, 3.2, 4.7, 1.4)`$ and $`(6.3, 3.3, 6.0, 2.5)`$ with unit weights. Rows 2 and 3 are the closest pair. With unit weights D6 of [weighted-statistics.md](weighted-statistics.md) gives half their squared distance, and R's computed height is $`\hat{h}_1 = 1.7`$. Merge 2 joins row 1, and $`\hat{h}_2 = 15.786666666666667`$. R records the merge rows $`(-2, -3)`$ and $`(1, -1)`$ and the order $`(2, 3, 1)`$. The port returns the merge rows $`(1, 2)`$ and $`(3, 0)`$ and the order $`(1, 2, 0)`$. In `tests/fixtures/plot/tree_n3` the cut at $`h = 1.7`$ gives the labels $`(1, 2, 2)`$, and the cut at $`h = 15.786666666666667`$ gives $`(1, 1, 1)`$. A cut exactly at a height includes that merge.

## Numerical notes

- Heights are totals. The `method` field says "ward.D", and each height is the total WWCSS after the merge (D2).
- Ties. Exact ties are common with integer or duplicated data. Steps 4 and K make the port take R's pick, and the choice can change later merges.
- Zero weights. Zero-weight singletons merge only through `Inf` picks. Every height is NaN while a zero-weight row is still a singleton. A cut at a height then stops with R's `NA` message, as the trees `tree_zero_w1` and `tree_zero_w2` in `tests/fixtures/plot/` record.
- Signed weights. Costs and heights can be negative. In `tests/fixtures/whclust/selfmerge_last` the weights are 1, 1, -1 and -1. The last merge joins cluster 1 with itself. R's order is then $`(3, 4, 3, 4)`$. It repeats rows 3 and 4 and leaves out rows 1 and 2, so P3 fails. The recorded cut into $`k = 1`$ gives the labels $`(1, 1, 2, 2)`$, and the port returns the same labels less one.
- Overflow. In `tests/fixtures/whclust/inf_single` the rows are 0, `1e160` and `-1e160`. Both picks land on `Inf`, and both heights are `Inf`. In `tests/fixtures/whclust/selfmerge_mid` the weights are all 1, and overflow alone leads to a self-merge before the last step.
- Cuts at infinity. A cut at $`h = +\infty`$ gives $`n`$ clusters, since no entry of $`(\hat{h}_1, \dots, \hat{h}_{n-1}, +\infty)`$ exceeds it and $`q = 1`$. The cut at $`h = -\infty`$ also gives $`n`$ clusters when $`\hat{h}_1 \gt -\infty`$. `tests/fixtures/plot/tree_n3` records both.
- Default weights. `Whclust` has the default `w = rep(1, nrow(x))`, which applies only when `w` is left out. A `w` passed as `NULL` has length 0, so R stops at the length check of Step 1. `whclust(x, w=None)` gives every row a weight of 1.
- Memory. The kernel holds one $`n \times n`$ array of doubles, `8 * n * n` bytes. R copies its $`r \times r`$ matrix at every step.

## Tests

The fixtures were recorded with R 4.6.1 and WCluster 1.3.0. `tests/fixtures/whclust/` holds Iris with unit and unequal weights, 200 rows of `quakes`, one-column data and a grid with duplicated rows. It also holds two sets with zero weights, the three-row case above and the `Inf` and self-merge cases above. Three cases record R's stops for one row, two rows and all-zero weights. `tests/fixtures/plot/` holds R's `cutree` for every `k`, for a set of heights and for each stop. `tests/fixtures/dn_wcluster/` holds `DN.Whclust` on nuggets, with one variable among the cases.

- Steps 1 to 8 and the `Inf` picks: `tests/wcluster/test_whclust.py::test_reference_matches_r`, `tests/wcluster/test_whclust.py::test_reference_matches_r_on_the_large_fixtures` and `tests/wcluster/test_whclust.py::test_reference_stops_where_r_stops` run `_whclust_py`. They compare merge and order exactly and heights bit for bit, with any NaN matching any NaN. They also check how often each case picked an `Inf` or merged a cluster with itself.
- Step K: `tests/wcluster/test_whclust.py::test_whclust_matches_r` and `tests/wcluster/test_whclust.py::test_whclust_stops_where_r_stops` run `whclust` on the same fixtures. `tests/wcluster/test_whclust.py::test_kernel_matches_reference_on_tie_heavy_data` and `tests/wcluster/test_whclust.py::test_kernel_matches_reference_on_a_large_fuzz` compare the kernel with `_whclust_py` on small random inputs in six families: unit, integer, zero and signed weights, duplicated rows and values near `1e154`. `tests/wcluster/test_whclust.py::test_fuzz_reaches_the_inf_pick_and_the_self_merge` checks that those inputs reach both paths.
- Coverage and inputs: `tests/wcluster/test_whclust.py::test_fixtures_cover_every_whclust_quirk`, `tests/wcluster/test_whclust.py::test_whclust_takes_any_layout_and_dtype` and `tests/wcluster/test_whclust.py::test_whclust_needs_three_rows`.
- P3 and the linkage: `tests/wcluster/test_whclust.py::test_to_linkage_drives_scipy_dendrogram_in_r_order` checks that SciPy's dendrogram leaves equal R's order on the weighted Iris case. `tests/test_result.py::test_whclust_result_to_linkage_counts_cluster_sizes`, `tests/test_result.py::test_to_linkage_keeps_heights_bit_for_bit` and `tests/test_result.py::test_to_linkage_refuses_a_self_merge` cover `to_linkage`.
- Nuggets: `tests/wcluster/test_dn.py::test_dn_whclust_matches_r`, `tests/wcluster/test_dn.py::test_one_variable_nuggets_stop_where_r_stops`, which also runs `dn_whclust` on one variable, and `tests/wcluster/test_dn.py::test_dn_functions_need_a_datanuggets`.
- Cutting: `tests/wcluster/test_cutree.py::test_every_k_matches_r`, `tests/wcluster/test_cutree.py::test_every_h_matches_r`, `tests/wcluster/test_cutree.py::test_stops_match_r`, `tests/wcluster/test_cutree.py::test_k_truncates_like_as_integer`, `tests/wcluster/test_cutree.py::test_k_wins_over_h` and `tests/wcluster/test_cutree.py::test_one_cut_per_call`. `tests/wcluster/test_cutree.py::test_numpy_loop_matches_c_on_fuzzed_trees` compares `_cutree` with a line by line copy of R's C on random trees, with and without self-merges. `tests/wcluster/test_cutree.py::test_r_merge_inverts_the_port_numbering` covers the numbering.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. Used: Section 2.
- Dey, R., Duan, Y., Cabrera, J., and Cheng, G. (2025). WCluster: Clustering and PCA with Weights, and Data Nuggets Clustering. R package version 1.3.0. Used: the functions `Whclust`, `forder`, `DN.Whclust`, `distw`, `distwij` and `wwcss`, and the help pages of `Whclust`, `DN.Whclust` and `distw`.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. Used: `stats::cutree`, `src/library/stats/src/hclust-utils.c` (`cutree`) and `which.min`.
- Ward, J. H., Jr. "Hierarchical Grouping to Optimize an Objective Function." Journal of the American Statistical Association 58(301), 236-244, 1963. DOI: `10.1080/01621459.1963.10500845`.

## Not covered by a test

- The remark under D1 that rounding can make the merged pair differ from the pair with the smallest exact cost. No test computes exact costs.
- The remark under P2 that computed heights need not be monotone. No fixture has computed heights that decrease. `tests/wcluster/test_cutree.py::test_stops_match_r` checks the stop on a tree whose heights the test reverses.
- The two-row stop with a NaN cost (Step 1). The fixture `n2` and `tests/wcluster/test_whclust.py::test_whclust_needs_three_rows` use unit weights.
- The `NA` stop of C2 for $`k \le -2^{31}`$. `tests/wcluster/test_cutree.py::test_stops_match_r` checks NaN, $`+\infty`$ and `3e9`.
