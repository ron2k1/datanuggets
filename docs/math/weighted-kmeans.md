# Weighted k-means

Draft. This page has not yet been reviewed by the authors of the R packages.

`wkmeans` clusters rows that carry weights, as WCluster's `Wkmeans` does. A start is a random partition turned into weighted centers. Each pass then runs 20 guarded Lloyd steps and one sweep of single-row moves, and every step is scored by the weighted within-cluster sum of squares (WWCSS). `dn_wkmeans` runs the same procedure on the nugget centers with the nugget weights. `cluster_predict` sends new rows to the nearest weighted cluster center. The weighted mean, `wss` and `wwcss` are defined in full on [weighted-statistics.md](weighted-statistics.md).

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `wkmeans` | `src/datanuggets/wcluster/wkmeans.py`, `wkmeans`, with the refinement in `src/datanuggets/wcluster/wkmeansjit.py`, `refine` | WCluster 1.3.0, `Wkmeans`, with its helpers `init`, `Wkmeans_single` and `wdist_mat` |
| `dn_wkmeans` | `src/datanuggets/wcluster/dn.py`, `dn_wkmeans` | WCluster 1.3.0, `DN.Wkmeans` |
| `cluster_predict` | `src/datanuggets/wcluster/predict.py`, `cluster_predict` | WCluster 1.3.0, `cluster.predict` |
| `dn_cluster_predict` | `src/datanuggets/wcluster/dn.py`, `dn_cluster_predict` | WCluster 1.3.0, `DNcluster.predict` |

## Notation

The shared symbols are in [README.md](README.md#notation). This page adds:

- $`y = (y_1, \dots, y_n)`$, labels in $`\lbrace 1, \dots, k \rbrace`$. Python's `labels[i - 1]` is $`y_i - 1`$.
- $`C_c(y) = \lbrace i : y_i = c \rbrace`$ is cluster $`c`$, and $`W_c(y) = \sum_{i \in C_c(y)} w_i`$ is its total weight.
- $`c_{(1)} \lt \dots \lt c_{(q)}`$, the labels present in $`y`$, with $`q = q(y)`$. A label below $`\max_i y_i`$ that is not present is a gap.
- $`t(y) = (t_1, \dots, t_q)`$, R's `table(y)`, with $`t_r = \lvert C_{c_{(r)}}(y) \rvert`$. R reads it by position, so $`t_a`$ is the size of label $`a`$ only when no gap lies below $`a`$.
- $`\gamma_1, \dots, \gamma_q \in \mathbb{R}^p`$, the centers a distance is taken to.
- $`T`$, `max_iter`. R calls it `max.iterations`, and `K` inside `Wkmeans_single`.
- $`U`$ and $`U_{00}`$, the running WWCSS values that R calls `uu0` and `uu00`.
- $`\boldsymbol{\xi}_l`$ is the center of nugget $`l`$, and $`a(i)`$ is the nugget of row $`i`$.

## Definitions

D1. Weighted mean. R: WCluster 1.3.0, `wmean`. Python: `src/datanuggets/wcluster/wstats.py`, `wmean`. Kernel: `src/datanuggets/wcluster/wstatsjit.py`, `wmean_rows`. For a set of rows $`S`$ with $`W_S = \sum_{i \in S} w_i \ne 0`$,

```math
\mu(S) = \frac{1}{W_S} \sum_{i \in S} w_i \mathbf{x}_i, \qquad \mu_c(y) = \mu\left(C_c(y)\right).
```

D2. Weighted sum of squares. R: WCluster 1.3.0, `wss`. Python: `src/datanuggets/wcluster/wstats.py`, `wss`. Kernel: `src/datanuggets/wcluster/wstatsjit.py`, `wss_rows`.

```math
\mathrm{wss}(S) = \sum_{i \in S} w_i \lVert \mathbf{x}_i - \mu(S) \rVert^2 .
```

D3. WWCSS. R: WCluster 1.3.0, `wwcss`, which groups by `factor(cl)`. Python: `src/datanuggets/wcluster/wstats.py`, `wwcss`. Kernel: `src/datanuggets/wcluster/wkmeansjit.py`, `_total`. Only the labels present count, added in increasing label order:

```math
\Omega(y) = \sum_{r=1}^{q(y)} \mathrm{wss}\left(C_{c_{(r)}}(y)\right).
```

D4. Weighted distance. R: WCluster 1.3.0, `wdist_mat`, which returns `diag(w) %*% (m11 + m22 - 2 * m12)`. Python: `src/datanuggets/wcluster/predict.py`, `_wdist_mat`. Kernel: `src/datanuggets/wcluster/wstatsjit.py`, `assign`. For centers $`\gamma_1, \dots, \gamma_q`$, with $`\delta_{ic}`$ equal to $`\lVert \mathbf{x}_i - \gamma_c \rVert^2`$ in exact arithmetic,

```math
\delta_{ic} = \left(\lVert \mathbf{x}_i \rVert^2 + \lVert \gamma_c \rVert^2\right) - 2\, \mathbf{x}_i^\top \gamma_c, \qquad D_{ic} = w_i \, \delta_{ic}.
```

D5. Nearest center. R: `which.min` in R 4.6.1, as `init`, `Wkmeans_single` and `cluster.predict` call it. Python: `src/datanuggets/wcluster/wstatsjit.py`, `assign`. The minimum is taken over the entries of row $`i`$ of $`D`$ that are not NaN. The first index wins a tie, and a row whose entries are all NaN has no value:

```math
\nu_i(\gamma; w) = \min \left\lbrace c : D_{ic} = \min_{c'} D_{ic'} \right\rbrace .
```

D6. Lloyd map. R: WCluster 1.3.0, `Wkmeans_single`, its inner loop. Python: `src/datanuggets/wcluster/wkmeansjit.py`, `refine`. Its values are positions $`r = 1, \dots, q(y)`$. So a gap in $`y`$ does not carry over, and a position that no row picks becomes a new gap. With $`c_{(1)} \lt \dots \lt c_{(q)}`$ the labels present in $`y`$,

```math
L(y)_i = \nu_i\left(\mu_{c_{(1)}}(y), \dots, \mu_{c_{(q)}}(y); w\right), \qquad i = 1, \dots, n.
```

## Algorithm

`Wkmeans` runs `init` and then `Wkmeans_single` once per start, and keeps the best start.

Step 1. Arguments. $`k`$, `num_init` and `max_iter` are floored. With user centers, `num_init` becomes 1. R then loops over `1:num.init` and `1:max.iterations`. For a finite value $`v \lt 1`$ after the floor, R's `1:v` counts down from 1 to $`v`$. So the loop over starts runs $`2 - v`$ times, twice when $`v = 0`$. The loop over passes runs at most $`2 - v`$ times, since Step 7e can stop it early. The port raises `ValueError` when `num_init` or `max_iter` is below 1 at this point. R: WCluster 1.3.0, `Wkmeans` and `Wkmeans_single`. Python: `src/datanuggets/wcluster/wkmeans.py`, `_prepare`.

Step 2. Seed. R calls `set.seed(seed)` once, and every start draws from that one stream in order. `RDraws(seed)` replays R's stream. An integer seed does not ([randomness.md](randomness.md)).

Step 3. Random partition. Draw

```math
\tilde{y} = \pi\left(1, 2, \dots, k, u_1, \dots, u_{n-k}\right),
```

where the $`u_r`$ are independent and uniform on $`\lbrace 1, \dots, k \rbrace`$ and $`\pi`$ is a uniform random permutation of the $`n`$ entries. Redraw until $`\min_c \lvert C_c(\tilde{y}) \rvert \ge 2`$, which can succeed only when $`n \ge 2k`$. R: WCluster 1.3.0, `init`, whose `ff(k, n)` is `sample(c(1:k, sample(k, n - k, replace = TRUE)))`. Python: `src/datanuggets/wcluster/wkmeans.py`, `_draw_labels` and `_init`.

Step 4. Start labels. Set $`\gamma_c = \mu_c(\tilde{y})`$ for $`c = 1, \dots, k`$ and $`y^{(0)}_i = \nu_i(\gamma; w)`$. With user centers, $`\gamma`$ is the given matrix and Step 3 is skipped. R takes that matrix as $`p`$ by $`k`$, and the port as $`k`$ by $`p`$. R: WCluster 1.3.0, `init`. Python: `src/datanuggets/wcluster/wkmeans.py`, `_init`.

Step 5. Return to Step 3 while $`y^{(0)}`$ has fewer than $`k`$ distinct entries. With user centers, R stops instead with "For the initialization, no data assigned to one(or multiple) of the clusters...". R draws without a limit. The port counts every partition drawn in Step 3, including redraws. It raises `ValueError` after `MAX_DRAWS` = 1,000 partitions in one start. A start found within the cap is R's start. Before any draw, the port also raises `ValueError` without user centers when $`k \le n \lt 2k`$, or when fewer than $`k`$ distinct pairs of row and weight exist. R would draw forever in both cases. R: WCluster 1.3.0, `init`. Python: `src/datanuggets/wcluster/wkmeans.py`, `_prepare`, `_init` and `MAX_DRAWS`.

Steps 6 to 8 are `Wkmeans_single`, run from $`y \leftarrow y^{(0)}`$. R: WCluster 1.3.0, `Wkmeans_single`. Python: `src/datanuggets/wcluster/wkmeansjit.py`, `refine`.

Step 6. Set $`k \leftarrow \max_i y_i`$ and $`t \leftarrow t(y)`$.

Step 7. For each pass $`\ell = 1, \dots, T`$:

- 7a. $`U_{00} \leftarrow \Omega(y)`$.
- 7b. Lloyd steps, 20 of them. Set $`y^{\mathrm{in}} \leftarrow y`$, $`y' \leftarrow L(y^{\mathrm{in}})`$ and $`U \leftarrow \Omega(y')`$. If $`U \gt U_{00}`$, set $`y \leftarrow y^{\mathrm{in}}`$. Otherwise set $`y \leftarrow y'`$ and $`U_{00} \leftarrow U`$. The table $`t`$ is not refreshed here.
- 7c. $`U \leftarrow \Omega(y)`$.
- 7d. Single-row moves. Loop over $`i = 1, \dots, n`$ and, inside that, over $`b = 1, \dots, k`$. Let $`a = y_i`$ with the current $`y`$. Skip if $`a = b`$. If $`a`$ exceeds the length of $`t`$, R stops with "missing value where TRUE/FALSE needed". Skip if $`t_a \le 1`$. Otherwise let $`y'`$ be $`y`$ with $`y'_i = b`$. If $`\Omega(y') \le U`$, set $`y \leftarrow y'`$, $`U \leftarrow \Omega(y')`$ and $`t \leftarrow t(y)`$.
- 7e. Stop if $`y`$ equals $`y^{\mathrm{in}}`$ of the 20th Lloyd step.

Step 8. Return $`(U, y)`$.

Step 9. Best start. A later start replaces the best one only when the best value is strictly greater than its value. So the first start with the smallest value wins. R: WCluster 1.3.0, `Wkmeans`. Python: `src/datanuggets/wcluster/wkmeans.py`, `wkmeans`.

Step 10. Result. Let $`y^\ast`$ be the best labels. `labels` is $`y^\ast - 1`$, with any gap kept. `centers` holds $`\mu_{c_{(r)}}(y^\ast)`$ for the present labels in increasing order, and `wwcss` holds their $`\mathrm{wss}`$. `total_wwcss` is $`\Omega(y^\ast)`$. R recomputes these with `wmean` and `wwcss(..., groupSum = TRUE)`. Python: `src/datanuggets/wcluster/wkmeans.py`, `_result`.

Step 11. `dn_wkmeans` runs Steps 1 to 10 on $`(\boldsymbol{\xi}_l, w_l)`$ for $`l = 1, \dots, m`$, which gives nugget labels $`y_1, \dots, y_m`$. Each original row takes its nugget's label, $`y^{\mathrm{row}}_i = y_{a(i)}`$. R takes the table's columns 2 to `ncol - 2` as the data and its `Weight` column as the weights. It gives each row the nugget label at the position named by the row's entry of `Data Nugget Assignments`. These entries are nugget IDs. The functions `create.DN` and `refine.DN` of datanugget 1.5.0 number the nuggets 1 to $`m`$ in table order, so each ID is also the nugget's position. The port returns `labels[assignments]` as `row_labels`. Both stop on a datanugget with one variable. R: WCluster 1.3.0, `DN.Wkmeans`. Python: `src/datanuggets/wcluster/dn.py`, `dn_wkmeans`.

Step 12. `cluster_predict`. The training rows $`\mathbf{x}_i`$ with labels $`\mathrm{cl}_i`$ and weights $`w_i`$ give the centers $`\mu_{c_{(r)}}(\mathrm{cl})`$, for the present labels in increasing order. A new row $`\tilde{\mathbf{x}}_h`$ goes to $`\hat{r}_h`$, the first $`r`$ at which $`\tilde{\delta}_{hr}`$ is smallest. Here $`\tilde{\delta}_{hr}`$ is the $`\delta`$ of D4 between $`\tilde{\mathbf{x}}_h`$ and center $`r`$, with unit weights. R returns the position $`\hat{r}_h`$, and the port returns $`\hat{r}_h - 1`$. The label is `np.unique(cl)[pos]`. The training weights enter only through the centers. R: WCluster 1.3.0, `cluster.predict`, which calls `wdist_mat` with `w = rep(1, nrow(newx))`. Python: `src/datanuggets/wcluster/predict.py`, `cluster_predict`.

Step 13. `dn_cluster_predict` is Step 12 with the nugget centers as the training rows and the nugget weights as their weights. `cl` holds one label per nugget. A datanugget `newx` is read through its table. R takes columns 2 to $`p + 1`$, with $`p`$ the number of variables of the training datanugget, and the port takes the same columns as the first $`p`$ of `[centers, weights, scales]`. R: WCluster 1.3.0, `DNcluster.predict`. Python: `src/datanuggets/wcluster/dn.py`, `dn_cluster_predict`.

Two consequences follow from the steps.

Monotone value. The value a start returns is at most $`\Omega(y^{(0)})`$, for any weights on which R does not stop. Step 7b keeps $`U_{00} = \Omega(y^{\mathrm{in}})`$ and accepts a Lloyd step only when $`\Omega`$ does not rise. Step 7c recomputes $`\Omega`$ on labels whose value is $`U_{00}`$. Step 7d accepts a move only when $`\Omega`$ does not rise.

Clustering nuggets. Suppose each nugget center is the mean of its rows and each weight is its row count, and every cluster has $`W_c \gt 0`$. Let $`S_l = \sum_{i : a(i) = l} \lVert \mathbf{x}_i - \boldsymbol{\xi}_l \rVert^2`$. Write $`\Omega(y; w)`$ for the $`\Omega`$ of D3 on the nugget centers with weights $`w`$, and $`\Omega_{\mathrm{rows}}(y^{\mathrm{row}}; \mathbf{1})`$ for the WWCSS of the original rows with unit weights. For any nugget labels $`y`$, in exact arithmetic,

```math
\Omega_{\mathrm{rows}}\left(y^{\mathrm{row}}; \mathbf{1}\right) = \Omega(y; w) + \sum_{l=1}^{m} S_l .
```

The last sum does not depend on $`y`$, so lowering $`\Omega(y; w)`$ lowers the WWCSS of the original rows by the same amount. Proof: the rows of cluster $`c`$ have mean $`\mu_c(y)`$, since the rows of nugget $`l`$ sum to $`w_l \boldsymbol{\xi}_l`$. For the rows of nugget $`l`$, $`\sum_{i : a(i) = l} \lVert \mathbf{x}_i - \mu_c \rVert^2 = S_l + w_l \lVert \boldsymbol{\xi}_l - \mu_c \rVert^2`$, because the cross term sums to zero. Sum over the nuggets of each cluster, then over the clusters. A nugget with no rows has weight 0 and adds nothing to either side. The proof of Theorem 1 in Beavers et al. (2025) uses the same cancellation.

## Numerical notes

Two means. `wss` and `wmean` use the same mean in exact arithmetic, but R computes it two ways. `wss` takes `c(t(x) %*% w)/sum(w)`, a double dot product over all rows divided by a long double sum. `wmean` calls `weighted.mean`, whose numerator is a long double sum over the rows of nonzero weight. The two need not agree in the last bits. The port keeps both, so the Lloyd centers come from `wmean` and $`\Omega`$ from the `wss` mean. [weighted-statistics.md](weighted-statistics.md) gives both sums, and [numerical-parity.md](numerical-parity.md) the long double arithmetic.

The Lloyd shortcut. R always runs the 20 Lloyd steps of 7b. A step depends only on its input. So after a revert or after a step that changes nothing, every later step repeats it. The kernel leaves the loop there. It has the same $`y`$, $`U_{00}`$ and $`y^{\mathrm{in}}`$ that R has after step 20.

Trial moves. R recomputes all of $`\Omega(y')`$ for each trial in 7d. The kernel recomputes only clusters $`a`$ and $`b`$, each on its rows in original order, and adds the cached values of the other clusters in label order. The same values in the same order give the same sum, so each trial score has R's bits.

The size table. $`t`$ is refreshed only after an accepted move, and it is read by position. So the guard $`t_a \gt 1`$ can let a move empty a label, and it can block a move that would leave every label filled. A label past the end of $`t`$ stops R. Since $`b`$ runs over all $`k`$ labels, a move can also refill an empty label.

Signs of weights. When every $`\delta_{ic}`$ is finite, $`D_{ic}`$ is the rounded product of $`w_i`$ and $`\delta_{ic}`$. For $`w_i \gt 0`$ rounding never reverses the order of two of the row's $`\delta_{ic}`$, but it can make two values that differ in the last bits equal. The row then gets the center with the smallest $`\delta_{ic}`$, as in unweighted k-means, unless rounding ties it with an earlier center. The first of the tied centers then wins. A row with $`w_i = 0`$ ties at every center and gets center 1. A row with $`w_i \lt 0`$ gets its farthest center, with the same rule for ties.

NaN and overflow. A cluster with $`W_c = 0`$ has no finite mean, and its $`\mathrm{wss}`$ and $`\Omega`$ come out NaN. An `if` on a NaN $`\Omega`$ in 7b or 7d stops R with "missing value where TRUE/FALSE needed". R's product with `diag(w)` adds 0 times every other row of a column. So one non-finite $`\delta`$ in a column makes every other entry of that column NaN. A Lloyd step that leaves some row with every entry NaN stops R inside `wwcss`. The port raises `ValueError` with R's message in each case. A NaN from the kernel matches R's as a NaN, not bit for bit. $`\delta_{ic}`$ can come out slightly below zero from rounding, and it is compared as it is.

One row. For $`n = 1`$, R's `diag(w)` of one number is an identity matrix of size `as.integer(w)`. So `Wkmeans` on one row runs only for $`w \ge 1`$ and $`w \lt 2`$, with $`D = \delta`$. One row always needs user centers, since a random start needs $`n \ge 2k`$. The port raises where R stops. For a finite $`w`$ with $`w \ge 2^{31}`$ or $`w \le -2^{31}`$, `as.integer(w)` is `NA`. R first warns "NAs introduced by coercion to integer range" and then stops with "invalid 'nrow' value (too large or NA)". The port raises `ValueError` with the stop's message and gives no warning. For $`w \ge 2^{26} + 1`$ below that size, `diag(w)` would have more than $`2^{52}`$ entries, and R stops with "too many elements specified". The port raises with that message. A smaller weight that is too large for R to allocate `diag(w)` makes R stop on memory. The port raises with "non-conformable arguments" there.

## Tests

The fixtures were recorded from R 4.6.1 with WCluster 1.3.0. For each case, `tests/fixtures/wkmeans/` holds every random draw, each start's `init` labels, each start's refined labels and value, R's "Iteration" messages and counts of the R quirks each case reaches. The cases cover one and several starts, $`k = 1`$, one column, `max_iter` of 1, user centers, positive and signed weights, gaps, reverts, the stale size table, a refilled label, and the three stops (`ucenters_empty`, `tb_na`, `nan_stop`).

- `tests/wcluster/test_wkmeans.py::test_wkmeans_matches_r_and_uses_r_draws` replays R's draws to the last one. It holds each start's `init` labels, refined labels, value and pass count to R's. It holds the final labels, centers, per-cluster WWCSS and total to R's bits. `tests/wcluster/test_wkmeans.py::test_reference_matches_r_and_uses_r_draws` and `tests/wcluster/test_wkmeans.py::test_reference_matches_r_on_the_large_fixtures` do the same for `tests/wcluster/wkmeans_reference.py` and also compare its quirk counts with R's. That file is a statement-for-statement copy of R's `Wkmeans` and `Wkmeans_single`, and it shares the package's argument checks, `init` and result.
- `tests/wcluster/test_wkmeans.py::test_fixtures_cover_every_wkmeans_quirk` checks that the fixtures reach every quirk: a redraw, a revert, an unchanged Lloyd step, a move the stale table decides, a refilled label, a gap, a label past the end of $`t`$ and a NaN test.
- `tests/wcluster/test_wkmeans.py::test_kernel_matches_reference_on_tie_heavy_data`, `tests/wcluster/test_wkmeans.py::test_fuzz_reaches_every_quirk` and `tests/wcluster/test_wkmeans.py::test_kernel_matches_reference_on_a_large_fuzz` hold the compiled kernel to the reference on generated data. The comparison covers the kernel's Lloyd shortcut and trial scores.
- `tests/wcluster/test_wkmeans.py::test_wkmeans_stops_where_r_stops`, `tests/wcluster/test_wkmeans.py::test_reference_stops_where_r_stops` and `tests/wcluster/test_wkmeans.py::test_arguments_stop_where_r_stops` check R's error messages, the last on the edge-case arguments in `tests/fixtures/wkmeans_args/table/`. `tests/wcluster/test_wkmeans.py::test_one_row_follows_rs_diag` covers one row.
- The draw cap: `tests/wcluster/test_wkmeans.py::test_the_draw_cap_counts_inner_and_outer_draws`, `tests/wcluster/test_wkmeans.py::test_each_start_gets_its_own_max_draws` and `tests/wcluster/test_wkmeans.py::test_the_cap_leaves_a_start_found_within_it_alone`. In `tests/wcluster/test_wkmeans.py::test_the_cap_allows_exactly_max_draws`, the data are one column holding the integers 0 to 15. With $`k = 8`$ after `set.seed(1)`, R's first start takes 2,179 partitions. `RDraws(1)` replays them. The port raises at the default cap and matches the uncapped run at a cap of 2,179.
- `tests/wcluster/test_dn.py::test_dn_wkmeans_matches_r`, `tests/wcluster/test_dn.py::test_dn_cluster_predict_matches_r` and `tests/wcluster/test_dn.py::test_one_variable_nuggets_stop_where_r_stops` compare the datanugget functions with R on `tests/fixtures/dn_wcluster/`.
- `tests/wcluster/test_dn.py::test_a_datanugget_newx_is_sliced_by_dns_width` compares the port with itself on the same nuggets. A datanugget `newx` gives the positions of the first $`p`$ columns of its centers, weights and scales passed as a matrix. The port raises when `[centers, weights, scales]` has fewer than $`p`$ columns.
- `tests/wcluster/test_predict.py::test_wdist_mat_matches_r` and `tests/wcluster/test_predict.py::test_cluster_predict_matches_r` compare distances and positions with R on `tests/fixtures/wstats/` and `tests/fixtures/wstats_constructed/`. `tests/wcluster/test_predict.py::test_overflow_rows_match_r`, `tests/wcluster/test_predict.py::test_returns_positions_not_labels`, `tests/wcluster/test_predict.py::test_first_index_wins_a_tie`, `tests/wcluster/test_predict.py::test_dead_cluster_is_never_predicted` and `tests/wcluster/test_predict.py::test_negative_squared_distance_is_kept` cover the cases of Step 12 and of the NaN notes.
- `tests/wcluster/test_wstatsjit.py::test_assign_is_which_min_over_wdist_mat` holds the kernel's assignment to the first minimum that is not NaN in each row of the port's `_wdist_mat`. `tests/wcluster/test_wstatsjit.py::test_assign_with_an_overflowing_row_kills_every_other_row` checks the kernel on a row whose distances overflow, which leaves only NaN in every other row.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. Used: Section 3.1, Algorithm 3 and the paragraph after it, and the proof of Theorem 1.
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0. Used: `create.DN` and `refine.DN`, for how they number the nuggets.
- Dey, R., Duan, Y., Cabrera, J., and Cheng, G. (2025). WCluster: Clustering and PCA with Weights, and Data Nuggets Clustering. R package version 1.3.0. Used: `Wkmeans`, `init`, `Wkmeans_single`, `wdist_mat`, `wmean`, `wss`, `wwcss`, `DN.Wkmeans`, `cluster.predict` and `DNcluster.predict`, and the help pages of the exported ones.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. Used: `which.min`, `diag`, `sum`, `weighted.mean`, `table`, `factor`, `apply`, `set.seed`, `sample` and `%*%`.

## Not covered by a test

- A Lloyd step that leaves some row with every distance NaN. R's labels become a list, and `factor` inside `wwcss` stops with "unimplemented type 'list' in 'orderVector1'". The port raises that message, and no fixture reaches it.
- A Lloyd step that leaves every row with every distance NaN. R's labels become `integer(0)`, and `wwcss` stops with "The length of cluster assignments should be the same as the number of observations." The port raises that message, and no fixture reaches it.
- A start whose `init` labels reach $`k`$ distinct entries while some row has every distance NaN. R's `apply` then returns a list, and R stops in `Wkmeans_single` with "invalid 'type' (list) of argument". The port raises that message in `_init`, and no fixture reaches it.
- R's positions for a datanugget `newx` whose number of variables differs from that of the training datanugget. The only such fixture case is R's stop for a training datanugget with one variable.
- The two consequences under Algorithm (Monotone value and Clustering nuggets) are proved on this page. No test checks them.
