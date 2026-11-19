# Weighted statistics

Draft. This page has not yet been reviewed by the authors of the R packages.

This page gives the math of four WCluster statistics and the floating-point steps the port replays to match R. The four are the weighted mean, the weighted sum of squares, the weighted within-cluster sum of squares (WWCSS) and the merge cost of the method of Ward (1963) between two clusters. [weighted-kmeans.md](weighted-kmeans.md) and [weighted-ward.md](weighted-ward.md) build on them. Algorithm 3 of Beavers et al. (2025, Section 3.1) defines a weighted within-cluster sum of squares of the nugget centers, and [weighted-kmeans.md](weighted-kmeans.md) covers that algorithm.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `wmean(x, cl, w=None)` | `src/datanuggets/wcluster/wstats.py`, `wmean` | WCluster 1.3.0, `wmean`, which calls R 4.6.1, `stats::weighted.mean` |
| `wss(x, w=None)` | `src/datanuggets/wcluster/wstats.py`, `wss` and `_wss` | WCluster 1.3.0, `wss` |
| `wwcss(x, cl, w=None, group_sum=False)` | `src/datanuggets/wcluster/wstats.py`, `wwcss` and `_group_wss` | WCluster 1.3.0, `wwcss` |
| `distw(x, cl, w=None)` | `src/datanuggets/wcluster/wstats.py`, `distw` and `_distwij` | WCluster 1.3.0, `distw` and `distwij` |

The port replays R's arithmetic with `r_sum` and `r_weighted_mean` in `src/datanuggets/_compat/arithmetic.py` and with `r_matvec` in `src/datanuggets/_compat/linalg.py`. The compiled kernels `rsum`, `wss_rows` and `wmean_rows` in `src/datanuggets/wcluster/wstatsjit.py` repeat these steps inside `whclust` and `wkmeans`.

## Notation

The symbols these pages share are in [README.md](README.md#notation). This page adds the following.

- $`\ell_i`$ is the label of row $`i`$, and $`\lambda_1 \lt \dots \lt \lambda_k`$ are the distinct labels in sorted order.
- $`C_c = \lbrace i : \ell_i = \lambda_c \rbrace`$ is cluster $`c`$.
- For a set of rows $`S`$, $`W_S`$ is its total weight (D1), $`\bar{\mathbf{x}}_S`$ its weighted mean (D2) and $`\mathrm{SS}(S)`$ its weighted sum of squares (D3).
- $`\mathrm{fl}(\cdot)`$ is a value rounded to double. A hat marks a computed value, such as $`\widehat{\mathrm{SS}}`$, and a tilde marks the computed mean of Step 4. Equations with none of these are exact arithmetic.
- $`\mathrm{WWCSS}(\mathcal{C})`$ is D4 for the clusters of a clustering $`\mathcal{C}`$.
- $`\mathrm{dsum}(v_1, \dots, v_r)`$ starts at $`+0`$ and adds $`v_1, \dots, v_r`$ in that order, rounding to double after each addition.
- $`\mathrm{ldsum}(v_1, \dots, v_r)`$ adds $`v_1, \dots, v_r`$ in that order in a long double accumulator with a 64-bit significand. It rounds once to double at the end and gives $`\pm\infty`$ past the largest double.

R's `sum()` of doubles is $`\mathrm{ldsum}`$ (R 4.6.1, `src/main/summary.c`, `rsum`). The port replays it in `r_sum`. [numerical-parity.md](numerical-parity.md) defines the same two sums in D2 and D3. R's `%*%` of a matrix and a vector is one $`\mathrm{dsum}`$ per output entry. `matprod` (R 4.6.1, `src/main/array.c`) first applies the quick test `mayHaveNaNOrInf` to each operand. The test adds the entries in pairs and is true when one of these sums is not finite. A NaN or an infinity makes it true, and so can two large finite entries whose sum passes the largest double. When the test is true, R calls `simple_matprod` in the same file. Otherwise it calls `DGEMV` of the reference BLAS (R 4.6.1, `src/extra/blas/blas.f`). `DGEMV` adds each column times its vector entry in column order, and `simple_matprod` adds in the same order. The port replays both in `r_matvec`.

## Definitions

Weights are real and finite. They can be zero or negative, as in R.

D1. Total weight.

```math
W_S = \sum_{i \in S} w_i
```

D2. Weighted mean. For $`W_S \neq 0`$ and $`j = 1, \dots, p`$:

```math
\bar{x}_{S,j} = \frac{1}{W_S} \sum_{i \in S} w_i x_{ij}
```

D3. Weighted sum of squares. For $`W_S \neq 0`$:

```math
\mathrm{SS}(S) = \sum_{i \in S} w_i \lVert \mathbf{x}_i - \bar{\mathbf{x}}_S \rVert^2 = \sum_{j=1}^{p} \sum_{i \in S} w_i \left( x_{ij} - \bar{x}_{S,j} \right)^2
```

`wss(x, w)` is $`\mathrm{SS}`$ of all $`n`$ rows (WCluster 1.3.0, `wss`).

D4. Weighted within-cluster sum of squares. Cluster $`c`$ contributes $`\mathrm{SS}(C_c)`$, and `wwcss` returns the total (WCluster 1.3.0, `wwcss`):

```math
\mathrm{WWCSS} = \sum_{c=1}^{k} \mathrm{SS}(C_c)
```

D5. Merge cost. For disjoint sets of rows $`A`$ and $`B`$ with $`W_A`$, $`W_B`$ and $`W_A + W_B`$ nonzero:

```math
d(A, B) = \mathrm{SS}(A \cup B) - \mathrm{SS}(A) - \mathrm{SS}(B)
```

The WCluster 1.3.0 help page for `distw` states it as "dist(A,B) = SS(A+B) - SS(A) - SS(B)", and `distwij` computes it for two clusters.

P1. Shift formula. If $`W_S \neq 0`$, then for every $`\mathbf{a} \in \mathbb{R}^p`$:

```math
\sum_{i \in S} w_i \lVert \mathbf{x}_i - \mathbf{a} \rVert^2 = \mathrm{SS}(S) + W_S \lVert \bar{\mathbf{x}}_S - \mathbf{a} \rVert^2
```

Proof. Write $`\mathbf{x}_i - \mathbf{a} = (\mathbf{x}_i - \bar{\mathbf{x}}_S) + (\bar{\mathbf{x}}_S - \mathbf{a})`$ and expand the square. The cross term vanishes because $`\sum_{i \in S} w_i \mathbf{x}_i = W_S \bar{\mathbf{x}}_S`$ by D2.

D6. Closed form of the merge cost. For disjoint $`A`$ and $`B`$ with $`W_A`$, $`W_B`$ and $`W_A + W_B`$ nonzero:

```math
d(A, B) = \frac{W_A W_B}{W_A + W_B} \lVert \bar{\mathbf{x}}_A - \bar{\mathbf{x}}_B \rVert^2
```

Proof. Let $`U = A \cup B`$, so $`W_U = W_A + W_B`$ and $`W_U \bar{\mathbf{x}}_U = W_A \bar{\mathbf{x}}_A + W_B \bar{\mathbf{x}}_B`$. Split the sum of D3 for $`U`$ into the rows of $`A`$ and the rows of $`B`$. Apply P1 to each part with $`\mathbf{a} = \bar{\mathbf{x}}_U`$ and subtract $`\mathrm{SS}(A) + \mathrm{SS}(B)`$:

```math
d(A, B) = W_A \lVert \bar{\mathbf{x}}_A - \bar{\mathbf{x}}_U \rVert^2 + W_B \lVert \bar{\mathbf{x}}_B - \bar{\mathbf{x}}_U \rVert^2
```

Here $`\bar{\mathbf{x}}_A - \bar{\mathbf{x}}_U = \frac{W_B}{W_U} (\bar{\mathbf{x}}_A - \bar{\mathbf{x}}_B)`$ and $`\bar{\mathbf{x}}_B - \bar{\mathbf{x}}_U = \frac{W_A}{W_U} (\bar{\mathbf{x}}_B - \bar{\mathbf{x}}_A)`$. The right side is then $`\frac{W_A W_B^2 + W_B W_A^2}{W_U^2} \lVert \bar{\mathbf{x}}_A - \bar{\mathbf{x}}_B \rVert^2`$, which is D6. Neither R nor the port evaluates D6. Both compute D5 from the pooled rows (Step 7).

P2. A merge adds its cost to the WWCSS. Let clustering $`\mathcal{C}'`$ come from clustering $`\mathcal{C}`$ by merging clusters $`A`$ and $`B`$. If every cluster of $`\mathcal{C}`$ and $`A \cup B`$ have nonzero total weight, then $`\mathrm{WWCSS}(\mathcal{C}') = \mathrm{WWCSS}(\mathcal{C}) + d(A, B)`$. Proof. By D4 the two totals share every term except $`\mathrm{SS}(A) + \mathrm{SS}(B)`$, which becomes $`\mathrm{SS}(A \cup B)`$. D5 is the difference. [weighted-ward.md](weighted-ward.md) builds its heights on P2.

## Algorithm

Step 1. Inputs. R's `wmean`, `wss` and `wwcss` call `as.data.frame`, and `distw` calls `as.matrix`. Each then stops unless `w` has one entry per row. The port reads `x` with `as_matrix` and `w` with `as_weights` (`src/datanuggets/_validate.py`). A vector `x` is one column, and a bool `x` counts as 0 and 1. NaN or infinite values in `x` or `w` raise `ValueError`. Weights that are not ints or floats raise `TypeError`.

Step 2. Clusters. Both number the clusters in sorted label order. R uses `as.numeric(factor(cl))` in `wmean` and `wwcss` and `sort(unique(cl))` in `distw`. The port's `as_labels` returns 0-based codes from `np.unique`. It accepts whole numbers below `1e15` in absolute value, and for these labels `factor` and `sort(unique())` give the same order.

Step 3. `wmean`. For $`c = 1, \dots, k`$ and $`j = 1, \dots, p`$, with the rows in their original order:

```math
\widehat{\bar{x}}_{C_c,j} = \mathrm{fl}\left( \frac{\mathrm{ldsum}\left( \mathrm{fl}(x_{ij} w_i) : i \in C_c,\ w_i \neq 0 \right)}{\mathrm{ldsum}\left( w_i : i \in C_c \right)} \right)
```

R's `wmean` applies `weighted.mean` to each column of each cluster. R 4.6.1, `stats::weighted.mean.default` computes `sum((x*w)[w != 0])/sum(w)`. With integer `x` and integer `w`, R multiplies in integer arithmetic. A product whose absolute value exceeds 2147483647 becomes NA with a warning, and that center is NA. The port converts both to float64 and computes the formula above, so it returns a number there. The port's `wmean` calls `r_weighted_mean` once per entry. When the division gives a NaN from operands that are not NaN, `r_weighted_mean` returns it with R's x86 bits `fff8000000000000`. R returns a $`p \times k`$ matrix with one column per cluster. The port returns a $`(k, p)`$ array with cluster $`c`$ in row $`c - 1`$.

Step 4. `wss`, the mean. For a set of rows $`S`$ in their original order, zero weights included:

```math
\widetilde{\bar{x}}_{S,j} = \mathrm{fl}\left( \frac{\mathrm{dsum}\left( \mathrm{fl}(x_{ij} w_i) : i \in S \right)}{\mathrm{ldsum}\left( w_i : i \in S \right)} \right)
```

R's `wss` computes `c(t(x) %*% w)/sum(w)`. The port's `_wss` computes `r_matvec(x.T, w) / r_sum(w)`.

Step 5. `wss`, the value.

```math
e_{ij} = \mathrm{fl}\left( x_{ij} - \widetilde{\bar{x}}_{S,j} \right), \qquad t_j = \mathrm{dsum}\left( \mathrm{fl}\left( \mathrm{fl}(e_{ij}^2)\, w_i \right) : i \in S \right), \qquad \widehat{\mathrm{SS}}(S) = \mathrm{ldsum}(t_1, \dots, t_p)
```

R's `wss` computes `sum(t(sweep(x, 2, weighted_mean)^2) %*% w)`. The port's `_wss` computes `r_sum(r_matvec((dev * dev).T, w))` with float warnings silenced, since R gives none.

Step 6. `wwcss`. Each cluster gets $`\widehat{\mathrm{SS}}(C_c)`$ by Steps 4 and 5 on its rows in their original order. The total is

```math
\widehat{\mathrm{WWCSS}} = \mathrm{ldsum}\left( \widehat{\mathrm{SS}}(C_1), \dots, \widehat{\mathrm{SS}}(C_k) \right)
```

R's `wwcss` applies `wss` to each cluster and adds the values with `sum`. The port uses `_group_wss` and then `r_sum`. With `group_sum=True` it returns both in a `WwcssResult`.

Step 7. `distw`. For $`c = 2, \dots, k`$ and $`c' = 1, \dots, c - 1`$, with the pooled rows in their original order:

```math
\hat{d}_{cc'} = \mathrm{fl}\left( \widehat{\mathrm{SS}}(C_c \cup C_{c'}) - \mathrm{ldsum}\left( \widehat{\mathrm{SS}}(C_{c'}), \widehat{\mathrm{SS}}(C_c) \right) \right)
```

R's `distw` calls `distwij` for each pair, and `distwij` subtracts the `wwcss` of the pooled rows from their `wss`. The port computes each cluster's own value once with `_group_wss`, and `_distwij` does the subtraction. The bits agree with R's recomputation, because a cluster's rows keep their order inside the pooled rows. The diagonal and the upper triangle hold R's `NA_real_`, the NaN with bits `7ff00000000007a2`. R fills them with `array(NA, dim = c(k, k))`, and the port writes `_NA_BITS` from `src/datanuggets/_compat/missing.py`. With one cluster R stops in `distwij` with "missing value where TRUE/FALSE needed", and the port raises `ValueError`.

Step 8. Compiled kernels. `wss_rows` replays Steps 4 and 5 on a list of rows. `wmean_rows` replays Step 3, and `rsum` is $`\mathrm{ldsum}`$. When its input holds a NaN or both infinities, `rsum` returns numpy's NaN with bits `7ff8000000000000`, so a NaN from `rsum` or `wss_rows` does not carry R's bits.

## Numerical notes

- Two forms of the mean. Step 3 uses a long double sum and drops zero-weight rows. Step 4 uses a double sum and keeps them. The two means can differ in the last bits, and `wss`, `wwcss` and `distw` always use Step 4.
- Summation order. The bits match R only because the port adds in R's order. The fixture manifests record R's reference BLAS (`Rblas.dll`), the `matprod` option "default" and a long double with 64 significand digits. R with another BLAS can add in another order.
- One-row clusters. A cluster of one row with $`w_i \neq 0`$ has $`\mathrm{SS} = 0`$ in exact arithmetic. Step 4 computes its mean as $`\mathrm{fl}(\mathrm{fl}(x_{ij} w_i) / w_i)`$, which need not equal $`x_{ij}`$, so $`\widehat{\mathrm{SS}}`$ can be a small nonzero value. R's recorded per-cluster values in the singleton cases of `tests/fixtures/wstats/` include such values.
- Costs below zero. Step 7 can subtract nearly equal numbers. With positive weights a computed cost can then fall below 0 by a rounding error in the sums of squares it subtracts. R's recorded `distw` values do so in `iris1_singletons_pw`, `iris3_singletons_pw`, `iris12_singletons_pl` and `hv_quakes_by_stations_w2pow`.
- Signed weights. Negative weights pass `as_weights`, as they pass R. Then $`W_S`$, $`\mathrm{SS}(S)`$ and $`d`$ can be negative in exact arithmetic. P1, D6 and P2 hold whenever the totals they need are nonzero.
- Zero totals. D2, D3 and D6 need nonzero totals. The code does not check them. It divides by the computed total $`\mathrm{ldsum}(w_i : i \in S)`$, and that total can be 0 when the exact total is not. The weights `1e20`, `1` and `-1e20` are an example. When every weight of a cluster is 0, its `wmean` row is NaN. When weights of both signs sum to 0, a `wmean` entry is an infinity with the sign of its numerator, or NaN where the numerator is 0 too. In both cases `wss` of the set is NaN, and so are its `wwcss` value, the total and every `distw` entry that includes the cluster. The port computes these without a warning, as R does.
- Labels. The port raises on labels that are not whole numbers below `1e15` in absolute value. R groups any labels that `factor` or `sort(unique())` accepts.
- Default weights. R's `wmean`, `wwcss` and `distw` default to `rep(1, length(x))`, evaluated after `x` is converted. `wmean` and `wwcss` then get one weight per column of the data frame, and `distw` one per cell of the matrix. The length check stops them unless `x` is square (`wmean`, `wwcss`) or has one column (`distw`). `wss` defaults to `rep(1, nrow(x))`. With `w = NULL` all four stop on the length check. In the port `w=None` means one weight of 1 per row in all four.

## Tests

The fixtures were recorded with R 4.6.1 and WCluster 1.3.0. `tests/fixtures/wstats/` holds cases on R's built-in data sets, some reduced to nuggets first. They include one-cluster cases, clusters of single rows, one-column data, integer data and clusters whose weights are all zero. `tests/fixtures/wstats_constructed/` adds weights near `1e299` that take the weighted sums of squares past the largest double, subnormal weights, powers of two as weights and values near `1e200` whose squares are infinite in double. `tests/fixtures/rcompat_wstats/` holds R's `sum()`, `%*%` and `weighted.mean` on their own. These include sums past the largest double, an infinity at a zero weight and a zero total weight.

The tests that compare a function of this page with R compare bits, and any NaN matches any NaN unless the list says otherwise. The tests of the arithmetic primitives compare NaN bits too. The tests of Step 8 compare the compiled kernels with the Python primitives, not with R.

- Step 3: `tests/wcluster/test_wstats.py::test_wmean_matches_r`, with `x` in C and Fortran order.
- Steps 4 and 5: `tests/wcluster/test_wstats.py::test_wss_matches_r`, with `x` in C and Fortran order.
- Step 6: `tests/wcluster/test_wstats.py::test_wwcss_matches_r`, with `x` in C and Fortran order, and `tests/wcluster/test_wstats.py::test_wwcss_total_is_the_long_double_sum`.
- Step 7: `tests/wcluster/test_wstats.py::test_distw_matches_r` with `x` in C and Fortran order, and `tests/wcluster/test_wstats.py::test_distw_needs_two_clusters`. The first checks the `NA_real_` bits.
- Zero totals: `tests/wcluster/test_wstats.py::test_zero_weight_cluster_is_nan_and_quiet` on the mtcars cases with a cluster of zero weights, and `tests/wcluster/test_wstats.py::test_cancelling_weights_give_infinities_as_in_r` on iris rows with weights 1 and -1, with R's values written into the test.
- Steps 1 and 2: `tests/wcluster/test_wstats.py::test_labels_need_not_start_at_zero`, `tests/wcluster/test_wstats.py::test_default_weights_are_ones`, `tests/wcluster/test_wstats.py::test_weight_and_label_errors`, `tests/wcluster/test_wstats.py::test_takes_1d_x_as_one_column_and_rejects_non_finite_x`, `tests/wcluster/test_wstats.py::test_accepts_a_dataframe`, `tests/test_compat_inputs.py::test_logical_x_and_newx_in_the_wcluster_functions` and `tests/test_compat_inputs.py::test_vector_x_in_the_wcluster_functions`.
- The arithmetic primitives: `tests/compat/test_rstats.py::test_r_weighted_mean_matches_r`, `tests/compat/test_rstats.py::test_matvec_matches_r_bitwise`, `tests/compat/test_rstats.py::test_r_matvec_never_returns_negative_zero`, `tests/compat/test_rstats.py::test_r_sum_boundaries`, `tests/compat/test_rstats.py::test_sum_matches_r_past_dbl_max_and_with_inf_or_nan` and `tests/compat/test_rstats.py::test_rcompat_wstats_covers_each_kind`.
- Step 8: `tests/wcluster/test_wstatsjit.py::test_rsum_matches_r_sum`, `tests/wcluster/test_wstatsjit.py::test_rsum_reads_only_the_first_n` and `tests/wcluster/test_wstatsjit.py::test_wss_rows_and_wmean_rows_match_the_python_primitives`.
- Fixture coverage: `tests/wcluster/test_wstats.py::test_fixtures_cover_the_edge_cases`.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. Used: Algorithm 3 (Section 3.1).
- Dey, R., Duan, Y., Cabrera, J., and Cheng, G. (2025). WCluster: Clustering and PCA with Weights, and Data Nuggets Clustering. R package version 1.3.0. Used: `wmean`, `wss`, `wwcss`, `distw`, `distwij` and the help page of `distw`.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. Used: `stats::weighted.mean.default`, `src/main/summary.c` (`rsum`), `src/main/array.c` (`matprod`, `simple_matprod`) and `src/extra/blas/blas.f` (`DGEMV`).
- Ward, J. H., Jr. "Hierarchical Grouping to Optimize an Objective Function." Journal of the American Statistical Association 58(301), 236-244, 1963. DOI: `10.1080/01621459.1963.10500845`. Used: Ward's method, as the WCluster 1.3.0 help page of `distw` cites it.

## Not covered by a test

- `distw` with signed weights and nonzero totals. No case in `tests/fixtures/wstats/` or `tests/fixtures/wstats_constructed/` has a negative weight. Signed weights with nonzero totals reach the R comparisons of `wmean` and `wwcss` through the cases `gap`, `recreate`, `revert` and `stale_tb` of `tests/fixtures/wkmeans/`. Each has a final cluster with weights of both signs. `tests/wcluster/test_wkmeans.py::test_wkmeans_matches_r_and_uses_r_draws` and `tests/wcluster/test_wkmeans.py::test_reference_matches_r_and_uses_r_draws` compare the centers and the WWCSS that `wkmeans` takes from `wmean` and `wwcss` with R's. Signed weights reach `distw` only through the cancelling case above, whose first cluster has a total of 0. The `selfmerge_last` case of `whclust` meets them in merge costs that `whclust` computes without calling `distw`.
- The NaN bits of `wmean`'s own output. The tests compare them by position. `tests/compat/test_rstats.py::test_r_weighted_mean_matches_r` checks the bits of `r_weighted_mean` alone.
- The NaN bits of `rsum` and `wss_rows` (Step 8). The tests of Step 8 compare them with `r_sum` and `_wss` through `assert_same_values` in `tests/rfixtures.py`, where any NaN matches any NaN.
- A `distw` entry for two clusters with nonzero totals whose pooled weights sum to 0.
- P1, D6 and P2, which this page proves.
- R's stop on its own default weights in `wmean`, `wwcss` and `distw` (Numerical notes, Default weights). No fixture records it.
