# Hartigan-Wong k-means for nugget splits

Draft. This page has not yet been reviewed by the authors of the R packages.

`refine.DN` splits a nugget with `stats::kmeans`, which runs the Hartigan-Wong algorithm (Hartigan and Wong, 1979) by default. R implements it in the Fortran routine `KMNS`. Its source file calls the routine "a minimal modification of AS136 to use double precision". Algorithm 2 of Beavers et al. (2025) names only "K-means clustering" in step 4.1. This page follows `kmeans` and `KMNS` in the one setting `refine_dn` uses: two clusters, `iter.max = 1000` and `nstart` starts after `set.seed(seed)`. [refine-dn.md](refine-dn.md) describes the splits themselves.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `kmeans` | `src/datanuggets/_clustering/hartigan_wong.py`, `kmeans` | R 4.6.1, `stats::kmeans` |
| `_one_run` | `src/datanuggets/_clustering/hartigan_wong.py`, `_one_run` | R 4.6.1, `stats::kmeans`, inner `do_one` |
| `kmns`, `hartigan_wong` | `src/datanuggets/_clustering/hartigan_wong.py`, `kmns`; `src/datanuggets/_clustering/kmnsjit.py`, `hartigan_wong` | R 4.6.1, `src/library/stats/src/kmns.f`, `KMNS` |
| `_initial_assignment`, `_initial_centers`, `_final_centers_and_wss` | `src/datanuggets/_clustering/kmnsjit.py`, same names | R 4.6.1, `kmns.f`, `KMNS` |
| `_optra` | `src/datanuggets/_clustering/kmnsjit.py`, `_optra` | R 4.6.1, `kmns.f`, `OPTRA` |
| `_qtran` | `src/datanuggets/_clustering/kmnsjit.py`, `_qtran` | R 4.6.1, `kmns.f`, `QTRAN` |
| `_transfer` | `src/datanuggets/_clustering/kmnsjit.py`, `_transfer` | R 4.6.1, `kmns.f`, the center updates in `OPTRA` and `QTRAN` |

## Notation

The shared symbols, such as $`p`$, $`k`$, $`c`$ and $`\mathbf{x}_i`$, are in [README.md](README.md#notation). Here $`\mathbf{x}_1, \dots, \mathbf{x}_n`$ are the rows handed to `kmeans`. They are a nugget's rows in the EV pass, or their projections with $`p = 1`$ in the shape pass. So $`n`$ is the nugget's weight $`w_l`$, and $`k = 2`$. This page adds:

| Symbol | Meaning | `kmns.f` | Python |
| --- | --- | --- | --- |
| $`a(i)`$ | the cluster of row $`i`$ | `IC1(I)` | `ic1` |
| $`b(i)`$ | the other cluster | `IC2(I)` | `ic2` |
| $`n_c`$ | the number of rows in cluster $`c`$ | `NC(L)` | `nc` |
| $`\bar{\mathbf{x}}_c`$, $`\bar x_{cj}`$ | the running center of cluster $`c`$, and its entry $`j`$ | `C(L,J)` | `c` |
| $`d_c(i)`$ | $`\lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert^2`$, summed over the columns in order | local sums such as `DA` | local sums such as `da` |
| $`\alpha^+_c`$, $`\alpha^-_c`$ | the size factors (D2) | `AN2(L)`, `AN1(L)` | `an2`, `an1` |
| $`D_i`$, $`R_i`$ | the costs of row $`i`$ (D3) | `D(I)`, `R2` | `d`, `r2` |
| $`\nu_c`$ | the step at which cluster $`c`$ last changed, plus $`n`$ in the quick-transfer stage (D5) | `NCP(L)` | `ncp` |
| $`Q`$ | the step budget of the quick-transfer stage (D7) | `iMaxQtr` | `max_qtran` |
| $`s`$, $`u`$ | quick-transfer steps so far, and steps since the last move | `ISTEP`, `ICOUN` | `istep`, `icoun` |
| $`n_s`$ | the number of starts | `nstart` | `nstart` |
| $`W`$ | the total within-cluster sum of squares (D1) | `sum(WSS)` in `kmeans` | `tot_withinss` |

## Definitions

D1. Objective. For an assignment $`a`$ with cluster means $`\bar{\mathbf{x}}_c`$:

```math
W = \sum_{c=1}^{2} \ \sum_{i : a(i) = c} \lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert^2 .
```

Here $`\bar{\mathbf{x}}_c`$ is the exact mean of the rows in cluster $`c`$. In exact arithmetic the running centers of D4 equal these means. The `kmns.f` header states the aim: divide the points into K clusters "so that the within cluster sum of squares is minimized". `kmeans` returns the two terms as `withinss` and their sum as `tot.withinss`.

D2. Size factors. For a cluster with $`n_c`$ rows:

```math
\alpha^+_c = \frac{n_c}{n_c + 1}, \qquad \alpha^-_c = \frac{n_c}{n_c - 1} \quad (n_c \ge 2) .
```

For $`n_c = 1`$, $`\alpha^-_c`$ is set to `BIG`, the REAL constant `1.E30` widened to double. Both stages skip a row whose cluster has one member. So `BIG` is stored and never multiplied, and it cannot reach a result.

D3. Costs. For a row $`i`$ whose cluster has at least 2 rows:

```math
D_i = d_{a(i)}(i) \, \alpha^-_{a(i)}, \qquad R_i = d_{b(i)}(i) \, \alpha^+_{b(i)} .
```

The `kmns.f` comments describe each move as one that reduces the within-cluster sum of squares. In exact arithmetic, with each center equal to its cluster's mean, $`D_i`$ is the drop in $`W`$ when row $`i`$ leaves $`a(i)`$ and $`R_i`$ is the rise in $`W`$ when it joins $`b(i)`$. This follows from D1. Let cluster $`c`$ hold row $`i`$ and $`n_c \ge 2`$ rows in all. Let $`\bar{\mathbf{x}}'_c`$ be the mean of its other $`n_c - 1`$ rows, so that $`\bar{\mathbf{x}}'_c - \bar{\mathbf{x}}_c = (\bar{\mathbf{x}}_c - \mathbf{x}_i)/(n_c - 1)`$. The deviations of those rows from $`\bar{\mathbf{x}}'_c`$ sum to zero. So removing row $`i`$ lowers the sum of squares of cluster $`c`$ by

```math
\lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert^2 + (n_c - 1) \lVert \bar{\mathbf{x}}'_c - \bar{\mathbf{x}}_c \rVert^2 = \Bigl( 1 + \frac{1}{n_c - 1} \Bigr) \lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert^2 = \frac{n_c}{n_c - 1} \lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert^2 .
```

With $`c = a(i)`$ this is $`D_i`$. Adding row $`i`$ to a cluster $`c`$ with $`n_c`$ rows is the reverse step for the enlarged cluster. Its mean is $`\bar{\mathbf{x}}_c + (\mathbf{x}_i - \bar{\mathbf{x}}_c)/(n_c + 1)`$, at distance $`\frac{n_c}{n_c + 1} \lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert`$ from $`\mathbf{x}_i`$. The identity above, read for the enlarged cluster of $`n_c + 1`$ rows, gives a rise of

```math
\frac{n_c + 1}{n_c} \cdot \frac{n_c^2}{(n_c + 1)^2} \lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert^2 = \frac{n_c}{n_c + 1} \lVert \mathbf{x}_i - \bar{\mathbf{x}}_c \rVert^2 .
```

With $`c = b(i)`$ this is $`R_i`$. Removing row $`i`$ leaves cluster $`b(i)`$ unchanged, so a move changes $`W`$ by $`R_i - D_i`$. A move with $`R_i \lt D_i`$ lowers $`W`$. Step 4 and D4 keep each center equal to its cluster's mean in exact arithmetic.

D4. Transfer. Write $`a`$ and $`b`$ for $`a(i)`$ and $`b(i)`$, and $`n_a`$ and $`n_b`$ for their sizes before the move. Moving row $`i`$ from cluster $`a`$ to cluster $`b`$ updates every column $`j`$:

```math
\bar x_{aj} \leftarrow \frac{n_a \bar x_{aj} - x_{ij}}{n_a - 1}, \qquad \bar x_{bj} \leftarrow \frac{n_b \bar x_{bj} + x_{ij}}{n_b + 1} .
```

The operations run in the order of `(C(L1,J) * AL1 - A(I,J)) / ALW`. Then $`n_a`$ drops by 1 and $`n_b`$ grows by 1. D2 is recomputed at the new sizes. In exact arithmetic the updated centers are the means of the new clusters.

D5. Update steps. In the optimal-transfer stage a move of row $`i`$ sets $`\nu_a = \nu_b = i`$, with $`a`$ and $`b`$ as in D4. In the quick-transfer stage a move at step $`s`$ sets both to $`s + n`$. `KMNS` starts every $`\nu_c`$ at $`-1`$. The quick-transfer stage tests row $`i`$ at step $`s`$ only if $`s \lt \nu_{a(i)}`$ or $`s \lt \nu_{b(i)}`$. That is the case when one of its two clusters changed at a quick-transfer step $`t`$ with $`s - n \lt t \lt s`$. In the first cycle ($`s \le n`$) a change in the `OPTRA` pass at a row after row $`s`$ also counts. A change at step $`s - n`$, where `QTRAN` last examined row $`i`$, does not count. With $`k = 2`$ every move changes both clusters, so $`\nu_1 = \nu_2`$ throughout.

D6. Live set. `OPTRA` keeps a live set that limits which other clusters it examines for a row. With $`k = 2`$ the only other cluster is $`b(i)`$, which `OPTRA` always examines. So the live set never changes a result here.

D7. Step budget. `kmeans` passes `KMNS` the budget

```math
Q = \min \bigl( 2^{31} - 1, \ 50 n \bigr) ,
```

computed as `as.integer(min(.Mimax, 50 * m))`. In that code `m` is `nrow(x)`, the $`n`$ of this page. `.Mimax` is R's largest integer.

## Algorithm

Step 1. Starts. With $`n_s = 1`$, `kmeans` draws two row numbers with `sample.int(n, 2)` and takes those rows as starts. If the two rows are equal, it draws the starts again from `unique(x)`. With $`n_s \ge 2`$ it skips the draw from all $`n`$ rows and draws the starts from `unique(x)`. `unique(x)` holds the first copy of each distinct row, in row order. If `unique(x)` has fewer than 2 rows, R stops with "more cluster centers than distinct data points.". For $`n_s \lt 2`$ with $`n_s \ne 1`$, no start is drawn. `centers` is then still the number 2, and the column check stops R with "argument is of length zero".

Step 2. Size check. `KMNS` returns IFAULT 3 when $`k \le 1`$ or $`k \ge n`$. Let $`N_{\min}`$ be `min.nugget.size` rounded down, as in [refine-dn.md](refine-dn.md#notation). `refine.DN` requires $`N_{\min} \ge 2`$. Its `split_nugget` returns before `kmeans` for a nugget with $`n \lt 2N_{\min}`$, and `_split_nugget` in `refine_dn` does the same. So a candidate that reaches `kmeans` has $`n \ge 4`$ rows, and this does not occur there.

Step 3. Initial assignment. Row $`i`$ goes to cluster 1 when $`d_1(i) \le d_2(i)`$ and to cluster 2 otherwise, with the starts as centers. A tie goes to start 1. $`b(i)`$ is the other cluster.

Step 4. Initial centers. Each center becomes the mean of its rows, summed in row order and divided by $`n_c`$. An empty cluster gives IFAULT 1. D2 sets the size factors, and every $`\nu_c`$ is set to $`-1`$.

Step 5. Optimal transfer (`OPTRA`). One pass over $`i = 1, \dots, n`$ skips a row whose cluster has one member. For every other row it recomputes $`D_i`$ and computes $`R_i`$ from the current centers. If $`R_i \lt D_i`$ the row moves (D4), and $`a(i)`$ and $`b(i)`$ swap. Otherwise $`b(i)`$ stays. `OPTRA` recomputes $`D_i`$ whenever $`\nu_{a(i)} \ne 0`$. With $`k = 2`$ each $`\nu_c`$ is $`-1`$ or a row number during this pass, so $`D_i`$ is always recomputed.

Step 6. If `OPTRA` moved no row, `KMNS` stops with IFAULT 0.

Step 7. Quick transfer (`QTRAN`). It cycles over the rows in order, one step per row, from $`s = u = 0`$. At each step:

- $`s`$ and $`u`$ grow by 1. If $`s \ge Q`$, `QTRAN` stops at once and `KMNS` sets IFAULT 4.
- A row whose cluster has one member is skipped.
- If $`s \le \nu_{a(i)}`$, $`D_i`$ is recomputed.
- If $`s \lt \nu_{a(i)}`$ or $`s \lt \nu_{b(i)}`$, the row moves when $`d_{b(i)}(i) \lt \mathrm{fl}(D_i / \alpha^+_{b(i)})`$. In exact arithmetic this is $`R_i \lt D_i`$. A move applies D4 and swaps $`a(i)`$ and $`b(i)`$. It also sets both $`\nu_c`$ to $`s + n`$ and sets $`u`$ to 0.
- `QTRAN` returns when $`u = n`$, after $`n`$ steps in a row without a move.

Step 8. End of the run. With $`k = 2`$, `KMNS` stops after the first `QTRAN`. ITER is 1 whether IFAULT is 0 or 4. So `iter.max = 1000` acts as 1. IFAULT 2 and the warning "did not converge in 1000 iterations" cannot occur.

Step 9. Final centers and WSS. `KMNS` recomputes each center from its rows: a sum in row order, divided by $`n_c`$. Then for each column $`j`$ in turn it adds $`(x_{ij} - \bar x_{a(i)j})^2`$ to the WSS of cluster $`a(i)`$, over the rows in order. The running centers of D4 are discarded.

Step 10. Faults. `kmeans` reads IFAULT in `do_one`:

- IFAULT 1 stops R with "empty cluster: try a better set of initial centers".
- IFAULT 3 stops R with "number of cluster centres must lie between 1 and nrow(x)".
- IFAULT 4 warns "Quick-TRANSfer stage steps exceeded maximum (= $`Q`$)" and keeps the run.
- IFAULT 2 raises ITER to at least `iter.max + 1`. `kmeans` then warns that it did not converge.

`refine.DN` (datanugget 1.5.0) calls `kmeans` inside `tryCatch` with an error handler and no warning handler. A stop is caught, and the nugget stays whole. Where a warning goes depends on `no.cores`. With `no.cores` at most 1, `run_splitting` runs the splits in the calling R session, and their warnings reach the caller. With `no.cores` above 1, it runs them through `foreach` and `%dopar%` on a cluster of R worker processes. The default `detectCores() - 1` is above 1 on a machine with 3 or more cores. Each worker evaluates a task in `parallel:::workCommand` (R 4.6.1), which catches errors and sends back only the value. So a warning raised on a worker does not reach the caller. `refine_dn` re-emits every warning as `DataNuggetWarning`, in candidate order, for every `n_jobs`.

Step 11. Restarts. With $`n_s \ge 2`$, `kmeans` runs starts 2 to $`\lfloor n_s \rfloor`$. Each one is drawn from `unique(x)` with `sample.int` and goes through steps 2 to 10. A later run replaces the best one only when its $`W`$ is strictly smaller. $`W`$ is R's `sum` of the two WSS values. So on a tie the earliest start wins. A stop in any run stops `kmeans`.

## Numerical notes

- `kmnsjit` performs the floating-point operations of `kmns.f` in the same order and in double. It uses no fused multiply-add. From the same starts it returns R's clusters, centers and WSS bit for bit.
- `QTRAN` stops summing $`d_{b(i)}(i)`$ as soon as a partial sum reaches the threshold. Partial sums of squares never decrease under rounding, so the early stop does not change the outcome. The port stops at the same term.
- `OPTRA` compares $`\mathrm{fl}(d_{b(i)}(i) \, \alpha^+_{b(i)})`$ with $`D_i`$. `QTRAN` compares $`d_{b(i)}(i)`$ with $`\mathrm{fl}(D_i / \alpha^+_{b(i)})`$. The two forms agree in exact arithmetic. In floating point they can disagree when the two sides lie within rounding of each other. The port keeps both forms.
- D4 updates the centers in place. After many moves the running centers can differ from the cluster means by rounding, and steps 5 and 7 use them as R does. Step 9 recomputes the centers, so the returned centers and WSS do not carry that drift.
- Step 11 compares $`W`$ as R's `sum`, which accumulates in long double. Source: R 4.6.1, `src/main/summary.c`, `rsum`. The port uses `r_sum` (`src/datanuggets/_compat/arithmetic.py`). [numerical-parity.md](numerical-parity.md) describes this sum.
- Steps 1 and 11 take their row numbers from `draws.sample`. With `RDraws` they are the draws of R's `sample.int` after `set.seed(seed)`. Other seeds give other starts, so the clusters can differ from R's. [randomness.md](randomness.md) covers the draws.
- Negating the rows and the starts negates every difference and leaves every $`d_c(i)`$ unchanged. The clusters stay the same and the centers are exactly negated. This matters in the shape pass, where LAPACK picks the sign of the eigenvector.

## Tests

The tests that compare with R read R's recorded results from `tests/fixtures/kmns/` and `tests/fixtures/kmeans/`.

- `tests/clustering/test_hartigan_wong.py::test_kmns_matches_r` runs `kmns` from R's recorded starts. It requires R's clusters, centers, sizes, WSS, ITER and IFAULT exactly. The cases with two clusters run on `faithful`, one-column `faithful` and `iris` with `iter.max = 1000` and $`Q = 50n`$. Other cases use 3 to 6 clusters on `faithful`, `iris`, `quakes` and `volcano`. They include a run cut off at `iter.max = 1` (IFAULT 2) and a run with $`Q = 50`$ (IFAULT 4).
- `tests/clustering/test_hartigan_wong.py::test_kmns_reports_the_faults_kmeans_stops_on` requires IFAULT 1 for two equal starts on `iris` and IFAULT 3 for one cluster.
- `tests/clustering/test_hartigan_wong.py::test_kmns_is_symmetric_under_negation` holds the negation note on one-column `faithful` with two clusters.
- `tests/clustering/test_hartigan_wong.py::test_kmns_kernel_matches_the_python_port` holds the compiled kernel bit for bit to `tests/clustering/kmns_reference.py`, a line-by-line port of `kmns.f` in plain Python. It uses grid and normal data with $`p`$ in 1, 2 and 5. Its cases cover 2 to 8 clusters, repeated starts, `iter.max` from 1 to 1000 and step budgets down to 1. Every IFAULT occurs.
- `tests/clustering/test_hartigan_wong.py::test_kmeans_matches_r` runs `kmeans` with R's recorded draws replayed and requires every draw to be used. It requires R's clusters, centers, WSS, sizes, total, ITER and warnings exactly. Its two-cluster cases use `iter.max = 10`. With two clusters step 8 holds for any `iter.max` of at least 1, so they run as they would with the 1000 of `refine_dn`. They use `faithful` with `nstart` 1, 25 and 25.7. They also use one-column `faithful` and the `iris` petal columns with `nstart` 25. A petal case with `nstart` 1 draws a repeated start. Other cases use 3 or 4 clusters. They run `nstart` 5 on `iris` and `nstart` 3 on `quakes`. Two cases with `nstart` 1 raise the two warnings of step 10: 4 clusters on `iris` with `iter.max = 1`, and 4 clusters on three-column data with many tied values, whose step budget runs out.
- `tests/clustering/test_hartigan_wong.py::test_kmeans_stops_where_r_stops` requires R's error for `nstart` 0, for 149 centers on the petal columns (fewer than 149 distinct rows) and for 5 centers on 5 rows.
- `tests/nuggets/test_refine.py::test_refine_matches_r` requires R's clusters for every k-means call `refine_dn` makes ([refine-dn.md](refine-dn.md#tests)). It requires R's input exactly in the EV pass. In the shape pass it requires the projected rows up to the eigenvector's sign, within `RTOL = 1e-8` and `ATOL = 1e-10` of `tests/rfixtures.py`. No R fixture of `refine.DN` warns. `tests/nuggets/test_refine.py::test_split_warnings_reach_the_caller_in_candidate_order` and `tests/nuggets/test_refine.py::test_split_nugget_carries_the_kmeans_warnings` hold step 10's warnings in `refine_dn` with a stub in place of `kmeans`.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. Algorithm 2, step 4.1.
- Hartigan, J. A., and Wong, M. A. (1979). "Algorithm AS 136: A K-Means Clustering Algorithm." Applied Statistics, 28(1), 100. doi:10.2307/2346830. Cited as R's `kmeans` help page cites it.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. Function `stats::kmeans` and its help page, `src/library/stats/src/kmns.f` (`KMNS`, `OPTRA`, `QTRAN`) and `src/main/summary.c` (`rsum`).
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0. Function `refine.DN`.

## Not covered by a test

- D3's reading of $`D_i`$ and $`R_i`$ as the change in $`W`$ is derived on this page. No test checks it.
- Step 10: no R fixture shows a `kmeans` warning reaching the caller of `refine.DN`. The R parallel path is read from the sources of `refine.DN` and `parallel`, not run. The stub tests of `refine_dn` run with `n_jobs = 1`.
- Step 11: no test case stops `kmeans` in a start after the first.
- Numerical notes: the early stop in `QTRAN` gives the outcome of a full sum because partial sums of squares never decrease. No test compares the two.
- Numerical notes: no test shows the `OPTRA` and `QTRAN` forms disagreeing.
