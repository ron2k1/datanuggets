# Refining data nuggets

Draft. This page has not yet been reviewed by the authors of the R packages.

`refine_dn` takes data and an assignment of every row to a nugget. It recomputes each nugget from its rows. Then it splits in two, by k-means, the nuggets whose first eigenvalue is large. With `shape_split=True` a second pass splits elongated or spread-out nuggets (D7) along their first eigenvector. Each split runs the k-means of [hartigan-wong.md](hartigan-wong.md).

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `refine_dn` | `src/datanuggets/nuggets/refine.py`, `refine_dn` | datanugget 1.5.0, `refine.DN` |
| `_split_nugget`, `_shape_ratio` | `src/datanuggets/nuggets/refine.py`, `_split_nugget` and `_shape_ratio` | datanugget 1.5.0, `refine.DN`, inner `split_nugget` and the shape check |
| `_shape_candidates` | `src/datanuggets/nuggets/refine.py`, `_shape_candidates` | datanugget 1.5.0, `refine.DN`, the shape pass |
| `_Nuggets.refresh` | `src/datanuggets/nuggets/refine.py`, `_Nuggets.refresh` | datanugget 1.5.0, `refine.DN`, inner `get_nugget_features` and `refresh_nuggets_info` |
| `_Nuggets.apply`, `_Nuggets.finalize` | `src/datanuggets/nuggets/refine.py`, `_Nuggets.apply` and `_Nuggets.finalize` | datanugget 1.5.0, `refine.DN`, inner `apply_splits` and the final renumbering |
| `r_quantile7`, `r_median`, `r_mad` | `src/datanuggets/_compat/arithmetic.py`, same names | R 4.6.1, `stats::quantile` (type 7), `stats::median`, `stats::mad` |
| `r_eigen_sym`, `r_matvec` | `src/datanuggets/_compat/linalg.py`, same names | R 4.6.1, `base::eigen` with `symmetric = TRUE`, and `%*%` |

## Notation

The shared symbols, such as $`n`$, $`p`$, $`m`$, $`\mathbf{x}_i`$ and the nugget weight $`w_l`$, are in [README.md](README.md#notation). Every row counts once, so $`w_l`$ is the number of rows of nugget $`l`$. This page adds:

| Symbol | Meaning | R | Python |
| --- | --- | --- | --- |
| $`a(i)`$ | the nugget of row $`i`$ | `DN.assignments` | `assignments` |
| $`I_l`$ | the rows of nugget $`l`$, in increasing order | `tmp.members[[l]]` | `members()[l]` |
| $`m_0`$ | the number of input nuggets | `length(DN.ids)` at the start | `dn.centers.shape[0]` |
| $`\bar{\mathbf{x}}_l`$, $`S_l`$, $`s_l`$ | center, sample covariance and scale of nugget $`l`$ | `DN.centers`, `S`, `DN.scale` | `centers`, `cov`, `scales` |
| $`\lambda_{1,l}`$, $`\lambda_{2,l}`$ | EV1 and EV2 of nugget $`l`$ (D3) | `DN.EV1`, `DN.EV2` | `ev1`, `ev2` |
| $`\mathbf{v}_l`$ | the unit eigenvector of $`\lambda_{1,l}`$ that LAPACK returns | first column of the `eigen` vectors | `vectors[:, 0]` |
| $`\alpha`$ | the split percentile | `EV.tol` | `ev_tol` |
| $`T`$ | the round limit, `max.splits` rounded down | `max.splits` | `max_splits` |
| $`N_{\min}`$ | the smallest child, `min.nugget.size` rounded down | `min.nugget.size` | `mns` |
| $`s_{\min}`$ | the smallest weight for a shape split, not rounded | `min.shape.size` | `min_shape_size` |
| $`\delta`$ | the elongation bound | `delta` | `delta` |
| $`\tau`$, $`\tau_s`$ | the EV threshold and the spread threshold | `EV.threshold`, `s.threshold` | `threshold`, `s_threshold` |
| $`\rho_l`$ | the ratio of nugget $`l`$ (D7) | `tmp.ratio` | `ratio` |
| $`G_1`$, $`G_2`$ | the two children of a split | `cluster == 1`, `cluster == 2` | `labels == 0`, `labels == 1` |

## Definitions

D1. Members.

```math
I_l = \lbrace i : a(i) = l \rbrace, \qquad w_l = \lvert I_l \rvert .
```

R builds the sets with `split(seq_len(n), factor(DN.assignments, levels = DN.ids))`. The port uses a stable sort (`_Nuggets.members`).

D2. Center, covariance and scale. For $`w_l \ge 2`$:

```math
\bar{\mathbf{x}}_l = \frac{1}{w_l} \sum_{i \in I_l} \mathbf{x}_i, \qquad S_l = \frac{1}{w_l - 1} \sum_{i \in I_l} (\mathbf{x}_i - \bar{\mathbf{x}}_l)(\mathbf{x}_i - \bar{\mathbf{x}}_l)^\top, \qquad s_l = \frac{\operatorname{tr} S_l}{p} .
```

A nugget with one row has that row as its center and $`s_l = 0`$. An empty nugget has $`s_l = 0`$ and a NaN center, R's `NA`. This is the scale of Algorithm 1 in Beavers et al. (2025). Its step 7 also sets the scale of a one-row nugget to 0.

D3. EV1 and EV2. For $`w_l \ge 2`$ and $`p \ge 2`$, $`\lambda_{1,l} \ge \lambda_{2,l}`$ are the two largest eigenvalues of $`S_l`$. For $`w_l \ge 2`$ and $`p = 1`$, no eigen routine runs and $`\lambda_{1,l} = \lambda_{2,l} = S_l`$, which equals $`s_l`$. For $`w_l \le 1`$, $`\lambda_{1,l} = \lambda_{2,l} = 0`$.

D4. Type-7 quantile. For sorted values $`e_{(1)} \le \dots \le e_{(r)}`$ and $`\alpha \in (0, 1)`$:

```math
h = 1 + (r - 1) \alpha, \qquad \gamma = h - \lfloor h \rfloor, \qquad Q_7(\alpha) = (1 - \gamma) \, e_{(\lfloor h \rfloor)} + \gamma \, e_{(\lceil h \rceil)} .
```

When $`\gamma = 0`$ or the two order statistics are equal, the code returns $`e_{(\lfloor h \rfloor)}`$ without interpolating. Source: R 4.6.1, `stats::quantile` (`src/library/stats/R/quantile.R`), and `r_quantile7`.

D5. EV threshold. After step 2, let $`E`$ list the finite positive $`\lambda_{1,l}`$ of the input nuggets, repeated values kept:

```math
E = \bigl( \lambda_{1,l} : 1 \le l \le m_0, \ \lambda_{1,l} \text{ finite}, \ \lambda_{1,l} \gt 0 \bigr), \qquad \tau = Q_7(\alpha) \text{ over } E .
```

$`\tau`$ is computed once, before the first round.

D6. EV candidates. In each round of the EV pass:

```math
C = \lbrace l : \lambda_{1,l} \text{ finite}, \ \lambda_{1,l} \gt \tau, \ w_l \ge 2N_{\min} \rbrace .
```

D7. Shape candidates. Let $`L = \lbrace l : s_l \text{ finite}, \ w_l \gt 1 \rbrace`$, one entry per nugget. With R's median, which averages the two middle values of an even count:

```math
\tau_s = \operatorname{med}_{l \in L} s_l + 2 \operatorname{mad}_{l \in L} s_l, \qquad \operatorname{mad}_{l \in L} s_l = 1.4826 \cdot \operatorname{med}_{l \in L} \bigl\lvert s_l - \operatorname{med}_{l' \in L} s_{l'} \bigr\rvert ,
```

and $`\tau_s = +\infty`$ when $`L`$ is empty. $`\rho_l = \lambda_{1,l} / \lambda_{2,l}`$ when $`\lambda_{2,l}`$ is finite and positive, and $`\rho_l = +\infty`$ otherwise. Then

```math
C^{\mathrm{sh}} = \lbrace l : w_l \ge \max(s_{\min}, 2N_{\min}), \ \lambda_{1,l} \text{ finite}, \ \lambda_{1,l} \gt 0, \ (\rho_l \gt \delta \text{ or } \lambda_{1,l} \gt \tau_s) \rbrace .
```

$`\tau_s`$ and $`\rho_l`$ are recomputed at every shape round.

D8. The split rule. A candidate's rows go to k-means as $`\mathbf{y}_i = \mathbf{x}_i`$ in the EV pass and as $`y_i = \mathbf{x}_i^\top \mathbf{v}_l`$ in the shape pass, for $`i \in I_l`$. In the shape pass $`\mathbf{v}_l`$ comes from `eigen` of the covariance of those rows. k-means runs with 2 centers, `iter.max = 1000` and `nstart`, right after `set.seed(seed)`. It returns $`G_1`$ and $`G_2`$. The split is kept when k-means returns without an error and

```math
\min \bigl( \lvert G_1 \rvert, \lvert G_2 \rvert \bigr) \ge N_{\min} .
```

In the shape pass the split must also satisfy

```math
\tfrac{1}{2} \bigl( \rho(G_1) + \rho(G_2) \bigr) \lt \delta ,
```

where $`\rho(G) = \mu_1(G) / \mu_2(G)`$ for the two largest eigenvalues of the sample covariance of the rows $`\mathbf{x}_i`$, $`i \in G`$. $`\rho(G) = +\infty`$ when $`\mu_2(G) \le 0`$. R also rejects the split when a child has fewer than 2 rows, where its ratio is `NA`. Both children of a kept split have at least $`N_{\min} \ge 2`$ rows, so this case does not arise.

D9. Applying a round. Every split of a round is computed from the state at the start of the round. Let $`K`$ be the number of positions before the round, empty ones included. Number the kept splits $`u = 1, \dots, q`$ in candidate order. The rows of $`G_2`$ of split $`u`$ move to the new position $`K + u`$. The rows of $`G_1`$ stay at the parent. D2 and D3 are recomputed for each parent and each new position, and only for those. No other position changes its rows.

## Algorithm

Step 1. Arguments. Among its checks, R requires $`\alpha \in (0, 1)`$, `max.splits` $`\ge 0`$, `min.nugget.size` $`\ge 2`$, $`s_{\min} \ge 10`$, $`\delta \gt 1`$ and a numeric `nstart`. The port also requires `max_splits` and `nstart` to be finite. Then $`T = \lfloor \text{max.splits} \rfloor`$ and $`N_{\min} = \lfloor \text{min.nugget.size} \rfloor`$.

Step 2. Features. Compute D1 to D3 for every input nugget from the assignments. The input's centers, weights and scales are not used.

Step 3. Threshold. If $`T = 0`$, go to step 6. If $`E`$ is empty, log "No data nugget has a positive first eigen value" and skip step 4. Otherwise compute $`\tau`$ (D5).

Step 4. EV pass. For rounds $`t = 1, \dots, T`$:

- Form $`C`$ (D6). Stop the pass if it is empty.
- Split each candidate, in increasing position order, with $`\mathbf{y}_i = \mathbf{x}_i`$ (D8).
- Stop the pass if no split is kept. Otherwise apply the kept splits (D9).

The note after Algorithm 2 of Beavers et al. (2025) says that "a limit can be placed on the number of times steps 2, 3, and/or 4 are executed". Here the limit is $`T`$ rounds.

Step 5. Shape pass. It runs only when `shape_split` is true, $`p \ge 2`$ and $`T \gt 0`$. For rounds $`t = 1, \dots, T`$ it repeats step 4 with $`C^{\mathrm{sh}}`$ (D7) and $`y_i = \mathbf{x}_i^\top \mathbf{v}_l`$, and it also applies the ratio rule of D8.

Step 6. Renumber. Drop every position with $`w_l = 0`$. Number the rest in position order: the input nuggets first, then the new ones in the order they were made. Set $`s_l = 0`$ where $`w_l \lt 2`$, which by D2 changes nothing. Check that the weights count the assignments. Return the centers, weights, scales and assignments.

## Numerical notes

- Centers are R's `colMeans`: a long double sum per column, divided in long double and rounded to double once. Source: R 4.6.1, `src/main/array.c`, `do_colsum`.
- $`S_l`$ is R's `stats::cov`. Its column means take two long double passes and are stored as doubles. The sums of products are long double, divided by $`w_l - 1`$. So the mean inside $`S_l`$ can differ from $`\bar{\mathbf{x}}_l`$ in the last bit. Source: R 4.6.1, `src/library/stats/src/cov.c`.
- $`s_l`$ sums the diagonal of $`S_l`$ in long double, as R's `sum` does. The port replays all three in `r_mean`, `r_cov` and `r_sum` (`src/datanuggets/_compat/arithmetic.py`) and in `group_moments` (`src/datanuggets/_compat/moments.py`). [numerical-parity.md](numerical-parity.md) describes this arithmetic.
- R's `eigen(S, symmetric = TRUE)` calls LAPACK `dsyevr` with range "A", the lower triangle and `abstol` 0. Source: R 4.6.1, `src/modules/lapack/Lapack.c`, `La_rs`. `r_eigen_sym` calls SciPy's `dsyevr` with the same arguments and reverses the output to decreasing order.
- The two LAPACK builds differ. For $`p \ge 2`$ the eigenvalues agree with R's to rounding but not always bit for bit. The tests hold them within a relative 1e-8 plus an absolute 1e-10 (`RTOL` and `ATOL` in `tests/rfixtures.py`). A candidate test or a ratio test whose two sides lie within that rounding of each other can therefore come out differently from R's. On every fixture the outcomes match R's (see Tests). For $`p = 1`$ no eigen routine runs, and $`\lambda_{1,l}`$ equals R's `cov` value bit for bit.
- `r_matvec` forms the projections of the nugget's rows on $`\mathbf{v}_l`$, summing column by column in the order of R's reference BLAS. $`\mathbf{v}_l`$ itself carries the eigen tolerance. LAPACK picks the sign of $`\mathbf{v}_l`$. With $`-\mathbf{v}_l`$ the projections and the drawn starts are negated, and k-means returns the same clusters. When $`\lambda_{1,l}`$ is a repeated eigenvalue, $`\mathbf{v}_l`$ is not unique and the projection depends on the vector each LAPACK build returns.
- R calls `set.seed(seed)` before every k-means call, so every split draws from the start of the same stream. The port gives each candidate `draws.fork()`, made in candidate order in the main process. `RDraws(seed)` restarts R's Mersenne-Twister and gives R's draws. An int seed restarts NumPy's stream, and a NumPy `Generator` spawns one child stream per candidate. Neither gives R's draws, so the splits can differ from R's. The default `seed=291102` is an int. Any other `Draws` object, such as the recorded replay the tests use, needs `n_jobs=1`. Forks and outcomes follow candidate order, so `n_jobs` does not change the result. [randomness.md](randomness.md) covers the draws.

## Tests

- `tests/nuggets/test_refine.py::test_refine_matches_r` runs every case in `tests/fixtures/refine_dn/` with R's recorded draws replayed. For every k-means call it requires R's input and R's clusters. EV-pass inputs are compared exactly. Shape-pass projections are compared within the eigen tolerance after the sign is aligned. Then it requires R's assignments, weights, centers and scales exactly. It also requires R's messages and warnings. This holds steps 2 to 6 and D1 to D9.
- The fixtures cover these inputs:
  - a `faithful` grid over $`\alpha \in \lbrace 0.5, 0.9 \rbrace`$, $`T \in \lbrace 0, 1, 5 \rbrace`$, $`N_{\min} \in \lbrace 2, 5 \rbrace`$ and $`\delta \in \lbrace 1.5, 2 \rbrace`$, with the shape split off and on and with `nstart` 1 and 25.
  - `iris`, its petal columns, `quakes`, scaled `volcano`, tied grid data and one-column `faithful`. Each runs with the defaults, with `nstart` 1, with a shape split and with a looser shape split. For one-column `faithful` the shape split cases check that the shape pass does not run (step 5).
  - a petal case with `nstart` 1 in which one k-means call draws a repeated start.
  - `faithful` with an empty input nugget, with one-row nuggets, with fractional arguments, with `nstart` 0, with two other input nugget sets and with a refined input refined again.
- `tests/nuggets/test_refine.py::test_bad_arguments_are_rejected` holds the argument checks of step 1, except the check that `nstart` is a number.
- `tests/nuggets/test_refine.py::test_initial_nugget_features_match_r` holds step 2 with `n_jobs` 1 and 4. Centers and scales match R's exactly. Eigenvalues match exactly for $`p = 1`$ and within the tolerance otherwise.
- `tests/nuggets/test_refine.py::test_every_rejection_path_is_exercised` shows the fixtures reach each rejection of D8: a k-means error, a child below $`N_{\min}`$ and a shape split rejected on its children's ratios.
- `tests/randomness/test_rdraws.py::test_refine_dn_with_rdraws_matches_r` runs every fixture with `RDraws` instead of the replay and requires R's outputs exactly. `tests/randomness/test_rdraws.py::test_fork_restarts_the_seed_like_set_seed` holds the restart.
- `tests/compat/test_rstats.py::test_vector_stats_match_r_bitwise` holds `sum`, `mean`, `median`, `mad` and the type-7 quantile to R bit for bit (D4, D7). `tests/compat/test_rstats.py::test_cov_and_scale_match_r_bitwise` holds $`S_l`$ and $`s_l`$ of D2 bit for bit. `tests/compat/test_rstats.py::test_eigen_matches_r_numerically` holds D3 within the tolerance. `tests/compat/test_rstats.py::test_matvec_matches_r_bitwise` holds the projection bit for bit.
- `tests/clustering/test_hartigan_wong.py::test_kmns_is_symmetric_under_negation` holds the sign statement on a recorded one-column case.
- No R fixture of `refine.DN` warns. `tests/nuggets/test_refine.py::test_split_warnings_reach_the_caller_in_candidate_order` uses a stub in place of `kmeans` to show that k-means warnings reach the caller as `DataNuggetWarning`, in candidate order.
- `tests/nuggets/test_refine.py::test_n_jobs_does_not_change_the_result` (int seed and `Generator`) and `tests/randomness/test_rdraws.py::test_rdraws_splits_in_parallel_like_serial` (`RDraws`) hold the `n_jobs` statement.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. Section 3, Algorithm 1, step 7, and Algorithm 2.
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0. Function `refine.DN` and its help page.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. Functions `stats::quantile`, `stats::median`, `stats::mad`, `stats::cov`, `stats::kmeans`, `colMeans`, `sum`, `%*%`, `set.seed` and `base::eigen`.

## Not covered by a test

- The rounding in step 1. The only fixture with fractional arguments passes `max.splits` 2.7, `min.nugget.size` 2.5 and `min.shape.size` 10.5. On that case the candidates, the kept splits and the number of rounds stay the same when these values are rounded down, rounded up or left as they are. So no test tells $`T = \lfloor \text{max.splits} \rfloor`$ and $`N_{\min} = \lfloor \text{min.nugget.size} \rfloor`$ from other roundings, or an unrounded $`s_{\min}`$ from a rounded one.
- The check that `nstart` is a number (step 1).
