# Creating data nuggets

Draft. This page has not yet been reviewed by the authors of the R packages.

`create_dn` reduces an $`n \times p`$ data matrix to $`m`$ data nuggets. Each nugget has a center, a weight and a scale. The method is Algorithm 1 of Beavers et al. (2025) as datanugget 1.5.0 implements it in `create.DN`. Its deletion step is public as `create_dn_centers`. Where the paper and the R package differ, the port follows the R package.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `create_dn` | `src/datanuggets/nuggets/create.py`, `create_dn`, `_allocate` and `_summarize` | datanugget 1.5.0, `create.DN` |
| `create_dn_centers` | `src/datanuggets/nuggets/centers.py`, `create_dn_centers` and `select_centers`; `src/datanuggets/nuggets/centersjit.py`, `delete_rounds` | datanugget 1.5.0, `create.DNcenters` |
| chunking, internal | `src/datanuggets/_compat/indexing.py`, `split_indices` | R 4.6.1, parallel, `splitIndices` |
| assignment, internal | `src/datanuggets/_compat/rfast.py`, `nearest_index` | Rfast 2.1.5.2, `dista` with `k = 1` and `index = TRUE` |

Nugget ids and assignments are 0-based in the port. They are 1-based in R and on this page. The port has no `no.cores` or `make.pbs` argument. Instead `n_jobs` runs stage 1 in threads and `progress` shows a bar over the stage-1 chunks. The port raises on NaN or infinite values in $`X`$. It also raises when `dn_num1`, `dn_num2` or `dn_num` is not a whole number of at least 1, where R checks only that these are numeric.

## Notation

The shared symbols are in [README.md](README.md#notation). Here $`\mathrm{fl}(\cdot)`$ is an operation rounded to double, and D4 defines the weight $`w_l`$. This page adds the following.

| Symbol | Meaning | Argument |
| --- | --- | --- |
| $`r`$ | the chunk size the caller asks for | `r`, R's `R` |
| $`C`$ | the fraction deleted per round | `delete_percent` |
| $`m_1`$, $`m_2`$ | the stage-1 total and the stage-2 count, the paper's $`M_{init}`$ and $`M`$ | `dn_num1`, `dn_num2` |
| $`\pi`$ | the random permutation of the rows | |
| $`G'`$, $`n_g`$, $`t_g`$ | the number of non-empty chunks, the size of chunk $`g`$ and its target | |
| $`D`$ | the Euclidean or Manhattan distance | `dist_metric` |
| $`Z`$ | the stacked stage-1 centers | |
| $`\mathbf{y}_l`$ | stage-2 center $`l`$ | |
| $`a_i`$, $`I_l`$ | the nugget of row $`i`$, and the rows of nugget $`l`$ in increasing order | |
| $`\bar{\mathbf{x}}`$ | the column means of $`X`$ | |
| $`s_l`$, $`\mathbf{c}_l`$ | the scale and the final center of nugget $`l`$ | |
| $`\varphi`$, $`\varepsilon`$ | the scale floor, and the machine epsilon of a double | |

In D1, $`k`$ indexes the breaks and $`i`$ is a position in the shuffled order. README.md uses these two letters for the number of clusters and for a row. In D3, $`a`$ and $`b`$ are rows of the input to `create.DNcenters` and $`u`$ is the number of its rows that remain.

## Definitions

D1. Chunks. Let $`G = \lceil n / r \rceil`$. With $`G = 1`$ the one chunk holds every position. For $`G \ge 2`$ the breaks are the doubles that `parallel::splitIndices` gets from R's `seq(1 - f, n + f, length.out = G + 1)`:

```math
f = \min\lbrace \mathrm{fl}((n-1)/1000),\ \mathrm{fl}(\mathrm{fl}(\mathrm{fl}(0.4)\,n)/G) \rbrace, \qquad b_0 = \mathrm{fl}(1 - f), \qquad b_G = \mathrm{fl}(n + f),
```

```math
h = \mathrm{fl}(\mathrm{fl}(b_G - b_0)/G), \qquad b_k = \mathrm{fl}(b_0 + \mathrm{fl}(k\,h)) \quad (0 \lt k \lt G).
```

Position $`i`$ falls in chunk $`g`$ when $`b_{g-1} \lt i \le b_g`$. In exact arithmetic $`b_k = (1 - f) + k\,(n - 1 + 2f)/G`$. That value can be a whole number, as $`b_{G/2} = (n+1)/2`$ is for even $`G`$ and odd $`n`$, and the rounding then decides which chunk holds that position. For $`n = 101`$ and $`r = 100`$ the exact break $`b_1`$ is 51, which would give chunks of 51 and 50 positions. R's break is just below 51, so its chunks hold 50 and 51 positions. Empty chunks are dropped and $`G'`$ chunks remain. Under the check $`r \ge 100`$ of Step 1 no chunk is empty, so $`G' = G`$. Chunk $`g`$ holds the rows $`\mathbf{x}_{\pi(i)}`$ for its positions $`i`$, in increasing order of $`i`$.

D2. Targets. Let $`q = \lfloor m_1 / G' \rfloor`$ and $`e = m_1 - G' q`$. Chunk $`g`$ gets

```math
t_g = \max\lbrace 1,\ \min\lbrace q + [g \le e],\ n_g \rbrace \rbrace,
```

where $`[g \le e]`$ is 1 for the first $`e`$ chunks and 0 for the others.

D3. Deletion rounds. `create.DNcenters` takes $`u_0`$ rows and a target $`t`$. If $`u_0 \le t`$ it returns every row with the warning "DN.num is greater than or equal to the number of rows. Returning original data". Otherwise it computes the distance matrix once and sets its diagonal to $`\theta = \mathrm{fl}(\max_{a,b} D_{ab} + 1)`$. While $`u \gt t`$ rows remain, one round does the following.

1. Set $`\delta = \min\lbrace \max\lbrace 1, \lfloor \mathrm{fl}(u C) \rfloor \rbrace,\ u - t \rbrace`$.
2. For each remaining row $`a`$, let $`\nu(a)`$ be the first remaining row $`b`$ in the original order that minimizes $`D_{ab}`$. Here $`D_{ab}`$ is the double that `Rfast::Dist` returns and $`D_{aa} = \theta`$. Let $`d(a) = D_{a\nu(a)}`$. When $`\theta`$ exceeds every off-diagonal distance, $`\nu(a) \ne a`$.
3. Sort the remaining rows by $`d`$. Equal distances keep the original order.
4. Pass 1. Walk the sorted rows and mark $`a`$ when neither $`a`$ nor $`\nu(a)`$ is marked. Stop at $`\delta`$ marks.
5. Pass 2. If fewer than $`\delta`$ rows are marked, walk the sorted rows again and mark each unmarked row until there are $`\delta`$ marks.
6. Delete the marked rows.

Each round deletes at least one row and at most $`u - t`$, so exactly $`t`$ rows remain. They are returned in their original order.

D4. Assignment and weights. Row $`i`$ goes to $`a_i`$, the lowest-numbered $`l`$ that minimizes $`D(\mathbf{x}_i, \mathbf{y}_l)`$ as `Rfast::dista` computes it in double. Then $`I_l = \lbrace i : a_i = l \rbrace`$ and $`w_l = \lvert I_l \rvert`$. A nugget can have $`w_l = 0`$.

D5. Scales. With $`\tilde w_l = \max(w_l, 1)`$ the sums are taken on data centred at $`\bar{\mathbf{x}}`$:

```math
\begin{aligned}
S^{(1)}_{lj} &= \sum_{i \in I_l} (x_{ij} - \bar x_j), \qquad S^{(2)}_{lj} = \sum_{i \in I_l} (x_{ij} - \bar x_j)^2, \\
V_{lj} &= \frac{\max\lbrace 0,\ S^{(2)}_{lj} - \tilde w_l\,(S^{(1)}_{lj} / \tilde w_l)^2 \rbrace}{\max(w_l - 1, 1)}, \qquad s_l = \frac{1}{p} \sum_{j=1}^{p} V_{lj} \quad (w_l \ge 2).
\end{aligned}
```

For $`w_l \ge 2`$ and in exact arithmetic, $`V_{lj}`$ is the sample variance of column $`j`$ over $`I_l`$. So $`s_l`$ is the trace of the sample covariance of the nugget's rows divided by $`p`$. That is the scale of Beavers et al. (2025), Algorithm 1, step 7. A nugget with $`w_l \lt 2`$ gets its scale from D7.

D6. Centers. The three values of `center_method` give the following.

- `"mean"`: $`\mathbf{c}_l = S^{(1)}_{l\cdot} / \tilde w_l + \bar{\mathbf{x}}`$ when $`w_l \ge 1`$. In exact arithmetic this is the mean of the nugget's rows. An empty nugget keeps $`\mathbf{c}_l = \mathbf{y}_l`$.
- `"original"`: $`\mathbf{c}_l = \mathbf{y}_l`$.
- `"random"`: for each non-empty nugget in the order $`l = 1, \dots, m`$, draw $`v`$ uniformly from $`\lbrace 1, \dots, w_l \rbrace`$ with R's `sample.int` and set $`\mathbf{c}_l`$ to the $`v`$-th row of $`I_l`$. An empty nugget keeps $`\mathbf{y}_l`$.

D7. Floor. Let $`\mathcal{S} = \lbrace s_l : w_l \ge 2 \text{ and } s_l \text{ finite} \rbrace`$. Then $`\varphi = \min \mathcal{S} / 100`$ when $`\mathcal{S}`$ is not empty and $`\varphi = \varepsilon`$ otherwise, with $`\varepsilon = 2^{-52}`$. If $`\varphi`$ is not finite or $`\varphi \le 0`$, it becomes $`\varepsilon`$. Every scale that is not finite is then set to $`\varphi`$. That covers each nugget with $`w_l \lt 2`$ and any multi-row nugget whose sums exceeded the range of a double.

## Algorithm

Step 1. Checks. $`r`$ must lie in $`[100, 10000]`$ and $`C`$ in $`(0, 1)`$. Also $`m_2 \le m_1`$ and $`n \ge 2`$.

Step 2. Permutation. Draw a random permutation $`\pi`$ of the $`n`$ row numbers. R calls `set.seed(seed)` and then `sample(n, n)`.

Step 3. Chunks. Split the positions into $`G'`$ chunks by D1.

Step 4. Targets. Give chunk $`g`$ the target $`t_g`$ by D2.

Step 5. Stage 1. Reduce the rows of each chunk to $`t_g`$ rows by D3. A chunk whose target equals its size passes through with the warning. Stack the results in chunk order into $`Z`$.

Step 6. Stage 2. Reduce $`Z`$ to $`m_2`$ rows by D3. These are $`\mathbf{y}_1, \dots, \mathbf{y}_m`$. If $`Z`$ has at most $`m_2`$ rows, all of them pass through with the warning and $`m`$ is the number of rows of $`Z`$.

Step 7. Assignment. Assign every row of $`X`$ to a nugget and count the weights by D4.

Step 8. Scales. Compute $`s_l`$ for each nugget with $`w_l \ge 2`$ by D5.

Step 9. Centers. Compute $`\mathbf{c}_l`$ by D6. Only `"random"` draws more random numbers.

Step 10. Floor. Set the missing scales to $`\varphi`$ by D7.

The result is the centers $`\mathbf{c}_l`$, the weights $`w_l`$, the scales $`s_l`$ and the assignments $`a_i`$.

## Numerical notes

Order of operations. The port follows R's arithmetic so that its output equals R's. $`\bar{\mathbf{x}}`$ and $`s_l`$ use the arithmetic of R's `colMeans` and `rowMeans`, which [numerical-parity.md](numerical-parity.md) describes along with the distances of Rfast's `Dist` and `dista`. $`S^{(1)}`$ and $`S^{(2)}`$ add the rows of each nugget in row order in double precision. R does the same in R 4.6.1, `src/main/unique.c`, `rowsum`.

Cancellation. D5 subtracts two nearly equal numbers when the rows of a nugget lie close together far from $`\bar{\mathbf{x}}`$. Digits are lost there, and the clamp at 0 keeps $`V_{lj}`$ from going negative. The port keeps R's one-pass formula so that its scales match R's.

Ties in the deletion rounds. $`\nu(a)`$ is the first minimizer, as `Rfast::rowMins` returns it. The sort in D3 is stable, as R's `order` is. The port keeps one distance matrix and the remaining rows in their original order. After the first round it searches again only the rows whose nearest neighbour was deleted. A row's nearest remaining neighbour changes only when that neighbour is deleted, so the result equals that of R's full search.

The diagonal. $`\theta`$ exceeds every off-diagonal distance when $`\mathrm{fl}(\max D + 1) \gt \max D`$. That holds for $`\max D \lt 2^{53}`$.

Ties in the assignment. The port takes the lowest-numbered of exactly tied centers, and Rfast's `dista` can take another of them. In the fixtures, every row where R's pick differs from the lowest-numbered tied center has an exact tie and more than 16 centers in all. On those rows both picks are at the same distance. Where the port's own assignment differs from R's, its weights differ from R's. Its scales can differ too, and so can its centers under `"mean"` and `"random"`. Under `"original"` the centers are the stage-2 centers whatever the assignment (D6).

Random numbers. Only `seed=RDraws(s)` reproduces the permutation and the `"random"` draws of R's `seed = s`. An integer seed draws from NumPy, as [randomness.md](randomness.md) describes. The chunk reductions draw no random numbers, so neither `n_jobs` nor `progress` changes the result.

## Tests

The fixtures in `tests/fixtures/create_dn` hold the output of R 4.6.1 with datanugget 1.5.0 and Rfast 2.1.5.2. They cover both distances, the three center methods, $`C`$ of 0.1 and 0.3, one chunk and several, a target remainder, a stage-1 total smaller than the number of chunks, chunks and a stage 2 that pass through, a single nugget, an empty nugget, duplicated rows, grid ties, one column, integer data and three seeds. Each case records every random draw and every input, target and result of `create.DNcenters`.

- `tests/nuggets/test_create.py::test_create_dn_hands_select_centers_what_r_hands_create_dncenters`: Steps 2 to 6 give every call of D3 the input rows and target that R gives it.
- `tests/nuggets/test_centers.py::test_create_dn_centers_matches_r`: D3 keeps R's rows on every recorded call.
- `tests/nuggets/test_create.py::test_create_dn_matches_r`: Steps 7 to 10. It carries on from R's assignments and holds the weights, centers and scales equal to R's values. Each draw the port asks for must match R's recorded draw at that place in population size and sample size, and the port must use every recorded draw. The port must raise as many warnings as R. The test also checks the port's own assignments against R's with the tie rule above.
- `tests/compat/test_rcompat.py::test_split_indices_matches_r`: the chunk sizes of D1 against R's `splitIndices`.
- `tests/nuggets/test_create.py::test_allocate_splits_dn_num1_like_r`: D2 on constructed chunk sizes.
- `tests/compat/test_rfast.py::test_row_min_first_index_matches_rfast` and `tests/nuggets/test_centers.py::test_row_min_is_numpys_first_argmin_over_the_active_rows`: the first minimizer in D3.
- `tests/nuggets/test_centers.py::test_select_centers_matches_the_python_rounds`: the compiled rounds against a direct transcription of R's rounds, on grid data and duplicated rows full of ties and on normal data.
- `tests/nuggets/test_centers.py::test_returns_input_with_warning_when_dn_num_covers_all_rows`: the pass-through of D3.
- `tests/compat/test_rfast.py::test_nearest_index_matches_rfast`, `tests/compat/test_rfast.py::test_r_parts_from_the_lowest_index_only_on_wide_ties` and `tests/compat/test_rfast.py::test_exact_ties_go_to_the_lowest_center`: D4 and its tie rule.
- `tests/randomness/test_rdraws.py::test_every_recorded_create_dn_draw`: `RDraws` gives R's draws.
- `tests/nuggets/test_create.py::test_n_jobs_does_not_change_result`: `n_jobs` does not change the centers, weights, scales or assignments.
- `tests/nuggets/test_create.py::test_progress_does_not_change_result`: `progress` does not change the assignments or the centers.
- `tests/nuggets/test_create.py::test_more_nuggets_than_rows_keeps_every_row`: when Steps 5 and 6 both pass through, every row becomes a nugget and every scale is finite.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`.
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0.
- Papadakis, M., Tsagris, M., Dimitriadis, M., Fafalios, S., Fasiolo, M., Borboudakis, G., Burkardt, J., and Zou, C. (2025). Rfast: A Collection of Efficient and Extremely Fast R Functions. R package version 2.1.5.2.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria.

## Not covered by a test

- D7 when no nugget has two or more rows, so that $`\varphi = \varepsilon`$. `tests/nuggets/test_create.py::test_more_nuggets_than_rows_keeps_every_row` reaches it and checks only that the scales are finite.
- D7 for a nugget with two or more rows whose sums exceed the range of a double.
- D6 for an empty nugget under `"random"`. The one fixture with an empty nugget, `iris_all_passthrough`, uses `"mean"`.
- The diagonal note when $`\max D \ge 2^{53}`$. No test of D3 has distances that large.
