# Randomness

Draft. This page has not yet been reviewed by the authors of the R packages.

Three of the ported functions draw random numbers: `create_dn`, `refine_dn` and `wkmeans`. `dn_wkmeans` draws through `wkmeans`. Each takes a `seed=` argument. The t-SNE helpers `tsne_2d` and `tsne_3d` of `datanuggets.projection` take a `seed=` of their own, which they pass to scikit-learn as `random_state`. This page does not cover them. An integer or a NumPy `Generator` draws from NumPy. An `RDraws(seed)` draws what R 4.6.1 draws after `set.seed(seed)` under R's default generator, so the port makes the same random choices as the R packages. This page defines `RDraws` and lists where each function draws. The paper names no random number generator and no seed.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `RDraws(seed)` | `src/datanuggets/_random/r_draws.py`, `RDraws` | `set.seed(seed)`, R 4.6.1 (base) |
| `RDraws.sample(n, size)`, `.permutation(n)`, `.index(m)` | `src/datanuggets/_random/r_draws.py`, `RDraws.sample` | `sample.int(n, size)`, `sample(n, n)`, `sample.int(m, 1L)`, R 4.6.1 (base) |
| `RDraws.sample_replace(n, size)` | `src/datanuggets/_random/r_draws.py`, `RDraws.sample_replace` | `sample.int(n, size, replace = TRUE)`, R 4.6.1 (base) |
| `RDraws.fork()` | `src/datanuggets/_random/r_draws.py`, `RDraws.fork` | `set.seed(seed)` called again, R 4.6.1 (base) |
| `RDraws.random_seed()` | `src/datanuggets/_random/r_draws.py`, `RDraws.random_seed` | `.Random.seed`, R 4.6.1 (base) |
| `NumpyDraws(seed)` | `src/datanuggets/_random/numpy_draws.py`, `NumpyDraws` | none |
| `as_draws(seed)` | `src/datanuggets/_random/seeds.py`, `as_draws` | none |
| `Draws` | `src/datanuggets/_random/interfaces.py`, `Draws` | none |
| `ReplayDraws` | `src/datanuggets/_random/replay.py`, `ReplayDraws` | none: it replays draws that R recorded |

`RDraws` is exported as `datanuggets.RDraws`. `src/datanuggets/_draws.py` re-exports these names under an older import path.

## Notation

The shared symbols are in [README.md](README.md#notation). This page adds:

| Symbol | Meaning |
| --- | --- |
| $`s`$ | the seed after `set.seed` converts it to an integer |
| $`z_t`$ | value $`t`$ of the seeding congruential sequence, an unsigned 32-bit integer |
| $`\mathrm{mt}[a]`$ | word $`a`$ of the Mersenne-Twister state, $`a = 0, \dots, 623`$ |
| $`h_a`$ | the word that D3 builds from $`\mathrm{mt}[a]`$ and the next state word |
| $`y`$ | a 32-bit word read from the state, then tempered (D3) |
| $`U`$ | a uniform value from R's `unif_rand` |
| $`\epsilon`$ | R's constant `i2_32m1`, close to $`(2^{32} - 1)^{-1}`$ (D4). It is not the machine epsilon $`\varepsilon`$ of README.md |
| $`N`$, $`q`$ | the population and the sample size of one `sample.int(N, q)` call |
| $`b`$ | the number of low bits kept from each attempt of one index draw (D5) |
| $`L`$ | the number of uniforms one attempt draws (D5) |
| $`v`$ | the value of one attempt, kept when $`v \lt N`$ (D5) |
| $`N'`$, $`d`$ | the length of the remaining pool and a position drawn in it (D6) |
| $`\oplus`$, $`\wedge`$, $`\vee`$ | bitwise exclusive or, and, or on 32-bit words |
| $`\ll`$, $`\gg`$ | left and right shifts of a 32-bit word |

## Definitions

D1. Seed. `set.seed(seed)` converts `seed` with `asInteger` and stops with "supplied seed is not a valid integer" when the result is `NA` (R 4.6.1, `src/main/RNG.c`, `do_setseed`). For a double, `asInteger` gives `NA` for NaN and for any value outside $`-2^{31} \lt x \lt 2^{31}`$. It warns in the second case and otherwise truncates toward zero (R 4.6.1, `src/main/coerce.c`, `IntegerFromReal`). `_r_seed` in `src/datanuggets/_random/r_draws.py` does the same. It refuses a `bool` and a string, which R coerces: `set.seed(TRUE)` seeds with 1 and `set.seed("12")` with 12. It also refuses `None`. R's `set.seed(NULL)` skips `asInteger` and calls `TimeToSeed` for a new seed (`do_setseed`).

D2. Seeding. Let $`z_0 = s \bmod 2^{32}`$ and $`z_{t+1} = (69069\, z_t + 1) \bmod 2^{32}`$. `RNG_Init` discards $`z_1, \dots, z_{50}`$ and stores $`z_{51}, \dots, z_{675}`$ in the 625 seed words. `FixupSeeds` then sets the first word, the position, to 624. So

```math
\mathrm{mt}[a] = z_{52+a}, \qquad a = 0, \dots, 623,
```

and the first draw regenerates the whole state. `.Random.seed` holds 10403, the position and the 624 words as signed 32-bit integers. The code 10403 stands for the kinds Mersenne-Twister, Inversion and Rejection. `PutRNGstate` writes it as the generator kind plus 100 times the normal kind plus 10000 times the sample kind. R: `src/main/RNG.c`, `RNG_Init`, `FixupSeeds` and `PutRNGstate`. Python: `RDraws.__init__`, `RDraws.random_seed`.

D3. State update and tempering. The generator is the Mersenne Twister of Matsumoto and Nishimura (1998), which `src/main/RNG.c` cites. When a draw finds the position at 624, `MT_genrand` first rewrites the state in place for $`a = 0, 1, \dots, 623`$ in that order:

```math
\begin{aligned}
h_a &= (\mathrm{mt}[a] \wedge \mathtt{0x80000000}) \vee (\mathrm{mt}[(a+1) \bmod 624] \wedge \mathtt{0x7fffffff}), \\
\mathrm{mt}[a] &\leftarrow \mathrm{mt}[(a+397) \bmod 624] \oplus (h_a \gg 1) \oplus (\mathtt{0x9908b0df} \cdot (h_a \wedge 1)).
\end{aligned}
```

Each draw reads the next word $`y`$ and tempers it:

```math
\begin{aligned}
y &\leftarrow y \oplus (y \gg 11), \\
y &\leftarrow y \oplus ((y \ll 7) \wedge \mathtt{0x9d2c5680}), \\
y &\leftarrow y \oplus ((y \ll 15) \wedge \mathtt{0xefc60000}), \\
y &\leftarrow y \oplus (y \gg 18).
\end{aligned}
```

R: `MT_genrand`. The port reads the tempered words from NumPy's `MT19937` bit generator (`random_raw`), started from the state of D2 at position 624. The tests compare the state after each call with R's `.Random.seed`.

D4. Uniform. $`U = \mathrm{fixup}(y \cdot 2^{-32})`$. `fixup` returns $`\epsilon/2`$ for an argument $`x \le 0`$ and $`{1 - \epsilon/2}`$ for $`x \ge 1`$, and returns $`x`$ otherwise. Here $`\epsilon`$ is R's constant `i2_32m1`, written as `2.328306437080797e-10`, close to $`(2^{32} - 1)^{-1}`$. Since $`y \lt 2^{32}`$, only $`y = 0`$ reaches the first case and the second never applies. R: `unif_rand`, `fixup`. Python: `RDraws._unif`.

D5. Index. Under Rejection, `R_unif_index(N)` returns 0 for $`N \le 0`$. Otherwise it sets $`b = \lceil \log_2 N \rceil`$, computed in double, and draws $`v = \mathrm{rbits}(b)`$ until $`v \lt N`$:

```math
\mathrm{rbits}(b) = \Big( \sum_{r=0}^{L-1} 65536^{L-1-r} \lfloor 65536\, U_r \rfloor \Big) \wedge (2^{b} - 1), \qquad L = \lfloor b/16 \rfloor + 1,
```

with fresh uniforms $`U_0, U_1, \dots`$ drawn in order. R: `R_unif_index`, `rbits`. Python: `RDraws._unif_index`.

D6. `sample.int`. Without replacement and without weights, `sample.int(N, q)` defaults to `useHash = N > 1e7 && q <= N/2`. It then takes one of three cases. Each case finds 0-based values with `R_unif_index`, and R returns each value plus 1, in $`\lbrace 1, \dots, N \rbrace`$. The port returns the 0-based values.

- Hash (`do_sample2`, when `useHash` is true): for each of the $`q`$ values, draw `R_unif_index(N)` until the value is new or 100 draws are made. Keep the last draw.
- Single (`useHash` false, $`q \lt 2`$): make $`q`$ draws of `R_unif_index(N)`.
- Pool (`useHash` false, $`q \ge 2`$): start from the list $`(0, 1, \dots, N - 1)`$ of length $`N' = N`$. For each value draw a position $`d`$ from `R_unif_index(N')` and take entry $`d`$ of the list, counting from 0. Then move the last entry into position $`d`$ and decrease $`N'`$ by one.

`sample.int(N, q, replace = TRUE)` makes $`q`$ draws of `R_unif_index(N)` and never hashes. R: `do_sample` in `src/main/random.c` and `do_sample2` in `src/main/unique.c`. Python: `RDraws.sample` and `RDraws.sample_replace`, with `RDraws._hash`, `RDraws._pool` and `r_use_hash`.

D7. NumPy draws. `NumpyDraws(seed)` uses `numpy.random.default_rng(seed)` for an integer and a `Generator` as given. Its methods call the Generator's `permutation`, `integers` and `choice(..., replace=False)`.

D8. Choosing the draws. `as_draws(seed)` returns `seed.fork()` for an `RDraws` and any other `Draws` object as given. It returns a `NumpyDraws` for an integer or a `Generator` and raises `TypeError` otherwise, for a `bool` too. `fork()` gives the stream for one `refine_dn` split. For an `RDraws` it is a new `RDraws` from the same seed. For a `NumpyDraws` it is a new one from the same integer, or the next child from `Generator.spawn`.

## Algorithm

Step 1. `create_dn` (R `create.DN`, datanugget 1.5.0). Algorithm 1 of Beavers et al. (2025) splits $`X`$ into subsets at random in step 1. Its step 6 can re-center each nugget on a random observation assigned to it. R calls `set.seed(seed)` and then `sample(obs.num, obs.num, replace = FALSE)`. Since $`q = N \gt N/2`$, `useHash` is false. Since `create.DN` stops for fewer than two rows, $`q \ge 2`$ and this takes the pool case. The port calls `permutation(n)` on the stream from `as_draws`, which for an `RDraws` starts at the seed on every call. With `center_method = "random"`, R then draws `sample.int(length(ii), 1L)` for each nugget with members, in nugget order. It draws nothing for an empty nugget. The port calls `index(w_l)` in the same order. Python: `src/datanuggets/nuggets/create.py`, `create_dn`. Details are in [create-dn.md](create-dn.md).

Step 2. `refine_dn` (R `refine.DN`, datanugget 1.5.0). Before each split R calls `set.seed(seed)` and then `stats::kmeans(centers = 2, iter.max = 1000, nstart = nstart)`. The port passes `fork()` to each split. Its `kmeans` draws as R 4.6.1's `stats::kmeans` does with $`k = 2`$. With `nstart = 1` it draws `sample.int(N, 2)` among the nugget's $`N`$ rows. If the two rows are equal it then draws `sample.int(N_u, 2)` among the $`N_u`$ distinct rows. With `nstart` of 2 or more it draws `sample.int(N_u, 2)` once for each start. R: `src/library/stats/R/kmeans.R`, `kmeans`. Python: `refine_dn` in `src/datanuggets/nuggets/refine.py` and `kmeans` in `src/datanuggets/_clustering/hartigan_wong.py`. Details are in [refine-dn.md](refine-dn.md) and [hartigan-wong.md](hartigan-wong.md).

Step 3. `wkmeans` (R `Wkmeans`, WCluster 1.3.0). R calls `set.seed(seed)` once. For each of the `num.init` starts, `init` draws labellings until every label is on at least two rows. A labelling is `ff(k, n)`, which is `sample(c(1:k, sample(k, n - k, replace = TRUE)))`. `init` then gives row $`i`$ the first label $`c`$ that minimizes $`w_i \lVert \mathbf{x}_i - \mathbf{u}_c \rVert^2`$, where $`\mathbf{u}_c`$ is the weighted mean of the rows labelled $`c`$. For $`w_i \gt 0`$ that is the nearest label mean. R computes it with WCluster's `wdist_mat` and `which.min`, and the port with `assign` in `src/datanuggets/wcluster/wstatsjit.py`. `init` draws again until all $`k`$ labels are used. The port's `_draw_labels` calls `sample_replace(k, n - k)` and then `permutation(n)` for each labelling. `_init` counts every labelling, inner and outer. It raises `ValueError` after `MAX_DRAWS` of them, where R keeps drawing. `MAX_DRAWS` is 1,000. Centers given by the user draw nothing. `dn_wkmeans` calls `wkmeans` and draws the same way. Python: `src/datanuggets/wcluster/wkmeans.py`, `_init` and `_draw_labels`. Details are in [weighted-kmeans.md](weighted-kmeans.md).

Step 4. Parallel runs. `refine_dn` with `n_jobs` other than 1 accepts only `RDraws` and `NumpyDraws`, whose `fork()` gives each split a stream of its own. `ReplayDraws.fork()` returns the one recorded stream, which worker processes cannot share. `refine_dn` raises `ValueError` for a `ReplayDraws` or any other `Draws` object. `wkmeans` with threads draws every start before it refines any, in start order. Refining draws nothing, so each start gets its serial draws. When a start stops in its refinement, the serial path draws no later start, as R does. The threaded path has already drawn the later starts, up to the first start whose draws fail. It raises the same error as the serial path. Python: `refine_dn` in `src/datanuggets/nuggets/refine.py` and `_wkmeans` in `src/datanuggets/wcluster/wkmeans.py`.

## Numerical notes

Every step of D2, D3 and D5 is exact integer arithmetic except $`b = \lceil \log_2 N \rceil`$ and the products of $`U`$ and 65536 in D5. Those products and D4 work in double. The product $`y \cdot 2^{-32}`$ is exact in double. So is the product of $`U`$ and 65536, which makes each 16-bit piece of `rbits` the top 16 bits of a tempered word.

R holds `rbits` in a 64-bit unsigned integer and the port in a Python integer. R's accumulator never wraps around: `sample.int` stops for $`N \gt 4.5 \cdot 10^{15}`$, so $`b \le 52`$ and $`L \le 4`$.

The bit count comes from the rounded logarithm. At $`N = 2^{49} + 1`$ the double $`\log_2 N`$ is exactly 49, so $`b = 49`$ although $`N - 1`$ needs 50 bits, and R never returns the value $`N`$. The port computes $`b`$ with `math.log2` and matches R there.

For $`N`$ above `INT_MAX` and $`q \le 1`$, the default `useHash` is true, so R and the port both take the hash case. With `useHash = FALSE`, R's `do_sample` has no single case above `INT_MAX` and runs the pool for $`q \le 1`$ too. With one value the pool, the single case and the hash case make the same draw.

## Tests

The fixtures in `tests/fixtures/rsample/` were recorded by R 4.6.1 on x86_64-w64-mingw32 with datanugget 1.5.0. `tests/randomness/test_rdraws.py::test_reference_is_pinned` checks those versions, the three RNG kinds, R's `useHash` default and the body of `sample.int`. `tests/randomness/test_rdraws.py::test_every_case_is_recorded` checks that each case is present.

- `tests/randomness/test_rdraws.py::test_seed_state_and_draws_match_r`: `.Random.seed` and `sample(10, 10)` for integer seeds, fractional seeds and seeds near $`\pm 2^{31}`$.
- `tests/randomness/test_rdraws.py::test_invalid_seeds_stop_like_r` and `::test_integer_seeds_take_the_same_path_as_r_doubles`: R's errors and warnings for invalid seeds.
- `tests/randomness/test_rdraws.py::test_seeds_r_coerces_are_refused`: D1, the refusal of a `bool`, a string and `None`.
- `tests/randomness/test_rdraws.py::test_switch_matches_r_and_the_draws_after_it` and `::test_the_other_branch_matches_r`: both sides of the hash switch at $`N = 10^7`$ and at $`N = 2^{24} + 1`$, and the draws that follow.
- `tests/randomness/test_rdraws.py::test_size_condition_on_the_sampler` and `::test_use_hash_matches_r_across_the_boundaries`: the condition $`q \le N/2`$.
- `tests/randomness/test_rdraws.py::test_bit_count_comes_from_log2_like_r`: the case $`N = 2^{49} + 1`$.
- `tests/randomness/test_rdraws.py::test_hash_keeps_the_value_after_100_repeats`: the 100-draw limit of the hash case.
- `tests/randomness/test_rdraws.py::test_sample_replace_matches_r`: `replace = TRUE`, including $`N`$ = `INT_MAX`.
- `tests/randomness/test_rdraws.py::test_fork_restarts_the_seed_like_set_seed` and `::test_create_dn_reseeds_an_rdraws_like_set_seed`: D8.
- `tests/randomness/test_rdraws.py::test_every_recorded_create_dn_draw`: every draw recorded in the `create_dn` fixtures.
- `tests/randomness/test_rdraws.py::test_refine_dn_with_rdraws_matches_r`: `refine_dn` with `RDraws` against R's results. `::test_rdraws_splits_in_parallel_like_serial` compares `n_jobs=2` with `n_jobs=1`.
- `tests/randomness/test_rdraws.py::test_rdraws_is_public`: the export.
- `tests/wcluster/test_wkmeans.py::test_rdraws_seed_matches_r`: `wkmeans` with `RDraws(291102)` against R's labels. `::test_n_jobs_does_not_change_the_result` compares threaded runs with `n_jobs=1`.
- `tests/wcluster/test_wkmeans.py::test_the_cap_allows_exactly_max_draws`, `::test_the_draw_cap_counts_inner_and_outer_draws`, `::test_init_gives_up_where_r_draws_forever` and `::test_user_centers_draw_nothing_and_keep_their_error`: the cap of Step 3.
- `tests/randomness/test_draws.py::test_as_draws_rejects_other_types`, `::test_numpy_fork_with_int_seed_restarts_the_stream` and `::test_numpy_fork_with_generator_spawns_reproducible_children`: D8.
- `tests/randomness/test_draws.py::test_fixture_draws_round_trip`: in the draws recorded for the `iris_euclidean_random` fixture of `create_dn`, the first is a permutation of all rows and every later draw takes one value, as Step 1 describes.
- `tests/nuggets/test_refine.py::test_n_jobs_does_not_change_the_result` and `::test_replayed_draws_need_n_jobs_1`: Step 4.
- `tests/wcluster/test_wkmeans.py::test_replayed_draws_give_rs_error_for_any_n_jobs`: Step 4, R's error for a start that stops in its refinement, with and without threads.

The parity tests of the ported functions replay R's recorded draws through `ReplayDraws`, which fails on any draw R did not make: `tests/nuggets/test_create.py::test_create_dn_matches_r`, `tests/clustering/test_hartigan_wong.py::test_kmeans_matches_r`, `tests/wcluster/test_wkmeans.py::test_wkmeans_matches_r_and_uses_r_draws` and `tests/wcluster/test_dn.py::test_dn_wkmeans_matches_r`.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. Algorithm 1, steps 1 and 6. Algorithm 3, step 1, and the paragraph after it.
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0. Functions `create.DN` and `refine.DN`.
- Dey, R., Duan, Y., Cabrera, J., and Cheng, G. (2025). WCluster: Clustering and PCA with Weights, and Data Nuggets Clustering. R package version 1.3.0. Functions `Wkmeans` and `init`.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. `src/main/RNG.c` (`do_setseed`, `RNG_Init`, `FixupSeeds`, `PutRNGstate`, `MT_genrand`, `fixup`, `unif_rand`, `rbits`, `R_unif_index`), `src/main/random.c` (`do_sample`), `src/main/unique.c` (`do_sample2`), `src/main/coerce.c` (`IntegerFromReal`) and `src/library/stats/R/kmeans.R` (`kmeans`).
- Matsumoto, M., and Nishimura, T. (1998). Mersenne Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number Generator. ACM Transactions on Modeling and Computer Simulation, 8(1), 3-30. Cited in `src/main/RNG.c`.

## Not covered by a test

- D4: the first case of `fixup`, which only $`y = 0`$ reaches. No test constructs that word.
- D7: that `NumpyDraws` calls `default_rng(seed)`, `permutation`, `integers` and `choice(..., replace=False)`. The tests check only that its draws repeat and lie in range, and that `sample` draws distinct values.
- Numerical notes: the draws for $`N`$ above `INT_MAX` with $`q \le 1`$, and R's pool above `INT_MAX`. The recorded case `bits_2p49p1` takes three values by the hash case.
- Step 4: the `ValueError` of `refine_dn` for a `Draws` object other than `ReplayDraws`.
