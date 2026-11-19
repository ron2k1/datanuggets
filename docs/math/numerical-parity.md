# Numerical parity with R

Draft. This page has not yet been reviewed by the authors of the R packages.

The nugget and WCluster ports do their sums, means, covariances, matrix products and distances through the functions in `src/datanuggets/_compat/`. R 4.6.1 on x86-64 accumulates many of these in C's `long double`, which there is the x87 extended format with a 64-bit significand. This page states how R rounds each computation and how the port copies it. The fixtures that hold the port to R were recorded by that build. Beavers et al. (2025) says nothing about floating-point rounding, so no statement on this page rests on it.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `r_sum(v)` | `src/datanuggets/_compat/arithmetic.py`, `r_sum` | `sum()`, R 4.6.1 (base) |
| `r_vec_mean(v)` | `src/datanuggets/_compat/arithmetic.py`, `r_vec_mean` | `mean()`, R 4.6.1 (base) |
| `r_mean(a, axis)` | `r_mean` in `src/datanuggets/_compat/arithmetic.py` and `line_means` in `src/datanuggets/_compat/ldjit.py` | `colMeans()` and `rowMeans()`, R 4.6.1 (base) |
| `r_cov(x)` | `r_cov` and `_r_cov_py` in `src/datanuggets/_compat/arithmetic.py`, and `cov_rows` in `src/datanuggets/_compat/ldjit.py` | `cov()` and `var()`, R 4.6.1 (stats) |
| `group_moments(x, groups, n_jobs)` | `group_moments` in `src/datanuggets/_compat/moments.py` and `_group` in `src/datanuggets/_compat/ldjit.py` | `colMeans()` and `cov()` of each group, R 4.6.1 |
| `r_weighted_mean(v, w)` | `src/datanuggets/_compat/arithmetic.py`, `r_weighted_mean` | `weighted.mean()`, R 4.6.1 (stats) |
| `r_median(v)`, `r_mad(v)`, `r_quantile7(v, prob)` | `src/datanuggets/_compat/arithmetic.py`, `r_median`, `r_mad` and `r_quantile7` | `median()`, `mad()` and `quantile(type = 7)`, R 4.6.1 (stats) |
| `r_matvec(a, v)`, `r_matmul(a, b)` | `src/datanuggets/_compat/linalg.py`, `r_matvec` and `r_matmul` | `%*%`, R 4.6.1 (base) with R's reference BLAS |
| `r_eigen_sym(s, vectors)` | `src/datanuggets/_compat/linalg.py`, `r_eigen_sym` | `eigen(symmetric = TRUE)`, R 4.6.1 (base) |
| `cross_distances`, `dist_matrix`, `nearest_index` | `src/datanuggets/_compat/rfast.py`, and `_sums`, `dist` and `nearest` in `src/datanuggets/_compat/rfastjit.py` | `dista()` and `Dist()`, Rfast 2.1.5.2 |
| `_non_finite_sum(v, quiet)` | `src/datanuggets/_compat/missing.py`, `_non_finite_sum` | the Inf or NaN that R's long double sum ends on, R 4.6.1 |
| `ldouble`, `ldjit` | `src/datanuggets/_compat/ldouble.py` and `src/datanuggets/_compat/ldjit.py` | R's `LDOUBLE` type, R 4.6.1 on x86-64 |

`split_indices` and `unique_first` in `src/datanuggets/_compat/indexing.py` choose indices and use no long double arithmetic. Their tests are `tests/compat/test_rcompat.py::test_split_indices_matches_r` and `tests/compat/test_rstats.py::test_unique_first_matches_r`. `create_dn` calls `r_mean` and `nearest_index` ([create-dn.md](create-dn.md)). `src/datanuggets/nuggets/centers.py` calls `dist_matrix`, and `get_dn_means` calls `r_mean` ([nugget-means-and-complexity.md](nugget-means-and-complexity.md)). `refine_dn` calls `group_moments`, `r_cov`, `r_eigen_sym`, `r_matvec`, `r_sum`, `r_vec_mean`, `r_median`, `r_mad` and `r_quantile7` ([refine-dn.md](refine-dn.md)). `kmeans` calls `r_sum` ([hartigan-wong.md](hartigan-wong.md)). `wmean`, `wss`, `wwcss`, `distw` and `wpca` call `r_weighted_mean`, `r_sum`, `r_matvec`, `r_matmul` and `r_eigen_sym` ([weighted-statistics.md](weighted-statistics.md), [weighted-pca.md](weighted-pca.md)).

## Notation

The shared symbols are in [README.md](README.md#notation). This page adds:

| Symbol | Meaning |
| --- | --- |
| $`\mathrm{rn}(y)`$ | $`y`$ rounded to the nearest integer, ties to the even one |
| $`\mathrm{fl}_{80}(y)`$ | $`y`$ rounded to the x87 extended format, D1 |
| $`\Omega`$ | the largest finite double, C's `DBL_MAX` |
| $`\mathrm{S}_{80}`$, $`\mathrm{S}`$, $`\mathrm{A}`$ | the three running sums of D2 |
| $`v_{(r)}`$ | the $`r`$-th smallest entry of a vector $`v`$ |
| $`z_c`$, $`\delta_j`$ | center $`c`$ of a distance call, and $`\delta_j = \mathrm{fl}(z_{cj} - x_{ij})`$ |

## Definitions

D1. Roundings. R's `LDOUBLE` is C's `long double`. On x86-64 every long double operation rounds to a 64-bit significand, to nearest with ties to even. A double keeps 53 bits on a grid no finer than $`{2^{-1074}}`$. For $`y \ne 0`$:

```math
\begin{aligned} \mathrm{fl}_{80}(y) &= \mathrm{rn}\big(y\, 2^{-e}\big)\, 2^{e}, \qquad e = \lfloor \log_2 \lvert y \rvert \rfloor - 63, \\ \mathrm{fl}(y) &= \mathrm{rn}\big(y\, 2^{-e}\big)\, 2^{e}, \qquad e = \max\big(\lfloor \log_2 \lvert y \rvert \rfloor - 52, -1074\big). \end{aligned}
```

Both map 0 to 0, and $`\mathrm{fl}`$ gives $`\pm\infty`$ once the rounded value reaches $`{2^{1024}}`$ in absolute value. The long double exponent range is ignored, since it holds every sum, product and quotient of finite doubles formed here. R's `(double)` cast of a long double applies $`\mathrm{fl}`$ once, so R's long double results are $`\mathrm{fl}(\mathrm{fl}_{80}(\cdot))`$. Arithmetic on plain doubles rounds each operation by $`\mathrm{fl}`$. Python: `_rounded` and `to_float` in `src/datanuggets/_compat/ldouble.py`, and `round128` and `round53` in `src/datanuggets/_compat/ldjit.py`.

D2. Running sums. Each adds from the left and rounds after every term. An empty sum is $`+0`$.

```math
\begin{aligned} \mathrm{S}_{80}(y_1, \dots, y_n) &= s_n, \qquad s_0 = +0, \quad s_i = \mathrm{fl}_{80}(s_{i-1} + y_i), \\ \mathrm{S}(y_1, \dots, y_n) &= s_n, \qquad s_0 = +0, \quad s_i = \mathrm{fl}(s_{i-1} + y_i), \\ \mathrm{A}(y_1, \dots, y_p) &= \mathrm{fl}\big(\mathrm{S}(y_1, y_3, y_5, \dots) + \mathrm{S}(y_2, y_4, y_6, \dots)\big). \end{aligned}
```

$`\mathrm{S}_{80}`$ is R's long double sum and $`\mathrm{S}`$ the double one. $`\mathrm{A}`$ keeps two double accumulators by position, as `_sums` in `src/datanuggets/_compat/rfastjit.py` does.

D3. `sum()`. For finite $`x_1, \dots, x_n`$ let $`s = \mathrm{S}_{80}(x_1, \dots, x_n)`$. `sum(x)` is $`+\infty`$ if $`s \gt \Omega`$ and $`-\infty`$ if $`s \lt -\Omega`$. Otherwise it is $`\mathrm{fl}(s)`$. The comparison comes before the rounding. R: `src/main/summary.c`, `rsum`. Python: `r_sum`, which compares with $`\Omega`$ as an exact integer.

With an Inf or NaN among the terms, only the non-finite terms decide the result, one add at a time. `_non_finite_sum` applies these rules to the bits. A number plus a NaN is that NaN, quieted. Infinities of opposite sign give the NaN with bits `fff8000000000000`. A NaN sum stays unless a quiet NaN with a larger significand follows, and the positive one wins a tie. A signaling NaN never replaces it. R's `NA` is the signaling NaN `7ff00000000007a2` (R 4.6.1, `src/main/arithmetic.c`, `R_ValueOfNA`). The fixtures hold these rules to R's recorded bits.

D4. `mean()`. For finite $`x`$ with $`n \ge 1`$ and $`s = \mathrm{S}_{80}(x_1, \dots, x_n)`$, R takes two passes. If $`\mathrm{fl}(s)`$ is finite:

```math
\begin{aligned} \hat\mu_0 &= \mathrm{fl}_{80}(s / n), \qquad t = \mathrm{S}_{80}\big(\mathrm{fl}_{80}(x_1 - \hat\mu_0), \dots, \mathrm{fl}_{80}(x_n - \hat\mu_0)\big), \\ \mathrm{mean}(x) &= \mathrm{fl}\big(\mathrm{fl}_{80}(\hat\mu_0 + \mathrm{fl}_{80}(t / n))\big). \end{aligned}
```

If $`\mathrm{fl}(s)`$ is infinite, R sums the terms already divided by $`n`$ and rounded to a double. It also divides each residual by $`n`$:

```math
\begin{aligned} \hat\mu_0 &= \mathrm{S}_{80}\big(\mathrm{fl}(x_1 / n), \dots, \mathrm{fl}(x_n / n)\big), \qquad t = \mathrm{S}_{80}(\tau_1, \dots, \tau_n), \qquad \tau_i = \mathrm{fl}_{80}\big(\mathrm{fl}_{80}(x_i - \hat\mu_0) / n\big), \\ \mathrm{mean}(x) &= \mathrm{fl}\big(\mathrm{fl}_{80}(\hat\mu_0 + t)\big). \end{aligned}
```

If $`\mathrm{fl}(\hat\mu_0)`$ is still infinite, R returns it with no second pass. With an Inf or NaN in $`x`$ the first sum of the second case is not finite, and R returns it. Each $`x_i / n`$ there is a double division, which quiets a signaling NaN, so the port calls `_non_finite_sum` with `quiet=True`. R: `src/main/summary.c`, `real_mean`. Python: `r_vec_mean`.

D5. `colMeans()` and `rowMeans()`. One long double pass, one division by the count and one rounding. There is no second pass and no comparison with $`\Omega`$:

```math
\mathrm{colMeans}(x)_j = \mathrm{fl}\Big(\mathrm{fl}_{80}\big(\mathrm{S}_{80}(x_{1j}, \dots, x_{nj}) / n\big)\Big), \qquad \mathrm{rowMeans}(x)_i = \mathrm{fl}\Big(\mathrm{fl}_{80}\big(\mathrm{S}_{80}(x_{i1}, \dots, x_{ip}) / p\big)\Big).
```

R: `src/main/array.c`, `do_colsum`. Python: `r_mean` and `line_means`. A line with no values gives the NaN `fff8000000000000`.

D6. `cov()` and `var()`. Both reach `cov_na_1` for a numeric matrix with the default `use = "everything"`. Each column mean takes two long double passes and is stored as a double $`\bar x_j`$. Each entry is a long double sum of long double products of the deviations from those doubles, divided by $`n - 1`$ and rounded once:

```math
\begin{aligned} \hat m_{0j} &= \mathrm{fl}_{80}\big(\mathrm{S}_{80}(x_{1j}, \dots, x_{nj}) / n\big), \qquad t_j = \mathrm{S}_{80}\big(\mathrm{fl}_{80}(x_{1j} - \hat m_{0j}), \dots, \mathrm{fl}_{80}(x_{nj} - \hat m_{0j})\big), \\ \bar x_j &= \mathrm{fl}\big(\mathrm{fl}_{80}(\hat m_{0j} + \mathrm{fl}_{80}(t_j / n))\big), \qquad d_{ij} = \mathrm{fl}_{80}(x_{ij} - \bar x_j), \\ \mathrm{cov}(x)_{ab} &= \mathrm{fl}\Big(\mathrm{fl}_{80}\Big(\mathrm{S}_{80}\big(\mathrm{fl}_{80}(d_{1a} d_{1b}), \dots, \mathrm{fl}_{80}(d_{na} d_{nb})\big) / (n - 1)\Big)\Big), \qquad b \le a. \end{aligned}
```

The upper triangle mirrors the lower one. When $`\mathrm{fl}(\hat m_{0j})`$ is infinite, R skips the second pass and $`\bar x_j = \mathrm{fl}(\hat m_{0j})`$. So the test is on the mean, where `mean()` tests the sum, and nothing here matches the divided pass of D4. R: `var` and `cov` in `src/library/stats/R/cor.R`, and `corcov`, `MEAN_` and `cov_na_1` in `src/library/stats/src/cov.c`. Python: `cov_rows` and `_r_cov_py`. Where some $`\bar x_j`$ is infinite the port returns no matrix (Step 4).

D7. `weighted.mean()`. R 4.6.1's `weighted.mean.default` computes `sum((x*w)[w != 0])/sum(w)`. For weights with no NaN that is $`\mathrm{fl}\big(\mathrm{sum}((\mathrm{fl}(v_i w_i))_{i : w_i \ne 0}) / \mathrm{sum}(w)\big)`$, with both sums by D3 in increasing $`i`$ and the division in double. Dropping the zero weights also drops an $`\infty \cdot 0`$ term. A NaN that the division makes from two non-NaN values is returned with the bits `fff8000000000000`, whatever the host's division gives. R: `src/library/stats/R/weighted.mean.R`, `weighted.mean.default`. Python: `r_weighted_mean`.

D8. `%*%`. For $`A`$ with $`q`$ columns and $`B`$ with $`q`$ rows, R's `%*%` under the default `matprod` option with R's reference BLAS gives the double sum $`(AB)_{ab} = \mathrm{S}\big(\mathrm{fl}(A_{a1} B_{1b}), \dots, \mathrm{fl}(A_{aq} B_{qb})\big)`$ from $`+0`$ over the inner index in order. `matprod` takes one of five routes, and each computes this sum without skipping zero terms:

- An empty dimension writes $`+0`$ in every entry.
- A NaN or Inf in either matrix goes to `simple_matprod`, a double loop in this order. R's check adds the entries two at a time, so two finite entries whose double sum is infinite also go there.
- A result with one column calls `dgemv` with `"N"`, and a result with one row calls `dgemv` with `"T"` on $`B`$.
- Any other shape calls `dgemm` with `"N", "N"`.

R: `matprod` and `simple_matprod` in `src/main/array.c`, and `DGEMV` and `DGEMM` in `src/extra/blas/blas.f`. Python: `r_matvec` and `r_matmul`, a zero array plus one rank-one update per inner index.

D9. Rfast distances. For a point $`x_i`$ and a center $`z_c`$ the port computes the following, with the correctly rounded square root:

```math
D^{\mathrm{euc}}_{ic} = \mathrm{fl}\Big(\sqrt{\mathrm{A}\big(\mathrm{fl}(\delta_1^2), \dots, \mathrm{fl}(\delta_p^2)\big)}\Big), \qquad D^{\mathrm{man}}_{ic} = \mathrm{A}\big(\lvert \delta_1 \rvert, \dots, \lvert \delta_p \rvert\big).
```

It uses these for `Rfast::Dist` at every $`p`$ and for `Rfast::dista` with the Manhattan metric. For `Rfast::dista` with the Euclidean metric it uses $`\mathrm{A}`$ up to 32 features and $`\mathrm{S}`$ above 32, where the constant `_ARMA_DOT_MAX` is 32. `create.DN` asks `dista` for the nearest center with `k = 1L, index = TRUE`. `nearest_index` returns the lowest-numbered center among exact ties, $`\hat c(i) = \min \big\lbrace c : D_{ic} = \min_{c'} D_{ic'} \big\rbrace`$, 0-based. Python: `src/datanuggets/_compat/rfast.py`, `cross_distances`, `dist_matrix` and `nearest_index`.

D10. `median()`, `mad()` and `quantile()`. For odd $`n`$, `median(v)` is $`v_{((n+1)/2)}`$. For even $`n`$ it is R's `mean()` of $`v_{(n/2)}`$ and $`v_{(n/2+1)}`$, by D4. Then $`\mathrm{mad}(v) = \mathrm{fl}\big(c_{\mathrm{mad}} \cdot \mathrm{median}(\lvert \mathrm{fl}(v_i - \mathrm{median}(v)) \rvert)\big)`$, where $`c_{\mathrm{mad}}`$ is the double nearest 1.4826 (`MAD_CONSTANT` in `src/datanuggets/_defaults.py`).

R's type-7 quantile at probability $`\rho`$ sets $`\eta = \mathrm{fl}(1 + \mathrm{fl}((n - 1)\rho))`$ with $`\mathrm{lo} = \lfloor \eta \rfloor`$, $`\mathrm{hi} = \lceil \eta \rceil`$ and $`g = \mathrm{fl}(\eta - \mathrm{lo})`$. It returns $`v_{(\mathrm{lo})}`$ when $`\eta = \mathrm{lo}`$ or $`v_{(\mathrm{hi})} = v_{(\mathrm{lo})}`$. Otherwise it returns $`\mathrm{fl}\big(\mathrm{fl}(\mathrm{fl}(1 - g)\, v_{(\mathrm{lo})}) + \mathrm{fl}(g\, v_{(\mathrm{hi})})\big)`$. The `fuzz` argument of `quantile.default` defaults to 0 for type 7, which makes $`\mathrm{lo}`$ the plain floor. R: `median.default` in `src/library/stats/R/median.R`, `mad` in `src/library/stats/R/mad.R` and `quantile.default` in `src/library/stats/R/quantile.R`. Python: `r_median`, `r_mad` and `r_quantile7`.

D11. `eigen()`. R's `eigen(symmetric = TRUE)` calls `La_rs`, which calls LAPACK's `dsyevr` with `uplo = "L"`, `range = "A"` and `abstol = 0` after a workspace query. `eigen` then puts the increasing values in decreasing order, with the vectors. `r_eigen_sym` rejects a non-finite matrix and otherwise calls the same routine with the same arguments through SciPy. It reverses the same way. R: `eigen` in `src/library/base/R/eigen.R` and `La_rs` in `src/modules/lapack/Lapack.c`.

## Algorithm

Step 1. Integer reference, `src/datanuggets/_compat/ldouble.py`. A long double is a pair $`(m, e)`$ standing for $`m \cdot 2^e`$. `from_float` takes a double's integer ratio, which is exact. `add`, `sub` and `mul` form the exact result in Python integers and round it once by D1. `div_int` keeps at least two quotient bits past the 64. It rounds them by D1 and counts a nonzero remainder as lying past the halfway point, which gives $`\mathrm{fl}_{80}(a / n)`$. `to_float` rounds once by $`\mathrm{fl}`$. `r_sum`, `r_vec_mean` and `_r_cov_py` run on this module. `r_median`, `r_mad` and `r_weighted_mean` run on it through them.

Step 2. Compiled long double, `src/datanuggets/_compat/ldjit.py`. A long double is a triple $`(s, m, e)`$ with $`m`$ a 64-bit unsigned integer whose top bit is set. Zero is $`(0, 0, 0)`$. `add` holds the exact sum in 128 bits. When the exponents differ by more than 63, the bits that fall off the smaller operand collapse into one sticky bit, which cannot change the rounding direction. `mul` builds the 128-bit product from 32-bit halves. `div_count` divides by a count one bit at a time. `to_double` rounds once by $`\mathrm{fl}`$. `as_double` does so and reads the double back, as R does when it stores a long double in a double variable.

Step 3. `r_mean` moves the averaged axis last. A finite line goes to `line_means`, which applies D5. A line with Inf or NaN gets `_non_finite_sum`, since dividing a non-finite long double by a count does not change it.

Step 4. `r_cov` raises `ValueError` below 2 rows. For finite data `cov_rows` applies D6. It returns `False` when some $`\bar x_j`$ is infinite. `r_cov` then calls `_r_cov_py`, which raises an error on that mean. So the port returns no matrix where R's `cov()` returns one with non-finite entries in that row and column. Data with Inf or NaN also goes to `_r_cov_py`, which raises an error on the first non-finite value.

Step 5. `group_moments` takes the groups in chunks of $`\max(1, \lfloor 2^{22} / p^2 \rfloor)`$, where $`{2^{22}}`$ is `MOMENTS_CHUNK`. For each group `_group` returns `False` if any entry is Inf or NaN. An empty group gets the NaN center `fff8000000000000`. Otherwise it computes the center by D5 and, with 2 or more rows, the covariance with `cov_rows`. With `n_jobs` other than 1 the groups run on threads, each group serial. A group that returned `False` is redone with `r_mean` and `r_cov`. With 2 or more rows the call to `r_cov` raises as in Step 4, so `group_moments` raises on such a group.

Step 6. `_sums` adds the terms into two accumulators by position, or all into one for a left-to-right sum, and adds the two at the end. `dist` computes the lower half of the matrix and mirrors it one 64 by 64 tile at a time (`MIRROR_TILE`). `nearest` moves its pick only on a strictly smaller distance.

## Numerical notes

The fixtures pin a 64-bit significand. R's `sum(c(1, 2^-53, 2^-63))` is `1 + 2^-52` and `sum(c(1, 2^-53, 2^-64))` is 1. With 64 bits the long double step at 1 is `2^-63`. The first long double sum is exact and lies past the double midpoint `1 + 2^-53`, so $`\mathrm{fl}`$ rounds it up. In the second, adding `2^-64` is a tie under $`\mathrm{fl}_{80}`$ that stays at `1 + 2^-53`, and that is a tie under $`\mathrm{fl}`$ that goes to 1. A double running sum gives 1 for both, and the correctly rounded sum is `1 + 2^-52` for both. Each manifest of the fixture groups named under Tests records `longdouble_digits` 64.

The order of a long double sum matters even at $`\Omega`$. R's sum of $`\Omega`$, `2^959` and `2^959` in that order is $`\Omega`$. In the order `2^959`, `2^959`, $`\Omega`$ it is $`+\infty`$. The long double step at $`\Omega`$ is `2^960` and $`\Omega`$'s 64-bit significand is even, so each added `2^959` is a tie that stays at $`\Omega`$. In the other order the two halves make `2^960` exactly, and $`\Omega`$ plus `2^960` is exact and above $`\Omega`$. `sum()` clamps before it rounds, and `mean()` rounds before it tests. $`\Omega`$ plus `2^960` gives $`+\infty`$ from `sum()`, although $`\mathrm{fl}`$ of that value is $`\Omega`$. `mean()` takes its second case only when $`\mathrm{fl}(s)`$ is infinite. `r_sum` compares in exact integers and `r_vec_mean` tests the rounded value.

`mean()` has a correction pass and `colMeans()` and `rowMeans()` do not, so the port keeps D4 and D5 apart. `cov()`'s column means equal `mean()` of each column whenever the column sum is finite as a double. Where that sum is infinite as a double, `mean()` takes its divided pass and `MEAN_` keeps the plain second pass, so the port keeps two copies of the mean code.

`%*%` never returns $`-0`$. Each sum in D8 starts at $`+0`$, and under round to nearest $`+0 + (-0) = +0`$ and $`y + (-y) = +0`$ (IEEE 754-2019, section 6.3). A sum of doubles is zero only when it is exactly zero, so a running sum from $`+0`$ never becomes $`-0`$. The five routes add the same products in the same order and agree wherever the result is finite. The port therefore does not model R's NaN check.

`Dist` is symmetric bit for bit. Swapping two rows negates each $`\delta_j`$ exactly, squares and absolute values drop the sign, and the order over $`j`$ is unchanged. So mirroring the lower half is exact. `group_moments` does not depend on the thread count or the chunk size. Each group runs the same serial kernel and no sum crosses groups. The center loop of `_group` is the loop of `line_means`, and its covariance is `cov_rows`, which `r_cov` also calls.

A NaN that an x86 floating-point operation makes from non-NaN values has the bits `fff8000000000000`, with the sign bit set. R's `0/0`, `Inf - Inf` and `colMeans()` of no values give these bits. Other computations in R give the positive NaN `7ff8000000000000`. Examples are `log(-1)` and `5 %% 0`, which returns R's constant `R_NaN` from `myfmod` in `src/main/arithmetic.c`. The port writes `fff8000000000000` as a constant instead of taking the host's.

The port and R differ in these places:

- The fixtures fix one build: R 4.6.1 on x86_64-w64-mingw32 with R's reference BLAS, LAPACK 3.12.1 and a 64-bit long double significand. The port always computes as that build does. An R with another BLAS or another long double width can round differently.
- `r_eigen_sym` uses SciPy's LAPACK build, not R's, and no test holds it to R's bits. `tests/compat/test_rstats.py::test_eigen_matches_r_numerically` holds its values, and its leading vector up to sign, within `RTOL = 1e-8` and `ATOL = 1e-10` in `tests/rfixtures.py`.
- On an exact distance tie the port takes the lowest-numbered center. R's Rfast does the same in the `rcompat_dista` cases with 12 centers. In those with 20 or more centers it can take another of the tied centers, even when only 2 are tied. The tests accept R's other pick only in a call with more than 16 centers (`R_TIES_AGREE_UP_TO` in `tests/rfixtures.py`). No fixture fixes that bound between 12 and 20.
- `r_cov` raises on data with Inf or NaN and on finite data whose column mean $`\bar x_j`$ is infinite. `group_moments` raises on such a group of 2 or more rows. R's `cov()` returns a matrix in both cases.
- `r_vec_mean([])` and `r_median([])` return 0.0, and `r_cov` raises below 2 rows. R's `mean()`, `median()` and `cov()` return `NaN`, `NA` and a matrix of `NA` there.
- The port does not model `options(matprod = "internal")`, `"blas"` or `"default.simd"`, nor R's `crossprod()`.

## Tests

The fixtures in `tests/fixtures/rcompat_means/`, `rcompat_stats/`, `rcompat_wstats/`, `rcompat_dist/`, `rcompat_dista/` and `get_dn_means/` were recorded by R 4.6.1 on x86_64-w64-mingw32 with Rfast 2.1.5.2 and LAPACK 3.12.1.

- `tests/compat/test_ldjit.py::test_binary_kernels_match_the_reference`, `::test_div_count_matches_the_reference`, `::test_from_bits_matches_the_reference`, `::test_to_double_matches_the_reference`, `::test_as_double_matches_the_reference`, `::test_to_double_keeps_the_sign_of_zero_and_infinity` and `::test_long_fuzz`: Step 2 gives the values of Step 1.
- `tests/compat/test_rstats.py::test_r_sum_boundaries`: the cases `sum_one_powm53_powm63`, `sum_one_powm53_powm64`, `sum_xmax_pow959_pow959`, `sum_pow959_pow959_xmax` and `sum_xmax_pow960` of the notes above, and other boundary vectors.
- `tests/compat/test_rstats.py::test_sum_matches_r_past_dbl_max_and_with_inf_or_nan` and `::test_mean_of_non_finite_values_matches_r`: D3 and D4 on `sum_inf_minus_inf`, `mean_inf_minus_inf`, the `sum_nanbits_*` and `mean_nanbits_*` cases and a sum past $`\Omega`$ that takes the second case of D4.
- `tests/compat/test_rstats.py::test_vector_stats_match_r_bitwise`: `sum()`, `mean()`, `median()`, `mad()` and type-7 quantiles bit for bit on the `vec_*` cases.
- `tests/compat/test_rcompat.py::test_r_mean_matches_r_colmeans_and_rowmeans`: D5 on `faithful`, `iris`, `mtcars`, `petal`, `quakes`, `volcano_scaled`, `nan_bits` and `non_finite`. Also `::test_r_mean_handles_sums_beyond_the_double_range` and `::test_r_mean_of_no_values_is_nan_like_r`. `tests/nuggets/test_means.py::test_empty_nugget_is_rs_negative_nan` holds the bits of D5 for a line with no values to R's `getDN.means` (datanugget 1.5.0), which calls `colMeans()`.
- `tests/compat/test_rstats.py::test_cov_and_scale_match_r_bitwise`: D6 on `cov_big_offset`, `cov_faithful`, `cov_iris`, `cov_iris_setosa`, `cov_mtcars`, `cov_one_col`, `cov_quakes`, `cov_two_rows` and `cov_volcano5`. Also `::test_r_cov_matches_the_replay_on_every_fixture_nugget` and `::test_r_cov_raises_on_inf_and_nan_like_the_replay`. Step 5: `::test_group_moments_match_the_replay_on_every_fixture_nugget`, `::test_group_moments_chunking_gives_the_same_result` and `::test_group_moments_leaves_non_finite_groups_to_the_replay`.
- `tests/compat/test_rstats.py::test_r_weighted_mean_matches_r` and `::test_rcompat_wstats_covers_each_kind`: D7, with a zero total. `::test_matvec_matches_r_bitwise`, `::test_r_matmul_matches_r`, `::test_r_matmul_never_returns_negative_zero` and `::test_r_matvec_never_returns_negative_zero`: D8, with a case that takes R's NaN route. `::test_eigen_matches_r_numerically`: the values of D11 and the leading vector up to its sign, within the tolerances above. `tests/wcluster/test_wpca.py::test_wpca_matches_r` judges every vector that `wpca` takes from `r_eigen_sym` by the rule `eigen_mismatches` in `tests/rfixtures.py`, which [weighted-pca.md](weighted-pca.md) states.
- `tests/compat/test_rfast.py::test_dist_matrix_is_bitwise_equal_to_rfast`, `::test_dist_matrix_matches_the_numpy_replay` and `::test_dist_matrix_mirrors_every_tile`: `Rfast::Dist` bit for bit and the mirror. The `rcompat_dist` cases with 4, 5, 11 and 40 features tell $`\mathrm{A}`$ from $`\mathrm{S}`$. So the first test holds $`\mathrm{A}`$ of D2 to R. `::test_rfast_full_order_ranks_the_distances` and `::test_nearest_index_matches_rfast`: `Rfast::dista`'s ranking of the centers and its nearest index. On `perm32`, `perm33` and `perm40` R's Euclidean ranking and nearest index fit $`\mathrm{A}`$ at 32 features and $`\mathrm{S}`$ at 33 and 40. R's Manhattan ranking and nearest index fit $`\mathrm{A}`$ at all three. So these two tests hold the switch at 32 features to R. `::test_dista_euclidean_sums_left_to_right_above_32_features` checks the switch against NumPy.
- `tests/compat/test_rfast.py::test_exact_ties_go_to_the_lowest_center` and `::test_r_parts_from_the_lowest_index_only_on_wide_ties`: the tie rule of D9, and the fixture rows where R picks another tied center, each in a call with more than 16 centers.

## References

- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. `src/main/summary.c` (`rsum`, `real_mean`), `src/main/array.c` (`matprod`, `simple_matprod`, `do_colsum`), `src/main/arithmetic.c` (`R_ValueOfNA`), `src/library/stats/src/cov.c` (`MEAN_`, `cov_na_1`, `corcov`), `src/extra/blas/blas.f` (`DGEMV`, `DGEMM`), `src/modules/lapack/Lapack.c` (`La_rs`), and the R functions `eigen` (base) and `var`, `cov`, `weighted.mean.default`, `median.default`, `mad` and `quantile.default` (stats).
- Papadakis, M., Tsagris, M., Dimitriadis, M., Fafalios, S., Fasiolo, M., Borboudakis, G., Burkardt, J., and Zou, C. (2025). Rfast: A Collection of Efficient and Extremely Fast R Functions. R package version 2.1.5.2. Functions `dista` and `Dist`.
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0. Function `create.DN`.
- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`.
- IEEE Std 754-2019. IEEE Standard for Floating-Point Arithmetic. Section 6.3.

## Not covered by a test

- `r_vec_mean([])` and `r_median([])` return 0.0. R's `mean(numeric(0))` is NaN by `real_mean`, and `median.default` returns `NA` for an empty vector.
- `r_cov` raises `ValueError` below 2 rows, where R's `cov_na_1` returns a matrix of `NA` (`src/library/stats/src/cov.c`, `COV_ini_na`).
- The bits `fff8000000000000` that `group_moments` writes for an empty group. `tests/compat/test_rstats.py::test_group_moments_leaves_non_finite_groups_to_the_replay` checks only that the center is NaN.
- The distance values of `Rfast::dista`. The fixtures hold only its ranking and nearest index, so the last bits of each distance are not compared with R.
- `r_cov` and `group_moments` on finite data whose column mean $`\bar x_j`$ is infinite.
- The correction pass in the first case of D4 and the second pass of the column means in D6. A single pass $`\mathrm{fl}(\mathrm{fl}_{80}(s / n))`$ on the sum $`s`$ of D4 gives R's bits too on `mean_pair`, on every `cov_*` case and on every `vec_*` case that takes the first case of D4. On the `vec_*` cases of even length R's `median()` also equals the plain double average of the two middle values. These passes rest on `real_mean` and `MEAN_` in R's source.
