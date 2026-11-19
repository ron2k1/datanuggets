# Weighted PCA

Draft. This page has not yet been reviewed by the authors of the R packages.

`wpca` computes principal components of data that carry a weight per row and a weight per column, as WCluster's `Wpca` does. The row weights act as counts in a weighted covariance matrix. The column weights rescale the variables before the eigen decomposition, and `corr = True` turns the matrix into a correlation matrix first. `dn_wpca` runs `wpca` on the nugget centers with the nugget weights as row weights. The center, the column weights, the centered data and the matrix handed to the eigen solver match R bit for bit, except in the one-column `corr` case of Numerical notes. The eigen decomposition comes from LAPACK and is compared with R's by the rule in Numerical notes.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `wpca` | `src/datanuggets/wcluster/wpca.py`, `wpca` and `_wpca_parts`, with the eigen solver `r_eigen_sym` in `src/datanuggets/_compat/linalg.py` | WCluster 1.3.0, `Wpca`, which calls R 4.6.1, `eigen` |
| `dn_wpca` | `src/datanuggets/wcluster/dn.py`, `dn_wpca` | WCluster 1.3.0, `DN.Wpca` |

## Notation

The shared symbols are in [README.md](README.md#notation). This page adds:

- $`w_i \ge 0`$ is the weight of row $`i`$, and $`W = \sum_i w_i`$ is the total row weight. $`\mathbf{w} = (w_1, \dots, w_n)^\top`$ and $`D_w = \mathrm{diag}(w_1, \dots, w_n)`$.
- $`\omega_j`$ is the column weight given for variable $`j`$, and $`v_j`$ is the scaled one of D1. $`\mathbf{v} = (v_1, \dots, v_p)^\top`$ and $`D_v = \mathrm{diag}(v_1, \dots, v_p)`$.
- $`\bar{\mathbf{x}}`$ is the weighted center and $`X_0`$ the centered $`n \times p`$ data. $`\mathbf{1}`$ is the $`n`$-vector of ones.
- $`C`$ is the weighted covariance matrix, $`S`$ the matrix handed to `eigen` without `corr` and $`\tilde{S}`$ the one with `corr`. $`A`$ stands for whichever of the two is used.
- $`\mathbf{d} = (d_1, \dots, d_p)`$ is the scale under `corr`, and $`D_d = \mathrm{diag}(\mathbf{d})`$.
- $`\lambda_1 \ge \dots \ge \lambda_p`$ are the eigenvalues of $`A`$, and $`\mathbf{u}_a`$ is the eigenvector of $`\lambda_a`$. $`U = (\mathbf{u}_1, \dots, \mathbf{u}_p)`$, $`\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_p)`$ and $`Z = X_0 U`$ holds the scores.
- $`\boldsymbol{\xi}_l`$ is the center of nugget $`l`$, and $`g(i)`$ is the nugget of row $`i`$.
- In the Algorithm a hat marks a computed double, such as $`\hat{W}`$ and $`\hat{S}`$. $`\hat{\lambda}_a`$ and $`\hat{U}`$ are the eigenvalues and vectors that Step 9 returns.

## Definitions

D1. Column weights. R: `wcol <- wcol/sum(wcol) * length(wcol)`. The scaled weights sum to $`p`$:

```math
v_j = \frac{p \, \omega_j}{\sum_{j'=1}^{p} \omega_{j'}} .
```

D2. Center. R: `cwmean(x, wrow)`, which is `csum(x * wrow)/sum(wrow)`. With $`W \ne 0`$, $`\bar{\mathbf{x}} = W^{-1} \sum_i w_i \mathbf{x}_i = W^{-1} X^\top \mathbf{w}`$.

D3. Centered data. R: `sweep(x, 2, wm_x)`. $`X_0 = X - \mathbf{1} \bar{\mathbf{x}}^\top`$.

D4. Weighted covariance. The row weights act as counts. With integer weights summing to at least 2, $`C`$ is the sample covariance of the data with row $`i`$ repeated $`w_i`$ times:

```math
C = \frac{1}{W - 1} X_0^\top D_w X_0 .
```

D5. Eigen input without `corr`. R: `x_w` is $`X_0`$ with row $`i`$ scaled by $`\sqrt{w_i}`$ and column $`j`$ by $`\sqrt{v_j}`$, and `x_wcov` is `t(x_w) %*% x_w/(sum(wrow) - 1)`. With $`X_w = D_w^{1/2} X_0 D_v^{1/2}`$,

```math
S = \frac{1}{W - 1} X_w^\top X_w = D_v^{1/2} \, C \, D_v^{1/2} .
```

D6. Eigen input with `corr`. R: `dd = sqrt(diag(x_wcov))` and `diag(1/dd) %*% x_wcov %*% diag(1/dd)`. With $`p \ge 2`$ and $`d_j = \sqrt{S_{jj}}`$, this is $`\tilde{S} = D_d^{-1} S D_d^{-1}`$. With one column R computes something else (Numerical notes), and the port still computes $`\tilde{S}`$. When every $`v_j \gt 0`$ and every $`C_{jj} \gt 0`$, the column weights cancel:

```math
\tilde{S}_{jj'} = \frac{S_{jj'}}{\sqrt{S_{jj} S_{j'j'}}} = \frac{C_{jj'}}{\sqrt{C_{jj} C_{j'j'}}} .
```

D7. Eigen decomposition. R 4.6.1, `eigen`. $`A \mathbf{u}_a = \lambda_a \mathbf{u}_a`$ with $`U^\top U = I`$ and the eigenvalues in decreasing order.

D8. Outputs. `sdev` holds $`\sqrt{\lambda_a}`$, `rotation` holds $`U`$ and `x` holds the scores $`Z = X_0 U`$, with no weights applied. `center` is $`\bar{\mathbf{x}}`$, `wrow` is $`\mathbf{w}`$ and `wcol` is $`\mathbf{v}`$. `scale` is $`\mathbf{d}`$ under `corr`. Without `corr`, R's `scale` is the single number 1 and the port's is a vector of $`p`$ ones.

D9. `dn_wpca` is `wpca` with the nugget centers as rows, the nugget weights as $`\mathbf{w}`$ and $`\omega_j = 1`$ by default. R takes the table's columns 2 to `ncol - 2` as the data and its `Weight` column as the weights. The scales are not used. On a datanugget with one variable, R's default `wcol` stops with "invalid 'times' argument", and the port raises `ValueError` there. R: WCluster 1.3.0, `DN.Wpca`. Python: `src/datanuggets/wcluster/dn.py`, `dn_wpca`.

Five statements follow from the definitions, in exact arithmetic.

P1. Sign. With $`W \gt 1`$, $`C`$ is positive semidefinite, and so are $`S`$ and $`\tilde{S}`$. With $`W \gt 0`$ and $`W \lt 1`$, $`C`$ is negative semidefinite. Then no eigenvalue of $`S`$ is positive, and a negative one gives a NaN `sdev` with R's warning.

P2. Trace. $`\sum_a \lambda_a = \mathrm{tr}(A)`$, which is $`\sum_j v_j C_{jj}`$ without `corr` and $`p`$ with it.

P3. Unit weights. With $`n \ge 2`$, every $`w_i = 1`$ and every $`\omega_j = 1`$, $`S`$ is R's `cov(x)` and $`\tilde{S}`$ is R's `cor(x)`.

P4. Signs and ties. If $`\mathbf{u}_a`$ is an eigenvector, so is $`-\mathbf{u}_a`$. Inside a tie, any orthonormal basis of the eigenspace is one. The rotation and the scores are fixed only up to these choices. Beavers et al. (2025, Section 3.3) compare the components of the full data under unit weights with those of the nuggets under the nugget weights. They multiply some components by -1 to make the similarity of the structures more apparent.

P5. Scores. Since $`\mathbf{w}^\top X_0 = 0`$, each score column has weighted mean zero. The weighted covariance of the scores is

```math
\frac{1}{W - 1} Z^\top D_w Z = U^\top C \, U ,
```

which is $`\Lambda`$ when the column weights are equal and `corr` is off. A row with $`w_i = 0`$ adds nothing to $`\bar{\mathbf{x}}`$ or $`C`$ and still gets a score.

Nugget identity. Suppose $`\boldsymbol{\xi}_l`$ is the mean of the original rows in nugget $`l`$ and $`w_l`$ is their count, so $`\sum_l w_l = n`$. Let $`n \gt 1`$. Let $`C_{\mathrm{rows}}`$ be $`C`$ of the original rows with unit weights and $`C_{\mathrm{nug}}`$ be $`C`$ of the nugget centers with weights $`w_l`$. With $`\Delta_l = \sum_{i : g(i) = l} (\mathbf{x}_i - \boldsymbol{\xi}_l)(\mathbf{x}_i - \boldsymbol{\xi}_l)^\top`$,

```math
C_{\mathrm{rows}} = C_{\mathrm{nug}} + \frac{1}{n - 1} \sum_{l=1}^{m} \Delta_l .
```

Proof: the weighted mean of the centers is the mean of the rows, since the rows of nugget $`l`$ sum to $`w_l \boldsymbol{\xi}_l`$. Expand each row's term around its nugget center, and the cross terms sum to zero within each nugget. With the data centered at zero, $`C_{\mathrm{nug}}`$ is the $`S_{\mathrm{DN}}`$ of Beavers et al. (2025, Section 3). The identity is then the decomposition in the proof of Theorem 1.

The page's $`\Delta_l`$ is the scatter of nugget $`l`$ about its center. Beavers et al. (2025, Section 3) use the letter for a covariance. In this page's indices, with $`\mathbf{X}_l`$ the rows of nugget $`l`$, their matrix for nugget $`l`$ is $`\mathrm{Cov}(\mathbf{X}_l)`$ and their total is $`\Delta = \frac{1}{n-1} \sum_{l=1}^{m} (w_l - 1) \, \mathrm{Cov}(\mathbf{X}_l)`$. For $`w_l \ge 2`$ and $`\mathrm{Cov}`$ the sample covariance, $`(w_l - 1) \, \mathrm{Cov}(\mathbf{X}_l) = \Delta_l`$. A nugget of one row has $`\Delta_l = 0`$. With its term in $`\Delta`$ taken as zero, their $`\Delta`$ is $`\frac{1}{n-1} \sum_l \Delta_l`$. With the data centered at zero, the identity then reads $`S = S_{\mathrm{DN}} + \Delta`$ in their symbols, where their $`S`$ is $`C_{\mathrm{rows}}`$.

## Algorithm

$`\mathrm{fl}(\cdot)`$ rounds to double. $`\mathrm{ldsum}`$ is R's `sum()`, a long double running sum rounded once, as defined in [weighted-statistics.md](weighted-statistics.md). Steps 2 to 8 are `_wpca_parts`, and Steps 9 to 11 are `wpca`.

Step 1. Inputs. A 1-D `x` is one column. `corr` must be a bool. Each weight vector must have one finite numeric entry per row or column. A negative row weight raises `ValueError`. In R its square root is NaN, which reaches every entry of $`S`$, and `eigen` stops.

Step 2. Total. $`\hat{W} = \mathrm{ldsum}(w_1, \dots, w_n)`$, computed by `r_sum`. With the weights $`(1, 2^{-53}, 2^{-64})`$, R's total is exactly 1, while the exact sum rounded once to double is $`\mathrm{fl}(W) = 1 + 2^{-52}`$. When $`\hat{W} - 1 = 0`$, R divides by zero and `eigen` stops with "infinite or missing values in 'x'". The port raises `ValueError` before that, and also when $`\hat{W} = 0`$. With `corr` and $`\hat{W} \lt 1`$, no $`S_{jj}`$ is positive. A negative $`S_{jj}`$ gives a NaN $`d_j`$ and R's warning from `sqrt`. A zero $`S_{jj}`$ gives an infinite $`d_j^{-1}`$ and adds no warning. Either way R's matrix gets NaN entries and `eigen` stops. The port raises.

Step 3. Column weights. $`\hat{v}_j = \mathrm{fl}\left(\mathrm{fl}(\omega_j / \mathrm{ldsum}(\omega)) \cdot p\right)`$. Forty-nine weights of 1 give $`\hat{v}_j = \mathrm{fl}(\mathrm{fl}(1/49) \cdot 49) = 1 - 2^{-53}`$, not 1. Other equal weights can give exactly 1, as forty-nine weights of 0.3 do. A zero total or weights of mixed sign put infinite or NaN entries into R's matrix, and `eigen` stops. The port raises `ValueError`.

Step 4. Center. R's `csum(x * wrow)` is `rep(1, n) %*% (x * wrow)`. Under the default `matprod` with R's reference BLAS, this is one running double sum $`s_j`$ per column of the products $`\mathrm{fl}(x_{ij} w_i)`$, in row order from $`+0`$ ([numerical-parity.md](numerical-parity.md) D8). R 4.6.1: `src/main/array.c`, `matprod`, and `DGEMV` in `src/extra/blas/blas.f`. Each product with an entry of `rep(1, n)` is exact. Then $`\hat{\bar{x}}_j = \mathrm{fl}(s_j / \hat{W})`$. The port computes the same sums with `r_matvec(x.T, wrow)` and divides by $`\hat{W}`$.

Step 5. Centered data. $`\hat{X}_{0,ij} = \mathrm{fl}(x_{ij} - \hat{\bar{x}}_j)`$.

Step 6. Scaled data. $`\hat{X}_{w,ij} = \mathrm{fl}\left(\mathrm{fl}\left(\hat{X}_{0,ij} \, \mathrm{fl}(\sqrt{w_i})\right) \mathrm{fl}(\sqrt{\hat{v}_j})\right)`$, with the row factor first as in R's inner `sweep`.

Step 7. Covariance. Let $`P_{jj'}`$ be the running double sum of $`\mathrm{fl}(\hat{X}_{w,ij} \hat{X}_{w,ij'})`$ over $`i = 1, \dots, n`$ in order from $`+0`$. This is R's `t(x_w) %*% x_w` by [numerical-parity.md](numerical-parity.md) D8, through `DGEMM` for $`p \ge 2`$ and `DGEMV` for $`p = 1`$ when the entries are finite. The port computes it with `r_matmul`. Then $`\hat{S}_{jj'} = \mathrm{fl}\left(P_{jj'} / \mathrm{fl}(\hat{W} - 1)\right)`$. Entries $`(j, j')`$ and $`(j', j)`$ add the same products in the same order, so $`\hat{S}`$ is symmetric bit for bit when the input is finite.

Step 8. Correlation. $`\hat{d}_j = \mathrm{fl}(\sqrt{\hat{S}_{jj}})`$ and $`\hat{\tilde{S}}_{jj'} = \mathrm{fl}\left(\mathrm{fl}(\mathrm{fl}(1/\hat{d}_j) \, \hat{S}_{jj'}) \cdot \mathrm{fl}(1/\hat{d}_{j'})\right)`$. The products with R's diagonal matrices add only exact zeros. Entry $`(j, j')`$ rounds after the factor of $`j`$ and entry $`(j', j)`$ after the factor of $`j'`$, so $`\hat{\tilde{S}}`$ is symmetric only to rounding. A column with $`\hat{S}_{jj} = 0`$ puts NaN entries into R's matrix, and `eigen` stops. The port raises `ValueError`.

Step 9. Eigen. R's `eigen` tests the matrix with `isSymmetric.matrix` (R 4.6.1, base). For $`p \ge 2`$ that function first compares rows 1, 2, $`p - 1`$ and $`p`$ with the matching columns by `all.equal` at tolerance $`{800 \varepsilon}`$. It then compares the whole matrix with its transpose by `all.equal` at tolerance $`{100 \varepsilon}`$. `all.equal.numeric` takes the mean absolute difference over the entries that differ and scales it by their mean absolute value, or by 1 when that mean is at most the tolerance. $`\hat{S}`$ is exactly symmetric by Step 7. The two entries of a mirrored pair of $`\hat{\tilde{S}}`$ round the same exact product $`\mathrm{fl}(1/\hat{d}_j) \, \hat{S}_{jj'} \, \mathrm{fl}(1/\hat{d}_{j'})`$ twice each, in different orders. So when no rounded value is subnormal they differ by at most about $`{2^{-51}}`$ of their size, far inside both tolerances. Both matrices pass, and R calls `La_rs`. `La_rs` calls LAPACK's `dsyevr` with all eigenvalues, `abstol` 0 and the lower triangle, so only the lower triangle of $`A`$ is read. `dsyevr` returns increasing eigenvalues, and `eigen` reverses the values and the vector columns. The port's `r_eigen_sym` calls SciPy's `dsyevr` with the same arguments and reverses the same way.

Step 10. Standard deviations. `sdev` is $`\mathrm{fl}(\sqrt{\hat{\lambda}_a})`$, with the values from the same call that gives the vectors. When some $`\hat{\lambda}_a \lt 0`$, its `sdev` is NaN and the port warns "NaNs produced" once, as R does.

Step 11. Scores. $`\hat{Z}_{ia}`$ is the running double sum of $`\mathrm{fl}(\hat{X}_{0,ij} \hat{U}_{ja})`$ over $`j = 1, \dots, p`$ in order from $`+0`$. This is R's `x_0 %*% vectors` by [numerical-parity.md](numerical-parity.md) D8, and the port computes it with `r_matmul`. A score whose sum leaves the double range is infinite with no warning, as in R.

## Numerical notes

Steps 2 to 8 give R's bits, apart from the one-column case below. The eigen output does not. R uses its bundled reference LAPACK, the port uses SciPy's, and two correct builds can differ in the last bits and in the sign of each vector. Inside a tie they can return different bases. The tests therefore judge `sdev`, `rotation` and `x` by the rule in `tests/rfixtures.py`, `eigen_mismatches`. R1: the rotation is orthonormal. R2: each vector's residual against R's eigenvalue, on the symmetric matrix built from the lower triangle of R's eigen input, is within a band of `EIG_K` = 1024 machine epsilons times the largest absolute eigenvalue of R's. R3: each `sdev` squared is within that band of R's eigenvalue, and none is negative. A NaN `sdev` passes only where R's eigenvalue is at most the band. R4: R's eigenvalues, in decreasing order, are split into clusters wherever two neighbors differ by more than `EIG_TIE` = $`{10^{-12}}`$ times that largest absolute eigenvalue. With $`U_c`$ the columns of $`U`$ in cluster $`c`$, each projector $`U_c U_c^\top`$ matches R's within a tolerance set by its gap to the other eigenvalues. R5: the scores projected on each cluster match R's row by row. A NaN or an infinity in the port's output fails every check it reaches. The rule ignores signs and the basis inside a tie, which is the freedom of P4.

Near-zero eigenvalues. A rank-deficient $`S`$, for example from more columns than rows, can give an eigenvalue just below zero from rounding. R then warns, while another LAPACK build can give one just above zero. The port's warning is held to R's when an eigenvalue of R's lies below the band, or when all of them lie above it. Inside the band the port warns exactly when its own `sdev` has a NaN.

One column with `corr`. R's `diag(1/dd)` of one number is an identity matrix of size `as.integer(1/dd)`. So R returns the unscaled $`S`$ when $`d_1^{-1}`$ is at least 1 and below 2, and stops otherwise. The port divides by $`d_1`$ and returns an `sdev` of 1 up to rounding. Where R runs, its `sdev` equals its `scale`, $`\hat{d}_1`$. Where R stops, the port returns a result.

## Tests

The fixtures in `tests/fixtures/wpca/` were recorded from R 4.6.1 with WCluster 1.3.0 and R's bundled LAPACK. They hold R's inputs, center, column weights, $`X_0`$, the eigen input, the covariance before `corr`, the eigenvalues, `sdev`, `rotation`, scores and warnings. The cases cover unit and weighted rows, unequal column weights, `corr` on and off, one column, a zero row weight, row weights summing to less than 1, a tied eigenvalue, a constant column, more columns than rows, row weights whose long double sum differs from other summation orders, and three stops (`one_iris_first5_petalwidth`, `sum_ld_stops`, `sum_ld_steps_stops`).

- `tests/wcluster/test_wpca.py::test_parts_match_r` holds the center, $`\hat{\mathbf{v}}`$, $`\hat{X}_0`$, the eigen input, the scale and, under `corr`, the covariance before scaling to R's bits. It runs on every case where R returns a result, except `petalwidth_1d_corr`, for row-major and column-major input. The volcano cases with 49 column weights of 1 check $`\hat{v}_j = 1 - 2^{-53}`$.
- `tests/wcluster/test_wpca.py::test_scores_product_matches_r` gives R's scores bit for bit from R's $`X_0`$ and rotation. `tests/wcluster/test_wpca.py::test_wpca_matches_r` holds the center, scale and weights to R's bits. It holds the scores to the product of R's $`X_0`$ with the port's rotation and applies the eigen rule.
- `tests/wcluster/test_wpca.py::test_fixtures_cover_the_edge_cases` checks that the cases include one column, a zero row weight, a tie, cases below and above the band, and a recorded $`\tilde{S}`$ that is not exactly symmetric. `tests/wcluster/test_wpca.py::test_eigen_reads_the_lower_triangle` checks that the port's eigen output from such a $`\tilde{S}`$ equals its output from the mirrored lower triangle, bit for bit.
- `tests/wcluster/test_wpca.py::test_eigen_rule_rejects_wrong_outputs`, `tests/wcluster/test_wpca.py::test_eigen_rule_fails_closed`, `tests/wcluster/test_wpca.py::test_eigen_rule_constants_are_the_plans` and `tests/wcluster/test_wpca.py::test_eigen_rule_accepts_a_rotated_tie` test the rule itself. `tests/wcluster/test_wpca.py::test_sdev_comes_from_the_vectors_call` and `tests/wcluster/test_wpca.py::test_warnings_match_r` cover Step 10.
- `tests/wcluster/test_wpca_sums.py::test_cases_are_pinned` checks that each long double case separates R's total from exactly the other sums it names. `tests/wcluster/test_wpca_sums.py::test_stops_where_r_stops` covers the weights $`(1, 2^{-53}, 2^{-64})`$ of Step 2, and `tests/wcluster/test_wpca_sums.py::test_runs_where_r_runs` the cases where R runs.
- `tests/wcluster/test_wpca.py::test_weights_summing_to_one_raise`, `tests/wcluster/test_wpca.py::test_one_row`, `tests/wcluster/test_wpca.py::test_corr_with_weights_summing_below_one_raise`, `tests/wcluster/test_wpca.py::test_constant_column_with_corr_raises`, `tests/wcluster/test_wpca.py::test_weight_rules`, `tests/wcluster/test_wpca.py::test_corr_must_be_a_bool` and `tests/wcluster/test_wpca.py::test_takes_1d_x_as_one_column_and_rejects_non_finite_x` cover Steps 1 to 3 and 8. `tests/wcluster/test_wpca.py::test_weights_summing_below_one_warn_once` covers P1, `tests/wcluster/test_wpca.py::test_overflow_is_quiet_as_in_r` Step 11, and `tests/wcluster/test_wpca.py::test_one_column_corr_departs_from_r` the one-column note.
- `tests/wcluster/test_dn.py::test_dn_wpca_matches_r` compares `dn_wpca` with R on `tests/fixtures/dn_wcluster/`, with and without `corr`. `tests/wcluster/test_dn.py::test_one_variable_nuggets_stop_where_r_stops` covers D9's stop.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. Used: Section 3, the proof of Theorem 1 and Section 3.3.
- Dey, R., Duan, Y., Cabrera, J., and Cheng, G. (2025). WCluster: Clustering and PCA with Weights, and Data Nuggets Clustering. R package version 1.3.0. Used: `Wpca`, `DN.Wpca` and their help pages.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria. Used: `eigen`, `isSymmetric.matrix`, `all.equal.numeric`, `sum`, `diag`, `src/main/array.c` (`matprod`), `src/extra/blas/blas.f` (`DGEMV`, `DGEMM`) and `src/modules/lapack/Lapack.c` (`La_rs`).

## Not covered by a test

- Negative row weights, column weights of mixed sign or zero total, a row-weight total of zero, `corr` with $`\hat{W} \lt 1`$ and `corr` with a zero $`\hat{S}_{jj}`$: the tests check the port's `ValueError`, and no fixture records R's stop.
- The one-column `corr` case where $`\hat{d}_1^{-1}`$ is below 1 or at least 2: the fixtures `tests/fixtures/compat/wpca1_corr_*` record R's results for single columns, including its stop with "non-conformable arguments" here, and no test reads them.
- Step 11's infinite score: `tests/wcluster/test_wpca.py::test_overflow_is_quiet_as_in_r` checks the port, and no fixture records R's output.
- Step 9's `isSymmetric.matrix` call: no fixture records its result.
- P2, P3, P5 and the nugget identity: no test checks them. They are statements in exact arithmetic.
- The cancellation of the column weights under `corr` in D6: no test checks it. The only fixture with unequal column weights is `mtcars_carb_invsd`, and it has `corr = FALSE`.
