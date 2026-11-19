# Nugget means and arithmetic complexity

Draft. This page has not yet been reviewed by the authors of the R packages.

This page covers two small functions from datanugget 1.5.0. `get_dn_means` recomputes the mean of the rows assigned to each nugget, whatever center method built the nuggets. `arithmetic_complexity` ports `AC`, a cost model for `create.DN` ([create-dn.md](create-dn.md)) on the $`\log_{10}`$ scale.

## Scope

| Python function | Code | R function |
| --- | --- | --- |
| `get_dn_means` | `src/datanuggets/nuggets/means.py`, `get_dn_means` | datanugget 1.5.0, `getDN.means` |
| `arithmetic_complexity` | `src/datanuggets/nuggets/complexity.py`, `arithmetic_complexity`, `_ac_state`, `_shrink` and `_r_log10` | datanugget 1.5.0, `AC` |

`get_dn_means` returns an $`m \times p`$ float array where R returns a data frame. With $`p = 1`$ R's frame has one row and $`m`$ columns, and the port still returns $`m`$ rows. The port raises `ValueError` where R ignores rows of `x` past the assigned ones or skips a row whose nugget id is out of range or not a whole number. It also raises on NaN or infinite values. R's `AC` stops on a matrix or a data.table because its class check gets a condition of length 2, and it accepts a plain data frame. `arithmetic_complexity` takes any 2-D input and reads only its shape. It raises where a loop of `AC` never ends and on a `dn_num1` or `dn_num2` that is not finite. It also raises on an input with no rows, where R's chunk count $`G`$ is 0.

The page does not compare $`A`$ and $`B`$ with the ranges that Beavers et al. (2025), "Appendix on computational cost" gives for $`\psi_1`$ and $`\psi_2`$.

## Notation

The shared symbols are in [README.md](README.md#notation). Here $`\mathrm{fl}(\cdot)`$ is an operation rounded to double. This page adds the following.

| Symbol | Meaning |
| --- | --- |
| $`I_l`$ | the rows assigned to nugget $`l`$, in increasing order |
| $`n_l`$ | the number of rows in $`I_l`$ |
| $`\boldsymbol{\mu}_l`$ | the mean of nugget $`l`$ |
| $`C`$ | the deletion fraction, `delete_percent` |
| $`r_0`$, $`m_1`$, $`m_2`$ | the floors of `r`, `dn_num1` and `dn_num2`, in the role of the paper's $`R`$, $`M_{init}`$ and $`M`$ |
| $`G`$ | the chunk count of `AC` |
| $`t`$ | the target of loop 1 |
| $`\kappa(v)`$, $`T_\tau`$ | the cut and the pass of D4 |
| $`A`$, $`B`$ | the counters of loops 1 and 2, in the role of the paper's $`\psi_1`$ and $`\psi_2`$ |
| $`W`$ | the operation count |

## Definitions

D1. Nugget mean. For $`n_l \ge 1`$,

```math
\boldsymbol{\mu}_l = \frac{1}{n_l} \sum_{i \in I_l} \mathbf{x}_i .
```

Each entry is computed as R's `colMeans` computes it. The column is summed in long double from the first row of $`I_l`$ to the last. The sum is divided by $`n_l`$ in long double and rounded to double once. R: R 4.6.1, `src/main/array.c`, `do_colsum`. Port: `src/datanuggets/_compat/arithmetic.py`, `r_mean`. [numerical-parity.md](numerical-parity.md) describes this arithmetic.

D2. Empty nugget. If $`n_l = 0`$, every entry of $`\boldsymbol{\mu}_l`$ is the long double zero divided by zero that R computes. On the platform the fixtures record, x86_64-w64-mingw32 with a 64-bit long double significand, this is a NaN with its sign bit set. Its bits are `fff8000000000000`.

D3. Floors and chunks. After its checks `AC` floors three arguments: $`r_0 = \lfloor r \rfloor`$ and likewise $`m_1`$ and $`m_2`$. It does not floor $`C`$. Then

```math
G = \lceil n / r_0 \rceil, \qquad t = \mathrm{fl}(m_1 / G).
```

$`t`$ is not rounded to a whole number.

D4. One pass. For a value $`v`$ let $`\kappa(v) = \lfloor \mathrm{fl}(C v) \rfloor`$. A pass toward a target $`\tau`$ maps $`v`$ to $`T_\tau(v)`$ as follows.

1. If $`\mathrm{fl}(v - \kappa(v)) \lt \tau`$, then $`T_\tau(v) = \mathrm{fl}(v - \mathrm{fl}(v - \tau))`$.
2. Otherwise, if $`\kappa(v) = 0`$, then $`T_\tau(v) = \mathrm{fl}(v - 1)`$.
3. Otherwise $`T_\tau(v) = \mathrm{fl}(v - \kappa(v))`$.

Case 1 jumps to the target. In exact arithmetic it lands on $`\tau`$. In doubles its two roundings can leave it beside $`\tau`$, and the loop then goes on.

D5. Counters. With $`T_\tau^{h}`$ for $`h`$ passes,

```math
A = 1 + \min\lbrace h \ge 0 : T_t^{h}(r_0) = t \rbrace, \qquad B = 1 + \min\lbrace h \ge 0 : T_{m_2}^{h}(m_1) = m_2 \rbrace .
```

If a set is empty, that loop of `AC` never ends.

D6. Count. Each loop exits only on equality. So when both have ended, R's `R` holds $`t`$ and its `DN.num1` holds $`m_2`$. `AC` returns $`\log_{10} W`$ with

```math
W = p \left( G \cdot \frac{t(t-1)}{2} \cdot A + \frac{m_2(m_2-1)}{2} \cdot B + (n+1)\, m_2 \right).
```

R evaluates $`W`$ in doubles in the order of its code, and the port uses the same order. As in R, $`\log_{10} W`$ is $`-\infty`$ when $`W = 0`$. It is NaN with the warning "NaNs produced" when $`W \lt 0`$.

## Algorithm

`get_dn_means` runs four steps.

Step 1. Inputs. Read the assignments and the number of nuggets $`m`$. R takes $`m`$ from the rows of the "Data Nuggets" frame. The values of the centers, weights and scales are not used. The result depends only on $`X`$, the assignments and $`m`$.

Step 2. Groups. Form $`I_l`$ for each nugget with its rows in increasing order. R collects them with `which`, and the port uses a stable argsort of the ids.

Step 3. Means. Compute $`\boldsymbol{\mu}_l`$ by D1, or by D2 for an empty nugget.

Step 4. Output. Return the $`m`$ means as the rows of an $`m \times p`$ array.

`arithmetic_complexity` runs five steps.

Step 1. Checks. These run on the values before flooring. $`r`$ must lie in $`[100, 10000]`$ and $`C`$ in $`(0, 1)`$. `dn_num2` must not exceed `dn_num1`.

Step 2. Floors. Floor the arguments and set $`G`$ and $`t`$ by D3. Only $`n`$ and $`p`$ are read from $`X`$.

Step 3. Loop 1. Start at $`v = r_0`$ with $`A = 1`$. While $`v \ne t`$, set $`v`$ to $`T_t(v)`$ and add 1 to $`A`$.

Step 4. Loop 2. Start at $`v = m_1`$ with $`B = 1`$. While $`v \ne m_2`$, set $`v`$ to $`T_{m_2}(v)`$ and add 1 to $`B`$.

Step 5. Result. Return $`\log_{10} W`$ by D6.

The port adds two guards to steps 3 and 4. The first raises before a loop whose target is below $`-1`$ and whose start is above it. From a whole-number start above such a target, every pass gives a whole number of at least $`\min(v, -1)`$, so the target is never reached. The second raises when the current value equals a value saved earlier. A pass depends only on $`v`$, so a repeated value means the loop cycles without reaching its target. The saved value is replaced at power-of-two intervals, so the check needs constant memory. It can fire some passes after the first repeat. It fires once the saved value lies on the cycle and the interval is at least as long as the cycle. A loop that never ends must repeat a value, since there are finitely many doubles, so one of the guards fires on it.

## Numerical notes

Order of summation. Long double addition is not associative, so the order of the rows in D1 can change a mean. Where the rows cancel, the change can be far larger than the last bit. In the fixture `get_dn_means/row_order`, R sums the first column of nugget 1 as (1e20 - 1e20) + 1 and gets a mean of 1/3 rounded to a double. Summed as (1 + 1e20) - 1e20, the same rows give a mean of 0. R and the port both add the rows of $`I_l`$ in increasing row order.

Mean centers. With `center_method="mean"` and $`n_l \ge 1`$, `create_dn` returns a center equal to $`\boldsymbol{\mu}_l`$ in exact arithmetic ([create-dn.md](create-dn.md), D6). The two are computed by different formulas, so their last bits can differ. For an empty nugget `create_dn` keeps the stage-2 center, and `get_dn_means` gives NaN.

Double rounding in the loops. The cut floors the rounded product $`\mathrm{fl}(C v)`$, and the jump of case 1 can land beside $`t`$. So $`A`$ and $`B`$ can differ from the counts that exact arithmetic gives. The port runs the loops in R's double arithmetic and in R's order of operations.

Whole-number targets. Let $`t`$ be a whole number with $`r_0 \gt t \ge 0`$. Every value then stays a whole number, and case 1 lands exactly on $`t`$. Each pass lowers $`v`$ by at least 1 without going below $`t`$. So $`A \ge 2`$ and $`A \le r_0 - t + 1`$. The same argument gives $`B \ge 2`$ and $`B \le m_1 - m_2 + 1`$ when $`m_1 \gt m_2 \ge 0`$ and $`m_1 \lt 2^{53}`$.

Fractional targets. Let $`t`$ not be a whole number, with $`r_0 \gt t \ge 0`$ and $`\mathrm{fl}(C r_0) \lt 1`$. Every cut is then 0. Loop 1 steps down by 1 from $`r_0`$ to $`\lfloor t \rfloor`$, which is below $`t`$, and the next pass jumps up to $`t`$. That jump is exact. The difference $`\lfloor t \rfloor - t`$ is minus the fractional part of $`t`$. The fractional part of a double is itself a double, so this subtraction is not rounded. Subtracting it from $`\lfloor t \rfloor`$ gives $`t`$, which is a double, so that result is not rounded either. So $`A = r_0 - \lfloor t \rfloor + 2`$. This count is above $`r_0 - t + 1`$, so the bound of the whole-number case does not hold for a fractional target.

## Tests

The fixtures in `tests/fixtures/get_dn_means` hold R's `getDN.means` on every nugget set of the `create_dn` fixtures. They also hold it on the sets passed to and returned by `refine.DN` in the `refine_dn` fixtures. They were recorded with R 4.6.1 and datanugget 1.5.0 on x86_64-w64-mingw32 with a 64-bit long double significand. They include sets with an empty nugget and sets with one column. A constructed set fixes the order of the sum.

- `tests/nuggets/test_means.py::test_get_dn_means_matches_r`: every entry to R's bits, NaN included.
- `tests/nuggets/test_means.py::test_empty_nugget_is_rs_negative_nan`: D2.
- `tests/nuggets/test_means.py::test_sums_each_nugget_in_row_order_like_r`: the row order of D1.
- `tests/nuggets/test_means.py::test_fixtures_cover_empty_nuggets_and_one_column`: the fixtures hold a set with an empty nugget and a set with one column and more than one nugget.
- `tests/nuggets/test_means.py::test_x_must_have_the_rows_dn_assigns`, `tests/nuggets/test_means.py::test_assignments_must_be_nugget_ids` and `tests/nuggets/test_means.py::test_x_must_be_finite`: the checks where the port raises.

The fixture `tests/fixtures/ac/grid` holds R's `AC` on the row and column counts of eight R datasets. These shapes are crossed with values of `R`, `delete.percent`, `DN.num1` and `DN.num2`. For each call it records what R returned and whether it warned. It also records $`G`$ and the counters $`A`$ and $`B`$ with the end values of `R` and `DN.num1`. It marks the calls that did not return.

- `tests/nuggets/test_complexity.py::test_counts_value_and_warning_match_r`: the chunk count, both counters and both end values equal R's on every call that returned. The value agrees within the tolerances `RTOL` and `ATOL` of `tests/rfixtures.py`, and the warning agrees too.
- `tests/nuggets/test_complexity.py::test_raises_where_r_never_returns`: the port raises on every call that did not return.
- `tests/nuggets/test_complexity.py::test_raises_at_once_where_r_counts_down_forever` and `tests/nuggets/test_complexity.py::test_returns_like_r_when_dn_num1_already_equals_a_target_below_minus_one`: the first guard.
- `tests/nuggets/test_complexity.py::test_grid_needs_r_double_rounding`: some grid calls have an $`A`$ that differs from the count in exact arithmetic.
- `tests/nuggets/test_complexity.py::test_reads_only_the_shape_of_any_2d_input` and `tests/nuggets/test_complexity.py::test_x_must_be_2d_with_rows`: the port gives R's value for an array, a data frame with a text column and a list of rows of the same shape. It raises on a 1-D input and on an input with no rows. With $`p = 0`$ the count is 0 and the result is $`-\infty`$.
- `tests/nuggets/test_complexity.py::test_rejects_what_r_rejects`: the checks of step 1.

## References

- Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`.
- Dey, R., Duan, Y., Beavers, T., Cabrera, J., Cheng, G., Qi, K., and Lubomirski, M. (2026). datanugget: Create, Optimize, and Refine Data Nuggets. R package version 1.5.0.
- R Core Team (2026). R: A Language and Environment for Statistical Computing. Version 4.6.1. R Foundation for Statistical Computing, Vienna, Austria.

## Not covered by a test

- The bounds on $`A`$ and $`B`$ of "Whole-number targets". No test asserts them.
- The count $`A = r_0 - \lfloor t \rfloor + 2`$ of "Fractional targets". No call in `tests/fixtures/ac/grid` that returned has a fractional target with $`r_0 \gt t \ge 0`$ and $`\mathrm{fl}(C r_0) \lt 1`$.
- That $`B`$, like $`A`$, can differ from the count in exact arithmetic. `test_grid_needs_r_double_rounding` checks $`A`$ only, and it does not tell which rounding changed $`A`$, the cut or the jump of case 1 in D4.
- That the mean centers of `create_dn` can differ from the means of `get_dn_means` ("Mean centers"). No test compares the two.
- What R's `AC` does with a matrix or a data.table, and what R's `getDN.means` does with extra rows of `x` or with ids that are out of range or not whole numbers. The tests check only the port's side.
