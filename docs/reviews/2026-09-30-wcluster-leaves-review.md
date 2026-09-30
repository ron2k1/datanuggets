# WCluster leaves review, 2026-09-30

The range under review is 4de33ae..fc9ffe2 on `feat/wcluster-leaves`, the port of WCluster's `wmean`, `wss`, `wwcss`, `distw`, `cluster.predict` and `Wpca`. Its findings are fixed in 659413a..6d052d9.

Two reviews ran in parallel.

- Codex (gpt-6-astra, reasoning effort max, read-only sandbox) ran on the laptop, Windows with Python 3.14.3. Its verdict was REJECT, on B3 and B4. Its final report is [2026-09-30-wcluster-leaves-codex.txt](2026-09-30-wcluster-leaves-codex.txt), verbatim.
- Three Opus reviewers each took one lens: parity with R, the person calling the package, and the tests and parity pages. For each lens a second Opus agent reran every finding and listed what the first had missed.

## Did the claims hold

Codex checked the branch's seven claims with a command each. F1 (`r_matmul` and `r_weighted_mean`), F2 (the weight and label checks), F4 (`cluster_predict`) and F5 (`wpca`) held. F3 (`wmean`, `wss`, `wwcss` and `distw`) was partial: all 784 of its comparisons matched, but B3 showed a docstring sentence was false. F7 (README and spec) was partial because of B4. F6 didn't hold, since `WpcaResult` and `WwcssResult` import from `datanuggets` but not from `datanuggets.wcluster`. The plan puts them at the top level only (docs/plans/2026-09-30-feature4-wcluster-leaves.md:627), so the claim overstated the plan and nothing changed. Part A of the Codex report has the command and verbatim output for each.

## Findings

| Finding | Reviewer | Fixed in | Evidence it was real |
|---|---|---|---|
| the tests and parity pages can't see the sign of a zero | parity F1, tests F4 | 659413a | two mutants passed, see below |
| the eigen rule passes NaN scores | tests lens, missed list | d0a6388 | all-NaN scores beside R's rotation gave no mismatch on four cases |
| the eigen rule passes a negative sdev | Codex B1 | d0a6388 | `test_wpca_matches_r[iris]` passed with every sdev negated |
| the docs say a zero-sum cluster's center is NaN | Codex B3, person lens missed list | 6d052d9 | `w = [1, -1]` gives `[[-inf]]`, and so does R |
| the README makes a row with no finite distance an error | Codex B4 | 6d052d9 | `newx = [[1e200]]` has one `Inf` distance and returns `[0]`, as R does |
| the README says `distw` and `cluster_predict` match R's NaN bits | parity lens missed list | 6d052d9 | the NaN table below |
| the ledger's deferred note on `wcol = [-1, 0, -1, -1]` says R gives +0 | tests F5 | ledger | R 4.6.1 gives -0 too: `1/r$wcol` prints `0.75 -Inf 0.75 0.75` |

### Sign of zero

`assert_array_equal` and `np.array_equal` compare with `==`, and `-0.0 == 0.0`. A port that lost R's negative zero, or made one where R has none, passed every WCluster test and still read "yes" on parity pages that called the comparison bit for bit. R's `%*%` sums from +0 in the reference BLAS, so `r_matvec` must never return -0. Two mutants showed the gap.

- `r_matvec` starting from its first product instead of +0 makes `wpca`'s center -0 on an all -0 column, where live R 4.6.1 gives +0. It passed 1190 tests. `test_r_matvec_never_returns_negative_zero` fails on it now.
- `_wpca_parts` returning `s + 0.0` turns the -0 that R leaves in the eigen input of the two `den_iris` cases into +0. It passed `test_wpca` at fc9ffe2. Four `test_parts_match_r` cases fail on it now.

`same_values` in `tests/rfixtures.py` compares finite entries as float64 bit patterns and lets any NaN match any NaN. Every float comparison in the WCluster tests and every parity page uses it now. No page row turned to NO.

### Eigen rule

`eigen_mismatches` flagged a value only when `off > tol` held, and a comparison with NaN is always false. Beside R's own rotation and sdev, all-NaN scores gave no mismatch on iris, tie_npk, quakes_stations and volcano_scaled. The exact check `got.x == r_matmul(cap_x_0, got.rotation)` at both call sites caught them, so no wrong output got through, but the rule alone didn't. R3 compared sdev squared, so a port returning -sdev passed `test_wpca_matches_r[iris]` in Codex's run.

Each clause now passes only a value shown to be inside its bound, `not off <= tol`. R3 also needs sdev to be zero or more, and the spectral norm runs only on a finite difference. `test_eigen_rule_fails_closed` feeds four outputs no LAPACK gives, on iris and on tie_npk: NaN scores, an infinite score, a NaN rotation and a negative sdev. Before the fix 7 of the 8 failed. The iris infinite score was already caught, since iris's rotation has no zero entry to turn `Inf` into NaN.

### Zero-sum weights

R's `weighted.mean` is `sum((x*w)[w != 0]) / sum(w)`. When every weight is zero that's 0/0, so NaN. When weights of both signs cancel, a nonzero sum is divided by zero, which gives an infinity, with NaN only in a column whose weighted sum is zero too. The port already did what R does. The docs said NaN in both cases, so a caller finding dead clusters with `np.isnan(centers)` would miss a cancelled one.

These values were recorded from R 4.6.1 by hand. `x` is the first six iris rows and three columns, with the first two rows of each order in cluster 1, weighted 1 and -1.

| rows in cluster 1 | R's center for cluster 1 | its `wwcss` | the `distw` entry |
|---|---|---|---|
| iris rows 3 and 1 | `fff0000000000000` in all three columns | NaN | NaN |
| iris rows 1 and 2 | `7ff0000000000000`, `7ff0000000000000`, `fff8000000000000` | NaN | NaN |

With rows 3, 1, 2, 4, 5, 6 as `x`, R's `cluster.predict` passes over the infinite center. For iris row 7 alone the distances are `7ff0000000000000` (`Inf`) and `3fc347ae147ae200`. For rows 7, 60 and 120 together the multiply by the identity turns the first column NaN. Both give position 2 in R, 1 in the port.

`test_cancelling_weights_give_infinities_as_in_r` and `test_infinite_center_is_passed_over_as_in_r` pin all of it. The code was right, so there was no RED to watch. A mutant that writes the old docstring as code, NaN whenever the weights sum to zero, fails both tests and the existing zero-sum `weighted.mean` fixture.

### NaN bits

`wmean`, `wss` and `wwcss` match R bit for bit in all 49 reported cases, NaN bits included. `distw` and `cluster_predict` don't match R's NaN bits, and the README said they did.

| output | port | R | where |
|---|---|---|---|
| `distw` cells R never fills | `7ff8000000000000` | `7ff00000000007a2`, R's NA | every case, 113,265 cells in the 45 reported |
| `cluster_predict` distances made NaN by the identity multiply | `7ff8000000000000` | `fff8000000000000` | 6 of the 39 reported cases, 2,985 cells |

The six are chick_diettime_time, mtcars_carb_am, mtcars_cyl_vs, mtcars_gear_am, mtcars_gear_am_row01 and mtcars_singletons_vs. Codex's report lists the constructed cases too. The README now says each number matches bit for bit and each NaN sits in the same place.

## Codex's sandbox failures

Codex's full-suite run ended `4 failed, 2080 passed, 2 skipped, 140 deselected, 1 error`. It traced all five to the read-only sandbox: `test_every_kernel_caches_when_numba_can` had no writable numba cache, and the other four found no temporary directory (the no-cache test's `tmp_path`, both `test_n_jobs_does_not_change_the_result` cases through joblib, and the workqueue test's child pytest). All five pass outside the sandbox.

## Test runs

After each fix the full suite ran on the laptop: 2086 passed after 659413a, 2094 after d0a6388 and 2098 after 6d052d9, each with 2 skipped (`needs os.fork`) and 140 deselected as slow. `ruff check` and `ruff format --check` pass. The fixtures come only from the laptop's R 4.6.1, as their manifests record. The values in the two new pins were read from R by hand and are written into the tests in hex.

## Unverified

- A mutant returning `q` in place of R's NaN in `r_weighted_mean` survives on the laptop, where 0/0 already gives `fff8000000000000`. It should die on arm64, whose 0/0 gives `7ff8000000000000`, in `test_r_weighted_mean_matches_r[wmean_mtcars_gear3_mpg_am_zero_total]`. That test passes on the mini, but the mutant wasn't run there.
- A mutant that warns only below -1e-300 survives in `wpca`. No input was built with an eigenvalue in [-1e-300, 0).
- Dropping R1 from the eigen rule survives, since R4 at the same floor catches a rotation that isn't orthonormal to first order. That wasn't proven for tied clusters.
- No real column in the wpca or wstats fixtures separates R's long double sum from `math.fsum`. Other R datasets weren't searched beyond nine checked against a plain double sum.
