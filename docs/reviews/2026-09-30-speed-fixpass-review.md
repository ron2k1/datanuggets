# Speed fix pass review, 2026-09-30

The range under review is 43637f2..63fa1bd on `feat/speed`, the fix pass on the compiled long double plan. Its findings are fixed in 1110370..99f2b7e.

Two reviewers ran in parallel.

- Codex (gpt-6-astra, reasoning effort max, read-only sandbox) ran on the laptop, Windows with Python 3.14.3. Its verdict was REJECT. Its final report is [2026-09-30-speed-fixpass-codex.txt](2026-09-30-speed-fixpass-codex.txt), verbatim.
- A companion review on Opus ran its probes on the mini, arm64 macOS.

## Did the claimed fixes land

Codex ran a command for each claimed fix and quoted the result line. F1 to F10 and F12 landed. F11 (the no-cache test finds every kernel module) was partial, and that's finding B2 below. The table in Part A of the Codex report has the command, inputs and verbatim output for each.

## Findings

| Finding | Reviewer | Fixed in | Evidence it was real |
|---|---|---|---|
| sum() and mean() return the wrong NaN | Codex B1, companion F1 | 1110370 | See below |
| mean() takes the wrong correction branch when the sum overflows | companion F2 | b3bd2b3 | `mean_overflow_careful` failed |
| a child forked during a launch inherits a held lock | companion F3 | b754139 | the child hung until SIGALRM |
| cov, %*% and vector stats tests can't see a sign flip | Codex C1 | 64ff99e | Codex's mutants passed the old tests |
| kernel discovery rests on an unchecked convention | Codex B2 | 3b43c78 | see the ruling below |
| the plan's `grep -rn` step can't produce no output | companion F4 | 99f2b7e | three `.pyc` matches in the main tree |
| the plan says every kernel is `nogil` with no `parallel` | companion nit | 99f2b7e | `_ldjit.group_moments_parallel` is `@compiled(parallel=True)` |
| `_layer_forbids_fork` cites parallel.py | companion nit | b754139 | numba 0.67.0 `omppool.cpp` lines 120 to 124 hold the check |

### NaN identity

R sums in x87 long double. On the laptop's R 4.6.1, `sum(c(Inf, -Inf, 1))` is `fff8000000000000`, the x87 default NaN with its sign bit set. HEAD returned `7ff8000000000000`, while 43637f2 had it right. `assert_array_equal` treats every NaN as equal, so the regression passed. From Codex's run:

```
fixture input= [inf, -inf, 1.0] expected= 0xfff8000000000000
HEAD r_sum actual=0x7ff8000000000000 same_bits=False
43637f2 r_sum actual=0xfff8000000000000 same_bits=True
existing assert_array_equal comparison: PASS
```

The companion showed the problem is wider than one sign bit. R's NA is a signaling NaN, and which NaN survives depends on the x87's rules for mixing NaNs. mean() quiets every value first by dividing it by n in SSE. New fixtures recorded on R 4.6.1 pin this down:

| input | sum() | mean() |
|---|---|---|
| NA, 1 | `7ff80000000007a2` | `7ff80000000007a2` |
| Inf, -Inf, NA | `fff8000000000000` | `7ff80000000007a2` |
| NaN payload 123, NA | `7ff8000000000123` | `7ff80000000007a2` |
| NA, signaling NaN `7ff4000000000000` | `7ff80000000007a2` | `7ffc000000000000` |
| `7ff8000000000000`, `fff8000000000000` (either order) | `7ff8000000000000` | `7ff8000000000000` |

Before the fix, 18 tests failed on the mini: 8 sum cases, 6 mean cases (5 NaN cases plus `mean_overflow_careful`), both rcompat means cases (`nan_bits`, and `non_finite`, whose row `Inf, 1, -Inf` came back `7ff8000000000000` where R's rowMeans gives `fff8000000000000`), and the two fork tests. After the fix, `_non_finite_sum` walks the Inf and NaN entries as bit patterns, and every fixture matches on arm64.

### Overflow branch

R 4.6.1's `real_mean` (summary.c, lines 479 to 521) has two correction passes. When the long double sum overflows a double, it sums x / n and then divides each residual by n before adding it. `r_vec_mean` used the finite branch for both. The fixture `mean_overflow_careful` holds seven doubles whose long double sum overflows a double. Three are within a factor of 2 of DBL_MAX, and the other four are between 2^958 and 2^971 in size. R gives `7fd6db6db6db6db5`.

### Fork lock

`test_a_child_forked_during_a_launch_gets_a_free_lock` failed with `assert not True`, because the copied lock was still held. `test_group_moments_runs_in_a_child_forked_during_another_threads_launch` got `(-14, b'')` from the child: exit by SIGALRM after its 60 s alarm, having sent nothing. Both pass once the at-fork hook binds a fresh lock.

### Comparator strength

Codex flipped the 12 zeros in `cov_two_rows` to -0.0, and `vec_huge`'s MAD to -0.0. Both tests passed. With `same_bits`, both mutants fail:

```
cov zeros -> -0.0 killed: ('cov', ['3f947ae147ae1452', '3fa9999999999980', '8000000000000000', ...
mad +0.0 -> -0.0 killed: ('mad', ['8000000000000000'], ['0000000000000000'])
```

eigen stays on `assert_allclose`, since LAPACK builds differ in their kernels.

### Ruling on B2

Codex injected a nested module and a factory that builds an `njit(cache=True)` kernel at call time. Neither exists in the code. The package is flat, and every kernel is `@compiled` in `_ldjit`. What the probe exposes is still real. The no-cache test covers only modules whose names end in `jit`, and it relies on every kernel going through `compiled()`. Nothing enforced either. The follow-up plan adds three kernel modules, so the fix is a test that parses every source file. It fails on numba's decorators outside `_compiled.py` and on `compiled()` outside a `*jit` module. A second test feeds the checker code that breaks both rules. Discovery now walks subpackages too.

## Codex's sandbox failures

Codex's full-suite run ended `4 failed, 1090 passed, 1 skipped, 140 deselected, 1 error`. Codex traced four of the five itself to `No usable temporary directory found`, since the read-only sandbox has no temp directory: both `test_n_jobs_does_not_change_the_result` cases (joblib's temp folder), the workqueue subprocess test (pytest's capture file in the child) and the no-cache test's `tmp_path`. The fifth, `test_every_kernel_caches_when_numba_can`, failed because Codex emptied numba's cache locators to get its run through the sandbox, so no kernel had a cache. All five pass on the mini.

## Test runs

Each of the six commits, 1110370 through 99f2b7e, ran the full suite on the mini, and none failed. `ruff check` and `ruff format --check` pass. The fixtures come only from the laptop's R 4.6.1 (x86-64, 64-bit long double significand), as their manifests record.

## Unverified

- Codex found no R fixture for the prototype's zero-column or empty-center cases, and didn't classify their behavior as a parity defect.
- Suppose a process launches numba threads on GNU OpenMP, forks, and the child imports `_rstats` for the first time. The at-fork hook wasn't registered at the fork, so the child doesn't know to run serially. Neither the laptop nor the mini has GNU OpenMP, so this is untested here.
