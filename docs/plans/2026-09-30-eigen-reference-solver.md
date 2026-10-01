# Eigen reference solver: evidence and plan

Status: proposal. No code, download or install until RON2K picks an option and approves its steps. No tolerance is loosened in the meantime: the eigen rule (R1 to R5, floor 1e-10, C 256, K 1024, tie 1e-12) keeps holding the scipy path to R.

## The question

Row 15 of the section 5 record in `2026-09-30-r-compat.md`: rotation signs, D1 (repeated top eigenvalue in refine) and WP-1, WP-2 (rank-deficient `wpca`). The audit traced all of them to the LAPACK build. RON2K asked whether a reference solver resolves the repeated-eigenvalue refine cases and the rank-deficient PCA cases in the pinned environment, and what building, distributing and running it would cost, without assuming a packaged replacement behaves like R's own.

## What was measured

A throwaway spike (kept outside the repo) called R 4.6.1's own `Rlapack.dll` from Python through ctypes, with the exact arguments R's `La_rs` passes: jobz 'N' for values only and 'V' otherwise, range 'A', uplo 'L', abstol 0, a workspace query first. It swapped that in for scipy inside the port's `r_eigen_sym` and reran the evidence. Environment: the pinned laptop (R 4.6.1 ucrt, LAPACK 3.12.1, Rlapack.dll md5 49aac4488ac7828f6ebaf100ab3cfb63, Rblas.dll md5 5fe8258e86d8a1a52a6c3db05faa0225), Python 3.14.3, numpy 2.5.3, scipy 1.18.1.

| Evidence | R's own Rlapack | scipy (shipped) |
|---|---|---|
| D1, 17 direct inputs: values, first vector, all vectors, projection, kmeans labels | 17 of 17 on every field | 8 of 17 |
| D1, 5 refine inputs, R's draws replayed | 5 of 5 exact | 1 of 5 |
| D1, the same 5 with `RDraws(291102)` | 5 of 5 exact | 2 of 5 |
| WP-2, 420 rank-deficient `wpca` inputs: sdev bits with NaN positions, R's warning, rotation bits | 420 of 420 | 9 of 420 |
| refine, 100 fixture cases replayed | 100 of 100 exact | 63 of 100 |
| refine, EV1 and EV2 for 3285 nuggets | 3285 of 3285 | 2777 and 2018 |
| 600 column-permuted spectra | 600 of 600 | 8 of 600 |
| `wpca` rotation and scores, 43 cases | 43 of 43 | 16 of 43 |

In every case the port's eigen input already had R's bits, so the eigen call was the only source of difference. With R's solver, the only `wpca` differences left are B3 (scale with corr=False) and the held one-column corr=True decision, neither of which involves eigen.

Two findings shape any design:
- jobz matters. Rlapack's own 'N' and 'V' modes disagree on 49 of the 420 inputs, so the backend must use 'N' exactly where R does (`only.values = TRUE`: refine's EV1, EV2 and shape ratio) and 'V' elsewhere. `r_eigen_sym` already maps `vectors` to jobz this way.
- The call tree matters. From `dsyevr_`, Rlapack.dll reaches 62 functions (20 of them compiler clones), 15 Rblas routines and R.dll's `xerbla_`. R compiles LAPACK as one file, so helpers are inlined or cloned. A replacement has to compile the same sources with the same flags, and copying a function list won't reproduce it.

## What it doesn't show

- It proves R's DLL reproduces R. It doesn't prove a DLL built by us does. That needs the build and a bit test, below.
- Nothing ran on Linux, macOS or arm64, on another R version, or on an R that uses an external LAPACK (common on Linux distributions, and Accelerate on macOS). On arm64 GCC contracts multiply-adds into FMA by default, which would likely change bits. That's reasoning, not a measurement.
- A gfortran other than R's 14.3.0 wasn't tried.
- The refine runs were serial. With `n_jobs` above 1, `refine.py:133` runs `_split_nugget` in loky worker processes, which call `r_eigen_sym` at `refine.py:209`. A backend chosen by patching the parent process wouldn't reach them, so the choice has to be made at import time.

## Options

### A. Ship our own build of R's LAPACK (Windows x64 first)

| Step | What it proves |
|---|---|
| 1. Download R 4.6.1's source (40,926,649 B) and Rtools45 (483,090,888 B), install Rtools45. Optionally LAPACK 3.12.1 from netlib (8,067,105 B) to diff against | nothing yet. Needs RON2K's approval |
| 2. Take `src/modules/lapack` and `src/extra/blas` and R's own `Makefile.win` / `MkRules` flags (R's Makeconf shows gcc 14.3.0, `-O2 -mfpmath=sse -msse2 -mstackrealign`) | the flags are R's, and any R-local source edits show up in the diff |
| 3. Build one LAPACK DLL and one BLAS DLL as R splits them, static libgfortran, our own `xerbla_` that returns an error instead of calling R's `error()` | a build exists |
| 4. Bit-test the build against R's Rlapack.dll on every spike input, both jobz modes. Only bit equality counts | R-equivalence on this platform, or the list of inputs where it fails |
| 5. Pick the backend at import time (a config value, overridable by an environment variable), record which one ran in every result's metadata, and keep scipy as the fallback | loky workers use the same backend as the parent |
| 6. Build platform wheels instead of the pure-Python wheel | the package installs |
| 7. Licensing read: LAPACK's modified BSD (bundled in R), R's GPL-2+, static libgfortran under the GCC Runtime Library Exception, libquadmath under LGPL-2.1, against the port's GPL-2.0-only | the wheels can ship |

Each other platform repeats steps 3, 4 and 6 and gets its own parity statement, such as "R 4.6.1, internal LAPACK, x86_64 Windows".

### B. Use the user's installed R at run time

Find R_HOME by asking R (`R RHOME`, `La_library()`), load its `Rlapack`, fall back to scipy, and record which one ran.

- For: it's exactly what the spike measured, and nothing is redistributed.
- Against: it loads R.dll (40 MB) into Python. An invalid LAPACK argument would reach R's `xerbla_` with R uninitialized, which most likely kills the process, and that path was never exercised. An R whose LAPACK is external has no `libRlapack` at all. The user's R version can drift from the pinned fixtures. The loky issue from step 5 applies too.

### Performance

One `dsyevr` call with vectors, pinned to one P-core, median microseconds:

| p | R's Rlapack, bare call | scipy, bare call | ratio |
|---|---|---|---|
| 2 | 3.5 | 1.9 | 1.86 |
| 4 | 5.4 | 4.4 | 1.23 |
| 10 | 10.3 | 10.9 | 0.95 |
| 30 | 53.7 | 63.2 | 0.85 |
| 61 | 273 | 231 | 1.18 |
| 200 | 6329 | 2957 | 2.14 |

Up to 61 columns the cost is close to scipy's. At 200 it's about twice, because R's BLAS is the reference BLAS. The spike's ctypes wrapper added about 30 microseconds a call, which a production wrapper with cached workspaces would mostly remove. That overhead wasn't measured end to end.

## Recommendation

A, Windows x64 first, because it's the only option that can be tested against R before it ships and doesn't depend on what R the user has. B is worth keeping as a development aid for recording evidence, not as the shipped path. Whichever is chosen, the scipy path stays the default until step 4 passes, and the eigen rule stays as it is.

## Needs RON2K's approval

1. The downloads and the Rtools45 install listed in step 1.
2. Moving from a pure-Python wheel to platform wheels.
3. The licensing read in step 7.
4. A or B, or B first for evidence and then A.
