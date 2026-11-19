# Limitations and open problems

This package is a research alpha. Its numerical methods are checked against 544 recorded R cases ([methodology](methodology.md#against-r)), but the API is still changing. The differences from R kept on purpose are in [compatibility.md](compatibility.md).

## What this means for your analysis

| Limitation | Practical consequence |
| --- | --- |
| Quadratic memory use | Reduce the intermediate and final nugget counts before running a large analysis. The hierarchy alone needs about 800 MB at 10,000 nuggets. |
| A reduced representation | Compare settings and check whether important structure survives. Weighted centers do not retain every detail of the original observations. |
| Different random streams and eigen calculations | Use [`RDraws`](compatibility.md#rdraws) for R's draws; seed numbers alone don't match R. Some degenerate inputs can change refinements or PCA results. |
| A changing API | Record the package version and commit used for an analysis. [Planned compatibility changes](#planned-changes) can alter arguments, shapes, labels, and random defaults. |

## Limitations

### Parity with R

- The recorded R reference data is included, but its R generators and comparison scripts are not. A fresh clone can verify the recorded cases but cannot regenerate R baselines on its own.

- Parity means one reference environment: R 4.6.1 on x86-64 with the reference BLAS, R's internal LAPACK and the x87 long double. Results from an R built another way can differ in the last bits, and on tied data that can move rows between nuggets. [reproducibility.md](reproducibility.md) records that environment.
- On an exact distance tie, `create_dn` assigns a row to the lowest-numbered of the tied centers. R's Rfast picks among them with an unstable sort, which gives the same center whenever there are 16 or fewer centers. With more, R can take another of the tied centers, so on such data the nuggets can differ from R's. [compatibility.md](compatibility.md#exact-distance-ties-in-create_dn) has the measured detail.
- The eigen output in `wpca` and `refine_dn` comes from scipy's LAPACK, so it's held to R's by a rule or a tolerance instead of matched bit for bit. In `refine_dn` the first eigenvalue decides which nuggets every split round tries, with or without shape splits. The eigen spike found two kinds of input where the difference goes past rounding. With a repeated top eigenvalue, `refine_dn`'s nuggets can differ from R's, and scipy matched R on 1 of 5 such refine inputs. On rank-deficient `wpca` inputs, the NaN positions in `sdev` and R's warning can differ. scipy reproduced R's `sdev` bits, NaN positions, warning and rotation bits together on 9 of 420. A rotation column can also come back with the opposite sign from R's.
- An integer seed or a NumPy `Generator` draws from NumPy, so the nuggets differ from R's for the same seed. Only [`RDraws`](compatibility.md#rdraws) gives R's draws: import it with `from datanuggets import RDraws` and pass `seed=RDraws(s)` to `create_dn`, `refine_dn`, `wkmeans` or `dn_wkmeans`. It reproduces R's `set.seed(s)` and `sample.int` under R's default generator and restarts from `s` on every call, as R's functions call `set.seed` each time. Results can still differ from R's on exact distance ties among more than 16 centers and where LAPACK eigenvalues decide, as the two items above describe.
- NaN and infinite values in the data or the weights raise in every port of a `datanugget` or `WCluster` function that reads them. `arithmetic_complexity` reads only the shape of `x`, as R's `AC` does, so it doesn't check them. R's behavior with them is recorded only for `cluster.predict` so far. `cluster_predict` raises on them too, and the other ports keep raising until R's behavior is recorded for them.
- Labels must be whole numbers smaller than 1e15 in absolute value. Fractional, bool and text labels raise.

### Memory and speed

- `whclust` keeps an `n` by `n` float64 matrix of merge costs, as R's `Whclust` does: 32 MB at 2,000 rows and 800 MB at 10,000.
- When weights that cancel or sums of squares that overflow make R merge a cluster with itself, `whclust` reruns the whole call through its plain-Python reference. That reference follows R's matrix steps one by one and is far slower than the compiled path.
- `wkmeans` gives up on a start after 1,000 random partitions (`MAX_DRAWS` in `datanuggets.wcluster.wkmeans`) that each leave a label on fewer than two rows or a cluster with no row, where R keeps drawing. A start found within the cap is R's start. Where R needs more draws than that, the port raises `ValueError` instead, which can happen near `n = 2k`. On the 16 values 0 to 15 in one column at `k = 8`, the port's start under `RDraws(1)` takes 2,179 partitions when the cap is lifted. So `wkmeans` with `seed=RDraws(1)` raises there, where R's two redraw loops have no cap and keep drawing (`test_the_cap_allows_exactly_max_draws`).
- `create_dn` builds a square distance matrix over the centers actually pooled for its second stage. At 10,000 centers, its default intermediate target, that matrix is 800 MB of float64; small inputs can produce fewer centers. Chunk allocation and other arrays also affect memory, so this is not a total-memory bound. The recorded Covertype run peaked near 0.9 GB.
- `refine_dn` without shape splits is slower than R on the full Covertype data, 1.5 seconds to 0.9. Two thirds of its time there goes to the long double means and covariances. R gets those from the x87 unit's hardware long double, and the port repeats them in 64-bit integer code. The benchmark is in [reproducibility.md](reproducibility.md).
- numba compiles the kernels on first use and caches them. On the development laptop (i7-13650HX, Windows 11), importing the package and making a first small call to each public function took 8.8 seconds with an empty cache and 0.9 seconds with a warm one on Python 3.14, and 8.3 and 0.9 seconds on 3.15. The cache is kept per Python version, so each new version compiles once. Where numba can't write its cache, every new process compiles again.

### Tests

- The plot tests check drawn coordinates and colors, not images. A change in how matplotlib renders them, such as its fonts or its antialiasing, wouldn't fail them.
- Windows has no `os.fork`, so the two tests of forked children run only on macOS and Linux.

### Packaging

- It needs Python 3.12 or newer with compatible NumPy, SciPy, and Numba installations. A platform-independent package wheel does not guarantee that its compiled dependencies support every platform. Numba's wheels limit it to Windows x64, macOS 12 or newer on Apple silicon, Linux x86-64 or ARM64 with glibc 2.27 or newer, and Windows on ARM. Its Windows on ARM wheels cover Python 3.12 to 3.14 only. No tests run on Linux ARM64 or Windows on ARM ([supported platforms](installation.md#supported-platforms)). [installation.md](installation.md) covers installing from PyPI, from GitHub and from a local checkout.

## Planned changes

These changes toward R's behavior are planned and not made yet. They change what an existing call accepts or returns.

- An integer `seed`, the default included, will draw what R draws with that seed, as `RDraws(seed)` does now, so its results will change.
- `create_dn` and `create_dn_centers` will round a fractional `dn_num1`, `dn_num2` or `dn_num` down, as R does, where they now raise `ValueError`.
- `create_dn_centers` will take its arguments in R's order, with `delete_percent` before `dn_num`.
- `progress` in `create_dn` and `refine_dn` will accept only `True` or `False`.
- `DataNuggets.to_frame()` will give R's 1-based nugget ids and R's column types.
- `refine_dn` will accept any number as `nstart`, NaN and infinity included, and return R's result, which leaves the nuggets unsplit where `kmeans` can't run.
- Fractional and bool cluster labels will be accepted and grouped as R's `factor()` groups them, to 15 significant digits, and a new `cluster_levels`, exported from `datanuggets`, will give their levels.
- Categorical labels will be ordered by their categories, as R orders a factor by its levels.
- `wmean` will return R's layout, `p` rows by `k` columns, in place of `(k, p)`.
- With `corr=False`, `wpca`'s `scale` will be a single 1.0, as in R, in place of a 1.0 per column.
- `wpca` and `dn_wpca` will accept the other `corr` values R reads as true or false, such as `1`, `0` and `"FALSE"`, where they now raise `TypeError`.
- `cluster_predict` will take its arguments in R's order, `(x, w, cl, newx)`, with `w` before `cl`.
- `cluster_predict` will return R's empty result for a new row whose distances are all NaN, in place of raising `ValueError`.

## Open problems

These are known and unsolved. [compatibility.md](compatibility.md) describes the current user-visible differences from R.

1. One-column `wpca` with `corr=True` waits on a decision between the port's `sdev` and R's result. The port's is 1 to within rounding, and R's is a side effect of `diag()`. Both independent checks give the port's 1.
2. R's behavior with NaN, NA and infinite values isn't recorded for `create_dn`, `refine_dn`, `create_dn_centers` or `get_dn_means`, so they keep raising until it is. The `WCluster` statistics need NA and NaN recorded separately before they can pass them through with R's payloads.
3. A reference eigen solver is not implemented. In the reference-solver experiment, R's own `Rlapack` reproduced R on all 420 rank-deficient `wpca` inputs, where scipy matched 9, on 100 of 100 refine cases against scipy's 63, and on all 43 `wpca` rotation and score cases against 16. The recommended option builds R's LAPACK sources with R's flags for Windows x64 first. It needs the R source and Rtools downloads, platform wheels in place of the pure-Python one, and a license review. Nothing was measured on Linux, macOS arm64 or another R version.
4. `optimal.DN` isn't ported. One-variable support differs by wrapper: `dn_wkmeans` rejects it; `dn_wpca` requires an explicit `wcol=[1]`; `dn_whclust` accepts it. See the function reference for the supported shapes. `dn_cluster_predict` reads a `DataNuggets` given as `newx` by `dn`'s width, as R does. `whclust` keeps R's `n` by `n` matrix without chunking it. `dn_cluster_predict` takes a matrix `newx`, where R stops, so that it matches `cluster_predict`.
5. A process that launches numba threads under GNU OpenMP and then forks, before it has ever imported the moments module, leaves the child without the hook that makes it run serially. No test covers this, since neither test machine has GNU OpenMP.
