# Limitations and open problems

This package is a research alpha. Its numerical methods have substantial test coverage, but the API is still changing and the flow workflow needs validation for each study. The differences from R kept on purpose are in [compatibility.md](compatibility.md).

**Publication hold:** the GNU sorting helpers need a compatible source-provenance record or replacement before this candidate can be redistributed publicly. See [third-party notices](../THIRD_PARTY.md#publication-hold-gnu-sorting-helpers). Both TestPyPI and PyPI uploads are blocked in the release workflow; check-only runs remain available.

## What this means for your analysis

| Limitation | Practical consequence |
| --- | --- |
| Quadratic memory use | Reduce the intermediate and pooled nugget counts before running a large study. The hierarchy alone needs about 800 MB at 10,000 nuggets. |
| A reduced representation | Compare settings and check whether important structure survives. Weighted centers do not retain every detail of the original observations. |
| Different random streams and eigen calculations | Matching R seed numbers is insufficient. Some degenerate inputs can change refinements or PCA results. |
| Experimental flow defaults | Inspect gates, choose channels and transformations for your panel, and validate biological interpretations. No automatic compensation is applied. |
| A changing API | Record the package version and commit used for a study. [Planned compatibility changes](#planned-changes) can alter arguments, shapes, labels, and random defaults. |

## CLI safeguards and boundaries

1. **Each run needs a new output folder.** The default creates `outputs/flow-<timestamp>/` relative to the current directory. An existing `--out` path is refused, even if empty. Failed or cancelled runs keep partial results with an `INCOMPLETE.txt` marker; inspect them only as incomplete work and retry in a new folder.
2. **At least two distinct fluorescence channels are required.** The CLI checks this before asking for populations. Its PCA figure needs two components; a one-dimensional plotting workflow is not supported.
3. **Clustering needs at least three occupied pooled nuggets.** Very small retained samples or `--dn-num2 2` on a single file can fall short. The CLI reports this before calling weighted Ward clustering.

The [CLI guide](cli.md) explains the output policy and errors. Finishing a run does not validate its biological interpretation.

## Limitations

### Parity with R

- The recorded R reference data is included, but its R generators and comparison scripts are not. A fresh clone can verify the recorded cases but cannot regenerate R baselines on its own.

- Parity means one reference environment: R 4.6.1 on x86-64 with the reference BLAS, R's internal LAPACK and the x87 long double. Results from an R built another way can differ in the last bits, and on tied data that can move rows between nuggets. [reproducibility.md](reproducibility.md) records that environment.
- The eigen output in `wpca` and `refine_dn` comes from scipy's LAPACK, so it's held to R's by a rule or a tolerance instead of matched bit for bit. In `refine_dn` the first eigenvalue decides which nuggets every split round tries, with or without shape splits. The eigen spike found two kinds of input where the difference goes past rounding. With a repeated top eigenvalue, `refine_dn`'s nuggets can differ from R's, and scipy matched R on 1 of 5 such refine inputs. On rank-deficient `wpca` inputs, the NaN positions in `sdev` and R's warning can differ. scipy reproduced R's `sdev` bits, NaN positions, warning and rotation bits together on 9 of 420. A rotation column can also come back with the opposite sign from R's.
- An integer seed or a NumPy `Generator` draws from NumPy, so the nuggets differ from R's for the same seed. Only [`RDraws`](compatibility.md#rdraws) gives R's draws, and it isn't exported from `datanuggets` yet.
- NaN and infinite values in the data or the weights raise everywhere. R's behavior with them is recorded only for `cluster.predict` so far, so the rest keep raising until it is.
- Labels must be whole numbers smaller than 1e15. Fractional, bool and text labels raise.

### Memory and speed

- `whclust` keeps an `n` by `n` float64 matrix of merge costs, as R's `Whclust` does: 32 MB at 2,000 rows and 800 MB at 10,000.
- When weights of both signs make R merge a cluster with itself, `whclust` reruns the whole call through its plain-Python reference, which follows R's matrix steps one by one and is far slower than the compiled path.
- `wkmeans` gives up after 10,000 starts that miss `k` clusters (`MAX_STARTS` in `datanuggets.wcluster.wkmeans`), where R keeps drawing. On an input where R would return after more starts than that, the port raises instead. No such input is known.
- `create_dn` builds a square distance matrix over the centers actually pooled for its second stage. At 10,000 centers, its default intermediate target, that matrix is 800 MB of float64; small inputs can produce fewer centers. Chunk allocation and other arrays also affect memory, so this is not a total-memory bound. The recorded Covertype run peaked near 0.9 GB.
- The flow command pools nuggets from every retained file. At its default of 1,000 per file, 44 files can produce 44,000 nuggets: about 15.5 GB for the hierarchy's matrix alone. Per-file reduction does not make the folder-scale hierarchy bounded in memory.
- `refine_dn` without shape splits is slower than R on the full Covertype data, 1.5 seconds to 0.9. Two thirds of its time there goes to the long double means and covariances. R gets those from the x87 unit's hardware long double, and the port repeats them in 64-bit integer code. The benchmark is in [reproducibility.md](reproducibility.md).
- numba compiles the kernels on first use and caches them. On the development laptop (i7-13650HX, Windows 11), importing the package and making a first small call to each public function took 8.8 seconds with an empty cache and 0.9 seconds with a warm one on Python 3.14, and 8.3 and 0.9 seconds on 3.15. The cache is kept per Python version, so each new version compiles once. Where numba can't write its cache, every new process compiles again.

### Tests

- The plot tests check drawn coordinates and colors, not images. A change in how matplotlib renders them, such as its fonts or its antialiasing, wouldn't fail them.
- Windows has no `os.fork`, so the two tests of forked children run only on macOS and Linux.
- The FCS parser and complete CLI are tested with an FCS 3.1 file reconstructed from measured fixture data. This does not cover every instrument's original file encoding or metadata.

### Flow cytometry

- The flow finder's cutoffs are Prof. Cabrera's: a peak's row above 10 and below 90 and its column below 90, with no lower limit on the column, the 0.85 and 0.9 quantiles, and `t11` to `t22`. They haven't been tuned for other panels or instruments. On the 44 PBMC files of the Zenodo record that [reproducibility.md](reproducibility.md) lists, Prof. Cabrera's finder found 2 to 7 populations per file.
- `datanuggets-flow` reads FCS files through flowio, from the `flow` extra. It clusters the events as stored, uncompensated, and warns when a file's spillover matrix isn't the identity. The functions `datanuggets.flow` exports take arrays and don't read FCS files. The quickstart example and the flow benchmark read them with flowio.
- The flow manuscript's differential projection pursuit analysis isn't implemented. `datanuggets-flow` clusters the cells of the gates you pick, with data nuggets per file and weighted Ward clustering of all the files' nuggets together, cut at `--k` clusters. It doesn't run projection pursuit, a varimax rotation, SVM regions or weighted k-means with an elbow rule. [methodology.md](methodology.md) describes the pipeline as implemented, and [cli.md](cli.md) its options.

### Packaging

- It needs Python 3.12 or newer with compatible NumPy, SciPy, and Numba installations. A platform-independent package wheel does not guarantee that its compiled dependencies support every platform. [installation.md](installation.md) covers the alpha, GitHub, and local-wheel install routes.

## Planned changes

These changes toward R's behavior are planned and not made yet. Most of them change what an existing call accepts or returns.

- An integer `seed`, the default included, will draw what R draws with that seed, as `RDraws(seed)` does now, so its results will change.
- `RDraws` will be exported from `datanuggets`, so it won't need to be imported from its module.
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

1. One-column `wpca` with `corr=True` waits on a decision. R's result there is a side effect of `diag()`, and both independent checks give the port's 1. Keeping the port's answer is the recommendation.
2. R's behavior with NaN, NA and infinite values isn't recorded for `create_dn`, `refine_dn`, `create_dn_centers`, `arithmetic_complexity` or `get_dn_means`, so they keep raising until it is. The `WCluster` statistics need NA and NaN recorded separately before they can pass them through with R's payloads.
3. A reference eigen solver is not implemented. In the reference-solver experiment, R's own `Rlapack` reproduced R on all 420 rank-deficient `wpca` inputs, where scipy matched 9, on 100 of 100 refine cases against scipy's 63, and on all 43 `wpca` rotation and score cases against 16. The recommended option builds R's LAPACK sources with R's flags for Windows x64 first. It needs the R source and Rtools downloads, platform wheels in place of the pure-Python one, and a license review. Nothing was measured on Linux, macOS arm64 or another R version.
4. `optimal.DN` isn't ported. The iterative clustering functions and DN wrappers also have compatibility boundaries: `wkmeans` gives up after 10,000 starts without `k` clusters, where R keeps drawing. One-variable support differs by wrapper: `dn_wkmeans` rejects it; `dn_wpca` requires an explicit `wcol=[1]`; `dn_whclust` accepts it. See the function reference for the supported shapes. `dn_cluster_predict` reads a `DataNuggets` given as `newx` by `dn`'s width, as R does. `whclust` keeps R's `n` by `n` matrix without chunking it. `dn_cluster_predict` takes a matrix `newx`, where R stops, so that it matches `cluster_predict`.
5. A process that launches numba threads under GNU OpenMP and then forks, before it has ever imported the moments module, leaves the child without the hook that makes it run serially. No test covers this, since neither test machine has GNU OpenMP.
6. `find_area`'s disc waits on a ruling. The port centres it on the peak, which changes which events each population holds: on `202.fcs` it puts 107,909 of the 282,270 events in a different set of populations, as [methodology.md](methodology.md) describes. Matching the reference finder's output instead means the mirrored disc, which `tests/flowcases.py` states and `compare.py`'s `finder his` row runs. `datanuggets-flow` gates with the mirrored disc by default, `--disc his`, and `--disc peak` uses the port's `find_area`.
