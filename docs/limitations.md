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
| A changing API | Record the package version and commit used for a study. Planned compatibility changes can alter arguments, shapes, labels, and random defaults. |

## CLI safeguards and boundaries

1. **Each run needs a new output folder.** The default creates `outputs/flow-<timestamp>/` relative to the current directory. An existing `--out` path is refused, even if empty. Failed or cancelled runs keep partial results with an `INCOMPLETE.txt` marker; inspect them only as incomplete work and retry in a new folder.
2. **At least two distinct fluorescence channels are required.** The CLI checks this before asking for populations. Its PCA figure needs two components; a one-dimensional plotting workflow is not supported.
3. **Clustering needs at least three occupied pooled nuggets.** Very small retained samples or `--dn-num2 2` on a single file can fall short. The CLI reports this before calling weighted Ward clustering. Keep more cells, or raise `--dn-num2`, which must stay below `--dn-num1`.

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
- `datanuggets-flow` reads FCS files through flowio, from the `flow` extra. It clusters the events as stored, uncompensated, and warns when a file's spillover matrix isn't the identity. The functions `datanuggets.flow` exports take arrays and don't read FCS files. `datanuggets.flow.pipeline.read_fcs` reads one for the command, and the quickstart example and the flow benchmark read them with flowio.
- The default channels are every channel of the first file whose name ends in `-A` and doesn't start with FSC or SSC. A viability dye recorded on such a channel is clustered with the markers unless `--exclude-channels` leaves it out or `--channels` lists the channels to use. Whether the default should leave such channels out is undecided.
- A gate folder records each file's path, size, SHA-256 and channels, not its spillover matrix. `--from-gates` reads each file again and repeats the spillover warning, but neither stage compensates, and the folder doesn't show whether a file needed it. Its masks fit only the bytes they were found in, so a file that moves or changes after gating has to be gated again.
- The flow manuscript's differential projection pursuit analysis isn't implemented. `datanuggets-flow` clusters the cells of the gates you pick, with data nuggets per file and weighted Ward clustering of all the files' nuggets together, cut at `--k` clusters. It doesn't run projection pursuit, a varimax rotation, SVM regions or weighted k-means with an elbow rule. [methodology.md](methodology.md) describes the pipeline as implemented, and [cli.md](cli.md) its options.

### Packaging

- It needs Python 3.12 or newer with compatible NumPy, SciPy, and Numba installations. A platform-independent package wheel does not guarantee that its compiled dependencies support every platform. [installation.md](installation.md) covers the alpha, GitHub, and local-wheel install routes.

## Open problems

These are known and unsolved. [compatibility.md](compatibility.md) describes the current user-visible differences from R.

1. Nine breaking changes toward R's behavior are approved in principle and held until they're confirmed as a set. B1 makes R's draws the default for an integer seed. B2 returns `wmean` as `p` by `k`. B3 makes `wpca`'s scale a single 1.0. B4 and B5 adopt R's argument order for `cluster_predict` and `create_dn_centers`. B6 makes `progress` take bools only. B7 gives `to_frame()` R's 1-based ids and types. B8 makes `cluster_predict` return R's empty results instead of raising. B9 orders categorical labels by category.
2. Eight compatibility tasks are pending. T4 accepts fractional and bool labels under `factor()`'s 15-digit rule and adds `cluster_levels`. Two recorded cases, `labels_faithful_float_merge` and `labels_faithful_big`, show a gap in that design. In both, `distw` builds three axes from the exact values where `factor()` makes two levels, so one list of levels can't describe both. T6 floors `dn_num1`, `dn_num2` and `dn_num` the way R does. T7 makes `refine_dn` accept any real `nstart` and return R's result, which leaves the nuggets unsplit when `kmeans` can't run. Today `refine_dn` raises `ValueError` on a NaN or infinite `nstart`, and the internal `kmeans` raises an `OverflowError` on an infinite one. T8 accepts R's other `corr` values. T9 exports `RDraws` and `cluster_levels`. T10 adds independent checks for the one-column `"random"` case of `create_dn`, where R's own table is malformed. T11 adds tests for the documented exceptions. T12 corrects the remaining docs and writes `docs/compatibility.md`. [compatibility.md](compatibility.md) now holds the differences from R, moved from the README, but the record's check for T12, reading each claim against the fixtures, hasn't been run on it.
3. One-column `wpca` with `corr=True` waits on a decision. R's result there is a side effect of `diag()`, and both independent checks give the port's 1. Keeping the port's answer is the recommendation.
4. R's behavior with NaN, NA and infinite values isn't recorded for `create_dn`, `refine_dn`, `create_dn_centers`, `arithmetic_complexity` or `get_dn_means`, so they keep raising until it is. The `WCluster` statistics need NA and NaN recorded separately before they can pass them through with R's payloads.
5. A reference eigen solver is not implemented. In the reference-solver experiment, R's own `Rlapack` reproduced R on all 420 rank-deficient `wpca` inputs, where scipy matched 9, on 100 of 100 refine cases against scipy's 63, and on all 43 `wpca` rotation and score cases against 16. The recommended option builds R's LAPACK sources with R's flags for Windows x64 first. It needs the R source and Rtools downloads, platform wheels in place of the pure-Python one, and a license review. Nothing was measured on Linux, macOS arm64 or another R version.
6. Five implementation choices from the speed work still need review. The stopping rule itself passes: no Python function takes more than 10% of wall time on Covertype, at 100,000 rows (5.8% for `create_dn`, 3.6% for `refine_dn`) or at full size (7.0% and 1.1%). Each deletion round is compiled whole, marking loop included, because the profile flagged the function and not the loop. The surviving rows are kept as a list of row numbers rather than a mask, and a row's nearest neighbour is searched again only when it was deleted. The row minimum keeps four running minima instead of being vectorized, which gives the same first index. Stage 1 runs on threads and the kernels release the GIL, so with `n_jobs` above 1 its threads now run them at once. `DIST_BLOCK_ROWS` and `ASSIGN_BLOCK_ROWS` are gone, since the kernels make no numpy temporaries to bound.
7. `optimal.DN` isn't ported. The iterative clustering functions and DN wrappers also have compatibility boundaries: `wkmeans` gives up after 10,000 starts without `k` clusters, where R keeps drawing. One-variable support differs by wrapper: `dn_wkmeans` rejects it; `dn_wpca` requires an explicit `wcol=[1]`; `dn_whclust` accepts it. See the function reference for the supported shapes. `dn_cluster_predict` reads a `DataNuggets` given as `newx` by `dn`'s width, as R does. `whclust` keeps R's `n` by `n` matrix without chunking it. `dn_cluster_predict` takes a matrix `newx`, where R stops, so that it matches `cluster_predict`.
8. Some behavior isn't covered by a recorded case or a test:
   - R hasn't recorded zero-column inputs or centers with no rows. The compiled distance functions raise `ValueError` on both, and no public call reaches them.
   - A process that launches numba threads under GNU OpenMP and then forks, before it has ever imported the moments module, leaves the child without the hook that makes it run serially. Neither test machine has GNU OpenMP.
   - Some exact checks use `same_values`, which lets any NaN match any NaN, so a wrong NaN payload there wouldn't fail.
   - The `create_dn`, `create_dn_centers`, `refine_dn`, `kmeans` and Rfast tests compare with `assert_array_equal`, which can't tell -0.0 from 0.0. A wrong sign of zero there would pass the suite. For `create_dn` and `refine_dn` it would show when the parity tables are rebuilt.
   - Three mutation-test cases remain undetected. One in `r_weighted_mean` survives on the laptop, where its NaN and R's have the same bits, and it wasn't run on arm64, where it should fail. One moves `wpca`'s warning threshold below -1e-300, where no recorded input has an eigenvalue. One drops the orthonormality check from the eigen rule, which another check covers, though that isn't proven for tied eigenvalues.
   - `RDraws` isn't checked against R for a sample of more than 2^31 - 1 values from the hash sampler, which only a population of more than four billion would reach.
9. Online documentation links require GitHub access while the repository is private. The README now uses absolute links suitable for PyPI, and the source archive includes the user guides and runnable example. Make the repository public or provide publicly accessible documentation before a public package release.
10. `find_area`'s disc waits on a ruling. The port centres it on the peak, which changes which events each population holds: on `202.fcs` it puts 107,909 of the 282,270 events in a different set of populations, as [methodology.md](methodology.md) describes. Matching the reference finder's output instead means the mirrored disc, which `tests/flowcases.py` states and `compare.py`'s `finder his` row runs. `datanuggets-flow` gates with the mirrored disc by default, `--disc his`, and `--disc peak` uses the port's `find_area`.
