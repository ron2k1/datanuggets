# Methodology

This page describes the research problem, the data nuggets method and the weighted analyses as this package implements them. It then says how each part relates to the papers and how it is checked. The functions in `datanuggets.nuggets` are ports of `datanugget` 1.5.0. Those in `datanuggets.wcluster` are ports of `WCluster` 1.3.0, except `cut_tree`, which is a port of R's `cutree`. They are held to R's output. The modules `datanuggets.colors`, `datanuggets.filters` and `datanuggets.projection` are not ports, and [compatibility.md](compatibility.md#plots) says how the plots in `datanuggets.plot` relate to R's drawings. The DESCRIPTION file of `datanugget` 1.5.0 describes the method in these words: "Data nuggets reduce a large dataset into a small collection of nuggets of data, each containing a center (location), weight (importance), and scale (variability) parameter."

Usage is elsewhere: [quickstart.md](quickstart.md) has first examples, and [compatibility.md](compatibility.md) lists the differences from R. The [math reference](https://github.com/ron2k1/datanuggets/blob/main/docs/math/README.md) gives each function's computation as equations. It is a draft that the authors of the R packages have not reviewed.

## The problem

The data nuggets paper (Beavers et al., Section 2) puts the memory and time of typical clustering methods, such as k-means, hierarchical clustering and PAM, at order P·N(N−1)/2 for N rows and P columns. Hierarchical clustering needs the full distance matrix before it can start. The usual way around this is to cluster a random sample, but a sample can miss small groups and the sparse edges of the data.

Data nuggets reduce the N rows to M groups, each described by a center, a weight (its row count) and a scale. The analyses then run on the M weighted centers. The paper puts the cost of building nuggets mainly in the N·M assignment step, after which the clustering costs are of order M² (Section 3). The paper's claim is that the nuggets keep the structure of the data, including its edges, better than a random sample or support points of the same size (Sections 3 and 3.2). This package doesn't test that claim. It reproduces the R implementation of the method.

## Data nuggets

### Creating nuggets

`create_dn` is a port of `create.DN` from `datanugget` 1.5.0. It follows Algorithm 1 of the paper as the R package implements it:

1. The rows are shuffled once with the seed and split into `ceiling(n / r)` chunks of nearly equal size, as R's `parallel::splitIndices` splits them.
2. Each chunk gets its share of `dn_num1` first-stage centers: `dn_num1` divided by the number of chunks, the remainder going to the first chunks, clamped to between 1 and the chunk's size.
3. Each chunk is reduced to its share by closest-pair deletion. The chunk's full distance matrix is computed, Euclidean or Manhattan. Each round deletes `max(1, floor(t * delete_percent))` of the `t` remaining points, without going below the target. The points are ordered by the distance to their nearest neighbour, and a point is deleted when neither it nor its nearest neighbour has been deleted yet, so one point of each closest pair goes. If that falls short, more points are deleted in the same order. Rounds repeat until the target count remains. A chunk with no more rows than its target passes through whole, with a warning.
4. The first-stage centers of all chunks are pooled and reduced to `dn_num2` by the same deletion.
5. Every row is assigned to its nearest second-stage center, with the same metric. A row exactly as near to several centers goes to the lowest-numbered of them. With more than 16 centers R's choice among them can differ ([compatibility.md](compatibility.md#exact-distance-ties-in-create_dn)).
6. Each nugget's weight is its row count. Its scale is the mean over columns of the sample variances of its rows, which is the trace of its covariance divided by p. Its center depends on `center_method`: `"mean"` takes the mean of its rows, `"random"` one of its rows drawn with the seed, and `"original"` keeps the second-stage center.

Deletion removes points from dense regions first, since their nearest neighbours are closest. Points in the sparse tails survive as centers, often with weight 1. A nugget with fewer than 2 rows gets a floor, the smallest finite scale among the nuggets with 2 or more rows divided by 100. A nugget whose scale is not finite gets the same floor. When there is no such scale, or the floor is 0, the floor is machine epsilon, about 2.2e-16. R does the same. The paper sets that scale to 0. An empty nugget keeps its second-stage center and weight 0.

The second stage builds a square distance matrix over the pooled first-stage centers. There are usually `dn_num1` of them, which makes 800 MB of float64 at the default 10,000. Smaller inputs pool fewer, since no chunk gives more centers than it has rows. When `ceiling(n / r)` is above `dn_num1`, every chunk still gives one center, so the pool is the number of chunks. A pool of `dn_num2` or fewer centers passes through whole, with a warning and no matrix. `arithmetic_complexity`, the port of R's `AC`, reports the arithmetic cost of a `create_dn` call on a log10 scale from the data's shape alone. The paper's appendix derives the cost of the algorithm.

### Refining nuggets

`refine_dn` is a port of `refine.DN`. It reads only the assignments of its input and recomputes every nugget's center (the mean of its rows), weight, scale and two largest covariance eigenvalues from the rows of `x`.

1. The split threshold is the type-7 quantile, at `ev_tol`, of the nuggets' positive first eigenvalues. It is computed once, before the first round.
2. For up to `max_splits` rounds, every nugget whose first eigenvalue is above the threshold and that has at least `2 * min_nugget_size` rows is split in two by k-means with k = 2, using a statement-by-statement port of the Hartigan-Wong Fortran behind R's `stats::kmeans`, with `nstart` starts and 1,000 iterations at most. A split is kept only if both children have at least `min_nugget_size` rows. The rounds stop early when no nugget qualifies or none splits.
3. With `shape_split=True` and two or more columns, up to `max_splits` more rounds split nuggets by shape. A candidate has at least `max(min_shape_size, 2 * min_nugget_size)` rows and is elongated, with its first eigenvalue over its second above `delta`, or spread out, with its first eigenvalue above the median plus twice the MAD of the scales of the nuggets with 2 or more rows. Its rows are projected on its first eigenvector and the projection is split by 2-means. The split is kept only if both children are large enough and the mean of their eigenvalue ratios is below `delta`.
4. Empty nuggets are dropped, the rest are renumbered from 0, and a nugget with one row gets scale 0.

R seeds again before every split, so each split draws from a fresh copy of the seed's stream. The port does the same with an int seed or an `RDraws`. With a NumPy `Generator` each split draws from a child stream spawned from it, in candidate order. Either way the result doesn't depend on `n_jobs`. The eigenvalues that decide the splits come from LAPACK, which is the one place where `refine_dn` can't be held to R bit for bit (see [Evaluation](#evaluation)).

### Defaults, from the code and the paper

The defaults are `datanugget` 1.5.0's, kept in `src/datanuggets/_defaults.py`. Where the paper and the R package differ, the port follows the package.

| Paper | Python | Default in the code | In the paper |
| --- | --- | --- | --- |
| R, the size of each random subset | `r` | 5000, allowed from 100 to 10,000 | 5,000 is the default (Section 3) |
| C, the share deleted per round | `delete_percent` | 0.1 | The text calls 1% the default. Figure 2 and the examples of Sections 3.2 and 3.3 use 0.1, and Section 4 uses 0.01 |
| Minit, the first-stage count | `dn_num1` | 10,000 | 10,000 on a 16 GB laptop |
| M, the number of nuggets | `dn_num2` | 2000 | 2,000 as a starting point |
| D, the distance metric | `dist_metric` | `"euclidean"`, or `"manhattan"` | Euclidean in every example |
| center choice | `center_method` | `"mean"`, `"random"` or `"original"` | mean or random |
| α, the split percentile | `ev_tol` | 0.9 | 0.5 for Figure 3, 0.25 in Section 4 |
| Nmin, the smallest nugget | `min_nugget_size` | 2 | 2 |
| a limit on rounds | `max_splits` | 5 | a limit "can be placed" |
| none | `shape_split`, `min_shape_size`, `delta`, `nstart` | `False`, 10, 2, 25 | Algorithm 2 states only the eigenvalue split |
| none | `seed` | 291102 | none |

The paper describes C as an artifact of slow loops in R and expects a later version to delete one point per iteration. `datanugget` 1.5.0 still deletes in rounds, and so does the port. An integer seed draws from NumPy, so it doesn't give R's nuggets. `seed=RDraws(291102)` draws what R draws with `set.seed(291102)` ([compatibility.md](compatibility.md#rdraws)). `datanugget`'s `optimal.DN` isn't ported ([compatibility.md](compatibility.md#not-ported)).

## Weighted analyses of the nuggets

The `WCluster` functions treat each row as a point with a weight. On nuggets, the rows are the centers and the weights are the row counts. The scales aren't used. The paper makes the same choice and justifies it with Theorem 1, which gives conditions under which the weighted covariance of the nugget centers approaches the sample covariance of the data as N grows (Section 3).

- `wpca` is `WCluster`'s `Wpca`: principal components with a weight per row and per column. The center is the weighted mean of the rows. The column weights are scaled to sum to p. The covariance of the centered data, scaled by the square roots of both weights, divides by the sum of the row weights minus 1, so the row weights act as counts. `corr=True` scales it to a correlation matrix. `sdev` holds the square roots of its eigenvalues and the scores are the centered data times the rotation. The help pages of `Wpca` and `DN.Wpca` cite Amaratunga and Cabrera (2009).
- `wkmeans` is weighted k-means as `WCluster`'s `Wkmeans` computes it. Each start draws random labels with every label on at least two rows, takes their weighted means as centers and gives each row its nearest center. It then runs up to `max_iter` rounds of 20 Lloyd steps, undoing a step that raises the weighted within-cluster sum of squares (WWCSS), followed by single-point moves that don't raise it. The start with the smallest WWCSS wins. The paper's Algorithm 3 starts from K nugget centers and uses single-point moves only. The port follows `WCluster`, quirks included, and [compatibility.md](compatibility.md#wkmeans-whclust-and-the-dn_-functions) lists where its arguments and stops differ from R's.
- `whclust` is `WCluster`'s `Whclust`: weighted hierarchical clustering by Ward's method (Ward, 1963). Every row starts as its own cluster, and each step merges the pair whose merge adds the least WWCSS, taking R's first pick on a tie. A merge's height is the total WWCSS after it, so the last height is the WWCSS of all rows as one cluster.
- `cut_tree` is R's `cutree` for one `k` or one height `h`. It numbers the clusters in the order they first appear along the rows. R's help page for `cutree` cites Becker, Chambers and Wilks (1988).
- `cluster_predict` is `cluster.predict`, with `cl` before `newx` and `w` keyword-only where R takes `(x, w, cl, newx)`. It sends each new row to the nearest weighted cluster mean, by squared Euclidean distance, and returns its 0-based position in sorted label order.
- `dn_wkmeans`, `dn_whclust`, `dn_wpca` and `dn_cluster_predict` are `WCluster`'s wrappers that run these on a `DataNuggets`. `dn_wkmeans` also gives each original row the cluster of its nugget.

## Relation to the papers

Beavers et al. (2025) define data nuggets (Algorithms 1 and 2, Section 3) and weighted k-means for nuggets (Algorithm 3, Section 3.1).

The package has no rule for choosing the number of clusters. For a tree over `n` rows, `whclust`'s heights give the WWCSS of every cut from 1 to `n - 1` clusters without reclustering: the height of merge `n - k`, counted from 1, equals `wwcss(x, cut_tree(tree, k=k), w)`, which is `tree.height[n - k - 1]` in Python. These are the WWCSS of Ward's nested cuts, not of a k-means fit at each k.

Cherasia, Cabrera, Fernholz and Fernholz (2023) use data nuggets for regression. They build the nuggets from the features alone, without the response. Of the three estimators they compare, weighted least squares on the nugget centers and mean responses, with the row counts as weights, performs best. In Section 6 they propose, without an example, keeping each nugget's count of each class for a binary or categorical response, and for a binary one splitting nuggets until none mixes outcomes. Their standard errors assume i.i.d. errors.

## Evaluation

### Against R

The port is checked against R's own output. The reference fixtures were recorded using R 4.6.1 with `datanugget` 1.5.0, `WCluster` 1.3.0 and Rfast 2.1.5.2 on x86-64 Windows, with R's reference BLAS and internal LAPACK 3.12.1. The generators traced the functions without modifying them and recorded the inputs, the outputs, every random draw and the internal values that matter: 544 cases in 23 groups. A few more values were computed in an R session and written into the tests as literals. Running the tests doesn't need R. The recorded data is included; the R generators are local maintainer tools and are not distributed. [reproducibility.md](reproducibility.md#what-r-recorded) lists the groups and what each holds, [development.md](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) says how to run the tests, and [architecture/package.md](https://github.com/ron2k1/datanuggets/blob/main/docs/architecture/package.md#tests) shows where the tests and fixtures live. [compatibility.md](compatibility.md) describes the known differences from R.

| Function | What R recorded | What is compared | Table |
| --- | --- | --- | --- |
| `create_dn` | 37 cases: iris, faithful and quakes under both metrics and all three center methods, and edge cases | assignments, weights, centers and scales, bit for bit, except a row exactly tied between centers when there are more than 16 of them ([compatibility.md](compatibility.md#exact-distance-ties-in-create_dn)) | [create_dn.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/create_dn.md) |
| `refine_dn` | 128 cases: every combination of its arguments on faithful, and six more datasets | every kmeans call's rows and clusters, and the outputs, bit for bit. The input nuggets' two largest eigenvalues and the shape-split projections within tolerance | [refine_dn.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/refine_dn.md) |
| `get_dn_means` | 293 nugget sets from the two groups above | bit for bit, an empty nugget's NaN sign included | [get_dn_means.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/get_dn_means.md) |
| `arithmetic_complexity` | 3481 calls | the pass counts equal, and the value within `rtol=1e-8` and `atol=1e-10`. It raises on the 19 calls R never returned from | [arithmetic_complexity.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/arithmetic_complexity.md) |
| `wmean`, `wss`, `wwcss` | 66 cases, 49 reported | bit for bit | [wmean.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/wmean.md), [wss.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/wss.md), [wwcss.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/wwcss.md) |
| `distw` | the 62 of those with two clusters or more, 45 reported | bit for bit, with R's NA where R leaves NA | [distw.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/distw.md) |
| `cluster_predict` | 51 cases and 7,392 new rows, 39 cases and 6,422 rows reported | the distances bit for bit, the positions equal | [cluster_predict.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/cluster_predict.md) |
| `wpca` | 29 cases, 25 reported | everything up to the eigen input bit for bit, the eigen output by a rule | [wpca.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/wpca.md) |
| `whclust` | 14 cases, 11 where R returns | `merge` and `order` equal, `height` bit for bit, R's message where R stops | [whclust.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/whclust.md) |
| `wkmeans` | 15 cases, 12 where R returns, and 28 argument combinations, 26 where R stops and 2 where it returns labels | labels equal, centers and WWCSS bit for bit, every draw read in R's order, R's message where R stops | [wkmeans.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/wkmeans.md) |
| `dn_` functions | a quakes datanugget of 150 nuggets of 1,000 rows | as for the functions they wrap | [dn_wkmeans.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/dn_wkmeans.md), [dn_whclust.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/dn_whclust.md), none for `dn_wpca` and `dn_cluster_predict` |
| `cut_tree` | 13 trees | 940 cuts at `k` and 1,729 at `h` | [cut_tree.md](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/cut_tree.md) |

Exact comparisons come in three strengths. `same_bits` compares every float64 bit, NaN payloads included. `same_values` compares the bits but lets any NaN match any NaN. NumPy's `assert_array_equal` also treats -0.0 and 0.0 as equal. The `WCluster`, compat, R primitive and `get_dn_means` tests use the first two. The `create_dn`, `refine_dn`, `kmeans` and Rfast tests use `assert_array_equal`, and the parity tables for `create_dn` and `refine_dn` are built with `same_values`.

The plot tests hold the dendrogram's lines to the polylines in R's PDF within 0.006 points, and the markers' sizes to the `pch = 16` disc in R's PDF within `rtol=1e-12`. Elsewhere tolerances appear only where LAPACK or the platform's `log10` is involved. `arithmetic_complexity`'s value uses `rtol=1e-8` and `atol=1e-10`. `refine_dn` uses the same tolerance for the two largest eigenvalues of each input nugget, on the first case of each dataset and input nugget set, and requires an exact match when there is one column. It also uses it for the projection a shape split hands to k-means, after the projection's sign is aligned with R's. Eigenvalues computed after a split aren't compared directly. The split decisions they steer are checked through the clusters of every k-means call and through the outputs, which match R's exactly. So no split decision in the fixture grid changes because of that rounding. `wpca`'s eigen output is held by a rule instead: the rotation must be orthonormal, each column must solve the eigen input for R's eigenvalue to within 1024 machine epsilons times the largest absolute value among R's eigenvalues, each `sdev` squared must lie within that same band of R's eigenvalue, a cluster of near-equal eigenvalues must span R's subspace, and the scores must be R's product of the centered data and the port's rotation. The rule fails on a NaN or an infinity, except a NaN `sdev` where R's eigenvalue is at most that band.

The tests feed R's recorded draws back through `ReplayDraws`, which raises if the port asks for a draw R didn't make, asks with different arguments, or leaves one unread. So the order of the random calls is checked as well as the numbers. `RDraws` reproduces R's `set.seed` and `sample.int` on 27 recorded cases, and `refine_dn` with `RDraws` returns R's nuggets on all 128 recorded cases.

The compiled kernels are checked against plain-Python references on fuzzed inputs built for ties, overflow and subnormals. They're kept under `tests/`, or in the package where the runtime also uses them. The references of `whclust` and `wkmeans` follow R's code statement by statement and count R's quirks, and a case named for a quirk was recorded only after a counting copy of R's function showed the case reaches it. On the first 10 columns of UCI Covertype at 10,000, 100,000 and 581,012 rows, run on an i7-13650HX laptop under Windows 11 with one thread and R's random draws replayed, `create_dn` and `refine_dn` matched R's assignments, weights, centers, scales, progress messages and warnings exactly in every run ([reproducibility.md](reproducibility.md#covertype-create_dn-and-refine_dn-against-r)). Those runs predate the lowest-index tie-break, and Covertype's whole-number columns hold many exact ties, so [compatibility.md](compatibility.md#exact-distance-ties-in-create_dn) is the current word on where `create_dn`'s assignments can differ from R's.

These results hold for one reference environment. The port doesn't use the host's long double. The [recorded suite run](reproducibility.md#the-test-suite-at-the-merge-into-main) of the code of 2026-10-09 passed on arm64 macOS too. Two outputs come from the host's libraries. The eigen output of `wpca` and `refine_dn` comes from the installed scipy's LAPACK, and the value of `arithmetic_complexity` comes from the platform's `log10`. The tests hold both to R's by a rule or a tolerance. A NaN that `wss`, `wwcss` or `distw` computes comes from the host's own arithmetic, and the tests let it match any NaN. An R built with another BLAS, or R on arm64 macOS, can round differently, and on tied data that can move rows between nuggets ([limitations.md](limitations.md)).

### What is tested and what is not

| Part | Status |
| --- | --- |
| Data nuggets, the `WCluster` functions, `cut_tree` | ported, and checked against R's recorded output |
| The paper's claims about structure preservation | not tested here |

## References

Amaratunga, D., and Cabrera, J. *Exploration and Analysis of DNA Microarray and Protein Array Data*. John Wiley & Sons, 2009 (Vol. 605), as the WCluster manual cites it. DOI: `10.1002/9780470317129`.

Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`.

Becker, R. A., Chambers, J. M., and Wilks, A. R. *The New S Language*. Chapman and Hall/CRC, London, 1988. ISBN 053409192X.

Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023. Published online in 2022. DOI: `10.1007/978-3-031-22687-8_20`.

Ward, J. H., Jr. "Hierarchical Grouping to Optimize an Objective Function." Journal of the American Statistical Association 58(301), 236-244, 1963. DOI: `10.1080/01621459.1963.10500845`.

Covertype data: Blackard, J. (1998). [Covertype](https://doi.org/10.24432/C50K5N) [Dataset]. UCI Machine Learning Repository. Licensed under CC BY 4.0.
