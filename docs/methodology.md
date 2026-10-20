# Methodology

This page describes the research problem, the data nuggets method and the weighted analyses as this package implements them, the flow cytometry gate finder, and the `datanuggets-flow` pipeline that joins them. It then says how each part relates to the papers and how it is checked. Most of the package is a port of two R packages and is held to R's output. The flow pipeline adds steps of its own, and those are marked as design choices where nothing in the repository validates them.

Usage is elsewhere: [quickstart.md](quickstart.md) has first examples, [cli.md](cli.md) documents every flag of `datanuggets-flow`, and [compatibility.md](compatibility.md) lists the differences from R.

## The problem

The data nuggets paper (Beavers et al., Section 2) puts the memory and time of typical clustering methods, such as k-means, hierarchical clustering and PAM, at order P·N(N−1)/2 for N rows and P columns. Hierarchical clustering needs the full distance matrix before it can start. The usual way around this is to cluster a random sample, but a sample can miss small groups and the sparse edges of the data.

Flow cytometry is a case where N is large. Each event is a cell measured on many channels, and one sample holds hundreds of thousands of events. The 44 files of the public PBMC record used in this repository hold 10,183,290 events, 20,844 to 516,810 a file. `whclust`, like R's `Whclust`, keeps an n by n matrix of merge costs, 32 MB at 2,000 rows and 800 MB at 10,000, so it can't run on a file's events directly.

Data nuggets reduce the N rows to M groups, each described by a center, a weight (its row count) and a scale. The analyses then run on the M weighted centers. The paper puts the cost of building nuggets mainly in the N·M assignment step, after which the clustering costs are of order M² (Section 3). The paper's claim is that the nuggets keep the structure of the data, including its edges, better than a random sample or support points of the same size (Sections 3 and 3.2). This package doesn't test that claim. It reproduces the R implementation of the method.

## Data nuggets

### Creating nuggets

`create_dn` is a port of `create.DN` from `datanugget` 1.5.0. It follows Algorithm 1 of the paper as the R package implements it:

1. The rows are shuffled once with the seed and split into `ceiling(n / r)` chunks of nearly equal size, as R's `parallel::splitIndices` splits them.
2. Each chunk gets its share of `dn_num1` first-stage centers: `dn_num1` divided by the number of chunks, the remainder going to the first chunks, clamped to between 1 and the chunk's size.
3. Each chunk is reduced to its share by closest-pair deletion. The chunk's full distance matrix is computed, Euclidean or Manhattan. Each round deletes `max(1, floor(t * delete_percent))` of the `t` remaining points, without going below the target. The points are ordered by the distance to their nearest neighbour, and a point is deleted when neither it nor its nearest neighbour has been deleted yet, so one point of each closest pair goes. If that falls short, more points are deleted in the same order. Rounds repeat until the target count remains. A chunk with no more rows than its target passes through whole, with a warning.
4. The first-stage centers of all chunks are pooled and reduced to `dn_num2` by the same deletion.
5. Every row is assigned to its nearest second-stage center, with the same metric.
6. Each nugget's weight is its row count. Its scale is the mean over columns of the sample variances of its rows, which is the trace of its covariance divided by p. Its center depends on `center_method`: `"mean"` takes the mean of its rows, `"random"` one of its rows drawn with the seed, and `"original"` keeps the second-stage center.

Deletion removes points from dense regions first, since their nearest neighbours are closest. Points in the sparse tails survive as centers, often with weight 1. A nugget with fewer than 2 rows gets the smallest scale among the nuggets with 2 or more rows, divided by 100, as R does. The paper sets that scale to 0. An empty nugget keeps its second-stage center and weight 0.

The second stage builds a `dn_num1` by `dn_num1` distance matrix, 800 MB of float64 at the default 10,000, whatever the row count. `arithmetic_complexity`, the port of R's `AC`, reports the arithmetic cost of a `create_dn` call on a log10 scale from the data's shape alone. The paper's appendix derives the cost of the algorithm.

### Refining nuggets

`refine_dn` is a port of `refine.DN`. It reads only the assignments of its input and recomputes every nugget's center (the mean of its rows), weight, scale and two largest covariance eigenvalues from the rows of `x`.

1. The split threshold is the type-7 quantile, at `ev_tol`, of the nuggets' positive first eigenvalues. It is computed once, before the first round.
2. For up to `max_splits` rounds, every nugget whose first eigenvalue is above the threshold and that has at least `2 * min_nugget_size` rows is split in two by k-means with k = 2, using a statement-by-statement port of the Hartigan-Wong Fortran behind R's `stats::kmeans`, with `nstart` starts and 1,000 iterations at most. A split is kept only if both children have at least `min_nugget_size` rows. The rounds stop early when no nugget qualifies or none splits.
3. With `shape_split=True` and two or more columns, up to `max_splits` more rounds split nuggets by shape. A candidate has at least `max(min_shape_size, 2 * min_nugget_size)` rows and is elongated, with its first eigenvalue over its second above `delta`, or spread out, with its first eigenvalue above the median plus twice the MAD of the scales of the nuggets with 2 or more rows. Its rows are projected on its first eigenvector and the projection is split by 2-means. The split is kept only if both children are large enough and the mean of their eigenvalue ratios is below `delta`.
4. Empty nuggets are dropped, the rest are renumbered from 0, and a nugget with one row gets scale 0.

R seeds again before every split, so each split draws from a fresh copy of the seed's stream. The port does the same, so the result doesn't depend on `n_jobs`. The eigenvalues that decide the splits come from LAPACK, which is the one place where `refine_dn` can't be held to R bit for bit (see [Evaluation](#evaluation)).

### Defaults, from the code and the paper

The defaults are `datanugget` 1.5.0's, kept in `src/datanuggets/_defaults.py`. Where the paper and the R package differ, the port follows the package.

| Paper | Python | Default in the code | In the paper |
| --- | --- | --- | --- |
| R, the size of each random subset | `r` | 5000, allowed from 100 to 10,000 | 5,000 is the default (Section 3) |
| C, the share deleted per round | `delete_percent` | 0.1 | The text calls 1% the default. Figure 2 and the examples of Sections 3.2 and 3.3 use 0.1, and the flow example of Section 4 uses 0.01 |
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

- `wpca` is principal components with a weight per row and per column. The center is the weighted mean of the rows. The column weights are scaled to sum to p. The covariance of the centered data, scaled by the square roots of both weights, divides by the sum of the row weights minus 1, so the row weights act as counts. `corr=True` scales it to a correlation matrix. `sdev` holds the square roots of its eigenvalues and the scores are the centered data times the rotation.
- `wkmeans` is weighted k-means as `WCluster`'s `Wkmeans` computes it. Each start draws random labels with every label on at least two rows, takes their weighted means as centers and gives each row its nearest center. It then runs up to `max_iter` rounds of 20 Lloyd steps, undoing a step that raises the weighted within-cluster sum of squares (WWCSS), followed by single-point moves that don't raise it. The start with the smallest WWCSS wins. The paper's Algorithm 3 starts from K nugget centers and uses single-point moves only. The port follows `WCluster`, quirks included, and [compatibility.md](compatibility.md#wkmeans-whclust-and-the-dn_-functions) lists where its arguments and stops differ from R's.
- `whclust` is weighted Ward hierarchical clustering. Every row starts as its own cluster, and each step merges the pair whose merge adds the least WWCSS, taking R's first pick on a tie. A merge's height is the total WWCSS after it, so the last height is the WWCSS of all rows as one cluster.
- `cut_tree` is R's `cutree` for one `k` or one height `h`. It numbers the clusters in the order they first appear along the rows.
- `cluster_predict` sends each new row to the nearest weighted cluster mean, by squared Euclidean distance, and returns its 0-based position in sorted label order.
- `dn_wkmeans`, `dn_whclust`, `dn_wpca` and `dn_cluster_predict` are `WCluster`'s wrappers that run these on a `DataNuggets`. `dn_wkmeans` also gives each original row the cluster of its nugget.

## Flow cytometry, as implemented

### The gate finder

`datanuggets.flow.lymphmacro_finder` is Prof. Cabrera's gate finder for the lymphocyte and macrophage populations, ported from his Python, which is his port of his R code. It works on two channels, FSC-A and SSC-A:

1. Both channels are shifted by the smaller of their two minimums, plus 3, and square rooted.
2. `f_plot` bins them into a 100 by 100 grid, each bin `floor(100 * (v - min) / (max - min)) + 1` clipped to 1 to 100, counts the events per cell, takes `log(1 + count)`, and sets cells below 0.1 times the largest to 0. Grid rows are SSC-A bins and columns are FSC-A bins.
3. `local_maxima` keeps each cell equal to the maximum of the 11 by 11 window around it, where that maximum is above the 0.85 quantile of the grid.
4. `refine_peaks` moves each candidate to the densest cell within 2 cells of it (`search_ind`), drops duplicates, keeps a peak whose row is above 10 and below 90, whose column is below 90 and whose density is above the 0.9 quantile of the grid, and sorts the peaks by row.
5. `find_area` keeps the cells closer than `t1` cells to the peak whose density is above `t2` times the peak's. The first peak takes `t11=22` and `t21=0.25`, the others `t12=20` and `t22=0.5`.
6. `hull_gate` takes the convex hull of the area's cells and keeps the events whose bin lies inside it. The population is every event inside the convex hull of those events, in FSC-A and SSC-A. It needs more than 3 cells and more than 3 kept events.

The result has one row per event and one column per population, in peak order. An event can be in more than one population, or in none. With the shipped finder on `202.fcs`, 57,797 of its 282,270 events are in more than one and 29,065 in none. The cutoffs are his and aren't retuned here.

### Where the finder departs from Cabrera's code

1. `find_area` centres its distance disc on the peak. His code builds the grid row by row and flattens the density column by column, so its density test reads the right cell while its disc sits on the mirrored cell `(idy, idx)`. On `202.fcs` this puts 107,909 of the 282,270 events in a different set of populations: his three hold 225,770, 58,245 and 27,590 events, the corrected ones 214,085, 66,149 and 37,742. On a peak on the diagonal the two agree. It's the one change in output, and it waits on a ruling.
2. `hull_gate` tests each grid cell once and looks the answer up per event, and tests the events against a triangulation of their hull's vertices, where his triangulates every kept event. Both cover the same polygon, so they can differ only on an event within rounding of the hull's edge.
3. With no peak left after his filters, the finder returns zero columns. His raises `IndexError`.
4. An area whose cells or kept events lie on one line, or so near one that Qhull calls them flat, gives an empty population. His raises `QhullError`.
5. NaN or infinite values, or a channel with one value, raise `ValueError`. His code bins them or divides by zero.
6. His unused `plot=False` argument is dropped, and `plot.gates` draws his figure with one matplotlib artist per population.

### The disc: `--disc his` and `--disc peak`

Whether the disc should sit on the peak or where his code puts it is an open ruling ([limitations.md](limitations.md#open-problems), problem 10). The library and the command line take different sides of it for now:

- `lymphmacro_finder` uses the corrected `find_area`, with the disc on the peak.
- `datanuggets-flow` defaults to `--disc his`, which gates with a copy of his `find_area` in `datanuggets.flow.pick`, so the populations and their numbers are the ones his figures show. `--disc peak` uses the corrected `find_area`.

The choice changes which events the clustering sees. In the comparison run on the Mac mini (Apple M4, 16 GB) over the 44 files of the record, the corrected disc gates 1.4% to 38.7% of a file's events differently from his, 8.0% at the median.

### The `datanuggets-flow` pipeline

`datanuggets-flow` (`src/datanuggets/flow/pick.py`) runs these steps:

1. Read each FCS file with flowio, as stored and uncompensated. A file whose spillover matrix isn't the identity gets a warning. A folder's files go in name order.
2. Gate each file on FSC-A against SSC-A, or the channels `--fsc` and `--ssc` name, with the finder above, with `--t11` and `--t21` given once or per file, and the area rule `--disc` names.
3. Draw every file's numbered populations in one figure, `gates.png`, and print each population's event count, share and median of the two scatter channels.
4. Pick populations file by file, at a prompt or with `--pops`. The kept events are the union of the picked populations, so an event in two of them counts once. A file left out, or with fewer than 2 kept events, isn't clustered.
5. Take the fluorescence channels: `--channels`, or by default every channel of the first file whose name ends in `-A` and doesn't start with FSC or SSC. Transform them as `arcsinh(x / cofactor)`, with `--cofactor` 6000 by default. A cofactor of 0 leaves them linear.
6. Make data nuggets of each file's kept events separately: `create_dn` with `dn_num2` set to `--dn-num2` (1000 by default) or the number of kept events if that is smaller, and every other argument at its default. There is no refine step.
7. Pool every file's nugget centers and weights, drop nuggets with weight 0, and build one weighted Ward tree with `whclust`.
8. Cut the tree at `--k` clusters, 8 by default, or at the number of nuggets with weight above 0 if that is smaller.
9. Give each kept event its nugget's cluster, and write per-file nuggets and events, per-file cluster counts, each cluster's weighted mean in the transformed units, and figures of the tree, a weighted PCA of the nuggets and the cluster centers.

[cli.md](cli.md) gives every flag, the prompt's grammar, the output files and the conditions on which the command stops.

### Design choices in the pipeline

These choices come from the code. Nothing in the repository measures how they affect the clusters.

- One tree for all files. The clusters are cut from a single tree over every file's nuggets, so a cluster has one definition, and cluster 3 in one file is the same region of marker space as cluster 3 in another. That makes the per-file percentages in `cluster_summary.csv` comparable. Clustering each file on its own would give labels with no correspondence between files.
- Nuggets per file. Each nugget holds the events of one file, so its cluster maps back to that file's events and the per-file counts are sums of whole nuggets. Each `create_dn` call also stays within one file's memory. A file gets at most `--dn-num2` nuggets whatever its event count, but its nuggets carry their event counts as weights in Ward's criterion.
- Gating first, with the populations picked per file. The finder numbers a file's populations in the order of its peaks, so the same population can get a different number in another file, and the pick is made file by file.
- Ward rather than k-means. The tree needs no random starts and can be cut at any k without rerunning. The paper's flow example used weighted k-means instead.
- The transform and the cluster count. arcsinh is close to linear for values well below the cofactor and close to logarithmic above it. No test or recorded result in the repository supports a particular cofactor or k.
- Uncompensated events. They are clustered as the file stores them, and a file with a real spillover matrix gets only a warning. The code's reason is that spectral files are unmixed already.

`create_dn` runs here with its default integer seed, which draws from NumPy. The shuffle is the only random step in the pipeline, so the same files and settings give the same nuggets in the same environment, but not the nuggets R would give.

### The clustering step from Python

The steps after gating can be run without the command line. With `files` a list holding each file's kept, transformed events as arrays of one row per event, this is what steps 6 to 8 do:

```python
import numpy as np
from datanuggets import create_dn, cut_tree, whclust, wmean

dns = [create_dn(x, dn_num2=min(1000, len(x)), n_jobs=-1) for x in files]
centers = np.vstack([d.centers for d in dns])
weights = np.concatenate([d.weights for d in dns]).astype(float)
use = weights > 0
tree = whclust(centers[use], weights[use])
labels = np.full(len(centers), -1)
labels[use] = cut_tree(tree, k=8)
cluster_means = wmean(centers[use], labels[use], weights[use])
```

The nuggets of file `i` are the next `dns[i].n_nuggets` entries of `labels`, and `dns[i].assignments` gives each of its events a nugget. The pooled tree holds an n by n matrix over all the files' nuggets, so with 1,000 nuggets a file, ten files make the 10,000-row case that needs 800 MB.

## Relation to the papers

Beavers et al. (2025) define data nuggets (Algorithms 1 and 2, Section 3), weighted k-means for nuggets (Algorithm 3, Section 3.1), and apply them to a flow cytometry dataset of over one million observations (Section 4). That example made nuggets of the whole dataset with mean centers, R = 5,000, C = 0.01, Minit = 10,000, M = 2,000 and Euclidean distance, refined them with α = 0.25 and Nmin = 2, compared weighted PCA of the nuggets with PCA of the full data and of a random sample, and clustered the refined nuggets with weighted k-means, choosing the number of clusters by comparing WWCSS across a range of k.

`datanuggets-flow` differs from that example in these ways:

| | Paper, Section 4 | `datanuggets-flow` |
| --- | --- | --- |
| Input | one dataset | one or more FCS files, gated on FSC-A and SSC-A first, with populations picked per file |
| Transform | none stated | `arcsinh(x / 6000)` by default |
| Nuggets | one set for the whole dataset, then refined | one set per file, not refined |
| `create_dn` settings | C = 0.01, M = 2,000 | the package defaults, C = 0.1, and 1,000 nuggets a file |
| Clustering | weighted k-means, k chosen from WWCSS | weighted Ward on the pooled nuggets, cut at a given k |
| PCA | weighted PCA compared with PCA of the data | a weighted PCA figure of the nuggets |

The gate finder isn't in the paper. It comes from Prof. Cabrera's own scripts.

A manuscript in revision by Dastgiri, Cabrera and coauthors applies data nuggets to flow cytometry in another way. It compresses the events with data nuggets and then uses differential projection pursuit, a search for projections that maximize an index of the difference between two conditions' densities. This package doesn't implement that analysis.

## Evaluation

### Against R

The port is checked against R's own output. The scripts in `r/` run R 4.6.1 with `datanugget` 1.5.0, `WCluster` 1.3.0 and Rfast 2.1.5.2 on x86-64 Windows, with R's reference BLAS and internal LAPACK 3.12.1. They trace the functions without modifying them and record the inputs, the outputs, every random draw and the internal values that matter: 544 cases in 23 groups. A few more values were checked by hand in R and written into the tests. Running the tests doesn't need R. [reproducibility.md](reproducibility.md#what-r-recorded) lists the groups and what each holds, [development.md](development.md) says how to run the tests, and [architecture.md](architecture.md) shows where the tests and fixtures live. Each known difference from R has a decision record in [the R-compatibility plan](plans/2026-09-30-r-compat.md).

| Function | What R recorded | What is compared | Table |
| --- | --- | --- | --- |
| `create_dn` | 37 cases: iris, faithful and quakes under both metrics and all three center methods, and edge cases | assignments, weights, centers and scales, bit for bit | [create_dn.md](validation/create_dn.md) |
| `refine_dn` | 128 cases: every combination of its arguments on faithful, and six more datasets | every kmeans call's rows and clusters, and the outputs, bit for bit. Eigenvalues and projections within tolerance | [refine_dn.md](validation/refine_dn.md) |
| `get_dn_means` | 293 nugget sets from the two groups above | bit for bit, an empty nugget's NaN sign included | [get_dn_means.md](validation/get_dn_means.md) |
| `arithmetic_complexity` | 3481 calls | the pass counts, and the value bit for bit. It raises on the 19 calls R never returned from | [arithmetic_complexity.md](validation/arithmetic_complexity.md) |
| `wmean`, `wss`, `wwcss` | 49 cases | bit for bit | [wmean.md](validation/wmean.md), [wss.md](validation/wss.md), [wwcss.md](validation/wwcss.md) |
| `distw` | the 45 of those with two clusters or more | bit for bit, with R's NA where R leaves NA | [distw.md](validation/distw.md) |
| `cluster_predict` | 39 cases, 6,422 new rows | the distances bit for bit, the positions equal | [cluster_predict.md](validation/cluster_predict.md) |
| `wpca` | 29 cases, 25 reported | everything up to the eigen input bit for bit, the eigen output by a rule | [wpca.md](validation/wpca.md) |
| `whclust` | 14 cases, 11 where R returns | `merge` and `order` equal, `height` bit for bit, R's message where R stops | none |
| `wkmeans` | 15 cases, 12 where R returns, and 28 runs on arguments R stops on | labels equal, centers and WWCSS bit for bit, every draw read in R's order, R's message where R stops | none |
| `dn_` functions | a quakes datanugget of 150 nuggets of 1,000 rows | as for the functions they wrap | none |
| `cut_tree` | 13 trees | 940 cuts at `k` and 1,729 at `h` | none |

Exact comparisons come in three strengths. `same_bits` compares every float64 bit, NaN payloads included. `same_values` compares the bits but lets any NaN match any NaN. NumPy's `assert_array_equal` also treats -0.0 and 0.0 as equal. The `WCluster`, compat, R primitive and `get_dn_means` tests use the first two. The `create_dn`, `refine_dn`, `kmeans` and Rfast tests use `assert_array_equal`, and the parity tables for `create_dn` and `refine_dn` are built with `same_values`.

Tolerances appear only where LAPACK or the platform's `log10` is involved. `refine_dn`'s eigenvalues and projections and `arithmetic_complexity`'s value use `rtol=1e-8` and `atol=1e-10`, and no split decision in the fixture grid changes because of that rounding. `wpca`'s eigen output is held by a rule instead: the rotation must be orthonormal, each column must solve the eigen input for R's eigenvalue to within 1024 machine epsilons times the largest eigenvalue, a cluster of near-equal eigenvalues must span R's subspace, and the scores must be R's product of the centered data and the port's rotation. The rule fails on a NaN or an infinity.

The tests feed R's recorded draws back through `ReplayDraws`, which raises if the port asks for a draw R didn't make, asks with different arguments, or leaves one unread. So the order of the random calls is checked as well as the numbers. `RDraws` reproduces R's `set.seed` and `sample.int` on 27 recorded cases, and `refine_dn` with `RDraws` returns R's nuggets on all 128 recorded cases.

The compiled kernels are checked against plain-Python references kept in the package, on fuzzed inputs built for ties, overflow and subnormals. The references of `whclust` and `wkmeans` follow R's code statement by statement and count R's quirks, and a case named for a quirk was recorded only after a counting copy of R's function showed the case reaches it. On the first 10 columns of UCI Covertype at 10,000, 100,000 and 581,012 rows, run on the laptop (i7-13650HX, Windows 11) with one thread and R's random draws replayed, `create_dn` and `refine_dn` matched R's assignments, weights, centers, scales, progress messages and warnings exactly in every run ([reproducibility.md](reproducibility.md#covertype-create_dn-and-refine_dn-against-r)).

These results hold for one reference environment. The port doesn't use the host's long double, so it gives the x86-64 results on arm64 macOS too, where the same tests pass. An R built with another BLAS, or R on arm64 macOS, can round differently, and on tied data that can move rows between nuggets ([limitations.md](limitations.md)).

### Against Cabrera's script

The reference for the flow finder is Prof. Cabrera's Python, not his R code. `benchmarks/flow/` holds his functions word for word, his whole script with only its two paths read from the command line, and his finder cut at its steps. `check_reference.py` runs his whole script with FlowKit and checks that its masks equal his pieces' masks. On the Mac mini (Apple M4, 16 GB) it matched on all four files it was given.

The suite compares with three fixtures recorded from his code on files of the record, each thinned to 50,000 events:

- The grid's bins and coordinates, the focal maxima and the refined peaks must equal his exactly. The grid's log counts must equal his to 1 unit in the last place, since on an x86 CPU without AVX-512 numpy's `log1p` is the C library's, which can differ in the last bit.
- `find_area` must equal the corrected rule written out as a plain distance test, and his recorded cells must equal the mirrored disc written out the same way.
- The gate on his cells must give his masks, and so must the whole finder given his area rule, or differ only on events within 1e-9 of the data's range from the hull's edge.
- His guards and his row and column limits are each tested on both sides of the line.

`compare.py` runs his code and ours piece by piece on any FCS file or folder, on the same arrays on both sides, and exits with 1 if a piece differs beyond the hull's edge. It pairs per-sample thresholds with the workbook's rows, where his script pairs them with the samples in FlowKit's `$FIL` order. On the same machine over all 44 files, the grid, the maxima and the peaks equal his on every file. The gate on his cells and the finder with his `find_area` equal his on 40, and on the other 4 one event each differs, lying exactly on the hull's edge. `stress.py` joined the 44 files into one sample of 10,183,290 events, and ours given his `find_area` gave his masks for every event. The record has no reference gates, so whether the gates are biologically right isn't checked, and the [flow gate spec](specs/2026-10-12-flow-gate-design.md) lists that as a non-goal. [reproducibility.md](reproducibility.md#flow-comparepy-piece-by-piece) has the comparison runs.

### What is tested and what is not

| Part | Status |
| --- | --- |
| Data nuggets, the `WCluster` functions, `cut_tree` | ported, and checked against R's recorded output |
| The gate finder's pieces | ported, and checked against his code on the fixtures and on the 44 files |
| `find_area`'s disc on the peak | a correction of his code, tested against an independent statement of the corrected rule. Whether it stands is an open ruling |
| `--disc his`, the copy of his `find_area` in `pick.py` | no test of its own. A check run on the laptop while this page was written found it giving his recorded cells, and with them his recorded masks, on all three fixtures |
| `datanuggets-flow`'s argument and population parsing, the union of picks, the file listing | unit tests in `tests/flow/test_flow_pick.py` |
| `datanuggets-flow`'s transform, per-file nuggets, pooling, Ward and `k` | design choices. No test runs the command end to end on FCS files, and no result in the repository validates the clusters |
| The paper's claims about structure preservation | not tested here |

The clustering steps of `datanuggets-flow` call `create_dn`, `whclust`, `cut_tree` and `wmean`, which are the R-checked functions above. What isn't checked is the pipeline that composes them.

## References

Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D., and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data Structure." Journal of Computational and Graphical Statistics 34(1), 2025. Published online in 2024. DOI: `10.1080/10618600.2024.2341896`. arXiv: `2403.03099`.

Dastgiri, M., Cabrera, J., et al. "Novel Machine Learning Approach to Differential Cell Flow Cytometry Analysis Based on Projection Pursuit." Manuscript in revision.

Flow data: [Zenodo record 22079206](https://zenodo.org/records/22079206), "Flow cytometry data of PBMC of critically ill, mechanically ventilated patients", licensed under CC BY 4.0.

Covertype data: Blackard, J. (1998). [Covertype](https://doi.org/10.24432/C50K5N) [Dataset]. UCI Machine Learning Repository. Licensed under CC BY 4.0.
