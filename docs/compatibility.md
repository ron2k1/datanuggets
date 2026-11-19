# Compatibility with R

`datanuggets` ports CRAN's `datanugget` 1.5.0 and `WCluster` 1.3.0. On the cases recorded from R that the tests compare with, it matches R bit for bit, except for the eigen solver's output and the differences on this page. Of the 46 `compat` cases, 39 aren't compared. They record what R does with labels, `corr` values, `nstart`, fractional nugget counts, predictions with no answer and the one-column `"random"` case of `create.DN` (`create_p1_random`), and the names and column types of `create.DN`'s and `refine.DN`'s results (`frame_types`), as evidence for decisions still open. This page lists what is ported, how R's names and arguments map to Python's, and where the behavior differs from R's. The recorded cases and the R build they came from are in [reproducibility.md](reproducibility.md), and the per-function parity tables are in [validation/](https://github.com/ron2k1/datanuggets/tree/main/docs/validation). What the port can't guarantee is in [limitations.md](limitations.md).

## Ported functions

Version `0.1.0a3` is a research alpha. Eighteen functions are ported:

| Python | R | R package |
| --- | --- | --- |
| `create_dn` | `create.DN` | `datanugget` |
| `create_dn_centers` | `create.DNcenters` | `datanugget` |
| `refine_dn` | `refine.DN` | `datanugget` |
| `get_dn_means` | `getDN.means` | `datanugget` |
| `arithmetic_complexity` | `AC` | `datanugget` |
| `wmean` | `wmean` | `WCluster` |
| `wss` | `wss` | `WCluster` |
| `wwcss` | `wwcss` | `WCluster` |
| `distw` | `distw` | `WCluster` |
| `wpca` | `Wpca` | `WCluster` |
| `cluster_predict` | `cluster.predict` | `WCluster` |
| `wkmeans` | `Wkmeans` | `WCluster` |
| `whclust` | `Whclust` | `WCluster` |
| `dn_wkmeans` | `DN.Wkmeans` | `WCluster` |
| `dn_whclust` | `DN.Whclust` | `WCluster` |
| `dn_wpca` | `DN.Wpca` | `WCluster` |
| `dn_cluster_predict` | `DNcluster.predict` | `WCluster` |
| `cut_tree` | `cutree` | `stats` |

The [function reference](api.md) shows each function's R usage and argument map beside the Python one.

`refine_dn` splits nuggets that are spread out or elongated with 2-means, through a port of the Hartigan-Wong k-means in R's `stats::kmeans`. `get_dn_means` averages the rows in each nugget. `arithmetic_complexity` gives the arithmetic cost of a `create_dn` call on a log10 scale, from the shape of the data alone.

The `WCluster` functions treat each row as a point with a weight, such as a nugget with its row count. `wmean` gives each cluster's weighted mean and `wss` the weighted sum of squares around the weighted mean. `wwcss` adds up each cluster's `wss`, and `distw` gives how much merging two clusters adds to it. `cluster_predict` sends new rows to the nearest weighted cluster mean. `wpca` is principal components with a weight per row and per column.

`wkmeans` is weighted k-means and `whclust` is weighted Ward hierarchical clustering, both as `WCluster` computes them, quirks included. The four `dn_` functions are `WCluster`'s wrappers that run `wkmeans`, `whclust`, `wpca` and `cluster_predict` on a `DataNuggets`, with the nugget centers as the data and the nugget weights as the row weights. `dn_wkmeans` also gives each original row the cluster of its nugget. They take `datanugget`'s output, but `datanugget` 1.5.0 doesn't define them, so they're listed under `WCluster`.

`cut_tree` is R's `stats::cutree` for a `whclust` or `dn_whclust` tree, which the `WCluster` examples use to cut a tree into clusters. The `datanuggets.plot` package draws the trees, clusters, nuggets and weighted PCA results, including the plots R's help examples draw. [Plots](#plots) below covers how they relate to R's drawings.

### RDraws

`RDraws` reproduces R's `set.seed` and `sample.int`. Passed as `seed=RDraws(291102)`, it makes `create_dn`, `refine_dn`, `wkmeans` and `dn_wkmeans` draw what R's `create.DN`, `refine.DN`, `Wkmeans` and `DN.Wkmeans` draw with that seed. Import it from `datanuggets`:

```python
from datanuggets import RDraws
```

### Not ported

`datanugget`'s `optimal.DN` and its elbow and difference plots are not implemented yet, and are listed among the [open problems](limitations.md#open-problems).

## The same pipeline in R and Python

The usual pipeline on R's `quakes` data, unscaled so that both sides read the same numbers:

```r
library(datanugget)
library(WCluster)
write.csv(quakes, "quakes.csv", row.names = FALSE)  # the same rows for Python
X <- as.matrix(quakes)

my.DN <- create.DN(x = X, center.method = "mean", R = 300, delete.percent = 0.1,
                   DN.num1 = 400, DN.num2 = 100, dist.metric = "euclidean",
                   seed = 291102, no.cores = 0, make.pbs = FALSE)
my.DN2 <- refine.DN(x = X, DN = my.DN, EV.tol = 0.9, max.splits = 5,
                    min.nugget.size = 2, seed = 291102, no.cores = 0, make.pbs = FALSE)
km <- DN.Wkmeans(datanugget = my.DN2, k = 3, num.init = 10,
                 max.iterations = 10, seed = 291102)
tree <- DN.Whclust(datanugget = my.DN2)
cl <- cutree(tree, k = 3)
pca <- DN.Wpca(datanugget = my.DN2, wcol = NULL, corr = FALSE)
pred <- DNcluster.predict(datanugget = my.DN2, cl = cl, newx = as.data.frame(X[1:25, ]))
```

```python
import pandas as pd
from datanuggets import (
    create_dn,
    cut_tree,
    dn_cluster_predict,
    dn_whclust,
    dn_wkmeans,
    dn_wpca,
    refine_dn,
)
from datanuggets import RDraws

X = pd.read_csv("quakes.csv").to_numpy(dtype=float)
seed = RDraws(291102)  # the draws of R's set.seed(291102)

dn = create_dn(
    X,
    center_method="mean",
    r=300,
    delete_percent=0.1,
    dn_num1=400,
    dn_num2=100,
    dist_metric="euclidean",
    seed=seed,
    n_jobs=1,
    progress=False,
)
dn2 = refine_dn(
    X, dn, ev_tol=0.9, max_splits=5, min_nugget_size=2, seed=seed, n_jobs=1, progress=False
)
km = dn_wkmeans(dn2, k=3, num_init=10, max_iter=10, seed=seed)
tree = dn_whclust(dn2)
cl = cut_tree(tree, k=3)
pca = dn_wpca(dn2, wcol=None, corr=False)
pred = dn_cluster_predict(dn2, cl, X[:25])
```

Where R returns 1-based values, Python returns them minus one: nugget ids, `assignments`, `kept`, cluster labels, `order`, and the positions `cluster_predict` and `dn_cluster_predict` give. Results are objects with fields where R returns named lists: `dn.assignments` is `my.DN[["Data Nugget Assignments"]]` minus one, `km.row_labels` is `km[["Cluster Assignments for original dataset"]]` minus one, and `dn.to_frame()` gives R's `Data Nuggets` table with 0-based ids.

An integer `seed` draws from NumPy, so it does not give R's draws for the same number. `seed=RDraws(291102)` gives them, as [RDraws](#rdraws) above describes. On the `create.DN` call above it gives R's recorded nuggets exactly. [Differences from R](#differences-from-r) below lists the rest, including exact distance ties among more than 16 centers, eigenvector signs and `merge` numbering.

## Differences from R

### Defaults and argument names

Defaults match R except for `n_jobs` and the default weights of `wmean`, `wwcss` and `distw`, which are described below. `create_dn` and `refine_dn` default to `n_jobs=1`, while R's `create.DN` and `refine.DN` default to `no.cores = detectCores() - 1`. R runs serially for any `no.cores` of 1 or less. Its help pages give 0 for that, and their runnable examples pass it. In the port use `n_jobs=1`: `n_jobs` of 0 or below -1 raises `ValueError`, and `n_jobs=-1` means all cores, where R's -1 runs serially. `create_dn` maps R's argument names as follows:

| R | Python |
| --- | --- |
| `x` | `x` |
| `center.method` | `center_method` |
| `R` | `r` |
| `delete.percent` | `delete_percent` |
| `DN.num1` | `dn_num1` |
| `DN.num2` | `dn_num2` |
| `dist.metric` | `dist_metric` |
| `seed` | `seed` |
| `no.cores` | `n_jobs` |
| `make.pbs` | `progress` |

`center_method` accepts `"mean"`, `"random"` or `"original"`. `dist_metric` accepts `"euclidean"` or `"manhattan"`. `progress=True` draws a bar over the stage-1 chunks only. R's `make.pbs` also draws a second bar over the points stage 2 deletes.

### Shared behavior

Nugget IDs and assignments are 0-based instead of R's 1-based values. `seed` accepts an integer, a NumPy `Generator` or an `RDraws`. An integer or a `Generator` draws from NumPy, so the same seed doesn't give the same nuggets as R. `RDraws(seed)` draws what R draws.

Every function that reads the values in the data rejects NaN and infinite values. What R does with them isn't recorded for the `datanugget` functions. In `cluster.predict` one NaN row in `newx` empties R's whole result. Probes outside the fixtures show `wmean`, `wss`, `wwcss` and `distw` passing NaN through their arithmetic, and `Wpca` stops in `eigen()` once the matrix it hands over isn't finite, as the recorded weight cases show. Recording R's behavior with them is one of the [open problems](limitations.md#open-problems).

A few differences are kept on purpose. `n_jobs=-1` means all cores. Warnings come as `DataNuggetWarning`, and those raised in worker processes are sent back and raised again in the caller's process. Progress messages go to the `datanuggets` logger in place of R's `message()`. The port draws from its own generator, so it doesn't advance the caller's global random state the way R's functions advance `.Random.seed`.

### Exact distance ties in create_dn

`create_dn` gives every row to its nearest second-stage center, as R's `create.DN` does with `Rfast::dista(k = 1, index = TRUE)`. When a row is exactly as far from two or more centers, compared as doubles with no tolerance, the port gives it to the lowest-numbered of them. Rfast takes whichever tied center C++'s `std::sort` leaves first. With 16 or fewer centers that is always the lowest-numbered one, so R and the port agree. With 17 or more centers R can take another of the tied centers, even when only two of them tie. The recorded `Rfast::dista` cases fit this rule. In those with 12 centers R takes the lowest-numbered of the tied centers on every tied row. In those with 20 to 30 centers it takes another on 142 rows, including rows where only two centers tie.

Exact ties need repeated or grid-like values, such as whole numbers or duplicated rows, and they are rare in continuous measurements. In 9 of the 37 recorded `create.DN` cases, between 1 and 12 rows tie this way and land in another nugget than in R. That changes those nuggets' weights and scales and can change their mean centers. With `center_method="random"` the draws for later nuggets can differ too. The tests hold the moved rows to the lowest-index rule, and hold the rest of those cases to R's output by carrying on from R's assignments.

### refine_dn

`refine_dn` maps its arguments the same way:

| R | Python |
| --- | --- |
| `x` | `x` |
| `DN` | `dn` |
| `EV.tol` | `ev_tol` |
| `max.splits` | `max_splits` |
| `min.nugget.size` | `min_nugget_size` |
| `shape.split` | `shape_split` |
| `min.shape.size` | `min_shape_size` |
| `delta` | `delta` |
| `nstart` | `nstart` |
| `seed` | `seed` |
| `no.cores` | `n_jobs` |
| `make.pbs` | `progress` |

`refine_dn` reads only the assignments from `dn`. Like R, it recomputes every center, weight and scale from the rows. It also drops empty nuggets and gives single-row nuggets a scale of 0, where `create_dn` keeps empty nuggets and floors single-row scales. Both come from R. R re-seeds before every split, so each split starts from a fresh copy of the seed's stream. The port does the same with an int seed or an `RDraws`. With a NumPy `Generator` each split draws from a child stream spawned from it in candidate order. With any of the three the result doesn't depend on `n_jobs`. With `n_jobs` other than 1 the splits of a round run in worker processes. Unlike R, `refine_dn` rejects a non-finite `nstart`, where R leaves every nugget unsplit, and a `shape_split` that isn't a bool. A NaN or infinite `max_splits` raises `ValueError` up front, where R stops later with its own error (at its argument check for NaN, at `seq_len()` for `Inf`).

### create_dn_centers

`create_dn_centers(rs, dn_num, *, delete_percent=0.1, dist_metric="euclidean")` maps R's arguments as follows:

| R | Python |
| --- | --- |
| `RS` | `rs` |
| `delete.percent` | `delete_percent` |
| `DN.num` | `dn_num` |
| `dist.metric` | `dist_metric` |
| `make.pbs` | not ported |

R's order puts `delete.percent` before `DN.num`. The port puts `dn_num` second and makes `delete_percent` and `dist_metric` keyword-only, so a call written in R's order, such as `create_dn_centers(rs, 0.1, 50)`, raises `TypeError` instead of taking 0.1 as the count. A [planned change](limitations.md#planned-changes) will adopt R's order. There is no progress bar. R returns the kept rows as a data frame with row names 1 to m and the kept rows' 1-based positions in its `"kept"` attribute. The port returns a `CentersResult`: `centers` holds the kept rows and `kept` their 0-based positions, R's minus one.

### get_dn_means

`get_dn_means(x, dn)` takes R's `x` and `DN` and returns a NumPy array with one row per nugget, where R returns a data frame. An empty nugget gets a row of R's -NaN. With one column R transposes its result into a single row, but `get_dn_means` still returns one row per nugget. R ignores any rows of `x` past the ones `dn` assigns. It also skips a row whose nugget id isn't a whole number from 1 to the nugget count. `get_dn_means` raises `ValueError` in both cases.

### arithmetic_complexity

`arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2)` names its arguments like `create_dn` and has no defaults, like R's `AC`. It reads only the shape of `x`. R 4.6.1's class check in `AC` stops on a matrix or a data.table, so R only takes a plain data frame. The port takes any 2-D input. R's first loop never returns when `floor(DN.num1)/G` is negative but not exactly -1 (`G = ceiling(nrow(x)/floor(R))`). Its second loop never returns when `floor(DN.num2)` is below -1 and below `floor(DN.num1)`. A loop also never returns when a pass rounds back to the value it started from, as with `DN.num1 = 1e17`, `DN.num2 = 1` and `delete.percent = 1e-18`. The port raises `ValueError` on these at once, even where R would count down for a long time before the cycle starts. It also raises on a `dn_num1` or `dn_num2` that isn't finite and on an `x` with no rows, where R stops, loops, or returns `Inf` or `NaN`.

### The WCluster statistics and wpca

In the `WCluster` functions `Wpca` becomes `wpca`, `cluster.predict` becomes `cluster_predict` and the `groupSum` argument of `wwcss` becomes `group_sum`. The other names are R's. A weight argument left as `None` means one weight per row, or per column for `wpca`'s `wcol`. R's `wmean`, `wwcss` and `distw` default to `rep(1, length(x))` instead, which has the wrong length. It stops unless the data is square (`wmean`, `wwcss`) or has one column (`distw`). A cluster whose weights sum to zero gets a NaN `wwcss` and NaN in every `distw` entry that touches it, as in R. Its `wmean` row is NaN when every weight is zero. When weights of both signs cancel, the row is infinite instead, with NaN in any column whose weighted sum is zero too, again as in R. `cluster_predict` doesn't pick such a cluster while another is at a finite distance. The other differences are these:

| Item | R | Python |
| --- | --- | --- |
| `wmean` layout | `p` rows by `k` columns | `(k, p)`, like every other center array in the package |
| `distw` with one cluster | stops ("missing value where TRUE/FALSE needed") | `ValueError` naming the cause |
| cluster labels | `wmean` and `wwcss` group by `factor()` (15 significant digits), and `cluster.predict` through `wmean`; `distw` by `sort(unique())`; text by locale collation | whole numbers smaller than 1e15 in size, sorted. Text, bool, fractional and NaN labels raise |
| `cluster.predict` result | 1-based position in sorted label order, not the label | 0-based position |
| `cluster.predict` argument order | `(x, w, cl, newx)`, and a swapped `w` and `cl` runs silently | `(x, cl, newx, *, w=None)` |
| `cluster.predict` for a row where every distance is NaN (NaN centers, another new row that overflows, or centers of infinities once there are two new rows) | `integer(0)` when every row is like that, a list with empty entries when only some are | `ValueError` |
| `cluster.predict` distances after R's identity multiply | `0 * Inf` from one overflowing row turns its whole column NaN, and a lone `Inf` stays | the same, computed without the multiply |
| a vector `newx` in `cluster.predict`, a vector `x` in `getDN.means`, `create.DN` or `refine.DN`, a vector `RS` in `create.DNcenters` | stops. The `datanugget` functions say `x must be of class "matrix", "data.frame", or "data.table"`, with `RS` in place of `x` for `create.DNcenters` | `ValueError`. A text vector raises `TypeError`, and so does a bool vector in `create_dn`, `create_dn_centers` and `refine_dn`. `get_dn_means` and `cluster_predict` read bools as 0 and 1, so a bool vector raises `ValueError` there |
| integer `x` with integer weights | when a value times its weight passes the integer range, R 4.6.1's `wmean` warns "NAs produced by integer overflow" and returns NA in that cluster's center for the column that overflows. `cluster.predict` gives the same warning and still returns a cluster for the new row. `Wpca` multiplies the same integers, and its code sets the resulting NA to 0 in its column sum, which gives a wrong center. `wss` returns a finite value with no warning, and `wwcss` and `distw` compute through `wss` | float64 arithmetic with no overflow and no warning. `wmean` returns the center R's `wmean` returns when `x` is stored as double, and `cluster_predict` sends the new row to the nearest center |
| NaN or infinite weights, text or bool weights | passed through, or stopped late in `eigen`. `Wpca` stops text and logical weights at its class check ('wrow must be of class "numeric" or "table"'); NaN or infinite weights reach `eigen` and stop there | `ValueError` or `TypeError` up front |
| `Wpca` row weights summing to 1, a zero total, a negative row weight, zero variance under `corr`, row weights summing below 1 under `corr`, column weights with a zero sum or mixed signs | stop inside `eigen` ("infinite or missing values in 'x'"), some after a "NaNs produced" warning | `ValueError` naming the cause |
| `Wpca` with one column and `corr = TRUE` | `diag()` of one number builds an identity: the covariance passes through unscaled for `0.5 < dd <= 1`, and R stops otherwise | the general formula, `sdev` 1 to rounding. It fixes an upstream bug |
| `Wpca` `scale` with `corr = FALSE` | the single number 1 | ones, one per column |
| `Wpca` `corr` | `TRUE`, `1`, `1L` and `"TRUE"` mean corr. `FALSE`, `0`, `0L`, `"FALSE"`, `"F"`, `"false"` and `"False"` mean no corr. `"T"`, `"true"`, `"True"`, `2` and `-1` stop with "object 'dd' not found" in a clean session and use the caller's `dd` otherwise. `NA` stops with "missing value where TRUE/FALSE needed" | a bool, else `TypeError` |
| `w = NULL` | stops, though the help pages say it means unit weights | `None` means unit weights |

`wpca`'s `rotation` columns, and the scores with them, can have the opposite sign from R's, since LAPACK picks each column's sign.

### wkmeans, whclust and the dn_ functions

`Wkmeans` becomes `wkmeans` and `Whclust` becomes `whclust`. In `wkmeans`, R's `dataset`, `cl.centers`, `obs.weights`, `num.init` and `max.iterations` become `x`, `centers`, `w`, `num_init` and `max_iter`, and `k` and `seed` keep their names. Every argument after `k` is keyword-only. R's order puts `cl.centers` before the weights, so a call written in R's order fails instead of taking the weights as centers. `whclust(x, w=None)` keeps R's names. The `dn_` functions take a `DataNuggets` as `dn` where R takes a datanugget, and `dn_wkmeans` names its other arguments like `wkmeans`, keyword-only too. `w=None` means unit weights, `n_jobs` is new and never changes the result, and the other defaults are R's. The other differences are these:

| Item | R | Python |
| --- | --- | --- |
| `Wkmeans` and `DN.Wkmeans` labels | 1-based | 0-based, R's label minus one. A gap in R's labels stays a gap |
| `Whclust` `merge` | a single row as its negative row number, a merged cluster as the 1-based step that made it | scipy's numbering, one to one: row `r` is `r`, and the cluster made at 0-based step `s` is `n + s`. `to_linkage()` gives the linkage that `scipy.cluster.hierarchy.dendrogram` takes |
| `Whclust` `order` | 1-based | 0-based |
| `Wkmeans` `cl.centers` | `p` rows by `k` columns | `centers`, `(k, p)` |
| `Wkmeans` with a vector `dataset` | stops, since `nrow()` of a vector is `NULL`: "invalid 'times' argument" with the default weights, "argument is of length zero" with given ones | one column |
| `Wkmeans` output | prints each start's WWCSS, and sends "Iteration" lines to `message()`, plus an "Initialization" line per start when `num.init` is above 1 | `logging.debug` on the `datanuggets` logger, one line per start |
| `Wkmeans` without user centers on at least `k` but fewer than `2k` rows, or on fewer than `k` distinct pairs of a row and its weight | draws starts forever | `ValueError` |
| `Wkmeans` starts whose random partitions keep leaving a label on fewer than two rows or keep missing `k` clusters | keeps drawing, maybe forever | `ValueError` after `MAX_DRAWS`, 1,000 random partitions in one start, both kinds of redraw counted together |
| `Wkmeans` on one row with `cl.centers` and a finite weight outside `[1, 2)` | `diag(w)` is an identity matrix of size `as.integer(w)`, so R stops. For a weight of size 2^31 or more, `as.integer(w)` is NA, and R warns "NAs introduced by coercion to integer range" before it stops with "invalid 'nrow' value (too large or NA)". From 67108865 (2^26 + 1) up to 2^31, R stops with "too many elements specified". A smaller weight that is too large for R to allocate `diag(w)` stops R on memory | `ValueError` with R's message and no warning. Where R stops on memory, the message is "non-conformable arguments" |
| `num.init` or `max.iterations` finite and below 1 after the floor | `1:v` counts down and R returns a result. The loop over starts runs 2 - v times (twice for 0). The loop over rounds runs at most 2 - v times, since it can stop early | `ValueError`. A `num_init` below 1 with `centers` is set to 1 first, as in R |
| `DNcluster.predict` with a matrix `newx` | stops ("newdata should be either a data frame or a datanugget object.") | runs, as `cluster_predict` does |

The `dn_` functions stop where R's do on a datanugget with one variable, where R's `[, 2:2]` drops the table to a vector: `dn_wkmeans` always, `dn_wpca` without `wcol`, and `dn_cluster_predict` with a `DataNuggets` as `newx`. `dn_whclust` runs, as R's does. The `one_variable` case records R's messages there, and what R returns where it runs. `dn_cluster_predict` reads a `DataNuggets` given as `newx` the way R reads a datanugget there. It takes as many columns of `newx`'s table as `dn` has variables, so a wider `newx` gives its first variables, and a narrower one passes its weights and scales as data. Both stop when the table runs out of columns.

### Result components

R returns named lists, a data frame or an `hclust`. The port returns the result classes in `datanuggets.result`, with these fields in place of R's components:

| R | Python |
| --- | --- |
| `create.DN` and `refine.DN`: `"Data Nuggets"`, a data frame of `Data Nugget`, `Center1` to `Centerp`, `Weight` and `Scale`, and `"Data Nugget Assignments"` | `DataNuggets`: `centers`, `weights`, `scales` and `assignments`. `to_frame()` gives R's table, with 0-based ids |
| `create.DNcenters`: the kept rows as a data frame, with their positions in the `"kept"` attribute | `CentersResult`: `centers` and `kept` |
| `Wkmeans`: `"Cluster Assignments"`, `"Cluster Centers"` and `"WWCSS"`, the last a list of `WWCSS` and `TotalWWCSS` | `WkmeansResult`: `labels`, `centers`, `wwcss` and `total_wwcss` |
| `DN.Wkmeans`: `Wkmeans`' list with the assignments renamed `"Cluster Assignments for data nuggets"`, plus `"Cluster Assignments for original dataset"` | `DnWkmeansResult`: `wkmeans`' fields, with the nuggets' clusters in `labels`, plus `row_labels` |
| `wwcss(groupSum = TRUE)`: `WWCSS` and `TotalWWCSS` | `WwcssResult`: `per_cluster` and `total` |
| `Whclust`: `merge`, `height` and `order`, plus `labels = NULL`, `method = "ward.D"`, `call` and `dist.method = "euclidean"` | `WhclustResult`: `merge`, `height` and `order` only. `to_linkage()`, `cut_tree` and `datanuggets.plot.dendrogram` do what R does with an `hclust` through `cutree` and `plot` |

The arrays carry none of R's names. R names `Wkmeans`' assignments after the data's row names, `wwcss`' values per cluster 1 to k and the nugget table's rows 1 to m.

### cut_tree

`cut_tree` takes one `k` or one `h` and returns one label per row, where R's `cutree` also takes a vector of them and returns a matrix with a column per cut. Its labels are R's minus one. With both `k` and `h`, `k` wins and `h` is ignored, as in R. A fractional `k` is truncated, as `as.integer` does.

The clusters are numbered in the order they first appear along the rows. On 13 trees R's `Whclust` and `DN.Whclust` made, `cut_tree` matches R at every `k` from 1 to n, 940 cuts, and at 1,729 values of `h`: every merge height, the midpoints between them, values below and above them all, and both infinities. It stops where R stops, with R's message. On a `k` of magnitude 2^31 or more, infinities included, R also warns "NAs introduced by coercion to integer range" before it stops. The port gives no warning there.

## Plots

`datanuggets.plot` needs the `plot` extra ([installation.md](installation.md)), and [quickstart.md](quickstart.md) has a worked example. Every plot function takes `ax=`, draws on `plt.gca()` without it, and returns the Axes it drew on.

`dendrogram(tree)` draws a `whclust` or `dn_whclust` tree as R's `plot.hclust` does. Leaves sit at 1 to n in the tree's `order` and each merge is a bar at its height. A leaf hangs below its parent's bar by `hang`, 0.1 by default, times the last height minus the first. When that amount is negative, every leaf sits at 0. On the eight recorded trees R draws, and on one of them again with `hang` at -1 and 0.5 and with its heights out of order, every line matches the polylines R wrote into a PDF to within 0.006 points, where R's PDF rounds to 0.005. Leaves are labelled with their 0-based row numbers, or with a list of names, or not at all with `labels=False`. It stops where `plot.hclust` stops, with R's message.

`nuggets(dn)` draws the nugget centers on two columns as the `DN.Whclust` and `DN.Wkmeans` examples do, R's `pch = 16` at `cex = 0.5`, a disc 2.7 points across, the size R's own PDF draws. Without labels it uses the examples' colors, `rgb(0, min(t, 1), 0)` with `t` a nugget's share of the weight over the 80% quantile of the shares, so a lighter green means more rows. Those colors equal the hex strings R gives for the 150 quakes nuggets the `dn_` tests use, for a weight vector with ties, and for the 32 fractional weights of `mtcars$wt`, and the values of `t` behind them equal R's bit for bit. On `mtcars$wt` numpy's pairwise sum would break that, though a plain left-to-right sum wouldn't, so R's long double sum is left to the tests aimed at it. With `labels`, each nugget takes its cluster's color.

`clusters(x, labels, *, w=None, centers=None)` is the `Whclust` and `Wkmeans` examples' `plot(x, col = labels)` with the centers on top: the rows on two columns in their cluster's color, sized by weight, and each cluster's center as a star with a white edge. With weights and the default `size="area"`, a point's marker size is `6 + 400 * w / max(w)`. It runs from 6 to 406 and grows with the weight. Without weights every point has marker size 20. A marker size is matplotlib's `s`, in points squared, and a disc of size `s` is `sqrt(s)` points across. `size="log"` gives R's `cex = log(w)`.

`scores(pca)`, `scree(pca)` and `loadings(pca)` draw a `wpca` or `dn_wpca` result: the scores on two components, with raw rows in gray behind them when `rows` is given, each component's share of the variance as a bar, and each variable's loadings as an arrow. R's examples draw no PCA plot. The views flip each component so its largest loading is positive, so they don't change with the sign LAPACK happens to pick. The result itself keeps R's signs.

`style()` is a `with` block in which new figures take a paper style: 7 to 9 point fonts, thin lines, the Okabe-Ito colors, which color-blind readers can tell apart, and TrueType fonts embedded in the PDF. The font type and the dpi of `savefig` are read when a figure is saved. A figure saved after the block gets matplotlib's own font type, Type 3 by default, even through `save`. A plain `savefig` after the block also uses matplotlib's own dpi, 100 by default. `save(fig, path, width="single")` sizes the figure to one journal column, 3.5 inches wide, or two with `"double"`, 7 inches, or any width in inches, with the height 0.75 times the width unless it's given. It saves at 300 dpi in the format the path's suffix names, such as `.pdf` or `.png`, and stops on a path without one. A PDF's page is exactly that size. A PNG's sides are the inches times 300, rounded down, so the default single-column PNG, 3.5 by 2.625 inches, is 1050 by 787 pixels. A `colors` list must not be empty, and a bad entry raises before anything is drawn. The exception is a list whose entries are all 3-item or all 4-item tuples or lists. There a bad entry such as `(2, 0, 0)` raises only when a label takes it, and in `scores` that can be after the raw `rows` are drawn. A flat `(r, g, b)` or `(r, g, b, a)` of numbers from 0 to 1 raises no error and is not read as one color. Each label takes one of its numbers. The colormap then spreads the numbers in use from its first color to its last. `nuggets` without `labels` does not read `colors`, so there no `colors` value raises. `colors=R_PALETTE` gives R's default `palette()`, and since the port's labels are R's minus one, label `l` then gets R's `col = l + 1`.

### Departures from R's drawings

The plots depart from R's drawings in six ways:

1. Leaf labels and cluster labels are 0-based, like every row number in the port.
2. Points are sized by area by default. R's `cex = log(w)` hides points of weight 1 and gives weights below 1 a negative size. It's still there as `size="log"`.
3. There are no titles by default, and the dendrogram's y range is matplotlib's. R sizes its window from the width of the leaf labels.
4. The PCA views flip each component's sign so its largest loading is positive.
5. `dendrogram` raises `ValueError` on a tree with NaN or infinite heights, which `whclust` gives when a row's weight is 0, or when large weights or data values make its sums overflow. `whclust` rejects infinite weights up front. R's `plot.hclust` warns and writes `-nan(ind)`, or coordinates far off the page, into the PDF, so there's no figure to match.
6. Centers are stars of a fixed size, in their cluster's color with a white edge. R's examples draw them as discs or asterisks at `cex = 5`, which cover the points around them.
