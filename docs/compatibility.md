# Compatibility with R

`datanuggets` ports CRAN's `datanugget` 1.5.0 and `WCluster` 1.3.0. On the 544 cases recorded from R it matches R bit for bit, except for the eigen solver's output and the differences on this page. This page lists what is ported, how R's names and arguments map to Python's, and where the behavior differs from R's. The recorded cases and the R build they came from are in [reproducibility.md](reproducibility.md), and the per-function parity tables are in [validation/](https://github.com/ron2k1/datanuggets/blob/main/docs/validation/). What the port can't guarantee is in [limitations.md](limitations.md).

## Ported functions

Version `0.1.0.dev0` is in development. Eighteen functions are ported:

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

`refine_dn` splits nuggets that are spread out or elongated with 2-means, through a port of the Hartigan-Wong k-means in R's `stats::kmeans`. `get_dn_means` averages the rows in each nugget. `arithmetic_complexity` gives the arithmetic cost of a `create_dn` call on a log10 scale, from the shape of the data alone.

The `WCluster` functions treat each row as a point with a weight, such as a nugget with its row count. `wmean` gives each cluster's weighted mean and `wss` the weighted sum of squares around the weighted mean. `wwcss` adds up each cluster's `wss`, and `distw` gives how much merging two clusters adds to it. `cluster_predict` sends new rows to the nearest weighted cluster mean. `wpca` is principal components with a weight per row and per column.

`wkmeans` is weighted k-means and `whclust` is weighted Ward hierarchical clustering, both as `WCluster` computes them, quirks included. The four `dn_` functions are `WCluster`'s wrappers that run `wkmeans`, `whclust`, `wpca` and `cluster_predict` on a `DataNuggets`, with the nugget centers as the data and the nugget weights as the row weights. `dn_wkmeans` also gives each original row the cluster of its nugget. They take `datanugget`'s output, but `datanugget` 1.5.0 doesn't define them, so they're listed under `WCluster`.

`cut_tree` is R's `stats::cutree` for a `whclust` or `dn_whclust` tree, which the `WCluster` examples use to cut a tree into clusters. The `datanuggets.plot` package draws the trees, clusters, nuggets and weighted PCA results, including the plots R's help examples draw. [Plots](#plots) below covers how they relate to R's drawings.

### RDraws

`RDraws` reproduces R's `set.seed` and `sample.int`. Passed as `seed=RDraws(291102)`, it makes `create_dn`, `refine_dn`, `wkmeans` and `dn_wkmeans` draw what R's `create.DN`, `refine.DN`, `Wkmeans` and `DN.Wkmeans` draw with that seed. It isn't exported from `datanuggets` yet, so import it from its module:

```python
from datanuggets._random.r_draws import RDraws
```

### Not ported

`datanugget`'s `optimal.DN` and its elbow and difference plots are not implemented yet ([open problem 7](limitations.md#open-problems)).

`datanuggets.flow` doesn't come from R. It ports Prof. Cabrera's Python gate finder for flow cytometry, `lymphmacro_finder`, and is checked against the reference code piece by piece. Its departures from that code are in [methodology.md](methodology.md), and the `datanuggets-flow` command built on it is in [cli.md](cli.md).

## Differences from R

### Defaults and argument names

Defaults match R except for `n_jobs` and the default weights of `wmean`, `wwcss` and `distw`, which are described below. `create_dn` and `refine_dn` default to `n_jobs=1`, while R's `create.DN` and `refine.DN` default to `no.cores = detectCores() - 1`. `create_dn` maps R's argument names as follows:

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

`center_method` accepts `"mean"`, `"random"` or `"original"`. `dist_metric` accepts `"euclidean"` or `"manhattan"`.

### Shared behavior

Nugget IDs and assignments are 0-based instead of R's 1-based values. `seed` accepts an integer, a NumPy `Generator` or an `RDraws`. An integer or a `Generator` draws from NumPy, so the same seed doesn't give the same nuggets as R. `RDraws(seed)` draws what R draws.

Every function that reads the values in the data rejects NaN and infinite values. What R does with them isn't recorded for the `datanugget` functions. In `cluster.predict` one NaN row in `newx` empties R's whole result. Probes outside the fixtures show `wmean`, `wss`, `wwcss` and `distw` passing NaN through their arithmetic, and `Wpca` stops in `eigen()` once the matrix it hands over isn't finite, as the recorded weight cases show. [Open problem 4](limitations.md#open-problems) has the plan.

A few differences are kept on purpose. `n_jobs=-1` means all cores. Warnings come as `DataNuggetWarning`, and those raised in worker processes are sent back and raised again in the caller's process. Progress messages go to the `datanuggets` logger in place of R's `message()`. The port draws from its own generator, so it doesn't advance the caller's global random state the way R's functions advance `.Random.seed`. `create_dn`, `create_dn_centers` and `refine_dn` raise on a vector `x`, since R's behavior there isn't recorded.

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

`refine_dn` reads only the assignments from `dn`. Like R, it recomputes every center, weight and scale from the rows. It also drops empty nuggets and gives single-row nuggets a scale of 0, where `create_dn` keeps empty nuggets and floors single-row scales. Both come from R. R re-seeds before every split, so each split starts from a fresh copy of the seed's stream and the result doesn't depend on `n_jobs`. With `n_jobs` other than 1 the splits of a round run in worker processes. Unlike R, `refine_dn` rejects an `nstart` or `max_splits` that isn't finite and a `shape_split` that isn't a bool.

### get_dn_means

`get_dn_means(x, dn)` takes R's `x` and `DN` and returns a NumPy array with one row per nugget, where R returns a data frame. An empty nugget gets a row of R's -NaN. With one column R transposes its result into a single row, but `get_dn_means` still returns one row per nugget. R ignores any rows of `x` past the ones `dn` assigns. It also skips a row whose nugget id isn't a whole number from 1 to the nugget count. `get_dn_means` raises `ValueError` in both cases.

### arithmetic_complexity

`arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2)` names its arguments like `create_dn` and has no defaults, like R's `AC`. It reads only the shape of `x`. R 4.6.1's class check in `AC` stops on a matrix or a data.table, so R only takes a plain data frame. The port takes any 2-D input. Some arguments send R's loops around a cycle forever. A `DN.num2` below -1 does it unless `DN.num1` floors to the same integer. The port raises `ValueError` on these at once, even where R would count down for a long time before the cycle starts. It also raises on a `dn_num1` or `dn_num2` that isn't finite and on an `x` with no rows, where R stops, loops, or returns `Inf` or `NaN`.

### The WCluster statistics and wpca

In the `WCluster` functions `Wpca` becomes `wpca`, `cluster.predict` becomes `cluster_predict` and the `groupSum` argument of `wwcss` becomes `group_sum`. The other names are R's. A weight argument left as `None` means one weight per row, or per column for `wpca`'s `wcol`. R's `wmean`, `wwcss` and `distw` default to `rep(1, length(x))` instead, which has the wrong length. It stops unless the data is square (`wmean`, `wwcss`) or has one column (`distw`). A cluster whose weights sum to zero gets a NaN `wwcss` and NaN in every `distw` entry that touches it, as in R. Its `wmean` row is NaN when every weight is zero. When weights of both signs cancel, the row is infinite instead, with NaN in any column whose weighted sum is zero too, again as in R. `cluster_predict` doesn't pick such a cluster while another is at a finite distance. The other differences are these:

| Item | R | Python |
| --- | --- | --- |
| `wmean` layout | `p` rows by `k` columns | `(k, p)`, like every other center array in the package |
| `distw` with one cluster | stops ("missing value where TRUE/FALSE needed") | `ValueError` naming the cause |
| cluster labels | `wwcss` groups by `factor()` (15 significant digits), `distw` by `sort(unique())`, text by locale collation | whole numbers smaller than 1e15 in size, sorted. Text, bool, fractional and NaN labels raise |
| `cluster.predict` result | 1-based position in sorted label order, not the label | 0-based position |
| `cluster.predict` argument order | `(x, w, cl, newx)`, and a swapped `w` and `cl` runs silently | `(x, cl, newx, *, w=None)` |
| `cluster.predict` for a row where every distance is NaN (NaN centers, another new row that overflows, or centers of infinities once there are two new rows) | `integer(0)` when every row is like that, a list with empty entries when only some are | `ValueError` |
| `cluster.predict` distances after R's identity multiply | `0 * Inf` from one overflowing row turns its whole column NaN, and a lone `Inf` stays | the same, computed without the multiply |
| a vector `newx` in `cluster.predict`, a vector `x` in `getDN.means` | stops | `ValueError` |
| integer `x` with integer weights | `wmean` and `cluster.predict`: an NA center on overflow with a warning. `Wpca`: the NA becomes 0 in its column sum, a wrong center with a warning. `wss`, `wwcss`, `distw`: no integer products | float64 arithmetic, no overflow |
| NaN or infinite weights, text or bool weights | passed through or stopped late in `eigen` | `ValueError` or `TypeError` up front |
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
| `Wkmeans` output | prints each start's WWCSS, and sends "Iteration" lines to `message()` | `logging.debug` on the `datanuggets` logger |
| `Wkmeans` without user centers on at least `k` but fewer than `2k` rows, or on fewer than `k` distinct pairs of a row and its weight | draws starts forever | `ValueError` |
| `Wkmeans` starts that keep missing `k` clusters | keeps drawing, maybe forever | `ValueError` after `MAX_STARTS`, 10,000 starts |
| `num.init` or `max.iterations` below 1 | `1:0` runs the loop twice | `ValueError` |
| `DNcluster.predict` with a matrix `newx` | stops ("newdata should be either a data frame or a datanugget object.") | runs, as `cluster_predict` does |

The `dn_` functions stop where R's do on a datanugget with one variable, where R's `[, 2:2]` drops the table to a vector: `dn_wkmeans` always, `dn_wpca` without `wcol`, and `dn_cluster_predict` with a `DataNuggets` as `newx`. `dn_whclust` runs, as R's does. The `one_variable` case records R's messages there, and what R returns where it runs. `dn_cluster_predict` reads a `DataNuggets` given as `newx` the way R reads a datanugget there. It takes as many columns of `newx`'s table as `dn` has variables, so a wider `newx` gives its first variables, and a narrower one passes its weights and scales as data. Both stop when the table runs out of columns.

### cut_tree

`cut_tree` takes one `k` or one `h` and returns one label per row, where R's `cutree` also takes a vector of them and returns a matrix with a column per cut. Its labels are R's minus one. With both `k` and `h`, `k` wins and `h` is ignored, as in R. A fractional `k` is truncated, as `as.integer` does.

## Plots

`datanuggets.plot` needs the `plot` extra ([installation.md](installation.md)), and [quickstart.md](quickstart.md) has a worked example. Every plot function takes `ax=`, draws on `plt.gca()` without it, and returns the Axes it drew on.

`dendrogram(tree)` draws a `whclust` or `dn_whclust` tree as R's `plot.hclust` does. Leaves sit at 1 to n in the tree's `order` and each merge is a bar at its height. A leaf hangs below its parent's bar by `hang`, 0.1 by default, times the last height minus the first, and a negative `hang` puts every leaf at 0. On the eight recorded trees R draws, and on one of them again with `hang` at -1 and 0.5 and with its heights out of order, every line matches the polylines R wrote into a PDF to within 0.006 points, where R's PDF rounds to 0.005. Leaves are labelled with their 0-based row numbers, or with a list of names, or not at all with `labels=False`. It stops where `plot.hclust` stops, with R's message.

`cut_tree(tree, k=None, h=None)` cuts a tree into `k` clusters or at height `h`, as R's `cutree` does, and returns each row's 0-based cluster, numbered in the order the clusters first appear along the rows. On 13 trees R's `Whclust` and `DN.Whclust` made, it matches R at every `k` from 1 to n, 940 cuts, and at 1,729 values of `h`: every merge height, the midpoints between them, values below and above them all, and both infinities. It stops where R stops, with R's message.

`nuggets(dn)` draws the nugget centers on two columns as the `DN.Whclust` and `DN.Wkmeans` examples do, R's `pch = 16` at `cex = 0.5`, a disc 2.7 points across, the size R's own PDF draws. Without labels it uses the examples' colors, `rgb(0, min(t, 1), 0)` with `t` a nugget's share of the weight over the 80% quantile of the shares, so a lighter green means more rows. Those colors equal the hex strings R gives for the 150 quakes nuggets the `dn_` tests use, for a weight vector with ties, and for the 32 fractional weights of `mtcars$wt`, and the values of `t` behind them equal R's bit for bit. On `mtcars$wt` numpy's pairwise sum would break that, though a plain left-to-right sum wouldn't, so R's long double sum is left to the tests aimed at it. With `labels`, each nugget takes its cluster's color.

`clusters(x, labels, w=None, centers=None)` is the `Whclust` and `Wkmeans` examples' `plot(x, col = labels)` with the centers on top: the rows on two columns in their cluster's color, sized by weight, and each cluster's center as a star with a white edge. By default a point's area is proportional to its weight. `size="log"` gives R's `cex = log(w)`.

`scores(pca)`, `scree(pca)` and `loadings(pca)` draw a `wpca` or `dn_wpca` result: the scores on two components, with raw rows in gray behind them when `rows` is given, each component's share of the variance as a bar, and each variable's loadings as an arrow. R's examples draw no PCA plot. The views flip each component so its largest loading is positive, so they don't change with the sign LAPACK happens to pick. The result itself keeps R's signs.

`style()` is a `with` block in which new figures take a paper style: 7 to 9 point fonts, thin lines, the Okabe-Ito colors, which color-blind readers can tell apart, and TrueType fonts embedded in the PDF. `save(fig, path, width="single")` sizes the figure to one journal column, 3.5 inches wide, or two with `"double"`, 7 inches, or any width in inches, with the height 0.75 times the width unless it's given. It saves at 300 dpi in the format the path's suffix names, such as `.pdf` or `.png`, and stops on a path without one. A PDF's page is exactly that size. A PNG's sides are the inches times 300, rounded down, so the default single-column PNG, 3.5 by 2.625 inches, is 1050 by 787 pixels. A `colors` list must hold at least one matplotlib color, and every entry is checked before anything is drawn. `colors=R_PALETTE` gives R's default `palette()`, and since the port's labels are R's minus one, label `l` then gets R's `col = l + 1`.

### Departures from R's drawings

The plots depart from R's drawings in six ways:

1. Leaf labels and cluster labels are 0-based, like every row number in the port.
2. Points are sized by area by default. R's `cex = log(w)` hides points of weight 1 and gives weights below 1 a negative size. It's still there as `size="log"`.
3. There are no titles by default, and the dendrogram's y range is matplotlib's. R sizes its window from the width of the leaf labels.
4. The PCA views flip each component's sign so its largest loading is positive.
5. `dendrogram` raises `ValueError` on a tree with NaN or infinite heights, which `whclust` gives when a row's weight is 0 or infinite. R's `plot.hclust` warns and writes `-nan(ind)`, or coordinates far off the page, into the PDF, so there's no figure to match.
6. Centers are stars of a fixed size, in their cluster's color with a white edge. R's examples draw them as discs or asterisks at `cex = 5`, which cover the points around them.
