# Plotting: `datanuggets.plot` and `cut_tree`

- Status: designed and approved in brainstorming on 2026-10-10. RON2K approved the layout and the area-sized points, then asked to finish without more review stops ("approve all just finish and keep testing").
- Author: Ronil Basu (Rutgers)
- Upstream: R 4.6.1's `stats::cutree` and `stats:::plot.hclust` (C code `cutree` in `hclust-utils.c`, `C_dend` and `C_dendwindow` in `graphics/src/plot.c`), and the plots in the help examples of `WCluster` 1.3.0 and `datanugget` 1.5.0.
- Branch: `feat/plotting`, stacked on `feat/wcluster-iterative`, since it plots `WhclustResult` and `WpcaResult`.

## 1. Goal

RON2K asked that the package draw graphs, for three uses: what R draws, exploring results, and paper figures. R draws in its help examples, and every one of those plots should take one call on the port's results. The exploring views come from the parity guide's `dn` plot, which matched Cabrera's R side. Paper figures need one consistent style and exact sizes.

### Success criteria

1. `cut_tree` equals R's `cutree` on every recorded case and stops where R stops, with R's message.
2. `dendrogram` draws R's lines. On every recorded tree its segments equal the polylines R's `plot.hclust` drew, read back from R's PDF output, to within the PDF's rounding. It stops where `plot.hclust` stops.
3. `nuggets` without labels colors each nugget with exactly the hex color R's recipe gives.
4. Every plot function takes `ax=` and returns the Axes it drew on. The tests check the drawn data on the artists, not pixels.
5. `save` writes a PDF whose page is the requested size in points and a PNG whose pixel size is the requested size at 300 dpi.
6. `import datanuggets` doesn't import matplotlib. `import datanuggets.plot` without matplotlib raises `ImportError` naming the extra.
7. The suite passes on Python 3.14 and 3.15 on the laptop, the Mac mini and the Actions runner.

### Non-goals

- `optimal.DN`'s elbow and difference plots. They come with `optimal_dn`.
- Interactive plots (plotly) and ggplot2 styling (plotnine).
- Image comparisons with R. R's plots are checked by their coordinates and colors.
- R's vector `k` or `h` in `cutree`. One cut per call.
- R's dendrogram window, which `C_dendwindow` sizes from the label widths in inches. matplotlib picks the y range.

## 2. Scope and API

```python
# core: datanuggets.wcluster and datanuggets
def cut_tree(tree: WhclustResult, k=None, h=None) -> np.ndarray

# datanuggets.plot
def dendrogram(tree, *, labels=None, hang=0.1, ax=None) -> Axes
def clusters(x, labels, *, w=None, centers=None, dims=(0, 1), size="area", colors=None, ax=None) -> Axes
def nuggets(dn, *, labels=None, dims=(0, 1), colors=None, ax=None) -> Axes
def scores(wpca, *, w=None, labels=None, rows=None, dims=(0, 1), colors=None, ax=None) -> Axes
def scree(wpca, *, ax=None) -> Axes
def loadings(wpca, *, names=None, dims=(0, 1), ax=None) -> Axes
def style() -> ContextManager
def save(fig, path, *, width="single", height=None) -> Path
R_PALETTE: tuple[str, ...]
OKABE_ITO: tuple[str, ...]
```

| Module | Holds |
|---|---|
| `wcluster/cutree.py` | `cut_tree` |
| `plot/__init__.py` | the matplotlib check and the public names |
| `plot/tree.py` | `dendrogram` |
| `plot/points.py` | `clusters`, `nuggets`, R's weight colors, the size rules |
| `plot/pca.py` | `scores`, `scree`, `loadings`, the sign rule |
| `plot/style.py` | `style`, `save`, the palettes |

`datanuggets/__init__.py` exports `cut_tree` and never imports `plot`. With `ax=None` each function draws on `plt.gca()`.

## 3. `cut_tree`

R's `cutree(tree, k, h)` is an R wrapper around the C routine `cutree`. The port follows both.

The wrapper's checks, in R's order:

1. Neither `k` nor `h`: "either 'k' or 'h' must be specified".
2. With `k` (R ignores `h` then): `k` is truncated toward zero, as `as.integer` does. A NaN or infinite `k` stops with "missing value where TRUE/FALSE needed", R's message once `as.integer` has made it NA. Outside 1 to n it stops with "elements of 'k' must be between 1 and n".
3. With `h` only: heights with a NaN stop with "missing value where TRUE/FALSE needed" (`is.unsorted` gives NA), and unsorted heights with "the 'height' component of 'tree' is not sorted (increasingly)". Then `k` is n + 1 minus the 1-based position of the first entry of `c(height, Inf)` greater than `h`. So a cut at a merge's own height keeps that merge, and an `h` below every height gives n clusters. `which.max` of all FALSE is 1, so `h = Inf` gives n clusters too. `h = NaN` makes every comparison NA, and R fails later, naming a zero-column result: "length of 'dimnames' [2] not equal to array extent". The port raises that message.

The C routine walks the merges in order. It keeps, for each row, whether it's still alone and the last merge step that took it in. At merge n - k it numbers the clusters in order of first appearance along the rows, a row still alone getting the next number. `k = n` gives each row its own number. The port converts its scipy numbering back to R's (row `a` below n is `-(a + 1)`, cluster `n + s` is `s + 1`), replaces the inner "every row whose step is m" loops with numpy masks, which visit the same rows, and returns R's numbers minus one as `int64`. A self-merge tree runs through the same loop, as it does in R.

## 4. `dendrogram`

R's `plot.hclust` first runs `.validity.hclust`: the lengths of `height` and `order` ("'height' is of wrong length", "'order' is of wrong length"), then the sorted merge entries, in R's numbering, which must be exactly `-n, ..., -1, 1, ..., n - 2`, or it stops with "'merge' matrix has invalid contents". Every self-merge tree stops here (`selfmerge_mid` on its order's length, since its order drops rows), so `dendrogram` raises R's message where `to_linkage` raises its own. A tree with no merges stops in `seq_len(-1)`: "argument must be coercible to non-negative integer". One merge passes the check and stops in `C_dendwindow`: "invalid dendrogram input".

R draws a tree with NaN or infinite heights, which whclust gives for zero weights, with only a warning ("nonfinite axis=2 limits ... corrected now"). The probe on `zero_w1` and `inf_single` shows what it draws: the y axis "corrected" to about 4.6e307 to 1.2e308, every line below or above the page, and NaN coordinates written into the PDF as `-nan(ind)`, which isn't a PDF number. There's nothing to copy, so `dendrogram` raises `ValueError` for such a tree, a departure (section 12).

Then `C_dend` draws, and the port copies it:

- Leaf `i` sits at x = 1 + its position in `order` (R's `order(x$order)`), so x runs from 1 to n.
- The hang is `hang * (height[-1] - height[0])`, the last height minus the first, not the range. A non-finite `hang` stops with "invalid dendrogram input".
- `drawdend` recurses from the last merge, left child, then right child, then the merge itself. A merge child returns its anchor. A leaf returns (its x, the parent's height minus the hang), or (its x, 0) when `hang` is negative. The merge draws the polyline (xl, yl), (xl, h), (xr, h), (xr, yr), and its anchor is ((xl + xr) / 2, h).
- The port collects the n - 1 polylines in that order into one `LineCollection` with gid `"dendrogram"`, in the axes' line color.
- Leaf labels are the 0-based row numbers by default, rotated 90 degrees and hanging below each leaf. `labels=False` draws none and a sequence of n strings replaces them.
- The y label is "Height". The x axis and the top, right and bottom spines are hidden, as R draws only the y axis. There's no title, where R's default is "Cluster Dendrogram".

## 5. `clusters` and `nuggets`

`clusters` is R's `plot(x, cex = log(w), pch = 16, col = k)` from the `Whclust` and `Wkmeans` examples, with `points(centers, ...)` on top:

- It draws `x[:, dims]`, the row with label `l` in `colors[l % len(colors)]`. `colors` defaults to the current matplotlib color cycle. `colors=R_PALETTE` gives R's `palette()` colors, and since the port's labels are R's minus one, label `l` then gets R's `col = l + 1`.
- `size="area"` (the default) makes each point's area proportional to its weight, `s = 6 + 400 * w / max(w)` in points squared, as the parity guide's plot did. Without `w` every point gets `s = 20`. A negative or NaN weight raises `ValueError`.
- `size="log"` is R's rule. R draws `pch = 16` as a circle of radius 0.375 times `cex` times half the character height, which the pdf device makes 1.2 times its 12 points, so 2.7 points at `cex = 1` and `s = (5.4 * log(w))^2`. The review caught the first design's 4.5 points, which is grid's rule, and the `pch16` fixture reads the disc back from R's PDF. A point whose `log(w)` isn't positive gets size 0, as it vanishes in R.
- `centers`, one row per cluster in label order, are drawn above the points as stars in their cluster's color with a white edge. A black edge, as first designed, hid cluster 0's star under the paper style or R's palette, which both start with black.

The examples draw centers as big discs (`pch = 16, cex = 5`, `Wkmeans`) or asterisks (`pch = 8, cex = 5`, `DN.Wkmeans`). The port's stars with a white edge stay visible over points of the same color, black ones included.

`nuggets` draws a `DataNuggets`' centers, `pch = 16, cex = 0.5` as in the `DN.Whclust` and `DN.Wkmeans` examples, so `s = (5.4 * 0.5)^2`:

- With `labels`, nugget `j` is colored like `clusters`.
- Without, it uses the examples' weight colors, `col = sapply(w_trans, function(t) rgb(0, min(t, 1), 0))` with `w_trans = w / sum(w)` and then `w_trans / quantile(w_trans, 0.8)`. `sum` is R's long double sum and `quantile` R's type 7, through `r_sum` and `r_quantile7`, since numpy's `linear` quantile rounds differently. `rgb` maps a fraction to `int(255 * x + 0.5)`. Lighter green means more weight. All-zero weights stop in `quantile` ("missing values and NaN's not allowed if 'na.rm' is FALSE"), and a quantile of 0 with zero weights makes a 0 / 0 that stops in `rgb` ("color intensity nan, not in [0,1]"). The port raises both messages.

Both take two distinct `dims` within the columns, and labels with one entry per point, or raise `ValueError`.

## 6. PCA views

Eigenvectors are defined up to sign. As in the parity guide, component `j` is flipped so its largest loading by absolute value is positive (a zero sign counts as positive), and `rotation` and `x` flip together. This only affects the views. `WpcaResult` keeps R's signs.

- `scores` draws `wpca.x[:, dims]`, sized by `w` with the area rule when given, colored by `labels` when given. `rows`, a raw data matrix, are drawn behind in light gray as `(rows - center) @ rotation[:, dims]`, the same map that gives `x`. At most 50,000 rows are drawn, every `ceil(n / 50000)`-th. The axis labels give each component's share of the variance, "PC1 (61.3%)".
- `scree` draws `100 * lam / sum(lam)` as bars, with `lam = sdev ** 2` and NaN counted as 0 (an eigenvalue rounded just below zero gives a NaN `sdev`).
- `loadings` draws an arrow from the origin to each variable's loadings on `dims`, labelled with `names` or the 0-based column numbers.
- `scores` and `loadings` need at least two components, or raise `ValueError`.

## 7. `style` and `save`

`style()` is a `matplotlib.rc_context` for paper figures:

- font sizes 8 (axis labels), 9 (titles), 7 (ticks and legends)
- lines 0.8 and axes 0.6 points wide
- the Okabe-Ito color cycle, which stays distinguishable for color-blind readers
- no top or right spines
- TrueType fonts embedded in PDF and PostScript (`pdf.fonttype` and `ps.fonttype` 42)
- constrained layout and 300 dpi when saving

`save(fig, path, width="single", height=None)` sets the figure to 3.5 inches wide (`"single"`, one journal column), 7.0 (`"double"`) or a number of inches. `height` defaults to 0.75 times the width. It saves at 300 dpi without a tight bounding box, which would change the size, and returns the path. The format comes from the suffix. It doesn't create folders.

`OKABE_ITO` is `#000000, #E69F00, #56B4E9, #009E73, #F0E442, #0072B2, #D55E00, #CC79A7`. `R_PALETTE` is R 4's default `palette()`, `black, #DF536B, #61D04F, #2297E6, #28E2E5, #CD0BBC, #F5C710, gray62`, given as the hex R's `col2rgb` reads them as, since matplotlib doesn't know R's names such as `gray62`: `#000000, #DF536B, #61D04F, #2297E6, #28E2E5, #CD0BBC, #F5C710, #9E9E9E`.

## 8. Errors

| Input | Raises |
|---|---|
| `import datanuggets.plot` without matplotlib | `ImportError`: "datanuggets.plot needs matplotlib: pip install 'datanuggets[plot]'" |
| `cut_tree` stops | `ValueError` with R's message (section 3) |
| a tree `plot.hclust` stops on: a self-merge, a wrong `height` or `order` length, no merges, one merge | `ValueError` with R's message (section 4) |
| a non-finite `hang`, leaf labels of the wrong length | `ValueError`: "invalid dendrogram input" |
| a tree with NaN or infinite heights in `dendrogram` | `ValueError`, the port's message (departure 5) |
| `dims` not two distinct columns in range | `ValueError` |
| labels, weights or centers of the wrong length or width | `ValueError` |
| a negative, NaN or infinite weight, or all zero, with `size="area"` | `ValueError` |
| weights R's color recipe stops on, in `nuggets` | `ValueError` with R's message (section 5) |
| `size` other than `"area"` or `"log"`, `width` other than the two names or a positive number | `ValueError` |

## 9. Fixtures

`r/fixtures_plot.R`, run on the laptop's R 4.6.1 only, writes `tests/fixtures/plot/`. It reads the trees R's `Whclust` and `DN.Whclust` returned from the `whclust` and `dn_wcluster` groups, recorded on the same R, so there's nothing to rerun, and records:

- `cutree` at every `k` from 1 to n, and at every distinct height, the midpoints between them, below the lowest and above the highest, Inf and -Inf, plus R's stops: no `k` or `h`, `k` of 0, n + 1, NaN, Inf and 3e9, `h` of NaN, and `h` on trees with unsorted or NaN heights. `k = 2.9` is recorded as R's cut.
- `plot.hclust`'s polylines, for every tree with finite heights that R draws. Each is drawn with `plot(h, labels = FALSE, axes = FALSE, ann = FALSE)` into `pdf(file, compress = FALSE)`, which writes every polyline as plain `m`, `l` and `S` operators. The script reads the four-point paths back, maps them to user coordinates with `grconvertX` and `grconvertY`, checks that the bars sit at the heights and the leaves span 1 to n, and records them in drawing order with the points per user unit. One tree is also drawn with `hang = -1` and `hang = 0.5`. It records R's stops for the self-merge trees, a tree with no merges and a non-finite `hang`.
- R's weight colors: `w_trans` and the `rgb()` strings for the recorded `dn_wcluster` nugget weights and for a weight vector with ties, and the recipe's stops for all-zero weights and a quantile of 0.

Fixtures are never regenerated to make a test pass.

## 10. Testing

- `tests/wcluster/test_cutree.py`: every recorded cut and stop, and the numpy version against a statement-for-statement transliteration of the C loop on fuzzed trees, self-merges included.
- `tests/plot/`: matplotlib's Agg backend. `dendrogram`'s segments equal R's to within 0.006 points in each axis, converted to user units (the PDF rounds to 0.005 points). The other plots are checked on their artists: offsets, sizes, face colors and arrow ends. `save` is checked on the PDF MediaBox and the PNG size. A subprocess blocks matplotlib's import and checks that `import datanuggets` works and `import datanuggets.plot` raises the message above.
- Nothing in the slow tier. The tests skip if matplotlib is missing, but the dev group always has it.

## 11. Packaging and docs

- `pyproject.toml`: a `plot = ["matplotlib>=3.9"]` extra, and `matplotlib>=3.9` in the dev group. `uv.lock` regenerated. matplotlib 3.11.2 has wheels for 3.14 and 3.15.
- README: a "Plots" section that ports the `DN.Whclust` example, and the departures below. `docs/architecture.md`: the `plot` package.

## 12. Departures from R

1. Leaf labels and cluster labels are 0-based, like every row number in the port.
2. Points are sized by area by default. R's `cex = log(w)` hides weight-1 points and gives weights below 1 a negative size. It stays available as `size="log"`.
3. No titles by default, and the y range is matplotlib's, not `C_dendwindow`'s.
4. The PCA views flip component signs so the largest loading is positive. R's examples draw no PCA plot.
5. `dendrogram` raises `ValueError` for a tree with NaN or infinite heights. R's `plot.hclust` warns and writes `-nan(ind)` or coordinates far off the page into the device, so there's no figure to match. `Whclust` gives such trees when a row has weight 0 or a weight is infinite.
6. Centers are drawn as stars at a fixed size. R's examples use `pch = 16` at `cex = 5` for `Wkmeans` and `pch = 8` at `cex = 5` for `DN.Wkmeans`, which cover the points around them.
