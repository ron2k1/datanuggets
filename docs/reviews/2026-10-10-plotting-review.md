# Plotting review, 2026-10-10

The range under review is the [plotting plan](../plans/2026-10-10-plotting.md) on `feat/plotting`, 53aed29..a51ec22: `cut_tree` and the `plot` package (`dendrogram`, `nuggets`, `clusters`, `scores`, `scree`, `loadings`, `style`, `save`), against the [design](../specs/2026-10-10-plotting-design.md). Its findings are fixed in 52049eb..759dfb4 and the docs commit that adds this record.

One reviewer ran: Opus, read-only, on the laptop, Windows with Python 3.14.3. It ran the plot and `cut_tree` tests (126 passed), checked `uv lock --check`, ran the README's Plots example with warnings as errors, and tried mutants in memory without editing the source. It didn't run R. Its verdict was approve with fixes: no blockers, one major finding it marked suspected, six minors and two nits. The major one was real and changed what the port draws. The rest are about error paths, tests that missed a mutant and claims the docs overstated.

## Findings

| Finding | Severity | Fixed in | Evidence it was real |
|---|---|---|---|
| 1. the `pch = 16` disc is too wide | major, suspected | 52049eb, ab69321 | see below |
| 2. `scores` raises after it has drawn the gray rows | minor | 9a2366c | `scores(pc, rows=x0, colors=["notacolor"])` left the rows on the axes |
| 3. no test checks where a leaf label sits | minor | 01c1aac | labelling the leaves in drawing order passed every test |
| 4. two hang mutants survive | minor | 52049eb, f3135fa | see below |
| 5. the docs say `mtcars$wt` tells R's long double sum from a plain one | minor | 52049eb, docs commit | a left-to-right double sum passed all three color cases |
| 6. one label for many leaves doesn't stop as R does | minor | a916866 | `labels="abc"` drew a, b and c, `labels=True` raised `TypeError` |
| 7. `save` returns a path it didn't write when the path has no suffix | minor | 47de2bf | matplotlib wrote `nosuffix.png` and `save` returned `nosuffix` |
| 8. an empty palette raises `ZeroDivisionError` | minor | 9a2366c | `clusters(..., colors=[])` |
| 9. a single-column PNG can't be its exact size | nit | docs commit | see below |
| 10. a NaN merge entry gets the wrong message | nit, suspected | 06c04d4 | see below |

### The pch 16 disc

The port drew R's `pch = 16` with a 9 point diameter, from grid's rule of a radius 0.375 times the font size. The reviewer read base graphics' `GSymbol` instead, which scales the symbol by half the device's character height, 1.2 times the 12 point font, so a radius of 0.375 times 7.2, or 2.7 points. It couldn't run R and called the finding suspected.

`fixtures_plot.R` now settles it with R itself. It draws `plot(0, pch = 16)` into an uncompressed PDF at `cex` 1 and 0.5 and reads the disc's extent back from the four Bezier curves R writes, and the `pch16` case records a radius of 2.7 and 1.35 points. So the port's nuggets, at `cex = 0.5`, had 2.78 times R's area, and `size="log"` was off by the same factor. `PCH16` is now 5.4, and the tests take the sizes from the recorded disc rather than from a constant, so a wrong rule fails them. Spec section 5 has the new numbers.

### Hang

`dendrogram` hangs a leaf by `hang` times the last height minus the first, and puts every leaf at 0 once that's negative, as R's `C_dend` does. But every recorded tree had sorted heights, where the last minus the first is the range and its sign is `hang`'s. So two mutants passed: the range in place of the difference, and the sign test on the raw `hang`. `plot.hclust` doesn't need sorted heights, so the new `hang_heights` case has R draw `grid_dup`'s merges twice, once with the heights reversed and once with the last set equal to the first. Its lines now fail both mutants.

### The mtcars weights

The reviewer swapped the long double sum for numpy's pairwise `np.sum` and for Python's left-to-right `sum` in memory. Numpy's broke `mtcars$wt`'s `t` in the last bit. The left-to-right sum passed all three color cases. So the case rules out a pairwise sum but doesn't test the long double one, which other tests cover. The comment in `fixtures_plot.R` and the README now say so.

### PNG size

matplotlib's Agg truncates the pixel count: a side of 788.97 pixels comes out 788. So a PNG is exact only when its inches times 300 is a whole number, and the default single-column figure, 3.5 by 2.625 inches, is 1050 by 787 pixels. The PDF's page is exact at every size. Success criterion 5 in the spec and the README's `save` paragraph now say so. The size itself isn't changed. Rounding the inches so the pixels come out whole would change the PDF's page, which is the size a journal reads.

### Merge entries

The reviewer read `.validity.hclust` as stopping on NaN with "missing value where TRUE/FALSE needed". R 4.6.1's source is `any(as.integer(merge) != merge)`. `as.integer` gives NA for NaN, an infinity and anything at or beyond 2^31, so `any()` is TRUE once an entry it can hold is fractional, and NA otherwise. A probe of ten merges by hand confirmed the order: a fraction wins over a NaN, an infinity or 3e9, and 2^31 - 1 reaches the contents check. The port compared its own merge with its truncation, so a NaN got the integer message, an infinity was cast to garbage, and 2^31 fell through to "invalid contents". It now reads R's merge as doubles and stops in R's order, and nine of the probed merges are in the tests. The tenth, R's NA, has no numpy double apart from NaN.

## Found while fixing

- Python's `round` in place of R's `int(255 * v + 0.5)` in `rgb()` passed every test, which the reviewer listed as harmless. For `v = (2j + 1) / 510` with `j` even, `255 v` is exactly `j + 0.5`, and R gives `j + 1` where `round` gives `j`. 759dfb4 pins four of them, checked by hand.
- `colors="red"` was split into the colors r, e and d and stopped on `'e'`. It now raises `TypeError`, as the package does for other wrong-type arguments.

## Checked and found correct

The reviewer traced each of these:

- `cut_tree` checks in R's order: no merges, neither `k` nor `h`, `as.integer`'s NA at 2^31, the range. On the `h` path it checks NaN heights, unsorted heights, `h = NaN`, `which.max`'s first-maximum rule and `Inf` giving n. The C loop matches R's `hclust-utils.c`, self-merges, the `k = n` branch and first-appearance numbering included, and the fuzz test against the transliteration reaches self-merges and reused clusters.
- `dendrogram` follows `.validity.hclust`'s order, `seq_len(-1)` for no merges, `C_dendwindow`'s `n < 2` and its `hang` and label checks. A leaf's x is its rank in `order`, by a stable sort, and the polylines come in R's postorder with midpoint anchors. The largest error against R on the eight trees is 0.005 points, so the 0.006 tolerance is tight. The 3000-deep chain runs on an explicit stack.
- The weight colors match R on the 150 nuggets, the ties and `mtcars$wt`. Mutants for `np.quantile`, `np.sum` and a truncating `rgb` are caught.
- Per-point color lists keep scatter from mapping one RGB tuple through viridis at 3 or 4 points. `style()` restores rcParams, `save` forces the standard bounding box, and the PDF's page is exact with rasterized rows. A matplotlib `UserWarning` fails a test in setup, call and teardown.
- `clusters`, `nuggets`, `dendrogram` and `loadings` check their input before drawing. The rows' projection `(rows - center) @ rotation` equals `wpca`'s `x0 @ rotation`, unscaled even with `corr`.
- The README example prints `[107  58  35]`, writes a 504 by 396 point page, embeds TrueType fonts with no Type 3 ones, and calls `save` inside `with style()`.
- The README's counts added up from the fixtures before the new cases: 542 cases in 23 groups, 13 trees, 940 cuts at `k`, 1,729 values of `h` and 8 drawn trees.
- The `plot` extra and matplotlib in the dev group are in `uv.lock`, `import datanuggets` doesn't load matplotlib, and the `ImportError` names the extra.

## Test runs

At 759dfb4, after the fixes, with numpy 2.5.3, matplotlib 3.11.2 and numba 0.68.0:

- Laptop, Windows: 2817 passed and 2 skipped (no `os.fork` on Windows) on Python 3.14.3 and on 3.15.0. ruff is clean, and pyright finds no errors in `datanuggets.plot`.
- Mac mini, arm64: 2819 passed on 3.14 and on 3.15.
- Mutants: each mutant the reviewer described fails the tests after its fix. So do the ones written for the fixes: five of the merge check, three of the palette check, two of `save`'s suffix check, and Python's `round` in `rgb()`.
