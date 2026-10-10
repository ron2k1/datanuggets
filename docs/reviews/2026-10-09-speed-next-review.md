# Speed next targets review, 2026-10-09

The range under review is e3343b6..1102c31 on `feat/speed-next`: the [next targets plan](../plans/2026-09-30-speed-next-targets.md) built at the package layout's paths, plus the tiled mirror in `dist` and the numba 0.68 lock for Python 3.15. Its findings are fixed in 9aa567f..0d332df.

One reviewer ran: Opus, read-only, on the laptop, Windows with Python 3.14.3. It didn't run the suite or the benchmarks, since the laptop was timing Covertype at the time. It ran three short probes with its own numba cache. Its verdict was approve with minors, with no blockers.

## Findings

| Finding | Severity | Fixed in | Evidence it was real |
|---|---|---|---|
| zero columns, or zero centers in `nearest_index`, return uninitialized memory | minor | 9aa567f | see below |
| the profiler makes one wrapper per reference, not per kernel module | nit | 905573c | cProfile's `snapshot_stats` keys entries by file, line and name, and overwrites a repeated key |
| the lock raises the Linux glibc floor for 3.12 to 3.14 | nit | not changed | see the ruling below |
| the mirror test relies on what `np.empty` held | nit | 9aa567f | a skipped entry would read as leftover memory, which almost never matches by chance, but could |
| the plan's tech stack line still says numba 0.67 | nit | 0d332df | line 13 of the plan |

### Unfilled output

With zero columns the feature loop in `_sums` never runs, so nothing writes `out`. `cross_distances(np.zeros((3, 0)), np.zeros((2, 0)), m)` returned values like `1.34e154` and NaN for both metrics, and `dist_matrix` did the same. With zero centers, `nearest` reads `row[0]` of an empty array, which numba doesn't bounds-check, and `nearest_index(np.ones((3, 2)), np.ones((0, 2)), "euclidean")` returned `[0 0 0]`. The numpy code these replaced raised `TypeError` or `ValueError`, or returned `None`.

No public call reaches either: `as_matrix` rejects zero columns, stage 2 always has a center, and `kmeans` always asks for two. The wrappers raise `ValueError` now anyway, like the mismatched-column check. `test_distances_reject_inputs_the_kernels_cant_fill` failed before the fix with "DID NOT RAISE".

The mirror test now runs the `dist` kernel on a NaN-filled output. Cutting the mirror's rows one short fails it at all six sizes, and the restored kernel passes.

### Ruling on the glibc floor

numba 0.68.0 and llvmlite 0.50.0 ship `manylinux_2_27` and `_2_28` wheels, where 0.67.0 and 0.49.0 shipped `manylinux2014`, glibc 2.17. So `uv sync` from the lock on a glibc older than 2.27 builds llvmlite from source. The reviewer suggested a single `numba>=0.68` to match what the lock tests.

Not taken. pyproject keeps `numba>=0.67` below Python 3.15 so that a pip install on an old glibc can still take 0.67's wheels. Only the lock, which uv resolves to one version for every Python, moves to 0.68. The cost is that no locked environment tests 0.67 any more.

## Checked and found correct

The reviewer traced each of these against the code it replaces:

- Every kernel does its floating-point operations in the Python reference's order: Armadillo's even and odd sums, left to right for dista's Euclidean kernel past 32 features, `b - a`, no fused multiply-add. `dist` uses the even and odd sums at every width, as R's `Dist` fixture at 40 columns shows.
- The tiled mirror writes every entry above the diagonal exactly once and none on or below it, for sizes under 64, multiples of 64 and a partial last tile.
- `nearest` flags a tie exactly when the old `count_nonzero(d == mins) > 1` did, and the caller settles it with the same `gnu_sort_first`.
- Searching a row again only when its nearest neighbour was deleted is exact, including when a row's nearest neighbour is itself because the diagonal rounds back to the matrix maximum near 2^60. numba's mergesort argsort is stable. The four-lane minimum is a first-index argmin at every length. On degenerate inputs the kernel matched `_select_centers_py` with no mismatches.
- `kmnsjit` matches `_HW` statement for statement, with the same 1-based row numbers in LIVE and NCP, the same quick-transfer branches and the same ITER and IFAULT on every exit.
- No kernel writes shared state, so stage 1's threads can run them at once.
- The profiler's swap finds every kernel reference, including one in `missing` the old hard-coded list missed, and restores them in `finally`.
- The memory test sees numba's allocations: tracemalloc counted a 60 MB probe allocation inside a kernel.

## Test runs

At d8a3a68, after the fixes and with numba 0.68.0 everywhere:

- Laptop, Windows: 2463 passed and 2 skipped (no `os.fork` on Windows) on Python 3.14.3 and on 3.15.0. The slow fuzz, 10 million cases for each of the seven long double operations, passed on both.
- Mac mini, arm64: 2465 passed on 3.14.3 and on 3.15.0.
- Covertype, pinned to the laptop's performance cores, at 0d332df, whose source and benchmarks match d8a3a68 apart from the new `pin.py`: every output matched R's at 10,000 rows, 100,000 rows and full size, on both Pythons. Both stopping-rule profiles pass at 100,000 rows and at full size.
