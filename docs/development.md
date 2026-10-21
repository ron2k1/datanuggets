# Development

Setting up, running the tests and the lint, checking a build, and rebuilding the parity tables and the R fixtures. [architecture.md](architecture.md) shows where each part of the code, tests and fixtures lives, which way the imports run, and where new work goes.

## Setup

```sh
git clone --branch main https://github.com/ron2k1/datanuggets.git
cd datanuggets
uv sync --locked --python 3.14
```

[Installation](installation.md#editable-install-for-development) explains editable installs. This creates `.venv` with Python 3.14 and the locked development dependencies. Every command below runs from the repository root. Ordinary development needs only one Python version; testing 3.15 is a separate compatibility check of the same code.

## Tests

```sh
uv run pytest
uv run pytest -m slow
```

The first runs the default tier. `pyproject.toml` adds `-m 'not slow'` to every run, so the default leaves out the tests marked `slow`. The second runs only those: the long fuzz of the compiled long double kernels, 10 million cases per operation, and for `whclust` and `wkmeans` the plain-Python references on the large recorded cases and a larger fuzz of the compiled kernels against those references.

The tests read the R fixtures in `tests/fixtures/`, so they don't need R. Two tests need `os.fork` and skip on Windows.

pytest-xdist runs the tests in parallel:

```sh
uv run pytest -n auto
```

`-n auto` starts one worker per logical CPU, and each worker loads numba, scipy and its own copy of the fixtures it reads. If workers fail with `MemoryError`, give a smaller number, such as the `-n 2` that CI uses.

### Test Python 3.14 and 3.15

Keep the main `.venv` on 3.14 and create a second environment under the ignored `tmp/` folder. These are two installations of the same checkout, not two branches or package editions. Do not reuse one environment for both interpreters.

**Windows (PowerShell):** restore the environment setting after creating the second environment so later `uv` commands keep using their original target.

```powershell
$previousEnvironment = $env:UV_PROJECT_ENVIRONMENT
try {
    $env:UV_PROJECT_ENVIRONMENT = "tmp/venv315"
    uv sync --locked --python 3.15
} finally {
    if ($null -eq $previousEnvironment) {
        Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    } else {
        $env:UV_PROJECT_ENVIRONMENT = $previousEnvironment
    }
}
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pytest
.\tmp\venv315\Scripts\python.exe --version
.\tmp\venv315\Scripts\python.exe -m pytest
```

**macOS or Linux:** the environment setting below applies to that command only.

```sh
UV_PROJECT_ENVIRONMENT=tmp/venv315 uv sync --locked --python 3.15
.venv/bin/python --version
.venv/bin/python -m pytest
tmp/venv315/bin/python --version
tmp/venv315/bin/python -m pytest
```

Numba compiles every kernel again for a new Python version, so the first run there is slower. The Python version belongs to the environment, while `datanuggets.__version__` is the library version and is the same in both.

### What CI runs

| Workflow | When | What it runs |
| --- | --- | --- |
| `tests.yml` | every pull request, and inside a release | `uv sync --locked`, then the default tier on Linux for Python 3.12 to 3.15, and the slow tier on 3.14 and 3.15 |
| `matrix.yml` | a pull request gets the `matrix` label, and inside a release | the default tier on Windows and macOS for Python 3.12 to 3.15 |
| `package.yml` | every pull request, and inside a release | the build check below, then the sdist's default tests against the installed wheel |
| `bench.yml` | a pull request gets the `bench` label | the `WCluster` benchmark in R 4.6.1 and in the port on x86 Linux, about an hour |
| `release.yml` | by hand for TestPyPI, a pushed `v*` tag for PyPI | the three above, then the upload, as [releasing.md](releasing.md) describes |

The test steps run pytest with `-q -p no:cacheprovider -n 2` and `-m "not slow"` or `-m slow`. These workflows are on `main`; use their triggers in the table above.

## Lint

```sh
uv run ruff check
uv run ruff format --check
```

No workflow runs ruff, so run both before pushing. `uv run ruff format` rewrites the files the check flags. The settings in `pyproject.toml` set the line length to 100 and leave out `docs/`, whose snippets are illustrative, and `benchmarks/flow/cabrera_reference.py` and `benchmarks/flow/run_cabrera.py`, which keep Prof. Cabrera's code as written.

## Check a build

To check a build the way CI does, build into an empty folder and run the checker on it:

```sh
uv build --out-dir tmp/dist
uv run python scripts/check_dist.py tmp/dist
uvx --from twine==7.0.0 twine check --strict tmp/dist/*
```

The checker opens the sdist and the wheel. It fails on a file outside the allowlist in `pyproject.toml`, a cache file, a home-folder path, a name from the development setup, or versions that disagree, so leftovers from an earlier build in the same folder can fail it. `--expect-version` makes it fail unless the build has that version, which the release uses on a tag. The `package` workflow also installs the wheel alone in a new environment with the `frame`, `progress`, `plot` and `flow` extras, deletes `src/` from the unpacked sdist, and runs the sdist's own default tests against the installed wheel, including the FCS parser and CLI integration tests. [releasing.md](releasing.md) covers publishing.

## Parity tables

The tables in [validation/](validation/) compare the port with R on every fixture, one file per function. Rebuild them with:

```sh
uv run python scripts/parity_report.py
```

It overwrites the ten files in `docs/validation/`. `git diff docs/validation` then shows what a change did to them.

## Fixtures

The fixtures in `tests/fixtures/` were generated by running the R packages, and each case's manifest records the R version, the package versions, the BLAS and the long double width. Running the tests doesn't need R. Regenerating the fixtures needs R 4.6.1 with the pinned packages, and they are recorded on one machine only, the laptop with the pinned R. The generators are in `r/`, with the setup and the order to run them in [r/README.md](../r/README.md). [reproducibility.md](reproducibility.md) lists what each group of fixtures records, and has the benchmark scripts and their recorded results.

## What was done

The work landed in the order below. Most steps started from a written plan, linked in the table. Steps 4, 5, 10, 11, 12, 13 and 14 also had an independent review, and its findings were fixed or recorded. `RDraws` grew out of a spike and has no plan document, and the architecture document was written after the moves it describes.

| Step | Record | What it added |
| --- | --- | --- |
| 1 | PR #1, [plan](plans/2026-09-29-feature1-create-dn.md) | `create_dn` and `create_dn_centers`. Rfast's distances are replayed with Armadillo's two running sums over even and odd columns, and R's `colMeans` and `rowMeans` in their 80-bit long double |
| 2 | PR #2, [plan](plans/2026-09-29-feature2-refine-dn.md) | `refine_dn`, with a statement-by-statement port of the Hartigan-Wong `kmns` Fortran behind `stats::kmeans`, and R's `cov`, `mean`, `median`, `mad` and `quantile` with long double sums |
| 3 | PR #3, [plan](plans/2026-09-29-feature3-dn-helpers.md) | `get_dn_means`, and `arithmetic_complexity`, which repeats `AC`'s double arithmetic step by step |
| 4 | [plan](plans/2026-09-30-speed-compiled-long-double.md), [review](reviews/2026-09-30-speed-fixpass-review.md) | The long double replay moved from Python integers into numba kernels that give the same bits. `refine_dn` gets every nugget's center and covariance from one compiled call, threaded when `n_jobs` isn't 1 |
| 5 | PR #4, [plan](plans/2026-09-30-feature4-wcluster-leaves.md), [review](reviews/2026-09-30-wcluster-leaves-review.md) | The six `WCluster` functions, with the reference BLAS order of R's `%*%`, `weighted.mean` and `sum` in long double, and a tolerance rule for the output of `eigen()` |
| 6 | PR #5 | `RDraws`: R's Mersenne-Twister seeding and `sample.int`, including the hash sampler R uses when it draws at most half of more than 1e7 values |
| 7 | [plan](plans/2026-09-30-r-compat.md) | A decision record for every known difference from R, backed by 46 new recorded cases. The first four fixes: R's -NaN for an empty nugget, logical data read as 0 and 1, a vector `x` read as one column, and R's NA bits in `distw` |
| 8 | [plan](plans/2026-09-30-eigen-reference-solver.md) | A measured spike with no code shipped: R's own LAPACK reproduces R's eigen output on every input tried, where scipy's doesn't. It sets out the options for a reference solver |
| 9 | [architecture](architecture.md) | The package split into `nuggets/`, `wcluster/`, `_compat/`, `_random/` and `_clustering/`, with the tests in matching folders. The five old public module paths and the private `datanuggets._draws` still import. The other moved private modules are reachable only at their new paths |
| 10 | PR #7, [plan](plans/2026-09-30-speed-next-targets.md), [review](reviews/2026-10-09-speed-next-review.md) | Rfast's distances, `create.DNcenters`' deletion rounds and the Hartigan-Wong port compiled with numba, with the same bits. The deletion rounds work on one distance matrix instead of copying it every round. numba 0.68 in the lock, so the package runs on Python 3.15 |
| 11 | PR #8, [plan](plans/2026-10-09-wcluster-iterative-port.md), [review](reviews/2026-10-10-wcluster-iterative-review.md) | `Wkmeans`, `Whclust`, `DN.Wkmeans`, `DN.Whclust`, `DN.Wpca` and `DNcluster.predict`, with numba kernels that give the Python references' bits, R's quirks where it has them, and R's message in R's order where it stops. `Draws` gains `sample_replace`, so a hand-written `Draws` needs it too |
| 12 | PR #9, [spec](specs/2026-10-10-plotting-design.md), [plan](plans/2026-10-10-plotting.md), [review](reviews/2026-10-10-plotting-review.md) | `cut_tree` and the `plot` package. R recorded `cutree`'s cuts and stops, `plot.hclust`'s lines read back from its PDF, and the nugget colors of the help examples |
| 13 | PR #10, [spec](specs/2026-10-12-flow-gate-design.md), [plan](plans/2026-10-12-flow-gate.md), [review](reviews/2026-10-12-flow-gate-review.md) | `datanuggets.flow`, Prof. Cabrera's lymphocyte and macrophage gate finder ported from its Python version, with the distance disc centred on the peak, and `plot.gates` for the reference code's figure. The reference script, word for word, recorded its pieces on three files of a public PBMC record, and `compare.py` and `compare.ps1` check ours against the reference piece by piece on any FCS file or folder |
| 14 | PR #11, [spec](specs/2026-10-19-packaging-design.md), [plan](plans/2026-10-19-packaging.md), [review](reviews/2026-10-19-packaging-review.md) | The sdist built from an allowlist, and `scripts/check_dist.py`, which reads the built sdist and wheel and fails on a file off the list, a cache file, a home-folder path, a name from the development setup, or versions that disagree. `package.yml` runs the sdist's tests on the installed wheel, the Linux tests cover 3.12 to 3.15, `matrix.yml` adds Windows and macOS on a label, and `release.yml` publishes through Trusted Publishing. The version is written once, in `pyproject.toml` |
| 15 | PR #12 | `datanuggets-flow` and the `flow` extra: each FCS file gated, every file's numbered populations shown in one figure, the ones to keep picked file by file, and the kept cells clustered with data nuggets per file and weighted Ward clustering of all the files' nuggets together. `--disc his`, the default, gates with the `find_area` of Prof. Cabrera's `cfc.py`, so the populations are the ones `cfc.py`'s figures show |

PRs #4 and #5 were opened against the speed branch and closed there. Steps 4 to 9 reached `main` together through PR #6, and step 10 through PR #7. PRs #8 to #11 and step 15 (PR #12) are now merged into `main`.
