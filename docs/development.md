# Development

Setting up, running the tests and the lint, checking a build, and rebuilding the parity tables. [architecture.md](architecture.md) shows where each part of the code, tests and fixtures lives, which way the imports run, and where new work goes.

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
| `release.yml` | by hand for TestPyPI, a pushed `v*` tag for PyPI | the three above, then the upload, as [releasing.md](releasing.md) describes |

The test steps run pytest with `-q -p no:cacheprovider -n 2` and `-m "not slow"` or `-m slow`. These workflows are on `main`; use their triggers in the table above.

## Lint

```sh
uv run ruff check
uv run ruff format --check
```

No workflow runs ruff, so run both before pushing. `uv run ruff format` rewrites the files the check flags. The settings in `pyproject.toml` set the line length to 100, sort a helper module beside its tests, such as `tests/clustering/kmns_reference.py`, with the first-party imports, and leave out `docs/`, whose snippets are illustrative, and `benchmarks/flow/cabrera_reference.py` and `benchmarks/flow/run_cabrera.py`, which keep Prof. Cabrera's code as written.

## Check a build

To check a build the way CI does, build into an empty folder and run the checker on it:

```sh
uv build --out-dir tmp/dist
uv run python scripts/check_dist.py tmp/dist
uvx --from twine==7.0.0 twine check --strict tmp/dist/*
```

The checker opens the sdist and the wheel. It fails on a file outside the allowlist in `pyproject.toml`, a cache file, a home-folder path, a name from the development setup, or versions that disagree, so leftovers from an earlier build in the same folder can fail it. `--expect-version` makes it fail unless the build has that version, which the release uses on a tag. The `package` workflow also installs the wheel alone in a new environment with the `frame`, `progress`, `plot`, `flow`, and `cli` extras, deletes `src/` from the unpacked sdist, and runs the sdist's own default tests against the installed wheel, including the CSV and FCS CLI integration tests. [releasing.md](releasing.md) covers publishing.

## Parity tables

The tables in [validation/](validation/) compare the port with R on every fixture, one file per function. Rebuild them with:

```sh
uv run python scripts/parity_report.py
```

It overwrites the ten files in `docs/validation/`. `git diff docs/validation` then shows what a change did to them.

## Fixtures

The reference fixtures in `tests/fixtures/` were generated by running the R packages. Each case's manifest records the R version, package versions, BLAS and long double width. They stay in Git and in the source archive, so the Python tests run without R. [reproducibility.md](reproducibility.md) describes the recorded cases and benchmark results.

The R generators, R comparison scripts and their optional benchmark workflow are local maintainer tools, excluded from Git and release archives. A fresh clone can verify the recorded cases but cannot regenerate the R baselines on its own. Do not replace reference fixtures just to make a failing test pass; investigate the difference first.

## Repository contents

Keep the Python source, tests, reference fixtures, examples and public documentation tracked. Internal specs, plans and reviews belong in the ignored `docs/specs/`, `docs/plans/` and `docs/reviews/` folders. R source files (`*.R` and `*.r`), the local `r/` folder and `.github/workflows/bench.yml` are also ignored. The R-derived numerical code is Python and remains part of the package, with its license and attribution.

## Project history

The [merged pull requests](https://github.com/ron2k1/datanuggets/pulls?q=is%3Apr+is%3Amerged) record implementation history. [compatibility.md](compatibility.md) and [limitations.md](limitations.md) describe the current behavior and unresolved issues.
