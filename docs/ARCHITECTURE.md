# Architecture

How the datanuggets repository fits together: what each folder holds, which way the package's imports run, the tests that keep those lines, and where new work goes. [architecture/package.md](architecture/package.md) has the package's own layout.

## What each folder holds

| Folder | Holds |
|---|---|
| `src/datanuggets/` | the Python port of R's datanugget and WCluster packages. It is the only code in the wheel, and [architecture/package.md](architecture/package.md) maps it |
| `docs/` | user guides at the top level ([README.md](README.md) lists them), the package's layout in `docs/architecture/`, the parity pages in `docs/validation/`, and the draft math reference in `docs/math/` |
| `tests/` | the package's pytest suite, its recorded fixtures, and the package's two boundary tests |
| `scripts/` | maintainer tools that write ten of the parity pages in `docs/validation/` and check a built sdist and wheel |
| `benchmarks/` | timings and comparisons |
| `examples/` | the runnable example `first_analysis.py` |
| `.github/workflows/` | `tests.yml`, `matrix.yml`, `package.yml` and `release.yml` for the package ([development.md](development.md#what-ci-runs) describes them) |

The wheel holds the package alone. The sdist adds the package's tests, the user guides and the runnable example, as [architecture/package.md](architecture/package.md#what-ships) lists.

## Which way the dependencies run

`src/datanuggets` imports nothing from `tests/`, `scripts/` or `benchmarks/`. Inside it, only the `plot` package reaches matplotlib. [package.md](architecture/package.md#dependency-direction) gives the order its modules import each other in.

## The boundary tests

Two tests at the top of `tests/` keep the package's boundaries. Both run with the rest of the suite, and each can also be run alone from the repository root.

| Test | Fails when | Run it alone |
|---|---|---|
| `tests/test_import_graph.py` | the package has an import cycle, or a module outside `datanuggets.plot` reaches matplotlib or `datanuggets.plot` | `uv run pytest tests/test_import_graph.py` |
| `tests/test_public_paths.py` | a name `docs/api.md` documents, a name in an `__all__`, or a private name listed in the test's `OUTSIDE` table stops importing from its path | `uv run pytest tests/test_public_paths.py` |

`test_import_graph.py` reads the imports from the source, including those inside functions. `test_public_paths.py` imports each module and checks its names. `uv run pytest -n 2` runs every test not marked `slow`. [development.md](development.md) covers the package's setup and lint.

## Adding a feature

A feature goes into the package with its tests, then into the docs:

- **Package.** A function: [package.md](architecture/package.md#where-new-code-goes).
- **Docs.** A user guide goes in `docs/`. When it should ship in the sdist, it is also added to the sdist list in `pyproject.toml` and to `scripts/check_dist.py`. A change to the package's layout updates [package.md](architecture/package.md). This page changes only when a folder or a boundary does.
