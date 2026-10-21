# Installation

Use Python 3.12 or newer. The package depends on NumPy, SciPy, Joblib, and Numba; pip installs them automatically. Your Python version and platform also need to be supported by those dependencies.

The repository currently requires GitHub access, and the package is not on PyPI. `datanugget`, without the final **s**, is an unrelated Python package.

## Install from GitHub

Install Git first, and make sure your Git credentials can access the repository. These docs describe the `feat/flow-pick` development branch:

```sh
python -m pip install "datanuggets[frame,plot] @ git+https://github.com/ron2k1/datanuggets.git@feat/flow-pick"
python -c "import datanuggets; print(datanuggets.__version__)"
```

The current version is `0.1.0.dev0`. The branch can change, so record its commit when reproducing an analysis. Omitting the branch suffix installs the repository's default branch, which may have a different set of features.

## Extras

Choose the optional features you need:

| Extra | Enables | Adds |
| --- | --- | --- |
| `frame` | `DataNuggets.to_frame()` | pandas |
| `progress` | Progress bars during creation and refinement | tqdm |
| `plot` | `datanuggets.plot` | matplotlib |
| `flow` | The `datanuggets-flow` command | flowio, matplotlib, pandas |

Combine extras inside the brackets. For the flow command:

```sh
python -m pip install "datanuggets[flow] @ git+https://github.com/ron2k1/datanuggets.git@feat/flow-pick"
datanuggets-flow --help
```

The array-based gate-finding functions in `datanuggets.flow` need only the core dependencies. Reading FCS files requires flowio. Read the [CLI guide](cli.md) before processing files, including its warning about reusing output directories.

## From a local checkout

```sh
git clone --branch feat/flow-pick https://github.com/ron2k1/datanuggets.git
cd datanuggets
python -m pip install ".[frame,plot]"
```

For an editable install, use `python -m pip install -e ".[frame,plot]"`. Edits under `src/` then take effect without reinstalling.

Contributors who use uv can instead run `uv sync --locked` from the checkout. This installs the locked development environment, including tests and optional dependencies. See the [contributor guide](development.md) for its toolchain and checks.

## From a built wheel

With uv installed, build from the checkout:

```sh
uv build
python -m pip install "dist/datanuggets-0.1.0.dev0-py3-none-any.whl[plot]"
```

The wheel contains Python code and has no platform-specific extension of its own. NumPy, SciPy, and Numba do have compiled components, so the wheel's `any` tag does not guarantee support on every platform. Build and installation checks are described in [development](development.md#check-a-build).

## The first call compiles

Numba compiles numerical kernels the first time they run. Initial calls can therefore take longer than later ones. It caches compiled code separately for each Python version; an unwritable cache can cause recompilation in new processes. If necessary, set `NUMBA_CACHE_DIR` to a writable folder.

Continue with the [walkthrough](quickstart.md) or the [function reference](api.md).
