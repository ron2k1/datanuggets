# Installation

The first alpha candidate is `0.1.0a1`; it has not been published to PyPI. Install a supplied wheel or use GitHub `main` for the current source. Publication remains [on hold](../THIRD_PARTY.md#publication-hold-gnu-sorting-helpers). `datanugget`, without the final **s**, is an unrelated Python package.

## MacBook: install once from a wheel

This installs both the Python library and the CSV command. You need Python 3.14 and the current `datanuggets-0.1.0a1-py3-none-any.whl` file from the maintainer. You do **not** need Git or uv, or a copy of the full repository. If Python is not installed, install Python 3.14 for macOS from [python.org](https://www.python.org/downloads/macos/), then open Terminal.

Save the wheel in Downloads. Run these commands once; pip downloads the required dependencies, so this step needs an internet connection:

```sh
python3.14 -m venv "$HOME/.venvs/datanuggets"
"$HOME/.venvs/datanuggets/bin/python" -m pip install --force-reinstall "$HOME/Downloads/datanuggets-0.1.0a1-py3-none-any.whl[cli]"
"$HOME/.venvs/datanuggets/bin/python" -m datanuggets --help
```

If you installed an earlier alpha wheel, rerun the pip command with the newly supplied wheel. Unpublished builds still use `0.1.0a1`, and an older build may not contain the CSV command. You do not need to recreate the environment or reinstall before each analysis.

For each new Terminal session, activate that environment and run your analysis:

```sh
source "$HOME/.venvs/datanuggets/bin/activate"
datanuggets analyze "$HOME/Documents/study.csv" \
  --columns measurement_a measurement_b \
  --transform standardize \
  --nuggets 200 \
  --analysis kmeans pca \
  --clusters 3
```

Replace the file path and column names with yours. Results go under `outputs/` in Terminal's current working folder. Use `cd "$HOME/Documents"` first if you want them there. Your dataset can stay anywhere you can read it; it does not belong inside the installed package. The [CSV guide](cli.md#analyze-a-csv) explains the transformations and results.

Activation is optional: use `"$HOME/.venvs/datanuggets/bin/datanuggets" analyze ...` directly instead. For Python scripts, run `"$HOME/.venvs/datanuggets/bin/python" your_script.py`, or select that interpreter in your editor. Then `from datanuggets import create_dn` uses this installation.

## Choose a Python version

There is **one datanuggets codebase**. Python 3.14 and 3.15 are two versions of the interpreter that can run it; there is no separate 3.14 edition or 3.15 edition of this library. Normal use needs only one interpreter and one environment.

The package declares Python 3.12 or newer, and CI tests 3.12 through 3.15. NumPy, SciPy, Joblib, and Numba install automatically; they must also support your Python version and platform. The commands below use **3.14** as a consistent starting point. Substitute `3.15` if you want to use that interpreter instead. Contributors testing both should use [separate environments](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md#test-python-314-and-315).

Install your chosen Python version, then create an environment. Git is needed only for a source checkout or GitHub install. Run these commands in a folder where you want to work.

**Windows (PowerShell):**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below; activation is optional.

**macOS or Linux:**

```sh
python3.14 -m venv .venv
. .venv/bin/activate
python --version
```

Outside an activated environment, names such as `python`, `python3`, and `py` may resolve to different installed versions. Check `python --version`; the explicit version used to create the environment determines its interpreter.

## Install the alpha from PyPI

Once `0.1.0a1` is published, install it explicitly:

```sh
python -m pip install "datanuggets[cli]==0.1.0a1"
python -c "import datanuggets; print(datanuggets.__version__)"
```

The `a1` suffix means alpha 1. An exact version selects this prerelease without `--pre`; avoid an unpinned install when reproducing results. Omit `[cli]` for the core NumPy API, or use `[flow]` for the FCS command. Before publication, use the GitHub or local-wheel routes.

## Install from GitHub

Make sure Git is installed and your Git credentials can access the repository. In your activated environment:

```sh
python -m pip install "datanuggets[cli] @ git+https://github.com/ron2k1/datanuggets.git@main"
python -c "import datanuggets; print(datanuggets.__version__)"
datanuggets --help
```

The candidate version is `0.1.0a1`. `main` can change between releases. Record the source commit when reproducing an analysis; replacing `@main` with a full commit hash installs that exact revision.

## Extras

Choose the optional features you need:

| Extra | Enables | Adds |
| --- | --- | --- |
| `cli` | `datanuggets analyze`, CSV tables, and plots | pandas, matplotlib |
| `frame` | `DataNuggets.to_frame()` | pandas |
| `progress` | Progress bars during creation and refinement | tqdm |
| `plot` | `datanuggets.plot` | matplotlib |
| `flow` | The `datanuggets-flow` command | flowio, matplotlib, pandas, scikit-learn |

Combine extras inside the brackets. Once the alpha is published, add the flow command without changing your package version:

```sh
python -m pip install "datanuggets[flow]==0.1.0a1"
datanuggets-flow --help
```

For a source install instead, use `python -m pip install "datanuggets[flow] @ git+https://github.com/ron2k1/datanuggets.git@main"`. This selects the current `main` commit, which may differ from the published alpha.

If you skipped activation on Windows, run `.\.venv\Scripts\datanuggets-flow.exe --help` instead.

The array-based gate-finding functions in `datanuggets.flow` need only the core dependencies. Reading FCS files requires flowio. The [FCS guide](cli.md#fcs-pipeline) explains that pipeline's preprocessing and output folders. The general [CSV command](cli.md#analyze-a-csv) uses the `cli` extra.

## From a local checkout

Clone first, then create and activate an environment inside the checkout using the commands above:

```sh
git clone --branch main https://github.com/ron2k1/datanuggets.git
cd datanuggets
```

In that environment:

```sh
python -m pip install ".[cli]"
python examples/first_analysis.py
```

The example uses 1,000 recorded observations from R's `quakes` dataset. It saves a cluster plot, nugget table, and observation labels in a new `outputs/first-analysis-<timestamp>/` folder inside the checkout. No R installation, input CSV, or dataset download is needed. This is a demonstration of the workflow, not an earthquake model.

### Editable install for development

Use `python -m pip install -e ".[cli]"` to make edits under `src/` take effect without reinstalling. Contributors who use uv can instead run `uv sync --locked --python 3.14` from the checkout to install the locked development environment, including tests and optional dependencies. See [development](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) for checks and testing on 3.15.

## From a built wheel

If you already have a wheel, install it into your chosen environment with pip:

```sh
python -m pip install --force-reinstall "path/to/datanuggets-0.1.0a1-py3-none-any.whl[cli]"
python -m datanuggets --help
```

Building the wheel is a separate maintainer step: run `uv build` from a checkout with uv installed, then distribute the file under `dist/` when redistribution is permitted. Recipients do not need the build tools. See the [MacBook commands](#macbook-install-once-from-a-wheel) for a complete setup using a wheel in Downloads.

The wheel contains Python code and has no platform-specific extension of its own. NumPy, SciPy, and Numba do have compiled components, so the wheel's `any` tag does not guarantee support on every platform. The source archive also includes user guides, the runnable example, and recorded test fixtures; the wheel links to the online guides. Build checks are described in [development](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md#check-a-build).

## The first call compiles

Numba compiles numerical kernels the first time they run. Initial calls can therefore take longer than later ones. It caches compiled code separately for each Python version; an unwritable cache can cause recompilation in new processes. If necessary, set `NUMBA_CACHE_DIR` to a writable folder.

Continue with the [CSV command](cli.md#analyze-a-csv), [Python walkthrough](quickstart.md), or [function reference](api.md).
