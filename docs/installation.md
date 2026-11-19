# Installation

This guide installs `0.1.0a3`, the first alpha on PyPI, or the current source from GitHub `main`. `datanugget`, without the final **s**, is an unrelated Python package.

## Supported platforms

datanuggets itself is pure Python, but NumPy, SciPy and Numba are compiled, and Numba and its llvmlite dependency publish wheels for fewer platforms than the others. The package installs on:

| System | Requirement |
| --- | --- |
| Windows | x64 |
| macOS | Apple silicon, macOS 12 or newer |
| Linux | x86-64 or ARM64, glibc 2.27 or newer |

Numba has no wheels for Intel Macs or for musl Linux such as Alpine, so pip tries to build it from source there, which needs LLVM and usually fails. Numba also publishes Windows on ARM wheels for Python 3.12 to 3.14, but no tests run on that platform or on Linux ARM64.

## Mac: install once from PyPI

This installs the Python library and its `frame` and `plot` extras into an environment of its own, on a Mac with Apple silicon. The extras add pandas and matplotlib, which the [Python walkthrough](quickstart.md) uses. You do **not** need Git or uv, or a copy of the repository. If Python is not installed, install Python 3.14 for macOS from [python.org](https://www.python.org/downloads/macos/), then open Terminal.

Run these commands once. pip downloads the package and its dependencies, so this step needs an internet connection:

```sh
python3.14 -m venv "$HOME/.venvs/datanuggets"
"$HOME/.venvs/datanuggets/bin/python" -m pip install "datanuggets[frame,plot]==0.1.0a3"
"$HOME/.venvs/datanuggets/bin/python" -c "import datanuggets; print(datanuggets.__version__)"
```

`0.1.0a3` is a pre-release. pip installs a pre-release when the requirement names one, as `==0.1.0a3` does, or when `--pre` is passed. An install without a version also gets the newest pre-release while the project has no final release.

To move to a later version, rerun the pip command with the new version number. You do not need to recreate the environment or reinstall before each analysis.

For Python scripts, run `"$HOME/.venvs/datanuggets/bin/python" your_script.py`, or select that interpreter in your editor. Then `from datanuggets import create_dn` uses this installation.

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

Install the exact version:

```sh
python -m pip install "datanuggets==0.1.0a3"
python -c "import datanuggets; print(datanuggets.__version__)"
```

The `a3` suffix means alpha 3. `0.1.0a1` and `0.1.0a2` were never uploaded. An exact version selects this pre-release without `--pre`; avoid an unpinned install when reproducing results.

## Install from GitHub

Make sure Git is installed. In your activated environment:

```sh
python -m pip install "datanuggets @ git+https://github.com/ron2k1/datanuggets.git@main"
python -c "import datanuggets; print(datanuggets.__version__)"
```

`main` can be ahead of the latest release, `0.1.0a3`, and can change between releases. Record the source commit when reproducing an analysis; replacing `@main` with a full commit hash installs that exact revision.

## Extras

Choose the optional features you need:

| Extra | Enables | Adds |
| --- | --- | --- |
| `frame` | `DataNuggets.to_frame()` | pandas |
| `progress` | Progress bars during creation and refinement | tqdm |
| `plot` | `datanuggets.plot` | matplotlib |
| `tsne` | `tsne_2d` and `tsne_3d` in `datanuggets.projection` | scikit-learn |

Combine extras inside the brackets. To add extras without changing your package version:

```sh
python -m pip install "datanuggets[frame,plot]==0.1.0a3"
```

For a source install instead, use `python -m pip install "datanuggets[frame,plot] @ git+https://github.com/ron2k1/datanuggets.git@main"`. This selects the current `main` commit, which may differ from the published alpha.

## From a local checkout

Clone first, then create and activate an environment inside the checkout using the commands above:

```sh
git clone --branch main https://github.com/ron2k1/datanuggets.git
cd datanuggets
```

In that environment:

```sh
python -m pip install ".[frame,plot]"
python examples/first_analysis.py
```

The example uses 1,000 recorded observations from R's `quakes` dataset. It saves a cluster plot, nugget table, and observation labels in a new `outputs/first-analysis-<timestamp>/` folder inside the checkout. No R installation, input CSV, or dataset download is needed. This is a demonstration of the workflow, not an earthquake model.

### Editable install for development

Use `python -m pip install -e ".[frame,plot]"` to make edits under `src/` take effect without reinstalling. Contributors who use uv can instead run `uv sync --locked --python 3.14` from the checkout to install the locked development environment, including tests and optional dependencies. See [development](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md) for checks and testing on 3.15.

## From a built wheel

If you already have a wheel, install it into your chosen environment with pip:

```sh
python -m pip install --force-reinstall "path/to/datanuggets-0.1.0a3-py3-none-any.whl[frame,plot]"
python -c "import datanuggets; print(datanuggets.__version__)"
```

To build one yourself, run `uv build` from a checkout with uv installed. The wheel and the source archive land in `dist/`.

The wheel contains Python code and has no platform-specific extension of its own. NumPy, SciPy, and Numba do have compiled components, so the wheel's `any` tag does not mean every platform is supported. [Supported platforms](#supported-platforms) lists where they install. The source archive also includes user guides, the runnable example, and recorded test fixtures; the wheel links to the online guides. Build checks are described in [development](https://github.com/ron2k1/datanuggets/blob/main/docs/development.md#check-a-build).

## The first call compiles

Numba compiles numerical kernels the first time they run. Initial calls can therefore take longer than later ones. It caches compiled code separately for each Python version; an unwritable cache can cause recompilation in new processes. If necessary, set `NUMBA_CACHE_DIR` to a writable folder.

Continue with the [Python walkthrough](quickstart.md) or [function reference](api.md).
