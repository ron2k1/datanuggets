# Installation

datanuggets needs Python 3.12 or newer. pip installs its dependencies: numpy 2.0 or newer, scipy 1.13 or newer, joblib 1.4 or newer and numba 0.67 or newer. On Python 3.15 it needs numba 0.68 or newer, the first numba with 3.15 wheels. The suite passes on Python 3.14 and 3.15. Its default tier, without the long fuzz, also passes on 3.12 and 3.13.

It isn't on PyPI. `pip install datanugget`, without the s, gets an unrelated package of code snippets.

## Install from GitHub

The repository is private for now, so pip's `git clone` needs a GitHub account with access to it and git credentials that it can use. Until the open pull requests merge, install from the `feat/flow-pick` branch:

```sh
pip install "git+https://github.com/ron2k1/datanuggets.git@feat/flow-pick"
```

Check the install:

```sh
python -c "import datanuggets; print(datanuggets.__version__)"
```

It prints `0.1.0.dev0`. The version is read from the installed metadata.

### Which branch holds what

`main` doesn't yet hold everything these docs describe. It has `create_dn`, `create_dn_centers`, `refine_dn`, `get_dn_means`, `arithmetic_complexity` and the six `WCluster` functions `wmean`, `wss`, `wwcss`, `distw`, `wpca` and `cluster_predict`, with the `frame` and `progress` extras. The rest waits on four open pull requests, stacked so that each one targets the branch of the one before:

| Pull request | Branch | Adds |
| --- | --- | --- |
| #8 | `feat/wcluster-iterative` | `wkmeans`, `whclust` and the four `dn_` functions |
| #9 | `feat/plotting` | `cut_tree`, `datanuggets.plot` and the `plot` extra |
| #10 | `feat/flow-gate` | `datanuggets.flow` and `plot.gates` |
| #11 | `chore/packaging` | the checked build, the CI matrix and the release workflow |

`feat/flow-pick` builds on all four and adds the `datanuggets-flow` command and the `flow` extra. It is the only branch with both.

Without `@feat/flow-pick`, pip installs `main`. Asking `main` for an extra it doesn't define, such as `plot`, prints `WARNING: datanuggets 0.1.0.dev0 does not provide the extra 'plot'` and installs the package without it. `import datanuggets.plot` then fails with `ModuleNotFoundError`, and there is no `datanuggets-flow` command. Once the pull requests merge, the commands on this page work without the `@feat/flow-pick` suffix.

## Extras

Each extra adds the dependencies of one optional feature. `import datanuggets` loads none of them.

| Extra | Adds | Needed for |
| --- | --- | --- |
| `frame` | pandas 2.2 or newer | `DataNuggets.to_frame()` |
| `progress` | tqdm 4.66 or newer | `progress=True` in `create_dn` and `refine_dn` |
| `plot` | matplotlib 3.9 or newer | `datanuggets.plot` |
| `flow` | flowio 1.4 (below 1.5), matplotlib 3.9 or newer, pandas 2.2 or newer | the `datanuggets-flow` command |

Name the extras in brackets before the URL:

```sh
pip install "datanuggets[plot] @ git+https://github.com/ron2k1/datanuggets.git@feat/flow-pick"
pip install "datanuggets[flow] @ git+https://github.com/ron2k1/datanuggets.git@feat/flow-pick"
pip install "datanuggets[frame,progress,plot] @ git+https://github.com/ron2k1/datanuggets.git@feat/flow-pick"
```

A feature whose extra is missing stops with the install it needs. `to_frame()` raises `ImportError: to_frame() needs pandas: pip install 'datanuggets[frame]'`, and `progress=True` and `import datanuggets.plot` raise the same kind of error for tqdm and matplotlib. `datanuggets-flow` exits with status 1 and prints `datanuggets-flow needs the flow extra: pip install "datanuggets[flow]"`.

The gate finder in `datanuggets.flow` needs only numpy and scipy, so it imports without the `flow` extra. Only the command reads FCS files, through flowio. [cli.md](cli.md) documents the command.

## From a built wheel

The wheel is pure Python, `datanuggets-0.1.0.dev0-py3-none-any.whl`, so one file installs on every platform and on every Python from 3.12. numpy, scipy and numba come from PyPI as binary wheels. To build it from a clone:

```sh
git clone --branch feat/flow-pick https://github.com/ron2k1/datanuggets.git
cd datanuggets
uv build
pip install "dist/datanuggets-0.1.0.dev0-py3-none-any.whl[plot]"
```

`uv build` writes the sdist and the wheel to `dist/`. Extras go in brackets after the file name, as above. The `package` workflow builds the same two files on every pull request, checks them, and uploads them as the `dist` artifact of the run, which GitHub keeps for 7 days. [development.md](development.md#check-a-build) shows how to run those checks locally.

## Editable install for development

```sh
git clone --branch feat/flow-pick https://github.com/ron2k1/datanuggets.git
cd datanuggets
uv sync
```

`uv sync` creates `.venv/` from `uv.lock` and installs the package from `src/` in editable mode, so changes to the source take effect without reinstalling. It also installs the `dev` dependency group: pytest, pytest-xdist and ruff, the dependencies of every extra (pandas, tqdm, matplotlib and flowio), and openpyxl for the flow benchmark. Run commands in that environment with `uv run`, such as `uv run datanuggets-flow --help`.

CI runs `uv sync --locked`, which fails instead of updating `uv.lock` when it no longer matches `pyproject.toml`. Without uv, `pip install -e ".[flow]"` from the clone gives an editable install too, but without the lock's versions or the `dev` group.

[development.md](development.md) covers the tests, the lint and the build check.

## The first call compiles

numba compiles each kernel the first time it runs and caches the result, so the first calls in a new environment take seconds where later processes are much faster. The cache is kept per Python version, so a new version compiles once more. Where numba can't write its cache, every new process compiles again. numba's `NUMBA_CACHE_DIR` environment variable points the cache at a folder it can write. [limitations.md](limitations.md#memory-and-speed) gives the measured times.
