# Releasing

`release.yml` publishes the package. Run by hand from the Actions tab, it uploads to TestPyPI. A pushed `v*` tag uploads to PyPI. Either way it first runs `tests.yml` (Linux, 3.12 to 3.15), `matrix.yml` (Windows and macOS, 3.12 to 3.15) and `package.yml` (the build, `scripts/check_dist.py`, `twine check` and the installed-wheel test), and it uploads only the two files `package.yml` checked.

Both uploads use Trusted Publishing. No token is stored in the repo or its secrets. The index trusts a run of `release.yml` from this repo in a named GitHub environment, and the job gets a short-lived token for that one upload. The upload action also attaches a signed attestation of where each file was built.

## One-time setup

These need the account owner, and nothing can be uploaded until they're done.

1. On [test.pypi.org](https://test.pypi.org/manage/account/publishing/) and on [pypi.org](https://pypi.org/manage/account/publishing/), add a pending trusted publisher. The two sites have separate accounts, and PyPI needs two-factor sign-in.

   | Field | TestPyPI | PyPI |
   |---|---|---|
   | PyPI project name | `datanuggets` | `datanuggets` |
   | Owner | `ron2k1` | `ron2k1` |
   | Repository name | `datanuggets` | `datanuggets` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `testpypi` | `pypi` |

   A pending publisher doesn't reserve the name. The first upload creates the project, so until then someone else can take `datanuggets`.

2. In the repo's Settings, under Environments, create `testpypi` and `pypi`. On `pypi`, add yourself as a required reviewer and limit deployments to tags matching `v*`. Each PyPI upload then waits for your approval, and a run from a branch can't use the environment.

## A dry run on TestPyPI

GitHub lists a workflow under Run workflow only once its file is on the default branch, so this works after the branch that adds `release.yml` is merged.

1. In Actions, pick release, then Run workflow on `main`.
2. When it finishes, the files are at [test.pypi.org/p/datanuggets](https://test.pypi.org/p/datanuggets). To install from there, with the dependencies from PyPI:

   ```sh
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datanuggets==0.1.0.dev0
   ```

TestPyPI takes each version once, like PyPI. A second dry run needs a new version in `pyproject.toml`, such as `0.1.0.dev1`.

## A release

1. Set the version in `pyproject.toml`, such as `0.1.0`, and merge it to `main`. That's the only place the version is written. `datanuggets.__version__` reads it from the installed metadata.
2. Do a dry run of that commit, as above.
3. Tag the same commit and push the tag:

   ```sh
   git tag -a v0.1.0 -m "datanuggets 0.1.0"
   git push origin v0.1.0
   ```

4. Approve the `pypi` job when it asks. It runs the checker again on the downloaded files with `--expect-version 0.1.0`, so a tag that doesn't match the built version stops the upload. Then the files go up to [pypi.org/p/datanuggets](https://pypi.org/p/datanuggets).

A version can't be uploaded twice or replaced. A broken release is fixed with a new version, and the broken one can be yanked on PyPI, which keeps it installable only by its exact version.

## The full matrix on a pull request

`matrix.yml` doesn't run on every push, since macOS and Windows minutes cost 10 and 2 times as much on a private repo. Add the `matrix` label to a pull request to run it. Remove the label and add it again to run it on a later push.
