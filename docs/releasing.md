# Releasing

The repository is prepared for a future PyPI release; the current installation route is [GitHub `main`](installation.md). Updating documentation or merging a pull request does not publish a package. Publishing starts only when a maintainer runs the release workflow or pushes a release tag.

## What the workflow does

`.github/workflows/release.yml` is on `main`. A manual run uploads to TestPyPI; a pushed `v*` tag uploads to PyPI. Its first job requires the selected commit to be in `main`'s history. A tag must also match `pyproject.toml`: for example, `v0.1.0a1` for version `0.1.0a1`.

Before uploading, the workflow runs:

- Linux tests on Python 3.12–3.15, including the slow tier on 3.14 and 3.15.
- Windows and macOS tests on Python 3.12–3.15.
- A source archive and wheel build, the distribution audit, `twine check`, and tests against the installed wheel.

Only the checked archives are uploaded. Both indexes use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/), so a repository API token is not required. The publishing action creates provenance attestations by default; those identify the repository, workflow and commit publicly, including when the source repository is private.

## One-time account setup

The account owner must configure publishing before the first upload. These settings are separate from the files committed to the repository.

1. Create a pending trusted publisher in [TestPyPI](https://test.pypi.org/manage/account/publishing/) and [PyPI](https://pypi.org/manage/account/publishing/), using the following exact values. The sites have separate accounts.

   | Field | TestPyPI | PyPI |
   | --- | --- | --- |
   | Project name | `datanuggets` | `datanuggets` |
   | Owner | `ron2k1` | `ron2k1` |
   | Repository | `datanuggets` | `datanuggets` |
   | Workflow | `release.yml` | `release.yml` |
   | Environment | `testpypi` | `pypi` |

   A pending publisher does not reserve a project name; verify availability before planning the first release.

2. In GitHub Settings → Environments, create `testpypi` and `pypi`. Restrict `testpypi` to branch `main` and `pypi` to tags matching `v*`. Add deployment approval where the repository's plan supports it, and limit release-tag access to maintainers.
3. Before a public release, make sure the repository and the README's documentation links are accessible to the intended audience. Confirm the license, credits, compatibility notes and known limitations describe the version being released.

## Test a release candidate

For a first public alpha, a version such as `0.1.0a1` communicates its status. The following commands use that version as an example; substitute the version actually merged to `main`.

1. Update `project.version` in `pyproject.toml`, refresh `uv.lock` with `uv lock`, and merge both changes to `main`. The library reads `datanuggets.__version__` from installed package metadata.
2. In GitHub Actions, select **release → Run workflow → main**. This uploads to TestPyPI after all checks pass.
3. In a fresh environment, download the exact wheel from TestPyPI without dependencies, then install that local wheel with dependencies from PyPI:

   ```sh
   python -m pip download --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: --dest tmp/testpypi datanuggets==0.1.0a1
   python -m pip install --index-url https://pypi.org/simple/ "tmp/testpypi/datanuggets-0.1.0a1-py3-none-any.whl[frame,plot,flow]"
   python -c "import datanuggets; print(datanuggets.__version__)"
   datanuggets-flow --help
   ```

   This makes the candidate's source explicit. An `--extra-index-url` does not give TestPyPI priority over PyPI; pip considers matching candidates across indexes. See [pip's package selection documentation](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages).

4. Run `python examples/first_analysis.py` from a checkout of that same commit, using the fresh environment. It must produce a plot and two CSV files under `outputs/`. Exercise the CLI with representative FCS files before advertising the flow workflow for a study.

TestPyPI requires a new version for each new set of release files. Bump the alpha suffix for another candidate, for example `0.1.0a2`.

## Publish the tested commit

After the candidate passes, tag the exact commit tested above and push that tag:

```sh
git switch main
git pull --ff-only origin main
git tag -a v0.1.0a1 TESTED_COMMIT_SHA -m "datanuggets 0.1.0a1"
git push origin v0.1.0a1
```

Replace `TESTED_COMMIT_SHA` with the candidate's full commit hash. The workflow reruns its checks and confirms the built version matches the tag before uploading to [PyPI](https://pypi.org/p/datanuggets/). Verify the published metadata, install that exact version in a fresh environment, and then update the README and installation guide to use PyPI. For an alpha, show a pinned version such as `python -m pip install "datanuggets[frame,plot]==0.1.0a1"`.

Published files cannot be replaced. Fix a broken release with a new version; yanking can discourage installation of the broken version without deleting its history.

## Run the full platform matrix on a pull request

Add the `matrix` label to run the Windows and macOS jobs before merging. Remove and re-add the label to rerun them after a later push. The release workflow always runs this matrix.
