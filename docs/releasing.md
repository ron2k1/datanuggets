# Releasing

The first release is `0.1.0a3`. `0.1.0a1` was built and tested as a candidate, and neither it nor `0.1.0a2` was uploaded. Merging a pull request publishes nothing. A release goes out only when a maintainer runs the release workflow with TestPyPI selected or pushes a `v*` tag. The tag publishes the Python package to PyPI.

## What the workflows do

`.github/workflows/release.yml` publishes the package. A manual run defaults to `destination: check`, which builds and tests the candidate without uploading it. Select `destination: testpypi` to upload it to TestPyPI. A pushed `v*` tag uploads it to PyPI. The first job requires the commit to be in `main`'s history, and a tag must match the version in `pyproject.toml`, for example `v0.1.0a3` for version `0.1.0a3`.

Before uploading, the workflow runs:

- Linux tests on Python 3.12 to 3.15, including the slow tier on 3.14 and 3.15.
- Windows and macOS tests on Python 3.12 to 3.15.
- A source archive and wheel build, the distribution audit, strict `twine check`, a core-only wheel smoke test on recorded data, and the source archive's tests against an installed wheel with all extras.

Only the checked archives are uploaded. Both indexes use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/), so the repository stores no API token. The publishing action creates provenance attestations by default. They name the repository, workflow and commit publicly.

## One-time setup

The account owner configures these once, before the first release. They live in the GitHub and index settings, not in the committed files.

### Protect `main` and the release tags

1. In **Settings > Rules > Rulesets**, add a branch ruleset that targets the default branch. Turn on **Require a pull request before merging** and **Require status checks to pass**, and select the checks that the `tests`, `matrix` and `package` workflows report on pull requests. Keep **Restrict deletions** and **Block force pushes** on.
2. Add a tag ruleset that targets the pattern `v*`. Turn on **Restrict creations**, **Restrict updates** and **Restrict deletions**, and put the **Repository admin** role on the bypass list. Only an admin can then create a release tag, and a pushed tag can't be moved to another commit or deleted.

### Create the GitHub environments

In **Settings > Environments**, create `testpypi` and `pypi`. Under **Deployment branches and tags**, choose **Selected branches and tags**. Limit `testpypi` to the branch `main` and `pypi` to tags matching `v*`. Turn on **Required reviewers** in both and add yourself. Leave **Prevent self-review** off while you are the only maintainer, or no one can approve a deployment. Each upload then waits on the run page until a reviewer approves it.

On a Free, Pro or Team plan, GitHub offers **Required reviewers** only while the repository is public. In a private repository on one of those plans, no approval step exists, and each upload starts as soon as its checks pass. The branch and tag limits still apply in a private repository on a Pro or Team plan. See [GitHub's reference on environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments#required-reviewers).

### Add the trusted publishers

Both index accounts need a verified email address and two-factor authentication. Keep the recovery codes somewhere safe. See [PyPI account setup](https://docs.pypi.org/your-account/managing-your-account/).

Create a pending trusted publisher in [TestPyPI](https://test.pypi.org/manage/account/publishing/) and in [PyPI](https://pypi.org/manage/account/publishing/) with these exact values. The two sites have separate accounts.

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| Project name | `datanuggets` | `datanuggets` |
| Owner | `ron2k1` | `ron2k1` |
| Repository | `datanuggets` | `datanuggets` |
| Workflow | `release.yml` | `release.yml` |
| Environment | `testpypi` | `pypi` |

A pending publisher does not reserve the project name. Check that `datanuggets` is still free on both indexes before the first release.

### Permissions and notices

Keep the upstream notices and the dataset attributions in both archives. On 2026-11-01, Prof. Javier Cabrera approved releasing the port publicly under GPL-2.0-only, as the `datanuggets` package on PyPI ([release approval](../THIRD_PARTY.md#release-approval)). Review permissions again before adding externally supplied code or data.

A successful check run shows the build and tests pass. It does not show that the setup above is done.

## Prepare the release

Before building, update the README, the installation guide, the quickstart, the compatibility notes, the docs index (`docs/README.md`), the function-reference version and the release notes. A search of `README.md` and `docs/` for the old version string finds any other mention. The README is embedded in the distribution metadata, so an edit after upload can't repair that release's PyPI description.

In `CITATION.cff`, set `version` to the release version and `date-released` to the day the tag is pushed. `tests/scripts/test_citation.py` fails while `version` differs from `project.version` in `pyproject.toml`.

## Test a release candidate

`0.1.0a3` means the third alpha of 0.1.0. Use the exact version in install instructions so users choose the prerelease on purpose. For later candidates, use the version actually merged to `main`.

1. Update `project.version` in `pyproject.toml`. Then, from the repository root, refresh the lock file:

   ```sh
   uv lock
   ```

   `uv.lock` records the package's version. Merge the version, the lock file, `CITATION.cff`, the release notes and the documentation together to `main`. The library reads `datanuggets.__version__` from the installed package metadata.
2. In GitHub Actions, select **release > Run workflow > main > check**. Review all checks and download the `dist` artifact. Then run it again with **testpypi**. If the `testpypi` environment has required reviewers, approve the deployment on the run page. Record the full commit hash of that run.
3. In a fresh environment, download the exact wheel from TestPyPI without dependencies, then install that local wheel with dependencies from PyPI:

   ```sh
   python -m pip download --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: --dest tmp/testpypi datanuggets==0.1.0a3
   python -m pip install --index-url https://pypi.org/simple/ "tmp/testpypi/datanuggets-0.1.0a3-py3-none-any.whl[frame,plot]"
   python -c "import datanuggets; print(datanuggets.__version__)"
   ```

   This makes the candidate's source explicit. An `--extra-index-url` does not give TestPyPI priority over PyPI, because pip considers matching candidates across indexes. See [pip's package selection documentation](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages).

4. Run `python examples/first_analysis.py` from a checkout of that same commit, using the fresh environment. It must produce a plot and two CSV files under `outputs/`.

TestPyPI needs a new version for each new set of release files. Bump the alpha suffix for another candidate, for example `0.1.0a4`.

## Publish the tested commit

Once the candidate passes, tag the exact commit tested above and push that tag:

```sh
git switch main
git pull --ff-only origin main
git tag -a v0.1.0a3 TESTED_COMMIT_SHA -m "datanuggets 0.1.0a3"
git push origin v0.1.0a3
```

Replace `TESTED_COMMIT_SHA` with the candidate's full commit hash. The tag starts `release.yml`. The release workflow reruns its checks. If the `pypi` environment has required reviewers, it waits for approval of the deployment. Without them, pushing the tag is the last step before the upload. The workflow then confirms that the built files carry the tag's version and uploads them to [PyPI](https://pypi.org/p/datanuggets/). The tag run rebuilds and checks its own archives. It does not promote the TestPyPI files byte for byte.

Then verify the PyPI page's metadata and documentation links, and install the exact version in a fresh environment with the README's pinned command, `python -m pip install "datanuggets[frame,plot]==0.1.0a3"`.

Published package files can't be replaced. Fix a broken release with a new version. Yanking it discourages installs of the broken version without deleting its history.

## Ship an update

Every later release takes the same path. Merge the changes with a new version, bumped as step 1 of [Test a release candidate](#test-a-release-candidate) lists, check it, try it on TestPyPI and push its tag. Users upgrade with `pip install` at the new version.

## Windows and macOS tests

`matrix.yml` runs the Windows and macOS tests on every pull request to `main`, so a platform break shows before the merge. The release workflow runs them again on the commit it publishes.
