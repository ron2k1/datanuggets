# Releasing

The first candidate is `0.1.0a1`. Updating documentation or merging a pull request does not publish a package. The release workflow can check a candidate without uploading it. Once the provenance hold below is resolved, publication requires selecting TestPyPI or pushing a release tag.

**Publication hold:** the GNU sorting helpers need a compatible source-provenance record or replacement before this candidate can be redistributed publicly. See [third-party notices](../THIRD_PARTY.md#publication-hold-gnu-sorting-helpers). Both TestPyPI and PyPI uploads are blocked in the release workflow; check-only runs remain available.

## What the workflow does

`.github/workflows/release.yml` is on `main`. A manual run defaults to `destination: check`, which builds and tests without uploading. After the source-provenance hold is resolved and its explicit guard is removed in review, select `destination: testpypi` to upload to TestPyPI; a pushed `v*` tag uploads to PyPI. Until then, either publication attempt fails at the gate before building or requesting index credentials. The first job requires the selected commit to be in `main`'s history. A tag must also match `pyproject.toml`: for example, `v0.1.0a1` for version `0.1.0a1`.

The hold protects this workflow revision. A tag on an older commit runs that commit's workflow, which may predate the guard. Keep Trusted Publishing unconfigured or disabled on both indexes until the provenance issue is resolved; do not use historical tags to bypass the hold.

Before uploading, the workflow runs:

- Linux tests on Python 3.12–3.15, including the slow tier on 3.14 and 3.15.
- Windows and macOS tests on Python 3.12–3.15.
- A source archive and wheel build, the distribution audit, strict `twine check`, a core-only wheel smoke test on recorded data, and the source archive's tests against an installed wheel with all extras.

Only the checked archives are uploaded. Both indexes use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/), so a repository API token is not required. The publishing action creates provenance attestations by default; those identify the repository, workflow and commit publicly, including when the source repository is private.

## One-time account setup

After the provenance hold is resolved, the account owner must configure publishing before the first upload. These settings are separate from the files committed to the repository. Both index accounts need a verified email address and two-factor authentication; keep recovery codes somewhere safe. See [PyPI account setup](https://docs.pypi.org/your-account/managing-your-account/).

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
3. Before a public release, make sure the repository and the README's documentation links are publicly accessible. Preserve upstream notices and dataset attribution in both archives. [Third-party notices](../THIRD_PARTY.md) record the maintainer's confirmation of Prof. Javier Cabrera's redistribution permission on 2026-10-21; review permissions again when adding externally supplied code or data.
4. Update the README, installation guide, function-reference version, and release notes **before building**. The README is embedded in the distribution metadata, so an edit after upload cannot repair that release's PyPI description.

Before the first upload, confirm all four steps above. A successful check run establishes build and test results, not account setup, public documentation access, or redistribution permission.

## Test a release candidate

`0.1.0a1` means the first alpha of 0.1.0. Use its exact version in install instructions so users intentionally select the prerelease. For later candidates, substitute the version actually merged to `main`.

1. Update `project.version` in `pyproject.toml`, refresh `uv.lock` with `uv lock`, and merge the version, release notes, and documentation together to `main`. The library reads `datanuggets.__version__` from installed package metadata.
2. In GitHub Actions, select **release → Run workflow → main → check**. Review all checks and download the `dist` artifact. After account setup and release permission are confirmed, run again with **testpypi** to upload the candidate. Record the full commit hash of that run.
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

Only after the provenance hold is resolved, publishing is configured, and the candidate passes, tag the exact commit tested above and push that tag:

```sh
git switch main
git pull --ff-only origin main
git tag -a v0.1.0a1 TESTED_COMMIT_SHA -m "datanuggets 0.1.0a1"
git push origin v0.1.0a1
```

Replace `TESTED_COMMIT_SHA` with the candidate's full commit hash. The workflow reruns its checks and confirms the built version matches the tag before uploading to [PyPI](https://pypi.org/p/datanuggets/). Verify the published metadata and documentation links, then install that exact version in a fresh environment using the README's pinned command: `python -m pip install "datanuggets[frame,plot]==0.1.0a1"`. The tag run rebuilds and checks its own archives; it does not promote the TestPyPI files byte for byte.

Published files cannot be replaced. Fix a broken release with a new version; yanking can discourage installation of the broken version without deleting its history.

## Run the full platform matrix on a pull request

Add the `matrix` label to run the Windows and macOS jobs before merging. Remove and re-add the label to rerun them after a later push. The release workflow always runs this matrix.
