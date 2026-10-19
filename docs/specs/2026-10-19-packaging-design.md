# Packaging and release: design

The package builds, but nothing guards what the build puts in the sdist and wheel, CI never tests what a user installs, and there's no way to publish. This spec covers the four steps that fix that: a checked, allowlisted build, the installed-wheel test, the metadata, and the release workflow. A public repo, a changelog, `CITATION.cff` and a docs site come later.

## Problems and fixes

1. **Untracked scratch folders reach the sdist.** Hatchling reads only the root `.gitignore`, so a folder that ignores itself through a nested `.gitignore` is swept in. A build from the working tree on 2026-10-19 carried 16 such files. Fix: `[tool.hatch.build.targets.sdist]` takes an allowlist, `only-include`, of `src/datanuggets`, `tests`, `r`, `LICENSE`, `README.md` and `pyproject.toml`. Anything not on it stays out, including folders that don't exist yet.
2. **The sdist carries the development record.** The same build held `docs/` (plans, reviews and review transcripts, eight of them with home-folder paths), `benchmarks/` (two scripts call a helper from the development machine), `.github/` and `uv.lock`. Fix: the allowlist in 1. The tests for the release checker, `tests/scripts/`, are excluded too, since the checker itself isn't shipped.
3. **Nothing checks the built files.** Fix: `scripts/check_dist.py` opens the sdist and the wheel and fails on:
   - a file outside the allowlist, or a required file missing,
   - a cache file (`__pycache__`, `.pyc`, numba's `.nbi` and `.nbc`, OS litter),
   - an absolute path, a `..` part, or a link, in the archive,
   - a home-folder path in any file's bytes (`X:\Users\name`, `/Users/name/`, `/home/name/`),
   - a denied word in any file's bytes: tool and machine names from the development setup. The checker holds them only as SHA-256 digests, so it doesn't publish the words it hunts for,
   - a version that differs between the sdist, the wheel and their metadata, or from `--expect-version`.

   A missing folder, a missing or extra archive, or an archive it can't read also fails. It runs on every PR and before every upload.
4. **The README is the PyPI page.** Open problem 9 names the scratch folder, and fix 1 resolves it. Item 9 is replaced by the packaging problem that stays open, so item 10 keeps its number.
5. **CI tests the source tree, never the built package.** Fix: a `package` workflow builds with `uv build`, runs the checker and `twine check --strict`, installs the wheel alone into a new venv, deletes `src/` from the unpacked sdist, and runs the sdist's own default tests against the installed wheel. It's `R CMD check` on the built tarball, for Python.
6. **Python 3.12 and 3.13 aren't tested, and Windows and macOS are tested only by hand.** Fix: PRs run Linux on 3.12 to 3.15, with the slow tier on 3.14 and 3.15 as before. A `matrix` workflow runs Windows and macOS on 3.12 to 3.15, when a PR gets the `matrix` label, by hand, and inside every release. The repo is private on GitHub Pro, 3,000 minutes a month, with macOS billed at 10 times and Windows at 2 times, so a full matrix on every push would take about 110 billed minutes. Once the repo is public the label condition can go.
7. **The version is written twice.** Fix: `pyproject.toml` holds it. `datanuggets.__version__` reads the installed metadata. The release job refuses a tag that isn't `v` plus the built version.
8. **No classifiers, keywords or links.** Fix: classifiers (Alpha, Science/Research, OS independent, Python 3.12 to 3.15, two Scientific/Engineering topics; no license classifier, since `license` is an SPDX expression), keywords, `license-files`, and `[project.urls]` for the source, the issues, both CRAN packages and the paper. `authors` stays the Python code's author. The README credits every R author; naming them on PyPI is theirs to agree to.
9. **No release path.** Fix: `release.yml`. Run by hand, it publishes to TestPyPI; on a pushed `v*` tag, to PyPI. Both use Trusted Publishing, with no stored token, from the GitHub environments `testpypi` and `pypi`. It uploads the files the `package` job built and checked, after the Linux, Windows and macOS tests pass. Actions are pinned by commit SHA.
10. **Owner-only steps.** Registering the trusted publishers on test.pypi.org and pypi.org and creating the two environments need the account owner. `docs/releasing.md` lists the values. Nothing is uploaded by this work.

## Already clean

Checked on 2026-10-19 and kept by the checker: the wheel holds only the package's `.py` files and its metadata, the sdist's tar entries have uid and gid 0 and a fixed timestamp, the fixtures hold no user or path strings, and the metadata has no email.
