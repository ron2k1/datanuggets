# Packaging and release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A build whose sdist and wheel are allowlisted and checked for leaks, tested as installed, described by full metadata, and published only through Trusted Publishing.

**Architecture:** One stdlib script, `scripts/check_dist.py`, is the independent gate on the built files. Three workflows reuse each other: `package.yml` builds and checks, `tests.yml` (Linux) and `matrix.yml` (Windows, macOS) test, and `release.yml` calls all three before it uploads.

**Tech Stack:** hatchling, uv, twine 7.0.0, GitHub Actions, pypa/gh-action-pypi-publish v1.14.2.

**Spec:** [docs/specs/2026-10-19-packaging-design.md](../specs/2026-10-19-packaging-design.md)

## Global Constraints

- `requires-python = ">=3.12"`; CI covers 3.12, 3.13, 3.14 and 3.15.
- sdist `only-include`: `src/datanuggets`, `tests`, `r`, `LICENSE`, `README.md`, `pyproject.toml`; `exclude`: `tests/scripts`.
- Wheel name must end `-py3-none-any.whl`.
- Actions pinned by SHA: checkout `3d3c42e5aac5ba805825da76410c181273ba90b1` (v7.0.1), setup-uv `1c37ad07a6a961277cf70c0d37d6f313000f5884` (v10.3.0), upload-artifact `cf430e030ddbb5b0abf93d22962f4752f3646cd9` (v7.0.2), download-artifact `9000827ccba6bdab643e8b6fd33ac0654aef8333` (v8.0.2), gh-action-pypi-publish `dc37677b2e1c63e2034f94d8a5b11f265b73ba33` (v1.14.2).
- `id-token: write` only on the two publish jobs; every workflow's top level is `contents: read`.
- The checker never contains a denied word in plain text, and neither do its tests.

## Review Focus

- A JSON fixture with an escaped Windows path (`C:\\Users\\name`) must still be caught: the home-path pattern allows doubled separators.
- A denied word inside a longer word (`subagentsx`) is not a hit; the checker matches whole alphanumeric tokens, case-folded.
- A dist folder holding `uv build`'s `.gitignore` must not fail; any other extra file must.
- A tag `v0.1.0` against a built `0.1.0.dev0` must stop the PyPI job before upload.
- An unreadable or truncated archive must fail, not pass with zero files checked.

---

### Task 1: The release checker

**Files:**
- Create: `scripts/check_dist.py`
- Test: `tests/scripts/test_check_dist.py`

**Interfaces:**
- Produces: `check(dist: Path, expect_version: str | None = None, deny: frozenset[str] = DENY) -> list[str]` (problem lines, empty when clean); `digest(word: str) -> str`; CLI `python scripts/check_dist.py DIST [--expect-version V]` exits 0 or 1; `python scripts/check_dist.py --digest WORD` prints a digest.

- [ ] **Step 1: Write the failing tests.** Helpers build a minimal good sdist (`datanuggets-1.0/` with `PKG-INFO` holding `Version: 1.0`, `pyproject.toml`, `LICENSE`, `README.md`, `src/datanuggets/__init__.py`, `tests/conftest.py`) and wheel (`datanuggets/__init__.py`, `datanuggets-1.0.dist-info/METADATA` with `Version: 1.0`, `.../licenses/LICENSE`) in `tmp_path`, with an `extra` hook per archive. Tests: `test_clean_build_passes` (`check(...) == []`), `test_file_outside_allowlist` (sdist `docs/plan.md` and a hidden scratch folder, both named in the problems), `test_wheel_outside_package`, `test_cache_files` (`__pycache__/x.pyc`, `x.nbi`), `test_home_paths` (`C:\Users\name\x`, `C:\\Users\\name`, `/Users/name/x`, `/home/name/x`), `test_denied_word` (injected `deny={digest("zorblax")}`, found as `ZORBLAX`; not found in `zorblaxes`), `test_missing_required`, `test_version_mismatch` (sdist 1.0 vs wheel 1.1, and `expect_version="2.0"`), `test_unsafe_member` (`../evil`, absolute, symlink), `test_dist_folder_shape` (missing folder, no wheel, two wheels, an extra file, `.gitignore` allowed), `test_unreadable_archive` (truncated `.tar.gz`), `test_cli` (exit codes and `--digest`).
- [ ] **Step 2: Run them.** `uv run pytest tests/scripts -q` fails on the missing module.
- [ ] **Step 3: Implement `scripts/check_dist.py`.** Stdlib only. The test imports it with `importlib.util.spec_from_file_location`. Denied words: lowercased `[A-Za-z0-9]+` tokens, all-digit tokens skipped, deduplicated per file, hashed with SHA-256. Home paths: `(?i)\b[a-z]:[\\/]{1,2}users[\\/]{1,2}[^\\/\s"']+`, `/Users/[^/\s]+/`, `/home/[^/\s]+/` on bytes. Problems name the archive, member and line.
- [ ] **Step 4: Run them.** All pass. Then run the checker on the current build, before Task 2. It must flag the scratch folder, `docs/`, `benchmarks/` and the home paths, which proves the shipped digests on real files.
- [ ] **Step 5: Commit.**

### Task 2: Allowlist, metadata, one version

**Files:**
- Modify: `pyproject.toml`, `src/datanuggets/__init__.py`
- Test: `tests/test_version.py`

**Interfaces:**
- Consumes: Task 1's CLI.
- Produces: `datanuggets.__version__` from `importlib.metadata.version("datanuggets")`.

- [ ] **Step 1: Write `test_version_matches_pyproject`.** It reads `pyproject.toml` beside `tests/` with `tomllib` and asserts `datanuggets.__version__ == project.version`. It fails while `__init__.py` holds its own literal only if they differ, so also assert that `__init__.py`'s source has no version literal.
- [ ] **Step 2: Run it.** It fails on the literal.
- [ ] **Step 3: Edit `pyproject.toml`** (sdist table, classifiers, keywords, `license-files`, `[project.urls]`) **and `__init__.py`.**
- [ ] **Step 4: Verify.** `uv lock --check`; `uv run pytest -q`; `uv build --out-dir tmp/dist`; `uv run --no-project python scripts/check_dist.py tmp/dist` prints OK; `uvx --from twine==7.0.0 twine check --strict tmp/dist/*` passes.
- [ ] **Step 5: Commit.**

### Task 3: The package workflow

**Files:** Create `.github/workflows/package.yml`.

- [ ] **Step 1: Write it.** Triggers: `pull_request`, `workflow_dispatch`, `workflow_call`. One ubuntu job on Python 3.14: build, checker, twine, then the installed-wheel test (unpack sdist, delete `src/`, new venv with the wheel plus `[frame,progress,plot]`, `pytest==9.1.1` and `pytest-xdist==3.8.0`, assert `datanuggets.__file__` is under `site-packages`, run `pytest -n 2 -m "not slow"` from the sdist), upload `dist` with 7-day retention.
- [ ] **Step 2: Run the same steps locally** in `tmp/` and on Windows; the default tier passes against the installed wheel.
- [ ] **Step 3: Commit.**

### Task 4: Test matrix

**Files:** Modify `.github/workflows/tests.yml`; create `.github/workflows/matrix.yml`.

- [ ] **Step 1:** `tests.yml` gains `workflow_call` and Python 3.12 and 3.13; the slow tier runs only on 3.14 and 3.15.
- [ ] **Step 2:** `matrix.yml`: `pull_request: types: [labeled]`, `workflow_dispatch`, `workflow_call`; job `if` mirrors `bench.yml` with the label `matrix`; `windows-latest` and `macos-latest` by 3.12 to 3.15, default tier.
- [ ] **Step 3: Verify locally** the default tier on Windows with 3.12 and 3.13 (`UV_PROJECT_ENVIRONMENT=tmp/venv312`).
- [ ] **Step 4: Commit.**

### Task 5: Release workflow and the owner's steps

**Files:** Create `.github/workflows/release.yml`, `docs/releasing.md`.

- [ ] **Step 1:** `release.yml` calls `tests.yml`, `matrix.yml` and `package.yml`; job `testpypi` (`workflow_dispatch` only, environment `testpypi`, `repository-url: https://test.pypi.org/legacy/`) and job `pypi` (tag push only, environment `pypi`, checkout plus `check_dist.py dist --expect-version "${GITHUB_REF_NAME#v}"` before upload). Both download the `dist` artifact.
- [ ] **Step 2:** `docs/releasing.md`: the publisher values (owner `ron2k1`, repo `datanuggets`, workflow `release.yml`, environments), environment protection, the dispatch-needs-main rule, TestPyPI's one-upload-per-version rule, and the release steps.
- [ ] **Step 3: Commit.**

### Task 6: README

**Files:** Modify `README.md`.

- [ ] **Step 1:** Replace open problem 9 and its intro clause; state the Python versions the suite was seen to pass on; add the build check and the release doc to Development; add step 14 to What was done.
- [ ] **Step 2:** Rebuild and rerun the checker and twine.
- [ ] **Step 3: Commit.**
