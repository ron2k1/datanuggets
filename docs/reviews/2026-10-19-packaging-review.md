# Packaging review, 2026-10-19

The range under review is the [packaging plan](../plans/2026-10-19-packaging.md) on `chore/packaging`, feat/flow-gate..1e67938: the sdist allowlist and metadata in `pyproject.toml`, `scripts/check_dist.py` and its tests, the `package`, `tests`, `matrix` and `release` workflows, `docs/releasing.md` and the README, against the [design](../specs/2026-10-19-packaging-design.md). Its findings are fixed in f5bde5d, 14fe283 and 4fdbb40, and the docs commit that adds this record.

One reviewer ran: Opus, on the laptop, Windows with Python 3.14, read-only on the repo with experiments in a scratch folder. It built the sdist and wheel, recovered the ten denied words from their digests by tokenizing the repo and the local config, and scanned every member of both files, every decompressed `.npz` member and all tar and zip metadata for them, in UTF-8 and UTF-16LE: none was there. It then built archives that should fail and fed them to the checker. Its verdict: one major finding to fix before any `v*` tag is pushed, one more before the flow fixtures are recorded again, and the rest minor or nits. Nothing leaked into this build.

## Findings

| Finding | Severity | Fixed in | Evidence it was real |
|---|---|---|---|
| 1. `docs/releasing.md` promises a required reviewer on `pypi`, which a private repo on GitHub Pro can't have | major | 14fe283 | GitHub's docs, see below |
| 2. the checker can't read inside a compressed member, and the flow fixtures are deflated `.npz` zips | major | f5bde5d | an `.npz` holding a home path and a denied word passed |
| 3. the macOS and Linux home paths need a trailing `/`, and a doubly escaped Windows path passes | minor | f5bde5d | `{"cwd": "/Users/alice"}`, `HOME=/home/alice` and five more passed |
| 4. a run from any other label cancels a pending `matrix` run | minor | 4fdbb40 | GitHub's concurrency docs, see below |
| 5. tar and zip headers aren't scanned, though the spec says the owner fields are kept | minor | f5bde5d | a word or path in uname, gname, a PAX value, the zip comment or a member comment passed |
| 6. a dispatch from any branch reaches TestPyPI | minor | 14fe283 | nothing checked the ref |
| 7. camelCase names and words next to digits aren't caught | nit | f5bde5d | `ZorblaxHelper` and `zorblax2` passed |
| 8. an unknown zip method or an encrypted member raises instead of reporting | nit | f5bde5d | `NotImplementedError` and `RuntimeError` escaped `check()` |
| 9. the cache pattern is case-sensitive and narrow | nit | f5bde5d | `X.PYC`, `.hypothesis/`, `r/.Rhistory` and five more passed |
| 10. three problem lines print file names unmasked | nit | f5bde5d | a dist file named after the test word was printed whole |
| 11. a wrong tag is refused only after the tests, about 110 billed minutes | nit | 14fe283 | the check sat in the `pypi` job |

### The release gate

GitHub's deployments reference says: "If you are on a GitHub Free, GitHub Pro, or GitHub Team plan, required reviewers are only available for public repositories." Deployment branches and tags are available to private repos on Pro. So while the repo is private, a pushed `v*` tag on any commit would have gone up to PyPI as soon as the tests passed, with no one asked.

`release.yml` now starts with a `gate` job that the tests, the matrix and the build all need. It checks out the full history, refuses a commit that `git merge-base --is-ancestor` doesn't find on `origin/main`, and, on a tag, refuses one that isn't `v` and the version in `pyproject.toml`. That fixes findings 6 and 11 as well: a dispatch from a branch stops there, and a wrong tag fails in seconds. The `pypi` job still runs the checker on the downloaded files with `--expect-version`. The gate's shell was run against the real refs: it passes `origin/main`, refuses this branch's head, and refuses `v0.1.0` against `0.1.0.dev0`. `docs/releasing.md` now says there's no approval step while the repo is private, sets the deployment rules, `main` for `testpypi` and `v*` tags for `pypi`, and says to add a required reviewer once the repo is public.

### Inside compressed members

The three `tests/fixtures/flow/*.npz` files are zips whose members are all deflated, since `benchmarks/flow/record.py` saves them with `np.savez_compressed`. The reviewer built an sdist with a deflated member holding `/home/alice/data` and the test word, and `check()` returned nothing. They're the only fixtures recorded on a development machine, so a re-recording that saved a path or a host name would have shipped.

The checker now opens a member that starts like a zip and scans each inner member's name, bytes and header, one level down. A zip inside that, and a member in gzip, PNG, PDF, xz, bzip2, 7z or zstd, fails as one the checker can't read. The sdist holds 6,077 `.bin`, 544 `.json` and 3 `.npz` files and no other compressed one, so this costs nothing today, and the three `.npz` files are now read and clean.

### Home paths

Seven forms passed: `/Users/alice` and `/home/alice` without a trailing slash (in JSON, after `HOME=` and quoted), a lowercase `/users/alice/x`, a Windows path with four backslashes per separator, the same path in UTF-16LE, and `/root/.cache/x`. The macOS and Linux patterns no longer need the slash, but must start a path, so `https://example.com/home/page` isn't one. Windows takes up to four separators. A lowercase `/users/` stays out, since macOS writes `/Users` and the GitHub API's paths are `/users/`. UTF-16 text and `/root` stay out too: nothing in the build writes UTF-16, and `/root` holds no user name. `C:\Users\Public` and `/home/runner` are flagged, which fails closed.

### The matrix's concurrency

GitHub's concurrency docs say a new run cancels any pending run in its group. With one group per ref and `cancel-in-progress: false`, a `matrix` run started after a push waits for the running one, and a run from any other label then cancels it, so the newer commit's matrix never runs. The group now carries the label's name, so other labels' skipped runs have their own, and a newer `matrix` run cancels an older one.

### Headers, names, errors and caches

Each of these passed with a word or a path in it: a tar entry's `uname` and `gname`, a member's PAX `comment`, a global PAX header, the zip comment and a member's comment. The checker now fails on any tar entry with an owner, which hatchling's never have, scans PAX values other than the name (global ones once, though tarfile copies them into every member's) and zip comments and extra fields, and the spec says the timestamp is the one thing it doesn't keep.

A camelCase part, or a run of letters beside digits, now counts as a word, so `ZorblaxHelper` and `zorblax2` are caught, and `zorblaxes`, `xzorblax` and `ZORBLAXES` still aren't. A zip member with an unknown method or the encryption flag is a "can't be read" line. The cache pattern takes any case and adds `.hypothesis/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `.Rhistory`, `.RData`, `.orig`, `.rej` and `~`. The folder, the unexpected file, the wrong archive names and the versions are masked in the output.

Declined: bytes added before a wheel's first header, since no build step here writes them.

## Checked and found sound

The wheel holds only `datanuggets/` and four dist-info files, with no `.data` folder or entry points. `upload-artifact` skips hidden files, so `uv build`'s `.gitignore` never reaches the upload. Both publish jobs are in the top-level workflow, where Trusted Publishing has to run. `needs` stops the upload when any called workflow fails or is skipped. Inside the called workflows, `github.event_name` and `github.workflow` are the caller's, so the matrix runs inside a release and the groups can't collide. The wheel is built from the sdist, and the files checked are the files uploaded. The installed-wheel test can't import the source tree, and at 1e67938 its 2930 tests were the source tree's 2942 less the 12 checker tests the sdist leaves out. The doc's claims about pending publishers, two-factor sign-in, dispatch needing the default branch, filename reuse and yanking hold.

The reviewer couldn't confirm whether attestations are made for a private repo. The doc says they're on by default and what they publish.

## Test runs at 4fdbb40

- The default tier: 2947 passed, the 2942 reviewed plus the 5 new checker tests, and 2 skipped, the two that need `os.fork`, on Windows with Python 3.14.
- `tests/scripts`: 17 passed.
- The build of this tree: `check_dist.py` OK, in 3 s with the `.npz` members read; `twine check --strict` passes on both files. The build from before the allowlist still has 79 problems.
- actionlint is clean, and zizmor's only notes are the `self-repository` hints, declined in 8979a6f.
