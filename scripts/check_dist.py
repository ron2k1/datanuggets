"""Check a built sdist and wheel before they ship, the way R CMD check reads the built tarball.

    python scripts/check_dist.py dist [--expect-version 0.1.0]
    python scripts/check_dist.py --digest WORD

The dist folder must hold one sdist and one pure-Python wheel of the same version. Every file in
them must be on the allowlist below and not a cache file, and no file's name or bytes may hold a
home-folder path or a denied word. It prints one line per problem and exits with 1 on any,
including a folder or an archive it can't read. --digest prints the digest that adds a word to
DENY."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

NAME = "datanuggets"

# The sdist's allowlist, relative to its top folder: pyproject.toml's only-include and exclude,
# and the PKG-INFO and .gitignore the build adds. It's kept apart from pyproject.toml on purpose,
# so that a change to only-include has to get past this too.
SDIST_FILES = frozenset({"PKG-INFO", ".gitignore", "pyproject.toml", "LICENSE", "README.md"})
SDIST_DIRS = ("src/datanuggets/", "tests/", "r/")
SDIST_EXCLUDED = ("tests/scripts/",)
SDIST_REQUIRED = (
    "PKG-INFO",
    "pyproject.toml",
    "LICENSE",
    "README.md",
    "src/datanuggets/__init__.py",
    "tests/conftest.py",
)

# Caches a build can sweep in from a working tree: Python's, numba's (cache=True writes .nbi and
# .nbc files beside the source) and the files the OS leaves
CACHE = re.compile(
    r"(^|/)(__pycache__|\.ipynb_checkpoints)/|\.(pyc|pyo|nbi|nbc)$"
    r"|(^|/)(\.DS_Store|Thumbs\.db|desktop\.ini)$"
)

# Home folders on Windows, with doubled separators for JSON-escaped paths, on macOS and on Linux
HOME = re.compile(
    rb"(?i:\b[a-z]:[\\/]{1,2}users[\\/]{1,2}[^\\/\s\"']+)|/Users/[^/\s]+/|/home/[^/\s]+/"
)

# SHA-256 of the lowercased words no release may hold: tool and machine names from the
# development setup. Only the digests are kept, so this file doesn't publish the words it looks
# for. A word is a whole run of letters and digits, so it's found inside snake_case and
# kebab-case names but not inside a longer word.
DENY = frozenset(
    {
        "155521b950c68ca76d57111cd3377198d205b0a7667b89d5222f08afc67d6961",
        "567781541aa1d043041ec08be14e1c519629784e3952d0d079ac87640242ba5e",
        "57de4cf40144bdf7d00010f2f5557a7d642c2b9705309bfade167dd313e2ca93",
        "60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c",
        "7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef",
        "94a168d2da574b00ec6c787377278465271cf2b97e68af076fc3eee87eba8664",
        "b225390a8984c8de4206c746772ab5ef2abb0a02bcb043da10f6d41c274d1a02",
        "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131",
        "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
        "d298391b07433318a02bd922ed5b90112a4302606c4928b54e3fb1c18130d92e",
    }
)
TOKEN = re.compile(rb"[a-z0-9]+")
NAME_TOKEN = re.compile(r"[A-Za-z0-9]+")
VERSION = re.compile(rb"^Version: *(\S+)\s*$", re.MULTILINE)


def digest(word: str) -> str:
    return hashlib.sha256(word.lower().encode()).hexdigest()


def _denied(token: bytes, deny: frozenset[str]) -> bool:
    """token is lowercase. No denied word is all digits, which skips most of a fixture's tokens."""
    return not token.isdigit() and hashlib.sha256(token).hexdigest() in deny


def _mask(name: str, deny: frozenset[str]) -> str:
    """name with each denied word as ***, so a problem line doesn't repeat the word it found."""
    return NAME_TOKEN.sub(
        lambda m: "***" if _denied(m.group().lower().encode(), deny) else m.group(), name
    )


def _line(data: bytes, pos: int) -> int:
    return data.count(b"\n", 0, pos) + 1


def _unsafe(name: str) -> bool:
    return (
        name.startswith("/")
        or "\\" in name
        or re.match(r"[A-Za-z]:", name) is not None
        or ".." in name.split("/")
    )


def _scan(where: str, name: str, data: bytes, deny: frozenset[str]) -> list[str]:
    shown = _mask(name, deny)
    out = []
    if shown != name:
        out.append(f"{where}: {shown}: denied word in the name")
    lines = sorted({_line(data, m.start()) for m in HOME.finditer(data)})
    out += [f"{where}: {shown}:{n}: home-folder path" for n in lines]
    low = data.lower()
    for word in sorted(t for t in set(TOKEN.findall(low)) if _denied(t, deny)):
        first = next(m.start() for m in TOKEN.finditer(low) if m.group() == word)
        tag = hashlib.sha256(word).hexdigest()[:12]
        out.append(f"{where}: {shown}:{_line(data, first)}: denied word (sha256 {tag})")
    return out


def _member(where: str, name: str, data: bytes, allowed: bool, deny: frozenset[str]) -> list[str]:
    if not allowed:
        return [f"{where}: {_mask(name, deny)}: not on the allowlist"]
    if CACHE.search(name):
        return [f"{where}: {_mask(name, deny)}: cache file"]
    return _scan(where, name, data, deny)


def _sdist_allowed(rel: str) -> bool:
    if rel.startswith(SDIST_EXCLUDED):
        return False
    return rel in SDIST_FILES or rel.startswith(SDIST_DIRS)


def _read_sdist(path: Path, version: str, deny: frozenset[str]) -> tuple[list[str], bytes | None]:
    where, top = path.name, f"{NAME}-{version}/"
    problems, seen, meta = [], set(), None
    try:
        with tarfile.open(path, "r:gz") as tf:
            for m in tf:
                shown = _mask(m.name, deny)
                if _unsafe(m.name):
                    problems.append(f"{where}: {shown}: unsafe path")
                elif m.isdir():
                    continue
                elif not m.isfile():
                    problems.append(f"{where}: {shown}: not a regular file")
                elif not m.name.startswith(top):
                    problems.append(f"{where}: {shown}: outside {top}")
                else:
                    rel = m.name[len(top) :]
                    seen.add(rel)
                    f = tf.extractfile(m)
                    data = f.read() if f else b""
                    if rel == "PKG-INFO":
                        meta = data
                    problems += _member(where, rel, data, _sdist_allowed(rel), deny)
    except (tarfile.TarError, OSError, EOFError, zlib.error) as e:
        return problems + [f"{where}: can't be read ({type(e).__name__})"], None
    problems += [f"{where}: missing {r}" for r in SDIST_REQUIRED if r not in seen]
    return problems, meta


def _read_wheel(path: Path, version: str, deny: frozenset[str]) -> tuple[list[str], bytes | None]:
    where, info = path.name, f"{NAME}-{version}.dist-info/"
    problems, seen, meta = [], set(), None
    try:
        with zipfile.ZipFile(path) as zf:
            for zi in zf.infolist():
                # orig_filename is the name as stored: on Windows, filename has \ turned into /
                name = zi.orig_filename
                shown = _mask(name, deny)
                if _unsafe(name):
                    problems.append(f"{where}: {shown}: unsafe path")
                elif zi.is_dir():
                    continue
                elif (zi.external_attr >> 16) & 0o170000 == 0o120000:
                    problems.append(f"{where}: {shown}: not a regular file")
                else:
                    seen.add(name)
                    data = zf.read(zi)
                    if name == info + "METADATA":
                        meta = data
                    allowed = name.startswith((f"{NAME}/", info))
                    problems += _member(where, name, data, allowed, deny)
    except (zipfile.BadZipFile, OSError, EOFError, zlib.error) as e:
        return problems + [f"{where}: can't be read ({type(e).__name__})"], None
    required = (f"{NAME}/__init__.py", info + "METADATA", info + "licenses/LICENSE")
    problems += [f"{where}: missing {r}" for r in required if r not in seen]
    return problems, meta


def check(dist: Path, expect_version: str | None = None, deny: frozenset[str] = DENY) -> list[str]:
    """Every problem with the sdist and wheel in dist, one line each. Empty means they can ship."""
    dist = Path(dist)
    if not dist.is_dir():
        return [f"{dist}: not a folder"]
    sdists, wheels, problems = [], [], []
    for p in sorted(dist.iterdir()):
        if p.name == ".gitignore":  # uv build writes one into its output folder
            continue
        if p.name.endswith(".tar.gz"):
            sdists.append(p)
        elif p.name.endswith(".whl"):
            wheels.append(p)
        else:
            problems.append(f"{dist}: unexpected file {p.name}")
    if len(sdists) != 1:
        problems.append(f"{dist}: expected one sdist, found {len(sdists)}")
    if len(wheels) != 1:
        problems.append(f"{dist}: expected one wheel, found {len(wheels)}")
    if len(sdists) != 1 or len(wheels) != 1:
        return problems
    sdist, wheel = sdists[0], wheels[0]
    sm = re.fullmatch(rf"{NAME}-([^/]+)\.tar\.gz", sdist.name)
    wm = re.fullmatch(rf"{NAME}-([^-]+)-py3-none-any\.whl", wheel.name)
    if not sm:
        problems.append(f"{sdist.name}: not an sdist of {NAME}")
    if not wm:
        problems.append(f"{wheel.name}: not a pure-Python wheel of {NAME}")
    if not sm or not wm:
        return problems
    sv, wv = sm[1], wm[1]
    sp, smeta = _read_sdist(sdist, sv, deny)
    wp, wmeta = _read_wheel(wheel, wv, deny)
    problems += sp + wp
    for where, label, meta, v in (
        (sdist.name, "PKG-INFO", smeta, sv),
        (wheel.name, "METADATA", wmeta, wv),
    ):
        if meta is None:  # unreadable or missing, which is reported already
            continue
        got = VERSION.search(meta)
        if got is None:
            problems.append(f"{where}: {label} has no Version")
        elif got[1].decode() != v:
            problems.append(f"{where}: {label} says version {got[1].decode()}, the file name {v}")
    if sv != wv:
        problems.append(f"version mismatch: sdist {sv}, wheel {wv}")
    if expect_version is not None and sv != expect_version:
        problems.append(f"built {sv}, expected {expect_version}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("dist", nargs="?", type=Path, help="the folder uv build wrote")
    ap.add_argument("--expect-version", help="fail unless the build has this version")
    ap.add_argument("--digest", metavar="WORD", help="print the digest of WORD for DENY")
    args = ap.parse_args(argv)
    if args.digest is not None:
        print(digest(args.digest))
        return 0
    if args.dist is None:
        ap.error("the dist folder is required")
    problems = check(args.dist, args.expect_version)
    for line in problems:
        print(line)
    if problems:
        print(f"{len(problems)} problem(s) in {args.dist}")
        return 1
    names = sorted(p.name for p in args.dist.iterdir() if p.name != ".gitignore")
    print(f"OK: {' and '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
