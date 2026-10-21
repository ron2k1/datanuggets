"""Check a built sdist and wheel before they ship, the way R CMD check reads the built tarball.

    python scripts/check_dist.py dist [--expect-version 0.1.0]
    python scripts/check_dist.py --digest WORD

The dist folder must hold one sdist and one pure-Python wheel of the same version. Every file in
them must be on the allowlist below and not a cache file, and no file's name, bytes or archive
header may hold a home-folder path or a denied word. A zip inside them, such as numpy's .npz, is
opened and read the same way; any other compressed file fails, since its bytes can't be read. It
prints one line per problem and exits with 1 on any, including a folder or an archive it can't
read. --digest prints the digest that adds a word to DENY."""

from __future__ import annotations

import argparse
import hashlib
import io
import lzma
import re
import sys
import tarfile
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path

NAME = "datanuggets"

# The sdist's allowlist, relative to its top folder: pyproject.toml's only-include and exclude,
# and the PKG-INFO and .gitignore the build adds. It's kept apart from pyproject.toml on purpose,
# so that a change to only-include has to get past this too.
USER_GUIDES = (
    "docs/README.md",
    "docs/installation.md",
    "docs/quickstart.md",
    "docs/api.md",
    "docs/cli.md",
    "docs/limitations.md",
    "docs/compatibility.md",
    "docs/methodology.md",
    "docs/reproducibility.md",
    "examples/first_analysis.py",
)
SDIST_FILES = frozenset(
    {"PKG-INFO", ".gitignore", "pyproject.toml", "LICENSE", "README.md", *USER_GUIDES}
)
SDIST_DIRS = ("src/datanuggets/", "tests/", "r/")
SDIST_EXCLUDED = ("tests/scripts/",)
SDIST_REQUIRED = (
    "PKG-INFO",
    "pyproject.toml",
    "LICENSE",
    "README.md",
    "src/datanuggets/__init__.py",
    "tests/conftest.py",
    *USER_GUIDES,
)

# Caches and leftovers a build can sweep in from a working tree: Python's and its tools', numba's
# (cache=True writes .nbi and .nbc files beside the source), R's, editors' and the OS's
CACHE = re.compile(
    r"(?i)(^|/)(__pycache__|\.ipynb_checkpoints|\.hypothesis|\.mypy_cache|\.pytest_cache"
    r"|\.ruff_cache)/|\.(pyc|pyo|nbi|nbc|orig|rej)$|~$"
    r"|(^|/)(\.DS_Store|Thumbs\.db|desktop\.ini|\.coverage(\..*)?|\.Rhistory|\.RData)$"
)

# Home folders: on Windows, with up to four separators for paths escaped once or twice in JSON,
# and on macOS and Linux, where the path must start there, so a URL's /home/ or /Users/ isn't one
HOME = re.compile(
    rb"(?i:\b[a-z]:[\\/]{1,4}users[\\/]{1,4}[^\\/\s\"']+)"
    rb"|(?<![\w.~-])/(?:Users|home)/[^/\s\"']+"
)

# SHA-256 of the lowercased words no release may hold: tool and machine names from the
# development setup. Only the digests are kept, so this file doesn't publish the words it looks
# for. A word is a whole run of letters and digits, or a camelCase part or a run of letters or
# digits inside one, so it's found in snake_case, kebab-case and camelCase names and next to a
# digit, but not inside a longer word.
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
TOKEN = re.compile(rb"[a-z0-9]+")  # run on lowercased bytes
PART = re.compile(rb"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")  # run on the bytes as they are
NAME_TOKEN = re.compile(r"[A-Za-z0-9]+")
NAME_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
VERSION = re.compile(rb"^Version: *(\S+)\s*$", re.MULTILINE)

# Compressed formats other than zip, whose bytes a scan can't read
ZIP = (b"PK\x03\x04", b"PK\x05\x06")
COMPRESSED = (
    (b"\x1f\x8b", "gzip"),
    (b"\x89PNG", "png"),
    (b"%PDF", "pdf"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
)
BZIP2 = re.compile(rb"BZh[1-9]1AY&SY")
TAR_ERRORS = (tarfile.TarError, OSError, EOFError, zlib.error)
# a member with an unknown method raises NotImplementedError, an encrypted one RuntimeError
ZIP_ERRORS = (
    zipfile.BadZipFile,
    OSError,
    EOFError,
    zlib.error,
    lzma.LZMAError,
    NotImplementedError,
    RuntimeError,
    ValueError,
)


def digest(word: str) -> str:
    return hashlib.sha256(word.lower().encode()).hexdigest()


def _denied(token: bytes, deny: frozenset[str]) -> bool:
    """token is lowercase. No denied word is all digits, which skips most of a fixture's tokens."""
    return not token.isdigit() and hashlib.sha256(token).hexdigest() in deny


def _mask(name: str, deny: frozenset[str]) -> str:
    """name with each denied word as ***, so a problem line doesn't repeat the word it found."""

    def part(m: re.Match[str]) -> str:
        return "***" if _denied(m.group().lower().encode(), deny) else m.group()

    def token(m: re.Match[str]) -> str:
        return "***" if part(m) == "***" else NAME_PART.sub(part, m.group())

    return NAME_TOKEN.sub(token, name)


def _hits(data: bytes, deny: frozenset[str]) -> dict[bytes, int]:
    """Each denied word in data, with the offset of its first whole token or part."""
    sources = ((TOKEN, data.lower()), (PART, data))
    words = {t.lower() for rx, src in sources for t in set(rx.findall(src))}
    return {
        w: min(m.start() for rx, src in sources for m in rx.finditer(src) if m.group().lower() == w)
        for w in words
        if _denied(w, deny)
    }


def _line(data: bytes, pos: int) -> int:
    return data.count(b"\n", 0, pos) + 1


def _unsafe(name: str) -> bool:
    return (
        name.startswith("/")
        or "\\" in name
        or re.match(r"[A-Za-z]:", name) is not None
        or ".." in name.split("/")
    )


def _scan_bytes(where: str, label: str, data: bytes, deny: frozenset[str]) -> list[str]:
    """label is masked already."""
    lines = sorted({_line(data, m.start()) for m in HOME.finditer(data)})
    out = [f"{where}: {label}:{n}: home-folder path" for n in lines]
    for word, pos in sorted(_hits(data, deny).items()):
        tag = hashlib.sha256(word).hexdigest()[:12]
        out.append(f"{where}: {label}:{_line(data, pos)}: denied word (sha256 {tag})")
    return out


def _scan(
    where: str, name: str, data: bytes, deny: frozenset[str], nested: bool = False
) -> list[str]:
    shown = _mask(name, deny)
    out = [f"{where}: {shown}: denied word in the name"] if shown != name else []
    if data.startswith(ZIP):
        return out + _scan_zip(where, shown, data, deny, nested)
    kind = next((k for magic, k in COMPRESSED if data.startswith(magic)), None)
    if kind is None and BZIP2.match(data):
        kind = "bzip2"
    if kind is not None:
        return out + [f"{where}: {shown}: compressed ({kind}), which the checker doesn't open"]
    return out + _scan_bytes(where, shown, data, deny)


def _zip_header(where: str, label: str, zi: zipfile.ZipInfo, deny: frozenset[str]) -> list[str]:
    return _scan_bytes(where, f"{label} (zip header)", zi.comment + b"\n" + zi.extra, deny)


def _scan_zip(where: str, shown: str, data: bytes, deny: frozenset[str], nested: bool) -> list[str]:
    """A zip inside the sdist or wheel, such as numpy's .npz: every member, one level down."""
    if nested:
        return [f"{where}: {shown}: a zip inside a zip, which the checker doesn't open"]
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            out += _scan_bytes(where, f"{shown} (archive comment)", zf.comment, deny)
            for zi in zf.infolist():
                if not zi.is_dir():
                    name = f"{shown}[{zi.orig_filename}]"
                    out += _zip_header(where, _mask(name, deny), zi, deny)
                    out += _scan(where, name, zf.read(zi), deny, nested=True)
    except ZIP_ERRORS as e:
        return out + [f"{where}: {shown}: can't be read ({type(e).__name__})"]
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


def _pax(headers: Mapping[str, str], skip: Mapping[str, str]) -> bytes:
    """The PAX values not in skip, as bytes. The name keys are checked as names."""
    values = (v for k, v in headers.items() if k not in ("path", "linkpath") and skip.get(k) != v)
    return "\n".join(values).encode("utf-8", "surrogateescape")


def _read_sdist(path: Path, version: str, deny: frozenset[str]) -> tuple[list[str], bytes | None]:
    where, top = _mask(path.name, deny), f"{NAME}-{version}/"
    problems, seen, meta = [], set(), None
    try:
        with tarfile.open(path, "r:gz") as tf:
            for m in tf:
                shown = _mask(m.name, deny)
                # a member's PAX headers carry the archive's global ones too, which are read once
                header = _pax(m.pax_headers, tf.pax_headers)
                problems += _scan_bytes(where, f"{shown} (tar header)", header, deny)
                if m.uid or m.gid or m.uname or m.gname:
                    problems.append(f"{where}: {shown}: has an owner (uid, gid, uname or gname)")
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
            problems += _scan_bytes(where, "(archive header)", _pax(tf.pax_headers, {}), deny)
    except TAR_ERRORS as e:
        return problems + [f"{where}: can't be read ({type(e).__name__})"], None
    problems += [f"{where}: missing {r}" for r in SDIST_REQUIRED if r not in seen]
    return problems, meta


def _read_wheel(path: Path, version: str, deny: frozenset[str]) -> tuple[list[str], bytes | None]:
    where, info = _mask(path.name, deny), f"{NAME}-{version}.dist-info/"
    problems, seen, meta = [], set(), None
    try:
        with zipfile.ZipFile(path) as zf:
            problems += _scan_bytes(where, "(archive comment)", zf.comment, deny)
            for zi in zf.infolist():
                # orig_filename is the name as stored: on Windows, filename has \ turned into /
                name = zi.orig_filename
                shown = _mask(name, deny)
                problems += _zip_header(where, shown, zi, deny)
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
    except ZIP_ERRORS as e:
        return problems + [f"{where}: can't be read ({type(e).__name__})"], None
    required = (f"{NAME}/__init__.py", info + "METADATA", info + "licenses/LICENSE")
    problems += [f"{where}: missing {r}" for r in required if r not in seen]
    return problems, meta


def check(dist: Path, expect_version: str | None = None, deny: frozenset[str] = DENY) -> list[str]:
    """Every problem with the sdist and wheel in dist, one line each. Empty means they can ship."""
    dist = Path(dist)
    shown = _mask(str(dist), deny)
    if not dist.is_dir():
        return [f"{shown}: not a folder"]
    sdists, wheels, problems = [], [], []
    for p in sorted(dist.iterdir()):
        if p.name == ".gitignore":  # uv build writes one into its output folder
            continue
        if p.name.endswith(".tar.gz"):
            sdists.append(p)
        elif p.name.endswith(".whl"):
            wheels.append(p)
        else:
            problems.append(f"{shown}: unexpected file {_mask(p.name, deny)}")
    if len(sdists) != 1:
        problems.append(f"{shown}: expected one sdist, found {len(sdists)}")
    if len(wheels) != 1:
        problems.append(f"{shown}: expected one wheel, found {len(wheels)}")
    if len(sdists) != 1 or len(wheels) != 1:
        return problems
    sdist, wheel = sdists[0], wheels[0]
    sm = re.fullmatch(rf"{NAME}-([^/]+)\.tar\.gz", sdist.name)
    wm = re.fullmatch(rf"{NAME}-([^-]+)-py3-none-any\.whl", wheel.name)
    if not sm:
        problems.append(f"{_mask(sdist.name, deny)}: not an sdist of {NAME}")
    if not wm:
        problems.append(f"{_mask(wheel.name, deny)}: not a pure-Python wheel of {NAME}")
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
            problems.append(f"{_mask(where, deny)}: {label} has no Version")
        elif got[1].decode() != v:
            said = _mask(got[1].decode(), deny)
            problems.append(f"{_mask(where, deny)}: {label} says version {said}, the file name {v}")
    if sv != wv:
        problems.append(f"version mismatch: sdist {_mask(sv, deny)}, wheel {_mask(wv, deny)}")
    if expect_version is not None and sv != expect_version:
        problems.append(f"built {_mask(sv, deny)}, expected {expect_version}")
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
        print(f"{len(problems)} problem(s) in {_mask(str(args.dist), DENY)}")
        return 1
    names = sorted(p.name for p in args.dist.iterdir() if p.name != ".gitignore")
    print(f"OK: {' and '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
