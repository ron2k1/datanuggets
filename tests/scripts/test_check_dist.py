"""The release checker, scripts/check_dist.py, on small sdists and wheels built here.

The checker isn't part of the package, so the sdist leaves this folder out and these tests run
only from a clone. The denied words here are made up: a test that held a real one would put it in
the repo the checker guards."""

import importlib.util
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_dist.py"
_spec = importlib.util.spec_from_file_location("check_dist", SCRIPT)
assert _spec is not None and _spec.loader is not None
check_dist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_dist)

METADATA = "Metadata-Version: 2.4\nName: datanuggets\nVersion: {v}\n"
SDIST = {
    "PKG-INFO": METADATA,
    "pyproject.toml": '[project]\nname = "datanuggets"\n',
    "LICENSE": "GPL-2.0\n",
    "README.md": "# datanuggets\n",
    "src/datanuggets/__init__.py": "",
    "tests/conftest.py": "",
    "tests/fixtures/case.json": '{"x": [1.5, 2e-300]}',
    **dict.fromkeys(check_dist.USER_GUIDES, "# User guide\n"),
}
WHEEL = {
    "datanuggets/__init__.py": "",
    "datanuggets-{v}.dist-info/METADATA": METADATA,
    "datanuggets-{v}.dist-info/licenses/LICENSE": "GPL-2.0\n",
}
FAKE = frozenset({check_dist.digest("zorblax")})


def _bytes(data):
    return data if isinstance(data, bytes) else data.encode()


def _tar(path, files, links=(), attrs=None, pax=None):
    """attrs sets TarInfo fields by member name; pax is the archive's global PAX header."""
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT, pax_headers=pax or {}) as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(_bytes(data))
            for field, value in (attrs or {}).get(name, {}).items():
                setattr(info, field, value)
            tf.addfile(info, io.BytesIO(_bytes(data)))
        for name in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = "../../etc/passwd"
            tf.addfile(info)


def _zip(path, files, compression=zipfile.ZIP_STORED, comment=b"", attrs=None):
    """attrs sets ZipInfo fields, such as comment and extra, by member name."""
    with zipfile.ZipFile(path, "w", compression) as zf:
        zf.comment = comment
        for name, data in files.items():
            # ZipInfo turns os.sep into / on Windows, so a backslash name is set after it
            info = zipfile.ZipInfo("x")
            info.filename = name
            info.compress_type = compression
            for field, value in (attrs or {}).get(name, {}).items():
                setattr(info, field, value)
            zf.writestr(info, _bytes(data))


def _zip_bytes(files, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    _zip(buf, files, compression)
    return buf.getvalue()


def _patch_central(path, name, offset, value):
    """Overwrite 2 bytes of name's central directory entry: 8 is the flags, 10 the method."""
    data = bytearray(path.read_bytes())
    entry = data.find(b"PK\x01\x02")
    while entry != -1:
        size = int.from_bytes(data[entry + 28 : entry + 30], "little")
        if data[entry + 46 : entry + 46 + size] == name.encode():
            data[entry + offset : entry + offset + 2] = value.to_bytes(2, "little")
            path.write_bytes(bytes(data))
            return
        entry = data.find(b"PK\x01\x02", entry + 4)
    raise AssertionError(f"{name} not in {path}")


def build(dist, version="1.0", wheel_version=None, sdist=None, wheel=None, drop=()):
    """A good sdist and wheel in dist, with members changed, added or dropped by name. sdist
    names are relative to its top folder; a value of None drops the member."""
    wv = wheel_version or version
    dist.mkdir(exist_ok=True)
    s = {k: v.replace("{v}", version) for k, v in SDIST.items()} | (sdist or {})
    w = {k.replace("{v}", wv): v.replace("{v}", wv) for k, v in WHEEL.items()} | (wheel or {})
    s = {f"datanuggets-{version}/{k}": v for k, v in s.items() if k not in drop and v is not None}
    w = {k: v for k, v in w.items() if k not in drop and v is not None}
    _tar(dist / f"datanuggets-{version}.tar.gz", s)
    _zip(dist / f"datanuggets-{wv}-py3-none-any.whl", w)
    return dist


def has(problems, *parts):
    return any(all(p in line for p in parts) for line in problems)


def test_clean_build_passes(tmp_path):
    assert check_dist.check(build(tmp_path / "dist")) == []
    assert check_dist.check(build(tmp_path / "dist2"), expect_version="1.0") == []


def test_source_archive_requires_user_guides_and_example(tmp_path):
    got = check_dist.check(build(tmp_path / "dist", drop=check_dist.USER_GUIDES))
    for member in check_dist.USER_GUIDES:
        assert has(got, ".tar.gz", member, "missing"), member
    assert len(got) == len(check_dist.USER_GUIDES)


def test_files_outside_the_allowlist(tmp_path):
    got = check_dist.check(
        build(
            tmp_path / "dist",
            sdist={
                "docs/plan.md": "x",
                ".scratch/sdd/notes.md": "x",
                "benchmarks/run.py": "x",
                "uv.lock": "x",
                "tests/scripts/test_check_dist.py": "x",
                "src/other/__init__.py": "x",
            },
            wheel={"tests/test_x.py": "x", "other/__init__.py": "x"},
        )
    )
    for member in (
        "docs/plan.md",
        ".scratch/sdd/notes.md",
        "benchmarks/run.py",
        "uv.lock",
        "tests/scripts/test_check_dist.py",
        "src/other/__init__.py",
    ):
        assert has(got, ".tar.gz", member, "not on the allowlist"), member
    for member in ("tests/test_x.py", "other/__init__.py"):
        assert has(got, ".whl", member, "not on the allowlist"), member
    assert len(got) == 8


def test_cache_files(tmp_path):
    got = check_dist.check(
        build(
            tmp_path / "dist",
            sdist={
                "src/datanuggets/__pycache__/create.cpython-314.pyc": b"\0",
                "tests/.DS_Store": b"\0",
                "tests/OLD.PYC": b"\0",
                "tests/__PYCACHE__/a.py": b"\0",
                "tests/.hypothesis/examples/x": b"\0",
                "tests/.mypy_cache/x.json": b"\0",
                "tests/.coverage": b"\0",
                "r/.Rhistory": b"\0",
                "r/setup.R.orig": b"\0",
                "r/setup.R~": b"\0",
            },
            wheel={
                "datanuggets/nuggets/create.create_dn-12.py314.nbi": b"\0",
                "datanuggets/nuggets/create.create_dn-12.py314.1.nbc": b"\0",
                "datanuggets/old.pyc": b"\0",
            },
        )
    )
    for member in (
        "__pycache__/create.cpython-314.pyc",
        "tests/.DS_Store",
        "py314.nbi",
        "py314.1.nbc",
        "datanuggets/old.pyc",
        "tests/OLD.PYC",
        "tests/__PYCACHE__/a.py",
        "tests/.hypothesis/examples/x",
        "tests/.mypy_cache/x.json",
        "tests/.coverage",
        "r/.Rhistory",
        "r/setup.R.orig",
        "r/setup.R~",
    ):
        assert has(got, member, "cache file"), member
    assert len(got) == 13


def test_home_folder_paths(tmp_path):
    lines = [
        "# datanuggets",
        r"run C:\Users\someone\repo\x.py",
        r'{"path": "D:\\Users\\someone\\x"}',
        "c:/users/someone/x",
        "/Users/someone/repo",
        "/home/someone/repo",
        # without a trailing separator, and a Windows path JSON-escaped twice
        '{"cwd": "/Users/someone"}',
        "HOME=/home/someone",
        '"/home/someone"',
        r'"C:\\\\Users\\\\someone\\\\x"',
        # not home folders
        "https://api.github.com/users/someone/repos",
        "https://example.com/home/page and https://example.com/Users/page",
        "/usr/local/lib",
        "C:/Program Files/R",
    ]
    got = check_dist.check(build(tmp_path / "dist", sdist={"README.md": "\n".join(lines)}))
    for line in range(2, 11):
        assert has(got, f"README.md:{line}:", "home-folder path"), line
    assert len(got) == 9


def test_denied_words(tmp_path):
    dist = build(
        tmp_path / "dist",
        sdist={"tests/test_y.py": "a = 1\n# made with ZORBLAX\nb = 2\n# zorblax_helper again\n"},
        wheel={
            "datanuggets/zorblax_notes.py": "",
            "datanuggets/fine.py": "zorblaxes xzorblax Zorblaxes\n",
        },
    )
    got = check_dist.check(dist, deny=FAKE)
    assert has(got, ".tar.gz", "tests/test_y.py:2:", "denied word")
    assert has(got, ".whl", "datanuggets/***_notes.py", "denied word in the name")
    assert not has(got, "fine.py")
    # one line per member and word, at its first line
    assert len(got) == 2
    # a denied word is shown by digest and masked in names, so a CI log doesn't repeat it
    assert not has(got, "zorblax") and not has(got, "ZORBLAX")


def test_denied_words_are_digests():
    assert len(check_dist.DENY) >= 10
    assert all(len(d) == 64 and set(d) <= set("0123456789abcdef") for d in check_dist.DENY)
    assert check_dist.digest("Zorblax") == check_dist.digest("zorblax")


def test_required_members(tmp_path):
    got = check_dist.check(
        build(
            tmp_path / "dist",
            drop=("tests/conftest.py", "datanuggets-1.0.dist-info/licenses/LICENSE"),
        )
    )
    assert has(got, ".tar.gz", "missing tests/conftest.py")
    assert has(got, ".whl", "missing datanuggets-1.0.dist-info/licenses/LICENSE")
    assert len(got) == 2


def test_versions_agree(tmp_path):
    got = check_dist.check(build(tmp_path / "a", wheel_version="1.1"))
    assert has(got, "sdist 1.0", "wheel 1.1")
    got = check_dist.check(
        build(tmp_path / "b", sdist={"PKG-INFO": METADATA.replace("{v}", "1.0.1")})
    )
    assert has(got, "PKG-INFO", "1.0.1", "1.0")
    got = check_dist.check(build(tmp_path / "c"), expect_version="2.0")
    assert has(got, "built 1.0", "expected 2.0")
    got = check_dist.check(build(tmp_path / "d", sdist={"PKG-INFO": "Name: datanuggets\n"}))
    assert has(got, "PKG-INFO", "no Version")


def test_unsafe_members(tmp_path):
    dist = build(tmp_path / "dist")
    _tar(
        dist / "datanuggets-1.0.tar.gz",
        {f"datanuggets-1.0/{k}": v.replace("{v}", "1.0") for k, v in SDIST.items()}
        | {"datanuggets-1.0/../evil": "x", "/etc/evil": "x", "other-1.0/x": "x"},
        links=["datanuggets-1.0/tests/link"],
    )
    _zip(
        dist / "datanuggets-1.0-py3-none-any.whl",
        {k.replace("{v}", "1.0"): v.replace("{v}", "1.0") for k, v in WHEEL.items()}
        | {"../evil": "x", "/abs/evil": "x", "datanuggets\\win.py": "x"},
    )
    got = check_dist.check(dist)
    for member in (
        "datanuggets-1.0/../evil",
        "/etc/evil",
        "../evil",
        "/abs/evil",
        "datanuggets\\win.py",
    ):
        assert has(got, member, "unsafe path"), member
    assert has(got, "other-1.0/x", "outside datanuggets-1.0/")
    assert has(got, "tests/link", "not a regular file")
    assert len(got) == 7


def test_dist_folder_shape(tmp_path):
    assert has(check_dist.check(tmp_path / "missing"), "not a folder")
    (tmp_path / "empty").mkdir()
    got = check_dist.check(tmp_path / "empty")
    assert has(got, "one sdist", "found 0") and has(got, "one wheel", "found 0")
    dist = build(tmp_path / "two")
    (dist / "datanuggets-1.0-cp314-cp314-win_amd64.whl").write_bytes(b"")
    assert has(check_dist.check(dist), "one wheel", "found 2")
    dist = build(tmp_path / "extra")
    (dist / "notes.txt").write_text("x")
    (dist / ".gitignore").write_text("*")
    got = check_dist.check(dist)
    assert has(got, "unexpected file notes.txt") and not has(got, ".gitignore")
    assert len(got) == 1
    dist = tmp_path / "platform"
    dist.mkdir()
    _tar(dist / "datanuggets-1.0.tar.gz", {})
    _zip(dist / "datanuggets-1.0-cp314-cp314-win_amd64.whl", {})
    assert has(check_dist.check(dist), "win_amd64", "not a pure-Python wheel")


def test_unreadable_archives(tmp_path):
    dist = build(tmp_path / "dist")
    sdist = dist / "datanuggets-1.0.tar.gz"
    sdist.write_bytes(sdist.read_bytes()[:40])
    (dist / "datanuggets-1.0-py3-none-any.whl").write_bytes(b"PK\x03\x04 not a zip")
    got = check_dist.check(dist)
    assert has(got, ".tar.gz", "can't be read") and has(got, ".whl", "can't be read")


def test_denied_words_inside_names(tmp_path):
    dist = build(
        tmp_path / "dist",
        wheel={
            "datanuggets/a.py": "ZorblaxHelper = 1\n",
            "datanuggets/b.py": "x = 1\nhost = 'zorblax2'\n",
            "datanuggets/ZorblaxTools.py": "",
            "datanuggets/c.py": "Zorblaxes zorblaxes xzorblax ZORBLAXES\n",
        },
    )
    got = check_dist.check(dist, deny=FAKE)
    assert has(got, "datanuggets/a.py:1:", "denied word")
    assert has(got, "datanuggets/b.py:2:", "denied word")
    assert has(got, "datanuggets/***Tools.py", "denied word in the name")
    assert not has(got, "c.py") and not has(got, "orblax")
    assert len(got) == 3


def test_compressed_members(tmp_path):
    # numpy's savez_compressed writes deflated zips, which a scan of the raw bytes can't read
    npz = _zip_bytes({"arr_0.npy": "x = /home/someone/data", "zorblax.npy": "1", "ok.npy": "1"})
    got = check_dist.check(
        build(
            tmp_path / "dist",
            sdist={
                "tests/fixtures/a.npz": npz,
                "tests/fixtures/nest.npz": _zip_bytes({"deep.npz": _zip_bytes({"x": "y"})}),
                "tests/fixtures/broken.npz": b"PK\x03\x04 not a zip",
                "tests/fixtures/b.gz": b"\x1f\x8b\x08\x00rest",
                "tests/fixtures/c.png": b"\x89PNG\r\n\x1a\nrest",
                "tests/fixtures/d.pdf": b"%PDF-1.7 rest",
            },
            wheel={"datanuggets/data.zip": _zip_bytes({"n.txt": r"C:\Users\someone\x"})},
        ),
        deny=FAKE,
    )
    assert has(got, ".tar.gz", "tests/fixtures/a.npz[arr_0.npy]:1:", "home-folder path")
    assert has(got, "tests/fixtures/a.npz[***.npy]", "denied word in the name")
    assert not has(got, "ok.npy")
    assert has(got, "nest.npz[deep.npz]", "a zip inside a zip")
    assert has(got, "broken.npz", "can't be read")
    for member, kind in (("b.gz", "gzip"), ("c.png", "png"), ("d.pdf", "pdf")):
        assert has(got, member, f"compressed ({kind})"), member
    assert has(got, ".whl", "datanuggets/data.zip[n.txt]:1:", "home-folder path")
    assert len(got) == 8


def test_archive_metadata(tmp_path):
    dist = build(tmp_path / "dist")
    _tar(
        dist / "datanuggets-1.0.tar.gz",
        {f"datanuggets-1.0/{k}": v.replace("{v}", "1.0") for k, v in SDIST.items()},
        attrs={
            "datanuggets-1.0/LICENSE": {"uid": 1000, "uname": "someone"},
            "datanuggets-1.0/README.md": {"pax_headers": {"comment": "made in /home/someone/x"}},
        },
        pax={"comment": "zorblax"},
    )
    _zip(
        dist / "datanuggets-1.0-py3-none-any.whl",
        {k.replace("{v}", "1.0"): v.replace("{v}", "1.0") for k, v in WHEEL.items()},
        comment=b"zorblax",
        attrs={
            "datanuggets/__init__.py": {
                "comment": rb"C:\Users\someone",
                "extra": b"\xfe\xca\x07\x00zorblax",
            }
        },
    )
    got = check_dist.check(dist, deny=FAKE)
    assert has(got, ".tar.gz", "LICENSE", "has an owner")
    assert has(got, ".tar.gz", "README.md (tar header)", "home-folder path")
    assert has(got, ".tar.gz", "(archive header)", "denied word")
    assert has(got, ".whl", "(archive comment)", "denied word")
    assert has(got, ".whl", "datanuggets/__init__.py (zip header)", "home-folder path")
    assert has(got, ".whl", "datanuggets/__init__.py (zip header)", "denied word")
    # the global header's keys show up in every member's, but are reported once
    assert len(got) == 6


def test_unreadable_wheel_members(tmp_path):
    for flag_or_method, value in ((10, 99), (8, 1)):  # an unknown method, then encrypted
        dist = build(tmp_path / f"dist{value}", wheel={"datanuggets/a.py": "x"})
        _patch_central(
            dist / "datanuggets-1.0-py3-none-any.whl", "datanuggets/a.py", flag_or_method, value
        )
        got = check_dist.check(dist)
        assert has(got, ".whl", "can't be read"), value


def test_dist_names_are_masked(tmp_path):
    dist = build(tmp_path / "dist")
    (dist / "zorblax.txt").write_text("x")
    got = check_dist.check(dist, deny=FAKE)
    assert has(got, "unexpected file ***.txt")
    other = tmp_path / "other"
    other.mkdir()
    _tar(other / "zorblax-1.0.tar.gz", {})
    _zip(other / "zorblax-1.0-py3-none-any.whl", {})
    got += check_dist.check(other, deny=FAKE)
    assert has(got, "***-1.0.tar.gz", "not an sdist")
    assert has(got, "***-1.0-py3-none-any.whl", "not a pure-Python wheel")
    assert not has(got, "zorblax")


def test_command_line(tmp_path):
    def run(*args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
        )

    good = build(tmp_path / "good")
    out = run(str(good), "--expect-version", "1.0")
    assert out.returncode == 0 and out.stdout.startswith("OK"), out.stdout + out.stderr
    bad = build(tmp_path / "bad", sdist={"docs/x.md": "x"})
    out = run(str(bad))
    assert out.returncode == 1 and "docs/x.md" in out.stdout
    assert run(str(good), "--expect-version", "1.1").returncode == 1
    out = run("--digest", "Zorblax")
    assert out.returncode == 0 and out.stdout.strip() == check_dist.digest("zorblax")
