"""The version is written once, in pyproject.toml, and the package reads it from its metadata."""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import datanuggets

# tests/ sits beside pyproject.toml in a clone and in the sdist
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject():
    with PYPROJECT.open("rb") as f:
        want = tomllib.load(f)["project"]["version"]
    assert datanuggets.__version__ == want == version("datanuggets")


def test_init_holds_no_version_literal():
    # a second copy would agree until the first release that forgets to bump it
    source = Path(datanuggets.__file__).read_text(encoding="utf-8")
    assert not re.search(r"__version__\s*=\s*['\"]", source)
