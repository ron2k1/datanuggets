"""CITATION.cff against pyproject.toml. CITATION.cff isn't in the sdist, which leaves this folder
out, so this test runs only from a clone."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_citation_version_matches_pyproject():
    """A citation made from CITATION.cff must name the version that pyproject.toml releases."""
    with (ROOT / "pyproject.toml").open("rb") as f:
        want = tomllib.load(f)["project"]["version"]
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    # the top-level key only: the references' versions are indented
    found = re.search(r"^version:\s*[\"']?([^\"'\s]+)[\"']?(?:\s+#.*)?\s*$", cff, re.MULTILINE)
    assert found is not None, "CITATION.cff has no top-level version"
    assert found.group(1) == want
