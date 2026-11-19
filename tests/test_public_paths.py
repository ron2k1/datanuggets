"""Every name a user or a script imports by module stays importable from that module: the names
docs/api.md documents, every __all__, and the private names code outside the package imports. A
split that moves a name must re-export it from its old module, keeping one implementation of
it."""

import ast
import importlib
import re
from pathlib import Path

import pytest

import datanuggets

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "docs" / "api.md"
# The source of the package the tests import: src/datanuggets in a checkout, and the installed
# wheel's when package.yml runs the sdist's tests with src/ deleted
PACKAGE = Path(datanuggets.__file__).resolve().parent

# api.md's sections whose `### `name`` headings document a name of one module
SECTION_MODULES = {
    "Creating and refining nuggets": "datanuggets",
    "Weighted summaries": "datanuggets",
    "Clustering and prediction": "datanuggets",
    "Weighted PCA": "datanuggets",
    "Nugget analysis": "datanuggets",
    "Result objects": "datanuggets",
    "Warnings": "datanuggets",
    "Plotting": "datanuggets.plot",
}

# Private names, names of private modules and names no __all__ lists that code outside the
# package uses: benchmarks/covertype.py, benchmarks/wcluster.py and scripts/parity_report.py.
# Those files import some by name and read the others from a module they import.
OUTSIDE = {
    "datanuggets._clustering.hartigan_wong": ("kmns",),
    "datanuggets._compat.arithmetic": ("r_mean", "r_cov"),
    "datanuggets._compat.linalg": ("r_matmul",),
    "datanuggets._compat.moments": ("group_moments",),
    "datanuggets._compat.rfast": ("cross_distances", "nearest_index"),
    "datanuggets._random.r_draws": ("RDraws",),
    "datanuggets._random.replay": ("ReplayDraws",),
    "datanuggets.nuggets.centers": ("select_centers",),
    "datanuggets.nuggets.complexity": ("_ac_state",),
    "datanuggets.nuggets.create": ("nearest_index",),
    "datanuggets.wcluster.predict": ("_wdist_mat",),
    "datanuggets.wcluster.wpca": ("_wpca_parts",),
}

HEADING = re.compile(r"^(#{2,3}) (.+?)\s*$")
MODULE_HEADING = re.compile(r"^`(datanuggets(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`$")
NAME_HEADING = re.compile(r"^`([A-Za-z_][A-Za-z0-9_]*)`$")
BLOCK_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\b")


def _documented() -> dict[str, set[str]]:
    """Each module api.md documents names of, and those names, from its name headings and the
    ```text blocks of signatures under a heading that is a module's name in backticks."""
    found: dict[str, set[str]] = {}
    section = heading = None
    fence = None  # the fence's language while inside one
    for line in API.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fence = line[3:].strip() if fence is None else None
            continue
        if fence is not None:
            module = heading.strip("`") if heading and MODULE_HEADING.match(heading) else None
            name = BLOCK_LINE.match(line)
            if fence == "text" and module and name:
                found.setdefault(module, set()).add(name.group(1))
            continue
        m = HEADING.match(line)
        if m:
            level, text = m.groups()
            if level == "##":
                section, heading = text, None
            else:
                heading = text
                name = NAME_HEADING.match(text)
                if name and section in SECTION_MODULES:
                    found.setdefault(SECTION_MODULES[section], set()).add(name.group(1))
    return found


DOCUMENTED = _documented()


def _all_modules() -> list[str]:
    """Every module of the package whose source assigns __all__, found without importing it."""
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assigned = {
            t.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        }
        if "__all__" in assigned:
            parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
            names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def _check(module: str, names) -> None:
    """Every name imports from module, as the one object its defining module holds."""
    mod = importlib.import_module(module)
    missing = [name for name in names if not hasattr(mod, name)]
    assert missing == [], f"{module} no longer has {missing}"
    for name in names:
        obj = getattr(mod, name)
        home = getattr(obj, "__module__", None)
        ours = isinstance(home, str) and home.split(".")[0] == "datanuggets"
        if ours and callable(obj) and getattr(obj, "__name__", None) == name:
            assert getattr(importlib.import_module(home), name) is obj, (module, name, home)


def test_the_docs_are_read_for_every_module_they_document():
    """The reader of api.md finds names for each module the guide documents. It takes name
    headings only from the sections SECTION_MODULES names, so a new section of name headings
    needs an entry there."""
    assert sorted(DOCUMENTED) == sorted(
        {
            "datanuggets",
            "datanuggets.colors",
            "datanuggets.filters",
            "datanuggets.plot",
            "datanuggets.projection",
        }
    )


@pytest.mark.parametrize("module", sorted(DOCUMENTED))
def test_documented_names_import_from_their_documented_module(module):
    _check(module, sorted(DOCUMENTED[module]))


ALL_MODULES = _all_modules()


def test_the_source_is_read_for_every_module_that_assigns_all():
    """The reader of the package's source fails closed: it finds the modules whose __all__ the
    next test checks, so a source tree it can't find can't leave every __all__ unchecked."""
    assert {
        "datanuggets",
        "datanuggets.nuggets",
        "datanuggets.plot",
        "datanuggets.wcluster",
    } <= set(ALL_MODULES)


@pytest.mark.parametrize("module", ALL_MODULES)
def test_every_all_name_imports_from_its_module(module):
    names = importlib.import_module(module).__all__
    assert names, module
    _check(module, names)


@pytest.mark.parametrize("module", sorted(OUTSIDE))
def test_private_names_code_outside_the_package_imports_stay(module):
    _check(module, OUTSIDE[module])
