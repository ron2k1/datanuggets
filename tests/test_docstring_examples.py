"""Each ported function's docstring has an Examples section, as R's help pages do, and the
examples run and print what the docstring shows."""

import doctest
import inspect
import sys

import pytest

import datanuggets

FUNCTIONS = sorted(n for n in datanuggets.__all__ if inspect.isfunction(getattr(datanuggets, n)))
OPTIONAL = ("pandas", "polars", "sklearn", "matplotlib", "tqdm")


def test_every_ported_function_is_listed():
    assert FUNCTIONS == [
        "arithmetic_complexity",
        "cluster_predict",
        "create_dn",
        "create_dn_centers",
        "cut_tree",
        "distw",
        "dn_cluster_predict",
        "dn_whclust",
        "dn_wkmeans",
        "dn_wpca",
        "get_dn_means",
        "refine_dn",
        "whclust",
        "wkmeans",
        "wmean",
        "wpca",
        "wss",
        "wwcss",
    ]


def _doctests(fn, name):
    return doctest.DocTestFinder(recurse=False).find(fn, name, globs={})


class _NotInstalled:
    """A meta path finder that reports each optional package as missing. The error names the
    top-level package, as it does on a plain install. importlib.util.find_spec raises for these
    names too, where a plain install returns None."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        """Raise for an optional package. Return None for the rest, so the next finder runs."""
        root = name.split(".")[0]
        if root in OPTIONAL:
            raise ModuleNotFoundError(f"No module named {root!r}", name=root)


def _block_optional(monkeypatch):
    """Make each optional package raise ImportError on import, as on a plain install. That holds
    for an import in an example and for one in the function it calls. Loaded optional modules
    leave sys.modules, submodules included, because "from tqdm.auto import tqdm" takes a loaded
    tqdm.auto without looking at tqdm. The finder then refuses a fresh import. A None entry in
    sys.modules blocks an import too, but the error then names the submodule. Python 3.15 imports
    that submodule's parent to format the error, and the blocked parent raises a second error
    out of the doctest runner. datanuggets.plot leaves sys.modules too once a test module has
    loaded it. Otherwise "from datanuggets import plot" returns the loaded module without
    importing matplotlib."""
    for m in list(sys.modules):
        if m.split(".")[0] in OPTIONAL or m.split(".")[:2] == ["datanuggets", "plot"]:
            monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(sys, "meta_path", [_NotInstalled, *sys.meta_path])
    monkeypatch.delattr(datanuggets, "plot", raising=False)


def _failures(tests):
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for t in tests:
        runner.run(t)
    return runner.failures


@pytest.mark.parametrize("name", FUNCTIONS)
def test_docstring_examples_run(name, monkeypatch):
    fn = getattr(datanuggets, name)
    # cleandoc: Python 3.13 and later strip a docstring's indentation when compiling, 3.12 keeps it
    doc = inspect.cleandoc(fn.__doc__)
    assert "\nExamples\n--------\n" in doc, f"{name} has no Examples section"
    tests = _doctests(fn, name)
    examples = [e for t in tests for e in t.examples]
    assert examples, f"{name} has an Examples section with no >>> line"
    assert any(e.want for e in examples), f"{name}'s examples print nothing to compare"
    # The examples have to run on a plain install, so they may not import an optional package.
    _block_optional(monkeypatch)
    failures = _failures(tests)
    assert failures == 0, f"{failures} example line(s) of {name} failed"


@pytest.mark.parametrize(
    ("package", "line"),
    [
        ("pandas", "import pandas"),
        ("tqdm", "from tqdm.auto import tqdm"),
        ("pandas", "_ = DataNuggets([[0.0]], [1.0], [0.0], [0]).to_frame()"),
        ("matplotlib", "from datanuggets import plot"),
    ],
)
def test_an_example_that_imports_an_optional_package_fails(package, line, monkeypatch):
    """The check above can fail. Each line runs with its package installed and fails once the
    package is blocked. The third line imports pandas inside the package, in to_frame. The last
    line reaches matplotlib through datanuggets.plot, which the first run loads."""
    pytest.importorskip(package)

    def fake():
        pass

    fake.__doc__ = f"Examples\n--------\n>>> from datanuggets import DataNuggets\n>>> {line}\n"
    tests = _doctests(fake, "fake")
    assert _failures(tests) == 0
    _block_optional(monkeypatch)
    assert _failures(tests) == 1
