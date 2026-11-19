"""The package's import graph, read from its source with every import counted: those inside
functions and under TYPE_CHECKING too, since each is an edge a later change can turn into a
load-time one. The graph has no cycles, and only the modules that draw reach matplotlib or
datanuggets.plot, so the numerical core never depends on a plotting library."""

import ast
from pathlib import Path

import datanuggets

# The source of the package the tests import: src/datanuggets in a checkout, and the installed
# wheel's when package.yml runs the sdist's tests with src/ deleted
PACKAGE = Path(datanuggets.__file__).resolve().parent

# The modules that draw, or run something that draws, outside the plotting package. Only
# datanuggets.plot (every one of its modules) draws, so there are none. No other module may import
# matplotlib or datanuggets.plot, directly or through another module.
DRAWS: set[str] = set()
PLOT = "datanuggets.plot"


def _modules() -> dict[str, tuple[Path, bool]]:
    """Each module's dotted name, its file and whether it is a package's __init__."""
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
        package = parts[-1] == "__init__"
        found[".".join(parts[:-1] if package else parts)] = (path, package)
    return found


def _graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Each module's imports of the package's modules, and its imports of other top-level
    packages. An imported name that is a submodule counts as that module; any other counts as the
    module it is imported from."""
    modules = _modules()
    inside, outside = {}, {}
    for name, (path, package) in modules.items():
        edges, others = set(), set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    here = name.split(".") if package else name.split(".")[:-1]
                    here = here[: len(here) - node.level + 1]
                    base = ".".join([*here, *([node.module] if node.module else [])])
                targets = [
                    f"{base}.{a.name}" if f"{base}.{a.name}" in modules else base
                    for a in node.names
                ]
            else:
                continue
            for target in targets:
                if target.split(".")[0] != "datanuggets":
                    others.add(target.split(".")[0])
                    continue
                while target not in modules:
                    target = target.rsplit(".", 1)[0]
                if target != name:
                    edges.add(target)
        inside[name], outside[name] = edges, others
    return inside, outside


INSIDE, OUTSIDE = _graph()


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """The graph's strongly connected components of more than one module (Tarjan's)."""
    index, low, stack, on, found = {}, {}, [], set(), []

    def visit(v):
        index[v] = low[v] = len(index)
        stack.append(v)
        on.add(v)
        for w in sorted(graph[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                found.append(sorted(component))

    for v in sorted(graph):
        if v not in index:
            visit(v)
    return found


def _draws_directly(name: str) -> bool:
    return "matplotlib" in OUTSIDE[name] or any(
        t == PLOT or t.startswith(PLOT + ".") for t in INSIDE[name]
    )


def _reaching_draw() -> set[str]:
    """The modules that import matplotlib or datanuggets.plot, or import one that does."""
    reach = {name for name in INSIDE if _draws_directly(name)}
    grown = True
    while grown:
        grown = False
        for name, edges in INSIDE.items():
            if name not in reach and edges & reach:
                reach.add(name)
                grown = True
    return reach


def test_the_graph_is_read_from_the_source():
    """The reader fails closed: it finds the modules and the edges the checks rest on, so a
    reader that silently found nothing can't pass them."""
    assert {"datanuggets", "datanuggets.projection", "datanuggets.plot"} <= set(INSIDE)
    assert "datanuggets.projection" in INSIDE["datanuggets.plot.pca"]
    assert "datanuggets.wcluster.cutree" in INSIDE["datanuggets.colors"]  # in a function
    assert "matplotlib" in OUTSIDE["datanuggets.plot.points"]
    assert "pandas" in OUTSIDE["datanuggets.result"]  # in a function
    assert _cycles({"a": {"b"}, "b": {"a"}, "c": {"a"}}) == [["a", "b"]]


def test_the_package_has_no_import_cycles():
    assert _cycles(INSIDE) == []


def test_only_the_modules_that_draw_reach_matplotlib_or_plot():
    reach = _reaching_draw()
    assert {PLOT, "datanuggets.plot.pca"} <= reach
    core = sorted(n for n in reach if n not in DRAWS and n != PLOT and not n.startswith(PLOT + "."))
    assert core == [], f"these modules reach matplotlib or datanuggets.plot: {core}"


def test_the_core_and_helper_modules_import_no_drawing_module():
    """datanuggets, what import datanuggets loads, and the helper modules colors, filters and
    projection reach no module that draws, through any import."""
    reach = _reaching_draw()
    assert {
        "datanuggets",
        "datanuggets.nuggets",
        "datanuggets.wcluster",
        "datanuggets.colors",
        "datanuggets.filters",
        "datanuggets.projection",
    }.isdisjoint(reach)
