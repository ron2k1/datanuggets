"""dendrogram against R's plot.hclust: the polylines R's pdf device wrote, R's stops, and trees
deeper than Python's recursion limit."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from datanuggets import WhclustResult
from datanuggets.plot import dendrogram
from datanuggets.plot.tree import NONFINITE_STOP
from rfixtures import load_group

CASES = {c.id: c for c in load_group("plot")}
TREES = [c for c in CASES.values() if c.id.startswith("tree_") and c.id != "tree_hang"]
REFUSED = [c for c in TREES if c.manifest["plot_error"]]
NONFINITE = [c for c in TREES if c.manifest["nonfinite"] and not c.manifest["plot_error"]]
DRAWN = [c for c in TREES if c not in REFUSED and c not in NONFINITE]


def port_tree(case, height="height"):
    m = case.array("merge").astype(np.int64)
    n = m.shape[0] + 1
    return WhclustResult(
        np.where(m < 0, -m - 1, n + m - 1),
        case.array(height),
        case.array("order").astype(np.int64) - 1,
    )


def segments(ax):
    (lc,) = [c for c in ax.collections if c.get_gid() == "dendrogram"]
    return np.array(lc.get_segments())


def assert_r_lines(got, want, scale_x, scale_y):
    """Equal within 0.006 pt, the 0.005 pt R's pdf device rounds to plus a margin, in user units
    by the drawing's own points per unit."""
    want = want.reshape(-1, 4, 2)
    assert got.shape == want.shape
    tol = np.array([0.006 / scale_x, 0.006 / scale_y])
    assert np.all(np.abs(got - want) <= tol), np.max(np.abs(got - want), axis=(0, 1))


@pytest.mark.parametrize("case", DRAWN, ids=[c.id for c in DRAWN])
def test_lines_match_r(case):
    _, ax = plt.subplots()
    assert dendrogram(port_tree(case), labels=False, ax=ax) is ax
    m = case.manifest
    assert_r_lines(segments(ax), case.array("segs"), m["scale_x"], m["scale_y"])


@pytest.mark.parametrize(("hang", "suffix"), [(-1, "m1"), (0.5, "05")])
def test_hang_matches_r(hang, suffix):
    case = CASES["tree_hang"]
    _, ax = plt.subplots()
    dendrogram(port_tree(case), labels=False, hang=hang, ax=ax)
    m = case.manifest
    assert_r_lines(
        segments(ax), case.array(f"segs_{suffix}"), m[f"scale_x_{suffix}"], m[f"scale_y_{suffix}"]
    )


@pytest.mark.parametrize("heights", ["rev", "eq"])
def test_hang_is_the_last_height_minus_the_first(heights):
    """R hangs leaves by hang times the last height minus the first, not the range, and puts them
    at 0 once that's negative. R drew grid_dup's merges with the heights reversed, so the last is
    below the first, and with the last set to the first, so there's no hang but a range."""
    case = CASES["hang_heights"]
    _, ax = plt.subplots()
    dendrogram(port_tree(case, f"height_{heights}"), labels=False, ax=ax)
    m = case.manifest
    assert_r_lines(
        segments(ax),
        case.array(f"segs_{heights}"),
        m[f"scale_x_{heights}"],
        m[f"scale_y_{heights}"],
    )


@pytest.mark.parametrize("case", REFUSED, ids=[c.id for c in REFUSED])
def test_refuses_where_r_stops(case):
    with pytest.raises(ValueError) as e:
        dendrogram(port_tree(case))
    assert str(e.value) == case.manifest["plot_error"]


def test_the_recorded_trees_cover_each_outcome():
    assert len(DRAWN) == 8
    assert sorted(c.id for c in REFUSED) == ["tree_selfmerge_last", "tree_selfmerge_mid"]
    assert sorted(c.id for c in NONFINITE) == ["tree_inf_single", "tree_zero_w1", "tree_zero_w2"]


@pytest.mark.parametrize("case", NONFINITE, ids=[c.id for c in NONFINITE])
def test_refuses_nonfinite_heights(case):
    """R draws these off the page with a warning, so the port refuses them (departure 5)."""
    with pytest.raises(ValueError) as e:
        dendrogram(port_tree(case))
    assert str(e.value) == NONFINITE_STOP


def test_stops_match_r():
    e = CASES["stops"].manifest["errors"]
    tree = port_tree(CASES["stops"])
    for kwargs, key in [({"hang": float("nan")}, "hang_nan"), ({"hang": float("inf")}, "hang_inf")]:
        with pytest.raises(ValueError) as exc:
            dendrogram(tree, **kwargs)
        assert str(exc.value) == e[key]
    none = WhclustResult(np.empty((0, 2), np.int64), np.empty(0), [0])
    one = WhclustResult(np.array([[0, 1]], np.int64), np.array([1.0]), [0, 1])
    for bad, key in [(none, "none_plot"), (one, "one_plot")]:
        with pytest.raises(ValueError) as exc:
            dendrogram(bad)
        assert str(exc.value) == e[key], key


NOT_INTEGER = "'merge' component in dendrogram must be integer"
NA_IF = "missing value where TRUE/FALSE needed"


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        (-1, np.nan, NA_IF),
        (-1, np.inf, NA_IF),
        (-1, -np.inf, NA_IF),
        (-1, 2.0**31, NA_IF),
        (-1, 2.0**31 - 1, "'merge' matrix has invalid contents"),
        (-1, 3e9 + 0.5, NA_IF),
        (-1.5, np.nan, NOT_INTEGER),
        (-1.5, np.inf, NOT_INTEGER),
    ],
)
def test_merge_entries_stop_like_r(x, y, message):
    """.validity.hclust tests any(as.integer(merge) != merge). A fractional entry as.integer can
    hold gives TRUE, so it wins, and otherwise NaN, Inf or an entry at 2^31 or beyond gives NA.
    R's merge is rbind(c(x, y), c(-3, 1)) for three rows, each stop checked by hand on R 4.6.1."""
    rm = np.array([[x, y], [-3.0, 1.0]])
    merge = np.where(rm < 0, -rm - 1, rm + 2)  # R's -k is row k - 1 and its cluster j is j + 2
    tree = WhclustResult(merge, np.array([1.0, 2.0]), np.array([2, 0, 1]))
    with pytest.raises(ValueError) as e:
        dendrogram(tree)
    assert str(e.value) == message


def test_a_fraction_beside_a_huge_entry_is_not_integer_like_r():
    merge = np.array([[0.0, 3e9 + 2], [2.0, 3.5]])  # R's rbind(c(-1, 3e9), c(-3, 1.5))
    tree = WhclustResult(merge, np.array([1.0, 2.0]), np.array([2, 0, 1]))
    with pytest.raises(ValueError, match=f"^{NOT_INTEGER}$"):
        dendrogram(tree)


def test_leaf_labels():
    tree = port_tree(CASES["tree_n3"])
    _, ax = plt.subplots()
    dendrogram(tree, ax=ax)
    assert sorted(t.get_text() for t in ax.texts) == ["0", "1", "2"]
    _, ax = plt.subplots()
    dendrogram(tree, labels=["a", "b", "c"], ax=ax)
    assert sorted(t.get_text() for t in ax.texts) == ["a", "b", "c"]
    _, ax = plt.subplots()
    dendrogram(tree, labels=False, ax=ax)
    assert len(ax.texts) == 0
    with pytest.raises(ValueError, match="invalid dendrogram input"):
        dendrogram(tree, labels=["a", "b"])


@pytest.mark.parametrize("labels", ["abc", True, 5, np.int64(3), np.True_])
def test_one_label_for_many_leaves_stops_like_r(labels):
    """plot.hclust takes as.character(labels), one string for any of these, so C_dendwindow
    stops on the length. Checked by hand on R 4.6.1 for "abc", TRUE and 5."""
    with pytest.raises(ValueError, match="^invalid dendrogram input$"):
        dendrogram(port_tree(CASES["tree_n3"]), labels=labels)


def test_numpy_false_draws_no_labels_like_false():
    _, ax = plt.subplots()
    dendrogram(port_tree(CASES["tree_n3"]), labels=np.False_, ax=ax)
    assert len(ax.texts) == 0


def test_each_label_sits_under_its_own_row():
    """Row r's label hangs at its leaf: x is 1 plus r's place in order, y its merge's height minus
    the hang. Labels assigned in drawing order would keep the same set of texts."""
    tree = port_tree(CASES["tree_dn"])
    n = tree.order.size
    parent = np.empty(n)
    for s, pair in enumerate(np.asarray(tree.merge)):
        parent[pair[pair < n]] = tree.height[s]
    leaf_y = parent - 0.1 * (tree.height[-1] - tree.height[0])
    rank = np.argsort(tree.order)
    names = [f"r{r}" for r in range(n)]
    for labels, text in ((None, [str(r) for r in range(n)]), (names, names)):
        _, ax = plt.subplots()
        dendrogram(tree, labels=labels, ax=ax)
        got = {t.get_text(): t.xy for t in ax.texts}
        assert got == {text[r]: (1.0 + rank[r], leaf_y[r]) for r in range(n)}


def test_axes_look_like_r():
    _, ax = plt.subplots()
    dendrogram(port_tree(CASES["tree_n3"]), ax=ax)
    assert ax.get_ylabel() == "Height" and not ax.xaxis.get_visible()
    assert not any(ax.spines[s].get_visible() for s in ("top", "right", "bottom"))


def test_dendrogram_draws_a_3000_deep_chain():
    n = 3000
    merge = np.array([(0, 1)] + [(n + s - 1, s + 1) for s in range(1, n - 1)], dtype=np.int64)
    tree = WhclustResult(merge, np.arange(1.0, n), np.arange(n))
    _, ax = plt.subplots()
    dendrogram(tree, labels=False, ax=ax)
    assert segments(ax).shape == (n - 1, 4, 2)
