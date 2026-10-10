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


def port_tree(case):
    m = case.array("merge").astype(np.int64)
    n = m.shape[0] + 1
    return WhclustResult(
        np.where(m < 0, -m - 1, n + m - 1),
        case.array("height"),
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
