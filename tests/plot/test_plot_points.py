"""clusters and nuggets: R's sizes and colors, labels with gaps, and any array-like input."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from datanuggets import DataNuggets, wkmeans
from datanuggets._random.r_draws import RDraws
from datanuggets.plot import OKABE_ITO, R_PALETTE, clusters, nuggets
from datanuggets.plot.points import _rgb_green, _weight_fractions, weight_colors
from rfixtures import assert_same_values, load_group

CASES = {c.id: c for c in load_group("plot")}
DN = {c.id: c for c in load_group("dn_wcluster")}["dn"]
WKMEANS = {c.id: c for c in load_group("wkmeans")}


def drawn(ax, gid):
    (c,) = [c for c in ax.collections if c.get_gid() == gid]
    return c


@pytest.mark.parametrize(
    ("w", "t", "hex_"),
    [("w", "t", "hex"), ("w_ties", "t_ties", "hex_ties"), ("w_wt", "t_wt", "hex_wt")],
)
def test_weight_colors_match_r(w, t, hex_):
    c = CASES["colors"]
    assert_same_values(_weight_fractions(c.array(w)), c.array(t))
    assert weight_colors(c.array(w)) == c.manifest[hex_]


@pytest.mark.parametrize(
    ("w", "key"), [([0.0, 0, 0], "all_zero"), ([0.0] * 9 + [1], "quantile_zero")]
)
def test_weight_colors_stop_where_r_stops(w, key):
    with pytest.raises(ValueError) as e:
        weight_colors(w)
    assert str(e.value) == CASES["colors"].manifest["errors"][key]


@pytest.mark.parametrize(("v", "key"), [(float("nan"), "rgb_nan"), (-0.5, "rgb_neg")])
def test_rgb_stops_match_r(v, key):
    with pytest.raises(ValueError) as e:
        _rgb_green(v)
    assert str(e.value) == CASES["colors"].manifest["errors"][key]


def test_no_weights_give_no_colors():
    """R's sapply over numeric(0) gives an empty list, with no quantile stop."""
    assert weight_colors([]) == []


def test_negative_weights_stop_in_rgb():
    """R's recipe with a negative weight: t = -1/3 reaches rgb, which prints it with %g."""
    with pytest.raises(ValueError, match=r"^color intensity -0\.333333, not in \[0,1\]$"):
        weight_colors([-1.0, 3, 3])


def test_nuggets_draws_r_colors_and_size():
    dn = DataNuggets(
        DN.array("centers"), DN.array("weights"), DN.array("scales"), DN.array("assignments") - 1
    )
    _, ax = plt.subplots()
    assert nuggets(dn, ax=ax) is ax
    pts = drawn(ax, "nuggets")
    assert np.array_equal(pts.get_offsets(), dn.centers[:, :2])
    assert [to_hex(c).upper() for c in pts.get_facecolor()] == CASES["colors"].manifest["hex"]
    assert np.all(pts.get_sizes() == (9 * 0.5) ** 2)


def test_nuggets_by_labels_and_dims():
    dn = DataNuggets(
        DN.array("centers"), DN.array("weights"), DN.array("scales"), DN.array("assignments") - 1
    )
    labels = np.arange(dn.n_nuggets) % 3
    _, ax = plt.subplots()
    nuggets(dn, labels=labels, dims=(2, 4), colors=R_PALETTE, ax=ax)
    pts = drawn(ax, "nuggets")
    assert np.array_equal(pts.get_offsets(), dn.centers[:, [2, 4]])
    assert [to_hex(c) for c in pts.get_facecolor()] == [to_hex(R_PALETTE[v]) for v in labels]


def test_clusters_sizes_and_colors():
    x = np.array([[0.0, 1], [1, 2], [2, 0], [3, 3]])
    labels = np.array([0, 1, 9, 7])  # 9 wraps to color 1
    w = np.array([1.0, 2, 4, 8])
    _, ax = plt.subplots()
    clusters(x, labels, w=w, colors=R_PALETTE, ax=ax)
    pts = drawn(ax, "points")
    assert np.array_equal(pts.get_sizes(), 6 + 400 * w / 8)
    want = [to_hex(R_PALETTE[v % 8]) for v in labels]
    assert [to_hex(c) for c in pts.get_facecolor()] == want
    _, ax = plt.subplots()
    clusters(x, labels, w=w, size="log", ax=ax)
    assert np.allclose(drawn(ax, "points").get_sizes(), np.where(w > 1, (9 * np.log(w)) ** 2, 0))


def test_centers_follow_the_labels_present():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 2))
    labels = np.repeat([3, 0, 2], 20)  # label 1 left empty, as wkmeans can, and 3 seen first
    centers = np.array([x[labels == v].mean(axis=0) for v in (0, 2, 3)])
    _, ax = plt.subplots()
    clusters(x, labels, centers=centers, colors=R_PALETTE, ax=ax)
    stars = drawn(ax, "centers")
    assert np.array_equal(stars.get_offsets(), centers)
    assert [to_hex(c) for c in stars.get_facecolor()] == [to_hex(R_PALETTE[v]) for v in (0, 2, 3)]


def test_center_stars_stand_out_from_their_own_points():
    """OKABE_ITO and R_PALETTE start with black, where a black edge would hide the star."""
    for palette in (OKABE_ITO, R_PALETTE):
        _, ax = plt.subplots()
        clusters(np.zeros((3, 2)), [0, 1, 1], centers=np.zeros((2, 2)), colors=palette, ax=ax)
        stars = drawn(ax, "centers")
        assert to_hex(stars.get_facecolor()[0]) == "#000000"
        assert {to_hex(c) for c in stars.get_edgecolor()} == {"#ffffff"}


@pytest.mark.parametrize("case_id", ["gap", "iris_k3"])
def test_clusters_on_a_wkmeans_result(case_id):
    """The recorded runs, replayed: gap is R's own k = 3 run that left a label empty."""
    case = WKMEANS[case_id]
    a = case.args
    x, w = case.array("x"), case.array("w")
    res = wkmeans(
        x, a["k"], w=w, num_init=a["num_init"], max_iter=a["max_iter"], seed=RDraws(a["seed"])
    )
    assert np.array_equal(res.labels, case.array("labels").astype(np.int64) - 1)
    present = np.unique(res.labels)
    assert (present.size < a["k"]) == (case_id == "gap")
    _, ax = plt.subplots()
    # gap's weights are negative, which Wkmeans takes and area sizing refuses, so no w here
    clusters(x, res.labels, centers=res.centers, colors=R_PALETTE, ax=ax)
    stars = drawn(ax, "centers")
    assert np.array_equal(stars.get_offsets(), res.centers[:, :2])
    assert [to_hex(c) for c in stars.get_facecolor()] == [to_hex(R_PALETTE[v]) for v in present]


def test_clusters_takes_frames_lists_and_ints():
    x = np.array([[0, 1], [1, 2], [2, 0]])
    for xx in (x, x.tolist(), pd.DataFrame(x), np.asfortranarray(x)):
        _, ax = plt.subplots()
        clusters(xx, [0, 1, 1], w=[1, 2, 3], ax=ax)
        assert np.array_equal(np.asarray(drawn(ax, "points").get_offsets()), x)


def test_clusters_draws_on_the_current_axes():
    _, ax = plt.subplots()
    assert clusters(np.zeros((3, 2)), [0, 1, 1]) is ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dims": (0, 0)},
        {"dims": (0, 2)},
        {"dims": (0,)},
        {"dims": (-1, 0)},
        {"w": [1, 2]},
        {"w": [-1, 1, 1]},
        {"w": [0, 0, 0]},
        {"w": [1, np.nan, 1]},
        {"w": [1, np.inf, 1]},
        {"size": "radius"},
        {"size": "log"},
        {"centers": np.zeros((3, 2))},
        {"centers": np.zeros((2, 3))},
    ],
)
def test_clusters_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        clusters(np.zeros((3, 2)), [0, 1, 1], **kwargs)


@pytest.mark.parametrize("labels", [[0, 1], [0.5, 1, 1], [[0, 1, 1]], ["a", "b", "b"]])
def test_clusters_rejects_bad_labels(labels):
    with pytest.raises(ValueError):
        clusters(np.zeros((3, 2)), labels)


def test_bad_centers_draw_nothing():
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="one row per label present"):
        clusters(np.zeros((3, 2)), [0, 1, 1], centers=np.zeros((3, 2)), ax=ax)
    assert len(ax.collections) == 0


def test_clusters_rejects_a_vector_x():
    with pytest.raises(ValueError, match="two-dimensional"):
        clusters(np.zeros(3), [0, 1, 1])
