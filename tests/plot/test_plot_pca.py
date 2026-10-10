"""The PCA views on R's DN.Wpca output: the sign rule, shares, projections and arrows."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from datanuggets import WpcaResult
from datanuggets.plot import R_PALETTE, loadings, scores, scree
from datanuggets.plot.pca import _signs
from rfixtures import load_group

WPCA = {c.id: c for c in load_group("dn_wcluster")}["wpca"]
QUAKES = ["lat", "long", "depth", "mag", "stations"]


def r_wpca(**change):
    c = WPCA
    fields = {
        "sdev": c.array("sdev"),
        "rotation": c.array("rotation"),
        "x": c.array("scores"),
        "center": c.array("center"),
        "scale": c.array("scale"),
        "wrow": c.array("wrow"),
        "wcol": c.array("wcol"),
    }
    return WpcaResult(**{**fields, **change})


def drawn(ax, gid):
    (c,) = [c for c in ax.collections if c.get_gid() == gid]
    return c


def test_signs_make_the_largest_loading_positive():
    rot = r_wpca().rotation
    s = _signs(rot)
    flipped = rot * s
    big = flipped[np.argmax(np.abs(flipped), axis=0), np.arange(rot.shape[1])]
    assert np.all(big > 0) and set(s.tolist()) <= {-1.0, 1.0}


def test_signs_keep_a_zero_column_positive():
    assert _signs(np.zeros((3, 2))).tolist() == [1.0, 1.0]


def test_views_ignore_the_eigenvector_signs():
    """R's eigen can return either sign, so flipping a component in the result changes no view."""
    pc = r_wpca()
    flip = np.array([-1.0, 1, -1, 1, 1])
    other = r_wpca(rotation=pc.rotation * flip, x=pc.x * flip)
    got = []
    for res in (pc, other):
        _, ax = plt.subplots()
        scores(res, ax=ax)
        _, ax2 = plt.subplots()
        loadings(res, ax=ax2)
        q = drawn(ax2, "loadings")
        got.append((np.asarray(drawn(ax, "scores").get_offsets()), q.U, q.V))
    for a, b in zip(*got):
        assert np.array_equal(a, b)


def test_scores_are_the_flipped_scores_with_shares_on_the_axes():
    pc = r_wpca()
    _, ax = plt.subplots()
    assert scores(pc, w=pc.wrow, ax=ax) is ax
    s = _signs(pc.rotation)
    pts = drawn(ax, "scores")
    assert np.array_equal(pts.get_offsets(), (pc.x * s)[:, :2])
    assert np.array_equal(pts.get_sizes(), 6 + 400 * pc.wrow / pc.wrow.max())
    lam = pc.sdev**2
    assert ax.get_xlabel() == f"PC1 ({100 * lam[0] / lam.sum():.1f}%)"
    assert ax.get_ylabel() == f"PC2 ({100 * lam[1] / lam.sum():.1f}%)"


def test_scores_on_other_dims_with_labels():
    pc = r_wpca()
    labels = np.arange(pc.x.shape[0]) % 3
    _, ax = plt.subplots()
    scores(pc, labels=labels, dims=(2, 0), colors=R_PALETTE, ax=ax)
    pts = drawn(ax, "scores")
    assert np.array_equal(pts.get_offsets(), (pc.x * _signs(pc.rotation))[:, [2, 0]])
    assert [to_hex(c) for c in pts.get_facecolor()] == [to_hex(R_PALETTE[v]) for v in labels]
    assert ax.get_xlabel().startswith("PC3 (") and ax.get_ylabel().startswith("PC1 (")


def test_scores_without_labels_take_the_first_color_at_any_count():
    """matplotlib's cycle colors are RGB tuples, which scatter would value-map for 3 points."""
    pc = r_wpca()
    first = to_hex(plt.rcParams["axes.prop_cycle"].by_key()["color"][0])
    for n in (3, 4, 150):
        fig, ax = plt.subplots()
        scores(r_wpca(x=pc.x[:n], wrow=pc.wrow[:n]), ax=ax)
        fig.canvas.draw()  # a value-mapped c only takes its colormap's colors when drawn
        pts = drawn(ax, "scores")
        assert pts.get_array() is None
        assert {to_hex(c) for c in pts.get_facecolor()} == {first}


def test_rows_project_like_the_scores():
    pc = r_wpca()
    x0 = WPCA.array("x")  # the nugget centers Wpca took
    _, ax = plt.subplots()
    scores(pc, rows=x0, ax=ax)
    got = np.asarray(drawn(ax, "rows").get_offsets())
    assert np.allclose(got, (pc.x * _signs(pc.rotation))[:, :2], rtol=1e-12, atol=1e-9)


def test_rows_are_thinned_to_50000():
    pc = r_wpca()
    rows = np.tile(WPCA.array("x"), (700, 1))  # 105,000 rows
    _, ax = plt.subplots()
    scores(pc, rows=rows, ax=ax)
    assert len(drawn(ax, "rows").get_offsets()) == len(rows[::3])
    _, ax = plt.subplots()
    scores(pc, rows=rows[:50_000], ax=ax)
    assert len(drawn(ax, "rows").get_offsets()) == 50_000


def test_scree_bars():
    pc = r_wpca()
    _, ax = plt.subplots()
    assert scree(pc, ax=ax) is ax
    lam = pc.sdev**2
    assert np.array_equal([b.get_height() for b in ax.patches], 100 * lam / lam.sum())
    assert [b.get_x() + b.get_width() / 2 for b in ax.patches] == [1, 2, 3, 4, 5]


def test_scree_counts_a_nan_sdev_as_zero():
    """An eigenvalue rounded just below zero gives R a NaN sdev."""
    sdev = r_wpca().sdev.copy()
    sdev[-1] = np.nan
    _, ax = plt.subplots()
    scree(r_wpca(sdev=sdev), ax=ax)
    heights = np.array([b.get_height() for b in ax.patches])
    assert heights[-1] == 0 and np.isclose(heights.sum(), 100)


def test_loadings_arrows():
    pc = r_wpca()
    _, ax = plt.subplots()
    assert loadings(pc, names=QUAKES, ax=ax) is ax
    q = drawn(ax, "loadings")
    rot = pc.rotation * _signs(pc.rotation)
    assert np.array_equal(q.U, rot[:, 0]) and np.array_equal(q.V, rot[:, 1])
    assert sorted(t.get_text() for t in ax.texts) == sorted(QUAKES)
    _, ax = plt.subplots()
    loadings(pc, ax=ax)
    assert sorted(t.get_text() for t in ax.texts) == ["0", "1", "2", "3", "4"]


def test_one_component_has_a_scree_but_no_plane():
    pc = r_wpca()
    one = WpcaResult(
        pc.sdev[:1], pc.rotation[:1, :1], pc.x[:, :1], pc.center[:1], pc.scale, pc.wrow, pc.wcol[:1]
    )
    _, ax = plt.subplots()
    scree(one, ax=ax)
    assert len(ax.patches) == 1
    for f in (scores, loadings):
        with pytest.raises(ValueError, match="two components"):
            f(one)


@pytest.mark.parametrize(
    ("view", "kwargs"),
    [
        (scores, {"dims": (0, 5)}),
        (scores, {"dims": (1, 1)}),
        (scores, {"rows": np.zeros((4, 3))}),
        (scores, {"labels": [0, 1]}),
        (scores, {"w": [1.0, 2.0]}),
        (scores, {"colors": []}),
        (scores, {"colors": ["notacolor"], "rows": np.zeros((4, len(QUAKES)))}),
        (loadings, {"names": ["lat", "long"]}),
        (loadings, {"dims": (0, 5)}),
    ],
)
def test_bad_input_draws_nothing(view, kwargs):
    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        view(r_wpca(), ax=ax, **kwargs)
    assert len(ax.collections) == 0 and len(ax.texts) == 0
