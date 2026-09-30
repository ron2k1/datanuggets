"""cluster_predict and its distance matrix against WCluster's on the wstats groups."""

import warnings

import numpy as np
import pytest

from datanuggets.wcluster.predict import _wdist_mat, cluster_predict
from datanuggets.wcluster.wstats import wmean
from rfixtures import assert_same_values, from_hex, load_group

CASES = load_group("wstats") + load_group("wstats_constructed")
BY_ID = {c.id: c for c in CASES}
Z = [c for c in CASES if c.has("z")]
PRED = [c for c in CASES if c.has("pred")]
LIST = [c for c in Z if c.args.get("pred") == "list"]
OVF = ["ovf_one_row", "ovf_two_rows", "ovf_newx_row", "ovf_dead_first_nan_inf"]
DEAD = {"mtcars_gear_am": 0, "mtcars_gear_am_row01": 0, "mtcars_carb_am": 2, "mtcars_cyl_vs": 2}
ORDERS = pytest.mark.parametrize("order", ["C", "F"])


def _ids(cases):
    return [c.id for c in cases]


def _layout(a, order):
    a = np.asarray(a, dtype=np.float64)
    return np.asfortranarray(a) if order == "F" else np.ascontiguousarray(a)


def _inputs(case, order="C"):
    x = case.array("x")
    x = np.asfortranarray(x) if order == "F" else np.ascontiguousarray(x)
    newx = case.array("newx")
    newx = np.asfortranarray(newx) if order == "F" else np.ascontiguousarray(newx)
    return x, case.array("cl"), case.array("w"), newx


def _dead_columns(case) -> list[int]:
    """The clusters whose R center is NaN in every coordinate."""
    return np.flatnonzero(np.isnan(case.array("wmean")).all(axis=0)).tolist()


def _tie_rows(z: np.ndarray) -> np.ndarray:
    """The rows of z whose smallest entry appears more than once."""
    return np.flatnonzero((z == z.min(axis=1)[:, None]).sum(axis=1) > 1)


def _predict(case, order="C"):
    x, cl, w, newx = _inputs(case, order)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return cluster_predict(x, cl, newx, w=w)


@ORDERS
@pytest.mark.parametrize("case", Z, ids=_ids(Z))
def test_wdist_mat_matches_r(case, order):
    newx, want = case.array("newx"), case.array("z")
    centers = case.array("wmean").T  # R's wmean is p by k
    assert want.shape == (newx.shape[0], centers.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _wdist_mat(_layout(newx, order), _layout(centers, order))
    assert_same_values(got, want)  # any NaN matches any NaN, and each Inf keeps its place


@ORDERS
@pytest.mark.parametrize("case", PRED, ids=_ids(PRED))
def test_cluster_predict_matches_r(case, order):
    want = case.array("pred")
    got = _predict(case, order)
    assert got.dtype == np.intp and got.shape == want.shape
    np.testing.assert_array_equal(got, want - 1)  # R's which.min is 1-based


def test_fixtures_cover_the_edge_cases():
    for cases in (Z, PRED, LIST):
        assert cases
    assert len(BY_ID) == len(CASES)  # case ids are unique across both groups
    for case in Z:
        assert case.has("newx") and case.has("wmean")
        assert case.has("pred") or case.args.get("pred") == "list"
    assert _ids(LIST) == ["ovf_newx_row"]
    for cid, pos in DEAD.items():
        assert _dead_columns(BY_ID[cid]) == [pos]
    last = BY_ID["mtcars_cyl_vs"]
    assert _dead_columns(last) == [last.array("z").shape[1] - 1]  # the last of k
    assert BY_ID["mtcars_gear_am_row01"].array("newx").shape[0] == 1
    assert _tie_rows(BY_ID["mtcars_cyl1d_gear"].array("z")).size
    assert (BY_ID["iris50_singletons"].array("z") < 0).any()
    for cid in OVF:
        assert not np.isfinite(BY_ID[cid].array("z")).all()
    assert np.isinf(BY_ID["ovf_one_row"].array("z")).any()
    (row,) = BY_ID["ovf_dead_first_nan_inf"].array("z")
    assert np.isnan(row[0]) and np.isposinf(row[1:]).all() and row.size > 2


@pytest.mark.parametrize("cid", list(DEAD))
def test_dead_cluster_is_never_predicted(cid):
    case = BY_ID[cid]
    pos = DEAD[cid]
    z = case.array("z")
    # R's which.min skips the NaN column, where a plain argmin would stop on it
    assert (np.argmin(z, axis=1) == pos).any()
    assert np.isnan(z[:, pos]).all()
    got = _predict(case)
    assert not (got == pos).any()
    np.testing.assert_array_equal(got, case.array("pred") - 1)


def test_every_cluster_dead_raises():
    # R's which.min gives integer(0) for each all-NaN row, and cluster.predict returns
    # integer(0) for the whole call
    x, cl, w, newx = _inputs(BY_ID["mtcars_gear_am"])
    zeros = np.zeros_like(w)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="no usable distance"):
            cluster_predict(x, cl, newx, w=zeros)


@pytest.mark.parametrize("cid", OVF)
def test_overflow_rows_match_r(cid):
    case = BY_ID[cid]
    x, cl, w, newx = _inputs(case)
    want = case.array("z")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _wdist_mat(newx, case.array("wmean").T)
    assert_same_values(got, want)
    if case.args.get("pred") == "list":
        # a row with no finite distance makes R's which.min return integer(0) there, so apply()
        # hands back a list instead of a vector
        assert not case.has("pred")
        assert np.isnan(want).all(axis=1).any()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="no usable distance"):
                cluster_predict(x, cl, newx, w=w)
        return
    np.testing.assert_array_equal(_predict(case), case.array("pred") - 1)


# Recorded by hand on R 4.6.1 with WCluster 1.3.0, 2026-09-30: iris rows 3 and 1 with weights 1
# and -1 put the first center at -Inf in every column (test_wstats pins it). One new row, iris row
# 7, is at +Inf from it and which.min passes over it. With rows 7, 60 and 120, R's identity
# multiply turns that column NaN. R predicts cluster 2 for every row both times.
INF_CENTER = {
    "one_row": ([6], ["7ff0000000000000", "3fc347ae147ae200"]),
    "three_rows": (
        [6, 59, 119],
        ["fff8000000000000", "3fc347ae147ae200", "fff8000000000000", "401933d70a3d70b0"]
        + ["fff8000000000000", "402d7b3333333330"],
    ),
}


@pytest.mark.parametrize("name", list(INF_CENTER))
def test_infinite_center_is_passed_over_as_in_r(name):
    rows, z = INF_CENTER[name]
    iris = BY_ID["iris_species_petalwidth"].array("x")[:, :3]
    x, newx = iris[[2, 0, 1, 3, 4, 5]], iris[rows]
    cl, w = np.array([1, 1, 2, 2, 2, 2]), np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centers = wmean(x, cl, w)
        got = cluster_predict(x, cl, newx, w=w)
        got_z = _wdist_mat(newx, centers)
    assert np.isneginf(centers[0]).all()
    assert_same_values(got_z, from_hex(z).reshape(len(rows), 2))
    np.testing.assert_array_equal(got, np.ones(len(rows)))  # R's 2 is position 1


def test_nan_then_inf_row_takes_the_first_live_column():
    case = BY_ID["ovf_dead_first_nan_inf"]
    (row,) = case.array("z")
    assert np.isnan(row[0]) and np.isposinf(row[1:]).all()
    assert case.array("pred").tolist() == [2]  # R's which.min skips the NaN, ties go first
    np.testing.assert_array_equal(_predict(case), [1])


def test_first_index_wins_a_tie():
    case = BY_ID["mtcars_cyl1d_gear"]
    z = case.array("z")
    ties = _tie_rows(z)
    assert ties.size
    first = np.argmax(z == z.min(axis=1)[:, None], axis=1)
    np.testing.assert_array_equal(case.array("pred")[ties] - 1, first[ties])
    got = _predict(case)
    np.testing.assert_array_equal(got[ties], first[ties])
    np.testing.assert_array_equal(got, case.array("pred") - 1)


def test_negative_squared_distance_is_kept():
    case = BY_ID["iris50_singletons"]
    z = case.array("z")
    assert (z < 0).any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _wdist_mat(case.array("newx"), case.array("wmean").T)
    np.testing.assert_array_equal(got < 0, z < 0)
    np.testing.assert_array_equal(_predict(case), case.array("pred") - 1)


def test_returns_positions_not_labels():
    case = BY_ID["mtcars_cyl_carb"]
    x, cl, w, newx = _inputs(case)
    assert np.array_equal(np.unique(cl), [4.0, 6.0, 8.0])
    codes = np.searchsorted([4.0, 6.0, 8.0], cl)
    want = case.array("pred") - 1
    assert set(want.tolist()) == {0, 1, 2}
    for labels in (cl, cl.astype(np.int64), cl.astype(np.int32), codes):
        np.testing.assert_array_equal(cluster_predict(x, labels, newx, w=w), want)


def test_negative_labels_break_ties_in_sorted_order():
    case = BY_ID["iris_singletons_neg_pw"]
    _, cl, _, newx = _inputs(case)
    assert (cl < 0).all()
    # newx rows 3 and 4 are iris rows 102 and 143, the same measurements with the same weight, so
    # the tie between labels -102 and -143 goes to -143: first in sorted order, last in x
    assert_same_values(newx[3], newx[4])
    assert {3, 4} <= set(_tie_rows(case.array("z")).tolist())
    want = case.array("pred") - 1
    levels = np.unique(cl)
    assert want[3] == want[4] == np.searchsorted(levels, -143)
    np.testing.assert_array_equal(_predict(case), want)


def test_r_argument_order_fails():
    # R's order is (x, w, cl, newx), and w is keyword-only here
    x, cl, w, newx = _inputs(BY_ID["iris_species_petalwidth"])
    with pytest.raises(TypeError, match="positional"):
        cluster_predict(x, w, cl, newx)


def test_newx_must_have_x_columns():
    # R stops: "newdata should have the same variables as the learning dataset."
    x, cl, w, newx = _inputs(BY_ID["iris_species_petalwidth"])
    for bad in (newx[:, :3], np.hstack([newx, newx[:, :1]])):
        with pytest.raises(ValueError, match=r"newx must have the 4 columns of x"):
            cluster_predict(x, cl, bad, w=w)


def test_newx_rules():
    x, cl, w, newx = _inputs(BY_ID["iris_species_petalwidth"])
    # R stops on a NULL newx with "newdata should be a data frame."
    with pytest.raises(TypeError, match="newx must contain only numeric columns"):
        cluster_predict(x, cl, None, w=w)
    with pytest.raises(TypeError, match="newx must contain only numeric columns"):
        cluster_predict(x, cl, newx.astype(str), w=w)
    with pytest.raises(ValueError, match="newx must be 2-D"):
        cluster_predict(x, cl, newx[0], w=w)
    for value in (np.nan, np.inf, -np.inf):
        bad = newx.copy()
        bad[2, 1] = value
        with pytest.raises(ValueError, match="newx contains NaN or infinite values"):
            cluster_predict(x, cl, bad, w=w)


def test_weight_and_label_errors():
    x, cl, w, newx = _inputs(BY_ID["iris_species_petalwidth"])
    # R stops: "The length of weights should be the same as the number of observations."
    for bad in (w[:-1], w[: x.shape[1]]):
        with pytest.raises(ValueError, match=r"w must have one weight per row \(150\)"):
            cluster_predict(x, cl, newx, w=bad)
    # R stops: "The length of cluster assignments should be the same as the number of
    # observations."
    with pytest.raises(ValueError, match=r"cl must have one label per row \(150\)"):
        cluster_predict(x, cl[:-1], newx, w=w)
    with pytest.raises(ValueError, match="x must be 2-D"):
        cluster_predict(x[:, 0], cl, newx, w=w)
