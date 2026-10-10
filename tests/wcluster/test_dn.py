"""The DN wrappers against WCluster's DN.Wkmeans, DN.Whclust, DN.Wpca and DNcluster.predict on
datanuggets from create.DN, and where R stops."""

import warnings

import numpy as np
import pytest

from datanuggets._compat.linalg import r_matmul
from datanuggets._random.replay import ReplayDraws
from datanuggets.result import DataNuggets
from datanuggets.wcluster.dn import dn_cluster_predict, dn_whclust, dn_wkmeans, dn_wpca
from rfixtures import assert_same_values, eigen_mismatches, load_group

CASES = {c.id: c for c in load_group("dn_wcluster")}
TAB = CASES["dn"]


def nuggets(suffix=""):
    """R's datanugget table as a DataNuggets, its assignments 0-based."""
    return DataNuggets(
        TAB.array("centers" + suffix),
        TAB.array("weights" + suffix),
        TAB.array("scales" + suffix),
        TAB.array("assignments" + suffix).astype(np.int64) - 1,
    )


DN, DN2 = nuggets(), nuggets("2")
CL = CASES["predict_frame"].array("cl")


def first_vars(dn, q):
    return DataNuggets(dn.centers[:, :q], dn.weights, dn.scales, dn.assignments)


def test_fixture_nuggets_are_the_ones_r_clustered():
    assert DN.n_nuggets == 150 and DN2.n_nuggets == 60
    assert DN.centers.shape[1] == DN2.centers.shape[1] == 5
    assert CL.tolist() == CASES["wkmeans"].array("labels").tolist()


def test_dn_wkmeans_matches_r():
    case = CASES["wkmeans"]
    a = case.args
    draws = ReplayDraws(case.draws())
    got = dn_wkmeans(DN, a["k"], num_init=a["num_init"], max_iter=a["max_iter"], seed=draws)
    draws.assert_exhausted()
    assert got.labels.tolist() == (case.array("labels").astype(np.int64) - 1).tolist()
    assert_same_values(got.centers, case.array("centers"))
    assert_same_values(got.wwcss, case.array("wwcss"))
    assert_same_values(got.total_wwcss, case.array("total").reshape(()))
    assert got.row_labels.tolist() == (case.array("row_labels").astype(np.int64) - 1).tolist()


def test_dn_whclust_matches_r():
    case = CASES["whclust"]
    n = DN.n_nuggets
    merge = case.array("merge").astype(np.int64)
    got = dn_whclust(DN)
    assert got.merge.tolist() == np.where(merge < 0, -merge - 1, n + merge - 1).tolist()
    assert_same_values(got.height, case.array("height"))
    assert got.order.tolist() == (case.array("order").astype(np.int64) - 1).tolist()


@pytest.mark.parametrize("cid", ["wpca", "wpca_corr"])
def test_dn_wpca_matches_r(cid):
    case = CASES[cid]
    assert_same_values(case.array("x"), DN.centers)
    assert_same_values(case.array("wrow"), DN.weights)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = dn_wpca(DN, corr=case.args["corr"])
    assert [str(w.message) for w in rec] == case.manifest["warnings"] == []
    if case.args["corr"]:
        assert_same_values(got.scale, case.array("scale"))
    else:
        assert case.array("scale").tolist() == [1.0]  # R's single 1, which wpca gives as ones
        assert got.scale.tolist() == [1.0] * DN.centers.shape[1]
    assert_same_values(got.center, case.array("center"))
    assert_same_values(got.wcol, case.array("wcol"))
    assert_same_values(got.wrow, case.array("wrow"))
    assert_same_values(got.x, r_matmul(case.array("cap_x_0"), got.rotation))
    wrong = eigen_mismatches(
        case.array("cap_s"),
        case.array("values"),
        case.array("rotation"),
        case.array("scores"),
        case.array("cap_x_0"),
        got.sdev,
        got.rotation,
        got.x,
    )
    assert wrong == []


@pytest.mark.parametrize("cid", ["predict_frame", "predict_dn"])
def test_dn_cluster_predict_matches_r(cid):
    case = CASES[cid]
    newx = TAB.array("newx") if cid == "predict_frame" else DN2
    got = dn_cluster_predict(DN, case.array("cl"), newx)
    assert (got + 1).tolist() == case.array("labels").tolist()


def test_a_datanugget_newx_is_sliced_by_dns_width():
    # R takes newx's table columns 2:(p + 1) for dn's p: a wider newx gives its first p
    # centers, and a narrower one passes its Weight and Scale columns as data.
    narrow = first_vars(DN, 3)
    want = dn_cluster_predict(narrow, CL, DN2.centers[:, :3])
    assert dn_cluster_predict(narrow, CL, DN2).tolist() == want.tolist()
    short = np.column_stack([DN2.centers[:, :3], DN2.weights, DN2.scales])
    want = dn_cluster_predict(DN, CL, short)
    assert dn_cluster_predict(DN, CL, first_vars(DN2, 3)).tolist() == want.tolist()
    with pytest.raises(ValueError, match="undefined columns selected"):
        dn_cluster_predict(DN, CL, first_vars(DN2, 1))


def test_dn_functions_need_a_datanuggets():
    for fn in (
        lambda: dn_whclust(np.zeros((4, 2))),
        lambda: dn_wpca(None),
        lambda: dn_wkmeans({}, 2),
        lambda: dn_cluster_predict([], [0], np.zeros((1, 2))),
    ):
        with pytest.raises(TypeError, match='class "datanugget"'):
            fn()


def test_cluster_labels_must_cover_the_nuggets():
    with pytest.raises(ValueError, match="number of data nuggets"):
        dn_cluster_predict(DN, np.zeros(DN.n_nuggets - 1), DN.centers[:3])


def test_one_variable_nuggets_stop_where_r_stops():
    one = first_vars(DN, 1)
    with pytest.raises(ValueError, match="argument is of length zero"):
        dn_wkmeans(one, 2)
    with pytest.raises(ValueError, match="invalid 'times' argument"):
        dn_wpca(one)
    assert dn_wpca(one, wcol=[1.0]).sdev.shape == (1,)
    assert dn_whclust(one).merge.shape == (DN.n_nuggets - 1, 2)
    for newx in (one, DN2):  # dn's width slices newx's table, whatever newx's own width
        with pytest.raises(ValueError, match="argument is of length zero"):
            dn_cluster_predict(one, CL, newx)
    assert dn_cluster_predict(one, CL, TAB.array("newx")[:5, :1]).shape == (5,)


def test_r_positional_order_fails():
    with pytest.raises(TypeError):
        dn_wkmeans(DN, 3, None, 1)
