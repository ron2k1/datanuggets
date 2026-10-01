"""The port against R where R's own coercions decide what an input means: logical data and
one-dimensional x. The fixtures are tests/fixtures/compat, recorded by r/fixtures_compat.R, and
each case holds what R returned or the error it stopped with."""

import numpy as np
import pandas as pd
import pytest

from datanuggets import (
    DataNuggets,
    cluster_predict,
    create_dn,
    create_dn_centers,
    distw,
    get_dn_means,
    refine_dn,
    wmean,
    wpca,
    wss,
    wwcss,
)
from datanuggets.wcluster.wpca import _wpca_parts
from rfixtures import assert_same_values, eigen_mismatches, load_group, same_bits

COMPAT = {c.id: c for c in load_group("compat")}


def check_wcluster(case, x, newx):
    """Every WCluster function on x against R's recorded outputs for the same data."""
    cl, w = case.array("cl"), case.array("w")
    assert same_bits(wss(x, w), case.array("wss")[0])
    assert same_bits(wmean(x, cl, w).T, case.array("wmean"))  # R's layout is p by k
    got = wwcss(x, cl, w, group_sum=True)
    assert same_bits(got.per_cluster, case.array("wwcss_group"))
    assert same_bits(got.total, case.array("wwcss_total")[0])
    assert same_bits(distw(x, cl, w), case.array("distw"))  # NA bits in the unfilled cells
    np.testing.assert_array_equal(cluster_predict(x, cl, newx, w=w) + 1, case.array("pred"))
    # Wpca with its defaults: unit weights and no corr. The eigen output by the eigen rule, with
    # R's eigenvalues as sdev squared, which is within the rule's band of them
    xm = case.array("wpca_matrix")
    n, p = xm.shape
    _, _, x0, _, s = _wpca_parts(xm, np.ones(n), np.ones(p), False)
    sdev = case.array("wpca_sdev")
    assert np.isfinite(sdev).all()
    pc = wpca(x)
    assert_same_values(pc.center, case.array("wpca_center"))
    wrong = eigen_mismatches(
        s,
        sdev**2,
        case.array("wpca_rotation"),
        case.array("wpca_x"),
        x0,
        pc.sdev,
        pc.rotation,
        pc.x,
    )
    assert wrong == []


def same_nuggets(a: DataNuggets, b: DataNuggets) -> None:
    for name in ("centers", "weights", "scales", "assignments"):
        assert same_bits(getattr(a, name), getattr(b, name)), name


# ---- logical data: R's arithmetic reads TRUE and FALSE as 1 and 0 ----


def test_logical_x_and_newx_in_the_wcluster_functions():
    case = COMPAT["data_mtcars_vsam_logical"]
    assert case.args["x_logical"] and case.args["newx_logical"]
    check_wcluster(case, case.array("x").astype(bool), case.array("newx").astype(bool))


def test_logical_newx_against_numeric_x():
    case = COMPAT["predict_logical_newx"]
    newx = case.array("newx").astype(bool)
    got = cluster_predict(case.array("x"), case.array("cl"), newx, w=case.array("w"))
    np.testing.assert_array_equal(got + 1, case.array("pred"))


def test_get_dn_means_on_logical_x():
    case = COMPAT["means_logical"]
    dn = DataNuggets(
        case.array("centers"),
        case.array("weights"),
        case.array("scales"),
        case.array("assignments") - 1,
    )
    assert same_bits(get_dn_means(case.array("x").astype(bool), dn), case.array("means"))


def mixed_frame(case) -> pd.DataFrame:
    """The recorded numeric matrix as R's data frame had it, with its logical columns as bool."""
    x = case.array("x")
    frame = pd.DataFrame({f"c{j}": x[:, j] for j in range(x.shape[1])})
    for j in case.args["logical_columns"]:
        frame[f"c{j - 1}"] = x[:, j - 1].astype(bool)
    return frame


def test_a_frame_mixing_logical_and_numeric_columns_is_the_numeric_matrix():
    # R's as.matrix() turns the frame into the numeric matrix, so R's results were identical
    case = COMPAT["mixed_logical_frame"]
    assert case.args["create_identical"] and case.args["centers_identical"]
    assert case.args["refine_identical"]
    x, frame = case.array("x"), mixed_frame(case)
    assert (frame.dtypes == bool).sum() == 2
    args = {"r": 100, "dn_num1": 20, "dn_num2": 8}
    dn = create_dn(x, **args)
    same_nuggets(create_dn(frame, **args), dn)
    a = create_dn_centers(frame.iloc[:30], dn_num=10)
    b = create_dn_centers(x[:30], dn_num=10)
    assert same_bits(a.centers, b.centers) and a.kept.tolist() == b.kept.tolist()
    same_nuggets(refine_dn(frame, dn), refine_dn(x, dn))


def test_all_logical_input_stops_where_r_stops():
    case = COMPAT["mixed_logical_frame"]
    x = case.array("x")
    logical = x[:, [j - 1 for j in case.args["logical_columns"]]].astype(bool)
    dn = create_dn(x, r=100, dn_num1=20, dn_num2=8)
    args = {"r": 100, "dn_num1": 20, "dn_num2": 8}
    assert case.args["all_logical_create_error"] == "x must contain only numeric columns"
    with pytest.raises(TypeError, match="^x must contain only numeric columns$"):
        create_dn(logical, **args)
    with pytest.raises(TypeError, match="^x must contain only numeric columns$"):
        create_dn(pd.DataFrame(logical), **args)
    assert case.args["all_logical_refine_error"] == "x must contain only numeric columns"
    with pytest.raises(TypeError, match="^x must contain only numeric columns$"):
        refine_dn(logical, dn)
    # R names its argument RS
    assert case.args["all_logical_centers_error"] == "RS must contain only numeric columns"
    with pytest.raises(TypeError, match="^rs must contain only numeric columns$"):
        create_dn_centers(logical, dn_num=10)


def test_a_frame_with_a_text_column_is_not_numeric():
    # as.matrix() of a frame with a character column is a character matrix
    case = COMPAT["mixed_logical_frame"]
    frame = mixed_frame(case).assign(name=[f"car{i}" for i in range(32)])
    with pytest.raises(TypeError, match="^x must contain only numeric columns$"):
        create_dn(frame, r=100, dn_num1=20, dn_num2=8)
    with pytest.raises(TypeError, match="^x must contain only numeric columns$"):
        wss(frame)
