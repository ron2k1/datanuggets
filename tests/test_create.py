import warnings

import numpy as np
import pandas as pd
import pytest

import datanuggets.nuggets.create as create_mod
from datanuggets import DataNuggetWarning, create_dn
from datanuggets._random.replay import ReplayDraws
from datanuggets.nuggets.create import _allocate
from rfixtures import create_args, load_group

CASES = load_group("create_dn")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_create_dn_matches_r(case):
    draws = ReplayDraws(case.draws())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dn = create_dn(case.array("x"), **create_args(case), seed=draws)
    draws.assert_exhausted()
    np.testing.assert_array_equal(dn.assignments + 1, case.array("assignments"))
    np.testing.assert_array_equal(dn.weights, case.array("weights"))
    np.testing.assert_array_equal(dn.centers, case.array("centers"))
    np.testing.assert_array_equal(dn.scales, case.array("scales"))
    ours = [w for w in caught if issubclass(w.category, DataNuggetWarning)]
    assert len(ours) == len(case.manifest["warnings"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_create_dn_hands_select_centers_what_r_hands_create_dncenters(case, monkeypatch):
    calls = []
    real = create_mod.select_centers

    def spy(rs, dn_num, delete_percent, dist_metric):
        calls.append((rs.copy(), dn_num))
        return real(rs, dn_num, delete_percent, dist_metric)

    monkeypatch.setattr(create_mod, "select_centers", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        create_dn(case.array("x"), **create_args(case), seed=ReplayDraws(case.draws()))
    assert len(calls) == case.manifest["n_calls"]
    for i, (rs, dn_num) in enumerate(calls, start=1):
        np.testing.assert_array_equal(rs, case.array(f"call{i:02d}_rs"), err_msg=f"call {i}")
        assert dn_num == case.manifest["call_dn_num"][i - 1], f"call {i}"


@pytest.mark.parametrize(
    ("dn_num1", "sizes", "expected"),
    [
        (401, [250, 250, 250, 250], [101, 100, 100, 100]),  # remainder goes to the first chunks
        (7, [100] * 10, [1] * 10),  # fewer centers than chunks: every chunk still keeps one
        (10, [3, 100], [3, 5]),  # never more than the chunk holds
        (5, [2, 2, 2], [2, 2, 1]),
    ],
)
def test_allocate_splits_dn_num1_like_r(dn_num1, sizes, expected):
    assert _allocate(dn_num1, np.array(sizes)).tolist() == expected


QUAKES = next(c for c in CASES if c.id == "quakes_euclidean_mean")
SMALL = {"r": 300, "dn_num1": 400, "dn_num2": 100}


def test_n_jobs_does_not_change_result():
    x = QUAKES.array("x")
    a = create_dn(x, **SMALL, seed=11, n_jobs=1)
    b = create_dn(x, **SMALL, seed=11, n_jobs=2)
    for field in ("centers", "weights", "scales", "assignments"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_progress_does_not_change_result():
    x = QUAKES.array("x")
    a = create_dn(x, **SMALL, progress=False)
    b = create_dn(x, **SMALL, progress=True)
    np.testing.assert_array_equal(a.assignments, b.assignments)
    np.testing.assert_array_equal(a.centers, b.centers)


def test_integer_input_equals_float_input():
    x = np.rint(QUAKES.array("x") * 10)
    a = create_dn(x.astype(np.int64), **SMALL)
    b = create_dn(x, **SMALL)
    np.testing.assert_array_equal(a.assignments, b.assignments)
    np.testing.assert_array_equal(a.centers, b.centers)


def test_numeric_dataframe_equals_array():
    x = QUAKES.array("x")
    frame = pd.DataFrame(x, columns=["lat", "long", "depth", "mag", "stations"])
    a, b = create_dn(frame, **SMALL), create_dn(x, **SMALL)
    np.testing.assert_array_equal(a.assignments, b.assignments)


def test_non_numeric_frame_rejected():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="only numeric columns"):
        create_dn(frame, r=100, dn_num1=2, dn_num2=1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_and_inf_rejected(bad):
    x = QUAKES.array("x").copy()
    x[3, 1] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        create_dn(x, **SMALL)


def test_more_nuggets_than_rows_keeps_every_row():
    x = QUAKES.array("x")[:150]
    with pytest.warns(DataNuggetWarning):
        dn = create_dn(x, r=5000, dn_num1=300, dn_num2=300)
    assert dn.n_nuggets == 150
    assert dn.weights.sum() == 150
    assert np.isfinite(dn.scales).all()


def test_to_frame_matches_r_layout():
    frame = create_dn(QUAKES.array("x"), **SMALL).to_frame()
    assert list(frame.columns) == [
        "Data Nugget",
        "Center1",
        "Center2",
        "Center3",
        "Center4",
        "Center5",
        "Weight",
        "Scale",
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"center_method": "median"}, ValueError, "center_method must be one of"),
        ({"r": 99}, ValueError, r"r must be within \[100, 10000\]"),
        ({"r": 10_001}, ValueError, r"r must be within \[100, 10000\]"),
        ({"delete_percent": 1.0}, ValueError, r"delete_percent must be within \(0, 1\)"),
        ({"dn_num1": 10, "dn_num2": 20}, ValueError, "dn_num2 must be less than dn_num1"),
        ({"dist_metric": "cosine"}, ValueError, "dist_metric must be one of"),
        ({"seed": 1.5}, TypeError, "seed must be"),
        ({"n_jobs": 0}, ValueError, "n_jobs"),
    ],
)
def test_rejects_bad_arguments(kwargs, error, match):
    args = {"r": 100, "dn_num1": 20, "dn_num2": 10} | kwargs
    with pytest.raises(error, match=match):
        create_dn(QUAKES.array("x"), **args)


def test_rejects_single_row_and_1d_input():
    with pytest.raises(ValueError, match="at least two observations"):
        create_dn(np.zeros((1, 3)), r=100, dn_num1=2, dn_num2=1)
    with pytest.raises(ValueError, match="2-D"):
        create_dn(np.zeros(10), r=100, dn_num1=2, dn_num2=1)
    with pytest.raises(ValueError, match="at least one column"):
        create_dn(np.zeros((10, 0)), r=100, dn_num1=2, dn_num2=1)
