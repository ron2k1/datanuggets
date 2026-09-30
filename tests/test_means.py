"""get_dn_means against R's getDN.means on every nugget set in the create and refine fixtures."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from datanuggets import DataNuggets, get_dn_means
from rfixtures import load_group, means_inputs

(ORDER,) = [m for m in load_group("get_dn_means") if m.id == "row_order"]
MEANS = [m for m in load_group("get_dn_means") if m.id != "row_order"]
PAIRS = [(means, name) for means in MEANS for name in means.manifest["arrays"]]
X, DN = means_inputs(next(m for m in MEANS if m.id == "create_dn"), "faithful_euclidean_mean")


@pytest.mark.parametrize(("means", "name"), PAIRS, ids=[f"{m.id}-{n}" for m, n in PAIRS])
def test_get_dn_means_matches_r(means, name):
    x, dn = means_inputs(means, name)
    want = means.array(name)
    k, p = dn.centers.shape
    if p == 1 and k > 1:
        # R's t(sapply(...)) turns one column's means into a single row
        assert want.shape == (1, k)
        want = want.T
    np.testing.assert_array_equal(get_dn_means(x, dn), want)


def test_sums_each_nugget_in_row_order_like_r():
    # Reversed rows change no real set's means, so a constructed set recorded from R pins R's order
    x = ORDER.array("x")
    dn = DataNuggets(
        np.zeros((2, 2)), np.array([3, 3]), np.zeros(2), ORDER.array("assignments") - 1
    )
    want = ORDER.array("means")
    assert want[0, 0] != 0  # R sums 1e20 - 1e20 + 1, not 1 - 1e20 + 1e20
    np.testing.assert_array_equal(get_dn_means(x, dn), want)


def test_fixtures_cover_empty_nuggets_and_one_column():
    sets = [means_inputs(m, n)[1] for m, n in PAIRS]
    assert any((dn.weights == 0).any() for dn in sets)
    assert any(dn.centers.shape[1] == 1 and dn.n_nuggets > 1 for dn in sets)


def test_dn_must_be_datanuggets():
    with pytest.raises(TypeError, match="dn must be a DataNuggets"):
        get_dn_means(X, {"centers": DN.centers, "assignments": DN.assignments})


def test_x_must_be_numeric_and_2d():
    with pytest.raises(ValueError, match="2-D"):
        get_dn_means(X[:, 0], DN)
    with pytest.raises(TypeError, match="numeric"):
        get_dn_means(X.astype(str), DN)


@pytest.mark.parametrize("rows", [np.arange(100), np.r_[0:272, 0:10]], ids=["fewer", "more"])
def test_x_must_have_the_rows_dn_assigns(rows):
    # R stops on fewer rows (subscript out of bounds) and silently drops extra ones
    with pytest.raises(ValueError, match="one entry per row"):
        get_dn_means(X[rows], DN)


def test_x_must_have_dn_columns():
    # R stops too: the means get dn's column names, and the lengths differ
    with pytest.raises(ValueError, match="columns"):
        get_dn_means(np.hstack([X, X[:, :1]]), DN)


def test_assignments_must_be_nugget_ids():
    for bad in (-1, DN.n_nuggets, 0.5):
        a = DN.assignments.astype(np.float64)
        a[5] = bad
        with pytest.raises(ValueError, match="nugget ids|whole numbers"):
            get_dn_means(X, replace(DN, assignments=a))


def test_x_must_be_finite():
    # R's colMeans passes NA through to that nugget's mean. The package rejects NaN everywhere.
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        get_dn_means(bad, DN)


def test_accepts_a_dataframe():
    frame = pd.DataFrame({"eruptions": X[:, 0], "waiting": X[:, 1].astype(np.int64)})
    np.testing.assert_array_equal(get_dn_means(frame, DN), get_dn_means(X, DN))
