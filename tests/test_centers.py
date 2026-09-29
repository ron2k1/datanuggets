import warnings

import numpy as np
import pytest

from datanuggets import DataNuggetWarning, create_dn_centers
from rfixtures import load_group

CASES = load_group("create_dn")
CALLS = [(c, i) for c in CASES for i in range(1, c.manifest["n_calls"] + 1)]


@pytest.mark.parametrize(("case", "i"), CALLS, ids=[f"{c.id}-call{i:02d}" for c, i in CALLS])
def test_create_dn_centers_matches_r(case, i):
    rs = case.array(f"call{i:02d}_rs")
    dn_num = int(case.manifest["call_dn_num"][i - 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        out = create_dn_centers(
            rs,
            dn_num,
            delete_percent=case.args["delete.percent"],
            dist_metric=case.args["dist.metric"],
        )
    np.testing.assert_array_equal(out.kept + 1, case.array(f"call{i:02d}_kept"))
    np.testing.assert_array_equal(out.centers, rs[out.kept])


def test_returns_input_with_warning_when_dn_num_covers_all_rows():
    rs = np.arange(12.0).reshape(6, 2)
    with pytest.warns(DataNuggetWarning, match="greater than or equal"):
        out = create_dn_centers(rs, 6)
    np.testing.assert_array_equal(out.kept, np.arange(6))
    np.testing.assert_array_equal(out.centers, rs)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"delete_percent": 0.0}, ValueError, r"delete_percent must be within \(0, 1\)"),
        ({"delete_percent": 1.0}, ValueError, r"delete_percent must be within \(0, 1\)"),
        ({"dist_metric": "cosine"}, ValueError, "dist_metric must be one of"),
    ],
)
def test_rejects_bad_arguments(kwargs, error, match):
    with pytest.raises(error, match=match):
        create_dn_centers(np.zeros((10, 2)), 3, **kwargs)


def test_rejects_bad_dn_num():
    with pytest.raises(ValueError, match="dn_num must be a whole number >= 1"):
        create_dn_centers(np.zeros((10, 2)), 0)
