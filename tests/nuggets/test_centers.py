import tracemalloc
import warnings

import numpy as np
import pytest

from datanuggets import DataNuggetWarning, create_dn_centers
from datanuggets.nuggets import centersjit
from datanuggets.nuggets.centers import _select_centers_py, select_centers
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


@pytest.mark.parametrize("delete_percent", [0.001, 0.1, 0.5, 0.99])
@pytest.mark.parametrize("kind", ["grid", "dups", "normal"])
def test_select_centers_matches_the_python_rounds(kind, delete_percent):
    """Ties everywhere (a small grid, duplicated rows) are where a row's nearest neighbour and the
    order of equal distances decide which point goes."""
    rng = np.random.default_rng(int(delete_percent * 1000) + len(kind))
    for n in (2, 3, 17, 120, 301):
        if kind == "grid":
            rs = rng.integers(0, 3, size=(n, 2)).astype(float)
        elif kind == "dups":
            rs = rng.standard_normal((n, 3))
            rs[n // 2 :] = rs[: n - n // 2]
        else:
            rs = rng.standard_normal((n, 4))
        for dn_num in sorted({1, max(1, n // 10), max(1, n // 2), max(1, n - 1)}):
            for metric in ("euclidean", "manhattan"):
                got = select_centers(rs, dn_num, delete_percent, metric)
                want = _select_centers_py(rs, dn_num, delete_percent, metric)
                np.testing.assert_array_equal(got[0], want[0], err_msg=f"{n} {dn_num} {metric}")
                assert got[1] == want[1]


@pytest.mark.parametrize("t", [*range(1, 20), 37, 301])
def test_row_min_is_numpys_first_argmin_over_the_active_rows(t):
    """The four running minima against numpy's argmin, on rows full of ties, for every remainder
    past the last group of four and a scattered set of active rows."""
    rng = np.random.default_rng(t)
    n = t + 25
    d = rng.integers(0, 3, size=(n, n)).astype(float)
    act = np.sort(rng.choice(n, size=t, replace=False))
    for a in range(n):
        want = int(d[a, act].argmin())
        assert centersjit._nearest(d, act, t, a) == (want, d[a, act[want]])


def test_select_centers_holds_one_distance_matrix():
    """Stage 2 builds a dn_num1 x dn_num1 matrix, 800 MB at the default 10,000. The rounds must
    work inside it, not copy it down each round as R does."""
    rs = np.random.default_rng(0).standard_normal((1500, 3))
    matrix = rs.shape[0] ** 2 * 8
    tracemalloc.start()
    try:
        select_centers(rs, 150, 0.1, "euclidean")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.1 * matrix
