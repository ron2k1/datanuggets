"""R's long-double statistics, unique() and %*% replayed bit for bit; eigen() to Numeric tier."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import pytest

import datanuggets._rstats as rstats
from datanuggets._rcompat import _r_mean_py, unique_first
from datanuggets._rstats import (
    _r_cov_py,
    group_moments,
    r_cov,
    r_eigen_sym,
    r_mad,
    r_matvec,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
from rfixtures import ATOL, RTOL, fixture_data, load_group, same_bits

CASES = load_group("rcompat_stats")


def _kind(kind):
    return [c for c in CASES if c.args["kind"] == kind]


COV = _kind("cov")
EIGEN = [c for c in COV if c.has("values")]


@pytest.mark.parametrize("case", COV, ids=lambda c: c.id)
def test_cov_and_scale_match_r_bitwise(case):
    x = case.array("x")
    s = r_cov(x)
    np.testing.assert_array_equal(s, case.array("cov"))
    assert r_sum(np.diag(s)) / x.shape[1] == case.array("scale")[0]


@pytest.mark.parametrize("case", EIGEN, ids=lambda c: c.id)
def test_eigen_matches_r_numerically(case):
    s = case.array("cov")
    np.testing.assert_allclose(r_eigen_sym(s), case.array("values"), rtol=RTOL, atol=ATOL)
    values, vectors = r_eigen_sym(s, vectors=True)
    np.testing.assert_allclose(values, case.array("values_v"), rtol=RTOL, atol=ATOL)
    got, want = vectors[:, 0], case.array("vectors")[:, 0]
    np.testing.assert_allclose(got * np.sign(got @ want), want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", EIGEN, ids=lambda c: c.id)
def test_matvec_matches_r_bitwise(case):
    got = r_matvec(case.array("x"), case.array("vectors")[:, 0])
    np.testing.assert_array_equal(got, case.array("proj"))


@pytest.mark.parametrize("case", _kind("vec"), ids=lambda c: c.id)
def test_vector_stats_match_r_bitwise(case):
    v = case.array("v").tolist()
    assert r_sum(v) == case.array("sum")[0]
    assert r_vec_mean(v) == case.array("mean")[0]
    assert r_median(v) == case.array("median")[0]
    assert r_mad(v) == case.array("mad")[0]
    got = [r_quantile7(v, q) for q in case.args["probs"]]
    np.testing.assert_array_equal(got, case.array("q"))


@pytest.mark.parametrize("case", _kind("mean"), ids=lambda c: c.id)
def test_mean_of_non_finite_values_matches_r(case):
    np.testing.assert_array_equal(r_vec_mean(case.array("v").tolist()), case.array("mean")[0])


@pytest.mark.parametrize("case", _kind("unique"), ids=lambda c: c.id)
def test_unique_first_matches_r(case):
    np.testing.assert_array_equal(unique_first(case.array("x")) + 1, case.array("first"))


DATA = fixture_data()
BIGGEST = max((k for k in DATA if DATA[k][1]), key=lambda k: DATA[k][0].size)


def _nuggets(a):
    """Each nugget's rows in ascending order, for 0-based assignments a."""
    order = np.argsort(a, kind="stable")
    return np.split(order, np.cumsum(np.bincount(a))[:-1])


@pytest.mark.parametrize("key", sorted(DATA))
def test_r_cov_matches_the_replay_on_every_fixture_nugget(key):
    x, sets = DATA[key]
    for rows in [np.arange(x.shape[0])] + [g for a in sets for g in _nuggets(a)]:
        if rows.size >= 2:
            assert same_bits(r_cov(x[rows]), _r_cov_py(x[rows])), rows[:5]


def test_r_cov_uses_the_compiled_path_for_finite_data(monkeypatch):
    def replay(x):
        raise AssertionError("r_cov fell back to the Python replay")

    monkeypatch.setattr(rstats, "_r_cov_py", replay)
    x = DATA[BIGGEST][0]
    assert r_cov(x).shape == (x.shape[1], x.shape[1])


EDGE_COLUMNS = {
    "subnormal": [5e-324, -5e-324, 1e-310, 0.0, 2.5e-320, -1e-315],
    "signed_zero": [0.0, -0.0, -0.0, 0.0, -0.0, 0.0],
    "constant": [3.7] * 6,
    "big_offset": [1e15 + 0.125 * i for i in range(6)],
    "near_one": [1.0, 1.0 + 2**-52, 1.0, 1.0 - 2**-53, 1.0, 1.0 + 2**-51],
    "huge": [1e300, -1e300, 1e300, -1e300, 1e300, 1.7e308],
}


def test_r_cov_edge_columns_match_the_replay():
    # every pair of columns meets in one cov() call, and "huge" gives covariances past DBL_MAX
    x = np.column_stack(list(EDGE_COLUMNS.values()))
    got = r_cov(x)
    assert same_bits(got, _r_cov_py(x))
    assert np.isinf(got[-1, -1])


def test_r_cov_takes_any_layout_and_dtype():
    x = DATA[BIGGEST][0]
    want = _r_cov_py(x)
    for view in (np.asfortranarray(x), np.repeat(x, 2, axis=0)[::2]):
        assert same_bits(r_cov(view), want)
    ints = np.rint(x * 10).astype(np.int64)
    assert same_bits(r_cov(ints), _r_cov_py(ints.astype(np.float64)))


@pytest.mark.parametrize(("bad", "error"), [(np.inf, OverflowError), (np.nan, ValueError)])
def test_r_cov_raises_on_inf_and_nan_like_the_replay(bad, error):
    x = np.array([[1.0, 2.0], [bad, 3.0], [4.0, 5.0]])
    with pytest.raises(error, match="cannot convert"):
        r_cov(x)


MOMENT_SETS = [(k, i) for k in sorted(DATA) for i in range(len(DATA[k][1]))]


def _same_moments(got, want) -> bool:
    return len(got) == len(want) and all(
        same_bits(c, wc) and (v is None if wv is None else same_bits(v, wv))
        for (c, v), (wc, wv) in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize(("key", "index"), MOMENT_SETS, ids=[f"{k}#{i}" for k, i in MOMENT_SETS])
def test_group_moments_match_the_replay_on_every_fixture_nugget(key, index, n_jobs):
    x, sets = DATA[key]
    groups = _nuggets(sets[index])
    got = list(group_moments(x, groups, n_jobs))
    want = [(_r_mean_py(x[g], 0), _r_cov_py(x[g]) if g.size >= 2 else None) for g in groups]
    assert _same_moments(got, want)


def test_group_moments_chunking_gives_the_same_result(monkeypatch):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    p = x.shape[1]
    for chunk in (1, 3 * p * p):  # one group per call, then three
        monkeypatch.setattr(rstats, "MOMENTS_CHUNK", chunk)  # rstats' own binding of it
        assert _same_moments(list(group_moments(x, groups)), want), chunk
        assert _same_moments(list(group_moments(x, groups, 4)), want), chunk


@pytest.mark.parametrize("n_jobs", [-1, 10**6])
def test_group_moments_caps_and_restores_threads(monkeypatch, n_jobs):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    before = numba.get_num_threads()
    asked = []
    real = numba.set_num_threads

    def spy(k):
        asked.append(k)
        real(k)

    monkeypatch.setattr(numba, "set_num_threads", spy)
    assert _same_moments(list(group_moments(x, groups, n_jobs)), want)
    assert asked == [numba.config.NUMBA_NUM_THREADS, before]
    assert numba.get_num_threads() == before


def test_group_moments_from_two_threads_at_once():
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    start = threading.Barrier(2)

    def run():
        start.wait()
        return list(group_moments(x, groups, 4))

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results = [f.result() for f in futures]
    assert all(_same_moments(got, want) for got in results)


def test_group_moments_leaves_non_finite_groups_to_the_replay():
    x = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0], [6.0, 7.0]])
    groups = [np.array([1]), np.array([2, 3]), np.array([], dtype=np.int64)]
    (c0, v0), (c1, v1), (c2, v2) = group_moments(x, groups)
    assert same_bits(c0, [np.inf, 3.0]) and v0 is None
    assert same_bits(c1, _r_mean_py(x[[2, 3]], 0)) and same_bits(v1, _r_cov_py(x[[2, 3]]))
    assert np.isnan(c2).all() and v2 is None
    with pytest.raises(OverflowError):
        list(group_moments(x, [np.array([0, 1])]))
