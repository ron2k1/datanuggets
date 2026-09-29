"""R's long-double statistics, unique() and %*% replayed bit for bit; eigen() to Numeric tier."""

import numpy as np
import pytest

from datanuggets._rcompat import unique_first
from datanuggets._rstats import (
    r_cov,
    r_eigen_sym,
    r_mad,
    r_matvec,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
from rfixtures import ATOL, RTOL, load_group

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
