"""wstatsjit's kernels against the Python primitives they replay, bit for bit."""

import numpy as np
import pytest

from datanuggets._compat._compiled import frozen
from datanuggets._compat.arithmetic import r_sum
from datanuggets.wcluster import wstatsjit
from datanuggets.wcluster.predict import _wdist_mat
from datanuggets.wcluster.wstats import _wss, wmean
from rfixtures import assert_same_values

RNG = np.random.default_rng(20261009)
BIG = 1.7976931348623157e308


def data(kind, n=40, p=3):
    if kind == "grid":
        return RNG.integers(0, 4, (n, p)).astype(float), RNG.integers(-1, 4, n).astype(float)
    if kind == "huge":
        return RNG.normal(size=(n, p)) * 1e154, RNG.integers(1, 4, n).astype(float)
    return RNG.normal(size=(n, p)), RNG.uniform(0.5, 3, n)


@pytest.mark.parametrize(
    "v",
    [
        [1.0, 1e-30, -1.0],
        [BIG, BIG, -BIG],
        [BIG, 1e292],
        [-0.0, -0.0],
        [1e-320, -1e-320, 5e-324],
        [np.inf, 1.0],
        [np.inf, -np.inf],
        [-np.inf, -np.inf],
        [np.nan, 1.0],
        [],
        list(RNG.normal(size=1000) * 10.0 ** RNG.integers(-20, 20, 1000)),
    ],
)
def test_rsum_matches_r_sum(v):
    a = np.array(v, dtype=float)
    got = wstatsjit.rsum(np.ascontiguousarray(a), a.size)
    assert_same_values(np.float64(got), np.float64(r_sum(a)))


def test_rsum_reads_only_the_first_n():
    a = np.array([1.0, 2.0, np.nan])
    assert wstatsjit.rsum(a, 2) == 3.0


@pytest.mark.parametrize("kind", ["grid", "huge", "normal"])
def test_wss_rows_and_wmean_rows_match_the_python_primitives(kind):
    x, w = data(kind)
    p = x.shape[1]
    acc, mean, out, buf, wbuf = np.empty(p), np.empty(p), np.empty(p), np.empty(40), np.empty(40)
    for _ in range(50):
        rows = np.sort(RNG.choice(40, RNG.integers(1, 40), replace=False)).astype(np.int64)
        got = wstatsjit.wss_rows(frozen(x), frozen(w), rows, rows.size, acc, mean, wbuf)
        with np.errstate(all="ignore"):
            assert_same_values(np.float64(got), np.float64(_wss(x[rows], w[rows])))
        wstatsjit.wmean_rows(frozen(x), frozen(w), rows, rows.size, out, buf, wbuf)
        assert_same_values(out, wmean(x[rows], np.zeros(rows.size), w[rows])[0])


def test_row_norms_is_wdist_mats_m11():
    x, _ = data("normal")
    want = np.zeros(40)
    for c in range(3):
        want += x[:, c] * x[:, c]
    assert_same_values(wstatsjit.row_norms(frozen(x)), want)


@pytest.mark.parametrize("kind", ["grid", "huge", "normal"])
def test_assign_is_which_min_over_wdist_mat(kind):
    x, w = data(kind)
    for k in (1, 2, 4):
        centers = x[RNG.choice(40, k, replace=False)] + RNG.integers(0, 2, (k, x.shape[1]))
        if kind == "grid" and k > 1:
            centers[0] = np.nan  # a NaN center, as a cluster whose weights sum to zero gives
        z = _wdist_mat(x, centers, w)
        want = np.array(
            [0 if np.isnan(r).all() else np.flatnonzero(r == np.nanmin(r))[0] + 1 for r in z]
        )
        out = np.empty(40, np.int64)
        dead = wstatsjit.assign(
            frozen(x),
            frozen(w),
            frozen(np.ascontiguousarray(centers)),
            k,
            wstatsjit.row_norms(frozen(x)),
            np.empty((40, k)),
            out,
        )
        assert out.tolist() == want.tolist()
        assert dead == int((want == 0).sum())


def test_assign_with_an_overflowing_row_kills_every_other_row():
    # row 2's z is Inf in every column, so R's diag(w) %*% z puts 0 * Inf, NaN, in every other
    # row, and row 2 keeps its own Inf, which which.min takes as the first minimum
    x = np.array([[0.0, 1.0], [2.0, 2.0], [1e200, 0.0]])
    c = np.array([[0.0, 0.0], [1.0, 1.0]])
    w = np.array([2.0, -1.0, 3.0])
    out = np.empty(3, np.int64)
    dead = wstatsjit.assign(
        frozen(x), frozen(w), frozen(c), 2, wstatsjit.row_norms(frozen(x)), np.empty((3, 2)), out
    )
    assert out.tolist() == [0, 0, 1] and dead == 2
