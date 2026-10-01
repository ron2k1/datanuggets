import numpy as np
import pytest

from datanuggets._compat.rfast import (
    arma_sum_last,
    cross_distances,
    dist_matrix,
    gnu_sort_first,
    gnu_sort_order,
    nearest_index,
)
from rfixtures import load_group

DIST = load_group("rcompat_dist")
DISTA = load_group("rcompat_dista")


def test_arma_sum_uses_even_odd_accumulators():
    terms = np.array([[1e16, 1.0, -1e16, 1.0]])
    # Armadillo: (1e16 + -1e16) + (1.0 + 1.0) = 2; a left-to-right sum gives 1.
    assert arma_sum_last(terms)[0] == 2.0
    assert arma_sum_last(np.array([[3.0]]))[0] == 3.0


@pytest.mark.parametrize("case", DIST, ids=lambda c: c.id)
def test_dist_matrix_is_bitwise_equal_to_rfast(case):
    got = dist_matrix(case.array("x"), case.args["metric"], block_rows=7)
    np.testing.assert_array_equal(got, case.array("dist"))


@pytest.mark.parametrize("case", DISTA, ids=lambda c: c.id)
def test_gnu_sort_order_reproduces_rfast_full_order(case):
    d = cross_distances(case.array("xnew"), case.array("x"), case.args["metric"], kernel="dista")
    expected = case.array("order_full") - 1
    for r in range(d.shape[0]):
        np.testing.assert_array_equal(gnu_sort_order(d[r]), expected[r], err_msg=f"row {r}")


@pytest.mark.parametrize("case", DISTA, ids=lambda c: c.id)
def test_nearest_index_matches_rfast(case):
    got = nearest_index(case.array("xnew"), case.array("x"), case.args["metric"])
    np.testing.assert_array_equal(got + 1, case.array("nearest"))


@pytest.mark.parametrize("p", [32, 33])
def test_dista_euclidean_sums_left_to_right_above_32_features(p):
    rng = np.random.default_rng(p)
    a, b = rng.standard_normal((50, p)), rng.standard_normal((40, p))
    arma = cross_distances(a, b, "euclidean")
    seq = np.zeros((50, 40))
    for j in range(p):
        seq += (b[None, :, j] - a[:, None, j]) ** 2
    np.sqrt(seq, out=seq)
    assert not np.array_equal(arma, seq)  # the data tells the two orders apart
    want = arma if p <= 32 else seq
    np.testing.assert_array_equal(cross_distances(a, b, "euclidean", kernel="dista"), want)
    np.testing.assert_array_equal(
        cross_distances(a, b, "manhattan", kernel="dista"), cross_distances(a, b, "manhattan")
    )


def test_row_min_first_index_matches_rfast():
    (case,) = load_group("rcompat_rowmins")
    np.testing.assert_array_equal(case.array("m").argmin(axis=1) + 1, case.array("index"))


def test_nearest_index_block_size_invariant():
    case = next(c for c in DISTA if c.id == "grid_dups_euclidean")
    a = nearest_index(case.array("xnew"), case.array("x"), "euclidean")
    b = nearest_index(case.array("xnew"), case.array("x"), "euclidean", block_rows=5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("depth_limit", [None, 0])
def test_gnu_sort_order_sorts(depth_limit):
    rng = np.random.default_rng(0)
    values = rng.integers(0, 5, size=300).astype(float)  # heavy ties
    order = gnu_sort_order(values, depth_limit=depth_limit)
    assert sorted(order.tolist()) == list(range(300))
    assert np.all(np.diff(values[order]) >= 0)


@pytest.mark.parametrize("depth_limit", [None, 0, 1])
def test_gnu_sort_first_is_head_of_full_order(depth_limit):
    rng = np.random.default_rng(1)
    for size in (1, 2, 5, 16, 17, 18, 33, 64, 100, 257, 1000):
        for levels in (1, 2, 3, 10):
            values = rng.integers(0, levels, size=size).astype(float)
            expected = gnu_sort_order(values, depth_limit=depth_limit)[0]
            assert gnu_sort_first(values, depth_limit=depth_limit) == expected, (size, levels)


def test_small_rows_break_ties_by_first_index():
    # 16 or fewer centers: std::sort is a plain (stable) insertion sort
    assert gnu_sort_first(np.array([3.0, 1.0, 2.0, 1.0, 1.0])) == 1
