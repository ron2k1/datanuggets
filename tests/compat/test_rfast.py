import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from datanuggets._compat import rfastjit
from datanuggets._compat.rfast import (
    _cross_distances_py,
    arma_sum_last,
    cross_distances,
    dist_matrix,
    gnu_sort_first,
    gnu_sort_order,
    nearest_index,
)
from rfixtures import load_group, same_bits

DIST = load_group("rcompat_dist")
DISTA = load_group("rcompat_dista")


def test_arma_sum_uses_even_odd_accumulators():
    terms = np.array([[1e16, 1.0, -1e16, 1.0]])
    # Armadillo: (1e16 + -1e16) + (1.0 + 1.0) = 2; a left-to-right sum gives 1.
    assert arma_sum_last(terms)[0] == 2.0
    assert arma_sum_last(np.array([[3.0]]))[0] == 3.0


@pytest.mark.parametrize("case", DIST, ids=lambda c: c.id)
def test_dist_matrix_is_bitwise_equal_to_rfast(case):
    got = dist_matrix(case.array("x"), case.args["metric"])
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


def _fuzz_rows(rng, kind, m, p):
    """Rows built to stress the kernels: ties, huge spreads that overflow to inf, subnormals."""
    if kind == "grid":
        return rng.integers(0, 3, size=(m, p)).astype(float)
    if kind == "huge":
        return rng.standard_normal((m, p)) * 1e154
    if kind == "tiny":
        return rng.standard_normal((m, p)) * 1e-310
    return rng.standard_normal((m, p)) * 10.0 ** rng.integers(-3, 4, size=(m, 1))


KINDS = ["grid", "huge", "tiny", "mixed"]
WIDTHS = [1, 2, 3, 7, 32, 33, 54]


def _replay(a, b, metric, kernel="dist"):
    with np.errstate(over="ignore"):  # "huge" rows overflow to inf, in R too
        return _cross_distances_py(a, b, metric, kernel=kernel)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("p", WIDTHS)
def test_cross_distances_match_the_numpy_replay(p, kind):
    rng = np.random.default_rng(p * 10 + KINDS.index(kind))
    a, b = _fuzz_rows(rng, kind, 23, p), _fuzz_rows(rng, kind, 31, p)
    for metric in ("euclidean", "manhattan"):
        for kernel in ("dist", "dista"):
            want = _replay(a, b, metric, kernel)
            got = cross_distances(a, b, metric, kernel=kernel)
            assert same_bits(got, want), (metric, kernel)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("p", WIDTHS)
def test_dist_matrix_matches_the_numpy_replay(p, kind):
    x = _fuzz_rows(np.random.default_rng(100 + p * 10 + KINDS.index(kind)), kind, 41, p)
    for metric in ("euclidean", "manhattan"):
        assert same_bits(dist_matrix(x, metric), _replay(x, x, metric)), metric


@pytest.mark.parametrize("n", [63, 64, 65, 128, 129, 200])
def test_dist_matrix_mirrors_every_tile(n):
    """Dist mirrors its lower half one tile at a time: sizes on and around the tile edges, so a
    missed tile or a row cut short at a tile boundary leaves an upper entry unwritten."""
    assert rfastjit.MIRROR_TILE == 64
    x = np.random.default_rng(n).integers(0, 50, size=(n, 3)).astype(float)
    for metric in ("euclidean", "manhattan"):
        got = dist_matrix(x, metric)
        assert same_bits(got, _replay(x, x, metric)), metric
        assert same_bits(got, got.T), metric


def _nearest_index_py(points, centers, metric):
    d = _replay(points, centers, metric, "dista")
    best = d.argmin(axis=1)
    mins = d[np.arange(d.shape[0]), best]
    for r in np.flatnonzero(np.count_nonzero(d == mins[:, None], axis=1) > 1):
        best[r] = gnu_sort_first(d[r])
    return best


@pytest.mark.parametrize("k", [1, 2, 5, 16, 17, 40, 200])
@pytest.mark.parametrize("kind", KINDS)
def test_nearest_index_matches_the_numpy_replay(kind, k):
    rng = np.random.default_rng(1000 + k * 10 + KINDS.index(kind))
    for p in (2, 33):
        points, centers = _fuzz_rows(rng, kind, 150, p), _fuzz_rows(rng, kind, k, p)
        centers[k // 2 :] = centers[: k - k // 2]  # duplicated centers: exact ties
        for metric in ("euclidean", "manhattan"):
            want = _nearest_index_py(points, centers, metric)
            np.testing.assert_array_equal(nearest_index(points, centers, metric), want)


def test_distances_take_any_layout_and_dtype():
    rng = np.random.default_rng(7)
    a = rng.integers(-50, 50, size=(30, 12))
    b = np.asfortranarray(rng.standard_normal((40, 24)))[::2, ::2]
    a64, b64 = a.astype(float), np.ascontiguousarray(b)
    for metric in ("euclidean", "manhattan"):
        assert same_bits(cross_distances(a, b, metric), _replay(a64, b64, metric))
        assert same_bits(dist_matrix(b, metric), dist_matrix(b64, metric))
        np.testing.assert_array_equal(nearest_index(a, b, metric), nearest_index(a64, b64, metric))


def test_distances_reject_mismatched_columns():
    for a, b in ((np.ones((3, 2)), np.ones((4, 3))), (np.ones((3, 3)), np.ones((4, 2)))):
        with pytest.raises(ValueError, match="columns"):
            cross_distances(a, b, "euclidean")
        with pytest.raises(ValueError, match="columns"):
            nearest_index(a, b, "euclidean")


def _all_distances(a, b):
    out = []
    for metric in ("euclidean", "manhattan"):
        out += [cross_distances(a, b, metric, kernel=k) for k in ("dist", "dista")]
        out += [dist_matrix(b, metric), nearest_index(a, b, metric)]
    return out


def test_read_only_inputs_run_the_same_compiled_kernels():
    rng = np.random.default_rng(11)
    a, b = rng.standard_normal((30, 40)), rng.standard_normal((20, 40))  # 40 > 32: dista in order
    want = _all_distances(a, b)
    kernels = [k for k in vars(rfastjit).values() if isinstance(k, CPUDispatcher)]
    compiled = [len(k.signatures) for k in kernels]
    a, b = a.copy(), b.copy()
    a.setflags(write=False)
    b.setflags(write=False)
    got = _all_distances(a, b)
    assert all(same_bits(g, w) for g, w in zip(got, want, strict=True))
    assert [len(k.signatures) for k in kernels] == compiled  # no second signature to compile
