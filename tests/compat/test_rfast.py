import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from datanuggets._compat import rfastjit
from datanuggets._compat._compiled import frozen
from datanuggets._compat.rfast import cross_distances, dist_matrix, nearest_index
from rfast_reference import _cross_distances_py, arma_sum_last
from rfixtures import assert_r_nearest, load_group, same_bits, wide_ties

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


def _dista(case):
    return cross_distances(case.array("xnew"), case.array("x"), case.args["metric"], kernel="dista")


@pytest.mark.parametrize("case", DISTA, ids=lambda c: c.id)
def test_nearest_index_matches_rfast(case):
    """R's pick on every row but a tie in a case with more than 16 centers, where the port takes
    the lowest-numbered of the tied."""
    got = nearest_index(case.array("xnew"), case.array("x"), case.args["metric"])
    assert_r_nearest(got, case.array("nearest") - 1, _dista(case))


def test_r_parts_from_the_lowest_index_only_on_wide_ties():
    """R's own picks differ from the lowest index on 142 fixture rows, so the exception above is
    exercised. Every one of them is a tie in a case with more than 16 centers."""
    parted = 0
    for case in DISTA:
        lowest, wide = wide_ties(_dista(case))
        off = case.array("nearest") - 1 != lowest
        assert not (off & ~wide).any(), case.id
        parted += int(off.sum())
    assert parted == 142


@pytest.mark.parametrize("case", DISTA, ids=lambda c: c.id)
def test_rfast_full_order_ranks_the_distances(case):
    """R's whole dista order for each row, every center ranked, puts the port's distances in
    order, so the two rank the centers alike up to ties."""
    d = _dista(case)
    order = case.array("order_full") - 1
    np.testing.assert_array_equal(
        np.sort(order, axis=1), np.broadcast_to(np.arange(d.shape[1]), d.shape)
    )
    assert (np.diff(np.take_along_axis(d, order, axis=1), axis=1) >= 0).all()


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


@pytest.mark.parametrize(
    ("k", "tied"),
    [
        (5, [1, 3, 4]),
        (16, [9, 15]),
        (17, [0, 1]),  # the fewest centers where R can take another of the tied
        (40, list(range(3, 40, 2))),  # 19 tied, wider than 16
        (40, list(range(40))),  # every center tied
        (300, list(range(120, 300, 3))),
    ],
)
def test_exact_ties_go_to_the_lowest_center(k, tied):
    """The tied centers sit exactly 1 from the points, alternately on either side, and the rest
    are farther, each at its own distance."""
    centers = np.arange(2.0, 2.0 + k)[:, None]
    centers[tied, 0] = np.where(np.arange(len(tied)) % 2 == 0, 1.0, -1.0)
    for metric in ("euclidean", "manhattan"):
        got = nearest_index(np.zeros((3, 1)), centers, metric)
        np.testing.assert_array_equal(got, [tied[0]] * 3, err_msg=metric)


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
        got = np.full((n, n), np.nan)  # an entry the kernel skips stays NaN, whatever memory held
        rfastjit.dist(frozen(x), frozen(np.ascontiguousarray(x.T)), metric == "manhattan", got)
        assert same_bits(got, _replay(x, x, metric)), metric
        assert same_bits(got, got.T), metric
        assert same_bits(dist_matrix(x, metric), got), metric


def _nearest_index_py(points, centers, metric):
    return _replay(points, centers, metric, "dista").argmin(axis=1)  # the first of a tie


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


def test_distances_reject_inputs_the_kernels_cant_fill():
    """With no columns the kernels never write their output, and with no centers nearest reads
    past an empty row, so either would hand back whatever memory held. Public calls can't get
    here (as_matrix rejects zero columns, and every caller has a center), but nothing else stops
    an internal one."""
    for metric in ("euclidean", "manhattan"):
        with pytest.raises(ValueError, match="no columns"):
            cross_distances(np.zeros((3, 0)), np.zeros((2, 0)), metric)
        with pytest.raises(ValueError, match="no columns"):
            dist_matrix(np.zeros((3, 0)), metric)
        with pytest.raises(ValueError, match="no columns"):
            nearest_index(np.zeros((3, 0)), np.zeros((2, 0)), metric)
        with pytest.raises(ValueError, match="no centers"):
            nearest_index(np.ones((3, 2)), np.ones((0, 2)), metric)


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
