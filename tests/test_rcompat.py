import math
from fractions import Fraction

import numpy as np
import pytest

from datanuggets._compat import rcompat
from datanuggets._compat.rcompat import _r_mean_py, r_mean, split_indices
from rfixtures import fixture_data, hex_bits, load_group, same_bits


def test_split_indices_matches_r():
    (case,) = load_group("rcompat_split_indices")
    nx, ncl, nchunks = case.array("nx"), case.array("ncl"), case.array("nchunks")
    lengths = case.array("chunk_len")
    pos = 0
    for n, c, k in zip(nx.tolist(), ncl.tolist(), nchunks.tolist(), strict=True):
        chunks = split_indices(n, c)
        expected = lengths[pos : pos + k].tolist()
        pos += k
        assert [ch.size for ch in chunks] == expected, (n, c)
        np.testing.assert_array_equal(np.concatenate(chunks), np.arange(n))
    assert pos == lengths.size


@pytest.mark.parametrize("case", load_group("rcompat_means"), ids=lambda c: c.id)
def test_r_mean_matches_r_colmeans_and_rowmeans(case):
    x = case.array("x")
    for mean in (r_mean, _r_mean_py):  # bit for bit, so a NaN has to be the one R gives
        for axis, want in ((0, case.array("col_means")), (1, case.array("row_means"))):
            got = mean(x, axis)
            assert same_bits(got, want), (mean.__name__, axis, hex_bits(got), hex_bits(want))


def test_r_mean_handles_sums_beyond_the_double_range():
    # the partial sum 2e308 overflows a double but not R's long double
    np.testing.assert_array_equal(
        r_mean(np.array([[1e308, 1e308, -1e308]]), axis=1), [float(Fraction(1e308) / 3)]
    )
    got = r_mean(np.array([[math.inf, 1.0], [math.inf, -math.inf]]), axis=1)
    assert got[0] == math.inf and math.isnan(got[1])


def test_r_mean_of_no_values_is_nan_like_r():
    # R's colMeans of a 0-row matrix divides 0 by 0
    got = r_mean(np.empty((0, 3)), axis=0)
    assert got.shape == (3,) and np.isnan(got).all()
    assert np.isnan(r_mean(np.empty((2, 0)), axis=1)).all()


DATA = fixture_data()
BIGGEST = max(DATA, key=lambda k: DATA[k][0].size)


@pytest.mark.parametrize("key", sorted(DATA))
def test_r_mean_matches_the_replay_on_every_fixture_matrix(key):
    x = DATA[key][0]
    for axis in (0, 1):
        assert same_bits(r_mean(x, axis), _r_mean_py(x, axis)), axis


def test_r_mean_uses_the_compiled_path_for_finite_lines(monkeypatch):
    def replay(v):
        raise AssertionError("r_mean fell back to the Python replay")

    monkeypatch.setattr(rcompat, "_ldouble_mean", replay)
    x = DATA[BIGGEST][0]
    assert r_mean(x, 0).shape == (x.shape[1],)


EDGE_LINES = [
    [5e-324, -5e-324, 1e-310, 2.5e-320],
    [-0.0, -0.0, -0.0],
    [0.0, -0.0],
    [-1.5e-323, 5e-324],
    [1e308, 1e308, -1e308],
    [1.7e308, 1.7e308, 1.7e308, 1.7e308],
    [1e16, 1.0, -1e16],
    [1.0, 2.0**-60, 2.0**-60, 2.0**-60],
]


def test_r_mean_edge_lines_match_the_replay():
    for line in EDGE_LINES:
        a = np.array([line])
        assert same_bits(r_mean(a, 1), _r_mean_py(a, 1)), line


def test_r_mean_takes_any_layout_and_dtype():
    x = DATA[BIGGEST][0]
    for axis in (0, 1):
        want = _r_mean_py(x, axis)
        for view in (np.asfortranarray(x), np.repeat(x, 2, axis=0)[::2]):
            assert same_bits(r_mean(view, axis), want), axis
    ints = np.rint(x * 10).astype(np.int64)
    assert same_bits(r_mean(ints, 0), _r_mean_py(ints.astype(np.float64), 0))
    cube = np.stack([x, -x])
    assert same_bits(r_mean(cube, 1), _r_mean_py(cube, 1))


def test_r_mean_mixes_finite_and_non_finite_lines():
    x = np.array(
        [
            [1.0, 2.0, 4.0],
            [math.inf, 1.0, 2.0],
            [0.1, 0.2, 0.3],
            [math.nan, 1.0, 1.0],
            [math.inf, -math.inf, 0.0],
        ]
    )
    got, want = r_mean(x, 1), _r_mean_py(x, 1)
    np.testing.assert_array_equal(got, want)  # NaN matches NaN here
    assert same_bits(got[[0, 2]], want[[0, 2]])
