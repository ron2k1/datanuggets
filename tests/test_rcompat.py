import math
from fractions import Fraction

import numpy as np
import pytest

from datanuggets._rcompat import r_mean, split_indices
from rfixtures import load_group


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
    np.testing.assert_array_equal(r_mean(x, axis=0), case.array("col_means"))
    np.testing.assert_array_equal(r_mean(x, axis=1), case.array("row_means"))


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
