"""f_plot against his grid: the same z, x, y and zz on the recorded files, and by hand."""

import numpy as np
import pytest

from datanuggets.flow import f_plot
from datanuggets.flow.grid import sqrt_coords
from flowcases import NAMES, load_case

# four events on a 2 by 2 grid: bins (1, 1), (1, 2), (1, 2), (2, 2)
X, Y = [0, 0, 0, 3], [0, 3, 3, 3]


@pytest.mark.parametrize("name", NAMES)
def test_f_plot_equals_his_grid(name):
    c = load_case(name)
    g = f_plot(*sqrt_coords(c.fsc, c.ssc), nr=100, nc=100, scale="l")
    for got, want in ((g.z, c.z), (g.x, c.gx), (g.y, c.gy), (g.zz, c.zz)):
        assert got.dtype == want.dtype
        assert np.array_equal(got, want)


def test_f_plot_counts_and_transposes():
    g = f_plot(X, Y, nr=2, nc=2, alpha=0)
    assert g.zz.tolist() == [[1, 1], [1, 2], [1, 2], [2, 2]]
    # his table(zx, zy) is [[1, 2], [0, 1]], returned as t(z)
    assert g.z.tolist() == [[1, 0], [2, 1]]
    assert g.x.tolist() == [0, 3]
    assert g.y.tolist() == [0, 3]


def test_f_plot_bins_float32_by_their_float64_values():
    # 4,003,041 is bin 20's lower edge from 369,044 to 2**22. In float32, 20 times 3,633,997 rounds
    # half to even down to 72,679,936, so float32 arithmetic would put the event in bin 19
    x = np.array([369044, 4003041, 2**22], dtype=np.float32)
    lo, hi = x.min(), x.max()
    assert np.floor(20 * (x[1] - lo) / (hi - lo)) + 1 == 19
    assert f_plot(x, [0, 1, 2]).zz[:, 0].tolist() == [1, 20, 20]


def test_alpha_cuts_below_the_fraction_of_the_maximum():
    assert f_plot(X, Y, nr=2, nc=2, alpha=0.6).z.tolist() == [[0, 0], [2, 0]]


@pytest.mark.parametrize(("scale", "f"), [("l", np.log1p), ("d", lambda v: np.log1p(np.log1p(v)))])
def test_scales(scale, f):
    g = f_plot(X, Y, nr=2, nc=2, scale=scale, alpha=0)
    assert np.array_equal(g.z, f(np.array([[1.0, 0.0], [2.0, 1.0]])))


@pytest.mark.parametrize(
    ("x", "y", "match"),
    [
        ([0, np.nan], [0, 1], "x has NaN"),
        ([0, 1], [0, np.inf], "y has NaN"),
        ([2, 2], [0, 1], "x has one value"),
        ([0, 1, 2], [0, 1], "same length"),
        ([], [], "non-empty"),
        ([[0, 1]], [[0, 1]], "one-dimensional"),
    ],
)
def test_f_plot_rejects_what_it_cant_bin(x, y, match):
    with pytest.raises(ValueError, match=match):
        f_plot(x, y)


def test_f_plot_checks_its_options():
    with pytest.raises(ValueError, match="scale"):
        f_plot([0, 1], [0, 1], scale="log")
    with pytest.raises(ValueError, match="nr"):
        f_plot([0, 1], [0, 1], nr=0)
    with pytest.raises(TypeError, match="alpha"):
        f_plot([0, 1], [0, 1], alpha="0.1")
