"""find_area: the corrected rule on the recorded peaks and on random grids, the blob his version
misses, and his cells as the mirrored disc on real data."""

import numpy as np
import pytest

from datanuggets.flow import find_area
from flowcases import NAMES, FlowCase, T, fixed_area, load_case, mirrored_area


def _thresholds(k):
    return T[0] if k == 0 else T[1]


def test_a_blob_off_the_diagonal_keeps_its_121_cells():
    x = np.zeros((100, 100))
    x[15:26, 65:76] = 5.0
    cells = find_area(x, 20, 70, t1=22, t2=0.25)
    assert len(cells) == 121
    assert set(map(tuple, cells.tolist())) == {(i, j) for i in range(16, 27) for j in range(66, 77)}
    assert len(mirrored_area(x, 20, 70, 22, 0.25)) == 0  # what his version keeps


@pytest.mark.parametrize("seed", range(20))
def test_find_area_equals_the_rule_stated_directly(seed):
    rng = np.random.default_rng(seed)
    x = rng.gamma(1.0, size=(100, 100))
    idx, idy = (int(v) for v in rng.integers(1, 101, size=2))
    assert np.array_equal(find_area(x, idx, idy, t1=20, t2=0.5), fixed_area(x, idx, idy, 20, 0.5))


@pytest.mark.parametrize("name", NAMES)
def test_find_area_equals_the_recorded_fixed_cells(name):
    c = load_case(name)
    for k, (i, j) in enumerate(c.peaks):
        got = find_area(c.z.T, i, j, *_thresholds(k))
        assert np.array_equal(got, FlowCase.cells(c.fixed_area, k))


@pytest.mark.parametrize("name", NAMES)
def test_his_cells_are_the_mirrored_disc(name):
    c = load_case(name)
    for k, (i, j) in enumerate(c.peaks):
        want = FlowCase.cells(c.his_area, k)
        assert np.array_equal(mirrored_area(c.z.T, i, j, *_thresholds(k)), want)


def test_on_the_diagonal_his_rule_and_ours_agree():
    x = np.random.default_rng(1).gamma(1.0, size=(100, 100))
    assert np.array_equal(find_area(x, 50, 50, t1=20, t2=0.5), mirrored_area(x, 50, 50, 20, 0.5))
