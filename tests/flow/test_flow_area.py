"""find_area: the corrected rule on the recorded peaks and on random grids, the blob cfc.py's
version misses, and cfc.py's cells as the mirrored disc on real data, stated directly and as
datanuggets.flow.mirrored_area."""

import numpy as np
import pytest

from datanuggets.flow import find_area
from datanuggets.flow import mirrored_area as flow_mirrored_area
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


@pytest.mark.parametrize("name", NAMES)
def test_flow_mirrored_area_gives_the_recorded_his_area_cells(name):
    c = load_case(name)
    for k, (i, j) in enumerate(c.peaks):
        got = flow_mirrored_area(c.z.T, i, j, *_thresholds(k))
        assert np.array_equal(got, FlowCase.cells(c.his_area, k))
        assert np.array_equal(got, mirrored_area(c.z.T, i, j, *_thresholds(k)))


def test_both_area_rules_take_a_nested_list_as_find_area_does():
    x = np.random.default_rng(2).gamma(1.0, size=(30, 30))
    for rule in (find_area, flow_mirrored_area):
        assert np.array_equal(rule(x.tolist(), 10, 20, 8, 0.5), rule(x, 10, 20, 8, 0.5))


def test_on_the_diagonal_his_rule_and_ours_agree():
    x = np.random.default_rng(1).gamma(1.0, size=(100, 100))
    assert np.array_equal(find_area(x, 50, 50, t1=20, t2=0.5), mirrored_area(x, 50, 50, 20, 0.5))
