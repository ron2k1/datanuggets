"""hull_gate and lymphmacro_finder: his masks on his cells (spec criterion 3), his pipeline with
the corrected area end to end, his masks again when given his area rule, and the departures."""

import numpy as np
import pytest

from datanuggets.flow import hull_gate, lymphmacro_finder
from datanuggets.flow.gate import _finder
from flowcases import NAMES, FlowCase, assert_gate_close, load_case, mirrored_area


@pytest.mark.parametrize("name", NAMES)
def test_hull_gate_on_his_cells_gives_his_masks(name):
    c = load_case(name)
    for k in range(len(c.peaks)):
        cells = FlowCase.cells(c.his_area, k)
        got = hull_gate(c.fsc, c.ssc, c.zz, cells)
        assert_gate_close(c.fsc, c.ssc, got, c.his_masks[:, k], c.zz, cells)


@pytest.mark.parametrize("name", NAMES)
def test_finder_gives_his_pipeline_with_the_corrected_area(name):
    c = load_case(name)
    got = lymphmacro_finder(c.fsc, c.ssc)
    assert got.shape == c.fixed_masks.shape
    for k in range(got.shape[1]):
        cells = FlowCase.cells(c.fixed_area, k)
        assert_gate_close(c.fsc, c.ssc, got[:, k], c.fixed_masks[:, k], c.zz, cells)


@pytest.mark.parametrize("name", NAMES)
def test_finder_with_his_area_rule_gives_his_masks(name):
    c = load_case(name)
    got = _finder(c.fsc, c.ssc, 11, 11, 22, 0.25, 20, 0.5, mirrored_area)
    assert got.shape == c.his_masks.shape
    for k in range(got.shape[1]):
        cells = FlowCase.cells(c.his_area, k)
        assert_gate_close(c.fsc, c.ssc, got[:, k], c.his_masks[:, k], c.zz, cells)


def test_finder_takes_lists_and_float32():
    c = load_case(NAMES[0])
    want = lymphmacro_finder(c.fsc, c.ssc)
    got = lymphmacro_finder(c.fsc.astype(np.float32), c.ssc.astype(np.float32))
    assert np.array_equal(got, want)
    n = 5000
    got = lymphmacro_finder(c.fsc[:n].tolist(), c.ssc[:n].tolist())
    assert np.array_equal(got, lymphmacro_finder(c.fsc[:n], c.ssc[:n]))


def test_no_peak_gives_zero_columns():
    rng = np.random.default_rng(0)
    n = 20_000
    fsc = rng.uniform(0, 1e6, n)
    ssc = np.where(rng.random(n) < 0.5, rng.normal(0, 1e3, n), rng.normal(1e6, 1e3, n))
    got = lymphmacro_finder(fsc, ssc)  # every peak in the first or last rows, so filtered out
    assert got.shape == (n, 0)
    assert got.dtype == bool


def test_collinear_cells_give_an_empty_population():
    c = load_case(NAMES[0])
    line = np.column_stack((np.arange(40, 46), np.full(6, 50)))
    assert not hull_gate(c.fsc, c.ssc, c.zz, line).any()


def test_his_guard_needs_more_than_3_cells():
    rng = np.random.default_rng(0)
    n = 200
    fsc, ssc = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    zz = np.tile([40, 50], (n, 1))
    tri = np.array([[30, 30], [30, 70], [70, 50]])  # holds bin (40, 50) and could make a hull
    assert not hull_gate(fsc, ssc, zz, tri).any()
    assert hull_gate(fsc, ssc, zz, np.vstack((tri, [50, 50]))).all()  # the same hull, 4 cells


def test_his_guard_needs_more_than_3_kept_events():
    square = np.array([[40, 40], [40, 60], [60, 40], [60, 60]])
    fsc = np.array([0.0, 10.0, 0.0, 2.0, 10.0])
    ssc = np.array([0.0, 0.0, 10.0, 2.0, 10.0])
    zz = np.array([[50, 50], [50, 50], [50, 50], [90, 90], [50, 50]])
    # the first 3 events are kept and could make a hull holding event 4, whose bin isn't kept
    assert not hull_gate(fsc[:4], ssc[:4], zz[:4], square).any()
    assert hull_gate(fsc, ssc, zz, square).all()


def test_collinear_events_give_an_empty_population():
    n = 400
    fsc = np.linspace(1.0, 100.0, n)
    ssc = np.full(n, 7.0)
    ssc[0] = 0.0  # one event off the line, in a bin of its own
    zz = np.column_stack((np.full(n, 2), np.arange(n) // 4 + 1))
    zz[0] = (1, 1)
    cells = np.array([[2, 10], [2, 40], [3, 10], [3, 40]])
    assert not hull_gate(fsc, ssc, zz, cells).any()  # the kept events all have ssc 7


def test_nearly_collinear_events_give_an_empty_population():
    rng = np.random.default_rng(0)
    n = 2000
    fsc = rng.uniform(4e5, 1e6, n)
    ssc = 3e5 + 1e-9 * rng.standard_normal(n)  # ConvexHull takes it, Delaunay calls it flat
    zz = np.full((n, 2), 50)
    cells = np.array([[40, 40], [40, 60], [60, 40], [60, 60]])
    assert not hull_gate(fsc, ssc, zz, cells).any()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_finder_rejects_nan_and_inf(bad):
    fsc = np.arange(100.0)
    fsc[3] = bad
    with pytest.raises(ValueError, match="fsc_a has NaN"):
        lymphmacro_finder(fsc, np.arange(100.0))


def test_finder_rejects_a_constant_channel_and_unequal_lengths():
    with pytest.raises(ValueError, match="ssc_a has one value"):
        lymphmacro_finder(np.arange(100.0), np.full(100, 5.0))
    with pytest.raises(ValueError, match="same length"):
        lymphmacro_finder(np.arange(100.0), np.arange(99.0))
