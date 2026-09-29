import numpy as np
import pytest

from datanuggets import CentersResult, DataNuggets


def _dn(**over):
    fields = {
        "centers": np.array([[0.0, 1.0], [np.nan, np.nan]]),
        "weights": np.array([3, 0]),
        "scales": np.array([0.5, 0.0]),
        "assignments": np.array([0, 0, 0]),
    }
    fields.update(over)
    return DataNuggets(**fields)


def test_result_copies_the_callers_arrays():
    weights = np.array([3, 0])
    dn = _dn(weights=weights)
    weights[0] = 99
    assert dn.weights[0] == 3
    assert weights.flags.writeable


def test_result_arrays_are_read_only():
    with pytest.raises(ValueError):
        _dn().centers[0, 0] = 5.0


def test_lists_become_arrays():
    assert isinstance(_dn(weights=[3, 0]).weights, np.ndarray)


def test_equal_by_value_with_nan_centers():
    assert _dn() == _dn()
    assert _dn() != _dn(weights=np.array([2, 1]))
    assert _dn() != _dn(assignments=np.array([0, 0]))


def test_not_equal_to_other_types():
    assert _dn() != "nuggets"
    assert CentersResult(centers=np.zeros((1, 2)), kept=np.array([0])) != _dn()


def test_results_are_unhashable():
    with pytest.raises(TypeError):
        hash(_dn())
    with pytest.raises(TypeError):
        hash(CentersResult(centers=np.zeros((1, 2)), kept=np.array([0])))


def test_centers_result_equality():
    a = CentersResult(centers=np.eye(2), kept=np.array([0, 1]))
    assert a == CentersResult(centers=np.eye(2), kept=np.array([0, 1]))
    assert a != CentersResult(centers=np.eye(2), kept=np.array([1, 0]))
