import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
import scipy.cluster.hierarchy

import datanuggets
import datanuggets.wcluster
from datanuggets import CentersResult, DataNuggets
from datanuggets.result import (
    DnWkmeansResult,
    WhclustResult,
    WkmeansResult,
    WpcaResult,
    WwcssResult,
)

# Each WCluster leaf and the module that defines it. The modules are reached through importlib,
# because the exported functions wpca, whclust and wkmeans shadow their modules as attributes of
# datanuggets.wcluster.
WCLUSTER_LEAVES = {
    "cluster_predict": "datanuggets.wcluster.predict",
    "distw": "datanuggets.wcluster.wstats",
    "dn_cluster_predict": "datanuggets.wcluster.dn",
    "dn_whclust": "datanuggets.wcluster.dn",
    "dn_wkmeans": "datanuggets.wcluster.dn",
    "dn_wpca": "datanuggets.wcluster.dn",
    "whclust": "datanuggets.wcluster.whclust",
    "wkmeans": "datanuggets.wcluster.wkmeans",
    "wmean": "datanuggets.wcluster.wstats",
    "wpca": "datanuggets.wcluster.wpca",
    "wss": "datanuggets.wcluster.wstats",
    "wwcss": "datanuggets.wcluster.wstats",
}
RESULTS = {
    "DnWkmeansResult": DnWkmeansResult,
    "WhclustResult": WhclustResult,
    "WkmeansResult": WkmeansResult,
    "WpcaResult": WpcaResult,
    "WwcssResult": WwcssResult,
}


def test_wcluster_leaves_are_exported_from_both_packages():
    for name, module in WCLUSTER_LEAVES.items():
        defined = getattr(importlib.import_module(module), name)
        assert getattr(datanuggets.wcluster, name, None) is defined, name
        assert getattr(datanuggets, name, None) is defined, name
    assert datanuggets.wcluster.__all__ == sorted(WCLUSTER_LEAVES)
    for name, cls in RESULTS.items():
        assert getattr(datanuggets, name, None) is cls, name
    assert set(WCLUSTER_LEAVES) | set(RESULTS) <= set(datanuggets.__all__)
    assert datanuggets.__all__ == sorted(datanuggets.__all__)
    for name in datanuggets.__all__:
        assert hasattr(datanuggets, name), name


# Each datanugget function and its module in datanuggets.nuggets. The module of the same name at
# the package root is its old import path, which forwards to it.
NUGGET_FUNCTIONS = {
    "arithmetic_complexity": "complexity",
    "create_dn": "create",
    "create_dn_centers": "centers",
    "get_dn_means": "means",
    "refine_dn": "refine",
}


def test_nugget_functions_are_exported_and_their_old_paths_forward():
    nuggets = importlib.import_module("datanuggets.nuggets")
    for name, module in NUGGET_FUNCTIONS.items():
        defined = getattr(importlib.import_module(f"datanuggets.nuggets.{module}"), name)
        assert defined.__module__ == f"datanuggets.nuggets.{module}", name
        assert getattr(nuggets, name) is defined, name
        assert getattr(datanuggets, name) is defined, name
        old = importlib.import_module(f"datanuggets.{module}")
        assert getattr(old, name) is defined, name
        assert old.__all__ == [name], name
    assert nuggets.__all__ == sorted(NUGGET_FUNCTIONS)


# The old module paths stay attributes of the package after a bare `import datanuggets`, as they
# were when its __init__ imported them, so `datanuggets.create.create_dn` still resolves. It runs
# in a fresh interpreter, because here any test that imported an old path has already bound it.
OLD_MODULES = [*sorted(NUGGET_FUNCTIONS.values()), "_draws"]


def test_a_bare_import_binds_the_old_module_paths():
    code = (
        "import sys\n"
        "import datanuggets\n"
        f"for name in {OLD_MODULES!r}:\n"
        "    assert getattr(datanuggets, name) is sys.modules['datanuggets.' + name], name\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]


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


def _wwcss(**over):
    values = {"per_cluster": np.array([1.5, np.nan]), "total": np.float64(np.nan)}
    values.update(over)
    return WwcssResult(**values)


def _wpca(**over):
    values = {
        "sdev": np.array([2.0, np.nan]),
        "rotation": np.eye(2),
        "x": np.array([[1.0, 0.0], [0.0, -1.0], [-1.0, 1.0]]),
        "center": np.array([0.5, 0.25]),
        "scale": np.ones(2),
        "wrow": np.array([1.0, 2.0, 1.0]),
        "wcol": np.ones(2),
    }
    values.update(over)
    return WpcaResult(**values)


def test_wcluster_result_fields_follow_the_spec():
    assert [f.name for f in fields(WwcssResult)] == ["per_cluster", "total"]
    assert [f.name for f in fields(WpcaResult)] == [
        "sdev",
        "rotation",
        "x",
        "center",
        "scale",
        "wrow",
        "wcol",
    ]


def test_wcluster_results_are_read_only_copies():
    per = np.array([1.5, 2.5])
    rotation = np.eye(2)
    ww, pc = _wwcss(per_cluster=per), _wpca(rotation=rotation)
    per[0] = rotation[0, 0] = 99.0
    assert ww.per_cluster[0] == 1.5 and pc.rotation[0, 0] == 1.0
    assert ww.total.shape == () and ww.total.dtype == np.float64
    for arr in (ww.per_cluster, ww.total, *(getattr(pc, f.name) for f in fields(pc))):
        assert not arr.flags.writeable
    with pytest.raises(ValueError):
        ww.per_cluster[0] = 5.0
    with pytest.raises(FrozenInstanceError):
        pc.sdev = np.zeros(2)


def test_wcluster_results_equal_by_value_with_nan():
    assert _wwcss() == _wwcss()
    assert _wwcss() != _wwcss(total=np.float64(1.5))
    assert _wwcss() != _wwcss(per_cluster=np.array([1.5, 2.5]))
    assert _wpca() == _wpca()
    assert _wpca() != _wpca(rotation=-np.eye(2))
    assert _wpca() != _wpca(wcol=np.array([0.5, 1.5]))
    assert _wwcss() != _wpca()
    with pytest.raises(TypeError):
        hash(_wpca())


def _whclust(**over):
    values = {
        "merge": np.array([[2, 3], [0, 1], [5, 4]]),
        "height": np.array([0.5, 1.0, 4.0]),
        "order": np.array([0, 1, 2, 3]),
    }
    values.update(over)
    return WhclustResult(**values)


def test_iterative_result_fields_follow_the_spec():
    assert [f.name for f in fields(WkmeansResult)] == ["labels", "centers", "wwcss", "total_wwcss"]
    assert [f.name for f in fields(DnWkmeansResult)] == [
        "labels",
        "centers",
        "wwcss",
        "total_wwcss",
        "row_labels",
    ]
    assert [f.name for f in fields(WhclustResult)] == ["merge", "height", "order"]


def test_whclust_result_to_linkage_counts_cluster_sizes():
    z = _whclust().to_linkage()
    assert z.dtype == np.float64
    assert z.tolist() == [[2, 3, 0.5, 2], [0, 1, 1.0, 2], [5, 4, 4.0, 4]]
    assert scipy.cluster.hierarchy.is_valid_linkage(z)


def test_to_linkage_keeps_heights_bit_for_bit():
    h = np.array([0.1 + 0.2, -1e-300, np.inf])
    assert _whclust(height=h).to_linkage()[:, 2].tobytes() == h.tobytes()


def test_to_linkage_refuses_a_self_merge():
    with pytest.raises(ValueError, match="itself"):
        _whclust(
            merge=np.array([[2, 3], [0, 1], [4, 4]]), order=np.array([2, 3, 2, 3])
        ).to_linkage()


def test_iterative_results_are_frozen_and_compare_by_value():
    a = WkmeansResult(np.array([0, 1]), np.ones((2, 2)), np.ones(2), np.float64(2))
    assert a == WkmeansResult(np.array([0, 1]), np.ones((2, 2)), np.ones(2), np.float64(2))
    assert a != WkmeansResult(np.array([1, 0]), np.ones((2, 2)), np.ones(2), np.float64(2))
    with pytest.raises(ValueError):
        a.labels[0] = 1
    d = DnWkmeansResult(a.labels, a.centers, a.wwcss, a.total_wwcss, np.array([1, 0, 0]))
    assert d.row_labels.tolist() == [1, 0, 0] and not d.row_labels.flags.writeable
    assert d != a
    assert _whclust() == _whclust() and _whclust() != _whclust(order=np.array([1, 0, 2, 3]))
    with pytest.raises(TypeError):
        hash(_whclust())
