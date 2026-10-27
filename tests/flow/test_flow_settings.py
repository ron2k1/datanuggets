"""The gate finder's settings. The constants fixed for parity keep the values of Cabrera's
cfc.py, and the tunable defaults live once, in _defaults, so a signature can't drift from them."""

import inspect

import pytest

from datanuggets import _defaults, flow
from datanuggets.flow import gate, grid, peaks

# each constant fixed for parity with cfc.py, at the value its code has
FIXED = {
    gate: {"MSIZE": 100, "GRID_SCALE": "l", "GRID_ALPHA": 0.1, "MIN_POINTS": 3},
    grid: {"SQRT_SHIFT": 3},
    peaks: {
        "LOCAL_MAX_QUANTILE": 0.85,
        "LOCAL_MAX_EDGES": "reflect",
        "PEAK_QUANTILE": 0.9,
        "ROW_MIN": 10,
        "ROW_MAX": 90,
        "COL_MAX": 90,
        "SEARCH_THRESHOLD": 2,
        "AREA_T1": 35,
        "AREA_T2": 0.85,
    },
}
# the finder's tunable defaults, as cfc.py calls it, and the functions that take each
FINDER = {"n_row": 11, "n_col": 11, "t11": 22, "t21": 0.25, "t12": 20, "t22": 0.5}
TAKES = {
    "lymphmacro_finder": tuple(FINDER),
    "finder_peaks": ("n_row", "n_col"),
    "gate_peaks": tuple(FINDER),
    "local_maxima": ("n_row", "n_col"),
}


def _default(f, name):
    return inspect.signature(f).parameters[name].default


def test_parity_constants_keep_cabreras_values():
    for module, constants in FIXED.items():
        got = {name: getattr(module, name, None) for name in constants}
        assert got == constants, module.__name__
    assert _default(flow.f_plot, "alpha") == gate.GRID_ALPHA
    assert _default(flow.local_maxima, "quantile") == peaks.LOCAL_MAX_QUANTILE
    for f in (flow.search_ind, flow.refine_peaks):
        assert _default(f, "threshold") == peaks.SEARCH_THRESHOLD, f.__name__
    for f in (flow.find_area, flow.mirrored_area):
        assert (_default(f, "t1"), _default(f, "t2")) == (peaks.AREA_T1, peaks.AREA_T2), f.__name__


@pytest.mark.parametrize("name", TAKES)
def test_finder_signature_defaults_are_the_central_defaults(name):
    central = {k: getattr(_defaults, f"FINDER_{k.upper()}", None) for k in FINDER}
    assert central == FINDER
    got = {k: _default(getattr(flow, name), k) for k in TAKES[name]}
    assert got == {k: central[k] for k in TAKES[name]}
