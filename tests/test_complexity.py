"""arithmetic_complexity against R's AC over the row and column counts of real R datasets."""

import math
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from datanuggets import DataNuggetWarning, arithmetic_complexity
from datanuggets.complexity import _ac_state
from rfixtures import ATOL, RTOL, exact_ac_count, load_group

(GRID,) = load_group("ac")
ARGS = ("r", "delete_percent", "dn_num1", "dn_num2")
GOOD = {"r": 5000, "delete_percent": 0.1, "dn_num1": 10_000, "dn_num2": 2000}


def _rows(hung: bool) -> list[dict]:
    names = ("n", "p", *ARGS, "warned", "g", "r_end", "a", "dn_num1_end", "b", "value")
    cols = {name: GRID.array(name).tolist() for name in names}
    return [
        {name: col[i] for name, col in cols.items()}
        for i in np.flatnonzero(GRID.array("hung") == int(hung)).tolist()
    ]


def _call(row: dict) -> float:
    x = np.broadcast_to(0.0, (row["n"], row["p"]))
    return arithmetic_complexity(x, *(row[a] for a in ARGS))


def test_counts_value_and_warning_match_r():
    rows, bad = _rows(hung=False), []
    for row in rows:
        state = _ac_state(row["n"], *(row[a] for a in ARGS))[:5]
        want = (row["g"], row["r_end"], row["a"], row["dn_num1_end"], row["b"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = _call(row)
        warned = sum(issubclass(w.category, DataNuggetWarning) for w in caught)
        close = np.isclose(value, row["value"], rtol=RTOL, atol=ATOL, equal_nan=True)
        if state != want or not close or warned != row["warned"]:
            bad.append((row, state, value, warned))
    assert not bad, f"{len(bad)} of {len(rows)} calls differ from R, first: {bad[:2]}"
    assert any(row["warned"] for row in rows) and any(row["value"] == -math.inf for row in rows)


def test_raises_where_r_never_returns():
    rows = _rows(hung=True)
    assert rows
    for row in rows:
        with pytest.raises(ValueError, match="never returns"):
            _call(row)


def test_grid_needs_r_double_rounding():
    """In some rows double rounding changes R's count A, so a port that ran the loop in exact
    arithmetic would fail test_counts_value_and_warning_match_r."""
    differs = 0
    for row in _rows(hung=False):
        r, dn1 = math.floor(row["r"]), math.floor(row["dn_num1"])
        target = Fraction(dn1, -(-row["n"] // r))
        differs += exact_ac_count(r, target, row["delete_percent"]) != row["a"]
    assert differs > 0


def test_reads_only_the_shape_of_any_2d_input():
    # R 4.6.1's AC stops on a matrix or a data.table and takes any plain data frame, text too
    (want,) = [
        row["value"]
        for row in _rows(hung=False)
        if (row["n"], row["p"]) == (272, 2) and all(row[a] == GOOD[a] for a in ARGS)
    ]
    frame = pd.DataFrame({"label": ["a"] * 272, "count": range(272)})
    for x in (np.zeros((272, 2)), frame, [[0.0, 0.0]] * 272):
        assert arithmetic_complexity(x, **GOOD) == pytest.approx(want, rel=RTOL, abs=ATOL)


def test_numpy_scalars_act_like_python_numbers():
    x = np.zeros((272, 2))
    got = arithmetic_complexity(
        x, np.int64(5000), np.float64(0.1), np.int64(10_000), np.int32(2000)
    )
    assert got == arithmetic_complexity(x, **GOOD)


BAD_ARGS = [
    ({"r": "5000"}, TypeError, "r must be a number"),
    ({"r": True}, TypeError, "r must be a number"),
    ({"r": 99.5}, ValueError, "r must be within"),
    ({"r": 10_000.5}, ValueError, "r must be within"),
    ({"r": math.nan}, ValueError, "r must be within"),
    ({"delete_percent": "0.1"}, TypeError, "delete_percent must be a number"),
    ({"delete_percent": 0}, ValueError, "delete_percent must be within"),
    ({"delete_percent": 1}, ValueError, "delete_percent must be within"),
    ({"dn_num1": "1e4"}, TypeError, "dn_num1 must be a number"),
    ({"dn_num2": None}, TypeError, "dn_num2 must be a number"),
    # R stops on NaN, stops or loops on most infinite values and returns Inf when both are Inf
    ({"dn_num1": math.nan}, ValueError, "dn_num1 must be finite"),
    ({"dn_num1": math.inf, "dn_num2": math.inf}, ValueError, "dn_num1 must be finite"),
    ({"dn_num2": -math.inf}, ValueError, "dn_num2 must be finite"),
    # R compares before flooring, so 100.7 > 100.5 stops though both floor to 100
    ({"dn_num1": 100.5, "dn_num2": 100.7}, ValueError, "dn_num2 must not exceed dn_num1"),
]


@pytest.mark.parametrize(("kwargs", "exc", "match"), BAD_ARGS)
def test_rejects_what_r_rejects(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        arithmetic_complexity(np.zeros((272, 2)), **{**GOOD, **kwargs})


def test_x_must_be_2d_with_rows():
    with pytest.raises(ValueError, match="2-D"):
        arithmetic_complexity(np.zeros(272), **GOOD)
    # R returns NaN for a data frame with no rows
    with pytest.raises(ValueError, match="at least one row"):
        arithmetic_complexity(np.zeros((0, 2)), **GOOD)
    # and -Inf, log10(0), for one with no columns, which is kept
    assert arithmetic_complexity(np.zeros((272, 0)), **GOOD) == -math.inf
