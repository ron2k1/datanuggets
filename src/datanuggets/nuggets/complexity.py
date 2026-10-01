"""Port of R's AC (datanugget 1.5.0): the arithmetic complexity of create_dn."""

from __future__ import annotations

import math
import warnings

import numpy as np

from datanuggets._defaults import R_MAX, R_MIN
from datanuggets._validate import as_real, check_delete_percent
from datanuggets.exceptions import DataNuggetWarning


def arithmetic_complexity(x, r, delete_percent, dn_num1, dn_num2) -> float:
    """R's AC: the arithmetic complexity of create_dn(x, r=r, ...) on a log10 scale.

    Only x's shape is read, and there are no defaults, as in R. AC counts G = ceiling(n / r)
    chunks thinned from r rows to dn_num1 / G centers in A passes and dn_num1 thinned to
    dn_num2 in B passes, each pass costing a pairwise distance matrix, plus n + 1 rows measured
    against dn_num2 centers, all times the column count. A and B come from R's two while loops,
    run here in the same double arithmetic, so they match R's counts exactly. Some arguments
    make those loops cycle and R never returns. A dn_num2 below -1 is one, unless dn_num1 floors
    to the same integer. This raises ValueError for them.
    """
    shape = np.shape(x)
    if len(shape) != 2:
        raise ValueError(f"x must be 2-D (a matrix or data frame), got {len(shape)}-D")
    n, p = shape
    if n == 0:
        raise ValueError("x must have at least one row")
    r = as_real("r", r)
    if not R_MIN <= r <= R_MAX:
        raise ValueError(f"r must be within [{R_MIN}, {R_MAX}], got {r!r}")
    delete_percent = check_delete_percent(delete_percent)
    dn_num1 = as_real("dn_num1", dn_num1)
    dn_num2 = as_real("dn_num2", dn_num2)
    for name, value in (("dn_num1", dn_num1), ("dn_num2", dn_num2)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if dn_num2 > dn_num1:
        raise ValueError("dn_num2 must not exceed dn_num1")
    g, r, a, dn_num1, b, dn_num2 = _ac_state(n, r, delete_percent, dn_num1, dn_num2)
    count = p * (
        g * ((r * (r - 1) / 2) * a) + (dn_num1 * (dn_num1 - 1) / 2) * b + (n + 1) * dn_num2
    )
    return _r_log10(count)


def _ac_state(
    n: int, r: float, delete_percent: float, dn_num1: float, dn_num2: float
) -> tuple[float, float, int, float, int, float]:
    """AC after its argument checks: floor R, DN.num1 and DN.num2, set G and run both loops.
    Returns (G, R, A, DN.num1, B, DN.num2) as R holds them when it returns."""
    r, dn_num1, dn_num2 = (float(math.floor(v)) for v in (r, dn_num1, dn_num2))
    g = float(math.ceil(n / r))
    r, a = _shrink("r", r, dn_num1 / g, delete_percent)
    dn_num1, b = _shrink("dn_num1", dn_num1, dn_num2, delete_percent)
    return g, r, a, dn_num1, b, dn_num2


def _shrink(name: str, value: float, target: float, delete_percent: float) -> tuple[float, int]:
    """One of AC's while loops: cut value by floor(delete_percent * value) until it equals
    target, jumping to target when a cut would pass it and stepping by 1 when the cut floors to
    0. Returns the final value and R's counter, 1 plus the number of passes. Every operation is
    R's double arithmetic in R's order, because the rounding of delete_percent * value and of
    value - (value - target) changes the count on real inputs.

    The loop is deterministic, so a value that comes back means R never returns. Brent's cycle
    check compares each value with one saved at power-of-two intervals, in constant memory
    however long the loop runs. A target below -1 is out of reach from any value above it, since
    the value never goes below -1, so that raises before the loop. The loop could otherwise step
    down by 1 for as many passes as value is large before its cycle begins.
    """
    if target < -1 and value > target:
        raise ValueError(
            f"R's AC never returns for these arguments: {name} never goes below -1, so it never"
            f" reaches {target!r}"
        )
    count = 1
    saved, horizon, steps = value, 1, 0
    while value != target:
        cut = float(math.floor(delete_percent * value))
        if value - cut < target:
            value = value - (value - target)
        elif cut == 0:
            value = value - 1
        else:
            value = value - cut
        count += 1
        if value == saved:
            raise ValueError(
                f"R's AC never returns for these arguments: {name} cycles through {value!r}"
                f" without reaching {target!r}"
            )
        steps += 1
        if steps == horizon:
            saved, horizon, steps = value, 2 * horizon, 0
    return value, count


def _r_log10(v: float) -> float:
    """R's log10: -Inf at 0, and NaN with R's warning below 0."""
    if v < 0:
        warnings.warn("NaNs produced", DataNuggetWarning, stacklevel=3)
        return math.nan
    if v == 0:
        return -math.inf
    return math.log10(v)  # NaN and Inf pass through, as in R
