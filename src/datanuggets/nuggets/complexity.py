# Adapted from datanugget 1.5.0's AC (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

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

    Only x's shape is read, and there are no defaults, as in R. AC counts
    G = ceiling(n / floor(r)) chunks thinned from r rows to dn_num1 / G centers, then dn_num1
    centers thinned to dn_num2. A and B, the counters of R's two while loops, start at 1 and
    add one per thinning pass. Each chunk adds A times the pairs among dn_num1 / G rows, and
    the second stage adds B times the pairs among dn_num2 rows. Then (n + 1) * dn_num2 is
    added, and the total is multiplied by the column count. The loops run here in R's double
    arithmetic, so A and B match R's counts exactly. Some arguments make those loops cycle and
    R never returns. The first loop cycles when floor(dn_num1) / G is negative but not
    exactly -1. The second cycles when floor(dn_num2) is below -1 and below floor(dn_num1).
    A loop also cycles when a pass rounds back to the value it started from, as with
    dn_num1=1e17, dn_num2=1 and delete_percent=1e-18. This raises ValueError for them.

    Parameters
    ----------
    x : array_like, shape (n, p)
        Any 2-D array or data frame with at least one row. Only its shape is read.
    r : float
        Chunk-size control within [100, 10000], floored before counting.
    delete_percent : float
        Thinning fraction, within (0, 1).
    dn_num1 : float
        First-stage center target, finite and floored before counting.
    dn_num2 : float
        Second-stage center target, finite and floored before counting. It must not exceed
        ``dn_num1``, compared before flooring.

    Returns
    -------
    float
        ``log10(p * (G * A * R1 * (R1 - 1) / 2 + B * D * (D - 1) / 2 + (n + 1) * D))``, where
        ``G = ceil(n / floor(r))``, R1 is where the first loop ends (``floor(dn_num1) / G``),
        ``D = floor(dn_num2)``, and A and B are R's loop counters. A count of 0 gives
        ``-inf`` and a negative count gives NaN, as R's log10 does.

    Raises
    ------
    TypeError
        If ``r``, ``delete_percent``, ``dn_num1`` or ``dn_num2`` is a bool or not a real
        number.
    ValueError
        If ``x`` is not 2-D or has no rows, ``r`` is outside [100, 10000],
        ``delete_percent`` is outside (0, 1), ``dn_num1`` or ``dn_num2`` is not finite, or
        ``dn_num2 > dn_num1``. Also for the arguments on which R's loops never return: a
        target below -1 that the loop starts above, or a loop that cycles.

    Warns
    -----
    DataNuggetWarning
        "NaNs produced", R's log10 warning, when the count is negative.

    See Also
    --------
    create_dn : The function whose work this counts.

    Notes
    -----
    R counterpart: ``datanugget::AC``, ported from datanugget 1.5.0. Differences from R:

    - R's AC accepts only a plain data frame. This reads the shape of any 2-D input.
    - Arguments on which R's loops never return raise ValueError.
    - ``dn_num2 > dn_num1`` raises "dn_num2 must not exceed dn_num1", where R says
      "DN.num2 must be less than DN.num1".
    - R returns a length-one numeric vector. This returns a Python float.

    References
    ----------
    .. [1] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.
    .. [2] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import arithmetic_complexity
    >>> x = np.zeros((300, 3))
    >>> v = arithmetic_complexity(x, r=100, delete_percent=0.1, dn_num1=60, dn_num2=20)
    >>> round(v, 4)
    4.7501
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
