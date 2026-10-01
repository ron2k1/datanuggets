"""wpca's row-weight total against R's long double sum(wrow), on weights where the rounding decides
whether Wpca runs.

The cases are recorded by r/fixtures_wpca.R on R 4.6.1 (x86_64-w64-mingw32, 64-bit long double)
with WCluster 1.3.0. Every case has wcol = c(1, 1) and corr = FALSE. In R:

    x <- cbind(c(1, 4, 2), c(3, 1, 5))
    WCluster::Wpca(x, c(1, 2^-53, 2^-64), c(1, 1), FALSE)        # stops
    WCluster::Wpca(x, c(1, 2^-53, 2^-62), c(1, 1), FALSE)        # runs
    WCluster::Wpca(x, c(1 + 2^-51, 2^-53, 2^-64), c(1, 1), FALSE) # runs
    WCluster::Wpca(x, c(1, 2^-53, 2^-53), c(1, 1), FALSE)        # runs
    WCluster::Wpca(rbind(x, c(6, 2)), c(1, 2^-53, 2^-64, 2^-64), c(1, 1), FALSE) # stops
"""

import math

import numpy as np
import pytest

from datanuggets._compat import ldouble as ld
from datanuggets._compat.rstats import r_sum
from datanuggets.wcluster.wpca import _wpca_parts, wpca
from rfixtures import assert_same_values, hex_bits, load_group, same_bits

CASES = {c.id: c for c in load_group("wpca") if c.id.startswith("sum_ld_")}
X3 = [[1.0, 3.0], [4.0, 1.0], [2.0, 5.0]]
X4 = [*X3, [6.0, 2.0]]
# Each case's x and wrow, and the other sums that get its total wrong. sum_ld_near is sum_ld_stops
# with 2^-62 for 2^-64. sum_ld_steps_stops adds two 2^-64s: R rounds each one away, and a sum that
# adds them together first keeps them.
INPUTS = {
    "sum_ld_stops": (X3, [1.0, 2.0**-53, 2.0**-64], {"fsum"}),
    "sum_ld_near": (X3, [1.0, 2.0**-53, 2.0**-62], {"double"}),
    "sum_ld_ties_down": (X3, [1.0 + 2.0**-51, 2.0**-53, 2.0**-64], {"fsum"}),
    "sum_ld_double_loses": (X3, [1.0, 2.0**-53, 2.0**-53], {"double"}),
    "sum_ld_steps_stops": (X4, [1.0, 2.0**-53, 2.0**-64, 2.0**-64], {"fsum", "once", "reversed"}),
}
STOPS = ["sum_ld_stops", "sum_ld_steps_stops"]
RUNS = [cid for cid in INPUTS if cid not in STOPS]


def double_sum(values):
    total = 0.0
    for v in values:
        total += v
    return total


def once_sum(values):
    """The exact sum rounded once to a 64-bit long double, then to double."""
    parts = [ld.from_float(float(v)) for v in values]
    e = min(pe for _, pe in parts)
    return ld.to_float(ld._rounded(sum(m << (pe - e) for m, pe in parts), e))


# Sums that differ from R's: math.fsum rounds the exact sum once to double, a double sum rounds
# after every addition, and the last two keep 64 bits but round once or add in the other order
OTHER_SUMS = {
    "fsum": math.fsum,
    "double": double_sum,
    "once": once_sum,
    "reversed": lambda values: r_sum(values[::-1]),
}


def test_every_case_is_recorded():
    assert set(CASES) == set(INPUTS)


@pytest.mark.parametrize("cid", list(INPUTS))
def test_cases_are_pinned(cid):
    case = CASES[cid]
    versions = case.manifest["versions"]
    assert versions["R"].startswith("R version 4.6.1 ")
    assert versions["platform"] == "x86_64-w64-mingw32"
    assert versions["longdouble_digits"] == 64
    assert versions["WCluster"] == "1.3.0"
    assert case.args == {"corr": False}
    x, wrow, wrong = INPUTS[cid]
    assert case.array("x").tolist() == x
    assert case.array("wcol_in").tolist() == [1.0, 1.0]
    assert case.array("wrow").tolist() == wrow
    # the case separates R from the sums it names, and from no other
    r_total = float(case.array("sum_wrow")[0])
    for name, total in OTHER_SUMS.items():
        assert (total(wrow) != r_total) == (name in wrong), name


@pytest.mark.parametrize("cid", STOPS)
def test_stops_where_r_stops(cid):
    case = CASES[cid]
    x, wrow, wcol = case.array("x"), case.array("wrow"), case.array("wcol_in")
    assert case.manifest["error"] == "infinite or missing values in 'x'"
    # R's sum is exactly 1, so the covariance divides by 0. The exact sum is above 1.
    assert hex_bits(case.array("sum_wrow")) == ["3ff0000000000000"]
    assert same_bits([r_sum(wrow)], case.array("sum_wrow"))
    assert math.fsum(wrow) == 1.0 + 2.0**-52
    with pytest.raises(ValueError, match="sum to exactly 1"):
        wpca(x, wrow, wcol)


@pytest.mark.parametrize("cid", RUNS)
def test_runs_where_r_runs(cid):
    case = CASES[cid]
    x, wrow, wcol = case.array("x"), case.array("wrow"), case.array("wcol_in")
    assert case.manifest["warnings"] == []
    assert same_bits([r_sum(wrow)], case.array("sum_wrow"))
    got = wpca(x, wrow, wcol)
    assert_same_values(got.center, case.array("center"))
    assert_same_values(_wpca_parts(x, wrow, wcol, False)[4], case.array("cap_s"))
    assert np.isfinite(got.sdev).all()
