"""wpca's row-weight total against R's long double sum(wrow), on weights where the rounding decides
whether Wpca runs.

The cases are recorded by r/fixtures_wpca.R on R 4.6.1 (x86_64-w64-mingw32, 64-bit long double)
with WCluster 1.3.0. Every case has x = cbind(c(1, 4, 2), c(3, 1, 5)), wcol = c(1, 1) and
corr = FALSE, and differs only in wrow. In R:

    x <- cbind(c(1, 4, 2), c(3, 1, 5))
    WCluster::Wpca(x, c(1, 2^-53, 2^-64), c(1, 1), FALSE)        # stops
    WCluster::Wpca(x, c(1, 2^-53, 2^-62), c(1, 1), FALSE)        # runs
    WCluster::Wpca(x, c(1 + 2^-51, 2^-53, 2^-64), c(1, 1), FALSE) # runs
    WCluster::Wpca(x, c(1, 2^-53, 2^-53), c(1, 1), FALSE)        # runs
"""

import math

import numpy as np
import pytest

from datanuggets._rstats import r_sum
from datanuggets.wcluster.wpca import _wpca_parts, wpca
from rfixtures import assert_same_values, hex_bits, load_group, same_bits

CASES = {c.id: c for c in load_group("wpca") if c.id.startswith("sum_ld_")}
X = [[1.0, 3.0], [4.0, 1.0], [2.0, 5.0]]
# Each case's wrow, and the other sums that get its total wrong: math.fsum rounds the exact sum
# once, a double sum rounds after every addition. sum_ld_near is sum_ld_stops with 2^-62 for 2^-64.
WROW = {
    "sum_ld_stops": ([1.0, 2.0**-53, 2.0**-64], {"fsum"}),
    "sum_ld_near": ([1.0, 2.0**-53, 2.0**-62], {"double"}),
    "sum_ld_ties_down": ([1.0 + 2.0**-51, 2.0**-53, 2.0**-64], {"fsum"}),
    "sum_ld_double_loses": ([1.0, 2.0**-53, 2.0**-53], {"double"}),
}
RUNS = [cid for cid in WROW if cid != "sum_ld_stops"]


def double_sum(values):
    total = 0.0
    for v in values:
        total += v
    return total


def test_every_case_is_recorded():
    assert set(CASES) == set(WROW)


@pytest.mark.parametrize("cid", list(WROW))
def test_cases_are_pinned(cid):
    case = CASES[cid]
    versions = case.manifest["versions"]
    assert versions["R"].startswith("R version 4.6.1 ")
    assert versions["platform"] == "x86_64-w64-mingw32"
    assert versions["longdouble_digits"] == 64
    assert versions["WCluster"] == "1.3.0"
    assert case.args == {"corr": False}
    assert case.array("x").tolist() == X
    assert case.array("wcol_in").tolist() == [1.0, 1.0]
    wrow, wrong = WROW[cid]
    assert case.array("wrow").tolist() == wrow
    # the case separates R from the sums it names, and from no other
    r_total = float(case.array("sum_wrow")[0])
    assert (math.fsum(wrow) != r_total) == ("fsum" in wrong)
    assert (double_sum(wrow) != r_total) == ("double" in wrong)


def test_stops_where_r_stops():
    case = CASES["sum_ld_stops"]
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
