"""whclust against R's Whclust: the reference and the kernel on every fixture, and the kernel
against the reference on tie-heavy data."""

import importlib
import re

import numpy as np
import pytest
import scipy.cluster.hierarchy

from datanuggets._compat._compiled import frozen
from datanuggets.wcluster.whclust import whclust
from rfixtures import assert_same_values, load_group

# The exported function whclust shadows its module as an attribute of datanuggets.wcluster
wh = importlib.import_module("datanuggets.wcluster.whclust")

CASES = load_group("whclust")
RETURNED = [c for c in CASES if not c.manifest["error"]]
STOPPED = [c for c in CASES if c.manifest["error"]]
SMALL = [c for c in RETURNED if c.array("x").shape[0] <= 40]
LARGE = [c for c in RETURNED if c.array("x").shape[0] > 40]


def _ids(cases):
    return [c.id for c in cases]


def expected(case):
    n = case.array("x").shape[0]
    merge = case.array("merge").astype(np.int64)
    merge = np.where(merge < 0, -merge - 1, n + merge - 1)
    return merge, case.array("height"), case.array("order").astype(np.int64) - 1


def check_reference(case):
    counts = {}
    n = case.array("x").shape[0]
    got = wh._to_result(n, *wh._whclust_py(case.array("x"), case.array("w"), counts=counts))
    merge, height, order = expected(case)
    assert got.merge.tolist() == merge.tolist()
    assert_same_values(got.height, height)
    assert got.order.tolist() == order.tolist()
    for q, v in counts.items():
        assert v == case.manifest["counts"][q], q


def test_fixtures_cover_every_whclust_quirk():
    counts = {q: sum(c.manifest["counts"][q] for c in CASES) for q in ("inf_pick", "self_merge")}
    assert all(counts.values()), counts
    assert {c.id for c in STOPPED} == {"n1", "n2", "no_candidate"}
    assert SMALL and LARGE


@pytest.mark.parametrize("case", SMALL, ids=_ids(SMALL))
def test_reference_matches_r(case):
    check_reference(case)


@pytest.mark.slow
@pytest.mark.parametrize("case", LARGE, ids=_ids(LARGE))
def test_reference_matches_r_on_the_large_fixtures(case):
    check_reference(case)


@pytest.mark.parametrize("case", STOPPED, ids=_ids(STOPPED))
def test_reference_stops_where_r_stops(case):
    counts = {}
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        wh._whclust_py(case.array("x"), case.array("w"), counts=counts)
    for q, v in counts.items():
        assert v == case.manifest["counts"][q], q


@pytest.mark.parametrize("case", RETURNED, ids=_ids(RETURNED))
def test_whclust_matches_r(case):
    got = whclust(case.array("x"), case.array("w"))
    merge, height, order = expected(case)
    assert got.merge.tolist() == merge.tolist()
    assert_same_values(got.height, height)
    assert got.order.tolist() == order.tolist()


@pytest.mark.parametrize("case", STOPPED, ids=_ids(STOPPED))
def test_whclust_stops_where_r_stops(case):
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        whclust(case.array("x"), case.array("w"))


FAMILIES = ["unit", "int", "zero", "signed", "dup", "huge"]


def fuzz_inputs(seed, family):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 14))
    p = int(rng.integers(1, 4))
    x = rng.integers(0, 3, (n, p)).astype(float)
    if family == "dup":
        x[rng.integers(0, n, n // 2)] = x[0]
    if family == "huge":
        x = x * 1e154 + rng.normal(size=(n, p)) * 1e153
    w = {
        "unit": np.ones(n),
        "int": rng.integers(1, 4, n),
        "zero": rng.integers(0, 3, n),
        "signed": rng.choice([-1, 1], n),
        "dup": rng.integers(1, 3, n),
        "huge": rng.integers(1, 3, n),
    }[family].astype(float)
    return x, w


def run_both(x, w, counts):
    try:
        ref = wh._to_result(x.shape[0], *wh._whclust_py(x, w, counts=counts))
    except ValueError as e:
        with pytest.raises(ValueError, match=re.escape(str(e))):
            whclust(x, w)
        return
    got = whclust(x, w)
    assert got.merge.tolist() == ref.merge.tolist()
    assert_same_values(got.height, ref.height)
    assert got.order.tolist() == ref.order.tolist()


@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_tie_heavy_data(family):
    for seed in range(40):
        run_both(*fuzz_inputs(seed, family), {})


def test_fuzz_reaches_the_inf_pick_and_the_self_merge():
    total = {"inf_pick": 0, "self_merge": 0}
    for family in ("zero", "signed", "huge"):
        for seed in range(40):
            c = {}
            run_both(*fuzz_inputs(seed, family), c)
            for q in total:
                total[q] += c.get(q, 0)
    assert all(v > 0 for v in total.values()), total


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_a_large_fuzz(family):
    for seed in range(40, 1040):
        run_both(*fuzz_inputs(seed, family), {})


def test_whclust_takes_any_layout_and_dtype():
    case = next(c for c in RETURNED if c.id == "zero_w1")
    x, w = case.array("x"), case.array("w")
    want = whclust(x, w)
    for xv in (np.asfortranarray(x), np.repeat(x, 2, axis=1)[:, ::2], frozen(x.copy())):
        assert whclust(xv, list(w)) == want
    assert whclust(x, np.repeat(w, 2)[::2]) == want
    grid = next(c for c in RETURNED if c.id == "grid_dup").array("x")
    assert whclust(grid.astype(np.int64)) == whclust(grid)


def test_whclust_needs_three_rows():
    with pytest.raises(ValueError, match="subscript out of bounds"):
        whclust(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="missing value where TRUE/FALSE needed"):
        whclust(np.zeros((1, 2)))


def test_to_linkage_drives_scipy_dendrogram_in_r_order():
    case = next(c for c in RETURNED if c.id == "iris_w1372")
    got = whclust(case.array("x"), case.array("w"))
    leaves = scipy.cluster.hierarchy.dendrogram(got.to_linkage(), no_plot=True)["leaves"]
    assert leaves == got.order.tolist()
