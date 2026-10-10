"""The whclust reference against R's Whclust on every fixture."""

import re

import numpy as np
import pytest

from datanuggets.wcluster import whclust as wh
from rfixtures import assert_same_values, load_group

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
