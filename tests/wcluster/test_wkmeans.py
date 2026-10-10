"""wkmeans against R's Wkmeans: the draws, each start's labels and result, and the answer, or
the error R stops with."""

import re

import numpy as np
import pytest

from datanuggets._random.replay import ReplayDraws
from datanuggets.wcluster import wkmeans as wk
from rfixtures import assert_same_values, load_group

CASES = load_group("wkmeans")
RETURNED = [c for c in CASES if not c.manifest["error"]]
STOPPED = [c for c in CASES if c.manifest["error"]]
SMALL = [c for c in RETURNED if c.array("x").shape[0] <= 160]
LARGE = [c for c in RETURNED if c.array("x").shape[0] > 160]
_MISSING = re.escape("missing value where TRUE/FALSE needed")


def _ids(cases):
    return [c.id for c in cases]


def run_reference(case, draws, **extra):
    a = case.args
    centers = case.array("centers_in") if case.has("centers_in") else None
    return wk._wkmeans_py(
        case.array("x"),
        a["k"],
        w=case.array("w"),
        centers=centers,
        num_init=a["num_init"],
        max_iter=a["max_iter"],
        seed=draws,
        **extra,
    )


def check_counts(case, counts):
    for q in wk.QUIRKS:
        assert counts[q] == case.manifest["counts"][q], q


def check(case, got, trace):
    starts = case.array("init_labels").astype(np.int64)
    assert [t[0].tolist() for t in trace] == starts.tolist()
    singles = case.array("single_labels").astype(np.int64)
    assert len(trace) == singles.shape[0]
    for i, (_, u, labels) in enumerate(trace):
        assert labels.tolist() == singles[i].tolist()
        assert_same_values(np.float64(u), np.float64(case.array("single_wwcss")[i]))
    assert got.labels.tolist() == (case.array("labels").astype(np.int64) - 1).tolist()
    assert_same_values(got.centers, case.array("centers"))
    assert_same_values(got.wwcss, case.array("wwcss"))
    assert_same_values(got.total_wwcss, case.array("total").reshape(()))


def check_reference(case):
    trace, counts = [], wk._counts()
    draws = ReplayDraws(case.draws())
    got = run_reference(case, draws, trace=trace, counts=counts)
    draws.assert_exhausted()
    check(case, got, trace)
    check_counts(case, counts)


def test_fixtures_cover_every_wkmeans_quirk():
    for q in wk.QUIRKS:
        assert sum(c.manifest["counts"][q] for c in CASES), q
    assert set(_ids(STOPPED)) == {"ucenters_empty", "tb_na", "nan_stop"}
    assert SMALL and LARGE


@pytest.mark.parametrize("case", SMALL, ids=_ids(SMALL))
def test_reference_matches_r_and_uses_r_draws(case):
    check_reference(case)


@pytest.mark.slow
@pytest.mark.parametrize("case", LARGE, ids=_ids(LARGE))
def test_reference_matches_r_on_the_large_fixtures(case):
    check_reference(case)


@pytest.mark.parametrize("case", STOPPED, ids=_ids(STOPPED))
def test_reference_stops_where_r_stops(case):
    counts = wk._counts()
    draws = ReplayDraws(case.draws())
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        run_reference(case, draws, counts=counts)
    draws.assert_exhausted()
    check_counts(case, counts)


@pytest.mark.parametrize(
    "bad, exc, msg",
    [
        ({"k": "3"}, TypeError, 'k must be of class "numeric"'),
        ({"k": True}, TypeError, 'k must be of class "numeric"'),
        ({"k": 0.5}, ValueError, "k must be"),
        ({"num_init": 0.9}, ValueError, "num_init"),
        ({"max_iter": 0}, ValueError, "max_iter"),
        ({"centers": np.zeros((2, 4))}, ValueError, "centers must be"),
        ({"w": np.ones(3)}, ValueError, "w"),
    ],
)
def test_arguments_are_checked_as_r_checks_them(bad, exc, msg):
    args = {"k": 3, "w": None, "centers": None, "num_init": 1, "max_iter": 10} | bad
    x = np.arange(40.0).reshape(10, 4)
    with pytest.raises(exc, match=re.escape(msg)):
        wk._prepare(x, args["k"], args["w"], args["centers"], args["num_init"], args["max_iter"])


def test_hang_guards_raise_at_their_boundary():
    x = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]])
    wk._wkmeans_py(x, 3, seed=1)  # exactly k distinct pairs and n == 2k: runs
    with pytest.raises(ValueError, match="2k"):
        wk._wkmeans_py(x[:5], 3, seed=1)
    two_values = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    with pytest.raises(ValueError, match="distinct"):
        wk._wkmeans_py(two_values, 3, seed=1)
    # A zero weight makes a third (row, weight) pair. Its distances are all 0, so which.min
    # gives it label 1, and a start whose label 1 is neither row's nearest reaches k = 3. Init
    # ends there, with the zero-weight row alone in label 1, whose WWCSS is then NaN: R stops.
    with pytest.raises(ValueError, match=_MISSING):
        wk._wkmeans_py(two_values, 3, w=np.array([1.0, 1, 0, 1, 1, 1]), seed=1)


def test_init_gives_up_where_r_draws_forever(monkeypatch):
    monkeypatch.setattr(wk, "MAX_STARTS", 50)  # zero weights: every center is NaN, every start
    with pytest.raises(ValueError, match="keep drawing"):
        wk._wkmeans_py(np.arange(12.0).reshape(6, 2), 2, w=np.zeros(6), seed=1)


def test_one_row_follows_rs_diag():
    x = np.array([[1.0, 2.0]])
    c = np.zeros((1, 2))
    assert wk._wkmeans_py(x, 1, w=np.array([1.5]), centers=c).labels.tolist() == [0]
    with pytest.raises(ValueError, match="non-conformable arguments"):
        wk._wkmeans_py(x, 1, w=np.array([2.0]), centers=c)
    with pytest.raises(ValueError, match=re.escape("invalid 'nrow' value (< 0)")):
        wk._wkmeans_py(x, 1, w=np.array([-1.0]), centers=c)
