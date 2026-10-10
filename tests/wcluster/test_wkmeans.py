"""wkmeans against R's Wkmeans: the draws, each start's labels and result, and the answer, or
the error R stops with, for the reference and the kernel, and the kernel against the reference
on tie-heavy data."""

import importlib
import re

import numpy as np
import pytest

from datanuggets._compat._compiled import frozen
from datanuggets._random.r_draws import RDraws
from datanuggets._random.replay import ReplayDraws
from datanuggets.wcluster.wkmeans import wkmeans
from rfixtures import assert_same_values, load_group

# The exported function wkmeans shadows its module as an attribute of datanuggets.wcluster
wk = importlib.import_module("datanuggets.wcluster.wkmeans")

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
    for i, (_, u, labels, _passes) in enumerate(trace):
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


@pytest.mark.parametrize("case", RETURNED, ids=_ids(RETURNED))
def test_wkmeans_matches_r_and_uses_r_draws(case):
    trace = []
    draws = ReplayDraws(case.draws())
    got = wk._wkmeans(
        case.array("x"),
        case.args["k"],
        w=case.array("w"),
        centers=case.array("centers_in") if case.has("centers_in") else None,
        num_init=case.args["num_init"],
        max_iter=case.args["max_iter"],
        seed=draws,
        n_jobs=1,
        trace=trace,
    )
    draws.assert_exhausted()
    check(case, got, trace)


@pytest.mark.parametrize("case", STOPPED, ids=_ids(STOPPED))
def test_wkmeans_stops_where_r_stops(case):
    draws = ReplayDraws(case.draws())
    with pytest.raises(ValueError, match=re.escape(case.manifest["error"].strip())):
        wkmeans(
            case.array("x"),
            case.args["k"],
            w=case.array("w"),
            centers=case.array("centers_in") if case.has("centers_in") else None,
            num_init=case.args["num_init"],
            max_iter=case.args["max_iter"],
            seed=draws,
        )
    draws.assert_exhausted()


def fuzz_inputs(seed, family):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 30))
    p = int(rng.integers(1, 4))
    k = int(rng.integers(1, max(2, n // 3)))
    x = rng.integers(0, 4, (n, p)).astype(float)
    if family == "huge":
        x = x * 1e153  # 1e154 overflows the distances, and init never reaches k clusters
    w = {
        "unit": np.ones(n),
        "int": rng.integers(1, 4, n),
        "zero": rng.integers(0, 3, n),
        "signed": rng.choice([-1, 1, 2], n),
        "huge": rng.integers(1, 3, n),
        # Mostly negative: a negative-weight row takes its farthest center, which empties middle
        # labels, and moving it out of its cluster can raise that cluster's WWCSS, so the
        # emptied label can stay empty. That's the only way R returns a gap.
        "negative": rng.choice([-2.41, -1.19, 0.73], n),
    }[family].astype(float)
    return x, k, w


def run_both(x, k, w, seed, counts):
    def go(fn, **kw):
        try:
            got = fn(x, k, w=w, centers=None, num_init=3, max_iter=4, seed=RDraws(seed), **kw)
            return got, None
        except ValueError as e:
            return None, str(e)

    ref_trace = []
    ref, ref_err = go(wk._wkmeans_py, counts=counts, trace=ref_trace)
    for n_jobs in (1, 3):
        trace = []
        got, got_err = go(wk._wkmeans, n_jobs=n_jobs, trace=trace)
        assert got_err == ref_err
        if ref is not None:
            assert got.labels.tolist() == ref.labels.tolist()
            assert_same_values(got.centers, ref.centers)
            assert_same_values(got.wwcss, ref.wwcss)
            assert_same_values(got.total_wwcss, ref.total_wwcss)
            # R messages "Iteration l" once per pass, so a start's pass count is R's output too
            assert [t[3] for t in trace] == [t[3] for t in ref_trace]


FAMILIES = ["unit", "int", "zero", "signed", "huge", "negative"]

# R's gap case's grid, with weights from R's gap search drawn by numpy. A returned gap needs a
# singleton emptied through R's stale table after every row of the top label, since that row
# would hit tb_na, with no refill behind it, in the last iteration: a sweep of 5,000 seeds per k
# found 5 at k = 3 and 2 at k = 4. These are the first, so the default run checks a gap without
# the thousands of seeds before them.
GRID = np.array([[a, b] for b in range(3) for a in range(4)], float)
GAP_SEEDS = [(1061, 3), (1943, 3), (2114, 4)]


def grid_inputs(seed, k):
    return GRID, k, np.random.default_rng(seed).choice([-0.73, 1.19, 2.41], GRID.shape[0])


@pytest.fixture
def few_starts(monkeypatch):
    # Zero weights and overflowing distances can fail init on every start, where R draws
    # forever. Both paths share the cap, so a lower one ends those runs in the same error sooner.
    monkeypatch.setattr(wk, "MAX_STARTS", 100)


@pytest.mark.usefixtures("few_starts")
@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_tie_heavy_data(family):
    for seed in range(25):
        run_both(*fuzz_inputs(seed, family), seed, wk._counts())


@pytest.mark.usefixtures("few_starts")
def test_fuzz_reaches_every_quirk():
    total = wk._counts()
    runs = [(fuzz_inputs(s, f), s) for f in ("negative", "signed") for s in range(10)]
    runs += [(grid_inputs(s, k), s) for s, k in GAP_SEEDS]
    for inputs, seed in runs:
        c = wk._counts()
        run_both(*inputs, seed, c)
        for q in total:
            total[q] += c[q]
    assert all(total[q] > 0 for q in wk.QUIRKS), total


@pytest.mark.slow
@pytest.mark.usefixtures("few_starts")
@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_matches_reference_on_a_large_fuzz(family):
    for seed in range(25, 625):
        run_both(*fuzz_inputs(seed, family), seed, wk._counts())


def test_int_seed_is_reproducible():
    x = CASES[0].array("x")
    assert wkmeans(x, 3, seed=5) == wkmeans(x, 3, seed=5)


def test_rdraws_seed_matches_r():
    case = next(c for c in RETURNED if c.id == "iris_k3_init5")
    got = wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102))
    assert got.labels.tolist() == (case.array("labels").astype(np.int64) - 1).tolist()


def test_n_jobs_does_not_change_the_result():
    case = next(c for c in RETURNED if c.id == "iris_k3_init5")
    one = wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102), n_jobs=1)
    assert wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102), n_jobs=4) == one
    assert wkmeans(case.array("x"), 3, num_init=5, seed=RDraws(291102), n_jobs=-1) == one


def test_wkmeans_takes_any_layout_and_dtype():
    case = next(c for c in RETURNED if c.id == "quakes500_k4_w")
    x, w = case.array("x"), case.array("w")
    want = wkmeans(x, 4, w=w, seed=RDraws(291102))
    for xv in (np.asfortranarray(x), np.repeat(x, 2, axis=1)[:, ::2], frozen(x.copy())):
        assert wkmeans(xv, 4, w=list(w), seed=RDraws(291102)) == want
    grid = next(c for c in RETURNED if c.id == "gap").array("x")
    assert wkmeans(grid.astype(np.int64), 3, seed=1) == wkmeans(grid, 3, seed=1)


def test_r_positional_order_fails():
    with pytest.raises(TypeError):
        wkmeans(np.zeros((4, 2)), 2, None, np.ones(4))
