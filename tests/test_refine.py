"""refine_dn against R's refine.DN: every kmeans call, message, warning and output."""

import logging
import warnings
from dataclasses import replace

import numpy as np
import pytest

import datanuggets.refine as refine_module
from datanuggets import DataNuggets, DataNuggetWarning, refine_dn
from datanuggets._draws import NumpyDraws, ReplayDraws
from datanuggets._hartigan_wong import KMeansError, KMeansResult
from datanuggets._validate import check_assignments
from datanuggets.refine import _Nuggets, _split_nugget
from rfixtures import ATOL, RTOL, load_group, refine_inputs

CASES = load_group("refine_dn")
# The rejection reasons refine_dn logs at DEBUG, keyed by the flag R's fixture records
REASONS = {
    "kmeans_error": "kmeans error",
    "child_too_small": "a child is smaller than min_nugget_size",
}
SHAPE_REJECTED = "mean EV1/EV2 of the children >= delta"


def _case(case_id):
    return next(c for c in CASES if c.id == case_id)


@pytest.fixture
def kmeans_calls(monkeypatch):
    """The data and 0-based clusters (None on an error) of every kmeans call refine_dn makes."""
    calls = []
    real = refine_module.kmeans

    def spy(x, k, draws, **kwargs):
        call = [np.array(x), None]
        calls.append(call)
        result = real(x, k, draws, **kwargs)
        call[1] = result.cluster
        return result

    monkeypatch.setattr(refine_module, "kmeans", spy)
    return calls


def _r_kmeans_calls(case):
    """R's kmeans calls from the fixture: (data, 1-based clusters or None on an error)."""
    shapes = zip(
        case.array("km_nrow").tolist(),
        case.array("km_ncol").tolist(),
        case.array("km_ok").tolist(),
        strict=True,
    )
    xs, clusters = case.array("km_x"), case.array("km_cluster")
    calls, i, j = [], 0, 0
    for rows, cols, ok in shapes:
        x = xs[i : i + rows * cols].reshape(rows, cols)
        i += rows * cols
        cluster = clusters[j : j + rows] if ok else None
        j += rows if ok else 0
        calls.append((x, cluster))
    assert (i, j) == (xs.size, clusters.size)
    return calls


def _run(case, caplog):
    x, dn, kwargs = refine_inputs(case)
    draws = ReplayDraws(case.draws())
    caplog.set_level(logging.DEBUG, logger="datanuggets")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = refine_dn(x, dn, seed=draws, **kwargs)
    draws.assert_exhausted()
    return got, caught


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_refine_matches_r(case, kmeans_calls, caplog):
    got, caught = _run(case, caplog)
    p = case.array("x").shape[1]
    want = _r_kmeans_calls(case)
    assert len(kmeans_calls) == len(want)
    for (gx, gc), (wx, wc) in zip(kmeans_calls, want, strict=True):
        if wx.shape[1] == p:
            np.testing.assert_array_equal(gx, wx)
        else:  # a shape split: the projection on an eigenvector whose sign LAPACK picks
            sign = 1.0 if gx[:, 0] @ wx[:, 0] >= 0 else -1.0
            np.testing.assert_allclose(gx * sign, wx, rtol=RTOL, atol=ATOL)
        if wc is None:
            assert gc is None
        else:
            np.testing.assert_array_equal(gc + 1, wc)
    np.testing.assert_array_equal(got.assignments + 1, case.array("assignments"))
    np.testing.assert_array_equal(got.weights, case.array("weights"))
    np.testing.assert_array_equal(got.centers, case.array("centers"))
    np.testing.assert_array_equal(got.scales, case.array("scales"))
    records = [r for r in caplog.records if r.name == "datanuggets"]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert infos == case.manifest["messages"]
    texts = [str(w.message) for w in caught if w.category is DataNuggetWarning]
    assert texts == case.manifest["warnings"]
    debug = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
    for flag, reason in REASONS.items():
        assert any(reason in m for m in debug) == case.manifest["paths"][flag], flag


def test_every_rejection_path_is_exercised(caplog):
    """R's calls show kmeans errors and too-small children (the parity test ties each case's
    Python reasons to them). A shape split rejected on its children's EV1/EV2 leaves no trace in
    R's calls, so find one on the Python side: exact output parity on that case means R rejected
    the same split."""
    for flag in REASONS:
        assert any(c.manifest["paths"][flag] for c in CASES), flag
    for case in CASES:
        if not case.args["shape.split"] or case.array("x").shape[1] < 2:
            continue
        caplog.clear()
        _run(case, caplog)
        if any(SHAPE_REJECTED in r.getMessage() for r in caplog.records):
            return
    pytest.fail("no fixture rejects a shape split on its children's EV1/EV2")


def _first_case_per_input():
    first = {}
    for c in CASES:
        first.setdefault((c.manifest["dataset"], c.manifest["input"]), c)
    return list(first.values())


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("case", _first_case_per_input(), ids=lambda c: c.id)
def test_initial_nugget_features_match_r(case, n_jobs):
    x, dn, _ = refine_inputs(case)
    state = _Nuggets(x, check_assignments(dn, *x.shape), dn.centers.shape[0], n_jobs)
    state.refresh(range(state.size), state.members())
    np.testing.assert_array_equal(state.centers, case.array("init_centers"))
    np.testing.assert_array_equal(state.scales, case.array("init_scale"))
    if x.shape[1] == 1:  # EV1 = EV2 = the variance, which is exact
        np.testing.assert_array_equal(state.ev1, case.array("init_ev1"))
        np.testing.assert_array_equal(state.ev2, case.array("init_ev2"))
    else:
        np.testing.assert_allclose(state.ev1, case.array("init_ev1"), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(state.ev2, case.array("init_ev2"), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "make_seed", [lambda: 11, lambda: np.random.default_rng(11)], ids=["int", "generator"]
)
def test_n_jobs_does_not_change_the_result(make_seed):
    x, dn, kwargs = refine_inputs(_case("faithful_ev0.9_ms5_mns2_shape_d2_ns25"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        one = refine_dn(x, dn, seed=make_seed(), n_jobs=1, **kwargs)
        two = refine_dn(x, dn, seed=make_seed(), n_jobs=2, **kwargs)
        four = refine_dn(x, dn, seed=make_seed(), n_jobs=4, **kwargs)
    assert one.n_nuggets > np.count_nonzero(dn.weights)  # something was split
    assert one == two == four


def test_split_warnings_reach_the_caller_in_candidate_order(monkeypatch, caplog):
    """No R fixture warns (refine.DN runs kmeans with iter.max = 1000), so a stub supplies the
    warnings: two from a split that goes ahead, one carried by a kmeans error. refine_dn re-emits
    each as a DataNuggetWarning, in candidate order, pointing at the caller's line."""
    real = refine_module.kmeans
    calls = []

    def stub(x, k, draws, **kwargs):
        calls.append(x.shape[0])
        result = real(x, k, draws, **kwargs)
        if len(calls) == 1:
            return replace(result, warnings=("w1a", "w1b"))
        if len(calls) == 2:
            raise KMeansError("boom", ("w2",))
        return result

    monkeypatch.setattr(refine_module, "kmeans", stub)
    x, dn, kwargs = refine_inputs(_case("faithful_ev0.5_ms1_mns2_noshape_d2_ns1"))
    caplog.set_level(logging.DEBUG, logger="datanuggets")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        refine_dn(x, dn, seed=7, n_jobs=1, **kwargs)
    ours = [w for w in caught if w.category is DataNuggetWarning]
    assert len(calls) > 2
    assert [str(w.message) for w in ours] == ["w1a", "w1b", "w2"]
    assert [w.filename for w in ours] == [__file__] * 3
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert sum("not split: kmeans error: boom" in m for m in debug) == 1


def _stub_kmeans(monkeypatch, cluster=None, warned=("w",)):
    """refine's kmeans returning cluster with warned, or raising KMeansError when cluster is None."""

    def stub(x, k, draws, **kwargs):
        if cluster is None:
            raise KMeansError("boom", warned)
        labels = np.array(cluster, dtype=np.intp)
        return KMeansResult(
            cluster=labels,
            centers=np.zeros((2, x.shape[1])),
            withinss=np.zeros(2),
            size=np.bincount(labels, minlength=2),
            tot_withinss=0.0,
            iter=1,
            ifault=0,
            warnings=warned,
        )

    monkeypatch.setattr(refine_module, "kmeans", stub)


# Two elongated children: EV1/EV2 is far above delta, so a shape split is rejected
_LINES = np.array([[t, 0.01 * (-1) ** t + 10 * g] for g in (0, 1) for t in range(6)], dtype=float)
SPLIT_PATHS = [
    ("kmeans error", None, False, "kmeans error: boom"),
    ("child too small", [0] * 11 + [1], False, "a child is smaller than min_nugget_size"),
    ("shape rejected", [0] * 6 + [1] * 6, True, SHAPE_REJECTED),
    ("split", [0] * 6 + [1] * 6, False, None),
]


@pytest.mark.parametrize(
    "cluster, project, reason", [p[1:] for p in SPLIT_PATHS], ids=[p[0] for p in SPLIT_PATHS]
)
def test_split_nugget_carries_the_kmeans_warnings(monkeypatch, cluster, project, reason):
    _stub_kmeans(monkeypatch, cluster, warned=("first", "second"))
    out = _split_nugget(_LINES, NumpyDraws(1), 2, 1, project, 2)
    assert out.reason == reason
    assert out.warned == ("first", "second")
    if reason is None:
        np.testing.assert_array_equal(out.labels, cluster)
    else:
        assert out.labels is None


def _tiny():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    dn = DataNuggets(
        centers=np.zeros((2, 2)),
        weights=np.array([3, 1]),
        scales=np.zeros(2),
        assignments=np.array([0, 0, 0, 1]),
    )
    return x, dn


BAD_ARGS = [
    ({"ev_tol": 0}, ValueError, r"ev_tol must be within \(0, 1\)"),
    ({"ev_tol": 1}, ValueError, r"ev_tol must be within \(0, 1\)"),
    ({"ev_tol": "0.5"}, TypeError, "ev_tol must be a number"),
    ({"max_splits": -1}, ValueError, "max_splits must be a finite number >= 0"),
    ({"max_splits": float("inf")}, ValueError, "max_splits must be a finite number >= 0"),
    ({"min_nugget_size": 1.5}, ValueError, "min_nugget_size must be at least 2"),
    ({"min_nugget_size": float("nan")}, ValueError, "min_nugget_size must be at least 2"),
    ({"shape_split": 1}, TypeError, "shape_split must be True or False"),
    ({"min_shape_size": 9}, ValueError, "min_shape_size must be at least 10"),
    ({"delta": 1}, ValueError, "delta must be greater than 1"),
    ({"nstart": float("nan")}, ValueError, "nstart must be finite"),
    ({"n_jobs": 0}, ValueError, "n_jobs must be a positive int or -1"),
    ({"seed": "abc"}, TypeError, "seed must be an int"),
]


@pytest.mark.parametrize("kwargs, exc, match", BAD_ARGS)
def test_bad_arguments_are_rejected(kwargs, exc, match):
    x, dn = _tiny()
    with pytest.raises(exc, match=match):
        refine_dn(x, dn, **kwargs)


def test_dn_must_be_datanuggets():
    x, dn = _tiny()
    with pytest.raises(TypeError, match="dn must be a DataNuggets"):
        refine_dn(x, {"assignments": dn.assignments})


def test_x_needs_a_row():
    _, dn = _tiny()
    with pytest.raises(ValueError, match="at least one observation"):
        refine_dn(np.zeros((0, 2)), replace(dn, assignments=np.array([], dtype=int)))


def test_assignments_must_cover_every_row():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="one entry per row of x"):
        refine_dn(x[:3], dn)


@pytest.mark.parametrize("bad", [[0, 0, 0, 2], [-1, 0, 0, 1]])
def test_assignments_must_be_nugget_ids(bad):
    x, dn = _tiny()
    with pytest.raises(ValueError, match=r"nugget ids in \[0, 2\)"):
        refine_dn(x, replace(dn, assignments=np.array(bad)))


def test_assignments_must_be_whole_numbers():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="whole numbers"):
        refine_dn(x, replace(dn, assignments=np.array([0, 0.5, 0, 1])))
    whole = refine_dn(x, replace(dn, assignments=np.array([0.0, 0.0, 0.0, 1.0])))
    assert whole == refine_dn(x, dn)


def test_dn_from_data_with_other_columns_is_rejected():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="x has 3 columns"):
        refine_dn(np.hstack([x, x[:, :1]]), dn)


def test_replayed_draws_need_n_jobs_1():
    x, dn = _tiny()
    with pytest.raises(ValueError, match="n_jobs=1"):
        refine_dn(x, dn, seed=ReplayDraws([]), n_jobs=2)
