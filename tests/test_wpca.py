"""wpca against WCluster's Wpca: every step up to eigen() bit for bit, and eigen's output by the
residual and subspace rule in tests/rfixtures.py."""

import importlib
import warnings

import numpy as np
import pytest

from datanuggets._rstats import r_eigen_sym, r_matmul, r_sum
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.wcluster.wpca import _wpca_parts, wpca
from rfixtures import (
    EIG_C,
    EIG_FLOOR,
    EIG_K,
    EIG_TIE,
    assert_same_values,
    eigen_mismatches,
    load_group,
    same_bits,
)

EPS = float(np.finfo(float).eps)
# The plan's rule as literals, so a looser rule in rfixtures can't loosen the near misses too
PLAN_FLOOR, PLAN_C, PLAN_K, PLAN_TIE = 1e-10, 256.0, 1024.0, 1e-12
GROUP = load_group("wpca")
(STOP,) = [c for c in GROUP if c.id == "one_iris_first5_petalwidth"]
CASES = [c for c in GROUP if c is not STOP]
BY_ID = {c.id: c for c in CASES}
QUIRK = "petalwidth_1d_corr"
MATCHED = [c for c in CASES if c.id != QUIRK]
CORR = [c for c in CASES if c.args["corr"]]
# A 1 x 1 matrix is always symmetric, so only the corr = TRUE cases with p > 1 can pin the triangle
CORR_WIDE = [c for c in CORR if c.array("x").shape[1] > 1]


def band(values):
    return EIG_K * EPS * max(float(np.max(np.abs(values))), 1e-300)


BELOW = [c for c in CASES if c.array("values").min() < -band(c.array("values"))]
CLEAN = [c for c in CASES if (c.array("values") > band(c.array("values"))).all()]
DEN_IRIS = {f"den_iris_first{k}_petalwidth" for k in (2, 3, 4)}

SWAPPED = ["iris", "quakes_stations", "volcano_scaled"]
WEIGHTED = ["quakes_stations", "mtcars_carb_invsd", "w_iris_petalwidth"]
OTHER_SDEV = {
    "iris": "iris_corr",
    "quakes_stations": "quakes_stations_corr",
    "volcano_scaled": "nug_volcano_scaled",
}
# near misses: just past the plan's R3 band, and a turn that R2 lets through but R4 must not
NEAR_BAND = ["iris", "quakes_stations", "volcano_scaled"]
NEAR_TURN = ["volcano_scaled", "w_mtcars_wt", "wcol_volcano49_default", "mtcars_carb_invsd"]
MUTANTS = (
    [("swap", i) for i in SWAPPED]
    + [("flip", i) for i in SWAPPED]
    + [("weighted", i) for i in WEIGHTED]
    + [("sdev", i) for i in OTHER_SDEV]
    + [("band", i) for i in NEAR_BAND]
    + [("turn", i) for i in NEAR_TURN]
)


def ids(cases):
    return [c.id for c in cases]


def inputs(case):
    return case.array("x"), case.array("wrow"), case.array("wcol_in"), bool(case.args["corr"])


def run(*args, **kwargs):
    """wpca and every warning it raised, of any category."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = wpca(*args, **kwargs)
    return got, rec


def r_scale(case):
    """R's scale as the port returns it: dd with corr, and ones in place of R's single 1."""
    scale = case.array("scale")
    if case.args["corr"]:
        return scale
    assert scale.tolist() == [1.0]
    return np.ones(case.array("x").shape[1])


def is_tied(values):
    norm = float(np.max(np.abs(values)))
    return np.abs(np.diff(values)) <= EIG_TIE * norm


def test_fixtures_cover_the_edge_cases():
    for cases in (CASES, MATCHED, CORR, CORR_WIDE, BELOW, CLEAN, MUTANTS):
        assert cases
    assert all(i in BY_ID for _, i in MUTANTS)
    assert all(i in BY_ID for i in OTHER_SDEV.values())
    assert any(c.array("x").shape[1] == 1 for c in CASES)
    assert any((c.array("wrow") == 0).any() for c in CASES)
    # below -band with R's one warning: the three den_iris cases and nothing else
    assert set(ids(BELOW)) == DEN_IRIS
    assert all(c.manifest["warnings"] == ["NaNs produced"] for c in BELOW)
    assert all(c.manifest["warnings"] == [] for c in CLEAN)
    assert any(is_tied(c.array("values")).any() for c in CLEAN)
    assert is_tied(BY_ID["tie_npk"].array("values")).any()
    assert any(not np.array_equal(c.array("cap_s"), c.array("cap_s").T) for c in CORR)
    assert QUIRK in BY_ID and BY_ID[QUIRK].args["corr"]
    assert STOP.manifest["error"] == "infinite or missing values in 'x'"
    assert set(STOP.manifest["arrays"]) == {"x", "wrow", "wcol_in"}


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("case", MATCHED, ids=ids(MATCHED))
def test_parts_match_r(case, order):
    x, wrow, wcol, corr = inputs(case)
    center, wcol_out, x0, scale, s = _wpca_parts(np.asarray(x, order=order), wrow, wcol, corr)
    assert_same_values(center, case.array("center"))
    assert_same_values(wcol_out, case.array("wcol"))
    assert_same_values(x0, case.array("cap_x_0"))
    assert_same_values(s, case.array("cap_s"))
    assert_same_values(scale, r_scale(case))
    if corr:
        cov = _wpca_parts(np.asarray(x, order=order), wrow, wcol, False)[4]
        assert_same_values(cov, case.array("cap_cov"))


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_scores_product_matches_r(case):
    got = r_matmul(case.array("cap_x_0"), case.array("rotation"))
    assert_same_values(got, case.array("scores"))


@pytest.mark.parametrize("case", MATCHED, ids=ids(MATCHED))
def test_wpca_matches_r(case):
    got, _ = run(*inputs(case))
    assert_same_values(got.center, case.array("center"))
    assert_same_values(got.scale, r_scale(case))
    assert_same_values(got.wcol, case.array("wcol"))
    assert_same_values(got.wrow, case.array("wrow"))
    # the scores are R's product of R's centered data with whatever rotation LAPACK gave
    assert_same_values(got.x, r_matmul(case.array("cap_x_0"), got.rotation))
    wrong = eigen_mismatches(
        case.array("cap_s"),
        case.array("values"),
        case.array("rotation"),
        case.array("scores"),
        case.array("cap_x_0"),
        got.sdev,
        got.rotation,
        got.x,
    )
    assert wrong == []


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_warnings_match_r(case):
    got, rec = run(*inputs(case))
    assert all(issubclass(w.category, DataNuggetWarning) for w in rec)
    msgs = [str(w.message) for w in rec]
    values = case.array("values")
    if values.min() < -band(values):
        assert msgs == ["NaNs produced"] == case.manifest["warnings"]
    if (values > band(values)).all():
        assert msgs == [] == case.manifest["warnings"]
    # Inside the band an eigenvalue may take either sign on another LAPACK, so only this holds
    assert msgs == (["NaNs produced"] if np.isnan(got.sdev).any() else [])


def r_outputs(case):
    return case.array("sdev"), case.array("rotation"), case.array("scores")


def mismatches(case, sdev, rotation, scores):
    return eigen_mismatches(
        case.array("cap_s"),
        case.array("values"),
        case.array("rotation"),
        case.array("scores"),
        case.array("cap_x_0"),
        sdev,
        rotation,
        scores,
    )


@pytest.mark.parametrize(("kind", "cid"), MUTANTS, ids=[f"{k}-{i}" for k, i in MUTANTS])
def test_eigen_rule_rejects_wrong_outputs(kind, cid):
    case = BY_ID[cid]
    sdev, rotation, scores = r_outputs(case)
    assert mismatches(case, sdev, rotation, scores) == []
    if kind == "swap":
        perm = [1, 0, *range(2, rotation.shape[1])]
        rotation, scores = rotation[:, perm], scores[:, perm]
        assert not np.array_equal(rotation, case.array("rotation"))
        assert not np.array_equal(scores, case.array("scores"))
    elif kind == "flip":
        rotation = rotation.copy()
        rotation[:, 0] = -rotation[:, 0]
        assert not np.array_equal(rotation, case.array("rotation"))
    elif kind == "weighted":
        x0 = case.array("cap_x_0")
        xw = (x0 * np.sqrt(case.array("wrow"))[:, None]) * np.sqrt(case.array("wcol"))[None, :]
        assert not np.array_equal(xw, x0)
        scores = r_matmul(xw, rotation)
        assert not np.array_equal(scores, case.array("scores"))
    elif kind == "sdev":
        sdev = BY_ID[OTHER_SDEV[cid]].array("sdev")
        assert sdev.shape == case.array("sdev").shape
        assert not np.array_equal(sdev, case.array("sdev"), equal_nan=True)
    elif kind == "band":
        values = case.array("values")
        plan_band = PLAN_K * EPS * float(np.max(np.abs(values)))
        sdev = sdev.copy()
        sdev[0] = np.sqrt(values[0] + 2 * plan_band)
        miss = abs(sdev[0] * sdev[0] - values[0])
        assert plan_band < miss < 3 * plan_band
    else:
        values = case.array("values")
        gaps = np.abs(np.diff(values)) / float(np.max(np.abs(values)))
        apart = np.flatnonzero(gaps > PLAN_TIE)
        i = int(apart[np.argmin(gaps[apart])])
        # R4 allows PLAN_C eps / gap for this pair and R2 about PLAN_K eps / gap, so turn between
        theta = np.sqrt(PLAN_C * PLAN_K) * EPS / gaps[i]
        assert max(PLAN_FLOOR, PLAN_C * EPS / gaps[i]) < theta < PLAN_K * EPS / gaps[i]
        c, s = np.cos(theta), np.sin(theta)
        turn = np.array([[c, -s], [s, c]])
        rotation, scores = rotation.copy(), scores.copy()
        rotation[:, [i, i + 1]] = rotation[:, [i, i + 1]] @ turn
        scores[:, [i, i + 1]] = scores[:, [i, i + 1]] @ turn
        assert not np.array_equal(rotation, case.array("rotation"))
    wrong = mismatches(case, sdev, rotation, scores)
    assert wrong != []
    if kind == "band":
        assert all(m.startswith("R3") for m in wrong)
    if kind == "turn":
        assert any(m.startswith("R4") for m in wrong)
        assert not any(m.startswith("R2") for m in wrong)


def test_eigen_rule_constants_are_the_plans():
    assert (EIG_FLOOR, EIG_C, EIG_K, EIG_TIE) == (PLAN_FLOOR, PLAN_C, PLAN_K, PLAN_TIE)


def test_eigen_rule_accepts_a_rotated_tie():
    case = BY_ID["tie_npk"]
    tied = np.flatnonzero(is_tied(case.array("values")))
    assert tied.size
    i, j = int(tied[0]), int(tied[0]) + 1
    sdev, rotation, scores = r_outputs(case)
    c, s = np.cos(0.6), np.sin(0.6)
    turn = np.array([[c, -s], [s, c]])
    rotation, scores = rotation.copy(), scores.copy()
    rotation[:, [i, j]] = rotation[:, [i, j]] @ turn
    scores[:, [i, j]] = scores[:, [i, j]] @ turn
    assert not np.array_equal(rotation, case.array("rotation"))
    assert not np.array_equal(scores, case.array("scores"))
    assert mismatches(case, sdev, rotation, scores) == []


def test_sdev_comes_from_the_vectors_call(monkeypatch):
    mod = importlib.import_module("datanuggets.wcluster.wpca")
    real = mod.r_eigen_sym
    calls = []

    def spy(s, vectors=False):
        calls.append(vectors)
        return real(s, vectors=vectors)

    monkeypatch.setattr(mod, "r_eigen_sym", spy)
    case = BY_ID["iris"]
    got = wpca(*inputs(case))
    assert calls == [True]
    values, _ = real(case.array("cap_s"), vectors=True)
    assert same_bits(got.sdev, np.sqrt(values))


@pytest.mark.parametrize("case", CORR_WIDE, ids=ids(CORR_WIDE))
def test_eigen_reads_the_lower_triangle(case):
    s = case.array("cap_s")
    assert not np.array_equal(s, s.T)
    lower = np.tril(s) + np.tril(s, -1).T
    values, vectors = r_eigen_sym(s, vectors=True)
    values_l, vectors_l = r_eigen_sym(lower, vectors=True)
    assert same_bits(values, values_l)
    assert same_bits(vectors, vectors_l)


def test_one_column_corr_departs_from_r():
    case = BY_ID[QUIRK]
    # R's diag() of one number builds an identity, so R's sdev is the unscaled one, dd itself
    assert_same_values(case.array("sdev"), case.array("scale"))
    assert case.array("sdev").tolist() != [1.0]
    got, rec = run(*inputs(case))
    assert rec == []
    assert_same_values(got.scale, case.array("scale"))
    assert_same_values(got.center, case.array("center"))
    assert got.sdev.shape == (1,)
    assert abs(got.sdev[0] - 1.0) <= 4 * EPS


def test_weights_summing_to_one_raise():
    # R divides by sum(wrow) - 1 = 0 and eigen() stops on the infinite covariance
    assert STOP.manifest["error"] == "infinite or missing values in 'x'"
    x, wrow, wcol = STOP.array("x"), STOP.array("wrow"), STOP.array("wcol_in")
    assert r_sum(wrow) - 1.0 == 0
    with pytest.raises(ValueError, match="sum to exactly 1"):
        wpca(x, wrow, wcol)


@pytest.mark.parametrize("case", BELOW, ids=ids(BELOW))
def test_weights_summing_below_one_warn_once(case):
    assert r_sum(case.array("wrow")) < 1
    assert case.manifest["warnings"] == ["NaNs produced"]
    got, rec = run(*inputs(case))
    assert [(w.category, str(w.message)) for w in rec] == [(DataNuggetWarning, "NaNs produced")]
    assert np.isnan(got.sdev).any()


@pytest.mark.parametrize("cid", sorted(DEN_IRIS))
def test_corr_with_weights_summing_below_one_raise(cid):
    # R 4.6.1 warns "NaNs produced" from sqrt() of the negative diagonal, then eigen() stops
    # with "infinite or missing values in 'x'". Only the columns that vary in these rows are
    # kept, so the weight sum is the only cause.
    x, wrow, wcol, _ = inputs(BY_ID[cid])
    varies = np.ptp(x, axis=0) > 0
    x, wcol = x[:, varies], wcol[varies]
    assert r_sum(wrow) < 1
    assert x.shape[1] >= 2
    with pytest.raises(ValueError, match="corr=True needs wrow to sum to more than 1"):
        wpca(x, wrow, wcol, corr=True)


def test_weight_rules():
    x = BY_ID["iris"].array("x")
    n, p = x.shape
    neg = np.ones(n)
    neg[3] = -1.0
    with pytest.raises(ValueError, match="negative"):
        wpca(x, neg)
    for wcol in ([1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -0.5], [-1.0, -1.0, -1.0, 2.0]):
        with pytest.raises(ValueError, match="wcol must have a nonzero sum and weights of one"):
            wpca(x, wcol=np.array(wcol))
    with pytest.raises(ValueError, match="one weight per row"):
        wpca(x, np.ones(n - 1))
    with pytest.raises(ValueError, match="one weight per row"):
        wpca(x, np.ones((n, 1)))
    with pytest.raises(ValueError, match="one weight per column"):
        wpca(x, wcol=np.ones(p + 1))
    with pytest.raises(ValueError, match="one weight per column"):
        wpca(x, wcol=np.ones(n))
    with pytest.raises(TypeError, match="wrow must be numeric"):
        wpca(x, np.array(["1"] * n))
    with pytest.raises(TypeError, match="wcol must be numeric"):
        wpca(x, wcol=np.array(["1"] * p))
    with pytest.raises(TypeError, match="wrow must be numeric"):
        wpca(x, np.ones(n, dtype=bool))
    with pytest.raises(TypeError, match="wcol must be numeric"):
        wpca(x, wcol=np.ones(p, dtype=bool))
    for bad in (np.nan, np.inf):
        w = np.ones(n)
        w[0] = bad
        with pytest.raises(ValueError, match="wrow contains NaN or infinite"):
            wpca(x, w)
        w = np.ones(p)
        w[0] = bad
        with pytest.raises(ValueError, match="wcol contains NaN or infinite"):
            wpca(x, wcol=w)


def test_constant_column_with_corr_raises():
    x = BY_ID["mtcars_cyl8_constant"].array("x")
    assert (np.ptp(x, axis=0) == 0).any()
    with pytest.raises(ValueError, match="positive weighted variance in every column"):
        wpca(x, corr=True)


def test_one_row():
    x = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="sum to exactly 1"):
        wpca(x, np.array([1.0]))
    with pytest.raises(ValueError, match="sum to zero"):
        wpca(x, np.array([0.0]))
    for w in (0.5, 2.0):
        # R's Wpca(matrix(c(1, 2, 3), 1), wrow = w) returns sdev 0 0 0 and no warning
        got, rec = run(x, np.array([w]))
        assert rec == []
        assert_same_values(got.sdev, np.zeros(3))
        assert_same_values(got.center, x[0])
        assert_same_values(got.x, np.zeros((1, 3)))


def test_overflow_is_quiet_as_in_r():
    # R 4.6.1: Wpca(matrix(c(1e300, -1e300, 0, 1, 2, 3), 3), corr = TRUE) stops in eigen, no warning
    x = np.array([[1e300, 1.0], [-1e300, 2.0], [0.0, 3.0]])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="infinite or missing values in 'x'"):
            wpca(x, corr=True)
    assert rec == []
    # R 4.6.1: Wpca(matrix(c(1.5e308, 1, 3, 2, 1.5e308, 2, 1, 2), 4), wrow = c(0, 1, 1, 1)) gives a
    # finite sdev and an infinite second score in row 1, with no warning
    x = np.array([[1.5e308, 1.5e308], [1.0, 2.0], [3.0, 1.0], [2.0, 2.0]])
    got, rec = run(x, np.array([0.0, 1.0, 1.0, 1.0]))
    assert rec == []
    assert np.isfinite(got.sdev).all()
    assert np.isinf(got.x[0, 1])
    assert np.isfinite(got.x[0, 0])
    assert np.isfinite(got.x[1:]).all()


def test_corr_must_be_a_bool():
    x = BY_ID["iris"].array("x")
    for corr in (1, 0, 1.0, "TRUE", None):
        with pytest.raises(TypeError, match="corr must be True or False"):
            wpca(x, corr=corr)
    assert wpca(x, corr=np.bool_(True)) == wpca(x, corr=True)


def test_rejects_1d_and_non_finite_x():
    x = BY_ID["iris"].array("x")
    with pytest.raises(ValueError, match="2-D"):
        wpca(x[:, 0])
    for bad in (np.nan, np.inf, -np.inf):
        y = x.copy()
        y[2, 1] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            wpca(y)
    with pytest.raises(TypeError, match="numeric"):
        wpca(x.astype(str))
    with pytest.raises(ValueError, match="at least one row"):
        wpca(np.empty((0, 4)))


def test_default_weights_are_ones():
    x = BY_ID["iris"].array("x")
    n, p = x.shape
    got = wpca(x)
    assert got == wpca(x, np.ones(n), np.ones(p), False)
    assert_same_values(got.wrow, np.ones(n))
    assert_same_values(got.wcol, np.ones(p))
    counts = wpca(x, np.full(n, 2), np.full(p, 3))
    assert counts.wrow.dtype == np.float64
    assert counts == wpca(x, np.full(n, 2.0), np.full(p, 3.0))
