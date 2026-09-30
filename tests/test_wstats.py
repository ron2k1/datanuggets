"""wmean, wss, wwcss and distw against WCluster's on every case of the wstats groups."""

import warnings

import numpy as np
import pytest

from datanuggets.wcluster.wstats import distw, wmean, wss, wwcss
from rfixtures import FIXTURE_ROOT, assert_same_values, load_group

CASES = load_group("wstats") + load_group("wstats_constructed")
REAL = {c.id: c for c in load_group("wstats")}
BY_ID = {c.id: c for c in CASES}
WMEAN = [c for c in CASES if c.has("wmean")]
WSS = [c for c in CASES if c.has("wss")]
WWCSS = [c for c in CASES if c.has("wwcss_group") and c.has("wwcss_total")]
DISTW = [c for c in CASES if c.has("distw")]
K3 = ["iris_species_petallength_k3", "mtcars_cyl_qsec_k3", "usarrests_regions123_pop_k3"]
K3 += ["plantgrowth_group_k3"]
ORDERS = pytest.mark.parametrize("order", ["C", "F"])


def _ids(cases):
    return [c.id for c in cases]


def _inputs(case, order="C"):
    x = case.array("x")
    x = np.asfortranarray(x) if order == "F" else np.ascontiguousarray(x)
    return x, case.array("cl"), case.array("w")


def _k(case) -> int:
    return np.unique(case.array("cl")).size


@ORDERS
@pytest.mark.parametrize("case", WMEAN, ids=_ids(WMEAN))
def test_wmean_matches_r(case, order):
    x, cl, w = _inputs(case, order)
    want = case.array("wmean")
    assert want.shape == (x.shape[1], _k(case))  # R returns p by k
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = wmean(x, cl, w)
    assert_same_values(got, want.T)


@ORDERS
@pytest.mark.parametrize("case", WSS, ids=_ids(WSS))
def test_wss_matches_r(case, order):
    x, _, w = _inputs(case, order)
    (want,) = case.array("wss")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = wss(x, w)
    assert isinstance(got, float)
    assert_same_values(got, want)


@ORDERS
@pytest.mark.parametrize("case", WWCSS, ids=_ids(WWCSS))
def test_wwcss_matches_r(case, order):
    x, cl, w = _inputs(case, order)
    groups, (total,) = case.array("wwcss_group"), case.array("wwcss_total")
    assert groups.shape == (_k(case),)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = wwcss(x, cl, w)
        res = wwcss(x, cl, w, group_sum=True)
    assert isinstance(got, float)
    assert_same_values(got, total)
    assert res.total.shape == ()
    assert_same_values(res.per_cluster, groups)
    assert_same_values(res.total, total)


@ORDERS
@pytest.mark.parametrize("case", DISTW, ids=_ids(DISTW))
def test_distw_matches_r(case, order):
    x, cl, w = _inputs(case, order)
    want = case.array("distw")
    k = _k(case)
    assert k >= 2 and want.shape == (k, k)
    assert np.isnan(want[np.triu_indices(k)]).all()  # R's NA on the diagonal and above
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = distw(x, cl, w)
    assert np.isnan(got[np.triu_indices(k)]).all()
    low = np.tril_indices(k, -1)
    assert_same_values(got[low], want[low])
    assert_same_values(got, want)


def _dead_column(case) -> list[int]:
    """The clusters whose R center is NaN in every coordinate."""
    return np.flatnonzero(np.isnan(case.array("wmean")).all(axis=0)).tolist()


def _has_exact_tie(z: np.ndarray) -> bool:
    return any((row == np.nanmin(row)).sum() > 1 for row in z if not np.isnan(row).all())


def test_fixtures_cover_the_edge_cases():
    for cases in (WMEAN, WSS, WWCSS, DISTW, K3):
        assert cases
    assert len(BY_ID) == len(CASES)  # case ids are unique across both groups
    assert set(K3) <= set(REAL)
    assert REAL["quakes_int2_mag5"].manifest["arrays"]["x"]["dtype"] == "<i4"
    assert REAL["quakes_int_by_mag"].manifest["arrays"]["x"]["dtype"] == "<i4"
    assert REAL["mtcars_cyl_carb"].manifest["arrays"]["cl"]["dtype"] == "<f8"
    for cid in ("iris_one_cluster", "mtcars_one_carb", "volcano_one_ones"):
        assert _k(REAL[cid]) == 1
        assert not REAL[cid].has("distw")
    for cid in ("faithful1d", "mtcars_cyl1d_gear", "plantgrowth_group_k3"):
        assert REAL[cid].array("x").shape[1] == 1
    for cid in ("mtcars_cyl_carb_row15", "iris_species_pw_row150", "mtcars_gear_am_row01"):
        assert REAL[cid].array("newx").shape[0] == 1
    assert _dead_column(REAL["mtcars_gear_am"]) == [0]
    assert _dead_column(REAL["mtcars_carb_am"]) == [2]
    assert _k(REAL["mtcars_carb_am"]) == 6
    assert _dead_column(REAL["mtcars_cyl_vs"]) == [_k(REAL["mtcars_cyl_vs"]) - 1]
    assert (REAL["iris_singletons_neg_pw"].array("cl") < 0).all()
    assert (REAL["iris50_singletons"].array("z") < 0).any()
    assert _has_exact_tie(REAL["mtcars_cyl1d_gear"].array("z"))


# The feature may add 8 MB allocated to tests/fixtures, measured with du -sk on the laptop
# (12 MB before it). wpca and rcompat_wstats take about 2 MB of that. On this disk 4.5 MB of
# arrays comes to about 5.8 MB allocated, so the statistics groups stay inside their share. Only
# the .bin files count here: a manifest's line endings depend on how it was checked out.
ARRAY_BUDGET = 4_500_000


def test_fixture_groups_stay_within_their_share_of_the_budget():
    stored = sum(
        f.stat().st_size
        for group in ("wstats", "wstats_constructed")
        for f in (FIXTURE_ROOT / group).rglob("*.bin")
    )
    assert stored <= ARRAY_BUDGET


@pytest.mark.parametrize(
    ("cid", "pos"), [("mtcars_gear_am", 0), ("mtcars_carb_am", 2), ("mtcars_cyl_vs", -1)]
)
def test_zero_weight_cluster_is_nan_and_quiet(cid, pos):
    case = REAL[cid]
    x, cl, w = _inputs(case)
    k = _k(case)
    pos %= k
    dead = np.arange(k) == pos
    want_mean, want_groups = case.array("wmean").T, case.array("wwcss_group")
    # R: the cluster's weights sum to zero, so its center, its wss and the total are NaN
    assert np.isnan(want_mean[dead]).all() and np.isfinite(want_mean[~dead]).all()
    assert np.isnan(want_groups[dead]).all() and np.isfinite(want_groups[~dead]).all()
    assert np.isnan(case.array("wwcss_total")).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean = wmean(x, cl, w)
        res = wwcss(x, cl, w, group_sum=True)
        pairs = distw(x, cl, w)
    assert np.isnan(mean[dead]).all() and np.isfinite(mean[~dead]).all()
    assert np.isnan(res.per_cluster[dead]).all() and np.isfinite(res.per_cluster[~dead]).all()
    assert np.isnan(res.total)
    assert np.isnan(pairs[pos, :pos]).all() and np.isnan(pairs[pos + 1 :, pos]).all()
    assert_same_values(mean, want_mean)
    assert_same_values(res.per_cluster, want_groups)
    assert_same_values(pairs, case.array("distw"))


def test_labels_need_not_start_at_zero():
    case = REAL["mtcars_cyl_carb"]
    x, cl, w = _inputs(case)
    assert cl.dtype == np.float64 and np.array_equal(np.unique(cl), [4.0, 6.0, 8.0])
    codes = np.searchsorted([4.0, 6.0, 8.0], cl)
    for labels in (cl, cl.astype(np.int64), cl.astype(np.int32)):
        assert_same_values(wmean(x, labels, w), wmean(x, codes, w))
        assert wwcss(x, labels, w, group_sum=True) == wwcss(x, codes, w, group_sum=True)
        assert_same_values(distw(x, labels, w), distw(x, codes, w))
    assert_same_values(wmean(x, codes, w), case.array("wmean").T)


def test_default_weights_are_ones():
    # R's own default, rep(1, length(x)), is one weight per column in wmean and wwcss (x is a
    # data frame by then), so they stop unless x is square, and one per cell in distw (a
    # matrix), so it stops unless x has one column. wss defaults to
    # rep(1, nrow(x)), which works. The port takes None as one weight of 1 per row.
    case = REAL["iris_species_ones"]
    x, cl, w = _inputs(case)
    assert np.array_equal(w, np.ones(x.shape[0]))
    assert_same_values(wmean(x, cl), case.array("wmean").T)
    assert_same_values(wss(x), case.array("wss")[0])
    assert_same_values(wwcss(x, cl), case.array("wwcss_total")[0])
    assert_same_values(distw(x, cl), case.array("distw"))
    x, cl, _ = _inputs(REAL["mtcars_gear_vs"])
    ones = np.ones(x.shape[0])
    assert_same_values(wmean(x, cl), wmean(x, cl, ones))
    assert wss(x) == wss(x, ones)
    assert wwcss(x, cl, group_sum=True) == wwcss(x, cl, ones, group_sum=True)
    assert_same_values(distw(x, cl), distw(x, cl, ones))


def test_distw_needs_two_clusters():
    # R's loop runs i over 2:1, reads the missing second label as NA and stops with
    # "missing value where TRUE/FALSE needed"
    case = REAL["iris_one_cluster"]
    x, cl, w = _inputs(case)
    assert _k(case) == 1
    with pytest.raises(ValueError, match="at least two clusters"):
        distw(x, cl, w)


CALLS = {
    "wmean": lambda x, cl, w: wmean(x, cl, w),
    "wss": lambda x, cl, w: wss(x, w),
    "wwcss": lambda x, cl, w: wwcss(x, cl, w),
    "distw": lambda x, cl, w: distw(x, cl, w),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_weight_and_label_errors(name):
    call = CALLS[name]
    x, cl, w = _inputs(REAL["iris_species_petalwidth"])
    # R stops in every one: "The length of weights should be the same as the number of
    # observations."
    for bad in (w[:-1], w[: x.shape[1]], np.ones((x.shape[0], 1))):
        with pytest.raises(ValueError, match=r"w must have one weight per row \(150\)"):
            call(x, cl, bad)
    # R stops on text weights and takes TRUE and FALSE as 1 and 0
    for bad in (w.astype(str), w > 1):
        with pytest.raises(TypeError, match="w must be numeric"):
            call(x, cl, bad)
    # R gives NA or NaN results for an NA or infinite weight
    for value in (np.nan, np.inf):
        bad = w.copy()
        bad[7] = value
        with pytest.raises(ValueError, match="w contains NaN or infinite values"):
            call(x, cl, bad)
    if name == "wss":
        return  # wss takes no labels
    # R stops: "The length of cluster assignments should be the same as the number of
    # observations."
    with pytest.raises(ValueError, match=r"cl must have one label per row \(150\)"):
        call(x, cl[:-1], w)
    # R groups text and logical labels in its own order, which the port doesn't reproduce
    for bad in (cl.astype(str), cl > 1):
        with pytest.raises(TypeError, match="cl must hold numeric cluster labels"):
            call(x, bad, w)
    # R crashes on an NA label ("NA/NaN argument") and groups fractional labels by factor()
    for value in (0.5, np.nan, np.inf):
        bad = cl.astype(np.float64)
        bad[3] = value
        with pytest.raises(ValueError, match="cl must hold whole numbers"):
            call(x, bad, w)
    bad = cl.astype(np.float64)
    bad[3] = 1e15
    with pytest.raises(ValueError, match="smaller than 1e15"):
        call(x, bad, w)


@pytest.mark.parametrize("name", list(CALLS))
def test_rejects_1d_and_non_finite_x(name):
    call = CALLS[name]
    x, cl, w = _inputs(REAL["iris_species_petalwidth"])
    with pytest.raises(ValueError, match="x must be 2-D"):
        call(x[:, 0], cl, w)
    with pytest.raises(TypeError, match="x must contain only numeric columns"):
        call(x.astype(str), cl, w)
    for value in (np.nan, np.inf, -np.inf):
        bad = x.copy()
        bad[2, 1] = value
        with pytest.raises(ValueError, match="x contains NaN or infinite values"):
            call(bad, cl, w)


def test_accepts_a_dataframe():
    pd = pytest.importorskip("pandas")
    case = REAL["iris_species_petalwidth"]
    x, cl, w = _inputs(case)
    frame = pd.DataFrame(x, columns=["sl", "sw", "pl", "pw"])
    assert_same_values(wmean(frame, pd.Series(cl), pd.Series(w)), wmean(x, cl, w))
    assert wss(frame, w) == wss(x, w)
    assert wwcss(frame, cl, w) == wwcss(x, cl, w)
    assert_same_values(distw(frame, cl, w), case.array("distw"))


@pytest.mark.parametrize("cid", K3)
def test_wwcss_total_is_the_long_double_sum(cid):
    case = REAL[cid]
    x, cl, w = _inputs(case)
    g, (want,) = case.array("wwcss_group"), case.array("wwcss_total")
    assert g.shape == (3,)
    # R's sum() adds the three in long double: a double sum lands on another value
    assert want != float(np.float64(g[0]) + g[1] + g[2])
    assert_same_values(wwcss(x, cl, w), want)
    assert_same_values(wwcss(x, cl, w, group_sum=True).total, want)
