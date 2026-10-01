"""Hartigan-Wong k-means against R's own C_kmns and stats::kmeans, bit for bit."""

import numpy as np
import pytest

from datanuggets._clustering.hartigan_wong import KMeansError, kmeans, kmns
from datanuggets._random.replay import ReplayDraws
from rfixtures import load_group

KMNS = load_group("kmns")
KMEANS = load_group("kmeans")
RUNS = [c for c in KMNS if c.args["ifault"] in (0, 2, 4)]
STOPS = [c for c in KMNS if c.args["ifault"] in (1, 3)]


def _kmns(case, x=None, c0=None):
    a = case.args
    x = case.array("x") if x is None else x
    c0 = case.array("c0") if c0 is None else c0
    return kmns(x, c0, a["iter.max"], a["isteps"])


def test_fixtures_cover_every_fault_code():
    assert {c.args["ifault"] for c in KMNS} == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("case", RUNS, ids=lambda c: c.id)
def test_kmns_matches_r(case):
    z = _kmns(case)
    assert z.ifault == case.args["ifault"]
    np.testing.assert_array_equal(np.array(z.cluster) + 1, case.array("c1"))
    np.testing.assert_array_equal(z.centers, case.array("centers"))
    np.testing.assert_array_equal(z.size, case.array("nc"))
    np.testing.assert_array_equal(z.withinss, case.array("wss"))
    assert z.iter == case.array("iter")[0]


@pytest.mark.parametrize("case", STOPS, ids=lambda c: c.id)
def test_kmns_reports_the_faults_kmeans_stops_on(case):
    assert _kmns(case).ifault == case.args["ifault"]


def test_kmns_is_symmetric_under_negation():
    """refine_dn clusters projections onto an eigenvector whose sign LAPACK picks. Negating
    the data and starts must give the same clusters and exactly negated centers."""
    case = next(c for c in KMNS if c.id == "faithful1d_k2")
    z = _kmns(case)
    neg = _kmns(case, -case.array("x"), -case.array("c0"))
    assert neg.cluster == z.cluster
    np.testing.assert_array_equal(neg.centers, -z.centers)
    np.testing.assert_array_equal(neg.withinss, z.withinss)


def test_kmeans_fixtures_surface_every_reachable_fault():
    """What kmeans() makes of kmns's fault codes, end to end. ifault 1 (an empty cluster) needs
    repeated starts, which kmeans() never passes kmns."""
    errors = {c.manifest["error"] for c in KMEANS}
    assert "number of cluster centres must lie between 1 and nrow(x)" in errors  # ifault 3
    warned = [w for c in KMEANS for w in c.manifest["warnings"]]
    assert any(w.startswith("did not converge in") for w in warned)  # ifault 2
    assert any(w.startswith("Quick-TRANSfer stage steps exceeded maximum") for w in warned)  # 4


def _run(case, draws):
    a = case.args
    return kmeans(case.array("x"), a["k"], draws, nstart=a["nstart"], iter_max=a["iter.max"])


@pytest.mark.parametrize("case", [c for c in KMEANS if not c.manifest["error"]], ids=lambda c: c.id)
def test_kmeans_matches_r(case):
    draws = ReplayDraws(case.draws())
    got = _run(case, draws)
    draws.assert_exhausted()
    np.testing.assert_array_equal(got.cluster + 1, case.array("cluster"))
    np.testing.assert_array_equal(got.centers, case.array("centers"))
    np.testing.assert_array_equal(got.withinss, case.array("withinss"))
    np.testing.assert_array_equal(got.size, case.array("size"))
    assert got.tot_withinss == case.array("tot_withinss")[0]
    assert got.iter == case.array("iter")[0]
    assert list(got.warnings) == case.manifest["warnings"]


@pytest.mark.parametrize("case", [c for c in KMEANS if c.manifest["error"]], ids=lambda c: c.id)
def test_kmeans_stops_where_r_stops(case):
    draws = ReplayDraws(case.draws())
    with pytest.raises(KMeansError) as err:
        _run(case, draws)
    draws.assert_exhausted()
    assert str(err.value) == case.manifest["error"]
    assert list(err.value.warnings) == case.manifest["warnings"]
