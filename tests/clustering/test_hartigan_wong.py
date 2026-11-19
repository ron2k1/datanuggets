"""Hartigan-Wong k-means against R's own C_kmns and stats::kmeans, bit for bit."""

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from datanuggets._clustering import kmnsjit
from datanuggets._clustering.hartigan_wong import KMeansError, kmeans, kmns
from datanuggets._random.replay import ReplayDraws
from kmns_reference import _kmns_py
from rfixtures import load_group, same_bits

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


def _same_kmns(got, want):
    assert (got.ifault, got.iter) == (want.ifault, want.iter)
    assert got.cluster == want.cluster
    assert got.size == want.size
    assert same_bits(got.centers, want.centers)
    assert same_bits(got.withinss, want.withinss)


@pytest.mark.parametrize("p", [1, 2, 5])
@pytest.mark.parametrize("kind", ["grid", "normal"])
def test_kmns_kernel_matches_the_python_port(kind, p):
    """Starts that repeat a row or a value (empty clusters), tiny step budgets (ifault 2 and 4),
    and k from 2 up to m and past it (ifault 3), on tied and untied data."""
    rng = np.random.default_rng(p * 7 + len(kind))
    faults = set()
    for _ in range(300):
        m = int(rng.integers(3, 60))
        if kind == "grid":
            a = rng.integers(0, 10, size=(m, p)).astype(float)
        else:
            a = rng.standard_normal((m, p))
        k = int(rng.integers(2, min(8, m + 1) + 1))
        repeats = k > m or rng.random() < 0.2
        c = a[rng.integers(0, m, size=k) if repeats else rng.choice(m, size=k, replace=False)]
        iter_max = int(rng.choice([1, 2, 10, 1000]))
        max_qtran = int(rng.choice([1, 3, 50 * m], p=[0.15, 0.15, 0.7]))
        want = _kmns_py(a, c, iter_max, max_qtran)
        _same_kmns(kmns(a, c, iter_max, max_qtran), want)
        faults.add(want.ifault)
    assert faults == {0, 1, 2, 3, 4}


def test_kmns_takes_any_layout_and_dtype():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 9, size=(40, 6))
    c = np.asfortranarray(a[:4].astype(float))
    want = _kmns_py(a[:, ::2].astype(float), np.ascontiguousarray(c[:, ::2]), 50, 2000)
    _same_kmns(kmns(a[:, ::2], c[:, ::2], 50, 2000), want)


def test_kmns_rejects_mismatched_columns():
    for a, c in ((np.ones((5, 3)), np.ones((2, 4))), (np.ones((5, 4)), np.ones((2, 3)))):
        with pytest.raises(ValueError, match="columns"):
            kmns(a, c, 10, 100)


def test_kmns_read_only_data_runs_the_same_compiled_kernels():
    case = max(RUNS, key=lambda c: c.array("x").size)
    want = _kmns(case)
    kernels = [k for k in vars(kmnsjit).values() if isinstance(k, CPUDispatcher)]
    compiled = [len(k.signatures) for k in kernels]
    x, c0 = case.array("x"), case.array("c0")
    x.setflags(write=False)
    c0.setflags(write=False)
    _same_kmns(_kmns(case, x, c0), want)
    assert [len(k.signatures) for k in kernels] == compiled  # no second signature to compile
