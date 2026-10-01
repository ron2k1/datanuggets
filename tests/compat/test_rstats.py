"""R's long double stats, unique(), %*% and weighted.mean bit for bit. eigen() to Numeric tier."""

import os
import signal
import subprocess
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import numba
import numpy as np
import pytest
from numba.core.registry import CPUDispatcher
from numba.np import ufunc as numba_ufunc

from datanuggets._compat import arithmetic, moments
from datanuggets._compat.arithmetic import (
    _r_cov_py,
    _r_mean_py,
    r_cov,
    r_mad,
    r_mean,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
    r_weighted_mean,
)
from datanuggets._compat.indexing import unique_first
from datanuggets._compat.linalg import r_eigen_sym, r_matmul, r_matvec
from datanuggets._compat.moments import group_moments
from rfixtures import ATOL, RTOL, fixture_data, hex_bits, load_group, same_bits

CASES = load_group("rcompat_stats")
WCASES = load_group("rcompat_wstats")


def _kind(kind):
    return [c for c in CASES if c.args["kind"] == kind]


def _bitwise(got, want, *label):
    # == and assert_array_equal pass -0.0 for 0.0 and any NaN for any other NaN
    assert same_bits(got, want), (*label, hex_bits(got), hex_bits(want))


COV = _kind("cov")
EIGEN = [c for c in COV if c.has("values")]


@pytest.mark.parametrize("case", COV, ids=lambda c: c.id)
def test_cov_and_scale_match_r_bitwise(case):
    x = case.array("x")
    s = r_cov(x)
    _bitwise(s, case.array("cov"), "cov")
    _bitwise(r_sum(np.diag(s)) / x.shape[1], case.array("scale")[0], "scale")


@pytest.mark.parametrize("case", EIGEN, ids=lambda c: c.id)
def test_eigen_matches_r_numerically(case):
    s = case.array("cov")
    np.testing.assert_allclose(r_eigen_sym(s), case.array("values"), rtol=RTOL, atol=ATOL)
    values, vectors = r_eigen_sym(s, vectors=True)
    np.testing.assert_allclose(values, case.array("values_v"), rtol=RTOL, atol=ATOL)
    got, want = vectors[:, 0], case.array("vectors")[:, 0]
    np.testing.assert_allclose(got * np.sign(got @ want), want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", EIGEN, ids=lambda c: c.id)
def test_matvec_matches_r_bitwise(case):
    got = r_matvec(case.array("x"), case.array("vectors")[:, 0])
    _bitwise(got, case.array("proj"))


@pytest.mark.parametrize("case", _kind("vec"), ids=lambda c: c.id)
def test_vector_stats_match_r_bitwise(case):
    v = case.array("v").tolist()
    for name, stat in (("sum", r_sum), ("mean", r_vec_mean), ("median", r_median), ("mad", r_mad)):
        _bitwise(stat(v), case.array(name)[0], name)
    _bitwise([r_quantile7(v, q) for q in case.args["probs"]], case.array("q"), "q")


@pytest.mark.parametrize("case", _kind("mean"), ids=lambda c: c.id)
def test_mean_of_non_finite_values_matches_r(case):
    # which NaN comes back (sign, NA's payload) is part of R's answer
    _bitwise(r_vec_mean(case.array("v").tolist()), case.array("mean")[0])


@pytest.mark.parametrize("case", _kind("sum"), ids=lambda c: c.id)
def test_sum_matches_r_past_dbl_max_and_with_inf_or_nan(case):
    _bitwise(r_sum(case.array("v").tolist()), case.array("sum")[0])


@pytest.mark.parametrize("case", _kind("unique"), ids=lambda c: c.id)
def test_unique_first_matches_r(case):
    np.testing.assert_array_equal(unique_first(case.array("x")) + 1, case.array("first"))


def _wkind(kind):
    return [c for c in WCASES if c.args["kind"] == kind]


MATMUL = _wkind("matmul")
WMEAN = _wkind("wmean")
WSUM = _wkind("sum")
NAN_MATMUL = "matmul_mtcars_gear_am_centers_nan"
NEGZERO_MATMUL = "matmul_negzero_column"
ZERO_TOTAL_WMEAN = "wmean_mtcars_gear3_mpg_am_zero_total"
INF_AT_ZERO_WMEAN = "wmean_iris_sepallength_inf_at_zero_weight"


def _wcase(case_id):
    return next(c for c in WCASES if c.id == case_id)


def test_rcompat_wstats_covers_each_kind():
    assert MATMUL and WMEAN and WSUM
    ids = {c.id for c in WCASES}
    assert {NAN_MATMUL, NEGZERO_MATMUL, ZERO_TOTAL_WMEAN, INF_AT_ZERO_WMEAN} <= ids
    # R's [w != 0] drops the Inf * 0 term, which would otherwise make the mean NaN
    inf_at_zero = _wcase(INF_AT_ZERO_WMEAN)
    v, w = inf_at_zero.array("v"), inf_at_zero.array("w")
    assert np.isinf(v[w == 0]).any() and np.isfinite(inf_at_zero.array("mean")).all()
    prod = _wcase(NAN_MATMUL).array("prod")
    assert np.isnan(prod[:, 0]).all() and np.isfinite(prod[:, 1:]).all()
    b = _wcase(NEGZERO_MATMUL).array("b")
    assert (b[:, 0] == 0).all() and np.signbit(b[:, 0]).all()
    zero = _wcase(ZERO_TOTAL_WMEAN)
    assert (zero.array("w") == 0).all()
    assert hex_bits(zero.array("mean")) == ["fff8000000000000"]  # the x87 and SSE default NaN


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("case", MATMUL, ids=lambda c: c.id)
def test_r_matmul_matches_r(case, order):
    a = np.asarray(case.array("a"), order=order)
    b = np.asarray(case.array("b"), order=order)
    got, want = r_matmul(a, b), case.array("prod")
    nan = np.isnan(want)
    # which NaN x * NaN gives depends on the machine, so NaN is compared by position only
    np.testing.assert_array_equal(np.isnan(got), nan)
    _bitwise(got[~nan], want[~nan])


def test_r_matmul_never_returns_negative_zero():
    case = _wcase(NEGZERO_MATMUL)
    a, b = case.array("a"), case.array("b")
    assert (a > 0).all() and np.signbit(b[:, 0]).all()  # every product in column 0 is -0
    got = r_matmul(a, b)
    assert (got[:, 0] == 0).all() and not np.signbit(got[:, 0]).any()
    _bitwise(got, case.array("prod"))


def test_r_matvec_never_returns_negative_zero():
    # R's matrix times vector goes through dgemv, whose y starts at +0 as dgemm's does, in the
    # same order. On R 4.6.1, as.matrix(iris[1:5, 1:4]) %*% rep(-0, 4) is five +0.
    case = _wcase(NEGZERO_MATMUL)
    a, v = case.array("a"), case.array("b")[:, 0]
    assert (a > 0).all() and np.signbit(v).all()
    got = r_matvec(a, v)
    assert (got == 0).all() and not np.signbit(got).any()
    _bitwise(got, case.array("prod")[:, 0])


@pytest.mark.parametrize("case", WMEAN, ids=lambda c: c.id)
def test_r_weighted_mean_matches_r(case):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a zero total gives NaN or Inf without a numpy warning
        got = r_weighted_mean(case.array("v"), case.array("w"))
    _bitwise(got, case.array("mean")[0])


@pytest.mark.parametrize("case", WSUM, ids=lambda c: c.id)
def test_r_sum_boundaries(case):
    _bitwise(r_sum(case.array("v")), case.array("sum")[0])


DATA = fixture_data()
BIGGEST = max((k for k in DATA if DATA[k][1]), key=lambda k: DATA[k][0].size)


def _nuggets(a):
    """Each nugget's rows in ascending order, for 0-based assignments a."""
    order = np.argsort(a, kind="stable")
    return np.split(order, np.cumsum(np.bincount(a))[:-1])


@pytest.mark.parametrize("key", sorted(DATA))
def test_r_cov_matches_the_replay_on_every_fixture_nugget(key):
    x, sets = DATA[key]
    for rows in [np.arange(x.shape[0])] + [g for a in sets for g in _nuggets(a)]:
        if rows.size >= 2:
            assert same_bits(r_cov(x[rows]), _r_cov_py(x[rows])), rows[:5]


def test_r_cov_uses_the_compiled_path_for_finite_data(monkeypatch):
    def replay(x):
        raise AssertionError("r_cov fell back to the Python replay")

    monkeypatch.setattr(arithmetic, "_r_cov_py", replay)
    x = DATA[BIGGEST][0]
    assert r_cov(x).shape == (x.shape[1], x.shape[1])


EDGE_COLUMNS = {
    "subnormal": [5e-324, -5e-324, 1e-310, 0.0, 2.5e-320, -1e-315],
    "signed_zero": [0.0, -0.0, -0.0, 0.0, -0.0, 0.0],
    "constant": [3.7] * 6,
    "big_offset": [1e15 + 0.125 * i for i in range(6)],
    "near_one": [1.0, 1.0 + 2**-52, 1.0, 1.0 - 2**-53, 1.0, 1.0 + 2**-51],
    "huge": [1e300, -1e300, 1e300, -1e300, 1e300, 1.7e308],
}


def test_r_cov_edge_columns_match_the_replay():
    # every pair of columns meets in one cov() call, and "huge" gives covariances past DBL_MAX
    x = np.column_stack(list(EDGE_COLUMNS.values()))
    got = r_cov(x)
    assert same_bits(got, _r_cov_py(x))
    assert np.isinf(got[-1, -1])


def test_r_cov_takes_any_layout_and_dtype():
    x = DATA[BIGGEST][0]
    want = _r_cov_py(x)
    for view in (np.asfortranarray(x), np.repeat(x, 2, axis=0)[::2]):
        assert same_bits(r_cov(view), want)
    ints = np.rint(x * 10).astype(np.int64)
    assert same_bits(r_cov(ints), _r_cov_py(ints.astype(np.float64)))


@pytest.mark.parametrize(("bad", "error"), [(np.inf, OverflowError), (np.nan, ValueError)])
def test_r_cov_raises_on_inf_and_nan_like_the_replay(bad, error):
    x = np.array([[1.0, 2.0], [bad, 3.0], [4.0, 5.0]])
    with pytest.raises(error, match="cannot convert"):
        r_cov(x)


MOMENT_SETS = [(k, i) for k in sorted(DATA) for i in range(len(DATA[k][1]))]


def _same_moments(got, want) -> bool:
    return len(got) == len(want) and all(
        same_bits(c, wc) and (v is None if wv is None else same_bits(v, wv))
        for (c, v), (wc, wv) in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize(("key", "index"), MOMENT_SETS, ids=[f"{k}#{i}" for k, i in MOMENT_SETS])
def test_group_moments_match_the_replay_on_every_fixture_nugget(key, index, n_jobs):
    x, sets = DATA[key]
    groups = _nuggets(sets[index])
    got = list(group_moments(x, groups, n_jobs))
    want = [(_r_mean_py(x[g], 0), _r_cov_py(x[g]) if g.size >= 2 else None) for g in groups]
    assert _same_moments(got, want)


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_group_moments_uses_the_compiled_path_for_finite_data(monkeypatch, n_jobs):
    def replay(*args):
        raise AssertionError("group_moments fell back to r_mean or r_cov")

    monkeypatch.setattr(moments, "r_mean", replay)
    monkeypatch.setattr(moments, "r_cov", replay)
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    assert len(list(group_moments(x, groups, n_jobs))) == len(groups)


def _all_outputs(x, groups):
    moments = [a for m in group_moments(x, groups) for a in m if a is not None]
    return [r_mean(x, 0), r_mean(x, 1), r_cov(x), *moments]


def test_read_only_input_runs_the_same_compiled_kernels():
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = _all_outputs(x, groups)
    kernels = [k for k in vars(moments._ldjit).values() if isinstance(k, CPUDispatcher)]
    compiled = [len(k.signatures) for k in kernels]
    frozen = x.copy()
    frozen.setflags(write=False)
    got = _all_outputs(frozen, groups)
    assert all(same_bits(g, w) for g, w in zip(got, want, strict=True))
    assert [len(k.signatures) for k in kernels] == compiled  # no second signature to compile


def test_group_moments_chunking_gives_the_same_result(monkeypatch):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    p = x.shape[1]
    for chunk in (1, 3 * p * p):  # one group per call, then three
        monkeypatch.setattr(moments, "MOMENTS_CHUNK", chunk)  # moments' own binding of it
        assert _same_moments(list(group_moments(x, groups)), want), chunk
        assert _same_moments(list(group_moments(x, groups, 4)), want), chunk


@pytest.mark.parametrize("n_jobs", [-1, 10**6])
def test_group_moments_caps_and_restores_threads(monkeypatch, n_jobs):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    before = numba.get_num_threads()
    asked = []
    real = numba.set_num_threads

    def spy(k):
        asked.append(k)
        real(k)

    monkeypatch.setattr(numba, "set_num_threads", spy)
    assert _same_moments(list(group_moments(x, groups, n_jobs)), want)
    assert asked == [numba.config.NUMBA_NUM_THREADS, before]
    assert numba.get_num_threads() == before


def test_group_moments_from_two_threads_at_once(monkeypatch):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = list(group_moments(x, groups))
    monkeypatch.setattr(moments, "MOMENTS_CHUNK", x.shape[1] ** 2)  # a launch per group
    start = threading.Barrier(2)

    def run():
        start.wait()
        return list(group_moments(x, groups, 4))

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results = [f.result() for f in futures]
    assert all(_same_moments(got, want) for got in results)


def test_group_moments_from_two_threads_under_workqueue():
    # workqueue aborts the process when two launches overlap, so a missing lock fails here on
    # every machine, not only where workqueue is the default layer
    test = f"{__file__}::test_group_moments_from_two_threads_at_once"
    env = {**os.environ, "NUMBA_THREADING_LAYER": "workqueue"}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    # an abort dumps its traceback from the top of stderr
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[:3000]


def test_group_moments_leaves_non_finite_groups_to_the_replay():
    x = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0], [6.0, 7.0]])
    groups = [np.array([1]), np.array([2, 3]), np.array([], dtype=np.int64)]
    (c0, v0), (c1, v1), (c2, v2) = group_moments(x, groups)
    assert same_bits(c0, [np.inf, 3.0]) and v0 is None
    assert same_bits(c1, _r_mean_py(x[[2, 3]], 0)) and same_bits(v1, _r_cov_py(x[[2, 3]]))
    assert np.isnan(c2).all() and v2 is None
    with pytest.raises(OverflowError):
        list(group_moments(x, [np.array([0, 1])]))


def test_group_moments_rejects_rows_past_the_end():
    x = np.arange(12.0).reshape(4, 3)
    with pytest.raises(IndexError):
        list(group_moments(x, [np.array([2, 4])]))


@pytest.mark.parametrize("rows", [[True, False, True, False], [-4, -2]], ids=["mask", "negative"])
def test_group_moments_reads_rows_the_way_numpy_does(rows):
    x = np.arange(12.0).reshape(4, 3)
    rows = np.array(rows)
    ((center, cov),) = group_moments(x, [rows])
    assert same_bits(center, [3.0, 4.0, 5.0]) and same_bits(cov, _r_cov_py(x[rows]))


LAYERS = [
    ("omp", "GNU", True),
    ("omp", "MS", False),
    ("omp", "Intel", False),
    ("workqueue", "GNU", False),
    ("tbb", "GNU", False),
]


@pytest.mark.parametrize(("layer", "vendor", "forbids"), LAYERS)
def test_only_gnu_openmp_forbids_a_launch_after_fork(monkeypatch, layer, vendor, forbids):
    monkeypatch.setattr(numba, "threading_layer", lambda: layer)
    fake = SimpleNamespace(openmp_vendor=vendor)
    monkeypatch.setattr(numba_ufunc, "omppool", fake, raising=False)
    assert moments._layer_forbids_fork() is forbids


def test_no_parallel_launch_yet_leaves_fork_safe(monkeypatch):
    def unset():
        raise ValueError("Threading layer is not initialized.")

    monkeypatch.setattr(numba, "threading_layer", unset)
    assert moments._layer_forbids_fork() is False


def _moment_bytes(moments) -> bytes:
    return b"".join(c.tobytes() + (b"" if v is None else v.tobytes()) for c, v in moments)


def _forked_child_moments(x, groups, n_jobs) -> tuple[int, bytes]:
    """group_moments in a forked child: its exit code and the bytes it sent. A child still
    running after 60 s dies of SIGALRM, so a deadlock fails the test instead of hanging it."""
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # the child sends its bits and exits without returning into pytest
        code = 1
        try:
            signal.alarm(60)
            with os.fdopen(write, "wb") as pipe:
                pipe.write(_moment_bytes(group_moments(x, groups, n_jobs)))
            code = 0
        except BaseException as e:
            os.write(2, f"child: {e!r}\n".encode())
            raise  # the finally exits first, so this only satisfies the linter
        finally:
            os._exit(code)
    os.close(write)
    with os.fdopen(read, "rb") as pipe:
        sent = pipe.read()
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), sent


@contextmanager
def _launch_in_progress():
    """Another thread inside a launch, holding _THREADS, until the block ends."""
    held, done = threading.Event(), threading.Event()

    def launch():
        with moments._THREADS:
            held.set()
            done.wait()

    thread = threading.Thread(target=launch)
    thread.start()
    held.wait()
    try:
        yield
    finally:
        done.set()
        thread.join()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() from a process with threads
def test_group_moments_runs_serially_in_a_child_forked_after_gnu_openmp(monkeypatch):
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = _moment_bytes(group_moments(x, groups, 2))  # the parent launches before the fork

    def launch(*args):
        raise AssertionError("numba threads launched")

    monkeypatch.setattr(moments, "_layer_forbids_fork", lambda: True)
    monkeypatch.setattr(moments._ldjit, "group_moments_parallel", launch)
    assert _forked_child_moments(x, groups, 2) == (0, want)
    with pytest.raises(AssertionError, match="numba threads launched"):  # only the child is serial
        list(group_moments(x, groups, 2))


def test_a_child_forked_during_a_launch_gets_a_free_lock(monkeypatch):
    # the fork copies _THREADS held, but not the thread that would release it
    monkeypatch.setattr(moments, "_THREADS", moments._THREADS)  # restored after the test
    monkeypatch.setattr(moments, "_FORKED_UNSAFE", moments._FORKED_UNSAFE)
    with _launch_in_progress():
        moments._after_fork_in_child()  # what os.register_at_fork runs in the child
        assert not moments._THREADS.locked()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() from a process with threads
def test_group_moments_runs_in_a_child_forked_during_another_threads_launch():
    x, sets = DATA[BIGGEST]
    groups = _nuggets(sets[0])
    want = _moment_bytes(group_moments(x, groups))
    with _launch_in_progress():
        got = _forked_child_moments(x, groups, 2)
    assert got == (0, want)
