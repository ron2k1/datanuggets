import importlib
import pickle

import numpy as np
import pytest

from datanuggets._random.interfaces import Draws
from datanuggets._random.numpy_draws import NumpyDraws
from datanuggets._random.replay import DrawRecord, ReplayDraws, ReplayMismatch
from datanuggets._random.seeds import as_draws
from rfixtures import load_group


def test_replay_returns_zero_based_values_in_order():
    draws = ReplayDraws([DrawRecord(3, 3, np.array([2, 3, 1])), DrawRecord(4, 1, np.array([4]))])
    np.testing.assert_array_equal(draws.permutation(3), [1, 2, 0])
    assert draws.index(4) == 3
    draws.assert_exhausted()


def test_replay_rejects_a_draw_r_did_not_make():
    draws = ReplayDraws([DrawRecord(3, 3, np.array([2, 3, 1]))])
    with pytest.raises(ReplayMismatch, match=r"sample.int\(5, 5\)"):
        draws.permutation(5)


def test_replay_rejects_extra_draws():
    draws = ReplayDraws([])
    with pytest.raises(ReplayMismatch, match="made only 0"):
        draws.index(2)


def test_replay_reports_unconsumed_draws():
    draws = ReplayDraws([DrawRecord(2, 1, np.array([1]))])
    with pytest.raises(ReplayMismatch, match="consumed 0"):
        draws.assert_exhausted()


def test_numpy_draws_are_reproducible():
    a, b = NumpyDraws(291102), NumpyDraws(291102)
    np.testing.assert_array_equal(a.permutation(50), b.permutation(50))
    assert a.index(7) == b.index(7)
    assert sorted(NumpyDraws(1).permutation(10).tolist()) == list(range(10))
    assert NumpyDraws(1).index(1) == 0


def test_as_draws_accepts_int_generator_and_draws():
    assert isinstance(as_draws(5), NumpyDraws)
    assert isinstance(as_draws(np.random.default_rng(5)), NumpyDraws)
    replay = ReplayDraws([])
    assert as_draws(replay) is replay


@pytest.mark.parametrize("bad", [1.5, True, "7", None])
def test_as_draws_rejects_other_types(bad):
    with pytest.raises(TypeError, match="seed"):
        as_draws(bad)


def test_fixture_draws_round_trip():
    case = next(c for c in load_group("create_dn") if c.id == "iris_euclidean_random")
    records = case.draws()
    assert records[0].n == 150 and records[0].size == 150
    assert sorted(records[0].value.tolist()) == list(range(1, 151))
    assert all(r.size == 1 for r in records[1:])


def test_numpy_sample_draws_distinct_values_in_range():
    got = NumpyDraws(1).sample(10, 4)
    assert got.shape == (4,)
    assert len(set(got.tolist())) == 4
    assert got.min() >= 0 and got.max() < 10


def test_numpy_fork_with_int_seed_restarts_the_stream():
    draws = NumpyDraws(5)
    first = draws.fork().sample(1000, 5)
    np.testing.assert_array_equal(draws.fork().sample(1000, 5), first)
    np.testing.assert_array_equal(NumpyDraws(5).sample(1000, 5), first)


def test_numpy_fork_with_generator_spawns_reproducible_children():
    a = NumpyDraws(np.random.default_rng(3))
    b = NumpyDraws(np.random.default_rng(3))
    a1, a2 = a.fork().sample(10**6, 3), a.fork().sample(10**6, 3)
    assert not np.array_equal(a1, a2)
    np.testing.assert_array_equal(b.fork().sample(10**6, 3), a1)


def test_replay_sample_returns_zero_based_values_and_checks_the_call():
    draws = ReplayDraws([DrawRecord(10, 2, np.array([3, 7]))])
    np.testing.assert_array_equal(draws.sample(10, 2), [2, 6])
    draws.assert_exhausted()
    with pytest.raises(ReplayMismatch):
        ReplayDraws([DrawRecord(10, 2, np.array([3, 7]))]).sample(9, 2)


def test_replay_fork_shares_the_recorded_stream():
    draws = ReplayDraws([])
    assert draws.fork() is draws


def test_both_implementations_satisfy_the_protocol():
    assert isinstance(NumpyDraws(1), Draws)
    assert isinstance(ReplayDraws([]), Draws)


# Each draw source and its module in datanuggets._random. datanuggets._draws, their old import
# path, forwards to the same objects, so scripts that import it keep working, and so do pickles
# that name it, since unpickling looks a class up by module and name.
RANDOM_NAMES = {
    "Draws": "interfaces",
    "NumpyDraws": "numpy_draws",
    "RDraws": "r_draws",
    "r_use_hash": "r_draws",
    "DrawRecord": "replay",
    "ReplayDraws": "replay",
    "ReplayMismatch": "replay",
    "as_draws": "seeds",
}


def test_draw_sources_live_in_random_and_their_old_path_forwards():
    old = importlib.import_module("datanuggets._draws")
    for name, module in RANDOM_NAMES.items():
        defined = getattr(importlib.import_module(f"datanuggets._random.{module}"), name)
        assert defined.__module__ == f"datanuggets._random.{module}", name
        assert getattr(old, name) is defined, name
    assert sorted(old.__all__) == sorted(RANDOM_NAMES)


def test_a_pickle_naming_the_old_path_loads():
    # protocol 0 writes each class as "module\nname" text, so the old path can be put back in.
    # The bytes loaded are the ones this test just made.
    draws = NumpyDraws(7)
    new = pickle.dumps(draws, protocol=0)
    assert b"datanuggets._random.numpy_draws\nNumpyDraws" in new
    restored = pickle.loads(new.replace(b"datanuggets._random.numpy_draws", b"datanuggets._draws"))
    assert type(restored) is NumpyDraws
    np.testing.assert_array_equal(restored.permutation(20), NumpyDraws(7).permutation(20))
