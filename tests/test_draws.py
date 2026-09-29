import numpy as np
import pytest

from datanuggets._draws import (
    DrawRecord,
    NumpyDraws,
    ReplayDraws,
    ReplayMismatch,
    as_draws,
)
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
