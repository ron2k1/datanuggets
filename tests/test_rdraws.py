"""RDraws against R 4.6.1's set.seed() and sample.int(), recorded by r/fixtures_rsample.R with
.Random.seed around every call. Every case calls the sampler directly, so the switch to R's hash
sampler above n = 1e7 is checked without a large dataset."""

import itertools
import math
import warnings

import numpy as np
import pytest

from datanuggets import DataNuggetWarning, create_dn, refine_dn
from datanuggets._random.numpy_draws import NumpyDraws
from datanuggets._random.r_draws import RDraws, r_use_hash
from datanuggets._random.replay import ReplayDraws
from datanuggets._random.seeds import as_draws
from rfixtures import create_args, load_group, refine_inputs

GROUP = {c.id: c for c in load_group("rsample")}
CREATE = {c.id: c for c in load_group("create_dn")}
SEEDS = [c for c in GROUP.values() if c.id.startswith("seed_") and c.id != "seed_invalid"]
SWITCH = [c for c in GROUP.values() if c.id.startswith("switch_")]
SIZE = [c for c in GROUP.values() if c.id.startswith("size_")]
REFINE = {c.id: c for c in load_group("refine_dn")}
RULE = "(n > 1e+07 && !replace && is.null(prob) && size <= n/2)"
SAMPLE_INT = [
    "{",
    "    stopifnot(length(n) == 1L)",
    "    if (useHash) {",
    "        stopifnot(is.null(prob), !replace)",
    "        .Internal(sample2(n, size))",
    "    }",
    "    else .Internal(sample(n, size, replace, prob))",
    "}",
]


def by_id(cases):
    return pytest.mark.parametrize("case", cases, ids=[c.id for c in cases])


def ints(case, name):
    return case.array(name).astype(np.int64).tolist()


def test_every_case_is_recorded():
    assert {c.id for c in SWITCH} == {
        "switch_2p24p1_size2",
        "switch_2p24p1_size1",
        "switch_1e7_size1200",
        "switch_1e7p1_size1200",
    }
    assert {c.id for c in SIZE} == {"size_5000001_of_10000002", "size_5000002_of_10000002"}
    assert len(SEEDS) == 12 and "use_hash_table" in GROUP and "seed_invalid" in GROUP
    assert "bits_2p49p1" in GROUP
    # the cases tell the branches apart where they should
    differ = {c.id for c in SWITCH if c.manifest["branches_differ"]}
    assert differ == {"switch_2p24p1_size2", "switch_1e7_size1200", "switch_1e7p1_size1200"}
    for c in SWITCH:
        assert (ints(c, "first") != ints(c, "other")) == c.manifest["branches_differ"]
    hashed, pooled = (GROUP[f"size_{s}_of_10000002"] for s in (5000001, 5000002))
    split = hashed.manifest["first_split"]
    assert ints(hashed, "head")[: split - 1] == ints(pooled, "head")[: split - 1]
    assert ints(hashed, "head")[split - 1] != ints(pooled, "head")[split - 1]


@by_id(list(GROUP.values()))
def test_reference_is_pinned(case):
    versions = case.manifest["versions"]
    assert versions["R"].startswith("R version 4.6.1 ")
    assert versions["platform"] == "x86_64-w64-mingw32"
    assert versions["datanugget"] == "1.5.0"
    rng = case.manifest["rng"]
    assert rng["kind"] == ["Mersenne-Twister", "Inversion", "Rejection"]
    assert rng["use_hash"] == RULE
    assert rng["sample_int"] == SAMPLE_INT


@by_id(SEEDS)
def test_seed_state_and_draws_match_r(case):
    draws = RDraws(case.args["seed"])
    got = draws.random_seed()
    assert got.dtype == np.int32 and got.shape == (626,)
    assert got.tolist() == ints(case, "state0")
    assert (draws.sample(10, 10) + 1).tolist() == ints(case, "first")
    assert draws.random_seed().tolist() == ints(case, "state1")


def test_invalid_seeds_stop_like_r():
    case = GROUP["seed_invalid"]
    for seed, error, warned in zip(
        case.array("seeds").tolist(),
        case.manifest["errors"],
        case.manifest["warnings"],
        strict=True,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match=f"^{error}$"):
                RDraws(seed)
        assert [str(w.message) for w in caught] == warned, seed
        assert all(w.category is DataNuggetWarning for w in caught)


def test_integer_seeds_take_the_same_path_as_r_doubles():
    # set.seed's asInteger stops at 2^31 and -2^31, whose int32 is NA, and keeps +-(2^31 - 1).
    # R records these as doubles, and a Python int or numpy integer at the same value must agree.
    case = GROUP["seed_invalid"]
    recorded = dict(
        zip(
            case.array("seeds").tolist(),
            zip(case.manifest["errors"], case.manifest["warnings"], strict=True),
            strict=True,
        )
    )
    for value in (2.0**31, -(2.0**31)):
        error, warned = recorded[value]
        for seed in (int(value), np.int64(value)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(ValueError, match=f"^{error}$"):
                    RDraws(seed)
            assert [str(w.message) for w in caught] == warned, seed
    for c in (GROUP["seed_intmax"], GROUP["seed_neg_intmax"]):
        for seed in (int(c.args["seed"]), np.int64(c.args["seed"])):
            assert RDraws(seed).random_seed().tolist() == ints(c, "state0"), seed


@pytest.mark.parametrize("seed", [True, "12", None])
def test_seeds_r_coerces_are_refused(seed):
    # R's set.seed(TRUE) and set.seed("12") seed with 1 and 12, and set.seed(NULL) picks a seed
    # from the clock. RDraws refuses all three.
    with pytest.raises(TypeError, match="seed must be a real number"):
        RDraws(seed)


@by_id(SWITCH)
def test_switch_matches_r_and_the_draws_after_it(case):
    n, size = case.args["n"], case.args["size"]
    assert r_use_hash(n, size) == case.manifest["use_hash"]
    draws = RDraws(case.args["seed"])
    assert draws.random_seed().tolist() == ints(case, "state0")
    assert (draws.sample(n, size) + 1).tolist() == ints(case, "first")
    assert draws.random_seed().tolist() == ints(case, "state1")
    assert (draws.sample(n, size) + 1).tolist() == ints(case, "second")
    assert draws.random_seed().tolist() == ints(case, "state2")
    assert (draws.sample(10, 10) + 1).tolist() == ints(case, "third")
    assert draws.random_seed().tolist() == ints(case, "state3")


@by_id(SWITCH)
def test_the_other_branch_matches_r(case):
    n, size = case.args["n"], case.args["size"]
    draws = RDraws(case.args["seed"])
    other = draws._values(n, size, not case.manifest["use_hash"])
    assert [v + 1 for v in other] == ints(case, "other")


@by_id(SIZE)
def test_size_condition_on_the_sampler(case):
    n, size = case.args["n"], case.args["size"]
    assert r_use_hash(n, size) == case.manifest["use_hash"]
    head = ints(case, "head")
    got = RDraws(case.args["seed"])._values(n, size, r_use_hash(n, size))
    assert [v + 1 for v in itertools.islice(got, len(head))] == head


def test_bit_count_comes_from_log2_like_r():
    # R_unif_index draws ceil(log2(n)) bits. At n = 2^49 + 1 that is 49, though n - 1 = 2^49
    # needs 50, so R never draws the last value and an exact bit count draws something else.
    case = GROUP["bits_2p49p1"]
    n, size = int(case.args["n"]), case.args["size"]
    assert math.ceil(math.log2(n)) == 49 and (n - 1).bit_length() == 50
    assert r_use_hash(n, size) and case.manifest["use_hash"]
    draws = RDraws(case.args["seed"])
    assert draws.random_seed().tolist() == ints(case, "state0")
    assert (draws.sample(n, size) + 1).tolist() == ints(case, "first")
    assert draws.random_seed().tolist() == ints(case, "state1")


def test_use_hash_matches_r_across_the_boundaries():
    case = GROUP["use_hash_table"]
    rows = list(zip(ints(case, "n"), ints(case, "size"), ints(case, "use_hash"), strict=True))
    assert [r_use_hash(n, size) for n, size, _ in rows] == [bool(h) for _, _, h in rows]
    assert {h for _, _, h in rows} == {0, 1}
    # datanugget never samples with replacement or weights, and either turns R's hash off
    assert not any(ints(case, "with_replace")) and not any(ints(case, "with_prob"))


def test_hash_keeps_the_value_after_100_repeats():
    # unique.c's do_sample2 tries a position at most 100 times and then keeps the repeat
    draws, calls = RDraws(1), []

    def always_seven(n):
        calls.append(n)
        return 7

    draws._unif_index = always_seven
    assert list(draws._hash(2 * 10**7, 2)) == [7, 7]
    assert len(calls) == 101


@pytest.mark.parametrize(
    ("n", "size", "message"),
    [
        (3, 4, "cannot take a sample larger than the population when 'replace = FALSE'"),
        (3, -1, "invalid 'size' argument"),
        (-1, 0, "invalid first argument"),
        (0, 1, "invalid first argument"),
    ],
)
def test_sample_stops_where_r_stops(n, size, message):
    with pytest.raises(ValueError, match=message):
        RDraws(1).sample(n, size)


def test_fork_restarts_the_seed_like_set_seed():
    draws = RDraws(55)
    first = draws.fork().sample(1000, 5)
    draws.sample(1000, 5)
    assert draws.fork().sample(1000, 5).tolist() == first.tolist()


def test_every_recorded_create_dn_draw():
    count = 0
    for case in load_group("create_dn"):
        draws = RDraws(case.args["seed"])
        for rec in case.draws():
            assert (draws.sample(rec.n, rec.size) + 1).tolist() == rec.value.tolist(), case.id
            count += 1
    assert count > 400


@by_id(list(REFINE.values()))
def test_refine_dn_with_rdraws_matches_r(case):
    x, dn, kwargs = refine_inputs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        got = refine_dn(x, dn, seed=RDraws(case.args.get("seed", 291102)), **kwargs)
    np.testing.assert_array_equal(got.assignments + 1, case.array("assignments"))
    np.testing.assert_array_equal(got.weights, case.array("weights"))
    np.testing.assert_array_equal(got.centers, case.array("centers"))
    np.testing.assert_array_equal(got.scales, case.array("scales"))


def test_rdraws_splits_in_parallel_like_serial():
    x, dn, kwargs = refine_inputs(REFINE["faithful_ev0.9_ms5_mns2_shape_d2_ns25"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        one = refine_dn(x, dn, seed=RDraws(291102), n_jobs=1, **kwargs)
        two = refine_dn(x, dn, seed=RDraws(291102), n_jobs=2, **kwargs)
    np.testing.assert_array_equal(one.assignments, two.assignments)
    np.testing.assert_array_equal(one.centers, two.centers)


def test_create_dn_reseeds_an_rdraws_like_set_seed():
    # create.DN calls set.seed(seed) before it draws, so one RDraws passed to two calls gives
    # R's result both times
    case = CREATE["iris_euclidean_random"]
    draws = RDraws(case.args["seed"])
    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataNuggetWarning)
            dn = create_dn(case.array("x"), **create_args(case), seed=draws)
        np.testing.assert_array_equal(dn.assignments + 1, case.array("assignments"))
        np.testing.assert_array_equal(dn.weights, case.array("weights"))
        np.testing.assert_array_equal(dn.centers, case.array("centers"))
        np.testing.assert_array_equal(dn.scales, case.array("scales"))
    # the caller's RDraws is left where it was, and other Draws pass through as they are
    assert draws.random_seed().tolist() == RDraws(case.args["seed"]).random_seed().tolist()
    replay = ReplayDraws([])
    assert as_draws(replay) is replay


def test_numpy_stays_the_default():
    assert isinstance(as_draws(5), NumpyDraws)
    assert isinstance(as_draws(np.random.default_rng(5)), NumpyDraws)
