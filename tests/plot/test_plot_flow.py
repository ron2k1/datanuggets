"""gates: his figure as Line2D markers, his sizes, alphas and colors."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from datanuggets.plot import gates


def _data():
    rng = np.random.default_rng(0)
    fsc, ssc = rng.uniform(0, 10, 200), rng.uniform(0, 10, 200)
    masks = np.zeros((200, 3), dtype=bool)
    masks[:50, 0] = True
    masks[100:120, 2] = True  # population 2 is empty
    return fsc, ssc, masks


def test_gates_draws_the_events_then_each_population():
    fsc, ssc, masks = _data()
    ax = gates(fsc, ssc, masks, title="Sample 1: FSC-A vs SSC-A")
    events, *pops = ax.get_lines()
    assert events.get_gid() == "events"
    assert [p.get_gid() for p in pops] == ["population", "population"]
    assert np.array_equal(events.get_xdata(), fsc)
    assert np.array_equal(events.get_ydata(), ssc)
    assert np.array_equal(pops[1].get_xdata(), fsc[masks[:, 2]])
    assert np.array_equal(pops[1].get_ydata(), ssc[masks[:, 2]])
    assert events.get_markersize() == pytest.approx(0.5**0.5)
    assert pops[0].get_markersize() == pytest.approx(5**0.5)
    assert (events.get_alpha(), pops[0].get_alpha()) == (0.3, 0.8)
    assert events.get_color() == "gray"
    assert all(line.get_linestyle() == "None" for line in ax.get_lines())
    assert all(line.get_rasterized() for line in ax.get_lines())
    assert ax.get_title() == "Sample 1: FSC-A vs SSC-A"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("FSC-A", "SSC-A")
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["Unlabeled", "Pop 1", "Pop 3"]


def test_gates_skips_empty_populations_in_the_color_cycle():
    fsc, ssc, masks = _data()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _, *pops = gates(fsc, ssc, masks).get_lines()
    assert [p.get_color() for p in pops] == cycle[:2]  # Pop 3 takes the second color, as in his


def test_gates_with_no_population_draws_the_events():
    fsc, ssc, _ = _data()
    ax = gates(fsc, ssc, np.zeros((200, 0), dtype=bool))
    assert [line.get_gid() for line in ax.get_lines()] == ["events"]
    assert ax.get_title() == ""


def test_gates_draws_on_the_axes_given():
    fsc, ssc, masks = _data()
    _, (a, b) = plt.subplots(1, 2)
    assert gates(fsc, ssc, masks, ax=b) is b
    assert not a.get_lines()


@pytest.mark.parametrize(
    ("masks", "match"),
    [
        (np.zeros((199, 2), dtype=bool), "one row per event"),
        (np.zeros((200, 2)), "bool"),
        (np.zeros(200, dtype=bool), "bool"),
    ],
)
def test_gates_rejects_bad_masks(masks, match):
    fsc, ssc, _ = _data()
    with pytest.raises(ValueError, match=match):
        gates(fsc, ssc, masks)


def test_gates_rejects_channels_of_different_lengths():
    fsc, ssc, masks = _data()
    with pytest.raises(ValueError, match="same length"):
        gates(fsc, ssc[:-1], masks)
