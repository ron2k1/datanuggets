"""gates: his figure as Line2D markers, his sizes, alphas and colors. peaks: the grid, its\nlocal maxima and the kept peaks labelled with their Pop numbers."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from datanuggets.flow import FinderPeaks
from datanuggets.plot import gates, peaks


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


def test_gates_takes_the_axes_cycle_and_edge_width_his_scatter_takes():
    fsc, ssc, masks = _data()
    with mpl.rc_context({"patch.linewidth": 2.5}):
        fig, (his, ours) = plt.subplots(1, 2)
        for ax in (his, ours):
            ax.set_prop_cycle(color=["k", "m"])
        his.scatter(fsc, ssc, s=0.5, c="gray", alpha=0.3)  # his calls, as his script makes them
        want = [his.scatter(fsc[masks[:, j]], ssc[masks[:, j]], s=5, alpha=0.8) for j in (0, 2)]
        gates(fsc, ssc, masks, ax=ours)
    _, *pops = ours.get_lines()
    assert [to_hex(p.get_color()) for p in pops] == [to_hex(w.get_facecolor()[0]) for w in want]
    assert [w.get_linewidths()[0] for w in want] == [2.5, 2.5]
    assert [line.get_markeredgewidth() for line in ours.get_lines()] == [2.5, 2.5, 2.5]
    plt.close(fig)


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


def _peaks():
    density = np.zeros((20, 30))
    density[5, 8], density[12, 20], density[3, 25] = 3.0, 2.0, 1.0
    return FinderPeaks(
        density=density,
        candidates=np.array([[6, 9], [13, 21], [4, 26]]),
        peaks=np.array([[6, 9], [13, 21]]),
        fsc=np.linspace(1e5, 4e6, 30),
        ssc=np.linspace(2e4, 4e6, 20),
    )


def test_peaks_draws_the_grid_maxima_and_numbered_peaks():
    fp = _peaks()
    fig, ax = plt.subplots()
    assert peaks(fp, title="File 1", ax=ax) is ax
    (im,) = ax.get_images()
    assert im.get_array().shape == fp.density.shape
    assert im.get_array().mask[0, 0]  # empty bins are left blank
    cand, kept = ax.get_lines()
    assert (cand.get_gid(), kept.get_gid()) == ("candidates", "peaks")
    assert np.array_equal(cand.get_xdata(), fp.candidates[:, 1])
    assert np.array_equal(cand.get_ydata(), fp.candidates[:, 0])
    assert np.array_equal(kept.get_xdata(), fp.peaks[:, 1])
    assert [t.get_text() for t in ax.texts] == ["Pop 1", "Pop 2"]
    assert [t.xy for t in ax.texts] == [(9, 6), (21, 13)]
    assert ax.get_title() == "File 1"
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["800k", "1.6M", "2.4M", "3.2M", "4M"]
    assert ax.get_xticks()[-1] == pytest.approx(30)  # 4e6 is the last column's value
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["Local maxima (3)", "Kept peaks (2)"]
    plt.close(fig)


def test_peaks_with_no_peak_draws_the_grid():
    fp = _peaks()._replace(candidates=np.empty((0, 2), int), peaks=np.empty((0, 2), int))
    fig, ax = plt.subplots()
    peaks(fp, ax=ax)
    assert not ax.texts
    assert [len(line.get_xdata()) for line in ax.get_lines()] == [0, 0]
    plt.close(fig)


def test_peaks_rejects_raw_values_that_do_not_fit_the_grid():
    with pytest.raises(ValueError, match="per column and row"):
        peaks(_peaks()._replace(fsc=np.arange(5.0)))
