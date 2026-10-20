"""local_maxima, search_ind and refine_peaks against his candidates and peaks, and against
generic_filter with his focal rule on random grids."""

import numpy as np
import pytest
from scipy.ndimage import generic_filter

from datanuggets.flow import local_maxima, refine_peaks, search_ind
from flowcases import NAMES, load_case


@pytest.mark.parametrize("name", NAMES)
def test_local_maxima_equal_his_candidates(name):
    c = load_case(name)
    assert np.array_equal(local_maxima(c.z.T), c.candidates)


@pytest.mark.parametrize("name", NAMES)
def test_refine_peaks_equal_his_peaks(name):
    c = load_case(name)
    assert np.array_equal(refine_peaks(c.z.T, c.candidates), c.peaks)


@pytest.mark.parametrize("seed", range(20))
def test_local_maxima_equal_his_focal_rule(seed):
    rng = np.random.default_rng(seed)
    x = np.round(rng.gamma(1.0, size=(37, 41)), 1)  # rounded, so windows have ties
    q = np.quantile(x, 0.85)
    local = generic_filter(x, lambda w: w.max() if w.max() > q else -1, size=(11, 11))
    assert np.array_equal(local_maxima(x), np.argwhere(x == local) + 1)


def test_search_ind_takes_the_first_maximum_in_row_order():
    x = np.zeros((9, 9))
    x[3, 5] = x[4, 2] = 7.0  # 1-based (4, 6) and (5, 3), both in the window around (5, 5)
    assert search_ind(x, 5, 5).tolist() == [4, 6]


def test_search_ind_clips_the_window_at_the_edges():
    x = np.zeros((9, 9))
    x[0, 0] = 1.0
    assert search_ind(x, 1, 2).tolist() == [1, 1]
    assert search_ind(x, 9, 9).tolist() == [7, 7]  # an all-zero window gives its first cell


def test_refine_peaks_filters_and_sorts_as_he_does():
    x = np.zeros((100, 100))
    cells = [(50, 50, 5.0), (20, 30, 4.0), (95, 40, 9.0), (60, 95, 9.0), (5, 5, 9.0)]
    for i, j, v in cells:
        x[i - 1, j - 1] = v
    peaks = refine_peaks(x, np.array([(i, j) for i, j, _ in cells]))
    assert peaks.tolist() == [[20, 30], [50, 50]]


def test_refine_peaks_limits_sit_where_his_do():
    x = np.zeros((100, 100))
    # his limits are row > 10, row < 90 and column < 90, with no lower limit on the column
    cells = [(10, 50), (11, 20), (89, 50), (90, 20), (50, 89), (40, 90), (30, 1)]
    for i, j in cells:
        x[i - 1, j - 1] = 9.0
    peaks = refine_peaks(x, np.array(cells))
    assert peaks.tolist() == [[11, 20], [30, 1], [50, 89], [89, 50]]


def test_refine_peaks_with_no_candidate_is_empty():
    assert refine_peaks(np.ones((100, 100)), np.empty((0, 2), dtype=int)).shape == (0, 2)
