"""The peaks of the density grid: Cabrera's focal maxima, refined by search_ind and filtered the
way cfc.py filters them, and the cells around each: find_area, and mirrored_area, cfc.py's
find_area as written."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter

from datanuggets._defaults import FINDER_N_COL, FINDER_N_ROW

ROW_MIN = 10  # cfc.py's filters on a refined peak: row > 10, row < 90, column < 90
ROW_MAX = 90
COL_MAX = 90
PEAK_QUANTILE = 0.9  # a peak's density must be above this quantile of the grid
LOCAL_MAX_QUANTILE = 0.85  # a local maximum must be above this quantile of the grid
LOCAL_MAX_EDGES = "reflect"  # the focal window's edges: generic_filter's, which cfc.py uses
SEARCH_THRESHOLD = 2  # search_ind's half width, as cfc.py calls it
AREA_T1 = 35  # cfc.py's find_area defaults, never reached: the finder always passes t11 to t22
AREA_T2 = 0.85


def local_maxima(
    z, *, n_row=FINDER_N_ROW, n_col=FINDER_N_COL, quantile=LOCAL_MAX_QUANTILE
) -> np.ndarray:
    """cfc.py's focal step: the 1-based (row, col) of every cell equal to the maximum of the n_row
    by n_col window around it, when that maximum is above the quantile of z. maximum_filter's
    reflect edges are generic_filter's, so the windows are cfc.py's."""
    z = np.asarray(z, dtype=float)
    m = maximum_filter(z, size=(n_row, n_col), mode=LOCAL_MAX_EDGES)
    return np.argwhere(z == np.where(m > np.quantile(z, quantile), m, -1.0)) + 1


def search_ind(x, idx, idy, threshold=SEARCH_THRESHOLD) -> np.ndarray:
    """The 1-based cell of the largest x in the window of half width threshold around (idx, idy),
    clipped to the grid, the first in row order on a tie, as cfc.py's argmax takes it."""
    r, c = math.ceil(idx), math.ceil(idy)
    r0, r1 = max(r - threshold, 1), min(r + threshold, x.shape[0])
    c0, c1 = max(c - threshold, 1), min(c + threshold, x.shape[1])
    w = x[r0 - 1 : r1, c0 - 1 : c1]
    i, j = np.unravel_index(np.argmax(w), w.shape)
    return np.array([r0 + i, c0 + j])


def refine_peaks(x, candidates, *, threshold=SEARCH_THRESHOLD) -> np.ndarray:
    """cfc.py's peak step: search_ind from each candidate, np.unique, its row and column limits, a
    density above the 0.9 quantile, then np.argsort on the row. The same calls in its order,
    since unique's sort and argsort decide which peak comes first and gets t11 and t21."""
    x = np.asarray(x, dtype=float)
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.int64)
    p = np.unique(np.array([search_ind(x, i, j, threshold) for i, j in candidates]), axis=0)
    p = p[(p[:, 0] < ROW_MAX) & (p[:, 1] < COL_MAX) & (p[:, 0] > ROW_MIN)]
    p = p[x[p[:, 0] - 1, p[:, 1] - 1] > np.quantile(x, PEAK_QUANTILE)]
    return p[np.argsort(p[:, 0])]


def find_area(x, idx, idy, t1=AREA_T1, t2=AREA_T2) -> np.ndarray:
    """The 1-based cells (i, j) of x closer than t1 to the peak (idx, idy) with x[i, j] above t2
    times the peak's density, column by column as cfc.py's come. cfc.py's version reads each cell's
    density right but centres the distance on the mirrored peak (idy, idx), so off the diagonal
    it keeps the wrong cells. This one centres it on the peak."""
    x = np.asarray(x, dtype=float)
    di = np.arange(1, x.shape[0] + 1)[:, None] - idx
    dj = np.arange(1, x.shape[1] + 1)[None, :] - idy
    keep = (np.sqrt(di * di + dj * dj) < t1) & (x > t2 * x[int(idx) - 1, int(idy) - 1])
    j, i = np.nonzero(keep.T)
    return np.column_stack((i, j)) + 1


def mirrored_area(x, idx, idy, t1=AREA_T1, t2=AREA_T2) -> np.ndarray:
    """cfc.py's find_area as Cabrera wrote it: each cell's density read right, its distance
    measured from the mirrored peak (idy, idx). datanuggets-flow --disc his gates with it, so the
    populations and their numbers are the ones cfc.py's figures show; find_area measures from the
    peak itself, which is --disc peak."""
    x = np.asarray(x, dtype=float)
    rows, cols = x.shape
    ii, jj = np.meshgrid(np.arange(1, rows + 1), np.arange(1, cols + 1), indexing="ij")
    gr = np.column_stack([ii.ravel(), jj.ravel()])
    d = np.linalg.norm(gr - np.array([[idx, idy]]), axis=1)
    mask = (d < t1) & (x.flatten(order="F") > t2 * x[int(idx) - 1, int(idy) - 1])
    return gr[mask][:, [1, 0]]
