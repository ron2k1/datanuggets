"""hull_gate and lymphmacro_finder: the events of each population, inside the convex hull of the
events whose bin lies in the hull of the population's area."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from datanuggets._defaults import (
    FINDER_N_COL,
    FINDER_N_ROW,
    FINDER_T11,
    FINDER_T12,
    FINDER_T21,
    FINDER_T22,
)
from datanuggets.flow.grid import SQRT_SHIFT, channel, f_plot, sqrt_coords
from datanuggets.flow.peaks import find_area, local_maxima, refine_peaks

MSIZE = 100  # cfc.py's grid, 100 by 100 bins
GRID_SCALE = "l"  # its cells hold log(1 + count)
GRID_ALPHA = 0.1  # cells below 0.1 of its maximum go to 0: f_plot's default, implicit in cfc.py
MIN_POINTS = 3  # cfc.py's guard: a hull needs more than 3 cells, and more than 3 kept events


class FinderPeaks(NamedTuple):
    """The finder's view of a file: density, its log(1 + count) grid with SSC-A bins as rows and
    FSC-A bins as columns, cells below cfc.py's 0.1 of its maximum at 0; candidates, the 1-based
    (row, col) of every local maximum; peaks, the refined peaks in the finder's order, so peak k
    seeds population k; and fsc and ssc, the raw channel value at each column and row."""

    density: np.ndarray
    candidates: np.ndarray
    peaks: np.ndarray
    fsc: np.ndarray
    ssc: np.ndarray


def _inside(vertices: np.ndarray, q: np.ndarray) -> np.ndarray:
    """cfc.py's test, Delaunay(...).find_simplex(q) >= 0, on a triangulation of the hull's
    vertices."""
    return Delaunay(vertices).find_simplex(q) >= 0


def _bins(zz) -> np.ndarray:
    """zz as f_plot gives it: one (x bin, y bin) row per event, integers counting from 1. A 0 bin
    would read the grid's last cell, and silently."""
    zz = np.asarray(zz)
    if zz.ndim != 2 or zz.shape[1] != 2:
        raise ValueError(f"zz must have two columns, an (x bin, y bin) per event, got {zz.shape}")
    if not np.issubdtype(zz.dtype, np.integer):
        raise ValueError(f"zz must be integer bins, got {zz.dtype}")
    if len(zz) and zz.min() < 1:
        raise ValueError("zz must count its bins from 1, as f_plot does")
    return zz


def bin_mask(zz, cells) -> np.ndarray:
    """The events whose bin is inside the convex hull of the cells. Each grid cell is tested once
    and looked up per event, since every event in a bin has the bin's coordinates."""
    zz = _bins(zz)
    cells = np.asarray(cells)
    if len(cells) <= MIN_POINTS:
        return np.zeros(len(zz), dtype=bool)
    try:
        hull = ConvexHull(cells)
    except QhullError:  # cells on one line: cfc.py's ConvexHull raises here
        return np.zeros(len(zz), dtype=bool)
    nr, nc = int(zz[:, 0].max()), int(zz[:, 1].max())
    a, b = np.indices((nr, nc)) + 1
    inside = _inside(cells[hull.vertices], np.column_stack((a.ravel(), b.ravel())))
    return inside.reshape(nr, nc)[zz[:, 0] - 1, zz[:, 1] - 1]


def hull_gate(fsc_a, ssc_a, zz, cells) -> np.ndarray:
    """cfc.py's gate for one population: the events inside the convex hull of the events bin_mask
    keeps, tested against a triangulation of that hull's vertices rather than of every kept
    event. Empty when the cells or the kept events can't make a hull."""
    raw = np.column_stack((np.asarray(fsc_a, dtype=float), np.asarray(ssc_a, dtype=float)))
    if len(_bins(zz)) != len(raw):
        raise ValueError(f"zz must have one row per event, {len(raw)}, got {len(zz)}")
    points = raw[bin_mask(zz, cells)]
    if len(points) <= MIN_POINTS:
        return np.zeros(len(raw), dtype=bool)
    try:
        return _inside(points[ConvexHull(points).vertices], raw)
    except QhullError:  # points on or within rounding of one line: cfc.py's Delaunay raises here
        return np.zeros(len(raw), dtype=bool)


def lymphmacro_finder(
    fsc_a,
    ssc_a,
    *,
    n_row=FINDER_N_ROW,
    n_col=FINDER_N_COL,
    t11=FINDER_T11,
    t21=FINDER_T21,
    t12=FINDER_T12,
    t22=FINDER_T22,
    area=find_area,
):
    """Cabrera's gate finder: one bool column per population, one row per event, populations in
    cfc.py's order. The first peak's area takes t11 and t21, the others t12 and t22. area(x, i, j,
    t1, t2) gives a peak's cells: find_area, the disc on the peak, or mirrored_area, cfc.py's
    find_area as written. No peak left after cfc.py's filters gives zero columns, where cfc.py
    raises IndexError."""
    return _finder(fsc_a, ssc_a, n_row, n_col, t11, t21, t12, t22, area)


def _finder(fsc_a, ssc_a, n_row, n_col, t11, t21, t12, t22, area) -> np.ndarray:
    return _masks(_grid_peaks(fsc_a, ssc_a, n_row, n_col), t11, t21, t12, t22, area)


def finder_peaks(fsc_a, ssc_a, *, n_row=FINDER_N_ROW, n_col=FINDER_N_COL) -> FinderPeaks:
    """The density grid lymphmacro_finder gates on, its local maxima and the peaks it keeps, for
    drawing with plot.peaks. n_row and n_col are the finder's focal window."""
    return _view(_grid_peaks(fsc_a, ssc_a, n_row, n_col))


def gate_peaks(
    fsc_a,
    ssc_a,
    *,
    n_row=FINDER_N_ROW,
    n_col=FINDER_N_COL,
    t11=FINDER_T11,
    t21=FINDER_T21,
    t12=FINDER_T12,
    t22=FINDER_T22,
    area=find_area,
) -> tuple[np.ndarray, FinderPeaks]:
    """lymphmacro_finder's masks and finder_peaks' view of the file, from one density grid. A file
    that is gated and drawn needs both, and each finds the same grid, local maxima and peaks, so
    this finds them once. It takes lymphmacro_finder's arguments."""
    found = _grid_peaks(fsc_a, ssc_a, n_row, n_col)
    return _masks(found, t11, t21, t12, t22, area), _view(found)


def _masks(found, t11, t21, t12, t22, area) -> np.ndarray:
    """One column per peak of _grid_peaks' result: the events of the area grown from it, with t11
    and t21 for the first peak and t12 and t22 for the others."""
    fsc_a, ssc_a, grid, x, _, peaks = found
    masks = np.zeros((len(fsc_a), len(peaks)), dtype=bool)
    for k, (i, j) in enumerate(peaks):
        t1, t2 = (t11, t21) if k == 0 else (t12, t22)
        masks[:, k] = hull_gate(fsc_a, ssc_a, grid.zz, area(x, i, j, t1, t2))
    return masks


def _view(found) -> FinderPeaks:
    """_grid_peaks' result as FinderPeaks, the grid's positions in raw channel units."""
    fsc_a, ssc_a, grid, x, candidates, peaks = found
    shift = min(ssc_a.min(), fsc_a.min()) - SQRT_SHIFT  # sqrt_coords' shift, undone
    return FinderPeaks(x, candidates, peaks, grid.y**2 + shift, grid.x**2 + shift)


def _grid_peaks(fsc_a, ssc_a, n_row, n_col):
    """cfc.py's grid of sqrt SSC-A (rows) by sqrt FSC-A (columns), its local maxima and the
    refined peaks, with the checked channels and the grid itself for the gates."""
    fsc_a, ssc_a = channel("fsc_a", fsc_a), channel("ssc_a", ssc_a)
    if len(fsc_a) != len(ssc_a):
        raise ValueError(
            f"fsc_a and ssc_a must have the same length, got {len(fsc_a)} and {len(ssc_a)}"
        )
    grid = f_plot(
        *sqrt_coords(fsc_a, ssc_a), nr=MSIZE, nc=MSIZE, scale=GRID_SCALE, alpha=GRID_ALPHA
    )
    x = grid.z.T
    candidates = local_maxima(x, n_row=n_row, n_col=n_col)
    return fsc_a, ssc_a, grid, x, candidates, refine_peaks(x, candidates)
