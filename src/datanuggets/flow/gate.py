"""hull_gate and lymphmacro_finder: the events of each population, inside the convex hull of the
events whose bin lies in the hull of the population's area."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from datanuggets.flow.grid import channel, f_plot, sqrt_coords
from datanuggets.flow.peaks import find_area, local_maxima, refine_peaks

MSIZE = 100  # his grid, 100 by 100 bins
MIN_POINTS = 3  # his guard: a hull needs more than 3 cells, and more than 3 kept events


def _inside(vertices: np.ndarray, q: np.ndarray) -> np.ndarray:
    """His test, Delaunay(...).find_simplex(q) >= 0, on a triangulation of the hull's vertices."""
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
    except QhullError:  # cells on one line: his ConvexHull raises here
        return np.zeros(len(zz), dtype=bool)
    nr, nc = int(zz[:, 0].max()), int(zz[:, 1].max())
    a, b = np.indices((nr, nc)) + 1
    inside = _inside(cells[hull.vertices], np.column_stack((a.ravel(), b.ravel())))
    return inside.reshape(nr, nc)[zz[:, 0] - 1, zz[:, 1] - 1]


def hull_gate(fsc_a, ssc_a, zz, cells) -> np.ndarray:
    """His gate for one population: the events inside the convex hull of the events bin_mask
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
    except QhullError:  # kept events on or within rounding of one line: his Delaunay raises here
        return np.zeros(len(raw), dtype=bool)


def lymphmacro_finder(fsc_a, ssc_a, *, n_row=11, n_col=11, t11=22, t21=0.25, t12=20, t22=0.5):
    """His gate finder: one bool column per population, one row per event, populations in his
    order. The first peak's area takes t11 and t21, the others t12 and t22. No peak left after
    his filters gives zero columns, where his code raises IndexError."""
    return _finder(fsc_a, ssc_a, n_row, n_col, t11, t21, t12, t22, find_area)


def _finder(fsc_a, ssc_a, n_row, n_col, t11, t21, t12, t22, area) -> np.ndarray:
    fsc_a, ssc_a = channel("fsc_a", fsc_a), channel("ssc_a", ssc_a)
    if len(fsc_a) != len(ssc_a):
        raise ValueError(
            f"fsc_a and ssc_a must have the same length, got {len(fsc_a)} and {len(ssc_a)}"
        )
    grid = f_plot(*sqrt_coords(fsc_a, ssc_a), nr=MSIZE, nc=MSIZE, scale="l")
    x = grid.z.T
    peaks = refine_peaks(x, local_maxima(x, n_row=n_row, n_col=n_col))
    masks = np.zeros((len(fsc_a), len(peaks)), dtype=bool)
    for k, (i, j) in enumerate(peaks):
        t1, t2 = (t11, t21) if k == 0 else (t12, t22)
        masks[:, k] = hull_gate(fsc_a, ssc_a, grid.zz, area(x, i, j, t1, t2))
    return masks
