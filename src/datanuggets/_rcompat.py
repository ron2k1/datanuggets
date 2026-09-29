"""Exact re-implementations of base-R behavior the ported code depends on."""

from __future__ import annotations

import numpy as np


def _r_seq_length_out(start: float, stop: float, n: int) -> np.ndarray:
    """R's seq(start, stop, length.out = n): start + i * by, with the last value set to stop."""
    if n == 1:
        return np.array([start])
    by = (stop - start) / (n - 1)
    out = start + np.arange(n, dtype=np.float64) * by
    out[-1] = stop
    return out


def split_indices(nx: int, ncl: int) -> list[np.ndarray]:
    """R's parallel::splitIndices(nx, ncl), 0-based. Empty chunks are kept, as in R."""
    if ncl == 0:
        return []
    i = np.arange(1, nx + 1)
    if ncl == 1 or nx == 1:
        return [i - 1]
    fuzz = min((nx - 1) / 1000, 0.4 * nx / ncl)
    breaks = _r_seq_length_out(1 - fuzz, nx + fuzz, ncl + 1)
    bins = np.searchsorted(breaks, i, side="left") - 1  # cut(): right-closed (b_j, b_j+1]
    return [np.flatnonzero(bins == j) for j in range(ncl)]
