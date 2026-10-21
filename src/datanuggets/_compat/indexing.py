# R parallel::splitIndices adaptation; upstream notices are in THIRD_PARTY.md.
# Copyright (C) 1995-2025 The R Core Team; derived from snow by Luke Tierney.
# Python adaptation by Ronil Basu, recorded 2026-10-21.
# GPL version 2; supplied without warranty. See LICENSE and THIRD_PARTY.md.

"""R's index helpers the ported code depends on, 0-based: parallel::splitIndices and the rows
unique() keeps."""

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


def unique_first(x: np.ndarray) -> np.ndarray:
    """0-based positions of the rows R's unique(x) keeps from a numeric matrix: the first
    occurrence of each distinct row, in row order. R 4.x compares the rows exactly, with
    0 equal to -0, and so do Python float tuples. Finite data only."""
    first: dict[tuple[float, ...], int] = {}
    for i, row in enumerate(np.asarray(x, dtype=np.float64).tolist()):
        first.setdefault(tuple(row), i)
    return np.fromiter(first.values(), dtype=np.intp, count=len(first))
