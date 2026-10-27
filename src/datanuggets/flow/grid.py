"""f_plot: Cabrera's density grid of two channels, its bins counted with one bincount."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datanuggets._validate import as_count, as_real, check_choice

SCALES = ("raw", "l", "d")  # cfc.py's counts, log(1 + z), log(1 + log(1 + z))
SQRT_SHIFT = 3  # cfc.py adds it to each channel, less the smaller minimum, before the square root


@dataclass(frozen=True)
class DensityGrid:
    """cfc.py's f_plot result. z is the nc by nr grid (its t(z)), x and y the nr and nc evenly
    spaced positions from each channel's minimum to its maximum, zz each event's 1-based (x bin,
    y bin)."""

    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    zz: np.ndarray


def channel(name: str, v) -> np.ndarray:
    """v as a float64 vector cfc.py's binning can use: finite, with more than one value."""
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} has NaN or infinite values")
    if a.min() == a.max():
        raise ValueError(f"{name} has one value, so it can't be binned")
    return a


def sqrt_coords(fsc_a, ssc_a) -> tuple[np.ndarray, np.ndarray]:
    """cfc.py's transform: SSC-A and FSC-A shifted by the smaller channel's minimum, plus
    SQRT_SHIFT, square rooted. x is SSC-A and y is FSC-A, as cfc.py has them."""
    min_val = min(np.min(ssc_a), np.min(fsc_a))
    return np.sqrt((-min_val) + SQRT_SHIFT + ssc_a), np.sqrt((-min_val) + SQRT_SHIFT + fsc_a)


def _bins(v: np.ndarray, n: int) -> np.ndarray:
    """cfc.py's bin of each value, floor(n (v - min) / (max - min)) + 1 clipped to 1 to n, in its
    order of operations so every bin is cfc.py's."""
    lo, hi = v.min(), v.max()
    return np.clip(np.floor(n * (v - lo) / (hi - lo)).astype(int) + 1, 1, n)


def f_plot(x, y, *, nr=20, nc=20, scale="raw", alpha=0.1) -> DensityGrid:
    """cfc.py's f_plot: the nr by nc table of the binned events, scaled, with cells below alpha
    times the maximum set to 0. Its padding for R's table() isn't needed by bincount, so it's
    gone."""
    x, y = channel("x", x), channel("y", y)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    nr, nc = as_count("nr", nr), as_count("nc", nc)
    check_choice("scale", scale, SCALES)
    alpha = as_real("alpha", alpha)
    zx, zy = _bins(x, nr), _bins(y, nc)
    z = np.bincount((zx - 1) * nc + (zy - 1), minlength=nr * nc).reshape(nr, nc).astype(float)
    if scale == "l":
        z = np.log1p(z)
    elif scale == "d":
        z = np.log1p(np.log1p(z))
    z[z < z.max() * alpha] = 0
    return DensityGrid(
        z=z.T,
        x=np.linspace(x.min(), x.max(), nr),
        y=np.linspace(y.min(), y.max(), nc),
        zz=np.column_stack((zx, zy)),
    )
