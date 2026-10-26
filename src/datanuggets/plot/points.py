"""clusters and nuggets: the scatter plots of WCluster's and datanugget's help examples."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array

from datanuggets._compat.arithmetic import r_quantile7, r_sum

# R's pch = 16 at cex 1, in points across: radius 0.375 times half the character height, which
# R's pdf device makes 1.2 times its 12 points, so 2.7 pt, as R's own PDF draws it
PCH16 = 5.4
NUGGET_CEX = 0.5  # the help examples' nugget size
CENTER_SIZE = 120.0  # star markers for cluster centers, points squared
PLAIN_SIZE = 20.0  # a point without a weight
# quantile's stop when the weight fractions have a NaN, from 0 / 0 or an infinite weight
QUANTILE_NAN_STOP = "missing values and NaN's not allowed if 'na.rm' is FALSE"


def _matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected a two-dimensional array, got {a.ndim} dimensions")
    return a


def _dims(dims, p: int) -> tuple[int, int]:
    d = tuple(int(v) for v in dims)
    if len(d) != 2 or d[0] == d[1] or not all(0 <= v < p for v in d):
        raise ValueError(f"dims must be two distinct columns from 0 to {p - 1}, got {dims}")
    return d[0], d[1]


def _labels(labels, n: int) -> np.ndarray:
    a = np.asarray(labels)
    if a.shape != (n,) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"labels must be {n} integers, one per point")
    return a


def _palette(colors) -> list:
    """The colors labels index into, the color cycle by default. Checks every one before a view
    draws, so a bad one can't leave a half-drawn plot."""
    if colors is None:
        return plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if isinstance(colors, str):
        raise TypeError("colors must be a sequence of colors, not one string")
    palette = list(colors)
    if not palette:
        raise ValueError("colors must hold at least one color")
    try:
        to_rgba_array(palette)
    except (TypeError, ValueError) as e:
        raise ValueError(f"colors must all be matplotlib colors: {e}") from e
    return palette


def _label_colors(labels, palette) -> list:
    return [palette[int(v) % len(palette)] for v in labels]


def _sizes(w, n: int, size: str) -> np.ndarray:
    """Marker areas in points squared: proportional to weight ("area", 6 to 406) or to its
    square root ("sqrt", 6 to 406, so heavy points dominate less), or R's cex = log(w) for
    pch = 16 ("log"), where a log that isn't positive draws nothing."""
    if size not in ("area", "sqrt", "log"):
        raise ValueError(f"size must be 'area', 'sqrt' or 'log', got {size!r}")
    if w is None:
        if size == "log":
            raise ValueError("size='log' needs w")
        return np.full(n, PLAIN_SIZE)
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"w has {w.size} entries for {n} points")
    if size == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            cex = np.log(w)
        return np.where(cex > 0, (PCH16 * cex) ** 2, 0.0)
    if not (np.all(np.isfinite(w)) and np.all(w >= 0) and w.max() > 0):
        raise ValueError(f"size={size!r} needs finite weights, none negative and not all zero")
    if size == "sqrt":
        w = np.sqrt(w)
    return 6.0 + 400.0 * w / w.max()


def _weight_fractions(w) -> np.ndarray:
    """The examples' t: w / sum(w), then over its 80% quantile, with R's long double sum and
    type 7 quantile. No weights give no fractions, as R's sapply over numeric(0) does."""
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        return w
    with np.errstate(divide="ignore", invalid="ignore"):
        t = w / r_sum(w)
        if np.isnan(t).any():
            raise ValueError(QUANTILE_NAN_STOP)
        return t / r_quantile7(t, 0.8)


def _rgb_stop(v: float) -> str:
    """ScaleColor's message, which prints the value with %g (NaN as "nan")."""
    return f"color intensity {v:g}, not in [0,1]"


def _rgb_green(v: float) -> str:
    """R's rgb(0, v, 0): ScaleColor's checks, then 255 v + 0.5 truncated."""
    if not (v >= 0 and v <= 1):
        raise ValueError(_rgb_stop(v))
    return f"#00{int(255 * v + 0.5):02X}00"


def weight_colors(w) -> list[str]:
    """The help examples' nugget colors: the fraction of weight over its 80% quantile, capped at
    1, as rgb(0, t, 0). Lighter green means more weight. The hex strings are R's."""
    return [_rgb_green(v) for v in np.minimum(_weight_fractions(w), 1.0).tolist()]


def clusters(x, labels, *, w=None, centers=None, dims=(0, 1), size="area", colors=None, ax=None):
    """The Whclust and Wkmeans examples' plot(x, col = labels) with points(centers): the rows of
    x on columns dims, label l in colors[l % len(colors)] (the color cycle by default, R_PALETTE
    for R's), sized by w (see size), and centers, one row per label present in label order, as
    stars in their cluster's color with a white edge. Returns the Axes."""
    x = _matrix(x)
    n, p = x.shape
    i, j = _dims(dims, p)
    labels = _labels(labels, n)
    s = _sizes(w, n, size)
    palette = _palette(colors)
    c = None
    present = np.unique(labels)
    if centers is not None:
        c = _matrix(centers)
        if c.shape != (present.size, p):
            raise ValueError(f"centers must be {present.size} by {p}, one row per label present")
    ax = plt.gca() if ax is None else ax
    ax.scatter(x[:, i], x[:, j], s=s, c=_label_colors(labels, palette), linewidths=0, gid="points")
    if c is not None:
        ax.scatter(
            c[:, i],
            c[:, j],
            s=CENTER_SIZE,
            marker="*",
            c=_label_colors(present, palette),
            edgecolors="white",  # OKABE_ITO and R_PALETTE start with black
            linewidths=0.6,
            zorder=3,
            gid="centers",
        )
    return ax


def nuggets(dn, *, labels=None, dims=(0, 1), colors=None, ax=None):
    """The DN examples' plot of nugget centers on columns dims, pch = 16 at cex = 0.5, colored
    by weight as R's examples do (weight_colors), or by labels like clusters. Returns the
    Axes."""
    c = _matrix(dn.centers)
    m, p = c.shape
    i, j = _dims(dims, p)
    if labels is None:
        face = weight_colors(dn.weights)
    else:
        face = _label_colors(_labels(labels, m), _palette(colors))
    ax = plt.gca() if ax is None else ax
    ax.scatter(c[:, i], c[:, j], s=(PCH16 * NUGGET_CEX) ** 2, c=face, linewidths=0, gid="nuggets")
    return ax
