"""gates: Cabrera's figure of the gate finder's populations on FSC-A against SSC-A, and peaks:
the density grid they are found on, with its local maxima and the peaks that seed them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator

# cfc.py's scatter sizes, in points squared, and alphas. Drawn as Line2D markers with markersize the
# square root of its s, and scatter's edge, patch.linewidth points in the face color, so a figure
# of a million events draws as one path per population rather than one marker each
EVENT_SIZE = 0.5
POP_SIZE = 5.0
EVENT_ALPHA = 0.3
POP_ALPHA = 0.8


def gates(fsc_a, ssc_a, masks, *, title=None, ax=None):
    """cfc.py's figure: every event in gray, then each population that has an event in the next
    color of the axes' cycle, so an empty population takes no color, as cfc.py's scatter calls do.
    masks is the finder's output, one bool column per population. Returns the axes."""
    fsc = np.asarray(fsc_a, dtype=float)
    ssc = np.asarray(ssc_a, dtype=float)
    if fsc.ndim != 1 or fsc.shape != ssc.shape:
        raise ValueError("fsc_a and ssc_a must be one-dimensional and the same length")
    m = np.asarray(masks)
    if m.ndim != 2 or m.dtype != bool:
        raise ValueError("masks must be a two-dimensional bool array, one column per population")
    if len(m) != len(fsc):
        raise ValueError(f"masks must have one row per event, {len(fsc)}, got {len(m)}")
    ax = plt.gca() if ax is None else ax
    style = {"ls": "none", "mew": plt.rcParams["patch.linewidth"], "rasterized": True}
    ax.plot(
        fsc,
        ssc,
        "o",
        ms=EVENT_SIZE**0.5,
        color="gray",
        alpha=EVENT_ALPHA,
        label="Unlabeled",
        gid="events",
        **style,
    )
    for j in range(m.shape[1]):
        if m[:, j].any():  # no color given, so the axes' cycle gives the next one
            ax.plot(
                fsc[m[:, j]],
                ssc[m[:, j]],
                "o",
                ms=POP_SIZE**0.5,
                alpha=POP_ALPHA,
                label=f"Pop {j + 1}",
                gid="population",
                **style,
            )
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("FSC-A")
    ax.set_ylabel("SSC-A")
    ax.legend(markerscale=4)
    return ax


def peaks(fp, *, title=None, ax=None):
    """The finder's density grid, FSC-A across and SSC-A up as in gates, every local maximum as
    an open circle and each kept peak as a cross labelled with the Pop number it seeds in gates.
    fp is flow.finder_peaks' result; ticks are in the channels' raw units. Returns the axes."""
    density = np.asarray(fp.density, dtype=float)
    if density.ndim != 2:
        raise ValueError("density must be a two-dimensional grid")
    nr, nc = density.shape
    fsc, ssc = np.asarray(fp.fsc, dtype=float), np.asarray(fp.ssc, dtype=float)
    if fsc.shape != (nc,) or ssc.shape != (nr,):
        raise ValueError(f"fsc and ssc must give a value per column and row, {nc} and {nr}")
    cand = np.asarray(fp.candidates).reshape(-1, 2)
    kept = np.asarray(fp.peaks).reshape(-1, 2)
    ax = plt.gca() if ax is None else ax
    im = ax.imshow(
        np.where(density > 0, density, np.nan),
        origin="lower",
        extent=(0.5, nc + 0.5, 0.5, nr + 0.5),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.figure.colorbar(im, ax=ax, label="log(1 + events per bin)")
    ax.plot(
        cand[:, 1],
        cand[:, 0],
        "o",
        ms=7,
        mfc="none",
        mec="white",
        ls="none",
        label=f"Local maxima ({len(cand)})",
        gid="candidates",
    )
    ax.plot(
        kept[:, 1],
        kept[:, 0],
        "x",
        ms=9,
        mew=2,
        color="red",
        ls="none",
        label=f"Kept peaks ({len(kept)})",
        gid="peaks",
    )
    for k, (i, j) in enumerate(kept, start=1):
        ax.annotate(
            f"Pop {k}",
            (j, i),
            xytext=(5, 5),
            textcoords="offset points",
            color="red",
            fontweight="bold",
        )
    for axis, raw in ((ax.xaxis, fsc), (ax.yaxis, ssc)):  # round raw values at their bins
        ticks = MaxNLocator(5).tick_values(raw[0], raw[-1])
        ticks = ticks[(ticks >= raw[0]) & (ticks <= raw[-1])]
        axis.set_major_locator(FixedLocator(np.interp(ticks, raw, np.arange(1, len(raw) + 1))))
        axis.set_major_formatter(FixedFormatter([_short(t) for t in ticks]))
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("FSC-A (sqrt-scaled grid)")
    ax.set_ylabel("SSC-A (sqrt-scaled grid)")
    ax.legend(loc="upper left", facecolor="0.75")  # gray, so the white circle shows
    return ax


def _short(v: float) -> str:
    """A tick value short enough not to crowd the sqrt scale's top end: 2.4M, 800k, 0."""
    for unit, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(v) >= unit:
            return f"{v / unit:g}{suffix}"
    return f"{v:g}"
