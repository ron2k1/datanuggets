"""gates: Cabrera's figure of the gate finder's populations on FSC-A against SSC-A."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# his scatter sizes, in points squared, and alphas. Drawn as Line2D markers with markersize the
# square root of his s, and scatter's edge, patch.linewidth points in the face color, so a figure
# of a million events draws as one path per population rather than one marker each
EVENT_SIZE = 0.5
POP_SIZE = 5.0
EVENT_ALPHA = 0.3
POP_ALPHA = 0.8


def gates(fsc_a, ssc_a, masks, *, title=None, ax=None):
    """His figure: every event in gray, then each population that has an event in the next color
    of the axes' cycle, so an empty population takes no color, as his scatter calls do. masks is
    the finder's output, one bool column per population. Returns the axes."""
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
