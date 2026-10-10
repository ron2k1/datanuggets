"""Paper figures: the Okabe-Ito colors, small type, TrueType fonts in PDFs, and exact sizes."""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
from cycler import cycler

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

# R 4's default palette(), so label l can take R's col = l + 1
R_PALETTE = ("black", "#DF536B", "#61D04F", "#2297E6", "#28E2E5", "#CD0BBC", "#F5C710", "gray62")
# Okabe and Ito's eight colors, which stay apart for readers with color-blindness
OKABE_ITO = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
WIDTHS = {"single": 3.5, "double": 7.0}  # one and two journal columns, inches
DPI = 300

# Typed by matplotlib's rc keys, so a checker catches a misspelled one
PAPER_RC: dict[RcKeyType, Any] = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.prop_cycle": cycler(color=list(OKABE_ITO)),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
    "savefig.dpi": DPI,
}


def style():
    """A with-block in which new figures take the paper settings. rcParams come back as they
    were when it ends, even on an exception."""
    return matplotlib.rc_context(PAPER_RC)


def _inches(value, name: str) -> float:
    """A width or height in inches: a WIDTHS name, or a positive finite number, numpy's too."""
    v = WIDTHS.get(value) if isinstance(value, str) else value
    inches = float(v) if isinstance(v, numbers.Real) and not isinstance(v, bool) else math.nan
    if not (math.isfinite(inches) and inches > 0):
        raise ValueError(f"{name} must be 'single', 'double' or a positive number of inches")
    return inches


def save(fig, path, *, width="single", height=None) -> Path:
    """Saves fig at width ("single", 3.5 inches, "double", 7, or inches) by height (inches,
    default three quarters of the width) at 300 dpi, the format from the suffix. A tight
    bounding box would change the size, so it's off whatever rcParams say."""
    w = _inches(width, "width")
    h = 0.75 * w if height is None else _inches(height, "height")
    fig.set_size_inches(w, h)
    path = Path(path)
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path, dpi=DPI)
    return path
