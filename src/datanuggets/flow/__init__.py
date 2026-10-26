"""Cabrera's gate finder for the lymphocyte and macrophage populations on FSC-A against SSC-A,
from Prof. Cabrera's Python port of the original R code, checked piece by piece against it."""

from datanuggets.flow.gate import FinderPeaks, finder_peaks, hull_gate, lymphmacro_finder
from datanuggets.flow.grid import DensityGrid, f_plot
from datanuggets.flow.peaks import find_area, local_maxima, mirrored_area, refine_peaks, search_ind

__all__ = [
    "DensityGrid",
    "FinderPeaks",
    "f_plot",
    "find_area",
    "finder_peaks",
    "hull_gate",
    "local_maxima",
    "lymphmacro_finder",
    "mirrored_area",
    "refine_peaks",
    "search_ind",
]
