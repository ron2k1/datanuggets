"""Cabrera's gate finder for the lymphocyte and macrophage populations on FSC-A against SSC-A,
from his Python port of his R code, checked piece by piece against it."""

from datanuggets.flow.gate import hull_gate, lymphmacro_finder
from datanuggets.flow.grid import DensityGrid, f_plot
from datanuggets.flow.peaks import find_area, local_maxima, refine_peaks, search_ind

__all__ = [
    "DensityGrid",
    "f_plot",
    "find_area",
    "hull_gate",
    "local_maxima",
    "lymphmacro_finder",
    "refine_peaks",
    "search_ind",
]
