"""Plots of the port's results: R's dendrogram, the cluster and nugget scatters of the help
examples, weighted PCA views, Cabrera's figure of the flow gates and the density peaks they grow
from, and a style for paper figures.
Needs the plot extra, matplotlib, which import datanuggets never loads."""

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise ImportError("datanuggets.plot needs matplotlib: pip install 'datanuggets[plot]'") from exc

from datanuggets.plot.flow import gates, peaks
from datanuggets.plot.pca import loadings, scores, scree
from datanuggets.plot.points import clusters, nuggets
from datanuggets.plot.style import OKABE_ITO, PAPER_RC, R_PALETTE, save, style
from datanuggets.plot.tree import dendrogram

__all__ = [
    "OKABE_ITO",
    "PAPER_RC",
    "R_PALETTE",
    "clusters",
    "dendrogram",
    "gates",
    "loadings",
    "nuggets",
    "peaks",
    "save",
    "scores",
    "scree",
    "style",
]
