"""Native Python port of the R packages datanugget and WCluster."""

from importlib.metadata import version as _version

from datanuggets.exceptions import DataNuggetWarning
from datanuggets.nuggets import (
    arithmetic_complexity,
    create_dn,
    create_dn_centers,
    get_dn_means,
    refine_dn,
)
from datanuggets.result import (
    CentersResult,
    DataNuggets,
    DnWkmeansResult,
    WhclustResult,
    WkmeansResult,
    WpcaResult,
    WwcssResult,
)
from datanuggets.wcluster import (
    cluster_predict,
    cut_tree,
    distw,
    dn_cluster_predict,
    dn_whclust,
    dn_wkmeans,
    dn_wpca,
    whclust,
    wkmeans,
    wmean,
    wpca,
    wss,
    wwcss,
)

# pyproject.toml holds the version; this reads it back from the installed metadata
__version__ = _version("datanuggets")

__all__ = [
    "CentersResult",
    "DataNuggetWarning",
    "DataNuggets",
    "DnWkmeansResult",
    "WhclustResult",
    "WkmeansResult",
    "WpcaResult",
    "WwcssResult",
    "arithmetic_complexity",
    "cluster_predict",
    "create_dn",
    "create_dn_centers",
    "cut_tree",
    "distw",
    "dn_cluster_predict",
    "dn_whclust",
    "dn_wkmeans",
    "dn_wpca",
    "get_dn_means",
    "refine_dn",
    "whclust",
    "wkmeans",
    "wmean",
    "wpca",
    "wss",
    "wwcss",
]
