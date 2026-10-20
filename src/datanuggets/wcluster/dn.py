"""WCluster's DN.Wkmeans, DN.Whclust, DN.Wpca and DNcluster.predict: the WCluster functions on a
datanugget, its centers as the data and its weights as the row weights."""

from __future__ import annotations

import numpy as np

from datanuggets.result import DataNuggets, DnWkmeansResult, WhclustResult, WpcaResult
from datanuggets.wcluster.predict import cluster_predict
from datanuggets.wcluster.whclust import whclust
from datanuggets.wcluster.wkmeans import wkmeans
from datanuggets.wcluster.wpca import wpca

_ZERO = "argument is of length zero"


def _check(dn) -> DataNuggets:
    if not isinstance(dn, DataNuggets):
        raise TypeError(
            'dn must be a DataNuggets, as R\'s datanugget must be of class "datanugget", '
            f"got {type(dn).__name__}"
        )
    return dn


def _one_variable(dn: DataNuggets, what: str) -> None:
    if dn.centers.shape[1] == 1:
        raise ValueError(
            f"{what} on a datanugget with one variable: R's [, 2:2] drops its table to a "
            f'vector, and R stops with "{_ZERO}"'
        )


def dn_wkmeans(
    dn, k, *, centers=None, num_init=1, max_iter=10, seed=291102, n_jobs=1
) -> DnWkmeansResult:
    """DN.Wkmeans: wkmeans on the nuggets with their weights, plus row_labels, each original
    row's cluster, the one its nugget gets. Pass seed=RDraws(seed) to match R. R stops on a
    datanugget with one variable, and so does this."""
    _check(dn)
    _one_variable(dn, "DN.Wkmeans")
    res = wkmeans(
        dn.centers,
        k,
        w=dn.weights,
        centers=centers,
        num_init=num_init,
        max_iter=max_iter,
        seed=seed,
        n_jobs=n_jobs,
    )
    return DnWkmeansResult(
        res.labels, res.centers, res.wwcss, res.total_wwcss, res.labels[dn.assignments]
    )


def dn_whclust(dn) -> WhclustResult:
    """DN.Whclust: whclust on the nuggets with their weights."""
    _check(dn)
    return whclust(dn.centers, dn.weights)


def dn_wpca(dn, wcol=None, corr=False) -> WpcaResult:
    """DN.Wpca: wpca of the nuggets with their weights as row weights and wcol (ones by
    default) as column weights. R's default wcol stops on a datanugget with one variable, and
    so does this."""
    _check(dn)
    if wcol is None:
        if dn.centers.shape[1] == 1:
            raise ValueError(
                "DN.Wpca's default wcol on a datanugget with one variable: R's "
                "rep(1, ncol(vector)) stops with \"invalid 'times' argument\""
            )
        wcol = np.ones(dn.centers.shape[1])
    return wpca(dn.centers, wrow=dn.weights, wcol=wcol, corr=corr)


def dn_cluster_predict(dn, cl, newx) -> np.ndarray:
    """DNcluster.predict: the cluster each row of newx is nearest to, the centers being the
    weighted means of the nuggets in each cluster of cl, as 0-based positions in sorted label
    order, as cluster_predict gives them. newx is a matrix, a data frame or a DataNuggets. R
    takes a datanugget newx's table columns by dn's number of variables, so a wider one gives
    its first variables and a narrower one passes its Weight and Scale columns as data, and
    this does the same."""
    _check(dn)
    if np.size(cl) != dn.n_nuggets:
        raise ValueError(
            "The length of cluster assignments should be the same as the number of data "
            f"nuggets. Got {np.size(cl)} for {dn.n_nuggets}"
        )
    if isinstance(newx, DataNuggets):
        p = dn.centers.shape[1]
        if p == 1:
            raise ValueError(
                "DNcluster.predict with a datanugget newx and a dn with one variable: R's "
                f'[, 2:2] drops newx\'s table to a vector, and R stops with "{_ZERO}"'
            )
        table = np.column_stack([newx.centers, newx.weights, newx.scales])
        if p > table.shape[1]:
            raise ValueError(
                f"newx's table has {table.shape[1]} columns after its ids, fewer than dn's {p} "
                'variables, and R stops with "undefined columns selected"'
            )
        newx = table[:, :p]
    return cluster_predict(dn.centers, cl, newx, w=dn.weights)
