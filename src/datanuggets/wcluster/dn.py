# Adapted from WCluster 1.3.0's DN wrappers (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""WCluster's DN.Wkmeans, DN.Whclust, DN.Wpca and DNcluster.predict: the WCluster functions on a
datanugget, its centers as the data and its weights as the row weights."""

from __future__ import annotations

import numpy as np

from datanuggets._defaults import SEED
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
    dn, k, *, centers=None, num_init=1, max_iter=10, seed=SEED, n_jobs=1
) -> DnWkmeansResult:
    """DN.Wkmeans: wkmeans on the nuggets with their weights, plus row_labels, each original
    row's cluster, the one its nugget gets. Pass seed=RDraws(seed) to match R. R stops on a
    datanugget with one variable, and so does this.

    Parameters
    ----------
    dn : DataNuggets
        The data nuggets. Their centers are the data and their weights the row weights.
    k : float
        Number of clusters, floored. It must be at least 1 after the floor.
    centers : array_like, shape (k, p), optional
        Starting centers, one row per cluster, in place of the random starts. R takes them as
        p by k.
    num_init : float, default 1
        Number of random starts, floored. Without ``centers`` it must be at least 1. With
        ``centers`` it is set to 1, and only its type is checked.
    max_iter : float, default 10
        Maximum rounds of Lloyd steps and moves per start, floored. It must be at least 1.
    seed : int, numpy.random.Generator or RDraws, default 291102
        Source of the random starts. ``RDraws(seed)`` starts from ``seed`` on every call, as
        R's ``set.seed(seed)`` does, and draws what R draws. An int seeds a new numpy
        Generator on every call, so the same int gives the same result, but not R's. A numpy
        Generator is drawn from in place. Each random start advances it, so a second call with
        the same Generator draws other starts. A call with ``centers`` draws nothing. The
        default 291102 is R's default seed. As an int it draws from numpy, and
        ``RDraws(291102)`` gives R's default draws.
    n_jobs : int, default 1
        Threads that refine the starts: a positive int, or -1 for every core. It is used only
        when ``num_init`` is above 1 and no ``centers`` are given. It is checked even where it
        is not used, unless another argument raises first. It changes neither the result nor
        the error raised.

    Returns
    -------
    DnWkmeansResult
        ``labels``, ``centers``, ``wwcss`` and ``total_wwcss`` as ``wkmeans`` returns them for
        the nuggets, and ``row_labels``, shape (n,), each original row's cluster.

    Raises
    ------
    TypeError
        If ``dn`` is not a DataNuggets. Also for the argument types ``wkmeans`` rejects.
    ValueError
        If the nuggets have one variable, where R stops with "argument is of length zero".
        Also wherever ``wkmeans`` raises it on the nuggets, their weights and the other
        arguments.
    IndexError
        If ``dn.assignments`` is neither an integer nor a bool array, or is a bool array whose
        length is not the number of nuggets. Also if it holds an id at or beyond the number of
        nuggets or below minus that number. ``row_labels`` is ``labels[dn.assignments]``, so
        negative ids count from the end and a bool array with one entry per nugget selects
        labels, as numpy indexing does.

    See Also
    --------
    wkmeans : The clustering this runs.
    dn_cluster_predict : Assign new rows to the clusters.

    Notes
    -----
    R counterpart: ``WCluster::DN.Wkmeans``, ported from WCluster 1.3.0. Differences from R:

    - Labels and row labels are R's minus one.
    - R names the fields "Cluster Assignments for data nuggets" and "Cluster Assignments for
      original dataset". Here they are ``labels`` and ``row_labels``.
    - The arguments after ``k`` are keyword-only, and ``centers`` is (k, p), as in
      ``wkmeans``.
    - An int seed, the default included, draws from numpy and does not give R's clusters.
      ``RDraws(seed)`` does.
    - ``n_jobs`` has no R counterpart.

    References
    ----------
    .. [1] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.
    .. [2] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import RDraws, create_dn, dn_wkmeans
    >>> g = np.array([(i, j) for i in range(6) for j in range(6)], dtype=float)
    >>> x = np.vstack([g, g + 10])
    >>> dn = create_dn(x, r=100, dn_num1=40, dn_num2=10, seed=RDraws(3))
    >>> dn.n_nuggets
    10
    >>> res = dn_wkmeans(dn, 2, max_iter=5, seed=RDraws(3))
    >>> res.labels.tolist()
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    >>> [round(float(v), 4) for v in res.centers[:, 0]]
    [2.5, 12.5]
    >>> [round(float(v), 4) for v in res.wwcss]
    [148.8084, 160.4048]
    >>> np.bincount(res.row_labels).tolist()
    [36, 36]
    """
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
    """DN.Whclust: whclust on the nuggets with their weights.

    Parameters
    ----------
    dn : DataNuggets
        The data nuggets. Their centers are the data and their weights the row weights.

    Returns
    -------
    WhclustResult
        The tree over the nuggets, as ``whclust`` returns it: ``merge`` in scipy's numbering,
        ``height`` the total WWCSS after each merge, and ``order`` 0-based.

    Raises
    ------
    TypeError
        If ``dn`` is not a DataNuggets.
    ValueError
        Wherever ``whclust`` raises it on the nuggets and their weights, including fewer
        than three nuggets, with R's message.

    See Also
    --------
    whclust : The clustering this runs.
    cut_tree : Cut the tree into clusters.

    Notes
    -----
    R counterpart: ``WCluster::DN.Whclust``, ported from WCluster 1.3.0. The differences from
    R are those of ``whclust``: ``merge`` is in scipy's numbering, ``order`` is 0-based, and
    R's ``labels``, ``method``, ``call`` and ``dist.method`` are not ported.

    References
    ----------
    .. [1] Ward, J. H., Jr. "Hierarchical Grouping to Optimize an Objective Function." Journal of
       the American Statistical Association 58(301), 236-244, 1963.
       DOI: ``10.1080/01621459.1963.10500845``.
    .. [2] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.
    .. [3] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import RDraws, create_dn, cut_tree, dn_whclust
    >>> g = np.array([(i, j) for i in range(6) for j in range(6)], dtype=float)
    >>> x = np.vstack([g, g + 10])
    >>> dn = create_dn(x, r=100, dn_num1=40, dn_num2=10, seed=RDraws(3))
    >>> tree = dn_whclust(dn)
    >>> tree.merge.shape
    (9, 2)
    >>> k2 = cut_tree(tree, k=2)
    >>> k2.tolist()
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    >>> np.bincount(k2).tolist()
    [5, 5]
    >>> np.bincount(k2[dn.assignments]).tolist()
    [36, 36]
    """
    _check(dn)
    return whclust(dn.centers, dn.weights)


def dn_wpca(dn, wcol=None, corr=False) -> WpcaResult:
    """DN.Wpca: wpca of the nuggets with their weights as row weights and wcol (ones by
    default) as column weights. R's default wcol stops on a datanugget with one variable, and
    so does this.

    Parameters
    ----------
    dn : DataNuggets
        The data nuggets. Their centers are the data and their weights the row weights.
    wcol : array_like, shape (p,), optional
        One weight per variable, all of one sign with a nonzero sum. None means ones.
    corr : bool, default False
        If True, use the weighted correlation matrix instead of the covariance matrix.

    Returns
    -------
    WpcaResult
        The components of the nuggets, as ``wpca`` returns them. ``x`` holds one row of
        scores per nugget.

    Raises
    ------
    TypeError
        If ``dn`` is not a DataNuggets. Also for the argument types ``wpca`` rejects.
    ValueError
        If the nuggets have one variable and ``wcol`` is None, where R stops with "invalid
        'times' argument". Also wherever ``wpca`` raises it on the nuggets, their weights,
        ``wcol`` and ``corr``.
    scipy.linalg.LinAlgError
        If LAPACK's dsyevr reports an error.

    Warns
    -----
    DataNuggetWarning
        "NaNs produced", once when an eigenvalue is below zero.

    See Also
    --------
    wpca : The analysis this runs.

    Notes
    -----
    R counterpart: ``WCluster::DN.Wpca``, ported from WCluster 1.3.0. The differences from R
    are those of ``wpca``.

    References
    ----------
    .. [1] Amaratunga, D., and Cabrera, J. *Exploration and Analysis of DNA Microarray and Protein
       Array Data*. John Wiley & Sons, 2009 (Vol. 605), as the WCluster manual cites it.
       DOI: ``10.1002/9780470317129``.
    .. [2] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.
    .. [3] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import RDraws, create_dn, dn_wpca
    >>> g = np.array([(i, j, k) for i in range(4) for j in range(4) for k in range(4)])
    >>> dn = create_dn(g * [3.0, 2.0, 1.0], r=100, dn_num1=40, dn_num2=12, seed=RDraws(3))
    >>> res = dn_wpca(dn)
    >>> [round(float(v), 4) for v in res.sdev]
    [3.3014, 2.0303, 0.6125]
    >>> res.rotation.shape
    (3, 3)
    >>> res.x.shape
    (12, 3)
    """
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
    this does the same.

    Parameters
    ----------
    dn : DataNuggets
        The clustered data nuggets. Their centers are the data and their weights the row
        weights.
    cl : array_like, shape (n_nuggets,)
        One numeric cluster label per nugget, such as ``dn_wkmeans(dn, k).labels``. Labels
        must be whole numbers smaller than 1e15 in size.
    newx : array_like, shape (m, p), or DataNuggets
        The rows to assign. A DataNuggets contributes its centers, then its weights and
        scales as further columns, cut to the first p columns. Its weights are not used as
        weights.

    Returns
    -------
    numpy.ndarray, shape (m,)
        Each new row's nearest center as an intp position, 0-based in sorted label order.
        ``np.unique(cl)[pos]`` gives the labels.

    Raises
    ------
    TypeError
        If ``dn`` is not a DataNuggets. Also for the argument types ``cluster_predict``
        rejects.
    ValueError
        If ``cl`` does not have one entry per nugget, with R's message. With a DataNuggets
        ``newx``: if ``dn`` has one variable, where R stops with "argument is of length
        zero", or if ``newx`` has fewer than p columns of centers, weights and scales, where
        R stops with "undefined columns selected". Also wherever ``cluster_predict`` raises
        it.

    See Also
    --------
    cluster_predict : The assignment this runs.
    dn_wkmeans : Cluster the nuggets.

    Notes
    -----
    R counterpart: ``WCluster::DNcluster.predict``, ported from WCluster 1.3.0. Differences
    from R:

    - Positions are 0-based. R returns them 1-based.
    - ``newx`` can be a matrix. R stops with "newdata should be either a data frame or a
      datanugget object."
    - The other differences are those of ``cluster_predict``.

    References
    ----------
    .. [1] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.
    .. [2] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import RDraws, create_dn, dn_cluster_predict, dn_wkmeans
    >>> g = np.array([(i, j) for i in range(6) for j in range(6)], dtype=float)
    >>> x = np.vstack([g, g + 10])
    >>> dn = create_dn(x, r=100, dn_num1=40, dn_num2=10, seed=RDraws(3))
    >>> cl = dn_wkmeans(dn, 2, max_iter=5, seed=RDraws(3)).labels
    >>> dn_cluster_predict(dn, cl, np.array([[0.5, 1.0], [11.0, 12.0]])).tolist()
    [0, 1]
    >>> new = create_dn(x + 0.5, r=100, dn_num1=40, dn_num2=10, seed=RDraws(13))
    >>> np.bincount(dn_cluster_predict(dn, cl, new)).tolist()
    [5, 5]
    """
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
