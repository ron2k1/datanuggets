# Adapted from datanugget 1.5.0's create.DN (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""create.DN: reduce a dataset to data nuggets (center, weight, scale)."""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
from joblib import Parallel, delayed

from datanuggets._compat.arithmetic import r_mean
from datanuggets._compat.indexing import split_indices
from datanuggets._compat.rfast import nearest_index
from datanuggets._defaults import (
    CENTER_METHODS,
    DELETE_PERCENT,
    DIST_METRICS,
    DN_NUM1,
    DN_NUM2,
    R_CHUNK,
    R_MAX,
    R_MIN,
    SEED,
)
from datanuggets._random.interfaces import Draws
from datanuggets._random.seeds import as_draws
from datanuggets._validate import (
    as_count,
    as_matrix,
    as_real,
    check_choice,
    check_delete_percent,
    check_n_jobs,
)
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.nuggets.centers import PASSTHROUGH_WARNING, select_centers
from datanuggets.result import DataNuggets

log = logging.getLogger("datanuggets")


def create_dn(
    x,
    center_method: str = "mean",
    r: float = R_CHUNK,
    delete_percent: float = DELETE_PERCENT,
    dn_num1: int = DN_NUM1,
    dn_num2: int = DN_NUM2,
    dist_metric: str = "euclidean",
    seed=SEED,
    n_jobs: int = 1,
    progress: bool = False,
) -> DataNuggets:
    """Reduce a numeric dataset to data nuggets: a center, a weight and a scale each.

    Port of R's create.DN (datanugget 1.5.0). The rows are shuffled once and split into
    ``ceil(n / r)`` chunks. Each chunk is reduced to its share of ``dn_num1`` centers by
    repeatedly deleting one point of each closest pair. The pooled centers are reduced the same
    way to ``dn_num2``, and every row is assigned to its nearest center. A nugget's weight is
    its row count and its scale is the mean of its columns' sample variances. Nugget ids and
    assignments are 0-based.

    Parameters
    ----------
    x : array_like, shape (n, p)
        Numeric data with at least two rows, as a 2-D array or a data frame. A frame mixing
        bool and numeric columns reads True as 1 and False as 0.
    center_method : {"mean", "random", "original"}, default "mean"
        How each final center is set once the rows are assigned. "mean" takes the mean of the
        nugget's rows. "random" takes one of its rows at random. "original" keeps the selected
        center. Under "mean" and "random" an empty nugget keeps its selected center.
    r : float, default 5000
        Chunk-size control within [100, 10000]. A fraction is accepted. The rows are split
        into ``ceil(n / r)`` chunks.
    delete_percent : float, default 0.1
        Fraction of the remaining points deleted in each thinning round, within (0, 1).
    dn_num1 : int, default 10000
        Target count of centers pooled from the chunks, a whole number of at least 1. Each
        chunk gets ``dn_num1 // chunks`` and the remainder goes to the first chunks one each.
        Each share is clamped to [1, chunk size], so the pooled count can differ from it.
    dn_num2 : int, default 2000
        Target count of final nuggets, a whole number of at least 1 and at most ``dn_num1``.
    dist_metric : {"euclidean", "manhattan"}, default "euclidean"
        Distance used to select centers and to assign rows. A row tied between centers goes to
        the lowest-numbered one.
    seed : int, numpy.random.Generator or RDraws, default 291102
        Source of the chunk shuffle and of the "random" centers. An int starts a new numpy
        generator on every call, so the same int gives the same nuggets. A Generator is drawn
        from directly, so the call advances it. ``RDraws(seed)`` draws what R's
        ``set.seed(seed)`` draws.
    n_jobs : int, default 1
        Threads for the chunks: a positive int, or -1 for every core. The result does not
        depend on it.
    progress : bool, default False
        Show a tqdm bar over the chunks. Needs the ``progress`` extra.

    Returns
    -------
    DataNuggets
        ``centers`` (m, p) and ``scales`` (m,) as float64, and ``weights`` (m,) and
        ``assignments`` (n,) as int64, with m at most ``dn_num2``. Empty nuggets are kept. A
        nugget with fewer than two rows, or with a scale that is not finite, gets a floor as
        its scale: the smallest finite scale among nuggets with two or more rows divided by
        100. The floor is float64 eps when there is no such scale or that quotient is 0.

    Raises
    ------
    ValueError
        If ``r`` is outside [100, 10000], ``delete_percent`` is outside (0, 1), ``dn_num1`` or
        ``dn_num2`` is not a whole number of at least 1, ``dn_num2 > dn_num1``,
        ``center_method`` or ``dist_metric`` is not one of its choices, or ``n_jobs`` is not
        a positive int or -1. Also if ``x`` is not 2-D, has no columns, holds NaN or infinity,
        or has fewer than two rows.
    TypeError
        If ``r``, ``delete_percent``, ``dn_num1`` or ``dn_num2`` is a bool or not a real
        number, if ``x`` is not numeric or is all bool, or if ``seed`` is not an int, a numpy
        Generator or a Draws object.
    ImportError
        If ``progress=True`` and tqdm is not installed.

    Warns
    -----
    DataNuggetWarning
        R's "DN.num is greater than or equal to the number of rows. Returning original data",
        once for each selection whose input has no more rows than its target.

    See Also
    --------
    refine_dn : Split nuggets that are spread out or elongated.
    get_dn_means : The column means of each nugget's rows.

    Notes
    -----
    R counterpart: ``datanugget::create.DN``, ported from datanugget 1.5.0. Differences from R:

    - Nugget ids and assignments are 0-based, where R's are 1-based.
    - R returns a list of class "datanugget" holding the "Data Nuggets" table and the
      "Data Nugget Assignments". This returns a DataNuggets, and ``to_frame()`` rebuilds the
      table.
    - ``dn_num1`` and ``dn_num2`` must be whole numbers. R checks only that they are numeric.
    - ``n_jobs`` defaults to 1, where R's ``no.cores`` defaults to ``detectCores() - 1``. Only
      the chunks run in parallel here. R's ``no.cores`` also parallelizes the assignment.
    - ``progress`` draws one bar over the chunks. R's ``make.pbs`` also draws one over the
      second-stage deletions.
    - An int seed draws from numpy and does not give R's nuggets. ``RDraws(seed)`` does.
    - On an exact distance tie a row goes to the lowest-numbered center. R's Rfast picks the
      same center when there are 16 centers or fewer.
    - ``dn_num2 > dn_num1`` raises "dn_num2 must be less than dn_num1", where R says
      "DN.num2 must be less than DN.num1". Equal values are accepted, as in R.

    References
    ----------
    .. [1] Beavers, T. E., Cheng, G., Duan, Y., Cabrera, J., Lubomirski, M., Amaratunga, D.,
       and Teigler, J. E. "Data Nuggets: A Method for Reducing Big Data While Preserving Data
       Structure." Journal of Computational and Graphical Statistics 34(1), 330-342, 2025.
       Published online in 2024. DOI: ``10.1080/10618600.2024.2341896``.
    .. [2] Cherasia, K. E., Cabrera, J., Fernholz, L. T., and Fernholz, R. "Data Nuggets in
       Supervised Learning." In M. Yi and K. Nordhausen (eds.), *Robust and Multivariate
       Statistical Methods: Festschrift in Honor of David E. Tyler*, 429-449. Springer, 2023.
       Published online in 2022. DOI: ``10.1007/978-3-031-22687-8_20``.

    Examples
    --------
    The data are the 300 points of a 10 by 10 by 3 grid, one point per row.

    >>> import numpy as np
    >>> from datanuggets import RDraws, create_dn
    >>> x = np.indices((10, 10, 3)).reshape(3, -1).T.astype(float)
    >>> x.shape
    (300, 3)
    >>> dn = create_dn(x, r=100, delete_percent=0.1, dn_num1=60, dn_num2=20,
    ...                seed=RDraws(291102))
    >>> dn.centers.shape
    (20, 3)
    >>> dn.weights.tolist()
    [24, 13, 12, 19, 12, 16, 8, 7, 14, 8, 21, 16, 26, 18, 9, 25, 16, 13, 11, 12]
    >>> int(dn.weights.sum())
    300
    >>> dn.assignments[:10].tolist()
    [7, 7, 7, 7, 7, 14, 17, 14, 14, 17]
    >>> round(float(dn.scales[0]), 4)
    0.8345
    """
    check_choice("center_method", center_method, CENTER_METHODS)
    r = as_real("r", r)
    if not R_MIN <= r <= R_MAX:
        raise ValueError(f"r must be within [{R_MIN}, {R_MAX}], got {r!r}")
    delete_percent = check_delete_percent(delete_percent)
    dn_num1 = as_count("dn_num1", dn_num1)
    dn_num2 = as_count("dn_num2", dn_num2)
    if dn_num2 > dn_num1:
        raise ValueError("dn_num2 must be less than dn_num1")
    check_choice("dist_metric", dist_metric, DIST_METRICS)
    check_n_jobs(n_jobs)
    bar = _progress_bar() if progress else None
    draws = as_draws(seed)
    x = as_matrix(x)
    n = x.shape[0]
    if n < 2:
        raise ValueError("x must contain at least two observations")

    perm = draws.permutation(n)
    chunks = [c for c in split_indices(n, math.ceil(n / r)) if c.size > 0]
    m_each = _allocate(dn_num1, np.array([c.size for c in chunks]))

    log.info("Creating initial set of data nugget centers...")

    def stage1(i: int) -> tuple[np.ndarray, bool]:
        rs = x[perm[chunks[i]]]
        kept, passthrough = select_centers(rs, int(m_each[i]), delete_percent, dist_metric)
        return rs[kept], passthrough

    ids = range(len(chunks)) if bar is None else bar(range(len(chunks)), desc="stage 1 chunks")
    if n_jobs == 1:
        parts = [stage1(i) for i in ids]
    else:
        parts = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(stage1)(i) for i in ids)
    for _, passthrough in parts:
        if passthrough:
            warnings.warn(PASSTHROUGH_WARNING, DataNuggetWarning, stacklevel=2)
    centers1 = np.vstack([c for c, _ in parts])

    log.info("Creating final set of data nugget centers...")
    kept2, passthrough = select_centers(centers1, dn_num2, delete_percent, dist_metric)
    if passthrough:
        warnings.warn(PASSTHROUGH_WARNING, DataNuggetWarning, stacklevel=2)
    stage2 = centers1[kept2]

    log.info("Assigning observations to the nearest data nuggets...")
    assignments = nearest_index(x, stage2, dist_metric)

    log.info("Recalculating the data nugget centers and calculating the data nugget scales...")
    centers, weights, scales = _summarize(x, assignments, stage2, center_method, draws)
    return DataNuggets(centers=centers, weights=weights, scales=scales, assignments=assignments)


def _progress_bar():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("progress=True needs tqdm: pip install 'datanuggets[progress]'") from exc
    return tqdm


def _allocate(dn_num1: int, chunk_sizes: np.ndarray) -> np.ndarray:
    """R: m.each <- DN.num1 %/% splits, remainder to the first chunks, clamped to
    [1, chunk size]."""
    s = chunk_sizes.size
    m = np.full(s, dn_num1 // s, dtype=np.int64)
    rem = dn_num1 - int(m.sum())
    if rem > 0:
        m[:rem] += 1
    return np.maximum(1, np.minimum(m, chunk_sizes))


def _summarize(
    x: np.ndarray, assignments: np.ndarray, stage2: np.ndarray, center_method: str, draws: Draws
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights, scales and final centers with R's formulas (one-pass sums on centred data)."""
    k, p = stage2.shape
    weights = np.bincount(assignments, minlength=k)

    g_mean = r_mean(x, axis=0)  # colMeans: long-double sum, as R
    xc = x - g_mean
    s1 = np.zeros((k, p))
    s2 = np.zeros((k, p))
    np.add.at(s1, assignments, xc)  # R rowsum(): sequential, row order
    np.add.at(s2, assignments, xc * xc)
    n_safe = np.maximum(weights, 1)[:, None]
    mean_c = s1 / n_safe
    ss = s2 - n_safe * (mean_c * mean_c)
    ss[ss < 0] = 0.0
    var = ss / np.maximum(weights - 1, 1)[:, None]
    scales = r_mean(var, axis=1)  # rowMeans
    scales[weights < 2] = np.nan

    if center_method == "mean":
        centers = mean_c + g_mean
        empty = weights == 0
        centers[empty] = stage2[empty]
    elif center_method == "original":
        centers = stage2.copy()
    else:
        order = np.argsort(assignments, kind="stable")
        bounds = np.concatenate(([0], np.cumsum(weights)))
        centers = np.empty((k, p))
        for i in range(k):
            members = order[bounds[i] : bounds[i + 1]]
            centers[i] = stage2[i] if members.size == 0 else x[members[draws.index(members.size)]]

    multi = scales[(weights > 1) & np.isfinite(scales)]
    floor = multi.min() / 100 if multi.size else np.finfo(np.float64).eps
    if not np.isfinite(floor) or floor <= 0:
        floor = np.finfo(np.float64).eps
    scales[~np.isfinite(scales)] = floor
    return centers, weights, scales
