"""create.DN: reduce a dataset to data nuggets (center, weight, scale)."""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
from joblib import Parallel, delayed

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
from datanuggets._draws import Draws, as_draws
from datanuggets._rcompat import r_mean, split_indices
from datanuggets._rfast import nearest_index
from datanuggets._validate import (
    as_count,
    as_matrix,
    as_real,
    check_choice,
    check_delete_percent,
    check_n_jobs,
)
from datanuggets.centers import PASSTHROUGH_WARNING, select_centers
from datanuggets.exceptions import DataNuggetWarning
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
    """Port of R's create.DN (datanugget 1.5.0).

    The rows are shuffled once, split into ceiling(n / r) chunks, each chunk is reduced
    to its share of dn_num1 centers, the pooled centers are reduced to dn_num2, and every
    row is assigned to its nearest center. Nugget ids and assignments are 0-based.
    progress=True shows a tqdm bar over the stage-1 chunks (needs the `progress` extra).
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
    """R: m.each <- DN.num1 %/% splits, remainder to the first chunks, clamped to [1, chunk size]."""
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
