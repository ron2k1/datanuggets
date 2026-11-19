# Adapted from datanugget 1.5.0's refine.DN (GPL-2), notices in THIRD_PARTY.md.
# Python adaptation by Ronil Basu, recorded 2026-11-01. See LICENSE.

"""refine.DN: split data nuggets that are spread out or elongated.

When the splits are R's, everything that reaches the result (memberships, colMeans centers, long
double scales) matches R bit for bit. Split decisions compare eigenvalues from LAPACK's dsyevr
in scipy's build. The tests compare the two largest eigenvalues of the input nuggets with R's
recorded values, within a relative tolerance of 1e-8 and an absolute one of 1e-10. With RDraws
the port makes R's split decisions in every recorded R case.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from datanuggets._clustering.hartigan_wong import KMeansError, kmeans
from datanuggets._compat.arithmetic import (
    r_cov,
    r_mad,
    r_median,
    r_quantile7,
    r_sum,
    r_vec_mean,
)
from datanuggets._compat.linalg import r_eigen_sym, r_matvec
from datanuggets._compat.moments import group_moments
from datanuggets._defaults import (
    DELTA,
    EV_TOL,
    KMEANS_ITER_MAX,
    MAX_SPLITS,
    MIN_NUGGET_SIZE,
    MIN_SHAPE_SIZE,
    NSTART,
    SEED,
)
from datanuggets._random.interfaces import Draws
from datanuggets._random.numpy_draws import NumpyDraws
from datanuggets._random.r_draws import RDraws
from datanuggets._random.seeds import as_draws
from datanuggets._validate import as_matrix, as_real, check_assignments, check_n_jobs
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.nuggets.create import _progress_bar
from datanuggets.result import DataNuggets

log = logging.getLogger("datanuggets")


def refine_dn(
    x,
    dn: DataNuggets,
    ev_tol: float = EV_TOL,
    max_splits: float = MAX_SPLITS,
    min_nugget_size: float = MIN_NUGGET_SIZE,
    shape_split: bool = False,
    min_shape_size: float = MIN_SHAPE_SIZE,
    delta: float = DELTA,
    nstart: float = NSTART,
    seed=SEED,
    n_jobs: int = 1,
    progress: bool = False,
) -> DataNuggets:
    """Split data nuggets that are spread out or elongated.

    Port of R's refine.DN (datanugget 1.5.0). Only dn's assignments and the shape of its
    centers are used: every nugget's center, weight, scale and two largest eigenvalues are
    recomputed from its rows of x. For up to max_splits rounds, nuggets whose first eigenvalue
    is above the ev_tol quantile of the positive ones (taken once, before the first round) are
    split in two by Hartigan-Wong k-means. With shape_split=True and two or more columns, up to
    max_splits more rounds split elongated nuggets (EV1/EV2 above delta, or EV1 above the
    median plus twice the mad of the scales) along their first eigenvector. A shape split is
    kept only if the mean EV1/EV2 of its two children is below delta. Empty nuggets are
    dropped and the rest renumbered from 0.

    R calls set.seed(seed) before every split, so each candidate split here draws from a
    stream of its own (see seed). The streams are made in the main process in candidate order,
    so the result is the same for any n_jobs. n_jobs other than 1 runs each round's splits in
    joblib worker processes and computes the nuggets' centers and covariances on up to that
    many numba threads (all cores for -1). progress=True shows a tqdm bar per round.

    Parameters
    ----------
    x : array_like, shape (n, p)
        The data dn was built from, in the same row order. It must have n rows and the
        columns of ``dn.centers``.
    dn : DataNuggets
        The nuggets to refine. Only ``dn.assignments`` and the shape of ``dn.centers`` are
        read. The assignments are 0-based ids in [0, m), with m the row count of
        ``dn.centers``.
    ev_tol : float, default 0.9
        Quantile, within (0, 1), of the nuggets' positive first eigenvalues that sets the
        splitting threshold. It is the type-7 quantile, taken once before the first round.
    max_splits : float, default 5
        The eigenvalue pass and the shape pass each run at most this many rounds. It is floored,
        and it must be finite and at least 0. With 0 the nuggets are only recomputed from x.
    min_nugget_size : float, default 2
        Smallest child a split may leave, at least 2 and floored. A nugget needs at least
        ``2 * floor(min_nugget_size)`` rows to be split.
    shape_split : bool, default False
        Run the shape pass after the eigenvalue pass. It has no effect with one column or with
        ``max_splits=0``.
    min_shape_size : float, default 10
        Smallest nugget the shape pass considers, at least 10 and not floored. The pass uses
        ``max(min_shape_size, 2 * floor(min_nugget_size))``.
    delta : float, default 2
        EV1/EV2 ratio above which the shape pass splits a nugget, greater than 1. A shape split
        is kept only when the mean EV1/EV2 of its two children is below it.
    nstart : float, default 25
        Random starts for each two-means split. A value of 2 or more is floored. A value other
        than 1 that is below 2 makes every split fail, as R does with ``nstart=0``. It must be
        finite.
    seed : int, numpy.random.Generator or RDraws, default 291102
        Source of the k-means starts. An int draws from numpy and restarts the seed's stream
        for each candidate split. A Generator gives each candidate split a child stream
        spawned from it (``Generator.spawn``). That draws nothing from the Generator but
        advances its seed sequence, so a second call with the same Generator gets other
        streams. ``RDraws(seed)`` draws what R's ``set.seed(seed)`` draws and also restarts
        for each candidate split.
    n_jobs : int, default 1
        Worker processes for each round's splits, and numba threads for the moments: a
        positive int, or -1 for every core. The result does not depend on it.
    progress : bool, default False
        Show a tqdm bar per split round. Needs the ``progress`` extra.

    Returns
    -------
    DataNuggets
        A new set of nuggets with an assignment for every row of x. Each center is the column
        mean of its rows, whatever ``center_method`` built dn. Each scale is the trace of the
        nugget's sample covariance divided by p, and a nugget of one row has scale 0. Empty
        nuggets are dropped and the rest are renumbered from 0 in position order.

    Raises
    ------
    TypeError
        If ``dn`` is not a DataNuggets, ``x`` is not numeric or is all bool, or
        ``shape_split`` is not a bool. Also if a numeric argument is a bool or not a real
        number, or ``seed`` is not an int, a numpy Generator or a Draws object.
    ValueError
        If an argument is out of its range above, ``max_splits`` or ``nstart`` is not finite,
        or ``n_jobs`` is not a positive int or -1. Also if ``x`` is not 2-D, has no rows or
        columns, holds NaN or infinity, or does not match dn's rows and columns, or if
        ``dn.assignments`` holds ids that are not whole numbers in [0, m). A replayed or
        custom Draws with ``n_jobs`` other than 1 raises too. With two or more columns, a nugget
        whose sample covariance overflows to infinity raises R's eigen() message, "infinite or
        missing values in 'x'".
    ImportError
        If ``progress=True`` and tqdm is not installed.
    scipy.linalg.LinAlgError
        If LAPACK's dsyevr fails on a nugget's covariance. In R, an unsuccessful result from
        the LAPACK code that eigen() calls is an error giving a positive error code.

    Warns
    -----
    DataNuggetWarning
        A split's k-means warning, with R's stats::kmeans text: "Quick-TRANSfer stage steps
        exceeded maximum (= N)". A run with two clusters stops after its first quick-transfer
        stage, so R's warning "did not converge in 1000 iterations" cannot occur here. Warnings
        from worker processes arrive in the main process.

    See Also
    --------
    create_dn : Build the nuggets this refines.
    get_dn_means : The column means of each nugget's rows.

    Notes
    -----
    R counterpart: ``datanugget::refine.DN``, ported from datanugget 1.5.0. When the splits
    are R's, the memberships, centers and scales match R bit for bit. The split decisions
    compare eigenvalues from LAPACK's dsyevr in scipy's build. The tests compare the two
    largest eigenvalues of the input nuggets with R's recorded values, within a relative
    tolerance of 1e-8 and an absolute one of 1e-10. With ``RDraws`` the port makes R's split
    decisions in every recorded R case. A decision that differs from R's can change later
    splits.
    Differences from R:

    - Ids and assignments are 0-based, where R's are 1-based.
    - ``x`` must also have dn's column count. R checks only the rows.
    - ``shape_split`` must be a bool. R does no type check.
    - A non-finite ``max_splits`` or ``nstart`` raises ValueError.
    - ``n_jobs`` defaults to 1, where R's ``no.cores`` defaults to ``detectCores() - 1``.
    - An int seed draws from numpy and does not give R's splits. ``RDraws(seed)`` does.

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
    >>> from datanuggets import RDraws, create_dn, refine_dn
    >>> x = np.indices((10, 10, 3)).reshape(3, -1).T.astype(float)
    >>> dn = create_dn(x, r=100, delete_percent=0.1, dn_num1=60, dn_num2=20,
    ...                seed=RDraws(291102))
    >>> dn2 = refine_dn(x, dn, seed=RDraws(291102))
    >>> dn.n_nuggets, dn2.n_nuggets
    (20, 22)
    >>> dn2.weights.tolist()
    [13, 13, 12, 19, 12, 16, 8, 7, 14, 8, 21, 16, 26, 18, 9, 12, 16, 13, 11, 12, 11, 13]
    >>> int(dn2.weights.sum())
    300
    >>> dn2.assignments[:10].tolist()
    [7, 7, 7, 7, 7, 14, 17, 14, 14, 17]
    """
    x = as_matrix(x)
    if not isinstance(dn, DataNuggets):
        raise TypeError(
            f"dn must be a DataNuggets from create_dn or refine_dn, not {type(dn).__name__}"
        )
    ev_tol = as_real("ev_tol", ev_tol)
    if not 0 < ev_tol < 1:
        raise ValueError(f"ev_tol must be within (0, 1), got {ev_tol!r}")
    max_splits = as_real("max_splits", max_splits)
    if not 0 <= max_splits < math.inf:
        raise ValueError(f"max_splits must be a finite number >= 0, got {max_splits!r}")
    min_nugget_size = as_real("min_nugget_size", min_nugget_size)
    if not min_nugget_size >= 2:
        raise ValueError(f"min_nugget_size must be at least 2, got {min_nugget_size!r}")
    if not isinstance(shape_split, (bool, np.bool_)):
        raise TypeError(f"shape_split must be True or False, got {shape_split!r}")
    min_shape_size = as_real("min_shape_size", min_shape_size)
    if not min_shape_size >= 10:
        raise ValueError(f"min_shape_size must be at least 10, got {min_shape_size!r}")
    delta = as_real("delta", delta)
    if not delta > 1:
        raise ValueError(f"delta must be greater than 1, got {delta!r}")
    nstart = as_real("nstart", nstart)
    if not math.isfinite(nstart):
        raise ValueError(f"nstart must be finite, got {nstart!r}")
    check_n_jobs(n_jobs)
    draws = as_draws(seed)
    if n_jobs != 1 and not isinstance(draws, (NumpyDraws, RDraws)):
        raise ValueError(
            "replayed or custom Draws need n_jobs=1: their stream can't be copied into worker "
            "processes. Pass an int seed, a numpy Generator or RDraws(seed) to split in parallel."
        )
    bar = _progress_bar() if progress else None
    n, p = x.shape
    if n == 0:
        raise ValueError("x must contain at least one observation")
    state = _Nuggets(x, check_assignments(dn, n, p), dn.centers.shape[0], n_jobs)
    state.refresh(range(state.size), state.members())
    max_splits = math.floor(max_splits)
    mns = math.floor(min_nugget_size) if math.isfinite(min_nugget_size) else min_nugget_size

    def run_round(cands: np.ndarray, project: bool, desc: str) -> bool:
        """One round of refine.DN's run_splitting and apply_splits. False if nothing split."""
        members = state.members()
        jobs = [
            delayed(_split_nugget)(x[members[c]], draws.fork(), mns, nstart, project, delta)
            for c in cands.tolist()
        ]
        if n_jobs == 1:
            results = (f(*args, **kwargs) for f, args, kwargs in jobs)
        else:
            results = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
        if bar is not None:
            results = bar(results, total=len(jobs), desc=desc)
        splits = []
        for c, out in zip(cands.tolist(), results, strict=True):
            for message in out.warned:
                warnings.warn(message, DataNuggetWarning, stacklevel=3)
            if out.labels is None:
                log.debug("nugget %d not split: %s", c, out.reason)
            else:
                splits.append((c, members[c], out.labels))
        if not splits:
            return False
        state.refresh(state.apply(splits), state.members())
        return True

    if max_splits > 0:
        positive = [v for v in state.ev1.tolist() if math.isfinite(v) and v > 0]
        if not positive:
            log.info("No data nugget has a positive first eigen value")
        else:
            threshold = r_quantile7(positive, ev_tol)
            log.info("Splitting data nuggets according to max EV...")
            for attempt in range(1, max_splits + 1):
                ev1 = state.ev1
                cands = np.flatnonzero(
                    np.isfinite(ev1) & (ev1 > threshold) & (state.weights >= 2 * mns)
                )
                if cands.size == 0:
                    break
                log.info("split #%d", attempt)
                if not run_round(cands, project=False, desc=f"EV split #{attempt}"):
                    break
            log.info("complete!")

    if shape_split and p >= 2 and max_splits > 0:
        log.info("Splitting data nuggets according to shape...")
        min_size = max(min_shape_size, 2 * mns)
        for attempt in range(1, max_splits + 1):
            cands = _shape_candidates(state, min_size, delta)
            if cands.size == 0:
                break
            log.info("split #%d", attempt)
            if not run_round(cands, project=True, desc=f"shape split #{attempt}"):
                break
        log.info("complete")

    log.info("Renumbering the data nuggets...")
    result = state.finalize()
    log.info("completed!")
    return result


@dataclass(frozen=True)
class _Outcome:
    """One split attempt's answer to the main process: each row's child (0 or 1), or None and
    the reason the nugget stays whole, plus the warnings kmeans raised."""

    labels: np.ndarray | None
    reason: str | None
    warned: tuple[str, ...] = ()


def _split_nugget(
    obs: np.ndarray,
    draws: Draws,
    min_nugget_size: float,
    nstart: float,
    project: bool,
    delta: float,
) -> _Outcome:
    """refine.DN's split_nugget, plus the shape check on the children when project is True."""
    if obs.shape[0] < 2 * min_nugget_size:
        return _Outcome(None, "fewer than 2 * min_nugget_size rows")
    data = obs
    if project:
        _, vectors = r_eigen_sym(r_cov(obs), vectors=True)
        data = r_matvec(obs, vectors[:, 0])[:, None]
    try:
        km = kmeans(data, 2, draws, nstart=nstart, iter_max=KMEANS_ITER_MAX)
    except KMeansError as err:
        return _Outcome(None, f"kmeans error: {err}", err.warnings)
    if np.bincount(km.cluster, minlength=2).min() < min_nugget_size:
        return _Outcome(None, "a child is smaller than min_nugget_size", km.warnings)
    if project:
        ratios = [_shape_ratio(obs[km.cluster == g]) for g in (0, 1)]
        if any(math.isnan(r) for r in ratios) or r_vec_mean(ratios) >= delta:
            return _Outcome(None, "mean EV1/EV2 of the children >= delta", km.warnings)
    return _Outcome(km.cluster, None, km.warnings)


def _shape_ratio(obs: np.ndarray) -> float:
    """EV1/EV2 of one child of a shape split: NaN below 2 rows, Inf when EV2 <= 0."""
    if obs.shape[0] < 2:
        return math.nan
    ev = r_eigen_sym(r_cov(obs))
    if ev.size < 2 or ev[1] <= 0:
        return math.inf
    return float(ev[0] / ev[1])


def _shape_candidates(state: _Nuggets, min_size: float, delta: float) -> np.ndarray:
    """Positions big enough to split that are elongated (EV1/EV2 > delta) or spread out (EV1
    above the median plus twice the mad of the scales of nuggets with 2 or more rows)."""
    w, ev1, ev2 = state.weights, state.ev1, state.ev2
    multi = state.scales[np.isfinite(state.scales) & (w > 1)].tolist()
    s_threshold = r_median(multi) + 2 * r_mad(multi) if multi else math.inf
    ratio = np.full(ev1.shape, math.inf)
    np.divide(ev1, ev2, out=ratio, where=np.isfinite(ev2) & (ev2 > 0))
    big = (w >= min_size) & np.isfinite(ev1) & (ev1 > 0)
    return np.flatnonzero(big & ((ratio > delta) | (ev1 > s_threshold)))


class _Nuggets:
    """refine.DN's working state, one entry per position (R's id minus 1). A split appends its
    new nugget at the end, so positions stay stable until finalize() renumbers them."""

    def __init__(self, x: np.ndarray, assignments: np.ndarray, k: int, n_jobs: int = 1):
        self.x = x
        self.n_jobs = n_jobs
        self.assignments = assignments.copy()
        self.centers = np.full((k, x.shape[1]), np.nan)
        self.weights = np.zeros(k, dtype=np.int64)
        self.scales = np.zeros(k)
        self.ev1 = np.zeros(k)
        self.ev2 = np.zeros(k)

    @property
    def size(self) -> int:
        return self.weights.size

    def members(self) -> list[np.ndarray]:
        """R's split(seq_len(n), factor(assignments, levels = ids)): each position's rows in
        ascending order."""
        order = np.argsort(self.assignments, kind="stable")
        counts = np.bincount(self.assignments, minlength=self.size)
        return np.split(order, np.cumsum(counts)[:-1])

    def refresh(self, positions, members: list[np.ndarray]) -> None:
        """refresh_nuggets_info: recompute these positions' center (colMeans), weight, scale
        (the trace of the sample covariance over p) and two largest eigenvalues from their rows.
        An empty nugget has a NaN center, a single row has scale and eigenvalues 0, and with one
        column EV2 = EV1."""
        positions = list(positions)
        p = self.x.shape[1]
        moments = group_moments(self.x, [members[c] for c in positions], self.n_jobs)
        for c, (center, cov) in zip(positions, moments, strict=True):
            self.centers[c] = center
            self.weights[c] = members[c].size
            if cov is None:
                self.scales[c] = self.ev1[c] = self.ev2[c] = 0.0
                continue
            self.scales[c] = r_sum(np.diag(cov)) / p
            ev = cov.ravel() if p == 1 else r_eigen_sym(cov)
            self.ev1[c] = float(ev[0])
            self.ev2[c] = float(ev[1] if ev.size > 1 else ev[0])

    def apply(self, splits: list[tuple[int, np.ndarray, np.ndarray]]) -> list[int]:
        """apply_splits: each split's rows in child 1 (R's cluster 2) move to a new position,
        appended in split order. Returns the parent and child positions to refresh."""
        k, new = self.size, len(splits)
        self.centers = np.vstack([self.centers, np.full((new, self.x.shape[1]), np.nan)])
        self.weights = np.concatenate([self.weights, np.zeros(new, dtype=np.int64)])
        self.scales = np.concatenate([self.scales, np.zeros(new)])
        self.ev1 = np.concatenate([self.ev1, np.zeros(new)])
        self.ev2 = np.concatenate([self.ev2, np.zeros(new)])
        touched = []
        for i, (c, rows, labels) in enumerate(splits):
            self.assignments[rows[labels == 1]] = k + i
            touched += [c, k + i]
        return touched

    def finalize(self) -> DataNuggets:
        """Drop empty nuggets, renumber the rest in position order, zero the scale of
        single-row nuggets and check R's stopifnot() invariants."""
        keep = self.weights > 0
        assignments = (np.cumsum(keep) - 1)[self.assignments]
        weights = self.weights[keep]
        scales = self.scales[keep]
        scales[weights < 2] = 0.0
        if not np.array_equal(np.bincount(assignments, minlength=weights.size), weights):
            raise AssertionError("refine_dn: the weights no longer match the assignments")
        return DataNuggets(
            centers=self.centers[keep], weights=weights, scales=scales, assignments=assignments
        )
