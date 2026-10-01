"""R's summary statistics as refine.DN and WCluster use them.

sum, mean, median, mad, type-7 quantile, cov, weighted.mean and %*% (a matrix times a vector,
or two matrices) are replayed bit for bit. R accumulates sums in long double (x87 extended
precision on x86-64). cov and group_moments, which gives every nugget's colMeans and cov in one
call, run that arithmetic compiled, in _ldjit. sum and mean replay it with Python integers
through _ldouble, which is also the reference the compiled code is tested against. eigen()
calls LAPACK's dsyevr with R's arguments, so eigenvalues agree with R's to rounding on the
scale of the largest one, not bit for bit: LAPACK builds differ in their kernels. A second
eigenvalue near zero can be far off relative to itself.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterator

import numba
import numpy as np
from scipy.linalg import LinAlgError, lapack

from datanuggets._compat import ldjit as _ldjit
from datanuggets._compat import ldouble as ld
from datanuggets._compat.rcompat import _R_NAN, _non_finite_sum, r_mean
from datanuggets._defaults import MAD_CONSTANT, MOMENTS_CHUNK


def _ld_sum(v: list[float]) -> tuple[int, int]:
    s = ld.ZERO
    for x in v:
        s = ld.add(s, ld.from_float(x))
    return s


_DBL_MAX = (2**53 - 1) << 971  # DBL_MAX as an exact integer


def r_sum(v) -> float:
    """sum() of a double vector, as R's rsum: a long double running sum that's +-Inf once it's
    past +-DBL_MAX, even where rounding would give DBL_MAX back, and otherwise rounds to double
    once. With Inf or NaN in v only those decide."""
    v = [float(x) for x in v]
    if not all(map(math.isfinite, v)):
        return _non_finite_sum(v)
    m, e = _ld_sum(v)
    if abs(m) << max(e, 0) > _DBL_MAX << max(-e, 0):  # |m| * 2**e > DBL_MAX, exactly
        return math.copysign(math.inf, m)
    return ld.to_float((m, e))


def r_vec_mean(v) -> float:
    """mean() of a double vector, as R 4.6.1's real_mean: the long double sum divided by n,
    plus the mean of the residuals from a second pass. If the sum overflows double, R sums
    x / n instead and divides each residual by n before adding it, so neither pass overflows."""
    v = [float(x) for x in v]
    n = len(v)
    if not all(map(math.isfinite, v)):
        return _non_finite_sum(v, quiet=True)  # R's sum of x / n, and no second pass
    s = _ld_sum(v)
    finite_s = math.isfinite(ld.to_float(s))
    if finite_s:
        s = ld.div_int(s, n)
    else:
        s = _ld_sum([x / n for x in v])  # each x / n is rounded to double first, as in R
    if math.isfinite(ld.to_float(s)):
        t = ld.ZERO
        for x in v:
            r = ld.sub(ld.from_float(x), s)
            t = ld.add(t, r if finite_s else ld.div_int(r, n))
        s = ld.add(s, ld.div_int(t, n) if finite_s else t)
    return ld.to_float(s)


def r_median(v) -> float:
    """median(): the middle value, or R's mean() of the two middle values."""
    s = sorted(float(x) for x in v)
    half = (len(s) + 1) // 2
    if len(s) % 2:
        return s[half - 1]
    return r_vec_mean(s[half - 1 : half + 1])


def r_mad(v) -> float:
    """mad() with R's default constant, 1.4826 exactly."""
    c = r_median(v)
    return MAD_CONSTANT * r_median([abs(float(x) - c) for x in v])


def r_quantile7(v, prob: float) -> float:
    """quantile(v, prob) with R's default type 7."""
    s = sorted(float(x) for x in v)
    index = 1 + max(len(s) - 1, 0) * prob
    lo, hi = math.floor(index), math.ceil(index)
    qs = s[lo - 1]
    if index > lo and s[hi - 1] != qs:
        h = index - lo
        qs = (1 - h) * qs + h * s[hi - 1]
    return qs


def r_cov(x: np.ndarray) -> np.ndarray:
    """cov(x): each column mean in two long double passes, stored as a double, then long double
    sums of the products of deviations divided by n - 1. Compiled for finite data. Inf or NaN
    goes to the Python replay, which raises OverflowError or ValueError."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n, p = x.shape
    if n < 2:
        raise ValueError("cov needs at least two rows")
    out = np.empty((p, p))
    rows = np.arange(n, dtype=np.int64)
    if np.isfinite(x).all() and _ldjit.cov_rows(_ldjit.as_bits(x), rows, out):
        return out
    return _r_cov_py(x)


def _r_cov_py(x: np.ndarray) -> np.ndarray:
    """cov(x) replayed with Python integers: r_cov's path for data with Inf or NaN, which it
    rejects as from_float does, and the reference the compiled path is tested against."""
    n, p = x.shape
    if n < 2:
        raise ValueError("cov needs at least two rows")
    dev = []
    for j in range(p):
        col = [ld.from_float(v) for v in x[:, j].tolist()]
        s = ld.ZERO
        for v in col:
            s = ld.add(s, v)
        mean = ld.div_int(s, n)
        if math.isfinite(ld.to_float(mean)):
            s = ld.ZERO
            for v in col:
                s = ld.add(s, ld.sub(v, mean))
            mean = ld.add(mean, ld.div_int(s, n))
        mean = ld.from_float(ld.to_float(mean))  # R keeps the column means as doubles
        dev.append([ld.sub(v, mean) for v in col])
    out = np.empty((p, p))
    for i in range(p):
        for j in range(i + 1):
            s = ld.ZERO
            for a, b in zip(dev[i], dev[j], strict=True):
                s = ld.add(s, ld.mul(a, b))
            out[i, j] = out[j, i] = ld.to_float(ld.div_int(s, n - 1))
    return out


_THREADS = threading.Lock()
_FORKED_UNSAFE = False  # a child forked after a GNU OpenMP launch, which must run serially


def group_moments(
    x: np.ndarray, groups: list[np.ndarray], n_jobs: int = 1
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """(colMeans, cov) of x[g] for each g in groups, in order, bit for bit r_mean(x[g], 0) and
    r_cov(x[g]), with cov None below 2 rows. A group with no rows has R's NaN center. Groups run
    compiled, a chunk at a time, and n_jobs other than 1 spreads each chunk over numba threads
    (all of them for -1). A group with Inf or NaN is redone through r_mean and r_cov. Each g
    indexes rows as x[g] would, so a mask or negative index works and a row past the end raises
    IndexError. A child forked after a GNU OpenMP launch runs serially at any n_jobs, since numba
    can't launch threads there."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    bits = _ldjit.as_bits(x)
    p = x.shape[1]
    per = max(1, MOMENTS_CHUNK // (p * p))
    rows = np.arange(x.shape[0])
    for lo in range(0, len(groups), per):
        chunk = [rows[g] for g in groups[lo : lo + per]]  # numpy reads masks, negatives, bounds
        counts = np.array([g.size for g in chunk], dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        order = np.concatenate(chunk).astype(np.int64)
        centers = np.empty((len(chunk), p))
        covs = np.empty((len(chunk), p, p))
        ok = np.empty(len(chunk), dtype=np.bool_)
        args = (bits, order, starts, counts, centers, covs, ok)
        if n_jobs == 1 or _FORKED_UNSAFE:
            _ldjit.group_moments_serial(*args)
        else:
            _run_threaded(_ldjit.group_moments_parallel, args, n_jobs)
        for i, g in enumerate(chunk):
            if ok[i]:
                yield centers[i], (covs[i] if g.size >= 2 else None)
            else:
                yield r_mean(x[g], 0), (r_cov(x[g]) if g.size >= 2 else None)


def _run_threaded(kernel, args, n_jobs: int) -> None:
    """kernel(*args) on min(n_jobs, NUMBA_NUM_THREADS) numba threads, or all of them for -1.
    One parallel launch at a time: numba's workqueue threading layer aborts the process on
    concurrent launches. The caller's thread count is restored afterwards."""
    cap = numba.config.NUMBA_NUM_THREADS
    threads = cap if n_jobs == -1 else min(n_jobs, cap)
    with _THREADS:
        before = numba.get_num_threads()
        numba.set_num_threads(threads)
        try:
            kernel(*args)
        finally:
            numba.set_num_threads(before)


def _layer_forbids_fork() -> bool:
    """True when this process has launched numba threads on GNU OpenMP, the one layer numba
    can't launch again in a forked child (numba's omppool.cpp raises SIGTERM in the child)."""
    try:
        layer = numba.threading_layer()
    except ValueError:  # no parallel launch yet, so a child inherits nothing
        return False
    if layer != "omp":
        return False
    from numba.np.ufunc import omppool

    return omppool.openmp_vendor == "GNU"


def _after_fork_in_child() -> None:
    """A fork copies _THREADS as it was, held if another thread was launching, but not that
    thread, so nothing in the child would ever release it. The child gets a fresh lock."""
    global _FORKED_UNSAFE, _THREADS
    _THREADS = threading.Lock()
    _FORKED_UNSAFE = _layer_forbids_fork()


if hasattr(os, "register_at_fork"):  # Windows has no fork
    os.register_at_fork(after_in_child=_after_fork_in_child)


def r_matvec(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """a %*% v as R computes it with reference BLAS dgemv: y = 0, then y += a[:, j] * v[j]
    column by column, every product and sum rounded to double."""
    y = np.zeros(a.shape[0])
    for j in range(a.shape[1]):
        y += a[:, j] * v[j]
    return y


def r_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a %*% b as R computes it with reference BLAS dgemm: c = 0, then c += a[:, k] * b[k, :]
    for k in order, every product and sum rounded to double. A sum starts at +0, so no entry
    comes back as -0."""
    c = np.zeros((a.shape[0], b.shape[1]))
    for k in range(a.shape[1]):
        c += np.multiply.outer(a[:, k], b[k, :])
    return c


def r_weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    """weighted.mean(v, w): sum((v * w)[w != 0]) / sum(w), both sums R's long double sum and
    the division in double. A zero total gives NaN or an infinity, as in R. A NaN made by the
    division is the x86 default NaN R gets, whatever this machine's own division gives."""
    with np.errstate(all="ignore"):
        num = r_sum((v * w)[w != 0])
        den = r_sum(w)
        q = float(np.float64(num) / np.float64(den))
    if math.isnan(q) and not (math.isnan(num) or math.isnan(den)):
        return _R_NAN
    return q


def r_eigen_sym(s: np.ndarray, vectors: bool = False):
    """eigen(s, symmetric = TRUE): LAPACK dsyevr on the lower triangle with R's arguments
    (range "A", abstol 0, optimal workspace), values decreasing. With vectors=True, returns
    (values, vectors), the vectors as columns in the same order."""
    s = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError("infinite or missing values in 'x'")
    lwork, liwork, info = lapack.dsyevr_lwork(s.shape[0], lower=1)
    if info == 0:
        w, z, _, _, info = lapack.dsyevr(
            s,
            compute_v=int(vectors),
            range="A",
            lower=1,
            abstol=0.0,
            lwork=int(lwork),
            liwork=int(liwork),
        )
    if info != 0:
        raise LinAlgError(f"error code {info} from Lapack routine 'dsyevr'")
    values = w[::-1].copy()
    if not vectors:
        return values
    return values, z[:, ::-1].copy()
