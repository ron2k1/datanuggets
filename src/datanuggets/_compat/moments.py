"""Every nugget's colMeans and cov in one compiled call, as refine.DN needs them.

group_moments gives, group by group, bit for bit what arithmetic's r_mean and r_cov give, and can
spread each chunk of groups over numba threads. One parallel launch runs at a time, and a child
forked after a GNU OpenMP launch runs serially, since numba can't launch threads there.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator

import numba
import numpy as np

from datanuggets._compat import ldjit as _ldjit
from datanuggets._compat.arithmetic import r_cov, r_mean
from datanuggets._defaults import MOMENTS_CHUNK

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
