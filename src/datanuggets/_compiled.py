"""numba's njit, with its on-disk cache wherever numba has somewhere to put it, and the read-only
views every kernel input goes through."""

from __future__ import annotations

import numpy as np
from numba import njit


def compiled(**options):
    """njit(cache=True, **options), or plain njit(**options) when numba finds no writable cache
    location (a read-only install whose user has no writable home). Without the cache every run
    compiles its kernels again on first call, which is slower to start but gives the same bits."""

    def decorate(fn):
        try:
            return njit(cache=True, **options)(fn)
        except RuntimeError as e:
            if "no locator available" not in str(e):
                raise
            return njit(**options)(fn)

    return decorate


def frozen(a: np.ndarray) -> np.ndarray:
    """A read-only view of a. numba compiles a separate kernel for read-only arrays, so making every
    input read-only keeps one compiled kernel for writable and read-only callers alike."""
    view = a.view()
    view.flags.writeable = False
    return view
