"""Input coercion and argument checks, mirroring the stop() conditions of the R code."""

from __future__ import annotations

import numbers

import numpy as np


def as_matrix(x, name: str = "x") -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain only numeric columns")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (a matrix or data frame), got {arr.ndim}-D")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column")
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


_LABEL_LIMIT = 1e15  # R's factor() tells labels apart by 15 significant digits


def as_weights(w, n: int, name: str = "w", per: str = "row") -> np.ndarray:
    """One weight per row (or column) as float64. None means ones: R's own default,
    rep(1, length(x)), is one weight per column in wmean and wwcss, which stops unless x is
    square, and one per cell in distw, which stops unless x has one column."""
    if w is None:
        return np.ones(n)
    arr = np.asarray(w)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric")
    if arr.shape != (n,):
        raise ValueError(f"{name} must have one weight per {per} ({n}), got shape {arr.shape}")
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_labels(cl, n: int, name: str = "cl") -> tuple[np.ndarray, np.ndarray]:
    """The sorted distinct labels and each row's 0-based position among them. Labels are whole
    numbers smaller than 1e15 in size: the labels R's factor() and sort(unique()) group and
    order alike, which is how wwcss and distw read them."""
    arr = np.asarray(cl)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold numeric cluster labels")
    if arr.shape != (n,):
        raise ValueError(f"{name} must have one label per row ({n}), got shape {arr.shape}")
    if n == 0:
        raise ValueError(f"{name} is empty: there are no rows to cluster")
    size = np.abs(arr.astype(np.float64))
    if not np.isfinite(size).all() or not np.array_equal(size, np.floor(size)):
        raise ValueError(f"{name} must hold whole numbers")
    if size.max() >= _LABEL_LIMIT:
        raise ValueError(f"{name} must hold labels smaller than 1e15 in size")
    levels, codes = np.unique(arr, return_inverse=True)
    return levels, codes.reshape(-1).astype(np.intp)


def check_choice(name: str, value, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, choices))}, got {value!r}")
    return value


def as_real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def as_count(name: str, value) -> int:
    v = as_real(name, value)
    if not v.is_integer() or v < 1:
        raise ValueError(f"{name} must be a whole number >= 1, got {value!r}")
    return int(v)


def check_delete_percent(value) -> float:
    v = as_real("delete_percent", value)
    if not 0 < v < 1:
        raise ValueError(f"delete_percent must be within (0, 1), got {value!r}")
    return v


def check_assignments(dn, n: int, p: int) -> np.ndarray:
    """dn's assignments as 0-based nugget ids (intp), after checking that dn describes n rows
    with p columns."""
    if dn.centers.ndim != 2 or dn.centers.shape[1] != p:
        raise ValueError(f"dn has centers of shape {dn.centers.shape}, but x has {p} columns")
    k = dn.centers.shape[0]
    a = dn.assignments
    if a.shape != (n,):
        raise ValueError(
            f"dn.assignments must have one entry per row of x ({n}), got shape {a.shape}"
        )
    if a.dtype.kind not in "iu" and (a.dtype.kind != "f" or not np.array_equal(a, np.floor(a))):
        raise ValueError("dn.assignments must be whole numbers (nugget ids)")
    if a.size and (a.min() < 0 or a.max() >= k):
        raise ValueError(
            f"dn.assignments must be nugget ids in [0, {k}), got values from {a.min()} to {a.max()}"
        )
    return a.astype(np.intp)


def check_n_jobs(n_jobs) -> int:
    """joblib's n_jobs: a positive int, or -1 for every core."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be a positive int or -1, got {n_jobs!r}")
    return n_jobs
