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


def check_n_jobs(n_jobs) -> int:
    """joblib's n_jobs: a positive int, or -1 for every core."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be a positive int or -1, got {n_jobs!r}")
    return n_jobs
