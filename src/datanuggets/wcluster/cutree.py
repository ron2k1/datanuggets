"""cut_tree: R's cutree for one k or one h, on a whclust tree."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import DTypeLike

from datanuggets.result import WhclustResult

# R's message when an if() meets NA: as.integer(NaN), is.unsorted() of NaN heights
NA_STOP = "missing value where TRUE/FALSE needed"
# cutree's h = NaN: which.max of all NA is empty, so k is empty and colnames<- fails
H_NAN_STOP = "length of 'dimnames' [2] not equal to array extent"
INT_EDGE = 2.0**31  # as.integer gives NA at or beyond +-2^31


def r_merge(merge: np.ndarray, dtype: DTypeLike = np.int64) -> np.ndarray:
    """R's merge numbering from the port's, in dtype: row a below n is -(a + 1), and the cluster
    merge s made is s + 1."""
    m = np.asarray(merge, dtype=dtype)
    n = m.shape[0] + 1
    return np.where(m < n, -(m + 1), m - n + 1)


def _scalar(v, name: str) -> float:
    if np.ndim(v) != 0:
        raise ValueError(f"cut_tree takes one k or one h, not an array of {name}")
    return float(v)


def _k(k, n: int) -> int:
    """as.integer(k), then cutree's range check."""
    k = _scalar(k, "k")
    if math.isnan(k) or k >= INT_EDGE or k <= -INT_EDGE:
        raise ValueError(NA_STOP)
    k = int(k)  # truncates toward zero, as as.integer does
    if k < 1 or k > n:
        raise ValueError(f"elements of 'k' must be between 1 and {n}")
    return k


def _k_from_h(height: np.ndarray, h, n: int) -> int:
    """cutree's k for a height: n + 1 minus the 1-based position of the first entry of
    c(height, Inf) above h."""
    if np.isnan(height).any():
        raise ValueError(NA_STOP)
    if np.any(height[1:] < height[:-1]):
        raise ValueError("the 'height' component of 'tree' is not sorted (increasingly)")
    h = _scalar(h, "h")
    if math.isnan(h):
        raise ValueError(H_NAN_STOP)
    above = np.append(height, np.inf) > h
    return n - int(np.argmax(above))  # argmax of all False is 0, as which.max gives 1


def _cutree(rm: np.ndarray, k: int) -> np.ndarray:
    """R's C cutree for one k on R's merge, 0-based. Each row keeps whether it's still alone and
    the last merge step that took it in. The masks visit the rows R's inner loops visit."""
    n = rm.shape[0] + 1
    if k == n:
        return np.arange(n, dtype=np.int64)
    alone = np.ones(n, dtype=bool)
    step_of = np.zeros(n, dtype=np.int64)
    for step in range(1, n - k + 1):
        m1, m2 = int(rm[step - 1, 0]), int(rm[step - 1, 1])
        if m1 < 0 and m2 < 0:
            step_of[[-m1 - 1, -m2 - 1]] = step
            alone[[-m1 - 1, -m2 - 1]] = False
        elif m1 < 0 or m2 < 0:
            row, other = (-m1 - 1, m2) if m1 < 0 else (-m2 - 1, m1)
            step_of[step_of == other] = step
            step_of[row] = step
            alone[row] = False
        else:
            step_of[(step_of == m1) | (step_of == m2)] = step
    # Number the clusters by first appearance along the rows, a lone row its own cluster
    key = np.where(alone, -np.arange(1, n + 1), step_of)
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(first.size)
    return rank[inverse.ravel()]


def cut_tree(tree: WhclustResult, k=None, h=None) -> np.ndarray:
    """R's cutree(tree, k, h): each row's cluster when the tree is cut into k clusters, or at
    height h, as R's label minus one. k wins when both are given, and a fractional k is
    truncated as R's as.integer does. Raises ValueError with R's message where R stops."""
    n1 = tree.merge.shape[0]
    if n1 < 1:
        raise ValueError("invalid 'tree' ('merge' component)")
    n = n1 + 1
    if k is None and h is None:
        raise ValueError("either 'k' or 'h' must be specified")
    k = _k(k, n) if k is not None else _k_from_h(np.asarray(tree.height, dtype=float), h, n)
    return _cutree(r_merge(tree.merge), k)
