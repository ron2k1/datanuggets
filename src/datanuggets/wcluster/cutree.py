# R cutree adaptations; upstream notices are in THIRD_PARTY.md.
# Copyright (C) 1995-2025 The R Core Team.
# Python adaptation by Ronil Basu, recorded 2026-10-21.
# GPL version 2; supplied without warranty. See LICENSE and THIRD_PARTY.md.

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
    truncated as R's as.integer does. Raises ValueError with R's message where R stops.

    With h, the cluster count is n + 1 minus the 1-based position of the first height above
    h, with a last height of infinity appended. So an h below the first height gives n
    clusters, a finite h at or above the last height gives 1, and an infinite h gives n.
    Clusters are numbered by first appearance in row order, not dendrogram order.

    Parameters
    ----------
    tree : WhclustResult
        The tree to cut. Only ``tree.merge`` and ``tree.height`` are read, so any object with
        those works if its ``merge`` is a NumPy array in scipy's numbering.
    k : float, optional
        One cluster count. It is truncated toward zero and must then be in [1, n].
    h : float, optional
        One cut height, in the units of ``tree.height``. Used only when ``k`` is None.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Each leaf's cluster as int64, R's label minus one. For a tree of nuggets n is the
        number of nuggets.

    Raises
    ------
    ValueError
        With R's message where R stops. A ``k`` that is NaN or at least 2^31 in size,
        infinities included, gives "missing value where TRUE/FALSE needed". That check comes
        before truncation. Any other ``k`` is truncated toward zero, and a result outside
        [1, n] gives "elements of 'k' must be between 1 and n". R's other stops are for
        neither ``k`` nor ``h`` given, a NaN ``h``, a cut by height on NaN or decreasing
        heights, and a tree with an empty merge. Also if ``k`` or ``h`` is an array or a
        string that ``float()`` can't parse.
    TypeError
        If ``k`` or ``h`` is another type that ``float()`` rejects.

    See Also
    --------
    whclust, dn_whclust

    Notes
    -----
    R counterpart: ``stats::cutree``, as R 4.6.1 runs it. Differences from R:

    - ``k`` and ``h`` take one value each, and an array raises. R takes vectors and returns
      one column per value.
    - Cluster numbers are R's minus one, and the result is an int64 array with no names. R's
      cutree names its result by the tree's ``labels``, so an hclust tree of named rows gives
      a named vector. R's Whclust sets ``labels`` to NULL even for named rows, so R's cutree
      of a Whclust tree has no names either. WhclustResult has no ``labels``.

    References
    ----------
    .. [1] Becker, R. A., Chambers, J. M., and Wilks, A. R. *The New S Language*. Chapman and
       Hall/CRC, London, 1988. ISBN 053409192X.

    Examples
    --------
    >>> import numpy as np
    >>> from datanuggets import cut_tree, whclust
    >>> x = np.array([[0, 0], [1, 0], [0, 3], [10, 10], [12, 10], [10, 14]], dtype=float)
    >>> tree = whclust(x, np.array([1, 2, 1, 1, 1, 2]))
    >>> cut_tree(tree, k=2).tolist()
    [0, 0, 0, 1, 1, 1]
    >>> cut_tree(tree, k=3).tolist()
    [0, 0, 0, 1, 1, 2]
    >>> [round(float(v), 4) for v in tree.height]
    [0.6667, 2.6667, 9.75, 26.75, 479.875]
    >>> cut_tree(tree, h=20).tolist()
    [0, 0, 0, 1, 1, 2]
    """
    n1 = tree.merge.shape[0]
    if n1 < 1:
        raise ValueError("invalid 'tree' ('merge' component)")
    n = n1 + 1
    if k is None and h is None:
        raise ValueError("either 'k' or 'h' must be specified")
    k = _k(k, n) if k is not None else _k_from_h(np.asarray(tree.height, dtype=float), h, n)
    return _cutree(r_merge(tree.merge), k)
