"""dendrogram: R's plot.hclust for a whclust tree, line for line."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from datanuggets.wcluster.cutree import INT_EDGE, NA_STOP, r_merge

BAD_INPUT = "invalid dendrogram input"  # C_dend's and C_dendwindow's one message
# .validity.hclust's seq_len(n - 2) for a tree with no merges
NO_MERGE_STOP = "argument must be coercible to non-negative integer"
# The port's own: R warns and draws such a tree off the page (README departure 5)
NONFINITE_STOP = "the tree has NaN or infinite heights, which R's plot.hclust draws off the page"


def _r_merge_checked(tree) -> np.ndarray:
    """.validity.hclust's checks in R's order, then C_dendwindow's: R's merge, or ValueError
    with R's message."""
    merge = np.asarray(tree.merge)
    if merge.ndim != 2 or merge.shape[1] != 2:
        raise ValueError("invalid dendrogram")
    # any(as.integer(merge) != merge) on R's merge in doubles: TRUE once an entry as.integer can
    # hold is fractional, and otherwise NA for NaN, Inf or an entry at 2^31 or beyond
    rf = r_merge(merge, float)
    held = np.isfinite(rf) & (np.abs(rf) < INT_EDGE)
    if np.any(held & (rf != np.trunc(rf))):
        raise ValueError("'merge' component in dendrogram must be integer")
    if not held.all():
        raise ValueError(NA_STOP)
    n1 = merge.shape[0]
    n = n1 + 1
    if np.size(tree.height) != n1:
        raise ValueError("'height' is of wrong length")
    if np.size(tree.order) != n:
        raise ValueError("'order' is of wrong length")
    if n1 == 0:
        raise ValueError(NO_MERGE_STOP)
    rm = rf.astype(np.int64)
    want = np.concatenate([np.arange(-n, 0), np.arange(1, n - 1)])
    if not np.array_equal(np.sort(rm, axis=None), want):
        raise ValueError("'merge' matrix has invalid contents")
    if n1 < 2:
        raise ValueError(BAD_INPUT)
    return rm


def _check_heights(height: np.ndarray) -> None:
    """Whclust gives NaN heights for zero weights and Inf ones for infinite weights. R's
    plot.hclust only warns on them and writes coordinates no figure can show, so the port stops."""
    if not np.isfinite(height).all():
        raise ValueError(NONFINITE_STOP)


def _drawdend(rm, height, xpos, hang):
    """C_dend's drawdend from the last merge, as an explicit stack: the left subtree, the right
    subtree, then the merge's polyline. A leaf hangs hang below its merge, or sits at 0 when
    hang is negative. Returns the polylines in R's drawing order and each leaf's (row, x, y)."""
    lines, leaves, anchor = [], [], {}
    stack = [(rm.shape[0], False)]
    while stack:
        node, ready = stack.pop()
        left, right = int(rm[node - 1, 0]), int(rm[node - 1, 1])
        if not ready:
            stack.append((node, True))
            stack.extend((k, False) for k in (right, left) if k > 0)  # left pops first
            continue
        y = float(height[node - 1])
        ends = []
        for k in (left, right):
            if k > 0:
                ends.append(anchor.pop(k))
            else:
                end = (float(xpos[-k - 1]), y - hang if hang >= 0 else 0.0)
                leaves.append((-k - 1, *end))
                ends.append(end)
        (xl, yl), (xr, yr) = ends
        lines.append(((xl, yl), (xl, y), (xr, y), (xr, yr)))
        anchor[node] = (0.5 * (xl + xr), y)
    return lines, leaves


def _leaf_text(labels, n: int):
    if labels is None:
        return [str(i) for i in range(n)]
    if np.ndim(labels) == 0:
        # R's FALSE draws none. Anything else single, a string too, is one label for n leaves
        if isinstance(labels, (bool, np.bool_)) and not labels:
            return None
        raise ValueError(BAD_INPUT)
    text = [str(v) for v in labels]
    if len(text) != n:
        raise ValueError(BAD_INPUT)
    return text


def dendrogram(tree, *, labels=None, hang=0.1, ax=None):
    """Draws a whclust tree as R's plot(tree) does: leaf i at x = 1 + its place in tree.order,
    each merge a bracket at its height, leaves hanging hang times the last height minus the
    first below their merge, or at 0 when that's negative. labels are the 0-based row numbers by
    default, a sequence of n strings, or False for none. Only the y axis is drawn, labelled
    "Height". Raises ValueError with R's message where plot.hclust stops, such as for a tree
    with a self-merge, and for a tree with NaN or infinite heights, which R draws off the page.
    Returns the Axes."""
    rm = _r_merge_checked(tree)
    n = rm.shape[0] + 1
    hang = float(hang)
    if not math.isfinite(hang):
        raise ValueError(BAD_INPUT)
    text = _leaf_text(labels, n)
    height = np.asarray(tree.height, dtype=float)
    _check_heights(height)
    xpos = np.argsort(np.asarray(tree.order), kind="stable") + 1.0  # R's order(x$order)
    lines, leaves = _drawdend(rm, height, xpos, hang * (height[-1] - height[0]))
    ax = plt.gca() if ax is None else ax
    rc = plt.rcParams
    ax.add_collection(
        LineCollection(
            lines, colors=rc["axes.edgecolor"], linewidths=rc["lines.linewidth"], gid="dendrogram"
        )
    )
    ax.autoscale_view()
    if text is not None:
        for row, x, y in leaves:
            ax.annotate(
                text[row],
                (x, y),
                xytext=(0, -2),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="top",
                fontsize=rc["xtick.labelsize"],
                annotation_clip=False,
            )
    ax.set_ylabel("Height")
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    return ax
