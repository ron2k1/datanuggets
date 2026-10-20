"""Weighted PCA views: scores, the scree and the loadings, with each component's sign fixed so
its largest loading is positive. WpcaResult keeps R's signs, only the views flip."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from datanuggets.plot.points import _dims, _label_colors, _labels, _matrix, _palette, _sizes

MAX_ROWS = 50_000  # raw rows drawn behind the scores, every ceil(n / MAX_ROWS)-th


def _signs(rotation) -> np.ndarray:
    """+1 or -1 per component: -1 where the loading largest by absolute value is negative."""
    r = np.asarray(rotation, dtype=float)
    big = r[np.argmax(np.abs(r), axis=0), np.arange(r.shape[1])]
    return np.where(big < 0, -1.0, 1.0)


def _share(wpca) -> np.ndarray:
    """Each component's percent of the variance. An eigenvalue rounded just below zero gives a
    NaN sdev, counted as 0."""
    lam = np.nan_to_num(np.asarray(wpca.sdev, dtype=float) ** 2)
    return 100.0 * lam / lam.sum()


def _plane(wpca, dims) -> tuple[int, int]:
    q = np.asarray(wpca.rotation).shape[1]
    if q < 2:
        raise ValueError("a PCA plane needs at least two components")
    return _dims(dims, q)


def scores(wpca, *, w=None, labels=None, rows=None, dims=(0, 1), colors=None, ax=None):
    """The scores on components dims, sized by w (area proportional to weight) and colored by
    labels when given, else all in the first color. rows, raw data, are drawn behind in light
    gray, projected as the scores were: (rows - center) @ rotation, at most 50,000 of them. The
    axis labels give each component's share of the variance. Returns the Axes."""
    i, j = _plane(wpca, dims)
    s = _signs(wpca.rotation)
    x = np.asarray(wpca.x, dtype=float) * s
    n = x.shape[0]
    # One color per point: the cycle's colors are RGB tuples, and a lone tuple as scatter's c
    # is value-mapped through a colormap when there are 3 points
    labels = np.zeros(n, dtype=np.int64) if labels is None else _labels(labels, n)
    face = _label_colors(labels, _palette(colors))
    size = _sizes(w, n, "area")
    proj = None
    if rows is not None:
        r = _matrix(rows)
        center = np.asarray(wpca.center, dtype=float)
        if r.shape[1] != center.size:
            raise ValueError(f"rows must have {center.size} columns, one per variable")
        r = r[:: max(1, math.ceil(r.shape[0] / MAX_ROWS))]
        proj = (r - center) @ (np.asarray(wpca.rotation, dtype=float) * s)[:, [i, j]]
    ax = plt.gca() if ax is None else ax
    if proj is not None:
        ax.scatter(proj[:, 0], proj[:, 1], s=2, c="0.8", linewidths=0, rasterized=True, gid="rows")
    ax.scatter(x[:, i], x[:, j], s=size, c=face, linewidths=0, gid="scores")
    share = _share(wpca)
    ax.set_xlabel(f"PC{i + 1} ({share[i]:.1f}%)")
    ax.set_ylabel(f"PC{j + 1} ({share[j]:.1f}%)")
    return ax


def scree(wpca, *, ax=None):
    """Each component's percent of the variance as a bar. Returns the Axes."""
    share = _share(wpca)
    ax = plt.gca() if ax is None else ax
    k = np.arange(1, share.size + 1)
    ax.bar(k, share, color=_palette(None)[0], gid="scree")
    ax.set_xticks(k)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance explained (%)")
    return ax


def loadings(wpca, *, names=None, dims=(0, 1), ax=None):
    """An arrow from the origin to each variable's loadings on components dims, labelled with
    names or the 0-based column numbers. Returns the Axes."""
    i, j = _plane(wpca, dims)
    rot = np.asarray(wpca.rotation, dtype=float) * _signs(wpca.rotation)
    p = rot.shape[0]
    text = [str(v) for v in (range(p) if names is None else names)]
    if len(text) != p:
        raise ValueError(f"names must have {p} entries, one per variable")
    ax = plt.gca() if ax is None else ax
    ax.axhline(0, color="0.85", linewidth=0.5, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.5, zorder=0)
    ax.quiver(
        np.zeros(p),
        np.zeros(p),
        rot[:, i],
        rot[:, j],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        gid="loadings",
    )
    for k in range(p):
        ax.text(1.1 * rot[k, i], 1.1 * rot[k, j], text[k], ha="center", va="center")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC{i + 1}")
    ax.set_ylabel(f"PC{j + 1}")
    return ax
