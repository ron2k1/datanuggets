"""Cluster colours for any number of clusters, as lowercase "#rrggbb" hex strings.

Up to CLUSTER_PARENT_K clusters take matplotlib's default colours (tab10) as they are, in the
order of its default colour cycle. More clusters are grouped in families, by the cluster of the
tree cut into CLUSTER_PARENT_K they fall in (tree_parents) or in blocks of neighbouring cluster
numbers (block_parents, for k-means, which has no tree). Family j's first cluster keeps TAB10[j],
and its others take colours near TAB10[j], each picked as far as it can be from the colours
already given (CIEDE2000), so siblings stay apart from each other and from other families, every
colour reads on a light and a dark background, and no two clusters share one. Needs no
matplotlib."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from datanuggets._defaults import (
    CLUSTER_BACKGROUNDS,
    CLUSTER_GRID_LEVELS,
    CLUSTER_MIN_CONTRAST,
    CLUSTER_OTHER_DE00,
    CLUSTER_PARENT_K,
    CLUSTER_SCORE_DECIMALS,
    CLUSTER_SIBLING_DE00,
)

if TYPE_CHECKING:
    from datanuggets.result import WhclustResult

# matplotlib's default colour cycle, tab10, as lowercase hex
TAB10: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
PARENT_K = CLUSTER_PARENT_K

# sRGB (IEC 61966-2-1) linear RGB to CIE XYZ under D65; each row's sum is the white point's
_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_25_7 = 25.0**7


def _check_k(k) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a whole number from 1, got {k!r}")
    return k


def _rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def _linear(rgb) -> np.ndarray:
    """sRGB channels in 0..1 as linear light."""
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _luminance(rgb) -> np.ndarray:
    """WCAG 2's relative luminance of sRGB channels in 0..1."""
    return _linear(rgb) @ _XYZ[1]


def _lab(rgb) -> np.ndarray:
    """CIELAB (D65) of sRGB channels in 0..1, one colour per row."""
    t = (_linear(rgb) @ _XYZ.T) / _XYZ.sum(axis=1)
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t * (29 / 6) ** 2 / 3 + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _de2000(lab1, lab2) -> np.ndarray:
    """The CIEDE2000 colour difference of CIELAB colours, row by row with numpy broadcasting,
    kL = kC = kH = 1, as Sharma, Wu and Dalal (2005) state it."""
    lab1, lab2 = np.asarray(lab1, dtype=float), np.asarray(lab2, dtype=float)
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    c7 = c**7
    g1 = 1.5 - 0.5 * np.sqrt(c7 / (c7 + _25_7))  # 1 + G
    a1p, a2p = g1 * a1, g1 * a2
    c1, c2 = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    gray = c1 * c2 == 0
    dh = h2 - h1
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    d_h = np.where(gray, 0.0, 2 * np.sqrt(c1 * c2) * np.sin(np.radians(dh / 2)))
    hsum = h1 + h2
    h = np.where(
        np.abs(h1 - h2) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)
    )
    h = np.where(gray, hsum, h)
    t = (
        1
        - 0.17 * np.cos(np.radians(h - 30))
        + 0.24 * np.cos(np.radians(2 * h))
        + 0.32 * np.cos(np.radians(3 * h + 6))
        - 0.20 * np.cos(np.radians(4 * h - 63))
    )
    lbar2 = ((l1 + l2) / 2 - 50) ** 2
    cbar = (c1 + c2) / 2
    cb7 = cbar**7
    r_t = (
        -2
        * np.sqrt(cb7 / (cb7 + _25_7))
        * np.sin(np.radians(60 * np.exp(-(((h - 275) / 25) ** 2))))
    )
    d_l = (l2 - l1) / (1 + 0.015 * lbar2 / np.sqrt(20 + lbar2))
    d_c = (c2 - c1) / (1 + 0.045 * cbar)
    d_h = d_h / (1 + 0.015 * cbar * t)
    return np.sqrt(d_l * d_l + d_c * d_c + d_h * d_h + r_t * d_c * d_h)


def _hex(rgb8) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in rgb8)


class _Pool:
    """The colours a family can take: the sRGB colours whose channels each take one of levels
    evenly spaced 8-bit values (round(255 i / (levels - 1))), whose WCAG contrast against every
    one of CLUSTER_BACKGROUNDS is at least CLUSTER_MIN_CONTRAST, and that aren't a family's own
    tab10 colour; in r, g, b order. Each one's region is the family whose tab10 colour is nearest
    it by CIEDE2000."""

    def __init__(self, levels: int):
        v = np.unique(np.round(np.linspace(0, 255, levels)).astype(np.int64))
        grid = np.stack(np.meshgrid(v, v, v, indexing="ij"), axis=-1).reshape(-1, 3)
        y = _luminance(grid / 255)
        keep = np.ones(len(grid), dtype=bool)
        for background in CLUSTER_BACKGROUNDS:
            yb = float(_luminance(_rgb(background)))
            keep &= (np.maximum(y, yb) + 0.05) / (np.minimum(y, yb) + 0.05) >= CLUSTER_MIN_CONTRAST
        bases = {int(c[1:], 16) for c in TAB10[:PARENT_K]}
        code = (grid[:, 0] << 16) | (grid[:, 1] << 8) | grid[:, 2]
        keep &= ~np.isin(code, list(bases))
        self.rgb8 = grid[keep]
        self.lab = _lab(self.rgb8 / 255)
        self.bases = _lab(np.array([_rgb(c) for c in TAB10[:PARENT_K]]))
        near = np.column_stack([_de2000(self.lab, b) for b in self.bases])
        self.region = np.argmin(np.round(near, CLUSTER_SCORE_DECIMALS), axis=1)

    def __len__(self) -> int:
        return len(self.rgb8)


@functools.cache
def _pool(levels: int) -> _Pool:
    return _Pool(levels)


def _pool_for(k: int) -> _Pool:
    """The pool at CLUSTER_GRID_LEVELS, or at twice as many levels until it holds k - PARENT_K
    colours. Levels stop at 256."""
    levels = CLUSTER_GRID_LEVELS
    while len(_pool(levels)) + PARENT_K < k and levels < 256:
        levels = min(2 * levels, 256)
    return _pool(levels)


@functools.lru_cache(maxsize=16)
def _family_colors(parents: tuple[int, ...]) -> tuple[str, ...]:
    """cluster_colors for more than PARENT_K clusters, given each one's family.

    Family j's first cluster takes TAB10[j]. Then, one colour at a time, the family with the
    smallest share of its clusters coloured (the lowest-numbered family on a tie) takes, from the
    pool colours in its own region not yet given, the one with the highest score
    min(d_own / CLUSTER_SIBLING_DE00, d_other / CLUSTER_OTHER_DE00): d_own is its least CIEDE2000
    to the colours of the family whose region holds it, d_other to every other family's. Scores
    are rounded to CLUSTER_SCORE_DECIMALS and a tie goes to the first in pool order, so a
    last-bit difference in a platform's arithmetic can't change a pick. A family whose region is
    used up takes the pool colour farthest from every colour given. Family j's colours go to its
    clusters in cluster order, in the order they were picked."""
    k = len(parents)
    need = np.bincount(np.asarray(parents, dtype=np.int64), minlength=PARENT_K)
    pool = _pool_for(k)
    near_own = np.full(len(pool), math.inf)
    near_other = np.full(len(pool), math.inf)
    free = np.ones(len(pool), dtype=bool)
    picked: list[list[str]] = [[] for _ in range(PARENT_K)]

    def give(j: int, lab: np.ndarray) -> None:
        d = _de2000(pool.lab, lab)
        own = pool.region == j
        np.minimum(near_own, d, out=near_own, where=own)
        np.minimum(near_other, d, out=near_other, where=~own)

    for j in range(PARENT_K):
        if need[j]:
            picked[j].append(TAB10[j])
            give(j, pool.bases[j])
    while True:
        short = [j for j in range(PARENT_K) if len(picked[j]) < need[j]]
        if not short:
            break
        j = min(short, key=lambda f: (len(picked[f]) / need[f], f))
        allowed = free & (pool.region == j)
        if allowed.any():
            score = np.minimum(near_own / CLUSTER_SIBLING_DE00, near_other / CLUSTER_OTHER_DE00)
        else:
            allowed, score = free, np.minimum(near_own, near_other)
        score = np.where(allowed, np.round(score, CLUSTER_SCORE_DECIMALS), -math.inf)
        i = int(np.argmax(score))
        free[i] = False
        picked[j].append(_hex(pool.rgb8[i]))
        give(j, pool.lab[i])
    taken = [0] * PARENT_K
    colors = []
    for p in parents:
        colors.append(picked[p][taken[p]])
        taken[p] += 1
    return tuple(colors)


def block_parents(k: int) -> list[int]:
    """Each of k clusters' family when there is no tree: neighbouring cluster numbers share one,
    c * PARENT_K // k, for more than PARENT_K clusters; each cluster is its own for PARENT_K or
    fewer."""
    k = _check_k(k)
    if k <= PARENT_K:
        return list(range(k))
    return [c * PARENT_K // k for c in range(k)]


def tree_parents(tree: WhclustResult, k: int) -> list[int]:
    """Each of the k clusters cut_tree(tree, k=k) gives, its parent: the cluster of the tree cut
    into PARENT_K that its members fall in. The cuts nest, so every member has the same one.
    Each cluster is its own parent for k up to PARENT_K."""
    from datanuggets.wcluster.cutree import cut_tree

    k = _check_k(k)
    if k <= PARENT_K:
        return list(range(k))
    child = cut_tree(tree, k=k)
    parent = cut_tree(tree, k=PARENT_K)
    found = [-1] * k
    for c, p in zip(child.tolist(), parent.tolist(), strict=True):
        found[c] = p
    return found


def cluster_colors(k: int, parents: Sequence[int] | None = None) -> list[str]:
    """k lowercase "#rrggbb" colours, entry c for cluster c (from 0). Up to PARENT_K clusters:
    TAB10[:k], parents ignored. More: parents (block_parents(k) by default) gives each cluster's
    family, 0 to PARENT_K - 1. Family j's first cluster takes TAB10[j]; its others take colours
    near it, picked one at a time as far as they can be from those already given (see
    _family_colors), every one readable on CLUSTER_BACKGROUNDS. Every colour differs when parents
    names all PARENT_K families, as block_parents and tree_parents do. The same parents give the
    same colours. Each pick scores the whole pool, so the time grows faster than k. The pool also
    grows in steps of about 8 times: 6,636 colours up to 6,644 clusters, 53,595 up to 53,603, and
    more beyond that. One laptop took 0.1 s for 100 clusters, 1 s for 1,000, 3.5 s for 3,000 and
    11 s for 6,644, then 2 minutes for 6,645."""
    k = _check_k(k)
    if k <= PARENT_K:
        return list(TAB10[:k])
    parents = block_parents(k) if parents is None else [int(p) for p in parents]
    if len(parents) != k:
        raise ValueError(f"parents must give one family per cluster, {k}, got {len(parents)}")
    if any(not 0 <= p < PARENT_K for p in parents):
        raise ValueError(f"parents must be family numbers from 0 to {PARENT_K - 1}")
    return list(_family_colors(tuple(parents)))
