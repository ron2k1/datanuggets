"""The datanuggets-flow figures, each drawn on a figure from figure(figsize=...). The default,
matplotlib's Figure, is held by no pyplot window, so a backend can draw run after run and leave
nothing open; the CLI passes plt.figure, so its figures show. They return the figures unsaved, and
save_figures writes them. Needs matplotlib, which the flow extra installs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from datanuggets import plot, wpca
from datanuggets.plot.points import _label_colors, _palette


def _panels(n: int, size: tuple[float, float], figure) -> tuple[Figure, np.ndarray]:
    """A figure of n panels of size inches, two across, and its axes in reading order, those past
    the n-th hidden."""
    cols = min(n, 2)
    rows = -(-n // cols)
    fig = (figure or Figure)(figsize=(size[0] * cols, size[1] * rows))
    axs = fig.subplots(rows, cols, squeeze=False).ravel()
    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs


def local_maxima_figure(samples: Sequence[Mapping], *, figure: Callable | None = None) -> Figure:
    """local_maxima.png, the figure the populations are chosen from: one panel per file of its
    density grid, local maxima and the peak each Pop grows from (plot.peaks of its "peaks")."""
    fig, axs = _panels(len(samples), (6.5, 5.2), figure)
    for i, s in enumerate(samples):
        plot.peaks(s["peaks"], title=f"File {i + 1}: {s['path'].name}", ax=axs[i])
    fig.tight_layout()
    return fig


def gates_figure(samples: Sequence[Mapping], *, figure: Callable | None = None) -> Figure:
    """gates.png: one panel per file of its events in gray and each of its populations in color
    (plot.gates of its "fsc", "ssc" and "masks")."""
    fig, axs = _panels(len(samples), (5.5, 4.8), figure)
    for i, s in enumerate(samples):
        title = f"File {i + 1}: {s['path'].name}"
        plot.gates(s["fsc"], s["ssc"], s["masks"], title=title, ax=axs[i])
    fig.tight_layout()
    return fig


def result_figures(
    pool,
    centers,
    labels: Sequence[str],
    cofactor: float,
    tsne_xy,
    *,
    figure: Callable | None = None,
) -> dict[str, Figure]:
    """The clustering's figures by file name, in the order a run saves them: dendrogram.png,
    clusters_pca.png, clusters_tsne.png (left out when tsne_xy, the t-SNE of the nuggets in use,
    is None) and cluster_centers.png. centers is the cluster_centers table, one column per label."""
    new = figure or Figure
    use, k = pool.use, pool.k
    figs = {}
    fig = new(figsize=(8, 4))
    ax = fig.subplots()
    plot.dendrogram(pool.tree, labels=False, hang=-1, ax=ax)
    ax.set_title(f"Whclust of {use.sum():,} nuggets")
    figs["dendrogram.png"] = fig
    fig = new(figsize=(6, 6))
    ax = fig.subplots()
    pca = wpca(pool.centers[use], wrow=pool.weights[use])
    plot.scores(pca, w=pool.weights[use], labels=pool.cl[use], ax=ax)
    ax.set_title(f"Nuggets of the kept cells, {k} clusters")
    figs["clusters_pca.png"] = fig
    if tsne_xy is not None:
        fig = new(figsize=(6.5, 6))
        ax = fig.subplots()
        plot.clusters(tsne_xy, pool.cl[use], w=pool.weights[use], size="sqrt", ax=ax)
        colors = _label_colors(range(k), _palette(None))  # plot.clusters' own, by construction
        handles = [
            Line2D([], [], ls="none", marker="o", color=color, label=str(c + 1))
            for c, color in enumerate(colors)
        ]
        ax.legend(handles=handles, title="cluster", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
        ax.set_title(f"t-SNE of {use.sum():,} nuggets, {k} clusters; area ∝ √weight")
        fig.tight_layout()
        figs["clusters_tsne.png"] = fig
    fig = new(figsize=(8, 8))
    ax = fig.subplots()
    im = ax.imshow(centers[list(labels)].to_numpy().T, aspect="auto", cmap="viridis")
    ax.set_xticks(range(k), [str(c) for c in range(1, k + 1)])
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("cluster")
    fig.colorbar(im, ax=ax, label=f"arcsinh(x / {cofactor:g})" if cofactor else "x")
    fig.tight_layout()
    figs["cluster_centers.png"] = fig
    return figs


def save_figures(figs: Mapping[str, Figure], out: Path, dpi: int) -> list[Path]:
    """Each figure saved in the folder out under its name, at dpi; the paths, in figs' order."""
    paths = []
    for name, fig in figs.items():
        fig.savefig(out / name, dpi=dpi)
        paths.append(out / name)
    return paths
