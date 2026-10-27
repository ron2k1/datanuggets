"""The flow run's library steps, which a backend calls without the CLI: its figures, drawn on
figures no pyplot window holds."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datanuggets import cut_tree, whclust, wmean
from datanuggets.flow import finder_peaks
from flowcases import LABELS, load_case


def _pool(seed=0):
    """A pool as the clustering step gives it: 40 nuggets, the weight-0 ones out of use, cut
    into 3 clusters."""
    rng = np.random.default_rng(seed)
    centers, weights = rng.normal(size=(40, 3)), rng.integers(0, 20, 40).astype(float)
    use = weights > 0
    tree = whclust(centers[use], weights[use])
    cl = np.full(len(centers), -1)
    cl[use] = cut_tree(tree, k=3)
    return SimpleNamespace(centers=centers, weights=weights, use=use, tree=tree, cl=cl, k=3)


def test_result_figures_leave_no_pyplot_figures(tmp_path, close_figures):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    from datanuggets.flow.figures import (
        gates_figure,
        local_maxima_figure,
        result_figures,
        save_figures,
    )

    case = load_case("202")
    sample = {
        "path": Path("202.fcs"),
        "fsc": case.fsc,
        "ssc": case.ssc,
        "masks": case.his_masks,
        "peaks": finder_peaks(case.fsc, case.ssc),
    }
    pool = _pool()
    use = pool.use
    centers = pd.DataFrame(
        wmean(pool.centers[use], pool.cl[use], pool.weights[use]), columns=LABELS
    )
    xy = np.random.default_rng(1).normal(size=(use.sum(), 2))
    plt.close("all")
    figs = {
        "local_maxima.png": local_maxima_figure([sample] * 3),  # two rows, the fourth panel hidden
        "gates.png": gates_figure([sample]),
        **result_figures(pool, centers, LABELS, 6000.0, xy),
    }
    assert plt.get_fignums() == []
    assert all(isinstance(fig, Figure) for fig in figs.values())
    assert list(figs) == [
        "local_maxima.png",
        "gates.png",
        "dendrogram.png",
        "clusters_pca.png",
        "clusters_tsne.png",
        "cluster_centers.png",
    ]
    assert sum(not ax.get_visible() for ax in figs["local_maxima.png"].axes) == 1
    no_tsne = result_figures(pool, centers, LABELS, 0, None)
    assert list(no_tsne) == ["dendrogram.png", "clusters_pca.png", "cluster_centers.png"]
    saved = save_figures({"gates.png": figs["gates.png"]}, tmp_path, 110)
    assert saved == [tmp_path / "gates.png"]
    assert saved[0].read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == []
