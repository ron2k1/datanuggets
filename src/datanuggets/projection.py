"""The weighted PCA of nugget centers as numpy arrays, with any rows projected into it, and 2-D
and 3-D t-SNEs of nugget centers. Needs no matplotlib; the t-SNEs need scikit-learn, which the
tsne extra installs."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from datanuggets._defaults import TSNE_ARGS, TSNE_PERPLEXITY, TSNE_SEED
from datanuggets.wcluster.wpca import wpca

TSNE_MISSING = 't-SNE needs scikit-learn: pip install "datanuggets[tsne]"'


class PcaView(NamedTuple):
    """A weighted PCA of nugget centers as numpy arrays, from pca_view or view_of."""

    center: np.ndarray  # (p,) weighted column means
    rotation: np.ndarray  # (p, q), one column per component
    scores: np.ndarray  # (n, q) of the centers given
    sdev: np.ndarray  # (q,)
    variance_ratio: np.ndarray  # (q,), a NaN sdev counted as 0; sums to 1 if any sdev is positive


def component_signs(rotation) -> np.ndarray:
    """+1 or -1 per component: -1 where the loading largest by absolute value is negative, so
    multiplying makes it positive. plot.pca's views flip each component by these."""
    r = np.asarray(rotation, dtype=float)
    big = r[np.argmax(np.abs(r), axis=0), np.arange(r.shape[1])]
    return np.where(big < 0, -1.0, 1.0)


def view_of(result, *, fix_signs: bool) -> PcaView:
    """The PcaView of a WpcaResult: its own LAPACK signs, or with each component flipped so its
    largest loading is positive, as plot.scores draws it. Weights summing below 1 can leave no
    sdev positive. Then variance_ratio is all NaN and NumPy issues a RuntimeWarning for the
    0 / 0, with either fix_signs."""
    signs = component_signs(result.rotation) if fix_signs else 1.0
    lam = np.nan_to_num(np.asarray(result.sdev, dtype=float) ** 2)
    return PcaView(
        center=np.asarray(result.center, dtype=float),
        rotation=np.asarray(result.rotation, dtype=float) * signs,
        scores=np.asarray(result.x, dtype=float) * signs,
        sdev=np.asarray(result.sdev, dtype=float),
        variance_ratio=lam / lam.sum(),
    )


def pca_view(centers, weights, *, fix_signs: bool) -> PcaView:
    """wpca(centers, wrow=weights) as a PcaView: each component flipped so its largest loading
    is positive, as plot.scores draws it (fix_signs=True), or with LAPACK's signs. Weights summing
    below 1 can bring wpca's DataNuggetWarning and the RuntimeWarning view_of describes."""
    return view_of(wpca(centers, wrow=weights), fix_signs=fix_signs)


def project(x, view: PcaView, n_components: int | None = None) -> np.ndarray:
    """Rows of x, one column per variable on the scale the centers were, in the view's space:
    (x - center) @ rotation[:, :n_components], every component by default. The centers project
    onto their own scores."""
    x = np.asarray(x, dtype=float)
    p, q = view.rotation.shape
    if x.ndim != 2 or x.shape[1] != p:
        raise ValueError(f"x must have {p} columns, one per variable, got shape {x.shape}")
    m = q if n_components is None else n_components
    if not isinstance(m, int) or not 1 <= m <= q:
        raise ValueError(f"n_components must be a whole number from 1 to {q}, got {m!r}")
    return (x - view.center) @ view.rotation[:, :m]


def _tsne(centers, dims: int, perplexity: float, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise ImportError(TSNE_MISSING) from e
    centers = np.asarray(centers, dtype=float)
    perplexity = min(perplexity, (len(centers) - 1) / 3)
    args = {**TSNE_ARGS, "n_components": dims}
    return TSNE(**args, perplexity=perplexity, random_state=seed).fit_transform(centers)


def tsne_2d(centers, *, perplexity: float = TSNE_PERPLEXITY, seed: int = TSNE_SEED) -> np.ndarray:
    """A 2-D t-SNE of the centers, shaped by TSNE_ARGS. Its perplexity is capped at (n - 1) / 3
    for n centers, as scikit-learn needs it below n. seed is its random_state, so the same call
    gives the same result."""
    return _tsne(centers, 2, perplexity, seed)


def tsne_3d(
    centers, *, perplexity: float = TSNE_PERPLEXITY, seed: int = TSNE_SEED
) -> np.ndarray | None:
    """A 3-D t-SNE of the centers: a fit of its own, not tsne_2d plus an axis, with the same
    TSNE_ARGS, perplexity cap and seed. None for centers of fewer than three columns or rows,
    which its PCA start can't lift to three components."""
    if min(np.shape(centers)) < 3:
        return None
    return _tsne(centers, 3, perplexity, seed)
