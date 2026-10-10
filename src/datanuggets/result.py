"""Result containers."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


def _freeze(value) -> np.ndarray:
    """A read-only copy: a result never shares memory with, or locks, the caller's array."""
    arr = np.array(value, copy=True)
    arr.setflags(write=False)
    return arr


class _ArrayFields:
    """Base for the frozen result dataclasses: every field is stored as a read-only copy, and ==
    compares the arrays by value. Dataclass equality would compare them with ==, whose truth
    value is ambiguous. NaN equals NaN, since empty nuggets have NaN centers. Like numpy arrays,
    results are unhashable."""

    __hash__ = None

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _freeze(getattr(self, f.name)))

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name), equal_nan=True)
            for f in fields(self)
        )


@dataclass(frozen=True, eq=False)
class CentersResult(_ArrayFields):
    """create_dn_centers output: the kept rows and their 0-based positions in the input."""

    centers: np.ndarray
    kept: np.ndarray


@dataclass(frozen=True, eq=False)
class DataNuggets(_ArrayFields):
    """Data nuggets: one center, weight and scale per nugget, plus each row's nugget (0-based)."""

    centers: np.ndarray
    weights: np.ndarray
    scales: np.ndarray
    assignments: np.ndarray

    @property
    def n_nuggets(self) -> int:
        return self.centers.shape[0]

    def to_frame(self):
        """R's "Data Nuggets" table: Data Nugget, Center1..Centerp, Weight, Scale (ids 0-based)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame() needs pandas: pip install 'datanuggets[frame]'") from exc
        p = self.centers.shape[1]
        df = pd.DataFrame(self.centers, columns=[f"Center{j + 1}" for j in range(p)])
        df.insert(0, "Data Nugget", np.arange(self.n_nuggets))
        df["Weight"] = self.weights
        df["Scale"] = self.scales
        return df


@dataclass(frozen=True, eq=False)
class WwcssResult(_ArrayFields):
    """wwcss(group_sum=True) output: each cluster's weighted within-cluster sum of squares, in
    sorted label order, and their total as a 0-d float64. A cluster whose weights sum to zero
    is NaN, and so is the total then, as in R."""

    per_cluster: np.ndarray
    total: np.ndarray


@dataclass(frozen=True, eq=False)
class WpcaResult(_ArrayFields):
    """wpca output, with R's field names in R's order. sdev holds the square roots of the
    eigenvalues, largest first, and rotation the eigenvectors as columns. x is the scores,
    named as R names it: the centered data times rotation, unscaled even with corr=True, as in
    R. center is the weighted column means. scale is what corr=True divides the covariance by,
    the square roots of its diagonal, and ones without corr. wrow is the row weights and wcol
    the column weights normalised to sum to the number of columns."""

    sdev: np.ndarray
    rotation: np.ndarray
    x: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    wrow: np.ndarray
    wcol: np.ndarray


@dataclass(frozen=True, eq=False)
class WkmeansResult(_ArrayFields):
    """wkmeans output. labels is each row's cluster, R's label minus one. R can leave a label
    empty, and then labels has a gap there too. centers (one row per label present, in label
    order) and wwcss (one value per label present) follow R's wmean and wwcss, and total_wwcss
    is their sum as R adds it, a 0-d float64."""

    labels: np.ndarray
    centers: np.ndarray
    wwcss: np.ndarray
    total_wwcss: np.ndarray


@dataclass(frozen=True, eq=False)
class DnWkmeansResult(_ArrayFields):
    """dn_wkmeans output: wkmeans' four fields for the nuggets, plus row_labels, each original
    row's cluster, which is the cluster of the nugget it belongs to."""

    labels: np.ndarray
    centers: np.ndarray
    wwcss: np.ndarray
    total_wwcss: np.ndarray
    row_labels: np.ndarray


@dataclass(frozen=True, eq=False)
class WhclustResult(_ArrayFields):
    """whclust output in scipy's numbering. Row s of merge joins two clusters, where cluster r
    below n is row r and cluster n + s is the one row s made, and each row keeps R's column
    order. height is R's total WWCSS after each merge, and order is the rows in dendrogram order,
    0-based. When weights cancel, R can merge a cluster with itself, and then order repeats or
    drops rows, as R's does."""

    merge: np.ndarray
    height: np.ndarray
    order: np.ndarray

    def to_linkage(self) -> np.ndarray:
        """scipy's (n-1, 4) linkage: merge, height and each new cluster's size, for
        scipy.cluster.hierarchy.dendrogram. Raises ValueError for a self-merge, which a scipy
        linkage can't hold."""
        n = self.merge.shape[0] + 1
        if np.any(self.merge[:, 0] == self.merge[:, 1]):
            raise ValueError("merge joins a cluster with itself, which a scipy linkage can't hold")
        size = np.ones(2 * n - 1)
        z = np.empty((n - 1, 4))
        for s, (a, b) in enumerate(self.merge.tolist()):
            size[n + s] = size[a] + size[b]
            z[s] = (a, b, self.height[s], size[n + s])
        return z
