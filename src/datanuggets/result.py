"""Result containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _freeze(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CentersResult:
    """create_dn_centers output: the kept rows and their 0-based positions in the input."""

    centers: np.ndarray
    kept: np.ndarray

    def __post_init__(self):
        _freeze(self.centers)
        _freeze(self.kept)


@dataclass(frozen=True)
class DataNuggets:
    """Data nuggets: one center, weight and scale per nugget, plus each row's nugget (0-based)."""

    centers: np.ndarray
    weights: np.ndarray
    scales: np.ndarray
    assignments: np.ndarray

    def __post_init__(self):
        for arr in (self.centers, self.weights, self.scales, self.assignments):
            _freeze(arr)

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
