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
