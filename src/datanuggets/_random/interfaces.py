"""The Draws interface: the R sampling calls the ported code makes, answered 0-based."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Draws(Protocol):
    def permutation(self, n: int) -> np.ndarray:
        """R's sample(n, n), 0-based."""
        ...

    def index(self, m: int) -> int:
        """R's sample.int(m, 1), 0-based."""
        ...

    def sample(self, n: int, size: int) -> np.ndarray:
        """R's sample.int(n, size): size distinct values from range(n), 0-based."""
        ...

    def fork(self) -> Draws:
        """The stream for one refine_dn split, where R calls set.seed(seed) again."""
        ...
