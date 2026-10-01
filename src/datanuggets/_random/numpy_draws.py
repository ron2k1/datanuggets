"""Draws from numpy's Generator, what an int or Generator seed gives users."""

from __future__ import annotations

import numpy as np


class NumpyDraws:
    def __init__(self, seed: int | np.integer | np.random.Generator):
        if isinstance(seed, np.random.Generator):
            self._seed, self._rng = None, seed
        else:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

    def permutation(self, n: int) -> np.ndarray:
        return self._rng.permutation(n)

    def index(self, m: int) -> int:
        return int(self._rng.integers(m))

    def sample(self, n: int, size: int) -> np.ndarray:
        return self._rng.choice(n, size, replace=False)

    def fork(self) -> NumpyDraws:
        """An int seed restarts the same stream on every call, like R's set.seed(seed). A
        Generator hands out a new child stream per call (Generator.spawn), so forks made in the
        same order give the same streams."""
        if self._seed is not None:
            return NumpyDraws(self._seed)
        return NumpyDraws(self._rng.spawn(1)[0])
