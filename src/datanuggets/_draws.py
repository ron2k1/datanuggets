"""Source of randomness. Users get numpy; parity tests replay the draws R recorded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class ReplayMismatch(AssertionError):
    """Python requested a random draw that R did not make, or in a different order."""


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


@dataclass(frozen=True)
class DrawRecord:
    n: int
    size: int
    value: np.ndarray  # 1-based, as R returned it


class ReplayDraws:
    def __init__(self, records: list[DrawRecord]):
        self._records = list(records)
        self._pos = 0

    def _next(self, n: int, size: int) -> np.ndarray:
        if self._pos >= len(self._records):
            raise ReplayMismatch(
                f"draw #{self._pos + 1} sample.int({n}, {size}) requested, "
                f"but R made only {len(self._records)} draws"
            )
        rec = self._records[self._pos]
        if rec.n != n or rec.size != size:
            raise ReplayMismatch(
                f"draw #{self._pos + 1}: Python asked for sample.int({n}, {size}), "
                f"R did sample.int({rec.n}, {rec.size})"
            )
        self._pos += 1
        return rec.value - 1

    def permutation(self, n: int) -> np.ndarray:
        return self._next(n, n).astype(np.intp)

    def index(self, m: int) -> int:
        return int(self._next(m, 1)[0])

    def sample(self, n: int, size: int) -> np.ndarray:
        return self._next(n, size).astype(np.intp)

    def fork(self) -> ReplayDraws:
        """R's re-seeding is already in the recording, so forks share the one stream."""
        return self

    def assert_exhausted(self) -> None:
        if self._pos != len(self._records):
            raise ReplayMismatch(f"R made {len(self._records)} draws, Python consumed {self._pos}")


def as_draws(seed) -> Draws:
    if isinstance(seed, Draws):
        return seed
    if isinstance(seed, np.random.Generator) or (
        isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
    ):
        return NumpyDraws(seed)
    raise TypeError(
        f"seed must be an int, a numpy Generator, or a Draws object, not {type(seed).__name__}"
    )
