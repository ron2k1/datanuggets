"""The draws R recorded for a fixture, replayed in order. A draw R didn't make, or made in
another order, raises ReplayMismatch, so parity tests check the call sequence as well as the
numbers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ReplayMismatch(AssertionError):
    """Python requested a random draw that R did not make, or in a different order."""


@dataclass(frozen=True)
class DrawRecord:
    n: int
    size: int
    value: np.ndarray  # 1-based, as R returned it
    replace: bool = False


def _call(n: int, size: int, replace: bool) -> str:
    return f"sample.int({n}, {size}{', replace = TRUE' if replace else ''})"


class ReplayDraws:
    def __init__(self, records: list[DrawRecord]):
        self._records = list(records)
        self._pos = 0

    def _next(self, n: int, size: int, replace: bool = False) -> np.ndarray:
        if self._pos >= len(self._records):
            raise ReplayMismatch(
                f"draw #{self._pos + 1} {_call(n, size, replace)} requested, "
                f"but R made only {len(self._records)} draws"
            )
        rec = self._records[self._pos]
        if rec.n != n or rec.size != size or rec.replace != replace:
            raise ReplayMismatch(
                f"draw #{self._pos + 1}: Python asked for {_call(n, size, replace)}, "
                f"R did {_call(rec.n, rec.size, rec.replace)}"
            )
        self._pos += 1
        return rec.value - 1

    def permutation(self, n: int) -> np.ndarray:
        return self._next(n, n).astype(np.intp)

    def index(self, m: int) -> int:
        return int(self._next(m, 1)[0])

    def sample(self, n: int, size: int) -> np.ndarray:
        return self._next(n, size).astype(np.intp)

    def sample_replace(self, n: int, size: int) -> np.ndarray:
        return self._next(n, size, True).astype(np.intp)

    def fork(self) -> ReplayDraws:
        """R's re-seeding is already in the recording, so forks share the one stream."""
        return self

    def assert_exhausted(self) -> None:
        if self._pos != len(self._records):
            raise ReplayMismatch(f"R made {len(self._records)} draws, Python consumed {self._pos}")
