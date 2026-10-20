"""R's own draws: set.seed(seed) under R's default RNGkind, then sample.int() as R 4.6.1 runs
it."""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Iterator

import numpy as np

from datanuggets.exceptions import DataNuggetWarning

_M32 = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1), RNG.c's i2_32m1, which fixup() uses
_TWO_M32 = 2.3283064365386963e-10  # 2^-32, MT_genrand's scale
_MT_KINDS = 10403  # .Random.seed[1] for Mersenne-Twister, Inversion, Rejection: R's defaults
_HASH_TRIES = 100  # unique.c's do_sample2 keeps the last value after 100 repeats in a row


def _r_seed(seed) -> int:
    """set.seed's asInteger(seed): truncated toward zero, and R's stop where that is NA."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Real):
        raise TypeError(f"seed must be a real number, not {type(seed).__name__}")
    if isinstance(seed, numbers.Integral):
        value = int(seed)
        in_range = -(2**31) < value < 2**31
    else:
        x = float(seed)
        if math.isnan(x):  # NaN and NA become NA without a warning
            raise ValueError("supplied seed is not a valid integer")
        in_range = -2147483648.0 < x < 2147483648.0
        value = int(x) if in_range else 0
    if not in_range:
        warnings.warn(
            "NAs introduced by coercion to integer range", DataNuggetWarning, stacklevel=3
        )
        raise ValueError("supplied seed is not a valid integer")
    return value


def r_use_hash(n: int, size: int) -> bool:
    """sample.int's default useHash, n > 1e7 && size <= n/2, with replace = FALSE and prob =
    NULL, which is how every datanugget call samples."""
    return n > 1e7 and size <= n / 2


class RDraws:
    """R's own draws: set.seed(seed) under R's default RNGkind (Mersenne-Twister, Inversion,
    Rejection), then sample.int() as R 4.6.1 runs it, hash branch included. Pass RDraws(seed)
    as seed= and create_dn and refine_dn draw exactly what R's create.DN and refine.DN draw.
    Each call starts from seed, as the set.seed(seed) inside them does, so one RDraws can be
    passed to many calls."""

    def __init__(self, seed):
        self._seed = _r_seed(seed)
        s = self._seed & _M32  # the cast to Int32, which is unsigned
        for _ in range(50):  # RNG_Init's initial scrambling
            s = (69069 * s + 1) & _M32
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & _M32
            key[j] = s
        # FixupSeeds sets the first word, mti, to 624, so the first draw regenerates the state
        self._bg = np.random.MT19937()
        self._bg.state = {"bit_generator": "MT19937", "state": {"key": key[1:], "pos": 624}}
        self._block: list[int] = []  # the unread words of the current 624, last one first

    def random_seed(self) -> np.ndarray:
        """R's .Random.seed: the RNG kinds, the Mersenne-Twister position, then its state."""
        key = self._bg.state["state"]["key"].astype(np.uint32).view(np.int32)
        return np.concatenate([[_MT_KINDS, 624 - len(self._block)], key]).astype(np.int32)

    def _unif(self) -> float:
        """unif_rand(): MT_genrand kept inside (0, 1) by fixup()."""
        if not self._block:
            self._block = self._bg.random_raw(624).tolist()[::-1]
        u = self._block.pop() * _TWO_M32
        if u <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - u <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return u

    def _unif_index(self, dn: int) -> int:
        """R_unif_index(dn) under Rejection: rbits(bits) until a value below dn. bits comes
        from log2 as in R, not from the exact bit length."""
        if dn <= 0:
            return 0
        bits = math.ceil(math.log2(dn))
        while True:
            v = 0
            for _ in range(0, bits + 1, 16):  # rbits: 16 bits per uniform, then the mask
                v = 65536 * v + math.floor(self._unif() * 65536)
            v &= (1 << bits) - 1
            if dn > v:
                return v

    def _hash(self, n: int, size: int) -> Iterator[int]:
        """do_sample2: draw from all n, and draw again (up to 100 times) on a repeat."""
        seen: set[int] = set()
        for _ in range(size):
            v, tries = self._unif_index(n), 1
            while v in seen and tries < _HASH_TRIES:
                v, tries = self._unif_index(n), tries + 1
            seen.add(v)
            yield v

    def _pool(self, n: int, size: int) -> Iterator[int]:
        """do_sample: draw a position among the n left, then move the last one into it."""
        pool = np.arange(n, dtype=np.int64)
        for _ in range(size):
            j = self._unif_index(n)
            yield int(pool[j])
            n -= 1
            pool[j] = pool[n]

    def _values(self, n: int, size: int, use_hash: bool) -> Iterator[int]:
        """sample.int(n, size, useHash = use_hash)'s values in order, 0-based, drawn lazily."""
        if use_hash:
            return self._hash(n, size)
        if size < 2:
            return (self._unif_index(n) for _ in range(size))
        return self._pool(n, size)

    def sample(self, n: int, size: int) -> np.ndarray:
        n, size = int(n), int(size)
        if n < 0 or n > 4.5e15 or (size > 0 and n == 0):
            raise ValueError("invalid first argument")
        if size < 0:
            raise ValueError("invalid 'size' argument")
        if size > n:
            raise ValueError(
                "cannot take a sample larger than the population when 'replace = FALSE'"
            )
        return np.fromiter(self._values(n, size, r_use_hash(n, size)), dtype=np.intp, count=size)

    def sample_replace(self, n: int, size: int) -> np.ndarray:
        """sample.int(n, size, replace = TRUE): do_sample's replace branch, one R_unif_index(n)
        per value. R never hashes a replace draw."""
        n, size = int(n), int(size)
        if n < 0 or n > 4.5e15 or (size > 0 and n == 0):
            raise ValueError("invalid first argument")
        if size < 0:
            raise ValueError("invalid 'size' argument")
        return np.fromiter((self._unif_index(n) for _ in range(size)), dtype=np.intp, count=size)

    def permutation(self, n: int) -> np.ndarray:
        return self.sample(n, n)

    def index(self, m: int) -> int:
        return int(self.sample(m, 1)[0])

    def fork(self) -> RDraws:
        """refine.DN's set.seed(seed) before a split: the stream from its start."""
        return RDraws(self._seed)
