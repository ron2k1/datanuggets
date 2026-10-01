"""as_draws: what create_dn and refine_dn make of their seed= argument."""

from __future__ import annotations

import numpy as np

from datanuggets._random.interfaces import Draws
from datanuggets._random.numpy_draws import NumpyDraws
from datanuggets._random.r_draws import RDraws


def as_draws(seed) -> Draws:
    if isinstance(seed, RDraws):  # create.DN's set.seed(seed): start from the seed every call
        return seed.fork()
    if isinstance(seed, Draws):
        return seed
    if isinstance(seed, np.random.Generator) or (
        isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
    ):
        return NumpyDraws(seed)
    raise TypeError(
        f"seed must be an int, a numpy Generator, or a Draws object, not {type(seed).__name__}"
    )
