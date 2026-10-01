"""Port of R's getDN.means (datanugget 1.5.0)."""

from __future__ import annotations

import numpy as np

from datanuggets._rcompat import r_mean
from datanuggets._validate import as_matrix, check_assignments
from datanuggets.result import DataNuggets


def get_dn_means(x, dn: DataNuggets) -> np.ndarray:
    """The column means of each nugget's rows of x, as a (k, p) float64 array.

    Row i is the mean of the rows assigned to nugget i, whatever center method built dn. An
    empty nugget gets a row of R's 0 / 0, the NaN with its sign bit set (fff8000000000000)
    that x86 R gives, where numpy's own NaN has it clear. Each mean is R's colMeans, summed in
    long double, so the result matches R bit for bit. R returns a data frame, transposed when x
    has one column. This returns k rows either way. x must have exactly the rows dn assigns,
    each to a nugget id in [0, k). R ignores extra rows and skips a row whose id is out of range
    or fractional. This raises ValueError for both.
    """
    x = as_matrix(x, logical=True)
    if not isinstance(dn, DataNuggets):
        raise TypeError(
            f"dn must be a DataNuggets from create_dn or refine_dn, not {type(dn).__name__}"
        )
    n, p = x.shape
    assignments = check_assignments(dn, n, p)
    k = dn.centers.shape[0]
    order = np.argsort(assignments, kind="stable")  # R's which(): each nugget's rows in order
    bounds = np.concatenate(([0], np.cumsum(np.bincount(assignments, minlength=k))))
    means = np.empty((k, p))
    for i in range(k):
        means[i] = r_mean(x[order[bounds[i] : bounds[i + 1]]], axis=0)
    return means
