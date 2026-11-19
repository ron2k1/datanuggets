"""DataNuggetWarning, the warning datanuggets issues where R warns."""

from __future__ import annotations


class DataNuggetWarning(UserWarning):
    """Warning issued where R warns, with R's message.

    Notes
    -----
    These functions emit it:

    - ``create_dn`` and ``create_dn_centers``: "DN.num is greater than or equal to the number
      of rows. Returning original data", when a center selection has no more rows than its
      target. ``create_dn`` checks each chunk and then the pooled centers.
    - ``refine_dn``: a split's k-means warning, with R's stats::kmeans text: "Quick-TRANSfer
      stage steps exceeded maximum (= N)".
    - ``arithmetic_complexity``: "NaNs produced", when the count it takes the log10 of is
      negative.
    - ``wpca``, ``dn_wpca`` and ``datanuggets.projection.pca_view``, which calls ``wpca``:
      "NaNs produced", once when an eigenvalue is below zero.
    - ``RDraws``: "NAs introduced by coercion to integer range", before its ValueError for a
      seed that is not NaN and does not truncate to an integer strictly between -2^31 and
      2^31.
    """
