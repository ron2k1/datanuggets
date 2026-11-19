"""The weight and label checks of the WCluster ports, each next to what R does with the input."""

import numpy as np
import pytest

from datanuggets._validate import as_labels, as_weights


def test_weight_rules():
    # None is ones. R's own default, rep(1, length(x)), is one weight per column in wmean and
    # wwcss, which stops unless x is square, and one per cell in distw, which stops unless x
    # has one column
    got = as_weights(None, 3)
    assert got.dtype == np.float64 and np.array_equal(got, np.ones(3))
    for w in ([1, 2, 3], np.array([1, 2, 3], dtype=np.int32)):
        got = as_weights(w, 3)
        assert got.dtype == np.float64 and got.flags.c_contiguous
        assert np.array_equal(got, [1.0, 2.0, 3.0])
    # R stops: "The length of weights should be the same as the number of observations."
    with pytest.raises(ValueError, match=r"w must have one weight per row \(3\), got shape \(2,\)"):
        as_weights([1.0, 2.0], 3)
    # R reads a matrix by its length, so a 3 by 1 matrix of weights passes there
    with pytest.raises(ValueError, match=r"w must have one weight per row \(3\)"):
        as_weights(np.ones((3, 1)), 3)
    # R stops on text weights: "non-numeric argument to binary operator"
    with pytest.raises(TypeError, match="w must be numeric"):
        as_weights(["1", "2", "3"], 3)
    # R takes TRUE and FALSE as the weights 1 and 0
    with pytest.raises(TypeError, match="w must be numeric"):
        as_weights([True, False, True], 3)
    # R gives an NA center for an NA weight and a NaN center for an infinite one
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="w contains NaN or infinite values"):
            as_weights([1.0, bad, 1.0], 3)
    # R takes negative and zero weights as they are
    assert np.array_equal(as_weights([-1, 0, 2], 3), [-1.0, 0.0, 2.0])
    # Wpca stops: "The length of column weights should be the same as the number of columns
    # of data." and 'wcol must be of class "numeric" or "table"'
    with pytest.raises(ValueError, match=r"wcol must have one weight per column \(4\)"):
        as_weights([1.0, 1.0], 4, name="wcol", per="column")
    with pytest.raises(TypeError, match="wcol must be numeric"):
        as_weights(list("abcd"), 4, name="wcol", per="column")


def test_label_rules():
    # R's factor() sorts the distinct labels and gives each row its label's position, which
    # wmean, wwcss and distw all read the same way
    for cl, levels, codes in (
        ([8, 4, 6, 4], [4, 6, 8], [2, 0, 1, 0]),
        (np.array([8, 4, 6, 4], dtype=np.int32), [4, 6, 8], [2, 0, 1, 0]),
        ([-1, -3, -1, -2], [-3, -2, -1], [2, 0, 2, 1]),
        (np.array([8.0, 4.0, 6.0, 4.0]), [4.0, 6.0, 8.0], [2, 0, 1, 0]),
    ):
        got_levels, got_codes = as_labels(cl, 4)
        assert np.array_equal(got_levels, levels)
        assert got_codes.dtype == np.intp and np.array_equal(got_codes, codes)
    # R's factor() prints -0 and 0 alike, so they're one level, "0"
    levels, codes = as_labels(np.array([-0.0, 0.0, 1.0]), 3)
    assert levels.size == 2 and np.array_equal(codes, [0, 0, 1])
    # R sorts text labels in its locale's collation, which Python can't reproduce
    with pytest.raises(TypeError, match="cl must hold numeric cluster labels"):
        as_labels(["b", "a", "b"], 3)
    # R's factor() makes the levels FALSE and TRUE
    with pytest.raises(TypeError, match="cl must hold numeric cluster labels"):
        as_labels([True, False, True], 3)
    # R's factor() tells labels apart by their 15 digit text, so 0.1 + 0.2 and 0.3 are one
    # level there. R stops on NA with "NA/NaN argument" and makes NaN, Inf and -Inf clusters
    # of their own, with -Inf sorted first and Inf and NaN last
    for bad in (0.1 + 0.2, 1.5, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="cl must hold whole numbers"):
            as_labels([1.0, bad, 2.0], 3)
    # R's factor() makes 1e15 and 1e15 + 1 one level, "1e+15". Below 1e15 every whole number
    # keeps its own level
    for big in (1e15, -1e15, 10**15, 2**62):
        with pytest.raises(ValueError, match="smaller than 1e15"):
            as_labels([1, big], 2)
    levels, _ = as_labels([1e15 - 1, 1e15 - 2], 2)
    assert levels.size == 2
    # R stops: "The length of cluster assignments should be the same as the number of
    # observations."
    with pytest.raises(ValueError, match=r"cl must have one label per row \(3\), got shape \(2,\)"):
        as_labels([1, 2], 3)
    with pytest.raises(ValueError, match=r"cl must have one label per row \(2\)"):
        as_labels([[1], [2]], 2)
    # R warns "no non-missing arguments to max" and stops with "result would be too long a
    # vector" when there are no rows
    with pytest.raises(ValueError, match="cl is empty"):
        as_labels([], 0)
