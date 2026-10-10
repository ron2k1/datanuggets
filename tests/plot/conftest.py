"""Every plot test draws on the Agg backend, closes its figures, and fails on a UserWarning,
which from matplotlib means an argument it had to guess at."""

import warnings

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield
    plt.close("all")
