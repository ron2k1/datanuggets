"""Every plot test draws on the Agg backend and closes its figures."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
