"""close_figures, for the tests that run datanuggets-flow: it closes the pyplot figures a run
leaves open. matplotlib is imported only when a test asks for it, so the finder's tests don't need
it."""

import pytest


@pytest.fixture
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
