"""The weighted-PCA view of nugget centers, any rows projected into it, and the nuggets' t-SNEs,
on real data (R's quakes, each column standardized). None of colors, filters or projection needs
matplotlib or pandas to import."""

import functools
import subprocess
import sys

import numpy as np
import pytest

from datanuggets import DataNuggetWarning, _defaults, create_dn, wpca
from datanuggets.projection import PcaView, pca_view, project, tsne_2d, tsne_3d
from rfixtures import quakes

# what TSNE_ARGS holds, written out so a change to it fails here
TSNE_ARGS = {"n_components": 2, "init": "pca", "learning_rate": "auto", "max_iter": 1000}


@pytest.fixture(scope="module")
def nuggets():
    x = quakes()
    dn = create_dn(x, dn_num1=400, dn_num2=120, seed=291102, n_jobs=1)
    use = dn.weights > 0
    return x, dn.centers[use], dn.weights[use].astype(float)


def test_raw_signs_view_is_wpcas_own(nuggets):
    _, centers, weights = nuggets
    view = pca_view(centers, weights, fix_signs=False)
    ref = wpca(centers, wrow=weights)
    assert isinstance(view, PcaView)
    assert np.array_equal(view.center, ref.center)
    assert np.array_equal(view.rotation, ref.rotation)
    assert np.array_equal(view.scores, ref.x)
    assert np.array_equal(view.sdev, ref.sdev)
    lam = ref.sdev**2
    np.testing.assert_allclose(view.variance_ratio, lam / lam.sum(), rtol=1e-15)
    assert view.variance_ratio.sum() == pytest.approx(1.0)


def test_fixed_signs_make_each_largest_loading_positive(nuggets):
    _, centers, weights = nuggets
    raw = pca_view(centers, weights, fix_signs=False)
    view = pca_view(centers, weights, fix_signs=True)
    big = view.rotation[np.argmax(np.abs(view.rotation), axis=0), np.arange(view.rotation.shape[1])]
    assert (big > 0).all()
    signs = np.sign(view.rotation[0] / raw.rotation[0])
    assert set(signs) <= {-1.0, 1.0}
    assert np.array_equal(view.rotation, raw.rotation * signs)
    assert np.array_equal(view.scores, raw.scores * signs)
    assert np.array_equal(view.sdev, raw.sdev)


def test_signs_match_the_plotted_view(nuggets):
    pytest.importorskip("matplotlib")
    from datanuggets.plot.pca import _signs

    _, centers, weights = nuggets
    ref = wpca(centers, wrow=weights)
    view = pca_view(centers, weights, fix_signs=True)
    assert np.array_equal(view.rotation, ref.rotation * _signs(ref.rotation))


@pytest.mark.parametrize("fix", [False, True])
def test_weights_summing_below_1_give_an_all_nan_variance_ratio(nuggets, fix):
    _, centers, weights = nuggets
    with (
        pytest.warns(DataNuggetWarning, match="NaNs produced"),
        pytest.warns(RuntimeWarning, match="invalid value"),
    ):
        view = pca_view(centers, weights * (0.5 / weights.sum()), fix_signs=fix)
    assert np.isnan(view.sdev).all()
    assert np.isnan(view.variance_ratio).all()


@pytest.mark.parametrize("fix", [False, True])
def test_projecting_the_centers_gives_their_scores(nuggets, fix):
    x, centers, weights = nuggets
    view = pca_view(centers, weights, fix_signs=fix)
    np.testing.assert_allclose(project(centers, view), view.scores, rtol=1e-9, atol=1e-12)
    two = project(x[:500], view, 2)
    assert two.shape == (500, 2)
    np.testing.assert_allclose(two, (x[:500] - view.center) @ view.rotation[:, :2], rtol=1e-12)


def test_project_refuses_rows_of_the_wrong_width(nuggets):
    _, centers, weights = nuggets
    view = pca_view(centers, weights, fix_signs=True)
    with pytest.raises(ValueError, match=f"must have {centers.shape[1]} columns"):
        project(np.zeros((4, 2)), view)
    with pytest.raises(ValueError, match="n_components"):
        project(centers, view, centers.shape[1] + 1)


def test_tsne_args_are_pinned_and_read_only():
    assert _defaults.TSNE_ARGS == TSNE_ARGS
    with pytest.raises(TypeError):  # read only: one caller can't change them for every later fit
        _defaults.TSNE_ARGS["max_iter"] = 250


def test_tsne_2d_is_scikit_learns_with_the_pinned_arguments(nuggets):
    sklearn = pytest.importorskip("sklearn.manifold")

    _, centers, _ = nuggets
    got = tsne_2d(centers, perplexity=20.0, seed=3)
    assert got.shape == (len(centers), 2)
    want = sklearn.TSNE(**TSNE_ARGS, perplexity=20.0, random_state=3).fit_transform(centers)
    assert np.array_equal(got, want)
    assert np.array_equal(tsne_2d(centers[:10]), tsne_2d(centers[:10], perplexity=30.0, seed=0))


def test_tsne_gets_the_pinned_arguments_and_repeats_for_a_seed(monkeypatch, nuggets):
    """The arguments that shape the picture are TSNE_ARGS'; perplexity and the seed are the
    caller's, the perplexity capped at (n - 1) / 3 for n centers."""
    sklearn = pytest.importorskip("sklearn.manifold")

    seen, init = [], sklearn.TSNE.__init__

    @functools.wraps(init)  # scikit-learn reads the parameters from __init__'s signature
    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        init(self, *args, **kwargs)

    monkeypatch.setattr(sklearn.TSNE, "__init__", spy)
    centers = nuggets[1][:40]
    first = tsne_2d(centers, perplexity=50.0, seed=7)
    again = tsne_2d(centers, perplexity=50.0, seed=7)
    tsne_2d(centers, perplexity=5.0)
    assert seen == [
        {**TSNE_ARGS, "perplexity": 13.0, "random_state": 7},
        {**TSNE_ARGS, "perplexity": 13.0, "random_state": 7},
        {**TSNE_ARGS, "perplexity": 5.0, "random_state": 0},
    ]
    assert first.shape == (40, 2)
    assert np.array_equal(first, again)


def test_tsne_3d_is_a_fit_of_its_own_with_the_same_settings(nuggets):
    """Three components fitted from scratch, not the 2-D fit plus an axis: TSNE_ARGS with
    n_components 3, the same perplexity cap and seed."""
    sklearn = pytest.importorskip("sklearn.manifold")

    _, centers, _ = nuggets
    got = tsne_3d(centers, perplexity=20.0, seed=3)
    assert got.shape == (len(centers), 3)
    args = {**TSNE_ARGS, "n_components": 3}
    want = sklearn.TSNE(**args, perplexity=20.0, random_state=3).fit_transform(centers)
    assert np.array_equal(got, want)
    assert np.array_equal(tsne_3d(centers[:10]), tsne_3d(centers[:10], perplexity=3.0, seed=0))


def test_tsne_3d_needs_three_columns(nuggets):
    """Its PCA start can't lift two columns to three components, so there is none."""
    pytest.importorskip("sklearn")
    assert tsne_3d(nuggets[1][:, :2]) is None


def test_tsne_3d_needs_three_centers(nuggets):
    """Nor can it lift two centers, as a run that keeps two nuggets has: none, not an error."""
    pytest.importorskip("sklearn")
    assert tsne_3d(nuggets[1][:2]) is None


@pytest.mark.parametrize("fit", [tsne_2d, tsne_3d])
def test_tsne_without_scikit_learn_says_how_to_get_it(monkeypatch, nuggets, fit):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.manifold", None)
    with pytest.raises(ImportError) as e:
        fit(nuggets[1])
    assert str(e.value) == 't-SNE needs scikit-learn: pip install "datanuggets[tsne]"'


def test_new_modules_import_without_matplotlib_or_pandas():
    code = (
        "import sys\n"
        "sys.modules.update(dict.fromkeys(('matplotlib', 'pandas', 'sklearn')))\n"
        "import datanuggets.colors, datanuggets.filters, datanuggets.projection\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
