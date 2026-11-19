"""style() and save(): the paper settings, their scope, and exact output sizes."""

import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from datanuggets import plot


def test_palettes():
    # R 4.6.1's col2rgb(palette()), as hex: matplotlib doesn't know R's name gray62
    assert plot.R_PALETTE == (
        "#000000",
        "#DF536B",
        "#61D04F",
        "#2297E6",
        "#28E2E5",
        "#CD0BBC",
        "#F5C710",
        "#9E9E9E",
    )
    assert plot.OKABE_ITO == (
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    )
    for color in plot.R_PALETTE + plot.OKABE_ITO:
        assert to_hex(color).upper() == color


def test_style_sets_the_paper_settings():
    with plot.style():
        rc = matplotlib.rcParams
        assert rc["pdf.fonttype"] == 42 and rc["ps.fonttype"] == 42
        assert rc["axes.prop_cycle"].by_key()["color"] == list(plot.OKABE_ITO)
        assert rc["font.size"] == 8 and rc["xtick.labelsize"] == 7
        assert not rc["axes.spines.top"] and not rc["axes.spines.right"]


def test_style_restores_rcparams():
    before = dict(matplotlib.rcParams)
    with pytest.raises(RuntimeError), plot.style():
        raise RuntimeError
    assert dict(matplotlib.rcParams) == before


@pytest.mark.parametrize(
    ("width", "height", "box"),
    [
        ("single", None, (252, 189)),
        ("double", 3.0, (504, 216)),
        (5, 2, (360, 144)),
        (np.int64(5), np.float64(2), (360, 144)),
    ],
)
def test_save_pdf_page_size(tmp_path, width, height, box):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = plot.save(fig, tmp_path / "f.pdf", width=width, height=height)
    media = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", path.read_bytes())
    assert media is not None, "no MediaBox in the PDF"
    assert (float(media[1]), float(media[2])) == box


def test_save_png_is_300_dpi(tmp_path):
    from PIL import Image

    fig, _ = plt.subplots()
    path = plot.save(fig, tmp_path / "f.png", width="double", height=3.0)
    assert Image.open(path).size == (2100, 900)


def test_save_ignores_a_tight_bbox_setting(tmp_path):
    from PIL import Image

    fig, _ = plt.subplots()
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        path = plot.save(fig, tmp_path / "f.png", width=4, height=2)
    assert Image.open(path).size == (1200, 600)


def test_style_embeds_truetype_fonts(tmp_path):
    with plot.style():
        fig, ax = plt.subplots()
        ax.set_title("Height")
        data = plot.save(fig, tmp_path / "f.pdf").read_bytes()
    assert b"/FontFile2" in data and b"/Type3" not in data


@pytest.mark.parametrize("bad", ["triple", 0, -1, float("nan"), True])
def test_save_rejects_bad_widths(tmp_path, bad):
    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        plot.save(fig, tmp_path / "f.pdf", width=bad)


@pytest.mark.parametrize("name", ["f", "f."])
def test_save_needs_a_suffix_for_the_format(tmp_path, name):
    """matplotlib would write f.png in the default savefig.format, a file other than the path
    save returns. The figure keeps its size and nothing is written."""
    fig = plt.figure(figsize=(2, 1))
    with pytest.raises(ValueError, match=r"^path needs a suffix such as \.pdf or \.png"):
        plot.save(fig, tmp_path / name)
    assert list(tmp_path.iterdir()) == [] and tuple(fig.get_size_inches()) == (2, 1)
