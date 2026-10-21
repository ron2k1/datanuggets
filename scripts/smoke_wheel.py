"""Exercise a core-only installed wheel using quakes from an extracted source archive.

Run with a fresh environment containing only the wheel and its required dependencies:
    python scripts/smoke_wheel.py path/to/extracted/sdist
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import datanuggets as dn


def main():
    package = Path(dn.__file__).resolve()
    assert "site-packages" in package.parts, package
    for extra in ("pandas", "matplotlib", "flowio", "tqdm"):
        assert importlib.util.find_spec(extra) is None, f"Expected core-only install: {extra}"

    fixture = Path(sys.argv[1]) / "tests/fixtures/create_dn/quakes_euclidean_mean"
    spec = json.loads((fixture / "manifest.json").read_text())["arrays"]["x"]
    raw = np.fromfile(fixture / spec["file"], dtype=spec["dtype"]).reshape(spec["shape"])
    x = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    nuggets = dn.create_dn(x, dn_num1=400, dn_num2=100, seed=42)
    labels = dn.cut_tree(dn.dn_whclust(nuggets), k=5)
    pca = dn.dn_wpca(nuggets)
    assert nuggets.weights.sum() == len(x)
    assert len(labels) == nuggets.n_nuggets
    assert len(np.unique(labels)) == 5
    assert len(labels[nuggets.assignments]) == len(x)
    assert np.isfinite(pca.sdev).all()
    print(f"Core-only wheel {dn.__version__}: {len(x)} observations, {nuggets.n_nuggets} nuggets")


if __name__ == "__main__":
    main()
