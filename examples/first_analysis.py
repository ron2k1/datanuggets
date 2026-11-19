"""Reduce and cluster the recorded R quakes data; save results under outputs/."""

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datanuggets import create_dn, cut_tree, dn_whclust, plot


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/fixtures/create_dn/quakes_euclidean_mean"
    spec = json.loads((fixture / "manifest.json").read_text())["arrays"]["x"]
    raw = np.fromfile(fixture / spec["file"], dtype=spec["dtype"]).reshape(spec["shape"])
    features = ["latitude", "longitude", "depth", "magnitude", "stations"]
    frame = pd.DataFrame(raw, columns=features)

    # Scale all five features; retain their order when working with the result.
    scale = raw.std(axis=0)
    scale[scale == 0] = 1
    x = (raw - raw.mean(axis=0)) / scale
    dn = create_dn(x, dn_num1=400, dn_num2=100, seed=42)
    labels = cut_tree(dn_whclust(dn), k=5)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "outputs" / f"first-analysis-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    dn.to_frame().to_csv(out / "nuggets.csv", index=False)
    frame.assign(cluster=labels[dn.assignments]).to_csv(out / "observations.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot.clusters(dn.centers, labels, w=dn.weights, ax=ax)
    ax.set(
        xlabel="Latitude (standardized)",
        ylabel="Longitude (standardized)",
        title="Quakes: five clusters of weighted nuggets",
    )
    fig.tight_layout()
    fig.savefig(out / "clusters.png", dpi=150)
    plt.close(fig)
    print(f"{len(x):,} observations -> {dn.n_nuggets} nuggets; weights sum to {dn.weights.sum()}")
    print(f"Saved plot and tables to {out}")


if __name__ == "__main__":
    main()
