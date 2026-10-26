"""Data nuggets for your own CSV: what was read, the transformation, the nuggets, the clusters.

    python examples/try_dn.py path/to/data.csv 4

Every numeric column is used (delete ID columns first) and rows with a missing value are left
out. Each column is standardized as R's scale() does, nuggets are made from the standardized
rows, and the nuggets' Ward tree is cut into k clusters (default 4). The tables and plot go to
outputs/<file name>/. The seed draws from NumPy, so the nuggets differ from R's for the same seed.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datanuggets import create_dn, cut_tree, dn_whclust, dn_wpca, plot

path = Path(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
df = pd.read_csv(path)
num = df.select_dtypes("number").replace([np.inf, -np.inf], np.nan).dropna()
print(f"1. read {path.name}: {len(df):,} rows, {df.shape[1]} columns")
print(f"   using {num.shape[1]} numeric columns: {', '.join(map(str, num.columns))}")
print(
    f"   left out: {[c for c in df.columns if c not in num.columns]} (not numeric), "
    f"{len(df) - len(num):,} rows with a missing value"
)

mean, sd = num.mean(), num.std().replace(0, 1)
x = ((num - mean) / sd).to_numpy()
print("\n2. transformation: each column -> (value - mean) / sd, as R's scale()")
print(
    pd.DataFrame(
        {
            "mean": mean,
            "sd": sd,
            "min": num.min(),
            "max": num.max(),
            "new min": x.min(0),
            "new max": x.max(0),
        }
    )
    .round(4)
    .to_string()
)

# Nugget targets scaled to the data size, as dn.R sets them
n = len(x)
dn = create_dn(x, dn_num1=min(2000, int(0.4 * n)), dn_num2=min(300, max(5, int(0.13 * n))), seed=42)
labels = cut_tree(dn_whclust(dn), k=k)
pc = dn_wpca(dn)
nug = pd.DataFrame(dn.centers * sd.to_numpy() + mean.to_numpy(), columns=num.columns)
nug["weight"], nug["cluster"] = dn.weights, labels
print(f"\n3. data nuggets: {n:,} rows -> {dn.n_nuggets} nuggets (centers in original units)")
print(nug.sort_values("weight", ascending=False).head(10).round(3).to_string())
print(f"\n4. clusters of nuggets: rows per cluster {np.bincount(labels[dn.assignments]).tolist()}")

out = Path("outputs") / path.stem
out.mkdir(parents=True, exist_ok=True)
nug.to_csv(out / "nuggets.csv", index_label="nugget")
rows = df.loc[num.index].assign(nugget=dn.assignments, cluster=labels[dn.assignments])
rows.to_csv(out / "rows.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
plot.scores(pc, w=dn.weights, labels=labels, rows=x, ax=ax[0])
ax[0].set_title("rows (gray), nuggets (area = weight), colored by cluster")
plot.scree(pc, ax=ax[1])
plot.loadings(pc, names=list(map(str, num.columns)), ax=ax[2])
fig.tight_layout()
fig.savefig(out / "nuggets.png", dpi=120)
print(f"\nsaved {out}/nuggets.csv, rows.csv, nuggets.png")
if sys.platform == "darwin":
    subprocess.run(["open", str(out / "nuggets.png")], check=False)
