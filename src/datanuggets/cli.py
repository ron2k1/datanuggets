"""Analyze a user's CSV with explicit preprocessing and reproducible, separate outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np

from datanuggets import __version__, create_dn, cut_tree, dn_whclust, dn_wkmeans, dn_wpca, refine_dn
from datanuggets.result import DataNuggets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datanuggets", description=__doc__)
    parser.add_argument("--version", action="version", version=f"datanuggets {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="create and analyze nuggets from a numeric CSV")
    analyze.add_argument("input", type=Path, help="UTF-8 CSV with a header row")
    analyze.add_argument("--columns", nargs="+", required=True, help="numeric feature column names")
    analyze.add_argument(
        "--transform",
        nargs="+",
        default=["none"],
        choices=["none", "standardize", "log1p", "arcsinh"],
        help="apply transformations in this order (default: none)",
    )
    analyze.add_argument("--cofactor", type=float, default=5.0, help="arcsinh divisor (default: 5)")
    analyze.add_argument(
        "--nuggets", type=int, default=200, help="final nugget target (default: 200)"
    )
    analyze.add_argument("--intermediate-nuggets", type=int, default=1000)
    analyze.add_argument(
        "--analysis",
        nargs="+",
        choices=["kmeans", "hclust", "pca", "none"],
        default=["kmeans", "pca"],
        help="analyses to run (default: kmeans pca); none creates nuggets",
    )
    analyze.add_argument("--clusters", type=int, default=3)
    analyze.add_argument("--refine", action="store_true", help="refine nuggets before analysis")
    analyze.add_argument("--max-splits", type=int, default=2, help="refinement rounds (default: 2)")
    analyze.add_argument("--num-init", type=int, default=10, help="k-means random starts")
    analyze.add_argument("--max-iter", type=int, default=100, help="k-means iteration limit")
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--jobs", type=int, default=1, help="workers; -1 uses all cores")
    analyze.add_argument("--no-plots", action="store_true", help="write tables without figures")
    analyze.add_argument(
        "--out", type=Path, help="new output folder (default: outputs/analysis-...)"
    )
    return parser


def _check_args(args) -> None:
    for flag in ("nuggets", "intermediate_nuggets", "clusters", "num_init", "max_iter"):
        if getattr(args, flag) < 1:
            raise ValueError(f"--{flag.replace('_', '-')} must be positive")
    if args.intermediate_nuggets < args.nuggets:
        raise ValueError("--intermediate-nuggets must be at least --nuggets")
    if args.seed < 0 or args.max_splits < 0:
        raise ValueError("--seed and --max-splits must be nonnegative")
    if args.jobs == 0 or args.jobs < -1:
        raise ValueError("--jobs must be -1 or a positive integer")
    if not np.isfinite(args.cofactor) or args.cofactor <= 0:
        raise ValueError("--cofactor must be positive and finite")
    for name in ("columns", "transform", "analysis"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f"--{name} must not contain duplicates")
        if name != "columns" and "none" in values and len(values) > 1:
            raise ValueError(f"--{name} none cannot be combined with other choices")
    if args.analysis != ["none"] and len(args.columns) < 2:
        raise ValueError(
            "analysis needs at least two feature columns; use --analysis none otherwise"
        )


def _read_csv(path: Path, columns: list[str]) -> tuple[np.ndarray, str]:
    """Validate structure without silently skipping rows or renaming duplicate headers."""
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    with io.TextIOWrapper(io.BytesIO(content), encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not header or any(not name.strip() for name in header):
            raise ValueError("CSV needs a nonempty header name for every column")
        if len(set(header)) != len(header):
            raise ValueError("CSV has duplicate column names")
        missing = [name for name in columns if name not in header]
        if missing:
            raise ValueError(f"CSV is missing selected columns: {', '.join(missing)}")
        positions = [header.index(name) for name in columns]
        rows = []
        for row0, record in enumerate(reader):
            if len(record) != len(header):
                raise ValueError(
                    f"CSV data row {row0} has {len(record)} fields; expected {len(header)}"
                )
            values = []
            for name, position in zip(columns, positions, strict=True):
                try:
                    value = float(record[position])
                except ValueError:
                    raise ValueError(
                        f"CSV data row {row0}, column {name!r}: expected a number"
                    ) from None
                if not np.isfinite(value):
                    raise ValueError(f"CSV data row {row0}, column {name!r}: value must be finite")
                values.append(value)
            rows.append(values)
    if len(rows) < 2:
        raise ValueError("CSV needs at least two data rows")
    return np.asarray(rows, dtype=float), digest


def _transform(raw: np.ndarray, names: list[str], cofactor: float):
    x = raw.copy()
    fitted = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for name in names:
            settings = {"name": name}
            if name == "standardize":
                mean, scale = x.mean(axis=0), x.std(axis=0)
                scale[scale == 0] = 1.0
                if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                    raise ValueError("standardization overflowed; rescale large input values first")
                x = (x - mean) / scale
                settings.update(mean=mean.tolist(), scale=scale.tolist(), ddof=0)
            elif name == "log1p":
                if (x < 0).any():
                    raise ValueError("log1p requires nonnegative values at this step")
                x = np.log1p(x)
            elif name == "arcsinh":
                x = np.arcsinh(x / cofactor)
                settings["cofactor"] = cofactor
            if not np.isfinite(x).all():
                raise ValueError(f"{name} produced nonfinite values; check the data and cofactor")
            fitted.append(settings)
    return x, fitted


def _occupied(dn: DataNuggets) -> DataNuggets:
    """Remove empty nuggets without changing centers or losing original row assignments."""
    keep = dn.weights > 0
    mapping = np.full(dn.n_nuggets, -1, dtype=np.int64)
    mapping[keep] = np.arange(keep.sum())
    return DataNuggets(dn.centers[keep], dn.weights[keep], dn.scales[keep], mapping[dn.assignments])


def _analyze(args, dn):
    labels, pca, tree = {}, None, None
    if "kmeans" in args.analysis:
        if dn.n_nuggets < 2 * args.clusters:
            raise ValueError(
                f"k-means needs at least {2 * args.clusters} occupied nuggets for "
                f"{args.clusters} clusters; got {dn.n_nuggets}. Reduce --clusters or increase --nuggets"
            )
        result = dn_wkmeans(
            dn,
            args.clusters,
            num_init=args.num_init,
            max_iter=args.max_iter,
            seed=args.seed,
            n_jobs=args.jobs,
        )
        labels["kmeans"] = result.labels
    if "hclust" in args.analysis:
        if dn.n_nuggets < max(3, args.clusters):
            raise ValueError(
                "hierarchical clustering needs at least three nuggets and one per cluster"
            )
        tree = dn_whclust(dn)
        labels["hclust"] = cut_tree(tree, k=args.clusters)
    if "pca" in args.analysis:
        pca = dn_wpca(dn)
        if not all(np.isfinite(v).all() for v in (pca.sdev, pca.x, pca.rotation)):
            raise ValueError(
                "PCA returned nonfinite results; check redundant features, scaling, and nugget count"
            )
        if not np.isfinite(np.square(pca.sdev).sum()) or np.square(pca.sdev).sum() <= 0:
            raise ValueError("PCA needs positive variance among nugget centers")
    return labels, pca, tree


def _write_tables(out, pd, raw, columns, dn, labels, pca, tree):
    nuggets = pd.DataFrame(dn.centers, columns=[f"feature:{name}" for name in columns])
    nuggets.insert(0, "nugget0", np.arange(dn.n_nuggets))
    nuggets["weight"], nuggets["scale"] = dn.weights, dn.scales
    assignments = pd.DataFrame({"row0": np.arange(len(raw)), "nugget0": dn.assignments})
    summary = []
    for method, cluster in labels.items():
        nuggets[f"{method}_cluster"] = cluster
        row_labels = cluster[dn.assignments]
        assignments[f"{method}_cluster"] = row_labels
        for label in np.unique(row_labels):
            selected = row_labels == label
            record = {
                "method": method,
                "cluster": int(label),
                "rows": int(selected.sum()),
                "percent": float(100 * selected.mean()),
            }
            record.update(zip((f"mean:{name}" for name in columns), raw[selected].mean(axis=0)))
            summary.append(record)
    nuggets.to_csv(out / "nuggets.csv", index=False)
    assignments.to_csv(out / "assignments.csv", index=False)
    if summary:
        pd.DataFrame(summary).to_csv(out / "cluster_summary.csv", index=False)
    if pca is not None:
        components = [f"PC{i + 1}" for i in range(len(pca.sdev))]
        scores = pd.DataFrame(pca.x, columns=components)
        scores.insert(0, "weight", dn.weights)
        scores.insert(0, "nugget0", np.arange(dn.n_nuggets))
        scores.to_csv(out / "pca_scores.csv", index=False)
        loadings = pd.DataFrame(pca.rotation, columns=components)
        loadings.insert(0, "feature", columns)
        loadings.to_csv(out / "pca_loadings.csv", index=False)
        variance = np.square(pca.sdev)
        pd.DataFrame(
            {
                "component": components,
                "sdev": pca.sdev,
                "variance_ratio": variance / variance.sum(),
            }
        ).to_csv(out / "pca_variance.csv", index=False)
        pd.DataFrame({"feature": columns, "center": pca.center}).to_csv(
            out / "pca_center.csv", index=False
        )
    if tree is not None:
        pd.DataFrame(tree.to_linkage(), columns=["left", "right", "height", "nuggets"]).to_csv(
            out / "hierarchy.csv",
            index=False,
        )
    return summary


def _plots(out, plt, columns, dn, labels, pca, tree):
    folder = out / "plots"
    folder.mkdir()
    sizes = 15 + 100 * dn.weights / dn.weights.max()

    def scatter(values, filename, title, axes, cluster=None):
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            points = ax.scatter(
                values[:, 0],
                values[:, 1],
                s=sizes,
                c=cluster,
                cmap="tab10" if cluster is not None else None,
                alpha=0.75,
                linewidths=0,
            )
            if cluster is not None:
                handles, legend_labels = points.legend_elements(num=None)
                ax.legend(handles, legend_labels, title="Cluster", loc="best")
            ax.set(xlabel=axes[0], ylabel=axes[1], title=title)
            fig.tight_layout()
            fig.savefig(folder / filename, dpi=150)
        finally:
            plt.close(fig)

    plane = (
        dn.centers[:, :2]
        if len(columns) >= 2
        else np.column_stack([dn.centers[:, 0], np.zeros(dn.n_nuggets)])
    )
    axes = [f"{name} (analysis units)" for name in columns[:2]]
    if len(axes) == 1:
        axes.append("")
    scatter(plane, "nuggets.png", "Data nuggets (size reflects weight)", axes)
    for method, cluster in labels.items():
        scatter(plane, f"{method}_clusters.png", f"{method}: first two features", axes, cluster)
    if pca is not None:
        ratios = np.square(pca.sdev) / np.square(pca.sdev).sum()
        scatter(
            pca.x[:, :2],
            "pca.png",
            "Weighted PCA of nugget centers",
            [f"PC{i + 1} ({ratios[i]:.1%})" for i in range(2)],
            next(iter(labels.values()), None),
        )
    if tree is not None:
        from datanuggets import plot

        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            plot.dendrogram(tree, labels=False, ax=ax)
            fig.tight_layout()
            fig.savefig(folder / "dendrogram.png", dpi=150)
        finally:
            plt.close(fig)


def _run(args) -> None:
    _check_args(args)
    out = args.out or Path("outputs") / (
        f"analysis-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    )
    out = out.expanduser()
    if out.exists() or out.is_symlink():
        raise ValueError(f"output path already exists: {out}; choose a new --out folder")
    try:
        import pandas as pd
    except ImportError:
        raise ValueError(
            "CSV analysis needs pandas; reinstall the package with the [cli] extra"
        ) from None
    plt = None
    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ValueError(
                "plots need matplotlib; install the [cli] extra or use --no-plots"
            ) from None
    source = args.input.expanduser()
    raw, digest = _read_csv(source, args.columns)
    x, fitted = _transform(raw, args.transform, args.cofactor)
    # Exclusive creation also protects against another process claiming the path after validation.
    out.mkdir(parents=True, exist_ok=False)
    incomplete = out / "INCOMPLETE.txt"
    incomplete.write_text(
        "This run has not completed. Files may be partial. Rerun in a new output folder.\n",
        encoding="utf-8",
    )
    try:
        print(
            f"Read {len(raw):,} rows and {len(args.columns)} features. Creating nuggets...",
            flush=True,
        )
        dn = create_dn(
            x,
            dn_num1=args.intermediate_nuggets,
            dn_num2=args.nuggets,
            seed=args.seed,
            n_jobs=args.jobs,
        )
        if args.refine:
            dn = refine_dn(x, dn, max_splits=args.max_splits, seed=args.seed, n_jobs=args.jobs)
        dn = _occupied(dn)
        labels, pca, tree = _analyze(args, dn)
        summary = _write_tables(out, pd, raw, args.columns, dn, labels, pca, tree)
        if plt is not None:
            _plots(out, plt, args.columns, dn, labels, pca, tree)
        metadata = {
            "schema_version": 1,
            "complete": True,
            "created_at": datetime.now(UTC).isoformat(),
            "input": {"name": source.name, "sha256": digest},
            "features": args.columns,
            "transformations": fitted,
            "row_count": len(raw),
            "nugget_count": dn.n_nuggets,
            "analyses": [] if args.analysis == ["none"] else args.analysis,
            "parameters": {
                key: value
                for key, value in vars(args).items()
                if key not in {"command", "input", "out", "columns", "transform", "analysis"}
            },
            "versions": {
                "datanuggets": __version__,
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "scipy", "numba", "pandas")},
            },
            "conventions": {
                "indices": "zero-based; row0 follows CSV data records, excluding the header",
                "features": "nugget centers and PCA use transformed analysis units",
                "cluster_summary": "means use original selected feature values",
                "pca": "scores and variance describe nugget centers, not within-nugget variation",
            },
        }
        (out / "run.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        incomplete.unlink()
    except BaseException:
        print(f"Run incomplete; partial outputs are in {out.resolve()}", file=sys.stderr)
        raise
    print(f"Created {dn.n_nuggets:,} occupied nuggets representing {int(dn.weights.sum()):,} rows.")
    for record in summary:
        print(
            f"{record['method']} cluster {record['cluster']}: {record['rows']:,} rows ({record['percent']:.1f}%)"
        )
    if pca is not None:
        ratios = np.square(pca.sdev) / np.square(pca.sdev).sum()
        print(
            "Nugget-center PCA variance: "
            + ", ".join(f"PC{i + 1} {v:.1%}" for i, v in enumerate(ratios))
        )
    print(f"Outputs are in {out.resolve()}")


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    try:
        _run(args)
    except (ValueError, OSError, csv.Error, UnicodeError, np.linalg.LinAlgError) as exc:
        raise SystemExit(f"datanuggets: {exc}") from None
    except KeyboardInterrupt:
        raise SystemExit("datanuggets: interrupted") from None


if __name__ == "__main__":
    main()
