"""Pick the cells to cluster from the gate figures, then cluster them: each FCS file is gated on
FSC-A against SSC-A with Cabrera's lymphmacro_finder, its numbered populations are shown and
counted, you choose which to keep (one, several, or all, file by file), and the kept events'
fluorescence channels go into data nuggets per file and weighted Ward clustering (Whclust) of
all the files' nuggets together, so a cluster is the same cells in every file.

    datanuggets-flow FOLDER                   asks which populations to keep in each file
    datanuggets-flow FOLDER --pops 1,2        keeps Pop 1 and Pop 2 of every file
    datanuggets-flow a.fcs b.fcs --pops "1,2;1"   Pop 1 and 2 of a.fcs, Pop 1 of b.fcs
    datanuggets-flow FOLDER --t11 22,20,18,18 --t21 0.25,0.5,0.5,0.5   cfc.py's thresholds

A folder's files go in name order, so per-file thresholds and --pops follow that order.

Needs the flow extra: pip install "datanuggets[flow]". Events are read as stored, so a file
with a real compensation matrix gets a warning; spectral (Aurora) files are unmixed already."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SPILL_TOL = 1e-3  # how far from the identity a file's spillover matrix may be before a warning


def mirrored_area(x, idx, idy, t1=35, t2=0.85):
    """cfc.py's find_area as he wrote it: each cell's density read right, its distance measured
    from the mirrored peak (idy, idx). --disc his gates with it, so the populations and their
    numbers are the ones his cfc.py figures show; datanuggets.flow.find_area measures from the
    peak itself, which is --disc peak."""
    rows, cols = x.shape
    ii, jj = np.meshgrid(np.arange(1, rows + 1), np.arange(1, cols + 1), indexing="ij")
    gr = np.column_stack([ii.ravel(), jj.ravel()])
    d = np.linalg.norm(gr - np.array([[idx, idy]]), axis=1)
    mask = (d < t1) & (x.flatten(order="F") > t2 * x[int(idx) - 1, int(idy) - 1])
    return gr[mask][:, [1, 0]]


def per_file(text: str, n: int, name: str) -> list[float]:
    """One number for every file, or one per file separated by commas, as n numbers."""
    try:
        v = [float(t) for t in str(text).split(",")]
    except ValueError as e:
        raise SystemExit(f"{name} takes numbers like 22 or 22,20,18,18, got {text!r}") from e
    if len(v) not in (1, n):
        raise SystemExit(f"{name} gives {len(v)} values for {n} files; give one, or one per file")
    return v * n if len(v) == 1 else v


def parse_pops(text: str, n: int) -> list[int]:
    """'1', '1,2', '1 2', '1-3' or 'all' as the sorted population numbers, 1 to n. 'none' gives
    no populations, which leaves the file out."""
    t = text.strip().lower()
    if t == "all":
        return list(range(1, n + 1))
    if t == "none":
        return []
    pops = set()
    for part in t.replace(",", " ").split():
        lo, dash, hi = part.partition("-")
        if not lo.isdigit() or (dash and not hi.isdigit()):
            raise ValueError(f"{part!r} isn't a population number or a range like 1-3")
        a, b = int(lo), int(hi) if dash else int(lo)
        pops.update(range(min(a, b), max(a, b) + 1))
    if not pops:
        raise ValueError("no population given; type numbers like 1,2, or all, or none")
    bad = [p for p in sorted(pops) if not 1 <= p <= n]
    if bad:
        raise ValueError(f"there is no Pop {bad[0]}; this file has Pop 1 to Pop {n}")
    return sorted(pops)


def pick_events(masks: np.ndarray, pops: list[int]) -> np.ndarray:
    """The events in any of the populations pops (numbered from 1), as one bool per event. An
    event in two of them counts once."""
    masks = np.asarray(masks, dtype=bool)
    if not pops:
        return np.zeros(len(masks), dtype=bool)
    return masks[:, [p - 1 for p in pops]].any(axis=1)


def read_fcs(path) -> tuple[np.ndarray, list[str], list[str]]:
    """A file's events, one column per channel, with each channel's $PnN name and $PnS label
    ('' when it has none). Warns when the file's spillover matrix isn't the identity."""
    try:
        import flowio
    except ImportError as e:  # an optional dependency
        raise ImportError('reading FCS files needs flowio: pip install "datanuggets[flow]"') from e
    fd = flowio.FlowData(str(path))
    c = fd.channel_count
    pnn = [fd.text[f"p{k}n"] for k in range(1, c + 1)]
    pns = [fd.text.get(f"p{k}s", "").strip() for k in range(1, c + 1)]
    spill = fd.text.get("spillover") or fd.text.get("spill")
    if spill:
        v = spill.split(",")
        m = int(v[0])
        s = np.asarray(v[1 + m :], dtype=float).reshape(m, m)
        if np.abs(s - np.eye(m)).max() > SPILL_TOL:
            print(
                f"warning: {Path(path).name} has a compensation matrix, but its events are "
                "clustered as stored, uncompensated"
            )
    return np.asarray(fd.as_array(preprocess=True), dtype=float), pnn, pns


def _fcs_paths(items) -> list[Path]:
    paths = []
    for item in map(Path, items):
        if item.is_dir():  # a folder's .fcs files by name, leaving out macOS's ._ copies
            paths += sorted(
                p
                for p in item.iterdir()
                if p.suffix.lower() == ".fcs" and not p.name.startswith(".")
            )
        elif item.is_file():
            paths.append(item)
        else:
            raise SystemExit(f"{item}: no such file or folder")
    if not paths:
        raise SystemExit(f"no .fcs files in {', '.join(map(str, items))}")
    return paths


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="datanuggets-flow",
        description="Gate FCS files on FSC-A vs SSC-A, keep the populations you pick, and cluster "
        "their cells with data nuggets and weighted Ward clustering.",
    )
    p.add_argument("paths", nargs="+", help="FCS files, or folders of them")
    p.add_argument(
        "--pops",
        help='populations to keep, e.g. "1,2" for every file or "1,2;1;1" '
        "file by file; without it you are asked",
    )
    p.add_argument("--out", default="datanuggets-out", help="folder for the outputs")
    p.add_argument("--k", type=int, default=8, help="clusters cut from the tree")
    p.add_argument(
        "--cofactor",
        type=float,
        default=6000,
        help="fluorescence is clustered as arcsinh(x / cofactor); 0 keeps it linear",
    )
    p.add_argument("--dn-num2", type=int, default=1000, help="nuggets per file")
    p.add_argument(
        "--channels",
        help="comma-separated channels to cluster (default: every -A channel but FSC and SSC)",
    )
    p.add_argument("--fsc", default="FSC-A")
    p.add_argument("--ssc", default="SSC-A")
    p.add_argument(
        "--disc",
        choices=("his", "peak"),
        default="his",
        help="his: cfc.py's gates, so the populations match his figures; peak: "
        "datanuggets.flow.find_area, the disc centred on the peak",
    )
    p.add_argument(
        "--t11",
        default="22",
        help="the first peak's distance threshold, one for all files or one per file, e.g. "
        "22,20,18,18 as in cfc.py",
    )
    p.add_argument(
        "--t21",
        default="0.25",
        help="the first peak's density threshold, the same way, e.g. 0.25,0.5,0.5,0.5",
    )
    p.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    return p


def _live_input(prompt, plt, show):
    """input(), with the figure window kept responsive (zoom, pan, resize) until something is
    typed in the terminal: input() alone stops the window's event loop while it waits."""
    print(prompt, end="", flush=True)
    if show:
        try:
            import msvcrt  # Windows: wait for the first key press

            while not msvcrt.kbhit():
                plt.pause(0.1)
        except ImportError:  # macOS and Linux: wait for a line
            import select

            while not select.select([sys.stdin], [], [], 0)[0]:
                plt.pause(0.1)
    try:
        return input()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\ncancelled; nothing was clustered") from None


def _ask(samples, plt, show):
    """Asks, file by file, which populations to keep. Enter keeps the previous file's choice."""
    if not sys.stdin.isatty():
        raise SystemExit('no terminal to ask in; pass --pops, e.g. --pops "1,2"')
    if show:
        plt.show(block=False)
        plt.pause(0.5)
    print(
        "\nWhich populations hold the cells you want? Each file's Pop numbers are the ones in "
        "its own legend in the figure, and the same number can be a different circle in "
        'another file. Type numbers like 1 or 1,2 or 1-3, "all", or "none" to leave the file '
        "out; Enter repeats the previous file's choice, Ctrl+C stops. Click this terminal "
        "before typing."
    )
    prev, chosen = None, []
    for i, s in enumerate(samples):
        n = s["masks"].shape[1]
        print(f"\nFile {i + 1}, {s['path'].name}:")
        print("\n".join(s["table"]))
        if n == 0:
            print("  no populations found, so this file is left out")
            chosen.append([])
            continue
        while True:
            hint = f" [{','.join(map(str, prev)) or 'none'}]" if prev is not None else ""
            text = _live_input(f"Keep which, Pop 1 to {n}{hint}: ", plt, show)
            if not text.strip() and prev is not None:
                text = ",".join(map(str, prev)) if prev else "none"
            try:
                pops = parse_pops(text, n)
                break
            except ValueError as e:
                print(f"  {e}")
        chosen.append(pops)
        prev = pops
    return chosen


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    try:
        import matplotlib
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            'datanuggets-flow needs the flow extra: pip install "datanuggets[flow]"'
        ) from e
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from datanuggets import create_dn, cut_tree, plot, whclust, wmean, wpca
    from datanuggets.flow import find_area
    from datanuggets.flow.gate import _finder

    if args.k < 1:
        raise SystemExit(f"--k is {args.k}; it takes 1 or more clusters")
    if not 2 <= args.dn_num2 < 10000:
        raise SystemExit(f"--dn-num2 is {args.dn_num2}; it takes 2 to 9999 nuggets per file")
    paths = _fcs_paths(args.paths)
    t11 = per_file(args.t11, len(paths), "--t11")
    t21 = per_file(args.t21, len(paths), "--t21")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("nuggets_*.csv"), *out.glob("events_*.csv")]:
        old.unlink()  # an earlier run's, which could be taken for this run's
    spec = args.pops.split(";") if args.pops else None
    if spec is not None and len(spec) not in (1, len(paths)):
        raise SystemExit(
            f"--pops gives {len(spec)} choices for {len(paths)} files; give one for "
            "all of them or one per file, separated by ;"
        )

    # 1. Each file's populations, counted and drawn on one figure
    cols = min(len(paths), 2)
    rows = -(-len(paths) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), squeeze=False)
    samples = []
    for i, path in enumerate(paths):
        events, pnn, pns = read_fcs(path)
        for c in (args.fsc, args.ssc):
            if c not in pnn:
                raise SystemExit(f"{path.name} has no channel {c}; its channels: {', '.join(pnn)}")
        fsc, ssc = events[:, pnn.index(args.fsc)], events[:, pnn.index(args.ssc)]
        area = mirrored_area if args.disc == "his" else find_area
        masks = _finder(fsc, ssc, 11, 11, t11[i], t21[i], 20, 0.5, area)
        table = [
            f"  Pop {j + 1}: {m.sum():>9,} events ({100 * m.mean():4.1f}%), median FSC-A "
            f"{np.median(fsc[m]) if m.any() else 0:>9,.0f}, SSC-A "
            f"{np.median(ssc[m]) if m.any() else 0:>9,.0f}"
            for j, m in enumerate(masks.T)
        ]
        print(f"\nFile {i + 1}, {path.name}: {len(fsc):,} events")
        print("\n".join(table))
        plot.gates(fsc, ssc, masks, title=f"File {i + 1}: {path.name}", ax=axs.flat[i])
        samples.append({"path": path, "masks": masks, "pnn": pnn, "pns": pns, "table": table})
    for ax in axs.flat[len(paths) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "gates.png", dpi=110)
    print(f"\nThe populations are drawn in {out / 'gates.png'}")

    # 2. The populations to keep in each file
    if spec is None:
        chosen = _ask(samples, plt, not args.no_show)
    else:
        try:
            chosen = [
                parse_pops(spec[0 if len(spec) == 1 else i], s["masks"].shape[1])
                for i, s in enumerate(samples)
            ]
        except ValueError as e:
            raise SystemExit(f"--pops: {e}") from e

    # 3. The kept events' fluorescence channels, the same ones in every file
    pnn, pns = samples[0]["pnn"], samples[0]["pns"]
    if args.channels:
        channels = [c.strip() for c in args.channels.split(",")]
    else:
        channels = [c for c in pnn if c.endswith("-A") and not c.startswith(("FSC", "SSC"))]
    for s in samples:
        missing = [c for c in channels if c not in s["pnn"]]
        if missing:
            raise SystemExit(f"{s['path'].name} has no channel {missing[0]}")
    labels = [f"{pns[pnn.index(c)]} ({c})" if pns[pnn.index(c)] else c for c in channels]

    kept, picks = [], []
    for i, (s, pops) in enumerate(zip(samples, chosen)):
        keep = pick_events(s["masks"], pops)
        picks.append(
            {
                "file_number": i + 1,
                "file": s["path"].name,
                "populations": ",".join(map(str, pops)) or "none",
                "events": len(keep),
                "kept": int(keep.sum()),
                "percent": 100 * keep.mean(),
            }
        )
        if keep.sum() < 2:
            continue  # left out, or too few cells for data nuggets
        events, pnn_i, _ = read_fcs(s["path"])  # read again, so only one file is in memory
        x = events[np.ix_(np.flatnonzero(keep), [pnn_i.index(c) for c in channels])]
        x = np.arcsinh(x / args.cofactor) if args.cofactor else x
        kept.append((i + 1, s["path"].name, np.flatnonzero(keep), np.ascontiguousarray(x)))
    pd.DataFrame(picks).to_csv(out / "selection.csv", index=False)
    print("\n" + pd.DataFrame(picks).to_string(index=False))
    if not kept:
        raise SystemExit("no file has cells kept, so there is nothing to cluster")

    # 4. Data nuggets of each file's kept cells, then Whclust of all their nuggets together
    dns = []
    for num, name, rows_i, x in kept:
        dn = create_dn(x, dn_num2=min(args.dn_num2, len(x)), n_jobs=-1)
        print(f"File {num}, {name}: {len(x):,} cells, {dn.n_nuggets:,} nuggets")
        dns.append(dn)
    centers = np.vstack([dn.centers for dn in dns])
    weights = np.concatenate([dn.weights for dn in dns]).astype(float)
    use = weights > 0  # a nugget no cell joined has no say in the clusters
    if use.sum() < 2:
        raise SystemExit("the kept cells make fewer than 2 nuggets, too few to cluster")
    k = min(args.k, int(use.sum()))
    tree = whclust(centers[use], weights[use])
    cl = np.full(len(centers), -1)
    cl[use] = cut_tree(tree, k=k)

    # 5. Outputs. N in nuggets_N.csv and events_N.csv is the file's number, as the prompt and
    #    selection.csv number it. Clusters are numbered from 1; event0 is the event's row in its
    #    file and nugget0 the nugget's row in nuggets_N.csv, both numbered from 0
    summary, start = [], 0
    for (num, name, rows_i, _), dn in zip(kept, dns):
        part = cl[start : start + dn.n_nuggets]
        start += dn.n_nuggets
        nug = pd.DataFrame(dn.centers, columns=labels)
        nug.insert(0, "nugget0", np.arange(dn.n_nuggets))
        nug["weight"], nug["scale"], nug["cluster"] = dn.weights, dn.scales, part + 1
        nug.to_csv(out / f"nuggets_{num}.csv", index=False)
        ev = part[dn.assignments]
        pd.DataFrame({"event0": rows_i, "nugget0": dn.assignments, "cluster": ev + 1}).to_csv(
            out / f"events_{num}.csv", index=False
        )
        counts = np.bincount(ev, minlength=k)
        summary += [
            {
                "file_number": num,
                "file": name,
                "cluster": c + 1,
                "cells": int(counts[c]),
                "percent": 100 * counts[c] / len(ev),
            }
            for c in range(k)
        ]
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "cluster_summary.csv", index=False)
    print("\nPercent of each file's kept cells in each cluster:")
    print("(columns are file numbers; selection.csv names the files)")
    table = summary.pivot(index="cluster", columns="file_number", values="percent")
    print(table.round(1).to_string())
    cc = pd.DataFrame(wmean(centers[use], cl[use], weights[use]), columns=labels)
    cc.insert(0, "cluster", np.arange(1, k + 1))
    cc.to_csv(out / "cluster_centers.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot.dendrogram(tree, labels=False, hang=-1, ax=ax)
    ax.set_title(f"Whclust of {use.sum():,} nuggets")
    fig.savefig(out / "dendrogram.png", dpi=150)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot.scores(wpca(centers[use], wrow=weights[use]), w=weights[use], labels=cl[use], ax=ax)
    ax.set_title(f"Nuggets of the kept cells, {k} clusters")
    fig.savefig(out / "clusters_pca.png", dpi=150)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cc[labels].to_numpy().T, aspect="auto", cmap="viridis")
    ax.set_xticks(range(k), [str(c) for c in range(1, k + 1)])
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("cluster")
    fig.colorbar(im, ax=ax, label=f"arcsinh(x / {args.cofactor:g})" if args.cofactor else "x")
    fig.tight_layout()
    fig.savefig(out / "cluster_centers.png", dpi=150)
    print(f"\nOutputs are in {out.resolve()}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
