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
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from datanuggets._defaults import DN_NUM1, FLOW_GATE_DPI, FLOW_RESULT_DPI
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.flow import pipeline

if TYPE_CHECKING:
    import pandas as pd

    from datanuggets.result import DataNuggets, WhclustResult


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
    """pipeline.read_fcs, its spillover warning printed as a line of the run's output. Every step
    reads through this name, so a test can stand in for the FCS reader."""
    with warnings.catch_warnings(record=True, action="always", category=DataNuggetWarning) as w:
        found = pipeline.read_fcs(path)
    for caught in w:
        if issubclass(caught.category, DataNuggetWarning):
            print(f"warning: {caught.message}")
        else:  # flowio's own, shown as they would have been
            warnings.warn_explicit(caught.message, caught.category, caught.filename, caught.lineno)
    return found


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
    p.add_argument(
        "--out",
        help="new output folder (default: outputs/flow-<timestamp>); existing paths are refused",
    )
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
        help="his: cfc.py's gates, so the populations match cfc.py's figures; peak: "
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
        "\nWhich populations hold the cells you want? Each file's Pop numbers label the peaks "
        "in its own panel of the local maxima figure, and the same number can be a different "
        'peak in another file. Type numbers like 1 or 1,2 or 1-3, "all", or "none" to leave the file '
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


def _pyplot(no_show: bool):
    """matplotlib's pyplot, once the flow extra is known to be there, drawing off screen with
    --no-show. pandas and scikit-learn are checked here too, for the tables and the t-SNE the
    steps make, so a missing one stops the run before any file is read."""
    try:
        import matplotlib
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            'datanuggets-flow needs the flow extra: pip install "datanuggets[flow]"'
        ) from e
    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _check_args(args) -> tuple[list[Path], list[float], list[float], Path, list[str] | None]:
    """Validate arguments without writing files; return inputs, thresholds, output and choices."""
    if args.k < 1:
        raise SystemExit(f"--k is {args.k}; it takes 1 or more clusters")
    if not 2 <= args.dn_num2 < DN_NUM1:  # create_dn's dn_num1, which dn_num2 stays under
        raise SystemExit(
            f"--dn-num2 is {args.dn_num2}; it takes 2 to {DN_NUM1 - 1} nuggets per file"
        )
    paths = _fcs_paths(args.paths)
    t11 = per_file(args.t11, len(paths), "--t11")
    t21 = per_file(args.t21, len(paths), "--t21")
    out = (
        Path(args.out)
        if args.out
        else Path("outputs") / datetime.now(UTC).strftime("flow-%Y%m%d-%H%M%S-%fZ")
    )
    spec = args.pops.split(";") if args.pops else None
    if spec is not None and len(spec) not in (1, len(paths)):
        raise SystemExit(
            f"--pops gives {len(spec)} choices for {len(paths)} files; give one for "
            "all of them or one per file, separated by ;"
        )
    return paths, t11, t21, out, spec


def _gate_files(args, paths, t11, t21, out, plt) -> list[dict]:
    """Each file gated with Cabrera's lymphmacro_finder and its populations counted, and the
    density grid, local maxima and the peak each Pop grows from drawn on one figure,
    local_maxima.png, the one the populations are chosen from. gates.png waits for the choice."""
    from datanuggets.flow import find_area, finder_peaks, lymphmacro_finder, mirrored_area
    from datanuggets.flow.figures import local_maxima_figure, save_figures

    area = mirrored_area if args.disc == "his" else find_area
    samples = []
    for i, path in enumerate(paths):
        events, pnn, pns = read_fcs(path)
        for c in (args.fsc, args.ssc):
            if c not in pnn:
                raise SystemExit(f"{path.name} has no channel {c}; its channels: {', '.join(pnn)}")
        fsc, ssc = events[:, pnn.index(args.fsc)], events[:, pnn.index(args.ssc)]
        masks = lymphmacro_finder(fsc, ssc, t11=t11[i], t21=t21[i], area=area)
        table = [
            f"  Pop {j + 1}: {m.sum():>9,} events ({100 * m.mean():4.1f}%), median {args.fsc} "
            f"{np.median(fsc[m]) if m.any() else 0:>9,.0f}, {args.ssc} "
            f"{np.median(ssc[m]) if m.any() else 0:>9,.0f}"
            for j, m in enumerate(masks.T)
        ]
        print(f"\nFile {i + 1}, {path.name}: {len(fsc):,} events")
        print("\n".join(table))
        fp = finder_peaks(fsc, ssc)
        print(f"  {len(fp.candidates)} local maxima, {len(fp.peaks)} kept as peaks")
        fsc, ssc = fsc.copy(), ssc.copy()  # for gates.png; copies, so events can be freed
        sample = {"path": path, "masks": masks, "pnn": pnn, "pns": pns, "table": table}
        samples.append(sample | {"peaks": fp, "fsc": fsc, "ssc": ssc})
    figs = {"local_maxima.png": local_maxima_figure(samples, figure=plt.figure)}
    save_figures(figs, out, FLOW_GATE_DPI)
    print(f"\nThe local maxima and each Pop's peak are drawn in {out / 'local_maxima.png'}")
    return samples


def _draw_gates(samples, plt, out) -> None:
    """gates.png, once the populations are chosen: each file's events in gray and each of its
    populations in color, one panel per file."""
    from datanuggets.flow.figures import gates_figure, save_figures

    save_figures({"gates.png": gates_figure(samples, figure=plt.figure)}, out, FLOW_GATE_DPI)
    print(f"The populations are drawn in {out / 'gates.png'}")


def _choose_pops(samples, spec, plt, show) -> list[list[int]]:
    """The populations to keep in each file: --pops, or asked file by file."""
    if spec is None:
        return _ask(samples, plt, show)
    try:
        return [
            parse_pops(spec[0 if len(spec) == 1 else i], s["masks"].shape[1])
            for i, s in enumerate(samples)
        ]
    except ValueError as e:
        raise SystemExit(f"--pops: {e}") from e


def _choose_channels(samples, names) -> tuple[list[str], list[str]]:
    """The fluorescence channels to cluster, the same ones in every file, and their labels:
    --channels, or every -A channel of the first file but FSC and SSC. A label is the channel's
    $PnS with its $PnN, or the $PnN alone."""
    pnn, pns = samples[0]["pnn"], samples[0]["pns"]
    if names:
        channels = [c.strip() for c in names.split(",")]
    else:
        channels = [c for c in pnn if c.endswith("-A") and not c.startswith(("FSC", "SSC"))]
    if len(set(channels)) < 2:
        raise SystemExit(
            "choose at least two distinct fluorescence channels for clustering and PCA"
        )
    if len(set(channels)) != len(channels):
        raise SystemExit("--channels repeats a channel; list each channel only once")
    for s in samples:
        missing = [c for c in channels if c not in s["pnn"]]
        if missing:
            raise SystemExit(
                f"{s['path'].name} has no channel {missing[0]}; its channels: {', '.join(s['pnn'])}"
            )
    return channels, [f"{pns[pnn.index(c)]} ({c})" if pns[pnn.index(c)] else c for c in channels]


def _keep_events(samples, chosen, channels, cofactor, out) -> list[tuple]:
    """Each file's kept events as (file number, file name, their rows in the file, their channels
    as arcsinh(x / cofactor), or as stored with cofactor 0), and selection.csv. A file with fewer
    than 2 kept events is left out."""
    import pandas as pd

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
        x = np.arcsinh(x / cofactor) if cofactor else x
        kept.append((i + 1, s["path"].name, np.flatnonzero(keep), np.ascontiguousarray(x)))
    pd.DataFrame(picks).to_csv(out / "selection.csv", index=False)
    print("\n" + pd.DataFrame(picks).to_string(index=False))
    if not kept:
        raise SystemExit("no file has cells kept, so there is nothing to cluster")
    return kept


class _Pool(NamedTuple):
    """Every kept file's nuggets, pooled in the files' order, and their clusters."""

    dns: list[DataNuggets]  # one per kept file
    centers: np.ndarray
    weights: np.ndarray
    use: np.ndarray  # the nuggets some cell joined; the others have no say in the clusters
    tree: WhclustResult  # Whclust of the nuggets in use
    cl: np.ndarray  # each nugget's cluster, numbered from 0, or -1 for one not in use
    k: int


def _cluster(kept, dn_num2, k) -> _Pool:
    """Data nuggets of each file's kept cells, then Whclust of all the files' nuggets together,
    cut into k clusters, or into as many as there are nuggets in use when that's fewer."""
    from datanuggets import create_dn, cut_tree, whclust

    dns = []
    for num, name, _, x in kept:
        dn = create_dn(x, dn_num2=min(dn_num2, len(x)), n_jobs=-1)
        print(f"File {num}, {name}: {len(x):,} cells, {dn.n_nuggets:,} nuggets")
        dns.append(dn)
    centers = np.vstack([dn.centers for dn in dns])
    weights = np.concatenate([dn.weights for dn in dns]).astype(float)
    use = weights > 0  # a nugget no cell joined has no say in the clusters
    if use.sum() < 3:
        raise SystemExit(
            "the kept cells make fewer than 3 occupied nuggets, too few for weighted Ward "
            "clustering; keep more cells or increase --dn-num2"
        )
    k = min(k, int(use.sum()))
    tree = whclust(centers[use], weights[use])
    cl = np.full(len(centers), -1)
    cl[use] = cut_tree(tree, k=k)
    return _Pool(dns, centers, weights, use, tree, cl, k)


def _write_tables(kept, pool, labels, out) -> pd.DataFrame:
    """nuggets_N.csv and events_N.csv for each kept file, then cluster_summary.csv and
    cluster_centers.csv, whose table it returns. N is the file's number, as the prompt and
    selection.csv number it. Clusters are numbered from 1; event0 is the event's row in its file
    and nugget0 the nugget's row in nuggets_N.csv, both numbered from 0."""
    import pandas as pd

    from datanuggets import wmean

    summary, start = [], 0
    for (num, name, rows_i, _), dn in zip(kept, pool.dns):
        part = pool.cl[start : start + dn.n_nuggets]
        start += dn.n_nuggets
        nug = pd.DataFrame(dn.centers, columns=labels)
        nug.insert(0, "nugget0", np.arange(dn.n_nuggets))
        nug["weight"], nug["scale"], nug["cluster"] = dn.weights, dn.scales, part + 1
        nug.to_csv(out / f"nuggets_{num}.csv", index=False)
        ev = part[dn.assignments]
        pd.DataFrame({"event0": rows_i, "nugget0": dn.assignments, "cluster": ev + 1}).to_csv(
            out / f"events_{num}.csv", index=False
        )
        counts = np.bincount(ev, minlength=pool.k)
        summary += [
            {
                "file_number": num,
                "file": name,
                "cluster": c + 1,
                "cells": int(counts[c]),
                "percent": 100 * counts[c] / len(ev),
            }
            for c in range(pool.k)
        ]
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "cluster_summary.csv", index=False)
    print("\nPercent of each file's kept cells in each cluster:")
    print("(columns are file numbers; selection.csv names the files)")
    table = summary.pivot(index="cluster", columns="file_number", values="percent")
    print(table.round(1).to_string())
    use = pool.use
    cc = pd.DataFrame(wmean(pool.centers[use], pool.cl[use], pool.weights[use]), columns=labels)
    cc.insert(0, "cluster", np.arange(1, pool.k + 1))
    cc.to_csv(out / "cluster_centers.csv", index=False)
    return cc


def _draw_figures(pool, cc, labels, cofactor, plt, out) -> None:
    """dendrogram.png, clusters_pca.png, clusters_tsne.png and cluster_centers.png."""
    from datanuggets.flow.figures import result_figures, save_figures

    xy = _tsne(pool.centers[pool.use])
    figs = result_figures(pool, cc, labels, cofactor, xy, figure=plt.figure)
    save_figures(figs, out, FLOW_RESULT_DPI)


def _tsne(x) -> np.ndarray:
    """A 2-D t-SNE of the nugget centers, seeded so a run repeats. Perplexity is lowered from 30
    when there are too few nuggets for it, as scikit-learn requires it below their number."""
    from sklearn.manifold import TSNE

    perplexity = min(30.0, (len(x) - 1) / 3)
    return TSNE(perplexity=perplexity, init="pca", random_state=0).fit_transform(x)


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    plt = _pyplot(args.no_show)
    paths, t11, t21, out, spec = _check_args(args)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"output path already exists: {out}; choose a new --out folder") from None
    incomplete = out / "INCOMPLETE.txt"
    incomplete.write_text(
        "This run has not completed. Its tables and figures may be partial.\n"
        "A successful run removes this marker; rerun in a new output folder.\n",
        encoding="utf-8",
    )
    try:
        samples = _gate_files(args, paths, t11, t21, out, plt)
        channels, labels = _choose_channels(samples, args.channels)
        chosen = _choose_pops(samples, spec, plt, not args.no_show)
        _draw_gates(samples, plt, out)
        kept = _keep_events(samples, chosen, channels, args.cofactor, out)
        pool = _cluster(kept, args.dn_num2, args.k)
        cc = _write_tables(kept, pool, labels, out)
        _draw_figures(pool, cc, labels, args.cofactor, plt, out)
        incomplete.unlink()
    except (Exception, SystemExit, KeyboardInterrupt):
        print(f"\nRun incomplete; partial outputs are in {out.resolve()}", file=sys.stderr)
        raise
    print(f"\nOutputs are in {out.resolve()}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
