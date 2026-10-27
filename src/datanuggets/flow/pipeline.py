"""The datanuggets-flow run as library steps, which a backend calls without the CLI: nothing here
reads the terminal or prints to it. A step reports what the CLI prints as a StageEvent, whose
message the CLI prints and whose data a backend reads. read_fcs reads a file with flowio, the flow
extra's reader, which loads only when a file is read; pandas loads in the steps that make tables,
scikit-learn only for the t-SNE, and matplotlib only for the figures."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import numpy as np

from datanuggets import create_dn, cut_tree, whclust, wmean
from datanuggets._defaults import (
    FLOW_CHANNEL_SUFFIX,
    FLOW_GATE_DPI,
    FLOW_RESULT_DPI,
    FLOW_SCATTER_PREFIXES,
    FLOW_SPILL_TOL,
    TSNE_ARGS,
)
from datanuggets.exceptions import DataNuggetWarning, FlowError
from datanuggets.flow.gate import gate_peaks
from datanuggets.flow.settings import check_channels

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from datanuggets.flow.gate import FinderPeaks
    from datanuggets.flow.settings import ClusterSettings, GateSettings
    from datanuggets.result import DataNuggets, WhclustResult

MIN_KEPT_EVENTS = 2  # create_dn needs two rows; a file keeping fewer is left out
MIN_OCCUPIED_NUGGETS = 3  # the fewest nuggets in use that weighted Ward clustering is run on


def read_fcs(path) -> tuple[np.ndarray, list[str], list[str]]:
    """A file's events, one column per channel, with each channel's $PnN name and $PnS label
    ('' when it has none). The events are flowio's, as stored: a spillover matrix further than
    FLOW_SPILL_TOL from the identity gives a DataNuggetWarning, since nothing compensates them."""
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
        if np.abs(s - np.eye(m)).max() > FLOW_SPILL_TOL:
            warnings.warn(
                f"{Path(path).name} has a compensation matrix, but its events are clustered as "
                "stored, uncompensated",
                DataNuggetWarning,
                stacklevel=2,
            )
    return np.asarray(fd.as_array(preprocess=True), dtype=float), pnn, pns


class StageEvent(NamedTuple):
    """What a step reports as it goes: its stage ("gate", "select", "nuggets" or "cluster"), the
    file it is about (numbered from 1; None for the whole run) of n_files, message, the lines the
    CLI prints for it, and data, the same facts as values."""

    stage: str
    file: int | None
    n_files: int
    message: str
    data: Mapping[str, object]


Reader = Callable[[Path], tuple[np.ndarray, list[str], list[str]]]  # read_fcs, or a stand-in
OnEvent = Callable[[StageEvent], None]


class PopStats(NamedTuple):
    """A population's size and place: its events, as a percent of the file's, the median of each
    scatter channel over them (0 for an empty population), and its peak's grid cell."""

    pop: int  # numbered from 1
    events: int
    percent: float
    median_fsc: float
    median_ssc: float
    peak: tuple[int, int]


class Sample(TypedDict):
    """A gated file, as gate_files gives it. It is a dict, so the CLI's prompt reads it as before;
    fsc and ssc are kept for gates.png, and are None in a sample state.load_gates reads, whose
    gates.png is drawn already."""

    number: int  # numbered from 1, in the order the files were given
    path: Path
    n_events: int
    pnn: list[str]
    pns: list[str]
    masks: np.ndarray  # one bool column per population, one row per event
    peaks: FinderPeaks  # what local_maxima.png draws
    stats: list[PopStats]
    table: list[str]  # population_table of stats
    local_maxima: int
    t11: float
    t21: float
    fsc: np.ndarray | None
    ssc: np.ndarray | None


class Kept(NamedTuple):
    """A file's kept cells as data nuggets: the file's number and name, and the cells' rows in
    it, numbered from 0."""

    number: int
    name: str
    rows: np.ndarray
    dn: DataNuggets


class Pool(NamedTuple):
    """Every kept file's nuggets, pooled in the files' order, and their clusters."""

    centers: np.ndarray
    weights: np.ndarray
    use: np.ndarray  # the nuggets some cell joined; the others have no say in the clusters
    tree: WhclustResult  # Whclust of the nuggets in use
    cl: np.ndarray  # each nugget's cluster, numbered from 0, or -1 for one not in use
    k: int


class Tables(NamedTuple):
    """The run's tables: nuggets_N.csv and events_N.csv by file number N, cluster_summary.csv
    and cluster_centers.csv."""

    nuggets: dict[int, pd.DataFrame]
    events: dict[int, pd.DataFrame]
    summary: pd.DataFrame
    centers: pd.DataFrame


class RunResult(NamedTuple):
    """What cluster_stage made: the channels clustered and their labels, selection.csv's table,
    each kept file's nuggets, their pool and clusters, the tables, the t-SNE of the nuggets in use
    (None without it), and the files written, in the order they were written."""

    channels: list[str]
    labels: list[str]
    selection: pd.DataFrame
    kept: list[Kept]
    pool: Pool
    tables: Tables
    tsne: np.ndarray | None
    files: list[Path]


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


def pick_events(masks: np.ndarray, pops: Sequence[int]) -> np.ndarray:
    """The events in any of the populations pops (numbered from 1), as one bool per event. An
    event in two of them counts once."""
    masks = np.asarray(masks, dtype=bool)
    if not pops:
        return np.zeros(len(masks), dtype=bool)
    return masks[:, [p - 1 for p in pops]].any(axis=1)


def selection_choices(spec: str, n_files: int) -> list[str]:
    """spec's choices, separated by ;, when it gives one for every file or one for each of
    n_files, and a FlowError naming pops when it gives neither. The rule is kept here alone: the
    CLI counts --pops with it before any file is read, and parse_selection reads its choices."""
    choices = spec.split(";")
    if len(choices) not in (1, n_files):
        raise FlowError(
            f"{{param}} gives {len(choices)} choices for {n_files} files; give one for all of "
            "them or one per file, separated by ;",
            "pops",
        )
    return choices


def parse_selection(spec: str, counts: Sequence[int]) -> list[list[int]]:
    """The populations to keep in each file, from one choice for every file ("1,2") or one per
    file separated by ; ("1,2;none;1"), each read by parse_pops. counts holds each file's number
    of populations."""
    choices = selection_choices(spec, len(counts))
    try:
        return [parse_pops(choices[0 if len(choices) == 1 else i], n) for i, n in enumerate(counts)]
    except ValueError as e:
        raise FlowError(f"{{param}}: {e}", "pops") from e


def pops_spec(chosen: Sequence[Sequence[int]]) -> str:
    """chosen as parse_selection reads it, one choice per file: [[1, 2], [], [1]] is
    "1,2;none;1"."""
    return ";".join(",".join(map(str, pops)) or "none" for pops in chosen)


def population_stats(fsc, ssc, masks, peaks) -> list[PopStats]:
    """Each population's PopStats, from the file's scatter channels, its masks and the grid cell
    each population grows from (FinderPeaks.peaks, one row per mask column)."""
    return [
        PopStats(
            j + 1,
            int(m.sum()),
            100 * float(m.mean()),
            float(np.median(fsc[m])) if m.any() else 0.0,
            float(np.median(ssc[m])) if m.any() else 0.0,
            (int(peaks[j][0]), int(peaks[j][1])),
        )
        for j, m in enumerate(np.asarray(masks, dtype=bool).T)
    ]


def population_table(stats: Sequence[PopStats], fsc_name: str, ssc_name: str) -> list[str]:
    """The line the CLI prints for each population, with its scatter channels' names."""
    return [
        f"  Pop {p.pop}: {p.events:>9,} events ({p.percent:4.1f}%), median {fsc_name} "
        f"{p.median_fsc:>9,.0f}, {ssc_name} {p.median_ssc:>9,.0f}"
        for p in stats
    ]


def gate_files(
    paths: Sequence[Path],
    settings: GateSettings,
    *,
    read: Reader = read_fcs,
    on_event: OnEvent | None = None,
) -> list[Sample]:
    """Each file read and gated on settings.fsc against settings.ssc with Cabrera's finder, its
    populations counted, in the order given. A "gate" event per file carries its lines. The
    settings are checked against the number of files before any is read, since a backend calls
    this without the CLI's checks, and a file without either scatter channel stops the run before
    it is gated."""
    if not paths:
        raise FlowError("no files to gate")
    settings.check(len(paths))
    samples = []
    for i, path in enumerate(map(Path, paths)):
        events, pnn, pns = read(path)
        for param, channel in (("fsc", settings.fsc), ("ssc", settings.ssc)):
            if channel not in pnn:
                raise FlowError(
                    f"{path.name} has no channel {channel}; its channels: {', '.join(pnn)}", param
                )
        # copies, so the events can be freed while gates.png waits for them
        fsc = events[:, pnn.index(settings.fsc)].copy()
        ssc = events[:, pnn.index(settings.ssc)].copy()
        finder = settings.for_file(i)
        masks, peaks = gate_peaks(fsc, ssc, **finder)
        stats = population_stats(fsc, ssc, masks, peaks.peaks)
        table = population_table(stats, settings.fsc, settings.ssc)
        samples.append(
            Sample(
                number=i + 1,
                path=path,
                n_events=len(fsc),
                pnn=pnn,
                pns=pns,
                masks=masks,
                peaks=peaks,
                stats=stats,
                table=table,
                local_maxima=len(peaks.candidates),
                t11=finder["t11"],
                t21=finder["t21"],
                fsc=fsc,
                ssc=ssc,
            )
        )
        if on_event:
            lines = [
                f"\nFile {i + 1}, {path.name}: {len(fsc):,} events",
                "\n".join(table),
                f"  {len(peaks.candidates)} local maxima, {len(peaks.peaks)} kept as peaks",
            ]
            data = {"events": len(fsc), "local_maxima": len(peaks.candidates), "populations": stats}
            on_event(StageEvent("gate", i + 1, len(paths), "\n".join(lines), data))
    return samples


def gate_stage(
    paths: Sequence[Path],
    out: Path,
    settings: GateSettings,
    *,
    read: Reader = read_fcs,
    on_event: OnEvent | None = None,
    figure: Callable[..., Figure] | None = None,
) -> list[Sample]:
    """The gate stage in the folder out, which must exist: each file gated by gate_files,
    local_maxima.png and gates.png drawn on figure (see flow.figures), and the gate folder's
    record saved by state.save_gates, gates.json last. cluster_stage clusters the samples it
    gives, or the ones state.load_gates reads back from out."""
    from datanuggets.flow.figures import gates_figure, local_maxima_figure, save_figures
    from datanuggets.flow.state import save_gates

    samples = gate_files(paths, settings, read=read, on_event=on_event)
    figs = {
        "local_maxima.png": local_maxima_figure(samples, figure=figure),
        "gates.png": gates_figure(samples, figure=figure),
    }
    save_figures(figs, out, FLOW_GATE_DPI)
    save_gates(out, samples, settings)
    return samples


def resolve_channels(
    samples: Sequence[Mapping], settings: ClusterSettings
) -> tuple[list[str], list[str]]:
    """The fluorescence channels to cluster, the same ones in every file, and their labels:
    settings.channels, or every -A channel of the first file but FSC and SSC, less
    settings.exclude_channels, refused by check_channels when they aren't two or more distinct
    ones. A label is the channel's $PnS with its $PnN, or the $PnN alone."""
    first = samples[0]
    pnn, pns = first["pnn"], first["pns"]
    for c in settings.exclude_channels:
        if c not in pnn:
            raise FlowError(
                f"{{param}} names {c}, which {first['path'].name} doesn't have; its channels: "
                f"{', '.join(pnn)}",
                "exclude_channels",
            )
    if settings.channels is not None:
        channels = list(settings.channels)
    else:
        channels = [
            c
            for c in pnn
            if c.endswith(FLOW_CHANNEL_SUFFIX) and not c.startswith(FLOW_SCATTER_PREFIXES)
        ]
    channels = [c for c in channels if c not in settings.exclude_channels]
    check_channels(channels)
    for s in samples:
        missing = [c for c in channels if c not in s["pnn"]]
        if missing:
            raise FlowError(
                f"{s['path'].name} has no channel {missing[0]}; its channels: {', '.join(s['pnn'])}",
                "channels",
            )
    return channels, [f"{pns[pnn.index(c)]} ({c})" if pns[pnn.index(c)] else c for c in channels]


def selection_table(samples: Sequence[Mapping], chosen: Sequence[Sequence[int]]) -> pd.DataFrame:
    """selection.csv: each file's number, name and populations kept, and its events, how many of
    them are kept and what percent."""
    import pandas as pd

    rows = []
    for i, (s, pops) in enumerate(zip(samples, chosen)):
        keep = pick_events(s["masks"], pops)
        rows.append(
            {
                "file_number": i + 1,
                "file": s["path"].name,
                "populations": ",".join(map(str, pops)) or "none",
                "events": len(keep),
                "kept": int(keep.sum()),
                "percent": 100 * keep.mean(),
            }
        )
    return pd.DataFrame(rows)


def make_nuggets(
    samples: Sequence[Mapping],
    chosen: Sequence[Sequence[int]],
    channels: Sequence[str],
    settings: ClusterSettings,
    *,
    read: Reader = read_fcs,
    on_event: OnEvent | None = None,
) -> list[Kept]:
    """Data nuggets of each file's kept cells, one file at a time: the file is read again, so one
    file's events are in memory at once, and its channels go to create_dn as arcsinh(x /
    cofactor), or as stored with cofactor 0. A file keeping fewer than MIN_KEPT_EVENTS is left
    out without being read. A "nuggets" event per file carries its line."""
    keeps = [pick_events(s["masks"], pops) for s, pops in zip(samples, chosen)]
    if not any(keep.sum() >= MIN_KEPT_EVENTS for keep in keeps):
        raise FlowError("no file has cells kept, so there is nothing to cluster")
    kept = []
    for i, (s, keep) in enumerate(zip(samples, keeps)):
        if keep.sum() < MIN_KEPT_EVENTS:
            continue  # left out, or too few cells for data nuggets
        rows = np.flatnonzero(keep)
        x = _kept_cells(s, rows, channels, settings.cofactor, read)
        dn = create_dn(
            x,
            dn_num1=settings.dn_num1,
            dn_num2=min(settings.dn_num2, len(x)),
            seed=settings.seed,
            n_jobs=settings.n_jobs,
        )
        name = s["path"].name
        kept.append(Kept(i + 1, name, rows, dn))
        if on_event:
            line = f"File {i + 1}, {name}: {len(x):,} cells, {dn.n_nuggets:,} nuggets"
            data = {"cells": len(x), "nuggets": dn.n_nuggets}
            on_event(StageEvent("nuggets", i + 1, len(samples), line, data))
    return kept


def _kept_cells(sample, rows, channels, cofactor, read) -> np.ndarray:
    """The rows' channels, read from the file again and transformed as make_nuggets clusters
    them. The file must still have the events and channels it was gated with, since the masks
    choose rows by number."""
    name = sample["path"].name
    events, pnn, _ = read(sample["path"])
    if len(events) != sample["n_events"]:
        raise FlowError(
            f"{name} isn't the file that was gated: it has {len(events):,} events, where the gated "
            f"file had {sample['n_events']:,}"
        )
    if list(pnn) != list(sample["pnn"]):
        raise FlowError(
            f"{name} isn't the file that was gated: its channels are {', '.join(pnn)}, where the "
            f"gated file's were {', '.join(sample['pnn'])}"
        )
    x = events[np.ix_(rows, [pnn.index(c) for c in channels])]
    return np.ascontiguousarray(np.arcsinh(x / cofactor) if cofactor else x)


def cluster_nuggets(kept: Sequence[Kept], k: int) -> Pool:
    """Whclust of all the kept files' nuggets together, cut into k clusters, or into as many as
    there are nuggets in use when that's fewer."""
    centers = np.vstack([kp.dn.centers for kp in kept])
    weights = np.concatenate([kp.dn.weights for kp in kept]).astype(float)
    use = weights > 0  # a nugget no cell joined has no say in the clusters
    if use.sum() < MIN_OCCUPIED_NUGGETS:
        raise FlowError(
            f"the kept cells make fewer than {MIN_OCCUPIED_NUGGETS} occupied nuggets, too few for "
            "weighted Ward clustering; keep more cells or increase {param}",
            "dn_num2",
        )
    k = min(k, int(use.sum()))
    tree = whclust(centers[use], weights[use])
    cl = np.full(len(centers), -1)
    cl[use] = cut_tree(tree, k=k)
    return Pool(centers, weights, use, tree, cl, k)


def tsne_embedding(centers: np.ndarray, settings: ClusterSettings) -> np.ndarray:
    """A 2-D t-SNE of the nugget centers, seeded with settings.tsne_seed so a run repeats, its
    shape set by TSNE_ARGS. The perplexity is capped at (n - 1) / 3 for n nuggets, as scikit-learn
    needs it below n."""
    from sklearn.manifold import TSNE

    perplexity = min(settings.tsne_perplexity, (len(centers) - 1) / 3)
    tsne = TSNE(**TSNE_ARGS, perplexity=perplexity, random_state=settings.tsne_seed)
    return tsne.fit_transform(centers)


def result_tables(kept: Sequence[Kept], pool: Pool, labels: Sequence[str]) -> Tables:
    """The run's tables. N is the file's number, as the prompt and selection.csv number it.
    Clusters are numbered from 1; event0 is the event's row in its file and nugget0 the nugget's
    row in nuggets_N.csv, both numbered from 0."""
    import pandas as pd

    labels = list(labels)
    nuggets, events, summary, start = {}, {}, [], 0
    for kp in kept:
        dn = kp.dn
        part = pool.cl[start : start + dn.n_nuggets]
        start += dn.n_nuggets
        nug = pd.DataFrame(dn.centers, columns=labels)
        nug.insert(0, "nugget0", np.arange(dn.n_nuggets))
        nug["weight"], nug["scale"], nug["cluster"] = dn.weights, dn.scales, part + 1
        nuggets[kp.number] = nug
        ev = part[dn.assignments]
        events[kp.number] = pd.DataFrame(
            {"event0": kp.rows, "nugget0": dn.assignments, "cluster": ev + 1}
        )
        counts = np.bincount(ev, minlength=pool.k)
        summary += [
            {
                "file_number": kp.number,
                "file": kp.name,
                "cluster": c + 1,
                "cells": int(counts[c]),
                "percent": 100 * counts[c] / len(ev),
            }
            for c in range(pool.k)
        ]
    use = pool.use
    centers = pd.DataFrame(
        wmean(pool.centers[use], pool.cl[use], pool.weights[use]), columns=labels
    )
    centers.insert(0, "cluster", np.arange(1, pool.k + 1))
    return Tables(nuggets, events, pd.DataFrame(summary), centers)


def write_tables(tables: Tables, out: Path) -> list[Path]:
    """nuggets_N.csv and events_N.csv for each kept file, then cluster_summary.csv and
    cluster_centers.csv, in the folder out; their paths, in that order."""
    named = []
    for num, nuggets in tables.nuggets.items():
        named += [(f"nuggets_{num}.csv", nuggets), (f"events_{num}.csv", tables.events[num])]
    named += [("cluster_summary.csv", tables.summary), ("cluster_centers.csv", tables.centers)]
    for name, table in named:
        table.to_csv(out / name, index=False)
    return [out / name for name, _ in named]


def cluster_stage(
    samples: Sequence[Mapping],
    chosen: Sequence[Sequence[int]],
    out: Path,
    settings: ClusterSettings,
    *,
    read: Reader = read_fcs,
    on_event: OnEvent | None = None,
    figure: Callable[..., Figure] | None = None,
) -> RunResult:
    """The clustering stage in the folder out, which must exist: the channels resolved,
    selection.csv, the kept cells' nuggets, Whclust of them, the tables, the t-SNE unless
    settings.tsne is off, and the figures, drawn on figure (see flow.figures). Events: "select"
    with selection.csv's table, "nuggets" per kept file, "cluster" with each cluster's percent.
    The settings, and that chosen gives one choice per sample, are checked before anything is read
    or written: a backend calls this without the CLI's checks, and a short chosen would drop its
    missing files from the tables without a word."""
    settings.check()
    if not samples:
        raise FlowError("no gated files to cluster")
    if len(chosen) != len(samples):
        raise FlowError(
            f"{{param}} gives {len(chosen)} choices for {len(samples)} files; give one per file",
            "pops",
        )
    channels, labels = resolve_channels(samples, settings)
    selection = selection_table(samples, chosen)
    selection.to_csv(out / "selection.csv", index=False)
    files = [out / "selection.csv"]
    emit = on_event or (lambda event: None)
    n = len(samples)
    table = "\n" + selection.to_string(index=False)
    emit(StageEvent("select", None, n, table, {"selection": selection}))
    kept = make_nuggets(samples, chosen, channels, settings, read=read, on_event=on_event)
    pool = cluster_nuggets(kept, settings.k)
    tables = result_tables(kept, pool, labels)
    files += write_tables(tables, out)
    percent = tables.summary.pivot(index="cluster", columns="file_number", values="percent")
    lines = [
        "\nPercent of each file's kept cells in each cluster:",
        "(columns are file numbers; selection.csv names the files)",
        percent.round(1).to_string(),
    ]
    emit(StageEvent("cluster", None, n, "\n".join(lines), {"percent": percent}))
    xy = tsne_embedding(pool.centers[pool.use], settings) if settings.tsne else None
    from datanuggets.flow.figures import result_figures, save_figures

    figs = result_figures(pool, tables.centers, labels, settings.cofactor, xy, figure=figure)
    files += save_figures(figs, out, FLOW_RESULT_DPI)
    return RunResult(channels, labels, selection, kept, pool, tables, xy, files)
