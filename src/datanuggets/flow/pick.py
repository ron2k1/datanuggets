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
with a real compensation matrix gets a warning; spectral (Aurora) files are unmixed already.

The steps are datanuggets.flow.pipeline's; this module reads the arguments, asks, and prints the
steps' events and errors."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from datanuggets._defaults import FLOW_GATE_DPI, FLOW_OUT_DIR, FLOW_RUN_NAME
from datanuggets.exceptions import DataNuggetWarning, FlowError
from datanuggets.flow import pipeline

# parse_pops and pick_events are pipeline's, and still importable from here
from datanuggets.flow.pipeline import parse_pops, pick_events  # noqa: F401
from datanuggets.flow.settings import ClusterSettings, GateSettings, cli_flag


def per_file(text: str, n: int, name: str) -> list[float]:
    """One number for every file, or one per file separated by commas, as n numbers."""
    try:
        v = [float(t) for t in str(text).split(",")]
    except ValueError as e:
        raise SystemExit(f"{name} takes numbers like 22 or 22,20,18,18, got {text!r}") from e
    if len(v) not in (1, n):
        raise SystemExit(f"{name} gives {len(v)} values for {n} files; give one, or one per file")
    return v * n if len(v) == 1 else v


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
    """The flags. One that sets a GateSettings or ClusterSettings field, under the field's name,
    defaults to None, so a run without it takes the settings' default, which its help shows."""
    gate, cluster = GateSettings(), ClusterSettings()
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
    p.add_argument("--k", type=int, help=f"clusters cut from the tree (default: {cluster.k})")
    p.add_argument(
        "--cofactor",
        type=float,
        help="fluorescence is clustered as arcsinh(x / cofactor); 0 keeps it linear "
        f"(default: {cluster.cofactor:g})",
    )
    p.add_argument(
        "--dn-num1",
        type=int,
        help="create_dn's dn_num1, the nuggets its first pass makes from a file's kept cells "
        f"before they are reduced to --dn-num2 (default: {cluster.dn_num1})",
    )
    p.add_argument("--dn-num2", type=int, help=f"nuggets per file (default: {cluster.dn_num2})")
    p.add_argument("--seed", type=int, help=f"create_dn's seed (default: {cluster.seed})")
    p.add_argument(
        "--jobs",
        type=int,
        dest="n_jobs",
        metavar="JOBS",
        help="cores create_dn uses, -1 for every one; the results are the same for any "
        f"(default: {cluster.n_jobs})",
    )
    channels = p.add_mutually_exclusive_group()
    channels.add_argument(
        "--channels",
        help="comma-separated channels to cluster (default: every -A channel but FSC and SSC)",
    )
    channels.add_argument(
        "--exclude-channels", help="comma-separated channels to leave out of the default ones"
    )
    p.add_argument("--fsc", help=f"the channel gated as FSC (default: {gate.fsc})")
    p.add_argument("--ssc", help=f"the channel gated as SSC (default: {gate.ssc})")
    p.add_argument(
        "--disc",
        choices=("his", "peak"),
        help="his: cfc.py's gates, so the populations match cfc.py's figures; peak: "
        f"datanuggets.flow.find_area, the disc centred on the peak (default: {gate.disc})",
    )
    p.add_argument(
        "--t11",
        help="the first peak's distance threshold, one for all files or one per file, e.g. "
        f"22,20,18,18 as in cfc.py (default: {gate.t11[0]})",
    )
    p.add_argument(
        "--t21",
        help="the first peak's density threshold, the same way, e.g. 0.25,0.5,0.5,0.5 "
        f"(default: {gate.t21[0]})",
    )
    p.add_argument(
        "--t12",
        type=float,
        help=f"every other peak's distance threshold, in every file (default: {gate.t12})",
    )
    p.add_argument(
        "--t22",
        type=float,
        help=f"every other peak's density threshold, in every file (default: {gate.t22})",
    )
    p.add_argument(
        "--n-row",
        type=int,
        help=f"rows of the window the finder's local maxima are found in (default: {gate.n_row})",
    )
    p.add_argument("--n-col", type=int, help=f"columns of that window (default: {gate.n_col})")
    p.add_argument(
        "--no-tsne",
        dest="tsne",
        action="store_false",
        default=None,
        help="leave out the t-SNE of the nuggets, clusters_tsne.png, and the need for scikit-learn",
    )
    p.add_argument(
        "--tsne-perplexity",
        type=float,
        help="the t-SNE's perplexity, lowered to (nuggets - 1) / 3 when that's smaller "
        f"(default: {cluster.tsne_perplexity:g})",
    )
    p.add_argument("--tsne-seed", type=int, help=f"the t-SNE's seed (default: {cluster.tsne_seed})")
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


def _pyplot(no_show: bool, *, tsne: bool):
    """matplotlib's pyplot, once the flow extra is known to be there, drawing off screen with
    --no-show. pandas is checked here too, for the tables, and scikit-learn when the t-SNE is
    drawn, so a missing one stops the run before any file is read."""
    try:
        import matplotlib
        import pandas  # noqa: F401

        if tsne:
            import sklearn  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            'datanuggets-flow needs the flow extra: pip install "datanuggets[flow]"'
        ) from e
    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _flags(cls, args, **values):
    """cls's settings from the flags named for them, values standing in for flags that are read
    another way first. A flag left out, or a value of None, keeps cls's default."""
    given = {f.name: getattr(args, f.name, None) for f in fields(cls)} | values
    return cls(**{name: v for name, v in given.items() if v is not None})


def _gate_settings(args, t11, t21) -> GateSettings:
    """GateSettings from the gate flags, with --t11 and --t21 as _check_args reads them."""
    return _flags(GateSettings, args, t11=tuple(t11), t21=tuple(t21))


def _settings(args) -> ClusterSettings:
    """The settings the flags give, checked before any file is read, so a bad value stops the run
    first; the clustering's are returned. --t11 and --t21 wait for the files to be counted."""
    for name in ("tsne_perplexity", "tsne_seed"):
        if args.tsne is False and getattr(args, name) is not None:
            raise SystemExit(f"{cli_flag(name)} has no effect with --no-tsne")
    _flags(GateSettings, args, t11=None, t21=None).check()
    cluster = _flags(
        ClusterSettings,
        args,
        channels=_channel_names(args.channels),
        exclude_channels=_channel_names(args.exclude_channels),
    )
    return cluster.check()


def _check_args(args) -> tuple[list[Path], list[float], list[float], Path, str | None]:
    """Validate arguments without writing files; return inputs, thresholds, output and --pops.
    --t11 or --t21 left out gives the settings' one value for every file."""
    paths = _fcs_paths(args.paths)
    gate = GateSettings()
    t11 = per_file(args.t11, len(paths), "--t11") if args.t11 is not None else list(gate.t11)
    t21 = per_file(args.t21, len(paths), "--t21") if args.t21 is not None else list(gate.t21)
    _gate_settings(args, t11, t21).check()
    out = (
        Path(args.out)
        if args.out
        else Path(FLOW_OUT_DIR) / datetime.now(UTC).strftime(FLOW_RUN_NAME)
    )
    choices = args.pops.split(";") if args.pops else None
    if choices is not None and len(choices) not in (1, len(paths)):
        raise SystemExit(
            f"--pops gives {len(choices)} choices for {len(paths)} files; give one for "
            "all of them or one per file, separated by ;"
        )
    return paths, t11, t21, out, args.pops or None


def _print_event(event: pipeline.StageEvent) -> None:
    """A step's lines, as the run prints them."""
    print(event.message)


def _read(path):
    """read_fcs, looked up when called, so a test's stand-in for it is the reader the steps use."""
    return read_fcs(path)


def _gate_files(args, paths, t11, t21, out, plt) -> list[dict]:
    """Each file gated by pipeline.gate_files, its populations printed as it goes, then the
    density grid, local maxima and the peak each Pop grows from drawn on one figure,
    local_maxima.png, the one the populations are chosen from. gates.png waits for the choice."""
    from datanuggets.flow.figures import local_maxima_figure, save_figures

    settings = _gate_settings(args, t11, t21)
    samples = pipeline.gate_files(paths, settings, read=_read, on_event=_print_event)
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
    """The populations to keep in each file: --pops, or asked file by file, the answers then
    printed as the --pops that repeats them without the prompt."""
    if spec is not None:
        return pipeline.parse_selection(spec, [s["masks"].shape[1] for s in samples])
    chosen = _ask(samples, plt, show)
    print(f'\nTo repeat this choice: --pops "{pipeline.pops_spec(chosen)}"')
    return chosen


def _channel_names(names: str | None) -> tuple[str, ...] | None:
    """--channels or --exclude-channels as ClusterSettings takes it: its names, or None when it
    is left out."""
    return tuple(c.strip() for c in names.split(",")) if names else None


def _choose_channels(samples, names, exclude=None) -> tuple[list[str], list[str]]:
    """The fluorescence channels to cluster and their labels, by pipeline.resolve_channels, for
    --channels and --exclude-channels given as names. The run checks them before asking for
    populations, so a bad channel stops it first."""
    settings = ClusterSettings(
        channels=_channel_names(names), exclude_channels=_channel_names(exclude) or ()
    )
    try:
        return pipeline.resolve_channels(samples, settings)
    except FlowError as e:
        raise SystemExit(e.cli(cli_flag)) from None


def main(argv=None) -> None:
    """datanuggets-flow. A FlowError, a bad setting's or a step's, stops it with its message, the
    setting written as its flag."""
    args = _parser().parse_args(argv)
    try:
        _run(args)
    except FlowError as e:
        raise SystemExit(e.cli(cli_flag)) from None


def _run(args) -> None:
    """The run: every check, then the output folder, the steps, and the figures shown."""
    cluster = _settings(args)
    plt = _pyplot(args.no_show, tsne=cluster.tsne)
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
        _choose_channels(samples, args.channels, args.exclude_channels)
        chosen = _choose_pops(samples, spec, plt, not args.no_show)
        _draw_gates(samples, plt, out)
        pipeline.cluster_stage(
            samples, chosen, out, cluster, read=_read, on_event=_print_event, figure=plt.figure
        )
        incomplete.unlink()
    except (Exception, SystemExit, KeyboardInterrupt):
        print(f"\nRun incomplete; partial outputs are in {out.resolve()}", file=sys.stderr)
        raise
    print(f"\nOutputs are in {out.resolve()}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
