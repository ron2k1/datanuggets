"""Pick the cells to cluster from the gate figures, then cluster them: each FCS file is gated on
FSC-A against SSC-A with Cabrera's lymphmacro_finder, its numbered populations are shown and
counted, you choose which to keep (one, several, or all, file by file), and the kept events'
fluorescence channels go into data nuggets per file and weighted Ward clustering (Whclust) of
all the files' nuggets together, so a cluster is the same cells in every file.

    datanuggets-flow FOLDER                   asks which populations to keep in each file
    datanuggets-flow FOLDER --pops 1,2        keeps Pop 1 and Pop 2 of every file
    datanuggets-flow a.fcs b.fcs --pops "1,2;1"   Pop 1 and 2 of a.fcs, Pop 1 of b.fcs
    datanuggets-flow FOLDER --t11 22,20,18,18 --t21 0.25,0.5,0.5,0.5   cfc.py's thresholds
    datanuggets-flow FOLDER --gates-only --out GATES   gates and draws the files, then stops
    datanuggets-flow --from-gates GATES       asks over GATES' figures, then clusters

A folder's files go in name order, so per-file thresholds and --pops follow that order. The two
stages give the one-shot run's tables: --from-gates clusters with the masks GATES holds, reading
each file only once its size and SHA-256 are the ones it was gated with.

Needs the flow extra: pip install "datanuggets[flow]". Events are read as stored, so a file
with a real compensation matrix gets a warning; spectral (Aurora) files are unmixed already.

The steps are datanuggets.flow.pipeline's; this module reads the arguments, asks, and prints the
steps' events and errors."""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from contextlib import contextmanager
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from datanuggets._defaults import FLOW_GATE_DPI, FLOW_GATES_NAME, FLOW_OUT_DIR, FLOW_RUN_NAME
from datanuggets.exceptions import DataNuggetWarning, FlowError
from datanuggets.flow import pipeline, state

# parse_pops and pick_events are pipeline's, and still importable from here
from datanuggets.flow.pipeline import parse_pops, pick_events  # noqa: F401
from datanuggets.flow.settings import ClusterSettings, GateSettings, cli_flag


def per_file(text: str | None, name: str) -> tuple[float, ...] | None:
    """--t11 or --t21 as GateSettings takes it: one number for every file, or one per file
    separated by commas, or None when it is left out. GateSettings.check counts them against the
    files, so the rule is kept in one place."""
    if text is None:
        return None
    try:
        return tuple(float(t) for t in str(text).split(","))
    except ValueError as e:
        raise SystemExit(f"{name} takes numbers like 22 or 22,20,18,18, got {text!r}") from e


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
    p.add_argument("paths", nargs="*", help="FCS files, or folders of them; none with --from-gates")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument(
        "--gates-only",
        action="store_true",
        help="gate the files, draw local_maxima.png and gates.png, save the gates in --out, and "
        "stop, without asking",
    )
    stage.add_argument(
        "--from-gates",
        metavar="GATES",
        help="choose populations over the figures of GATES, a --gates-only folder, and cluster "
        "its files, which must not have changed",
    )
    p.add_argument(
        "--pops",
        help='populations to keep, e.g. "1,2" for every file or "1,2;1;1" '
        "file by file; without it you are asked",
    )
    p.add_argument(
        "--out",
        help="new output folder (default: outputs/flow-<timestamp>, or "
        "outputs/flow-gates-<timestamp> with --gates-only); existing paths are refused",
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


def _pyplot(no_show: bool, *, tables: bool, tsne: bool):
    """matplotlib's pyplot, once the flow extra is known to be there, drawing off screen with
    --no-show. pandas is checked here too when the run makes tables, and scikit-learn when the
    t-SNE is drawn, so a missing one stops the run before any file is read."""
    try:
        import matplotlib

        if tables:
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


def _given(args, cls) -> list[str]:
    """The settings of cls whose flags were given."""
    return [f.name for f in fields(cls) if getattr(args, f.name, None) is not None]


def _mode_rules(args) -> None:
    """Refuses a flag the mode can't use, before any setting is checked or file read: FCS paths
    or a gate setting with --from-gates, which reuses its folder's files and gates, and a
    clustering setting or --pops with --gates-only, which stops before them."""
    if args.from_gates is None and not args.paths:
        raise SystemExit("give FCS files or folders, or --from-gates GATES")
    if args.from_gates is not None and args.paths:
        raise SystemExit(
            "--from-gates reads the FCS files its gate folder records; give no FCS paths"
        )
    gate = _given(args, GateSettings) if args.from_gates is not None else []
    if gate:
        raise SystemExit(
            f"{cli_flag(gate[0])} sets the gate stage, and --from-gates reuses the gates in "
            f"{args.from_gates}; run --gates-only again to change it"
        )
    cluster = _given(args, ClusterSettings) + ["pops"] * (args.pops is not None)
    if args.gates_only and cluster:
        raise SystemExit(
            f"{cli_flag(cluster[0])} sets the clustering stage, which --gates-only stops before"
        )


def _flags(cls, args, **values):
    """cls's settings from the flags named for them, values standing in for flags that are read
    another way first. A flag left out, or a value of None, keeps cls's default."""
    given = {f.name: getattr(args, f.name, None) for f in fields(cls)} | values
    return cls(**{name: v for name, v in given.items() if v is not None})


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


def _out(args, name: str) -> Path:
    """--out, or a folder under FLOW_OUT_DIR named for the UTC time by the pattern name, so each
    run has its own."""
    return Path(args.out) if args.out else Path(FLOW_OUT_DIR) / datetime.now(UTC).strftime(name)


def _check_args(args) -> tuple[list[Path], GateSettings, Path]:
    """Validate arguments without writing files; return inputs, the gate settings and output.
    The gate settings are built once, here, with --t11 and --t21 as given and checked against the
    files; one left out gives the settings' one value for every file. --pops is counted against
    the files too, by the rule parse_selection reads it with."""
    paths = _fcs_paths(args.paths)
    t11, t21 = per_file(args.t11, "--t11"), per_file(args.t21, "--t21")
    gate = _flags(GateSettings, args, t11=t11, t21=t21).check(len(paths))
    out = _out(args, FLOW_GATES_NAME if args.gates_only else FLOW_RUN_NAME)
    if args.pops is not None:
        pipeline.selection_choices(args.pops, len(paths))
    return paths, gate, out


def _print_event(event: pipeline.StageEvent) -> None:
    """A step's lines, as the run prints them."""
    print(event.message)


def _read(path):
    """read_fcs, looked up when called, so a test's stand-in for it is the reader the steps use."""
    return read_fcs(path)


def _gate_files(paths, settings, out, plt) -> list[dict]:
    """Each file gated by pipeline.gate_files, its populations printed as it goes, then the
    density grid, local maxima and the peak each Pop grows from drawn on one figure,
    local_maxima.png, the one the populations are chosen from. gates.png waits for the choice."""
    from datanuggets.flow.figures import local_maxima_figure, save_figures

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


@contextmanager
def _marked(out: Path):
    """The new folder out, marked incomplete until the steps in the with block finish. A step
    that fails, or a Ctrl+C, leaves the marker and says where the partial outputs are."""
    state.begin(out)
    try:
        yield
    except (Exception, SystemExit, KeyboardInterrupt):
        print(f"\nRun incomplete; partial outputs are in {out.resolve()}", file=sys.stderr)
        raise
    state.finish(out)


def _run(args) -> None:
    """The run in its mode: every check, then the output folder, the steps, and the figures
    shown. --gates-only stops once the gates are saved; --from-gates goes to _from_gates."""
    # --pops "", as a script that expands an unset variable gives it, is no --pops in every mode:
    # the run asks, and --gates-only doesn't count it as a clustering flag
    args.pops = args.pops or None
    _mode_rules(args)
    cluster = _settings(args)
    if args.from_gates is not None:
        _from_gates(args, cluster)
        return
    stop = args.gates_only
    plt = _pyplot(args.no_show, tables=not stop, tsne=cluster.tsne and not stop)
    paths, gate, out = _check_args(args)
    with _marked(out):
        samples = _gate_files(paths, gate, out, plt)
        if stop:
            _draw_gates(samples, plt, out)
            state.save_gates(out, samples, gate)
        else:
            _choose_channels(samples, args.channels, args.exclude_channels)
            chosen = _choose_pops(samples, args.pops, plt, not args.no_show)
            _draw_gates(samples, plt, out)
            pipeline.cluster_stage(
                samples, chosen, out, cluster, read=_read, on_event=_print_event, figure=plt.figure
            )
    if stop:
        where = out.resolve()
        print(f"\nGates are in {where}")
        print(f'Choose populations and cluster with: datanuggets-flow --from-gates "{where}"')
    else:
        print(f"\nOutputs are in {out.resolve()}")
    if not args.no_show:
        plt.show()


def _from_gates(args, cluster: ClusterSettings) -> None:
    """--from-gates: the gate folder and its files checked, --out refused if it is taken, and
    the channels resolved, all before the populations are asked over the folder's figures, so a
    run that can't go ahead stops before the question. The run folder is made only after it;
    the gate figures are copied in as they are, and run.json records the gate folder and the
    choice. The gate folder itself is only read."""
    from datanuggets.flow.figures import png_figure

    plt = _pyplot(args.no_show, tables=True, tsne=cluster.tsne)
    gates = state.load_gates(Path(args.from_gates))
    samples = gates.samples
    out = state.check_new(_out(args, FLOW_RUN_NAME))
    _choose_channels(samples, args.channels, args.exclude_channels)
    show = not args.no_show
    figures = [gates.path / "local_maxima.png", gates.path / "gates.png"]
    print(f"\nThe local maxima and each Pop's peak are drawn in {figures[0]}")
    print(f"The populations are drawn in {figures[1]}")
    if show:
        for path in figures:
            png_figure(path, FLOW_GATE_DPI, figure=plt.figure)
    chosen = _choose_pops(samples, args.pops, plt, show)
    with _marked(out):
        for path in figures:
            shutil.copyfile(path, out / path.name)
        run = pipeline.cluster_stage(
            samples, chosen, out, cluster, read=_read, on_event=_print_event, figure=plt.figure
        )
        state.write_run(out, gates.path, cluster, chosen, run.channels, run.labels)
    print(f"\nOutputs are in {out.resolve()}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
