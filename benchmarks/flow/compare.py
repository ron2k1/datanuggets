"""Cabrera's gate finder against datanuggets.flow, piece by piece, on real FCS files. Each piece
runs on the same arrays for his code and ours, timed as the median of --repeat runs, with whether
the outputs match. Writes a picture per file, his figure, ours and the events gated differently,
and appends a row per piece to summary.csv as each file ends. A file that raises is reported and
skipped.

python benchmarks/flow/compare.py <file.fcs | folder> [--repeat 3] [--out tmp/flow] [--no-png]
python benchmarks/flow/compare.py --metadata study.xlsx --dir <folder> --t0 22,20 --t00 0.25,0.5

--t0 and --t00 are his loop's t11 and t21 per file, in the order the files run (the workbook's
order with --metadata). Exit status 1 if the grid, maxima or peaks differ, or his gate on his cells,
or his finder against ours given his find_area, differs by more than events on the hull's edge,
or a file fails to run.
"""

import argparse
import csv
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

import cabrera_pieces as his
import matplotlib.pyplot as plt
import numpy as np
from cabrera_reference import find_area as his_find_area
from fcs import fcs_files, read_fsc_ssc, study_files
from scipy.spatial import ConvexHull, QhullError

from datanuggets import flow
from datanuggets.flow.gate import _finder, bin_mask
from datanuggets.plot import gates

T12, T22 = 20, 0.5  # his thresholds for every peak after the first
EDGE_TOL = 1e-9  # an event this close to the hull's edge, over the data's range, may go either way
CHUNK = 65_536  # events per edge_distance block
FIELDS = ["file", "events", "piece", "his_s", "ours_s", "speedup", "match"]
PIECES = [
    "1 grid",
    "2 maxima",
    "3 peaks",
    "4 area",
    "5 gate",
    "finder his",
    "finder fixed",
    "6 graph",
]


def timed(fn, repeat):
    """fn's result and its median seconds over repeat runs, at least one."""
    t0 = time.perf_counter()
    out = fn()
    secs = [time.perf_counter() - t0]
    for _ in range(repeat - 1):
        t0 = time.perf_counter()
        fn()
        secs.append(time.perf_counter() - t0)
    return out, statistics.median(secs)


def edge_distance(points, q):
    """Each row of q's distance to the boundary of the convex hull of points, in chunks of q so a
    big difference doesn't hold a q by hull array at once."""
    v = points[ConvexHull(points).vertices]
    d = np.roll(v, -1, axis=0) - v
    out = np.empty(len(q))
    for s in range(0, len(q), CHUNK):
        rel = q[s : s + CHUNK, None, :] - v[None]
        t = np.clip((rel * d[None]).sum(-1) / (d * d).sum(-1), 0.0, 1.0)
        out[s : s + CHUNK] = np.sqrt(((rel - t[..., None] * d[None]) ** 2).sum(-1)).min(axis=1)
    return out


def render(fig):
    """The figure drawn to Agg, as RGBA pixels, and the figure closed."""
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def our_graph(fsc, ssc, masks, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    gates(fsc, ssc, masks, title=title, ax=ax)
    fig.tight_layout()
    return fig


def differ(a, b):
    """The events whose populations differ, the narrower mask padded with empty populations."""
    k = max(a.shape[1], b.shape[1])
    pad = [np.pad(m, ((0, 0), (0, k - m.shape[1]))) for m in (a, b)]
    return (pad[0] != pad[1]).any(axis=1)


def gate_match(fsc, ssc, zz, cells, got, want):
    """Equal, or how many events differ and whether each sits on the edge of the hull."""
    if np.array_equal(got, want):
        return "equal", True
    raw = np.column_stack((fsc, ssc))
    span = max(np.ptp(fsc), np.ptp(ssc))
    far = 0.0
    for k, c in enumerate(cells):
        diff = np.flatnonzero(got[:, k] != want[:, k])
        if len(diff):
            try:
                far = max(far, edge_distance(raw[bin_mask(zz, c)], raw[diff]).max() / span)
            except (QhullError, ValueError):
                far = np.inf
    n = int((got != want).any(axis=1).sum())
    ok = far <= EDGE_TOL
    return f"{n} events differ, farthest {far:.1e} of the range from the hull edge", ok


def area_match(his_cells, ours):
    """Cells his find_area keeps and ours doesn't, and back, over the peaks."""
    if len(his_cells) != len(ours):  # the peaks differ, which row 3 has said
        return f"his {len(his_cells)} areas, ours {len(ours)}"
    his_only = ours_only = 0
    for h, o in zip(his_cells, ours, strict=True):
        hs, os_ = {tuple(r) for r in h.astype(int)}, {tuple(r) for r in o}
        his_only += len(hs - os_)
        ours_only += len(os_ - hs)
    if his_only == ours_only == 0:
        return "equal"
    return f"his keeps {his_only} cells ours doesn't, ours {ours_only} his doesn't (the disc)"


def compare(path, t11, t21, repeat, out, png):
    """One file's rows, printed, and whether pieces 1 to 3, 5 and finder his held."""
    fsc, ssc = read_fsc_ssc(path)
    print(f"\n{path.name}: {len(fsc):,} events, t11={t11:g} t21={t21:g}, median of {repeat}")
    rows = []

    def row(piece, his_s, ours_s, match):
        rows.append(
            dict(
                zip(
                    FIELDS,
                    [path.name, len(fsc), piece, his_s, ours_s, his_s / ours_s, match],
                    strict=True,
                )
            )
        )
        print(f"  {piece:<12} {his_s:10.4f} {ours_s:10.4f} {his_s / ours_s:8.1f}x  {match}")

    print(f"  {'piece':<12} {'his s':>10} {'ours s':>10} {'speedup':>9}  match")
    x0, y0 = his.transform(fsc, ssc)
    hd, hs = timed(lambda: his.grid(x0, y0), repeat)
    od, os_ = timed(lambda: flow.f_plot(x0, y0, nr=100, nc=100, scale="l"), repeat)
    grid_ok = all(np.array_equal(hd[k], getattr(od, k)) for k in ("z", "x", "y", "zz"))
    row("1 grid", hs, os_, "equal" if grid_ok else "DIFFER")
    hx, x = hd["z"].T, od.z.T

    hm, hs = timed(lambda: his.maxima(hd), repeat)
    om, os_ = timed(lambda: flow.local_maxima(x), repeat)
    max_ok = np.array_equal(hm, om)
    row("2 maxima", hs, os_, "equal" if max_ok else "DIFFER")

    hp, hs = timed(lambda: his.peaks(hx, hm), repeat)
    op, os_ = timed(lambda: flow.refine_peaks(x, om), repeat)
    peak_ok = np.array_equal(hp, op)
    row("3 peaks", hs, os_, f"equal, {len(op)} peaks" if peak_ok else "DIFFER")
    if len(hp) == 0:
        print("  no peak after his filters: his code raises here, ours gives zero populations")
        return rows, grid_ok and max_ok and peak_ok

    ts = [(t11, t21)] + [(T12, T22)] * (len(op) - 1)
    ha, hs = timed(lambda: his.areas(hx, hp, t11, t21, T12, T22), repeat)
    oa, os_ = timed(
        lambda: [flow.find_area(x, i, j, *ts[k]) for k, (i, j) in enumerate(op)], repeat
    )
    cells = [np.asarray(ha[ha["y"] == k + 1][["x", "z"]]) for k in range(len(hp))]
    row("4 area", hs, os_, area_match(cells, oa))

    hg, hs = timed(lambda: his.gate(fsc, ssc, hd, ha, len(hp)), repeat)
    og, os_ = timed(
        lambda: np.column_stack([flow.hull_gate(fsc, ssc, od.zz, c) for c in cells]), repeat
    )
    text, gate_ok = gate_match(fsc, ssc, od.zz, cells, og, hg)
    row("5 gate", hs, os_, f"on his cells: {text}")

    # his finder end to end against ours twice: given his find_area, which must give his masks,
    # and as shipped, with the disc centred on the peak, which moves events by design
    hf, hs = timed(lambda: his.finder(fsc, ssc, t11=t11, t21=t21), repeat)
    oh, os_ = timed(lambda: _finder(fsc, ssc, 11, 11, t11, t21, T12, T22, his_find_area), repeat)
    if oh.shape == hf.shape:
        text, his_ok = gate_match(fsc, ssc, od.zz, cells, oh, hf)
    else:
        text, his_ok = f"his {hf.shape[1]} populations, ours {oh.shape[1]}", False
    row("finder his", hs, os_, f"his find_area: {text}")
    of, os_ = timed(lambda: flow.lymphmacro_finder(fsc, ssc, t11=t11, t21=t21), repeat)
    moved = differ(hf, of)
    shape = f"his {hf.shape[1]} populations, ours {of.shape[1]}, "
    row("finder fixed", hs, os_, f"{shape}{int(moved.sum()):,} events gated differently (the disc)")

    a = np.column_stack((fsc, ssc))
    title = f"{path.stem}: FSC-A vs SSC-A"
    hi, hs = timed(lambda: render(his.graph(a, hf, title)), repeat)
    oi, os_ = timed(lambda: render(our_graph(fsc, ssc, hf, title)), repeat)
    pixels = np.abs(hi[..., :3].astype(int) - oi[..., :3]).mean() if hi.shape == oi.shape else -1
    row("6 graph", hs, os_, f"his masks both, mean pixel difference {pixels:.2f} of 255")

    if png:
        mine = render(our_graph(fsc, ssc, of, title))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, img, name in zip(
            axes, (hi, mine), ("His script", "datanuggets.flow, the disc on the peak"), strict=False
        ):
            ax.imshow(img)
            ax.set_title(name)
            ax.axis("off")
        style = {"ls": "none", "marker": "o", "rasterized": True}
        axes[2].plot(fsc, ssc, ms=0.5**0.5, color="gray", alpha=0.3, **style)
        axes[2].plot(fsc[moved], ssc[moved], ms=5**0.5, color="red", **style)
        axes[2].set_title(f"{int(moved.sum()):,} events gated differently end to end")
        axes[2].set_xlabel("FSC-A")
        axes[2].set_ylabel("SSC-A")
        fig.tight_layout()
        fig.savefig(out / f"{path.stem}_compare.png", dpi=100)
        plt.close(fig)
    return rows, grid_ok and max_ok and peak_ok and gate_ok and his_ok


def summarize(rows):
    """Each piece's median, min and max speedup over the files."""
    print(f"\nspeedup over {len({r['file'] for r in rows})} files: median, min, max")
    for piece in PIECES:
        s = [r["speedup"] for r in rows if r["piece"] == piece]
        if s:
            print(f"  {piece:<12} {statistics.median(s):8.1f}x {min(s):8.1f}x {max(s):8.1f}x")


def numbers(tokens):
    """--t0 22 20 and --t0 22,20 both give [22.0, 20.0], and so does "22 20", the one string
    PowerShell passes for an unquoted 22,20."""
    return [float(v) for t in tokens or [] for v in re.split(r"[,\s]+", t) if v]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cabrera's gate finder against datanuggets.flow")
    p.add_argument("target", nargs="?", help="an FCS file or a folder of them")
    p.add_argument("--metadata", help="his study workbook, the files in its second sheet")
    p.add_argument("--dir", help="the folder the workbook's files are in")
    p.add_argument("--t0", nargs="+", help="t11 per file, as 22 20 or 22,20, default 22")
    p.add_argument("--t00", nargs="+", help="t21 per file, default 0.25")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", default="tmp/flow")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args(argv)
    if (args.target is None) == (args.metadata is None or args.dir is None):
        p.error("give an FCS file or folder, or --metadata and --dir")
    if args.repeat < 1:
        p.error("--repeat must be at least 1")
    files = study_files(args.metadata, args.dir) if args.target is None else fcs_files(args.target)
    t0 = numbers(args.t0) or [22.0] * len(files)
    t00 = numbers(args.t00) or [0.25] * len(files)
    if len(t0) != len(files) or len(t00) != len(files):
        p.error(f"--t0 and --t00 need one value per file, {len(files)}")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    summary = out / "summary.csv"
    rows, ok, failed = [], True, []
    for f, t11, t21 in zip(files, t0, t00, strict=True):
        try:
            r, held = compare(f, t11, t21, args.repeat, out, not args.no_png)
        except Exception:  # noqa: BLE001  one file that can't run shouldn't cost the others their rows
            traceback.print_exc()
            failed.append(f.name)
            continue
        append(summary, r)  # as each file ends, so a stopped run keeps what it did
        rows += r
        ok &= held
    if len(files) > 1:
        summarize(rows)
    verdict = "pieces 1-3, 5 and finder his match" if ok else "A PIECE THAT MUST MATCH DIFFERS"
    if failed:
        verdict += f"; {len(failed)} FAILED TO RUN: {', '.join(failed)}"
    print(f"\n{verdict}; rows in {summary}")
    return 0 if ok and not failed else 1


def append(summary, rows):
    new = not summary.exists()
    with summary.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
