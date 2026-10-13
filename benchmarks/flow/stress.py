"""Every file's events joined into one sample and the gate finder run once on it: ours in this
process, as shipped and given his find_area, and his in a child process stopped at --cap seconds,
since his Delaunay of every kept event grows with the sample. Appends a row to stress.csv. Exit
status 1 if his run fails, or if ours given his find_area differs from his beyond the hull's edge.

python benchmarks/flow/stress.py <folder> [--cap 1800] [--out tmp/flow]
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from cabrera_reference import find_area as his_find_area
from compare import T12, T22, differ, gate_match
from fcs import fcs_files, read_fsc_ssc

from datanuggets import flow
from datanuggets.flow.gate import MSIZE, _finder
from datanuggets.flow.grid import sqrt_coords

FIELDS = ["files", "events", "ours_s", "ours_his_area_s", "his_s", "his", "ours_vs_his"]
STDERR_LINES = 15  # of a failed his run's stderr, printed


def his_child(npy: Path) -> None:
    """His finder on the saved events: the masks next to them, the seconds on stdout."""
    import cabrera_pieces as his

    fsc, ssc = np.load(npy)
    t0 = time.perf_counter()
    masks = his.finder(fsc, ssc)
    secs = time.perf_counter() - t0
    np.save(npy.with_name("stress_his_masks.npy"), masks)
    print(secs)


def his_cells(fsc, ssc):
    """The events' bins and the cells his find_area keeps for each peak, which his gate and ours
    both take."""
    grid = flow.f_plot(*sqrt_coords(fsc, ssc), nr=MSIZE, nc=MSIZE, scale="l")
    x = grid.z.T
    peaks = flow.refine_peaks(x, flow.local_maxima(x))
    ts = [(22, 0.25)] + [(T12, T22)] * (len(peaks) - 1)
    return grid.zz, [his_find_area(x, i, j, *ts[k]) for k, (i, j) in enumerate(peaks)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="the gate finder on every file's events at once")
    p.add_argument("target", nargs="?", help="a folder of FCS files")
    p.add_argument("--cap", type=float, default=1800.0, help="seconds his run may take")
    p.add_argument("--out", default="tmp/flow")
    p.add_argument("--his", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.his is not None:
        his_child(args.his)
        return 0
    if args.target is None:
        p.error("give a folder of FCS files")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    files = fcs_files(args.target)
    fsc, ssc = (np.concatenate(c) for c in zip(*(read_fsc_ssc(f) for f in files), strict=True))
    print(f"{len(files)} files, {len(fsc):,} events")
    t0 = time.perf_counter()
    ours = flow.lymphmacro_finder(fsc, ssc)
    ours_s = time.perf_counter() - t0
    print(f"ours {ours_s:.2f} s, {ours.shape[1]} populations")
    t0 = time.perf_counter()
    ours_his = _finder(fsc, ssc, 11, 11, 22, 0.25, T12, T22, his_find_area)
    ours_his_s = time.perf_counter() - t0
    print(f"ours given his find_area {ours_his_s:.2f} s")

    npy = out / "stress_events.npy"
    masks_npy = npy.with_name("stress_his_masks.npy")
    np.save(npy, np.vstack((fsc, ssc)))
    his_s, versus, ok = None, "", True
    cmd = [sys.executable, __file__, "--his", str(npy)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.cap, check=True)
        his_s = float(r.stdout.split()[-1])
        his = np.load(masks_npy)
        status = f"{his_s:.2f} s, {his.shape[1]} populations"
        if his.shape == ours_his.shape:
            same, ok = gate_match(fsc, ssc, *his_cells(fsc, ssc), ours_his, his)
        else:
            same, ok = f"his {his.shape[1]} populations, ours {ours_his.shape[1]}", False
        moved = int(differ(his, ours).sum())
        versus = f"given his find_area: {same}; as shipped: {moved:,} events moved (the disc)"
    except subprocess.TimeoutExpired:  # a measurement, his run is too slow to finish
        status = f"stopped at {args.cap / 60:g} min"
    except subprocess.CalledProcessError as exc:
        status = f"failed, exit {exc.returncode}"
        print("\n".join(exc.stderr.splitlines()[-STDERR_LINES:]), file=sys.stderr)
        ok = False
    finally:
        npy.unlink(missing_ok=True)
        masks_npy.unlink(missing_ok=True)
    print(f"his {status}")
    if his_s is not None:
        print(f"ours against his {versus}")
        print(f"speedup {his_s / ours_s:.1f}x")

    csv_path = out / "stress.csv"
    new = not csv_path.exists()
    his_text = "" if his_s is None else f"{his_s:.3f}"
    values = [len(files), len(fsc), f"{ours_s:.3f}", f"{ours_his_s:.3f}", his_text, status, versus]
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(zip(FIELDS, values, strict=True)))
    print(f"row in {csv_path}")
    if not ok:
        print("HIS FINDER AND OURS GIVEN HIS find_area DIFFER, or his run failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
