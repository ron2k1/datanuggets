"""Times the WCluster ports on benchmarks/wcluster.R's workloads and compares every output with
R's bit for bit, times the workloads R is too slow to run, and checks the speed gates in
docs/specs/2026-10-09-wcluster-iterative-port-design.md section 10. Exits nonzero on a FAIL.

Run from the repo root after benchmarks/wcluster.R, in the same pinned sitting:
    uv run --with psutil python benchmarks/pin.py <path to uv> run python benchmarks/wcluster.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from datanuggets import DataNuggets, dn_whclust, whclust, wkmeans
from datanuggets._random.r_draws import RDraws
from datanuggets._random.replay import ReplayDraws
from rfixtures import Case, same_values

OUT = ROOT / "tmp" / "wcluster" / "out"
COVERTYPE = ROOT / "tmp" / "covertype"
WEIGHTS = [1.0, 3.0, 7.0, 2.0]  # repeating, as wcluster.R weights quakes
QUAKES_WKMEANS = (100, 250, 500, 1000)


def load(path: Path) -> Case | None:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    return Case(path, json.loads(manifest.read_text(encoding="utf-8")))


def timed(fn):
    """fn's value and its seconds: the best of 3 when a run takes under a second, else one.
    wcluster.R takes R's wkmeans time as the best of 3 too, and times its whclust once, since a
    run takes seconds to minutes."""
    t0 = time.perf_counter()
    value = fn()
    best = time.perf_counter() - t0
    if best < 1:
        for _ in range(2):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
    return value, best


def same_whclust(got, case: Case) -> bool:
    n = case.array("x").shape[0]
    merge = case.array("merge").astype(np.int64)
    merge = np.where(merge < 0, -merge - 1, n + merge - 1)
    return (
        np.array_equal(got.merge, merge)
        and same_values(got.height, case.array("height"))
        and np.array_equal(got.order, case.array("order").astype(np.int64) - 1)
    )


def same_wkmeans(got, case: Case) -> bool:
    return (
        np.array_equal(got.labels, case.array("labels").astype(np.int64) - 1)
        and same_values(got.centers, case.array("centers"))
        and same_values(got.wwcss, case.array("wwcss"))
        and same_values(got.total_wwcss, case.array("total").reshape(()))
    )


def line(name: str, r_s, py_s: float, bits: str) -> None:
    r = "not run" if r_s is None else f"{r_s:.2f}"
    speedup = "" if r_s is None else f"{r_s / py_s:,.0f}x"
    print(f"{name:26s} {r:>9s} {py_s:10.4f} {speedup:>9s}  {bits}")


def against_r() -> dict:
    """Each R workload's (R s, Python s, equal to R), or None where R's output is missing."""
    results = {}
    for n in (200, 400, 800):
        case = load(OUT / f"whclust{n}")
        if case is None:
            results[f"whclust{n}"] = None
            continue
        x, w = case.array("x"), case.array("w")
        got, secs = timed(lambda x=x, w=w: whclust(x, w))
        results[f"whclust{n}"] = (case.manifest["secs"], secs, same_whclust(got, case))
    for n in QUAKES_WKMEANS:
        case = load(OUT / f"wkmeans{n}")
        if case is None:
            results[f"wkmeans{n}"] = None
            continue
        x, w, records = case.array("x"), case.array("w"), case.draws()

        def run(x=x, w=w, records=records):
            draws = ReplayDraws(records)
            got = wkmeans(x, 4, w=w, seed=draws)
            draws.assert_exhausted()
            return got

        got, secs = timed(run)
        results[f"wkmeans{n}"] = (case.manifest["secs"], secs, same_wkmeans(got, case))
    return results


def covertype_rows(n: int) -> np.ndarray:
    """The first n rows of Covertype's 10 continuous columns, as covertype.R reads them."""
    return np.loadtxt(
        COVERTYPE / "covtype.data.gz", delimiter=",", usecols=range(10), max_rows=n
    ).astype(np.float64)


def main() -> int:
    x = covertype_rows(10_000)
    w = np.resize(WEIGHTS, x.shape[0])
    t0 = time.perf_counter()
    whclust(x[:10], w[:10])
    wkmeans(x[:10], 4, w=w[:10], seed=1)
    print(f"first calls on 10 rows (compile or cache load): {time.perf_counter() - t0:.2f} s\n")

    print(f"{'workload':26s} {'R s':>9s} {'Python s':>10s} {'speedup':>9s}  bits")
    rs = against_r()
    for name, res in rs.items():
        if res is None:
            print(f"{name:26s} R output missing: run benchmarks/wcluster.R")
        else:
            line(name, res[0], res[1], "yes" if res[2] else "NO")
    _, wh2000 = timed(lambda: whclust(x[:2000], w[:2000]))
    line("whclust covertype 2000", None, wh2000, "")
    _, wk10000 = timed(lambda: wkmeans(x, 4, w=w, seed=RDraws(291102)))
    line("wkmeans covertype 10000", None, wk10000, "")
    case = load(COVERTYPE / "out" / "covertype_100000_create")
    if case is None:
        print("dn_whclust                 nuggets missing: run benchmarks/covertype.R 100000")
        dn, dn2000 = None, float("inf")
    else:
        dn = DataNuggets(
            case.array("centers"),
            case.array("weights"),
            case.array("scales"),
            case.array("assignments").astype(np.int64) - 1,
        )
        _, dn2000 = timed(lambda: dn_whclust(dn))
        line(f"dn_whclust {dn.n_nuggets} nuggets", None, dn2000, "")

    def ran(*names):
        return all(rs[k] is not None and rs[k][2] for k in names)

    wk = [f"wkmeans{n}" for n in QUAKES_WKMEANS]
    gates = [
        (
            "whclust quakes 200 and 400 under 1 s",
            ran("whclust200", "whclust400") and max(rs["whclust200"][1], rs["whclust400"][1]) < 1,
        ),
        (
            "whclust quakes 800 under 1 s, equal to R",
            ran("whclust800") and rs["whclust800"][1] < 1,
        ),
        ("whclust 2,000 Covertype rows under 10 s", wh2000 < 10),
        (
            "wkmeans quakes 50x faster at each n",
            ran(*wk) and all(rs[k][0] / rs[k][1] >= 50 for k in wk),
        ),
        ("wkmeans 10,000 Covertype rows under 30 s", wk10000 < 30),
        (
            "dn_whclust 2,000 nuggets under 10 s",
            dn is not None and dn.n_nuggets == 2000 and dn2000 < 10,
        ),
    ]
    print()
    for row, ok in gates:
        print(f"{'PASS' if ok else 'FAIL'}  {row}")
    return 0 if all(ok for _, ok in gates) else 1


if __name__ == "__main__":
    sys.exit(main())
