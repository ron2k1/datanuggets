"""Checks spec section 9.5's stopping rule on Covertype: no function written in Python in
src/datanuggets may have a cProfile tottime above 10% of the run's wall time, for create_dn and
for refine_dn (shape splits off), both with n_jobs=1 on R's replayed draws.

cProfile can't see into numba, so a kernel's time would land in the Python function that called
it. Callers reach kernels through their module (ldjit, rfastjit, centersjit, kmnsjit). For the
run, each of those references is swapped for a copy whose kernels are wrapped in Python functions
named compiled_<kernel>, which book the time under that name instead. Run from the repo root
after benchmarks/covertype.R with the same size:

    uv run python benchmarks/profile_stop.py [10000 | 100000]

Exits 1 if the rule fails, no kernel time was booked, or an output stops matching R's.
"""

import cProfile
import os
import pstats
import sys
import types
from pathlib import Path

from numba.core.dispatcher import Dispatcher

import datanuggets
from covertype import compare, load, run, stage_lines, warm_up
from datanuggets import DataNuggets, create_dn, refine_dn

SRC = os.path.normcase(str(Path(datanuggets.__file__).resolve().parent))
LIMIT = 0.10
TOP = 8


def booked(name, kernel):
    """A Python function that calls kernel, named compiled_<name> so cProfile books its time."""

    def call(*args):
        return kernel(*args)

    label = f"compiled_{name}"
    call.__code__ = call.__code__.replace(co_name=label, co_qualname=label)
    return call


class Booked:
    """A kernel module's attributes, with every kernel swapped for its booked wrapper."""

    def __init__(self, module):
        for name, value in vars(module).items():
            if not name.startswith("__"):
                setattr(self, name, booked(name, value) if isinstance(value, Dispatcher) else value)


def kernel_refs() -> list[tuple[types.ModuleType, str, types.ModuleType]]:
    """(module, name, kernel module) for every reference a datanuggets module holds to a module
    of kernels, a datanuggets module whose name ends in jit."""
    refs = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("datanuggets."):
            for name, value in list(vars(mod).items()):
                if (
                    isinstance(value, types.ModuleType)
                    and value.__name__.startswith("datanuggets.")
                    and value.__name__.endswith("jit")
                ):
                    refs.append((mod, name, value))
    return refs


def profiled(fn):
    prof = cProfile.Profile()
    prof.enable()
    try:
        result = fn()
    finally:
        prof.disable()
    return result, pstats.Stats(prof)


def report(label: str, stats: pstats.Stats, secs: float, checks: dict) -> bool:
    ours, compiled = [], []
    for (path, line, name), (_, _, tottime, _, _) in stats.stats.items():
        if name.startswith("compiled_"):
            compiled.append((tottime, name))
        elif os.path.normcase(os.path.abspath(path)).startswith(SRC):
            ours.append((tottime, f"{Path(path).name}:{line} {name}"))
    ours.sort(reverse=True)
    compiled.sort(reverse=True)
    parity = all(v is True for v in checks.values())
    worst = ours[0][0] / secs if ours else 0.0
    print(f"{label}: {secs:.2f} s wall, outputs {'match R' if parity else f'DIFFER {checks}'}")
    print(f"  Python in src/datanuggets by own time (limit {LIMIT:.0%} of wall):")
    for tottime, where in ours[:TOP]:
        print(f"    {tottime / secs:6.1%} {tottime:8.2f} s  {where}")
    print("  compiled kernels:")
    for tottime, name in compiled:
        print(f"    {tottime / secs:6.1%} {tottime:8.2f} s  {name}")
    if not compiled:
        print("  no kernel time was booked: a caller holds a kernel by name, not its module")
    passed = worst <= LIMIT
    print(f"  stopping rule: {'PASS' if passed else 'FAIL'} (largest {worst:.1%})")
    return parity and passed and bool(compiled)


def main(size: str) -> int:
    cid = f"covertype_{size}"
    cr = load(cid + "_create")
    x = cr.array("x")
    warm_up(x)
    refs = kernel_refs()
    for mod, name, jit in refs:
        setattr(mod, name, Booked(jit))
    try:
        (dn, secs, extra), stats = profiled(
            lambda: run(lambda d: create_dn(x, seed=d), cr, stage_lines(cr.manifest["messages"]))
        )
        good = report(f"create_dn {cid}", stats, secs, {**compare(dn, cr), **extra})
        r_dn = DataNuggets(
            centers=cr.array("centers"),
            weights=cr.array("weights"),
            scales=cr.array("scales"),
            assignments=cr.array("assignments") - 1,
        )
        rc = load(cid + "_refine")
        (got, secs, extra), stats = profiled(lambda: run(lambda d: refine_dn(x, r_dn, seed=d), rc))
        good &= report(f"refine_dn {cid}", stats, secs, {**compare(got, rc), **extra})
    finally:
        for mod, name, jit in refs:
            setattr(mod, name, jit)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "100000"))
