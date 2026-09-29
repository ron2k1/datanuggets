"""Runs a command and reports its wall time and peak working set, across its child processes
(Rscript.exe hands the work to a child Rterm). Windows only, since it reads peak_wset.

    uv run --with psutil python benchmarks/measure.py <cmd> [args]
"""

import subprocess
import sys
import time

import psutil

t0 = time.perf_counter()
proc = subprocess.Popen(sys.argv[1:])
root = psutil.Process(proc.pid)
peaks: dict[int, int] = {}
while proc.poll() is None:
    try:
        for p in [root, *root.children(recursive=True)]:
            try:
                peaks[p.pid] = max(peaks.get(p.pid, 0), p.memory_info().peak_wset)
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    time.sleep(0.2)
secs = time.perf_counter() - t0
peak = max(peaks.values(), default=0)
print(
    f"MEASURE exit={proc.returncode} wall={secs:.1f}s peak_wset={peak / 2**20:.0f} MiB", flush=True
)
sys.exit(proc.returncode)
