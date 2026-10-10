"""Runs a command on the given CPUs with Windows power throttling off, for its whole process tree.
Windows only.

    uv run --with psutil python benchmarks/pin.py <cmd> [args]
    PIN_MASK=0x555 (hex affinity mask, the default: one thread on each of six P-cores)

On a hybrid CPU, Windows can put a background console process on an efficiency core and throttle
it. On my laptop (i7-13650HX, P-cores on logical CPUs 0 to 11, E-cores on 12 to 19) one compiled
kernel took 15 ms on a P-core and 42 ms on an E-core, so the same benchmark run swung by 2x
between sittings. Children inherit the affinity but not the throttling state, so it is switched
off for each descendant as it appears. Check which CPUs are which on another machine first.
"""

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

import psutil

MASK = int(os.environ.get("PIN_MASK", "0x555"), 16)
CPUS = [i for i in range(MASK.bit_length()) if MASK >> i & 1]
PROCESS_SET_INFORMATION = 0x0200
PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.OpenProcess.restype = wintypes.HANDLE
k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.SetProcessInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
k32.CloseHandle.argtypes = [wintypes.HANDLE]


class PowerThrottlingState(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


# Version 1, control EXECUTION_SPEED, state 0: this process is never throttled.
NEVER_THROTTLE = PowerThrottlingState(1, 1, 0)
settled: set[int] = set()


def settle(pid: int) -> None:
    if pid in settled:
        return
    psutil.Process(pid).cpu_affinity(CPUS)
    handle = k32.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess {pid}")
    try:
        size = ctypes.sizeof(NEVER_THROTTLE)
        if not k32.SetProcessInformation(
            handle, PROCESS_POWER_THROTTLING, ctypes.byref(NEVER_THROTTLE), size
        ):
            raise OSError(ctypes.get_last_error(), f"SetProcessInformation {pid}")
    finally:
        k32.CloseHandle(handle)
    settled.add(pid)


settle(os.getpid())
cmd = sys.argv[1:]
if os.path.exists(cmd[0]):  # CreateProcess can't find a relative path written with "/"
    cmd[0] = os.path.abspath(cmd[0])
proc = subprocess.Popen(cmd)
root = psutil.Process(proc.pid)
while proc.poll() is None:
    try:
        for p in [root, *root.children(recursive=True)]:
            try:
                settle(p.pid)
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    time.sleep(0.05)
sys.exit(proc.returncode)
