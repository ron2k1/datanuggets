"""Record tests/fixtures/flow/202_fluor.npz: three fluorescence channels of 202.fcs from Zenodo
record 22079206, for the same events as 202.npz (every 5th event, the first 50,000), read the way
datanuggets.flow.pick.read_fcs reads a file:

uv run python benchmarks/flow/record_fluor.py path/to/202.fcs --out tests/fixtures/flow
"""

import argparse
from pathlib import Path

import flowio
import numpy as np

N_EVENTS = 50_000
CHANNELS = ("FJComp-Alexa Fluor 700-A", "FJComp-BV510-A", "FJComp-Spark Blue-A")  # CD8, CD3, CD4


def record(path: Path, out: Path) -> None:
    fd = flowio.FlowData(str(path))
    c = fd.channel_count
    pnn = [fd.text[f"p{k}n"] for k in range(1, c + 1)]
    pns = [fd.text.get(f"p{k}s", "").strip() for k in range(1, c + 1)]
    events = np.asarray(fd.as_array(preprocess=True), dtype=float)
    k = max(1, len(events) // N_EVENTS)
    ev = events[::k][:N_EVENTS]
    with np.load(out / "202.npz") as g:  # the same events as the recorded gate fixture
        assert np.array_equal(ev[:, pnn.index("FSC-A")], g["fsc"].astype(float))
        assert np.array_equal(ev[:, pnn.index("SSC-A")], g["ssc"].astype(float))
    idx = [pnn.index(ch) for ch in CHANNELS]
    assert idx == sorted(idx)  # in the file's channel order
    fluor = ev[:, idx]
    assert np.array_equal(fluor.astype(np.float32).astype(float), fluor)  # float32 loses nothing
    np.savez_compressed(
        out / "202_fluor.npz",
        fluor=fluor.astype(np.float32),
        pnn=np.array(CHANNELS),
        pns=np.array([pns[i] for i in idx]),
    )
    print(f"{out / '202_fluor.npz'}: {fluor.shape}, {[pns[i] for i in idx]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record 202_fluor.npz from 202.fcs.")
    ap.add_argument("fcs", type=Path, help="202.fcs")
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures/flow"))
    args = ap.parse_args()
    record(args.fcs, args.out)


if __name__ == "__main__":
    main()
