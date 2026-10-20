"""Run his whole script and check it against his pieces: for each sample his loop read, the masks
of run_cabrera.py equal cabrera_pieces.finder's on flowio's arrays, with the thresholds his loop
gave that sample. Runs where FlowKit is, in the ephemeral env:

uv run --no-project --with flowkit --with openpyxl --with matplotlib --with flowio \\
    python benchmarks/flow/check_reference.py <metadata.xlsx> <folder>
"""

import runpy
import sys
from pathlib import Path

import cabrera_pieces
import flowio
import numpy as np
from fcs import read_fsc_ssc, study_files


def main() -> int:
    here = Path(__file__).parent
    metadata, folder = sys.argv[1], sys.argv[2]
    sys.argv = [str(here / "run_cabrera.py"), metadata, folder]
    g = runpy.run_path(str(here / "run_cabrera.py"), run_name="__main__")
    by_fil = {flowio.FlowData(str(p)).text["fil"]: p for p in study_files(metadata, folder)}
    ok = 0
    for i, ind in enumerate(g["res"]):
        sample = g["dt2"][i]
        path = by_fil[sample.id]
        fsc, ssc = read_fsc_ssc(path)
        cols = [sample.get_channel_index("FSC-A"), sample.get_channel_index("SSC-A")]
        a = g["dt_exp"][i][:, cols]
        same_read = np.array_equal(a[:, 0], fsc) and np.array_equal(a[:, 1], ssc)
        same_masks = np.array_equal(
            cabrera_pieces.finder(fsc, ssc, t11=g["t0"][i], t21=g["t00"][i]), ind
        )
        ok += same_read and same_masks
        print(
            f"sample {i + 1} {sample.id} ({path.name}) t11={g['t0'][i]} t21={g['t00'][i]} "
            f"populations {ind.shape[1]}: flowio = FlowKit {same_read}, "
            f"his script = his pieces {same_masks}"
        )
    print(f"{ok} of {len(g['res'])} samples match")
    return 0 if ok == len(g["res"]) else 1


if __name__ == "__main__":
    sys.exit(main())
