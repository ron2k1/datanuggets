"""Record the flow fixtures: his pieces on files thinned to every k-th event for 50,000 events,
the inputs and each piece's output saved to <out>/<file>.npz:

uv run python benchmarks/flow/record.py <files...> --out tmp/flow/fixtures
"""

import argparse
from pathlib import Path

import cabrera_pieces as his
import cabrera_reference
import numpy as np
from fcs import read_fsc_ssc

N_EVENTS = 50_000


def find_area_fixed(x, idx, idy, t1=35, t2=0.85):
    """His find_area with the disc centred on the peak: R's column-major grid, each cell's own
    density, cells as (row, col)."""
    rows, cols = x.shape
    gr = np.column_stack(
        [np.tile(np.arange(1, rows + 1), cols), np.repeat(np.arange(1, cols + 1), rows)]
    )
    d = np.linalg.norm(gr - np.array([[idx, idy]]), axis=1)
    keep = (d < t1) & (x.flatten(order="F") > t2 * x[int(idx) - 1, int(idy) - 1])
    return gr[keep]


def record(path: Path, out: Path) -> None:
    fsc, ssc = read_fsc_ssc(path)
    k = max(1, len(fsc) // N_EVENTS)
    fsc, ssc = fsc[::k][:N_EVENTS], ssc[::k][:N_EVENTS]
    for v in (fsc, ssc):  # float32 in the file, so float32 in the fixture loses nothing
        assert np.array_equal(v.astype(np.float32).astype(float), v)
    f_density = his.grid(*his.transform(fsc, ssc))
    x = f_density["z"].T
    candidates = his.maxima(f_density)
    peaks = his.peaks(x, candidates)
    dat = his.areas(x, peaks)
    masks = his.gate(fsc, ssc, f_density, dat, len(peaks))
    assert np.array_equal(masks, his.finder(fsc, ssc))
    assert np.array_equal(masks, cabrera_reference.lymphmacro_finder(fsc, ssc))
    original = his.find_area
    his.find_area = find_area_fixed
    try:
        fixed = his.areas(x, peaks)
        fixed_masks = his.gate(fsc, ssc, f_density, fixed, len(peaks))
    finally:
        his.find_area = original
    np.savez_compressed(
        out / f"{path.stem}.npz",
        fsc=fsc.astype(np.float32),
        ssc=ssc.astype(np.float32),
        z=f_density["z"],
        gx=f_density["x"],
        gy=f_density["y"],
        zz=f_density["zz"].astype(np.uint8),
        candidates=candidates.astype(np.int16),
        peaks=peaks.astype(np.int16),
        his_area=dat.values.astype(np.int16),
        fixed_area=fixed.values.astype(np.int16),
        his_masks=np.packbits(masks, axis=0),
        fixed_masks=np.packbits(fixed_masks, axis=0),
    )
    print(
        f"{path.name}: every {k}th event, {len(fsc)} events, {len(candidates)} candidates, "
        f"{len(peaks)} populations {masks.sum(axis=0).tolist()}, "
        f"with the disc on the peak {fixed_masks.sum(axis=0).tolist()}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the flow fixtures from his pieces.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="tests/fixtures/flow")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        record(Path(f).expanduser(), out)


if __name__ == "__main__":
    main()
