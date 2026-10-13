"""The recorded flow cases (tests/fixtures/flow, from benchmarks/flow/record.py), two
independent statements of the area rule, and the gate check of spec criterion 3."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

FLOW = Path(__file__).parent / "fixtures" / "flow"
NAMES = tuple(sorted(p.stem for p in FLOW.glob("*.npz")))
EDGE_TOL = 1e-9  # of the data's range: how near the hull's edge an event gated differently may be
T = ((22, 0.25), (20, 0.5))  # his t11, t21 for the first peak, t12, t22 for the rest


@dataclass(frozen=True)
class FlowCase:
    name: str
    fsc: np.ndarray
    ssc: np.ndarray
    z: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    zz: np.ndarray
    candidates: np.ndarray
    peaks: np.ndarray
    his_area: np.ndarray
    fixed_area: np.ndarray
    his_masks: np.ndarray
    fixed_masks: np.ndarray

    @staticmethod
    def cells(area: np.ndarray, k: int) -> np.ndarray:
        """Population k's cells (k from 0) in an area table of (row, col, population)."""
        return area[area[:, 2] == k + 1, :2]


@cache
def load_case(name: str) -> FlowCase:
    with np.load(FLOW / f"{name}.npz") as f:
        n = len(f["fsc"])

        def unpack(key: str) -> np.ndarray:
            return np.unpackbits(f[key], axis=0, count=n).astype(bool)

        def ints(key: str) -> np.ndarray:
            return f[key].astype(np.int64)

        return FlowCase(
            name=name,
            fsc=f["fsc"].astype(float),
            ssc=f["ssc"].astype(float),
            z=f["z"],
            gx=f["gx"],
            gy=f["gy"],
            zz=ints("zz"),
            candidates=ints("candidates"),
            peaks=ints("peaks"),
            his_area=ints("his_area"),
            fixed_area=ints("fixed_area"),
            his_masks=unpack("his_masks"),
            fixed_masks=unpack("fixed_masks"),
        )


def _area(x, keep) -> np.ndarray:
    a, b = np.indices(x.shape) + 1
    return np.column_stack((a.T[keep.T], b.T[keep.T]))  # column by column, as his come


def mirrored_area(x, idx, idy, t1, t2) -> np.ndarray:
    """What his find_area keeps, stated directly: each cell's own density, the distance disc
    centred on the mirrored peak (idy, idx). Cells as 1-based (row, col)."""
    a, b = np.indices(x.shape) + 1
    return _area(x, (np.hypot(a - idy, b - idx) < t1) & (x > t2 * x[idx - 1, idy - 1]))


def fixed_area(x, idx, idy, t1, t2) -> np.ndarray:
    """The corrected rule stated directly: the same with the disc on the peak."""
    a, b = np.indices(x.shape) + 1
    return _area(x, (np.hypot(a - idx, b - idy) < t1) & (x > t2 * x[idx - 1, idy - 1]))


def edge_distance(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Each row of q's distance to the boundary of the convex hull of points."""
    v = points[ConvexHull(points).vertices]
    d = np.roll(v, -1, axis=0) - v
    rel = q[:, None, :] - v[None]
    t = np.clip((rel * d[None]).sum(-1) / (d * d).sum(-1), 0.0, 1.0)
    return np.sqrt(((rel - t[..., None] * d[None]) ** 2).sum(-1)).min(axis=1)


def assert_gate_close(fsc, ssc, got, want, zz, cells) -> None:
    """Spec criterion 3: the masks are equal, or every event that differs lies within EDGE_TOL of
    the data's range from the edge of the hull of the events whose bin is in the cells' hull."""
    from datanuggets.flow.gate import bin_mask

    if np.array_equal(got, want):
        return
    raw = np.column_stack((fsc, ssc))
    diff = np.flatnonzero(got != want)
    span = max(np.ptp(fsc), np.ptp(ssc))
    far = edge_distance(raw[bin_mask(zz, cells)], raw[diff]).max()
    assert far <= EDGE_TOL * span, (
        f"{len(diff)} events differ, the farthest {far:.3g} from the edge"
    )
