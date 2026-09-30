"""Load the R-generated parity fixtures (manifest.json + raw little-endian .bin arrays)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np

from datanuggets._draws import DrawRecord
from datanuggets.result import DataNuggets

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
RTOL = 1e-8
ATOL = 1e-10


@dataclass(frozen=True)
class Case:
    path: Path
    manifest: dict

    @property
    def id(self) -> str:
        return self.manifest["case"]

    @property
    def args(self) -> dict:
        return self.manifest["args"]

    def has(self, name: str) -> bool:
        return name in self.manifest["arrays"]

    def array(self, name: str) -> np.ndarray:
        meta = self.manifest["arrays"][name]
        data = np.fromfile(self.path / meta["file"], dtype=meta["dtype"])
        return data.reshape(meta["shape"])

    def draws(self) -> list[DrawRecord]:
        sizes = self.array("draw_size")
        values = self.array("draw_values").astype(np.int64)
        records, pos = [], 0
        for n, size in zip(self.array("draw_n").tolist(), sizes.tolist(), strict=True):
            records.append(DrawRecord(n, size, values[pos : pos + size]))
            pos += size
        assert pos == values.size, f"{self.id}: draw_values has {values.size - pos} extra entries"
        return records


def load_group(group: str) -> list[Case]:
    root = FIXTURE_ROOT / group
    dirs = (
        sorted(p for p in root.iterdir() if (p / "manifest.json").is_file())
        if root.is_dir()
        else []
    )
    if not dirs:
        raise FileNotFoundError(f"no fixtures under {root}; regenerate them with the scripts in r/")
    return [Case(p, json.loads((p / "manifest.json").read_text(encoding="utf-8"))) for p in dirs]


# create.DN's argument names in the fixture manifests, and create_dn's
CREATE_ARGS = {
    "center.method": "center_method",
    "R": "r",
    "delete.percent": "delete_percent",
    "DN.num1": "dn_num1",
    "DN.num2": "dn_num2",
    "dist.metric": "dist_metric",
}


def create_args(case: Case) -> dict:
    """create_dn's keyword arguments for a create_dn fixture, seed aside."""
    return {py: case.args[r] for r, py in CREATE_ARGS.items()}


# refine.DN's argument names in the fixture manifests, and refine_dn's
REFINE_ARGS = {
    "EV.tol": "ev_tol",
    "max.splits": "max_splits",
    "min.nugget.size": "min_nugget_size",
    "shape.split": "shape_split",
    "min.shape.size": "min_shape_size",
    "delta": "delta",
    "nstart": "nstart",
}


def refine_inputs(case: Case) -> tuple[np.ndarray, DataNuggets, dict]:
    """x, the input nuggets (assignments made 0-based) and refine_dn's keyword arguments."""
    dn = DataNuggets(
        centers=case.array("in_centers"),
        weights=case.array("in_weights"),
        scales=case.array("in_scales"),
        assignments=case.array("in_assignments") - 1,
    )
    return case.array("x"), dn, {py: case.args[r] for r, py in REFINE_ARGS.items()}


def means_inputs(means: Case, name: str) -> tuple[np.ndarray, DataNuggets]:
    """x and the nugget set a get_dn_means fixture holds R's means for under `name`, read from
    the create_dn or refine_dn case it came from (assignments made 0-based)."""
    path = FIXTURE_ROOT / means.args["source"] / name
    case = Case(path, json.loads((path / "manifest.json").read_text(encoding="utf-8")))
    pre = means.args["prefix"]
    dn = DataNuggets(
        centers=case.array(pre + "centers"),
        weights=case.array(pre + "weights"),
        scales=case.array(pre + "scales"),
        assignments=case.array(pre + "assignments") - 1,
    )
    return case.array("x"), dn


def exact_ac_count(value, target, delete_percent) -> int:
    """One of AC's while loops run in exact rational arithmetic, returning R's counter. It's
    what the code says without double rounding. Where it disagrees with R's count, only a port
    doing R's double operations in R's order can match."""
    value, target, dp = Fraction(value), Fraction(target), Fraction(delete_percent)
    count = 1
    while value != target:
        if count > 100_000:
            raise AssertionError("the exact loop did not end")
        cut = math.floor(dp * value)
        if value - cut < target:
            value = target
        elif cut == 0:
            value -= 1
        else:
            value -= cut
        count += 1
    return count


def same_bits(a, b) -> bool:
    """Equal as float64 bit patterns: -0.0 isn't 0.0, and a NaN equals only the same NaN."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def hex_bits(a) -> list[str]:
    """Each float64 as its 16 hex digits, so a failed same_bits shows which NaN came back."""
    return [f"{b:016x}" for b in np.asarray(a, dtype=np.float64).reshape(-1).view(np.uint64)]


@cache
def fixture_data() -> dict[str, tuple[np.ndarray, list[np.ndarray]]]:
    """Each distinct finite x in the create_dn, refine_dn, rcompat_means and rcompat_stats
    fixtures, keyed "group/case" by the first case holding it, with the distinct 0-based nugget
    assignments recorded for it. The compiled long double tests compare against the Python
    replay, not R, so any real matrix serves."""
    first: dict[tuple, str] = {}
    out: dict[str, tuple[np.ndarray, dict[bytes, np.ndarray]]] = {}
    for group in ("create_dn", "refine_dn", "rcompat_means", "rcompat_stats"):
        for case in load_group(group):
            if not case.has("x"):
                continue
            x = np.ascontiguousarray(case.array("x"), dtype=np.float64)
            if x.ndim != 2 or not np.isfinite(x).all():
                continue
            key = first.setdefault((x.shape, x.tobytes()), f"{group}/{case.id}")
            sets = out.setdefault(key, (x, {}))[1]
            for name in ("assignments", "in_assignments"):
                if case.has(name):
                    a = case.array(name).astype(np.int64) - 1
                    sets.setdefault(a.tobytes(), a)
    return {key: (x, list(sets.values())) for key, (x, sets) in out.items()}
