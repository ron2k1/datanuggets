"""Load the R-generated parity fixtures (manifest.json + raw little-endian .bin arrays)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np

from datanuggets._random.replay import DrawRecord
from datanuggets.result import DataNuggets

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
RTOL = 1e-8
ATOL = 1e-10
EIG_FLOOR, EIG_C, EIG_K, EIG_TIE = 1e-10, 256.0, 1024.0, 1e-12


def eigen_mismatches(s, values_r, vectors_r, scores_r, x0, sdev, vectors, scores) -> list[str]:
    """Where a port's eigen output differs from R's by more than two correct LAPACK builds can.
    Eigenvalues within EIG_TIE of each other form a cluster, and only a cluster's subspace is
    determined, so vectors are compared through their projectors. R1 orthonormal vectors. R2
    each vector's residual against R's eigenvalue. R3 sdev squared against R's eigenvalue, and
    no sdev below zero (a NaN sdev needs a value at most the band above zero). R4 each cluster's
    projector against R's, within max(EIG_FLOOR, EIG_C eps / gap). R5 the same for the scores,
    row by row. Every check passes only a value shown to be within its bound, so a NaN or an
    infinity in the port's output is a mismatch."""
    eps = float(np.finfo(float).eps)
    p = values_r.size
    for name, a in (("sdev", sdev), ("rotation", vectors), ("scores", scores)):
        if a.shape[-1] != p:
            raise ValueError(f"{name} has {a.shape[-1]} columns, R has {p}")
    norm = max(float(np.max(np.abs(values_r))), 1e-300)
    band = EIG_K * eps * norm
    lower = np.tril(s) + np.tril(s, -1).T
    out = []
    with np.errstate(all="ignore"):
        orth = float(np.max(np.abs(vectors.T @ vectors - np.eye(p))))
        if not orth <= EIG_FLOOR:
            out.append(f"R1 rotation is not orthonormal: {orth:.3e}")
        res = np.linalg.norm(lower @ vectors - vectors * values_r, axis=0)
        bad = np.flatnonzero(~(res <= band))
        out += [f"R2 column {i}: residual {res[i]:.3e} > {band:.3e}" for i in bad]
        for i in range(p):
            if np.isnan(sdev[i]):
                if not values_r[i] <= band:
                    out.append(f"R3 sdev {i} is NaN but R's eigenvalue is {values_r[i]:.3e}")
            elif not (sdev[i] >= 0 and abs(sdev[i] ** 2 - values_r[i]) <= band):
                out.append(f"R3 sdev {i} is {sdev[i]:.17g}, R's eigenvalue {values_r[i]:.17g}")
        cid = np.concatenate(([0], np.cumsum(np.abs(np.diff(values_r)) > EIG_TIE * norm)))
        rows = np.linalg.norm(x0, axis=1)
        for c in range(int(cid[-1]) + 1):
            idx, rest = np.flatnonzero(cid == c), np.flatnonzero(cid != c)
            gap = np.inf
            if rest.size:
                gap = float(np.min(np.abs(values_r[idx][:, None] - values_r[rest][None, :])))
                gap /= norm
            tol = max(EIG_FLOOR, EIG_C * eps / gap)
            v, vr = vectors[:, idx], vectors_r[:, idx]
            dp = v @ v.T - vr @ vr.T
            # the spectral norm raises on NaN rather than returning it
            sin = float(np.linalg.norm(dp, 2)) if np.isfinite(dp).all() else math.inf
            if not sin <= tol:
                out.append(f"R4 columns {idx[0]}..{idx[-1]}: subspace off by {sin:.3e} > {tol:.3e}")
            d = np.linalg.norm(scores[:, idx] @ v.T - scores_r[:, idx] @ vr.T, axis=1)
            off = ~(d <= (tol + 8 * p * eps) * rows)
            if off.any():
                out.append(f"R5 columns {idx[0]}..{idx[-1]}: scores off in {int(off.sum())} rows")
    return out


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
        flags = (
            self.array("draw_replace").astype(bool).tolist()
            if self.has("draw_replace")
            else [False] * sizes.size
        )
        records, pos = [], 0
        for n, size, rep in zip(self.array("draw_n").tolist(), sizes.tolist(), flags, strict=True):
            records.append(DrawRecord(n, size, values[pos : pos + size], rep))
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
        raise FileNotFoundError(
            f"no fixtures under {root}; restore tests/fixtures from the source checkout or archive"
        )
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


def from_hex(bits) -> np.ndarray:
    """The float64 values whose bit patterns hex_bits prints, for values recorded from R by hand."""
    return np.array([int(b, 16) for b in bits], dtype=np.uint64).view(np.float64)


def _same_entries(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float64).reshape(-1)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    return (nan_a & nan_b) | (~nan_a & ~nan_b & (a.view(np.uint64) == b.view(np.uint64)))


def same_values(a, b) -> bool:
    """Equal as float64 bit patterns, except that any NaN matches any NaN. -0.0 isn't 0.0, which
    == and assert_array_equal miss. NaN bits aren't compared because R's NA and the x86 default
    NaN differ, and where a WCluster port keeps R's NaN bits the tests use same_bits instead."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(_same_entries(a, b).all())


def assert_same_values(got, want) -> None:
    """same_values as an assertion that shows the first entry that differs, in hex."""
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, f"shape {got.shape}, want {want.shape}"
    same = _same_entries(got, want)
    if not same.all():
        bad = np.flatnonzero(~same)
        i = int(bad[0])
        g, w = got.reshape(-1)[i], want.reshape(-1)[i]
        raise AssertionError(
            f"{bad.size} of {same.size} entries differ, first at "
            f"{np.unravel_index(i, got.shape)}: {hex_bits(g)[0]} ({g!r}), want "
            f"{hex_bits(w)[0]} ({w!r})"
        )


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
