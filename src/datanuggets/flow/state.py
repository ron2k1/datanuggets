"""The folders a datanuggets-flow run writes, and the gate folder its two stages share. A run's
folder holds INCOMPLETE.txt until the run finishes. A gate folder holds local_maxima.png, gates.png,
gates.npz, each file's masks and finder grid, and gates.json, written last: the settings, the
constants the finder keeps for parity, the package versions, and each file's path, size, SHA-256,
channels and populations. The masks are the gates: the clustering stage reads a file only once its
size and SHA-256 match, and never gates it again. run.json records what a run from a gate folder
clustered. Nothing here needs the flow extra; gates.json records file paths and hashes, so treat a
gate folder like the FCS files themselves."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

import numpy as np

from datanuggets.exceptions import FlowError
from datanuggets.flow import gate, grid, peaks
from datanuggets.flow.pipeline import PopStats, Sample, pops_spec, population_table
from datanuggets.flow.settings import ClusterSettings, GateSettings

GATES_SCHEMA = "datanuggets-flow-gates/1"
RUN_SCHEMA = "datanuggets-flow-run/1"
MARKER = "INCOMPLETE.txt"
MARKER_TEXT = (
    "This run has not completed. Its tables and figures may be partial.\n"
    "A successful run removes this marker; rerun in a new output folder.\n"
)
GATE_FILES = ("local_maxima.png", "gates.png", "gates.npz", "gates.json")
NPZ_KEYS = ("masks", "density", "candidates", "peaks", "grid_fsc", "grid_ssc")  # _N for file N
# what each file's entry in gates.json holds
ENTRY_KEYS = (
    "number",
    "name",
    "path",
    "size",
    "sha256",
    "n_events",
    "pnn",
    "pns",
    "t11",
    "t21",
    "local_maxima",
    "populations",
)
# the constants the finder keeps for parity with cfc.py, which no flag changes
FIXED = {
    "MSIZE": gate.MSIZE,
    "GRID_SCALE": gate.GRID_SCALE,
    "GRID_ALPHA": gate.GRID_ALPHA,
    "SQRT_SHIFT": grid.SQRT_SHIFT,
    "LOCAL_MAX_QUANTILE": peaks.LOCAL_MAX_QUANTILE,
    "LOCAL_MAX_EDGES": peaks.LOCAL_MAX_EDGES,
    "PEAK_QUANTILE": peaks.PEAK_QUANTILE,
    "ROW_MIN": peaks.ROW_MIN,
    "ROW_MAX": peaks.ROW_MAX,
    "COL_MAX": peaks.COL_MAX,
    "SEARCH_THRESHOLD": peaks.SEARCH_THRESHOLD,
    "MIN_POINTS": gate.MIN_POINTS,
    "FIRST_PEAK_RULE": "t11/t21 for the first peak by SSC row, t12/t22 for the rest",
}
READER = {"library": "flowio", "preprocess": True, "compensation": "not applied"}
# the packages whose versions a record keeps, beside Python's
PACKAGES = (
    "datanuggets",
    "numpy",
    "scipy",
    "numba",
    "joblib",
    "flowio",
    "matplotlib",
    "pandas",
    "scikit-learn",
)


class GateFolder(NamedTuple):
    """A gate folder load_gates accepted: its path, the settings it was gated with, its files as
    samples, the masks and populations as gated, and the package versions gates.json records,
    which changed_versions compares with the ones installed now."""

    path: Path
    settings: GateSettings
    samples: list[Sample]
    versions: Mapping[str, str | None]


def check_new(out: Path) -> Path:
    """out, when nothing is there yet, not even a broken link, so a run never writes into an
    earlier one's folder."""
    if os.path.lexists(out):
        raise FlowError(f"output path already exists: {out}; choose a new {{param}} folder", "out")
    return out


def begin(out: Path) -> Path:
    """The new folder out, with the marker a finished run removes."""
    check_new(out).mkdir(parents=True)
    (out / MARKER).write_text(MARKER_TEXT, encoding="utf-8")
    return out


def finish(out: Path) -> None:
    """The run in out is complete: its marker goes."""
    (out / MARKER).unlink()


def file_sha256(path) -> str:
    """The SHA-256 of the file at path, as hex."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _versions() -> dict[str, str | None]:
    """The version of each package a run's results depend on, None for one not installed."""
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found | {"python": platform.python_version()}


def changed_versions(
    recorded: Mapping[str, str | None],
) -> dict[str, tuple[str | None, str | None]]:
    """Each package whose version now differs from the one a record's versions hold, as
    (recorded, now), in the record's order. Only the packages both name are compared, so a record
    from a version that kept a different list isn't reported for that alone."""
    now = _versions()
    return {
        name: (was, now[name]) for name, was in recorded.items() if name in now and was != now[name]
    }


def _record(schema: str) -> dict[str, object]:
    """What gates.json and run.json both begin with."""
    created = datetime.now(UTC).isoformat(timespec="seconds")
    return {"schema": schema, "created": created, "versions": _versions()}


def _write_json(path: Path, record: Mapping[str, object]) -> Path:
    """record as UTF-8 JSON, labels as typed, and refusing NaN, which JSON doesn't have."""
    text = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def save_gates(folder: Path, samples: Sequence[Mapping], settings: GateSettings) -> Path:
    """gates.npz and then gates.json in folder, beside local_maxima.png and gates.png: the record
    load_gates reads. A folder without gates.json is refused, so one whose saving stopped
    partway isn't read as whole. The path of gates.json."""
    arrays, files = {}, []
    for s in samples:
        n, path = s["number"], Path(s["path"]).absolute()
        found = (np.packbits(s["masks"], axis=0), *s["peaks"])  # FinderPeaks' fields in order
        arrays |= {f"{key}_{n}": a for key, a in zip(NPZ_KEYS, found, strict=True)}
        files.append(
            {
                "number": n,
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
                "n_events": s["n_events"],
                "pnn": list(s["pnn"]),
                "pns": list(s["pns"]),
                "t11": s["t11"],
                "t21": s["t21"],
                "local_maxima": s["local_maxima"],
                "populations": [p._asdict() for p in s["stats"]],
            }
        )
    np.savez(folder / "gates.npz", **arrays)
    record = _record(GATES_SCHEMA) | {
        "settings": settings.to_dict(),
        "fixed": FIXED,
        "reader": READER,
        "files": files,
    }
    return _write_json(folder / "gates.json", record)


def _refused(message: str) -> FlowError:
    return FlowError(message, "from_gates")


def _gap(record: Mapping[str, object]) -> str | None:
    """What a record of the gates schema lacks that load_gates reads, or None when it lacks
    nothing. A record save_gates wrote is whole, so this finds a hand edit that dropped a key."""
    settings, files = record.get("settings"), record.get("files")
    if not isinstance(settings, dict):
        return "it has no settings" if settings is None else "its settings aren't a JSON object"
    if not isinstance(files, list) or not files:
        return "it has no files" if not files else "its files aren't a JSON list"
    for i, entry in enumerate(files, 1):
        if not isinstance(entry, dict):
            return f"file {i}'s entry isn't a JSON object"
        missing = [key for key in ENTRY_KEYS if key not in entry]
        if missing:
            return f"file {i}'s entry has no {missing[0]}"
    return None


def load_gates(folder: Path) -> GateFolder:
    """The gate folder save_gates wrote, its files checked, cheapest first: it is a folder, its
    gate stage finished, it has every file, gates.json is a complete record of this schema whose
    settings this version has, gates.npz has each file's arrays, each file's masks fit its events
    and populations, and every FCS file is where it was, with its size and SHA-256 as gated. A
    FlowError (from_gates) names the folder or file it refuses."""
    folder = Path(folder)
    if not folder.is_dir():
        raise _refused(f"{folder} isn't a folder")
    if (folder / MARKER).exists():
        raise _refused(
            f"{folder}'s gate stage did not finish; run --gates-only again into a new folder"
        )
    for name in GATE_FILES:
        if not (folder / name).is_file():
            raise _refused(f"{folder} isn't a gate folder: it has no {name}")
    path = folder / "gates.json"
    try:
        record = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise _refused(f"{path} isn't UTF-8 JSON: {e}") from None
    if not isinstance(record, dict) or record.get("schema") != GATES_SCHEMA:
        raise _refused(f"{path} isn't a {GATES_SCHEMA} record")
    gap = _gap(record)
    if gap:
        raise _refused(f"{path} isn't a complete {GATES_SCHEMA} record: {gap}")
    try:
        settings = GateSettings.from_dict(record["settings"])
    except FlowError as e:
        raise _refused(f"{path}: {e}") from None
    samples = _samples(folder / "gates.npz", record["files"], settings)
    for entry in record["files"]:
        check_identity(entry)
    # never refused: the masks are the gates, whatever packages found them
    versions = record.get("versions")
    return GateFolder(folder, settings, samples, versions if isinstance(versions, dict) else {})


def _samples(npz: Path, files: Sequence[Mapping], settings: GateSettings) -> list[Sample]:
    """Each recorded file as a sample, its masks and finder grid from npz. fsc and ssc are None:
    gates.png, the one figure that needs them, is drawn already."""
    samples = []
    with np.load(npz, allow_pickle=False) as arrays:
        for entry in files:
            n, n_events = entry["number"], entry["n_events"]
            for key in (f"{key}_{n}" for key in NPZ_KEYS):
                if key not in arrays.files:
                    raise _refused(f"{npz} has no {key}")
            packed, want = arrays[f"masks_{n}"], (-(-n_events // 8), len(entry["populations"]))
            if packed.dtype != np.uint8 or packed.shape != want:
                raise _refused(
                    f"{npz}'s masks_{n} doesn't fit file {n}: its shape is {packed.shape}, where "
                    f"{n_events:,} events and {want[1]} populations take {want}"
                )
            stats = [PopStats(**(p | {"peak": tuple(p["peak"])})) for p in entry["populations"]]
            samples.append(
                Sample(
                    number=n,
                    path=Path(entry["path"]),
                    n_events=n_events,
                    pnn=entry["pnn"],
                    pns=entry["pns"],
                    masks=np.unpackbits(packed, axis=0, count=n_events).astype(bool),
                    peaks=gate.FinderPeaks(*(arrays[f"{key}_{n}"] for key in NPZ_KEYS[1:])),
                    stats=stats,
                    table=population_table(stats, settings.fsc, settings.ssc),
                    local_maxima=entry["local_maxima"],
                    t11=entry["t11"],
                    t21=entry["t21"],
                    fsc=None,
                    ssc=None,
                )
            )
    return samples


def check_identity(entry: Mapping[str, object]) -> None:
    """Refuses a recorded file that is gone or isn't the file that was gated: its size, then,
    since that costs a read, its SHA-256. The masks choose its events by row, so they fit only
    the bytes they were found in."""
    path = Path(entry["path"])
    if not path.is_file():
        raise _refused(f"{path}, gated as file {entry['number']}, is missing")
    size = path.stat().st_size
    if size != entry["size"]:
        raise _refused(
            f"{path} isn't the file that was gated: it has {size:,} bytes, where the gated file "
            f"had {entry['size']:,}"
        )
    if file_sha256(path) != entry["sha256"]:
        raise _refused(
            f"{path} isn't the file that was gated: its SHA-256 differs from the gated file's"
        )


def write_run(
    out: Path,
    gates: Path,
    settings: ClusterSettings,
    chosen: Sequence[Sequence[int]],
    channels: Sequence[str],
    labels: Sequence[str],
) -> Path:
    """run.json in out, for a run from the gate folder gates: the clustering's settings, the
    populations kept as --pops gives them, and the channels clustered with their labels."""
    record = _record(RUN_SCHEMA) | {
        "gates": str(Path(gates).absolute()),
        "settings": settings.to_dict(),
        "pops": pops_spec(chosen),
        "channels": list(channels),
        "labels": list(labels),
    }
    return _write_json(out / "run.json", record)
