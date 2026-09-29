"""Load the R-generated parity fixtures (manifest.json + raw little-endian .bin arrays)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datanuggets._draws import DrawRecord

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
