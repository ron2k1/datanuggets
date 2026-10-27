"""The flow pipeline's settings, one frozen dataclass per stage: GateSettings for the gate finder,
ClusterSettings for data nuggets, Whclust and the t-SNE. Their defaults are datanuggets-flow's,
from _defaults. check() refuses a bad value with a FlowError naming the setting, which the CLI
shows as its flag (cli_flag) and the library by its name. to_dict and from_dict carry the
settings through JSON, so a stage can record the ones it ran with."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Self

from datanuggets._defaults import (
    DN_NUM1,
    FINDER_N_COL,
    FINDER_N_ROW,
    FINDER_T11,
    FINDER_T12,
    FINDER_T21,
    FINDER_T22,
    FLOW_COFACTOR,
    FLOW_DISC,
    FLOW_DN_NUM2,
    FLOW_FSC,
    FLOW_K,
    FLOW_N_JOBS,
    FLOW_SSC,
    SEED,
    TSNE_PERPLEXITY,
    TSNE_SEED,
)
from datanuggets._validate import check_n_jobs
from datanuggets.exceptions import FlowError
from datanuggets.flow.gate import MSIZE
from datanuggets.flow.peaks import find_area, mirrored_area

AREAS = {"his": mirrored_area, "peak": find_area}  # disc's area rules
FLAGS = {"n_jobs": "--jobs", "tsne": "--no-tsne"}  # the settings whose flag isn't their name


def cli_flag(name: str) -> str:
    """The datanuggets-flow flag that sets name: --dn-num2 for dn_num2, and --jobs for n_jobs."""
    return FLAGS.get(name, "--" + name.replace("_", "-"))


def _finite(v) -> bool:
    """v is a finite number. A bool isn't, though Python counts it as one."""
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _whole(v, low, high=math.inf) -> bool:
    """v is an int, not a bool, from low to high."""
    return isinstance(v, int) and not isinstance(v, bool) and low <= v <= high


def _refused(name: str, value, takes: str) -> FlowError:
    """The error for a setting whose value is out of range, saying what it takes."""
    return FlowError(f"{{param}} is {value}; it takes {takes}", name)


class _Settings:
    """to_dict and from_dict for the settings dataclasses."""

    def to_dict(self) -> dict[str, object]:
        """The settings as JSON writes them, tuples as lists."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """The settings to_dict gave, lists back as tuples. A key that isn't a setting is refused,
        so a record from another version isn't read as if it held the same settings."""
        names = {f.name for f in fields(cls)}
        for key in data:
            if key not in names:
                kind = cls.__name__.removesuffix("Settings").lower()
                raise FlowError(f"{{param}} isn't a {kind} setting", key)
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


@dataclass(frozen=True, kw_only=True)
class GateSettings(_Settings):
    """The gate finder's settings: the scatter channels it gates on, disc, the area rule (his for
    cfc.py's mirrored_area, peak for find_area, the disc on the peak), and lymphmacro_finder's
    thresholds and window. t11 and t21, the first peak's, give one value for every file or one
    per file, as cfc.py's per-file values do."""

    fsc: str = FLOW_FSC
    ssc: str = FLOW_SSC
    disc: str = FLOW_DISC
    t11: tuple[float, ...] = (FINDER_T11,)
    t21: tuple[float, ...] = (FINDER_T21,)
    t12: float = FINDER_T12
    t22: float = FINDER_T22
    n_row: int = FINDER_N_ROW
    n_col: int = FINDER_N_COL

    def check(self, n_files: int | None = None) -> Self:
        """self, when the finder takes every value. With n_files, t11 and t21 must also give one
        value, or one per file."""
        if self.disc not in AREAS:
            raise FlowError(f"{{param}} is {self.disc!r}; it takes {' or '.join(AREAS)}", "disc")
        for name in ("t11", "t21"):
            got = len(getattr(self, name))
            if not got:
                raise FlowError("{param} gives no values; give one, or one per file", name)
            if n_files is not None and got not in (1, n_files):
                raise FlowError(
                    f"{{param}} gives {got} values for {n_files} files; give one, or one per file",
                    name,
                )
        for name, values in (("t11", self.t11), ("t12", (self.t12,))):
            for v in values:
                if not (_finite(v) and v > 0):
                    raise _refused(name, v, "a positive distance in grid cells")
        for name, values in (("t21", self.t21), ("t22", (self.t22,))):
            for v in values:
                if not (_finite(v) and v >= 0):
                    raise _refused(name, v, "a fraction of the peak's density, 0 or more")
        for name in ("n_row", "n_col"):
            if not _whole(getattr(self, name), 1, MSIZE):
                raise _refused(name, getattr(self, name), f"1 to {MSIZE} grid cells")
        return self

    def for_file(self, index: int) -> dict[str, object]:
        """gate_peaks' keyword arguments for the file at index, counting from 0: its own t11 and
        t21, or the ones every file shares, and disc's area rule."""
        return {
            "n_row": self.n_row,
            "n_col": self.n_col,
            "t11": self.t11[index if len(self.t11) > 1 else 0],
            "t21": self.t21[index if len(self.t21) > 1 else 0],
            "t12": self.t12,
            "t22": self.t22,
            "area": AREAS[self.disc],
        }


@dataclass(frozen=True, kw_only=True)
class ClusterSettings(_Settings):
    """The clustering stage's settings: the channels to cluster (None for every -A channel but
    FSC and SSC, less exclude_channels), arcsinh(x / cofactor) of them (0 keeps them linear),
    create_dn's dn_num1, dn_num2 per file, seed and n_jobs, the k clusters cut from the Whclust
    tree, and whether a t-SNE of the nuggets is drawn, with its perplexity and seed."""

    channels: tuple[str, ...] | None = None
    exclude_channels: tuple[str, ...] = ()
    cofactor: float = FLOW_COFACTOR
    dn_num1: int = DN_NUM1
    dn_num2: int = FLOW_DN_NUM2
    seed: int = SEED
    n_jobs: int = FLOW_N_JOBS
    k: int = FLOW_K
    tsne: bool = True
    tsne_perplexity: float = TSNE_PERPLEXITY
    tsne_seed: int = TSNE_SEED

    def check(self) -> Self:
        """self, when create_dn, cut_tree and the t-SNE take every value. The default channels
        are checked once a file's channels are known."""
        if not _whole(self.k, 1):
            raise _refused("k", self.k, "1 or more clusters")
        if not _whole(self.dn_num1, 3):  # room for dn_num2 of 2 below it
            raise _refused("dn_num1", self.dn_num1, "3 or more nuggets")
        if not _whole(self.dn_num2, 2, self.dn_num1 - 1):
            raise _refused("dn_num2", self.dn_num2, f"2 to {self.dn_num1 - 1} nuggets per file")
        if not (_finite(self.cofactor) and self.cofactor >= 0):
            raise _refused("cofactor", self.cofactor, "0 (linear) or a positive number")
        for name in ("seed", "tsne_seed"):
            if not _whole(getattr(self, name), 0):
                raise _refused(name, getattr(self, name), "a whole number from 0")
        try:
            check_n_jobs(self.n_jobs)
        except ValueError:
            raise _refused("n_jobs", self.n_jobs, "-1 (every core) or 1 or more") from None
        if self.channels is not None:
            if len(set(self.channels)) < 2:
                raise FlowError(
                    "choose at least two distinct fluorescence channels for clustering and PCA",
                    "channels",
                )
            if len(set(self.channels)) != len(self.channels):
                raise FlowError(
                    "{param} repeats a channel; list each channel only once", "channels"
                )
            if self.exclude_channels:
                raise FlowError("{param} can't be combined with channels", "exclude_channels")
        if not (_finite(self.tsne_perplexity) and self.tsne_perplexity > 0):
            raise _refused("tsne_perplexity", self.tsne_perplexity, "a positive number")
        return self
