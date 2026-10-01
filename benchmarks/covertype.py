"""Replays R's recorded draws on the Covertype benchmark and compares every output bit for bit.

Run from the repo root after benchmarks/covertype.R with the same size:
    uv run python benchmarks/covertype.py 10000 | 100000 | all
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from datanuggets import DataNuggets, DataNuggetWarning, create_dn, refine_dn
from datanuggets._compat import rcompat as _rcompat
from datanuggets._compat import rstats as _rstats
from datanuggets._draws import ReplayDraws
from rfixtures import Case

OUT = ROOT / "tmp" / "covertype" / "out"


def load(name: str) -> Case:
    path = OUT / name
    return Case(path, json.loads((path / "manifest.json").read_text(encoding="utf-8")))


def compare(got: DataNuggets, want: Case) -> dict:
    checks = {
        "assignments": np.array_equal(got.assignments + 1, want.array("assignments")),
        "weights": np.array_equal(got.weights, want.array("weights")),
        "centers": np.array_equal(got.centers, want.array("centers")),
        "scales": np.array_equal(got.scales, want.array("scales")),
    }
    if not checks["assignments"] and got.assignments.shape == want.array("assignments").shape:
        checks["assignments_differing"] = int(
            np.count_nonzero(got.assignments + 1 != want.array("assignments"))
        )
    return checks


class Infos(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        if record.levelno == logging.INFO:
            self.messages.append(record.getMessage())


def stage_lines(messages: list[str]) -> list[str]:
    """R's create.DN messages without its "completed!" lines, which the port doesn't log."""
    return [m.strip() for m in messages if m != "completed!"]


def warm_up(x: np.ndarray) -> None:
    """Load (or on a cold cache compile) every numba kernel, so no timing includes it."""
    head = np.ascontiguousarray(x[:50], dtype=np.float64)
    _rcompat.r_mean(head, 0)
    _rstats.r_cov(head)
    list(_rstats.group_moments(head, [np.arange(50)]))
    list(_rstats.group_moments(head, [np.arange(50)], n_jobs=2))


def run(fn, case: Case, want_messages=None):
    handler = Infos()
    logger = logging.getLogger("datanuggets")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    draws = ReplayDraws(case.draws())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            dn = fn(draws)
            secs = time.perf_counter() - t0
    finally:
        logger.removeHandler(handler)
    draws.assert_exhausted()
    texts = [str(w.message) for w in caught if w.category is DataNuggetWarning]
    if want_messages is None:
        want_messages = case.manifest["messages"]
    return (
        dn,
        secs,
        {
            "messages": handler.messages == want_messages,
            "warnings": texts == case.manifest["warnings"],
        },
    )


def main(size: str) -> None:
    cid = f"covertype_{size}"
    cr = load(cid + "_create")
    x = cr.array("x")
    warm_up(x)
    report = {"n": int(x.shape[0]), "p": int(x.shape[1])}

    dn, secs, extra = run(lambda d: create_dn(x, seed=d), cr, stage_lines(cr.manifest["messages"]))
    report["create"] = {
        "python_s": round(secs, 2),
        "r_s": cr.manifest["secs"],
        "nuggets": dn.n_nuggets,
        **compare(dn, cr),
        **extra,
    }

    # refine starts from R's create.DN output, so a create mismatch can't hide a refine one
    r_dn = DataNuggets(
        centers=cr.array("centers"),
        weights=cr.array("weights"),
        scales=cr.array("scales"),
        assignments=cr.array("assignments") - 1,
    )
    for suffix, shape in (("_refine", False), ("_refine_shape", True)):
        if not (OUT / (cid + suffix)).is_dir():
            continue
        rc = load(cid + suffix)
        got, secs, extra = run(lambda d, s=shape: refine_dn(x, r_dn, shape_split=s, seed=d), rc)
        report[suffix.lstrip("_")] = {
            "python_s": round(secs, 2),
            "r_s": rc.manifest["secs"],
            "nuggets": f"{r_dn.n_nuggets} -> {got.n_nuggets}",
            **compare(got, rc),
            **extra,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
