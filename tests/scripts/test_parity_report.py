"""The create_dn page that scripts/parity_report.py writes, from the recorded R cases.

The script isn't part of the package, so the sdist leaves this folder out and these tests run
only from a clone."""

import importlib.util
from pathlib import Path

import datanuggets.nuggets.create as create_mod
from rfixtures import load_group

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "parity_report.py"
_spec = importlib.util.spec_from_file_location("parity_report", SCRIPT)
assert _spec is not None and _spec.loader is not None
parity_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity_report)

# The rows R gives to another of the tied centers than the lowest-numbered one, per case. Each
# is a tie in a case with more than 16 centers
# (docs/compatibility.md#exact-distance-ties-in-create_dn).
MOVED = {
    "faithful_1d": 1,
    "grid_ties_euclidean_k30": 12,
    "grid_ties_manhattan_k30_random": 12,
    "iris_all_passthrough": 2,
    "iris_default_num1_warnpath": 1,
    "petal_duplicates": 1,
    "quakes_manhattan_mean": 1,
    "quakes_manhattan_original": 1,
    "quakes_manhattan_random": 1,
}
COLUMNS = [
    "case",
    "data",
    "nuggets",
    "assignments",
    "tie rows moved",
    "weights",
    "centers",
    "scales",
]


def _table(page: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    lines = [line for line in page.splitlines() if line.startswith("| ")]
    cells = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    head = cells[0]
    return head, {row[0]: dict(zip(head, row, strict=True)) for row in cells[1:]}


def test_the_create_dn_page_counts_the_moved_tie_rows_and_matches_r_elsewhere():
    real = create_mod.nearest_index
    head, rows = _table(parity_report.create_report())
    assert create_mod.nearest_index is real
    assert head == COLUMNS
    assert set(rows) == {case.id for case in load_group("create_dn")}
    for case_id, row in rows.items():
        assert row["tie rows moved"] == str(MOVED.get(case_id, "none")), case_id
        for column in ("assignments", "weights", "centers", "scales"):
            assert row[column] == "yes", (case_id, column)
