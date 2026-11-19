"""Row filters: which rows of a table filter_mask keeps.

A filter is written COLUMN OP VALUE, with whitespace on both sides of OP, one of OPS:

    MedInc >= 3
    species in setosa,versicolor
    [petal length] between 1,1.5

COLUMN is everything before the first operator, stripped, or a name in [brackets], which keeps a
name holding an operator word or edge spaces whole. VALUE is one value for a comparison, lo,hi for
between (both ends kept, lo <= hi), and one or more comma-separated values for in. A value that
reads as a finite number is a number; any other is text, which only ==, != and in take, and which
can't contain , [ or ]. Filters are ANDed. Needs no matplotlib or pandas."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

OPS = ("<", "<=", ">", ">=", "==", "!=", "between", "in")
NUMERIC_OPS = ("<", "<=", ">", ">=", "between")
TEXT_FORBIDDEN = (",", "[", "]")

Value = float | str | tuple[float, float] | tuple[float | str, ...]

# an operator with whitespace on both sides; the longer symbols first, so <= isn't read as <
_OP = r"(<=|>=|==|!=|<|>|between|in)"
_FIRST_OP = re.compile(rf"\s{_OP}\s")
# greedy: a value can't hold ], so a bracketed column ends at the last ] an operator follows, and
# a column like "a] > [b" reads back whole
_BRACKETED = re.compile(rf"^\[(.*)\]\s+{_OP}\s+(.*)$", re.DOTALL)
_BRACKETED_END = re.compile(rf"^\[(.*?)\]\s+{_OP}\s*$", re.DOTALL)


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _float(v, what: str) -> float:
    """A number as a float. An int past the float range is a ValueError, like every bad value,
    not the OverflowError float() raises."""
    try:
        return float(v)
    except OverflowError:
        raise ValueError(f"{what} got an integer too large for a float") from None


def _number(v, what: str) -> float:
    if not _is_number(v) or not math.isfinite(_float(v, what)):
        raise ValueError(f"{what} needs a number, got {v!r}")
    return float(v)


def _scalar(v, op: str) -> float | str:
    """A number as a float, or text, checked for what the grammar can carry. Text that reads as
    a finite number is that number, as the spec it formats to would read it."""
    if _is_number(v):
        if not math.isfinite(_float(v, op)):
            raise ValueError(f"{op} takes a finite number or text, got {v!r}")
        return float(v)
    if not isinstance(v, str):  # ValueError like every bad filter, so a caller catches one type
        raise ValueError(f"{op} takes a number or text, got {v!r}")  # noqa: TRY004
    if not v.strip() or v != v.strip():
        raise ValueError(f"{op} got an empty value or one with edge spaces: {v!r}")
    if any(c in v for c in TEXT_FORBIDDEN):
        raise ValueError(f"text values can't contain , [ or ]; got {v!r}")
    return _read_value(v)


@dataclass(frozen=True)
class RowFilter:
    """One filter: the column it reads, its operator (one of OPS) and its value. Numbers are kept
    as floats, so RowFilter("x", ">", 3) equals RowFilter("x", ">", 3.0), and text that reads as
    a finite number is kept as that number, as the grammar reads it: RowFilter("x", "==", "3")
    equals RowFilter("x", "==", 3.0), so parse_filter(format_filter(f)) == f for every f. A value
    that doesn't fit the operator raises ValueError."""

    column: str
    op: str
    value: Value

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"op must be one of {', '.join(OPS)}; got {self.op!r}")
        if not isinstance(self.column, str) or not self.column.strip():
            raise ValueError("a filter's column must be a nonempty name")
        v, op = self.value, self.op
        if op in ("<", "<=", ">", ">="):
            v = _number(v, op)
        elif op in ("==", "!="):
            v = _scalar(v, op)
        elif op == "between":
            if isinstance(v, str) or not isinstance(v, Sequence) or len(v) != 2:
                raise ValueError(f"between takes two numbers, lo,hi; got {v!r}")
            lo, hi = (_number(e, "between's lo,hi") for e in v)
            if not lo <= hi:
                raise ValueError(f"between takes lo,hi with lo <= hi; got {lo:g},{hi:g}")
            v = (lo, hi)
        else:
            if isinstance(v, str) or not isinstance(v, Sequence):
                raise ValueError(f"in takes a list of values; got {v!r}")
            if not v:
                raise ValueError("in takes at least one value")
            v = tuple(_scalar(e, op) for e in v)
        object.__setattr__(self, "value", v)

    @property
    def numeric(self) -> bool:
        """Every value is a number, so the column's cells must be numbers too."""
        if self.op in NUMERIC_OPS:
            return True
        values = self.value if self.op == "in" else (self.value,)
        return all(isinstance(v, float) for v in values)


def _read_value(text: str) -> float | str:
    """One value of a spec: a number when it reads as a finite one, else text."""
    try:
        v = float(text)
    except ValueError:
        return text
    return v if math.isfinite(v) else text


def _split(text: str, spec: str) -> list[str]:
    parts = [p.strip() for p in text.split(",")]
    if any(not p for p in parts):
        raise ValueError(f"{spec!r} has an empty value in {text!r}")
    return parts


def parse_filter(text: str) -> RowFilter:
    """The RowFilter a spec writes, as in the module's docstring. A spec the grammar can't read
    raises ValueError saying what is wrong."""
    spec = str(text).strip()
    if spec.startswith("["):
        m = _BRACKETED.match(spec)
        if not m:
            if _BRACKETED_END.match(spec):
                raise ValueError(f"{spec!r} has no value after its operator")
            raise ValueError(
                f"{spec!r} has no operator after its [bracketed] column; write [COLUMN] OP VALUE "
                f"with one of {' '.join(OPS)} between spaces"
            )
        column, op, rest = m.group(1), m.group(2), m.group(3).strip()
    else:
        padded = f" {spec} "  # an operator at either end still reads as one
        m = _FIRST_OP.search(padded)
        if not m:
            raise ValueError(
                f"{spec!r} has no operator; write COLUMN OP VALUE with one of "
                f"{' '.join(OPS)} between spaces"
            )
        column, op, rest = padded[: m.start()].strip(), m.group(1), padded[m.end() :].strip()
    if not column.strip():
        raise ValueError(f"{spec!r} has no column before its operator")
    if not rest:
        raise ValueError(f"{spec!r} has no value after its operator")
    if any(c in rest for c in "[]"):
        raise ValueError(f"{spec!r}: text values can't contain , [ or ]")
    if op == "between":
        parts = [p.strip() for p in rest.split(",")]
        values = [_read_value(p) for p in parts]
        if len(parts) != 2 or not all(isinstance(v, float) for v in values):
            raise ValueError(f"{spec!r}: between takes two numbers, lo,hi")
        value: Value = (values[0], values[1])
    elif op == "in":
        value = tuple(_read_value(p) for p in _split(rest, spec))
    else:
        if "," in rest:
            raise ValueError(f"{spec!r}: {op} takes one value; text values can't contain , [ or ]")
        value = _read_value(rest)
        if op in NUMERIC_OPS and not isinstance(value, float):
            raise ValueError(f"{spec!r}: {op} needs a number, got {rest!r}")
    try:
        return RowFilter(column, op, value)
    except ValueError as e:
        raise ValueError(f"{spec!r}: {e}") from None


def _format_value(v: float | str) -> str:
    if isinstance(v, str):
        return v
    return str(int(v)) if v.is_integer() and abs(v) < 2**53 else repr(v)


def format_filter(f: RowFilter) -> str:
    """f's canonical spec, which parse_filter reads back as f. The column is bracketed when it
    holds whitespace, starts with [ or is an operator, which a bare spec would read as one."""
    column = f.column
    if column.startswith("[") or column in OPS or any(c.isspace() for c in column):
        column = f"[{column}]"
    if f.op in ("between", "in"):
        value = ",".join(_format_value(v) for v in f.value)
    else:
        value = _format_value(f.value)
    return f"{column} {f.op} {value}"


def _numbers(cells: np.ndarray, f: RowFilter) -> np.ndarray:
    """The column as floats, and a ValueError naming the first cell that isn't a finite number."""
    if cells.dtype.kind in "fiu":
        x = cells.astype(float)
    else:
        try:
            x = cells.astype(float)
        except (TypeError, ValueError):
            for i, cell in enumerate(cells.tolist()):
                try:
                    float(cell)
                except (TypeError, ValueError):
                    raise _not_a_number(f, i, cell) from None
            raise
    bad = np.flatnonzero(~np.isfinite(x))
    if bad.size:
        raise _not_a_number(f, int(bad[0]), cells[bad[0]])
    return x


def _not_a_number(f: RowFilter, row: int, cell) -> ValueError:
    return ValueError(
        f"filter {format_filter(f)!r}: column {f.column!r}, data row {row} is not a finite "
        f"number ({cell!r})"
    )


def _text(cells: np.ndarray) -> np.ndarray:
    return cells.astype(str)


def _mixed_in(cells: np.ndarray, values) -> np.ndarray:
    """in with numbers and text: a cell matches a text value by its text, or a number by its
    value when it reads as a finite number."""
    words = [v for v in values if isinstance(v, str)]
    nums = [v for v in values if isinstance(v, float)]
    text = _text(cells)
    hit = np.isin(text, words)
    for i in np.flatnonzero(~hit):
        try:
            x = float(cells[i])
        except (TypeError, ValueError):
            continue
        hit[i] = math.isfinite(x) and x in nums
    return hit


def _one(cells: np.ndarray, f: RowFilter) -> np.ndarray:
    op, v = f.op, f.value
    if op == "in" and not f.numeric and any(isinstance(e, float) for e in v):
        return _mixed_in(cells, v)
    if f.numeric:
        x = _numbers(cells, f)
        if op == "between":
            return (x >= v[0]) & (x <= v[1])
        if op == "in":
            return np.isin(x, np.asarray(v, dtype=float))
    else:
        x = _text(cells)
        if op == "in":
            return np.isin(x, list(v))
    return {
        "<": np.less,
        "<=": np.less_equal,
        ">": np.greater,
        ">=": np.greater_equal,
        "==": np.equal,
        "!=": np.not_equal,
    }[op](x, v)


def filter_mask(
    get_column: Callable[[str], np.ndarray], filters: Sequence[RowFilter], n: int
) -> np.ndarray:
    """The rows every filter keeps, as n bools. get_column(name) gives a column's n cells, as
    numbers or as text: a number filter compares them as float64, and a cell that isn't a finite
    number under one raises ValueError naming the column and its 0-based data row; text compares
    by exact text."""
    keep = np.ones(n, dtype=bool)
    for f in filters:
        if not isinstance(f, RowFilter):
            raise TypeError(f"filters must be RowFilter, got {type(f).__name__}")
        cells = np.asarray(get_column(f.column))
        if cells.shape != (n,):
            raise ValueError(f"column {f.column!r} gives {cells.size} values for {n} rows")
        keep &= np.asarray(_one(cells, f), dtype=bool)
    return keep
