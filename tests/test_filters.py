"""Row filters: the COLUMN OP VALUE grammar, its canonical text, and the mask they give on real
data (Fisher's iris, as scikit-learn bundles it)."""

import numpy as np
import pytest

from datanuggets.filters import OPS, RowFilter, filter_mask, format_filter, parse_filter


@pytest.fixture(scope="module")
def iris():
    """Iris as CSV text columns, each cell the text a CSV file holds."""
    datasets = pytest.importorskip("sklearn.datasets")
    data = datasets.load_iris()
    names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    columns = {
        name: np.array([repr(float(v)) for v in data.data[:, j]], dtype=object)
        for j, name in enumerate(names)
    }
    columns["species"] = np.array(data.target_names[data.target], dtype=object)
    return columns, data


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("MedInc >= 3", RowFilter("MedInc", ">=", 3.0)),
        ("MedInc>=3", None),
        ("x < -1.5", RowFilter("x", "<", -1.5)),
        ("x <= 2e3", RowFilter("x", "<=", 2000.0)),
        ("x > 0", RowFilter("x", ">", 0.0)),
        ("x == 4", RowFilter("x", "==", 4.0)),
        ("x != 4", RowFilter("x", "!=", 4.0)),
        ("species == setosa", RowFilter("species", "==", "setosa")),
        ("species != setosa", RowFilter("species", "!=", "setosa")),
        ("species in setosa,versicolor", RowFilter("species", "in", ("setosa", "versicolor"))),
        ("code in 1, 2 ,x", RowFilter("code", "in", (1.0, 2.0, "x"))),
        ("code in 7", RowFilter("code", "in", (7.0,))),
        ("[FJComp-BV510-A] between 0,50000", RowFilter("FJComp-BV510-A", "between", (0.0, 5e4))),
        ("sepal length between 1, 1.5", RowFilter("sepal length", "between", (1.0, 1.5))),
        (
            "[FJComp-Alexa Fluor 700-A] > 100",
            RowFilter("FJComp-Alexa Fluor 700-A", ">", 100.0),
        ),
        ("[ margin in ] in a,b", RowFilter(" margin in ", "in", ("a", "b"))),
        ("[conc [mg/L]] >= 3", RowFilter("conc [mg/L]", ">=", 3.0)),
        ("  padded   ==  NaN ", RowFilter("padded", "==", "NaN")),
    ],
)
def test_parse_reads_every_op_and_both_column_forms(text, want):
    if want is None:
        with pytest.raises(ValueError, match="no operator"):
            parse_filter(text)
        return
    assert parse_filter(text) == want


@pytest.mark.parametrize(
    "f",
    [
        RowFilter("a", op, v)
        for op, v in [
            ("<", 1.0),
            ("<=", -2.5),
            (">", 1e-300),
            (">=", 123456789.125),
            ("==", 4.0),
            ("==", "setosa"),
            ("!=", 0.1),
            ("!=", "x y"),
            ("between", (0.0, 50000.0)),
            ("in", (1.0, "b", 2.5)),
            ("in", ("only",)),
        ]
    ]
    + [
        RowFilter("FJComp-Alexa Fluor 700-A", ">", 3.0),
        RowFilter(" edge ", "==", 1.0),
        RowFilter("[x]", "<", 2.0),
        RowFilter("conc [mg/L]", "between", (1.0, 2.0)),
        RowFilter("a in b", "in", ("c",)),
        # text that reads as a number is the number the spec would give
        RowFilter("a", "==", "3"),
        RowFilter("a", "!=", "-2.5"),
        RowFilter("a", "in", ("1e5", "abc")),
        RowFilter("a", "==", "1_000"),
        RowFilter("a", "in", ("+7", "7")),
        # text that reads as no finite number stays text
        RowFilter("a", "==", "nan"),
        RowFilter("a", "in", ("inf", "1e400")),
        # a bracketed column holding "] OP [" ends at its last ]
        RowFilter("a] > [b", ">", 1.0),
        RowFilter("x] == [y z", "==", "c"),
        # a column named like an operator, which a bare spec would read as the operator
        RowFilter("in", "==", 1.0),
        RowFilter("between", ">", 2.0),
        RowFilter("<", "==", "a"),
        RowFilter("!=", "in", ("x", 3.0)),
    ],
)
def test_format_round_trips(f):
    text = format_filter(f)
    assert parse_filter(text) == f
    assert format_filter(parse_filter(text)) == text


def test_text_that_reads_as_a_number_is_kept_as_that_number(iris):
    columns, _ = iris
    assert RowFilter("x", "==", "3") == RowFilter("x", "==", 3.0)
    assert RowFilter("x", "in", ("1e5", "abc")).value == (100000.0, "abc")
    assert RowFilter("x", "==", "nan").value == "nan" and not RowFilter("x", "==", "nan").numeric
    # so it masks as the spec does: by value, 5 matching the iris cell "5.0"
    typed = RowFilter("sepal_length", "==", "5")
    keep = filter_mask(columns.get, [typed], 150)
    assert np.array_equal(keep, filter_mask(columns.get, [parse_filter("sepal_length == 5")], 150))
    assert keep.sum() == (columns["sepal_length"] == "5.0").sum() > 0
    # every iris cell, typed as the text the CSV holds, round-trips as its number
    for name in ("sepal_length", "petal_width"):
        cells = sorted(set(columns[name]))
        for f in (RowFilter(name, "==", cells[0]), RowFilter(name, "in", tuple(cells))):
            assert parse_filter(format_filter(f)) == f and f.numeric


def test_format_writes_the_canonical_spec():
    assert format_filter(RowFilter("MedInc", ">=", 3)) == "MedInc >= 3"
    assert format_filter(RowFilter("x", "between", (0, 0.5))) == "x between 0,0.5"
    assert format_filter(RowFilter("s", "in", ("setosa", 2))) == "s in setosa,2"
    assert format_filter(RowFilter("a b", "<", 1.5)) == "[a b] < 1.5"
    assert RowFilter("x", "between", (0, 1)) == RowFilter("x", "between", (0.0, 1.0))
    assert set(OPS) == {"<", "<=", ">", ">=", "==", "!=", "between", "in"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no operator"),
        ("MedInc", "no operator"),
        (" >= 3", "no column"),
        ("[] >= 3", "no column"),
        ("[x >= 3", "no operator"),
        ("x >= ", "no value"),
        ("x >= abc", "needs a number"),
        ("x < inf", "needs a number"),
        ("x between 1", "two numbers"),
        ("x between 1,2,3", "two numbers"),
        ("x between 2,1", "lo <= hi"),
        ("x between a,2", "two numbers"),
        ("x in ", "no value"),
        ("x in a,,b", "empty value"),
        ("x == a]b", "can't contain"),
        ("x == [a", "can't contain"),
    ],
)
def test_bad_specs_name_the_problem(text, message):
    with pytest.raises(ValueError, match=message):
        parse_filter(text)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("x", "~", 1.0), "op must be one of"),
        (("", "<", 1.0), "column"),
        (("x", "<", "a"), "needs a number"),
        (("x", "between", (2.0, 1.0)), "lo <= hi"),
        (("x", "in", ()), "at least one"),
        (("x", "==", "a,b"), "can't contain"),
        (("x", "==", float("nan")), "finite"),
        # an int past the float range is a ValueError too, not an OverflowError
        (("x", ">", 10**400), "too large for a float"),
        (("x", "==", 10**400), "too large for a float"),
        (("x", "between", (0, 10**400)), "too large for a float"),
        (("x", "in", (1.0, -(10**400))), "too large for a float"),
    ],
)
def test_a_bad_filter_cant_be_made(args, message):
    with pytest.raises(ValueError, match=message):
        RowFilter(*args)


def test_iris_masks_match_direct_counts(iris):
    columns, data = iris
    n = len(data.target)
    setosa = filter_mask(columns.__getitem__, [parse_filter("species == setosa")], n)
    assert setosa.dtype == bool and setosa.shape == (n,)
    assert setosa.sum() == 50
    assert np.array_equal(setosa, data.target == 0)
    between = filter_mask(columns.__getitem__, [parse_filter("petal_length between 1,1.5")], n)
    petal = data.data[:, 2]
    assert between.sum() == np.count_nonzero((petal >= 1) & (petal <= 1.5))
    both = filter_mask(
        columns.__getitem__,
        [parse_filter("species in versicolor,virginica"), parse_filter("sepal_width < 3")],
        n,
    )
    assert np.array_equal(both, (data.target > 0) & (data.data[:, 1] < 3))
    assert filter_mask(columns.__getitem__, [], n).all()


def test_numeric_filters_work_on_float_columns_too():
    x = np.array([0.0, 5.0, 10.0, np.nan])
    got = filter_mask(lambda c: x[:3], [RowFilter("x", "between", (1.0, 10.0))], 3)
    assert got.tolist() == [False, True, True]
    got = filter_mask(lambda c: x[:3], [RowFilter("x", "in", (0.0, 10.0))], 3)
    assert got.tolist() == [True, False, True]


def test_a_cell_that_isnt_a_number_names_its_column_and_row():
    col = np.array(["1", "2", "", "4"], dtype=object)
    with pytest.raises(ValueError, match=r"column 'x', data row 2"):
        filter_mask(lambda c: col, [RowFilter("x", ">", 1.0)], 4)
    with pytest.raises(ValueError, match=r"column 'x', data row 3"):
        filter_mask(lambda c: np.array([1.0, 2.0, 3.0, np.inf]), [RowFilter("x", ">", 1.0)], 4)
    # a text filter compares text, so the empty cell is just another value
    assert filter_mask(lambda c: col, [RowFilter("x", "!=", "x")], 4).all()


def test_a_column_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="3 values for 4 rows"):
        filter_mask(lambda c: np.zeros(3), [RowFilter("x", ">", 1.0)], 4)
