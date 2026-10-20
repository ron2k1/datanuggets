import numpy as np
import pytest

from datanuggets.flow.pick import _fcs_paths, main, parse_pops, per_file, pick_events


def test_per_file_spreads_one_value_or_takes_one_per_file():
    assert per_file("22", 3, "--t11") == [22.0, 22.0, 22.0]
    assert per_file("22,20,18", 3, "--t11") == [22.0, 20.0, 18.0]
    with pytest.raises(SystemExit, match="2 values for 3 files"):
        per_file("22,20", 3, "--t11")
    with pytest.raises(SystemExit, match="takes numbers"):
        per_file("x", 3, "--t11")


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("1", [1]),
        ("1,2", [1, 2]),
        (" 2 1 ", [1, 2]),
        ("1-3", [1, 2, 3]),
        ("3-1", [1, 2, 3]),
        ("2,2", [2]),
        ("ALL", [1, 2, 3, 4]),
        ("none", []),
    ],
)
def test_parse_pops_reads_numbers_ranges_all_and_none(text, want):
    assert parse_pops(text, 4) == want


@pytest.mark.parametrize("text", ["5", "0", "1,x", "1-", "", "  ", "-2"])
def test_parse_pops_rejects_what_names_no_population(text):
    with pytest.raises(ValueError):
        parse_pops(text, 4)


def test_pick_events_is_the_union_counting_an_event_once():
    masks = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    assert pick_events(masks, [1, 2]).tolist() == [True, True, True, False, False]
    assert pick_events(masks, [3]).tolist() == [False, False, False, True, False]
    assert not pick_events(masks, []).any()


def test_fcs_paths_skips_dotfiles_and_other_files(tmp_path):
    for name in ("b.fcs", "a.FCS", "._a.fcs", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in _fcs_paths([tmp_path])] == ["a.FCS", "b.fcs"]


def test_fcs_paths_stops_on_an_empty_folder(tmp_path):
    with pytest.raises(SystemExit, match="no .fcs files"):
        _fcs_paths([tmp_path])


def test_pops_must_give_one_choice_or_one_per_file(tmp_path):
    for name in ("a.fcs", "b.fcs", "c.fcs"):
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(SystemExit, match="2 choices for 3 files"):
        main([str(tmp_path), "--pops", "1;2", "--no-show", "--out", str(tmp_path / "out")])


@pytest.mark.parametrize(
    ("flag", "value", "match"), [("--k", "0", "--k is 0"), ("--dn-num2", "10000", "2 to 9999")]
)
def test_bad_k_or_dn_num2_stops_before_any_file_is_read(tmp_path, flag, value, match):
    with pytest.raises(SystemExit, match=match):
        main([str(tmp_path / "missing"), flag, value, "--no-show"])
