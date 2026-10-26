import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from datanuggets.flow import pick
from datanuggets.flow.pick import _fcs_paths, main, parse_pops, per_file, pick_events
from flowcases import FLOW

LABELS = ["CD8 (FJComp-Alexa Fluor 700-A)", "CD3 (FJComp-BV510-A)", "CD4 (FJComp-Spark Blue-A)"]
COLUMNS = {
    "selection.csv": ["file_number", "file", "populations", "events", "kept", "percent"],
    "nuggets_1.csv": ["nugget0", *LABELS, "weight", "scale", "cluster"],
    "events_1.csv": ["event0", "nugget0", "cluster"],
    "cluster_summary.csv": ["file_number", "file", "cluster", "cells", "percent"],
    "cluster_centers.csv": ["cluster", *LABELS],
}
FIGURES = (
    "gates.png",
    "local_maxima.png",
    "dendrogram.png",
    "clusters_pca.png",
    "clusters_tsne.png",
    "cluster_centers.png",
)
# _digest of each table, recorded from pick.py before its main() was cut into steps
DIGESTS = {
    "selection.csv": "23b2c99d62f2c8dd40ec9f70602aef5ac59c0d001dfe8225c6589a2f302c133e",
    "nuggets_1.csv": "be16c779e8e5c25073b342d48263283546c1351fbf1699490627e1e0f4d8d101",
    "events_1.csv": "0cbebf6054bf803a9a970ee44a811bba3f60bfc5742e6bb2fdbae70efbeea144",
    "cluster_summary.csv": "22649c5a60c1f6d03a9c38374ae777825a76f03b4fc1f5be7a658fe57aa9b514",
    "cluster_centers.csv": "ac066c8d5c4d64447cc4d72ca4ceb7372212cf095369616c02eae28680ddb242",
}
DIGITS = 7  # significant digits of a number that isn't whole, in _digest


def _read_202(path):
    """202.fcs as read_fcs returns it, cut to FSC-A, SSC-A and three fluorescence channels."""
    with np.load(FLOW / "202.npz") as g, np.load(FLOW / "202_fluor.npz") as f:
        events = np.column_stack((g["fsc"], g["ssc"], f["fluor"])).astype(float)
        return events, ["FSC-A", "SSC-A", *f["pnn"].tolist()], ["", "", *f["pns"].tolist()]


def _cell(text: str) -> str:
    try:
        int(text)
        return text
    except ValueError:
        pass
    try:
        return format(float(text), f".{DIGITS}g")
    except ValueError:
        return text


def _digest(path) -> str:
    """SHA-256 of a CSV with \\n line ends and each number that isn't whole to DIGITS significant
    digits. pandas ends a line with os.linesep, and np.arcsinh's last bit differs between
    platforms, which moves a nugget's scale by up to about 1e-12 of itself."""
    with open(path, newline="") as f:
        rows = [[_cell(c) for c in row] for row in csv.reader(f)]
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    return hashlib.sha256(text.getvalue().encode()).hexdigest()


@pytest.fixture
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.mark.parametrize("asked", [False, True], ids=["pops", "asked"])
def test_main_clusters_real_events_end_to_end(tmp_path, monkeypatch, close_figures, asked):
    folder, out = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    (folder / "202.fcs").write_bytes(b"")  # for _fcs_paths to find; _read_202 reads it
    monkeypatch.setattr(pick, "read_fcs", _read_202)
    argv = [str(folder), "--no-show", "--out", str(out), "--dn-num2", "50", "--k", "3"]
    if asked:  # Pop 1 typed at the prompt instead
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(pick, "_live_input", lambda prompt, plt, show: "1")
    else:
        argv[1:1] = ["--pops", "1"]
    main(argv)
    assert sorted(p.name for p in out.iterdir()) == sorted([*COLUMNS, *FIGURES])
    for name, columns in COLUMNS.items():
        with open(out / name, newline="") as f:
            assert next(csv.reader(f)) == columns, name
    for name in FIGURES:
        assert (out / name).read_bytes().startswith(b"\x89PNG"), name
    assert {name: _digest(out / name) for name in COLUMNS} == DIGESTS


def test_populations_are_chosen_from_the_local_maxima_before_gates_are_drawn(
    tmp_path, monkeypatch, close_figures
):
    folder, out = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    (folder / "202.fcs").write_bytes(b"")
    monkeypatch.setattr(pick, "read_fcs", _read_202)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    seen = []

    def answer(prompt, plt, show):
        seen.append(sorted(p.name for p in out.iterdir()))
        raise SystemExit("\ncancelled; nothing was clustered")

    monkeypatch.setattr(pick, "_live_input", answer)
    with pytest.raises(SystemExit, match="cancelled"):
        main([str(folder), "--no-show", "--out", str(out)])
    assert seen == [["INCOMPLETE.txt", "local_maxima.png"]]
    assert not (out / "gates.png").exists()


# _digest of each table of the three-file run, recorded from pick.py before its main() was cut
# into steps
DIGESTS_3 = {
    "selection.csv": "0ef31e9c3c4252ed89080acc727502ac0f911580f0bbf297756fee03b164a627",
    "nuggets_1.csv": "128642490cf7274ee9176527b3face34f0339d9f7b3b55a0d93f1afb29459e95",
    "events_1.csv": "15dac77591ea8841683248b2e3d7647d84a6c9be66aaa11fb425e738ac833a5a",
    "nuggets_3.csv": "ee0609722e78ca1fc6b21220ab8f95071a49c4482cde2316686f078301e75d49",
    "events_3.csv": "ce82de3cef82518f7109ea757b7e74c780fb30707e54806fc17e86dccd4065d4",
    "cluster_summary.csv": "ab1c6f6bd1d16b522fbe43373a1dacbc276c48596f65adb0579266bca1e257eb",
    "cluster_centers.csv": "38f71313c81cee4ea83da8029c8626655948c7d78bb2f717f631c27404c90589",
}


def test_main_numbers_each_file_and_clusters_its_own_nuggets(tmp_path, monkeypatch, close_figures):
    """Three copies of 202.fcs, each with its own t11, the second left out: the other two keep
    their numbers, and the third file's slice of the pooled clusters is its own. With the peak's
    disc, the events as stored and two of the channels."""
    folder, out = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    for name in ("a.fcs", "b.fcs", "c.fcs"):
        (folder / name).write_bytes(b"")
    monkeypatch.setattr(pick, "read_fcs", _read_202)
    main(
        [
            str(folder),
            "--no-show",
            "--out",
            str(out),
            "--pops",
            "1;none;1,2",
            "--t11",
            "22,20,18",
            "--t21",
            "0.25,0.25,0.4",
            "--disc",
            "peak",
            "--cofactor",
            "0",
            "--channels",
            "FJComp-BV510-A, FJComp-Spark Blue-A",
            "--dn-num2",
            "40",
            "--k",
            "4",
        ]
    )
    tables = [
        "selection.csv",
        "nuggets_1.csv",
        "events_1.csv",
        "nuggets_3.csv",
        "events_3.csv",
        "cluster_summary.csv",
        "cluster_centers.csv",
    ]
    assert sorted(p.name for p in out.iterdir()) == sorted([*tables, *FIGURES])
    assert {name: _digest(out / name) for name in tables} == DIGESTS_3


def _sample(name, n):
    return {"path": Path(name), "masks": np.zeros((5, n), dtype=bool), "table": [f"  {name}"]}


def test_ask_takes_each_files_pops_and_enter_repeats_the_last(monkeypatch, capsys):
    answers = iter(["", "7", "1,2", "", "none", ""])
    prompts = []

    def typed(prompt, plt, show):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(pick, "_live_input", typed)
    names = [("a.fcs", 3), ("b.fcs", 3), ("c.fcs", 0), ("d.fcs", 2), ("e.fcs", 2)]
    got = pick._ask([_sample(*s) for s in names], None, False)
    assert got == [[1, 2], [1, 2], [], [], []]
    assert prompts[3] == "Keep which, Pop 1 to 3 [1,2]: "
    assert prompts[5] == "Keep which, Pop 1 to 2 [none]: "
    out = capsys.readouterr().out
    assert "no population given" in out and "there is no Pop 7" in out
    assert "File 3, c.fcs:\n  c.fcs\n  no populations found" in out


def test_ask_without_a_terminal_says_to_pass_pops(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="pass --pops"):
        pick._ask([_sample("a.fcs", 2)], None, False)


def test_live_input_without_a_window_reads_a_line(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda: "1,2")
    assert pick._live_input("Keep which: ", None, False) == "1,2"
    assert capsys.readouterr().out == "Keep which: "

    def closed():
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    with pytest.raises(SystemExit, match="nothing was clustered"):
        pick._live_input("Keep which: ", None, False)


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
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("existing", ["directory", "empty-directory", "file"])
def test_existing_output_is_preserved_before_reading_input(tmp_path, monkeypatch, existing):
    source, out = tmp_path / "202.fcs", tmp_path / "out"
    source.touch()
    if existing == "file":
        out.write_bytes(b"keep this file")
        before = out.read_bytes()
    else:
        out.mkdir()
        if existing == "directory":
            for name in (*COLUMNS, *FIGURES, "nuggets_9.csv"):
                (out / name).write_bytes(b"earlier results")
        before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(pick, "read_fcs", lambda path: pytest.fail("read before output check"))
    with pytest.raises(SystemExit, match="output path already exists"):
        main([str(source), "--pops", "1", "--no-show", "--out", str(out)])
    after = out.read_bytes() if out.is_file() else {p.name: p.read_bytes() for p in out.iterdir()}
    assert after == before


@pytest.mark.parametrize(
    "failure", [ValueError("bad FCS"), SystemExit("cancelled"), KeyboardInterrupt()]
)
def test_failed_run_is_marked_and_does_not_touch_previous_results(
    tmp_path, monkeypatch, capsys, failure
):
    source, old, out = tmp_path / "202.fcs", tmp_path / "previous", tmp_path / "new"
    source.touch()
    old.mkdir()
    (old / "events_1.csv").write_bytes(b"earlier events")

    def fail(*args):
        (out / "gates.png").write_bytes(b"partial figure")
        raise failure

    monkeypatch.setattr(pick, "_gate_files", fail)
    with pytest.raises(type(failure)):
        main([str(source), "--pops", "1", "--no-show", "--out", str(out)])
    assert (old / "events_1.csv").read_bytes() == b"earlier events"
    assert (out / "INCOMPLETE.txt").is_file()
    assert "Run incomplete" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="output path already exists"):
        main([str(source), "--pops", "1", "--no-show", "--out", str(out)])


def test_default_runs_have_separate_folders_under_outputs(tmp_path, monkeypatch):
    source = tmp_path / "202.fcs"
    source.touch()
    monkeypatch.chdir(tmp_path)

    def cancel(*args):
        raise SystemExit("cancelled")

    monkeypatch.setattr(pick, "_gate_files", cancel)
    for _ in range(2):
        with pytest.raises(SystemExit, match="cancelled"):
            main([str(source), "--no-show"])
    runs = list((tmp_path / "outputs").iterdir())
    assert len(runs) == 2
    assert all((run / "INCOMPLETE.txt").is_file() for run in runs)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["202.fcs", "outputs"]


@pytest.mark.parametrize("names", [None, "CD3-A", "CD3-A,CD3-A"])
def test_too_few_distinct_channels_stop_before_clustering(names):
    samples = [{"path": Path("one.fcs"), "pnn": ["FSC-A", "SSC-A", "CD3-A"], "pns": ["", "", ""]}]
    with pytest.raises(SystemExit, match="at least two distinct fluorescence channels"):
        pick._choose_channels(samples, names)


def test_two_occupied_nuggets_get_an_actionable_error():
    with np.load(FLOW / "202_fluor.npz") as data:
        x = data["fluor"][:2, :2].astype(float)
    kept = [(1, "202.fcs", np.arange(len(x)), x)]
    with pytest.raises(SystemExit, match="fewer than 3 occupied nuggets"):
        pick._cluster(kept, dn_num2=2, k=2)


@pytest.mark.parametrize(
    ("flag", "value", "match"), [("--k", "0", "--k is 0"), ("--dn-num2", "10000", "2 to 9999")]
)
def test_bad_k_or_dn_num2_stops_before_any_file_is_read(tmp_path, flag, value, match):
    with pytest.raises(SystemExit, match=match):
        main([str(tmp_path / "missing"), flag, value, "--no-show"])
