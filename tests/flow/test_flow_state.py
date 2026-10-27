"""Gate folders, the state a staged datanuggets-flow run keeps between its stages: what save_gates
writes and load_gates reads back, what load_gates refuses before the second stage reads anything,
the run folder's marker, and run.json."""

import json

import numpy as np
import pytest

from datanuggets.exceptions import FlowError
from datanuggets.flow import state
from datanuggets.flow.pipeline import gate_files
from datanuggets.flow.settings import ClusterSettings, GateSettings, cli_flag
from flowcases import read_202

MU = "CD3 µ"  # a label gates.json must keep as typed
MARKER_TEXT = (
    "This run has not completed. Its tables and figures may be partial.\n"
    "A successful run removes this marker; rerun in a new output folder.\n"
)


def _read(path):
    """202's events, its first fluorescence label not ASCII."""
    events, pnn, pns = read_202(path)
    return events, pnn, [*pns[:2], MU, *pns[3:]]


def _gated(tmp_path):
    """Two files, in/a.fcs and in/b.fcs, gated with a t11 each, and their gate folder, gates,
    saved by save_gates beside stand-ins for its figures."""
    folder, gates = tmp_path / "in", tmp_path / "gates"
    folder.mkdir()
    gates.mkdir()
    paths = [folder / "a.fcs", folder / "b.fcs"]
    for path in paths:
        path.write_bytes(path.name.encode())  # read by _read in name only
    settings = GateSettings(t11=(22, 20)).check(2)
    samples = gate_files(paths, settings, read=_read)
    for name in ("local_maxima.png", "gates.png"):
        (gates / name).write_bytes(b"png")
    state.save_gates(gates, samples, settings)
    return gates, samples, settings


def test_save_then_load_gates_round_trips(tmp_path):
    gates, samples, settings = _gated(tmp_path)
    loaded = state.load_gates(gates)
    assert (loaded.path, loaded.settings) == (gates, settings)
    assert sorted(p.name for p in gates.iterdir()) == sorted(state.GATE_FILES)
    keys = ("number", "path", "n_events", "pnn", "pns", "stats", "table", "local_maxima")
    for got, want in zip(loaded.samples, samples, strict=True):
        assert {k: got[k] for k in keys} == {k: want[k] for k in keys}
        assert np.array_equal(got["masks"], want["masks"])
        for field in want["peaks"]._fields:
            assert np.array_equal(getattr(got["peaks"], field), getattr(want["peaks"], field))
        assert got["fsc"] is None and got["ssc"] is None  # gates.png is drawn already
    assert [(s["t11"], s["t21"]) for s in loaded.samples] == [(22, 0.25), (20, 0.25)]
    assert loaded.samples[0]["pns"][2] == MU
    text = (gates / "gates.json").read_text(encoding="utf-8")
    assert MU in text and "NaN" not in text


def test_gates_json_is_written_last(tmp_path, monkeypatch):
    def full(path, record):
        raise OSError("no space left")

    monkeypatch.setattr(state, "_write_json", full)
    with pytest.raises(OSError, match="no space left"):
        _gated(tmp_path)
    gates = tmp_path / "gates"
    assert sorted(p.name for p in gates.iterdir()) == ["gates.npz", "gates.png", "local_maxima.png"]
    with pytest.raises(FlowError, match="it has no gates.json"):
        state.load_gates(gates)


def _json(change):
    def breaks(gates):
        path = gates / "gates.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        change(record)
        path.write_text(json.dumps(record), encoding="utf-8")

    return breaks


def _npz(change):
    def breaks(gates):
        path = gates / "gates.npz"
        with np.load(path) as f:
            arrays = dict(f)
        change(arrays)
        np.savez(path, **arrays)

    return breaks


def _fcs(data):
    def breaks(gates):
        path = gates.parent / "in" / "a.fcs"
        path.unlink() if data is None else path.write_bytes(data)

    return breaks


def _not_a_folder(gates):
    for path in gates.iterdir():
        path.unlink()
    gates.rmdir()
    gates.write_bytes(b"")


# each way a gate folder is refused: how it is broken and what the refusal says, naming the gate
# folder or the FCS file
REFUSED = {
    "not a folder": (_not_a_folder, "{gates} isn't a folder"),
    "unfinished": (
        lambda gates: (gates / "INCOMPLETE.txt").write_text(MARKER_TEXT),
        "{gates}'s gate stage did not finish",
    ),
    "no gates.json": (
        lambda gates: (gates / "gates.json").unlink(),
        "{gates} isn't a gate folder: it has no gates.json",
    ),
    "no png": (
        lambda gates: (gates / "gates.png").unlink(),
        "{gates} isn't a gate folder: it has no gates.png",
    ),
    "not JSON": (
        lambda gates: (gates / "gates.json").write_bytes(b'{"schema": '),
        "{json} isn't UTF-8 JSON",
    ),
    "not UTF-8": (
        lambda gates: (gates / "gates.json").write_bytes(b'{"schema": "\xff"}'),
        "{json} isn't UTF-8 JSON",
    ),
    "schema": (
        _json(lambda r: r.update(schema="datanuggets-flow-gates/2")),
        "{json} isn't a datanuggets-flow-gates/1 record",
    ),
    "setting": (_json(lambda r: r["settings"].update(t13=1)), "{json}: t13 isn't a gate setting"),
    "npz key": (_npz(lambda a: a.pop("peaks_2")), "{npz} has no peaks_2"),
    "mask shape": (
        _npz(lambda a: a.update(masks_1=a["masks_1"][:-1])),
        "{npz}'s masks_1 doesn't fit file 1",
    ),
    "fcs missing": (_fcs(None), "{fcs}, gated as file 1, is missing"),
    "fcs size": (_fcs(b"a.fcs, longer"), "{fcs} isn't the file that was gated: it has 13 bytes"),
    "fcs sha256": (_fcs(b"A.fcs"), "{fcs} isn't the file that was gated: its SHA-256"),
}


@pytest.mark.parametrize("case", REFUSED)
def test_load_gates_refuses(tmp_path, case):
    gates, _, _ = _gated(tmp_path)
    breaks, text = REFUSED[case]
    breaks(gates)
    with pytest.raises(FlowError) as e:
        state.load_gates(gates)
    assert e.value.param == "from_gates"
    paths = {
        "gates": gates,
        "json": gates / "gates.json",
        "npz": gates / "gates.npz",
        "fcs": (tmp_path / "in" / "a.fcs").absolute(),
    }
    assert text.format(**paths) in e.value.cli(cli_flag)


def test_begin_refuses_an_existing_path_and_finish_clears_the_marker(tmp_path):
    out, taken = tmp_path / "runs" / "new", tmp_path / "file"
    assert state.begin(out) == out
    assert [p.name for p in out.iterdir()] == [state.MARKER]
    assert (out / state.MARKER).read_text(encoding="utf-8") == MARKER_TEXT
    taken.write_bytes(b"keep this file")
    for path in (out, taken):
        for step in (state.begin, state.check_new):
            with pytest.raises(FlowError) as e:
                step(path)
            assert e.value.param == "out"
            assert e.value.cli(cli_flag) == (
                f"output path already exists: {path}; choose a new --out folder"
            )
    assert state.check_new(tmp_path / "other") == tmp_path / "other"
    assert not (tmp_path / "other").exists()
    state.finish(out)
    assert list(out.iterdir()) == []
    assert taken.read_bytes() == b"keep this file"


def test_write_run_records_the_gate_folder_and_settings(tmp_path):
    settings = ClusterSettings(channels=("CD3-A", "CD4-A"), dn_num2=50, k=3, tsne=False)
    channels, labels = ["CD3-A", "CD4-A"], ["CD3 (CD3-A)", "CD4-A"]
    path = state.write_run(tmp_path, tmp_path / "gates", settings, [[1, 2], []], channels, labels)
    assert path == tmp_path / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == state.RUN_SCHEMA
    assert record["gates"] == str(tmp_path / "gates")
    assert ClusterSettings.from_dict(record["settings"]) == settings
    assert (record["pops"], record["channels"], record["labels"]) == ("1,2;none", channels, labels)
    assert {"datanuggets", "numpy", "python"} <= set(record["versions"])
