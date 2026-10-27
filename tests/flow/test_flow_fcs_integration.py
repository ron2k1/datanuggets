"""Read an actual FCS file and run the CLI on the recorded 202 events.

The fixture reconstructs an FCS 3.1 file from measured data, with its recorded channel
names and labels. It exercises the parser without redistributing the original FCS file.
"""

import subprocess
import sys
import warnings

import flowio
import numpy as np
import pandas as pd
import pytest

from datanuggets.exceptions import DataNuggetWarning
from datanuggets.flow import pipeline
from datanuggets.flow.pick import read_fcs
from flowcases import FLOW, load_case


@pytest.fixture(scope="module")
def recorded_fcs(tmp_path_factory):
    path = tmp_path_factory.mktemp("recorded-fcs") / "202.fcs"
    with np.load(FLOW / "202.npz") as gate, np.load(FLOW / "202_fluor.npz") as fluor:
        events = np.column_stack((gate["fsc"], gate["ssc"], fluor["fluor"]))
        names = ["FSC-A", "SSC-A", *fluor["pnn"].tolist()]
        labels = ["", "", *fluor["pns"].tolist()]
    with path.open("wb") as handle:
        flowio.create_fcs(handle, events.ravel(), names, opt_channel_names=labels)
    return path, events, names, labels


def test_read_fcs_preserves_measured_events_and_channel_metadata(recorded_fcs):
    path, expected, names, labels = recorded_fcs
    events, pnn, pns = read_fcs(path)
    assert path.read_bytes().startswith(b"FCS3.1")
    assert events.shape == (50_000, 5)
    np.testing.assert_array_equal(events, expected)
    assert pnn == names
    assert pns == labels  # Missing $PnS fields become empty strings for FSC-A and SSC-A.


def _spilled(path, off):
    """A small FCS file whose spillover matrix is the identity plus off from SSC-A into CD3-A."""
    with path.open("wb") as handle:
        flowio.create_fcs(
            handle,
            np.arange(30, dtype=float),
            ["FSC-A", "SSC-A", "CD3-A"],
            metadata_dict={"SPILLOVER": f"2,SSC-A,CD3-A,1,{off},0,1"},
        )
    return path


def test_spillover_warns_only_off_the_identity_and_the_cli_prints_it(tmp_path, capsys):
    text = "c.fcs has a compensation matrix, but its events are clustered as stored, uncompensated"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning fails the read
        for name, off in (("a.fcs", 0), ("b.fcs", 1e-4)):
            pipeline.read_fcs(_spilled(tmp_path / name, off))
    path = _spilled(tmp_path / "c.fcs", 0.01)
    with pytest.warns(DataNuggetWarning) as caught:
        events, pnn, pns = pipeline.read_fcs(path)
    assert [str(w.message) for w in caught] == [text]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = read_fcs(path)  # the CLI's prints it as a line of the run's output instead
    assert capsys.readouterr().out == f"warning: {text}\n"
    np.testing.assert_array_equal(got[0], events)
    assert got[1:] == (pnn, pns) == (["FSC-A", "SSC-A", "CD3-A"], ["", "", ""])


def test_cli_reads_fcs_and_writes_consistent_cell_clusters(recorded_fcs, tmp_path):
    path, measured, names, labels = recorded_fcs
    out = tmp_path / "results"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "datanuggets.flow.pick",
            str(path),
            "--pops",
            "1",
            "--no-show",
            "--out",
            str(out),
            "--dn-num2",
            "50",
            "--k",
            "3",
        ],
        cwd=tmp_path,
        input="",
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Outputs are in" in run.stdout
    tables = {
        name: pd.read_csv(out / name)
        for name in (
            "selection.csv",
            "nuggets_1.csv",
            "events_1.csv",
            "cluster_summary.csv",
            "cluster_centers.csv",
        )
    }
    figures = (
        "gates.png",
        "local_maxima.png",
        "dendrogram.png",
        "clusters_pca.png",
        "clusters_tsne.png",
        "cluster_centers.png",
    )
    assert {p.name for p in out.iterdir()} == {*tables, *figures}
    for name in figures:
        image = (out / name).read_bytes()
        assert image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) > 1000, name

    selected = tables["selection.csv"].iloc[0]
    events = tables["events_1.csv"]
    nuggets = tables["nuggets_1.csv"]
    summary = tables["cluster_summary.csv"]
    centers = tables["cluster_centers.csv"]
    kept_rows = np.flatnonzero(load_case("202").his_masks[:, 0])
    assert selected["file"] == path.name
    assert selected["file_number"] == selected["populations"] == 1
    assert selected["events"] == len(measured)
    assert selected["kept"] == len(events) == len(kept_rows)
    assert selected["percent"] == pytest.approx(100 * len(events) / len(measured))
    np.testing.assert_array_equal(events["event0"], kept_rows)

    channel_labels = [f"{label} ({name})" for name, label in zip(names[2:], labels[2:])]
    assert list(nuggets.columns) == ["nugget0", *channel_labels, "weight", "scale", "cluster"]
    assert list(centers.columns) == ["cluster", *channel_labels]
    assert 3 <= len(nuggets) <= 50
    np.testing.assert_array_equal(nuggets["nugget0"], np.arange(len(nuggets)))
    np.testing.assert_array_equal(
        nuggets["weight"], np.bincount(events["nugget0"], minlength=len(nuggets))
    )
    np.testing.assert_array_equal(
        events["cluster"], nuggets["cluster"].to_numpy()[events["nugget0"]]
    )
    assert set(events["cluster"]) == {1, 2, 3}
    np.testing.assert_array_equal(summary["cluster"], [1, 2, 3])
    np.testing.assert_array_equal(summary["cells"], np.bincount(events["cluster"])[1:])
    np.testing.assert_allclose(summary["percent"], 100 * summary["cells"] / len(events))
    for row in centers.itertuples(index=False, name=None):
        members = nuggets[nuggets["cluster"] == row[0]]
        np.testing.assert_allclose(
            row[1:], np.average(members[channel_labels], axis=0, weights=members["weight"])
        )


def _flow(cwd, *args):
    """datanuggets-flow run as a command with --no-show, which must succeed."""
    run = subprocess.run(
        [sys.executable, "-m", "datanuggets.flow.pick", *map(str, args), "--no-show"],
        cwd=cwd,
        input="",
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_cli_staged_run_on_a_real_fcs_file_matches_the_one_shot(recorded_fcs, tmp_path):
    """--gates-only, then --from-gates with the clustering flags: every table is the one-shot
    run's, byte for byte, and the run folder holds the one-shot run's files and run.json."""
    path = recorded_fcs[0]
    one, gates, staged = tmp_path / "one", tmp_path / "gates", tmp_path / "staged"
    cluster = ["--pops", "1", "--dn-num2", "50", "--k", "3"]
    _flow(tmp_path, path, "--out", one, *cluster)
    _flow(tmp_path, path, "--gates-only", "--out", gates)
    _flow(tmp_path, "--from-gates", gates, "--out", staged, *cluster)
    names = {p.name for p in one.iterdir()}
    assert {p.name for p in staged.iterdir()} == names | {"run.json"}
    tables = sorted(name for name in names if name.endswith(".csv"))
    assert len(tables) == 5
    for name in tables:
        assert (staged / name).read_bytes() == (one / name).read_bytes(), name
