"""Exercise the CSV CLI with measured quakes data and public output contracts."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datanuggets import DataNuggets, cli, cut_tree, dn_whclust, dn_wkmeans, dn_wpca
from rfixtures import FIXTURE_ROOT

FEATURES = ["depth", "magnitude", "stations"]


@pytest.fixture(scope="module")
def quakes():
    """The first 120 measured earthquakes from the existing R parity fixture."""
    folder = FIXTURE_ROOT / "create_dn" / "quakes_seed7_random"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest["arrays"]["x"]
    values = np.fromfile(folder / meta["file"], dtype=meta["dtype"]).reshape(meta["shape"])
    return pd.DataFrame(values[:120].copy(), columns=["latitude", "longitude", *FEATURES])


@pytest.fixture
def source(tmp_path, quakes):
    path = tmp_path / "my earthquakes.csv"
    frame = quakes.copy()
    frame.insert(0, "sample_id", [f"quake-{i}" for i in range(len(frame))])
    frame.to_csv(path, index=False)
    return path


def _args(source, out=None, *, columns=FEATURES, analysis=("none",), plots=False):
    args = [
        "analyze",
        str(source),
        "--columns",
        *columns,
        "--nuggets",
        "20",
        "--intermediate-nuggets",
        "60",
        "--analysis",
        *analysis,
        "--seed",
        "42",
        "--jobs",
        "1",
    ]
    if not plots:
        args.append("--no-plots")
    if out is not None:
        args.extend(["--out", str(out)])
    return args


def _tables(out):
    return (
        pd.read_csv(out / "nuggets.csv", float_precision="round_trip"),
        pd.read_csv(out / "assignments.csv"),
        json.loads((out / "run.json").read_text(encoding="utf-8")),
    )


def _assert_failed(args, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code not in (0, None)
    output = capsys.readouterr()
    diagnostic = output.err + str(exc.value)
    assert diagnostic.strip()
    assert "Traceback (most recent call last)" not in diagnostic
    return diagnostic


def _assert_mapping(nuggets, assignments, expected):
    ids = assignments["nugget0"].to_numpy()
    np.testing.assert_array_equal(assignments["row0"], np.arange(len(expected)))
    np.testing.assert_array_equal(nuggets["nugget0"], np.arange(len(nuggets)))
    assert np.issubdtype(ids.dtype, np.integer)
    assert np.all((0 <= ids) & (ids < len(nuggets)))
    np.testing.assert_array_equal(nuggets["weight"], np.bincount(ids, minlength=len(nuggets)))
    assert nuggets["weight"].sum() == len(expected)
    centers = nuggets[[f"feature:{name}" for name in expected.columns]].to_numpy()
    for i, center in enumerate(centers):
        np.testing.assert_allclose(center, expected.to_numpy()[ids == i].mean(axis=0), atol=1e-12)
    assert np.isfinite(centers).all()
    assert np.isfinite(nuggets["scale"]).all()
    return ids


def test_analyze_preserves_rows_mass_and_input_with_provenance(source, tmp_path, quakes):
    before = source.read_bytes()
    out = tmp_path / "analysis"
    assert cli.main(_args(source, out)) is None
    nuggets, assignments, run = _tables(out)
    _assert_mapping(nuggets, assignments, quakes[FEATURES])
    assert source.read_bytes() == before
    assert not (out / "INCOMPLETE.txt").exists()
    assert not (out / "plots").exists()
    assert run["features"] == FEATURES
    assert run["row_count"] == len(quakes)
    assert run["nugget_count"] == len(nuggets)
    assert run["input"]["name"] == source.name
    assert run["input"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert run["schema_version"]
    assert run["complete"] is True
    assert run["analyses"] == []
    assert run["parameters"]["seed"] == 42
    assert run["versions"]["datanuggets"]


def test_checksum_describes_analyzed_bytes_if_input_changes(source, tmp_path, quakes, monkeypatch):
    before = source.read_bytes()
    read_csv = cli._read_csv

    def replace_after_read(path, columns):
        result = read_csv(path, columns)
        path.write_text("replacement data from another process\n", encoding="utf-8")
        return result

    monkeypatch.setattr(cli, "_read_csv", replace_after_read)
    out = tmp_path / "input-replaced"
    cli.main(_args(source, out))
    nuggets, assignments, run = _tables(out)
    _assert_mapping(nuggets, assignments, quakes[FEATURES])
    assert source.read_bytes() != before
    assert run["input"]["sha256"] == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize(
    "transforms", [("log1p",), ("arcsinh",), ("standardize",), ("arcsinh", "standardize")]
)
def test_transform_order_and_saved_parameters(source, tmp_path, quakes, transforms):
    out = tmp_path / "transformed"
    cli.main(_args(source, out) + ["--transform", *transforms, "--cofactor", "7"])
    nuggets, assignments, run = _tables(out)
    expected = quakes[FEATURES].to_numpy().copy()
    assert [step["name"] for step in run["transformations"]] == list(transforms)
    for name, step in zip(transforms, run["transformations"]):
        if name == "log1p":
            expected = np.log1p(expected)
        elif name == "arcsinh":
            assert step["cofactor"] == 7
            expected = np.arcsinh(expected / 7)
        elif name == "standardize":
            mean, scale = expected.mean(axis=0), expected.std(axis=0, ddof=0)
            np.testing.assert_allclose(step["mean"], mean)
            np.testing.assert_allclose(step["scale"], scale)
            expected = (expected - mean) / scale
    _assert_mapping(nuggets, assignments, pd.DataFrame(expected, columns=FEATURES))


def test_standardization_handles_constant_features(source, tmp_path, quakes):
    frame = quakes.copy()
    frame["stations"] = frame["stations"].iloc[0]
    frame.to_csv(source, index=False)
    out = tmp_path / "constant"
    cli.main(_args(source, out) + ["--transform", "standardize"])
    nuggets, _, run = _tables(out)
    assert run["transformations"][0]["scale"][-1] == 1
    np.testing.assert_array_equal(nuggets["feature:stations"], 0)


def test_multiple_analyses_match_core_apis_and_original_unit_summaries(source, tmp_path, quakes):
    out = tmp_path / "all-analyses"
    cli.main(
        _args(source, out, analysis=("kmeans", "hclust", "pca"))
        + ["--transform", "standardize", "--clusters", "3"]
    )
    nuggets, assignments, _ = _tables(out)
    expected = (quakes[FEATURES] - quakes[FEATURES].mean()) / quakes[FEATURES].std(ddof=0)
    ids = _assert_mapping(nuggets, assignments, expected)
    dn = DataNuggets(
        nuggets[[f"feature:{name}" for name in FEATURES]].to_numpy(),
        nuggets["weight"].to_numpy(),
        nuggets["scale"].to_numpy(),
        ids,
    )
    tree = dn_whclust(dn)
    expected_labels = {
        "kmeans": dn_wkmeans(dn, 3, num_init=10, max_iter=100, seed=42).labels,
        "hclust": cut_tree(tree, k=3),
    }
    linkage = pd.read_csv(out / "hierarchy.csv")
    np.testing.assert_allclose(linkage[["left", "right", "height", "nuggets"]], tree.to_linkage())
    summaries = pd.read_csv(out / "cluster_summary.csv")
    assert set(summaries["method"]) == set(expected_labels)
    for method, labels in expected_labels.items():
        np.testing.assert_array_equal(nuggets[f"{method}_cluster"], labels)
        row_labels = assignments[f"{method}_cluster"].to_numpy()
        np.testing.assert_array_equal(row_labels, labels[ids])
        rows = summaries[summaries["method"] == method]
        assert rows["rows"].sum() == len(quakes)
        assert rows["percent"].sum() == pytest.approx(100)
        for record in rows.to_dict("records"):
            selected = row_labels == record["cluster"]
            assert record["rows"] == selected.sum()
            assert record["percent"] == pytest.approx(100 * selected.mean())
            np.testing.assert_allclose(
                [record[f"mean:{name}"] for name in FEATURES],
                quakes.loc[selected, FEATURES].mean(),
            )
    pca = dn_wpca(dn)
    scores = pd.read_csv(out / "pca_scores.csv")
    loadings = pd.read_csv(out / "pca_loadings.csv")
    variance = pd.read_csv(out / "pca_variance.csv")
    center = pd.read_csv(out / "pca_center.csv")
    pcs = [f"PC{i + 1}" for i in range(len(FEATURES))]
    assert list(loadings["feature"]) == FEATURES
    assert list(center["feature"]) == FEATURES
    np.testing.assert_allclose(center["center"], pca.center)
    np.testing.assert_array_equal(scores["nugget0"], nuggets["nugget0"])
    np.testing.assert_array_equal(scores["weight"], dn.weights)
    # Eigenvectors may change sign after CSV round-tripping; reconstruction is invariant.
    np.testing.assert_allclose(
        scores[pcs].to_numpy() @ loadings[pcs].to_numpy().T, pca.x @ pca.rotation.T, atol=1e-12
    )
    np.testing.assert_allclose(variance["sdev"], pca.sdev)
    np.testing.assert_allclose(variance["variance_ratio"], pca.sdev**2 / (pca.sdev**2).sum())


def test_refinement_keeps_a_mapping_for_every_original_row(source, tmp_path, quakes):
    out = tmp_path / "refined"
    cli.main(_args(source, out) + ["--refine", "--max-splits", "1"])
    nuggets, assignments, _ = _tables(out)
    _assert_mapping(nuggets, assignments, quakes[FEATURES])


def test_plots_live_in_output_subfolder(source, tmp_path):
    out = tmp_path / "figures"
    cli.main(_args(source, out, analysis=("kmeans", "hclust", "pca"), plots=True))
    expected = {
        "nuggets.png",
        "kmeans_clusters.png",
        "hclust_clusters.png",
        "pca.png",
        "dendrogram.png",
    }
    assert {path.name for path in (out / "plots").iterdir()} == expected
    assert not list(out.glob("*.png"))
    for name in expected:
        image = (out / "plots" / name).read_bytes()
        assert image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) > 1000


def test_no_plots_does_not_import_matplotlib(source, tmp_path):
    script = """
import builtins
import sys
original_import = builtins.__import__
def without_matplotlib(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise AssertionError('matplotlib imported with --no-plots')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_matplotlib
from datanuggets.cli import main
main(sys.argv[1:])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *_args(source, tmp_path / "headless")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_module_entrypoint_help_works_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "datanuggets", "analyze", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for flag in ("--columns", "--transform", "--analysis", "--out"):
        assert flag in result.stdout


@pytest.mark.parametrize("kind", ["file", "empty-directory", "directory", "input-file"])
def test_existing_output_is_never_modified(source, tmp_path, capsys, kind):
    out = source if kind == "input-file" else tmp_path / "existing"
    if kind == "file":
        out.write_bytes(b"an unrelated file")
    elif kind in ("empty-directory", "directory"):
        out.mkdir()
        if kind == "directory":
            (out / "nuggets.csv").write_bytes(b"previous results")
            (out / "notes.txt").write_bytes(b"keep these notes")
    before = out.read_bytes() if out.is_file() else {p.name: p.read_bytes() for p in out.iterdir()}
    _assert_failed(_args(source, out), capsys)
    after = out.read_bytes() if out.is_file() else {p.name: p.read_bytes() for p in out.iterdir()}
    assert after == before


@pytest.mark.parametrize("dangling", [False, True])
def test_symlink_output_is_refused(source, tmp_path, capsys, dangling):
    target, out = tmp_path / "target", tmp_path / "linked-output"
    if not dangling:
        target.mkdir()
        (target / "notes.txt").write_bytes(b"do not touch")
    try:
        out.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")
    _assert_failed(_args(source, out), capsys)
    assert out.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert {p.name: p.read_bytes() for p in target.iterdir()} == {"notes.txt": b"do not touch"}


def test_output_claimed_during_validation_is_preserved(source, tmp_path, monkeypatch, capsys):
    out = tmp_path / "claimed"
    original_mkdir = Path.mkdir

    def claim_before_creation(path, *args, **kwargs):
        if path == out:
            original_mkdir(path)
            (path / "other-run.csv").write_bytes(b"another run claimed this folder")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", claim_before_creation)
    _assert_failed(_args(source, out), capsys)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {
        "other-run.csv": b"another run claimed this folder"
    }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ValueError("analysis could not finish"), "analysis could not finish"),
        (OSError("disk unavailable"), "disk unavailable"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_failed_run_keeps_partial_outputs_and_marks_them(
    source, tmp_path, monkeypatch, capsys, failure, message
):
    out = tmp_path / "failed"
    previous = tmp_path / "previous.csv"
    previous.write_bytes(b"previous results")

    def fail(*args, **kwargs):
        (out / "partial.csv").write_bytes(b"partial result")
        raise failure

    monkeypatch.setattr(cli, "create_dn", fail)
    assert message in _assert_failed(_args(source, out), capsys)
    assert (out / "INCOMPLETE.txt").is_file()
    assert (out / "partial.csv").read_bytes() == b"partial result"
    assert previous.read_bytes() == b"previous results"
    _assert_failed(_args(source, out), capsys)
    assert (out / "partial.csv").read_bytes() == b"partial result"


def test_default_runs_use_distinct_folders_under_outputs(source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        cli.main(_args(source))
    runs = list((tmp_path / "outputs").iterdir())
    assert len(runs) == 2
    assert all((run / "run.json").is_file() for run in runs)
    assert all(not (run / "INCOMPLETE.txt").exists() for run in runs)
    assert sorted(path.name for path in tmp_path.iterdir()) == [source.name, "outputs"]


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate-header",
        "empty-header",
        "short-row",
        "long-row",
        "missing-value",
        "nonnumeric",
        "nan",
        "inf",
        "one-row",
        "invalid-utf8",
        "missing-column",
        "duplicate-selection",
    ],
)
def test_bad_csv_is_rejected_without_silently_dropping_rows(source, tmp_path, capsys, problem):
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    selected = FEATURES
    if problem == "duplicate-header":
        rows[0][4] = rows[0][3]
    elif problem == "empty-header":
        rows[0][0] = ""
    elif problem == "short-row":
        rows[2].pop()
    elif problem == "long-row":
        rows[2].append("extra")
    elif problem in ("missing-value", "nonnumeric", "nan", "inf"):
        rows[2][3] = {"missing-value": "", "nonnumeric": "oops", "nan": "NaN", "inf": "inf"}[
            problem
        ]
    elif problem == "one-row":
        rows = rows[:2]
    elif problem == "missing-column":
        selected = ["depth", "absent"]
    elif problem == "duplicate-selection":
        selected = ["depth", "depth"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    if problem == "invalid-utf8":
        source.write_bytes(source.read_bytes().replace(b"quake-0", b"quake-\xff"))
    before = source.read_bytes()
    out = tmp_path / "bad-input"
    _assert_failed(_args(source, out, columns=selected), capsys)
    assert source.read_bytes() == before
    if (out / "run.json").exists():
        assert (out / "INCOMPLETE.txt").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--nuggets", "0"],
        ["--intermediate-nuggets", "10"],
        ["--clusters", "0"],
        ["--seed", "-1"],
        ["--jobs", "0"],
        ["--cofactor", "0"],
        ["--cofactor", "nan"],
        ["--cofactor", "inf"],
        ["--transform", "none", "standardize"],
        ["--analysis", "none", "pca"],
        ["--max-splits", "-1"],
    ],
)
def test_invalid_options_fail_concisely(source, tmp_path, capsys, extra):
    _assert_failed(_args(source, tmp_path / "invalid") + extra, capsys)


def test_log_transform_rejects_negative_data(source, tmp_path, capsys):
    _assert_failed(
        _args(source, tmp_path / "negative", columns=["latitude", "depth"])
        + ["--transform", "log1p"],
        capsys,
    )


def test_analysis_requires_multiple_features(source, tmp_path, capsys):
    _assert_failed(_args(source, tmp_path / "single", columns=["depth"], analysis=("pca",)), capsys)


def test_single_feature_nugget_creation_is_supported(source, tmp_path, quakes):
    out = tmp_path / "single"
    cli.main(_args(source, out, columns=["depth"]))
    nuggets, assignments, _ = _tables(out)
    _assert_mapping(nuggets, assignments, quakes[["depth"]])
