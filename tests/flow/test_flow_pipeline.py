"""The flow run's library steps, which a backend calls without the CLI: gating, the population
choice, the kept cells read file by file into nuggets, their clusters and t-SNE, the tables and
figures a run writes, and the events that carry the lines the CLI prints. None of them prints,
prompts or leaves a pyplot figure open."""

import builtins
import functools
import sys
from pathlib import Path

import numpy as np
import pytest

from datanuggets import create_dn, cut_tree, whclust, wmean
from datanuggets._defaults import TSNE_ARGS
from datanuggets.exceptions import FlowError
from datanuggets.flow import finder_peaks
from datanuggets.flow.pipeline import (
    Kept,
    Pool,
    PopStats,
    cluster_nuggets,
    cluster_stage,
    gate_files,
    gate_stage,
    make_nuggets,
    parse_selection,
    pops_spec,
    population_stats,
    population_table,
    resolve_channels,
    selection_choices,
    tsne_embedding,
)
from datanuggets.flow.settings import ClusterSettings, GateSettings, cli_flag
from flowcases import COLUMNS, DIGESTS, FIGURES, FLOW, LABELS, digest, load_case, read_202

PNN = ["FSC-A", "SSC-A", "CD3-A", "CD4-A"]  # the synthetic files' channels
RESULT_FIGURES = sorted(set(FIGURES) - {"local_maxima.png", "gates.png"})  # cluster_stage's


def _pool(seed=0):
    """A pool as the clustering step gives it: 40 nuggets, the weight-0 ones out of use, cut
    into 3 clusters."""
    rng = np.random.default_rng(seed)
    centers, weights = rng.normal(size=(40, 3)), rng.integers(0, 20, 40).astype(float)
    use = weights > 0
    tree = whclust(centers[use], weights[use])
    cl = np.full(len(centers), -1)
    cl[use] = cut_tree(tree, k=3)
    return Pool(centers, weights, use, tree, cl, 3)


def _files(kept, n=60):
    """Gated files a.fcs, b.fcs, ... of n events each, Pop 1 of file i its first kept[i] events
    and Pop 2 the ten after them, and a reader of their events that records each name it reads."""
    rng = np.random.default_rng(0)
    events, samples, seen = {}, [], []
    for i, k in enumerate(kept):
        name = f"{'abcd'[i]}.fcs"
        events[name] = rng.normal(size=(n, len(PNN)))
        masks = np.zeros((n, 2), dtype=bool)
        masks[:k, 0] = masks[k : k + 10, 1] = True
        samples.append({"path": Path(name), "n_events": n, "pnn": PNN, "masks": masks})

    def read(path):
        seen.append(path.name)
        return events[path.name], list(PNN), [""] * len(PNN)

    return samples, read, seen


def _gated_202():
    """202.fcs gated as datanuggets-flow gates it by default."""
    return gate_files([Path("202.fcs")], GateSettings(), read=read_202)


def test_cluster_nuggets_needs_three_occupied_nuggets():
    with np.load(FLOW / "202_fluor.npz") as data:
        x = data["fluor"][:2, :2].astype(float)
    kept = [Kept(1, "202.fcs", np.arange(len(x)), create_dn(x, dn_num2=2))]
    with pytest.raises(FlowError, match="fewer than 3 occupied nuggets") as e:
        cluster_nuggets(kept, k=2)
    assert e.value.param == "dn_num2"
    assert "increase dn_num2" in str(e.value)
    assert "increase --dn-num2" in e.value.cli(cli_flag)


def test_make_nuggets_reads_each_kept_file_once_in_order():
    """b.fcs keeps nothing and d.fcs one event, too few for nuggets: neither is read."""
    samples, read, seen = _files([20, 20, 20, 1])
    settings = ClusterSettings(dn_num2=5, n_jobs=1)
    kept = make_nuggets(samples, [[1], [], [1, 2], [1]], PNN[2:], settings, read=read)
    assert seen == ["a.fcs", "c.fcs"]
    assert [(k.number, k.name, len(k.rows)) for k in kept] == [(1, "a.fcs", 20), (3, "c.fcs", 30)]


def test_nothing_kept_raises_before_any_read():
    samples, read, seen = _files([20, 1])
    with pytest.raises(FlowError) as e:
        make_nuggets(samples, [[], [1]], PNN[2:], ClusterSettings(), read=read)
    assert str(e.value) == "no file has cells kept, so there is nothing to cluster"
    assert seen == []


def test_make_nuggets_matches_create_dn_file_by_file():
    events, pnn, _ = read_202(None)
    masks = load_case("202").his_masks
    sample = {"path": Path("202.fcs"), "n_events": len(events), "pnn": pnn, "masks": masks}
    settings = ClusterSettings(dn_num2=50, n_jobs=1)
    (kept,) = make_nuggets([sample], [[6]], pnn[2:], settings, read=read_202)
    rows = np.flatnonzero(masks[:, 5])
    x = np.arcsinh(events[np.ix_(rows, [2, 3, 4])] / settings.cofactor)
    want = create_dn(x, dn_num2=50, n_jobs=1)
    assert (kept.number, kept.name) == (1, "202.fcs")
    assert np.array_equal(kept.rows, rows)
    for field in ("centers", "weights", "scales", "assignments"):
        assert np.array_equal(getattr(kept.dn, field), getattr(want, field)), field


@pytest.mark.parametrize("change", ["rows", "channels"])
def test_make_nuggets_refuses_a_file_that_changed(change):
    samples, read, _ = _files([20])

    def changed(path):
        events, pnn, pns = read(path)
        return (events[:-1], pnn, pns) if change == "rows" else (events, pnn[::-1], pns)

    with pytest.raises(FlowError, match="a.fcs isn't the file that was gated"):
        make_nuggets(samples, [[1]], PNN[2:], ClusterSettings(), read=changed)


def test_population_table_keeps_the_cli_format():
    fsc, ssc = np.array([100.0, 300.0, 5e5, 7e5]), np.array([10.0, 30.0, 2e5, 4e5])
    masks = np.array([[1, 0], [1, 0], [0, 0], [0, 0]], dtype=bool)  # Pop 2 is empty
    stats = population_stats(fsc, ssc, masks, np.array([[3, 4], [5, 6]]))
    assert stats == [
        PopStats(1, 2, 50.0, 200.0, 20.0, (3, 4)),
        PopStats(2, 0, 0.0, 0.0, 0.0, (5, 6)),
    ]
    rows = [PopStats(1, 12345, 24.69, 412345.6, 98765.4, (3, 4)), stats[1]]
    assert population_table(rows, "FSC-A", "SSC-A") == [
        "  Pop 1:    12,345 events (24.7%), median FSC-A   412,346, SSC-A    98,765",
        "  Pop 2:         0 events ( 0.0%), median FSC-A         0, SSC-A         0",
    ]


def test_selection_choices_gives_one_for_all_or_one_per_file():
    """The rule on how many choices --pops gives, which the CLI checks before any file is read
    and parse_selection once the populations are counted."""
    assert selection_choices("1,2", 3) == ["1,2"]
    assert selection_choices("1;none;1,2", 3) == ["1", "none", "1,2"]
    with pytest.raises(FlowError) as e:
        selection_choices("1;2", 3)
    assert e.value.param == "pops"
    assert e.value.cli(cli_flag) == (
        "--pops gives 2 choices for 3 files; give one for all of them or one per file, "
        "separated by ;"
    )


def test_parse_selection_one_for_all_or_one_per_file():
    assert parse_selection("1,2", [3, 3, 3]) == [[1, 2]] * 3
    assert parse_selection("1;none;1,2", [2, 2, 2]) == [[1], [], [1, 2]]
    with pytest.raises(FlowError, match="pops gives 2 choices for 3 files"):
        parse_selection("1;2", [2, 2, 2])
    with pytest.raises(FlowError) as e:
        parse_selection("5", [4])
    assert str(e.value) == "pops: there is no Pop 5; this file has Pop 1 to Pop 4"


def test_pops_spec_round_trips():
    chosen = [[1, 2], [], [1]]
    assert pops_spec(chosen) == "1,2;none;1"
    assert parse_selection(pops_spec(chosen), [2, 2, 2]) == chosen


def test_resolve_channels_default_rule_and_errors():
    pnn = ["FSC-A", "SSC-A", "FSC-H", "CD3-A", "CD4-A", "Time"]
    a = {"path": Path("a.fcs"), "pnn": pnn, "pns": ["", "", "", "CD3", "", ""]}
    b = {"path": Path("b.fcs"), "pnn": pnn[:4], "pns": [""] * 4}
    assert resolve_channels([a], ClusterSettings()) == (
        ["CD3-A", "CD4-A"],
        ["CD3 (CD3-A)", "CD4-A"],
    )
    with pytest.raises(FlowError) as e:
        resolve_channels([a, b], ClusterSettings())
    assert str(e.value) == "b.fcs has no channel CD4-A; its channels: FSC-A, SSC-A, FSC-H, CD3-A"
    # check_channels' rule is tested on given lists through ClusterSettings.check; the list the
    # default rule gives is known only here
    one = {"path": Path("one.fcs"), "pnn": ["FSC-A", "SSC-A", "CD3-A"], "pns": [""] * 3}
    with pytest.raises(FlowError, match="at least two distinct fluorescence channels") as e:
        resolve_channels([one], ClusterSettings())
    assert e.value.param == "channels"


def test_resolve_channels_excludes_and_refuses_unknown_names():
    pnn = ["FSC-A", "SSC-A", "CD3-A", "CD4-A", "Live-A"]
    a = {"path": Path("a.fcs"), "pnn": pnn, "pns": [""] * 5}
    assert resolve_channels([a], ClusterSettings(exclude_channels=("Live-A",)))[0] == pnn[2:4]
    with pytest.raises(FlowError) as e:
        resolve_channels([a], ClusterSettings(exclude_channels=("CD9-A",)))
    assert e.value.param == "exclude_channels"
    assert e.value.cli(cli_flag) == (
        "--exclude-channels names CD9-A, which a.fcs doesn't have; its channels: "
        "FSC-A, SSC-A, CD3-A, CD4-A, Live-A"
    )
    with pytest.raises(FlowError, match="at least two distinct"):
        resolve_channels([a], ClusterSettings(exclude_channels=("Live-A", "CD4-A")))


@pytest.mark.parametrize(
    ("names", "settings", "param", "message"),
    [
        (
            ["a.fcs", "b.fcs", "c.fcs"],
            GateSettings(t11=(22, 20)),
            "t11",
            "t11 gives 2 values for 3 files; give one, or one per file",
        ),
        (
            ["a.fcs"],
            GateSettings(t11=(-5,)),
            "t11",
            "t11 is -5; it takes a positive distance in grid cells",
        ),
        (["a.fcs"], GateSettings(disc="bogus"), "disc", "disc is 'bogus'; it takes his or peak"),
        ([], GateSettings(), None, "no files to gate"),
    ],
)
def test_gate_stage_checks_its_settings_and_files_before_any_read(
    tmp_path, names, settings, param, message
):
    """A backend calls the stages without the CLI's checks, so a bad setting or no files at all
    stops gate_files, and gate_stage through it, with a FlowError before a file is read."""
    seen = []

    def read(path):
        seen.append(path.name)
        return read_202(path)

    paths = [Path(name) for name in names]
    for stage in (
        lambda: gate_files(paths, settings, read=read),
        lambda: gate_stage(paths, tmp_path, settings, read=read),
    ):
        with pytest.raises(FlowError) as e:
            stage()
        assert (e.value.param, str(e.value)) == (param, message)
    assert seen == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("kept", "chosen", "settings", "param", "message"),
    [
        ([20], [[1]], ClusterSettings(k=0), "k", "k is 0; it takes 1 or more clusters"),
        ([], [], ClusterSettings(), None, "no gated files to cluster"),
        (
            [20, 20, 20],
            [[1], [1]],
            ClusterSettings(),
            "pops",
            "pops gives 2 choices for 3 files; give one per file",
        ),
    ],
)
def test_cluster_stage_checks_its_settings_and_choices_before_any_read(
    tmp_path, kept, chosen, settings, param, message
):
    """A bad setting, no samples, or a choice list that doesn't give one choice per sample (which
    the tables would zip short) stops cluster_stage before it reads a file or writes one."""
    samples, read, seen = _files(kept)
    with pytest.raises(FlowError) as e:
        cluster_stage(samples, chosen, tmp_path, settings, read=read)
    assert (e.value.param, str(e.value)) == (param, message)
    assert seen == []
    assert list(tmp_path.iterdir()) == []


def test_library_stages_never_print_or_prompt(tmp_path, monkeypatch, capsys):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(builtins, "input", lambda *args: pytest.fail("the library asked"))
    plt.close("all")
    settings = ClusterSettings(dn_num2=50, k=3, n_jobs=1)
    cluster_stage(_gated_202(), [[6]], tmp_path, settings, read=read_202)
    assert capsys.readouterr() == ("", "")
    assert plt.get_fignums() == []


def test_cluster_stage_alone_writes_the_one_shot_tables(tmp_path):
    settings = ClusterSettings(dn_num2=50, k=3, n_jobs=1)
    run = cluster_stage(_gated_202(), [[1]], tmp_path, settings, read=read_202)
    assert {name: digest(tmp_path / name) for name in COLUMNS} == DIGESTS
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == sorted([*COLUMNS, *RESULT_FIGURES])
    assert sorted(p.name for p in run.files) == written
    assert run.labels == LABELS
    assert run.tsne.shape == (run.pool.use.sum(), 2)


def test_stage_events_carry_the_cli_lines(tmp_path):
    """The gate event's lines are the ones the CLI printed before the steps moved here, from the
    same f-string; the clustering's events follow in the order the CLI printed them."""
    events = []
    samples = gate_files([Path("202.fcs")], GateSettings(), read=read_202, on_event=events.append)
    case = load_case("202")
    table = [
        f"  Pop {j + 1}: {m.sum():>9,} events ({100 * m.mean():4.1f}%), median FSC-A "
        f"{np.median(case.fsc[m]) if m.any() else 0:>9,.0f}, SSC-A "
        f"{np.median(case.ssc[m]) if m.any() else 0:>9,.0f}"
        for j, m in enumerate(case.his_masks.T)
    ]
    fp = finder_peaks(case.fsc, case.ssc)
    (gate,) = events
    assert gate[:3] == ("gate", 1, 1)
    assert gate.message == "\n".join(
        [
            "\nFile 1, 202.fcs: 50,000 events",
            *table,
            f"  {len(fp.candidates)} local maxima, {len(fp.peaks)} kept as peaks",
        ]
    )
    assert gate.data["events"] == 50_000
    assert gate.data["populations"] == samples[0]["stats"]
    settings = ClusterSettings(dn_num2=50, k=3, n_jobs=1, tsne=False)
    run = cluster_stage(samples, [[6]], tmp_path, settings, read=read_202, on_event=events.append)
    assert [(e.stage, e.file) for e in events] == [
        ("gate", 1),
        ("select", None),
        ("nuggets", 1),
        ("cluster", None),
    ]
    assert events[1].message == "\n" + run.selection.to_string(index=False)
    assert events[2].message == f"File 1, 202.fcs: 2,292 cells, {run.kept[0].dn.n_nuggets} nuggets"
    assert events[3].message.startswith(
        "\nPercent of each file's kept cells in each cluster:\n"
        "(columns are file numbers; selection.csv names the files)\nfile_number"
    )


def test_cluster_stage_without_tsne_never_imports_sklearn(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.manifold", None)
    settings = ClusterSettings(dn_num2=50, k=3, n_jobs=1, tsne=False)
    run = cluster_stage(_gated_202(), [[6]], tmp_path, settings, read=read_202)
    assert run.tsne is None
    assert "clusters_tsne.png" not in {p.name for p in tmp_path.iterdir()}


def test_tsne_gets_the_pinned_arguments_and_repeats_for_a_seed(monkeypatch):
    """The arguments that shape the picture are TSNE_ARGS'; perplexity and the seed are the
    settings', the perplexity capped at (n - 1) / 3 for n nuggets."""
    from sklearn.manifold import TSNE

    seen, init = [], TSNE.__init__

    @functools.wraps(init)  # scikit-learn reads the parameters from __init__'s signature
    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        init(self, *args, **kwargs)

    monkeypatch.setattr(TSNE, "__init__", spy)
    x = np.random.default_rng(0).normal(size=(40, 3))
    capped = ClusterSettings(tsne_perplexity=50.0, tsne_seed=7)
    first, again = tsne_embedding(x, capped), tsne_embedding(x, capped)
    tsne_embedding(x, ClusterSettings(tsne_perplexity=5.0))
    assert seen == [
        {**TSNE_ARGS, "perplexity": 13.0, "random_state": 7},
        {**TSNE_ARGS, "perplexity": 13.0, "random_state": 7},
        {**TSNE_ARGS, "perplexity": 5.0, "random_state": 0},
    ]
    assert first.shape == (40, 2)
    assert np.array_equal(first, again)


def test_result_figures_leave_no_pyplot_figures(tmp_path, close_figures):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    from datanuggets.flow.figures import (
        gates_figure,
        local_maxima_figure,
        result_figures,
        save_figures,
    )

    case = load_case("202")
    sample = {
        "path": Path("202.fcs"),
        "fsc": case.fsc,
        "ssc": case.ssc,
        "masks": case.his_masks,
        "peaks": finder_peaks(case.fsc, case.ssc),
    }
    pool = _pool()
    use = pool.use
    centers = pd.DataFrame(
        wmean(pool.centers[use], pool.cl[use], pool.weights[use]), columns=LABELS
    )
    xy = np.random.default_rng(1).normal(size=(use.sum(), 2))
    plt.close("all")
    figs = {
        "local_maxima.png": local_maxima_figure([sample] * 3),  # two rows, the fourth panel hidden
        "gates.png": gates_figure([sample]),
        **result_figures(pool, centers, LABELS, 6000.0, xy),
    }
    assert plt.get_fignums() == []
    assert all(isinstance(fig, Figure) for fig in figs.values())
    assert list(figs) == [
        "local_maxima.png",
        "gates.png",
        "dendrogram.png",
        "clusters_pca.png",
        "clusters_tsne.png",
        "cluster_centers.png",
    ]
    assert sum(not ax.get_visible() for ax in figs["local_maxima.png"].axes) == 1
    no_tsne = result_figures(pool, centers, LABELS, 0, None)
    assert list(no_tsne) == ["dendrogram.png", "clusters_pca.png", "cluster_centers.png"]
    saved = save_figures({"gates.png": figs["gates.png"]}, tmp_path, 110)
    assert saved == [tmp_path / "gates.png"]
    assert saved[0].read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == []
