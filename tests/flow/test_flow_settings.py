"""The flow pipeline's settings. The constants fixed for parity keep the values of Cabrera's
cfc.py, the tunable defaults live once, in _defaults, so a signature can't drift from them, and the
settings classes refuse a bad value with the message the CLI prints, the setting named as its flag
there, a flag every setting has, and by its name in the library."""

import dataclasses
import inspect
import json
import math

import pytest

from datanuggets import _defaults, flow
from datanuggets.exceptions import FlowError
from datanuggets.flow import gate, grid, peaks, pick
from datanuggets.flow.settings import ClusterSettings, GateSettings, cli_flag

# each constant fixed for parity with cfc.py, at the value its code has
FIXED = {
    gate: {"MSIZE": 100, "GRID_SCALE": "l", "GRID_ALPHA": 0.1, "MIN_POINTS": 3},
    grid: {"SQRT_SHIFT": 3},
    peaks: {
        "LOCAL_MAX_QUANTILE": 0.85,
        "LOCAL_MAX_EDGES": "reflect",
        "PEAK_QUANTILE": 0.9,
        "ROW_MIN": 10,
        "ROW_MAX": 90,
        "COL_MAX": 90,
        "SEARCH_THRESHOLD": 2,
        "AREA_T1": 35,
        "AREA_T2": 0.85,
    },
}
# the finder's tunable defaults, as cfc.py calls it, and the functions that take each
FINDER = {"n_row": 11, "n_col": 11, "t11": 22, "t21": 0.25, "t12": 20, "t22": 0.5}
TAKES = {
    "lymphmacro_finder": tuple(FINDER),
    "finder_peaks": ("n_row", "n_col"),
    "gate_peaks": tuple(FINDER),
    "local_maxima": ("n_row", "n_col"),
}
# datanuggets-flow's defaults, as it has always run
GATE = FINDER | {"fsc": "FSC-A", "ssc": "SSC-A", "disc": "his", "t11": (22,), "t21": (0.25,)}
CLUSTER = {
    "channels": None,
    "exclude_channels": (),
    "cofactor": 6000,
    "dn_num1": 10_000,
    "dn_num2": 1000,
    "seed": 291102,
    "n_jobs": -1,
    "k": 8,
    "tsne": True,
    "tsne_perplexity": 30,
    "tsne_seed": 0,
}
TSNE_ARGS = {"n_components": 2, "init": "pca", "learning_rate": "auto", "max_iter": 1000}
# settings with no default left, for the JSON round trip
CHANGED = {
    "gate": GateSettings(
        fsc="FSC-H",
        ssc="SSC-H",
        disc="peak",
        t11=(22.0, 20.0, 18.0),
        t21=(0.25, 0.5, 0.5),
        t12=19.5,
        t22=0.4,
        n_row=9,
        n_col=13,
    ),
    "cluster": ClusterSettings(
        channels=("CD3-A", "CD4-A"),
        cofactor=150.0,
        dn_num1=5000,
        dn_num2=500,
        seed=7,
        n_jobs=2,
        k=5,
        tsne=False,
        tsne_perplexity=12.5,
        tsne_seed=3,
    ),
    "exclude": ClusterSettings(exclude_channels=("Live-A",)),
}
DISTANCE = "it takes a positive distance in grid cells"
FRACTION = "it takes a fraction of the peak's density, 0 or more"
CELLS = "it takes 1 to 100 grid cells"
COFACTOR = "it takes 0 (linear) or a positive number"
WHOLE = "it takes a whole number from 0"
JOBS = "it takes -1 (every core) or 1 or more"
TWO = "choose at least two distinct fluorescence channels for clustering and PCA"
# each rule check() keeps, as the settings, the setting named and the message; the rule on how
# many values t11 and t21 give is test_gate_settings_spread_per_file_thresholds'
BAD = [
    (GateSettings(disc="hist"), "disc", "disc is 'hist'; it takes his or peak"),
    (GateSettings(t11=(22, 0)), "t11", f"t11 is 0; {DISTANCE}"),
    (GateSettings(t12=math.nan), "t12", f"t12 is nan; {DISTANCE}"),
    (GateSettings(t21=(-0.5,)), "t21", f"t21 is -0.5; {FRACTION}"),
    (GateSettings(t22=math.inf), "t22", f"t22 is inf; {FRACTION}"),
    (GateSettings(t11=()), "t11", "t11 gives no values; give one, or one per file"),
    (GateSettings(n_row=0), "n_row", f"n_row is 0; {CELLS}"),
    (GateSettings(n_col=101), "n_col", f"n_col is 101; {CELLS}"),
    (GateSettings(n_col=5.5), "n_col", f"n_col is 5.5; {CELLS}"),
    (ClusterSettings(k=0), "k", "k is 0; it takes 1 or more clusters"),
    (ClusterSettings(dn_num1=2), "dn_num1", "dn_num1 is 2; it takes 3 or more nuggets"),
    (ClusterSettings(dn_num2=1), "dn_num2", "dn_num2 is 1; it takes 2 to 9999 nuggets per file"),
    (
        ClusterSettings(dn_num1=5, dn_num2=5),
        "dn_num2",
        "dn_num2 is 5; it takes 2 to 4 nuggets per file",
    ),
    (ClusterSettings(cofactor=-1.0), "cofactor", f"cofactor is -1.0; {COFACTOR}"),
    (ClusterSettings(cofactor=math.nan), "cofactor", f"cofactor is nan; {COFACTOR}"),
    (ClusterSettings(seed=-1), "seed", f"seed is -1; {WHOLE}"),
    (ClusterSettings(seed=True), "seed", f"seed is True; {WHOLE}"),
    (ClusterSettings(tsne_seed=0.5), "tsne_seed", f"tsne_seed is 0.5; {WHOLE}"),
    (ClusterSettings(n_jobs=0), "n_jobs", f"n_jobs is 0; {JOBS}"),
    (ClusterSettings(n_jobs=-2), "n_jobs", f"n_jobs is -2; {JOBS}"),
    (ClusterSettings(channels=("CD3-A", "CD3-A")), "channels", TWO),
    (
        ClusterSettings(channels=("CD3-A", "CD4-A", "CD3-A")),
        "channels",
        "channels repeats a channel; list each channel only once",
    ),
    (
        ClusterSettings(channels=("CD3-A", "CD4-A"), exclude_channels=("Live-A",)),
        "exclude_channels",
        "exclude_channels can't be combined with channels",
    ),
    (
        ClusterSettings(tsne_perplexity=0),
        "tsne_perplexity",
        "tsne_perplexity is 0; it takes a positive number",
    ),
]


def _default(f, name):
    return inspect.signature(f).parameters[name].default


def test_parity_constants_keep_cabreras_values():
    for module, constants in FIXED.items():
        got = {name: getattr(module, name, None) for name in constants}
        assert got == constants, module.__name__
    assert _default(flow.f_plot, "alpha") == gate.GRID_ALPHA
    assert _default(flow.local_maxima, "quantile") == peaks.LOCAL_MAX_QUANTILE
    for f in (flow.search_ind, flow.refine_peaks):
        assert _default(f, "threshold") == peaks.SEARCH_THRESHOLD, f.__name__
    for f in (flow.find_area, flow.mirrored_area):
        assert (_default(f, "t1"), _default(f, "t2")) == (peaks.AREA_T1, peaks.AREA_T2), f.__name__


@pytest.mark.parametrize("name", TAKES)
def test_finder_signature_defaults_are_the_central_defaults(name):
    central = {k: getattr(_defaults, f"FINDER_{k.upper()}", None) for k in FINDER}
    assert central == FINDER
    got = {k: _default(getattr(flow, name), k) for k in TAKES[name]}
    assert got == {k: central[k] for k in TAKES[name]}


def test_settings_defaults_are_the_cli_defaults():
    for s, defaults in ((GateSettings(), GATE), (ClusterSettings(), CLUSTER)):
        assert dataclasses.asdict(s) == defaults
        assert s.check() is s
    assert _defaults.TSNE_ARGS == TSNE_ARGS
    with pytest.raises(TypeError):  # read only: one caller can't change them for every later run
        _defaults.TSNE_ARGS["max_iter"] = 250


@pytest.mark.parametrize("name", CHANGED)
def test_settings_round_trip_through_json(name):
    s = CHANGED[name]
    data = s.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data  # lists, as JSON gives them back
    assert type(s).from_dict(json.loads(json.dumps(data))) == s


@pytest.mark.parametrize(
    ("cls", "key", "kind"), [(GateSettings, "t13", "gate"), (ClusterSettings, "disc", "cluster")]
)
def test_from_dict_rejects_unknown_keys(cls, key, kind):
    with pytest.raises(FlowError) as e:
        cls.from_dict(cls().to_dict() | {key: 1})
    assert (e.value.param, str(e.value)) == (key, f"{key} isn't a {kind} setting")


@pytest.mark.parametrize(
    ("s", "param", "message"), BAD, ids=[f"{p}-{i}" for i, (_, p, _) in enumerate(BAD)]
)
def test_check_rejects_bad_values(s, param, message):
    with pytest.raises(FlowError) as e:
        s.check()
    assert (e.value.param, str(e.value)) == (param, message)


def test_flow_error_renders_settings_as_flags():
    with pytest.raises(FlowError) as e:
        ClusterSettings(n_jobs=0).check()
    assert e.value.cli(cli_flag) == "--jobs is 0; it takes -1 (every core) or 1 or more"
    assert str(e.value) == "n_jobs is 0; it takes -1 (every core) or 1 or more"
    assert isinstance(e.value, ValueError)
    # user text is put in before raising, and its braces stay as typed
    typed = FlowError("{param} names {x}, which a.fcs doesn't have", "exclude_channels")
    assert typed.cli(cli_flag) == "--exclude-channels names {x}, which a.fcs doesn't have"
    assert str(FlowError("no file has cells kept")) == "no file has cells kept"
    flags = [cli_flag(n) for n in ("dn_num2", "tsne_seed", "tsne", "from_gates")]
    assert flags == ["--dn-num2", "--tsne-seed", "--no-tsne", "--from-gates"]


def test_every_setting_names_a_real_flag():
    """cli_flag of every setting is a datanuggets-flow flag, so an error the CLI shows names a
    flag that can be typed."""
    flags = pick._parser()._option_string_actions
    names = [f.name for cls in (GateSettings, ClusterSettings) for f in dataclasses.fields(cls)]
    assert [name for name in names if cli_flag(name) not in flags] == []


def test_gate_settings_spread_per_file_thresholds():
    s = GateSettings(t11=(22, 20, 18)).check(3)
    assert s.for_file(1) == {
        "n_row": 11,
        "n_col": 11,
        "t11": 20,
        "t21": 0.25,
        "t12": 20,
        "t22": 0.5,
        "area": flow.mirrored_area,
    }
    inspect.signature(flow.gate_peaks).bind(None, None, **s.for_file(2))  # gate_peaks takes them
    with pytest.raises(FlowError) as e:
        s.check(2)
    assert (e.value.param, str(e.value)) == (
        "t11",
        "t11 gives 3 values for 2 files; give one, or one per file",
    )
    with pytest.raises(FlowError) as e:
        GateSettings(t21=(0.25, 0.5)).check(3)
    assert str(e.value) == "t21 gives 2 values for 3 files; give one, or one per file"
    assert GateSettings(disc="peak").for_file(0)["area"] is flow.find_area
