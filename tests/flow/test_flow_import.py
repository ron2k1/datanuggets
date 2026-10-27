"""datanuggets.flow needs numpy and scipy only, import datanuggets doesn't load it, and its library
modules import without the flow extra's packages, which load only where a file is read or drawn."""

import subprocess
import sys

CODE = """
import sys
sys.modules["matplotlib"] = None
import numpy as np
import datanuggets
assert "datanuggets.flow" not in sys.modules
from datanuggets.flow import lymphmacro_finder
rng = np.random.default_rng(0)
fsc = np.concatenate([rng.normal(4e5, 3e4, 4000), rng.normal(7e5, 5e4, 2000)])
ssc = np.concatenate([rng.normal(2e5, 2e4, 4000), rng.normal(5e5, 6e4, 2000)])
print(lymphmacro_finder(fsc, ssc).shape)
"""
# the flow library's modules, which a backend imports without the CLI
LIBRARY = ("datanuggets.flow.settings", "datanuggets.flow.pipeline")
EXTRAS = ("matplotlib", "pandas", "sklearn", "flowio")


def test_flow_runs_without_matplotlib_and_stays_out_of_the_core_import():
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, check=True)
    assert out.stdout.strip().startswith("(6000, ")


def test_flow_library_modules_import_without_optional_extras():
    code = (
        "import importlib, sys\n"
        f"sys.modules.update(dict.fromkeys({EXTRAS!r}))\n"
        f"for name in {LIBRARY!r}:\n"
        "    importlib.import_module(name)\n"
        "assert 'datanuggets.flow.pick' not in sys.modules\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
