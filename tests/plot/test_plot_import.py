"""matplotlib stays optional: the core never imports it, and the plot package says what to
install when it's missing."""

import subprocess
import sys

CODE = """
import sys
sys.modules["matplotlib"] = None
import datanuggets
assert "matplotlib.pyplot" not in sys.modules
try:
    import datanuggets.plot
except ImportError as e:
    print(e)
"""


def test_core_works_without_matplotlib_and_plot_names_the_extra():
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, check=True)
    assert (
        out.stdout.strip() == "datanuggets.plot needs matplotlib: pip install 'datanuggets[plot]'"
    )


def test_import_datanuggets_does_not_load_matplotlib():
    code = "import sys, datanuggets; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
