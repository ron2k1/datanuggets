"""The kernels compile with or without a place on disk for numba's cache."""

import importlib
import pkgutil
import subprocess
import sys

import numpy as np
from numba.core.caching import FunctionCache
from numba.core.registry import CPUDispatcher

import datanuggets
from datanuggets._compiled import frozen
from datanuggets._rcompat import r_mean
from datanuggets._rstats import group_moments, r_cov
from rfixtures import fixture_data

# every module of compiled kernels, so one added later is checked without editing this file
JIT_MODULES = sorted(
    f"datanuggets.{m.name}"
    for m in pkgutil.iter_modules(datanuggets.__path__)
    if m.name.endswith("jit")
)

# numba finds no cache locator in a read-only install whose user has no writable home, which an
# empty locator list stands in for. Importing a module decorates its kernels, which is where a
# kernel that insists on a cache fails.
NO_CACHE_CHILD = """
import importlib
import sys
import numba.core.caching
import numpy as np

numba.core.caching.CacheImpl._locator_classes = []
for name in sys.argv[2:]:
    importlib.import_module(name)
from datanuggets._rcompat import r_mean
from datanuggets._rstats import group_moments, r_cov

x = np.load(sys.argv[1])
((center, cov),) = group_moments(x, [np.arange(x.shape[0])], 2)
print(r_mean(x, 0).tobytes().hex(), r_cov(x).tobytes().hex(), center.tobytes().hex(), cov.tobytes().hex())
"""


def test_frozen_is_a_read_only_view_of_the_same_memory():
    a = np.arange(6.0).reshape(2, 3)
    view = frozen(a)
    assert not view.flags.writeable and a.flags.writeable
    assert np.shares_memory(view, a) and view.dtype == a.dtype and view.shape == a.shape


def test_jit_modules_are_found():
    assert "datanuggets._ldjit" in JIT_MODULES


def test_every_kernel_caches_when_numba_can():
    for name in JIT_MODULES:
        kernels = [
            v for v in vars(importlib.import_module(name)).values() if isinstance(v, CPUDispatcher)
        ]
        assert kernels, name
        assert all(isinstance(k._cache, FunctionCache) for k in kernels), name


def test_kernels_compile_where_numba_has_nowhere_to_cache(tmp_path):
    data = fixture_data()
    x = max((data[k][0] for k in data), key=lambda a: a.size)
    np.save(tmp_path / "x.npy", x)
    run = subprocess.run(
        [sys.executable, "-c", NO_CACHE_CHILD, str(tmp_path / "x.npy"), *JIT_MODULES],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    ((center, cov),) = group_moments(x, [np.arange(x.shape[0])], 2)
    want = [r_mean(x, 0), r_cov(x), center, cov]
    assert run.stdout.split() == [a.tobytes().hex() for a in want]
