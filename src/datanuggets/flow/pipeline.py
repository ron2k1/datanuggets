"""The datanuggets-flow run as library steps, which a backend calls without the CLI: nothing here
reads the terminal or prints to it. read_fcs reads a file with flowio, the flow extra's reader,
which loads only when a file is read."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from datanuggets._defaults import FLOW_SPILL_TOL
from datanuggets.exceptions import DataNuggetWarning


def read_fcs(path) -> tuple[np.ndarray, list[str], list[str]]:
    """A file's events, one column per channel, with each channel's $PnN name and $PnS label
    ('' when it has none). The events are flowio's, as stored: a spillover matrix further than
    FLOW_SPILL_TOL from the identity gives a DataNuggetWarning, since nothing compensates them."""
    try:
        import flowio
    except ImportError as e:  # an optional dependency
        raise ImportError('reading FCS files needs flowio: pip install "datanuggets[flow]"') from e
    fd = flowio.FlowData(str(path))
    c = fd.channel_count
    pnn = [fd.text[f"p{k}n"] for k in range(1, c + 1)]
    pns = [fd.text.get(f"p{k}s", "").strip() for k in range(1, c + 1)]
    spill = fd.text.get("spillover") or fd.text.get("spill")
    if spill:
        v = spill.split(",")
        m = int(v[0])
        s = np.asarray(v[1 + m :], dtype=float).reshape(m, m)
        if np.abs(s - np.eye(m)).max() > FLOW_SPILL_TOL:
            warnings.warn(
                f"{Path(path).name} has a compensation matrix, but its events are clustered as "
                "stored, uncompensated",
                DataNuggetWarning,
                stacklevel=2,
            )
    return np.asarray(fd.as_array(preprocess=True), dtype=float), pnn, pns
