"""The benchmark's FCS reading: FSC-A and SSC-A as FlowKit's raw events hold them, and the files
of his study layout."""

from pathlib import Path

import flowio
import numpy as np
import pandas as pd


def read_fsc_ssc(path) -> tuple[np.ndarray, np.ndarray]:
    """FSC-A and SSC-A as float64. FlowKit's Sample.get_events(source="raw") is flowio's
    as_array(preprocess=True), which the mini showed equal on three of these files."""
    fd = flowio.FlowData(str(path))
    events = fd.as_array(preprocess=True)
    names = [fd.text[f"p{k}n"] for k in range(1, fd.channel_count + 1)]
    return events[:, names.index("FSC-A")], events[:, names.index("SSC-A")]


def fcs_files(target) -> list[Path]:
    """One FCS file, or every *.fcs in a folder, sorted by name."""
    p = Path(target).expanduser()
    files = sorted(p.glob("*.fcs")) if p.is_dir() else [p]
    if not files or not all(f.is_file() for f in files):
        raise FileNotFoundError(f"no FCS file at {p}")
    return files


def study_files(metadata, folder) -> list[Path]:
    """The files his script reads: the Filename column of the workbook's second sheet."""
    dt_meta = pd.read_excel(Path(metadata).expanduser(), sheet_name=1)
    return [Path(folder).expanduser() / f for f in dt_meta["Filename"]]
