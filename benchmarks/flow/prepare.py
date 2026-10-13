"""Put a folder of FCS files in Cabrera's study layout: a workbook whose second sheet has the
Filename column his script reads. With --source, the Zenodo sheet's columns come along.

usage: python benchmarks/flow/prepare.py <folder> [--source metadata_upload.xlsx] [--limit N]
           [--out <folder>/study_metadata.xlsx]
"""

import argparse
from pathlib import Path

import pandas as pd
from fcs import fcs_files

STUDY = "Zenodo 22079206, PBMC of critically ill, mechanically ventilated patients, CC-BY-4.0"


def main() -> None:
    ap = argparse.ArgumentParser(description="Put a folder of FCS files in his study layout.")
    ap.add_argument("folder")
    ap.add_argument("--source", help="the record's metadata sheet, with a file_name column")
    ap.add_argument("--limit", type=int, help="keep the first N files")
    ap.add_argument("--out")
    args = ap.parse_args()
    folder = Path(args.folder).expanduser()
    samples = pd.DataFrame({"Filename": [f.name for f in fcs_files(folder)]})
    if args.source:
        meta = pd.read_excel(Path(args.source).expanduser())
        samples = samples.merge(
            meta.rename(columns={"file_name": "Filename"}), on="Filename", how="left"
        )
    if args.limit:
        samples = samples.head(args.limit)
    out = Path(args.out).expanduser() if args.out else folder / "study_metadata.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        pd.DataFrame({"Study": [STUDY]}).to_excel(xw, sheet_name="Study", index=False)
        samples.to_excel(xw, sheet_name="Samples", index=False)
    print(f"wrote {out}: {len(samples)} files on sheet 2")


if __name__ == "__main__":
    main()
