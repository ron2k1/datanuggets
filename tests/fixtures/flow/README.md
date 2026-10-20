# Flow fixtures

Recorded by `benchmarks/flow/record.py` on the Mac mini on 2026-10-12, from Prof. Cabrera's own Python (`benchmarks/flow/cabrera_reference.py` and `cabrera_pieces.py`). Never regenerate them to make a test pass.

## Source

[Zenodo record 22079206](https://zenodo.org/records/22079206), "Flow cytometry data of PBMC of critically ill, mechanically ventilated patients", CC-BY-4.0. Three of its 44 FCS 3.1 files from a Cytek Aurora, each thinned to every k-th event and cut at 50,000 events:

| File | k | Populations, his | Populations, disc on the peak |
|---|---|---|---|
| `154.npz` | 6 | 2 | 2 |
| `202.npz` | 5 | 6 | 6 |
| `223.npz` | 1 | 9 | 9 |

## Keys

| Key | dtype | Holds |
|---|---|---|
| `fsc`, `ssc` | float32 | FSC-A and SSC-A as FlowKit's raw events hold them (float32 in the file, so nothing is lost) |
| `z`, `gx`, `gy` | float64 | his `f_plot(...)["z"]`, `["x"]` and `["y"]` for the 100 by 100 grid, log scale |
| `zz` | uint8 | his `["zz"]`, each event's 1-based (SSC bin, FSC bin) |
| `candidates` | int16 | his focal maxima, 1-based (row, col) |
| `peaks` | int16 | his refined, filtered and sorted peaks |
| `his_area` | int16 | his `find_area` cells for each peak as (row, col, population) |
| `fixed_area` | int16 | the same with the distance disc centred on the peak |
| `his_masks`, `fixed_masks` | uint8 | his gate's masks, from his cells and from the fixed cells, `np.packbits` along the events |

Unpack a mask with `np.unpackbits(f["his_masks"], axis=0, count=len(f["fsc"])).astype(bool)`. `tests/flowcases.py` loads all of it.
