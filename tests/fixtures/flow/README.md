# Flow fixtures

Recorded by `benchmarks/flow/record.py` on the Mac mini on 2026-10-12, from Prof. Cabrera's own Python (`benchmarks/flow/cabrera_reference.py` and `cabrera_pieces.py`). Never regenerate them to make a test pass.

## Source

Stefanie Kreutmair and Susanne Unger (2026), *Flow cytometry data of PBMC of critically ill, mechanically ventilated patients* (V1), Zenodo, [doi:10.5281/zenodo.22079206](https://doi.org/10.5281/zenodo.22079206). The data and these derived data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); use does not imply endorsement. Three of its 44 FCS 3.1 files from a Cytek Aurora were selected, thinned to every k-th event, cut at 50,000 events, and converted to NPZ with computed reference arrays:

| File | k | Populations, reference code | Populations, disc on the peak |
|---|---|---|---|
| `154.npz` | 6 | 2 | 2 |
| `202.npz` | 5 | 6 | 6 |
| `223.npz` | 1 | 9 | 9 |

## Keys

| Key | dtype | Holds |
|---|---|---|
| `fsc`, `ssc` | float32 | FSC-A and SSC-A as FlowKit's raw events hold them (float32 in the file, so nothing is lost) |
| `z`, `gx`, `gy` | float64 | the reference `f_plot(...)["z"]`, `["x"]` and `["y"]` for the 100 by 100 grid, log scale |
| `zz` | uint8 | its `["zz"]`, each event's 1-based (SSC bin, FSC bin) |
| `candidates` | int16 | the reference focal maxima, 1-based (row, col) |
| `peaks` | int16 | the reference refined, filtered and sorted peaks |
| `his_area` | int16 | the reference `find_area` cells for each peak as (row, col, population) |
| `fixed_area` | int16 | the same with the distance disc centred on the peak |
| `his_masks`, `fixed_masks` | uint8 | the reference gate's masks, from the reference cells and from the fixed cells, `np.packbits` along the events |

Unpack a mask with `np.unpackbits(f["his_masks"], axis=0, count=len(f["fsc"])).astype(bool)`. `tests/flowcases.py` loads all of it.

## The CLI's channels

`202_fluor.npz` holds three fluorescence channels of the same `202.fcs`, for the same events as `202.npz`: every 5th event, the first 50,000. `tests/flow/test_flow_pick.py` puts them beside `202.npz`'s FSC-A and SSC-A to run `datanuggets-flow` end to end on real events. Recorded by `benchmarks/flow/record_fluor.py` on the laptop on 2026-10-20 with flowio's `as_array(preprocess=True)`, as `datanuggets.flow.pick.read_fcs` reads a file, after checking that the FSC-A and SSC-A it read equal `202.npz`'s.

| Key | dtype | Holds |
|---|---|---|
| `fluor` | float32 | `FJComp-Alexa Fluor 700-A` (CD8), `FJComp-BV510-A` (CD3) and `FJComp-Spark Blue-A` (CD4), in the file's channel order (float32 in the file, so nothing is lost) |
| `pnn`, `pns` | str | those three channels' `$PnN` names and `$PnS` labels |
