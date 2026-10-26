# 20250618 study: interactive datanuggets-flow run

Outputs of one `datanuggets-flow` run, committed for review. Not for merging into `main`.

- **Input:** 4 FCS wells (B3 and C3 = treatment 1, B5 and C5 = treatment 2), about 1M events each. The data are raw (uncompensated, untransformed).
- **Code:** branch `feat/flow-local-maxima-figure` (#17). Populations were chosen at the prompt from `local_maxima.png`.
- **Command:** `datanuggets-flow "20250618_Study files" --out outputs/cellfc-interactive`, with defaults `--k 8`, `--dn-num2 1000`, `--cofactor 6000` and `--disc his`.

| File | Pops chosen | Kept events |
| --- | --- | --- |
| 1, B3 Well_003 | 2, 5 | 507,920 (52%) |
| 2, B5 Well_005 | 2 | 321,289 (27%) |
| 3, C3 Well_011 | 2, 4 | 563,774 (56%) |
| 4, C5 Well_013 | 2 | 334,702 (30%) |

The 24 clustered channels are every `-A` channel except FSC and SSC, `LIVE DEAD Blue-A` included. `cluster_summary.csv` gives each file's share of cells per cluster.

**Reviewer note:** B3 and C3 each contribute a second scatter population (Pop 5 and Pop 4) that B5 and C5 don't have. The clusters that are mostly treatment 1 (1, 2, 6 and 8) may reflect that selection rather than the treatment.

The t-SNE figure from #19 isn't in this folder. A rerun with the same `--pops` reproduces every table here exactly and adds `clusters_tsne.png`.
