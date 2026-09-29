# datanuggets

Native Python port of the R packages `datanugget` and `WCluster` (Beavers et al., 2024), validated against the R implementation to 1e-8 wherever the algorithm is deterministic.

Status: in development. Ported so far: `create_dn`, `create_dn_centers`.

## Development

    uv sync --python 3.14
    uv run pytest

The parity fixtures in `tests/fixtures/` were generated from R; regenerating them needs R (see `r/README.md`). Running the tests does not.

## License

GPL-2.0-only, the same license as the R packages. The algorithms are by the authors of `datanugget` and `WCluster`: Rituparna Dey, Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi and Mariusz Lubomirski.
