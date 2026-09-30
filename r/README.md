# R fixture generators (dev only)

These scripts run the original R packages and freeze their behavior into `tests/fixtures/`. The Python tests read those files and never call R.

1. Install R 4.6.1 (`winget install --id RProject.R --exact --version 4.6.1`).
2. From the repo root: `Rscript r/setup.R` (installs datanugget 1.5.0 into a private library, whose location `DATANUGGETS_R_LIB` overrides).
3. `Rscript r/fixtures_rcompat.R`, `Rscript r/fixtures_create_dn.R`, `Rscript r/fixtures_rstats.R`, `Rscript r/fixtures_kmeans.R` and `Rscript r/fixtures_refine_dn.R`.
4. `Rscript r/fixtures_get_dn_means.R` and `Rscript r/fixtures_ac.R`. The first reads the create_dn and refine_dn fixtures back, so it runs after them.

Arrays are raw little-endian float64 (`<f8`) or int32 (`<i4`), row-major. `manifest.json` gives each file's shape. Random draws are recorded with `trace(sample.int)` and internal calls with `trace(create.DNcenters)` and `trace(kmeans)`, so the packages run unmodified. `trace(AC)` records the loop counts inside `AC` the same way.
