# R fixture generators (dev only)

These scripts run the original R packages and freeze their behavior into `tests/fixtures/`. The Python tests read those files and never call R.

1. Install R 4.6.1 (`winget install --id RProject.R --exact --version 4.6.1`).
2. From the repo root: `Rscript r/setup.R` (installs datanugget 1.5.0 and WCluster 1.3.0 into a private library, whose location `DATANUGGETS_R_LIB` overrides).
3. `Rscript r/fixtures_rcompat.R`, `Rscript r/fixtures_create_dn.R`, `Rscript r/fixtures_rstats.R`, `Rscript r/fixtures_kmeans.R` and `Rscript r/fixtures_refine_dn.R`.
4. `Rscript r/fixtures_get_dn_means.R`, `Rscript r/fixtures_ac.R`, `Rscript r/fixtures_rwstats.R`, `Rscript r/fixtures_wstats.R` and `Rscript r/fixtures_wpca.R`. All but `fixtures_ac.R` read the create_dn fixtures back, and `fixtures_get_dn_means.R` reads the refine_dn ones too, so they run after them.
5. `Rscript r/fixtures_rsample.R` and `Rscript r/fixtures_compat.R`. Both build their own inputs from R's datasets, so they can run any time after step 2. `fixtures_compat.R` records the evidence behind `docs/plans/2026-09-30-r-compat.md`: R's value, warnings and error for each case, including the calls where R errors or never returns.
6. `Rscript r/fixtures_whclust.R` and `Rscript r/fixtures_wkmeans.R`. Both build their own inputs, and Whclust's 200-row case takes about five minutes. `r/wcluster_counted.R` holds copies of WCluster's iterative functions with counters added: every case runs the copy beside WCluster's own function, stops unless the two return identical values or stop with the same message, and records how often each R quirk fired, so a case proves it reaches the quirk it's named for.

Arrays are raw little-endian float64 (`<f8`) or int32 (`<i4`), row-major. `manifest.json` gives each file's shape. Random draws are recorded with `trace(sample.int)` and internal calls with `trace(create.DNcenters)` and `trace(kmeans)`, so the packages run unmodified. `trace(AC)` records the loop counts inside `AC` the same way, and `trace(Wpca)` records the centered data, the matrix `Wpca` hands to `eigen()` and the eigenvalues it gets back. `trace(init)` and `trace(Wkmeans_single)` record each Wkmeans start's labels and the labels and WWCSS it ends with.
