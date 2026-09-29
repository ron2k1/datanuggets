# Feature 1: create_dn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `create.DN` and `create.DNcenters` from R datanugget 1.5.0 to Python so that, with R's random draws replayed, assignments and weights match R exactly and centers and scales match to rtol 1e-8.

**Architecture:** R is installed as a dev-only oracle. R scripts write frozen fixtures (raw little-endian `.bin` arrays + `manifest.json`), recording every `sample.int` draw and every internal `create.DNcenters` call through `trace()`. Python code reproduces the R and Rfast semantics it depends on (`splitIndices`, Armadillo's summation order, libstdc++ `std::sort` tie order) in small private modules, each tested bit-for-bit against R, and `create_dn` is assembled from them.

**Tech Stack:** Python 3.12+ (dev on 3.14), numpy >= 2.0, joblib, optional tqdm/pandas, pytest, uv, ruff; R 4.6.1 with datanugget 1.5.0, Rfast, jsonlite.

**Spec:** `docs/specs/2026-09-29-datanuggets-python-port-design.md` (sections 3-6 govern this plan; section 8 build order is superseded by feature-by-feature execution, feature 1 = create_dn).

## Global Constraints

- Package name `datanuggets`, src layout, import `datanuggets`. R names map to snake_case (`create.DN` -> `create_dn`, `DN.num1` -> `dn_num1`, `R` -> `r`).
- Python indices are 0-based everywhere; tests add 1 when comparing with R.
- Algorithm defaults equal R's exactly: seed 291102, R 5000, delete.percent 0.1, DN.num1 10^4, DN.num2 2000, center.method "mean", dist.metric "euclidean".
- Parity tiers: Exact = `np.testing.assert_array_equal`; Numeric = `assert_allclose(rtol=1e-8, atol=1e-10)`; Rfast distance kernels = bitwise (`assert_array_equal`).
- Runtime deps for this feature: numpy >= 2.0, joblib >= 1.4 only. Optional extras: `frame` (pandas), `progress` (tqdm). scipy and pygam arrive with later features.
- Public signature (spec section 4): `create_dn(x, center_method="mean", r=5000, delete_percent=0.1, dn_num1=10_000, dn_num2=2000, dist_metric="euclidean", seed=291102, n_jobs=1, progress=False) -> DataNuggets`.
- `requires-python = ">=3.12"`. License GPL-2.0-only.
- Fixtures: `*.bin` must be `binary` in `.gitattributes`. Fixture generation is dev-only; CI and `pytest` never need R.
- Commits: one logical change each, imperative subject <= 70 chars plus a why/what body. Never add `Claude-Session:`, `Co-Authored-By`, or "Generated with Claude Code" lines (a repo-local `commit-msg` hook strips them; check with `git log --grep=Claude-Session` = 0).
- Work on branch `feat/create-dn`; `main` advances only by PR.
- Repo root holds only toolchain files (pyproject, lockfile, README, LICENSE, .gitattributes, .gitignore). Everything else lives in `src/`, `tests/`, `r/`, `scripts/`, `docs/`.
- Shell: PowerShell 7 on Windows 11. Run every command from the repo root `C:\Users\thegr\OneDrive\Desktop\Rutgers Research\datanuggets`.

## Review Focus

1. **Exact distance ties** (discretized measurements like iris, integer grids, duplicate rows): a user expects the same nugget R picks. Rfast breaks nearest-center ties with libstdc++ `std::sort`, not first index. Pinned by `rcompat_dista` full-order fixtures (Task 3) and the `grid_ties_*` create_dn cases (Task 5).
2. **Missing or infinite values in x**: R silently produces garbage; a user expects a clear error. Test `test_nan_and_inf_rejected` in Task 5.
3. **Mixed-type DataFrames and integer columns**: non-numeric columns must raise `TypeError`; integer columns must give the same result as the same values as float. Tests `test_non_numeric_frame_rejected`, `test_integer_input_equals_float_input` in Task 5.
4. **Large n during assignment**: the n x k distance matrix must never be materialized; results must not depend on the block size. Test `test_nearest_index_block_size_invariant` in Task 3.
5. **`n_jobs > 1`**: must return exactly the `n_jobs=1` result. Test `test_n_jobs_does_not_change_result` in Task 5.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE`, `.gitattributes`, `.gitignore` | toolchain and repo metadata |
| `r/setup.R` | installs the pinned R oracle packages into a private library |
| `r/fixture_io.R` | fixture writer (`writeBin` + manifest) and `trace()`-based capture helpers |
| `r/fixtures_rcompat.R` | fixtures for R/Rfast semantics: `splitIndices`, `Dist`, `dista`, `rowMins` |
| `r/fixtures_create_dn.R` | fixtures for `create.DN` with draws and every `create.DNcenters` call |
| `r/README.md` | how to regenerate fixtures |
| `src/datanuggets/__init__.py` | public API |
| `src/datanuggets/_defaults.py` | R defaults and internal block sizes |
| `src/datanuggets/exceptions.py` | `DataNuggetWarning` |
| `src/datanuggets/_validate.py` | input coercion and argument checks |
| `src/datanuggets/_draws.py` | `Draws` protocol, `NumpyDraws`, `ReplayDraws`, `DrawRecord`, `ReplayMismatch`, `as_draws` |
| `src/datanuggets/_rcompat.py` | R base semantics: `split_indices` |
| `src/datanuggets/_rfast.py` | Rfast semantics: Armadillo-order sums, distances, nearest index with `std::sort` ties |
| `src/datanuggets/result.py` | `DataNuggets`, `CentersResult` |
| `src/datanuggets/centers.py` | `create_dn_centers` |
| `src/datanuggets/create.py` | `create_dn` |
| `tests/rfixtures.py` | fixture loader + tolerance constants |
| `tests/test_draws.py`, `tests/test_rcompat.py`, `tests/test_rfast.py`, `tests/test_centers.py`, `tests/test_create.py` | tests |
| `scripts/parity_report.py` | writes `docs/validation/create_dn.md` from the fixtures |

---

### Task 1: Repo scaffold, R oracle, and fixtures

**Files:**
- Create: `pyproject.toml`, `README.md`, `LICENSE`, `.gitattributes`, `.gitignore`, `src/datanuggets/__init__.py`
- Create: `r/setup.R`, `r/fixture_io.R`, `r/fixtures_rcompat.R`, `r/fixtures_create_dn.R`, `r/README.md`
- Generated: `tests/fixtures/rcompat_split_indices/`, `tests/fixtures/rcompat_dist/`, `tests/fixtures/rcompat_dista/`, `tests/fixtures/rcompat_rowmins/`, `tests/fixtures/create_dn/`

**Interfaces:**
- Produces: fixture folders, each with `manifest.json` = `{group, case, args, arrays: {name: {file, dtype: "<f8"|"<i4", shape: [..]}}, versions, ...extra}` and one `.bin` per array, row-major.
- `create_dn` cases carry arrays `x, draw_n, draw_size, draw_values, centers, weights, scales, assignments, callNN_rs, callNN_kept` and extra keys `n_calls`, `call_dn_num` (list), `warnings` (list of strings).

- [ ] **Step 1: Branch and scaffold files**

```powershell
git checkout -b feat/create-dn
New-Item -ItemType Directory -Force src/datanuggets, tests, r, scripts, docs/validation | Out-Null
Invoke-WebRequest https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt -OutFile LICENSE
```

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "datanuggets"
version = "0.1.0.dev0"
description = "Native Python port of the R packages datanugget and WCluster (data nuggets for large-data reduction)."
readme = "README.md"
requires-python = ">=3.12"
license = "GPL-2.0-only"
authors = [{ name = "Ronil Basu" }]
dependencies = ["numpy>=2.0", "joblib>=1.4"]

[project.optional-dependencies]
frame = ["pandas>=2.2"]
progress = ["tqdm>=4.66"]

[dependency-groups]
dev = ["pytest>=8", "pytest-xdist>=3.6", "ruff>=0.6", "pandas>=2.2", "tqdm>=4.66"]

[tool.hatch.build.targets.wheel]
packages = ["src/datanuggets"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
src = ["src", "tests", "scripts"]
```

`.gitattributes`:

```
* text=auto eol=lf
*.bin binary
*.png binary
*.pdf binary
```

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
dist/
build/
tmp/
```

`README.md`:

```markdown
# datanuggets

Native Python port of the R packages `datanugget` and `WCluster` (Beavers et al., 2024), validated against the R implementation to 1e-8 wherever the algorithm is deterministic.

Status: in development. Ported so far: `create_dn`, `create_dn_centers`.

## Development

    uv sync --python 3.14
    uv run pytest

The parity fixtures in `tests/fixtures/` were generated from R; regenerating them needs R (see `r/README.md`). Running the tests does not.

## License

GPL-2.0-only, the same license as the R packages. The algorithms are by the authors of `datanugget` and `WCluster`: Rituparna Dey, Yajie Duan, Traymon Beavers, Javier Cabrera, Ge Cheng, Kunting Qi and Mariusz Lubomirski.
```

`src/datanuggets/__init__.py`:

```python
"""Native Python port of the R packages datanugget and WCluster."""

__version__ = "0.1.0.dev0"
```

- [ ] **Step 2: Install R 4.6.1**

```powershell
winget install --id RProject.R --exact --version 4.6.1 --scope user --silent --accept-package-agreements --accept-source-agreements
```

If `--scope user` is rejected, rerun without `--scope user` (a UAC prompt appears; this is a manual blocker: Telegram-ping RON2K via `~/.claude/tools/notify.sh` and wait). Then locate Rscript:

```powershell
$rs = (Get-ChildItem "$env:LOCALAPPDATA\Programs\R", "$env:ProgramFiles\R" -Recurse -Filter Rscript.exe -ErrorAction SilentlyContinue | Where-Object FullName -match 'R-4\.6\.1\\bin\\Rscript\.exe$' | Select-Object -First 1).FullName
& $rs --version
```

Expected: `Rscript (R) version 4.6.1`.

- [ ] **Step 3: Write `r/setup.R` and install the oracle packages**

```r
# Installs the pinned R oracle packages into a private library (dev only).
lib <- Sys.getenv("DATANUGGETS_R_LIB", unset = file.path(
  Sys.getenv(if (.Platform$OS.type == "windows") "LOCALAPPDATA" else "HOME"), "datanuggets-rlib"))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
install.packages(c("jsonlite", "datanugget"), lib = lib, repos = "https://cloud.r-project.org")
stopifnot(packageVersion("datanugget", lib.loc = lib) == "1.5.0")
cat("datanugget", as.character(packageVersion("datanugget", lib.loc = lib)),
    "| Rfast", as.character(packageVersion("Rfast", lib.loc = lib)), "\n")
```

Run: `& $rs r/setup.R`
Expected last line: `datanugget 1.5.0 | Rfast <version>`.

- [ ] **Step 4: Write `r/fixture_io.R`**

```r
# Shared helpers for the fixture generators. Run every r/*.R script from the repo root.
if (!file.exists("pyproject.toml")) stop("run from the datanuggets repo root")

lib <- Sys.getenv("DATANUGGETS_R_LIB", unset = file.path(
  Sys.getenv(if (.Platform$OS.type == "windows") "LOCALAPPDATA" else "HOME"), "datanuggets-rlib"))
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(jsonlite)
  library(datanugget)
})
stopifnot(packageVersion("datanugget") == "1.5.0")

versions <- function() list(
  R = R.version.string,
  datanugget = as.character(packageVersion("datanugget")),
  Rfast = as.character(packageVersion("Rfast")),
  lapack = La_version()
)

as_dbl <- function(m) {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  dimnames(m) <- NULL
  m
}

# Writes one array as raw little-endian bytes, row-major, and returns its manifest entry.
write_array <- function(dir, name, value) {
  if (is.data.frame(value)) value <- as.matrix(value)
  is_int <- is.integer(value) || is.logical(value)
  shape <- if (is.null(dim(value))) length(value) else dim(value)
  flat <- if (is.null(dim(value))) value else as.vector(t(value))
  path <- file.path(dir, paste0(name, ".bin"))
  con <- file(path, "wb")
  on.exit(close(con))
  if (is_int) {
    writeBin(as.integer(flat), con, size = 4L, endian = "little")
  } else {
    writeBin(as.double(flat), con, size = 8L, endian = "little")
  }
  list(file = basename(path), dtype = if (is_int) "<i4" else "<f8", shape = as.list(as.integer(shape)))
}

write_case <- function(group, case_id, args, arrays, extra = list()) {
  dir <- file.path("tests", "fixtures", group, case_id)
  if (dir.exists(dir)) unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE)
  meta <- Map(function(nm, v) write_array(dir, nm, v), names(arrays), arrays)
  manifest <- c(list(group = group, case = case_id, args = args, arrays = meta,
                     versions = versions()), extra)
  write_json(manifest, file.path(dir, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  invisible(dir)
}

no_args <- setNames(list(), character(0))

# trace() hooks record what the packages actually do, without modifying them.
.capture <- new.env()

capture_draws <- function(expr) {
  .capture$draws <- list()
  trace("sample.int", print = FALSE, where = baseenv(), exit = quote(
    .capture$draws[[length(.capture$draws) + 1L]] <- list(n = n, size = size, value = returnValue())
  ))
  on.exit(untrace("sample.int", where = baseenv()), add = TRUE)
  value <- expr
  list(value = value, draws = .capture$draws)
}

# create.DN reaches create.DNcenters two ways: stage 1 through a helper whose environment is
# clean.env (parent globalenv), which resolves the exported copy in package:datanugget; stage 2
# directly from the namespace. Trace both bindings, or stage-1 calls go unrecorded.
capture_centers <- function(expr) {
  .capture$calls <- list()
  hook <- quote(.capture$calls[[length(.capture$calls) + 1L]] <- list(
    RS = as.matrix(RS), DN.num = DN.num, kept = attr(returnValue(), "kept")))
  envs <- list(asNamespace("datanugget"), as.environment("package:datanugget"))
  for (e in envs) trace("create.DNcenters", exit = hook, print = FALSE, where = e)
  on.exit(for (e in envs) suppressWarnings(untrace("create.DNcenters", where = e)), add = TRUE)
  value <- expr
  list(value = value, calls = .capture$calls)
}

# Fail closed if trace() is not seeing calls made from inside base R.
local({
  chk <- capture_draws({ set.seed(1); sample(10, 10) })
  stopifnot(length(chk$draws) == 1L, identical(as.integer(chk$draws[[1]]$value), as.integer(chk$value)))
})
```

- [ ] **Step 5: Write `r/fixtures_rcompat.R`**

```r
# Fixtures pinning the R and Rfast semantics create.DN depends on.
source("r/fixture_io.R")

# parallel::splitIndices over a grid of (nx, ncl)
pairs <- expand.grid(nx = c(1, 2, 3, 7, 100, 150, 272, 999, 1000, 1001, 5000, 12345),
                     ncl = c(1, 2, 3, 4, 7, 10, 25))
pairs <- pairs[pairs$ncl <= pairs$nx, ]
lens <- list()
nchunks <- integer(nrow(pairs))
for (i in seq_len(nrow(pairs))) {
  ch <- parallel::splitIndices(pairs$nx[i], pairs$ncl[i])
  stopifnot(identical(as.integer(unlist(ch)), seq_len(pairs$nx[i])))  # contiguous, in order
  nchunks[i] <- length(ch)
  lens[[i]] <- lengths(ch)
}
write_case("rcompat_split_indices", "grid", no_args, list(
  nx = as.integer(pairs$nx), ncl = as.integer(pairs$ncl),
  nchunks = nchunks, chunk_len = as.integer(unlist(lens))))

# Rfast::Dist, compared bit for bit
dist_sets <- list(
  iris40 = as_dbl(iris[1:40, 1:4]),
  quakes40 = as_dbl(quakes[1:40, ]),
  mtcars = as_dbl(mtcars),                        # p = 11: odd p exercises the leftover accumulator
  faithful1d = as_dbl(faithful[1:40, "eruptions", drop = FALSE])
)
for (nm in names(dist_sets)) for (metric in c("euclidean", "manhattan")) {
  x <- dist_sets[[nm]]
  write_case("rcompat_dist", paste(nm, metric, sep = "_"), list(metric = metric),
             list(x = x, dist = unname(Rfast::Dist(x, method = metric))))
}

# Rfast::dista index output: the complete std::sort order and the k = 1 nearest index
grid <- as_dbl(expand.grid(a = 0:5, b = 0:5))
dista_sets <- list(
  iris = list(xnew = as_dbl(iris[, 1:4]), x = as_dbl(iris[seq(1, 150, by = 5), 1:4])),
  grid = list(xnew = grid, x = grid[c(1, 8, 15, 22, 29, 36, 3, 18, 33, 6, 31, 13), ]),
  grid_dups = list(xnew = grid, x = grid[c(8, 8, 15, 15, 22, 1, 36, 29, 29, 29, 3, 3, 18, 18,
                                           33, 6, 31, 13, 13, 20), ])
)
for (nm in names(dista_sets)) for (metric in c("euclidean", "manhattan")) {
  s <- dista_sets[[nm]]
  # dista(trans = TRUE, the default) returns one row per xnew row; std::sort runs over all
  # nrow(x) distances whatever k is, so k = nrow(x) exposes the complete order behind k = 1.
  full <- Rfast::dista(s$xnew, s$x, type = metric, k = nrow(s$x), index = TRUE)
  near <- Rfast::dista(s$xnew, s$x, type = metric, k = 1L, index = TRUE)
  stopifnot(identical(dim(full), c(nrow(s$xnew), nrow(s$x))), full[, 1] == near)
  storage.mode(full) <- "integer"
  write_case("rcompat_dista", paste(nm, metric, sep = "_"), list(metric = metric),
             list(xnew = s$xnew, x = s$x, order_full = unname(full), nearest = as.integer(near)))
}

# Rfast::rowMins index on a matrix full of ties
m <- unname(Rfast::Dist(grid, method = "manhattan"))
diag(m) <- max(m) + 1
write_case("rcompat_rowmins", "grid_manhattan", no_args,
           list(m = m, index = as.integer(Rfast::rowMins(m, value = FALSE))))
cat("rcompat fixtures written\n")
```

- [ ] **Step 6: Write `r/fixtures_create_dn.R`**

```r
# Fixtures for create.DN: inputs, every random draw, every create.DNcenters call, outputs.
source("r/fixture_io.R")

grid <- as_dbl(expand.grid(a = 0:5, b = 0:5))
datasets <- list(
  iris = as_dbl(iris[, 1:4]),
  faithful = as_dbl(faithful),
  quakes = as_dbl(quakes),
  faithful_1d = as_dbl(faithful[, "eruptions", drop = FALSE]),
  grid_ties = grid[rep(seq_len(nrow(grid)), 4), , drop = FALSE]  # every point 4 times
)

cases <- list()
add_case <- function(id, data, center.method, R, DN.num1, DN.num2,
                     delete.percent = 0.1, dist.metric = "euclidean", seed = 291102) {
  cases[[id]] <<- list(data = data, args = list(
    center.method = center.method, R = R, delete.percent = delete.percent,
    DN.num1 = DN.num1, DN.num2 = DN.num2, dist.metric = dist.metric, seed = seed))
}
sizes <- list(iris = c(100, 60, 20), faithful = c(100, 90, 30), quakes = c(300, 400, 100))
for (ds in names(sizes)) for (metric in c("euclidean", "manhattan")) {
  for (cm in c("mean", "random", "original")) {
    s <- sizes[[ds]]
    add_case(paste(ds, metric, cm, sep = "_"), ds, cm, s[1], s[2], s[3], dist.metric = metric)
  }
}
add_case("quakes_onechunk_dp30", "quakes", "mean", 5000, 500, 150, delete.percent = 0.3)
add_case("iris_default_num1_warnpath", "iris", "mean", 5000, 10^4, 50)
add_case("iris_dn2_equals_stage1", "iris", "mean", 100, 60, 60)
add_case("faithful_1d", "faithful_1d", "mean", 100, 90, 30)
add_case("grid_ties_euclidean_mean", "grid_ties", "mean", 100, 40, 12)
add_case("grid_ties_manhattan_random", "grid_ties", "random", 100, 40, 12,
         delete.percent = 0.3, dist.metric = "manhattan")
add_case("quakes_seed7_random", "quakes", "random", 300, 400, 100, seed = 7)

for (id in names(cases)) {
  cs <- cases[[id]]
  x <- datasets[[cs$data]]
  warns <- character()
  run <- function() withCallingHandlers(
    suppressMessages(do.call(create.DN, c(list(x = x), cs$args, list(no.cores = 1, make.pbs = FALSE)))),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  res <- capture_centers(capture_draws(run()))
  dn <- res$value$value
  draws <- res$value$draws
  calls <- res$calls

  n <- nrow(x)
  chunks <- parallel::splitIndices(n, ceiling(n / cs$args$R))
  chunks <- chunks[lengths(chunks) > 0L]
  if (length(calls) != length(chunks) + 1L) {
    stop(id, ": captured ", length(calls), " create.DNcenters calls, expected ", length(chunks) + 1L)
  }
  stopifnot(all(vapply(draws, function(d) length(d$value) == d$size, logical(1))))

  tab <- dn[["Data Nuggets"]]
  p <- ncol(x)
  arrays <- list(
    x = x,
    draw_n = as.integer(vapply(draws, function(d) as.numeric(d$n), numeric(1))),
    draw_size = as.integer(vapply(draws, function(d) as.numeric(d$size), numeric(1))),
    draw_values = as.integer(unlist(lapply(draws, function(d) d$value))),
    centers = unname(as.matrix(tab[, 1 + seq_len(p)])),
    weights = as.integer(tab[["Weight"]]),
    scales = as.double(tab[["Scale"]]),
    assignments = as.integer(dn[["Data Nugget Assignments"]])
  )
  for (i in seq_along(calls)) {
    arrays[[sprintf("call%02d_rs", i)]] <- unname(calls[[i]]$RS)
    arrays[[sprintf("call%02d_kept", i)]] <- as.integer(calls[[i]]$kept)
  }
  write_case("create_dn", id, cs$args, arrays, extra = list(
    n_calls = length(calls),
    call_dn_num = I(vapply(calls, function(cl) as.numeric(cl$DN.num), numeric(1))),
    warnings = I(warns)))
  cat(sprintf("%-30s nuggets=%4d draws=%4d calls=%2d warnings=%d\n",
              id, nrow(tab), length(draws), length(calls), length(warns)))
}
```

- [ ] **Step 7: Write `r/README.md`**

```markdown
# R fixture generators (dev only)

These scripts run the original R packages and freeze their behavior into `tests/fixtures/`. The Python tests read those files; they never call R.

1. Install R 4.6.1 (`winget install --id RProject.R --exact --version 4.6.1`).
2. From the repo root: `Rscript r/setup.R` (installs datanugget 1.5.0 into a private library; override its location with `DATANUGGETS_R_LIB`).
3. `Rscript r/fixtures_rcompat.R` and `Rscript r/fixtures_create_dn.R`.

Arrays are raw little-endian float64 (`<f8`) or int32 (`<i4`), row-major; `manifest.json` gives each file's shape. Random draws are recorded with `trace(sample.int)` and internal calls with `trace(create.DNcenters)`, so the packages run unmodified.
```

- [ ] **Step 8: Generate the fixtures**

```powershell
& $rs r/fixtures_rcompat.R
& $rs r/fixtures_create_dn.R
```

Expected: `rcompat fixtures written`, then 25 `create_dn` lines. `iris_default_num1_warnpath` and `iris_dn2_equals_stage1` each report `warnings=1`; every `*_random` case reports more draws than 1; every other case reports `draws=1`.

If the script stops with "captured N create.DNcenters calls, expected M": N = 1 means the `package:datanugget` binding was not traced, so stage-1 calls went unrecorded; N = 2M - 1 or 2M means a call fired two hooks (the second `trace()` wrapped an already traced function). For N = 1, check `class(get("create.DNcenters", envir = as.environment("package:datanugget")))` after tracing (it must be `functionWithTrace`). For double counting, trace only the namespace, then trace the package binding only if its class is not already `functionWithTrace`. Never relax the count check.

- [ ] **Step 9: Verify fixture hygiene and commit**

```powershell
(Get-ChildItem tests/fixtures -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
git add -A
git status --short | Select-Object -First 5
```

Expected: under 25 MB.

```powershell
git commit -m "Add repo scaffold and R oracle fixtures for create_dn" -m "Installs R 4.6.1 + datanugget 1.5.0 as a dev-only oracle and freezes create.DN's behavior (inputs, every sample.int draw, every internal create.DNcenters call, outputs) plus the Rfast and base-R semantics it relies on. Fixtures are raw little-endian bytes so inputs are bit-identical on both sides."
```

---

### Task 2: Fixture loader and the Draws abstraction

**Files:**
- Create: `src/datanuggets/_defaults.py`, `src/datanuggets/_draws.py`, `tests/rfixtures.py`, `tests/test_draws.py`

**Interfaces:**
- Produces: `DrawRecord(n: int, size: int, value: np.ndarray)` (R 1-based values); `ReplayMismatch(AssertionError)`; `Draws` protocol with `permutation(n) -> np.ndarray` (0-based) and `index(m) -> int` (0-based); `NumpyDraws(seed: int | np.random.Generator)`; `ReplayDraws(records)` with `.assert_exhausted()`; `as_draws(seed) -> Draws`.
- Produces (tests): `rfixtures.load_group(group) -> list[Case]`; `Case.id`, `Case.args`, `Case.manifest`, `Case.array(name)`, `Case.has(name)`, `Case.draws() -> list[DrawRecord]`; constants `RTOL = 1e-8`, `ATOL = 1e-10`.

- [ ] **Step 1: Set up the environment**

```powershell
uv sync --python 3.14
uv run python -c "import datanuggets, numpy; print(datanuggets.__version__, numpy.__version__)"
```

Expected: `0.1.0.dev0 2.x.y`.

- [ ] **Step 2: Write the failing tests `tests/test_draws.py`**

```python
import numpy as np
import pytest
from rfixtures import load_group

from datanuggets._draws import (
    DrawRecord,
    NumpyDraws,
    ReplayDraws,
    ReplayMismatch,
    as_draws,
)


def test_replay_returns_zero_based_values_in_order():
    draws = ReplayDraws([DrawRecord(3, 3, np.array([2, 3, 1])), DrawRecord(4, 1, np.array([4]))])
    np.testing.assert_array_equal(draws.permutation(3), [1, 2, 0])
    assert draws.index(4) == 3
    draws.assert_exhausted()


def test_replay_rejects_a_draw_r_did_not_make():
    draws = ReplayDraws([DrawRecord(3, 3, np.array([2, 3, 1]))])
    with pytest.raises(ReplayMismatch, match=r"sample.int\(5, 5\)"):
        draws.permutation(5)


def test_replay_rejects_extra_draws():
    draws = ReplayDraws([])
    with pytest.raises(ReplayMismatch, match="made only 0"):
        draws.index(2)


def test_replay_reports_unconsumed_draws():
    draws = ReplayDraws([DrawRecord(2, 1, np.array([1]))])
    with pytest.raises(ReplayMismatch, match="consumed 0"):
        draws.assert_exhausted()


def test_numpy_draws_are_reproducible():
    a, b = NumpyDraws(291102), NumpyDraws(291102)
    np.testing.assert_array_equal(a.permutation(50), b.permutation(50))
    assert a.index(7) == b.index(7)
    assert sorted(NumpyDraws(1).permutation(10).tolist()) == list(range(10))
    assert NumpyDraws(1).index(1) == 0


def test_as_draws_accepts_int_generator_and_draws():
    assert isinstance(as_draws(5), NumpyDraws)
    assert isinstance(as_draws(np.random.default_rng(5)), NumpyDraws)
    replay = ReplayDraws([])
    assert as_draws(replay) is replay


@pytest.mark.parametrize("bad", [1.5, True, "7", None])
def test_as_draws_rejects_other_types(bad):
    with pytest.raises(TypeError, match="seed"):
        as_draws(bad)


def test_fixture_draws_round_trip():
    case = next(c for c in load_group("create_dn") if c.id == "iris_euclidean_random")
    records = case.draws()
    assert records[0].n == 150 and records[0].size == 150
    assert sorted(records[0].value.tolist()) == list(range(1, 151))
    assert all(r.size == 1 for r in records[1:])
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_draws.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'rfixtures'` or `'datanuggets._draws'`.

- [ ] **Step 4: Implement `tests/rfixtures.py`**

```python
"""Load the R-generated parity fixtures (manifest.json + raw little-endian .bin arrays)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datanuggets._draws import DrawRecord

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
RTOL = 1e-8
ATOL = 1e-10


@dataclass(frozen=True)
class Case:
    path: Path
    manifest: dict

    @property
    def id(self) -> str:
        return self.manifest["case"]

    @property
    def args(self) -> dict:
        return self.manifest["args"]

    def has(self, name: str) -> bool:
        return name in self.manifest["arrays"]

    def array(self, name: str) -> np.ndarray:
        meta = self.manifest["arrays"][name]
        data = np.fromfile(self.path / meta["file"], dtype=meta["dtype"])
        return data.reshape(meta["shape"])

    def draws(self) -> list[DrawRecord]:
        sizes = self.array("draw_size")
        values = self.array("draw_values").astype(np.int64)
        records, pos = [], 0
        for n, size in zip(self.array("draw_n").tolist(), sizes.tolist(), strict=True):
            records.append(DrawRecord(n, size, values[pos : pos + size]))
            pos += size
        assert pos == values.size, f"{self.id}: draw_values has {values.size - pos} extra entries"
        return records


def load_group(group: str) -> list[Case]:
    root = FIXTURE_ROOT / group
    dirs = (
        sorted(p for p in root.iterdir() if (p / "manifest.json").is_file())
        if root.is_dir()
        else []
    )
    if not dirs:
        raise FileNotFoundError(f"no fixtures under {root}; regenerate them with the scripts in r/")
    return [Case(p, json.loads((p / "manifest.json").read_text(encoding="utf-8"))) for p in dirs]
```

- [ ] **Step 5: Implement `src/datanuggets/_defaults.py`**

```python
"""R's default argument values (datanugget 1.5.0) and internal sizes, kept in one place."""

SEED = 291102
R_CHUNK = 5000
R_MIN = 100
R_MAX = 10_000
DELETE_PERCENT = 0.1
DN_NUM1 = 10_000
DN_NUM2 = 2000
CENTER_METHODS = ("mean", "random", "original")
DIST_METRICS = ("euclidean", "manhattan")

# Rows per block when computing distances, so no n x k or n x n temporary exceeds memory.
DIST_BLOCK_ROWS = 1024
ASSIGN_BLOCK_ROWS = 1024  # 1024 x 2000 centers = 16 MB per temporary
```

- [ ] **Step 6: Implement `src/datanuggets/_draws.py`**

```python
"""Source of randomness. Users get numpy; parity tests replay the draws R recorded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class ReplayMismatch(AssertionError):
    """Python requested a random draw that R did not make, or in a different order."""


@runtime_checkable
class Draws(Protocol):
    def permutation(self, n: int) -> np.ndarray:
        """R's sample(n, n), 0-based."""
        ...

    def index(self, m: int) -> int:
        """R's sample.int(m, 1), 0-based."""
        ...


class NumpyDraws:
    def __init__(self, seed: int | np.random.Generator):
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def permutation(self, n: int) -> np.ndarray:
        return self._rng.permutation(n)

    def index(self, m: int) -> int:
        return int(self._rng.integers(m))


@dataclass(frozen=True)
class DrawRecord:
    n: int
    size: int
    value: np.ndarray  # 1-based, as R returned it


class ReplayDraws:
    def __init__(self, records: list[DrawRecord]):
        self._records = list(records)
        self._pos = 0

    def _next(self, n: int, size: int) -> np.ndarray:
        if self._pos >= len(self._records):
            raise ReplayMismatch(
                f"draw #{self._pos + 1} sample.int({n}, {size}) requested, "
                f"but R made only {len(self._records)} draws"
            )
        rec = self._records[self._pos]
        if rec.n != n or rec.size != size:
            raise ReplayMismatch(
                f"draw #{self._pos + 1}: Python asked for sample.int({n}, {size}), "
                f"R did sample.int({rec.n}, {rec.size})"
            )
        self._pos += 1
        return rec.value - 1

    def permutation(self, n: int) -> np.ndarray:
        return self._next(n, n).astype(np.intp)

    def index(self, m: int) -> int:
        return int(self._next(m, 1)[0])

    def assert_exhausted(self) -> None:
        if self._pos != len(self._records):
            raise ReplayMismatch(f"R made {len(self._records)} draws, Python consumed {self._pos}")


def as_draws(seed) -> Draws:
    if isinstance(seed, Draws):
        return seed
    if isinstance(seed, np.random.Generator) or (
        isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
    ):
        return NumpyDraws(seed)
    raise TypeError(
        f"seed must be an int, a numpy Generator, or a Draws object, not {type(seed).__name__}"
    )
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_draws.py -q`
Expected: `9 passed`.

- [ ] **Step 8: Commit**

```powershell
git add src/datanuggets/_defaults.py src/datanuggets/_draws.py tests/rfixtures.py tests/test_draws.py uv.lock
git commit -m "Add fixture loader and replayable Draws abstraction" -m "All randomness goes through Draws. ReplayDraws hands back R's recorded sample.int draws and raises ReplayMismatch if Python asks for a draw R never made, so parity tests check the call sequence as well as the numbers."
```

---

### Task 3: R and Rfast semantics (`_rcompat`, `_rfast`)

**Files:**
- Create: `src/datanuggets/_rcompat.py`, `src/datanuggets/_rfast.py`, `tests/test_rcompat.py`, `tests/test_rfast.py`
- Modify: `docs/specs/2026-09-29-datanuggets-python-port-design.md` (section 6, create_dn items 6-7: record the Rfast findings)

**Interfaces:**
- Produces: `split_indices(nx: int, ncl: int) -> list[np.ndarray]` (0-based, empty chunks kept, like R).
- Produces: `arma_sum_last(terms: np.ndarray) -> np.ndarray`; `cross_distances(a, b, metric) -> np.ndarray (m, k)`; `dist_matrix(x, metric, block_rows=DIST_BLOCK_ROWS) -> np.ndarray (n, n)`; `gnu_sort_order(values, *, depth_limit=None) -> np.ndarray` (full order, for tests); `gnu_sort_first(values, *, depth_limit=None) -> int` (O(k) head of that order, used in production); `nearest_index(points, centers, metric, block_rows=ASSIGN_BLOCK_ROWS) -> np.ndarray` (0-based, int).

- [ ] **Step 1: Write the failing tests `tests/test_rcompat.py`**

```python
import numpy as np
from rfixtures import load_group

from datanuggets._rcompat import split_indices


def test_split_indices_matches_r():
    (case,) = load_group("rcompat_split_indices")
    nx, ncl, nchunks = case.array("nx"), case.array("ncl"), case.array("nchunks")
    lengths = case.array("chunk_len")
    pos = 0
    for n, c, k in zip(nx.tolist(), ncl.tolist(), nchunks.tolist(), strict=True):
        chunks = split_indices(n, c)
        expected = lengths[pos : pos + k].tolist()
        pos += k
        assert [ch.size for ch in chunks] == expected, (n, c)
        np.testing.assert_array_equal(np.concatenate(chunks), np.arange(n))
    assert pos == lengths.size
```

- [ ] **Step 2: Write the failing tests `tests/test_rfast.py`**

```python
import numpy as np
import pytest
from rfixtures import load_group

from datanuggets._rfast import (
    arma_sum_last,
    cross_distances,
    dist_matrix,
    gnu_sort_first,
    gnu_sort_order,
    nearest_index,
)

DIST = load_group("rcompat_dist")
DISTA = load_group("rcompat_dista")


def test_arma_sum_uses_even_odd_accumulators():
    terms = np.array([[1e16, 1.0, -1e16, 1.0]])
    # Armadillo: (1e16 + -1e16) + (1.0 + 1.0) = 2; a left-to-right sum gives 1.
    assert arma_sum_last(terms)[0] == 2.0
    assert arma_sum_last(np.array([[3.0]]))[0] == 3.0


@pytest.mark.parametrize("case", DIST, ids=lambda c: c.id)
def test_dist_matrix_is_bitwise_equal_to_rfast(case):
    got = dist_matrix(case.array("x"), case.args["metric"], block_rows=7)
    np.testing.assert_array_equal(got, case.array("dist"))


@pytest.mark.parametrize("case", DISTA, ids=lambda c: c.id)
def test_gnu_sort_order_reproduces_rfast_full_order(case):
    d = cross_distances(case.array("xnew"), case.array("x"), case.args["metric"])
    expected = case.array("order_full") - 1
    for r in range(d.shape[0]):
        np.testing.assert_array_equal(gnu_sort_order(d[r]), expected[r], err_msg=f"row {r}")


@pytest.mark.parametrize("case", DISTA, ids=lambda c: c.id)
def test_nearest_index_matches_rfast(case):
    got = nearest_index(case.array("xnew"), case.array("x"), case.args["metric"])
    np.testing.assert_array_equal(got + 1, case.array("nearest"))


def test_row_min_first_index_matches_rfast():
    (case,) = load_group("rcompat_rowmins")
    np.testing.assert_array_equal(case.array("m").argmin(axis=1) + 1, case.array("index"))


def test_nearest_index_block_size_invariant():
    case = next(c for c in DISTA if c.id == "grid_dups_euclidean")
    a = nearest_index(case.array("xnew"), case.array("x"), "euclidean")
    b = nearest_index(case.array("xnew"), case.array("x"), "euclidean", block_rows=5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("depth_limit", [None, 0])
def test_gnu_sort_order_sorts(depth_limit):
    rng = np.random.default_rng(0)
    values = rng.integers(0, 5, size=300).astype(float)  # heavy ties
    order = gnu_sort_order(values, depth_limit=depth_limit)
    assert sorted(order.tolist()) == list(range(300))
    assert np.all(np.diff(values[order]) >= 0)


@pytest.mark.parametrize("depth_limit", [None, 0, 1])
def test_gnu_sort_first_is_head_of_full_order(depth_limit):
    rng = np.random.default_rng(1)
    for size in (1, 2, 5, 16, 17, 18, 33, 64, 100, 257, 1000):
        for levels in (1, 2, 3, 10):
            values = rng.integers(0, levels, size=size).astype(float)
            expected = gnu_sort_order(values, depth_limit=depth_limit)[0]
            assert gnu_sort_first(values, depth_limit=depth_limit) == expected, (size, levels)


def test_small_rows_break_ties_by_first_index():
    # 16 or fewer centers: std::sort is a plain (stable) insertion sort
    assert gnu_sort_first(np.array([3.0, 1.0, 2.0, 1.0, 1.0])) == 1
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_rcompat.py tests/test_rfast.py -q`
Expected: collection error `No module named 'datanuggets._rcompat'`.

- [ ] **Step 4: Implement `src/datanuggets/_rcompat.py`**

```python
"""Exact re-implementations of base-R behavior the ported code depends on."""

from __future__ import annotations

import numpy as np


def _r_seq_length_out(start: float, stop: float, n: int) -> np.ndarray:
    """R's seq(start, stop, length.out = n): start + i * by, with the last value set to stop."""
    if n == 1:
        return np.array([start])
    by = (stop - start) / (n - 1)
    out = start + np.arange(n, dtype=np.float64) * by
    out[-1] = stop
    return out


def split_indices(nx: int, ncl: int) -> list[np.ndarray]:
    """R's parallel::splitIndices(nx, ncl), 0-based. Empty chunks are kept, as in R."""
    if ncl == 0:
        return []
    i = np.arange(1, nx + 1)
    if ncl == 1 or nx == 1:
        return [i - 1]
    fuzz = min((nx - 1) / 1000, 0.4 * nx / ncl)
    breaks = _r_seq_length_out(1 - fuzz, nx + fuzz, ncl + 1)
    bins = np.searchsorted(breaks, i, side="left") - 1  # cut(): right-closed (b_j, b_j+1]
    return [np.flatnonzero(bins == j) for j in range(ncl)]
```

- [ ] **Step 5: Implement `src/datanuggets/_rfast.py`**

```python
"""Rfast's arithmetic, reproduced exactly.

Rfast::Dist and Rfast::dista sum per-feature terms with Armadillo's accumulate (two
accumulators over even and odd positions, then added), so a plain left-to-right or
numpy pairwise sum rounds differently. Rfast::dista(k = 1, index = TRUE) picks the
nearest center with libstdc++ std::sort, which is not stable: among exactly tied
distances the winner is whatever std::sort leaves first, not the lowest index.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from datanuggets._defaults import ASSIGN_BLOCK_ROWS, DIST_BLOCK_ROWS

_S_THRESHOLD = 16  # libstdc++ _S_threshold


def _arma_accumulate(term: Callable[[int], np.ndarray], p: int) -> np.ndarray:
    acc1 = acc2 = None
    i, j = 0, 1
    while j < p:
        t_i, t_j = term(i), term(j)
        acc1 = t_i if acc1 is None else np.add(acc1, t_i, out=acc1)
        acc2 = t_j if acc2 is None else np.add(acc2, t_j, out=acc2)
        i += 2
        j += 2
    if i < p:
        t_i = term(i)
        acc1 = t_i if acc1 is None else np.add(acc1, t_i, out=acc1)
    if acc2 is None:
        return acc1
    return np.add(acc1, acc2, out=acc1)


def arma_sum_last(terms: np.ndarray) -> np.ndarray:
    """Sum over the last axis in Armadillo's order."""
    terms = np.asarray(terms, dtype=np.float64)
    return _arma_accumulate(lambda j: terms[..., j].copy(), terms.shape[-1])


def cross_distances(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Distances between the rows of a (m x p) and b (k x p), as Rfast computes them."""
    if metric == "euclidean":

        def term(j: int) -> np.ndarray:
            d = b[None, :, j] - a[:, None, j]
            return d * d

    else:

        def term(j: int) -> np.ndarray:
            return np.abs(b[None, :, j] - a[:, None, j])

    total = _arma_accumulate(term, a.shape[1])
    if metric == "euclidean":
        np.sqrt(total, out=total)
    return total


def dist_matrix(x: np.ndarray, metric: str, block_rows: int = DIST_BLOCK_ROWS) -> np.ndarray:
    """Rfast::Dist(x, method = metric): full n x n matrix, built in row blocks."""
    n = x.shape[0]
    out = np.empty((n, n))
    for start in range(0, n, block_rows):
        out[start : start + block_rows] = cross_distances(x[start : start + block_rows], x, metric)
    return out


def nearest_index(
    points: np.ndarray, centers: np.ndarray, metric: str, block_rows: int = ASSIGN_BLOCK_ROWS
) -> np.ndarray:
    """Rfast::dista(points, centers, type = metric, k = 1, index = TRUE), 0-based."""
    n = points.shape[0]
    out = np.empty(n, dtype=np.intp)
    for start in range(0, n, block_rows):
        d = cross_distances(points[start : start + block_rows], centers, metric)
        best = d.argmin(axis=1)
        mins = d[np.arange(d.shape[0]), best]
        tied = np.count_nonzero(d == mins[:, None], axis=1) > 1
        for r in np.flatnonzero(tied):
            best[r] = gnu_sort_first(d[r])
        out[start : start + d.shape[0]] = best
    return out


def gnu_sort_first(values, *, depth_limit: int | None = None) -> int:
    """The index std::sort places first: gnu_sort_order(values)[0], in O(k) instead of O(k log k).

    std::sort loops on the left part of each partition and recurses into the right part, and
    those recursions never touch the left part, so only the partitions along the left edge
    decide the head. The leftmost segment always holds a minimal element (the pivot stays on
    the left), and the final insertion sort is stable, so the head is that segment's first
    minimal element. If the depth limit hits zero on the left edge, the heap sort decides it.
    """
    v = np.asarray(values, dtype=np.float64).tolist()
    n = len(v)
    a = list(range(n))
    depth = 2 * (n.bit_length() - 1) if depth_limit is None else depth_limit
    last = n
    while last > _S_THRESHOLD:
        if depth == 0:
            _heap_sort(a, v, 0, last)
            return a[0]
        depth -= 1
        last = _unguarded_partition_pivot(a, v, 0, last)
    head = min(v[i] for i in a[:last])
    return next(i for i in a[:last] if v[i] == head)


def gnu_sort_order(values, *, depth_limit: int | None = None) -> np.ndarray:
    """Order of indices 0..k-1 after libstdc++ std::sort with comparator v[i] < v[j].

    Reproduces introsort (median-of-three pivot, unguarded partition, heapsort fallback)
    followed by the final insertion sort, step for step, so ties come out as in R.
    """
    v = np.asarray(values, dtype=np.float64).tolist()
    a = list(range(len(v)))
    n = len(a)
    if n > 1:
        depth = 2 * (n.bit_length() - 1) if depth_limit is None else depth_limit
        _introsort_loop(a, v, 0, n, depth)
        _final_insertion_sort(a, v, 0, n)
    return np.asarray(a, dtype=np.intp)


def _introsort_loop(a, v, first, last, depth):
    while last - first > _S_THRESHOLD:
        if depth == 0:
            _heap_sort(a, v, first, last)
            return
        depth -= 1
        cut = _unguarded_partition_pivot(a, v, first, last)
        _introsort_loop(a, v, cut, last, depth)
        last = cut


def _move_median_to_first(a, v, result, i, j, k):
    if v[a[i]] < v[a[j]]:
        if v[a[j]] < v[a[k]]:
            t = j
        elif v[a[i]] < v[a[k]]:
            t = k
        else:
            t = i
    elif v[a[i]] < v[a[k]]:
        t = i
    elif v[a[j]] < v[a[k]]:
        t = k
    else:
        t = j
    a[result], a[t] = a[t], a[result]


def _unguarded_partition_pivot(a, v, first, last):
    mid = first + (last - first) // 2
    _move_median_to_first(a, v, first, first + 1, mid, last - 1)
    pivot = v[a[first]]
    lo, hi = first + 1, last
    while True:
        while v[a[lo]] < pivot:
            lo += 1
        hi -= 1
        while pivot < v[a[hi]]:
            hi -= 1
        if not lo < hi:
            return lo
        a[lo], a[hi] = a[hi], a[lo]
        lo += 1


def _final_insertion_sort(a, v, first, last):
    if last - first > _S_THRESHOLD:
        _insertion_sort(a, v, first, first + _S_THRESHOLD)
        for i in range(first + _S_THRESHOLD, last):
            _unguarded_linear_insert(a, v, i)
    else:
        _insertion_sort(a, v, first, last)


def _insertion_sort(a, v, first, last):
    if first == last:
        return
    for i in range(first + 1, last):
        val = a[i]
        if v[val] < v[a[first]]:
            a[first + 1 : i + 1] = a[first:i]
            a[first] = val
        else:
            _unguarded_linear_insert(a, v, i)


def _unguarded_linear_insert(a, v, last):
    val = a[last]
    nxt = last - 1
    while v[val] < v[a[nxt]]:
        a[last] = a[nxt]
        last = nxt
        nxt -= 1
    a[last] = val


def _heap_sort(a, v, first, last):
    """std::__partial_sort(first, last, last): make_heap, then sort_heap."""
    n = last - first
    if n >= 2:
        parent = (n - 2) // 2
        while True:
            _adjust_heap(a, v, first, parent, n, a[first + parent])
            if parent == 0:
                break
            parent -= 1
    while last - first > 1:
        last -= 1
        val = a[last]
        a[last] = a[first]
        _adjust_heap(a, v, first, 0, last - first, val)


def _adjust_heap(a, v, first, hole, length, val):
    top = hole
    child = hole
    while child < (length - 1) // 2:
        child = 2 * (child + 1)
        if v[a[first + child]] < v[a[first + child - 1]]:
            child -= 1
        a[first + hole] = a[first + child]
        hole = child
    if length % 2 == 0 and child == (length - 2) // 2:
        child = 2 * (child + 1)
        a[first + hole] = a[first + child - 1]
        hole = child - 1
    parent = (hole - 1) // 2
    while hole > top and v[a[first + parent]] < v[val]:
        a[first + hole] = a[first + parent]
        hole = parent
        parent = (hole - 1) // 2
    a[first + hole] = val
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_rcompat.py tests/test_rfast.py -q`
Expected: all pass (1 + 1 + 8 + 6 + 6 + 1 + 1 + 2 + 3 + 1 = 30 tests).

If `test_dist_matrix_is_bitwise_equal_to_rfast` fails only for p >= 3 cases, Armadillo's summation order differs from the even/odd model: compare against a left-to-right sum (`acc = t0; acc += t1; ...`) and keep whichever matches all eight cases, then update the module docstring. If `test_gnu_sort_order_reproduces_rfast_full_order` fails, print the first mismatching row with its distances and compare against the libstdc++ `bits/stl_algo.h` of the GCC in Rtools for R 4.6 before changing anything.

- [ ] **Step 7: Record the findings in the spec**

In `docs/specs/2026-09-29-datanuggets-python-port-design.md`, section 6, `create_dn`, make three replacements:

- In item 6, `full pairwise distance matrix (Rfast \`Dist\`)` becomes `full pairwise distance matrix (Rfast \`Dist\`, whose per-feature sums use Armadillo's even/odd two-accumulator order; \`_rfast.dist_matrix\` reproduces it bit for bit)`.
- In item 6, `(Rfast \`rowMins\`, first index on ties (verify))` becomes `(Rfast \`rowMins\`, first index on ties, pinned by the \`rcompat_rowmins\` fixture)`.
- Item 7 becomes:

```markdown
7. Assignment: nearest final center for every observation via `Rfast::dista(k = 1, index = TRUE)`. Rfast sorts the whole distance row with libstdc++ `std::sort` (not stable), so among exactly tied distances the winner is whichever index `std::sort` leaves first: the first index when there are at most 16 centers, otherwise an index decided by introsort's partitions. `_rfast.nearest_index` uses argmin when the minimum is unique and replays `std::sort` (`gnu_sort_first`, O(k)) only for tied rows. Verified against R's complete sort order in `rcompat_dista`.
```

- [ ] **Step 8: Commit**

```powershell
git add src/datanuggets/_rcompat.py src/datanuggets/_rfast.py tests/test_rcompat.py tests/test_rfast.py docs/specs
git commit -m "Reproduce splitIndices and Rfast distance/tie semantics exactly" -m "Rfast sums squared differences with Armadillo's even/odd accumulators and breaks nearest-center ties with libstdc++ std::sort, which is not stable. Both are reproduced step for step and checked against R bit for bit, including R's complete sort order on tie-heavy data."
```

---

### Task 4: `create_dn_centers`

**Files:**
- Create: `src/datanuggets/exceptions.py`, `src/datanuggets/_validate.py`, `src/datanuggets/result.py`, `src/datanuggets/centers.py`, `tests/test_centers.py`
- Modify: `src/datanuggets/__init__.py`

**Interfaces:**
- Consumes: `_rfast.dist_matrix(x, metric)`.
- Produces: `DataNuggetWarning(UserWarning)`; `as_matrix(x, name="x") -> np.ndarray` (2-D C-contiguous float64, finite); `check_choice(name, value, choices)`; `as_count(name, value) -> int`; `check_delete_percent(value) -> float`; `CentersResult(centers: np.ndarray, kept: np.ndarray)`; `DataNuggets(centers, weights, scales, assignments)` with `.n_nuggets` and `.to_frame()`; `select_centers(rs, dn_num, delete_percent, dist_metric) -> tuple[np.ndarray, bool]` (kept 0-based, True when the input was returned unchanged); `create_dn_centers(rs, dn_num, *, delete_percent=0.1, dist_metric="euclidean") -> CentersResult`.

- [ ] **Step 1: Write the failing tests `tests/test_centers.py`**

```python
import warnings

import numpy as np
import pytest
from rfixtures import load_group

from datanuggets import DataNuggetWarning, create_dn_centers

CASES = load_group("create_dn")
CALLS = [(c, i) for c in CASES for i in range(1, c.manifest["n_calls"] + 1)]


@pytest.mark.parametrize(("case", "i"), CALLS, ids=[f"{c.id}-call{i:02d}" for c, i in CALLS])
def test_create_dn_centers_matches_r(case, i):
    rs = case.array(f"call{i:02d}_rs")
    dn_num = int(case.manifest["call_dn_num"][i - 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataNuggetWarning)
        out = create_dn_centers(
            rs,
            dn_num,
            delete_percent=case.args["delete.percent"],
            dist_metric=case.args["dist.metric"],
        )
    np.testing.assert_array_equal(out.kept + 1, case.array(f"call{i:02d}_kept"))
    np.testing.assert_array_equal(out.centers, rs[out.kept])


def test_returns_input_with_warning_when_dn_num_covers_all_rows():
    rs = np.arange(12.0).reshape(6, 2)
    with pytest.warns(DataNuggetWarning, match="greater than or equal"):
        out = create_dn_centers(rs, 6)
    np.testing.assert_array_equal(out.kept, np.arange(6))
    np.testing.assert_array_equal(out.centers, rs)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"delete_percent": 0.0}, ValueError, r"delete_percent must be within \(0, 1\)"),
        ({"delete_percent": 1.0}, ValueError, r"delete_percent must be within \(0, 1\)"),
        ({"dist_metric": "cosine"}, ValueError, "dist_metric must be one of"),
    ],
)
def test_rejects_bad_arguments(kwargs, error, match):
    with pytest.raises(error, match=match):
        create_dn_centers(np.zeros((10, 2)), 3, **kwargs)


def test_rejects_bad_dn_num():
    with pytest.raises(ValueError, match="dn_num must be a whole number >= 1"):
        create_dn_centers(np.zeros((10, 2)), 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_centers.py -q`
Expected: `ImportError: cannot import name 'DataNuggetWarning' from 'datanuggets'`.

- [ ] **Step 3: Implement `src/datanuggets/exceptions.py`**

```python
"""Warnings raised by datanuggets."""


class DataNuggetWarning(UserWarning):
    """A condition R reports with warning(), e.g. asking for as many nuggets as rows."""
```

- [ ] **Step 4: Implement `src/datanuggets/_validate.py`**

```python
"""Input coercion and argument checks, mirroring the stop() conditions of the R code."""

from __future__ import annotations

import numbers

import numpy as np


def as_matrix(x, name: str = "x") -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain only numeric columns")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (a matrix or data frame), got {arr.ndim}-D")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column")
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def check_choice(name: str, value, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, choices))}, got {value!r}")
    return value


def as_real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def as_count(name: str, value) -> int:
    v = as_real(name, value)
    if not v.is_integer() or v < 1:
        raise ValueError(f"{name} must be a whole number >= 1, got {value!r}")
    return int(v)


def check_delete_percent(value) -> float:
    v = as_real("delete_percent", value)
    if not 0 < v < 1:
        raise ValueError(f"delete_percent must be within (0, 1), got {value!r}")
    return v
```

- [ ] **Step 5: Implement `src/datanuggets/result.py`**

```python
"""Result containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _freeze(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CentersResult:
    """create_dn_centers output: the kept rows and their 0-based positions in the input."""

    centers: np.ndarray
    kept: np.ndarray

    def __post_init__(self):
        _freeze(self.centers)
        _freeze(self.kept)


@dataclass(frozen=True)
class DataNuggets:
    """Data nuggets: one center, weight and scale per nugget, plus each row's nugget (0-based)."""

    centers: np.ndarray
    weights: np.ndarray
    scales: np.ndarray
    assignments: np.ndarray

    def __post_init__(self):
        for arr in (self.centers, self.weights, self.scales, self.assignments):
            _freeze(arr)

    @property
    def n_nuggets(self) -> int:
        return self.centers.shape[0]

    def to_frame(self):
        """R's "Data Nuggets" table: Data Nugget, Center1..Centerp, Weight, Scale (ids 0-based)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame() needs pandas: pip install 'datanuggets[frame]'") from exc
        p = self.centers.shape[1]
        df = pd.DataFrame(self.centers, columns=[f"Center{j + 1}" for j in range(p)])
        df.insert(0, "Data Nugget", np.arange(self.n_nuggets))
        df["Weight"] = self.weights
        df["Scale"] = self.scales
        return df
```

- [ ] **Step 6: Implement `src/datanuggets/centers.py`**

```python
"""create.DNcenters: reduce a sample to data nugget centers by deleting close pairs."""

from __future__ import annotations

import math
import warnings

import numpy as np

from datanuggets._defaults import DELETE_PERCENT, DIST_METRICS
from datanuggets._rfast import dist_matrix
from datanuggets._validate import as_count, as_matrix, check_choice, check_delete_percent
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import CentersResult

PASSTHROUGH_WARNING = (
    "DN.num is greater than or equal to the number of rows. Returning original data"
)


def select_centers(
    rs: np.ndarray, dn_num: int, delete_percent: float, dist_metric: str
) -> tuple[np.ndarray, bool]:
    """Kept row positions (0-based) and whether the input came back unchanged. Inputs pre-validated."""
    n_obs = rs.shape[0]
    keep = np.arange(n_obs)
    if n_obs <= dn_num:
        return keep, True

    d = dist_matrix(rs, dist_metric)
    np.fill_diagonal(d, d.max() + 1)
    tmp_num = n_obs
    while tmp_num > dn_num:
        delete_num = max(1, math.floor(tmp_num * delete_percent))
        delete_num = min(delete_num, tmp_num - dn_num)

        nn_index = d.argmin(axis=1)  # Rfast::rowMins: first index on ties
        nn_dist = d[np.arange(tmp_num), nn_index]
        order = np.argsort(nn_dist, kind="stable").tolist()  # R's order() is stable
        nn = nn_index.tolist()

        marked = [False] * tmp_num
        taken = 0
        for p in order:
            if taken >= delete_num:
                break
            if not marked[nn[p]] and not marked[p]:
                marked[p] = True
                taken += 1
        if taken < delete_num:
            for p in order:
                if taken >= delete_num:
                    break
                if not marked[p]:
                    marked[p] = True
                    taken += 1

        keep_mask = ~np.asarray(marked)
        keep = keep[keep_mask]
        d = d[np.ix_(keep_mask, keep_mask)]
        tmp_num = keep.size
    return keep, False


def create_dn_centers(
    rs, dn_num, *, delete_percent: float = DELETE_PERCENT, dist_metric: str = "euclidean"
) -> CentersResult:
    """Port of R's create.DNcenters: repeatedly delete one point of each closest pair."""
    delete_percent = check_delete_percent(delete_percent)
    check_choice("dist_metric", dist_metric, DIST_METRICS)
    dn_num = as_count("dn_num", dn_num)
    x = as_matrix(rs, "rs")
    kept, passthrough = select_centers(x, dn_num, delete_percent, dist_metric)
    if passthrough:
        warnings.warn(PASSTHROUGH_WARNING, DataNuggetWarning, stacklevel=2)
    return CentersResult(centers=x[kept], kept=kept)
```

- [ ] **Step 7: Export from `src/datanuggets/__init__.py`**

```python
"""Native Python port of the R packages datanugget and WCluster."""

from datanuggets.centers import create_dn_centers
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import CentersResult, DataNuggets

__version__ = "0.1.0.dev0"

__all__ = ["CentersResult", "DataNuggetWarning", "DataNuggets", "create_dn_centers"]
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_centers.py -q`
Expected: all pass (94 parity tests, one per captured call, plus 5 others).

- [ ] **Step 9: Commit**

```powershell
git add src/datanuggets tests/test_centers.py
git commit -m "Port create.DNcenters with exact parity on every captured call" -m "Each create.DNcenters call R made inside create.DN is replayed on R's exact input; the kept rows match R index for index. Rows are visited in stable order and nearest neighbors use first-index ties, as Rfast::rowMins does."
```

---

### Task 5: `create_dn`

**Files:**
- Create: `src/datanuggets/create.py`, `tests/test_create.py`
- Modify: `src/datanuggets/__init__.py`

**Interfaces:**
- Consumes: `as_draws`, `split_indices`, `nearest_index`, `select_centers`, `PASSTHROUGH_WARNING`, validators, `DataNuggets`.
- Produces: `create_dn(x, center_method="mean", r=5000, delete_percent=0.1, dn_num1=10_000, dn_num2=2000, dist_metric="euclidean", seed=291102, n_jobs=1, progress=False) -> DataNuggets`.

- [ ] **Step 1: Write the failing tests `tests/test_create.py`**

```python
import warnings

import numpy as np
import pandas as pd
import pytest
from rfixtures import ATOL, RTOL, load_group

from datanuggets import DataNuggetWarning, create_dn
from datanuggets._draws import ReplayDraws

CASES = load_group("create_dn")


def py_args(args: dict) -> dict:
    return {
        "center_method": args["center.method"],
        "r": args["R"],
        "delete_percent": args["delete.percent"],
        "dn_num1": args["DN.num1"],
        "dn_num2": args["DN.num2"],
        "dist_metric": args["dist.metric"],
    }


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_create_dn_matches_r(case):
    draws = ReplayDraws(case.draws())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dn = create_dn(case.array("x"), **py_args(case.args), seed=draws)
    draws.assert_exhausted()
    np.testing.assert_array_equal(dn.assignments + 1, case.array("assignments"))
    np.testing.assert_array_equal(dn.weights, case.array("weights"))
    np.testing.assert_allclose(dn.centers, case.array("centers"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(dn.scales, case.array("scales"), rtol=RTOL, atol=ATOL)
    ours = [w for w in caught if issubclass(w.category, DataNuggetWarning)]
    assert len(ours) == len(case.manifest["warnings"])


QUAKES = next(c for c in CASES if c.id == "quakes_euclidean_mean")
SMALL = {"r": 300, "dn_num1": 400, "dn_num2": 100}


def test_n_jobs_does_not_change_result():
    x = QUAKES.array("x")
    a = create_dn(x, **SMALL, seed=11, n_jobs=1)
    b = create_dn(x, **SMALL, seed=11, n_jobs=2)
    for field in ("centers", "weights", "scales", "assignments"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_progress_does_not_change_result():
    x = QUAKES.array("x")
    a = create_dn(x, **SMALL, progress=False)
    b = create_dn(x, **SMALL, progress=True)
    np.testing.assert_array_equal(a.assignments, b.assignments)
    np.testing.assert_array_equal(a.centers, b.centers)


def test_integer_input_equals_float_input():
    x = np.rint(QUAKES.array("x") * 10)
    a = create_dn(x.astype(np.int64), **SMALL)
    b = create_dn(x, **SMALL)
    np.testing.assert_array_equal(a.assignments, b.assignments)
    np.testing.assert_array_equal(a.centers, b.centers)


def test_numeric_dataframe_equals_array():
    x = QUAKES.array("x")
    frame = pd.DataFrame(x, columns=["lat", "long", "depth", "mag", "stations"])
    a, b = create_dn(frame, **SMALL), create_dn(x, **SMALL)
    np.testing.assert_array_equal(a.assignments, b.assignments)


def test_non_numeric_frame_rejected():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="only numeric columns"):
        create_dn(frame, r=100, dn_num1=2, dn_num2=1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_and_inf_rejected(bad):
    x = QUAKES.array("x").copy()
    x[3, 1] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        create_dn(x, **SMALL)


def test_more_nuggets_than_rows_keeps_every_row():
    x = QUAKES.array("x")[:150]
    with pytest.warns(DataNuggetWarning):
        dn = create_dn(x, r=5000, dn_num1=300, dn_num2=300)
    assert dn.n_nuggets == 150
    assert dn.weights.sum() == 150
    assert np.isfinite(dn.scales).all()


def test_to_frame_matches_r_layout():
    frame = create_dn(QUAKES.array("x"), **SMALL).to_frame()
    assert list(frame.columns) == [
        "Data Nugget",
        "Center1",
        "Center2",
        "Center3",
        "Center4",
        "Center5",
        "Weight",
        "Scale",
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"center_method": "median"}, ValueError, "center_method must be one of"),
        ({"r": 99}, ValueError, r"r must be within \[100, 10000\]"),
        ({"r": 10_001}, ValueError, r"r must be within \[100, 10000\]"),
        ({"delete_percent": 1.0}, ValueError, r"delete_percent must be within \(0, 1\)"),
        ({"dn_num1": 10, "dn_num2": 20}, ValueError, "dn_num2 must be less than dn_num1"),
        ({"dist_metric": "cosine"}, ValueError, "dist_metric must be one of"),
        ({"seed": 1.5}, TypeError, "seed must be"),
        ({"n_jobs": 0}, ValueError, "n_jobs"),
    ],
)
def test_rejects_bad_arguments(kwargs, error, match):
    args = {"r": 100, "dn_num1": 20, "dn_num2": 10} | kwargs
    with pytest.raises(error, match=match):
        create_dn(QUAKES.array("x"), **args)


def test_rejects_single_row_and_1d_input():
    with pytest.raises(ValueError, match="at least two observations"):
        create_dn(np.zeros((1, 3)), r=100, dn_num1=2, dn_num2=1)
    with pytest.raises(ValueError, match="2-D"):
        create_dn(np.zeros(10), r=100, dn_num1=2, dn_num2=1)
    with pytest.raises(ValueError, match="at least one column"):
        create_dn(np.zeros((10, 0)), r=100, dn_num1=2, dn_num2=1)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_create.py -q`
Expected: `ImportError: cannot import name 'create_dn' from 'datanuggets'`.

- [ ] **Step 3: Implement `src/datanuggets/create.py`**

```python
"""create.DN: reduce a dataset to data nuggets (center, weight, scale)."""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
from joblib import Parallel, delayed

from datanuggets._defaults import (
    CENTER_METHODS,
    DELETE_PERCENT,
    DIST_METRICS,
    DN_NUM1,
    DN_NUM2,
    R_CHUNK,
    R_MAX,
    R_MIN,
    SEED,
)
from datanuggets._draws import Draws, as_draws
from datanuggets._rcompat import split_indices
from datanuggets._rfast import nearest_index
from datanuggets._validate import (
    as_count,
    as_matrix,
    as_real,
    check_choice,
    check_delete_percent,
)
from datanuggets.centers import PASSTHROUGH_WARNING, select_centers
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import DataNuggets

log = logging.getLogger("datanuggets")


def create_dn(
    x,
    center_method: str = "mean",
    r: float = R_CHUNK,
    delete_percent: float = DELETE_PERCENT,
    dn_num1: int = DN_NUM1,
    dn_num2: int = DN_NUM2,
    dist_metric: str = "euclidean",
    seed=SEED,
    n_jobs: int = 1,
    progress: bool = False,
) -> DataNuggets:
    """Port of R's create.DN (datanugget 1.5.0).

    The rows are shuffled once, split into ceiling(n / r) chunks, each chunk is reduced
    to its share of dn_num1 centers, the pooled centers are reduced to dn_num2, and every
    row is assigned to its nearest center. Nugget ids and assignments are 0-based.
    progress=True shows a tqdm bar over the stage-1 chunks (needs the `progress` extra).
    """
    check_choice("center_method", center_method, CENTER_METHODS)
    r = as_real("r", r)
    if not R_MIN <= r <= R_MAX:
        raise ValueError(f"r must be within [{R_MIN}, {R_MAX}], got {r!r}")
    delete_percent = check_delete_percent(delete_percent)
    dn_num1 = as_count("dn_num1", dn_num1)
    dn_num2 = as_count("dn_num2", dn_num2)
    if dn_num2 > dn_num1:
        raise ValueError("dn_num2 must be less than dn_num1")
    check_choice("dist_metric", dist_metric, DIST_METRICS)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be a positive int or -1, got {n_jobs!r}")
    bar = _progress_bar() if progress else None
    draws = as_draws(seed)
    x = as_matrix(x)
    n = x.shape[0]
    if n < 2:
        raise ValueError("x must contain at least two observations")

    perm = draws.permutation(n)
    chunks = [c for c in split_indices(n, math.ceil(n / r)) if c.size > 0]
    m_each = _allocate(dn_num1, np.array([c.size for c in chunks]))

    log.info("Creating initial set of data nugget centers...")

    def stage1(i: int) -> tuple[np.ndarray, bool]:
        rs = x[perm[chunks[i]]]
        kept, passthrough = select_centers(rs, int(m_each[i]), delete_percent, dist_metric)
        return rs[kept], passthrough

    ids = range(len(chunks)) if bar is None else bar(range(len(chunks)), desc="stage 1 chunks")
    if n_jobs == 1:
        parts = [stage1(i) for i in ids]
    else:
        parts = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(stage1)(i) for i in ids)
    for _, passthrough in parts:
        if passthrough:
            warnings.warn(PASSTHROUGH_WARNING, DataNuggetWarning, stacklevel=2)
    centers1 = np.vstack([c for c, _ in parts])

    log.info("Creating final set of data nugget centers...")
    kept2, passthrough = select_centers(centers1, dn_num2, delete_percent, dist_metric)
    if passthrough:
        warnings.warn(PASSTHROUGH_WARNING, DataNuggetWarning, stacklevel=2)
    stage2 = centers1[kept2]

    log.info("Assigning observations to the nearest data nuggets...")
    assignments = nearest_index(x, stage2, dist_metric)

    log.info("Recalculating the data nugget centers and calculating the data nugget scales...")
    centers, weights, scales = _summarize(x, assignments, stage2, center_method, draws)
    return DataNuggets(centers=centers, weights=weights, scales=scales, assignments=assignments)


def _progress_bar():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("progress=True needs tqdm: pip install 'datanuggets[progress]'") from exc
    return tqdm


def _allocate(dn_num1: int, chunk_sizes: np.ndarray) -> np.ndarray:
    """R: m.each <- DN.num1 %/% splits, remainder to the first chunks, clamped to [1, chunk size]."""
    s = chunk_sizes.size
    m = np.full(s, dn_num1 // s, dtype=np.int64)
    rem = dn_num1 - int(m.sum())
    if rem > 0:
        m[:rem] += 1
    return np.maximum(1, np.minimum(m, chunk_sizes))


def _summarize(
    x: np.ndarray, assignments: np.ndarray, stage2: np.ndarray, center_method: str, draws: Draws
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights, scales and final centers with R's formulas (one-pass sums on centred data)."""
    k, p = stage2.shape
    weights = np.bincount(assignments, minlength=k)

    g_mean = x.mean(axis=0)
    xc = x - g_mean
    s1 = np.zeros((k, p))
    s2 = np.zeros((k, p))
    np.add.at(s1, assignments, xc)  # R rowsum(): sequential, row order
    np.add.at(s2, assignments, xc * xc)
    n_safe = np.maximum(weights, 1)[:, None]
    mean_c = s1 / n_safe
    ss = s2 - n_safe * (mean_c * mean_c)
    ss[ss < 0] = 0.0
    var = ss / np.maximum(weights - 1, 1)[:, None]
    scales = var.mean(axis=1)
    scales[weights < 2] = np.nan

    if center_method == "mean":
        centers = mean_c + g_mean
        empty = weights == 0
        centers[empty] = stage2[empty]
    elif center_method == "original":
        centers = stage2.copy()
    else:
        order = np.argsort(assignments, kind="stable")
        bounds = np.concatenate(([0], np.cumsum(weights)))
        centers = np.empty((k, p))
        for i in range(k):
            members = order[bounds[i] : bounds[i + 1]]
            centers[i] = stage2[i] if members.size == 0 else x[members[draws.index(members.size)]]

    multi = scales[(weights > 1) & np.isfinite(scales)]
    floor = multi.min() / 100 if multi.size else np.finfo(np.float64).eps
    if not np.isfinite(floor) or floor <= 0:
        floor = np.finfo(np.float64).eps
    scales[~np.isfinite(scales)] = floor
    return centers, weights, scales
```

- [ ] **Step 4: Export `create_dn` from `src/datanuggets/__init__.py`**

```python
"""Native Python port of the R packages datanugget and WCluster."""

from datanuggets.centers import create_dn_centers
from datanuggets.create import create_dn
from datanuggets.exceptions import DataNuggetWarning
from datanuggets.result import CentersResult, DataNuggets

__version__ = "0.1.0.dev0"

__all__ = ["CentersResult", "DataNuggetWarning", "DataNuggets", "create_dn", "create_dn_centers"]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_create.py -q`
Expected: all pass (25 parity cases + 19 others).

If a parity case fails on `assignments` only, find the first differing row, print its distances to all stage-2 centers, and check whether the row is tied (then the issue is in `gnu_sort_order`) or whether stage-2 centers already differ (then compare `create_dn_centers` on that case's last `callNN_rs`). If only `scales` fail at around 1e-12, confirm the failure is outside rtol 1e-8 before touching the formula.

- [ ] **Step 6: Run the whole suite and lint**

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: all tests pass; ruff reports no errors. Fix formatting with `uv run ruff format src tests` if needed, then rerun both.

- [ ] **Step 7: Commit**

```powershell
git add src/datanuggets tests/test_create.py
git commit -m "Port create.DN with exact assignment and weight parity" -m "With R's permutation (and per-nugget picks for center.method = random) replayed, all 25 fixture cases match R: assignments and weights exactly, centers and scales to rtol 1e-8. Scales use R's one-pass formula on globally centred data so round-off matches."
```

---

### Task 6: Parity report, review, and push

**Files:**
- Create: `scripts/parity_report.py`, `docs/validation/create_dn.md` (generated)

**Interfaces:**
- Consumes: `tests/rfixtures.load_group`, `create_dn`, `ReplayDraws`.

- [ ] **Step 1: Implement `scripts/parity_report.py`**

```python
"""Write docs/validation/create_dn.md: Python vs R on every create_dn fixture."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from rfixtures import load_group  # noqa: E402

from datanuggets import create_dn  # noqa: E402
from datanuggets._draws import ReplayDraws  # noqa: E402


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / scale)) if a.size else 0.0


def main() -> None:
    rows = []
    for case in load_group("create_dn"):
        a = case.args
        draws = ReplayDraws(case.draws())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dn = create_dn(
                case.array("x"),
                center_method=a["center.method"],
                r=a["R"],
                delete_percent=a["delete.percent"],
                dn_num1=a["DN.num1"],
                dn_num2=a["DN.num2"],
                dist_metric=a["dist.metric"],
                seed=draws,
            )
        draws.assert_exhausted()
        same_assign = np.array_equal(dn.assignments + 1, case.array("assignments"))
        same_weight = np.array_equal(dn.weights, case.array("weights"))
        rows.append(
            f"| {case.id} | {case.array('x').shape[0]} x {case.array('x').shape[1]} | "
            f"{dn.n_nuggets} | {'yes' if same_assign else '**NO**'} | "
            f"{'yes' if same_weight else '**NO**'} | "
            f"{rel_err(dn.centers, case.array('centers')):.1e} | "
            f"{rel_err(dn.scales, case.array('scales')):.1e} |"
        )
    header = (
        "# create_dn parity with R\n\n"
        "Generated by `scripts/parity_report.py` from the fixtures in `tests/fixtures/create_dn/`"
        " (R datanugget 1.5.0). R's random draws are replayed, so every difference below is"
        " arithmetic, not randomness.\n\n"
        "| case | data | nuggets | assignments identical | weights identical |"
        " max rel. error, centers | max rel. error, scales |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    out = ROOT / "docs" / "validation" / "create_dn.md"
    out.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate and read the report**

```powershell
uv run python scripts/parity_report.py
Get-Content docs/validation/create_dn.md | Select-Object -First 12
```

Expected: 25 rows, every "identical" column `yes`, every error at most 1e-8.

- [ ] **Step 3: Commit**

```powershell
uv run ruff check scripts
git add scripts/parity_report.py docs/validation/create_dn.md
git commit -m "Add create_dn parity report against R" -m "One row per fixture case: whether assignments and weights are identical and the largest relative error in centers and scales. This is the table to show the advisor."
```

- [ ] **Step 4: Whole-branch review**

Dispatch one fresh reviewer (Opus) over `git diff main...feat/create-dn` with the spec and this plan. Fix confirmed findings, rerun `uv run pytest -q`, commit fixes separately.

- [ ] **Step 5: Push and open the PR**

Ask RON2K first: creating the GitHub repo and the first push of `main` fall under the push-to-main hard rail, so this step runs only after an explicit yes. Then:

```powershell
git log --grep=Claude-Session --oneline | Measure-Object | Select-Object -ExpandProperty Count
gh repo create ron2k1/datanuggets --private --source . --remote origin
git push -u origin main
git push -u origin feat/create-dn
gh pr create --base main --head feat/create-dn --title "Feature 1: create_dn with exact R parity" --body-file docs/validation/create_dn.md
```

Expected: the trailer count is `0` before pushing; the PR URL is printed.
