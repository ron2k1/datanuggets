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

# R's arithmetic order depends on the build: Rfast::dista's Euclidean sums go through BLAS ddot
# for more than 32 features, and colMeans/rowMeans accumulate in long double. Record both, so a
# fixture regenerated on another BLAS or on a platform without 80-bit long double shows it.
blas_id <- function() {
  lib <- sessionInfo()$BLAS
  if (nzchar(lib)) return(lib)
  dll <- file.path(R.home("bin"), "Rblas.dll")  # Windows ships R's reference BLAS as Rblas.dll
  if (file.exists(dll)) paste("Rblas.dll md5", unname(tools::md5sum(dll))) else "unknown"
}

versions <- function() list(
  R = R.version.string,
  platform = R.version$platform,
  datanugget = as.character(packageVersion("datanugget")),
  Rfast = as.character(packageVersion("Rfast")),
  lapack = La_version(),
  blas = blas_id(),
  longdouble_digits = .Machine$longdouble.digits
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


# The draws capture_draws recorded, as the three arrays tests/rfixtures.py reads back.
draw_arrays <- function(draws) list(
  draw_n = as.integer(vapply(draws, function(d) as.numeric(d$n), numeric(1))),
  draw_size = as.integer(vapply(draws, function(d) as.numeric(d$size), numeric(1))),
  draw_values = as.integer(unlist(lapply(draws, function(d) d$value)))
)
