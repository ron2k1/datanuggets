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
# WCluster stays unattached: the generators call WCluster:: and WCluster::: by name.
stopifnot(packageVersion("WCluster") == "1.3.0")

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
  WCluster = as.character(packageVersion("WCluster")),
  matprod = getOption("matprod"),
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

# Reads back an array write_array wrote: row-major little-endian bytes, in R's own layout.
read_array <- function(dir, name) {
  meta <- read_json(file.path(dir, "manifest.json"))$arrays[[name]]
  if (is.null(meta)) stop("no array ", name, " in ", dir)
  shape <- as.integer(unlist(meta$shape))
  int <- meta$dtype == "<i4"
  con <- file(file.path(dir, meta$file), "rb")
  on.exit(close(con))
  v <- readBin(con, if (int) "integer" else "double", n = prod(shape),
               size = if (int) 4L else 8L, endian = "little")
  stopifnot(length(v) == prod(shape))
  if (length(shape) == 1L) v else t(matrix(v, nrow = shape[2], ncol = shape[1]))
}

# Fail closed if read_array does not invert write_array.
local({
  d <- tempfile()
  dir.create(d)
  m <- matrix(c(1.5, -2, 3, 4, 5, 6), 2)
  meta <- list(arrays = list(m = write_array(d, "m", m), i = write_array(d, "i", 3:1)))
  write_json(meta, file.path(d, "manifest.json"), auto_unbox = TRUE, digits = NA)
  stopifnot(identical(read_array(d, "m"), m), identical(read_array(d, "i"), 3:1))
  unlink(d, recursive = TRUE)
})

no_args <- setNames(list(), character(0))

# Doubles from their 16 hex digits, bit for bit, so a NaN's sign and payload are what the case says.
from_bits <- function(h) vapply(h, function(s) readBin(
  as.raw(strtoi(substring(s, seq(15, 1, -2), seq(16, 2, -2)), 16L)), "double", n = 1L), 0,
  USE.NAMES = FALSE)
local(stopifnot(identical(writeBin(from_bits("7ff00000000007a2"), raw(), endian = "little"),
                          as.raw(c(0xa2, 0x07, 0, 0, 0, 0, 0xf0, 0x7f)))))  # R's NA, unchanged

# NaNs that tell the FPU's rules apart: payload 123 is below NA's 7a2, and the signaling one's
# significand is above NA's once quieted.
nan_pay <- from_bits("7ff8000000000123")
nan_pos <- from_bits("7ff8000000000000")
nan_neg <- from_bits("fff8000000000000")
snan_big <- from_bits("7ff4000000000000")

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
