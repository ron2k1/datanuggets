# Benchmark on UCI Covertype (10 continuous columns): create.DN with R's defaults, then
# refine.DN on its output with and without shape splits, single core so the draws can be
# traced. Records every draw and the outputs under tmp/covertype/out/, the same way the
# fixture generators do, for benchmarks/covertype.py to replay. Run from the repo root:
#   Rscript benchmarks/covertype.R 10000 | 100000 | all [noshape]
# The data is https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
# saved as tmp/covertype/covtype.data.gz (CC BY 4.0, sha256 below).
source("r/fixture_io.R")

DATA <- "tmp/covertype/covtype.data.gz"
SHA256 <- "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
stopifnot(file.exists(DATA), identical(unname(tools::sha256sum(DATA)), SHA256))

size <- commandArgs(trailingOnly = TRUE)[1]
shapes <- if ("noshape" %in% commandArgs(trailingOnly = TRUE)) FALSE else c(FALSE, TRUE)
X <- as_dbl(read.csv(gzfile(DATA), header = FALSE)[, 1:10])
if (size != "all") {
  set.seed(20260929)
  X <- X[sort(sample.int(nrow(X), as.integer(size))), , drop = FALSE]
}
id <- paste0("covertype_", size)

write_out <- function(name, arrays, extra) {
  dir <- file.path("tmp", "covertype", "out", name)
  if (dir.exists(dir)) unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE)
  meta <- Map(function(nm, v) write_array(dir, nm, v), names(arrays), arrays)
  write_json(c(list(case = name, arrays = meta, versions = versions()), extra),
             file.path(dir, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

# Runs expr with draws traced, messages and warnings collected, and elapsed seconds timed.
run <- function(expr) {
  msgs <- character()
  warns <- character()
  invisible(gc())
  t0 <- proc.time()[["elapsed"]]
  res <- capture_draws(withCallingHandlers(expr,
    message = function(m) {
      msgs <<- c(msgs, sub("\n$", "", conditionMessage(m)))
      invokeRestart("muffleMessage")
    },
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }))
  list(dn = res$value, draws = res$draws, secs = proc.time()[["elapsed"]] - t0,
       messages = msgs, warnings = warns)
}

dn_arrays <- function(dn, prefix = "") {
  tab <- dn[["Data Nuggets"]]
  a <- list(unname(as.matrix(tab[, 1 + seq_len(ncol(X))])), as.integer(tab[["Weight"]]),
            as.double(tab[["Scale"]]), as.integer(dn[["Data Nugget Assignments"]]))
  setNames(a, paste0(prefix, c("centers", "weights", "scales", "assignments")))
}

cr <- run(create.DN(X, no.cores = 1, make.pbs = FALSE))
write_out(paste0(id, "_create"), c(list(x = X), draw_arrays(cr$draws), dn_arrays(cr$dn)),
          list(n = nrow(X), secs = cr$secs, messages = I(cr$messages), warnings = I(cr$warnings)))
cat(sprintf("%s create.DN  %8.1f s  nuggets %d  draws %d  warnings %d\n", id, cr$secs,
            nrow(cr$dn[["Data Nuggets"]]), length(cr$draws), length(cr$warnings)))

for (shape in shapes) {
  rr <- run(refine.DN(X, cr$dn, shape.split = shape, no.cores = 1, make.pbs = FALSE))
  name <- paste0(id, if (shape) "_refine_shape" else "_refine")
  write_out(name, c(draw_arrays(rr$draws), dn_arrays(rr$dn)),
            list(n = nrow(X), shape.split = shape, secs = rr$secs, messages = I(rr$messages),
                 warnings = I(rr$warnings)))
  cat(sprintf("%s refine.DN  %8.1f s  nuggets %d -> %d  draws %d  shape=%s\n", id, rr$secs,
              nrow(cr$dn[["Data Nuggets"]]), nrow(rr$dn[["Data Nuggets"]]), length(rr$draws),
              shape))
}
