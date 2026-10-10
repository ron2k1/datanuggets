# Times WCluster's Whclust and Wkmeans on quakes, the R column of the speed gates in
# docs/specs/2026-10-09-wcluster-iterative-port-design.md section 10, and writes each workload's
# inputs, outputs, Wkmeans' draws and the elapsed seconds under tmp/wcluster/out/ for
# benchmarks/wcluster.py to compare. The rows are quakes[1:n, ] with the weights 1, 3, 7, 2
# repeating, the workloads that gave the spec's R column. Run from the repo root under pin.py:
#   Rscript benchmarks/wcluster.R whclust200 | whclust400 | whclust800 | wkmeans
# whclust800 is the long one: 11 minutes on an M4, 49 on the laptop when it was short of memory.
source("r/fixture_io.R")

workload <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(workload %in% c("whclust200", "whclust400", "whclust800", "wkmeans"))

inputs <- function(n) list(x = as_dbl(quakes[seq_len(n), ]), w = rep(c(1, 3, 7, 2), length.out = n))

write_out <- function(name, arrays, extra) {
  dir <- file.path("tmp", "wcluster", "out", name)
  if (dir.exists(dir)) unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE)
  meta <- Map(function(nm, v) write_array(dir, nm, v), names(arrays), arrays)
  write_json(c(list(case = name, arrays = meta, versions = versions()), extra),
             file.path(dir, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

# Elapsed seconds of expr after a gc, as covertype.R times create.DN
timed <- function(expr) {
  invisible(gc())
  t0 <- proc.time()[["elapsed"]]
  value <- expr
  list(value = value, secs = proc.time()[["elapsed"]] - t0)
}

if (startsWith(workload, "whclust")) {
  n <- as.integer(sub("whclust", "", workload))
  a <- inputs(n)
  r <- timed(WCluster::Whclust(a$x, a$w))
  hc <- unclass(r$value)
  write_out(workload, c(a, list(merge = hc$merge, height = hc$height, order = hc$order)),
            list(n = n, secs = r$secs))
  cat(sprintf("whclust n=%4d  %9.1f s\n", n, r$secs))
} else {
  # Wkmeans' defaults: k = 4, one start, 10 iterations, seed 291102. Wkmeans sets the seed
  # itself, so every run is the same: the time is the best of 3 untraced runs, as wcluster.py
  # takes its own, and one more run traces sample.int for the draws, untimed.
  for (n in c(100, 250, 500, 1000)) {
    a <- inputs(n)
    run <- function() {
      out <- NULL
      capture.output(out <- WCluster::Wkmeans(a$x, 4, NULL, a$w))
      out
    }
    runs <- lapply(1:3, function(i) timed(run()))
    secs <- min(vapply(runs, function(r) r$secs, 0))
    traced <- capture_draws(run())
    out <- traced$value
    stopifnot(all(vapply(runs, function(r) identical(r$value, out), TRUE)))
    write_out(paste0("wkmeans", n), c(a, draw_arrays(traced$draws), list(
      labels = as.integer(out[[1]]), centers = unname(out[[2]]),
      wwcss = unname(out[[3]]$WWCSS), total = out[[3]]$TotalWWCSS)),
      list(n = n, k = 4, secs = secs))
    cat(sprintf("wkmeans n=%4d  %9.3f s\n", n, secs))
  }
}
