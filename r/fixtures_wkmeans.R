# Fixtures for WCluster's Wkmeans: the draws R made, each start's labels from init, each start's
# result from Wkmeans_single, and the answer, or the error R stops with. The counts
# r/wcluster_counted.R's copy took prove each case reaches the quirk it's named for. Seeds and
# weights for the quirk cases came from searches over set.seed(s) for the draws and
# set.seed(10000 + s) for the weights; the recorded arrays alone reproduce each case.
source("r/fixture_io.R")
source("r/wcluster_counted.R")

# init and Wkmeans_single are internal, so Wkmeans finds them in the namespace, where the trace
# goes. A start that stops leaves no return value, so it isn't recorded.
hook <- function(slot) substitute({
  v <- returnValue(NULL)
  if (!is.null(v)) .capture$S[[length(.capture$S) + 1L]] <- v
}, list(S = slot))

add_wkmeans <- function(id, x, k, w = rep(1, nrow(x)), centers = NULL, num.init = 1,
                        max.iterations = 10, seed = 291102, quirks = character()) {
  x <- as_dbl(x)
  w <- as.double(w)
  t0 <- Sys.time()
  args <- list(dataset = x, k = k, cl.centers = centers, obs.weights = w, num.init = num.init,
               max.iterations = max.iterations, seed = seed)
  pr <- prove(id, counted_wkmeans, wcluster_wkmeans, args, quirks)
  .capture$inits <- list()
  .capture$singles <- list()
  trace("init", where = W, print = FALSE, exit = hook(quote(inits)))
  trace("Wkmeans_single", where = W, print = FALSE, exit = hook(quote(singles)))
  res <- tryCatch(capture_draws(run_quiet(function() {
    out <- NULL
    capture.output(out <- WCluster::Wkmeans(x, k, centers, w, num.init, max.iterations, seed))
    out
  }, list())), finally = suppressMessages({
    untrace("init", where = W)
    untrace("Wkmeans_single", where = W)
  }))
  err <- if (inherits(res$value, "r_error")) as.character(res$value) else ""
  if (nzchar(err) != nzchar(if (inherits(pr$value, "r_error")) pr$value else ""))
    stop(id, ": the traced run and the proof disagree on whether R stops")
  arrays <- c(list(x = x, w = w), draw_arrays(res$draws))
  if (!is.null(centers)) arrays$centers_in <- t(centers)
  if (length(.capture$inits))
    arrays$init_labels <- do.call(rbind, lapply(.capture$inits, as.integer))
  if (length(.capture$singles)) {
    arrays$single_labels <- do.call(rbind, lapply(.capture$singles, function(s) as.integer(s[[2]])))
    arrays$single_wwcss <- vapply(.capture$singles, function(s) as.double(s[[1]]), 0)
  }
  if (!nzchar(err)) {
    out <- res$value
    if (!identical(unname(out[[1]]), unname(pr$value[[1]]))) stop(id, ": traced labels differ")
    arrays <- c(arrays, list(labels = as.integer(out[[1]]), centers = unname(out[[2]]),
                             wwcss = unname(out[[3]]$WWCSS), total = out[[3]]$TotalWWCSS))
  }
  write_case("wkmeans", id, list(k = k, num_init = num.init, max_iter = max.iterations,
                                 seed = seed), arrays, extra = list(error = err, counts = pr$counts))
  cnt <- unlist(pr$counts)[c("redraw", "revert", "fixed", "stale_decides", "recreate", "gap",
                             "tb_na", "nan_test")]
  cat(sprintf("wkmeans %-15s n=%3d draws=%3d starts=%d %5.1fs %s | %s\n", id, nrow(x),
              length(res$draws), length(.capture$inits), as.numeric(Sys.time() - t0, units = "secs"),
              if (nzchar(err)) gsub("\n", " ", err) else "ok",
              paste(names(cnt), cnt, sep = "=", collapse = " ")))
}

iris4 <- as_dbl(iris[, 1:4])
w1372 <- function(n) rep(c(1, 3, 7, 2), length.out = n)
grid <- as_dbl(expand.grid(0:3, 0:2))  # 12 integer points: ties everywhere
wset <- c(-0.73, 1.19, 2.41)  # weights of both signs, so a move can lower WWCSS by emptying
by_seed <- function(s, pool, n) {
  set.seed(10000 + s)
  sample(pool, n, TRUE)
}

add_wkmeans("iris_k3", iris4, 3)
add_wkmeans("iris_k3_init5", iris4, 3, num.init = 5)
add_wkmeans("quakes500_k4_w", quakes[1:500, ], 4, w = w1372(500))
add_wkmeans("ucenters", iris4, 3, centers = t(iris4[c(1, 51, 101), ]))
# Two equal centers: the second gets no rows, and R stops with its message for user centers
add_wkmeans("ucenters_empty", iris4, 3, centers = t(iris4[c(1, 1, 101), ]))
# Nine rows and k = 4: most draws leave a label with one row, so init draws again
add_wkmeans("small_k4", iris4[c(1:3, 51:53, 101:103), ], 4, quirks = "redraw")
add_wkmeans("k1", iris4, 1)
add_wkmeans("maxiter1", iris4, 3, max.iterations = 1)
add_wkmeans("one_col", faithful[, 1, drop = FALSE], 2)

# Weights of both signs make a Lloyd step raise WWCSS, which R reverts
add_wkmeans("revert", quakes[1:60, 1:2], 3, w = by_seed(1, wset, 60), max.iterations = 4,
            seed = 1, quirks = "revert")
# R's table of cluster sizes is refreshed only when a move is taken, so after the Lloyd steps
# it can say a cluster has more rows than it has, and that decides whether a move is tried
add_wkmeans("stale_tb", grid, 3, w = by_seed(15, wset, 12), max.iterations = 4, seed = 15,
            quirks = "stale_decides")
# A move into a label that has no rows left, which a stale table let empty
add_wkmeans("recreate", grid, 3, w = by_seed(4, wset, 12), max.iterations = 4, seed = 4,
            quirks = "recreate")
# The stale table is shorter than a row's label: tb[label] is NA and R stops
add_wkmeans("tb_na", grid, 3, w = by_seed(65, wset, 12), max.iterations = 4, seed = 65,
            quirks = "tb_na")
# A move empties a middle label and nothing refills it, so R returns labels with a gap
add_wkmeans("gap", grid, 3, w = by_seed(195, wset, 12), max.iterations = 4, seed = 195,
            quirks = "gap")
# Weights 1 and -1 can cancel in a cluster: its WWCSS is NaN and R's comparison stops
add_wkmeans("nan_stop", grid, 2, w = by_seed(1, c(-1, 1), 12), max.iterations = 4, seed = 1,
            quirks = "nan_test")
