# Fixtures for Hartigan-Wong k-means: R's own Fortran (stats:::C_kmns) from fixed starting
# centers (group kmns), and stats::kmeans end to end with its random draws (group kmeans).
source("r/fixture_io.R")

int_max <- .Machine$integer.max

# Calls C_kmns exactly as kmeans()'s do_one() does. Fails closed on an unexpected fault code.
add_direct <- function(id, x, rows, iter.max = 1000L, isteps = min(int_max, 50 * nrow(x)),
                       ifault = 0L) {
  x <- as_dbl(x)
  c0 <- x[rows, , drop = FALSE]
  m <- nrow(x)
  p <- ncol(x)
  k <- nrow(c0)
  Z <- .Fortran(stats:::C_kmns, x, as.integer(m), as.integer(p), centers = c0, as.integer(k),
                c1 = integer(m), c2 = integer(m), nc = integer(k), double(k), double(k),
                ncp = integer(k), D = double(m), iTran = c(as.integer(isteps), integer(k)),
                live = integer(k), iter = as.integer(iter.max), wss = double(k), ifault = 0L)
  if (Z$ifault != ifault) stop(id, ": expected ifault ", ifault, ", C_kmns gave ", Z$ifault)
  write_case("kmns", id,
             list(iter.max = as.integer(iter.max), isteps = as.integer(isteps), ifault = ifault),
             list(x = x, c0 = c0, c1 = Z$c1, centers = matrix(Z$centers, k), nc = Z$nc,
                  wss = Z$wss, iter = Z$iter, ifault = Z$ifault))
  cat(sprintf("kmns   %-20s ifault=%d iter=%d\n", id, Z$ifault, Z$iter))
}

iris4 <- iris[, 1:4]
f1 <- faithful[, 1, drop = FALSE]
add_direct("iris_k2", iris4, c(1, 51))
add_direct("iris_k3", iris4, c(1, 51, 101))
add_direct("iris_k5", iris4, c(3, 60, 77, 120, 140))
add_direct("iris_k3_bad_start", iris4, c(1, 2, 3))
add_direct("iris_k4_iter1", iris4, c(1, 2, 3, 4), iter.max = 1L, ifault = 2L)
add_direct("faithful_k2", faithful, c(10, 20))
add_direct("faithful_k4", faithful, c(1, 2, 3, 4))
add_direct("quakes_k3", quakes, c(5, 500, 900))
add_direct("quakes_k6", quakes, c(1, 100, 200, 300, 400, 500))
add_direct("faithful1d_k2", f1, c(1, 2))
add_direct("faithful1d_k3", f1, c(1, 2, 3))
add_direct("quakes_k4_isteps50", quakes, c(1, 2, 3, 4), isteps = 50L, ifault = 4L)
add_direct("iris_dup_start", iris4, c(1, 1), ifault = 1L)  # kmeans() stops: empty cluster
add_direct("iris_k1", iris4, 1, ifault = 3L)               # kmns.f needs 1 < k < m
add_direct("volcano_k3", scale(volcano), c(1, 40, 80))

add_kmeans <- function(id, x, k, nstart, seed = 291102, iter.max = 10L) {
  x <- as_dbl(x)
  warns <- character()
  err <- ""
  set.seed(seed)
  res <- capture_draws(tryCatch(
    withCallingHandlers(kmeans(x, k, iter.max = iter.max, nstart = nstart),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) {
      err <<- conditionMessage(e)
      NULL
    }))
  km <- res$value
  arrays <- c(list(x = x), draw_arrays(res$draws))
  if (!is.null(km)) {
    arrays <- c(arrays, list(cluster = km$cluster, centers = unname(km$centers),
                             withinss = km$withinss, size = km$size, iter = km$iter,
                             tot_withinss = km$tot.withinss))
  }
  write_case("kmeans", id, list(k = k, nstart = nstart, seed = seed, iter.max = iter.max),
             arrays, extra = list(error = err, warnings = I(warns)))
  cat(sprintf("kmeans %-20s draws=%d warnings=%d error=%s\n", id, length(res$draws),
              length(warns), err))
  invisible(list(draws = res$draws, error = err, warnings = warns))
}

petal <- as_dbl(iris[, 3:4])
add_kmeans("faithful_k2_n1", faithful, 2, 1)
add_kmeans("faithful_k2_n25", faithful, 2, 25)
add_kmeans("faithful_k2_n25.7", faithful, 2, 25.7)  # 2:25.7 runs 25 starts in all
add_kmeans("petal_k2_n25", petal, 2, 25)            # starts come from unique(x)
# nstart = 1 redraws from unique(x) when its two random rows are equal: find a seed that does
dup_seed <- Find(function(s) {
  set.seed(s)
  i <- sample.int(nrow(petal), 2L)
  all(petal[i[1], ] == petal[i[2], ])
}, seq_len(10000))
r <- add_kmeans("petal_k2_n1_dup", petal, 2, 1, seed = dup_seed)
stopifnot(length(r$draws) == 2L)
add_kmeans("faithful1d_k2_n25", f1, 2, 25)
add_kmeans("iris_k3_n5", iris4, 3, 5)
add_kmeans("quakes_k4_n3", quakes, 4, 3)
r <- add_kmeans("iris_k4_iter1", iris4, 4, 1, iter.max = 1L)
stopifnot(length(r$warnings) == 1L)
r <- add_kmeans("faithful_k2_n0", faithful, 2, 0)  # refine.DN with nstart = 0 hits this
stopifnot(nzchar(r$error), length(r$draws) == 0L)
r <- add_kmeans("petal_k149", petal, 149, 1)
stopifnot(r$error == "more cluster centers than distinct data points.", length(r$draws) == 1L)
r <- add_kmeans("faithful5_k5", faithful[1:5, ], 5, 1)  # k == nrow(x): kmns ifault 3
stopifnot(r$error == "number of cluster centres must lie between 1 and nrow(x)",
          length(r$draws) == 1L)
# Rounded normals tie often enough to cycle the quick-transfer stage past its 50 * m step cap
set.seed(4000040)
tied <- matrix(round(rnorm(12000) * 3, 1), ncol = 3)
r <- add_kmeans("tied3d_k4_qtran", tied, 4, 1, seed = 2, iter.max = 30L)
stopifnot(identical(r$warnings, "Quick-TRANSfer stage steps exceeded maximum (= 200000)"))
