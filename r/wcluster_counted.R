# WCluster 1.3.0's iterative functions, statement for statement, plus counters. They only prove
# that a fixture case reaches the quirk it's named for: prove() checks a copy returns exactly
# what WCluster returns (or stops with the same message), and every case records WCluster's own
# output. Sourced after r/fixture_io.R.
W <- asNamespace("WCluster")

QUIRKS <- c("redraw", "revert", "fixed", "stale_decides", "recreate", "gap", "tb_na", "nan_test",
            "inf_pick", "self_merge", "self_merge_mid")

new_counts <- function() {
  cnt <- new.env()
  for (nm in QUIRKS) assign(nm, 0L, cnt)
  cnt
}
bump <- function(cnt, nm) assign(nm, get(nm, cnt) + 1L, cnt)

counted_init <- function(x, w = rep(1, nrow(x)), k, u = NULL, cnt) {
  clcenter <- u
  z = NULL
  zz = NULL
  n <- nrow(x)
  ff <- function(k, n) sample(c(1:k, sample(k, n - k, replace = TRUE)))
  while (length(unique(zz)) < k) {
    if (!is.null(zz)) bump(cnt, "redraw")
    if (is.null(u)) {
      y <- ff(k, n)
      while (min(table(y)) < 2) {
        bump(cnt, "redraw")
        y <- ff(k, n)
      }
      clcenter <- W$wmean(x, cl = y, w)
    }
    z = W$wdist_mat(x, centers = clcenter, w = w)
    zz = apply(z, 1, which.min)
    zz = setNames(zz, row.names(x))
    if (length(unique(zz)) < k & !is.null(u)) {
      stop("For the initialization, no data assigned to one(or multiple) of the clusters based on the weighted\n         distance to the initial cluster centers chosen by users.")
    }
  }
  return(zz)
}

counted_single <- function(x, y, u = NULL, w = rep(1, length(y)), k, K = 30, cnt) {
  if (missing(y)) y = counted_init(x, w, k, u, cnt)
  n = length(y)
  k <- max(y)
  tb <- table(y)
  yy = y2 = y
  for (l in 1:K) {
    uu0 <- uu00 <- W$wwcss(x, cl = yy, w)
    for (ll in 1:20) {
      yy0 <- yy
      z = NULL
      kcl_centers <- W$wmean(x, cl = yy0, w = w)
      z = W$wdist_mat(x, centers = kcl_centers, w = w)
      yy = apply(z, 1, which.min)
      uu0 = W$wwcss(x, cl = yy, w)
      if (is.na(uu0 > uu00)) bump(cnt, "nan_test")
      if (uu0 > uu00) {
        bump(cnt, "revert")
        yy = yy0
      }
      else if (uu0 <= uu00) {
        if (ll < 20 && identical(as.vector(yy), as.vector(yy0))) bump(cnt, "fixed")
        uu00 = uu0
      }
    }
    uu0 = uu00 = W$wwcss(x, cl = yy, w)
    for (i in 1:n) {
      for (j in 1:k) {
        y2 = yy
        if (y2[i] != j) {
          if (is.na(tb[y2[i]])) bump(cnt, "tb_na")
          else if ((tb[y2[i]] > 1) != (sum(yy == y2[i]) > 1)) bump(cnt, "stale_decides")
        }
        if (y2[i] != j & tb[y2[i]] > 1) {
          y2[i] = j
          uu1 = W$wwcss(x, y2, w)
          if (is.na(uu1 <= uu0)) bump(cnt, "nan_test")
          if (uu1 <= uu0) {
            if (!(j %in% yy)) bump(cnt, "recreate")
            uu0 = uu1
            yy = y2
            tb = table(yy)
          }
        }
      }
    }
    if (all(yy0 == yy))
      break
  }
  if (length(unique(yy)) < max(yy)) bump(cnt, "gap")
  yy = setNames(yy, row.names(x))
  list(uu0, yy)
}

# Wkmeans' loop over starts, without its argument checks, prints and messages.
counted_wkmeans <- function(dataset, k, cl.centers = NULL, obs.weights = rep(1, nrow(dataset)),
                            num.init = 1, max.iterations = 10, seed = 291102, cnt) {
  dataset <- as.matrix(dataset)
  set.seed(seed)
  k = floor(k)
  num.init = floor(num.init)
  max.iterations = floor(max.iterations)
  if (!is.null(cl.centers)) num.init = 1
  obs.weights = as.vector(obs.weights)
  best.clusters = NULL
  for (U in 1:num.init) {
    result = counted_single(x = dataset, u = cl.centers, w = obs.weights, k = k,
                            K = max.iterations, cnt = cnt)
    clusterAss = result[[2]]
    wwcss_cl = result[[1]]
    if (is.null(best.clusters)) {
      best.clusters = clusterAss
      best.WWCSS.results = wwcss_cl
    }
    else if (best.WWCSS.results > wwcss_cl) {
      best.clusters = clusterAss
      best.WWCSS.results = wwcss_cl
    }
  }
  list(best.clusters, W$wwcss(x = dataset, cl = best.clusters, w = obs.weights, groupSum = TRUE))
}

# WCluster::Wkmeans with its prints captured, returning what counted_wkmeans returns.
wcluster_wkmeans <- function(...) {
  out <- NULL
  capture.output(out <- WCluster::Wkmeans(...))
  list(out[[1]], out[[3]])
}

counted_whclust <- function(x, w = rep(1, nrow(x)), cnt) {
  x = as.matrix(x)
  y = -(1:(n <- nrow(x)))
  kk = NULL
  y0 = sort(unique(y))
  m = n
  ws2 = rep(NA, n - 1)
  dd = W$distw(x, y, w)
  for (ii in 1:(n - 1)) {
    k = which.min(dd)
    if (length(k) == 1L && dd[k] == Inf) bump(cnt, "inf_pick")
    u = y0[k0 <- k %% m]
    v = y0[k1 <- k %/% m + if (k0 == 0) 0 else 1]
    if (k0 == 0) {
      u = y0[m]
      k0 <- m
    }
    if (k0 == k1) {
      bump(cnt, "self_merge")
      if (ii < n - 1) bump(cnt, "self_merge_mid")
    }
    kk = rbind(kk, c(u, v))
    y[y == u] = y[y == v] = ii
    y0 = sort(unique(y))
    m = m - 1
    if (ii < n - 1) {
      dd = dd[-c(k0, k1), -c(k0, k1)]
      newdd = cbind(rbind(dd, 0), Inf)
      newdd[m, 1:(m - 1)] = sapply(1:(m - 1), function(t) {
        W$distwij(y0[t], y0[m], y, x, w)
      })
      dd = newdd
    }
    ws2[ii] = W$wwcss(x, y, w)
  }
  list(merge = kk, height = ws2, order = W$forder(kk))
}

wcluster_whclust <- function(x, w) unclass(WCluster::Whclust(x, w))[c("merge", "height", "order")]

# Runs the counted copy f and WCluster's own g on the same arguments, quietly, and stops unless
# both return the same value or stop with the same message. Returns the counts and g's result,
# which is a string of class r_error when R stopped.
run_quiet <- function(fn, args) tryCatch(
  suppressMessages(withCallingHandlers(do.call(fn, args),
                                       warning = function(w) invokeRestart("muffleWarning"))),
  error = function(e) structure(conditionMessage(e), class = "r_error"))

prove <- function(id, f, g, args, quirks = character()) {
  cnt <- new_counts()
  mine <- run_quiet(f, c(args, list(cnt = cnt)))
  theirs <- run_quiet(g, args)
  if (!identical(unname(unclass(mine)), unname(unclass(theirs))))
    stop(id, ": the counted copy disagrees with WCluster")
  counts <- mget(QUIRKS, envir = cnt)
  for (q in quirks) if (counts[[q]] == 0L) stop(id, ": never reaches ", q)
  list(counts = counts, value = theirs)
}

# Counts alone, for searching: the counted copy without WCluster's run.
count_only <- function(f, args) {
  cnt <- new_counts()
  value <- run_quiet(f, c(args, list(cnt = cnt)))
  list(counts = mget(QUIRKS, envir = cnt), value = value)
}
