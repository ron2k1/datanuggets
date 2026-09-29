# Fixtures for refine.DN: the input nuggets, every random draw, every stats::kmeans call (the data
# it clustered and the clusters it returned), the messages and warnings, and the refined nuggets.
source("r/fixture_io.R")

grid <- as_dbl(expand.grid(a = 0:5, b = 0:5))
datasets <- list(
  faithful = as_dbl(faithful),
  iris = as_dbl(iris[, 1:4]),
  quakes = as_dbl(quakes),
  petal = as_dbl(iris[, 3:4]),                                     # heavy duplicates
  faithful_1d = as_dbl(faithful[, "eruptions", drop = FALSE]),
  volcano_scaled = as_dbl(scale(volcano)),                         # p = 61
  grid_ties = grid[rep(seq_len(nrow(grid)), 4), , drop = FALSE]    # every point 4 times
)
# create.DN's R, DN.num1 and DN.num2 for the input nuggets: about 10 to 20 rows each
dn_sizes <- list(faithful = c(100, 90, 20), iris = c(100, 60, 12), quakes = c(300, 400, 50),
                 petal = c(100, 100, 12), faithful_1d = c(100, 90, 20),
                 volcano_scaled = c(100, 40, 8), grid_ties = c(100, 60, 12))

make_dn <- function(ds, center.method = "mean") {
  s <- dn_sizes[[ds]]
  suppressWarnings(suppressMessages(create.DN(datasets[[ds]], center.method = center.method,
    R = s[1], DN.num1 = s[2], DN.num2 = s[3], no.cores = 1, make.pbs = FALSE)))
}

refine_args <- function(EV.tol = 0.9, max.splits = 5, min.nugget.size = 2, shape.split = FALSE,
                        min.shape.size = 10, delta = 2, nstart = 25, seed = 291102) {
  list(EV.tol = EV.tol, max.splits = max.splits, min.nugget.size = min.nugget.size,
       shape.split = shape.split, min.shape.size = min.shape.size, delta = delta,
       nstart = nstart, seed = seed)
}

# refine.DN reaches stats::kmeans through `::`, which resolves the namespace binding. Trace it
# and the attached copy, recording the data each call clusters and the clusters it returns
# (NULL when kmeans stops with an error, which refine.DN catches).
capture_kmeans <- function(expr) {
  .capture$km <- list()
  hook <- quote(.capture$km[[length(.capture$km) + 1L]] <- list(
    x = as.matrix(x), cluster = returnValue(NULL)$cluster))
  envs <- list(asNamespace("stats"), as.environment("package:stats"))
  for (e in envs) trace("kmeans", exit = hook, print = FALSE, where = e)
  on.exit(for (e in envs) suppressWarnings(untrace("kmeans", where = e)), add = TRUE)
  value <- expr
  list(value = value, calls = .capture$km)
}

run_refine <- function(x, dn, args) {
  msgs <- character()
  warns <- character()
  res <- capture_kmeans(capture_draws(withCallingHandlers(
    do.call(refine.DN, c(list(x = x, DN = dn), args, list(no.cores = 1, make.pbs = FALSE))),
    message = function(m) {
      msgs <<- c(msgs, sub("\n$", "", conditionMessage(m)))
      invokeRestart("muffleMessage")
    },
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })))
  list(dn = res$value$value, draws = res$value$draws, km = res$calls, messages = msgs,
       warnings = warns)
}

# refine.DN's get_nugget_features on the input nuggets, before any split
nugget_stats <- function(x, a, k) {
  p <- ncol(x)
  out <- vapply(seq_len(k), function(i) {
    obs <- x[a == i, , drop = FALSE]
    m <- nrow(obs)
    if (m == 0L) return(c(rep(NA_real_, p), 0, 0, 0))
    ctr <- colMeans(obs)
    if (m == 1L) return(c(ctr, 0, 0, 0))
    S <- cov(obs)
    ev <- if (p == 1L) as.numeric(S) else eigen(S, symmetric = TRUE, only.values = TRUE)$values
    c(ctr, sum(diag(S)) / p, ev[1], if (length(ev) > 1L) ev[2] else ev[1])
  }, numeric(p + 3L))
  list(init_centers = t(out[seq_len(p), , drop = FALSE]), init_scale = out[p + 1L, ],
       init_ev1 = out[p + 2L, ], init_ev2 = out[p + 3L, ])
}

km_arrays <- function(km) list(
  km_nrow = as.integer(vapply(km, function(k) as.numeric(nrow(k$x)), numeric(1))),
  km_ncol = as.integer(vapply(km, function(k) as.numeric(ncol(k$x)), numeric(1))),
  km_ok = as.integer(vapply(km, function(k) !is.null(k$cluster), logical(1))),
  km_x = as.double(unlist(lapply(km, function(k) t(k$x)))),   # row-major, calls back to back
  km_cluster = as.integer(unlist(lapply(km, function(k) k$cluster)))
)

all_paths <- list()
add_refine <- function(id, ds, dn, input, args = refine_args()) {
  force(dn)  # a lazy make_dn() argument would otherwise run, and draw, inside capture_draws
  x <- datasets[[ds]]
  p <- ncol(x)
  r <- run_refine(x, dn, args)
  n_rounds <- sum(grepl("^split #", r$messages))
  if (length(r$km) < n_rounds) {
    stop(id, ": ", n_rounds, " split rounds but only ", length(r$km), " kmeans calls recorded")
  }
  ok <- vapply(r$km, function(k) !is.null(k$cluster), logical(1))
  mns <- floor(args$min.nugget.size)
  paths <- list(
    kmeans_error = any(!ok),
    child_too_small = any(vapply(r$km[ok], function(k) min(tabulate(k$cluster, 2L)) < mns,
                                 logical(1))))
  in_tab <- dn[["Data Nuggets"]]
  in_a <- as.integer(dn[["Data Nugget Assignments"]])
  tab <- r$dn[["Data Nuggets"]]
  arrays <- c(
    list(x = x,
         in_centers = unname(as.matrix(in_tab[, 1 + seq_len(p)])),
         in_weights = as.integer(in_tab[["Weight"]]),
         in_scales = as.double(in_tab[["Scale"]]),
         in_assignments = in_a),
    nugget_stats(x, in_a, nrow(in_tab)),
    draw_arrays(r$draws),
    km_arrays(r$km),
    list(centers = unname(as.matrix(tab[, 1 + seq_len(p)])),
         weights = as.integer(tab[["Weight"]]),
         scales = as.double(tab[["Scale"]]),
         assignments = as.integer(r$dn[["Data Nugget Assignments"]])))
  write_case("refine_dn", id, args, arrays, extra = list(
    dataset = ds, input = input, messages = I(r$messages), warnings = I(r$warnings),
    paths = paths))
  all_paths[[id]] <<- paths
  cat(sprintf("%-40s nuggets %3d -> %3d  kmeans=%3d draws=%4d%s%s\n", id, nrow(in_tab),
              nrow(tab), length(r$km), length(r$draws),
              if (paths$kmeans_error) "  kmeans-error" else "",
              if (paths$child_too_small) "  small-child" else ""))
  invisible(r)
}

# The spec's grid on faithful: ev_tol x max_splits x min_nugget_size x shape_split x delta x nstart
dn_f <- make_dn("faithful")
for (ev in c(0.5, 0.9)) for (ms in c(0, 1, 5)) for (mns in c(2, 5)) for (shape in c(FALSE, TRUE))
  for (d in c(1.5, 2)) for (ns in c(1, 25)) {
    id <- sprintf("faithful_ev%s_ms%d_mns%d_%s_d%s_ns%d", ev, ms, mns,
                  if (shape) "shape" else "noshape", d, ns)
    add_refine(id, "faithful", dn_f, "mean", refine_args(
      EV.tol = ev, max.splits = ms, min.nugget.size = mns, shape.split = shape, delta = d,
      nstart = ns))
  }

for (ds in setdiff(names(datasets), "faithful")) {
  dn <- make_dn(ds)
  add_refine(paste0(ds, "_default"), ds, dn, "mean")
  add_refine(paste0(ds, "_shape"), ds, dn, "mean", refine_args(shape.split = TRUE))
  add_refine(paste0(ds, "_shape_loose"), ds, dn, "mean",
             refine_args(EV.tol = 0.5, shape.split = TRUE, delta = 1.5))
  add_refine(paste0(ds, "_nstart1"), ds, dn, "mean", refine_args(nstart = 1))
}

# kmeans(nstart = 0) stops with "argument is of length zero", so every split fails
add_refine("faithful_nstart0", "faithful", dn_f, "mean", refine_args(nstart = 0))
# refine.DN recomputes every center, so the input's center method must not matter
add_refine("faithful_input_random", "faithful", make_dn("faithful", "random"), "random")
add_refine("faithful_input_original", "faithful", make_dn("faithful", "original"), "original")
# max.splits and min.nugget.size are floored, min.shape.size is not, nstart runs floor(nstart)
add_refine("faithful_floored_args", "faithful", dn_f, "mean", refine_args(
  max.splits = 2.7, min.nugget.size = 2.5, shape.split = TRUE, min.shape.size = 10.5,
  delta = 1.5, nstart = 25.7))
# refine.DN applied to its own output
chain_in <- run_refine(datasets$faithful, dn_f, refine_args(shape.split = TRUE))$dn
add_refine("faithful_chain", "faithful", chain_in, "refined", refine_args(shape.split = TRUE))
# An empty nugget in the middle (create.DN keeps empty nuggets): refine drops it and renumbers
k_f <- nrow(dn_f[["Data Nuggets"]])
tab_e <- dn_f[["Data Nuggets"]][c(1:4, 4, 5:k_f), ]
tab_e[5, "Weight"] <- 0
tab_e[5, "Scale"] <- 0
tab_e[["Data Nugget"]] <- seq_len(k_f + 1L)
rownames(tab_e) <- NULL
a_f <- as.integer(dn_f[["Data Nugget Assignments"]])
dn_e <- structure(list(`Data Nuggets` = tab_e,
                       `Data Nugget Assignments` = ifelse(a_f >= 5L, a_f + 1L, a_f)),
                  class = "datanugget")
add_refine("faithful_empty_middle", "faithful", dn_e, "empty_middle")
# Every row its own nugget: no positive first eigenvalue, so no EV split round at all
n_f <- nrow(datasets$faithful)
dn_s <- structure(list(
  `Data Nuggets` = data.frame(`Data Nugget` = seq_len(n_f), Center1 = datasets$faithful[, 1],
                              Center2 = datasets$faithful[, 2], Weight = 1L, Scale = 0,
                              check.names = FALSE),
  `Data Nugget Assignments` = seq_len(n_f)), class = "datanugget")
add_refine("faithful_singletons", "faithful", dn_s, "singletons", refine_args(shape.split = TRUE))
# nstart = 1 on duplicate rows: find a seed where some split's two random starts are equal rows,
# so kmeans redraws from unique(x) and R makes more draws than kmeans calls
dn_p <- make_dn("petal")
dup_seed <- Find(function(s) {
  r <- run_refine(datasets$petal, dn_p, refine_args(nstart = 1, seed = s))
  length(r$draws) > length(r$km)
}, seq_len(200))
if (is.null(dup_seed)) stop("no seed in 1:200 makes a petal split redraw its starts")
add_refine("petal_nstart1_dups", "petal", dn_p, "mean", refine_args(nstart = 1, seed = dup_seed))

flags <- function(name) vapply(all_paths, function(p) p[[name]], logical(1))
if (!any(flags("kmeans_error"))) stop("no case has a kmeans call that stopped with an error")
if (!any(flags("child_too_small"))) stop("no case has a split child below min.nugget.size")
cat(length(all_paths), "cases\n")
