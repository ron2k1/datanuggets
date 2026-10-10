# Fixtures for WCluster's DN.Wkmeans, DN.Whclust, DN.Wpca and DNcluster.predict on datanuggets
# from create.DN. Case "dn" holds the two datanuggets and the data frame used as newx; each other
# case is one wrapper's output on them. datanugget 1.5.0 defines none of the four, so these are
# WCluster's, and the check below stops if a datanugget version starts shipping its own.
source("r/fixture_io.R")

wrappers <- c("DN.Wkmeans", "DN.Whclust", "DN.Wpca", "DNcluster.predict")
stopifnot(!any(wrappers %in% getNamespaceExports("datanugget")),
          !any(vapply(wrappers, exists, NA, envir = asNamespace("datanugget"), inherits = FALSE)))

make_dn <- function(x, R, DN.num1, DN.num2) suppressWarnings(suppressMessages(create.DN(
  x = as_dbl(x), center.method = "mean", R = R, delete.percent = 0.3, DN.num1 = DN.num1,
  DN.num2 = DN.num2, dist.metric = "euclidean", seed = 291102, no.cores = 1, make.pbs = FALSE)))

dn_arrays <- function(dn, suffix = "") {
  tab <- dn[["Data Nuggets"]]
  p <- ncol(tab) - 3L
  a <- list(centers = unname(as.matrix(tab[, 1L + seq_len(p)])),
            weights = as.double(tab[["Weight"]]), scales = as.double(tab[["Scale"]]),
            assignments = as.integer(dn[["Data Nugget Assignments"]]))
  setNames(a, paste0(names(a), suffix))
}

timed <- function(id, expr) {
  t0 <- Sys.time()
  value <- expr
  cat(sprintf("dn_wcluster %-14s %6.1fs\n", id, as.numeric(Sys.time() - t0, units = "secs")))
  value
}

# The quakes datanugget the create_dn group records as quakes_onechunk_dp30: 150 nuggets
dn <- make_dn(quakes, 5000, 500, 150)
tab <- dn_arrays(dn)
stopifnot(nrow(tab$centers) == 150L, identical(tab$centers, read_array(
  file.path("tests", "fixtures", "create_dn", "quakes_onechunk_dp30"), "centers")))
dn2 <- make_dn(quakes[1:400, ], 5000, 200, 60)
stopifnot(nrow(dn2[["Data Nuggets"]]) == 60L)
newx <- quakes[901:1000, ]
write_case("dn_wcluster", "dn", no_args,
           c(tab, dn_arrays(dn2, "2"), list(newx = as_dbl(newx))))

res <- timed("wkmeans", capture_draws({
  out <- NULL
  capture.output(out <- WCluster::DN.Wkmeans(dn, 3))
  out
}))
wk <- res$value
cl <- as.integer(wk[["Cluster Assignments for data nuggets"]])
write_case("dn_wcluster", "wkmeans", list(k = 3, num_init = 1, max_iter = 10, seed = 291102),
           c(draw_arrays(res$draws), list(
             labels = cl, centers = unname(wk[["Cluster Centers"]]),
             wwcss = unname(wk$WWCSS$WWCSS), total = wk$WWCSS$TotalWWCSS,
             row_labels = as.integer(wk[["Cluster Assignments for original dataset"]]))))

hc <- unclass(timed("whclust", WCluster::DN.Whclust(dn)))
write_case("dn_wcluster", "whclust", no_args,
           list(merge = hc$merge, height = hc$height, order = hc$order))

# Wpca's locals as it leaves, as r/fixtures_wpca.R reads them, so the eigen rule applies
.capture$wpca <- new.env()
trace("Wpca", where = asNamespace("WCluster"), print = FALSE, exit = quote(
  for (nm in c("x_0", "x_wcov", "x_e"))
    if (exists(nm, inherits = FALSE)) assign(nm, get(nm, inherits = FALSE), envir = .capture$wpca)))
add_wpca <- function(id, corr) {
  rm(list = ls(.capture$wpca), envir = .capture$wpca)
  warns <- character()
  value <- timed(id, withCallingHandlers(WCluster::DN.Wpca(dn, corr = corr),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }))
  cap <- mget(c("x_0", "x_wcov", "x_e"), envir = .capture$wpca, ifnotfound = list(NULL))
  stopifnot(!is.null(cap$x_0), !is.null(cap$x_wcov), !is.null(cap$x_e$values),
            identical(unname(value$x), unname(cap$x_0 %*% value$rotation)),
            identical(value$sdev, suppressWarnings(sqrt(cap$x_e$values))))
  write_case("dn_wcluster", id, list(corr = corr), list(
    x = tab$centers, wrow = tab$weights, wcol_in = rep(1, ncol(tab$centers)),
    sdev = value$sdev, rotation = unname(value$rotation), scores = unname(value$x),
    center = unname(value$center), scale = as.double(unname(value$scale)),
    wcol = unname(value$wcol), values = cap$x_e$values, cap_x_0 = unname(cap$x_0),
    cap_s = unname(cap$x_wcov)), extra = list(warnings = I(warns)))
}
add_wpca("wpca", FALSE)
add_wpca("wpca_corr", TRUE)
suppressMessages(untrace("Wpca", where = asNamespace("WCluster")))

# newx as a data frame, and as a datanugget, whose table DNcluster.predict slices by dn's width
for (case in list(list(id = "predict_frame", newx = newx), list(id = "predict_dn", newx = dn2))) {
  got <- timed(case$id, WCluster::DNcluster.predict(dn, cl, case$newx))
  stopifnot(is.integer(got), !anyNA(got))
  write_case("dn_wcluster", case$id, no_args, list(cl = cl, labels = unname(got)))
}

# dn cut to its first variable: R's [, 2:2] drops the table to a vector, so DN.Wkmeans, DN.Wpca's
# default wcol and DNcluster.predict with a datanugget newx stop, while DN.Whclust, DN.Wpca with a
# wcol and DNcluster.predict with a data frame run. A matrix newx stops for any dn.
one <- dn
one[["Data Nuggets"]] <- one[["Data Nuggets"]][, c(1, 2, ncol(tab$centers) + 2, ncol(tab$centers) + 3)]
stopifnot(identical(names(one[["Data Nuggets"]])[3:4], c("Weight", "Scale")))
stop_message <- function(expr) {
  msg <- tryCatch({
    capture.output(expr)
    NULL
  }, error = function(e) conditionMessage(e))
  if (is.null(msg)) stop("R returned where this case says it stops")
  msg
}
errors <- list(
  wkmeans = stop_message(WCluster::DN.Wkmeans(one, 2)),
  wpca = stop_message(WCluster::DN.Wpca(one)),
  predict_one = stop_message(WCluster::DNcluster.predict(one, cl, one)),
  predict_dn2 = stop_message(WCluster::DNcluster.predict(one, cl, dn2)),
  predict_matrix = stop_message(WCluster::DNcluster.predict(dn, cl, as.matrix(newx))))
hc1 <- unclass(timed("one_variable", WCluster::DN.Whclust(one)))
pc1 <- WCluster::DN.Wpca(one, wcol = 1)
pr1 <- WCluster::DNcluster.predict(one, cl, newx[1:5, 1, drop = FALSE])
write_case("dn_wcluster", "one_variable", no_args, list(
  cl = cl, merge = hc1$merge, height = hc1$height, order = hc1$order, wpca_sdev = pc1$sdev,
  predict_labels = unname(pr1)), extra = list(errors = errors))
