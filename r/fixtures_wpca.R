# Fixtures for WCluster's Wpca: inputs, outputs, and the internals it hands to eigen().
source("r/fixture_io.R")

# Wpca's locals as it leaves, read by a trace on the namespace binding. The hook also runs when
# eigen() stops, and then x_e is missing, so every capture is checked before it's used.
.capture$wpca <- new.env()
trace("Wpca", where = asNamespace("WCluster"), print = FALSE, exit = quote(
  for (nm in c("x_0", "x_wcov", "x_e"))
    if (exists(nm, inherits = FALSE)) assign(nm, get(nm, inherits = FALSE), envir = .capture$wpca)))

run_wpca <- function(x, wrow, wcol, corr) {
  rm(list = ls(.capture$wpca), envir = .capture$wpca)
  warns <- character()
  value <- withCallingHandlers(WCluster::Wpca(x, wrow, wcol, corr),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  cap <- mget(c("x_0", "x_wcov", "x_e"), envir = .capture$wpca, ifnotfound = list(NULL))
  stopifnot(!is.null(cap$x_0), !is.null(cap$x_wcov), !is.null(cap$x_e), !is.null(cap$x_e$values))
  # Fail closed unless the captures are the ones Wpca built its outputs from
  stopifnot(identical(unname(value$x), unname(cap$x_0 %*% value$rotation)),
            identical(value$sdev, suppressWarnings(sqrt(cap$x_e$values))))
  list(value = value, warnings = warns, x_0 = unname(cap$x_0), s = unname(cap$x_wcov),
       values = cap$x_e$values)
}

iris4 <- as_dbl(iris[, 1:4])
nug <- function(case) {
  d <- file.path("tests", "fixtures", "create_dn", case)
  list(x = read_array(d, "centers"), w = as.double(read_array(d, "weights")))
}
nq <- nug("quakes_euclidean_mean")
ni <- nug("iris_all_passthrough")
nv <- nug("volcano_scaled_mean")
npkx <- sapply(npk[, c("N", "P", "K")], function(f) as.numeric(as.character(f)))

cases <- list()
add_case <- function(id, x, wrow = NULL, wcol = NULL, corr = FALSE) {
  x <- as_dbl(x)
  if (is.null(wrow)) wrow <- rep(1, nrow(x))
  if (is.null(wcol)) wcol <- rep(1, ncol(x))
  cases[[id]] <<- list(x = x, wrow = as.double(unname(wrow)), wcol = as.double(unname(wcol)), corr = corr)
}
add_case("iris", iris4)
add_case("iris_corr", iris4, corr = TRUE)
add_case("quakes_stations", quakes, quakes$stations)
add_case("quakes_stations_corr", quakes, quakes$stations, corr = TRUE)
add_case("w_iris_petalwidth", iris4, iris$Petal.Width)
add_case("w_mtcars_wt", mtcars, mtcars$wt)
add_case("mtcars_carb_invsd", mtcars, mtcars$carb, 1 / apply(mtcars, 2, sd))
add_case("nug_quakes", nq$x, nq$w)
add_case("nug_quakes_corr", nq$x, nq$w, corr = TRUE)
add_case("nug_iris_passthrough", ni$x, ni$w)
add_case("faithful1d", faithful[, 1, drop = FALSE])
add_case("petalwidth_1d_corr", iris[, "Petal.Width", drop = FALSE], corr = TRUE)
add_case("volcano_scaled", scale(volcano))
add_case("nug_volcano_scaled", nv$x, nv$w)
add_case("mtcars_cyl8_constant", mtcars[mtcars$cyl == 8, c("mpg", "disp", "hp", "vs")])
add_case("wcol_volcano49_default", volcano[, 1:49])
add_case("wcol_volcano49_corr", volcano[, 1:49], corr = TRUE)
add_case("den_mtcars_first18_wt", mtcars[1:18, ], mtcars$wt[1:18])
add_case("den_trees_first2_girth", trees[1:2, ], trees$Girth[1:2])
for (k in 2:4) {
  add_case(sprintf("den_iris_first%d_petalwidth", k), iris4[1:k, , drop = FALSE], iris$Petal.Width[1:k])
}
add_case("tie_npk", npkx)
# Row weights whose total only a long double sum gets right. sum() adds 1, then 2^-53 exactly,
# then rounds once to 64 bits and once to 53. sum_ld_near is sum_ld_stops, below, with 2^-62 for
# 2^-64: math.fsum agrees with R, and a double sum drops 2^-53 and gives 1. sum_ld_ties_down: the
# 64-bit rounding leaves a tie that goes down to 1 + 2^-51, where fsum rounds up.
# sum_ld_double_loses: a double sum drops both 2^-53 and gives 1, where R keeps them.
x_ld <- cbind(c(1, 4, 2), c(3, 1, 5))
add_case("sum_ld_near", x_ld, c(1, 2^-53, 2^-62))
add_case("sum_ld_ties_down", x_ld, c(1 + 2^-51, 2^-53, 2^-64))
add_case("sum_ld_double_loses", x_ld, c(1, 2^-53, 2^-53))

# The band inside which an eigenvalue's sign isn't determined, as tests/rfixtures.py computes it
band <- function(v) 1024 * .Machine$double.eps * max(abs(v), 1e-300)

seen <- list(zero_wrow = character(), clean = character())
for (id in names(cases)) {
  cs <- cases[[id]]
  p <- ncol(cs$x)
  r <- run_wpca(cs$x, cs$wrow, cs$wcol, cs$corr)
  res <- r$value
  arrays <- list(
    x = cs$x, wrow = cs$wrow, wcol_in = cs$wcol,
    sdev = res$sdev, rotation = unname(res$rotation), scores = unname(res$x),
    center = unname(res$center), scale = as.double(unname(res$scale)), wcol = unname(res$wcol),
    values = r$values, cap_x_0 = r$x_0, cap_s = r$s
  )
  if (cs$corr) arrays$cap_cov <- run_wpca(cs$x, cs$wrow, cs$wcol, FALSE)$s
  if (startsWith(id, "sum_ld_")) arrays$sum_wrow <- sum(cs$wrow)

  # R's one warning is sqrt's, raised once exactly when an eigenvalue is negative
  stopifnot(all(r$warnings == "NaNs produced"), length(r$warnings) == any(r$values < 0))
  b <- band(r$values)
  if (any(cs$wrow == 0)) seen$zero_wrow <- c(seen$zero_wrow, id)
  if (all(r$values > b) && length(r$warnings) == 0) seen$clean <- c(seen$clean, id)
  if (startsWith(id, "den_iris_")) {
    stopifnot(min(r$values) < -b, length(r$warnings) == 1L)
  }
  if (id %in% c("den_mtcars_first18_wt", "den_trees_first2_girth")) {
    # the double subtraction of 1 and a long double sum through -1 part by one ulp here
    stopifnot(length(r$warnings) == 0L, !identical(sum(cs$wrow) - 1, sum(c(cs$wrow, -1))))
  }
  if (id == "tie_npk") {
    stopifnot(any(abs(diff(r$values)) <= 1e-12 * max(abs(r$values))))
  }
  if (cs$corr && p > 1) {
    stopifnot(!identical(r$s, t(r$s)))
  }
  if (startsWith(id, "wcol_volcano49_")) {
    stopifnot(p == 49L, (1 / p) * p != 1)
  }
  if (id == "faithful1d") stopifnot(p == 1L)
  if (id == "sum_ld_near") stopifnot(sum(cs$wrow) == 1 + 2^-52, (1 + 2^-53) + 2^-62 == 1)
  if (id == "sum_ld_ties_down") stopifnot(sum(cs$wrow) == 1 + 2^-51)
  if (id == "sum_ld_double_loses") stopifnot(sum(cs$wrow) == 1 + 2^-52, (1 + 2^-53) + 2^-53 == 1)

  write_case("wpca", id, list(corr = cs$corr), arrays, extra = list(warnings = I(r$warnings)))
  cat(sprintf("%-28s n=%4d p=%2d corr=%-5s warnings=%d min value %.3e band %.3e NaN sdev=%d\n",
              id, nrow(cs$x), p, cs$corr, length(r$warnings), min(r$values), b, sum(is.nan(res$sdev))))
}
stopifnot("nug_iris_passthrough" %in% seen$zero_wrow, "iris" %in% seen$clean)
cat("zero row weight in:", seen$zero_wrow, "\n")

# iris[1:5] with Petal.Width sums to exactly 1, so the covariance divides by zero and eigen() stops.
# Recorded as inputs only, with R's error.
x <- iris4[1:5, , drop = FALSE]
wrow <- iris$Petal.Width[1:5]
stopifnot(sum(wrow) - 1 == 0)
warns <- character()
err <- tryCatch(withCallingHandlers({ WCluster::Wpca(x, wrow, rep(1, 4), FALSE); NULL },
  warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
  error = function(e) e)
stopifnot(inherits(err, "error"), conditionMessage(err) == "infinite or missing values in 'x'",
          length(warns) == 0L)
write_case("wpca", "one_iris_first5_petalwidth", list(corr = FALSE),
           list(x = x, wrow = wrow, wcol_in = rep(1, 4)), extra = list(error = conditionMessage(err)))
cat(sprintf("%-28s stops: %s\n", "one_iris_first5_petalwidth", conditionMessage(err)))

# 1 + 2^-53 + 2^-64 is above 1, but the 64-bit rounding leaves 1 + 2^-53, a tie that the 53-bit
# rounding sends to 1. So R's sum(wrow) - 1 is 0 and eigen() stops, where math.fsum gives
# 1 + 2^-52 and a covariance.
wrow <- c(1, 2^-53, 2^-64)
stopifnot(sum(wrow) - 1 == 0, .Machine$longdouble.digits == 64L)
err <- tryCatch({ WCluster::Wpca(x_ld, wrow, c(1, 1), FALSE); NULL }, error = function(e) e)
stopifnot(inherits(err, "error"), conditionMessage(err) == "infinite or missing values in 'x'")
write_case("wpca", "sum_ld_stops", list(corr = FALSE),
           list(x = x_ld, wrow = wrow, wcol_in = c(1, 1), sum_wrow = sum(wrow)),
           extra = list(error = conditionMessage(err)))
cat(sprintf("%-28s stops: %s\n", "sum_ld_stops", conditionMessage(err)))

untrace("Wpca", where = asNamespace("WCluster"))
