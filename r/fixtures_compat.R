# Fixtures for the R-compatibility decisions in docs/plans/2026-09-30-r-compat.md: how R handles
# labels, logical and one-dimensional data, predictions with no answer, corr values, nstart,
# fractional nugget counts, the nugget table's types and create.DN's one-column "random" case.
# Every case records R's value, warnings and error, so a test checks the port against what R did,
# including where the port deliberately differs.
source("r/fixture_io.R")

G <- "compat"
unlink(file.path("tests", "fixtures", G), recursive = TRUE)

# R's value, warnings and error message for one call. Messages are muffled.
outcome <- function(expr) {
  warns <- character()
  err <- NULL
  value <- withCallingHandlers(
    tryCatch(expr, error = function(e) {
      err <<- conditionMessage(e)
      NULL
    }),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) invokeRestart("muffleMessage"))
  list(value = value, warnings = warns, error = err)
}
na_if_null <- function(v) if (is.null(v)) NA else v
done <- character()
save <- function(id, args, arrays, extra = list()) {
  stopifnot(!id %in% done)
  done <<- c(done, id)
  write_case(G, id, args, arrays, extra)
}

# A label vector as the fixture stores it: doubles as they are, logicals as 0/1 with kind
# "logical", factors as 0-based codes into their levels with kind "factor" or "ordered".
label_arrays <- function(cl) {
  if (is.factor(cl)) {
    lv <- levels(cl)
    list(arrays = list(cl_codes = as.integer(cl) - 1L),
         args = list(cl_kind = if (is.ordered(cl)) "ordered" else "factor", cl_categories = I(lv),
                     cl_categories_numeric = !anyNA(suppressWarnings(as.numeric(lv)))))
  } else if (is.logical(cl)) {
    list(arrays = list(cl = as.integer(cl)), args = list(cl_kind = "logical"))
  } else {
    stopifnot(is.double(cl))
    list(arrays = list(cl = cl), args = list(cl_kind = "double"))
  }
}

# cluster.predict's answer as arrays: a vector, or a list stored as each row's length (0 or 1)
# and the values in row order.
pred_arrays <- function(pred) {
  if (is.list(pred)) {
    list(kind = "list", arrays = list(pred_len = lengths(pred), pred = as.integer(unlist(pred))))
  } else {
    list(kind = if (length(pred)) "vector" else "empty", arrays = list(pred = as.integer(pred)))
  }
}

# ---- labels: factor() codes and levels for values that sit on 15-digit rounding edges ----
set.seed(20260930)
m15 <- sprintf("%.0f", floor(runif(1500, 1e14, 1e15)))
halfway <- as.numeric(paste0(substr(m15, 1, 1), ".", substr(m15, 2, 15), "5e",
                             sample(-30:30, 1500, replace = TRUE)))
edge <- c(0.1 + 0.2, 0.3, 1e15, 1e15 + 1, 1e15 + 2, 999999999999999.4, 999999999999999.6,
          2^53, 2^53 + 2, -0, 0, 5e-324, 1e-300, 1e300, -1e300, Inf, -Inf, 1 / 3, 2 / 3, 1e5,
          1e5 + 0.5, 1e21, 1e22, 123456789012345678, 9.999999999999995e14, 9.999999999999994e14,
          0.1234567890123455, 99999.99999999999, -2.5, 2.5)
for (nm in c("edge", "halfway")) {
  v <- get(nm)
  f <- factor(v)
  save(paste0("factor_codes_", nm), list(kind = "factor_codes", levels = I(levels(f))),
       list(cl = v, codes = as.integer(f)))
}

# ---- labels: the four WCluster functions on non-integer, logical and factor labels ----
labels_case <- function(id, x, cl, w, newx) {
  wm <- outcome(unname(WCluster::wmean(x, cl, w)))
  ww <- outcome(WCluster::wwcss(x, cl, w, groupSum = TRUE))
  dw <- outcome(unname(WCluster::distw(x, cl, w)))
  pr <- outcome(WCluster::cluster.predict(x, w, cl, newx))
  for (o in list(wm, ww, pr)) stopifnot(is.null(o$error), length(o$warnings) == 0L)
  stopifnot(length(dw$warnings) == 0L)
  la <- label_arrays(cl)
  pa <- pred_arrays(pr$value)
  f <- factor(cl)
  arrays <- c(list(x = x, w = w, newx = newx, codes = as.integer(f), wmean = wm$value,
                   wwcss_group = unname(ww$value$WWCSS), wwcss_total = ww$value$TotalWWCSS),
              la$arrays, pa$arrays)
  if (is.null(dw$error)) arrays$distw <- dw$value
  save(id, c(list(kind = "labels", levels = I(levels(f)), pred_kind = pa$kind,
                  distw_error = na_if_null(dw$error)), la$args), arrays)
  invisible(list(distw = dw, k = ncol(wm$value), pred = pr$value))
}
iris4 <- as_dbl(iris[, 1:4])
q <- as_dbl(quakes)
mt <- as_dbl(mtcars)
fa <- as_dbl(faithful)
rk <- as_dbl(rock)
lab <- list()
sp <- factor(iris$Species, levels = c("virginica", "setosa", "versicolor"))
lab$species_factor <- labels_case("labels_species_factor", iris4, sp, iris$Petal.Width,
                                  iris4[c(1, 51, 101, 150), ])
lab$species_ordered <- labels_case("labels_species_ordered", iris4,
                                   factor(sp, levels = levels(sp), ordered = TRUE),
                                   iris$Petal.Width, iris4[c(1, 51, 101, 150), ])
gear <- factor(mtcars$gear, levels = c(5, 3, 4))
lab$gear_factor <- labels_case("labels_gear_factor_5_3_4", mt, gear, mtcars$carb, mt[1:10, ])
lab$gear_ordered <- labels_case("labels_gear_ordered_5_3_4", mt,
                                factor(gear, levels = levels(gear), ordered = TRUE),
                                mtcars$carb, mt[1:10, ])
# a level with no rows: factor() drops it
lab$gear_unused <- labels_case("labels_gear_factor_unused", mt,
                               factor(mtcars$gear, levels = c(6, 5, 4, 3)), mtcars$carb, mt[1:5, ])
lab$quakes_mag <- labels_case("labels_quakes_mag", q, quakes$mag, as.double(quakes$stations),
                              q[1:40, ])
lab$rock_perm <- labels_case("labels_rock_perm", rk, rock$perm, rock$shape, rk[1:12, ])
lab$mtcars_am <- labels_case("labels_mtcars_am_logical", mt, mtcars$am == 1, mtcars$carb, mt[1:10, ])
lab$faithful_long <- labels_case("labels_faithful_long_logical", fa, faithful$waiting > 70,
                                 rep(1, 272), fa[1:20, ])
# 0.1 + 0.2 and 0.3 are one factor level but two sort(unique()) values
w3 <- ifelse(faithful$waiting > 80, 1, ifelse(faithful$waiting > 65, 0.3, 0.1 + 0.2))
lab$float_merge <- labels_case("labels_faithful_float_merge", fa, w3, faithful$eruptions, fa[1:20, ])
# 1e15 and 1e15 + 1 are one factor level, 2 is another
big <- ifelse(faithful$waiting > 80, 2, ifelse(faithful$waiting > 65, 1e15, 1e15 + 1))
lab$big <- labels_case("labels_faithful_big", fa, big, faithful$eruptions, fa[1:20, ])
lab$inf <- labels_case("labels_faithful_inf", fa,
                       ifelse(faithful$waiting > 80, Inf, ifelse(faithful$waiting > 65, 0, -Inf)),
                       rep(1, 272), fa[1:20, ])
nz <- ifelse(faithful$eruptions > 4, 5, ifelse(faithful$waiting > 70, 0, -0))
stopifnot(any(1 / nz == -Inf), any(1 / nz == Inf))
lab$negzero <- labels_case("labels_faithful_negzero", fa, nz, rep(1, 272), fa[1:20, ])
stopifnot(
  grepl("max", lab$species_factor$distw$error), is.null(lab$species_ordered$distw$error),
  grepl("max", lab$gear_factor$distw$error), is.null(lab$gear_ordered$distw$error),
  lab$gear_unused$k == 3,
  lab$float_merge$k == 2, identical(dim(lab$float_merge$distw$value), c(3L, 3L)),
  lab$float_merge$distw$value[2, 1] == 0, lab$float_merge$distw$value[3, 2] > 0,
  lab$big$k == 2, identical(dim(lab$big$distw$value), c(3L, 3L)),
  lab$inf$k == 3, lab$negzero$k == 2
)

# ---- data: logical and one-dimensional x in the WCluster functions and getDN.means ----
data_case <- function(id, x, cl, w, newx, wpca = TRUE) {
  s <- outcome(WCluster::wss(x, w))
  wm <- outcome(unname(WCluster::wmean(x, cl, w)))
  ww <- outcome(WCluster::wwcss(x, cl, w, groupSum = TRUE))
  dw <- outcome(unname(WCluster::distw(x, cl, w)))
  pr <- outcome(WCluster::cluster.predict(x, w, cl, newx))
  outs <- list(s, wm, ww, dw, pr)
  pc <- NULL
  if (wpca) {
    pc <- outcome(WCluster::Wpca(x))
    outs <- c(outs, list(pc))
  }
  for (o in outs) stopifnot(is.null(o$error), length(o$warnings) == 0L)
  xs <- if (is.null(dim(x))) as.double(x) else { m <- x; storage.mode(m) <- "integer"; dimnames(m) <- NULL; m }
  nx <- if (is.logical(newx)) { m <- newx; storage.mode(m) <- "integer"; dimnames(m) <- NULL; m } else newx
  arrays <- list(x = xs, cl = as.double(cl), w = as.double(w), newx = nx, wss = s$value,
                 wmean = wm$value, wwcss_group = unname(ww$value$WWCSS),
                 wwcss_total = ww$value$TotalWWCSS, distw = dw$value, pred = as.integer(pr$value))
  if (wpca) {
    arrays <- c(arrays, list(wpca_sdev = pc$value$sdev, wpca_rotation = pc$value$rotation,
                             wpca_x = unname(pc$value$x), wpca_center = unname(pc$value$center),
                             wpca_matrix = { xm <- as.matrix(x); storage.mode(xm) <- "double"; dimnames(xm) <- NULL; xm }))
  }
  save(id, list(kind = "data", x_logical = is.logical(x), x_vector = is.null(dim(x)),
                newx_logical = is.logical(newx)), arrays)
}
vsam <- as.matrix(mtcars[, c("vs", "am")] == 1)
data_case("data_mtcars_vsam_logical", vsam, as.double(mtcars$gear), mtcars$carb, vsam[1:10, ])
data_case("data_faithful_eruptions_vector", faithful$eruptions, as.double((faithful$waiting > 70) + 1),
          rep(1, 272), fa[1:15, 1, drop = FALSE])
# logical newx against a numeric x
local({
  pr <- outcome(WCluster::cluster.predict(as_dbl(mtcars[, c("vs", "am")]), mtcars$carb,
                                          as.double(mtcars$gear), vsam[1:10, ]))
  stopifnot(is.null(pr$error))
  m <- vsam[1:10, ]
  storage.mode(m) <- "integer"
  dimnames(m) <- NULL
  save("predict_logical_newx", list(kind = "predict", pred_kind = "vector",
                                    warnings = I(pr$warnings), error = NA),
       list(x = as_dbl(mtcars[, c("vs", "am")]), w = as.double(mtcars$carb),
            cl = as.double(mtcars$gear), newx = m, pred = as.integer(pr$value)))
})

# getDN.means on logical x, and on a vector, which it rejects
dn_mt <- suppressMessages(create.DN(mt, R = 100, DN.num1 = 20, DN.num2 = 8, no.cores = 1))
local({
  above <- sapply(mtcars, function(v) v > median(v))  # 32 x 11 logical, one column per center
  gm <- outcome(unname(as.matrix(getDN.means(above, dn_mt))))
  gv <- outcome(getDN.means(mtcars$mpg, dn_mt))
  stopifnot(is.logical(above), is.null(gm$error), !is.null(gv$error))
  m <- above
  storage.mode(m) <- "integer"
  dimnames(m) <- NULL
  tab <- dn_mt[["Data Nuggets"]]
  save("means_logical", list(kind = "means", vector_error = gv$error),
       list(x = m, centers = unname(as.matrix(tab[, 2:12])), weights = as.integer(tab$Weight),
            scales = as.double(tab$Scale), assignments = as.integer(dn_mt[["Data Nugget Assignments"]]),
            means = gm$value))
})

# create.DN, create.DNcenters and refine.DN on a data frame with logical columns: R turns it into
# the numeric matrix, so the result is identical. An all-logical matrix stops.
mixed <- data.frame(mtcars[, c("mpg", "disp", "hp", "wt")], vs = mtcars$vs == 1, am = mtcars$am == 1)
mixed_num <- as_dbl(mixed)
local({
  a <- suppressMessages(create.DN(mixed, R = 100, DN.num1 = 20, DN.num2 = 8, no.cores = 1))
  b <- suppressMessages(create.DN(mixed_num, R = 100, DN.num1 = 20, DN.num2 = 8, no.cores = 1))
  ca <- create.DNcenters(mixed[1:30, ], DN.num = 10)
  cb <- create.DNcenters(mixed_num[1:30, ], DN.num = 10)
  ra <- suppressMessages(refine.DN(mixed, b, no.cores = 1))
  rb <- suppressMessages(refine.DN(mixed_num, b, no.cores = 1))
  e1 <- outcome(create.DN(vsam, R = 100, DN.num1 = 20, DN.num2 = 8, no.cores = 1))
  e2 <- outcome(create.DNcenters(vsam, DN.num = 10))
  e3 <- outcome(refine.DN(vsam, b, no.cores = 1))
  e4 <- outcome(create.DN(as.data.frame(vsam), R = 100, DN.num1 = 20, DN.num2 = 8, no.cores = 1))
  stopifnot(identical(a, b), identical(unname(ca), unname(cb)), identical(ra, rb),
            !is.null(e1$error), !is.null(e2$error), !is.null(e3$error), !is.null(e4$error))
  save("mixed_logical_frame", list(kind = "mixed", logical_columns = I(c(5L, 6L)),
                                   create_identical = TRUE, centers_identical = TRUE,
                                   refine_identical = TRUE, all_logical_create_error = e1$error,
                                   all_logical_centers_error = e2$error,
                                   all_logical_refine_error = e3$error,
                                   all_logical_frame_error = e4$error),
       list(x = mixed_num))
})

# ---- predict: the cases where R has no answer for some or all rows ----
predict_case <- function(id, x, w, cl, newx) {
  pr <- outcome(WCluster::cluster.predict(x, w, cl, newx))
  pa <- if (is.null(pr$error)) pred_arrays(pr$value) else list(kind = "error", arrays = list())
  arrays <- c(list(x = x, w = as.double(w), cl = as.double(cl)),
              if (!is.null(dim(newx))) list(newx = newx) else list(newx_vector = newx), pa$arrays)
  save(id, list(kind = "predict", pred_kind = pa$kind, warnings = I(pr$warnings),
                error = na_if_null(pr$error)), arrays)
  pr
}
x8 <- iris4[1:8, 1:2]
cl8 <- as.double(rep(1:2, 4))
pz <- predict_case("predict_zero_rows", x8, rep(1, 8), cl8, x8[0, , drop = FALSE])
pn <- predict_case("predict_nan_row", x8, rep(1, 8), cl8, rbind(x8[1, ], c(NaN, 1), x8[3, ]))
pa <- predict_case("predict_na_row", x8, rep(1, 8), cl8, rbind(x8[1, ], c(NA, 1), x8[3, ]))
pl <- predict_case("predict_partial_list", rbind(c(1, 2), c(3, 4), c(5, 6)), c(1, 1, 1), c(1, 2, 2),
                   rbind(c(1, 2), c(1e200, 0), c(4, 4)))
pd <- predict_case("predict_all_centers_nan", x8, rep(0, 8), cl8, x8[1:3, ])
p1 <- predict_case("predict_vector_x", faithful$eruptions, rep(1, 272),
                   as.double((faithful$waiting > 70) + 1), fa[1:15, 1, drop = FALSE])
pv <- predict_case("predict_vector_newx", x8, rep(1, 8), cl8, c(5, 3))
stopifnot(
  identical(pz$value, integer(0)), length(pz$warnings) == 1L,
  identical(pn$value, integer(0)), length(pn$warnings) == 0L,
  identical(pa$value, integer(0)),
  is.list(pl$value), identical(lengths(pl$value), c(0L, 1L, 0L)),
  identical(pd$value, integer(0)),
  is.null(p1$error), length(p1$value) == 15L,
  !is.null(pv$error)
)

# ---- wpca: which corr values R accepts, with no global dd in this session ----
stopifnot(!exists("dd", envir = globalenv()))
local({
  vals <- list(TRUE, FALSE, 1, 0, 1L, 0L, "TRUE", "FALSE", "F", "false", "False", "T", "true",
               "True", 2, -1, NA)
  ref_t <- WCluster::Wpca(iris4, corr = TRUE)
  ref_f <- WCluster::Wpca(iris4, corr = FALSE)
  rows <- lapply(vals, function(v) {
    o <- outcome(WCluster::Wpca(iris4, corr = v))
    same <- if (!is.null(o$error)) "error" else if (identical(o$value, ref_t)) "TRUE" else
      if (identical(o$value, ref_f)) "FALSE" else "other"
    list(value = if (is.na(v)) "NA" else as.character(v), type = typeof(v), result = same,
         error = na_if_null(o$error))
  })
  stopifnot(!any(vapply(rows, function(r) r$result == "other", TRUE)),
            identical(ref_f$scale, 1))
  save("wpca_corr_values", list(kind = "wpca_corr", rows = rows),
       list(x = iris4, scale_true = ref_t$scale, scale_false = ref_f$scale))
})

# ---- wpca, one column, corr = TRUE: R's diag(1/dd) is an identity of order floor(1/dd) ----
local({
  cols <- list(petal_width = iris$Petal.Width, sepal_length = iris$Sepal.Length,
               petal_length = iris$Petal.Length, sepal_width = iris$Sepal.Width,
               eruptions = faithful$eruptions, waiting = faithful$waiting)
  for (nm in names(cols)) {
    x1 <- matrix(cols[[nm]], ncol = 1)
    n <- nrow(x1)
    ctr <- colSums(x1) / n
    x0 <- sweep(x1, 2, ctr)
    s <- t(x0) %*% x0 / (n - 1)
    d1 <- sqrt(diag(s))
    o <- outcome(WCluster::Wpca(x1, corr = TRUE))
    fixed <- diag(1 / d1, nrow = 1) %*% s %*% diag(1 / d1, nrow = 1)
    save(paste0("wpca1_corr_", nm), list(kind = "wpca1_corr", identity_order = floor(1 / d1),
                                          r_error = na_if_null(o$error)),
         c(list(x = x1, dd = d1, cov = s, cov2cor = cov2cor(s), fixed_sdev = sqrt(eigen(fixed)$values)),
           if (is.null(o$error)) list(r_sdev = o$value$sdev, r_scale = o$value$scale) else list()))
  }
})

# ---- nstart: refine.DN catches kmeans's error and leaves the nugget whole ----
dn_fa <- suppressMessages(create.DN(fa, R = 100, DN.num1 = 90, DN.num2 = 20, no.cores = 1))
unsplit <- suppressMessages(refine.DN(fa, dn_fa, max.splits = 0, no.cores = 1))
nstart_id <- function(v) gsub("\\.", "p", gsub("-", "neg", format(v)))
x40 <- fa[1:40, ]
logical_na <- outcome(refine.DN(fa, dn_fa, nstart = NA, no.cores = 1))
stopifnot(!is.null(logical_na$error))
for (v in c(NaN, NA, Inf, -Inf, 0, -1, 1.5, 2.5, 25)) {
  msgs <- character()
  warns <- character()
  res <- capture_draws(withCallingHandlers(
    refine.DN(fa, dn_fa, nstart = v, no.cores = 1),
    message = function(m) { msgs <<- c(msgs, sub("\n$", "", conditionMessage(m))); invokeRestart("muffleMessage") },
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }))
  r <- res$value
  tab <- r[["Data Nuggets"]]
  km <- outcome({ set.seed(1); stats::kmeans(x40, 2, iter.max = 1000, nstart = v) })
  save(paste0("nstart_", if (is.na(v) && !is.nan(v)) "NA" else nstart_id(v)),
       list(kind = "nstart", nstart = if (is.nan(v)) "NaN" else if (is.na(v)) "NA" else format(v),
            unsplit = identical(r, unsplit), messages = I(msgs), warnings = I(warns),
            kmeans_error = na_if_null(km$error), logical_na_error = logical_na$error),
       c(list(x = fa, nstart = as.double(v),
              in_centers = unname(as.matrix(dn_fa[["Data Nuggets"]][, 2:3])),
              in_weights = as.integer(dn_fa[["Data Nuggets"]]$Weight),
              in_scales = as.double(dn_fa[["Data Nuggets"]]$Scale),
              in_assignments = as.integer(dn_fa[["Data Nugget Assignments"]]),
              centers = unname(as.matrix(tab[, 2:3])), weights = as.double(tab$Weight),
              scales = as.double(tab$Scale), assignments = as.integer(r[["Data Nugget Assignments"]]),
              x40 = x40),
         draw_arrays(res$draws),
         if (is.null(km$error)) list(kmeans_size = km$value$size) else list()))
}

# ---- DN.num: create.DN floors fractional counts. Values below 1 are recorded, under a time
# limit, because AC's loops never end for some of them ----
local({
  a <- suppressMessages(create.DN(iris4, R = 100, DN.num1 = 60.7, DN.num2 = 20.9, no.cores = 1))
  b <- suppressMessages(create.DN(iris4, R = 100, DN.num1 = 60, DN.num2 = 20, no.cores = 1))
  ca <- create.DNcenters(iris4, DN.num = 10.5)
  cb <- create.DNcenters(iris4, DN.num = 10)
  edge_rows <- list()
  for (args in list(c(60, 0.5), c(60, 0), c(60, -1), c(0.5, 0.5), c(Inf, 20), c(60.2, 60.7))) {
    setTimeLimit(elapsed = 20, transient = TRUE)
    o <- outcome(create.DN(iris4, R = 100, DN.num1 = args[1], DN.num2 = args[2], no.cores = 1))
    setTimeLimit(elapsed = Inf)
    edge_rows[[length(edge_rows) + 1L]] <- list(
      dn_num1 = format(args[1]), dn_num2 = format(args[2]), error = na_if_null(o$error),
      nuggets = if (is.null(o$error)) nrow(o$value[["Data Nuggets"]]) else NA,
      warnings = I(o$warnings))
  }
  save("dn_num_fractional", list(kind = "dn_num", create_identical = identical(a, b),
                                 centers_identical = identical(ca, cb), edge = edge_rows),
       list(x = iris4))
})

# ---- the nugget table: column names, row names and column types ----
local({
  describe <- function(dn) {
    tab <- dn[["Data Nuggets"]]
    list(colnames = I(colnames(tab)), rownames = I(rownames(tab)),
         types = I(vapply(tab, typeof, "")), assignments_type = typeof(dn[["Data Nugget Assignments"]]),
         class = I(class(dn)), names = I(names(dn)))
  }
  created <- suppressMessages(create.DN(fa, R = 100, DN.num1 = 90, DN.num2 = 20, no.cores = 1))
  kept <- suppressMessages(refine.DN(fa, created, max.splits = 0, no.cores = 1))
  split <- suppressMessages(refine.DN(fa, created, no.cores = 1))
  stopifnot(nrow(split[["Data Nuggets"]]) > nrow(created[["Data Nuggets"]]))
  save("frame_types", list(kind = "frame", create = describe(created), refine_unsplit = describe(kept),
                           refine_split = describe(split)), list(x = fa))
})

# ---- create.DN, one column, "random" centers: R builds a malformed table ----
local({
  x1 <- as_dbl(faithful[, "eruptions", drop = FALSE])
  res <- capture_draws(suppressMessages(create.DN(x1, center.method = "random", R = 100,
                                                  DN.num1 = 90, DN.num2 = 30, no.cores = 1)))
  dn <- res$value
  tab <- dn[["Data Nuggets"]]
  k <- max(dn[["Data Nugget Assignments"]])
  stopifnot(nrow(tab) == k, ncol(tab) == k + 3L)
  m <- as.matrix(tab)
  dimnames(m) <- NULL
  save("create_p1_random", list(kind = "p1_random", colnames = I(colnames(tab)), k = k,
                                args = list(center.method = "random", R = 100, DN.num1 = 90,
                                            DN.num2 = 30, seed = 291102)),
       c(list(x = x1, table = m, assignments = as.integer(dn[["Data Nugget Assignments"]])),
         draw_arrays(res$draws)))
})

cat(G, length(done), "cases\n")
