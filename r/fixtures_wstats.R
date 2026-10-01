# Fixtures for WCluster's cluster statistics: wmean, wss, wwcss, distw and cluster.predict.
# One case feeds all five. Group wstats holds real data only (R's datasets and the repo's own
# create_dn nuggets read back). Group wstats_constructed holds built points and built weights:
# tie midpoints and their ulp nudges, heavy weights, and rows whose squares overflow.
source("r/fixture_io.R")
options(warn = 2)  # any warning from R stops the recording

ward <- function(m, k) cutree(hclust(dist(m), "ward.D2"), k)
bands <- cut(quakes$depth, c(0, 150, 300, 450, 700), labels = FALSE)

# Labels as the fixture stores them: whole numbers, integer unless a case asks for double.
lab <- function(v) {
  v <- as.vector(v)
  if (is.logical(v)) v <- as.integer(v)
  stopifnot(is.numeric(v), all(v == round(v)))
  as.integer(v)
}
# rock$perm and quakes$mag hold fractional labels, which the port rejects. Their tenths are whole
# numbers that R groups and orders the same way. record() checks R's answers don't move.
tenths <- function(v) {
  out <- as.integer(round(v * 10))
  stopifnot(all(abs(out / 10 - v) < 1e-9),
            identical(as.integer(factor(out)), as.integer(factor(v))))
  out
}
wts <- function(v) as.double(v)

strip <- function(p) if (is.list(p)) lapply(unname(p), unname) else unname(p)
tie_rows <- function(z) sum(apply(z, 1, function(r) {
  r <- r[!is.nan(r)]
  length(r) >= 2 && sum(r == min(r)) >= 2
}))

# R's five answers on exactly the arrays the fixture stores. orig_cl is the label vector the
# research used, when the stored one is a relabelling of it: every answer must be the same.
record <- function(group, id, x, cl, w, newx = x, predict = TRUE, orig_cl = NULL) {
  stopifnot(is.matrix(x), is.null(dimnames(x)), is.null(dim(cl)), is.null(attributes(cl)),
            is.null(attributes(w)), length(cl) == nrow(x), length(w) == nrow(x),
            is.integer(cl) || is.double(cl), is.integer(w) || is.double(w))
  s <- WCluster::wss(x, w)
  g <- WCluster::wwcss(x, cl, w, groupSum = TRUE)
  tot <- WCluster::wwcss(x, cl, w)
  stopifnot(identical(tot, g$TotalWWCSS))
  cen <- unname(WCluster::wmean(x, cl, w))
  k <- ncol(cen)
  stopifnot(length(g$WWCSS) == k)
  arrays <- list(x = x, cl = cl, w = w, wss = s, wwcss_group = unname(g$WWCSS),
                 wwcss_total = tot, wmean = cen)
  if (k >= 2) arrays$distw <- unname(WCluster::distw(x, cl, w))
  args <- no_args
  pred <- NULL
  if (predict) {
    stopifnot(is.matrix(newx), is.null(dimnames(newx)), ncol(newx) == ncol(x))
    pred <- WCluster::cluster.predict(x, w, cl, newx)
    z <- unname(WCluster:::wdist_mat(newx, WCluster::wmean(x, cl, w), rep(1, nrow(newx))))
    stopifnot(identical(strip(apply(z, 1, which.min)), strip(pred)))
    arrays$newx <- newx
    arrays$z <- z
    if (is.list(pred)) args <- list(pred = "list") else arrays$pred <- as.integer(pred)
  }
  if (!is.null(orig_cl)) {
    stopifnot(identical(unname(WCluster::wmean(x, orig_cl, w)), cen),
              identical(unname(WCluster::wwcss(x, orig_cl, w, groupSum = TRUE)$WWCSS),
                        arrays$wwcss_group))
    if (k >= 2) stopifnot(identical(unname(WCluster::distw(x, orig_cl, w)), arrays$distw))
    if (predict) stopifnot(identical(strip(WCluster::cluster.predict(x, w, orig_cl, newx)),
                                     strip(pred)))
  }
  write_case(group, id, args, arrays)
  invisible(c(arrays, list(k = k, pred_list = is.list(pred))))
}

# ---- group wstats: real data ----
G <- "wstats"
iris4 <- as_dbl(iris[, 1:4])
species <- lab(as.integer(iris$Species))
mt <- as_dbl(mtcars)
q <- as_dbl(quakes)
fa <- as_dbl(faithful)
cw <- data.frame(weight = ChickWeight$weight, Time = ChickWeight$Time,
                 Chick = as.integer(as.character(ChickWeight$Chick)),
                 Diet = as.integer(as.character(ChickWeight$Diet)))
eu <- as.data.frame(EuStockMarkets)
nug <- function(name) file.path("tests", "fixtures", "create_dn", name)
nq <- read_array(nug("quakes_euclidean_mean"), "centers")
nqw <- wts(read_array(nug("quakes_euclidean_mean"), "weights"))
ni <- read_array(nug("iris_all_passthrough"), "centers")
niw <- wts(read_array(nug("iris_all_passthrough"), "weights"))
nv <- read_array(nug("volcano_scaled_mean"), "centers")
nvw <- wts(read_array(nug("volcano_scaled_mean"), "weights"))

# Cases recorded for their distw or wwcss record no prediction (predict = FALSE below), and
# quakes_ward100_stations predicts its first 100 rows, which keeps the feature's fixtures inside
# 8 MB. Negative z and z ties stay covered by iris50_singletons, iris150_singletons_pw,
# mtcars_cyl1d_gear and chick_diettime_time, integer newx by quakes_int2_mag5, and negative labels
# by five rows of iris_singletons_neg_pw.
r <- list()
r$iris_species_ones <- record(G, "iris_species_ones", iris4, species, rep(1, 150))
r$iris_species_petalwidth <- record(G, "iris_species_petalwidth", iris4, species, wts(iris$Petal.Width))
r$mtcars_cyl_carb <- record(G, "mtcars_cyl_carb", mt, as.double(mtcars$cyl), wts(mtcars$carb))
r$mtcars_gear_vs <- record(G, "mtcars_gear_vs", mt, lab(mtcars$gear), wts(mtcars$vs))
r$mtcars_gear_am <- record(G, "mtcars_gear_am", mt, lab(mtcars$gear), wts(mtcars$am))
r$mtcars_carb_am <- record(G, "mtcars_carb_am", mt, lab(mtcars$carb), wts(mtcars$am))
r$mtcars_cyl_vs <- record(G, "mtcars_cyl_vs", mt, lab(mtcars$cyl), wts(mtcars$vs))
r$chick_diettime_time <- record(G, "chick_diettime_time", as_dbl(cw[, c("weight", "Time")]),
                                lab(as.integer(interaction(cw$Diet, cw$Time))), wts(cw$Time))
r$quakes_depth_stations <- record(G, "quakes_depth_stations", q, lab(bands), wts(quakes$stations))
r$faithful_waiting <- record(G, "faithful_waiting", fa, lab((faithful$waiting > 70) + 1), rep(1, 272))
r$faithful1d <- record(G, "faithful1d", as_dbl(faithful[, 1, drop = FALSE]),
                       lab((faithful$waiting > 70) + 1), rep(1, 272))
r$faithful1d_by_waiting <- record(G, "faithful1d_by_waiting",
                                  as_dbl(faithful[, "eruptions", drop = FALSE]),
                                  lab(faithful$waiting), rep(1, 272), predict = FALSE)
r$infert_stratum_parity <- record(G, "infert_stratum_parity",
                                  as_dbl(infert[, c("age", "parity", "induced", "case", "spontaneous")]),
                                  lab(infert$stratum), wts(infert$parity), predict = FALSE)
r$nug_quakes <- record(G, "nug_quakes", nq, lab(ward(nq, 4)), nqw,
                       newx = read_array(nug("quakes_euclidean_mean"), "x"))
r$nug_iris_passthrough <- record(G, "nug_iris_passthrough", ni, lab(ward(ni, 3)), niw)
r$nug_volcano_scaled <- record(G, "nug_volcano_scaled", nv, lab(ward(nv, 3)), nvw,
                               newx = read_array(nug("volcano_scaled_mean"), "x"))
r$volcano_ward4 <- record(G, "volcano_ward4", as_dbl(volcano), lab(ward(volcano, 4)), rep(1, 87))
r$iris50_singletons <- record(G, "iris50_singletons", as_dbl(iris[1:50, 1:4]), 1:50,
                              wts(iris$Petal.Width[1:50]))
r$iris150_singletons_pw <- record(G, "iris150_singletons_pw", iris4, 1:150, wts(iris$Petal.Width))
# rows 102 and 143 have the same measurements and weight, so their singleton centers tie and
# which.min picks -143, first in sorted label order
r$iris_singletons_neg_pw <- record(G, "iris_singletons_neg_pw", iris4, -(1:150), wts(iris$Petal.Width),
                                   newx = iris4[c(1, 51, 101, 102, 143), ])
r$iris3_singletons_pw <- record(G, "iris3_singletons_pw", as_dbl(iris[, 1:3]), 1:150,
                                wts(iris$Petal.Width), predict = FALSE)
r$iris12_singletons_pl <- record(G, "iris12_singletons_pl", as_dbl(iris[, 1:2]), 1:150,
                                 wts(iris$Petal.Length), predict = FALSE)
r$iris1_singletons_pw <- record(G, "iris1_singletons_pw", as_dbl(iris[, 1, drop = FALSE]), 1:150,
                                wts(iris$Petal.Width), predict = FALSE)
r$iris124_singletons_pl <- record(G, "iris124_singletons_pl", as_dbl(iris[, c(1, 2, 4)]), 1:150,
                                  wts(iris$Petal.Length), predict = FALSE)
r$quakes200_singletons_st <- record(G, "quakes200_singletons_st", as_dbl(quakes[1:200, ]), 1:200,
                                    wts(quakes$stations[1:200]), predict = FALSE)
r$mtcars_singletons_vs <- record(G, "mtcars_singletons_vs", mt, 1:32, wts(mtcars$vs))
r$nug_iris_passthrough_singletons <- record(G, "nug_iris_passthrough_singletons", ni,
                                            seq_len(nrow(ni)), niw, predict = FALSE)
r$iris_one_cluster <- record(G, "iris_one_cluster", iris4, rep(1L, 150), wts(iris$Petal.Width))
r$mtcars_one_carb <- record(G, "mtcars_one_carb", mt, rep(1L, 32), wts(mtcars$carb))
r$volcano_one_ones <- record(G, "volcano_one_ones", as_dbl(volcano), rep(1L, 87), rep(1, 87))
r$volcano_scaled_one_ones <- record(G, "volcano_scaled_one_ones", as_dbl(scale(volcano)),
                                    rep(1L, 87), rep(1, 87))
r$mtcars_cyl_carb_row15 <- record(G, "mtcars_cyl_carb_row15", mt, lab(mtcars$cyl), wts(mtcars$carb),
                                  newx = as_dbl(mtcars[15, ]))
r$iris_species_pw_row150 <- record(G, "iris_species_pw_row150", iris4, species, wts(iris$Petal.Width),
                                   newx = as_dbl(iris[150, 1:4]))
r$mtcars_gear_am_row01 <- record(G, "mtcars_gear_am_row01", mt, lab(mtcars$gear), wts(mtcars$am),
                                 newx = as_dbl(mtcars[1, ]))
r$mtcars_cyl1d_gear <- record(G, "mtcars_cyl1d_gear", as_dbl(mtcars[, "cyl", drop = FALSE]),
                              lab(mtcars$cyl), wts(mtcars$carb),
                              newx = as_dbl(data.frame(cyl = mtcars$gear)))
r$iris_species_petallength_k3 <- record(G, "iris_species_petallength_k3", iris4, species,
                                        wts(iris$Petal.Length))
r$mtcars_cyl_qsec_k3 <- record(G, "mtcars_cyl_qsec_k3", mt, lab(mtcars$cyl), wts(mtcars$qsec))
reg <- as.integer(state.region)
keep <- reg <= 3
r$usarrests_regions123_pop_k3 <- record(G, "usarrests_regions123_pop_k3", as_dbl(USArrests[keep, ]),
                                        lab(reg[keep]), wts(state.x77[keep, "Population"]))
r$plantgrowth_group_k3 <- record(G, "plantgrowth_group_k3", as_dbl(PlantGrowth[, "weight", drop = FALSE]),
                                 lab(as.integer(PlantGrowth$group)), rep(1, 30))
ord <- order(eu$DAX, seq_len(nrow(eu)))
cl2 <- rep(2L, nrow(eu))
cl2[ord[1:88]] <- 1L
r$eustock_dax_low88_k2 <- record(G, "eustock_dax_low88_k2", as_dbl(eu), cl2, wts(eu$DAX),
                                 predict = FALSE)
r$mtcars_carb_wt <- record(G, "mtcars_carb_wt", mt, lab(mtcars$carb), wts(mtcars$wt))
r$faithful_ward10_erupt <- record(G, "faithful_ward10_erupt", fa, lab(ward(faithful, 10)),
                                  wts(faithful$eruptions))
r$rock_perm_shape <- record(G, "rock_perm_shape", as_dbl(rock), tenths(rock$perm), wts(rock$shape),
                            orig_cl = rock$perm)
r$lcs_ward5_dpi <- record(G, "lcs_ward5_dpi", as_dbl(LifeCycleSavings),
                          lab(ward(scale(LifeCycleSavings), 5)), wts(LifeCycleSavings$dpi))
r$quakes_ward100_stations <- record(G, "quakes_ward100_stations", q, lab(ward(scale(quakes), 100)),
                                    wts(quakes$stations), newx = q[1:100, ])
r$volcano_ward30_col1 <- record(G, "volcano_ward30_col1", as_dbl(volcano), lab(ward(volcano, 30)),
                                wts(volcano[, 1]))
ua <- as_dbl(USArrests)
ua_s <- as_dbl(scale(ua))
r$usarrests_scaled_ward2_col1 <- record(G, "usarrests_scaled_ward2_col1", ua_s, lab(ward(ua_s, 2)),
                                        wts(ua[, 1]))
# integer storage: R gets an integer matrix and integer weights, as the fixture stores them
qi <- as.matrix(quakes[, c("depth", "stations")])
dimnames(qi) <- NULL
stopifnot(is.integer(qi), is.integer(quakes$stations))
r$quakes_int2_mag5 <- record(G, "quakes_int2_mag5", qi, lab(quakes$mag >= 5), quakes$stations,
                             orig_cl = quakes$mag >= 5)
r$quakes_int_by_mag <- record(G, "quakes_int_by_mag", qi, tenths(quakes$mag), quakes$stations,
                              predict = FALSE, orig_cl = quakes$mag)

# fail closed if a case lost the property a test relies on
nan_cols <- function(id) which(apply(is.nan(r[[id]]$wmean), 2, any))
stopifnot(
  identical(nan_cols("mtcars_gear_am"), 1L), all(is.nan(r$mtcars_gear_am$wmean[, 1])),
  r$mtcars_carb_am$k == 6, identical(nan_cols("mtcars_carb_am"), 3L),
  r$mtcars_cyl_vs$k == 3, identical(nan_cols("mtcars_cyl_vs"), 3L),
  identical(nan_cols("mtcars_gear_am_row01"), 1L), nrow(r$mtcars_gear_am_row01$z) == 1,
  all(is.nan(r$mtcars_gear_am_row01$z[, 1])), r$mtcars_gear_am_row01$pred != 1,
  all(r$mtcars_gear_am$pred != 1), all(r$mtcars_carb_am$pred != 3), all(r$mtcars_cyl_vs$pred != 3),
  r$chick_diettime_time$k == 48, length(nan_cols("chick_diettime_time")) == 4,
  tie_rows(r$chick_diettime_time$z) > 0,
  any(r$iris50_singletons$z < 0, na.rm = TRUE),
  tie_rows(r$mtcars_cyl1d_gear$z) > 0, tie_rows(r$iris150_singletons_pw$z) > 0,
  all(r$iris_singletons_neg_pw$cl < 0),
  is.double(r$mtcars_cyl_carb$cl), is.integer(r$mtcars_gear_am$cl),
  is.integer(r$quakes_int2_mag5$x), is.integer(r$quakes_int2_mag5$w),
  is.integer(r$quakes_int_by_mag$x), is.integer(r$quakes_int_by_mag$w),
  ncol(r$faithful1d$x) == 1, ncol(r$faithful1d_by_waiting$x) == 1, ncol(r$mtcars_cyl1d_gear$x) == 1,
  r$iris_one_cluster$k == 1, r$mtcars_one_carb$k == 1, r$volcano_one_ones$k == 1,
  r$volcano_scaled_one_ones$k == 1,
  nrow(r$mtcars_cyl_carb_row15$newx) == 1, nrow(r$iris_species_pw_row150$newx) == 1,
  ncol(r$nug_volcano_scaled$x) == 61, ncol(r$volcano_ward4$x) == 61,
  r$quakes_ward100_stations$k == 100, r$volcano_ward30_col1$k == 30, r$chick_diettime_time$k == 48,
  any(r$nug_iris_passthrough$w == 0), any(r$nug_iris_passthrough_singletons$w == 0),
  any(is.nan(r$mtcars_singletons_vs$distw)), any(is.nan(r$nug_iris_passthrough_singletons$distw)),
  !any(vapply(r, function(e) e$pred_list, TRUE)),
  identical(unname(iris4[102, ]), unname(iris4[143, ])), nrow(r$iris_singletons_neg_pw$z) == 5,
  tie_rows(r$iris_singletons_neg_pw$z) >= 2, identical(r$iris_singletons_neg_pw$pred[4:5], c(8L, 8L)),
  all(vapply(r[c("faithful1d_by_waiting", "infert_stratum_parity",
                 "iris3_singletons_pw", "iris12_singletons_pl", "iris1_singletons_pw",
                 "iris124_singletons_pl", "quakes200_singletons_st", "nug_iris_passthrough_singletons",
                 "eustock_dax_low88_k2", "quakes_int_by_mag")],
             function(e) is.null(e[["z"]]) && !is.null(e[["distw"]]), TRUE)),
  identical(dim(r$quakes_ward100_stations$z), c(100L, 100L)),
  is.integer(r$quakes_int2_mag5$newx), !is.null(r$quakes_int2_mag5$z)
)
# the k = 3 cases: R's long double total differs from a double sum of the groups
for (id in c("iris_species_petallength_k3", "mtcars_cyl_qsec_k3", "usarrests_regions123_pop_k3",
             "plantgrowth_group_k3")) {
  g <- r[[id]]$wwcss_group
  stopifnot(r[[id]]$k == 3, !identical(r[[id]]$wwcss_total, (g[1] + g[2]) + g[3]))
}
# a long double sum of two terms inside distw: the two groups' double sum differs
local({
  g <- r$eustock_dax_low88_k2$wwcss_group
  stopifnot(r$eustock_dax_low88_k2$k == 2, !identical(sum(c(g[1], g[2])), g[1] + g[2]))
})
# zero and negative distw entries among the singleton cases
local({
  lo <- unlist(lapply(c("iris3_singletons_pw", "iris12_singletons_pl", "iris1_singletons_pw",
                        "iris124_singletons_pl"), function(id) {
    d <- r[[id]]$distw
    d[lower.tri(d)]
  }))
  stopifnot(any(lo < 0, na.rm = TRUE), any(lo == 0, na.rm = TRUE))
})
cat(G, length(r), "cases\n")

# ---- group wstats_constructed: built points and built weights on real data ----
G <- "wstats_constructed"
rc <- list()
shift_ulp <- function(x, k) vapply(x, function(v) {
  stopifnot(is.finite(v), v != 0)
  r <- as.integer(writeBin(v, raw(), endian = "big"))
  hi <- sum(r[1:4] * 256^(3:0))
  lo <- sum(r[5:8] * 256^(3:0))
  lo <- lo + if (v < 0) -k else k
  if (lo >= 2^32) {
    lo <- lo - 2^32
    hi <- hi + 1
  } else if (lo < 0) {
    lo <- lo + 2^32
    hi <- hi - 1
  }
  readBin(as.raw(c((hi %/% 256^(3:0)) %% 256, (lo %/% 256^(3:0)) %% 256)), "double", n = 1,
          endian = "big")
}, 0)
stopifnot(shift_ulp(1, 1) == 1 + 2^-52, shift_ulp(1, -1) == 1 - 2^-53, shift_ulp(-1, 1) == -(1 - 2^-53),
          shift_ulp(-1, -1) == -(1 + 2^-52), shift_ulp(3.5, 2) == 3.5 + 2^-50)
# the midpoint of every pair of R's weighted centers, then each midpoint with its first
# coordinate moved by one and two ulp either way, every coordinate by one ulp either way, and its
# last coordinate by one ulp either way
tie_newx <- function(x, cl, w) {
  ctr <- WCluster::wmean(x, cl, w)
  pr <- t(combn(ncol(ctr), 2))
  mid <- t(apply(pr, 1, function(ij) (ctr[, ij[1]] + ctr[, ij[2]]) / 2))
  dimnames(mid) <- NULL
  nud <- function(f) t(apply(mid, 1, f))
  blocks <- list(mid = mid,
                 c1_up1 = nud(function(r) { r[1] <- shift_ulp(r[1], 1); r }),
                 c1_dn1 = nud(function(r) { r[1] <- shift_ulp(r[1], -1); r }),
                 c1_up2 = nud(function(r) { r[1] <- shift_ulp(r[1], 2); r }),
                 c1_dn2 = nud(function(r) { r[1] <- shift_ulp(r[1], -2); r }),
                 all_up1 = nud(function(r) shift_ulp(r, 1)), all_dn1 = nud(function(r) shift_ulp(r, -1)),
                 last_up1 = nud(function(r) { r[length(r)] <- shift_ulp(r[length(r)], 1); r }),
                 last_dn1 = nud(function(r) { r[length(r)] <- shift_ulp(r[length(r)], -1); r }))
  newx <- do.call(rbind, blocks)
  dimnames(newx) <- NULL
  newx
}
ties <- function(id, x, cl, w) record(G, id, x, cl, w, newx = tie_newx(x, cl, w))
rc$ties_iris <- ties("ties_iris", iris4, species, wts(iris$Petal.Width))
rc$ties_mtcars <- ties("ties_mtcars", mt, lab(mtcars$cyl), wts(mtcars$carb))
rc$ties_quakes <- ties("ties_quakes", q, lab(bands), wts(quakes$stations))
rc$ties_volcano <- ties("ties_volcano", as_dbl(volcano), lab(ward(volcano, 8)), rep(1, 87))
rc$ties_faithful <- ties("ties_faithful", fa, lab(ward(faithful, 10)), rep(1, 272))
# standardized data: x = scale(D), cl = ward(scale(D), K), w = column 1 of D or ones
scaled_ties <- function(id, d, k, ones) {
  raw <- as_dbl(d)
  sx <- as_dbl(scale(raw))
  w <- if (ones) rep(1, nrow(raw)) else raw[, 1]
  ties(id, sx, lab(ward(sx, k)), w)
}
rc$swiss_scaled_ward4_col1 <- scaled_ties("swiss_scaled_ward4_col1", swiss, 4, FALSE)
rc$mtcars_scaled_ward6_col1 <- scaled_ties("mtcars_scaled_ward6_col1", mtcars, 6, FALSE)
rc$lcs_scaled_ward2_ones <- scaled_ties("lcs_scaled_ward2_ones", LifeCycleSavings, 2, TRUE)

# heavy weights: real x, weights built from a real column, ratios up to 5e36. No newx.
hv <- function(id, x, cl, w, orig_cl = NULL) record(G, id, x, cl, w, predict = FALSE, orig_cl = orig_cl)
rc$hv_quakes_by_mag_w2pow <- hv("hv_quakes_by_mag_w2pow", q, tenths(quakes$mag), 2^quakes$stations,
                                orig_cl = quakes$mag)
rc$hv_chick_by_chick_w2pow <- hv("hv_chick_by_chick_w2pow", as_dbl(cw[, c("weight", "Time")]),
                                 lab(cw$Chick), 2^(cw$weight / 8))
rc$hv_quakes_by_stations_w2pow <- hv("hv_quakes_by_stations_w2pow", as_dbl(quakes[, 1:4]),
                                     lab(quakes$stations), 2^quakes$stations)
yr <- floor(as.numeric(time(EuStockMarkets)))
rc$hv_eustock_year_wdax_1e296 <- hv("hv_eustock_year_wdax_1e296", as_dbl(eu), lab(yr), eu$DAX * 1e296)
rc$hv_faithful_by_waiting_wsubnormal <- hv("hv_faithful_by_waiting_wsubnormal", fa,
                                           lab(faithful$waiting), faithful$waiting * 2^-1070)

# rows whose squares overflow. R's wdist_mat multiplies by an identity matrix, so 0 * Inf from
# another row makes an entry NaN, while a lone Inf survives.
ox <- rbind(c(1, 2), c(3, 4), c(1e200, 0))
rc$ovf_one_row <- record(G, "ovf_one_row", ox, c(1L, 1L, 2L), c(1, 1, 1), newx = rbind(c(1, 2)))
rc$ovf_two_rows <- record(G, "ovf_two_rows", ox, c(1L, 1L, 2L), c(1, 1, 1),
                          newx = rbind(c(1, 2), c(2, 2)))
rc$ovf_newx_row <- record(G, "ovf_newx_row", rbind(c(1, 2), c(3, 4), c(5, 6)), c(1L, 2L, 2L),
                          c(1, 1, 1), newx = rbind(c(1, 2), c(1e200, 0), c(4, 4)))
# the first cluster's weights sum to zero, so its center is NaN, and the one new row overflows
# against the live centers: which.min skips the NaN and takes the first of the tied Infs
rc$ovf_dead_first_nan_inf <- record(G, "ovf_dead_first_nan_inf", rbind(c(1, 2), c(3, 4), c(5, 6)),
                                    c(1L, 2L, 3L), c(0, 1, 1), newx = rbind(c(1e200, 0)))

stopifnot(
  identical(rc$ovf_one_row$z, matrix(c(2, Inf), 1)), identical(rc$ovf_one_row$pred, 1L),
  all(is.finite(rc$ovf_two_rows$z[, 1])), all(is.nan(rc$ovf_two_rows$z[, 2])),
  identical(rc$ovf_two_rows$pred, c(1L, 1L)),
  rc$ovf_newx_row$pred_list, all(is.nan(rc$ovf_newx_row$z[c(1, 3), ])),
  all(rc$ovf_newx_row$z[2, ] == Inf), is.null(rc$ovf_newx_row[["pred"]]),
  rc$ovf_dead_first_nan_inf$k == 3, is.nan(rc$ovf_dead_first_nan_inf$z[1, 1]),
  all(rc$ovf_dead_first_nan_inf$z[1, -1] == Inf), identical(rc$ovf_dead_first_nan_inf$pred, 2L),
  sum(vapply(rc[grep("^ties_", names(rc))], function(e) tie_rows(e$z), 0)) > 0,
  is.infinite(rc$hv_eustock_year_wdax_1e296$wss), is.finite(rc$hv_eustock_year_wdax_1e296$wwcss_total),
  any(rc$hv_quakes_by_stations_w2pow$distw < 0, na.rm = TRUE),
  any(rc$hv_quakes_by_stations_w2pow$distw == 0, na.rm = TRUE),
  all(vapply(rc[grep("^hv_", names(rc))], function(e) is.null(e[["z"]]), TRUE))
)
cat(G, length(rc), "cases\n")
