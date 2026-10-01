# Fixtures for the two R operations WCluster's statistics are built on: %*% of two matrices
# (reference BLAS, matprod "default") and weighted.mean (long double sums). Plus the boundary
# vectors that pin R's long double sum(): its order dependence and where it maps to +-Inf.
source("r/fixture_io.R")

neg <- function(v) 1 / v == -Inf  # TRUE where v is -0
dbl_vec <- function(v) {
  v <- as.double(v)
  names(v) <- NULL
  v
}

# %*%: every case uses R's built-in data, except the -0 column. mtcars times the centers of
# wmean(mtcars, gear, am) has a NaN first column, since no gear 3 car has am = 1.
bands <- cut(quakes$depth, c(0, 150, 300, 450, 700), labels = FALSE)
iris4 <- as_dbl(iris[, 1:4])
sv <- as_dbl(scale(volcano))
q <- as_dbl(quakes)
fa <- as_dbl(faithful)
mt <- as_dbl(mtcars)
negzero_b <- cbind(rep(-0, 4), dbl_vec(iris[6, 1:4]))  # built: a column of -0
matmul_sets <- list(
  matmul_iris_crossprod = list(a = t(iris4), b = iris4),
  matmul_volcano_scaled_crossprod = list(a = t(sv), b = sv),
  matmul_quakes_depthband_centers = list(
    a = q, b = as_dbl(WCluster::wmean(quakes, bands, quakes$stations))),
  matmul_faithful_row_times_matrix = list(a = as_dbl(matrix(faithful$waiting, nrow = 1)), b = fa),
  matmul_faithful_matrix_times_col = list(a = fa, b = as_dbl(matrix(dbl_vec(faithful[1, ]), ncol = 1))),
  matmul_mtcars_gear_am_centers_nan = list(
    a = mt, b = as_dbl(WCluster::wmean(mtcars, mtcars$gear, mtcars$am))),
  matmul_negzero_column = list(a = as_dbl(iris[1:5, 1:4]), b = negzero_b)
)
for (nm in names(matmul_sets)) {
  s <- matmul_sets[[nm]]
  prod <- s$a %*% s$b
  write_case("rcompat_wstats", nm, list(kind = "matmul"), list(a = s$a, b = s$b, prod = prod))
}
# fail closed if a case lost the property it's there for
local({
  p <- matmul_sets$matmul_mtcars_gear_am_centers_nan
  prod <- p$a %*% p$b
  stopifnot(all(is.nan(prod[, 1])), all(is.finite(prod[, -1])))
  z <- matmul_sets$matmul_negzero_column
  stopifnot(all(neg(z$b[, 1])), all(z$a > 0))
  prod <- z$a %*% z$b
  stopifnot(all(prod[, 1] == 0), !any(neg(prod[, 1])))  # R sums the -0 products to +0
})

# weighted.mean: sum((v * w)[w != 0]) / sum(w). The c(1, -1) weights are built: they sum to
# zero under a nonzero numerator, so R returns an infinity. The Inf at a zero weight is built
# too: [w != 0] drops its Inf * 0 = NaN term, so R's mean is finite where the plain sum is NaN.
nug <- file.path("tests", "fixtures", "create_dn", "quakes_euclidean_mean")
gear3 <- mtcars$gear == 3
wmean_sets <- list(
  wmean_iris_sepallength_petalwidth = list(v = iris$Sepal.Length, w = iris$Petal.Width),
  wmean_mtcars_mpg_vs = list(v = mtcars$mpg, w = mtcars$vs),
  wmean_mtcars_gear3_mpg_am_zero_total = list(v = mtcars$mpg[gear3], w = mtcars$am[gear3]),
  wmean_quakes_depth_stations = list(v = quakes$depth, w = quakes$stations),
  wmean_nug_quakes_center1 = list(v = read_array(nug, "centers")[, 1], w = read_array(nug, "weights")),
  wmean_iris_sepallength_weights_sum_zero = list(v = iris$Sepal.Length[1:2], w = c(1, -1)),
  wmean_iris_sepallength_inf_at_zero_weight = list(v = c(Inf, iris$Sepal.Length[1]), w = c(0, 1))
)
for (nm in names(wmean_sets)) {
  v <- dbl_vec(wmean_sets[[nm]]$v)
  w <- dbl_vec(wmean_sets[[nm]]$w)
  wmean_sets[[nm]]$mean <- weighted.mean(v, w)
  write_case("rcompat_wstats", nm, list(kind = "wmean"),
             list(v = v, w = w, mean = wmean_sets[[nm]]$mean))
}
stopifnot(is.nan(wmean_sets$wmean_mtcars_gear3_mpg_am_zero_total$mean),
          is.infinite(wmean_sets$wmean_iris_sepallength_weights_sum_zero$mean),
          any(wmean_sets$wmean_mtcars_mpg_vs$w == 0))
local({
  s <- wmean_sets$wmean_iris_sepallength_inf_at_zero_weight
  stopifnot(s$mean == iris$Sepal.Length[1], is.nan(sum(s$v * s$w) / sum(s$w)))
})

# sum(): R's rsum adds in long double and gives +-Inf past +-DBL_MAX, else rounds once to double.
xm <- .Machine$double.xmax
sum_sets <- list(
  sum_xmax_pow969 = c(xm, 2^969),  # exact in long double and past DBL_MAX: Inf
  sum_xmax_pow970 = c(xm, 2^970),
  sum_xmax_pow960 = c(xm, 2^960),  # one long double ulp at this size
  sum_xmax_pow959 = c(xm, 2^959),  # half an ulp: a tie, stays at DBL_MAX
  sum_xmax_pow959_pow959 = c(xm, 2^959, 2^959),  # each half-ulp add rounds back
  sum_pow959_pow959_xmax = c(2^959, 2^959, xm),  # same numbers, other order
  sum_xmax_pow917 = c(xm, 2^917),
  sum_xmax_pow969_neg_pow969 = c(xm, 2^969, -2^969),  # back below DBL_MAX before the end
  sum_1e308_twice_minus_once = c(1e308, 1e308, -1e308),  # past DBL_MAX midway, finite at the end
  sum_1e308_thrice_minus_twice = c(1e308, 1e308, 1e308, -1e308, -1e308),
  sum_neg_xmax_neg_pow969 = c(-xm, -2^969),
  sum_neg_xmax_neg_pow959 = c(-xm, -2^959),
  sum_xmax_xmax_neg_xmax_neg_xmax = c(xm, xm, -xm, -xm),
  sum_xmax_neg_xmax_xmax = c(xm, -xm, xm),
  sum_subnormal_thrice = rep(2^-1074, 3),
  sum_subnormal_alternating = c(2^-1074, -2^-1074, 2^-1074),
  sum_one_powm64_powm64 = c(1, 2^-64, 2^-64),  # below long double resolution, each add lost
  sum_powm64_powm64_one = c(2^-64, 2^-64, 1),  # small ones first: kept
  sum_one_powm53_powm53 = c(1, 2^-53, 2^-53),  # long double keeps both, rounds once at the end
  sum_one_powm53 = c(1, 2^-53),  # an exact tie at the cast to double: to even
  sum_one_powm53_powm63 = c(1, 2^-53, 2^-63),  # just above the tie in long double
  sum_one_powm53_powm64 = c(1, 2^-53, 2^-64)  # a long double tie: to even, back on the double tie
)
for (nm in names(sum_sets)) {
  v <- sum_sets[[nm]]
  write_case("rcompat_wstats", nm, list(kind = "sum"), list(v = v, sum = sum(v)))
}
cat(length(matmul_sets) + length(wmean_sets) + length(sum_sets), "cases\n")
