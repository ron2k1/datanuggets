# Fixtures for the R statistics refine.DN depends on: cov, sum, mean, median, mad and type-7
# quantile (long double accumulation), unique() on matrix rows, %*% and eigen(symmetric = TRUE).
source("r/fixture_io.R")

cov_sets <- list(
  iris = as_dbl(iris[, 1:4]),
  iris_setosa = as_dbl(iris[1:50, 1:4]),
  quakes = as_dbl(quakes),
  faithful = as_dbl(faithful),
  mtcars = as_dbl(mtcars),
  volcano5 = as_dbl(scale(volcano)[, 1:5]),
  big_offset = cbind(1e8 + faithful[, 1], faithful[, 2]),  # the mean's correction pass matters
  one_col = as_dbl(faithful[, 1, drop = FALSE]),
  two_rows = as_dbl(iris[1:2, 1:4])
)
for (nm in names(cov_sets)) {
  x <- cov_sets[[nm]]
  S <- cov(x)
  arrays <- list(x = x, cov = unname(S), scale = sum(diag(S)) / ncol(x))
  if (ncol(x) > 1L) {
    ev <- eigen(S, symmetric = TRUE)
    arrays <- c(arrays, list(
      values = eigen(S, symmetric = TRUE, only.values = TRUE)$values,
      values_v = ev$values,
      vectors = ev$vectors,
      proj = as.numeric(x %*% ev$vectors[, 1, drop = FALSE])))
  }
  write_case("rcompat_stats", paste0("cov_", nm), list(kind = "cov"), arrays)
}

set.seed(3)
vec_sets <- list(
  ev = eigen(cov(quakes))$values,
  scales = c(0.1, 0.2, 0.3, 1/3, 2/3, 0.7, 1e-17, 5),
  even = c(0.1 + 0.2, 0.3, 1/3, 1e-3),
  odd = rnorm(7),
  iris_col = iris[, 1],
  cancel = c(1e16, 1, -1e16, 3.5),
  ratios = c(1.9999999999999998, 2.0000000000000004),
  rn = rnorm(100) * 1e3,
  huge = c(1e308, 1e308, -1e307)  # the long double sum exceeds DBL_MAX: mean() re-sums x / n
)
probs <- c(0.5, 0.9, 0.25, 0.1, 0.999, 0.3)
for (nm in names(vec_sets)) {
  v <- as.double(vec_sets[[nm]])
  write_case("rcompat_stats", paste0("vec_", nm), list(kind = "vec", probs = probs), list(
    v = v, sum = sum(v), mean = mean(v), median = median(v), mad = mad(v),
    q = quantile(v, probs, names = FALSE)))
}

mean_sets <- list(inf = c(Inf, 2), inf_minus_inf = c(Inf, -Inf), neg_inf = c(-Inf, 1),
                  nan = c(NaN, 1), pair = c(1.5, 2.5000000000000004),
                  overflow_neg_inf = c(1.7e308, 1.7e308, -Inf),  # double sums give NaN here
                  # the sum overflows a double, so the correction pass divides each residual by n
                  overflow_careful = from_bits(c("7fe0000000000000", "7feffffffffffffd",
                                                 "fc00000000000000", "7c80000000000000",
                                                 "7fefffffffffffff", "fbd0000000000000",
                                                 "fca0000000000000")))
# Which NaN a sum or mean gives: Inf - Inf, NA's payload, and each order of two NaNs
nan_sets <- list(na = c(NA, 1), inf_minus_inf_na = c(Inf, -Inf, NA), na_inf_minus_inf = c(NA, Inf, -Inf),
                 na_pay = c(NA, nan_pay), pay_na = c(nan_pay, NA), pos_neg = c(nan_pos, nan_neg),
                 neg_pos = c(nan_neg, nan_pos), na_snan = c(NA, snan_big), snan_na = c(snan_big, NA))
names(nan_sets) <- paste0("nanbits_", names(nan_sets))
mean_sets <- c(mean_sets, nan_sets)
for (nm in names(mean_sets)) {
  v <- mean_sets[[nm]]
  write_case("rcompat_stats", paste0("mean_", nm), list(kind = "mean"), list(v = v, mean = mean(v)))
}

# sum() is R's rsum: a long double running sum that is +-Inf once it passes +-DBL_MAX, even where
# rounding would give DBL_MAX back. With Inf or NaN present only those decide.
big <- .Machine$double.xmax
sum_sets <- list(past_max = c(big, 2^960), past_neg_max = c(-big, -2^960), at_max = c(big, 0),
                 overflow_neg_inf = c(1.7e308, 1.7e308, -Inf),
                 overflow_inf = c(-1.7e308, -1.7e308, Inf),
                 inf_minus_inf = c(Inf, -Inf, 1), nan_inf = c(NaN, Inf))
sum_sets <- c(sum_sets, nan_sets)
for (nm in names(sum_sets)) {
  v <- sum_sets[[nm]]
  write_case("rcompat_stats", paste0("sum_", nm), list(kind = "sum"), list(v = v, sum = sum(v)))
}

# R 4.x compares matrix rows exactly (0 == -0), not by a 15-digit paste
unique_sets <- list(
  petal = as_dbl(iris[, 3:4]),
  signed_zero = rbind(c(0.1 + 0.2, 1), c(0.3, 1), c(-0, 2), c(0, 2), c(0.3, 1)),
  near_dup = rbind(c(1, 1), c(1 + 2^-52, 1), c(1, 1)),
  one_col = matrix(c(3, 1, 3, 2, 1, -0, 0), ncol = 1)
)
for (nm in names(unique_sets)) {
  x <- unique_sets[[nm]]
  write_case("rcompat_stats", paste0("unique_", nm), list(kind = "unique"),
             list(x = x, first = which(!duplicated(x))))
}
cat(length(cov_sets) + length(vec_sets) + length(mean_sets) + length(sum_sets) +
    length(unique_sets), "cases\n")
