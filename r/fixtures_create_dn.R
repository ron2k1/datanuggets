# Fixtures for create.DN: inputs, every random draw, every create.DNcenters call, outputs.
source("r/fixture_io.R")

grid <- as_dbl(expand.grid(a = 0:5, b = 0:5))
set.seed(40)
v40 <- rnorm(40)
datasets <- list(
  iris = as_dbl(iris[, 1:4]),
  faithful = as_dbl(faithful),
  quakes = as_dbl(quakes),
  faithful_1d = as_dbl(faithful[, "eruptions", drop = FALSE]),
  grid_ties = grid[rep(seq_len(nrow(grid)), 4), , drop = FALSE],  # every point 4 times
  # 6 constant rows, mathematically equidistant from 4 coordinate permutations of one vector:
  # p = 40 > 32, so R's assignment sums go through BLAS ddot and round-off decides every constant row
  perm40 = rbind(outer(seq(-0.25, 0.25, by = 0.1), rep(1, 40)), t(replicate(4, sample(v40))))
)
# The thinning breaks the last constant-versus-permutation tie toward the lower index, so the
# constants are deleted (leaving the permutations as the only centers) only when create.DN's
# shuffle keeps all 6 constants ahead of the permutations. Pick the first seed that does.
perm40_seed <- Find(function(s) { set.seed(s); all(sample(10, 10)[1:6] <= 6) }, seq_len(10000))

cases <- list()
add_case <- function(id, data, center.method, R, DN.num1, DN.num2,
                     delete.percent = 0.1, dist.metric = "euclidean", seed = 291102) {
  cases[[id]] <<- list(data = data, args = list(
    center.method = center.method, R = R, delete.percent = delete.percent,
    DN.num1 = DN.num1, DN.num2 = DN.num2, dist.metric = dist.metric, seed = seed))
}
sizes <- list(iris = c(100, 60, 20), faithful = c(100, 90, 30), quakes = c(300, 400, 100))
for (ds in names(sizes)) for (metric in c("euclidean", "manhattan")) {
  for (cm in c("mean", "random", "original")) {
    s <- sizes[[ds]]
    add_case(paste(ds, metric, cm, sep = "_"), ds, cm, s[1], s[2], s[3], dist.metric = metric)
  }
}
add_case("quakes_onechunk_dp30", "quakes", "mean", 5000, 500, 150, delete.percent = 0.3)
add_case("iris_default_num1_warnpath", "iris", "mean", 5000, 10^4, 50)
add_case("iris_dn2_equals_stage1", "iris", "mean", 100, 60, 60)
add_case("faithful_1d", "faithful_1d", "mean", 100, 90, 30)
add_case("grid_ties_euclidean_mean", "grid_ties", "mean", 100, 40, 12)
add_case("grid_ties_manhattan_random", "grid_ties", "random", 100, 40, 12,
         delete.percent = 0.3, dist.metric = "manhattan")
add_case("quakes_seed7_random", "quakes", "random", 300, 400, 100, seed = 7)
add_case("perm40_euclidean_mean", "perm40", "mean", 100, 4, 4, seed = perm40_seed)
add_case("perm40_manhattan_mean", "perm40", "mean", 100, 4, 4, dist.metric = "manhattan",
         seed = perm40_seed)

for (id in names(cases)) {
  cs <- cases[[id]]
  x <- datasets[[cs$data]]
  warns <- character()
  run <- function() withCallingHandlers(
    suppressMessages(do.call(create.DN, c(list(x = x), cs$args, list(no.cores = 1, make.pbs = FALSE)))),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  res <- capture_centers(capture_draws(run()))
  dn <- res$value$value
  draws <- res$value$draws
  calls <- res$calls

  n <- nrow(x)
  chunks <- parallel::splitIndices(n, ceiling(n / cs$args$R))
  chunks <- chunks[lengths(chunks) > 0L]
  if (length(calls) != length(chunks) + 1L) {
    stop(id, ": captured ", length(calls), " create.DNcenters calls, expected ", length(chunks) + 1L)
  }
  stopifnot(all(vapply(draws, function(d) length(d$value) == d$size, logical(1))))

  tab <- dn[["Data Nuggets"]]
  p <- ncol(x)
  if (id == "perm40_euclidean_mean") {
    # Fail closed unless the case separates the two summation orders: the final centers must be
    # the permutations, and Rfast::Dist (even/odd sums) must pick a different nearest center than
    # create.DN's dista (BLAS ddot) for at least one constant row
    fin <- calls[[2]]$RS[calls[[2]]$kept, , drop = FALSE]
    stopifnot(all(apply(fin, 1, function(r) diff(range(r)) > 0)))
    d <- unname(Rfast::Dist(rbind(x[1:6, ], fin)))[1:6, 6 + seq_len(nrow(fin))]
    stopifnot(any(apply(d, 1, which.min) != dn[["Data Nugget Assignments"]][1:6]))
  }
  arrays <- list(
    x = x,
    draw_n = as.integer(vapply(draws, function(d) as.numeric(d$n), numeric(1))),
    draw_size = as.integer(vapply(draws, function(d) as.numeric(d$size), numeric(1))),
    draw_values = as.integer(unlist(lapply(draws, function(d) d$value))),
    centers = unname(as.matrix(tab[, 1 + seq_len(p)])),
    weights = as.integer(tab[["Weight"]]),
    scales = as.double(tab[["Scale"]]),
    assignments = as.integer(dn[["Data Nugget Assignments"]])
  )
  for (i in seq_along(calls)) {
    arrays[[sprintf("call%02d_rs", i)]] <- unname(calls[[i]]$RS)
    arrays[[sprintf("call%02d_kept", i)]] <- as.integer(calls[[i]]$kept)
  }
  write_case("create_dn", id, cs$args, arrays, extra = list(
    n_calls = length(calls),
    call_dn_num = I(vapply(calls, function(cl) as.numeric(cl$DN.num), numeric(1))),
    warnings = I(warns)))
  cat(sprintf("%-30s nuggets=%4d draws=%4d calls=%2d warnings=%d\n",
              id, nrow(tab), length(draws), length(calls), length(warns)))
}
