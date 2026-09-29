# Fixtures pinning the R and Rfast semantics create.DN depends on.
source("r/fixture_io.R")

# parallel::splitIndices over a grid of (nx, ncl)
pairs <- expand.grid(nx = c(1, 2, 3, 7, 100, 150, 272, 999, 1000, 1001, 5000, 12345),
                     ncl = c(1, 2, 3, 4, 7, 10, 25))
pairs <- pairs[pairs$ncl <= pairs$nx, ]
lens <- list()
nchunks <- integer(nrow(pairs))
for (i in seq_len(nrow(pairs))) {
  ch <- parallel::splitIndices(pairs$nx[i], pairs$ncl[i])
  stopifnot(identical(as.integer(unlist(ch)), seq_len(pairs$nx[i])))  # contiguous, in order
  nchunks[i] <- length(ch)
  lens[[i]] <- lengths(ch)
}
write_case("rcompat_split_indices", "grid", no_args, list(
  nx = as.integer(pairs$nx), ncl = as.integer(pairs$ncl),
  nchunks = nchunks, chunk_len = as.integer(unlist(lens))))

# Rfast::Dist, compared bit for bit
dist_sets <- list(
  iris40 = as_dbl(iris[1:40, 1:4]),
  quakes40 = as_dbl(quakes[1:40, ]),
  mtcars = as_dbl(mtcars),                        # p = 11: odd p exercises the leftover accumulator
  faithful1d = as_dbl(faithful[1:40, "eruptions", drop = FALSE])
)
for (nm in names(dist_sets)) for (metric in c("euclidean", "manhattan")) {
  x <- dist_sets[[nm]]
  write_case("rcompat_dist", paste(nm, metric, sep = "_"), list(metric = metric),
             list(x = x, dist = unname(Rfast::Dist(x, method = metric))))
}

# Rfast::dista index output: the complete std::sort order and the k = 1 nearest index
grid <- as_dbl(expand.grid(a = 0:5, b = 0:5))
dista_sets <- list(
  iris = list(xnew = as_dbl(iris[, 1:4]), x = as_dbl(iris[seq(1, 150, by = 5), 1:4])),
  grid = list(xnew = grid, x = grid[c(1, 8, 15, 22, 29, 36, 3, 18, 33, 6, 31, 13), ]),
  grid_dups = list(xnew = grid, x = grid[c(8, 8, 15, 15, 22, 1, 36, 29, 29, 29, 3, 3, 18, 18,
                                           33, 6, 31, 13, 13, 20), ])
)
for (nm in names(dista_sets)) for (metric in c("euclidean", "manhattan")) {
  s <- dista_sets[[nm]]
  # dista(trans = TRUE, the default) returns one row per xnew row; std::sort runs over all
  # nrow(x) distances whatever k is, so k = nrow(x) exposes the complete order behind k = 1.
  full <- Rfast::dista(s$xnew, s$x, type = metric, k = nrow(s$x), index = TRUE)
  near <- Rfast::dista(s$xnew, s$x, type = metric, k = 1L, index = TRUE)
  stopifnot(identical(dim(full), c(nrow(s$xnew), nrow(s$x))), full[, 1] == near)
  storage.mode(full) <- "integer"
  write_case("rcompat_dista", paste(nm, metric, sep = "_"), list(metric = metric),
             list(xnew = s$xnew, x = s$x, order_full = unname(full), nearest = as.integer(near)))
}

# Rfast::rowMins index on a matrix full of ties
m <- unname(Rfast::Dist(grid, method = "manhattan"))
diag(m) <- max(m) + 1
write_case("rcompat_rowmins", "grid_manhattan", no_args,
           list(m = m, index = as.integer(Rfast::rowMins(m, value = FALSE))))
cat("rcompat fixtures written\n")
