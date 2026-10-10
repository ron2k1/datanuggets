# Fixtures for WCluster's Whclust: merge, height and order, or the error R stops with, plus the
# counts r/wcluster_counted.R's copy took, which prove each case reaches the quirk it's named
# for. Whclust draws nothing, so a case is its x and w.
source("r/fixture_io.R")
source("r/wcluster_counted.R")

add_whclust <- function(id, x, w, quirks = character()) {
  x <- as_dbl(x)
  w <- as.double(w)
  t0 <- Sys.time()
  pr <- prove(id, counted_whclust, wcluster_whclust, list(x = x, w = w), quirks)
  arrays <- list(x = x, w = w)
  err <- if (inherits(pr$value, "r_error")) as.character(pr$value) else ""
  if (!nzchar(err))
    arrays <- c(arrays, list(merge = pr$value$merge, height = pr$value$height,
                             order = pr$value$order))
  write_case("whclust", id, no_args, arrays, extra = list(error = err, counts = pr$counts))
  cat(sprintf("whclust %-16s n=%3d %5.1fs error=%s inf_pick=%d self_merge=%d\n", id, nrow(x),
              as.numeric(Sys.time() - t0, units = "secs"), gsub("\n", " ", err),
              pr$counts$inf_pick, pr$counts$self_merge))
}

iris4 <- as_dbl(iris[, 1:4])
w1372 <- function(n) rep(c(1, 3, 7, 2), length.out = n)

# iris has two equal rows, 102 and 143, so one distance is exactly zero
add_whclust("iris_unit", iris4, rep(1, 150))
add_whclust("iris_w1372", iris4, w1372(150))
add_whclust("quakes200_w1372", quakes[1:200, ], w1372(200))
# Integer points, each twice: many exact ties in distance, decided by R's first minimum
add_whclust("grid_dup", rbind(expand.grid(1:4, 1:3), expand.grid(1:4, 1:3)), rep(1, 24))
add_whclust("n3", iris4[c(1, 51, 101), ], rep(1, 3))
add_whclust("one_col", faithful[1:40, 1, drop = FALSE], rep(1, 40))

# A row with weight zero has NaN distances, so it merges only once every distance left is NaN:
# then which.min lands on the first Inf, row 1 of the first merged column
w <- rep(1, 30); w[7] <- 0
add_whclust("zero_w1", iris4[1:30, ], w, "inf_pick")
w[19] <- 0
add_whclust("zero_w2", iris4[1:30, ], w, "inf_pick")

# Weights 1, 1, -1, -1 leave two merged clusters whose pooled weight is zero, so their distance
# is NaN and which.min's first Inf is the merged column's own diagonal: a cluster merged with
# itself at the last step, which R returns without an error
add_whclust("selfmerge_last", c(0, 0.1, 5, 5.1), c(1, 1, -1, -1), c("inf_pick", "self_merge"))
# Three tight pairs 1e160 apart: each pair merges, then every distance between pairs overflows to
# Inf, and with no singleton left the first Inf is the diagonal again, now with three clusters
# left. R's counter m and its matrix then disagree in size for the last step.
b <- 1e160
e <- 2e144
add_whclust("selfmerge_mid", c(0, 1, b, b + e, -b, -b - e), rep(1, 6),
            c("inf_pick", "self_merge", "self_merge_mid"))
# Three singletons whose pairs all overflow: the first Inf is in a singleton's column. The merged
# pair's own wss is Inf too, so the last distance is Inf - Inf, NaN, and the pick is row 1 of the
# merged column.
add_whclust("inf_single", c(0, b, -b), rep(1, 3), "inf_pick")

# Where R stops: forder() reads merge[2, ] when n = 2, distw() needs two clusters, and with every
# weight zero every distance is NaN with no merged column to give an Inf
add_whclust("n2", iris4[1:2, ], c(1, 1))
add_whclust("n1", iris4[1, , drop = FALSE], 1)
add_whclust("no_candidate", iris4[1:5, ], rep(0, 5))
