# Fixtures for stats::cutree and stats:::plot.hclust on the trees R's Whclust and DN.Whclust
# returned (read back from the whclust and dn_wcluster groups, recorded on this same R), and for
# the weight colors WCluster's DN help examples draw nuggets with.
source("r/fixture_io.R")

tree_of <- function(dir) structure(list(merge = read_array(dir, "merge"),
  height = read_array(dir, "height"), order = read_array(dir, "order")), class = "hclust")
stop_message <- function(expr) tryCatch({ force(expr); "" }, error = conditionMessage)

# plot.hclust's polylines, read back from an uncompressed PDF. R's pdf device writes a polyline
# as "x y m", one "x y l" per further point and "S", in big points from the page's bottom left,
# rounded to two decimals. The map from user units to device units is linear in each axis, so
# grconvertX and grconvertY at 0 and 1 invert it.
# A tree with NaN or Inf heights that R doesn't stop on is drawn off the page with a warning, so
# only whether R stopped is recorded for it.
dend_segments <- function(tree, hang = 0.1) {
  f <- tempfile(fileext = ".pdf")
  pdf(f, width = 7, height = 5, compress = FALSE)
  err <- suppressWarnings(
    stop_message(plot(tree, labels = FALSE, axes = FALSE, ann = FALSE, hang = hang)))
  gx <- grconvertX(c(0, 1), "user", "device")
  gy <- grconvertY(c(0, 1), "user", "device")
  dev.off()
  nonfinite <- any(!is.finite(tree$height))
  if (nzchar(err) || nonfinite) { unlink(f); return(list(error = err, nonfinite = nonfinite)) }
  lines <- readLines(f, warn = FALSE)
  unlink(f)
  # The first line is binary, so match bytes, then split the ASCII lines that are left
  lines <- lines[grepl("^(-?[0-9.]+ -?[0-9.]+ [ml]|S)$", lines, useBytes = TRUE)]
  paths <- list(); cur <- NULL
  for (ln in lines) {
    if (ln == "S") {
      paths[[length(paths) + 1L]] <- cur
      cur <- NULL
      next
    }
    p <- strsplit(ln, " ", fixed = TRUE)[[1]]
    cur <- if (p[3] == "m") as.numeric(p[1:2]) else c(cur, as.numeric(p[1:2]))
  }
  n1 <- nrow(tree$merge)
  stopifnot(length(paths) == n1, all(lengths(paths) == 8L))
  segs <- do.call(rbind, paths)
  sx <- gx[2] - gx[1]; sy <- gy[2] - gy[1]
  segs[, c(1, 3, 5, 7)] <- (segs[, c(1, 3, 5, 7)] - gx[1]) / sx
  segs[, c(2, 4, 6, 8)] <- (segs[, c(2, 4, 6, 8)] - gy[1]) / sy
  # Fail closed: the bars sit at the heights, and the leaves span 1 to n
  tol_x <- 0.006 / sx; tol_y <- 0.006 / sy
  stopifnot(max(abs(sort(segs[, 4]) - sort(tree$height))) < tol_y,
            all(abs(segs[, 4] - segs[, 6]) < tol_y),
            abs(min(segs[, c(1, 7)]) - 1) < tol_x, abs(max(segs[, c(1, 7)]) - (n1 + 1)) < tol_x)
  list(error = "", nonfinite = FALSE, segs = segs, scale_x = sx, scale_y = sy)
}

add_tree <- function(id, dir) {
  tree <- tree_of(dir)
  n <- nrow(tree$merge) + 1L
  arrays <- list(merge = tree$merge, height = tree$height, order = tree$order,
                 cut_k = sapply(seq_len(n), function(k) cutree(tree, k = k)))
  hs <- sort(unique(tree$height))
  hs <- c(hs[1] - 1, hs, (hs[-1] + hs[-length(hs)]) / 2, hs[length(hs)] + 1, Inf, -Inf)
  cut_h <- tryCatch(sapply(hs, function(h) cutree(tree, h = h)), error = conditionMessage)
  if (is.character(cut_h)) cut_h_error <- cut_h else {
    cut_h_error <- ""
    arrays <- c(arrays, list(cut_h = hs, cut_h_labels = matrix(as.integer(cut_h), n)))
  }
  d <- dend_segments(tree)
  extra <- list(n = n, plot_error = d$error, nonfinite = d$nonfinite, cut_h_error = cut_h_error)
  if (!is.null(d$segs)) {
    arrays$segs <- d$segs
    extra <- c(extra, list(scale_x = d$scale_x, scale_y = d$scale_y))
  }
  write_case("plot", paste0("tree_", id), no_args, arrays, extra = extra)
  cat(sprintf("plot tree_%-16s n=%3d nonfinite=%s plot=[%s] cut_h=[%s]\n", id, n, d$nonfinite,
              d$error, cut_h_error))
}

for (d in Sys.glob("tests/fixtures/whclust/*")) {
  if ("merge" %in% names(read_json(file.path(d, "manifest.json"))$arrays))
    add_tree(basename(d), d)
}
add_tree("dn", "tests/fixtures/dn_wcluster/whclust")
add_tree("dn_one", "tests/fixtures/dn_wcluster/one_variable")

grid <- tree_of("tests/fixtures/whclust/grid_dup")
m1 <- dend_segments(grid, hang = -1)
h05 <- dend_segments(grid, hang = 0.5)
# Each hang gives its own window, so each drawing keeps its own points per user unit
write_case("plot", "tree_hang", no_args, list(merge = grid$merge, height = grid$height,
  order = grid$order, segs_m1 = m1$segs, segs_05 = h05$segs),
  extra = list(scale_x_m1 = m1$scale_x, scale_y_m1 = m1$scale_y,
               scale_x_05 = h05$scale_x, scale_y_05 = h05$scale_y))

n3 <- tree_of("tests/fixtures/whclust/n3")
nan_h <- n3; nan_h$height[1] <- NaN
unsorted <- n3; unsorted$height <- rev(n3$height) + c(0, 1)
none <- structure(list(merge = matrix(integer(0), 0, 2), height = numeric(0), order = 1L),
                  class = "hclust")
one <- structure(list(merge = matrix(c(-1L, -2L), 1), height = 1, order = 1:2), class = "hclust")
pdf(NULL)
errors <- suppressWarnings(list(
  no_k_or_h = stop_message(cutree(n3)), k_0 = stop_message(cutree(n3, k = 0)),
  k_4 = stop_message(cutree(n3, k = 4)), k_nan = stop_message(cutree(n3, k = NaN)),
  k_inf = stop_message(cutree(n3, k = Inf)), k_big = stop_message(cutree(n3, k = 3e9)),
  h_nan = stop_message(cutree(n3, h = NaN)), h_nan_height = stop_message(cutree(nan_h, h = 1)),
  h_unsorted = stop_message(cutree(unsorted, h = 1)), none_cut = stop_message(cutree(none, k = 1)),
  none_plot = stop_message(plot(none)), one_plot = stop_message(plot(one)),
  hang_nan = stop_message(plot(n3, hang = NaN)), hang_inf = stop_message(plot(n3, hang = Inf))))
dev.off()
stopifnot(all(nzchar(unlist(errors))))
write_case("plot", "stops", no_args, list(merge = n3$merge, height = n3$height, order = n3$order,
  k_2_9 = cutree(n3, k = 2.9)), extra = list(errors = errors))

# The DN.Whclust and DN.Wkmeans help examples' nugget colors, in their exact calls
recipe <- function(W) {
  w_trans = W / sum(W)
  w_trans = w_trans / quantile(w_trans, 0.8)
  col = sapply(w_trans, function(t){rgb(0, min(t, 1), 0)})
  list(t = unname(w_trans), hex = unname(col))
}
w <- read_array("tests/fixtures/dn_wcluster/dn", "weights")
ties <- c(1, 1, 1, 2, 2, 5, 5, 5, 9, 9)
a <- recipe(w); b <- recipe(ties)
write_case("plot", "colors", no_args, list(w = w, t = a$t, w_ties = ties, t_ties = b$t),
  extra = list(hex = I(a$hex), hex_ties = I(b$hex), errors = list(
    all_zero = stop_message(recipe(c(0, 0, 0))),
    quantile_zero = stop_message(recipe(c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1))),
    rgb_nan = stop_message(rgb(0, NaN, 0)), rgb_neg = stop_message(rgb(0, -0.5, 0)))))
