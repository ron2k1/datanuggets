# Fixtures for AC: its value and the loop state it returns with (G, R, A, DN.num1, B), over the
# row and column counts of real R datasets crossed with R, delete.percent and DN.num1/DN.num2.
# AC reads only nrow(x) and ncol(x). Some arguments send AC's loops into a cycle, so every call
# runs under a one-second limit, and a call that hits it is recorded as never returning.
source("r/fixture_io.R")

# R 4.6.1's AC accepts only a plain data.frame: class(x) %in% ... has length 2 for a matrix
frame <- function(d) {
  d <- as.data.frame(d)
  stopifnot(identical(class(d), "data.frame"))
  d
}
shapes <- lapply(list(faithful = faithful, iris = iris[, 1:4], quakes = quakes,
                      volcano = volcano, EuStockMarkets = EuStockMarkets,
                      airquality = airquality, women = women, trees = trees), frame)

pairs <- function(...) {
  do.call(rbind, lapply(list(...), function(v) data.frame(dn1 = v[1], dn2 = v[2])))
}
main <- merge(expand.grid(shape = names(shapes), R = c(100, 150, 250.7, 1000, 5000, 10000),
                          dp = c(0.01, 0.1, 0.3, 0.5, 0.9, 0.99), stringsAsFactors = FALSE),
              pairs(c(10^4, 2000), c(100, 20), c(100.9, 20.2), c(60, 60), c(1000, 1), c(3, 0),
                    c(0, 0), c(0, -1), c(5000, 4999), c(7, 3), c(2, 1), c(1, 1)))
# Cycles: a DN.num2 of -2 or less traps loop 2 between 0 and -1, a target DN.num1/G below -1 or
# strictly between -1 and 0 traps loop 1 the same way, and at 2^60 a cut that floors to 0
# leaves DN.num1 - 1 == DN.num1
hangs <- merge(expand.grid(shape = c("faithful", "women"), R = c(100, 5000), dp = c(0.1, 0.5),
                           stringsAsFactors = FALSE),
               pairs(c(1, -3), c(-5, -10), c(-0.5, -1)))
big <- data.frame(shape = "faithful", R = 5000, dp = 1e-20, dn1 = 2^60, dn2 = 0)
grid <- rbind(main, hangs, big)

ac_state <- NULL
trace("AC", where = asNamespace("datanugget"), print = FALSE, exit = quote(
  if (exists("my.AC", inherits = FALSE))
    ac_state <<- c(G = G, R = R, A = A, DN.num1 = DN.num1, B = B, value = my.AC)))

call_ac <- function(x, R, dp, dn1, dn2) {
  ac_state <<- NULL
  warned <- FALSE
  setTimeLimit(elapsed = 1, transient = TRUE)
  on.exit(setTimeLimit(elapsed = Inf))
  out <- tryCatch(withCallingHandlers(datanugget::AC(x, R, dp, dn1, dn2),
    warning = function(w) {
      if (conditionMessage(w) != "NaNs produced") stop("unexpected warning: ", conditionMessage(w))
      warned <<- TRUE
      invokeRestart("muffleWarning")
    }), error = function(e) {
      if (!grepl("reached elapsed time limit", conditionMessage(e))) stop(e)
      NULL
    })
  if (is.null(out)) {
    return(c(hung = 1, warned = 0, G = NA, R = NA, A = NA, DN.num1 = NA, B = NA, value = NA))
  }
  stopifnot(identical(out, unname(ac_state[["value"]])))
  c(hung = 0, warned = as.numeric(warned), ac_state)
}

rows <- t(vapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]
  call_ac(shapes[[g$shape]], g$R, g$dp, g$dn1, g$dn2)
}, numeric(8)))
untrace("AC", where = asNamespace("datanugget"))

write_case("ac", "grid", no_args, list(
  n = vapply(grid$shape, function(s) nrow(shapes[[s]]), integer(1), USE.NAMES = FALSE),
  p = vapply(grid$shape, function(s) ncol(shapes[[s]]), integer(1), USE.NAMES = FALSE),
  r = grid$R, delete_percent = grid$dp, dn_num1 = grid$dn1, dn_num2 = grid$dn2,
  hung = as.integer(rows[, "hung"]), warned = as.integer(rows[, "warned"]),
  g = rows[, "G"], r_end = rows[, "R"], a = rows[, "A"], dn_num1_end = rows[, "DN.num1"],
  b = rows[, "B"], value = rows[, "value"]), extra = list(shapes = grid$shape))
cat(nrow(grid), "calls,", sum(rows[, "hung"]), "never returned,", sum(rows[, "warned"]),
    "warned\n")
