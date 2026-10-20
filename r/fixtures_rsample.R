# Fixtures for R's own draws: set.seed() and sample.int(), with .Random.seed before and after each
# call, so RDraws is checked on the values, the state they leave, and the draws that follow. Every
# case calls sample.int() directly, so none needs a large dataset.
source("r/fixture_io.R")

stopifnot(identical(RNGkind(), c("Mersenne-Twister", "Inversion", "Rejection")))
rng <- list(kind = RNGkind(), use_hash = deparse1(formals(sample.int)$useHash),
            sample_int = deparse(body(sample.int)))
# sample.int switches to do_sample2, which draws from all n and redraws repeats, when this holds
stopifnot(rng$use_hash == "(n > 1e+07 && !replace && is.null(prob) && size <= n/2)")
# Evaluates base R's own default expression on the four values given, so the table below is R's
# rule and not a copy of it
use_hash <- function(n, size, replace = FALSE, prob = NULL)
  eval(formals(sample.int)$useHash, list(n = n, size = size, replace = replace, prob = prob))
seed_now <- function() get(".Random.seed", envir = globalenv())

# set.seed() and the draws after it. asInteger() truncates toward zero, so 2.7 seeds as 2.
seeds <- list(s291102 = 291102, s55 = 55, neg7 = -7, s0 = 0, s2 = 2, s2p7 = 2.7, neg2 = -2,
              neg2p7 = -2.7, intmax = 2147483647, neg_intmax = -2147483647,
              intmax_half = 2147483647.5, neg_intmax_frac = -2147483647.9)
states <- list()
for (nm in names(seeds)) {
  set.seed(seeds[[nm]])
  s0 <- seed_now()
  first <- sample.int(10, 10)
  states[[nm]] <- s0
  stopifnot(s0[1] == 10403L, s0[2] == 624L, length(s0) == 626L)
  write_case("rsample", paste0("seed_", nm), list(seed = seeds[[nm]]),
             list(state0 = s0, first = first, state1 = seed_now()), extra = list(rng = rng))
}
stopifnot(identical(states$s2p7, states$s2), identical(states$neg2p7, states$neg2),
          identical(states$intmax_half, states$intmax),
          identical(states$neg_intmax_frac, states$neg_intmax))

# Seeds asInteger() turns into NA. Out of range warns first; NaN and NA don't.
bad <- c(NaN, NA_real_, Inf, -Inf, 2^31, -2^31)
res <- lapply(bad, function(s) {
  w <- character()
  e <- tryCatch(withCallingHandlers({ set.seed(s); "ran" },
    warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") }),
    error = function(e) conditionMessage(e))
  list(error = e, warnings = w)
})
stopifnot(all(vapply(res, function(r) r$error == "supplied seed is not a valid integer", TRUE)))
write_case("rsample", "seed_invalid", no_args, list(seeds = bad),
           extra = list(rng = rng, errors = vapply(res, function(r) r$error, ""),
                        warnings = lapply(res, function(r) I(r$warnings))))

# The switch. Each case records sample.int(n, size), the same call again, then sample.int(10, 10),
# with the state after each, and what the other branch would have drawn from the same seed.
switch_case <- function(id, n, size, seed, differ) {
  hash <- use_hash(n, size)
  set.seed(seed)
  s0 <- seed_now()
  first <- sample.int(n, size)
  s1 <- seed_now()
  second <- sample.int(n, size)
  s2 <- seed_now()
  third <- sample.int(10, 10)
  s3 <- seed_now()
  set.seed(seed)
  other <- sample.int(n, size, useHash = !hash)
  stopifnot(identical(!identical(first, other), differ))
  write_case("rsample", id, list(n = n, size = size, seed = seed),
             list(state0 = s0, first = first, state1 = s1, second = second, state2 = s2,
                  third = third, state3 = s3, other = other),
             extra = list(rng = rng, use_hash = hash, branches_differ = differ))
  cat(sprintf("%-32s n=%.0f size=%d hash=%s branches differ=%s\n", id, n, size, hash, differ))
}
# n - 1 = 2^24 needs a bit fewer than n, so do_sample's second draw already differs from do_sample2's
switch_case("switch_2p24p1_size2", 2^24 + 1, 2L, 291102, TRUE)
switch_case("switch_2p24p1_size1", 2^24 + 1, 1L, 291102, FALSE)
# At 1e7 and 1e7 + 1 both branches take 24 bits, and seed 9 first splits them at draw 1130
switch_case("switch_1e7_size1200", 1e7, 1200L, 9, TRUE)
switch_case("switch_1e7p1_size1200", 1e7 + 1, 1200L, 9, TRUE)

# The size condition at an even n, where size = n/2 hashes and n/2 + 1 doesn't. A full sample is
# 5 million draws, so the fixture keeps each one's first values up to just past where they part.
n <- 1e7 + 2
half <- n / 2
stopifnot(use_hash(n, half), !use_hash(n, half + 1))
for (seed in 1:50) {
  set.seed(seed)
  a <- sample.int(n, half)
  set.seed(seed)
  b <- sample.int(n, half + 1)
  d <- which(a[1:20000] != b[1:20000])
  if (length(d)) break
}
stopifnot(length(d) > 0)
keep <- d[1] + 20L
for (sz in c(half, half + 1)) {
  set.seed(seed)
  v <- sample.int(n, sz)
  write_case("rsample", sprintf("size_%.0f_of_%.0f", sz, n), list(n = n, size = sz, seed = seed),
             list(head = head(v, keep)),
             extra = list(rng = rng, use_hash = use_hash(n, sz), first_split = d[1]))
}
cat(sprintf("size boundary at n=%.0f: seed %d, the branches first part at draw %d\n", n, seed, d[1]))

# R_unif_index takes its bit count from ceil(log2(dn)), which is 49 at n = 2^49 + 1 though 2^49
# needs 50 bits. The hash branch draws from all n without a pool, so a call this large is cheap.
n <- 2^49 + 1
stopifnot(ceiling(log2(n)) == 49, use_hash(n, 3))
set.seed(291102)
s0 <- seed_now()
first <- sample.int(n, 3)
write_case("rsample", "bits_2p49p1", list(n = n, size = 3L, seed = 291102),
           list(state0 = s0, first = first, state1 = seed_now()),
           extra = list(rng = rng, use_hash = use_hash(n, 3)))
cat(sprintf("bits at n=2^49+1: %s\n", paste(sprintf("%.0f", first), collapse = " ")))

# R's useHash across the n and size boundaries, and with replace or prob, which always turn it off
ns <- c(1e7 - 1, 1e7, 1e7 + 1, 1e7 + 2, 2^24 + 1, 2^31 - 1, 2^31 + 1)
tab <- do.call(rbind, lapply(ns, function(n) {
  h <- n / 2
  data.frame(n = n, size = unique(c(0, 1, 2, floor(h) - 1, floor(h), ceiling(h), ceiling(h) + 1, n)))
}))
tab$use_hash <- mapply(use_hash, tab$n, tab$size)
tab$with_replace <- mapply(function(n, s) use_hash(n, s, replace = TRUE), tab$n, tab$size)
tab$with_prob <- mapply(function(n, s) use_hash(n, s, prob = c(1, 1)), tab$n, tab$size)
stopifnot(any(tab$use_hash), !all(tab$use_hash), !any(tab$with_replace), !any(tab$with_prob))
write_case("rsample", "use_hash_table", no_args,
           list(n = tab$n, size = tab$size, use_hash = tab$use_hash,
                with_replace = tab$with_replace, with_prob = tab$with_prob),
           extra = list(rng = rng))
cat(sprintf("use_hash table: %d rows, %d hash\n", nrow(tab), sum(tab$use_hash)))

# sample.int(n, size, replace = TRUE): do_sample's replace branch, one R_unif_index(n) per value,
# then sample.int(10, 10) to show the state it leaves. Wkmeans' init draws
# sample(k, n - k, replace = TRUE).
replace_case <- function(id, n, size, seed) {
  set.seed(seed)
  s0 <- seed_now()
  first <- sample.int(n, size, replace = TRUE)
  s1 <- seed_now()
  then <- sample.int(10, 10)
  write_case("rsample", paste0("replace_", id), list(n = n, size = size, seed = seed),
             list(state0 = s0, first = first, state1 = s1, then = then, state2 = seed_now()),
             extra = list(rng = rng))
  cat(sprintf("replace %-14s %s
", id, paste(head(first, 8), collapse = " ")))
}
replace_case("k3_size147", 3, 147, 291102)            # iris with k = 3
replace_case("k4_size6", 4, 6, 55)
replace_case("k1_size5", 1, 5, 2)                     # rbits(0) still reads one uniform per value
replace_case("k7_size1", 7, 1, 291102)
replace_case("intmax_size3", 2147483647, 3, 291102)   # 31 bits, two uniforms per value
replace_case("k5_size0", 5, 0, 291102)
