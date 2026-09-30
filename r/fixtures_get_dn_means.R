# Fixtures for getDN.means: R's nugget means for every nugget set the create_dn and refine_dn
# fixtures hold (create.DN outputs, refine.DN inputs and refine.DN outputs), read back from their
# .bin files. Run after fixtures_create_dn.R and fixtures_refine_dn.R.
source("r/fixture_io.R")

# A "datanugget" object laid out as create.DN returns it. getDN.means reads only the table's
# shape, the center column names and the assignments.
as_dn <- function(centers, weights, scales, assignments) {
  tab <- data.frame(seq_len(nrow(centers)), centers, weights, scales)
  names(tab) <- c("Data Nugget", paste0("Center", seq_len(ncol(centers))), "Weight", "Scale")
  structure(list(`Data Nuggets` = tab, `Data Nugget Assignments` = assignments),
            class = "datanugget")
}

targets <- list(create_dn = c("create_dn", ""), refine_dn_in = c("refine_dn", "in_"),
                refine_dn_out = c("refine_dn", ""))
for (target in names(targets)) {
  group <- targets[[target]][1]
  prefix <- targets[[target]][2]
  dirs <- list.dirs(file.path("tests", "fixtures", group), recursive = FALSE)
  means <- lapply(dirs, function(dir) {
    part <- function(name) read_array(dir, paste0(prefix, name))
    dn <- as_dn(part("centers"), part("weights"), part("scales"), part("assignments"))
    unname(as.matrix(getDN.means(read_array(dir, "x"), dn)))
  })
  names(means) <- basename(dirs)
  write_case("get_dn_means", target, list(source = group, prefix = prefix), means)
  cat(target, length(means), "nugget sets\n")
}

# Row order inside a nugget changes a mean only once the long-double running sum rounds, and none
# of the real sets above rounds. One constructed set pins R's order (which() is ascending):
# 1e20 - 1e20 + 1 is 1, but in reverse 1 - 1e20 + 1e20 is 0.
x <- cbind(c(1e20, 5, -1e20, 7, 1, 3), 1:6)
assignments <- rep(1:2, 3)
dn <- as_dn(matrix(0, 2, 2), c(3, 3), c(0, 0), assignments)
write_case("get_dn_means", "row_order", no_args,
           list(x = x, assignments = assignments, means = unname(as.matrix(getDN.means(x, dn)))))
cat("row_order means", format(getDN.means(x, dn)[1, 1], digits = 17), "\n")
