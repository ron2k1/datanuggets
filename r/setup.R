# Installs the pinned R oracle packages into a private library (dev only).
lib <- Sys.getenv("DATANUGGETS_R_LIB", unset = file.path(
  Sys.getenv(if (.Platform$OS.type == "windows") "LOCALAPPDATA" else "HOME"), "datanuggets-rlib"))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
install.packages(c("jsonlite", "datanugget"), lib = lib, repos = "https://cloud.r-project.org")
stopifnot(packageVersion("datanugget", lib.loc = lib) == "1.5.0")
cat("datanugget", as.character(packageVersion("datanugget", lib.loc = lib)),
    "| Rfast", as.character(packageVersion("Rfast", lib.loc = lib)), "\n")
