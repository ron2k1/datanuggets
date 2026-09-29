"""R's default argument values (datanugget 1.5.0) and internal sizes, kept in one place."""

SEED = 291102
R_CHUNK = 5000
R_MIN = 100
R_MAX = 10_000
DELETE_PERCENT = 0.1
DN_NUM1 = 10_000
DN_NUM2 = 2000
CENTER_METHODS = ("mean", "random", "original")
DIST_METRICS = ("euclidean", "manhattan")

# Rows per block when computing distances, so no n x k or n x n temporary exceeds memory.
DIST_BLOCK_ROWS = 1024
ASSIGN_BLOCK_ROWS = 1024  # 1024 x 2000 centers = 16 MB per temporary
