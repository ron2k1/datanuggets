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


# refine.DN's defaults, and what it passes to stats::kmeans
EV_TOL = 0.9
MAX_SPLITS = 5
MIN_NUGGET_SIZE = 2
MIN_SHAPE_SIZE = 10
DELTA = 2
NSTART = 25
KMEANS_ITER_MAX = 1000
MAD_CONSTANT = 1.4826  # stats::mad's default, not scipy's 1.482602...
