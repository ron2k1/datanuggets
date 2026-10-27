"""R's default argument values (datanugget 1.5.0), the flow gate finder's, and internal sizes,
kept in one place."""

SEED = 291102
R_CHUNK = 5000
R_MIN = 100
R_MAX = 10_000
DELETE_PERCENT = 0.1
DN_NUM1 = 10_000
DN_NUM2 = 2000
CENTER_METHODS = ("mean", "random", "original")
DIST_METRICS = ("euclidean", "manhattan")

MOMENTS_CHUNK = 1 << 22  # covariance entries group_moments holds at once (32 MB)


# refine.DN's defaults, and what it passes to stats::kmeans
EV_TOL = 0.9
MAX_SPLITS = 5
MIN_NUGGET_SIZE = 2
MIN_SHAPE_SIZE = 10
DELTA = 2
NSTART = 25
KMEANS_ITER_MAX = 1000
MAD_CONSTANT = 1.4826  # stats::mad's default, not scipy's 1.482602...


# the flow gate finder's tunable values, as cfc.py calls it: the n_row by n_col window its local
# maxima are found in, and the area each peak grows, the cells closer than t1 grid cells with more
# than t2 times the peak's density; t11 and t21 for the first peak, t12 and t22 for the others
FINDER_N_ROW = 11
FINDER_N_COL = 11
FINDER_T11 = 22
FINDER_T21 = 0.25
FINDER_T12 = 20
FINDER_T22 = 0.5
