"""R's default argument values (datanugget 1.5.0), the flow pipeline's, and internal sizes, kept
in one place."""

from types import MappingProxyType

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


# datanuggets-flow's defaults: the scatter channels it gates on; his, cfc.py's mirrored_area, as
# the area rule, where the library's lymphmacro_finder defaults to find_area; the channels it
# clusters, every -A channel but the scatter ones; the arcsinh cofactor; nuggets per file, where
# create_dn's own default is DN_NUM2; every core for create_dn, which doesn't change its results;
# the clusters cut from the tree; and the output folder, named by the UTC time a run starts
FLOW_FSC = "FSC-A"
FLOW_SSC = "SSC-A"
FLOW_DISC = "his"
FLOW_CHANNEL_SUFFIX = "-A"
FLOW_SCATTER_PREFIXES = ("FSC", "SSC")
FLOW_COFACTOR = 6000.0
FLOW_DN_NUM2 = 1000
FLOW_N_JOBS = -1
FLOW_K = 8
FLOW_OUT_DIR = "outputs"
FLOW_RUN_NAME = "flow-%Y%m%d-%H%M%S-%fZ"
FLOW_GATES_NAME = "flow-gates-%Y%m%d-%H%M%S-%fZ"
FLOW_SPILL_TOL = 1e-3  # how far from the identity a file's spillover matrix may be before a warning
FLOW_GATE_DPI = 110  # local_maxima.png and gates.png
FLOW_RESULT_DPI = 150  # the clustering's figures

# the t-SNE of the clustered nuggets: its perplexity, lowered to (n - 1) / 3 for n nuggets when
# that's smaller, as scikit-learn needs it below n, and its seed; and the arguments that shape the
# picture, pinned so a change of scikit-learn's defaults can't move it
TSNE_PERPLEXITY = 30.0
TSNE_SEED = 0
TSNE_ARGS = MappingProxyType(
    {"n_components": 2, "init": "pca", "learning_rate": "auto", "max_iter": 1000}
)
