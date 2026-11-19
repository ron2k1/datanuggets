"""R's default argument values (datanugget 1.5.0), the helpers' settings, and internal sizes,
kept in one place."""

from types import MappingProxyType

SEED = 291102  # WCluster's Wkmeans and DN.Wkmeans default to it too
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


# the t-SNE of nugget centers: its perplexity, lowered to (n - 1) / 3 for n nuggets when
# that's smaller, as scikit-learn needs it below n, and its seed; and the arguments that shape the
# picture, pinned so a change of scikit-learn's defaults can't move it
TSNE_PERPLEXITY = 30.0
TSNE_SEED = 0
TSNE_ARGS = MappingProxyType(
    {"n_components": 2, "init": "pca", "learning_rate": "auto", "max_iter": 1000}
)


# cluster colours: up to CLUSTER_PARENT_K clusters take matplotlib's tab10 colours as they are;
# more fall in families, the cluster they fall in when the tree is cut into CLUSTER_PARENT_K. A
# family's first cluster keeps its tab10 colour and its others are picked near it from the sRGB
# colours whose channels take CLUSTER_GRID_LEVELS evenly spaced values (twice as many, up to
# 256, when that's too few) and whose WCAG contrast against each of CLUSTER_BACKGROUNDS (the
# light and the dark page) is at least CLUSTER_MIN_CONTRAST. Each pick maximises
# min(dE00 to its siblings / CLUSTER_SIBLING_DE00, dE00 to other families / CLUSTER_OTHER_DE00),
# with scores rounded to CLUSTER_SCORE_DECIMALS so a platform's last bit can't change a pick
CLUSTER_PARENT_K = 8
CLUSTER_GRID_LEVELS = 24
CLUSTER_BACKGROUNDS = ("#ffffff", "#111318")
CLUSTER_MIN_CONTRAST = 2.5
CLUSTER_SIBLING_DE00 = 5.0
CLUSTER_OTHER_DE00 = 3.0
CLUSTER_SCORE_DECIMALS = 6
