"""His lymphmacro_finder cut at its steps, each step his own lines, so each can be timed and
recorded on its own. finder() chains the steps, and record.py and check_reference.py check that it
gives his lymphmacro_finder's masks. graph() is his plotting block without plt.show()."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cabrera_reference import f_plot, find_area, search_ind
from scipy.ndimage import generic_filter
from scipy.spatial import ConvexHull, Delaunay


def transform(fsc_a, ssc_a):
    min_val = min(np.min(ssc_a), np.min(fsc_a))
    x0 = np.sqrt((-min_val) + 3 + ssc_a)
    y0 = np.sqrt((-min_val) + 3 + fsc_a)
    return x0, y0


def grid(x0, y0):
    msize = 100
    return f_plot(x0, y0, nr=msize, nc=msize, scale="l")


def maxima(f_density, n_row=11, n_col=11):
    x_matrix = f_density["z"].T
    threshold_val = np.quantile(f_density["z"], 0.85)

    def focal_func(window):
        m = np.max(window)
        return m if m > threshold_val else -1

    local_max = generic_filter(x_matrix, focal_func, size=(n_row, n_col))
    max_coords = np.argwhere(x_matrix == local_max)
    return np.column_stack([max_coords[:, 0] + 1, max_coords[:, 1] + 1])


def peaks(x_matrix, maxXY_rot):
    idx_f = []
    for pt in maxXY_rot:
        idx_f.append(search_ind(x_matrix, pt[0], pt[1]))
    idx_f = np.unique(np.array(idx_f), axis=0)
    mask = (idx_f[:, 0] < 90) & (idx_f[:, 1] < 90) & (idx_f[:, 0] > 10)
    idx_f = idx_f[mask]
    q90 = np.quantile(x_matrix, 0.9)
    idx_f = idx_f[np.array([x_matrix[int(i) - 1, int(j) - 1] for i, j in idx_f]) > q90]
    return idx_f[np.argsort(idx_f[:, 0])]


def areas(x_matrix, idx_f, t11=22, t21=0.25, t12=20, t22=0.5):
    idx_n = len(idx_f)
    tt = np.zeros((2, idx_n))
    tt[0, 0], tt[1, 0] = t11, t21
    tt[0, 1:], tt[1, 1:] = t12, t22
    dat_list = []
    for i in range(idx_n):
        area = find_area(x_matrix, idx_f[i, 0], idx_f[i, 1], t1=tt[0, i], t2=tt[1, i])
        area_with_id = np.column_stack([area, np.full(len(area), i + 1)])
        dat_list.append(area_with_id)
    return pd.DataFrame(np.vstack(dat_list), columns=["x", "z", "y"])


def gate(fsc_a, ssc_a, f_density, dat, idx_n):
    raw_data = np.column_stack([fsc_a, ssc_a])
    zz_grid = f_density["zz"]
    final_indices = np.zeros((len(raw_data), idx_n), dtype=bool)
    for i in range(1, idx_n + 1):
        v_cluster = dat[dat["y"] == i][["x", "z"]].values
        if len(v_cluster) > 3:
            hull = ConvexHull(v_cluster)
            delaunay = Delaunay(v_cluster[hull.vertices])
            jj_type = delaunay.find_simplex(zz_grid) >= 0
            points_to_gate = raw_data[jj_type]
            if len(points_to_gate) > 3:
                data_hull = Delaunay(points_to_gate)
                final_indices[:, i - 1] = data_hull.find_simplex(raw_data) >= 0
    return final_indices


def finder(fsc_a, ssc_a, n_row=11, n_col=11, t11=22, t21=0.25, t12=20, t22=0.5):
    x0, y0 = transform(fsc_a, ssc_a)
    f_density = grid(x0, y0)
    x_matrix = f_density["z"].T
    idx_f = peaks(x_matrix, maxima(f_density, n_row, n_col))
    dat = areas(x_matrix, idx_f, t11, t21, t12, t22)
    return gate(fsc_a, ssc_a, f_density, dat, len(idx_f))


def graph(a, ind, title):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(a[:, 0], a[:, 1], s=0.5, c="gray", alpha=0.3, label="Unlabeled")
    num_pops = ind.shape[1]
    for j in range(num_pops):
        gated_mask = ind[:, j]
        if np.any(gated_mask):
            plt.scatter(a[gated_mask, 0], a[gated_mask, 1], s=5, label=f"Pop {j + 1}", alpha=0.8)
    plt.title(title)
    plt.xlabel("FSC-A")
    plt.ylabel("SSC-A")
    plt.legend(markerscale=4)
    plt.tight_layout()
    return fig
