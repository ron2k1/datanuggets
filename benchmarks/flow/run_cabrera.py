"""Cabrera's whole script as he sent it on 2026-10-12, with three changes marked "changed:": the
workbook and folder come from the command line, and matplotlib draws on Agg so it runs unattended.
Nothing else is changed but trailing spaces. His loop reads the first four samples in FlowKit's
order, which sorts by the $FIL keyword, and pairs them with t0 and t00. FlowKit needs pandas<3, so
this runs outside the project's env:

uv run --no-project --with flowkit --with openpyxl --with matplotlib python benchmarks/flow/run_cabrera.py <metadata.xlsx> <folder>
"""

import sys  # changed: paths from the command line

import matplotlib  # changed: Agg, so the script runs unattended

matplotlib.use("Agg")

import pandas as pd
import flowkit as fk
import os
import numpy as np
from scipy.spatial import distance_matrix
import math
from scipy.ndimage import generic_filter
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt

# 1. Read the Excel metadata (equivalent to read_excel)
# /usr/local/bin/python3 /Users/cabrera/Documents/CellFC/FC/cfc.py
# sheet_name index is 0-based in Python, so sheet=2 in R is index 1
metadata_path = sys.argv[1]  # changed: was his 20250618 study workbook
dir_name = sys.argv[2]  # changed: was his 20250618 study folder
dt_meta = pd.read_excel(os.path.expanduser(metadata_path), sheet_name=1)
#/Users/cabrera/Documents/CellFC/20250618_Study files
# 2. Prepare the file paths
# Construct full paths by joining the directory name with the filename column
file_paths = [os.path.join(dir_name, f) for f in dt_meta['Filename']]

# 3. Read the flow files (equivalent to read.flowSet)
# FlowKit uses a 'Session' to manage a collection of samples
dt1 = fk.Session(fcs_samples=file_paths)
# Optional: To mimic truncate_max_range=FALSE, FlowKit by default
# reads the raw events. You can verify the sample data like this:
# sample = dt1.get_sample('sample_id')

dt2 = [sample for sample_id, sample in dt1.sample_lut.items()]


dt2_compensated = []

for sample in dt1.sample_lut.values():
    # Retrieve the internal spillover matrix
    spill = sample.metadata.get('spill')

    if spill is not None:
        # apply_compensation returns the compensated numpy array
        data = sample.apply_compensation(spill)
        dt2_compensated.append(data)
    else:
        # FIX: Explicitly request 'raw' source to avoid the AttributeError
        data = sample.get_events(source='raw')
        dt2_compensated.append(data)

print(f"Successfully processed {len(dt2_compensated)} samples.")


def f_plot(x, y, nr=20, nc=20, scale="raw", alpha=0.1):
    # Ensure inputs are numpy arrays for vector math
    x = np.array(x)
    y = np.array(y)

    # 1. Calculate the bin indices for x and y
    # This mimics the trunc(nr * (x - min) / (max - min)) + 1 logic
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Map data to 1-based indices (to match R logic)
    # clip ensures values don't exceed nr/nc
    zx_data = np.floor(nr * (x - x_min) / (x_max - x_min)).astype(int) + 1
    zx_data = np.clip(zx_data, 1, nr)

    zy_data = np.floor(nc * (y - y_min) / (y_max - y_min)).astype(int) + 1
    zy_data = np.clip(zy_data, 1, nc)

    # 2. Replicate the padding (zx = 1:nr followed by ones, etc.)
    # This is used in the R code to ensure the table covers all bins
    zx_pad = np.concatenate([np.arange(1, nr + 1), np.ones(nc, dtype=int)])
    zy_pad = np.concatenate([np.ones(nr, dtype=int), np.arange(1, nc + 1)])

    zx = np.concatenate([zx_pad, zx_data])
    zy = np.concatenate([zy_pad, zy_data])

    # 3. Create the 2D Frequency Table (Contingency table)
    # R's table(zx, zy) is equivalent to np.histogram2d
    z, x_edges, y_edges = np.histogram2d(
        zx, zy,
        bins=[nr, nc],
        range=[[0.5, nr + 0.5], [0.5, nc + 0.5]]
    )

    # 4. Adjust for the padding (subtracting 1 from first rows/cols)
    z[:, 0] = z[:, 0] - 1
    z[0, :] = z[0, :] - 1

    # 5. Apply Scaling
    if scale == "l":
        z = np.log1p(z) # log(1 + z)
    elif scale == "d":
        z = np.log1p(np.log1p(z)) # log(1 + log(1 + z))

    # 6. Apply Alpha Threshold
    z[z < (z.max() * alpha)] = 0

    # 7. Return results in a dictionary (equivalent to R's list)
    return {
        "z": z.T, # Transpose to match R's list(z = t(z))
        "x": np.linspace(x_min, x_max, nr),
        "y": np.linspace(y_min, y_max, nc),
        "zz": np.column_stack((zx_data, zy_data)),
        "zx": zx,
        "zy": zy
    }


def search_ind(x, idx, idy, threshold=2):
    """
    Finds the coordinates of the maximum value in a window around (idx, idy).
    """
    # 1. Define the bounds of the neighbor window
    # Note: R is 1-indexed, Python is 0-indexed.
    # We use math.ceil to match R's ceil() logic.
    start_x = max(int(math.ceil(idx)) - threshold, 1)
    end_x   = min(int(math.ceil(idx)) + threshold, x.shape[0])

    start_y = max(int(math.ceil(idy)) - threshold, 1)
    end_y   = min(int(math.ceil(idy)) + threshold, x.shape[1])

    # 2. Extract the sub-matrix (neighbor_x)
    # Python slicing [start:end] is exclusive of the end index.
    # We subtract 1 from starts because Python is 0-indexed.
    neighbor_x = x[start_x-1 : end_x, start_y-1 : end_y]

    # 3. Find index of maximum value (equivalent to which(arr.ind = TRUE))
    # np.argmax returns flattened index, unravel_index converts to 2D
    ind = np.unravel_index(np.argmax(neighbor_x, axis=None), neighbor_x.shape)

    # 4. Calculate the global coordinates
    # ind[0] is row (x), ind[1] is column (y)
    idx_f = start_x + ind[0]
    idy_f = start_y + ind[1]

    return np.array([idx_f, idy_f])

def find_area(x, idx, idy, t1=35, t2=0.85):
    """
    Finds grid coordinates within distance t1 and density > t2 * x[idx, idy].
    """
    rows, cols = x.shape

    # 1. Create the grid coordinates (1-based to match R logic)
    # R's cbind(rep(1:nrow, ncol), rep(1:ncol, each=nrow))
    ii, jj = np.meshgrid(np.arange(1, rows + 1), np.arange(1, cols + 1), indexing='ij')
    gr = np.column_stack([ii.ravel(), jj.ravel()])

    # 2. Calculate distance from the landmark point (idx, idy)
    # In R: id <- idx + (idy - 1) * nrow(x)
    # In Python: we can just use the coordinate directly for the distance calculation
    landmark_coord = np.array([[idx, idy]])

    # Calculate Euclidean distance from landmark to all points in gr
    # Using linalg.norm is much faster than computing a full N x N matrix
    d = np.linalg.norm(gr - landmark_coord, axis=1)

    # 3. Get the density at the landmark point
    # Note: R is 1-indexed, so subtract 1 for Python indexing
    val_landmark = x[int(idx) - 1, int(idy) - 1]

    # 4. Create the mask (Logical indexing)
    # d < t1 AND the value in x at those grid points > t2 * val_landmark
    # Flatten x to align with the flattened grid 'gr'
    x_flat = x.flatten(order='F') # 'F' order (column-major) to match R's vectorization

    mask = (d < t1) & (x_flat > t2 * val_landmark)

    # 5. Return the coordinates (swapped 2:1 to [col, row] to match R's gr[..., 2:1])
    result = gr[mask]
    return result[:, [1, 0]] # Swap columns to match R's output



def lymphmacro_finder(fsc_a, ssc_a, n_row=11, n_col=11, t11=22, t21=0.25,
                       t12=20, t22=0.5, plot=False):
    # 1. Coordinate Transformation
    # Mimics R's sqrt logic to stabilize variance/scale
    min_val = min(np.min(ssc_a), np.min(fsc_a))
    x0 = np.sqrt((-min_val) + 3 + ssc_a)
    y0 = np.sqrt((-min_val) + 3 + fsc_a)

    msize = 100
    # Use the f_plot function translated previously
    f_density = f_plot(x0, y0, nr=msize, nc=msize, scale="l")
    x_matrix = f_density['z'].T  # Aligning matrix orientation

    # 2. Local Maxima Detection (Focal Analysis)
    # Replaces R's raster::focal
    threshold_val = np.quantile(f_density['z'], 0.85)

    def focal_func(window):
        m = np.max(window)
        return m if m > threshold_val else -1

    # Apply moving window (11x11)
    local_max = generic_filter(x_matrix, focal_func, size=(n_row, n_col))

    # Find coordinates where the matrix equals the local max
    max_coords = np.argwhere(x_matrix == local_max)
    # Convert to 1-based indices to match search_ind logic
    maxXY_rot = np.column_stack([max_coords[:, 0] + 1, max_coords[:, 1] + 1])

    # 3. Refine Peaks with search_ind
    idx_f = []
    for pt in maxXY_rot:
        idx_f.append(search_ind(x_matrix, pt[0], pt[1]))
    idx_f = np.unique(np.array(idx_f), axis=0)

    # Filtering indices based on spatial constraints
    mask = (idx_f[:, 0] < 90) & (idx_f[:, 1] < 90) & (idx_f[:, 0] > 10)
    idx_f = idx_f[mask]

    # Sort and refine based on density quantile
    q90 = np.quantile(x_matrix, 0.9)
    idx_f = idx_f[np.array([x_matrix[int(i)-1, int(j)-1] for i, j in idx_f]) > q90]
    idx_f = idx_f[np.argsort(idx_f[:, 0])]
    idx_n = len(idx_f)

    # 4. Define Areas (find_area)
    tt = np.zeros((2, idx_n))
    tt[0, 0], tt[1, 0] = t11, t21 # First peak settings
    tt[0, 1:], tt[1, 1:] = t12, t22 # Subsequent peaks settings

    dat_list = []
    for i in range(idx_n):
        area = find_area(x_matrix, idx_f[i,0], idx_f[i,1], t1=tt[0,i], t2=tt[1,i])
        # Add a cluster ID column
        area_with_id = np.column_stack([area, np.full(len(area), i + 1)])
        dat_list.append(area_with_id)

    dat = pd.DataFrame(np.vstack(dat_list), columns=['x', 'z', 'y'])

    # 5. Gating Logic (Convex Hull and InHull)
    raw_data = np.column_stack([fsc_a, ssc_a])
    zz_grid = f_density['zz']
    final_indices = np.zeros((len(raw_data), idx_n), dtype=bool)

    for i in range(1, idx_n + 1):
        # Filter points for this cluster
        v_cluster = dat[dat['y'] == i][['x', 'z']].values

        # Simple InHull logic using Scipy
        if len(v_cluster) > 3:
            hull = ConvexHull(v_cluster)
            # Find points in the original grid (zz) inside this hull
            # We use Delaunay to check if points are inside
            delaunay = Delaunay(v_cluster[hull.vertices])
            jj_type = delaunay.find_simplex(zz_grid) >= 0

            # Now gate the original raw data points based on these grid points
            points_to_gate = raw_data[jj_type]
            if len(points_to_gate) > 3:
                data_hull = Delaunay(points_to_gate)
                final_indices[:, i-1] = data_hull.find_simplex(raw_data) >= 0

    return final_indices



import matplotlib.pyplot as plt
import numpy as np

# 1. Extract raw numerical matrices (equivalent to lapply @exprs)
# Based on our previous steps, dt2 is your list of FlowKit Sample objects
dt_exp = [sample.get_events(source='raw') for sample in dt2]

# 2. Setup parameters
res = []
t0 = [22, 20, 18, 18]
t00 = [0.25, 0.5, 0.5, 0.5]

# 3. Main Loop
for i in range(4):
    # Get the specific sample matrix
    # We use the Sample object to find the correct column indices by name
    sample_obj = dt2[i]
    fsc_idx = sample_obj.get_channel_index("FSC-A")
    ssc_idx = sample_obj.get_channel_index("SSC-A")

    # Slice the matrix (a <- dt_exp[[i]][,c(...)])
    a = dt_exp[i][:, [fsc_idx, ssc_idx]]

    # Run the gating finder
    # ind is a boolean matrix (Events x Populations)
    ind = lymphmacro_finder(a[:, 0], a[:, 1], t11=t0[i], t21=t00[i], plot=False)
    res.append(ind)

    # 4. Plotting
    plt.figure(figsize=(6, 6))

    # Draw background dots (equivalent to plot(a, pch="."))
    plt.scatter(a[:, 0], a[:, 1], s=0.5, c='gray', alpha=0.3, label='Unlabeled')

    # Draw gated populations (equivalent to points(a[ind[,j],]))
    # We iterate through columns of the boolean matrix 'ind'
    num_pops = ind.shape[1]
    for j in range(num_pops):
        gated_mask = ind[:, j]
        if np.any(gated_mask):
            plt.scatter(
                a[gated_mask, 0],
                a[gated_mask, 1],
                s=5,
                label=f'Pop {j+1}',
                alpha=0.8
            )

    plt.title(f"Sample {i+1}: FSC-A vs SSC-A")
    plt.xlabel("FSC-A")
    plt.ylabel("SSC-A")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.show()
