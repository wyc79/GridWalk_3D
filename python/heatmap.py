
import os

import pandas as pd
import seaborn as sns
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

EXCLUSION = [
    'AD261_2023-09-07_18-33-01'
]

x1 = 0
x2 = 510
y1 = 120
y2 = 480

x1_grid = 87-x1
x2_grid = 447-x1
y1_grid = 170-y1
y2_grid = 410-y1

result_path = '../GridWalking-wyc-2023-10-30/new_analyzed/all_coords.h5'

# reading and excluding outliers
df = pd.read_hdf(result_path)
df = df[~(df['video_name'].isin(EXCLUSION))]

for cond in ['AD', 'WT']:
    for month in pd.unique(df['month']):

        sub_df = df[df['type']==cond]
        sub_df = sub_df[sub_df['month']==month]

        x_coords = np.float32(sub_df[('mid_spine', 'x')])
        y_coords = np.float32(sub_df[('mid_spine', 'y')])

        x_mask = np.array(
            [1 if ((x>x1_grid) and (x<x2_grid)) else 0 for x in x_coords], dtype=int)
        y_mask = np.array(
            [1 if ((y>y1_grid) and (y<y2_grid)) else 0 for y in y_coords], dtype=int)

        mask = [True if m else False for m in x_mask * y_mask]
        x_coords_filtered = x_coords[mask]
        y_coords_filtered = y_coords[mask]
        x_bins = int((x2_grid - x1_grid)/10)+1
        y_bins = int((y2_grid - y1_grid)/10)+1

        counts, x_edges, y_edges = np.histogram2d(x_coords_filtered, y_coords_filtered, bins=(x_bins, y_bins))
        normalized_counts = counts / counts.sum()

        fig, ax = plt.subplots(1,1, figsize=[x_bins/2, y_bins/2], constrained_layout=True)
        sns.heatmap(normalized_counts, cmap="Blues", ax=ax)
        # hm = sns.jointplot(
        #     x=x_coords_filtered, y=y_coords_filtered, kind="hist",
        #     stat_func=lambda x, y: len(x) / len(x_coords_filtered))
        # fig = hm.get_figure()
        ax.set_title(f"{cond}_{month}")
        ax.tick_params(left=False, bottom=False)
        ax.collections[0].set_clim(0,0.01)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        fig.savefig(f"out/heatmap_{cond}_{month}.tiff")