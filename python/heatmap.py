
import os

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
from functools import partial

from transform_coords import *


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

EXCLUSION = [
    'AD261_2023-09-07_18-33-01'
]

x1 = 0
x2 = 510
y1 = 120
y2 = 480

x1_grid = 90-x1
x2_grid = 440-x1
y1_grid = 160-y1
y2_grid = 400-y1

##### 360px = 30cm; 240px = 20cm (px/12=cm)

x_bins = int((x2_grid - x1_grid)/10)+1
y_bins = int((y2_grid - y1_grid)/10)+1

_, x_edges, y_edges = np.histogram2d(
    np.linspace(x1_grid, x2_grid, num=100), 
    np.linspace(y1_grid, y2_grid, num=100),
    bins=(x_bins, y_bins), 
    range=[[x1_grid, x2_grid], [y1_grid, y2_grid]])


def normalized_count_heatmap(df):
    counts_arr = np.zeros([x_bins, y_bins])
    for vid in pd.unique(df['video_name']):
        sub_df = df[df['video_name']==vid]

        x_coords = np.array(sub_df[('mid_spine', 'x')], dtype=float)
        y_coords = np.array(sub_df[('mid_spine', 'y')], dtype=float)

        counts, _, _ = np.histogram2d(
            # x_coords_filtered, y_coords_filtered, 
            x_coords, y_coords, bins=[x_edges, y_edges],
            range=[[x1_grid, x2_grid], [y1_grid, y2_grid]])
        normalized_counts = counts / counts.sum()
        counts_arr = counts_arr + normalized_counts
    avg_norm_count = counts_arr / len(pd.unique(df['video_name']))
    return avg_norm_count, None


def average_speed_heatmap(df, smoothing_window=1):
    all_sum_grid = np.zeros([x_bins, y_bins])
    all_sum = 0
    vid_names = pd.unique(df['video_name'])
    for vid in vid_names:
        sub_df = df[df['video_name']==vid]
        # fps = 1/(sub_df[('time', '')][1] - sub_df[('time', '')][0])
        fps = 1/(sub_df["time"].iloc[1] - sub_df["time"].iloc[0])

        speed_sums = np.zeros((x_bins, y_bins))
        counts = np.zeros((x_bins, y_bins))

        x_coords = np.array(sub_df[('mid_spine', 'x')], dtype=float)
        y_coords = np.array(sub_df[('mid_spine', 'y')], dtype=float)

        speeds = get_spd_arr(x_coords, y_coords, threshold=1, smoothing_window=smoothing_window)
        speeds = speeds * fps / 12 * 10
        mask = filter_speed(speeds, 3)
        speeds = speeds[mask]
        all_sum = all_sum + np.mean(speeds)
        x_mid = (x_coords[1:] + x_coords[:-1])/2
        y_mid = (y_coords[1:] + y_coords[:-1])/2
        x_mid = x_mid[mask]
        y_mid = y_mid[mask]
        x_indices = np.clip(np.floor((x_mid - x1_grid) / 10).astype(int), 0, x_bins - 1)
        y_indices = np.clip(np.floor((y_mid - y1_grid) / 10).astype(int), 0, y_bins - 1)

        for i, spd in enumerate(speeds):
            speed_sums[x_indices[i], y_indices[i]] += spd
            counts[x_indices[i], y_indices[i]] += 1
        average_speeds = np.divide(speed_sums, counts, out=np.zeros_like(speed_sums), where=(counts!=0))
        all_sum_grid = all_sum_grid + average_speeds
        # all_sum_grid = all_sum_grid + speed_sums
    avg_speed_grid = all_sum_grid / len(vid_names)
    avg_speed = all_sum / len(vid_names)

    return avg_speed_grid, avg_speed

def average_ang_spd_heatmap(df, smoothing_window=1):
    # test_arr = np.transpose(np.array([x_coords, y_coords]))
    all_sum_grid = np.zeros([x_bins, y_bins])
    all_sum = 0
    vid_names = pd.unique(df['video_name'])
    for vid in vid_names:
        sub_df = df[df['video_name']==vid]
        # fps = 1/(sub_df[('time', '')][1] - sub_df[('time', '')][0])
        fps = 1/(sub_df["time"].iloc[1] - sub_df["time"].iloc[0])

        speed_sums = np.zeros((x_bins, y_bins))
        counts = np.zeros((x_bins, y_bins))

        x1_coords = np.array(sub_df[('base_neck', 'x')], dtype=float)
        y1_coords = np.array(sub_df[('base_neck', 'y')], dtype=float)
        x2_coords = np.array(sub_df[('mid_spine', 'x')], dtype=float)
        y2_coords = np.array(sub_df[('mid_spine', 'y')], dtype=float)
        center_x = (x1_coords + x2_coords)/2
        center_y = (y1_coords + y2_coords)/2

        bp1_xy = np.transpose(np.array([x1_coords, y1_coords]))
        bp2_xy = np.transpose(np.array([x2_coords, y2_coords]))

        angles = calc_angle_deg(bp1_xy, bp2_xy, smoothing_window=smoothing_window)
        # speeds = calc_angle_speed_deg(angles, fps=fps)
        speeds = np.diff(angles)
        speeds = np.abs(speeds)
        speeds = [360-ang if ang>180 else ang for ang in speeds]
        speeds = np.array(speeds, dtype=float) * fps
        mask = filter_speed(speeds, 3)
        speeds = speeds[mask]
        all_sum = all_sum + np.mean(speeds)

        x_mid = (center_x[1:] + center_x[:-1])/2
        y_mid = (center_y[1:] + center_y[:-1])/2
        x_mid = x_mid[mask]
        y_mid = y_mid[mask]
        x_indices = np.clip(np.floor((x_mid - x1_grid) / 10).astype(int), 0, x_bins)
        y_indices = np.clip(np.floor((y_mid - y1_grid) / 10).astype(int), 0, y_bins)

        for i, spd in enumerate(speeds):
            speed_sums[x_indices[i], y_indices[i]] += spd
            counts[x_indices[i], y_indices[i]] += 1
        average_speeds = np.divide(speed_sums, counts, out=np.zeros_like(speed_sums), where=(counts!=0))
        all_sum_grid = all_sum_grid + average_speeds
    avg_speed_grid = all_sum_grid / len(vid_names)
    avg_speed = all_sum / len(vid_names)
    return avg_speed_grid, avg_speed


def plot_heatmap(df, mode='count'):

    if mode == 'count':
        out_folder = "out/heatmaps/normailzed_count"
        os.makedirs(out_folder, exist_ok=True)
        cmap = "Blues"
        hm_fcn = normalized_count_heatmap
        cbar_lim = [0, 0.01]
    elif mode == 'speed':
        out_folder = "out/heatmaps/average_speed"
        os.makedirs(out_folder, exist_ok=True)
        cmap = "Reds"
        hm_fcn = partial(average_speed_heatmap, smoothing_window=1)
        cbar_lim = [0, 80]
        unit = "mm/s"

    elif mode == 'angle':
        out_folder = "out/heatmaps/average_angular_speed"
        os.makedirs(out_folder, exist_ok=True)
        cmap = "Greens"
        hm_fcn = partial(average_ang_spd_heatmap, smoothing_window=1)
        cbar_lim = [0, 100]
        unit = "deg/s"

    for cond in ['AD', 'WT']:
        for month in pd.unique(df['month']):

            # counts_arr = np.zeros([x_bins, y_bins])
            sub_df = df[df['type']==cond]
            sub_df = sub_df[sub_df['month']==month]

            hm, average_val = hm_fcn(sub_df)

            fig, ax = plt.subplots(1,1, figsize=[x_bins/2, y_bins/2], constrained_layout=True)
            sns.heatmap(hm, cmap=cmap, ax=ax)
            if average_val is not None:
                ax.set_title(f"{cond}_{month}: average = {np.round(average_val,4)} {unit}", fontsize=30, pad=20)
            else:
                ax.set_title(f"{cond}_{month}", fontsize=30, pad=20)
            ax.tick_params(left=False, bottom=False)
            ax.collections[0].set_clim(cbar_lim)
            ax.set_xticklabels('')
            ax.set_yticklabels('')
            fig.savefig(f"{out_folder}/heatmap_{cond}_{month}.tiff")

    return hm


if __name__=="__main__":
    result_path = '../GridWalking-wyc-2023-10-30/new_analyzed/sub_coords.h5'

    # reading and excluding outliers
    df = pd.read_hdf(result_path)
    df = df[~(df['video_name'].isin(EXCLUSION))]

    hm_count = plot_heatmap(df, mode='count')
    hm_speed = plot_heatmap(df, mode='speed')
    hm_angle = plot_heatmap(df, mode='angle')