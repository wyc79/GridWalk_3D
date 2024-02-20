
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

center_x = int((x1_grid + x2_grid)/2)
center_y = int((y1_grid + y2_grid)/2)
len_x = x2_grid - x1_grid
len_y = y2_grid - y1_grid

inner_range_x = np.array((int(center_x-1/10*len_x), int(center_x+1/10*len_x)))
inner_range_y = np.array((int(center_y-1/10*len_y), int(center_y+1/10*len_y)))
middle_range_x = np.array((int(center_x-3/10*len_x), int(center_x+3/10*len_x)))
middle_range_y = np.array((int(center_y-3/10*len_y), int(center_y+3/10*len_y)))
outer_range_x = np.array((x1_grid, x2_grid))
outer_range_y = np.array((y1_grid, y2_grid))

x_bins = int((x2_grid - x1_grid)/10)+1
y_bins = int((y2_grid - y1_grid)/10)+1

reference_grid = np.zeros([x_bins, y_bins])
inner_idx_x = np.clip(np.floor((np.array(inner_range_x) - x1_grid) / 10).astype(int), 0, x_bins)
inner_idx_y = np.clip(np.floor((np.array(inner_range_y) - y1_grid) / 10).astype(int), 0, y_bins)
middle_idx_x = np.clip(np.floor((np.array(middle_range_x) - x1_grid) / 10).astype(int), 0, x_bins)
middle_idx_y = np.clip(np.floor((np.array(middle_range_y) - y1_grid) / 10).astype(int), 0, y_bins)
reference_grid[inner_idx_x[0]:inner_idx_x[1]+2, inner_idx_y[0]:inner_idx_y[1]+2] += 1
reference_grid[middle_idx_x[0]:middle_idx_x[1]+2, middle_idx_y[0]:middle_idx_y[1]+2] += 1



def get_location(df):
    vid_names = pd.unique(df['video_name'])
    region_ratio = {0:[], 1:[], 2:[]}
    for vid in vid_names:
        sub_df = df[df['video_name']==vid]

        x_coords = np.array(sub_df[('mid_spine', 'x')], dtype=float)
        y_coords = np.array(sub_df[('mid_spine', 'y')], dtype=float)

        x_mid = (x_coords[1:] + x_coords[:-1])/2
        y_mid = (y_coords[1:] + y_coords[:-1])/2
        x_indices = np.clip(np.floor((x_mid - x1_grid) / 10).astype(int), 0, x_bins)
        y_indices = np.clip(np.floor((y_mid - y1_grid) / 10).astype(int), 0, y_bins)

        region_labels = np.zeros_like(x_indices)
        for i,_ in enumerate(x_indices):
            region_labels[i] = reference_grid[x_indices[i], y_indices[i]]

        for k in region_ratio.keys():
            region_ratio[k] += [np.sum(region_labels==k)/len(region_labels)]

    return vid_names, region_ratio


def get_speeds(df, smoothing_window=1):
    vid_names = pd.unique(df['video_name'])
    spd_arr = []
    region_spds = {0:[], 1:[], 2:[]}
    for vid in vid_names:
        sub_df = df[df['video_name']==vid]
        # fps = 1/(sub_df[('time', '')][1] - sub_df[('time', '')][0])
        fps = 1/(sub_df["time"].iloc[1] - sub_df["time"].iloc[0])

        x_coords = np.array(sub_df[('mid_spine', 'x')], dtype=float)
        y_coords = np.array(sub_df[('mid_spine', 'y')], dtype=float)

        speeds = get_spd_arr(x_coords, y_coords, threshold=1, smoothing_window=smoothing_window)
        speeds = speeds * fps / 12 * 10
        mask = filter_speed(speeds, 3)
        speeds = speeds[mask]
        x_mid = (x_coords[1:] + x_coords[:-1])/2
        y_mid = (y_coords[1:] + y_coords[:-1])/2
        x_mid = x_mid[mask]
        y_mid = y_mid[mask]
        x_indices = np.clip(np.floor((x_mid - x1_grid) / 10).astype(int), 0, x_bins)
        y_indices = np.clip(np.floor((y_mid - y1_grid) / 10).astype(int), 0, y_bins)

        region_labels = np.zeros_like(x_indices)
        for i,_ in enumerate(x_indices):
            region_labels[i] = reference_grid[x_indices[i], y_indices[i]]

        spd_arr = spd_arr + [np.mean(speeds)]

        for k in region_spds.keys():
            region_spds[k] += [np.mean(speeds[region_labels==k])]

    region_spds['all'] = spd_arr

    return vid_names, region_spds

def get_ang_speeds(df, smoothing_window=1):
    vid_names = pd.unique(df['video_name'])
    spd_arr = []
    region_spds = {0:[], 1:[], 2:[]}
    for vid in vid_names:
        sub_df = df[df['video_name']==vid]
        # fps = 1/(sub_df[('time', '')][1] - sub_df[('time', '')][0])
        fps = 1/(sub_df["time"].iloc[1] - sub_df["time"].iloc[0])

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
        # spd_arr = spd_arr + [np.mean(speeds)]

        x_mid = (center_x[1:] + center_x[:-1])/2
        y_mid = (center_y[1:] + center_y[:-1])/2
        x_mid = x_mid[mask]
        y_mid = y_mid[mask]
        x_indices = np.clip(np.floor((x_mid - x1_grid) / 10).astype(int), 0, x_bins)
        y_indices = np.clip(np.floor((y_mid - y1_grid) / 10).astype(int), 0, y_bins)

        region_labels = np.zeros_like(x_indices)
        for i,_ in enumerate(x_indices):
            region_labels[i] = reference_grid[x_indices[i], y_indices[i]]

        spd_arr = spd_arr + [np.mean(speeds)]

        for k in region_spds.keys():
            region_spds[k] += [np.mean(speeds[region_labels==k])]

    region_spds['all'] = spd_arr

    return vid_names, region_spds

def contains_special_strings(s, str_arr):
    for string in str_arr:
        if str(string) in s:
            return string
    
    return None

title_dict = {0: "Outer Region", 
            1: "Middle Region",
            2: "Center Region",
            'all': 'Overall'}


def plot_bar(res_df, filename, palette, y_label=None):
    hue = 'type'
    col = 'region'
    x = 'month'
    y = 'speed'

    hue_order = ['WT', 'AD']
    col_order = pd.unique(res_df[col])
    order = ['3-4M', '5-8M', '9-12M']
    

    # pairs = [[(o, ho) for ho in hue_order] for o in order]

    barplot = sns.catplot(
        x=x, y=y, hue=hue, data=res_df, kind='bar',
        order=order, hue_order=hue_order, legend=False,
        palette=palette, alpha=0.5,
        col=col, col_order=col_order,
        height=8, aspect=1., edgecolor='white', linewidth=2,
        ci=95, n_boot=1000, errwidth=2, capsize=.1,)

    barplot.fig.subplots_adjust(wspace=1)

    axes = barplot.axes[0]
    ##### ax.grid(axis='y', color='grey', linestyle='-', alpha=0.1, linewidth=1)

    if y_label is None:
        y_label = " "
    
    for ax in axes:
        region = contains_special_strings(ax.get_title(), pd.unique(res_df['region']))
        region_df = res_df[res_df['region']==region]

        sns.swarmplot(x=x, y=y, data=region_df, hue=hue, order=order, 
                    hue_order=hue_order, size=12, 
                    palette=palette, alpha=0.9, ax=ax, dodge=True,
                    edgecolor='white', linewidth=2)
        ax.legend().set_visible(False)
        ax.set_xlabel('Age', weight='bold',fontsize=35)
        ax.set_ylabel(y_label, weight='bold', fontsize=35)
        ax.set_xticklabels(['3-4M', '6-7M', '10-11M'], weight='bold',
                rotation=40, horizontalalignment='right')
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        ax.set_title(title_dict[region], fontsize=35, weight='bold')

        axis_width = 2
        for _, spine in ax.spines.items():
            spine.set_linewidth(axis_width)
        for axis in [ax.xaxis, ax.yaxis]:
            for tick in axis.get_major_ticks():
                tick.tick1line.set_linewidth(axis_width)
                tick.tick2line.set_linewidth(axis_width)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles=handles[:len(hue_order)], labels=labels[:len(hue_order)],
        loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)
    
    barplot.fig.tight_layout()
    barplot.savefig(filename) 

def coords2barplot(df, mode='speed'):
    res_df = pd.DataFrame()
    if mode == 'location':
        plot_fcn = get_location
        out_name = f"out/regions_location.tiff"
        palette = sns.color_palette(['#272927', '#355fe8'])
        y_label = "Ratio"

    elif mode == 'speed':
        plot_fcn = partial(get_speeds, smoothing_window=1)
        out_name = f"out/regions_speed.tiff"
        palette = sns.color_palette(['#272927', '#d1492e'])
        y_label = "Speed (mm/s)"

    elif mode == 'angle':
        plot_fcn = partial(get_ang_speeds, smoothing_window=1)
        out_name = f"out/regions_angle.tiff"
        palette = sns.color_palette(['#272927', '#008000'])
        y_label = "Speed (deg/s)"

    for cond in ['WT', 'AD']:
        for month in pd.unique(df['month']):

            # counts_arr = np.zeros([x_bins, y_bins])
            sub_df = df[df['type']==cond]
            sub_df = sub_df[sub_df['month']==month]
            vid_names, region_spds = plot_fcn(sub_df)
            for key, value in region_spds.items():
                temp_res = pd.DataFrame()
                temp_res['type'] = [cond] * len(value)
                temp_res['month'] = [month] * len(value)
                temp_res['region'] = [key] * len(value)
                temp_res['speed'] = value
                temp_res['video_names'] = vid_names
            
                res_df = pd.concat([res_df, temp_res], ignore_index=True)

    plot_bar(res_df, out_name, palette=palette, y_label=y_label)
    return res_df


if __name__=="__main__":
    result_path = '../GridWalking-wyc-2023-10-30/new_analyzed/sub_coords.h5'

    # reading and excluding outliers
    df = pd.read_hdf(result_path)
    df = df[~(df['video_name'].isin(EXCLUSION))]

    # palette = sns.color_palette(['#272927', '#008000'])
    
    _ = coords2barplot(df, mode='speed')
    _ = coords2barplot(df, mode='angle')
    _ = coords2barplot(df, mode='location')
    

    # ax.set_xlabel('Age', weight='bold',fontsize=35)
    # # ax.set_ylabel('percentage moving less than 1 px/fr', fontsize=30)
    # ax.set_ylabel('Travel speed (mm/s)', weight='bold', fontsize=35)
    # # ax.set_ylabel('Center:Peripheral Ratio', fontsize=30)
    # ax.set_xticklabels(['3-4M', '6-7M', '10-11M'], weight='bold',
    #                 rotation=40, horizontalalignment='right')

    # ax.tick_params(axis='x', labelsize=25)
    # ax.tick_params(axis='y', labelsize=25)

    # for i, bar in enumerate(ax.patches):  # iterate over the bars
    #     # Calculate the center x position of each bar
    #     x_value = bar.get_x() + bar.get_width() / 2
    #     ax.axvline(x=x_value, color='grey', linestyle='-', alpha=0.1, linewidth=1)  # add the gridline

    # axis_width = 2
    # for _, spine in ax.spines.items():
    #     spine.set_linewidth(axis_width)
    # for axis in [ax.xaxis, ax.yaxis]:
    #     for tick in axis.get_major_ticks():
    #         tick.tick1line.set_linewidth(axis_width)
    #         tick.tick2line.set_linewidth(axis_width)
