import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

example_file = \
    '../GridWalking-wyc-2023-10-30/new_analyzed/AD/3-4M/AD688_2023-09-11_11-06-19DLC_resnet50_GridWalkingDec12shuffle1_50000_filtered_transformed.csv'

df = pd.read_csv(example_file, header=None, skiprows=3)
header = pd.read_csv(example_file, header=None, nrows=3)
multi_level_header = pd.MultiIndex.from_arrays(header.values)
df.columns = multi_level_header
df.columns = df.columns.droplevel(0)

line_dict = [
    ['base_neck', 'L_shoulder'], ['base_neck', 'R_shoulder'],
    ['mid_spine', 'L_shoulder'], ['mid_spine', 'R_shoulder'],
    ['mid_spine', 'L_thigh'], ['mid_spine', 'R_thigh'],
    ['L_thigh', 'L_shoulder'], ['R_thigh', 'R_shoulder']
]

lateral_points = ['L_shoulder', 'R_shoulder', 'L_thigh', 'R_thigh']

df_line = pd.DataFrame()

for line in line_dict:
    df_line[f"{line[0]}-{line[1]}"] = np.sqrt(
        (df[(line[0], 'x')] - df[(line[1], 'x')])**2 + (df[(line[0], 'y')] - df[(line[1], 'y')])**2)
df_line['t'] = np.array(df_line.index, dtype=float)/10

n_plots = len(line_dict)+1
fig, ax = plt.subplots(n_plots, 1, figsize=[30, n_plots * 4], constrained_layout=True)
for i,line in enumerate(line_dict):
    col = f"{line[0]}-{line[1]}"
    ax[i].plot(df_line['t'], df_line[col])
    ax[i].set_ylabel(col,
                    rotation=0,
                    fontsize=30,
                    horizontalalignment='right')
    ax[i].tick_params(axis='x', labelsize=20)
    ax[i].tick_params(axis='y', labelsize=20)
    ax[i].set_xlim([20,50])

ax[i+1].plot(df_line['t'], df_line["L_thigh-L_shoulder"]/df_line["R_thigh-R_shoulder"])
ax[i+1].set_ylabel('Ratio',
                rotation=0,
                fontsize=30,
                horizontalalignment='right')
ax[i+1].tick_params(axis='x', labelsize=20)
ax[i+1].tick_params(axis='y', labelsize=20)
ax[i+1].set_xlim([20,50])


fig.suptitle("Length (in px)", fontsize=40)


fig, ax = plt.subplots(len(lateral_points), 1, figsize=[30, len(lateral_points) * 4], constrained_layout=True)
for i,bp in enumerate(lateral_points):
    # col = f"{line[0]}-{line[1]}"

    ax[i].plot(df_line['t'], df[(bp, 'x')], label='x')
    ax[i].plot(df_line['t'], df[(bp, 'y')], label='y')
    ax[i].set_ylabel(bp,
                    rotation=0,
                    fontsize=30,
                    horizontalalignment='right')
    ax[i].tick_params(axis='x', labelsize=20)
    ax[i].tick_params(axis='y', labelsize=20)
    ax[i].set_xlim([20,50])
    ax[i].legend(loc='upper right', fontsize=20)

fig.suptitle("Coordinates", fontsize=40)



# lateral_points = ['L_shoulder', 'R_shoulder']

lateral_points = ['L_thigh', 'R_thigh']

fig, ax = plt.subplots(1, 1, figsize=[30, 1 * 4], constrained_layout=True)
for i,bp in enumerate(lateral_points):
    # col = f"{line[0]}-{line[1]}"

    dx = np.diff(df[(bp, 'x')])
    dy = np.diff(df[(bp, 'y')])
    spd = np.sqrt(dx**2 + dy**2)

    ax.plot(df_line['t'][:-1], spd, label=bp)
    ax.set_ylabel("LR speed",
                    rotation=0,
                    fontsize=30,
                    horizontalalignment='right')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlim([20,25])
    ax.legend(loc='upper right', fontsize=20)

# fig.suptitle("Speed thigh", fontsize=40)