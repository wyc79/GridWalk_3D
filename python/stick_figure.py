
import math
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


example_file = \
    '../GridWalking-wyc-2023-10-30/new_analyzed/AD/3-4M/AD688_2023-09-11_11-06-19DLC_resnet50_GridWalkingDec12shuffle1_50000_filtered.h5'

df = pd.read_hdf(example_file)
df.columns = df.columns.droplevel(0)

# sub_df = df.head(10)
# sub_df.to_hdf('test_df.h5', key='test_df')

body_parts = pd.unique(df.columns.get_level_values(0))

skeleton = [
    ['nose', 'base_neck'], ['nose', 'L_shoulder'], ['nose', 'R_shoulder'], 
    ['base_neck', 'mid_spine'], ['base_neck', 'L_shoulder'], ['base_neck', 'R_shoulder'],
    ['L_shoulder', 'mid_spine'], ['R_shoulder', 'mid_spine'], ['mid_spine', 'hip'], 
    ['mid_spine', 'L_thigh'], ['mid_spine', 'R_thigh'], ['L_shoulder', 'L_thigh'], 
    ['R_shoulder', 'R_thigh'], ['L_thigh', 'hip'], ['R_thigh', 'hip'], 
    ['L_thigh', 'R_thigh'], ['hip', 'mid_tail'], ['mid_tail', 'tail_tip']
]

def rotate_point(x, y, angle):
    angle_rad = math.radians(angle)
    rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return rotated_x, rotated_y


L_length = np.sqrt((df[('L_shoulder', 'x')] - df[('L_thigh', 'x')])**2 + (df[('L_shoulder', 'y')] - df[('L_thigh', 'y')])**2)
R_length = np.sqrt((df[('R_shoulder', 'x')] - df[('R_thigh', 'x')])**2 + (df[('R_shoulder', 'y')] - df[('R_thigh', 'y')])**2)
ratio = L_length/R_length

coords_original = {part: df[part][['x', 'y']].values for part in body_parts}
coords = copy.deepcopy(coords_original)

for i in range(df.shape[0]):
    mid_point_x = (coords_original['mid_spine'][i,0] + coords_original['base_neck'][i,0]) / 2
    mid_point_y = (coords_original['mid_spine'][i,1] + coords_original['base_neck'][i,1]) / 2
    
    # Translate all points
    for bp in body_parts:
        coords[bp][i,0] = coords_original[bp][i,0] - mid_point_x
        coords[bp][i,1] = coords_original[bp][i,1] - mid_point_y
    
    # Calculate rotation angle to align base_neck and mid_spine horizontally
    dx = coords['base_neck'][i,0] - coords['mid_spine'][i,0]
    dy = coords['base_neck'][i,1] - coords['mid_spine'][i,1]
    angle = math.degrees(math.atan2(-dx, dy))

    # Rotate all points
    for bp in body_parts:
        coords[bp][i,0], coords[bp][i,1] = rotate_point(coords[bp][i,0], coords[bp][i,1], -angle)


xmax = df[body_parts[0]]['x'].max() + 50
ymax = df[body_parts[0]]['y'].max() + 50


fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # Left subplot twice as wide as the right

# Create two subplots
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

def update_plot(frame_num):
    ax1.clear()
    ax2.clear()

    ax1.set_xlim(0, xmax)
    ax1.set_ylim(0, ymax)
    ax2.set_xlim(-xmax/4, xmax/4)
    ax2.set_ylim(-ymax/2, ymax/2)

    # Draw x=0 and y=0 lines 
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axvline(0, color='gray', linestyle='--')

    for link in skeleton:
        if link[0] in coords_original and link[1] in coords_original:
            x_values = [coords_original[link[0]][frame_num, 0], coords_original[link[1]][frame_num, 0]]
            y_values = [coords_original[link[0]][frame_num, 1], coords_original[link[1]][frame_num, 1]]
            ax1.plot(x_values, y_values, '-', color='black')

            x_values = [coords[link[0]][frame_num, 0], coords[link[1]][frame_num, 0]]
            y_values = [coords[link[0]][frame_num, 1], coords[link[1]][frame_num, 1]]
            ax2.plot(x_values, y_values, '-', color='black')
    
    for part in body_parts:
        ax1.plot(coords_original[part][frame_num, 0], coords_original[part][frame_num, 1], 'o-', label=part)
        ax2.plot(coords[part][frame_num, 0], coords[part][frame_num, 1], 'o-', label=part)

    # ax1.set_title('Original Coordinates')
    get_val = lambda arr: np.round(arr[frame_num],2)
    ax2.set_title(f'L_length={get_val(L_length)}, R_length={get_val(R_length)}, LR_ratio={get_val(ratio)}')

    dx = coords_original['hip'][frame_num, 0] - coords_original['mid_spine'][frame_num-1, 0]
    dy = coords_original['hip'][frame_num, 1] - coords_original['mid_spine'][frame_num-1, 1]

    if (frame_num > 0) and (np.sqrt(dx**2 + dy**2)/12*10>3):
        ax1.set_title('moving', color='red')
    else:
        ax1.set_title('not moving', color='black')

# Creating the animation
ani = animation.FuncAnimation(fig, update_plot, frames=len(df), interval=100)

# Save the updated animation as a video file with vertical alignment
video_path = 'sticks/centered.mp4'
ani.save(video_path, writer='ffmpeg', fps=20)

# Output the video file path
print("Video saved to:", video_path)
