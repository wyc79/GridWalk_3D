# calculate the traverse distance of mid_spine
# + testing of different stuff

import os
import glob

import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


ROOT = '../GridWalking-wyc-2023-10-30/'
out_path = './out'

analyzed_files = glob.glob(os.path.join(ROOT, 'dlc_analyzed', "*50000_filtered.csv"))
result = pd.DataFrame(columns=['filename', 'type', 'total_distance'])

result_row_list = []

for fname in analyzed_files:
    df = pd.read_csv(fname)
    df = df.drop(columns=["scorer"])
    df.columns = pd.MultiIndex.from_arrays(df.iloc[0:2].values)
    df = (df.iloc[2:]).reset_index(drop=True)
    x_coords = np.array(df.loc[:, ('mid_spine','x')], dtype=float)
    y_coords = np.array(df.loc[:, ('mid_spine','y')], dtype=float)
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    distances = np.sqrt(dx**2 + dy**2)
    delta_d = np.mean(distances)
    # print(f"{fname}: {total_distance} pxs")

    dict1 = {
        'filename': fname.split('/')[-1], 
        'type': 'AD' if 'AD' in fname else 'WT',
        'avg_distance(px)': delta_d}
    result_row_list.append(dict1)

result = pd.DataFrame(result_row_list) 
result.to_csv(os.path.join(out_path, 'distance.csv'))



import cv2

# test_vid = '/Users/yuanchenwang/Library/CloudStorage/GoogleDrive-yuanchen.wang79@gmail.com/.shortcut-targets-by-id/1HwmmBaVNGnvstFsQoeaqCLyTIKZ0HoD2/LuLab_Xiaofei_Yixin sharing_Yuancheng/Extraction and Labeling/GridWalk_3D/GridWalking-wyc-2023-10-30/videos/WT282_2023-04-27_14-23-49.mp4'

test_vid = '/Users/yuanchenwang/Library/CloudStorage/GoogleDrive-yuanchen.wang79@gmail.com/.shortcut-targets-by-id/1HwmmBaVNGnvstFsQoeaqCLyTIKZ0HoD2/LuLab_Xiaofei_Yixin sharing_Yuancheng/data_organized/gridwalking/AD/9-12M/AD296_2023-09-07_17-34-27.mp4'
cv2.VideoCapture(test_vid).get(cv2.CAP_PROP_FPS)

import subprocess

video_path = test_vid # Replace with the path to your video file

# Get frame rate and duration using FFmpeg
command = f"ffmpeg -i '{video_path}'"
result = subprocess.Popen(
    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

output = result.stdout.read()

# Parse the output to extract frame rate and duration information
frame_rate_line = [line for line in output.split('\n') if 'Stream #0:0' in line and 'Video' in line][0]
duration_line = [line for line in output.split('\n') if 'Duration' in line][0]

frame_rate = frame_rate_line.split(",")[4].split()[0]
duration = duration_line.split(",")[0].split()[1]

print(f"Frame rate: {frame_rate} frames per second")
print(f"Duration: {duration}")

