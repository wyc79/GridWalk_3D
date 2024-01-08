import os
import glob
import math
import copy

import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

MODEL_NAME = 'DLC_resnet50_GridWalkingDec12shuffle1_50000'

analyzed_root = '../GridWalking-wyc-2023-10-30/new_analyzed'
body_parts = [
    'nose', 'base_neck', 'L_shoulder', 'R_shoulder', 'mid_spine', 
    'L_thigh', 'R_thigh', 'hip', 'mid_tail', 'tail_tip']

def rotate_point(x, y, angle):
    angle_rad = math.radians(angle)
    rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return rotated_x, rotated_y

for cond in ['AD', 'WT']:
    months = glob.glob(os.path.join(analyzed_root, f'{cond}/*'))
    for m in months:
        month = m.split('/')[-1]
        coord_files = glob.glob(
            os.path.join(analyzed_root, cond, month, '*_50000_filtered.h5'), recursive=True)
        for f in coord_files:
            df = pd.read_hdf(f)
            # print(f"{cond}, {m}, {f.split('/')[-1]}")

            transformed_df = pd.DataFrame(columns=df.columns)

            coords_original = {part: df[MODEL_NAME][part][['x', 'y']].values for part in body_parts}
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

                # print(f"{cond}, {m}")

            for bp in body_parts:
                transformed_df[(MODEL_NAME, bp,'x')] = coords[bp][:,0]
                transformed_df[(MODEL_NAME, bp,'y')] = coords[bp][:,1]
                transformed_df[(MODEL_NAME, bp,'likelihood')] = df[(MODEL_NAME, bp,'likelihood')]
            
            transformed_df.to_csv(f"{f[:-3]}_transformed.csv")

                    
