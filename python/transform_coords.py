import os
import glob
import math
import copy

import pandas as pd
import numpy as np

# from gait_analysis import calc_center_speed

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

MODEL_NAME = 'DLC_resnet50_GridWalkingDec12shuffle1_50000'

analyzed_root = '../GridWalking-wyc-2023-10-30/new_analyzed'
body_parts = [
    'nose', 'base_neck', 'L_shoulder', 'R_shoulder', 'mid_spine', 
    'L_thigh', 'R_thigh', 'hip', 'mid_tail', 'tail_tip']

lateral_points = ['L_shoulder', 'R_shoulder', 'L_thigh', 'R_thigh']

def rotate_point(x, y, angle):
    angle_rad = math.radians(angle)
    rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return rotated_x, rotated_y

def calc_angle_deg(bp1_xy, bp2_xy, smoothing_window=1):
    """
    modified from gaitinference
    calculates the angle of the orientation of the mouse in degrees, 
    bp1_xy: x & y coordinates of body part 1 (similar for bp2_xy)
    the calculated angle (vector) is from bp2 to bp1
    """

    bp1_xy = np.array(bp1_xy, dtype=float)
    bp2_xy = np.array(bp2_xy, dtype=float)

    line_offset_xy = bp1_xy - bp2_xy

    angle_rad = np.arctan2(line_offset_xy[:, 1], line_offset_xy[:, 0])
    angle_rad = _smooth(angle_rad, smoothing_window)

    return angle_rad * (180 / math.pi)

def calc_angle_speed_deg(angles, fps):
    """
    Calculate angular velocity from the given angles.
    """
    speed_deg = np.array(list(_gen_calc_angle_speed_deg(angles))) * fps
    # speed_deg = _smooth(speed_deg, smoothing_window)

    return speed_deg

def get_spd_arr(xs, ys, threshold=0, smoothing_window=1):
    xs = _smooth(xs, smoothing_window)
    ys = _smooth(ys, smoothing_window)
    dx = np.diff(np.array(xs, dtype=float))
    dy = np.diff(np.array(ys, dtype=float))
    spd = np.sqrt(dx**2 + dy**2)
    spd[spd<threshold] = 0
    return spd

def filter_speed(speeds, std=3):
    mask = np.ones_like(speeds)
    upper_limit = np.mean(speeds) + std*np.std(speeds)
    lower_limit = np.mean(speeds) - std*np.std(speeds)
    mask[speeds>upper_limit] = 0
    mask[speeds<lower_limit] = 0
    return np.array(mask, dtype=bool)

def _smooth(vec, smoothing_window):
    # changed np.float to np.double because of numpy version
    vec = np.array(vec, dtype=np.double)
    if smoothing_window <= 1 or len(vec) == 0:
        return vec
    else:
        assert smoothing_window % 2 == 1, 'expected smoothing_window to be odd'
        half_conv_len = smoothing_window // 2
        smooth_tgt = np.concatenate([
            np.full(half_conv_len, vec[0], dtype=vec.dtype),
            vec,
            np.full(half_conv_len, vec[-1], dtype=vec.dtype),
        ])

        smoothing_val = 1 / smoothing_window
        conv_arr = np.full(smoothing_window, smoothing_val)

        return np.convolve(smooth_tgt, conv_arr, mode='valid')
    

def _gen_calc_angle_speed_deg(angles):
    # we need smooth out the -180-180 breakpoint in order to calculate speed
    # correctly
    for i in range(len(angles) - 1):

        angle1 = angles[i]
        angle1 = angle1 % 360
        if angle1 < 0:
            angle1 += 360

        angle2 = angles[i + 1]
        angle2 = angle2 % 360
        if angle2 < 0:
            angle2 += 360

        diff1 = angle2 - angle1
        abs_diff1 = abs(diff1)
        diff2 = (360 + angle2) - angle1
        abs_diff2 = abs(diff2)
        diff3 = angle2 - (360 + angle1)
        abs_diff3 = abs(diff3)

        if abs_diff1 <= abs_diff2 and abs_diff1 <= abs_diff3:
            yield diff1
        elif abs_diff2 <= abs_diff3:
            yield diff2
        else:
            yield diff3

    yield 0






if __name__ == "__main__":
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
                    
                    # Calculate rotation angle to align base_neck and mid_spine vertically
                    dx = coords['base_neck'][i,0] - coords['mid_spine'][i,0]
                    dy = coords['base_neck'][i,1] - coords['mid_spine'][i,1]
                    rotate_angle = math.degrees(math.atan2(-dx, dy))

                    # Rotate all points
                    for bp in body_parts:
                        coords[bp][i,0], coords[bp][i,1] = rotate_point(coords[bp][i,0], coords[bp][i,1], -rotate_angle)

                    # print(f"{cond}, {m}")

                for bp in body_parts:
                    transformed_df[(MODEL_NAME, bp, 'x')] = coords[bp][:,0]
                    transformed_df[(MODEL_NAME, bp, 'y')] = coords[bp][:,1]
                    transformed_df[(MODEL_NAME, bp, 'likelihood')] = df[(MODEL_NAME, bp,'likelihood')]
                
                transformed_df[(MODEL_NAME, 'original', 'angle')] = calc_angle_deg(
                    coords_original['base_neck'], coords_original['mid_spine'])
                # transformed_df[(MODEL_NAME, 'original', 'center_speed')] = calc_center_speed(df, cm_per_px=int, fps=int, smoothing_window=int, start_index=__class__, stop_index=__class__)
                

                transformed_df[(MODEL_NAME, 'original_base_neck', 'x')] = df[(MODEL_NAME, 'base_neck', 'x')]
                transformed_df[(MODEL_NAME, 'original_base_neck', 'y')] = df[(MODEL_NAME, 'base_neck', 'y')]
                transformed_df[(MODEL_NAME, 'original_mid_spine', 'x')] = df[(MODEL_NAME, 'mid_spine', 'x')]
                transformed_df[(MODEL_NAME, 'original_mid_spine', 'y')] = df[(MODEL_NAME, 'mid_spine', 'y')]

                transformed_df.to_csv(f"{f[:-3]}_transformed.csv")
                # spd_df = pd.DataFrame(columns = lateral_points + ['mov_spd'])

                # for lp in lateral_points:
                #     lp_df = transformed_df.loc[:, transformed_df.columns.get_level_values(1) == lp]
                #     lp_df.columns = lp_df.columns.get_level_values(2)
                #     spd_df[lp] = get_dist_arr(lp_df['x'], lp_df['y'], threshold=1)
                
                # spd_df['mov_spd'] = (get_dist_arr(
                #     coords_original['base_neck'][:,0], coords_original['base_neck'][:,1], threshold=0
                # ) + get_dist_arr(
                #     coords_original['mid_spine'][:,0], coords_original['mid_spine'][:,1], threshold=0
                # ))/2

                # spd_df.to_csv(f"{f[:-3]}_speed.csv")

                
                

                        
