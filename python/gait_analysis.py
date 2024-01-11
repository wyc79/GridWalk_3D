import affine
import argparse
import h5py
import itertools
import math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import scipy.stats

import pandas as pd
import numpy as np

import borrowed.gaitinference as gait
# from borrowed.gaitinference import _smooth

# paw_speeds -> pt_spds
# base_tail_speeds -> mv_spds

# need to account for different fps & cm/px settings

FRAMES_PER_SECOND = 10
CM_PER_PIXEL = 1/12


def calc_speed(xy_pos, start_index=None, stop_index=None, smoothing_window=1, 
               cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND):
    # xy_pos = group['points'][start_index : stop_index, point_index, :].astype(np.double)
    xy_pos = np.array(xy_pos[start_index : stop_index, : ], dtype=float)
    xy_pos[:, 0] = gait._smooth(xy_pos[:, 0], smoothing_window)
    xy_pos[:, 1] = gait._smooth(xy_pos[:, 1], smoothing_window)

    xy_pos *= cm_per_px
    velocity = np.gradient(xy_pos, axis=0)
    speed_cm_per_sec = np.linalg.norm(velocity, axis=1) * fps
    return speed_cm_per_sec

def calc_center_speed(df, start_index=None, stop_index=None, smoothing_window=1,
                      cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND):
    base_neck_spd = calc_speed(
        np.array(df[[('original_base_neck','x'), ('original_base_neck','y')]], dtype=np.double), 
        start_index=start_index, stop_index=stop_index, 
        smoothing_window=smoothing_window, cm_per_px=cm_per_px, fps=fps)
    mid_spine_spd = calc_speed(
        np.array(df[[('original_mid_spine','x'), ('original_base_neck','y')]], dtype=np.double), 
        start_index=start_index, stop_index=stop_index, 
        smoothing_window=smoothing_window, cm_per_px=cm_per_px, fps=fps)
    return (base_neck_spd + mid_spine_spd)/2


if __name__ == '__main__':
    example_file = \
        '../GridWalking-wyc-2023-10-30/new_analyzed/AD/3-4M/AD688_2023-09-11_11-06-19DLC_resnet50_GridWalkingDec12shuffle1_50000_filtered_transformed.csv'
    df = pd.read_csv(example_file, header=None, skiprows=3)
    header = pd.read_csv(example_file, header=None, nrows=3)
    multi_level_header = pd.MultiIndex.from_arrays(header.values)
    df.columns = multi_level_header
    df.columns = df.columns.droplevel(0)

    # xy_pos = np.array(df[[('L_thigh','x'), ('L_thigh','y')]])
    l_speed = calc_speed(
        np.array(df[[('L_shoulder','x'), ('L_shoulder','y')]], dtype=np.double), 
        start_index=None, stop_index=None, smoothing_window=1,
        cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    r_speed = calc_speed(
        np.array(df[[('R_shoulder','x'), ('R_shoulder','y')]], dtype=np.double), 
        start_index=None, stop_index=None, smoothing_window=1,
        cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    angular_speed = list(gait.calc_angle_speed_deg(
        df[('original', 'angle')], smoothing_window=5))
    center_speed = calc_center_speed(
        df, start_index=None, stop_index=None, smoothing_window=1,
        cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    
    
    left_paw_accum = np.zeros(50)
    right_paw_accum = np.zeros(50)
    stride_count = 0

    tracks = list(gait.trackstridedet(
                l_speed,
                r_speed,
                center_speed,
                angular_speed,
                cm_per_px=CM_PER_PIXEL, 
                stationary_percentile=10))
    
    for track in tracks:
        strides = list(track.strides)
        left_steps = track.lrp_steps
        right_steps = track.rrp_steps
        if len(strides) >= 1:
            for stride in strides[1:-1]:
                gait.accum_steps(left_paw_accum, stride, left_steps)
                gait.accum_steps(right_paw_accum, stride, right_steps)
                stride_count += 1


    # for track in tracks:
    #     strides = list(track.strides)
    #     if len(strides) == 0:
    #         print('None')
    #     else:
    #         print(strides)
    #         test_track = track
    #         test_stride = strides


    if stride_count > 0:
        left_paw_accum /= stride_count
        right_paw_accum /= stride_count

    print(left_paw_accum)
    print(right_paw_accum)

    # for step in gait.stepdet(l_speed, center_speed, peakdelta=1, approx_still=0):
    #     print(step.start_frame)
    #     print(step.stop_frame_exclu)
