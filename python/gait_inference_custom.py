"""
modified from gaitinference.py
"""


import affine
import argparse
import h5py
import itertools
import math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import scipy.stats

from gait_analysis import *


FRAMES_PER_SECOND = 10


def _smooth(vec, smoothing_window):
    # changed np.float to np.double because of numpy version
    if smoothing_window <= 1 or len(vec) == 0:
        return vec.astype(np.double)
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


## Peak detection function borrowed from Eli Billauer
def peakdet(v, delta, x = None):
    """
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """

    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    v = (v - np.mean(v))/np.std(v)
    if len(v) != len(x):
        raise Exception('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        raise Exception('Input argument delta must be a scalar')
    if delta <= 0:
        raise Exception('Input argument delta must be positive')

    mn, mx = np.Inf, np.NINF
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                # maxtab.append((mxpos, mx))
                maxtab.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                # mintab.append((mnpos, mn))
                mintab.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)
    # return np.array(maxtab).reshape(-1, 2), np.array(mintab).reshape(-1, 2)


class FrameInterval(object):
    """
    A simple class for defining frame intervals. The start frame is inclusive and the stop
    frame is exclusive.
    """

    def __init__(self, start_frame, stop_frame_exclu):
        self.start_frame = start_frame
        self.stop_frame_exclu = stop_frame_exclu

    def __len__(self):
        return self.stop_frame_exclu - self.start_frame
    
class Track(FrameInterval):

    def __init__(self, start_frame, stop_frame_exclu):
        super().__init__(start_frame, stop_frame_exclu)

        self.lrp_steps = []
        self.rrp_steps = []
        self.strides = []

class Stride(FrameInterval):
    """
    A stride interval which is deliniated by foot strike events of the left rear paw
    """
    def __init__(self, start_frame, stop_frame_exclu):
        super().__init__(start_frame, stop_frame_exclu)

    # def __init__(self, start_frame, stop_frame_exclu, speed_cm_per_sec, angular_velocity, cm_per_px=CM_PER_PIXEL):
    #     super().__init__(start_frame, stop_frame_exclu)
        # self.speed_cm_per_sec = speed_cm_per_sec
        # self.angular_velocity = angular_velocity
        # self.cm_per_px = cm_per_px

    
def stepdet(paw_speeds, base_tail_speeds, peakdelta_sd=2, approx_still=15):
    
    speed_maxs, speed_mins = peakdet(paw_speeds, peakdelta_sd)
    speed_maxs = speed_maxs.astype(np.int32)
    speed_mins = speed_mins.astype(np.int32)

    for i, speed_max_frame in enumerate(speed_maxs):

        # print('speed_max_frame:', speed_max_frame)
        toe_off_index = speed_max_frame
        while (toe_off_index > 0
                and paw_speeds[toe_off_index] > approx_still
                and paw_speeds[toe_off_index] > base_tail_speeds[toe_off_index]):
            toe_off_index -= 1
        
        if paw_speeds[toe_off_index] <= approx_still and toe_off_index < len(paw_speeds) - 1:
            toe_off_index += 1

        if i > 0 and i < len(speed_mins):
            prev_speed_min_frame = speed_mins[i - 1]
            if prev_speed_min_frame > toe_off_index:
                toe_off_index = prev_speed_min_frame + 1
        
        strike_index = speed_max_frame
        while (strike_index < len(paw_speeds) - 1
                and paw_speeds[strike_index] > approx_still
                and paw_speeds[strike_index] > base_tail_speeds[strike_index]):
            strike_index += 1

        # if we stepped past the next local min we should adjust the strike index
        if i >= 0 and i < len(speed_mins):
            next_speed_min_frame = speed_mins[i]
            if next_speed_min_frame < strike_index:
                strike_index = next_speed_min_frame

        if strike_index > toe_off_index:
            yield FrameInterval(toe_off_index, strike_index+1)

def trackdet(base_tail_speeds, speed_thresh=5, fps=FRAMES_PER_SECOND):
    """
    Detect "track" intervals for the given `base_tail_speeds`
    """
    speed_over_thresh = base_tail_speeds >= speed_thresh
    grp_frame_index = 0
    for grp_key, grp_vals in itertools.groupby(speed_over_thresh):
        grp_count = len(list(grp_vals))
        if (grp_key) and (grp_count >= fps):
            yield Track(grp_frame_index, grp_frame_index + grp_count)

        grp_frame_index += grp_count

def intervals_overlap(inter1, inter2):
    return (
        inter1.start_frame < inter2.stop_frame_exclu
        and inter1.stop_frame_exclu > inter2.start_frame
    )

def trackstridedet_tail(tail_speeds, base_tail_speeds, stationary_percentile=30):
    tail_steps = list(stepdet(
        tail_speeds, base_tail_speeds, peakdelta_sd=1, 
        approx_still=np.median(base_tail_speeds)))
    tracks = trackdet(
        base_tail_speeds, fps=FRAMES_PER_SECOND,
        speed_thresh=np.percentile(base_tail_speeds, stationary_percentile))
    
    step_cursor = 0
    for track in tracks:

        # find steps that belong to the track
        while step_cursor < len(tail_steps):
            curr_lr_step = tail_steps[step_cursor]
            if intervals_overlap(track, curr_lr_step):
                track.lrp_steps.append(curr_lr_step)

            if curr_lr_step.start_frame >= track.stop_frame_exclu:
                break
            else:
                step_cursor += 1
    

def trackstridedet(lr_paw_speeds, rr_paw_speeds, base_tail_speeds, stationary_percentile=30):
    """
    This function will detect tracks along with the strides that belong to those tracks
    """
    lr_steps = list(stepdet(
        lr_paw_speeds, base_tail_speeds, peakdelta_sd=1, 
        approx_still=np.median(base_tail_speeds)))
    rr_steps = list(stepdet(
        rr_paw_speeds, base_tail_speeds, peakdelta_sd=1, 
        approx_still=np.median(base_tail_speeds)))
    tracks = trackdet(
        base_tail_speeds, fps=FRAMES_PER_SECOND,
        speed_thresh=np.percentile(base_tail_speeds, stationary_percentile))
    
    lr_step_cursor = 0
    rr_step_cursor = 0
    
    for track in tracks:

        # find steps that belong to the track
        while lr_step_cursor < len(lr_steps):
            curr_lr_step = lr_steps[lr_step_cursor]
            if intervals_overlap(track, curr_lr_step):
                track.lrp_steps.append(curr_lr_step)

            if curr_lr_step.start_frame >= track.stop_frame_exclu:
                break
            else:
                lr_step_cursor += 1

        while rr_step_cursor < len(rr_steps):
            curr_rr_step = rr_steps[rr_step_cursor]
            if intervals_overlap(track, curr_rr_step):
                track.rrp_steps.append(curr_rr_step)

            if curr_rr_step.start_frame >= track.stop_frame_exclu:
                break
            else:
                rr_step_cursor += 1

        # prev_stride_stop = track.start_frame
        # for step in track.lrp_steps:

        #     # strides will start with the finish of the previous stride (or
        #     # beginning of current track) and finish with the end of the left step
        #     stride_stop = min(step.stop_frame_exclu, track.stop_frame_exclu - 1)
        #     if (step.start_frame >= prev_stride_stop
        #             and step.start_frame < track.stop_frame_exclu):

        #         stride = Stride(
        #             prev_stride_stop,
        #             stride_stop)
        #         track.strides.append(stride)

        #     prev_stride_stop = stride_stop
        
        prev_stride_stop = track.start_frame
        for i, l_step in enumerate(track.lrp_steps):
            stride_stop = min(l_step.stop_frame_exclu, track.stop_frame_exclu - 1)
            if i == len(track.lrp_steps)-1:
                continue
            if (l_step.start_frame >= prev_stride_stop
                and l_step.start_frame < track.stop_frame_exclu):
                for r_step in track.rrp_steps:
                    if (r_step.start_frame > l_step.start_frame
                        and r_step.stop_frame_exclu >= l_step.stop_frame_exclu
                        and r_step.stop_frame_exclu <= track.lrp_steps[i+1].start_frame):
                        stride_stop = min(r_step.stop_frame_exclu, track.stop_frame_exclu)
                        stride = Stride(prev_stride_stop, stride_stop)
                        track.strides.append(stride)
            prev_stride_stop = stride_stop

        yield track
    # return tracks
                
def distance_normalize(sub_df, step=100):
    distance = np.array(np.sqrt(sub_df['x']**2 + sub_df['y']**2), dtype=np.double)
    # d_min = np.min(distance)
    # d_max = np.max(distance)
    # d_norm = (distance - d_min)/(d_max-d_min)
    d_norm = distance/72
    return interpolate(d_norm, step)

def x_normalize(sub_df, step=100):
    # todo: change to normalize by body length? now assumes body length=6cm*12px/cm
    x_vals = np.array(sub_df['x'], dtype=np.double)
    x_norm = x_vals/72
    return interpolate(x_norm, step)


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
    tail_speed = calc_speed(
        np.array(df[[('tail_tip','x'), ('tail_tip','y')]], dtype=np.double), 
        start_index=None, stop_index=None, smoothing_window=1,
        cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    # angular_speed = list(gait.calc_angle_speed_deg(
    #     df[('original', 'angle')], smoothing_window=5))
    # center_speed = calc_center_speed(
    #     df, start_index=None, stop_index=None, smoothing_window=1,
    #     cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    center_speed = calc_speed(
        np.array(df[[('hip','x'), ('hip','y')]], dtype=np.double), 
        start_index=None, stop_index=None, smoothing_window=1,
        cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    
    lr_steps = list(stepdet(
        paw_speeds=l_speed, base_tail_speeds=center_speed, 
        peakdelta_sd=1, approx_still=np.median(center_speed)))

    rr_steps = list(stepdet(
        paw_speeds=r_speed, base_tail_speeds=center_speed, 
        peakdelta_sd=1, approx_still=np.median(center_speed)))
    
    tracks = list(trackstridedet(
                l_speed,
                r_speed,
                center_speed,
                stationary_percentile=25,))
    

    fps = 10
    bodypart = 'tail_tip'
    step = 100
    sum_disp = np.zeros(step)
    stride_count = 0

    all_strides = []
    for track in tracks:
        strides = list(track.strides)
        if len(strides) == 0:
            # print('None')
            pass
        else:
            # change l/r detect logic
            # backtrack 
            for i, stride in enumerate(strides):
                start_frame = stride.start_frame
                stop_frame = stride.stop_frame_exclu
                if stop_frame - start_frame >= fps/2:
                    sub_df = df[[(bodypart, 'x'), (bodypart, 'y')]].iloc[start_frame:stop_frame,:]
                    sub_df.columns = sub_df.columns.droplevel(0)
                    # displacement = distance_normalize(sub_df, step=step)
                    displacement = x_normalize(sub_df, step=step)
                    sum_disp += displacement
                    plt.plot(displacement, color='gray')
                    # sum_disp += x_normalize(sub_df, step=step)
                    stride_count += 1
                    all_strides = all_strides + [stride]

    
    plt.plot(sum_disp/stride_count, color='red')
