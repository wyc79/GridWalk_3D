import affine
import argparse
import h5py
import itertools
import math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import scipy.stats

from borrowed import *

# paw_speeds -> pt_spds
# base_tail_speeds -> mv_spds


CM_PER_PIXEL = 1/12

def stepdet(pt_spds, mv_spds, peakdelta=10, approx_still=1):
    """
    generator which detects step events for a single paw
    """

    speed_maxs, speed_mins = peakdet(pt_spds, peakdelta)
    speed_maxs = speed_maxs[:, 0].astype(np.int32)
    speed_mins = speed_mins[:, 0].astype(np.int32)

    for i, speed_max_frame in enumerate(speed_maxs):

        # print('speed_max_frame:', speed_max_frame)
        toe_off_index = speed_max_frame
        while (toe_off_index > 0
                and pt_spds[toe_off_index] > approx_still
                and pt_spds[toe_off_index] > mv_spds[toe_off_index]):
            toe_off_index -= 1

        # we may need to step forward one frame
        if pt_spds[toe_off_index] <= approx_still and toe_off_index < len(pt_spds) - 1:
            toe_off_index += 1

        # if we stepped past the previous local min we should adjust the toe-off index
        if i > 0 and i < len(speed_mins):
            prev_speed_min_frame = speed_mins[i - 1]
            if prev_speed_min_frame > toe_off_index:
                toe_off_index = prev_speed_min_frame + 1

        strike_index = speed_max_frame
        while (strike_index < len(pt_spds) - 1
                and pt_spds[strike_index] > approx_still
                and pt_spds[strike_index] > mv_spds[strike_index]):
            strike_index += 1

        # if we stepped past the next local min we should adjust the strike index
        if i >= 0 and i < len(speed_mins):
            next_speed_min_frame = speed_mins[i]
            if next_speed_min_frame < strike_index:
                strike_index = next_speed_min_frame

        if strike_index > toe_off_index:
            yield FrameInterval(toe_off_index, strike_index)

def trackdet(mv_spds, speed_thresh=5):
    """
    Detect "track" intervals for the given `base_tail_speeds`
    """
    speed_over_thresh = mv_spds >= speed_thresh
    grp_frame_index = 0
    for grp_key, grp_vals in itertools.groupby(speed_over_thresh):
        grp_count = len(list(grp_vals))
        if grp_key:
            yield Track(grp_frame_index, grp_frame_index + grp_count)

        grp_frame_index += grp_count


def trackstridedet(lr_paw_speeds, rr_paw_speeds, mv_spds, angular_velocities, cm_per_px=CM_PER_PIXEL):
    """
    This function will detect tracks along with the strides that belong to those tracks
    """
    lr_steps = list(stepdet(lr_paw_speeds, mv_spds))
    rr_steps = list(stepdet(rr_paw_speeds, mv_spds))
    tracks = trackdet(mv_spds)

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

        # now that steps have been associated with the current track we need
        # to associate step pairs as strides. We start with the left rear paw
        # steps.
        prev_stride_stop = track.start_frame
        for step in track.lrp_steps:

            # strides will start with the finish of the previous stride (or
            # beginning of current track) and finish with the end of the left step
            stride_stop = min(step.stop_frame_exclu, track.stop_frame_exclu - 1)
            if (step.start_frame >= prev_stride_stop
                    and step.start_frame < track.stop_frame_exclu):

                speed_cm_per_sec = np.mean(
                    mv_spds[prev_stride_stop : stride_stop + 1])
                angular_velocity = np.mean(
                    angular_velocities[prev_stride_stop : stride_stop + 1])
                stride = Stride(
                    prev_stride_stop,
                    stride_stop + 1,
                    speed_cm_per_sec,
                    angular_velocity,
                    cm_per_px=cm_per_px)
                track.strides.append(stride)

            prev_stride_stop = stride_stop

        # now we assiciate the right rear paw step with the stride
        for stride in track.strides:
            for step in track.rrp_steps:
                if step.stop_frame_exclu > stride.start_frame:
                    if step.stop_frame_exclu < stride.stop_frame_exclu:
                        stride.rr_paw_strike_frame = step.stop_frame_exclu
                    break

            stride.lr_duty_factor = duty_factor(stride, track.lrp_steps)
            stride.rr_duty_factor = duty_factor(stride, track.rrp_steps)

        yield track

