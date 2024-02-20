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



NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

# we will reject any points with a lower confidence score
# MIN_CONF_THRESH = 200
# MIN_CONF_THRESH = 75
MIN_CONF_THRESH = 0.3

# the maximum length of a segment that we tolerate for good frames
# (eg. mid spine to base neck or right rear paw to base of tail)
MAX_SEGMENT_LEN_THRESH = 5

# TODO should I use timestamps instead of FRAMES_PER_SECOND?
FRAMES_PER_SECOND = 30
CM_PER_PIXEL = 19.5 * 2.54 / 400

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
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab).reshape(-1, 2), np.array(mintab).reshape(-1, 2)


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
    
### change stride definition
    
class Stride(FrameInterval):
    """
    A stride interval which is deliniated by foot strike events of the left rear paw
    """

    # pylint: disable=unsubscriptable-object

    def __init__(self, start_frame, stop_frame_exclu, speed_cm_per_sec, angular_velocity, cm_per_px=CM_PER_PIXEL):
        super().__init__(start_frame, stop_frame_exclu)
        self.speed_cm_per_sec = speed_cm_per_sec
        self.angular_velocity = angular_velocity
        self.cm_per_px = cm_per_px


def intervals_overlap(inter1, inter2):
    return (
        inter1.start_frame < inter2.stop_frame_exclu
        and inter1.stop_frame_exclu > inter2.start_frame
    )


def comp_inter_start(inter1, inter2):
    return inter1.start_frame - inter2.start_frame


class Track(FrameInterval):

    def __init__(self, start_frame, stop_frame_exclu):
        super().__init__(start_frame, stop_frame_exclu)

        self.lrp_steps = []
        self.rrp_steps = []
        self.strides = []

        self.confidence = 0

    def _stances(self, steps):
        prev_step_stop_exclu = self.start_frame

        for step in steps:
            if step.start_frame > prev_step_stop_exclu:
                yield FrameInterval(prev_step_stop_exclu, step.start_frame)
                prev_step_stop_exclu = step.start_frame

        if prev_step_stop_exclu < self.stop_frame_exclu:
            yield FrameInterval(prev_step_stop_exclu, self.stop_frame_exclu)

    @property
    def lrp_stances(self):
        return self._stances(self.lrp_steps)

    @property
    def rrp_stances(self):
        return self._stances(self.rrp_steps)

    @property
    def inner_strides(self):
        return self.strides[1:-1]

    # rewrote stride logic
    # @property
    # def good_strides(self):
    #     return [s for s in self.inner_strides if s.is_good]


def stepdet(paw_speeds, base_tail_speeds, peakdelta=10, approx_still=15):
    """
    generator which detects step events for a single paw
    """

    speed_maxs, speed_mins = peakdet(paw_speeds, peakdelta)
    speed_maxs = speed_maxs[:, 0].astype(np.int32)
    speed_mins = speed_mins[:, 0].astype(np.int32)

    for i, speed_max_frame in enumerate(speed_maxs):

        # print('speed_max_frame:', speed_max_frame)
        toe_off_index = speed_max_frame
        while (toe_off_index > 0
                and paw_speeds[toe_off_index] > approx_still
                and paw_speeds[toe_off_index] > base_tail_speeds[toe_off_index]):
            toe_off_index -= 1

        # we may need to step forward one frame
        if paw_speeds[toe_off_index] <= approx_still and toe_off_index < len(paw_speeds) - 1:
            toe_off_index += 1

        # if we stepped past the previous local min we should adjust the toe-off index
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
            yield FrameInterval(toe_off_index, strike_index)


def trackdet(base_tail_speeds, speed_thresh=5):
    """
    Detect "track" intervals for the given `base_tail_speeds`
    """
    speed_over_thresh = base_tail_speeds >= speed_thresh
    grp_frame_index = 0
    for grp_key, grp_vals in itertools.groupby(speed_over_thresh):
        grp_count = len(list(grp_vals))
        if grp_key:
            yield Track(grp_frame_index, grp_frame_index + grp_count)

        grp_frame_index += grp_count

def trackstridedet(lr_paw_speeds, rr_paw_speeds, base_tail_speeds, angular_velocities, 
                   cm_per_px=CM_PER_PIXEL, stationary_percentile=30, approx_still=5):
    """
    This function will detect tracks along with the strides that belong to those tracks
    """
    lr_steps = list(stepdet(lr_paw_speeds, base_tail_speeds, peakdelta=0.5, approx_still=approx_still))
    rr_steps = list(stepdet(rr_paw_speeds, base_tail_speeds, peakdelta=0.5, approx_still=approx_still))
    tracks = trackdet(base_tail_speeds, 
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
                    base_tail_speeds[prev_stride_stop : stride_stop + 1])
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
        ####### !!!!!!!!!!!!!
        for stride in track.strides:
            for step in track.rrp_steps:
                if step.stop_frame_exclu > stride.start_frame:
                    if step.stop_frame_exclu < stride.stop_frame_exclu:
                        stride.rr_paw_strike_frame = step.stop_frame_exclu
                    break

            # stride.lr_duty_factor = duty_factor(stride, track.lrp_steps)
            # stride.rr_duty_factor = duty_factor(stride, track.rrp_steps)

        yield track
