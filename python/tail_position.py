import os
import math
import glob
import itertools
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from pyestimate import sin_param_estimate

import scipy.fftpack as fftpack


FRAMES_PER_SECOND = 10
CM_PER_PIXEL = 19.5 * 2.54 / 400
TRACK_MIN = 50

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


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


def calc_speed(xy_pos, start_index=None, stop_index=None, smoothing_window=1, 
               cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND):
    # xy_pos = group['points'][start_index : stop_index, point_index, :].astype(np.double)
    xy_pos = np.array(xy_pos[start_index : stop_index, : ], dtype=float)
    xy_pos[:, 0] = _smooth(xy_pos[:, 0], smoothing_window)
    xy_pos[:, 1] = _smooth(xy_pos[:, 1], smoothing_window)

    xy_pos *= cm_per_px
    velocity = np.gradient(xy_pos, axis=0)
    speed_cm_per_sec = np.linalg.norm(velocity, axis=1) * fps
    return speed_cm_per_sec


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
    

def trackdet(base_tail_speeds, speed_thresh=5, fps=FRAMES_PER_SECOND):
    """
    Detect "track" intervals for the given `base_tail_speeds`
    """
    speed_over_thresh = base_tail_speeds >= speed_thresh
    grp_frame_index = 0
    for grp_key, grp_vals in itertools.groupby(speed_over_thresh):
        grp_count = len(list(grp_vals))
        if (grp_key) and (grp_count >= fps):
            yield FrameInterval(grp_frame_index, grp_frame_index + grp_count)
        grp_frame_index += grp_count


def tail_ang_filter(df, point1, point2, angle_thresh=150):
    """
    generate a mask based on angles: 1 if angle > 150; 0 otherwise
    to ensure the body is relatively straight
    """
    point1_coords = np.array(df[[(point1,'x'), (point1,'y')]], dtype=np.double)
    point2_coords = np.array(df[[(point2,'x'), (point2,'y')]], dtype=np.double)
    angles = [90-math.atan2(p2[1]-p1[1], p2[0]-p1[0])/math.pi*180 
              for p1, p2 in zip(point1_coords, point2_coords)]
    angles = [360-ang if ang>180 else ang for ang in angles]
    return np.array(angles) > angle_thresh

def perpendicular_distance(point, line_point1, line_point2):
    x1 = point[:,0]; y1 = point[:,1]
    x2 = line_point1[:,0]; y2 = line_point1[:,1]
    x3 = line_point2[:,0]; y3 = line_point2[:,1]
    numerator = (y3-y2)*x1-(x3-x2)*y1+x3*y2-y3*x2
    denominator = ((y3 - y2)**2 + (x3 - x2)**2)**0.5
    return numerator / denominator

def perpendicular_distance_rotate(point, line_point1, line_point2):
    mid_point_x = (line_point1[:,0] + line_point2[:,0]) / 2
    mid_point_y = (line_point1[:,1] + line_point2[:,1]) / 2

    point[:,0] = point[:,0] - mid_point_x
    point[:,1] = point[:,1] - mid_point_y

    def rotate_point(x, y, angle):
        angle_rad = np.radians(angle)
        rotated_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        rotated_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        return rotated_x, rotated_y
    dx = line_point1[:,0] - line_point2[:,0]
    dy = line_point1[:,1] - line_point2[:,1]
    angle = np.degrees(np.arctan2(-dx, dy))

    # result = np.zeros_like(point)
    result, _ = rotate_point(point[:,0], point[:,1], -angle)

    return result

bins = np.linspace(-3,3, 100)
def normalized_displacement(file):
    df = pd.read_csv(file, header=None, skiprows=3)
    header = pd.read_csv(file, header=None, nrows=3)
    multi_level_header = pd.MultiIndex.from_arrays(header.values)
    df.columns = multi_level_header
    df.columns = df.columns.droplevel(0)

    center_speed = calc_speed(
        np.array(df[[('original_mid_spine','x'), ('original_mid_spine','y')]], dtype=np.double), 
        start_index=None, stop_index=None, smoothing_window=1,
        cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    angle_mask = tail_ang_filter(df, 'mid_spine', 'hip', angle_thresh=150)
    # angle_mask = tail_ang_filter(df, 'hip', 'mid_tail', angle_thresh=150)
    # angle_mask = np.ones_like(center_speed)
    
    tracks = trackdet(
        center_speed, fps=FRAMES_PER_SECOND,
        speed_thresh=np.percentile(center_speed, 20))
    
    # track_start_end_points = []
    track_mask = np.zeros(df.shape[0], dtype=bool)
    for track in tracks:
        start = track.start_frame
        end = track.stop_frame_exclu
        track_len = end-start
        if (np.sum(angle_mask[start:end]) > track_len * 0.8) and\
            (end-start>TRACK_MIN):
            track_mask[start:end] = True
            # print(f"{start} - {end}: {end-start} frs")
            # track_start_end_points += [(start, end)]

    track_df = df[track_mask]

    diff_thigh = np.array(df[[('L_thigh', 'x'), ('L_thigh', 'y')]], dtype=np.double) - \
        np.array(df[[('R_thigh', 'x'), ('R_thigh', 'y')]], dtype=np.double)
    mean_thigh_width = np.nanmean(np.sqrt(diff_thigh[:,0]**2 + diff_thigh[:,1]**2))
    # norm_x = track_df[(poi, 'x')]/mean_thigh_width
    # abs_norm_x = np.abs(norm_x)

    x_hip = track_df[('hip', 'x')]/mean_thigh_width
    x_tailtip = track_df[('tail_tip', 'x')]/mean_thigh_width
    x_midtail = track_df[('mid_tail', 'x')]/mean_thigh_width

    ppd_tailtip = perpendicular_distance_rotate(
        np.array(track_df[[("tail_tip", 'x'), ("tail_tip", 'y')]], dtype=np.double), 
        np.array(track_df[[('mid_spine', 'x'), ('mid_spine', 'y')]], dtype=np.double), 
        np.array(track_df[[('hip', 'x'), ('hip', 'y')]], dtype=np.double))
    ppd_tailtip = ppd_tailtip / mean_thigh_width

    ppd_midtail = perpendicular_distance_rotate(
        np.array(track_df[[("mid_tail", 'x'), ("mid_tail", 'y')]], dtype=np.double), 
        np.array(track_df[[('mid_spine', 'x'), ('mid_spine', 'y')]], dtype=np.double), 
        np.array(track_df[[('hip', 'x'), ('hip', 'y')]], dtype=np.double))
    ppd_midtail = ppd_midtail / mean_thigh_width

    sho2thg_L = np.sqrt(
        (track_df[('L_shoulder', 'x')] - track_df[('L_thigh', 'x')])**2 + \
        (track_df[('L_shoulder', 'y')] - track_df[('L_thigh', 'y')])**2
        )/mean_thigh_width
    sho2thg_R = np.sqrt(
        (track_df[('R_shoulder', 'x')] - track_df[('R_thigh', 'x')])**2 + \
        (track_df[('R_shoulder', 'y')] - track_df[('R_thigh', 'y')])**2
        )/mean_thigh_width
    
    hip_ang = []
    for a,b,c in zip(
        np.array(track_df[[('mid_spine', 'x'), ('mid_spine', 'y')]]), 
        np.array(track_df[[('hip', 'x'), ('hip', 'y')]]), 
        np.array(track_df[[('mid_tail', 'x'), ('mid_tail', 'y')]])):
        hip_ang.append(angle_between_three_points(a,b,c))
    
    sho_ang = []
    for a,b,c in zip(
        np.array(track_df[[('L_shoulder', 'x'), ('L_shoulder', 'y')]]), 
        np.array(track_df[[('mid_spine', 'x'), ('mid_spine', 'y')]]), 
        np.array(track_df[[('R_shoulder', 'x'), ('R_shoulder', 'y')]])):
        sho_ang.append(angle_between_three_points(a,b,c))
    
    tail_ang = []
    for a,b,c in zip(
        np.array(track_df[[('hip', 'x'), ('hip', 'y')]]), 
        np.array(track_df[[('mid_tail', 'x'), ('mid_tail', 'y')]]), 
        np.array(track_df[[('tail_tip', 'x'), ('tail_tip', 'y')]])):
        tail_ang.append(angle_between_three_points(a,b,c))

    counts, _ = np.histogram(x_hip, bins=bins)
    # print(file)
    disp_df = pd.DataFrame({
        "time_points": np.array(df.iloc[:,0], dtype=int)[track_mask],
        "x_hip": np.array(x_hip, dtype=np.double),
        "x_tailtip": np.array(x_tailtip, dtype=np.double),
        "x_midtail": np.array(x_midtail, dtype=np.double),
        "ppd_tailtip": np.array(ppd_tailtip, dtype=np.double),
        "ppd_midtail": np.array(ppd_midtail, dtype=np.double),
        "thigh_width": np.array([mean_thigh_width]*len(x_hip), dtype=np.double),
        "sho2thg_L": np.array(sho2thg_L, dtype=np.double),
        "sho2thg_R": np.array(sho2thg_R, dtype=np.double),
        "sho_ang": np.array(sho_ang, dtype=np.double),
        "hip_ang": np.array(hip_ang, dtype=np.double),
        "tail_ang": np.array(tail_ang, dtype=np.double),
    })
    count_df = pd.DataFrame({
        "bin_center": (bins[1:] + bins[:-1])/2,
        "counts": 0 if np.sum(track_mask)==0 else counts/np.sum(track_mask)
    })
    meta_df = pd.DataFrame({
        "selected_timepoints": [0 if np.sum(track_mask)==0 else np.sum(track_mask)],
        "n_timepoints": [len(track_mask)],
        "selection_ratio": [np.sum(track_mask)/len(track_mask)],
        "thigh_width": [mean_thigh_width],
    })
    return disp_df, count_df, meta_df

def angle_between_three_points(A, B, C):
    # Convert points to numpy arrays
    A, B, C = np.array(A), np.array(B), np.array(C)

    if (A == B).all() or (B == C).all() or (A == C).all():
        return 0
    
    # Vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Dot product and magnitudes of vectors
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    # Calculate the angle in radians
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    theta = np.arccos(cos_theta)  # Angle in radians
    
    # Convert radians to degrees
    theta_degrees = np.degrees(theta)
    return theta_degrees

def plot_compare(disp_df_all, res_row, cycle_location, fs):
    start = int(res_row["start"])
    end = int(res_row["end"])
    amp = res_row["amplitude"]
    freq = res_row["frequency"]
    phase = res_row["phase"]
    print(f"amp:{amp}, freq:{freq}, phase:{phase}")
    sig = disp_df_all[cycle_location][start:end]
    ts = np.arange(len(sig))
    fig, ax = plt.subplots(1,1, figsize=[10, 4], constrained_layout=True)
    ax.plot(ts/fs, sig, label=cycle_location)
    # ax.plot(ts/fs, amp*np.sin(freq*ts+phase), 'r--', label='fitted sine')
    ax.plot(ts/fs, amp*np.cos(2*np.pi*freq*ts+phase), 'r--', label='fitted sine')
    ax.legend()

if __name__ == "__main__":

    # file = \
    #     '../GridWalking-wyc-2023-10-30/new_analyzed/AD/3-4M/AD688_2023-09-11_11-06-19DLC_resnet50_GridWalkingDec12shuffle1_50000_filtered_transformed.csv'
    
    data_root = "../GridWalking-wyc-2023-10-30/new_analyzed/"

    disp_df_all = pd.DataFrame()
    count_df_all = pd.DataFrame()
    meta_df_all = pd.DataFrame()
    
    for cond_root in glob.glob(f"{data_root}/*"):
        cond = cond_root.split('/')[-1]
        for month_root in glob.glob(f"{cond_root}/*"):
            mon = month_root.split('/')[-1]
            # counts_all[(cond, mon)] = []
            query = f"{month_root}/*_50000_filtered_transformed.csv"
            files = glob.glob(query)
            n_files = len(files)
            print(f"Query '{query}' return {n_files} results")
            # if n_files == 0: return None
            for file in files:
                disp_df, count_df, meta_df = normalized_displacement(file)
                disp_df["condition"] = cond
                disp_df["month"] = mon
                disp_df["filename"] = file.split('/')[-1]
                disp_df_all = pd.concat([disp_df_all, disp_df])
                disp_df_all = disp_df_all.reset_index(drop=True)
                
                count_df["condition"] = cond
                count_df["month"] = mon
                count_df["filename"] = file.split('/')[-1]
                count_df_all = pd.concat([count_df_all, count_df])
                count_df_all = count_df_all.reset_index(drop=True)

                meta_df["condition"] = cond
                meta_df["month"] = mon
                meta_df["filename"] = file.split('/')[-1]
                meta_df_all = pd.concat([meta_df_all, meta_df])
                meta_df_all = meta_df_all.reset_index(drop=True)
            #     counts_all[(cond, mon)] = counts_all[(cond, mon)] + [counts]
            # counts_all[(cond, mon)] = np.array(counts_all[(cond, mon)])

    disp_df_all.to_csv(f"gait/horizontal_displacement.csv")
    count_df_all.to_csv(f"gait/histogram_counts.csv")
    meta_df_all.to_csv(f"gait/meta.csv")

    print("Start calculating sin parameters ......")

    cycle_location = "x_midtail"
    
    l = TRACK_MIN
    count = 0
    i = 0
    tolerance = 5
    start_end_points = []
    while i < disp_df_all.shape[0]:
        start = i
        end = i + l
        if end < disp_df_all.shape[0] and \
            (disp_df_all["time_points"][end] - disp_df_all["time_points"][start] <= l + tolerance) and \
            (disp_df_all["time_points"][end] - disp_df_all["time_points"][start] > 0) and\
            (disp_df_all["filename"][end] == disp_df_all["filename"][start]):
            count += 1
            start_end_points += [(start, end)]
            i = end
        else:
            i += 1
    
    print("Signal segmented, estimating sin parameters ......")
    
    fit_df = pd.DataFrame()
    avg_df = pd.DataFrame()
    for i, (start, end) in tqdm(enumerate(start_end_points)):
        # ts = disp_df_all["time_points"][start:end]
        # plt.plot(ts, disp_df_all["x_hip"][start:end], label="x_hip")
        sub_df = disp_df_all[start:end].reset_index(drop=True)
        amp,f,phi = sin_param_estimate(
            sub_df[cycle_location], use_fft=False, 
            brute_Ns=10000, detrend_type='linear')
        freq = 2*np.pi*f
        # fname = disp_df_all["filename"][start]
        phase = phi+np.pi/2
        if phase > np.pi:
            phase = phase - np.pi
        fit_row = pd.DataFrame({
            "start":start, "end":end, 
            "amplitude":amp, "frequency":f, "phase":phi,
            "filename": sub_df["filename"][0]
        }, index=[i])
        fit_df = pd.concat([fit_df, fit_row])
        avg_vals = sub_df.mean(numeric_only=True)
        avg_row = pd.DataFrame({
            "filename": sub_df["filename"][0],
            "condition": sub_df["condition"][0],
            "month": sub_df["month"][0],
            "len_track": len(sub_df),
            "mean_sho2thg_L": avg_vals["sho2thg_L"],
            "mean_sho2thg_R": avg_vals["sho2thg_R"],
            "mean_sho_ang": avg_vals["sho_ang"],
            "mean_hip_ang": avg_vals["hip_ang"],   
            "mean_tail_ang": avg_vals["tail_ang"],   
            "mean_thigh_width": avg_vals["thigh_width"]
        }, index=[i])
        avg_df = pd.concat([avg_df, avg_row])
    fit_df.to_csv(f"gait/sin_params_{cycle_location}.csv")
    avg_df.to_csv(f"gait/tracks_vals.csv")

    avg_res_df = fit_df.groupby("filename").mean()
    avg_res_df = avg_res_df.merge(meta_df_all, how="inner", left_index=True, right_on="filename")
    avg_res_df = avg_res_df.drop(["start", "end"], axis=1)

    avg_res_df.to_csv(f"gait/avg_sin_params_{cycle_location}.csv")

    plot_compare(disp_df_all, fit_df.iloc[1,:], cycle_location, 10)
    

        
# (disp_df_all["filename"][end] == disp_df_all["filename"][start]):
    # amplitude, frequency, phase = sin_param_estimate(
    #     disp_df_all[cycle_location][start:end], use_fft=False, 
    #     brute_Ns=10000, detrend_type='linear')

    # plot_compare(disp_df_all, pd.DataFrame({"start":start, "end":end, 
    #     "amplitude":amplitude, "frequency":frequency, "phase":phase}, index=[0]), cycle_location, 10)


    # start = start_end_points[0][0]
    # end = start_end_points[0][1]
    
    # ts = disp_df_all["time_points"][start:end]
    
    # A,f,phi = sin_param_estimate(
    #     disp_df_all["x_hip"][start:end], use_fft=True, detrend_type='constant')
    # plt.plot(ts/10, disp_df_all["x_hip"][start:end], label="x_hip")
    # plt.plot(ts/10, A*np.sin(2*np.pi*f*ts+phi+np.pi/2), 'r--', label='fitted sine')

    # plt.legend()
        



    
    
    
    
    
    
    # df = pd.read_csv(file, header=None, skiprows=3)
    # header = pd.read_csv(file, header=None, nrows=3)
    # multi_level_header = pd.MultiIndex.from_arrays(header.values)
    # df.columns = multi_level_header
    # df.columns = df.columns.droplevel(0)

    # center_speed = calc_speed(
    #     np.array(df[[('hip','x'), ('hip','y')]], dtype=np.double), 
    #     start_index=None, stop_index=None, smoothing_window=1,
    #     cm_per_px=CM_PER_PIXEL, fps=FRAMES_PER_SECOND)
    # angel_mask = tail_ang_filter(df, 'hip', 'mid_tail', angle_thresh=150)
    
    # tracks = trackdet(
    #     center_speed, fps=FRAMES_PER_SECOND,
    #     speed_thresh=np.percentile(center_speed, 20))
    
    # track_mask = np.zeros(df.shape[0], dtype=bool)
    # for track in tracks:
    #     start = track.start_frame
    #     end = track.stop_frame_exclu
    #     track_len = end-start
    #     if np.sum(angel_mask[start:end]) > track_len * 0.8:
    #         track_mask[start:end] = True
    #         print(f"{start} - {end}: {end-start} frs")

    # track_df = df[track_mask]

    # diff_thigh = np.array(track_df[[('L_thigh', 'x'), ('L_thigh', 'y')]]) - \
    #     np.array(track_df[[('R_thigh', 'x'), ('R_thigh', 'y')]])
    # mean_thigh_width = np.mean(np.sqrt(diff_thigh[:,0]**2 + diff_thigh[:,1]**2))

    # norm_disp = np.abs(track_df[('tail_tip', 'x')])/mean_thigh_width

    

    

    