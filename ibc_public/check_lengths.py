'''
This script checks and compares the lengths of various files associated with the ArchiStandard task for subject 14, session 00, direction 'ap'.
I wanted to make sure that the timeseries, motion parameters, and event files are consistent in length.
'''

import os
import numpy as np
import pandas as pd
import nibabel as nib
from math import ceil, floor

subject_id = "04"
session_id = "00"
task_name = "ArchiStandard"
direction = "ap"

DT_PATH = f"/ptmp/hmueller2/Downloads/fmriprep_out/sub-{subject_id}/ses-{session_id}/postfmriprep/GLM/sub-{subject_id}_ses-{session_id}_task-{task_name}_dir-{direction}_cleaned_noscrub.dtseries.nii"
MOT_PATH = f"/ptmp/hmueller2/Downloads/fmriprep_out/sub-{subject_id}/ses-{session_id}/postfmriprep/regressors/sub-{subject_id}_ses-{session_id}_task-{task_name}_dir-{direction}_motion.txt"
EV_PATH = f"/ptmp/hmueller2/Downloads/ibc_raw/sub-{subject_id}/ses-{session_id}/func/sub-{subject_id}_ses-{session_id}_task-{task_name}_dir-{direction}_events.tsv"

def cifti_tr_and_scans(path):
    img = nib.load(path)
    ax0 = img.header.get_axis(0)
    ax1 = img.header.get_axis(1)
    ts_axis = ax0 if isinstance(ax0, nib.cifti2.SeriesAxis) else ax1 if isinstance(ax1, nib.cifti2.SeriesAxis) else None
    tr = float(getattr(ts_axis, "step", 2.0)) if ts_axis is not None else 2.0
    n_scans = ts_axis.size if ts_axis is not None else img.shape[0]
    return tr, n_scans

def motion_rows(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        rows = arr.size // 6 if arr.size % 6 == 0 else arr.size
    else:
        rows = arr.shape[0]
    return rows

def events_info(path):
    df = pd.read_csv(path, sep="\t")
    onset = df["onset"].to_numpy(dtype=float)
    duration = df["duration"].to_numpy(dtype=float) if "duration" in df.columns else np.zeros_like(onset)
    ev_min = float(np.nanmin(onset)) if onset.size else np.nan
    ev_max_end = float(np.nanmax(onset + duration)) if onset.size else np.nan
    return len(df), ev_min, ev_max_end

def main():
    print("Checking ArchiStandard AP lengths for sub-14...")
    for p in (DT_PATH, MOT_PATH, EV_PATH):
        print(f"  exists({p}): {os.path.exists(p)}")
    if not all(map(os.path.exists, (DT_PATH, MOT_PATH, EV_PATH))):
        return

    tr, n_scans = cifti_tr_and_scans(DT_PATH)
    mot_rows = motion_rows(MOT_PATH)
    ev_rows, ev_min, ev_max_end = events_info(EV_PATH)

    est_scans_floor = floor(ev_max_end / tr) if np.isfinite(ev_max_end) else np.nan
    est_scans_ceil = ceil(ev_max_end / tr) if np.isfinite(ev_max_end) else np.nan
    est_scans_round = int(round(ev_max_end / tr)) if np.isfinite(ev_max_end) else np.nan

    print("\nSummary")
    print("- dtseries:") #print(f"- dtseries: {DT_PATH}")
    print(f"  TR: {tr:.6f} s, timepoints: {n_scans}")
    print("- motion:") #print(f"- motion:   {MOT_PATH}")
    print(f"  rows: {mot_rows}")
    print("- events") #print(f"- events:   {EV_PATH}")
    print(f"  rows: {ev_rows}, min onset: {ev_min:.3f}s, max end: {ev_max_end:.3f}s")
    print(f"  est scans from events (floor/round/ceil): {est_scans_floor} / {est_scans_round} / {est_scans_ceil}")
    print("\nDiffs (motion - dtseries):", mot_rows - n_scans)
    if np.isfinite(ev_max_end):
        print("Diffs (events-derived - dtseries):")
        print(f"  floor: {est_scans_floor - n_scans}, round: {est_scans_round - n_scans}, ceil: {est_scans_ceil - n_scans}")

if __name__ == "__main__":
    main()