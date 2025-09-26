import sys
import os
import glob
import nibabel as nib
import numpy as np
from ibc_public.utils_pipeline import first_level

subject = sys.argv[1]
base_dir = '/ptmp/hmueller2/Downloads/fmriprep_out'
output_base = '/ptmp/hmueller2/Downloads/contrast_maps_fsLR'

subject_dir = os.path.join(base_dir, f'sub-{subject}')
session_dirs = sorted(glob.glob(os.path.join(subject_dir, 'ses-*')))

processed = []
skipped = []
seen_ids = set()

for session_dir in session_dirs:
    session = os.path.basename(session_dir)
    glm_dir = os.path.join(session_dir, 'postfmriprep', 'GLM')
    if not os.path.exists(glm_dir):
        continue

    func_files = sorted(glob.glob(os.path.join(glm_dir, f'sub-{subject}_{session}_task-*_dir-*_*cleaned_noscrub.dtseries.nii')))
    if not func_files:
        print(f"No functional files found for {subject} {session}")
        continue

    for func_path in func_files:
        fname = os.path.basename(func_path)
        parts = fname.split('_')
        task = [p.split('-')[1] for p in parts if p.startswith('task-')]
        direction = [p.split('-')[1] for p in parts if p.startswith('dir-')]
        run = [p.split('-')[1] for p in parts if p.startswith('run-')]
        task = task[0] if task else 'unknown'
        direction = direction[0] if direction else 'unknown'
        run = run[0] if run else None  # None if not present

        # Extract TR from CIFTI SeriesAxis (robust)
        try:
            img = nib.load(func_path)
            ax0 = img.header.get_axis(0)
            ax1 = img.header.get_axis(1)
            ts_axis = ax0 if isinstance(ax0, nib.cifti2.SeriesAxis) else ax1
            tr = float(getattr(ts_axis, "step", 2.0))
        except Exception as e:
            print(f"Could not read TR from {func_path}: {e}")
            tr = 2.0

        # Build file names depending on run presence
        if run:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_events.tsv'
            session_id = f'task-{task}_run-{run}_dir-{direction}'
        else:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_events.tsv'
            session_id = f'task-{task}_dir-{direction}'

        dedup_key = (session_id, os.path.basename(func_path))
        if dedup_key in seen_ids:
            skipped.append((session_id, "duplicate_run"))
            continue
        seen_ids.add(dedup_key)

        # Motion file path
        motion_path = os.path.join(session_dir, 'postfmriprep', 'regressors', motion_fname)
        if not os.path.exists(motion_path):
            motion_path = None

        # Onset/events file path
        onset_path = os.path.join('/ptmp/hmueller2/Downloads/ibc_raw', f'sub-{subject}', session, 'func', onset_fname)
        if not os.path.exists(onset_path):
            onset_path = None

        if not onset_path or not motion_path:
            print(f"Skipping run {session_id} for subject {subject} due to missing files.")
            skipped.append((session_id, "missing_motion_or_events"))
            continue

        anat_path = os.path.join(session_dir, 'anat', f'sub-{subject}_{session}_T1w.nii.gz')
        output_dir = os.path.join(output_base, f'sub-{subject}', session)  # keep utils' subfolder structure

        # Ensure motion regressors match n_scans from SeriesAxis
        try:
            # Determine number of scans (time points) from SeriesAxis
            if isinstance(ax0, nib.cifti2.SeriesAxis):
                n_scans = ax0.size
            elif isinstance(ax1, nib.cifti2.SeriesAxis):
                n_scans = ax1.size
            else:
                n_scans = img.shape[0]

            # Load motion and check shape
            motion = np.loadtxt(motion_path)
            if motion.ndim == 1:
                if motion.size % 6 == 0:
                    motion = motion.reshape(-1, 6)
                else:
                    motion = motion.reshape(-1, 1)
            diff = motion.shape[0] - n_scans
            if diff == 0:
                pass
            elif abs(diff) == 1:
                # allow off-by-one, pad or trim
                if diff == 1:
                    motion = motion[-n_scans:, :]
                else:
                    motion = np.pad(motion, ((1, 0), (0, 0)), mode='constant', constant_values=0.0)
            elif abs(diff) <= 5:
                # New: tolerate small mismatches (e.g. missing initial discarded volumes)
                if diff > 0:
                    # Trim extra rows from start
                    motion = motion[diff:, :]
                else:
                    # Pad at start with zeros
                    motion = np.pad(motion, ((abs(diff), 0), (0, 0)), mode='constant', constant_values=0.0)
                print(f"Adjusted motion length for {session_id}: original {motion.shape[0]-diff} -> {motion.shape[0]} (target {n_scans})")
            else:
                print(f"Skipping run {session_id}: motion rows {motion.shape[0]} != scans {n_scans} (diff {diff})")
                skipped.append((session_id, f"motion_len_mismatch_{motion.shape[0]}_{n_scans}"))
                continue
            motion = motion.astype(np.float32)
        except Exception as e:
            print(f"Error processing motion file {motion_path}: {e}")
            skipped.append((session_id, "motion_processing_error"))
            continue

        subject_dic = {
            'session_id': [session_id],
            'func': [func_path],
            'onset': [onset_path],
            'realignment_parameters': [motion],
            'output_dir': output_dir,
            'anat': anat_path,
            'hrf_model': 'spm',
            'high_pass': 0.01,
            'drift_model': 'cosine',
            'TR': tr
        }

        try:
            first_level(subject_dic, mesh='fsLR')
            processed.append(session_id)
        except Exception as e:
            print(f"Run {session_id} failed: {e}")
            skipped.append((session_id, "first_level_exception"))

# Summary
print(f"Subject {subject} summary:")
print(f"  Processed runs: {len(processed)}")
print(f"  Skipped runs  : {len(skipped)}")
if skipped:
    by_reason = {}
    for sid, reason in skipped:
        by_reason.setdefault(reason, 0)
        by_reason[reason] += 1
    print("  Skip reasons:")
    for r, c in sorted(by_reason.items(), key=lambda x: (-x[1], x[0])):
        print(f"    {r}: {c}")
# (No renaming or moving of outputs here; utils handle standard layout)