import sys
import os
import glob
import nibabel as nib
import numpy as np
import shutil
from ibc_public.utils_pipeline import first_level

subject = sys.argv[1]
base_dir = '/ptmp/hmueller2/Downloads/fmriprep_out'
output_base = '/ptmp/hmueller2/Downloads/contrast_maps_fsLR'

subject_dir = os.path.join(base_dir, f'sub-{subject}')
session_dirs = sorted(glob.glob(os.path.join(subject_dir, 'ses-*')))

for session_dir in session_dirs:
    session = os.path.basename(session_dir)
    glm_dir = os.path.join(session_dir, 'postfmriprep', 'GLM')
    if not os.path.exists(glm_dir):
        continue

    func_files = sorted(glob.glob(os.path.join(glm_dir, f'sub-{subject}_{session}_task-*_dir-*_*cleaned.dtseries.nii')))
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

        # Extract TR for this run
        try:
            img = nib.load(func_path)
            ts_axis = img.header.get_axis(1)
            tr = getattr(ts_axis, "step", 2.0)
        except Exception as e:
            print(f"Could not read TR from {func_path}: {e}")
            tr = 2.0

        # Build file names depending on run presence
        if run:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_events.tsv'
            session_id = f'task-{task}_run-{run}_dir-{direction}'
            run_part = f"_run-{run}"
        else:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_events.tsv'
            session_id = f'task-{task}_dir-{direction}'
            run_part = ""

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
            continue

        anat_path = os.path.join(session_dir, 'anat', f'sub-{subject}_{session}_T1w.nii.gz')
        output_dir = os.path.join(output_base, f'sub-{subject}', session)  # <-- store in session folder

        # Ensure motion file has the correct number of rows
        # As I removed the first 5(+1) columns in preprocessing, they have to be cut out from motion file
        try:
            n_scans = img.shape[0]
            if motion_path is not None:
                motion = np.loadtxt(motion_path)
            else:
                motion = np.random.randn(n_scans, 6)
            if motion.shape[0] > n_scans:
                motion = motion[-n_scans:]
        except Exception as e:
            print(f"Error processing motion file {motion_path}: {e}")
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

        # Run first level analysis
        first_level(subject_dic, mesh='individual')
'''
        # Rename and move output files
        for map_type, suffix in [
            ("z_score_maps", "ZMap"),
            ("stat_maps", "StatMap"),
            ("effect_size_maps", "EffectSizeMap"),
            ("effect_variance_maps", "EffectVarianceMap"),
        ]:
            map_dir = os.path.join(output_dir, map_type)
            if os.path.exists(map_dir):
                for fname in os.listdir(map_dir):
                    if fname.endswith(".gii"):
                        contrast = fname.split("_")[-1].replace(".gii", "")
                        new_name = (
                            f"sub-{subject}_{session}_task-{task}{run_part}_space-fsLR_{suffix}-{contrast}.gii"
                        )
                        os.rename(
                            os.path.join(map_dir, fname),
                            os.path.join(output_dir, new_name)
                        )

        # Move all files from subfolders to session folder and remove subfolders
        for subfolder in os.listdir(output_dir):
            subfolder_path = os.path.join(output_dir, subfolder)
            if os.path.isdir(subfolder_path) and subfolder.startswith("res_"):
                for file in os.listdir(subfolder_path):
                    shutil.move(
                        os.path.join(subfolder_path, file),
                        os.path.join(output_dir, file)
                    )
                shutil.rmtree(subfolder_path)
'''