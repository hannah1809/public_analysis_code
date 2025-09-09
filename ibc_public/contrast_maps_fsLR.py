import sys
import os
import glob
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

    # Find all dtseries.nii files for this session
    func_files = sorted(glob.glob(os.path.join(glm_dir, f'sub-{subject}_{session}_task-*_dir-*_*cleaned.dtseries.nii')))
    if not func_files:
        print(f"No functional files found for {subject} {session}")
        continue

    onset_files, motion_files, session_ids = [], [], []
    for func_path in func_files:
        fname = os.path.basename(func_path)
        parts = fname.split('_')
        task = [p.split('-')[1] for p in parts if p.startswith('task-')]
        direction = [p.split('-')[1] for p in parts if p.startswith('dir-')]
        run = [p.split('-')[1] for p in parts if p.startswith('run-')]
        task = task[0] if task else 'unknown'
        direction = direction[0] if direction else 'unknown'
        run = run[0] if run else None  # None if not present

        # Build file names depending on run presence
        if run:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_events.tsv'
            session_id = f'task-{task}_run-{run}_dir-{direction}'
        else:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_events.tsv'
            session_id = f'task-{task}_run-01_dir-{direction}'

        # Motion file path
        motion_path = os.path.join(session_dir, 'postfmriprep', 'regressors', motion_fname)
        if not os.path.exists(motion_path):
            motion_path = None

        # Onset/events file path
        onset_path = os.path.join('/ptmp/hmueller2/Downloads/ibc_raw', f'sub-{subject}', session, 'func', onset_fname)
        if not os.path.exists(onset_path):
            onset_path = None

        onset_files.append(onset_path)
        motion_files.append(motion_path)
        session_ids.append(session_id)

    anat_path = os.path.join(session_dir, 'anat', f'sub-{subject}_{session}_T1w.nii.gz')
    output_dir = os.path.join(output_base, f'sub-{subject}', session)

    subject_dic = {
        'session_id': session_ids,
        'func': func_files,
        'onset': onset_files,
        'realignment_parameters': motion_files,
        'output_dir': output_dir,
        'anat': anat_path,
        'hrf_model': 'spm',
        'high_pass': 0.01,
        'drift_model': 'cosine',
        'TR': 2.0
    }

    # As input is in fsLR space, individual mesh should be identical to fsLR ??!
    first_level(subject_dic, mesh='individual')  