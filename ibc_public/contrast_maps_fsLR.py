"""
Generate first-level contrast maps in fsLR space for IBC dataset subjects,
with optional merging (fixed-effects) across AP/PA directions.

Usage:
  python contrast_maps_fsLR.py <subject-id> [--no-merge]

Examples:
  python contrast_maps_fsLR.py 11
  python contrast_maps_fsLR.py 11 --no-merge

Notes:
- "Merging" here means: we estimate first-level GLMs per run (dir-ap and dir-pa
  as separate 'sessions' in a single call) and then compute fixed-effects maps
  across those sessions for the same (task, run) to produce dir-ffx outputs.
- If only one direction exists for a (task, run), it will be processed alone
  (no fixed-effects possible).
- Assumes files from a post-fmriprep pipeline with cleaned dtseries in CIFTI
  space, motion regressors, and BIDS-like events TSVs.
"""

import sys
import os
import glob
import nibabel as nib
import numpy as np

from ibc_public.utils_pipeline import first_level, fixed_effects_analysis
from ibc_public.fslr_ffx_fallback import do_ffx_dscalar

# -------------------
# Configurable roots
# -------------------
BASE_DIR = '/ptmp/hmueller2/Downloads/fmriprep_out'
OUTPUT_BASE = '/ptmp/hmueller2/Downloads/contrast_maps_fsLR'
RAW_IBC = '/ptmp/hmueller2/Downloads/ibc_raw'  # For events TSVs

# -------------------
# CLI parsing
# -------------------
if len(sys.argv) < 2:
    print("ERROR: subject-id required, e.g. `python contrast_maps_fsLR.py 11`")
    sys.exit(1)

subject = sys.argv[1]
merge_dirs = True
if len(sys.argv) > 2 and sys.argv[2] == '--no-merge':
    merge_dirs = False

subject_dir = os.path.join(BASE_DIR, f'sub-{subject}')
session_dirs = sorted(glob.glob(os.path.join(subject_dir, 'ses-*')))

processed_groups = []
processed_runs = []
skipped = []

def parse_fname_bits(fname):
    """Extract task, direction, and run from a dtseries filename parts."""
    parts = fname.split('_')
    task = [p.split('-')[1] for p in parts if p.startswith('task-')]
    direction = [p.split('-')[1] for p in parts if p.startswith('dir-')]
    run = [p.split('-')[1] for p in parts if p.startswith('run-')]
    return (task[0] if task else 'unknown',
            direction[0] if direction else 'unknown',
            run[0] if run else None)

def cifti_tr_and_scans(img):
    """Return TR and n_scans from a CIFTI dtseries image."""
    ax0 = img.header.get_axis(0)
    ax1 = img.header.get_axis(1)
    # Find the SeriesAxis
    ts_axis = ax0 if isinstance(ax0, nib.cifti2.SeriesAxis) else ax1 if isinstance(ax1, nib.cifti2.SeriesAxis) else None
    if ts_axis is not None:
        tr = float(getattr(ts_axis, "step", 2.0))
        n_scans = ts_axis.size
    else:
        # Fallback
        tr = 2.0
        n_scans = img.shape[0]
    return tr, n_scans

def load_and_align_motion(motion_path, target_len):
    """Load motion regressors and align/pad/trim to target length."""
    try:
        motion = np.loadtxt(motion_path)
        if motion.ndim == 1:
            if motion.size % 6 == 0:
                motion = motion.reshape(-1, 6)
            else:
                motion = motion.reshape(-1, 1)
        diff = motion.shape[0] - target_len
        if diff == 0:
            pass
        elif abs(diff) == 1:
            # allow off-by-one, pad or trim
            if diff == 1:
                motion = motion[-target_len:, :]
            else:
                motion = np.pad(motion, ((1, 0), (0, 0)), mode='constant', constant_values=0.0)
        elif abs(diff) <= 5:
            # tolerate small mismatches (e.g., missing initial discarded volumes)
            if diff > 0:
                motion = motion[diff:, :]
            else:
                motion = np.pad(motion, ((abs(diff), 0), (0, 0)), mode='constant', constant_values=0.0)
            print(f"Adjusted motion length: original {motion.shape[0]-diff} -> {motion.shape[0]} (target {target_len})")
        else:
            return None, f"motion_len_mismatch_{motion.shape[0]}_{target_len}"
        return motion.astype(np.float32), None
    except Exception as e:
        return None, f"motion_processing_error:{e}"

def build_subject_dict(records, output_dir, anat_path, tr_override=None):
    """Build a subject dictionary for ibc_public.utils_pipeline.first_level."""
    session_ids = [rec['session_id'] for rec in records]
    func_paths = [rec['func_path'] for rec in records]
    onset_paths = [rec['onset_path'] for rec in records]
    motions = [rec['motion'] for rec in records]
    # Optionally ensure identical TR across records; if not, use first and warn
    trs = [rec['tr'] for rec in records]
    tr = tr_override if tr_override is not None else trs[0]
    if any(abs(t - tr) > 1e-6 for t in trs):
        print(f"Warning: TR differs across directions {trs}; using TR={tr} from first record.")
    subject_dic = {
        'session_id': session_ids,
        'func': func_paths,
        'onset': onset_paths,
        'realignment_parameters': motions,
        'output_dir': output_dir,
        'anat': anat_path,
        'hrf_model': 'spm', # or make this configurable per protocol/task
        'high_pass': 1.0 / 128.0,  # match IBC default
        'drift_model': 'cosine',
        'TR': tr
    }
    return subject_dic

for session_dir in session_dirs:
    session = os.path.basename(session_dir)
    glm_dir = os.path.join(session_dir, 'postfmriprep', 'GLM')
    if not os.path.exists(glm_dir):
        continue

    # Find all cleaned dtseries runs for this session
    func_files = sorted(glob.glob(os.path.join(
        glm_dir, f'sub-{subject}_{session}_task-*_dir-*_*cleaned_noscrub.dtseries.nii')))
    if not func_files:
        print(f"No functional files found for sub-{subject} {session}")
        continue

    # Group runs by (task, run) so that ap/pa for the same run can be merged
    groups = {}  # key: (task, run) -> list of record dicts for ap/pa
    for func_path in func_files:
        fname = os.path.basename(func_path)
        task, direction, run = parse_fname_bits(fname)

        # Load CIFTI to get TR and number of time points
        try:
            img = nib.load(func_path)
            tr, n_scans = cifti_tr_and_scans(img)
        except Exception as e:
            print(f"Could not read CIFTI header for {func_path}: {e}")
            skipped.append((session, task, direction, run, "cifti_header_error"))
            continue

        # Build motion and onset filenames
        if run is not None:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_motion.txt'
            # Prefer BIDS-standard events name (no dir), but try a few variants
            onset_fname_candidates = [
                f'sub-{subject}_{session}_task-{task}_run-{run}_events.tsv',                       # BIDS-standard
                f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_events.tsv',        # legacy with dir
                f'sub-{subject}_{session}_task-{task}_acq-{direction}_run-{run}_events.tsv',        # occasional variant
            ]
            session_id = f'task-{task}_run-{run}_dir-{direction}'
        else:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_motion.txt'
            onset_fname_candidates = [
                f'sub-{subject}_{session}_task-{task}_events.tsv',                                  # BIDS-standard
                f'sub-{subject}_{session}_task-{task}_dir-{direction}_events.tsv',                  # legacy with dir
            ]
            session_id = f'task-{task}_dir-{direction}'

        motion_path = os.path.join(session_dir, 'postfmriprep', 'regressors', motion_fname)
        if not os.path.exists(motion_path):
            print(f"Missing motion file: {motion_path}")
            skipped.append((session, task, direction, run, "missing_motion"))
            continue

        # Find first existing events file among candidates
        onset_path = None
        for cand in onset_fname_candidates:
            p = os.path.join(RAW_IBC, f'sub-{subject}', session, 'func', cand)
            if os.path.exists(p):
                onset_path = p
                break
        if onset_path is None:
            print(f"Missing events file (tried): {[os.path.join(RAW_IBC, f'sub-{subject}', session, 'func', c) for c in onset_fname_candidates]}")
            skipped.append((session, task, direction, run, "missing_events"))
            continue

        motion, err = load_and_align_motion(motion_path, n_scans)
        if err:
            print(f"Skipping {session_id}: {err}")
            skipped.append((session, task, direction, run, err))
            continue

        rec = {
            'session_id': session_id,
            'func_path': func_path,
            'onset_path': onset_path,
            'motion': motion,
            'direction': direction,
            'tr': tr
        }
        key = (task, run)
        groups.setdefault(key, []).append(rec)

    # Prepare common paths
    anat_path = os.path.join(session_dir, 'anat', f'sub-{subject}_{session}_T1w.nii.gz')
    output_dir_base = os.path.join(OUTPUT_BASE, f'sub-{subject}', session)
    os.makedirs(output_dir_base, exist_ok=True)

    # For each (task, run) group, either merge AP/PA or process single direction
    for (task, run), records in groups.items():
        # Sort records so ap then pa if present (for reproducibility)
        records = sorted(records, key=lambda r: r['direction'])
        run_tag = f"run-{run}_" if run is not None else ""
        label = f"{session}_{task}_{run_tag}"

        # Default: enable compcorr except if you're processing mathlang
        compcorr_flag = True
        if isinstance(task, str) and task.lower().startswith('mathlang'):
            compcorr_flag = False

        if merge_dirs and len(records) >= 2:
            # If both directions exist, pass them together, then fixed-effects
            dirs = {r['direction'] for r in records}
            if 'ap' in dirs and 'pa' in dirs:
                print(f"Merging AP/PA for {label} (n={len(records)}).")
                subject_dic = build_subject_dict(records, output_dir_base, anat_path)
                try:
                    first_level(subject_dic, mesh='fsLR', compcorr=compcorr_flag)
                    fixed_effects_analysis(subject_dic, mesh='fsLR')  # produces dir-ffx
                    processed_groups.append((session, task, run, 'ap+pa'))
                except Exception as e:
                    print(f"FFX run failed for {label}: {e}")
                    skipped.append((session, task, 'ap+pa', run, "first_level_or_ffx_exception"))
                    continue
            else:
                # Only one direction present, fall back to single-run processing
                print(f"Only one direction present for {label}; processing single run.")
                for r in records:
                    single_dic = build_subject_dict([r], output_dir_base, anat_path)
                    try:
                        first_level(single_dic, mesh='fsLR', compcorr=compcorr_flag)
                        processed_runs.append((session, task, run, r['direction']))
                    except Exception as e:
                        print(f"Run failed for {label}{r['direction']}: {e}")
                        skipped.append((session, task, r['direction'], run, "first_level_exception"))
                        continue
        else:
            # Merging disabled or not enough directions: process each direction independently
            for r in records:
                print(f"Processing single direction for {label}{r['direction']}.")
                single_dic = build_subject_dict([r], output_dir_base, anat_path)
                try:
                    first_level(single_dic, mesh='fsLR', compcorr=compcorr_flag)
                    processed_runs.append((session, task, run, r['direction']))
                except Exception as e:
                    print(f"Run failed for {label}{r['direction']}: {e}")
                    skipped.append((session, task, r['direction'], run, "first_level_exception"))
                    continue

# Summary
print(f"\nSubject {subject} summary:")
print(f"  Groups merged (AP+PA): {len(processed_groups)}")
print(f"  Single-direction runs: {len(processed_runs)}")
print(f"  Skipped               : {len(skipped)}")
if skipped:
    by_reason = {}
    for sess, task, direction, run, reason in skipped:
        by_reason.setdefault(reason, 0)
        by_reason[reason] += 1
    print("  Skip reasons:")
    for reason, count in sorted(by_reason.items(), key=lambda x: (-x[1], x[0])):
        print(f"    {reason}: {count}")