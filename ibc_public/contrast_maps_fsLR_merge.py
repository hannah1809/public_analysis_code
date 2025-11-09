"""
My script to generate first-level contrast maps in fsLR space for IBC dataset subjects.
Taken from the the official IBC analysis code (Pinho et al., 2018).
Instead of giftis space, I use fsLR (ciftis) here. Thus, some functions had to be adapted.
"""

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
# Track all processed session_ids for fixed-effects grouping
all_session_ids = []

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
            all_session_ids.append(session_id)  # Track successful runs
        except Exception as e:
            print(f"Run {session_id} failed: {e}")
            skipped.append((session_id, "first_level_exception"))

# ========== Fixed-effects combination per task ==========
print(f"\n{'='*60}")
print(f"Running fixed-effects analysis to combine runs per task...")
print(f"{'='*60}")

def run_manual_fixed_effects(subject, contrast_base):
    """
    Manually implement fixed-effects combining across runs per task.
    Same logic as run_fixed_effects_only.py but integrated here.
    """
    # Collect all runs grouped by task
    runs_by_task = {}
    
    session_dirs = sorted(glob.glob(os.path.join(contrast_base, f'sub-{subject}', 'ses-*')))
    
    for session_dir in session_dirs:
        session = os.path.basename(session_dir)
        result_dirs = glob.glob(os.path.join(session_dir, 'res_task-*_space-fsLR_*'))
        
        for result_dir in result_dirs:
            dirname = os.path.basename(result_dir)
            
            # Parse task name
            parts = dirname.split('_')
            task = None
            for part in parts:
                if part.startswith('task-'):
                    task = part.replace('task-', '')
                    break
            
            if task is None:
                continue
            
            # Check if required directories exist
            effect_size_dir = os.path.join(result_dir, 'effect_size_maps')
            effect_var_dir = os.path.join(result_dir, 'effect_variance_maps')
            
            if not os.path.exists(effect_size_dir) or not os.path.exists(effect_var_dir):
                continue
            
            # Get list of contrasts (filter out F-tests)
            effect_files = glob.glob(os.path.join(effect_size_dir, '*.dscalar.nii'))
            
            if not effect_files:
                continue
            
            # Extract contrast names and check their shapes (only t-contrasts)
            contrasts = []
            for f in effect_files:
                contrast_name = os.path.basename(f).replace('.dscalar.nii', '')
                
                try:
                    img = nib.load(f)
                    data_shape = img.get_fdata().shape
                    
                    # Only include single contrasts (shape: (1, n_vertices))
                    if data_shape[0] == 1:
                        contrasts.append(contrast_name)
                    else:
                        print(f"    Skipping {contrast_name}: F-test with shape {data_shape}")
                except:
                    pass
            
            if not contrasts:
                continue
            
            if task not in runs_by_task:
                runs_by_task[task] = []
            
            runs_by_task[task].append({
                'session': session,
                'result_dir': result_dir,
                'dirname': dirname,
                'effect_size_dir': effect_size_dir,
                'effect_var_dir': effect_var_dir,
                'contrasts': contrasts
            })
    
    print(f"Found {len(runs_by_task)} unique tasks for fixed-effects")
    
    # Run fixed-effects for each task
    combined_count = 0
    for task, runs in sorted(runs_by_task.items()):
        if len(runs) < 2:
            print(f"  Skipping {task}: only {len(runs)} run(s)")
            continue
        
        print(f"\n  Task: {task} ({len(runs)} runs to combine)")
        
        # Get contrasts (intersection of all runs)
        contrast_sets = [set(run['contrasts']) for run in runs]
        contrasts = sorted(list(set.intersection(*contrast_sets)))
        
        if not contrasts:
            print(f"    No common contrasts found!")
            continue
        
        print(f"    Processing {len(contrasts)} contrasts")
        
        # Create output directory
        ffx_dir = os.path.join(contrast_base, f'sub-{subject}', f'res_task-{task}_space-fsLR_dir-ffx')
        os.makedirs(ffx_dir, exist_ok=True)
        
        ffx_effect_dir = os.path.join(ffx_dir, 'effect_size_maps')
        ffx_var_dir = os.path.join(ffx_dir, 'effect_variance_maps')
        ffx_z_dir = os.path.join(ffx_dir, 'z_score_maps')
        
        os.makedirs(ffx_effect_dir, exist_ok=True)
        os.makedirs(ffx_var_dir, exist_ok=True)
        os.makedirs(ffx_z_dir, exist_ok=True)
        
        # Process each contrast
        successful_contrasts = 0
        for contrast in contrasts:
            # Load effect sizes and variances from all runs
            effect_sizes = []
            variances = []
            
            for run in runs:
                effect_file = os.path.join(run['effect_size_dir'], f'{contrast}.dscalar.nii')
                var_file = os.path.join(run['effect_var_dir'], f'{contrast}.dscalar.nii')
                
                if not os.path.exists(effect_file) or not os.path.exists(var_file):
                    continue
                
                try:
                    effect_img = nib.load(effect_file)
                    var_img = nib.load(var_file)
                    
                    effect_data = effect_img.get_fdata()
                    var_data = var_img.get_fdata()
                    
                    # Ensure shape is (1, n_vertices)
                    if effect_data.shape[0] != 1:
                        continue
                    
                    effect_sizes.append(effect_data[0])
                    variances.append(var_data[0])
                    
                except Exception as e:
                    print(f"      Error loading {run['session']}/{contrast}: {e}")
                    continue
            
            if len(effect_sizes) < 2:
                continue
            
            # Convert to arrays
            effect_sizes = np.array(effect_sizes)  # Shape: (n_runs, n_vertices)
            variances = np.array(variances)
            
            # Fixed-effects: inverse-variance weighted average
            weights = 1.0 / (variances + 1e-10)
            weighted_effects = effect_sizes * weights
            combined_effect = np.sum(weighted_effects, axis=0) / np.sum(weights, axis=0)
            combined_variance = 1.0 / np.sum(weights, axis=0)
            z_score = combined_effect / np.sqrt(combined_variance + 1e-10)
            
            # Clean up
            z_score = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)
            combined_effect = np.nan_to_num(combined_effect, nan=0.0)
            combined_variance = np.nan_to_num(combined_variance, nan=0.0)
            
            # Load template for header
            template_img = nib.load(os.path.join(runs[0]['effect_size_dir'], 
                                                  f'{contrast}.dscalar.nii'))
            
            # Ensure shape is (1, n_vertices)
            combined_effect_2d = combined_effect.reshape(1, -1)
            combined_variance_2d = combined_variance.reshape(1, -1)
            z_score_2d = z_score.reshape(1, -1)
            
            # Create new CIFTI images with correct header
            brain_models_axis = template_img.header.get_axis(1)  # BrainModel axis
            scalar_axis = nib.cifti2.ScalarAxis(['combined_ffx'])
            new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
            
            # Save
            effect_out = nib.Cifti2Image(combined_effect_2d, header=new_header)
            nib.save(effect_out, os.path.join(ffx_effect_dir, f'{contrast}.dscalar.nii'))
            
            var_out = nib.Cifti2Image(combined_variance_2d, header=new_header)
            nib.save(var_out, os.path.join(ffx_var_dir, f'{contrast}.dscalar.nii'))
            
            z_out = nib.Cifti2Image(z_score_2d, header=new_header)
            nib.save(z_out, os.path.join(ffx_z_dir, f'{contrast}.dscalar.nii'))
            
            successful_contrasts += 1
        
        print(f"    ✓ Combined {successful_contrasts}/{len(contrasts)} contrasts")
        combined_count += 1
    
    return combined_count

if all_session_ids:
    try:
        n_tasks = run_manual_fixed_effects(subject, output_base)
        print(f"\n✓ Fixed-effects complete: {n_tasks} tasks processed")
    except Exception as e:
        print(f"\n✗ Fixed-effects analysis failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No runs processed successfully; skipping fixed-effects analysis")

# Summary
print(f"\n{'='*60}")
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