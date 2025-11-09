"""
Run fixed-effects analysis on already-computed first-level contrast maps.
This script combines runs (ap/pa/different runs) for each task.
"""

import sys
import os
import glob
import nibabel as nib
import numpy as np

def run_fixed_effects_for_subject(subject):
    """
    Manually implement fixed-effects combining across runs per task.
    """
    contrast_base = f'/ptmp/hmueller2/Downloads/contrast_maps_fsLR/sub-{subject}'
    output_base = contrast_base
    
    if not os.path.exists(contrast_base):
        print(f"No contrast maps found for subject {subject}")
        return
    
    # Find all session directories with contrast maps
    session_dirs = sorted(glob.glob(os.path.join(contrast_base, 'ses-*')))
    
    if not session_dirs:
        print(f"No session directories found in {contrast_base}")
        return
    
    print(f"Found {len(session_dirs)} sessions for subject {subject}")
    
    # Collect all runs grouped by task
    runs_by_task = {}
    
    for session_dir in session_dirs:
        session = os.path.basename(session_dir)
        
        # Find all result directories
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
            
            # Get list of contrasts
            effect_files = glob.glob(os.path.join(effect_size_dir, '*.dscalar.nii'))
            
            if not effect_files:
                continue
            
            # Extract contrast names and check their shapes
            contrasts = []
            for f in effect_files:
                contrast_name = os.path.basename(f).replace('.dscalar.nii', '')
                
                # Check if this is a single contrast (not F-test)
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
    
    print(f"\n{'='*60}")
    print(f"Found {len(runs_by_task)} unique tasks")
    for task, runs in sorted(runs_by_task.items()):
        print(f"  {task}: {len(runs)} run(s)")
    print(f"{'='*60}\n")
    
    # Run fixed-effects for each task
    combined_count = 0
    for task, runs in sorted(runs_by_task.items()):
        if len(runs) < 2:
            print(f"Skipping {task}: only {len(runs)} run(s)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Task: {task} ({len(runs)} runs)")
        print(f"{'='*60}")
        
        # Get contrasts (intersection of all runs)
        contrast_sets = [set(run['contrasts']) for run in runs]
        contrasts = sorted(list(set.intersection(*contrast_sets)))
        
        if not contrasts:
            print(f"  No common contrasts found!")
            continue
        
        print(f"  Processing {len(contrasts)} contrasts")
        
        # Create output directory
        ffx_dir = os.path.join(output_base, f'res_task-{task}_space-fsLR_dir-ffx')
        os.makedirs(ffx_dir, exist_ok=True)
        
        ffx_effect_dir = os.path.join(ffx_dir, 'effect_size_maps')
        ffx_var_dir = os.path.join(ffx_dir, 'effect_variance_maps')
        ffx_z_dir = os.path.join(ffx_dir, 'z_score_maps')
        
        os.makedirs(ffx_effect_dir, exist_ok=True)
        os.makedirs(ffx_var_dir, exist_ok=True)
        os.makedirs(ffx_z_dir, exist_ok=True)
        
        # Process each contrast
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
                        print(f"    Skipping {contrast} in {run['session']}: unexpected shape {effect_data.shape}")
                        continue
                    
                    effect_sizes.append(effect_data[0])
                    variances.append(var_data[0])
                    
                except Exception as e:
                    print(f"    Error loading {run['session']}: {e}")
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
            # Use the template's axis info to create proper scalar axis
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
        
        print(f"  ✓ Combined {len(contrasts)} contrasts")
        combined_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Fixed-effects complete: {combined_count} tasks processed")
    print(f"{'='*60}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_fixed_effects_only.py <subject>")
        sys.exit(1)
    
    subject = sys.argv[1]
    run_fixed_effects_for_subject(subject)