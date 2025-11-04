import os
import glob
import warnings
import numpy as np
import nibabel as nib

def _load_dscalar(fp):
    img = nib.load(fp)
    data = np.asanyarray(img.get_fdata()).squeeze()
    return img, data

def _save_dscalar_like(template_img, data, out_fp):
    # Preserve axes/header from the template
    data = np.asarray(data)
    # Ensure singleton dims like (N,) -> (N, 1) are acceptable; nibabel handles 1D dscalar
    out_img = nib.cifti2.Cifti2Image(data, header=template_img.header)
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    nib.save(out_img, out_fp)

def _ffx_from_effect_var(ap_eff, ap_var, pa_eff, pa_var):
    w_ap = np.where(ap_var > 0, 1.0 / ap_var, 0.0)
    w_pa = np.where(pa_var > 0, 1.0 / pa_var, 0.0)
    w_sum = w_ap + w_pa
    with np.errstate(divide='ignore', invalid='ignore'):
        eff_ffx = np.where(w_sum > 0, (w_ap * ap_eff + w_pa * pa_eff) / w_sum, 0.0)
        var_ffx = np.where(w_sum > 0, 1.0 / w_sum, np.inf)
        z_ffx = np.where(var_ffx > 0, eff_ffx / np.sqrt(var_ffx), 0.0)
    return eff_ffx, var_ffx, z_ffx

def _stouffer_z(z_ap, z_pa):
    zs = np.stack([z_ap, z_pa], axis=0)
    n = np.sum(~np.isnan(zs), axis=0)
    n = np.maximum(n, 1)
    return np.nansum(zs, axis=0) / np.sqrt(n)

def _list_contrasts(dir_root, subdir):
    folder = os.path.join(dir_root, subdir)
    if not os.path.isdir(folder):
        return set()
    names = []
    for fp in glob.glob(os.path.join(folder, "*.dscalar.nii")):
        names.append(os.path.basename(fp).replace(".dscalar.nii", ""))
    return set(names)

def _dir_root(output_dir_base, task, direction):
    return os.path.join(output_dir_base, f"res_task-{task}_space-fsLR_dir-{direction}")

def _ffx_root(output_dir_base, task):
    return os.path.join(output_dir_base, f"res_task-{task}_space-fsLR_dir-ffx")

def do_ffx_dscalar(output_dir_base, task, run=None, require_intersection=True):
    """
    Fixed-effects for fsLR dscalars produced by first_level:
    - Prefer effect_size_maps + effect_variance_maps if present in both ap & pa.
    - Else fallback to Stouffer z from stat_maps in both ap & pa.
    Writes merged dscalars into dir-ffx/effect_size_maps, effect_variance_maps, stat_maps.
    Returns number of contrasts written (z merged; eff/var only when available).
    """
    dir_ap = _dir_root(output_dir_base, task, "dir-ap")
    dir_pa = _dir_root(output_dir_base, task, "dir-pa")

    # Determine available contrasts
    ap_eff = _list_contrasts(dir_ap, "effect_size_maps")
    pa_eff = _list_contrasts(dir_pa, "effect_size_maps")
    ap_var = _list_contrasts(dir_ap, "effect_variance_maps")
    pa_var = _list_contrasts(dir_pa, "effect_variance_maps")
    ap_z   = _list_contrasts(dir_ap, "stat_maps")
    pa_z   = _list_contrasts(dir_pa, "stat_maps")

    eff_candidates = (ap_eff & pa_eff) if require_intersection else (ap_eff | pa_eff)
    var_candidates = (ap_var & pa_var) if require_intersection else (ap_var | pa_var)
    z_candidates   = (ap_z & pa_z)     if require_intersection else (ap_z | pa_z)

    ffx_root = _ffx_root(output_dir_base, task)
    for sub in ("effect_size_maps", "effect_variance_maps", "stat_maps"):
        os.makedirs(os.path.join(ffx_root, sub), exist_ok=True)

    written = 0

    # First, try proper effect/variance-based FFX
    valid_eff = sorted(eff_candidates & var_candidates)
    for contrast in valid_eff:
        ap_eff_fp = os.path.join(dir_ap, "effect_size_maps",     f"{contrast}.dscalar.nii")
        pa_eff_fp = os.path.join(dir_pa, "effect_size_maps",     f"{contrast}.dscalar.nii")
        ap_var_fp = os.path.join(dir_ap, "effect_variance_maps", f"{contrast}.dscalar.nii")
        pa_var_fp = os.path.join(dir_pa, "effect_variance_maps", f"{contrast}.dscalar.nii")
        if not (os.path.isfile(ap_eff_fp) and os.path.isfile(pa_eff_fp) and
                os.path.isfile(ap_var_fp) and os.path.isfile(pa_var_fp)):
            continue
        tpl_img, ap_eff_data = _load_dscalar(ap_eff_fp)
        _,      pa_eff_data = _load_dscalar(pa_eff_fp)
        _,      ap_var_data = _load_dscalar(ap_var_fp)
        _,      pa_var_data = _load_dscalar(pa_var_fp)

        eff_ffx, var_ffx, z_ffx = _ffx_from_effect_var(ap_eff_data, ap_var_data,
                                                       pa_eff_data, pa_var_data)
        _save_dscalar_like(tpl_img, eff_ffx, os.path.join(ffx_root, "effect_size_maps",     f"{contrast}.dscalar.nii"))
        _save_dscalar_like(tpl_img, var_ffx, os.path.join(ffx_root, "effect_variance_maps", f"{contrast}.dscalar.nii"))
        _save_dscalar_like(tpl_img, z_ffx,   os.path.join(ffx_root, "stat_maps",            f"{contrast}.dscalar.nii"))
        written += 1

    # Then, for any contrasts not handled above but present as ZMaps in both dirs, do Stouffer
    leftovers = sorted((z_candidates - set(valid_eff)))
    for contrast in leftovers:
        ap_z_fp = os.path.join(dir_ap, "stat_maps", f"{contrast}.dscalar.nii")
        pa_z_fp = os.path.join(dir_pa, "stat_maps", f"{contrast}.dscalar.nii")
        if not (os.path.isfile(ap_z_fp) and os.path.isfile(pa_z_fp)):
            continue
        tpl_img, ap_z_data = _load_dscalar(ap_z_fp)
        _,      pa_z_data = _load_dscalar(pa_z_fp)
        z_ffx = _stouffer_z(ap_z_data, pa_z_data)
        _save_dscalar_like(tpl_img, z_ffx, os.path.join(ffx_root, "stat_maps", f"{contrast}.dscalar.nii"))
        written += 1

    if written == 0:
        warnings.warn("do_ffx_dscalar: no contrasts merged; check per-direction outputs and naming.")
    return written