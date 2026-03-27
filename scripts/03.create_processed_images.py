#!/usr/bin/env python3
"""
Multi-scale per-source montages + optional multi-source comparison plot.

For each source:
- Load RAW, T_X, T_XSUB (X in {25,50,100,...} kpc).
- Build RT_X by convolving RAW with a circular X kpc kernel from redshift.
- Crop to a common NaN-free field of view (in beams) for each version.
- Save cropped FITS with updated WCS and a per-source montage PNG.
- Optionally produce a multi-source comparison grid (RAW | T_X | RT_X).
"""

# Standard library
import argparse, os, random, warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import astropy.units as u
from astropy.convolution import convolve_fft
from astropy.cosmology import Planck18 as COSMO
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

warnings.filterwarnings("ignore", category=FITSFixedWarning)

# dcreclass utilities
from dcreclass.utils.fits import (
    ARCSEC_PER_RAD,
    fwhm_major_as, beam_solid_angle_sr,
    read_fits_array_header_wcs, reproject_like,
    header_cluster_coord, robust_vmin_vmax, kernel_from_beams,
    wcs_after_center_crop_and_resize,
)
from dcreclass.data.processing import (
    circular_kernel_from_z, load_z_table, _canon_size,
    crop_to_side_arcsec_on_raw, find_pairs_in_tree, report_nans,
    _nan_free_centred_square_side_as, compute_global_nbeams_equalized,
    check_nan_fraction, process_images_for_scale,
)
from dcreclass.utils.annotation import (
    add_beam_patch_simple, add_scalebar_kpc_simple,
)

print("Running 03.create_processed_images.py")

# ------------------- manual per-source centre offsets (INPUT pixels) -------------------
OFFSETS_PX: Dict[str, Tuple[int, int]] = { # (dy, dx) to add to header centre pixel
    "PSZ2G048.10+57.16": (-100, 100),
    "PSZ2G066.34+26.14": (150, 200),
    "PSZ2G107.10+65.32": (-100, 100),
    "PSZ2G113.91-37.01": (50, 300),
    "PSZ2G121.03+57.02": (0, -200),
    "PSZ2G133.60+69.04": (-200, -200),
    "PSZ2G135.17+65.43": (-150, 50),
    "PSZ2G141.05-32.61": (50, 200),
    "PSZ2G143.44+53.66": (100, 100),
    "PSZ2G149.22+54.18": (-100, 0),
    "PSZ2G150.56+46.67": (-300, 200),
    "PSZ2G205.90+73.76": (-100, 100),
}

# ========================= per-source montage ================================

def make_multi_scale_montage(source_name: str,
                             raw_path: Path,
                             scales: List[float],
                             z: float,
                             global_nbeams: Dict,
                             root_dir: Path,
                             downsample_size=(1, 128, 128),
                             save_fits: bool = False,
                             cheat_rt: bool = False,
                             subtract_beam: bool = True,
                             force: bool = False,
                             out_png: Optional[Path] = None,
                             out_fits_dir: Optional[Path] = None,
                             suffix: str = "",
                             fov_arcsec: Optional[float] = None):
    """
    Build RT for multiple scales and create compact montage with layout:
      RAW    [RAW_ORIG_LARGE] [RAW_CROP_LARGE]
             "25kpc"  "50kpc"  "100kpc"
      RT     [o][c]   [o][c]   [o][c]
      T      [o][c]   [o][c]   [o][c]
      SUB    [o][c]   [o][c]   [o][c]
    Optionally save beam-cropped FITS for each scale with updated WCS.
    """
    # For fov_arcsec mode: compute shared side = min(Ω, min_{s'∈S} ℓ_{T_{s'}}^{NaN-free})
    # so that all scales are cropped to the same field of view.
    effective_fov_arcsec = fov_arcsec
    if fov_arcsec is not None:
        source_dir = raw_path.parent
        min_nanfree = float('inf')
        for _sc in scales:
            _sc_str = int(_sc) if _sc == int(_sc) else _sc
            _t = source_dir / f"{source_name}T{_sc_str}kpc.fits"
            if _t.exists():
                try:
                    _arr, _hdr, _ = read_fits_array_header_wcs(_t)
                    _nf = _nan_free_centred_square_side_as(_arr, _hdr)
                    if _nf > 0:
                        min_nanfree = min(min_nanfree, _nf)
                except Exception:
                    pass
        if min_nanfree < float('inf'):
            effective_fov_arcsec = min(fov_arcsec, min_nanfree)
            if effective_fov_arcsec < fov_arcsec:
                print(f"[fov_crop] {source_name}: clipping Ω={fov_arcsec:.0f}\" "
                      f"to NaN-free limit {min_nanfree:.0f}\" -> {effective_fov_arcsec:.0f}\"")

    processed_scales = []
    for scale in scales:
        source_dir = raw_path.parent
        scale_str  = int(scale) if scale == int(scale) else scale
        t_path     = source_dir / f"{source_name}T{scale_str}kpc.fits"
        sub_path   = source_dir / f"{source_name}T{scale_str}kpcSUB.fits"
        if not t_path.exists():
            print(f"[SKIP] {source_name}: T{scale}kpc.fits not found")
            continue
        if not sub_path.exists():
            sub_path = None
        try:
            _nb = global_nbeams.get(scale, {'T': 20.0, 'Blur': 20.0})
            processed = process_images_for_scale(
                source_name=source_name, raw_path=raw_path,
                t_path=t_path, sub_path=sub_path,
                z=z, fwhm_kpc=float(scale),
                target_nbeams_T=_nb.get('T',  20.0),
                target_nbeams_RT=_nb.get('Blur', 20.0),
                target_nbeams_I=global_nbeams.get('RAW', 20.0),
                downsample_size=downsample_size, cheat_rt=cheat_rt,
                subtract_beam=subtract_beam,
                offsets_px=OFFSETS_PX, fov_arcsec=effective_fov_arcsec)
            processed['scale']    = scale
            processed['t_path']   = t_path
            processed['sub_path'] = sub_path
            processed_scales.append(processed)
        except Exception as e:
            print(f"[ERROR] {source_name} @ {scale}kpc: {e}")
            continue
    if not processed_scales:
        raise RuntimeError(f"No scales could be processed for {source_name}")

    Ho, Wo = _canon_size(downsample_size)[-2:]

    if not force:
        all_outputs_exist = out_png.exists()
        if save_fits and all_outputs_exist:
            for data in processed_scales:
                sc = data['scale']; sc_str = int(sc) if sc == int(sc) else sc
                rt_l = f"Blur{sc_str}kpc"; t_l = f"T{sc_str}kpc"
                required = [
                    out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_{suffix}.fits",
                    out_fits_dir / f"{source_name}_{rt_l}_fmt_{Ho}x{Wo}_{suffix}.fits",
                    out_fits_dir / f"{source_name}_{t_l}_fmt_{Ho}x{Wo}_{suffix}.fits",
                ]
                if data['sub_path']:
                    required.append(out_fits_dir / f"{source_name}_{t_l}SUB_fmt_{Ho}x{Wo}_{suffix}.fits")
                if not all(f.exists() for f in required):
                    all_outputs_exist = False; break
        if all_outputs_exist:
            print(f"[SKIP] {source_name}: all outputs exist (use --force to regenerate)")
            return

    has_sub   = any(d['has_sub'] for d in processed_scales)
    nrows     = 4 if has_sub else 3
    n_scales  = len(processed_scales)
    ncols     = n_scales * 2
    row_heights = [2.0] + [1.0] * (nrows - 1)
    fig = plt.figure(figsize=(4 * ncols, sum(row_heights) * 4.3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows, ncols, figure=fig, height_ratios=row_heights,
                  left=0.03, right=0.98, top=0.93, bottom=0.04,
                  wspace=0.15, hspace=0.12)

    usable_h = 0.93 - 0.04
    y_below_raw = 0.93 - (row_heights[0] / sum(row_heights)) * usable_h + 0.005
    usable_w = 0.98 - 0.03
    for scale_idx, data in enumerate(processed_scales):
        sc_str = int(data['scale']) if data['scale'] == int(data['scale']) else data['scale']
        col_centre = (scale_idx * 2 + 1) / ncols
        x = 0.03 + col_centre * usable_w
        fig.text(x, y_below_raw, f"{sc_str} kpc",
                 fontsize=11, fontweight='bold', ha='center', va='bottom',
                 transform=fig.transFigure)

    row_label_names = ['Blur', 'T', 'SUB'] if has_sub else ['Blur', 'T']
    total_height = sum(row_heights)
    for row_idx, label in enumerate(row_label_names, start=1):
        y_top = 1.0 - sum(row_heights[:row_idx]) / total_height
        y_bot = 1.0 - sum(row_heights[:row_idx + 1]) / total_height
        fig.text(0.005, (y_top + y_bot) / 2, label, fontsize=13, fontweight='bold',
                 ha='left', va='center', transform=fig.transFigure)

    first_data = processed_scales[0]
    I_raw      = first_data['I_raw']
    I_fmt_np   = first_data['I_fmt_np']
    W_i_fmt    = first_data['W_i_fmt']
    W_common   = (WCS(first_data['H_tgt']).celestial
                  if hasattr(WCS(first_data['H_tgt']), 'celestial')
                  else WCS(first_data['H_tgt'], naxis=2))
    I_on_common_for_display = reproject_like(I_raw, first_data['H_raw'], first_data['H_tgt'])

    ax_raw_orig = fig.add_subplot(gs[0, :n_scales], projection=W_common)
    ax_raw_crop = fig.add_subplot(gs[0, n_scales:], projection=W_i_fmt)
    vmin_I, vmax_I = robust_vmin_vmax(I_on_common_for_display)
    ax_raw_orig.imshow(I_on_common_for_display, origin="lower", vmin=vmin_I, vmax=vmax_I)
    ax_raw_orig.set_title("RAW (original)", fontsize=12, pad=8)
    ax_raw_crop.imshow(I_fmt_np, origin="lower", vmin=vmin_I, vmax=vmax_I)
    nan_frac = check_nan_fraction(I_fmt_np, "")
    ax_raw_crop.set_title(f"RAW (cropped {Ho}x{Wo})"
                          + (f" {nan_frac*100:.1f}% NaN" if nan_frac > 0 else ""),
                          fontsize=12, pad=8)

    row_labels = ['Blur', 'T', 'SUB'] if has_sub else ['Blur', 'T']
    for row_idx, row_label in enumerate(row_labels, start=1):
        for scale_idx, data in enumerate(processed_scales):
            sc_str   = int(data['scale']) if data['scale'] == int(data['scale']) else data['scale']
            col_orig = scale_idx * 2
            col_crop = scale_idx * 2 + 1
            if row_label == 'Blur':
                arr_orig, arr_crop = data['RT_rawgrid'], data['RT_fmt_np']
            elif row_label == 'T':
                arr_orig, arr_crop = data['T_on_common'], data['T_fmt_np']
            else:
                arr_orig, arr_crop = data['SUB_on_common'], data['SUB_fmt_np']
            if row_label == 'SUB' and not data['has_sub']:
                arr_orig = np.zeros_like(data['T_on_common'])
                arr_crop = np.zeros((Ho, Wo))
            vmin, vmax = (robust_vmin_vmax(arr_orig)
                          if not (row_label == 'SUB' and not data['has_sub'])
                          else (0.0, 1.0))
            W_common_data = (WCS(data['H_tgt']).celestial
                             if hasattr(WCS(data['H_tgt']), 'celestial')
                             else WCS(data['H_tgt'], naxis=2))
            ax_orig = fig.add_subplot(gs[row_idx, col_orig], projection=W_common_data)
            ax_crop = fig.add_subplot(gs[row_idx, col_crop], projection=data['W_i_fmt'])
            if row_label == 'SUB' and not data['has_sub']:
                ax_orig.imshow(arr_orig, origin="lower", cmap='gray')
                ax_orig.set_title("N/A", fontsize=9)
                ax_crop.imshow(arr_crop, origin="lower", cmap='gray')
                ax_crop.set_title("N/A", fontsize=9)
            else:
                ax_orig.imshow(arr_orig, origin="lower", vmin=vmin, vmax=vmax)
                ax_crop.imshow(arr_crop, origin="lower", vmin=vmin, vmax=vmax)
                nf = check_nan_fraction(arr_crop, "")
                ax_crop.set_title(f"{nf*100:.0f}% NaN" if nf > 0 else "", fontsize=8)

    mode_str   = "header" if cheat_rt else "circular"
    nbeams_str = ", ".join([
        f"{int(d['scale']) if d['scale']==int(d['scale']) else d['scale']}kpc: "
        f"T={global_nbeams.get(d['scale'], {}).get('T', 20.0):.1f}b "
        f"Blur={global_nbeams.get(d['scale'], {}).get('Blur', 20.0):.1f}b"
        for d in processed_scales
    ]) + f"  I={global_nbeams.get('RAW', 20.0):.1f}b"
    center_note = processed_scales[0]['center_note'] if processed_scales else ""
    fig.suptitle(f"{source_name} — Multi-scale ({mode_str}) — z={z:.4f}\n"
                 f"Crop: {nbeams_str}\n{center_note}", fontsize=11, y=0.995)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    scales_str = ", ".join([f"{int(s) if s == int(s) else s}kpc"
                            for s in [d['scale'] for d in processed_scales]])
    print(f"[OK] {source_name} -- scales: {scales_str} -> {out_png}")

    if save_fits:
        out_fits_dir = (out_fits_dir or out_png.parent)
        out_fits_dir.mkdir(parents=True, exist_ok=True)
        for data in processed_scales:
            sc = data['scale']; sc_str = int(sc) if sc == int(sc) else sc
            rt_label = f"Blur{sc_str}kpc"; t_label = f"T{sc_str}kpc"
            H_i_fmt  = data['H_i_fmt']
            H_t_fmt  = data.get('H_t_fmt',  H_i_fmt)  # T/SUB-specific WCS
            H_rt_fmt = data.get('H_rt_fmt', H_i_fmt)  # RT-specific WCS + effective beam
            raw_fits_path = out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_{suffix}.fits"
            if not raw_fits_path.exists():
                fits.writeto(raw_fits_path, data['I_fmt_np'].astype(np.float32),
                             H_i_fmt, overwrite=True)
                report_nans(raw_fits_path)
            fits.writeto(out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                         data['RT_fmt_np'].astype(np.float32), H_rt_fmt, overwrite=True)
            fits.writeto(out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                         data['T_fmt_np'].astype(np.float32), H_t_fmt, overwrite=True)
            if data['has_sub']:
                sub_label = t_label + "SUB"
                fits.writeto(out_fits_dir / f"{source_name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                             data['SUB_fmt_np'].astype(np.float32), H_t_fmt, overwrite=True)
                report_nans(out_fits_dir / f"{source_name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits")
            report_nans(out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_{suffix}.fits")
            report_nans(out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}_{suffix}.fits")

# ========================= comparison plot ===================================

def _validate_source_has_scales(source_name: str, root: Path, scales: List[float]) -> bool:
    """Return True only if source has RAW and all T_Xkpc files."""
    src_dir = root / source_name
    if not (src_dir / f"{source_name}.fits").exists():
        return False
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        if not (src_dir / f"{source_name}T{scale_int}kpc.fits").exists():
            print(f"  Missing T{scale_int}kpc.fits for {source_name}")
            return False
    return True


def get_classified_sources_from_loader(root: Path,
                                       scales: List[float]) -> Tuple[List[str], List[str]]:
    """Discover DE (class 50) and NDE (class 51) sources with all required scale files."""
    from dcreclass.data import load_galaxies
    try:
        de_valid, nde_valid = [], []
        for galaxy_class, valid_list in [(50, de_valid), (51, nde_valid)]:
            label = "DE" if galaxy_class == 50 else "NDE"
            print(f"Loading {label} sources (class {galaxy_class}) from RAW files...")
            _, _, _, _, train_fns, eval_fns = load_galaxies(
                galaxy_classes=[galaxy_class], versions='RAW', fold=0, train=False,
                NORMALISE=False, AUGMENT=False, BALANCE=False, PRINTFILENAMES=True,
                USE_CACHE=False, DEBUG=False)
            for src in sorted(set(train_fns + eval_fns)):
                if _validate_source_has_scales(src, root, scales):
                    valid_list.append(src)
                else:
                    print(f"Skipping {label} source {src}: missing required T_X files")
        return de_valid, nde_valid
    except Exception as e:
        print(f"Error loading from load_galaxies: {e}")
        import traceback; traceback.print_exc()
        return [], []


def select_valid_random_sources(root: Path, scales: List[float],
                                n_de: int = 3, n_nde: int = 3
                                ) -> Tuple[List[str], List[str], List[str]]:
    """Randomly select n_de DE and n_nde NDE sources that have all required files."""
    de_sources, nde_sources = get_classified_sources_from_loader(root, scales)
    print(f"Found {len(de_sources)} valid DE / {len(nde_sources)} valid NDE sources")
    random.shuffle(de_sources); random.shuffle(nde_sources)
    selected = de_sources[:min(n_de, len(de_sources))] + nde_sources[:min(n_nde, len(nde_sources))]
    print(f"Selected DE:  {de_sources[:n_de]}")
    print(f"Selected NDE: {nde_sources[:n_nde]}")
    return selected, de_sources, nde_sources


def _load_redshift(source_name: str, slug_to_z: Dict[str, float]) -> float:
    z = slug_to_z.get(source_name, np.nan)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"Invalid redshift z={z} for {source_name}")
    return z


def _get_annotation_header(source_name: str, root: Path,
                            scale: Optional[float],
                            crop_fov_arcsec: float,
                            downsample_size: Tuple[int, int]) -> fits.Header:
    """Build a synthetic FITS header with correct pixel scale for comparison-plot annotations."""
    src_dir = root / source_name
    if scale is None:
        fits_path = src_dir / f"{source_name}.fits"
    else:
        scale_int = int(scale) if float(scale).is_integer() else scale
        fits_path = src_dir / f"{source_name}T{scale_int}kpc.fits"
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header.copy()
    target_ny, target_nx = downsample_size
    hdr['NAXIS1'] = target_nx
    hdr['NAXIS2'] = target_ny
    pixel_scale_deg = (crop_fov_arcsec / target_nx) / 3600.0
    if 'CD1_1' in hdr:
        hdr['CD1_1'] = -pixel_scale_deg; hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0;              hdr['CD2_2'] = pixel_scale_deg
    else:
        hdr['CDELT1'] = -pixel_scale_deg; hdr['CDELT2'] = pixel_scale_deg
    hdr['CRPIX1'] = (target_nx + 1) / 2.0
    hdr['CRPIX2'] = (target_ny + 1) / 2.0
    return hdr


def _add_comparison_annotations(ax, source_name: str, root: Path,
                                 scale: Optional[float], z: float,
                                 global_nbeams: Dict,
                                 downsample_size: Tuple[int, int],
                                 image_type: str, color: str = 'yellow',
                                 subtract_beam: bool = True,
                                 fov_arcsec: Optional[float] = None,
                                 cheat_rt: bool = False):
    """Add beam patch and scale bar to one panel of the comparison plot."""
    from dcreclass.data.processing import effective_rt_beam_deg
    try:
        ref_scale     = 50 if scale is None else scale
        ref_scale_int = int(ref_scale) if float(ref_scale).is_integer() else ref_scale
        t_path = root / source_name / f"{source_name}T{ref_scale_int}kpc.fits"
        _, H_ref, _ = read_fits_array_header_wcs(t_path)

        # Determine nbeams and fwhm_as per image type
        if image_type == 'RAW' or scale is None:
            raw_path_ann = root / source_name / f"{source_name}.fits"
            _, H_raw_ann, _ = read_fits_array_header_wcs(raw_path_ann)
            fwhm_as = fwhm_major_as(H_raw_ann)
            nbeams  = global_nbeams.get('RAW', 20.0)
        elif image_type == 'Blur':
            if cheat_rt:
                # Cheat mode: Blur image was convolved to match the T beam exactly
                fwhm_as = fwhm_major_as(H_ref)
            else:
                raw_path_ann = root / source_name / f"{source_name}.fits"
                _, H_raw_ann, _ = read_fits_array_header_wcs(raw_path_ann)
                try:
                    bmaj_rt_deg, _, _ = effective_rt_beam_deg(
                        z, H_raw_ann, fwhm_kpc=float(scale), subtract_beam=subtract_beam)
                    fwhm_as = bmaj_rt_deg * 3600.0
                except Exception:
                    fwhm_as = fwhm_major_as(H_ref)
            nbeams = global_nbeams.get(scale, {}).get('Blur', 20.0)
        else:  # T (taper)
            fwhm_as = fwhm_major_as(H_ref)
            nbeams  = global_nbeams.get(scale, {}).get('T', 20.0)

        # In fov_crop mode nbeams==0; use the fixed FOV directly
        if fov_arcsec is not None and nbeams == 0.0:
            side_as = float(fov_arcsec)
        else:
            side_as = nbeams * fwhm_as
        ann_hdr  = _get_annotation_header(source_name, root, scale, side_as, downsample_size)

        if image_type == 'Blur':
            if cheat_rt:
                # Cheat mode: Blur beam = T beam
                ann_hdr['BMAJ'] = H_ref['BMAJ']
                ann_hdr['BMIN'] = H_ref['BMIN']
                ann_hdr['BPA']  = H_ref.get('BPA', 0.0)
            else:
                raw_path = root / source_name / f"{source_name}.fits"
                _, H_raw_ann, _ = read_fits_array_header_wcs(raw_path)
                bmaj_deg, bmin_deg, bpa_deg = effective_rt_beam_deg(
                    z, H_raw_ann, fwhm_kpc=float(scale), subtract_beam=subtract_beam)
                ann_hdr['BMAJ'] = bmaj_deg
                ann_hdr['BMIN'] = bmin_deg
                ann_hdr['BPA']  = bpa_deg
            add_beam_patch_simple(ax, ann_hdr, color='cyan', loc='lower left', fontsize=6,
                                  x_offset=0.07)
        else:
            add_beam_patch_simple(ax, ann_hdr, color=color, loc='lower left', fontsize=6,
                                  x_offset=0.07)
        add_scalebar_kpc_simple(ax, ann_hdr, z, length_kpc=1000.0,
                                color='white', loc='lower right', fontsize=6)
    except Exception as e:
        tag = 'RAW' if scale is None else f"T{int(scale) if float(scale).is_integer() else scale}kpc"
        print(f"Warning: annotations skipped for {source_name} {tag}: {e}")


def _process_source_for_comparison(source_name: str,
                                   root: Path,
                                   scales: List[float],
                                   slug_to_z: Dict[str, float],
                                   global_nbeams: Dict,
                                   downsample_size: Tuple[int, int],
                                   subtract_beam: bool = True,
                                   cheat_rt: bool = False,
                                   fov_arcsec: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Load and crop a single source for the comparison plot."""
    z = _load_redshift(source_name, slug_to_z)
    print(f"  z={z:.4f} for {source_name}")

    I_raw, H_raw, W_raw = read_fits_array_header_wcs(root / source_name / f"{source_name}.fits")

    header_sky = header_cluster_coord(H_raw)
    if header_sky is None:
        yc, xc = (I_raw.shape[0] - 1) / 2.0, (I_raw.shape[1] - 1) / 2.0
    else:
        x_i, y_i = W_raw.world_to_pixel(header_sky)
        yc, xc = float(y_i), float(x_i)
    dy_px, dx_px = OFFSETS_PX.get(source_name, (0.0, 0.0))
    yc += dy_px; xc += dx_px

    Ho, Wo = downsample_size

    def _downsample_nan_safe(arr):
        nan_mask = np.isnan(arr)
        t = torch.from_numpy(np.nan_to_num(arr, nan=0.0)).float().unsqueeze(0).unsqueeze(0)
        m = torch.from_numpy(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            result = torch.nn.functional.interpolate(t, size=(Ho, Wo), mode="bilinear",
                                                     align_corners=False)
            mask_d = torch.nn.functional.interpolate(m, size=(Ho, Wo), mode="bilinear",
                                                     align_corners=False)
        out = result.squeeze(0).squeeze(0).cpu().numpy()
        out[mask_d.squeeze(0).squeeze(0).cpu().numpy() > 0.5] = np.nan
        return out

    from dcreclass.data.processing import effective_rt_beam_deg

    results = {}
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        key_t = f'T{scale_int}'; key_blur = f'Blur{scale_int}'
        T_nat, H_tgt, _ = read_fits_array_header_wcs(
            root / source_name / f"{source_name}T{scale_int}kpc.fits")

        # Per-type crop sizes
        if fov_arcsec is not None:
            side_as_T  = float(fov_arcsec)
            side_as_RT = float(fov_arcsec)
        else:
            side_as_T = global_nbeams.get(scale, {}).get('T', 20.0) * fwhm_major_as(H_tgt)
            try:
                bmaj_rt_deg, _, _ = effective_rt_beam_deg(
                    z, H_raw, fwhm_kpc=scale, subtract_beam=subtract_beam)
                fwhm_rt_as = bmaj_rt_deg * 3600.0
            except Exception:
                fwhm_rt_as = fwhm_major_as(H_tgt)
            side_as_RT = global_nbeams.get(scale, {}).get('Blur', 20.0) * fwhm_rt_as

        T_on_raw = reproject_like(T_nat, H_tgt, H_raw)
        (T_crop,), _, _ = crop_to_side_arcsec_on_raw(T_on_raw, H_raw, side_as_T, center=(yc, xc))
        results[key_t] = _downsample_nan_safe(T_crop)

        if cheat_rt:
            ker          = kernel_from_beams(H_raw, H_tgt)
            scale_factor = beam_solid_angle_sr(H_tgt) / beam_solid_angle_sr(H_raw)
        else:
            ker          = circular_kernel_from_z(z, H_raw, fwhm_kpc=scale, subtract_beam=subtract_beam)
            DA_kpc       = COSMO.angular_diameter_distance(z).to_value(u.kpc)
            sigma_r      = (scale / DA_kpc) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            scale_factor = 2.0 * np.pi * sigma_r ** 2 / beam_solid_angle_sr(H_raw)
        blurred_img = convolve_fft(I_raw, ker, boundary="fill", fill_value=np.nan,
                                   nan_treatment="interpolate", normalize_kernel=True,
                                   psf_pad=True, fft_pad=True, allow_huge=True)
        blurred_img = blurred_img * scale_factor
        (blurred_crop,), _, _ = crop_to_side_arcsec_on_raw(blurred_img, H_raw, side_as_RT, center=(yc, xc))
        results[key_blur] = _downsample_nan_safe(blurred_crop)
        print(f"  {key_t} side_T={side_as_T:.0f}\" side_RT={side_as_RT:.0f}\" -> {results[key_t].shape}")

    # RAW crop uses its own version-specific n_beams
    if fov_arcsec is not None:
        side_raw = float(fov_arcsec)
    else:
        side_raw = global_nbeams.get('RAW', 20.0) * fwhm_major_as(H_raw)
    (I_crop,), _, _ = crop_to_side_arcsec_on_raw(I_raw, H_raw, side_raw, center=(yc, xc))
    results['RAW'] = _downsample_nan_safe(I_crop)
    print(f"  RAW side={side_raw:.0f}\" -> {results['RAW'].shape}")
    return results


def _plot_comparison_row(source_name: str, root: Path,
                         scales: List[float], slug_to_z: Dict[str, float],
                         global_nbeams: Dict, downsample_size: Tuple[int, int],
                         gs, grid_row: int, n_cols: int,
                         is_first_row: bool, fig: plt.Figure,
                         annotate: bool = True,
                         subtract_beam: bool = True,
                         cheat_rt: bool = False,
                         fov_arcsec: Optional[float] = None) -> int:
    """Render one source row in the comparison grid. Returns next grid_row."""
    try:
        images = _process_source_for_comparison(
            source_name, root, scales, slug_to_z, global_nbeams, downsample_size,
            subtract_beam=subtract_beam, cheat_rt=cheat_rt, fov_arcsec=fov_arcsec)
        try:
            z = _load_redshift(source_name, slug_to_z)
        except Exception:
            z = None

        col_idx = 0
        cmap = plt.cm.viridis.copy(); cmap.set_bad('white', 1.0)

        ax = fig.add_subplot(gs[grid_row, col_idx])
        if 'RAW' in images:
            vmin, vmax = robust_vmin_vmax(images['RAW'])
            ax.imshow(images['RAW'], origin='lower', vmin=vmin, vmax=vmax,
                      cmap=cmap, interpolation='nearest')
            ax.axis('off')
            if is_first_row:
                ax.set_title('Reference', fontsize=10, fontweight='bold', pad=0)
            if z is not None and annotate:
                _add_comparison_annotations(ax, source_name, root, None, z,
                                            global_nbeams, downsample_size, 'RAW',
                                            subtract_beam=subtract_beam,
                                            fov_arcsec=fov_arcsec,
                                            cheat_rt=cheat_rt)
            ax.text(-0.015, 0.5, source_name, transform=ax.transAxes,
                    fontsize=9, va='center', ha='right', rotation=90)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='red')
        col_idx += 1

        for scale in scales:
            scale_int = int(scale) if float(scale).is_integer() else scale
            for key, img_type, col_title in [
                    (f'T{scale_int}',    'T',    f'Tap.\u2009{scale_int}\u2009kpc'),
                    (f'Blur{scale_int}', 'Blur', f'Blur\u2009{scale_int}\u2009kpc')]:
                ax = fig.add_subplot(gs[grid_row, col_idx])
                if key in images:
                    vmin, vmax = robust_vmin_vmax(images[key])
                    ax.imshow(images[key], origin='lower', vmin=vmin, vmax=vmax,
                              cmap='viridis', interpolation='nearest')
                    ax.axis('off')
                    if is_first_row:
                        ax.set_title(col_title, fontsize=10, fontweight='bold', pad=0)
                    if z is not None and annotate:
                        _add_comparison_annotations(ax, source_name, root, scale, z,
                                                    global_nbeams, downsample_size, img_type,
                                                    subtract_beam=subtract_beam,
                                                    fov_arcsec=fov_arcsec,
                                                    cheat_rt=cheat_rt)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8, color='red')
                col_idx += 1
        return grid_row + 1

    except Exception as e:
        print(f"Error processing {source_name}: {e}")
        import traceback; traceback.print_exc()
        for j in range(n_cols):
            ax = fig.add_subplot(gs[grid_row, j])
            ax.axis('off')
            ax.text(0.5, 0.5, 'ERROR', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='red')
        return grid_row + 1


def create_comparison_plot(sources: List[str],
                           de_sources_all: List[str],
                           nde_sources_all: List[str],
                           root: Path,
                           scales: List[float],
                           slug_to_z: Dict[str, float],
                           global_nbeams: Dict,
                           output_path: Path,
                           downsample_size: Tuple[int, int] = (128, 128),
                           figsize: Tuple[float, float] = (10, 9),
                           dpi: int = 200,
                           annotate: bool = True,
                           subtract_beam: bool = True,
                           cheat_rt: bool = False,
                           fov_arcsec: Optional[float] = None):
    """Multi-source comparison grid: RAW | T25kpc | RT25kpc | T50kpc | RT50kpc | ..."""
    de_indices  = [i for i, s in enumerate(sources) if s in de_sources_all]
    nde_indices = [i for i, s in enumerate(sources) if s in nde_sources_all]
    n_cols = 1 + len(scales) * 2
    height_ratios = ([1.0] * len(de_indices)
                     + ([0.15] if de_indices and nde_indices else [])
                     + [1.0] * len(nde_indices))
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(len(height_ratios), n_cols, figure=fig,
                            hspace=0.0, wspace=0.001,
                            width_ratios=[1.0] * n_cols,
                            height_ratios=height_ratios,
                            left=0.04, right=0.999, top=0.998, bottom=0.002)
    grid_row = 0
    for i in de_indices:
        grid_row = _plot_comparison_row(
            sources[i], root, scales, slug_to_z, global_nbeams, downsample_size,
            gs, grid_row, n_cols, is_first_row=(grid_row == 0), fig=fig, annotate=annotate,
            subtract_beam=subtract_beam, cheat_rt=cheat_rt, fov_arcsec=fov_arcsec)
    if de_indices and nde_indices:
        grid_row += 1
    for i in nde_indices:
        is_first = (grid_row == len(de_indices) + (1 if de_indices and nde_indices else 0))
        grid_row = _plot_comparison_row(
            sources[i], root, scales, slug_to_z, global_nbeams, downsample_size,
            gs, grid_row, n_cols, is_first_row=is_first, fig=fig, annotate=annotate,
            subtract_beam=subtract_beam, cheat_rt=cheat_rt, fov_arcsec=fov_arcsec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f"[comparison] Saved -> {output_path}")

# ========================= parallel helpers ==================================

def process_raw_only_source(source_name: str,
                             raw_path: Path,
                             global_nbeams: Dict,
                             downsample_size=(1, 128, 128),
                             out_fits_dir: Optional[Path] = None,
                             suffix: str = "",
                             force: bool = False,
                             fov_arcsec: Optional[float] = None):
    """Crop, downsample, and save only the RAW FITS for a source with no valid redshift.
    Used when --include-no-z is set: T/Blur images are skipped since they require z.
    """
    Ho, Wo = _canon_size(downsample_size)[-2:]
    out_fits_dir = Path(out_fits_dir)
    raw_fits_path = out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_{suffix}.fits"

    if not force and raw_fits_path.exists():
        print(f"[SKIP] {source_name}: RAW output already exists (use --force to regenerate)")
        return

    I_raw, H_raw, W_raw = read_fits_array_header_wcs(raw_path)

    # Determine crop side in arcseconds
    beam_fwhm_I_as = fwhm_major_as(H_raw)
    if fov_arcsec is not None:
        side_as = float(fov_arcsec)
    else:
        n_beams_raw = global_nbeams.get('RAW', 20.0)
        max_side_as = _nan_free_centred_square_side_as(I_raw, H_raw)
        side_as = min(n_beams_raw * beam_fwhm_I_as, max_side_as)

    # Find cluster centre (same logic as process_images_for_scale)
    header_sky = header_cluster_coord(H_raw)
    if header_sky is None:
        H0, W0 = I_raw.shape
        yc, xc = float(H0 // 2), float(W0 // 2)
    else:
        x_i, y_i = W_raw.world_to_pixel(header_sky)
        yc, xc = float(y_i), float(x_i)
    dy_px, dx_px = OFFSETS_PX.get(source_name, (0.0, 0.0))
    if dy_px or dx_px:
        yc += dy_px; xc += dx_px

    # Crop
    (I_crop,), (nyc, nxc), (cy, cx) = crop_to_side_arcsec_on_raw(
        I_raw, H_raw, side_as, center=(yc, xc))

    check_nan_fraction(I_crop, f"{source_name} I_crop (RAW-only)")

    # Downsample
    t = torch.from_numpy(I_crop).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        y = torch.nn.functional.interpolate(t, size=(Ho, Wo), mode='bilinear',
                                            align_corners=False)
    I_fmt_np = y.squeeze(0).squeeze(0).cpu().numpy()

    # Build WCS header for the cropped+resized image
    H0_raw, W0_raw = I_raw.shape
    _, H_i_fmt = wcs_after_center_crop_and_resize(
        H_raw, H0_raw, W0_raw, nyc, nxc, Ho, Wo,
        int(round(cy)), int(round(cx)))

    # Save
    out_fits_dir.mkdir(parents=True, exist_ok=True)
    fits.writeto(raw_fits_path, I_fmt_np.astype(np.float32), H_i_fmt, overwrite=True)
    report_nans(raw_fits_path)
    print(f"[OK] {source_name}: RAW-only FITS saved -> {raw_fits_path.name}")


def process_raw_only_wrapper(args_tuple):
    """Parallel wrapper for process_raw_only_source."""
    source_name, raw_path, args_dict = args_tuple
    try:
        process_raw_only_source(
            source_name=source_name, raw_path=raw_path,
            global_nbeams=args_dict['global_nbeams'],
            downsample_size=args_dict['down'],
            out_fits_dir=args_dict['out_fits_dir'],
            suffix=args_dict['suffix'],
            force=args_dict['force'],
            fov_arcsec=args_dict.get('fov_arcsec'))
        return (source_name, True, None)
    except Exception as e:
        import traceback
        return (source_name, False, traceback.format_exc())


def process_single_source_wrapper(args_tuple):
    """Wrapper for parallel per-source montage processing."""
    (source_name, raw_path, scale_values, z, global_nbeams, args_dict) = args_tuple
    try:
        out_png = args_dict['out'] / f"{source_name}_montage_multiscale_{args_dict['suffix']}.png"
        make_multi_scale_montage(
            source_name=source_name, raw_path=raw_path,
            scales=scale_values, z=z, global_nbeams=global_nbeams,
            root_dir=args_dict['root'], downsample_size=args_dict['down'],
            save_fits=args_dict['save_fits'], cheat_rt=args_dict['cheat_rt'],
            subtract_beam=args_dict.get('subtract_beam', True),
            force=args_dict['force'], out_png=out_png,
            out_fits_dir=args_dict['out_fits_dir'], suffix=args_dict['suffix'],
            fov_arcsec=args_dict.get('fov_arcsec'))
        return (source_name, True, None)
    except Exception as e:
        import traceback
        return (source_name, False, traceback.format_exc())


def _fov_from_fits_header(hdr) -> float:
    """Return FOV side length in arcsec from a FITS header (CDELT or CD matrix)."""
    n1 = int(hdr.get('NAXIS1', 0))
    if 'CD1_1' in hdr:
        import math
        cd11 = hdr['CD1_1']
        cd21 = hdr.get('CD2_1', 0.0)
        pix_deg = math.hypot(cd11, cd21)
    else:
        pix_deg = abs(hdr.get('CDELT1', 0.0))
    return pix_deg * 3600.0 * n1


def generate_diagnostic_histograms(root_dir: Path, scales: List[float],
                                   global_nbeams: Dict,
                                   output_path: Path,
                                   processed_dir: Optional[Path] = None,
                                   fov_arcsec: Optional[float] = None,
                                   mode: str = 'beam_crop'):
    """FOV distribution histograms for Reference, Tap. X kpc (solid), Blur X kpc (dashed).

    Reads FOV from the actual processed FITS files in *processed_dir* when available,
    falling back to analytical computation from raw files otherwise.
    """
    from astropy.io import fits as _fits
    from matplotlib.lines import Line2D
    print("\n[diagnostics] Generating FOV distribution histograms...")

    scale_palette = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
    scale_colors  = {sc: scale_palette[i % len(scale_palette)] for i, sc in enumerate(scales)}
    ref_color     = '#111111'
    alpha_hist    = 0.6   # transparency for step histograms
    alpha_vline   = 0.7   # transparency for mean lines

    fov_raw: List[float] = []
    fov_t:  Dict[float, List[float]] = {sc: [] for sc in scales}
    fov_rt: Dict[float, List[float]] = {sc: [] for sc in scales}

    # ── Read FOV from actual processed FITS files ──────────────────────────
    if processed_dir is not None and Path(processed_dir).is_dir():
        proc = Path(processed_dir)
        # Group files by source name (prefix before first '_')
        source_map: Dict[str, List[Path]] = {}
        for f in sorted(proc.glob("*.fits")):
            src = f.name.split('_')[0]
            source_map.setdefault(src, []).append(f)

        for src, files in source_map.items():
            file_by_tag = {f.name: f for f in files}

            def _fov_for_tag(tag: str) -> Optional[float]:
                """Find the processed file matching *tag* and return its FOV."""
                for fname, fpath in file_by_tag.items():
                    if f'_{tag}_fmt_' in fname:
                        try:
                            with _fits.open(fpath) as hdul:
                                return _fov_from_fits_header(hdul[0].header)
                        except Exception:
                            return None
                return None

            raw_fov = _fov_for_tag('RAW')
            if raw_fov is not None:
                fov_raw.append(raw_fov)

            for sc in scales:
                sc_int = int(sc) if float(sc).is_integer() else sc
                t_fov  = _fov_for_tag(f'T{sc_int}kpc')
                rt_fov = _fov_for_tag(f'Blur{sc_int}kpc')
                if t_fov is not None:
                    fov_t[sc].append(t_fov)
                if rt_fov is not None:
                    fov_rt[sc].append(rt_fov)

        print(f"[diagnostics] Read from processed dir: "
              f"RAW={len(fov_raw)}, "
              + ", ".join(f"T{int(sc)}kpc={len(fov_t[sc])}/Blur={len(fov_rt[sc])}"
                          for sc in scales))

    # ── Fallback: analytical computation from raw FITS ─────────────────────
    else:
        print("[diagnostics] No processed_dir — computing FOVs analytically from raw files.")
        ref_scale     = next((s for s in scales if (int(s) if float(s).is_integer() else s) == 50), scales[0])
        ref_scale_int = int(ref_scale) if float(ref_scale).is_integer() else ref_scale

        for src_dir in sorted(p for p in Path(root_dir).glob("*") if p.is_dir()):
            src = src_dir.name
            raw_path = src_dir / f"{src}.fits"
            if not raw_path.exists():
                continue
            try:
                raw_arr, raw_hdr, _ = read_fits_array_header_wcs(raw_path)
                nan_free_raw = _nan_free_centred_square_side_as(raw_arr, raw_hdr)
                fwhm_raw     = fwhm_major_as(raw_hdr)
            except Exception:
                continue

            if fov_arcsec is not None:
                raw_fov = (min(fov_arcsec, nan_free_raw) if nan_free_raw > 0
                           else fov_arcsec)
            else:
                raw_fov = global_nbeams.get('RAW', 20.0) * fwhm_raw
            fov_raw.append(raw_fov)

            for sc in scales:
                sc_int = int(sc) if float(sc).is_integer() else sc
                t_path = src_dir / f"{src}T{sc_int}kpc.fits"
                if not t_path.exists():
                    continue
                try:
                    t_arr, t_hdr, _ = read_fits_array_header_wcs(t_path)
                    fwhm_t     = fwhm_major_as(t_hdr)
                    nan_free_t = _nan_free_centred_square_side_as(t_arr, t_hdr)
                except Exception:
                    continue
                if fov_arcsec is not None:
                    t_fov  = min(fov_arcsec, nan_free_t)  if nan_free_t  > 0 else fov_arcsec
                    rt_fov = min(fov_arcsec, nan_free_raw) if nan_free_raw > 0 else fov_arcsec
                else:
                    t_fov  = global_nbeams.get(sc, {}).get('T',  20.0) * fwhm_t
                    rt_fov = global_nbeams.get(sc, {}).get('Blur', 20.0) * fwhm_t
                fov_t[sc].append(t_fov)
                fov_rt[sc].append(rt_fov)

    # ── Plot ───────────────────────────────────────────────────────────────
    all_vals = fov_raw[:]
    for sc in scales:
        all_vals += fov_t[sc] + fov_rt[sc]
    bins = np.linspace(min(all_vals), max(all_vals), 21) if all_vals else 20

    fig, ax = plt.subplots(figsize=(14, 6))

    if fov_raw:
        ax.hist(fov_raw, bins=bins, histtype='step', color=ref_color,
                linewidth=2.2, linestyle='-', alpha=alpha_hist)
        ax.axvline(np.mean(fov_raw), color=ref_color, linestyle='-',
                   linewidth=2.2, alpha=alpha_vline)

    for sc in scales:
        color = scale_colors[sc]
        if fov_t[sc]:
            ax.hist(fov_t[sc], bins=bins, histtype='step', color=color,
                    linewidth=2.2, linestyle='-', alpha=alpha_hist)
            ax.axvline(np.mean(fov_t[sc]), color=color, linestyle='-',
                       linewidth=2.2, alpha=alpha_vline)
        if fov_rt[sc]:
            ax.hist(fov_rt[sc], bins=bins, histtype='step', color=color,
                    linewidth=2.0, linestyle='--', alpha=alpha_hist)
            ax.axvline(np.mean(fov_rt[sc]), color=color, linestyle='--',
                       linewidth=2.2, alpha=alpha_vline)

    color_handles = [Line2D([0], [0], color=ref_color, linewidth=2.2, linestyle='-',
                            alpha=alpha_hist, label='Reference')]
    for sc in scales:
        sc_str = int(sc) if sc == int(sc) else sc
        color_handles.append(Line2D([0], [0], color=scale_colors[sc], linewidth=2.2,
                                    linestyle='-', alpha=alpha_hist,
                                    label=rf'{sc_str}$\,$kpc'))
    style_handles = [
        Line2D([0], [0], color='#888888', linestyle='-',  linewidth=2.2,
               alpha=alpha_hist, label='Tap. (solid mean)'),
        Line2D([0], [0], color='#888888', linestyle='--', linewidth=2.0,
               alpha=alpha_hist, label='Blur (dashed mean)'),
    ]
    ax.legend(handles=color_handles + style_handles, fontsize=10, ncol=2)
    ax.set_xlabel('Field of View (arcsec)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Sources',      fontsize=13, fontweight='bold')
    ax.set_title(f'FOV Distribution — {mode.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[diagnostics] Saved to {output_path}")
    for sc in scales:
        sc_str = int(sc) if sc == int(sc) else sc
        if fov_t[sc]:
            print(f"  Tap. {sc_str}kpc: mean={np.mean(fov_t[sc]):.0f}\"  (N={len(fov_t[sc])})")
        if fov_rt[sc]:
            print(f"  Blur {sc_str}kpc: mean={np.mean(fov_rt[sc]):.0f}\"  (N={len(fov_rt[sc])})")

# --------------------------------- CLI ---------------------------------------

def parse_tuple3(txt: str) -> Tuple[int, int, int]:
    vals = [int(v) for v in str(txt).strip().split(",")]
    if len(vals) == 2: return (1, vals[0], vals[1])
    if len(vals) == 3: return (vals[0], vals[1], vals[2])
    raise argparse.ArgumentTypeError("Use H,W or C,H,W")


def main():
    ap = argparse.ArgumentParser(
        description="Multi-scale per-source montages + optional comparison plot.")
    PSZ2_DIR      = Path("/users/mbredber/scratch/data/PSZ2")
    DEFAULT_ROOT  = PSZ2_DIR / "fits"
    DEFAULT_Z_CSV = PSZ2_DIR / "cluster_source_data.csv"

    ap.add_argument("--root",     type=Path,  default=DEFAULT_ROOT)
    ap.add_argument("--z-csv",    type=Path,  default=DEFAULT_Z_CSV)
    ap.add_argument("--out",      type=Path,  default=None,
                    help="Montage output dir (default: PSZ2/<mode>/montages/)")
    ap.add_argument("--down",     type=parse_tuple3, default="128,128")
    ap.add_argument("--scales",   type=str,   default="25, 50, 100")
    ap.add_argument("--fov-arcmin", type=float, default=50.0)
    ap.add_argument("--crop-mode", type=str, default="beam_crop",
                    choices=["beam_crop", "fov_crop", "pixel_crop"],
                    help="Cropping strategy (default: beam_crop).")
    ap.add_argument("--blur-method", type=str, default="circular",
                    choices=["circular", "circular_no_sub", "cheat"],
                    help="Blurring kernel (default: circular).")
    ap.add_argument("--fov-arcsec", type=float, default=300.0,
                    help="FOV size in arcseconds when --crop-mode fov_crop (default: 300).")
    ap.add_argument("--force",    action="store_true", default=False)
    ap.add_argument("--only-offsets", action="store_true")
    ap.add_argument("--only",     type=str,   default="")
    ap.add_argument("--save-fits", action="store_true", default=True)
    ap.add_argument("--fits-out", type=Path,  default=None,
                    help="Processed FITS output dir (default: PSZ2/<mode>/fits_files/)")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--only-one", type=str, default=None,
                    help="Debug: process only this single source.")
    ap.add_argument("--no-montage", action="store_true", default=False)
    ap.add_argument("--include-no-z", action="store_true", default=False,
                    help="Process RAW images for sources with no valid redshift "
                         "(T/Blur scales are skipped for these sources since they require z).")
    ap.add_argument("--comparison-plot", action="store_true", default=False)
    ap.add_argument("--no-annotate", action="store_true", default=False)
    ap.add_argument("--comp-out", type=Path,  default=None,
                    help="Comparison plot path (default: PSZ2/<mode>/comparison_plot.png)")
    ap.add_argument("--comp-sources", type=str, default=None)
    ap.add_argument("--n-de",  type=int, default=3)
    ap.add_argument("--n-nde", type=int, default=3)
    ap.add_argument("--comp-seed", type=int, default=10)
    ap.add_argument("--comp-figsize", type=str, default="10,9")

    args = ap.parse_args()

    crop_mode   = args.crop_mode
    blur_method = args.blur_method
    cheat_rt      = (blur_method == 'cheat')
    subtract_beam = (blur_method != 'circular_no_sub')
    fov_arcsec    = args.fov_arcsec if crop_mode == 'fov_crop' else None
    include_no_z  = args.include_no_z

    mode_subdir = f"{crop_mode}/{blur_method}"
    if args.out is None:
        args.out = PSZ2_DIR / crop_mode / blur_method / "montages"
    if args.fits_out is None:
        args.fits_out = PSZ2_DIR / crop_mode / blur_method / "fits_files"
    if args.comp_out is None:
        annotate_tag = "noannotate" if args.no_annotate else "annotate"
        args.comp_out = Path("/users/mbredber/scratch/figures/processing") / f"comparison_{mode_subdir.replace('/', '_')}_{annotate_tag}.png"

    print(f"[init] Loading redshift table from {args.z_csv}")
    slug_to_z = load_z_table(args.z_csv)
    print(f"[init] Loaded {len(slug_to_z)} redshifts")

    if args.only_one:
        args.only     = args.only_one
        args.n_workers = 1
        args.force    = True
        print(f"[debug] --only-one mode: processing {args.only_one!r} only.")

    scale_values = []
    for s in args.scales.split(","):
        s = s.strip()
        if s:
            try:
                scale_values.append(float(s))
            except Exception:
                print(f"[SKIP] invalid scale {s!r}")
    if not scale_values:
        print("[ERROR] No valid scales provided"); return

    only_names = set(s.strip() for s in args.only.split(",") if s.strip())
    suffix     = blur_method
    Ho, Wo       = _canon_size(args.down)[-2:]
    out_fits_dir = args.fits_out

    print(f"[init] Processing scales: {scale_values}")
    print(f"[init] Mode: {mode_subdir}" +
          (f" (FOV={fov_arcsec:.0f}\")" if fov_arcsec else ""))
    print("\n" + "=" * 80)
    print("STEP 1: Computing global crop size (NaN-free across all files)")
    print("=" * 80)
    DIAG_OUT = Path("/users/mbredber/scratch/figures/processing")
    diag_path = DIAG_OUT / f"diagnostics_nbeams_{mode_subdir.replace('/', '_')}.png"
    if crop_mode == 'fov_crop':
        global_nbeams = {'RAW': 0.0,
                         **{scale: {'T': 0.0, 'Blur': 0.0} for scale in scale_values}}
        print(f"[fov_crop] Skipping beam scan — using fixed FOV={fov_arcsec:.0f}\"")
    else:
        global_nbeams = compute_global_nbeams_equalized(
            args.root, scale_values, slug_to_z, subtract_beam=subtract_beam)
        print(f"  RAW: {global_nbeams.get('RAW', 0.0):.2f} beams")
        for scale in scale_values:
            sc_str = int(scale) if scale == int(scale) else scale
            nb = global_nbeams.get(scale, {'T': 0.0, 'Blur': 0.0})
            print(f"  T{sc_str}kpc: T={nb['T']:.2f}b  Blur={nb['Blur']:.2f}b")
    psz2_base = args.root.parent
    proc_dir  = psz2_base / crop_mode / blur_method / 'fits_files'
    generate_diagnostic_histograms(args.root, scale_values, global_nbeams, diag_path,
                                   processed_dir=proc_dir if proc_dir.is_dir() else None,
                                   fov_arcsec=fov_arcsec, mode=mode_subdir)

    if not args.no_montage:
        print("\n" + "=" * 80)
        print("STEP 2: Per-source multi-scale montages")
        print("=" * 80)
        source_names_seen = set()
        sources_to_process = []
        for source_name, raw_path, t_path, sub_path, y_chosen in \
                find_pairs_in_tree(args.root, scale_values[0]):
            if args.only_offsets and source_name not in OFFSETS_PX: continue
            if only_names and source_name not in only_names: continue
            if source_name not in source_names_seen:
                source_names_seen.add(source_name)
                sources_to_process.append((source_name, raw_path))
        print(f"[init] Found {len(sources_to_process)} sources to process")

        n_workers = args.n_workers if args.n_workers else cpu_count()
        print(f"[PARALLEL] Using {n_workers} workers")
        args_dict = dict(out=args.out, suffix=suffix, root=args.root, down=args.down,
                         save_fits=args.save_fits, cheat_rt=cheat_rt,
                         subtract_beam=subtract_beam,
                         force=args.force, out_fits_dir=out_fits_dir,
                         fov_arcsec=fov_arcsec)
        tasks = []; raw_only_tasks = []; n_skip_z = 0
        for source_name, raw_path in sources_to_process:
            z = slug_to_z.get(source_name, np.nan)
            if not cheat_rt and (not np.isfinite(z) or z <= 0):
                if include_no_z:
                    raw_only_tasks.append((source_name, raw_path, args_dict))
                else:
                    print(f"[SKIP] {source_name}: no valid redshift (z={z})")
                    n_skip_z += 1
                continue
            tasks.append((source_name, raw_path, scale_values, z, global_nbeams, args_dict))
        print(f"[PARALLEL] Processing {len(tasks)} sources "
              f"(skipped {n_skip_z} missing z"
              + (f", {len(raw_only_tasks)} queued for RAW-only" if raw_only_tasks else "")
              + ")...")
        if n_workers == 1:
            results = [process_single_source_wrapper(t) for t in tasks]
        else:
            with Pool(processes=n_workers) as pool:
                results = pool.map(process_single_source_wrapper, tasks)
        n_ok = sum(1 for _, ok, _ in results if ok)
        n_fail = len(results) - n_ok
        for name, ok, err in results:
            if not ok:
                print(f"[ERROR] {name}:\n{err}")
        print(f"\n[PARALLEL] Done. Wrote {n_ok} montages. Failed {n_fail}.")

        if raw_only_tasks:
            print(f"\n[RAW-only] Processing {len(raw_only_tasks)} z-less sources (RAW FITS only)...")
            raw_args_dict = {**args_dict, 'global_nbeams': global_nbeams}
            raw_only_tasks_with_nbeams = [(n, p, raw_args_dict) for n, p, _ in raw_only_tasks]
            if n_workers == 1:
                raw_results = [process_raw_only_wrapper(t) for t in raw_only_tasks_with_nbeams]
            else:
                with Pool(processes=n_workers) as pool:
                    raw_results = pool.map(process_raw_only_wrapper, raw_only_tasks_with_nbeams)
            n_raw_ok = sum(1 for _, ok, _ in raw_results if ok)
            n_raw_fail = len(raw_results) - n_raw_ok
            for name, ok, err in raw_results:
                if not ok:
                    print(f"[ERROR] {name}:\n{err}")
            print(f"[RAW-only] Done. Wrote {n_raw_ok} RAW FITS. Failed {n_raw_fail}.")

        print("\n" + "=" * 80)
        print("VERIFICATION REPORT: SUB File Coverage")
        print("=" * 80)
        for scale in scale_values:
            with_sub_in, without_sub_in, with_sub_out, missing_sub_out = [], [], [], []
            for name, raw_path, t_path, sub_path, y_chosen in find_pairs_in_tree(args.root, scale):
                if args.only_offsets and name not in OFFSETS_PX: continue
                if only_names and name not in only_names: continue
                t_label   = f"T{int(y_chosen) if float(y_chosen).is_integer() else y_chosen}kpc"
                sub_label = f"{t_label}SUB"
                if sub_path is not None and sub_path.exists():
                    with_sub_in.append(name)
                    out_sub = out_fits_dir / f"{name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits"
                    (with_sub_out if out_sub.exists() else missing_sub_out).append(name)
                else:
                    without_sub_in.append(name)
            total = len(with_sub_in) + len(without_sub_in)
            print(f"Scale: {scale:.0f} kpc | sources: {total} | "
                  f"SUB input: {len(with_sub_in)} | SUB output: {len(with_sub_out)}")
            if missing_sub_out:
                print(f"  WARNING: {len(missing_sub_out)} sources have SUB input but no output:")
                for n in sorted(missing_sub_out)[:10]: print(f"    - {n}")
                if len(missing_sub_out) > 10: print(f"    ... and {len(missing_sub_out)-10} more")
            if without_sub_in:
                print(f"  WARNING: {len(without_sub_in)} sources missing SUB input:")
                for n in sorted(without_sub_in)[:10]: print(f"    - {n}")
                if len(without_sub_in) > 10: print(f"    ... and {len(without_sub_in)-10} more")
    else:
        print("[montage] Skipped (--no-montage).")

    if args.comparison_plot and not args.only_one:
        print("\n" + "=" * 80)
        print("STEP 3: Multi-source comparison plot")
        print("=" * 80)
        annotate = not args.no_annotate
        figsize  = tuple(float(x) for x in args.comp_figsize.split(','))
        comp_nbeams = global_nbeams  # equalized dict already computed in STEP 1
        if True:
            if args.comp_sources:
                sources = [s.strip() for s in args.comp_sources.split(',') if s.strip()]
                de_sources_all, nde_sources_all = get_classified_sources_from_loader(
                    args.root, scale_values)
                sources = [s for s in sources
                           if s in de_sources_all or s in nde_sources_all
                           or print(f"Warning: {s} not in classified sources, skipping") is None]
                if not sources:
                    print("[ERROR] None of the provided comp-sources are valid — skipping.")
                    return
            else:
                if args.comp_seed is not None:
                    random.seed(args.comp_seed)
                sources, de_sources_all, nde_sources_all = select_valid_random_sources(
                    args.root, scale_values, n_de=args.n_de, n_nde=args.n_nde)
                if not sources:
                    print("[ERROR] No valid sources found — skipping comparison plot.")
                    return
            _nb50 = comp_nbeams.get(50.0, comp_nbeams.get(scale_values[0], {'T': 0.0}))
            print(f"[comparison] annotate={annotate}, "
                  f"nbeams_T50={_nb50.get('T', 0.0):.2f} Blur50={_nb50.get('Blur', 0.0):.2f}")
            create_comparison_plot(
                sources=sources,
                de_sources_all=de_sources_all,
                nde_sources_all=nde_sources_all,
                root=args.root,
                scales=scale_values,
                slug_to_z=slug_to_z,
                global_nbeams=comp_nbeams,
                output_path=args.comp_out,
                figsize=figsize,
                annotate=annotate,
                subtract_beam=subtract_beam,
                cheat_rt=cheat_rt,
                fov_arcsec=fov_arcsec,
            )
    else:
        print("[comparison] Skipped (use --comparison-plot to enable).")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
