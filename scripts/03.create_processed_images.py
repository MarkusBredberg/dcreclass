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
    arcsec_per_pix, fwhm_major_as, beam_solid_angle_sr,
    read_fits_array_header_wcs, reproject_like,
    header_cluster_coord, robust_vmin_vmax,
)
from dcreclass.data.processing import (
    circular_kernel_from_z, load_z_table, _canon_size,
    crop_to_side_arcsec_on_raw, find_pairs_in_tree, report_nans,
    _nan_free_centred_square_side_as, compute_global_nbeams_per_version,
    compute_global_nbeams_min_t50, check_nan_fraction,
    process_images_for_scale,
)
from dcreclass.utils.annotation import (
    add_beam_patch, add_scalebar_kpc,
    add_beam_patch_simple, add_scalebar_kpc_simple,
)

print("Running 03.create_processed_images.py")

# ------------------- manual per-source centre offsets (INPUT pixels) -------------------
OFFSETS_PX: Dict[str, Tuple[int, int]] = {
    "PSZ2G048.10+57.16": (-100, 100),
    "PSZ2G066.34+26.14": (150, 200),
    "PSZ2G107.10+65.32": (-100, 100),
    "PSZ2G113.91-37.01": (50, 300),
    "PSZ2G121.03+57.02": (0, -200),
    "PSZ2G133.60+69.04": (-200, -200),
    "PSZ2G135.17+65.43": (-150, 50),
    "PSZ2G141.05-32.61": (50, 200),
    "PSZ2G143.44+53.66": (100, 100),
    "PSZ2G150.56+46.67": (-300, 200),
    "PSZ2G205.90+73.76": (-100, 100),
}

# ========================= per-source montage ================================

def make_multi_scale_montage(source_name: str,
                             raw_path: Path,
                             scales: List[float],
                             z: float,
                             global_nbeams: Dict[float, float],
                             root_dir: Path,
                             downsample_size=(1, 128, 128),
                             save_fits: bool = False,
                             cheat_rt: bool = False,
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
            processed = process_images_for_scale(
                source_name=source_name, raw_path=raw_path,
                t_path=t_path, sub_path=sub_path,
                z=z, fwhm_kpc=float(scale),
                target_nbeams=global_nbeams.get(scale, 20.0),
                downsample_size=downsample_size, cheat_rt=cheat_rt,
                offsets_px=OFFSETS_PX, fov_arcsec=fov_arcsec)
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
                rt_l = f"RT{sc_str}kpc"; t_l = f"T{sc_str}kpc"
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

    row_label_names = ['RT', 'T', 'SUB'] if has_sub else ['RT', 'T']
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

    row_labels = ['RT', 'T', 'SUB'] if has_sub else ['RT', 'T']
    for row_idx, row_label in enumerate(row_labels, start=1):
        for scale_idx, data in enumerate(processed_scales):
            sc_str   = int(data['scale']) if data['scale'] == int(data['scale']) else data['scale']
            col_orig = scale_idx * 2
            col_crop = scale_idx * 2 + 1
            if row_label == 'RT':
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
    nbeams_str = ", ".join([f"{int(s) if s == int(s) else s}kpc: {global_nbeams.get(s, 20.0):.1f}b"
                            for s in [d['scale'] for d in processed_scales]])
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
            rt_label = f"RT{sc_str}kpc"; t_label = f"T{sc_str}kpc"
            H_i_fmt  = data['H_i_fmt']
            raw_fits_path = out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_{suffix}.fits"
            if not raw_fits_path.exists():
                fits.writeto(raw_fits_path, data['I_fmt_np'].astype(np.float32),
                             H_i_fmt, overwrite=True)
                report_nans(raw_fits_path)
            fits.writeto(out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                         data['RT_fmt_np'].astype(np.float32), H_i_fmt, overwrite=True)
            fits.writeto(out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                         data['T_fmt_np'].astype(np.float32), H_i_fmt, overwrite=True)
            if data['has_sub']:
                sub_label = t_label + "SUB"
                fits.writeto(out_fits_dir / f"{source_name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                             data['SUB_fmt_np'].astype(np.float32), H_i_fmt, overwrite=True)
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
                USE_CACHE=False, DEBUG=False, PREFER_PROCESSED=False)
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
                                 global_nbeams: float,
                                 downsample_size: Tuple[int, int],
                                 image_type: str, color: str = 'yellow'):
    """Add beam patch and scale bar to one panel of the comparison plot."""
    try:
        ref_scale     = 50 if scale is None else scale
        ref_scale_int = int(ref_scale) if float(ref_scale).is_integer() else ref_scale
        t_path = root / source_name / f"{source_name}T{ref_scale_int}kpc.fits"
        _, H_ref, _ = read_fits_array_header_wcs(t_path)
        fwhm_as  = fwhm_major_as(H_ref)
        side_as  = global_nbeams * fwhm_as
        ann_hdr  = _get_annotation_header(source_name, root, scale, side_as, downsample_size)
        if image_type == 'RT':
            DA_kpc   = COSMO.angular_diameter_distance(z).to_value(u.kpc)
            fwhm_deg = (float(scale) / DA_kpc) * ARCSEC_PER_RAD / 3600.0
            ann_hdr['BMAJ'] = fwhm_deg; ann_hdr['BMIN'] = fwhm_deg; ann_hdr['BPA'] = 0.0
            add_beam_patch_simple(ax, ann_hdr, color='cyan', loc='lower left', fontsize=6)
        else:
            add_beam_patch_simple(ax, ann_hdr, color=color, loc='lower left', fontsize=6)
        add_scalebar_kpc_simple(ax, ann_hdr, z, length_kpc=1000.0,
                                color='white', loc='lower right', fontsize=6)
    except Exception as e:
        tag = 'RAW' if scale is None else f"T{int(scale) if float(scale).is_integer() else scale}kpc"
        print(f"Warning: annotations skipped for {source_name} {tag}: {e}")


def _process_source_for_comparison(source_name: str,
                                   root: Path,
                                   scales: List[float],
                                   slug_to_z: Dict[str, float],
                                   global_nbeams: float,
                                   downsample_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
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

    results = {}
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        key_t = f'T{scale_int}'; key_rt = f'RT{scale_int}'
        T_nat, H_tgt, _ = read_fits_array_header_wcs(
            root / source_name / f"{source_name}T{scale_int}kpc.fits")
        side_as = global_nbeams * fwhm_major_as(H_tgt)

        T_on_raw = reproject_like(T_nat, H_tgt, H_raw)
        (T_crop,), _, _ = crop_to_side_arcsec_on_raw(T_on_raw, H_raw, side_as, center=(yc, xc))
        results[key_t] = _downsample_nan_safe(T_crop)

        ker   = circular_kernel_from_z(z, H_raw, fwhm_kpc=scale)
        I_smt = convolve_fft(I_raw, ker, boundary="fill", fill_value=np.nan,
                             nan_treatment="interpolate", normalize_kernel=True,
                             psf_pad=True, fft_pad=True, allow_huge=True)
        DA_kpc   = COSMO.angular_diameter_distance(z).to_value(u.kpc)
        sigma_r  = (scale / DA_kpc) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        omega_t  = 2.0 * np.pi * sigma_r ** 2
        RT_raw   = I_smt * (omega_t / beam_solid_angle_sr(H_raw))
        (RT_crop,), _, _ = crop_to_side_arcsec_on_raw(RT_raw, H_raw, side_as, center=(yc, xc))
        results[key_rt] = _downsample_nan_safe(RT_crop)
        print(f"  {key_t} side={side_as:.0f}\" -> {results[key_t].shape}")

    t50_key = next((f'T{int(s) if float(s).is_integer() else s}' for s in scales
                    if (int(s) if float(s).is_integer() else s) == 50), None)
    if t50_key is None:
        scale_int = int(scales[0]) if float(scales[0]).is_integer() else scales[0]
        t50_key = f'T{scale_int}'
    _, H_t50, _ = read_fits_array_header_wcs(
        root / source_name / f"{source_name}{t50_key}kpc.fits")
    side_raw = global_nbeams * fwhm_major_as(H_t50)
    (I_crop,), _, _ = crop_to_side_arcsec_on_raw(I_raw, H_raw, side_raw, center=(yc, xc))
    results['RAW'] = _downsample_nan_safe(I_crop)
    print(f"  RAW -> {results['RAW'].shape}")
    return results


def _plot_comparison_row(source_name: str, root: Path,
                         scales: List[float], slug_to_z: Dict[str, float],
                         global_nbeams: float, downsample_size: Tuple[int, int],
                         gs, grid_row: int, n_cols: int,
                         is_first_row: bool, fig: plt.Figure,
                         annotate: bool = True) -> int:
    """Render one source row in the comparison grid. Returns next grid_row."""
    try:
        images = _process_source_for_comparison(
            source_name, root, scales, slug_to_z, global_nbeams, downsample_size)
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
                ax.set_title('RAW', fontsize=10, fontweight='bold', pad=0)
            if z is not None and annotate:
                _add_comparison_annotations(ax, source_name, root, None, z,
                                            global_nbeams, downsample_size, 'RAW')
            ax.text(-0.015, 0.5, source_name, transform=ax.transAxes,
                    fontsize=9, va='center', ha='right', rotation=90)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='red')
        col_idx += 1

        for scale in scales:
            scale_int = int(scale) if float(scale).is_integer() else scale
            for key, img_type in [(f'T{scale_int}', 'T'), (f'RT{scale_int}', 'RT')]:
                ax = fig.add_subplot(gs[grid_row, col_idx])
                if key in images:
                    vmin, vmax = robust_vmin_vmax(images[key])
                    ax.imshow(images[key], origin='lower', vmin=vmin, vmax=vmax,
                              cmap='viridis', interpolation='nearest')
                    ax.axis('off')
                    if is_first_row:
                        ax.set_title(f'{key}kpc', fontsize=10, fontweight='bold', pad=0)
                    if z is not None and annotate:
                        _add_comparison_annotations(ax, source_name, root, scale, z,
                                                    global_nbeams, downsample_size, img_type)
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
                           global_nbeams: float,
                           output_path: Path,
                           downsample_size: Tuple[int, int] = (128, 128),
                           figsize: Tuple[float, float] = (10, 9),
                           dpi: int = 200,
                           annotate: bool = True):
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
            gs, grid_row, n_cols, is_first_row=(grid_row == 0), fig=fig, annotate=annotate)
    if de_indices and nde_indices:
        grid_row += 1
    for i in nde_indices:
        is_first = (grid_row == len(de_indices) + (1 if de_indices and nde_indices else 0))
        grid_row = _plot_comparison_row(
            sources[i], root, scales, slug_to_z, global_nbeams, downsample_size,
            gs, grid_row, n_cols, is_first_row=is_first, fig=fig, annotate=annotate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f"[comparison] Saved -> {output_path}")

# ========================= parallel helpers ==================================

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
            force=args_dict['force'], out_png=out_png,
            out_fits_dir=args_dict['out_fits_dir'], suffix=args_dict['suffix'],
            fov_arcsec=args_dict.get('fov_arcsec'))
        return (source_name, True, None)
    except Exception as e:
        import traceback
        return (source_name, False, traceback.format_exc())


def generate_diagnostic_histograms(root_dir: Path, scales: List[float],
                                   global_nbeams: Dict[float, float],
                                   output_path: Path):
    """FOV distribution histograms per version."""
    print("\n[diagnostics] Generating distribution histograms...")
    data = {scale: {'fov': []} for scale in scales}
    for src_dir in sorted(p for p in root_dir.glob("*") if p.is_dir()):
        for scale in scales:
            scale_str = f"{int(scale)}" if scale == int(scale) else f"{scale}"
            t_path = src_dir / f"{src_dir.name}T{scale_str}kpc.fits"
            if not t_path.exists(): continue
            try:
                arr, hdr, _ = read_fits_array_header_wcs(t_path)
                max_side_as = _nan_free_centred_square_side_as(arr, hdr)
                if max_side_as <= 0: continue
                beam_fwhm_as = fwhm_major_as(hdr)
                actual_side_as = min(global_nbeams[scale] * beam_fwhm_as, max_side_as)
                data[scale]['fov'].append(actual_side_as)
            except Exception:
                continue
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    for scale, color in zip(scales, colors):
        if data[scale]['fov']:
            sc_str = int(scale) if scale == int(scale) else scale
            ax.hist(data[scale]['fov'], bins=20, alpha=0.5, color=color,
                    label=f'T{sc_str}kpc (mu={np.mean(data[scale]["fov"]):.0f}")',
                    edgecolor='black', linewidth=1.2)
            ax.axvline(np.mean(data[scale]['fov']), color=color, linestyle='--',
                       linewidth=2.5, alpha=0.9)
    ax.set_xlabel('Field of View (arcsec)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Sources',      fontsize=13, fontweight='bold')
    ax.set_title('FOV Distribution Per Version', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[diagnostics] Saved to {output_path}")
    for scale in scales:
        if data[scale]['fov']:
            sc_str = int(scale) if scale == int(scale) else scale
            print(f"  T{sc_str}kpc: target={global_nbeams.get(scale, 20.0):.1f}, "
                  f"mean FOV={np.mean(data[scale]['fov']):.1f}\"")

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
    ap.add_argument("--cheat-rt",  action="store_true", default=False)
    ap.add_argument("--fov-crop",  action="store_true", default=False,
                    help="Crop all images to a fixed FOV in arcseconds (ignores beam count).")
    ap.add_argument("--fov-arcsec", type=float, default=300.0,
                    help="FOV size in arcseconds for --fov-crop mode (default: 300).")
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
    ap.add_argument("--comparison-plot", action="store_true", default=False)
    ap.add_argument("--no-annotate", action="store_true", default=False)
    ap.add_argument("--comp-out", type=Path,  default=None,
                    help="Comparison plot path (default: PSZ2/<mode>/comparison_plot.pdf)")
    ap.add_argument("--comp-sources", type=str, default=None)
    ap.add_argument("--n-de",  type=int, default=3)
    ap.add_argument("--n-nde", type=int, default=3)
    ap.add_argument("--comp-seed", type=int, default=10)
    ap.add_argument("--comp-figsize", type=str, default="10,9")

    args = ap.parse_args()

    if args.fov_crop and args.cheat_rt:
        ap.error("--fov-crop and --cheat-rt are mutually exclusive.")

    # Set mode-dependent output directories
    if args.fov_crop:
        mode_subdir = "fov_crop"
    elif args.cheat_rt:
        mode_subdir = "cheat_crop"
    else:
        mode_subdir = "beam_crop"
    if args.out is None:
        args.out = PSZ2_DIR / mode_subdir / "montages"
    if args.fits_out is None:
        args.fits_out = PSZ2_DIR / mode_subdir / "fits_files"
    if args.comp_out is None:
        args.comp_out = PSZ2_DIR / mode_subdir / "comparison_plot.pdf"

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

    only_names   = set(s.strip() for s in args.only.split(",") if s.strip())
    if args.fov_crop:
        suffix = "fov"
    elif args.cheat_rt:
        suffix = "cheat"
    else:
        suffix = "circ"
    fov_arcsec   = args.fov_arcsec if args.fov_crop else None
    Ho, Wo       = _canon_size(args.down)[-2:]
    out_fits_dir = args.fits_out

    print(f"[init] Processing scales: {scale_values}")
    print(f"[init] Mode: {mode_subdir}" +
          (f" (FOV={fov_arcsec:.0f}\")" if fov_arcsec else ""))
    print("\n" + "=" * 80)
    print("STEP 1: Computing global crop size (NaN-free across all files)")
    print("=" * 80)
    if args.fov_crop:
        global_nbeams = {scale: 0.0 for scale in scale_values}
        print(f"[fov_crop] Skipping beam scan — using fixed FOV={fov_arcsec:.0f}\"")
        diag_path = None
    else:
        global_nbeams = compute_global_nbeams_per_version(args.root, scale_values)
        for scale in scale_values:
            sc_str = int(scale) if scale == int(scale) else scale
            print(f"  T{sc_str}kpc: {global_nbeams[scale]:.1f} beams")
        diag_path = args.out.parent / "diagnostics_nbeams_distribution.png"
    if diag_path:
        generate_diagnostic_histograms(args.root, scale_values, global_nbeams, diag_path)

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
                         save_fits=args.save_fits, cheat_rt=args.cheat_rt,
                         force=args.force, out_fits_dir=out_fits_dir,
                         fov_arcsec=fov_arcsec)
        tasks = []; n_skip_z = 0
        for source_name, raw_path in sources_to_process:
            z = slug_to_z.get(source_name, np.nan)
            if not args.cheat_rt and (not np.isfinite(z) or z <= 0):
                print(f"[SKIP] {source_name}: no valid redshift (z={z})")
                n_skip_z += 1; continue
            tasks.append((source_name, raw_path, scale_values, z, global_nbeams, args_dict))
        print(f"[PARALLEL] Processing {len(tasks)} sources (skipped {n_skip_z} missing z)...")
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
        comp_nbeams = compute_global_nbeams_min_t50(args.root)
        if comp_nbeams is None:
            print("[ERROR] Could not compute global nbeams for comparison plot — skipping.")
        else:
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
            print(f"[comparison] annotate={annotate}, nbeams={comp_nbeams:.1f}")
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
            )
    else:
        print("[comparison] Skipped (use --comparison-plot to enable).")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
