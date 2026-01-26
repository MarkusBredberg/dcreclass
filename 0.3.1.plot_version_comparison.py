#!/usr/bin/env python3
"""
Plot comparison of RAW, T_Xkpc, and RT_Xkpc images for multiple sources in a grid layout.

UPDATED to match 0.3.0 approach:
- Uses circular (isotropic) kernel for RT generation based on REDSHIFT (header-free, generalizable)
- Uses equal-beams cropping system
- Applies zoom to crop side length (not FOV)
- NO vertical spacing between rows within groups
- White space ONLY between DE and NDE groups
- NO DE/NDE title labels
- Column titles: "RT{X}kpc" instead of "RT X"
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import random
import importlib.util
import sys
from astropy.convolution import convolve_fft
from astropy.io import fits

# Import necessary functions from 0.3.0.create_processed_images.py
def _import_create_processed_images():
    """Import module with dots in filename."""
    module_path = Path(__file__).parent / "0.3.0.create_processed_images.py"    
    spec = importlib.util.spec_from_file_location("create_processed_images", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["create_processed_images"] = module
    spec.loader.exec_module(module)
    return module

_cpi = _import_create_processed_images()

# Import necessary functions
read_fits_array_header_wcs = _cpi.read_fits_array_header_wcs
reproject_like = _cpi.reproject_like
circular_kernel_from_z = _cpi.circular_kernel_from_z  # Circular kernel from redshift
load_z_table = _cpi.load_z_table  # Redshift table loader
beam_solid_angle_sr = _cpi.beam_solid_angle_sr
robust_vmin_vmax = _cpi.robust_vmin_vmax
crop_to_fov_on_raw = _cpi.crop_to_fov_on_raw
header_cluster_coord = _cpi.header_cluster_coord
OFFSETS_PX = _cpi.OFFSETS_PX
arcsec_per_pix = _cpi.arcsec_per_pix
fwhm_major_as = _cpi.fwhm_major_as
beam_cov_world = _cpi.beam_cov_world
_cd_matrix_rad = _cpi._cd_matrix_rad


# ---------------------- Functions from 0.3.0 ----------------------

def compute_global_nbeams_min(root_dir):
    """
    Scan all subdirs under root_dir, find every *T50kpc.fits,
    compute n_beams = min(FOV_x, FOV_y)/FWHM, return the smallest.
    This ensures equal-beams cropping across all sources.
    """
    n_beams = []
    for tfile in Path(root_dir).rglob("*T50kpc.fits"):
        try:
            from astropy.io import fits
            with fits.open(tfile) as hdul:
                h = hdul[0].header
            fwhm = max(float(h["BMAJ"]), float(h["BMIN"])) * 3600.0
            # pixel scale
            if "CD1_1" in h:
                cd11 = h["CD1_1"]; cd22 = h.get("CD2_2", 0)
                cd12 = h.get("CD1_2", 0); cd21 = h.get("CD2_1", 0)
                dx = np.hypot(cd11, cd21)
                dy = np.hypot(cd12, cd22)
                asx = dx * 206264.806; asy = dy * 206264.806
            else:
                asx = abs(h.get("CDELT1",1))*3600.0
                asy = abs(h.get("CDELT2",1))*3600.0
            fovx = int(h["NAXIS1"])*asx
            fovy = int(h["NAXIS2"])*asy
            nb = min(fovx,fovy)/max(fwhm,1e-9)
            n_beams.append(nb)
        except Exception as e:
            print(f"[scan] skip {tfile}: {e}")
    if not n_beams:
        print("[scan] No T50kpc files found → default to None")
        return None
    nmin = min(n_beams)
    print(f"[scan] Using n_beams = {nmin:.2f} (smallest across {len(n_beams)} T50kpc frames)")
    return nmin


def crop_to_side_arcsec_on_raw(I, Hraw, side_arcsec, *arrs, center=None):
    """
    Square crop on RAW grid with side length in arcsec. If `center` is given,
    it must be (cy,cx) in INPUT pixels on RAW; otherwise use the image centre.
    This is the equal-beams cropping method from 0.3.0.
    """
    asx, asy = arcsec_per_pix(Hraw)
    nx = int(round(side_arcsec / asx))
    ny = int(round(side_arcsec / asy))
    m  = max(1, min(nx, ny))
    nx = min(m, I.shape[1]); ny = min(m, I.shape[0])

    if center is None:
        cy, cx = (I.shape[0] - 1)/2.0, (I.shape[1] - 1)/2.0
    else:
        cy, cx = float(center[0]), float(center[1])

    y0 = int(round(cy - ny/2)); x0 = int(round(cx - nx/2))
    y0 = max(0, min(y0, I.shape[0] - ny)); x0 = max(0, min(x0, I.shape[1] - nx))
    cy_eff, cx_eff = y0 + ny/2.0, x0 + nx/2.0

    out = [a[y0:y0+ny, x0:x0+nx] for a in (I,) + arrs]
    return out, (ny, nx), (cy_eff, cx_eff)


# ---------------------- Source Classifications ----------------------

def get_classified_sources_from_loader(root: Path, scales: List[float]) -> Tuple[List[str], List[str]]:
    """
    Use the existing load_galaxies function to discover DE and NDE sources.
    """
    try:
        from utils.data_loader import load_galaxies
        
        print("Loading DE sources (class 50) from RAW files...")
        _, _, _, _, de_train_fns, de_eval_fns = load_galaxies(
            galaxy_classes=[50],
            versions='RAW',
            fold=0,
            train=False,
            NORMALISE=False,
            AUGMENT=False,
            BALANCE=False,
            PRINTFILENAMES=True,
            USE_CACHE=False,
            DEBUG=False,
            PREFER_PROCESSED=False
        )
        de_sources = sorted(set(de_train_fns + de_eval_fns))
        
        print("Loading NDE sources (class 51) from RAW files...")
        _, _, _, _, nde_train_fns, nde_eval_fns = load_galaxies(
            galaxy_classes=[51],
            versions='RAW',
            fold=0,
            train=False,
            NORMALISE=False,
            AUGMENT=False,
            BALANCE=False,
            PRINTFILENAMES=True,
            USE_CACHE=False,
            DEBUG=False,
            PREFER_PROCESSED=False
        )
        nde_sources = sorted(set(nde_train_fns + nde_eval_fns))
        
        # Validate that sources have all required scales
        de_valid = []
        nde_valid = []
        
        for src in de_sources:
            if _validate_source_has_scales(src, root, scales):
                de_valid.append(src)
            else:
                print(f"Skipping DE source {src}: missing required T_X files")
        
        for src in nde_sources:
            if _validate_source_has_scales(src, root, scales):
                nde_valid.append(src)
            else:
                print(f"Skipping NDE source {src}: missing required T_X files")
        
        return de_valid, nde_valid
        
    except Exception as e:
        print(f"Error loading from load_galaxies: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def _validate_source_has_scales(source_name: str, root: Path, scales: List[float]) -> bool:
    """Check if a source has all required T_X files."""
    fits_root = Path(str(root).replace('/classified/', '/fits/'))
    src_dir = fits_root / source_name
    
    if not src_dir.exists():
        return False
    
    # Check RAW
    raw_path = src_dir / f"{source_name}.fits"
    if not raw_path.exists():
        return False
    
    # Check all T_X files
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        t_path = src_dir / f"{source_name}T{scale_int}kpc.fits"
        if not t_path.exists():
            print(f"  Missing {t_path.name} for {source_name}")
            return False
    
    return True


def validate_sources_have_all_files(sources: List[str], root: Path, scales: List[float]) -> List[str]:
    """Filter sources to only include those that have all required files."""
    valid_sources = []
    
    for source_name in sources:
        src_dir = root / source_name
        
        # Check for RAW file
        raw_path = src_dir / f"{source_name}.fits"
        if not raw_path.exists():
            print(f"Skipping {source_name}: RAW file not found")
            continue
        
        # Check for all T_X files
        has_all_scales = True
        for scale in scales:
            scale_int = int(scale) if float(scale).is_integer() else scale
            t_path = src_dir / f"{source_name}T{scale_int}kpc.fits"
            if not t_path.exists():
                print(f"Skipping {source_name}: T{scale_int}kpc file not found")
                has_all_scales = False
                break
        
        if has_all_scales:
            valid_sources.append(source_name)
    
    return valid_sources


def select_valid_random_sources(root: Path, scales: List[float], 
                                n_de: int = 3, n_nde: int = 3, 
                                max_attempts: int = 10) -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly select valid sources from DE and NDE classes.
    """
    print("No seed specified - sources will be randomly selected (different each run)")
    
    print("Discovering sources using load_galaxies...")
    de_sources, nde_sources = get_classified_sources_from_loader(root, scales)
    
    print(f"Found {len(de_sources)} valid DE sources")
    print(f"Found {len(nde_sources)} valid NDE sources")
    
    if de_sources:
        print(f"  DE samples: {', '.join(de_sources[:5])}" + (" ..." if len(de_sources) > 5 else ""))
    if nde_sources:
        print(f"  NDE samples: {', '.join(nde_sources[:5])}" + (" ..." if len(nde_sources) > 5 else ""))
    
    # Select from valid sources only
    n_de_actual = min(n_de, len(de_sources))
    n_nde_actual = min(n_nde, len(nde_sources))
    
    # Shuffle and select
    random.shuffle(de_sources)
    random.shuffle(nde_sources)
    selected_de = de_sources[:n_de_actual]
    selected_nde = nde_sources[:n_nde_actual]
    
    print(f"\nSelected sources:")
    print(f"  DE: {selected_de}")
    print(f"  NDE: {selected_nde}")
    
    return selected_de + selected_nde, de_sources, nde_sources


def find_source_files(root: Path, source_name: str, scales: List[float]) -> Dict[str, Path]:
    """Find RAW and T_X files for a given source."""
    files = {}
    src_dir = root / source_name
    
    # Find RAW file
    raw_path = src_dir / f"{source_name}.fits"
    if not raw_path.exists():
        raise ValueError(f"RAW file not found: {raw_path}")
    files['raw'] = raw_path
    
    # Find T_X files for each requested scale
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        t_path = src_dir / f"{source_name}T{scale_int}kpc.fits"
        if not t_path.exists():
            raise ValueError(f"T{scale_int}kpc file not found: {t_path}")
        files[f'T{scale_int}'] = t_path
    
    return files


def create_rt_image(raw_arr: np.ndarray, raw_hdr, z: float, fwhm_kpc: float) -> np.ndarray:
    """
    Create RT image by convolving RAW with circular kernel based on REDSHIFT.
    This is the generalizable, header-free method from 0.3.0 (circular mode).
    
    Args:
        raw_arr: RAW image array
        raw_hdr: RAW FITS header
        z: Redshift
        fwhm_kpc: Physical size of circular kernel in kpc
    
    Returns:
        RT image on RAW grid
    """
    # Build circular kernel from redshift (header-free, generalizable)
    ker = circular_kernel_from_z(z, raw_hdr, fwhm_kpc=fwhm_kpc)
    
    # Convolve RAW with circular kernel
    I_smt = convolve_fft(
        raw_arr, ker, boundary="fill", fill_value=np.nan,
        nan_treatment="interpolate", normalize_kernel=True,
        psf_pad=True, fft_pad=True, allow_huge=True
    )
    
    # Rescale flux (Jy/beam_raw → Jy/beam_target)
    # For circular kernel at physical scale, use same solid angle scaling
    # (this assumes T_X has similar beam area to circular target)
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
    
    # Calculate circular target beam area from physical scale
    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_rad = (fwhm_kpc / DA_kpc)
    sigma_rad = theta_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # Circular Gaussian solid angle
    omega_tgt = 2.0 * np.pi * sigma_rad**2
    omega_raw = beam_solid_angle_sr(raw_hdr)
    scale = omega_tgt / omega_raw
    
    RT = I_smt * scale  # Jy/beam_tgt
    
    return RT


def load_redshift(source_name: str, z_table_path: Path) -> float:
    """
    Load redshift for a given source from the redshift table.
    
    Args:
        source_name: Source name (e.g., "PSZ2G048.10+57.16")
        z_table_path: Path to redshift CSV file
    
    Returns:
        Redshift (float), or raises ValueError if not found
    """
    z_table = load_z_table(z_table_path)
    
    if source_name not in z_table:
        raise ValueError(f"Redshift not found for {source_name}")
    
    z = z_table[source_name]
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"Invalid redshift z={z} for {source_name}")
    
    return z


def process_source(source_name: str, 
                   root: Path,
                   scales: List[float],
                   z_table_path: Path,
                   global_nbeams: float,
                   fov_arcmin: float = 50.0,
                   downsample_size: Tuple[int, int] = (128, 128),
                   processed_dir: Path = Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits"), 
                   prefer_processed: bool = True) -> Dict[str, np.ndarray]:
    """
    Process a single source to generate all required images.
    """
    
    # ALWAYS try to load from processed files first
    if processed_dir and processed_dir.exists():
        Ho, Wo = downsample_size
        results = {}
        all_found = True
        
        print(f"  Checking for processed files for {source_name}...")
        
        # Try to load all versions from processed directory
        for scale in scales:
            scale_int = int(scale) if float(scale).is_integer() else scale
            
            # Check for RAW
            raw_proc = processed_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_circular.fits"
            if not raw_proc.exists():
                print(f"    Missing: {raw_proc.name}")
                all_found = False
                break
                
            # Check for T_X
            t_proc = processed_dir / f"{source_name}_T{scale_int}kpc_fmt_{Ho}x{Wo}_circular.fits"
            if not t_proc.exists():
                print(f"    Missing: {t_proc.name}")
                all_found = False
                break
                
            # Check for RT_X
            rt_proc = processed_dir / f"{source_name}_RT{scale_int}kpc_fmt_{Ho}x{Wo}_circular.fits"
            if not rt_proc.exists():
                print(f"    Missing: {rt_proc.name}")
                all_found = False
                break
        
        # If all files exist, load them
        if all_found:
            print(f"  ✓ Loading {source_name} from processed files")
            
            # Load RAW
            raw_arr = np.squeeze(fits.getdata(raw_proc)).astype(np.float32)
            if raw_arr.ndim == 3:
                raw_arr = raw_arr.mean(axis=0)
            results['RAW'] = raw_arr
            
            # Load T_X and RT_X for each scale
            for scale in scales:
                scale_int = int(scale) if float(scale).is_integer() else scale
                
                t_proc = processed_dir / f"{source_name}_T{scale_int}kpc_fmt_{Ho}x{Wo}_circular.fits"
                t_arr = np.squeeze(fits.getdata(t_proc)).astype(np.float32)
                if t_arr.ndim == 3:
                    t_arr = t_arr.mean(axis=0)
                results[f'T{scale_int}'] = t_arr
                
                rt_proc = processed_dir / f"{source_name}_RT{scale_int}kpc_fmt_{Ho}x{Wo}_circular.fits"
                rt_arr = np.squeeze(fits.getdata(rt_proc)).astype(np.float32)
                if rt_arr.ndim == 3:
                    rt_arr = rt_arr.mean(axis=0)
                results[f'RT{scale_int}'] = rt_arr
            
            return results
        else:
            print(f"  ✗ Some processed files missing, will generate from raw")
            
    print("No processed files found or preference disabled - processing from raw FITS files")
    
    # Load redshift
    z = load_redshift(source_name, z_table_path)
    print(f"  Loaded redshift z={z:.4f} for {source_name}")
    
    # Find all necessary files
    files = find_source_files(root, source_name, scales)
    
    if 'raw' not in files:
        raise ValueError(f"RAW file not found for {source_name}")
    
    # Load RAW image
    I_raw, H_raw, W_raw = read_fits_array_header_wcs(files['raw'])
    
    results = {}
    
    # Get center coordinates (with optional manual offset)
    header_sky = header_cluster_coord(H_raw)
    if header_sky is None:
        yc, xc = (I_raw.shape[0] - 1) / 2.0, (I_raw.shape[1] - 1) / 2.0
    else:
        x_i, y_i = W_raw.world_to_pixel(header_sky)
        yc, xc = float(y_i), float(x_i)
    
    # Apply manual offset if available
    dy_px, dx_px = OFFSETS_PX.get(source_name, (0.0, 0.0))
    yc += dy_px
    xc += dx_px
    
    # Downsample helper
    def _downsample(arr, Ho, Wo):
        """Downsample array while preserving NaN values."""
        # Create mask of NaN pixels
        nan_mask = np.isnan(arr)
        
        # Temporarily fill NaNs with zeros for interpolation
        arr_filled = np.nan_to_num(arr, nan=0.0)
        
        # Downsample the data
        t = torch.from_numpy(arr_filled).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y = torch.nn.functional.interpolate(t, size=(Ho, Wo), mode="bilinear", align_corners=False)
        result = y.squeeze(0).squeeze(0).cpu().numpy()
        
        # Downsample the mask
        mask_t = torch.from_numpy(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mask_down = torch.nn.functional.interpolate(mask_t, size=(Ho, Wo), mode="bilinear", align_corners=False)
        mask_result = mask_down.squeeze(0).squeeze(0).cpu().numpy()
        
        # Restore NaNs where mask is > 0.5 (majority NaN in that region)
        result[mask_result > 0.5] = np.nan
        
        return result
        
    Ho, Wo = downsample_size
    
    # Process each T_X scale with equal-beams cropping
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        key_t = f'T{scale_int}'
        key_rt = f'RT{scale_int}'
        
        if key_t not in files:
            print(f"Warning: {key_t} file not found for {source_name}")
            continue
        
        # Load T_X image to get its beam FWHM
        T_nat, H_tgt, W_tgt = read_fits_array_header_wcs(files[key_t])
        
        # Use THIS scale's FWHM for cropping
        fwhm_this_scale_as = fwhm_major_as(H_tgt)
        
        # Determine zoom factor based on scale
        if scale <= 30:  # 25kpc
            zoom_factor = 1.42  # zoom OUT
        elif scale >= 90:  # 100kpc
            zoom_factor = 0.49  # zoom IN 
        else:  # 50kpc
            zoom_factor = 1.0  # neutral zoom
        
        # Calculate equal-beams crop side with zoom
        side_as = global_nbeams * fwhm_this_scale_as * zoom_factor
        
        # Reproject T to RAW grid
        T_on_raw = reproject_like(T_nat, H_tgt, H_raw)
        #T_on_raw = np.nan_to_num(T_on_raw, nan=0.0)  # Fill NaNs from reprojection
        
        # Create RT image using circular kernel from redshift (header-free method)
        RT_on_raw = create_rt_image(I_raw, H_raw, z=z, fwhm_kpc=scale)
        #RT_on_raw = np.nan_to_num(RT_on_raw, nan=0.0)  # Fill NaNs from convolution
        
        # Equal-beams crop with zoom-adjusted side length
        (T_crop,), _, _ = crop_to_side_arcsec_on_raw(
            T_on_raw, H_raw, side_as, center=(yc, xc)
        )
        (RT_crop,), _, _ = crop_to_side_arcsec_on_raw(
            RT_on_raw, H_raw, side_as, center=(yc, xc)
        )
        
        # Fill NaN values with zeros to avoid white borders
        #T_crop = np.nan_to_num(T_crop, nan=0.0)
        #RT_crop = np.nan_to_num(RT_crop, nan=0.0)
        
        # Downsample to fixed 128x128 size
        results[key_t] = _downsample(T_crop, Ho, Wo)
        results[key_rt] = _downsample(RT_crop, Ho, Wo)
        
        print(f"  {key_t} (zoom={zoom_factor:.2f}, side={side_as:.1f}\") shape: {results[key_t].shape}")
        print(f"  {key_rt} (zoom={zoom_factor:.2f}, side={side_as:.1f}\") shape: {results[key_rt].shape}")
    
    # Process RAW with equal-beams crop (neutral zoom, using T50's FWHM)
    t50_key = None
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        if scale_int == 50:
            t50_key = f'T{scale_int}'
            break
    
    if t50_key is None or t50_key not in files:
        print(f"Warning: T50kpc not found for {source_name}, using first available scale for RAW")
        scale_int = int(scales[0]) if float(scales[0]).is_integer() else scales[0]
        t50_key = f'T{scale_int}'
    
    _, H_t50, _ = read_fits_array_header_wcs(files[t50_key])
    fwhm_t50_as = fwhm_major_as(H_t50)
    
    side_as_raw = global_nbeams * fwhm_t50_as * 1.0  # neutral zoom for RAW
    (I_crop,), _, _ = crop_to_side_arcsec_on_raw(
        I_raw, H_raw, side_as_raw, center=(yc, xc)
    )
    
    # Fill NaN values with zeros to avoid white borders
    #I_crop = np.nan_to_num(I_crop, nan=0.0)
    
    results['RAW'] = _downsample(I_crop, Ho, Wo)
    print(f"  RAW shape after downsample: {results['RAW'].shape}")
    
    return results


def create_comparison_plot(sources: List[str],
                           de_sources_all: List[str],
                           nde_sources_all: List[str],
                           root: Path,
                           scales: List[float],
                           z_table_path: Path,
                           global_nbeams: float,
                           output_path: Path,
                           fov_arcmin: float = 50.0,
                           downsample_size: Tuple[int, int] = (128, 128),
                           figsize: Tuple[float, float] = (12, 10),
                           dpi: int = 200,
                           add_class_labels: bool = True):
    """
    Create comparison plot showing multiple sources with different processing stages.
    Uses circular kernel from redshift (header-free, generalizable method).
    NO vertical spacing within groups, white space ONLY between DE and NDE groups.
    NO DE/NDE title labels.
    Column titles: "RT{X}kpc" instead of "RT X"
    """
    n_sources = len(sources)
    
    # Determine class for each source and group them
    de_indices = []
    nde_indices = []
    for i, src in enumerate(sources):
        if src in de_sources_all:
            de_indices.append(i)
        elif src in nde_sources_all:
            nde_indices.append(i)
    
    # Build column labels with T and RT paired together
    col_labels = ['RAW']
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        col_labels.append(f'T{scale_int}kpc')
        col_labels.append(f'RT{scale_int}kpc')  # UPDATED: No space in "RT{X}kpc"
    
    n_cols = len(col_labels)
    col_widths = [1.0] * n_cols  # All images are 128x128, equal width
    
    # Create figure with reduced spacing
    fig = plt.figure(figsize=figsize)
    
    import matplotlib.gridspec as gridspec
    
    # Calculate height ratios: no title rows, add space between DE and NDE groups
    height_ratios = []
    for _ in de_indices:
        height_ratios.append(1.0)
    if de_indices and nde_indices:
        height_ratios.append(0.15)  # Gap between DE and NDE
    for _ in nde_indices:
        height_ratios.append(1.0)
    
    # Create GridSpec with NO spacing within groups, only between groups
    n_grid_rows = len(height_ratios)
    gs = gridspec.GridSpec(n_grid_rows, n_cols, figure=fig,
                          hspace=0.0,
                          wspace=0.001,
                          width_ratios=col_widths,
                          height_ratios=height_ratios,
                          left=0.04, right=0.999,
                          top=0.998, bottom=0.002)
    
    # Process each source
    grid_row = 0
    
    # Process DE sources (no label row)
    for i in de_indices:
        source_name = sources[i]
        try:
            print(f"Processing {source_name}...")
            images = process_source(source_name, root, scales, z_table_path, 
                                   global_nbeams, fov_arcmin, downsample_size, prefer_processed=True)
            
            # Verify all images are the same size
            for key, img in images.items():
                if isinstance(img, np.ndarray) and img.shape != (downsample_size[0], downsample_size[1]):
                    print(f"Warning: {source_name} {key} has wrong shape {img.shape}, expected {downsample_size}")
            
            # Plot images in columns with T and RT paired
            col_idx = 0
            
            # RAW - always create subplot
            ax = fig.add_subplot(gs[grid_row, col_idx])
            if 'RAW' in images:
                vmin, vmax = robust_vmin_vmax(images['RAW'])
                # Create a copy and set NaNs to white by using set_bad
                cmap = plt.cm.viridis.copy()
                cmap.set_bad('white', 1.0)  # NaN pixels will be white
                ax.imshow(images['RAW'], origin='lower', vmin=vmin, vmax=vmax, 
                        cmap=cmap, interpolation='nearest')
                ax.axis('off')
                if grid_row == 0:  # First data row
                    ax.set_title('RAW', fontsize=10, fontweight='bold', pad=0)
                # Add source name on left - ROTATED 90 degrees
                ax.text(-0.015, 0.5, source_name, transform=ax.transAxes,
                    fontsize=9, va='center', ha='right', rotation=90)
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                       ha='center', va='center', fontsize=8, color='red')
            col_idx += 1
            
            # T_X and RT_X paired together - always create subplots
            for scale in scales:
                scale_int = int(scale) if float(scale).is_integer() else scale
                key_t = f'T{scale_int}'
                key_rt = f'RT{scale_int}'
                
                # T_X image - always create subplot
                ax = fig.add_subplot(gs[grid_row, col_idx])
                if key_t in images:
                    vmin, vmax = robust_vmin_vmax(images[key_t])
                    ax.imshow(images[key_t], origin='lower', vmin=vmin, vmax=vmax, 
                             cmap='viridis', interpolation='nearest')
                    ax.axis('off')
                    if grid_row == 0:
                        ax.set_title(f'T{scale_int}kpc', fontsize=10, fontweight='bold', pad=0)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                           ha='center', va='center', fontsize=8, color='red')
                col_idx += 1
                
                # RT_X image - always create subplot
                ax = fig.add_subplot(gs[grid_row, col_idx])
                if key_rt in images:
                    vmin, vmax = robust_vmin_vmax(images[key_rt])
                    ax.imshow(images[key_rt], origin='lower', vmin=vmin, vmax=vmax, 
                             cmap='viridis', interpolation='nearest')
                    ax.axis('off')
                    if grid_row == 0:
                        ax.set_title(f'RT{scale_int}kpc', fontsize=10, fontweight='bold', pad=0)  # UPDATED
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                           ha='center', va='center', fontsize=8, color='red')
                col_idx += 1
            
            grid_row += 1
            
        except Exception as e:
            print(f"Error processing {source_name}: {e}")
            import traceback
            traceback.print_exc()
            # Fill row with blank panels
            for j in range(n_cols):
                ax = fig.add_subplot(gs[grid_row, j])
                ax.axis('off')
                ax.text(0.5, 0.5, 'ERROR', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')
            grid_row += 1
    
    # Skip to NDE section (add empty spacing row)
    if de_indices and nde_indices:
        grid_row += 1
    
    # Process NDE sources (no label row)
    for i in nde_indices:
        source_name = sources[i]
        try:
            print(f"Processing {source_name}...")
            images = process_source(source_name, root, scales, z_table_path,
                                   global_nbeams, fov_arcmin, downsample_size, prefer_processed=True)
            
            # Verify all images are the same size
            for key, img in images.items():
                if isinstance(img, np.ndarray) and img.shape != (downsample_size[0], downsample_size[1]):
                    print(f"Warning: {source_name} {key} has wrong shape {img.shape}, expected {downsample_size}")
            
            # Plot images in columns with T and RT paired
            col_idx = 0
            
            # RAW - always create subplot
            ax = fig.add_subplot(gs[grid_row, col_idx])
            if 'RAW' in images:
                vmin, vmax = robust_vmin_vmax(images['RAW'])
                ax.imshow(images['RAW'], origin='lower', vmin=vmin, vmax=vmax, 
                         cmap='viridis', interpolation='nearest')
                ax.axis('off')
                # Add source name on left - ROTATED 90 degrees
                ax.text(-0.02, 0.5, source_name, transform=ax.transAxes,
                       fontsize=9, va='center', ha='right', rotation=90)
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                       ha='center', va='center', fontsize=8, color='red')
            col_idx += 1
            
            # T_X and RT_X paired together - always create subplots
            for scale in scales:
                scale_int = int(scale) if float(scale).is_integer() else scale
                key_t = f'T{scale_int}'
                key_rt = f'RT{scale_int}'
                
                # T_X image - always create subplot
                ax = fig.add_subplot(gs[grid_row, col_idx])
                if key_t in images:
                    vmin, vmax = robust_vmin_vmax(images[key_t])
                    ax.imshow(images[key_t], origin='lower', vmin=vmin, vmax=vmax, 
                             cmap='viridis', interpolation='nearest')
                    ax.axis('off')
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                           ha='center', va='center', fontsize=8, color='red')
                col_idx += 1
                
                # RT_X image - always create subplot
                ax = fig.add_subplot(gs[grid_row, col_idx])
                if key_rt in images:
                    vmin, vmax = robust_vmin_vmax(images[key_rt])
                    ax.imshow(images[key_rt], origin='lower', vmin=vmin, vmax=vmax, 
                             cmap='viridis', interpolation='nearest')
                    ax.axis('off')
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                           ha='center', va='center', fontsize=8, color='red')
                col_idx += 1
            
            grid_row += 1
            
        except Exception as e:
            print(f"Error processing {source_name}: {e}")
            import traceback
            traceback.print_exc()
            # Fill row with blank panels
            for j in range(n_cols):
                ax = fig.add_subplot(gs[grid_row, j])
                ax.axis('off')
                ax.text(0.5, 0.5, 'ERROR', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')
            grid_row += 1
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    
    print(f"Saved comparison plot to {output_path}")


def main():
    """Main entry point for the script."""
    ap = argparse.ArgumentParser(
        description="Create comparison plot of RAW, T_X, and RT_X images (using circular kernel from redshift)"
    )
    
    # Default paths
    DEFAULT_ROOT = Path("/users/mbredber/scratch/data/PSZ2/fits")
    DEFAULT_OUT = Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/comparison_plot.pdf")
    DEFAULT_Z_CSV = Path("/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv")
    
    # Arguments
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"Root directory with per-source subfolders. Default: {DEFAULT_ROOT}")
    ap.add_argument("--z-csv", type=Path, default=DEFAULT_Z_CSV,
                   help=f"CSV file with redshift table. Default: {DEFAULT_Z_CSV}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output PNG file path. Default: {DEFAULT_OUT}")
    ap.add_argument("--sources", type=str, default=None,
                   help="Comma-separated list of source names. If not provided, randomly selects 3 DE + 3 NDE sources.")
    ap.add_argument("--n-de", type=int, default=3,
                   help="Number of DE sources to randomly select. Default: 3")
    ap.add_argument("--n-nde", type=int, default=3,
                   help="Number of NDE sources to randomly select. Default: 3")
    ap.add_argument("--seed", type=int, default=10,
                   help="Random seed for source selection. Default: 10")
    ap.add_argument("--scales", type=str, default="25,50,100",
                   help="Comma-separated kpc scales. Default: 25,50,100")
    ap.add_argument("--fov-arcmin", type=float, default=50.0,
                   help="Field of view in arcminutes (UNUSED). Default: 50.0")
    ap.add_argument("--size", type=str, default="128,128",
                   help="Downsample size as H,W. Default: 128,128")
    ap.add_argument("--figsize", type=str, default="10,9",
                   help="Figure size as width,height in inches. Default: 10,9")
    ap.add_argument("--dpi", type=int, default=200,
                   help="Output DPI. Default: 200")
    ap.add_argument("--no-class-labels", action="store_true",
                   help="Ignored - kept for compatibility")
    ap.add_argument("--processed-dir", type=Path, 
                   default=Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits"),
                   help="Directory containing processed FITS files")
    ap.add_argument("--prefer-processed", action="store_true", default=True,
                   help="Use processed files if available")
    
    args = ap.parse_args()
    
    # Parse scales first
    scales = [float(s.strip()) for s in args.scales.split(',') if s.strip()]
    
    # Compute global minimum beam count for equal-beams cropping
    print("Computing global minimum beam count for equal-beams cropping...")
    global_nbeams = compute_global_nbeams_min(args.root)
    if global_nbeams is None:
        print("ERROR: Could not compute global beam count!")
        return
    # Apply scaling factor (matching 0.3.0)
    global_nbeams *= 1.85
    print(f"Using global_nbeams = {global_nbeams:.2f} (with 1.85x scaling factor)")
    
    # Parse arguments
    if args.sources is not None:
        # User provided explicit source list
        sources = [s.strip() for s in args.sources.split(',') if s.strip()]
        print(f"Using user-provided sources: {', '.join(sources)}")
        # Discover all sources to determine DE/NDE classification
        print("Discovering all sources for classification...")
        de_sources_all, nde_sources_all = get_classified_sources_from_loader(args.root, scales)
        # Validate user sources exist
        valid_sources = []
        for src in sources:
            if src in de_sources_all or src in nde_sources_all:
                valid_sources.append(src)
            else:
                print(f"Warning: {src} not found in classified sources, skipping")
        sources = valid_sources
        if not sources:
            print("ERROR: None of the provided sources are valid!")
            return
    else:
        # Randomly select sources
        print("Selecting random sources with seed:", args.seed)
        sources, de_sources_all, nde_sources_all = select_valid_random_sources(
            root=args.root,
            scales=scales,
            n_de=args.n_de,
            n_nde=args.n_nde,
        )
        if not sources:
            print("ERROR: No valid sources found with all required files!")
            return
        print(f"Randomly selected {len([s for s in sources if s in de_sources_all])} DE + {len([s for s in sources if s in nde_sources_all])} NDE sources:")
        de_selected = [s for s in sources if s in de_sources_all]
        nde_selected = [s for s in sources if s in nde_sources_all]
        if de_selected:
            print(f"  DE:  {', '.join(de_selected)}")
        if nde_selected:
            print(f"  NDE: {', '.join(nde_selected)}")
        if args.seed is not None:
            print(f"  (using seed={args.seed})")
    
    downsample_size = tuple(int(x) for x in args.size.split(','))
    figsize = tuple(float(x) for x in args.figsize.split(','))
    
    if len(downsample_size) != 2:
        raise ValueError("--size must be H,W (e.g., '128,128')")
    if len(figsize) != 2:
        raise ValueError("--figsize must be width,height (e.g., '15,12')")
    
    # Create plot
    create_comparison_plot(
        sources=sources,
        de_sources_all=de_sources_all,
        nde_sources_all=nde_sources_all,
        root=args.root,
        scales=scales,
        z_table_path=args.z_csv,
        global_nbeams=global_nbeams,
        output_path=args.out,
        fov_arcmin=args.fov_arcmin,
        downsample_size=downsample_size,
        figsize=figsize,
        dpi=args.dpi,
        add_class_labels=False  # Always False now
    )


if __name__ == "__main__":
    main()