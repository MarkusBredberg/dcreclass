#!/usr/bin/env python3
"""
Merged batch script with beam-sensitive cropping:
- For each source, load RAW, T_X, and T_XSUB (X in {25,50,100,...} kpc).
- Build RT_X by convolving RAW with a CIRCULAR X kpc kernel based on redshift.
  (e.g., RT25kpc uses a 25 kpc circular kernel, RT100kpc uses a 100 kpc circular kernel)
- Make a 4×2 montage (rows: RAW, RT_X, T_X, T_XSUB ; cols: original vs beam-cropped).
  Note: SUB only exists as T_XSUB (point-source-subtracted tapered image), not for RAW or RT.
- Save beam-cropped RAW/RT_X/T_X/T_XSUB FITS with updated WCS.

The circular kernel size is determined by the scale parameter (--scales), not from T_X header.
Beam-cropping: FOV determined by n_beams * FWHM, ensuring consistent angular coverage across redshifts.
"""

import argparse, torch, os, io, csv, re
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve_fft

# ------------------- manual per-source centre offsets (INPUT pixels) -------------------
OFFSETS_PX: Dict[str, Tuple[int,int]] = {
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

# ---------------------------- FITS / WCS utilities ----------------------------
ARCSEC_PER_RAD = 206264.80624709636 # 180.0 * 3600.0 / pi

def _cd_matrix_rad(h):
    """
    Extract 2×2 pixel→world Jacobian in radians/pixel from FITS header.
    Supports both CD and PC/CDELT conventions.
    """
    if 'CD1_1' in h:
        M = np.array([[h['CD1_1'], h.get('CD1_2', 0.0)],
                      [h.get('CD2_1', 0.0), h['CD2_2']]], float)
    else:
        pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
        pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
        cd1 =h.get('CDELT1', 1.0); cd2 =h.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    return M * (np.pi/180.0)

def arcsec_per_pix(h):
    """
    Compute effective pixel scales (|dx|, |dy|) in arcseconds from header WCS.
    """
    J = _cd_matrix_rad(h)
    dx = np.hypot(J[0,0], J[1,0])
    dy = np.hypot(J[0,1], J[1,1])
    return dx*ARCSEC_PER_RAD, dy*ARCSEC_PER_RAD

def fwhm_major_as(h):
    """
    Return the major axis FWHM from the FITS header beam in arcseconds.
    """
    return max(float(h["BMAJ"]), float(h["BMIN"])) * 3600.0

def _fwhm_as_to_sigma_rad(fwhm_as: float) -> float:
    """
    Convert FWHM [arcsec] → Gaussian sigma [radians].
    FWHM = 2*sqrt(2*ln 2)*sigma.
    """
    return (float(fwhm_as)/(2.0*np.sqrt(2.0*np.log(2.0)))) * (np.pi/(180.0*3600.0))

def beam_cov_world(h):
    """
    Return 2×2 covariance (σ^2) in world radians for the header beam.
    Accounts for elliptical beam with major/minor axes and position angle.
    """
    bmaj_as = abs(float(h['BMAJ']))*3600.0
    bmin_as = abs(float(h['BMIN']))*3600.0
    pa_deg  = float(h.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th      = np.deg2rad(pa_deg)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], float)
    S = np.diag([sx*sx, sy*sy])
    return R @ S @ R.T

def beam_solid_angle_sr(h):
    """
    Gaussian beam solid angle in steradians from BMAJ/BMIN [deg] in FITS header.
    """
    bmaj = abs(float(h['BMAJ'])) * np.pi/180.0
    bmin = abs(float(h['BMIN'])) * np.pi/180.0
    return (np.pi/(4.0*np.log(2.0))) * bmaj * bmin

def kernel_from_beams(raw_hdr, tgt_hdr):
    """
    Elliptical Gaussian kernel that maps RAW beam -> TARGET beam.
    In world coords: C_ker = C_tgt - C_raw (clip to PSD), then map to pixel coords.
    
    This is the "old method" used when cheat_rt=True, which derives the kernel
    from the T_X FITS header beam (elliptical with position angle).
    """
    C_raw = beam_cov_world(raw_hdr)
    C_tgt = beam_cov_world(tgt_hdr)
    C_ker_world = C_tgt - C_raw
    # PSD clip
    w, V = np.linalg.eigh(C_ker_world)
    w = np.clip(w, 0.0, None)
    C_ker_world = (V * w) @ V.T

    # world -> pixel: x_pix = J^{-1} x_world
    J    = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker_world @ Jinv.T
    wp, Vp = np.linalg.eigh(Cpix)
    wp = np.clip(wp, 1e-18, None)
    s_minor = float(np.sqrt(wp[0]))
    s_major = float(np.sqrt(wp[1]))
    theta   = float(np.arctan2(Vp[1,1], Vp[0,1]))
    nker    = int(np.ceil(8.0*max(s_major, s_minor))) | 1
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

def circular_cov_kpc(z, fwhm_kpc=50.0):
    """
    Return a circular 2×2 covariance matrix in world coords (radians^2)
    corresponding to a Gaussian with FWHM=fwhm_kpc at redshift z (kpc physical).
    
    This uses the angular diameter distance to convert physical kpc to angular size.
    """
    if z is None or not np.isfinite(z) or z <= 0:
        return None
    fwhm_kpc = float(fwhm_kpc)
    if fwhm_kpc <= 0:
        raise ValueError("fwhm_kpc must be positive")
    # angular diameter distance DA in kpc
    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_rad = (fwhm_kpc / DA_kpc)                    # small-angle in radians
    sigma = theta_rad / (2.0*np.sqrt(2.0*np.log(2.0))) # convert FWHM to sigma
    sigma2 = float(sigma**2)
    return np.array([[sigma2, 0.0],[0.0, sigma2]], float)

def load_z_table(csv_path):
    """
    Load redshift table from CSV file.
    Expected columns: 'slug' (source name) and 'z' (redshift).
    Returns dict mapping slug → z (float or np.nan).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Redshift table not found: {csv_path}")

    # Read the file and keep only non-empty lines; handle UTF-8 BOM
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()

    if not raw.strip():
        raise ValueError(
            "Redshift table appears empty after reading: "
            f"{csv_path} (first 80 bytes repr={raw[:80]!r})"
        )

    # Drop blank lines and comment lines
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError(
            "Redshift table has headers but no data rows after filtering. "
            f"First line: {lines[0]!r}" if lines else "No usable lines found."
        )

    # Parse via DictReader from the cleaned text
    rdr = csv.DictReader(io.StringIO("\n".join(lines)))
    if rdr.fieldnames is None or "slug" not in rdr.fieldnames or "z" not in rdr.fieldnames:
        raise ValueError(f"CSV missing required headers 'slug,z' in {csv_path}; got {rdr.fieldnames!r}")

    out = {}
    for row in rdr:
        slug = (row.get("slug") or "").strip()
        zstr = (row.get("z") or "").strip()
        if not slug:
            continue
        if zstr.lower() in ("", "nan", "none"):
            out[slug] = np.nan
        else:
            try:
                out[slug] = float(zstr)
            except Exception:
                out[slug] = np.nan

    if not out:
        raise ValueError(f"No rows parsed from {csv_path}")
    return out

def circular_kernel_from_z(z, raw_hdr, fwhm_kpc=50.0):
    """
    Build a circular Gaussian convolution kernel on the RAW pixel grid.
    
    The kernel convolves RAW beam → circular 50 kpc target beam.
    Uses redshift to compute physical size, then maps to pixel coordinates.
    
    Returns: Gaussian2DKernel for astropy.convolution
    """
    # Get RAW beam covariance in world coordinates (radians^2)
    C_raw = beam_cov_world(raw_hdr)
    
    # Get circular 50 kpc covariance in world coordinates (radians^2)
    C_circ = circular_cov_kpc(z, fwhm_kpc=fwhm_kpc)
    if C_circ is None:
        raise ValueError(f"Invalid redshift z={z} for circular kernel")
    
    # Kernel covariance: C_ker = C_circ - C_raw (in world coords)
    # We need to convolve RAW with this kernel to get the circular target
    C_ker_world = C_circ - C_raw
    
    # Clip to positive semi-definite (handle numerical issues)
    w, V = np.linalg.eigh(C_ker_world)
    w = np.clip(w, 0.0, None)
    C_ker_world = (V * w) @ V.T

    # Map world → pixel: x_pix = J^{-1} x_world
    J = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker_world @ Jinv.T
    
    # Extract eigenvalues/vectors to get ellipse parameters in pixel coords
    wp, Vp = np.linalg.eigh(Cpix)
    wp = np.clip(wp, 1e-18, None)
    s_minor = float(np.sqrt(wp[0]))  # sigma along minor axis [pixels]
    s_major = float(np.sqrt(wp[1]))  # sigma along major axis [pixels]
    theta   = float(np.arctan2(Vp[1,1], Vp[0,1]))  # rotation angle [radians]
    
    # Kernel size: ~8σ captures most of the Gaussian
    nker = int(np.ceil(8.0*max(s_major, s_minor))) | 1  # ensure odd size
    
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

def read_fits_array_header_wcs(fpath: Path):
    """
    Read FITS file and return (2D array, header, 2D WCS).
    Squeezes singleton dimensions and averages over frequency/Stokes if needed.
    """
    with fits.open(fpath, memmap=False) as hdul:
        header = hdul[0].header
        wcs_full = WCS(header)
        wcs2d = wcs_full.celestial if hasattr(wcs_full, "celestial") else WCS(header, naxis=2)
        arr = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                arr = np.asarray(hdu.data)
                break
    if arr is None: raise RuntimeError(f"No data-containing HDU in {fpath}")
    arr = np.squeeze(arr)
    if arr.ndim == 3: arr = np.nanmean(arr, axis=0)  # average over frequency/Stokes
    if arr.ndim != 2: raise RuntimeError(f"Expected 2D image; got {arr.shape}")
    return arr.astype(np.float32), header, wcs2d

def reproject_like(arr: np.ndarray, src_hdr, dst_hdr) -> np.ndarray:
    """
    Reproject array from source WCS to destination WCS.
    Tries to use reproject package if available, otherwise falls back to scipy zoom.
    """
    try:
        from reproject import reproject_interp
        w_src = (WCS(src_hdr).celestial if hasattr(WCS(src_hdr), "celestial") else WCS(src_hdr, naxis=2))
        w_dst = (WCS(dst_hdr).celestial if hasattr(WCS(dst_hdr), "celestial") else WCS(dst_hdr, naxis=2))
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        out, _ = reproject_interp((arr, w_src), w_dst, shape_out=(ny_out, nx_out), order='bilinear')
        return out.astype(np.float32)
    except Exception:
        # Fallback: simple zoom to target shape
        from scipy.ndimage import zoom as _zoom
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        ny_in, nx_in = arr.shape
        zy = ny_out / max(ny_in, 1); zx = nx_out / max(nx_in, 1)
        y = _zoom(arr, zoom=(zy, zx), order=1)
        y = y[:ny_out, :nx_out]
        if y.shape != (ny_out, nx_out):
            pad_y = ny_out - y.shape[0]; pad_x = nx_out - y.shape[1]
            y = np.pad(y, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode='edge')[:ny_out, :nx_out]
        return y.astype(np.float32)

def header_cluster_coord(header) -> Optional[SkyCoord]:
    """
    Extract cluster sky coordinates from FITS header.
    Tries multiple header keywords in order of preference.
    """
    if header.get('OBJCTRA') and header.get('OBJCTDEC'):
        return SkyCoord(header['OBJCTRA'], header['OBJCTDEC'], unit=(u.hourangle, u.deg))
    if header.get('RA_TARG') and header.get('DEC_TARG'):
        return SkyCoord(header['RA_TARG']*u.deg, header['DEC_TARG']*u.deg)
    if 'CRVAL1' in header and 'CRVAL2' in header:
        return SkyCoord(header['CRVAL1']*u.deg, header['CRVAL2']*u.deg)
    return None

def wcs_after_center_crop_and_resize(header, H0, W0, Hc, Wc, Ho, Wo, y0, x0):
    """
    Update WCS header after center crop and resize operations.
    
    Parameters:
    - header: Original FITS header
    - H0, W0: Original image dimensions
    - Hc, Wc: Crop dimensions
    - Ho, Wo: Output (resized) dimensions
    - y0, x0: Crop center coordinates
    
    Returns: (updated WCS, updated header)
    """
    y1, y2 = max(0, y0 - Hc // 2), min(H0, y0 + Hc // 2)
    x1, x2 = max(0, x0 - Wc // 2), min(W0, x0 + Wc // 2)
    width  = x2 - x1
    height = y2 - y1
    sx = width  / float(Wo)  # x scale factor
    sy = height / float(Ho)  # y scale factor

    new = header.copy()
    # Update reference pixel (CRPIX) accounting for crop offset and rescaling
    if "CRPIX1" in new and "CRPIX2" in new:
        new["CRPIX1"] = (new["CRPIX1"] - x1) / sx
        new["CRPIX2"] = (new["CRPIX2"] - y1) / sy

    # Update pixel scale (CD matrix or CDELT keywords)
    if all(k in new for k in ("CD1_1","CD1_2","CD2_1","CD2_2")):
        new["CD1_1"] *= sx; new["CD1_2"] *= sy
        new["CD2_1"] *= sx; new["CD2_2"] *= sy
    else:
        if "CDELT1" in new: new["CDELT1"] *= sx
        if "CDELT2" in new: new["CDELT2"] *= sy

    new["NAXIS1"] = Wo; new["NAXIS2"] = Ho
    wcs_new = (WCS(new).celestial if hasattr(WCS(new), "celestial") else WCS(new, naxis=2))
    return wcs_new, new

def robust_vmin_vmax(arr: np.ndarray, lo=1, hi=99):
    """
    Compute robust min/max from percentiles for display scaling.
    """
    finite = np.isfinite(arr)
    if not finite.any(): return 0.0, 1.0
    vals = arr[finite]
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if vmin == vmax: vmax = vmin + 1.0
    return float(vmin), float(vmax)

# ------------------------------- formatting ----------------------------------
def _canon_size(sz):
    """
    Canonicalize size specification to (C, H, W) tuple.
    """
    if isinstance(sz, (tuple, list)):
        if len(sz) == 2:  return (1, sz[0], sz[1])
        if len(sz) == 3:  return (sz[0], sz[1], sz[2])
    raise ValueError("size must be H,W or C,H,W")

def crop_to_fov_on_raw(I, Hraw, fov_arcmin, *arrs, center=None):
    """
    Square crop on RAW grid with side=fov_arcmin (arcmin). 
    
    If `center` is given, it must be (cy,cx) in INPUT pixels on RAW; 
    otherwise use the image centre.
    """
    asx, asy = arcsec_per_pix(Hraw)
    fov_as   = float(fov_arcmin) * 60.0
    nx_crop  = int(round(fov_as / asx))
    ny_crop  = int(round(fov_as / asy))
    m        = min(nx_crop, ny_crop)
    nx_crop  = min(m, I.shape[1])
    ny_crop  = min(m, I.shape[0])

    if center is None:
        cy, cx = (I.shape[0] - 1)/2.0, (I.shape[1] - 1)/2.0
    else:
        cy, cx = float(center[0]), float(center[1])

    # clamp the requested centre so the crop stays inside bounds
    y0 = int(round(cy - ny_crop/2)); x0 = int(round(cx - nx_crop/2))
    y0 = max(0, min(y0, I.shape[0] - ny_crop)); x0 = max(0, min(x0, I.shape[1] - nx_crop))
    cy_eff, cx_eff = y0 + ny_crop/2.0, x0 + nx_crop/2.0

    out = [a[y0:y0+ny_crop, x0:x0+nx_crop] for a in (I,) + arrs]
    return out, (ny_crop, nx_crop), (cy_eff, cx_eff)

def crop_to_side_arcsec_on_raw(I, Hraw, side_arcsec, *arrs, center=None):
    """
    Square crop on RAW grid with side length in arcsec. 
    
    If `center` is given, it must be (cy,cx) in INPUT pixels on RAW; 
    otherwise use the image centre.
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


# ------------------------------ IO helpers -----------------------------------
def find_pairs_in_tree(root: Path, desired_kpc: float) -> Iterable[Tuple[str, Path, Path, Path, float]]:
    """
    Yield (name, raw_path, t_path, sub_path, chosen_kpc) where chosen_kpc is the available T scale
    nearest to desired_kpc among files like <name>T{Y}kpc.fits and <name>T{Y}kpcSUB.fits.
    
    Returns:
    - name: Source name
    - raw_path: Path to RAW file (<name>.fits)
    - t_path: Path to T_X file (<name>T{Y}kpc.fits)
    - sub_path: Path to SUB file (<name>T{Y}kpcSUB.fits), or None if not found
    - chosen_kpc: The Y value chosen (closest to desired_kpc)
    """
    import re
    pat_t = re.compile(r"T([0-9]+(?:\.[0-9]+)?)kpc\.fits$", re.IGNORECASE)
    pat_sub = re.compile(r"T([0-9]+(?:\.[0-9]+)?)kpcSUB\.fits$", re.IGNORECASE)
    
    for src_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        name = src_dir.name
        raw_path = src_dir / f"{name}.fits"
        if not raw_path.exists():
            continue
        
        # Find T_X files
        candidates = []
        for fp in src_dir.glob(f"{name}T*kpc.fits"):
            # Skip SUB files in this pass
            if "SUB" in fp.name.upper():
                continue
            m = pat_t.search(fp.name)
            if m:
                y = float(m.group(1))
                candidates.append((abs(y - desired_kpc), y, fp))
        
        if candidates:
            _, ybest, fbest = sorted(candidates, key=lambda t: (t[0], t[1]))[0]
            
            # Look for corresponding SUB file
            sub_path = src_dir / f"{name}T{int(ybest) if ybest.is_integer() else ybest}kpcSUB.fits"
            if not sub_path.exists():
                sub_path = None
            
            yield name, raw_path, fbest, sub_path, ybest
            

def compute_global_nbeams_min(root_dir):
    """
    Scan all subdirs under root_dir, find every *Tkpc.fits,
    compute n_beams = min(FOV_x, FOV_y)/FWHM, return the smallest.
    
    This is used to ensure consistent cropping across all sources.
    """
    n_beams = []
    for tfile in Path(root_dir).rglob("*T*kpc.fits"):
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
        print("[scan] No T*kpc files found → default to None")
        return None
    nmin = min(n_beams)
    print(f"[scan] Using n_beams = {nmin:.2f} (smallest across {len(n_beams)} T*kpc frames)")
    return nmin

def report_nans(path: Path):
    """
    Report the number of NaN pixels in a FITS file (for debugging).
    """
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=float)
    n = np.isnan(arr).sum()
    if n > 0:
        print(f"[nancheck] {path}: {n} NaNs")

# ------------------------------ montage per source ---------------------------
def make_montage(source_name: str,
                 raw_path: Path,
                 t_path: Path,
                 sub_path: Optional[Path],
                 z: float,
                 rt_label: str,
                 t_label: str,
                 downsample_size=(1, 128, 128),
                 save_fits: bool = False,
                 fov_arcmin: float = 50.0,
                 fwhm_kpc: float = 50.0,
                 cheat_rt: bool = False,
                 y_chosen: Optional[float] = None,
                 force: bool = False,
                 out_png: Optional[Path] = None,
                 out_fits_dir: Optional[Path] = None,
                 ):
                     
    """
    Build RT on RAW grid, then create 3×3 or 4×2 WCS montage with beam-cropped outputs:
      rows: RAW, RT, T_X, (optionally SUB if available)
      cols: original | beam-cropped
    
    Save beam-cropped RAW/RT/T_X/SUB FITS with updated WCS for use in classifier.
    
    Parameters:
    - source_name: Name of the source (for labeling)
    - raw_path: Path to RAW FITS file
    - t_path: Path to T_X FITS file (for comparison)
    - sub_path: Path to T_XSUB FITS file (point-source-subtracted), or None
    - z: Redshift (used to compute circular kernel size when cheat_rt=False)
    - rt_label: Label for RT panel (e.g., "RT50kpc")
    - t_label: Label for T_X panel (e.g., "T50kpc")
    - downsample_size: Output size (C, H, W) for beam-cropped panels
    - save_fits: Whether to save beam-cropped FITS files
    - out_fits_dir: Directory for beam-cropped FITS outputs
    - fov_arcmin: Field of view in arcminutes for beam-cropped column (fallback)
    - fwhm_kpc: Physical size of circular kernel in kpc (default 50, only used when cheat_rt=False)
    - cheat_rt: If True, use T_X header beam (elliptical, old method); 
                if False (default), use circular kernel from redshift
    """
    # --- validate redshift (only required for circular kernel mode) ---
    if not cheat_rt:
        if not np.isfinite(z) or z <= 0:
            raise ValueError(f"Invalid redshift z={z} for {source_name} (required for circular kernel mode)")
    
    # --- load RAW and T_X
    I_raw,  H_raw,  W_raw  = read_fits_array_header_wcs(raw_path)
    T_nat,  H_tgt,  W_tgt  = read_fits_array_header_wcs(t_path)
    
    # --- load SUB if available
    if sub_path is not None and sub_path.exists():
        SUB_nat, H_sub, W_sub = read_fits_array_header_wcs(sub_path)
        has_sub = True
    else:
        SUB_nat, H_sub, W_sub = None, None, None
        has_sub = False
    
    # --- determine equal-beams crop side based on smallest n_beams across T50 set ---
    # (We cache the global minimum once at module level)
    if not hasattr(make_montage, "_global_nbeams"):
        # compute n_beams for this T50
        fwhm_as = fwhm_major_as(H_tgt)
        asx, asy = arcsec_per_pix(H_tgt)
        fovx_as = int(H_tgt["NAXIS1"]) * asx
        fovy_as = int(H_tgt["NAXIS2"]) * asy
        n_beams_here = min(fovx_as, fovy_as) / max(fwhm_as, 1e-9)
        # store first value; will be updated to min across calls
        make_montage._global_nbeams = n_beams_here
    else:
        fwhm_as = fwhm_major_as(H_tgt)
        asx, asy = arcsec_per_pix(H_tgt)
        fovx_as = int(H_tgt["NAXIS1"]) * asx
        fovy_as = int(H_tgt["NAXIS2"]) * asy
        n_beams_here = min(fovx_as, fovy_as) / max(fwhm_as, 1e-9)
        make_montage._global_nbeams = min(make_montage._global_nbeams, n_beams_here)

    # --- reproject T to RAW grid for consistent comparison
    T_on_raw = reproject_like(T_nat, H_tgt, H_raw)
    
    # --- reproject SUB to RAW grid if available
    if has_sub:
        SUB_on_raw = reproject_like(SUB_nat, H_sub, H_raw)
    else:
        SUB_on_raw = None

    # --- Build RT using either CIRCULAR kernel from redshift OR elliptical from header ---
    if cheat_rt:
        # OLD METHOD: Use T_X header beam (elliptical with position angle)
        ker = kernel_from_beams(H_raw, H_tgt)
        
        I_smt = convolve_fft(
            I_raw, ker, boundary="fill", fill_value=np.nan,
            nan_treatment="interpolate", normalize_kernel=True,
            psf_pad=True, fft_pad=True, allow_huge=True
        )
        
        scale = beam_solid_angle_sr(H_tgt) / beam_solid_angle_sr(H_raw)
        RT_rawgrid = I_smt * scale  # Jy/beam_tgt
    else:
        # NEW METHOD (default): Use circular kernel from redshift (no header beam info)
        ker = circular_kernel_from_z(z, H_raw, fwhm_kpc=fwhm_kpc)
        
        # Convolve RAW with the circular kernel
        I_smt = convolve_fft(
            I_raw, ker, boundary="fill", fill_value=np.nan,
            nan_treatment="interpolate", normalize_kernel=True,
            psf_pad=True, fft_pad=True, allow_huge=True
        )
        
        # Rescale flux: Jy/beam_raw → Jy/beam_circular
        # The circular target has the same solid angle as the T_X beam
        scale = beam_solid_angle_sr(H_tgt) / beam_solid_angle_sr(H_raw)
        RT_rawgrid = I_smt * scale  # Jy/beam_tgt
        
    # --- centres from header sky coord (shared if available)
    header_sky = header_cluster_coord(H_raw) or header_cluster_coord(H_tgt)
    if header_sky is None:
        H0_i, W0_i = I_raw.shape
        H0_t, W0_t = T_nat.shape
        yc_i, xc_i = H0_i // 2, W0_i // 2
        yc_t, xc_t = H0_t // 2, W0_t // 2
        center_note = "No header sky coords; used image centres."
    else:
        x_i, y_i = W_raw.world_to_pixel(header_sky)
        x_t, y_t = W_tgt.world_to_pixel(header_sky)
        yc_i, xc_i = float(y_i), float(x_i)
        yc_t, xc_t = float(y_t), float(x_t)
        center_note = f"Centered on RA={header_sky.ra.deg:.6f}, Dec={header_sky.dec.deg:.6f} deg."

    # optional per-source pixel offsets (applied consistently)
    dy_px, dx_px = OFFSETS_PX.get(source_name, (0.0, 0.0))
    if dy_px or dx_px:
        yc_i += dy_px; xc_i += dx_px
        yc_t += dy_px; xc_t += dx_px
        center_note += f" | manual offset (dy,dx)=({dy_px:.1f},{dx_px:.1f}) px"

    # --- extract kpc scale from label for zoom adjustment ---
    kpc_match = re.search(r'T(\d+(?:\.\d+)?)kpc', t_label)
    if kpc_match and False:
        kpc_value = float(kpc_match.group(1))
        # Apply scale-dependent zoom factor:
        zoom_25 = 1.42 # - 25kpc: zoom OUT
        zoom_100 = 0.49 # - 100kpc: zoom IN 
        if kpc_value <= 30:  # 25kpc and similar
            zoom_factor = zoom_25
        elif kpc_value >= 90:  # 100kpc and similar
            zoom_factor = zoom_100
        else:  # intermediate scales
            zoom_factor = 1.0  # neutral zoom for intermediate scales
    elif kpc_match and True:
        # Analytical zoom factor based on kpc scale
        kpc_value = float(kpc_match.group(1))
        zoom_factor = np.sqrt(fwhm_kpc / kpc_value) 
    else:
        zoom_factor = 1.0  # default if no match
    
    # --- equal-beams crop: side = global_min_beams * FWHM_T50 * zoom_factor ---
    fwhm_t50_as = fwhm_major_as(H_tgt)
    side_as = make_montage._global_nbeams * fwhm_t50_as * zoom_factor
    if getattr(make_montage, "GLOBAL_NBEAMS", None):
        # equal-beams crop using global min beam count with zoom adjustment
        fwhm_t50_as = fwhm_major_as(H_tgt)
        side_as = make_montage.GLOBAL_NBEAMS * fwhm_t50_as * zoom_factor
        if has_sub:
            (I_crop, RT_crop, T_crop, SUB_crop), (nyc, nxc), (cy_raw, cx_raw) = crop_to_side_arcsec_on_raw(
                I_raw, H_raw, side_as, RT_rawgrid, T_on_raw, SUB_on_raw, center=(yc_i, xc_i)
            )
        else:
            (I_crop, RT_crop, T_crop), (nyc, nxc), (cy_raw, cx_raw) = crop_to_side_arcsec_on_raw(
                I_raw, H_raw, side_as, RT_rawgrid, T_on_raw, center=(yc_i, xc_i)
            )
            SUB_crop = None
    else:
        # fallback: just use your default FOV crop with zoom adjustment
        adjusted_fov = fov_arcmin * zoom_factor
        if has_sub:
            (I_crop, RT_crop, T_crop, SUB_crop), (nyc, nxc), (cy_raw, cx_raw) = crop_to_fov_on_raw(
                I_raw, H_raw, adjusted_fov, RT_rawgrid, T_on_raw, SUB_on_raw, center=(yc_i, xc_i)
            )
        else:
            (I_crop, RT_crop, T_crop), (nyc, nxc), (cy_raw, cx_raw) = crop_to_fov_on_raw(
                I_raw, H_raw, adjusted_fov, RT_rawgrid, T_on_raw, center=(yc_i, xc_i)
            )
            SUB_crop = None
    
    # Optional downsample to a fixed display size (keeps the FOV content)
    Ho, Wo = _canon_size(downsample_size)[-2:]
    def _maybe_downsample(arr, Ho, Wo):
        """Downsample array to (Ho, Wo) using bilinear interpolation."""
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        with torch.no_grad():
            y = torch.nn.functional.interpolate(t, size=(Ho, Wo), mode="bilinear", align_corners=False)
        return y.squeeze(0).squeeze(0).cpu().numpy()

    I_fmt_np  = _maybe_downsample(I_crop,  Ho, Wo)
    RT_fmt_np = _maybe_downsample(RT_crop, Ho, Wo)
    T_fmt_np  = _maybe_downsample(T_crop,  Ho, Wo)
    if has_sub:
        SUB_fmt_np = _maybe_downsample(SUB_crop, Ho, Wo)
    else:
        SUB_fmt_np = None

    # --- WCS for formatted panels (FOV crop on RAW grid, then resize Ho×Wo)
    H0_i, W0_i = I_raw.shape
    Hc, Wc = nyc, nxc
    W_i_fmt, H_i_fmt = wcs_after_center_crop_and_resize(
        H_raw, H0_i, W0_i, Hc, Wc, Ho, Wo, int(round(cy_raw)), int(round(cx_raw))
    )
    W_rt_fmt, H_rt_fmt = W_i_fmt, H_i_fmt
    W_t_fmt,  H_t_fmt  = W_i_fmt, H_i_fmt   # T was reprojected to RAW, so use RAW WCS too
    W_sub_fmt, H_sub_fmt = W_i_fmt, H_i_fmt  # SUB also reprojected to RAW, so use RAW WCS too

    # --- plotting ranges. Only used for plotting. Not for FITS outputs.
    vmin_I, vmax_I   = robust_vmin_vmax(I_raw)
    vmin_RT, vmax_RT = robust_vmin_vmax(RT_rawgrid)
    vmin_T, vmax_T   = robust_vmin_vmax(T_nat)
    if has_sub:
        vmin_SUB, vmax_SUB = robust_vmin_vmax(SUB_nat)

    I_orig_np  = I_raw
    RT_orig_np = RT_rawgrid
    if has_sub:
        SUB_orig_np = SUB_on_raw  # Show SUB on RAW grid in original column

    # --- figure: 4×2 (if SUB) or 3×2 with WCS axes
    nrows = 4 if has_sub else 3
    fig = plt.figure(figsize=(12, 4.3*nrows), constrained_layout=True)
    
    ax00 = fig.add_subplot(nrows,2,1, projection=W_raw)      # RAW original
    ax01 = fig.add_subplot(nrows,2,2, projection=W_i_fmt)    # RAW beam-cropped
    ax10 = fig.add_subplot(nrows,2,3, projection=W_raw)      # RT original (on RAW grid)
    ax11 = fig.add_subplot(nrows,2,4, projection=W_rt_fmt)   # RT beam-cropped
    ax20 = fig.add_subplot(nrows,2,5, projection=W_raw)      # T original (RAW grid)
    ax21 = fig.add_subplot(nrows,2,6, projection=W_t_fmt)    # T beam-cropped
    
    if has_sub:
        ax30 = fig.add_subplot(nrows,2,7, projection=W_raw)      # SUB original (RAW grid)
        ax31 = fig.add_subplot(nrows,2,8, projection=W_sub_fmt)  # SUB beam-cropped

    im00 = ax00.imshow(I_orig_np,  origin="lower", vmin=vmin_I,  vmax=vmax_I);   ax00.set_title("RAW (original)")
    im01 = ax01.imshow(I_fmt_np,   origin="lower", vmin=vmin_I,  vmax=vmax_I);   ax01.set_title(f"RAW beam-cropped ({Ho}×{Wo})")
    
    # Different title depending on mode
    if cheat_rt:
        im10 = ax10.imshow(RT_orig_np, origin="lower", vmin=vmin_RT, vmax=vmax_RT);  ax10.set_title(f"{rt_label}=RAW⊗G_hdr (original, RAW grid, from T header)")
        im11 = ax11.imshow(RT_fmt_np,  origin="lower", vmin=vmin_RT, vmax=vmax_RT);  ax11.set_title(f"{rt_label} beam-cropped (header method)")
    else:
        im10 = ax10.imshow(RT_orig_np, origin="lower", vmin=vmin_RT, vmax=vmax_RT);  ax10.set_title(f"{rt_label}=RAW⊗G_circ (original, RAW grid, z={z:.3f})")
        im11 = ax11.imshow(RT_fmt_np,  origin="lower", vmin=vmin_RT, vmax=vmax_RT);  ax11.set_title(f"{rt_label} beam-cropped")
    
    im20 = ax20.imshow(T_on_raw,   origin="lower", vmin=vmin_T,  vmax=vmax_T);   ax20.set_title(f"{t_label} (original, RAW grid)")
    im21 = ax21.imshow(T_fmt_np,   origin="lower", vmin=vmin_T,  vmax=vmax_T);   ax21.set_title(f"{t_label} beam-cropped")
    
    if has_sub:
        sub_label = t_label + "SUB"
        im30 = ax30.imshow(SUB_orig_np, origin="lower", vmin=vmin_SUB, vmax=vmax_SUB);  ax30.set_title(f"{sub_label} (original, RAW grid)")
        im31 = ax31.imshow(SUB_fmt_np,  origin="lower", vmin=vmin_SUB, vmax=vmax_SUB);  ax31.set_title(f"{sub_label} beam-cropped")

    fig.colorbar(im00, ax=[ax00, ax01], shrink=0.85, label="RAW [Jy/beam_raw]")
    fig.colorbar(im10, ax=[ax10, ax11], shrink=0.85, label=f"{rt_label} [Jy/beam_circ]")
    fig.colorbar(im20, ax=[ax20, ax21], shrink=0.85, label=t_label)
    if has_sub:
        fig.colorbar(im30, ax=[ax30, ax31], shrink=0.85, label=sub_label)

    # Different suptitle depending on mode
    if cheat_rt:
        fig.suptitle(f"{source_name} — {rt_label} from T header (elliptical) vs {t_label} — {center_note}", fontsize=13)
    else:
        fig.suptitle(f"{source_name} — {rt_label} circular ({fwhm_kpc:.0f} kpc @ z={z:.3f}) vs {t_label} — {center_note}", fontsize=13)
    
    
    #out_png here
    t_label = f"T{int(y_chosen) if float(y_chosen).is_integer() else y_chosen}kpc"
    suffix = "cheat" if cheat_rt else "circular"
    
    # Info about SUB availability
    sub_info = " (with SUB)" if sub_path else " (no SUB)"
        
    # Check if outputs already exist (rsync-like behavior) unless --force is used
    if not force:
        if save_fits:
            
            required_files = [
                out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_{suffix}.fits",
                out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
            ]
            # Add SUB if it should exist
            if sub_path:
                sub_label = f"{t_label}SUB"
                required_files.append(out_fits_dir / f"{source_name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits")
            
            # Check if all required files exist
            all_exist = out_png.exists() and all(f.exists() for f in required_files)
            
            # Special case: if SUB input exists but output doesn't, don't skip
            if sub_path and sub_path.exists():
                sub_output = out_fits_dir / f"{source_name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits"
                if not sub_output.exists():
                    print(f"[REGEN] {source_name} {rt_label}: SUB input exists but output missing, regenerating...")
                    all_exist = False
            
            if all_exist:
                print(f"[SKIP] {source_name} {rt_label}: outputs already exist (use --force to regenerate)")
        else:
            # Only check PNG if not saving FITS
            if out_png.exists():
                print(f"[SKIP] {source_name} {rt_label}: montage already exists (use --force to regenerate)")

    
    if cheat_rt:
        print(f"[OK] {source_name} {rt_label} using T header (elliptical){sub_info} → {out_png}")
    else:
        print(f"[OK] {source_name} {rt_label} (z={z:.3f}) using circular beam {fwhm_kpc:.0f} kpc{sub_info} → {out_png}")
    n_ok += 1
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- optional FITS outputs for the beam-cropped panels
    if save_fits:
        out_fits_dir = (out_fits_dir or out_png.parent)
        out_fits_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  [DEBUG] Saving {source_name} with zoom_factor={zoom_factor:.2f}")
        
        # Save RAW, RT, T
        fits.writeto(out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_{suffix}.fits",
                     I_fmt_np.astype(np.float32), H_i_fmt, overwrite=True)
        fits.writeto(out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                    RT_fmt_np.astype(np.float32), H_rt_fmt, overwrite=True)
        fits.writeto(out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}{suffix}.fits",
                    T_fmt_np.astype(np.float32), H_t_fmt, overwrite=True)
        
        # Save SUB if available
        if has_sub:
            sub_label = t_label + "SUB"
            fits.writeto(out_fits_dir / f"{source_name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits",
                        SUB_fmt_np.astype(np.float32), H_sub_fmt, overwrite=True)
            report_nans(out_fits_dir / f"{source_name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits")
        
        report_nans(out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_{suffix}.fits")
        report_nans(out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_{suffix}.fits")
        report_nans(out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}_{suffix}.fits")


# --------------------------------- CLI ---------------------------------------
def parse_tuple3(txt: str) -> Tuple[int,int,int]:
    """Parse size specification like '128,128' or '1,128,128' into (C,H,W) tuple."""
    vals = [int(v) for v in str(txt).strip().split(",")]
    if len(vals) == 2: return (1, vals[0], vals[1])
    if len(vals) == 3: return (vals[0], vals[1], vals[2])
    raise argparse.ArgumentTypeError("Use H,W or C,H,W")

def main():
    ap = argparse.ArgumentParser(description="3×2 montages per source: RAW, RT=RAW⊗G_circ (circular kernel from z), and T_X.")
    DEFAULT_ROOT = Path("/users/mbredber/scratch/data/PSZ2/fits")
    DEFAULT_OUT  = Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/montages")
    DEFAULT_Z_CSV = Path("/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv")

    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Root directory with per-source subfolders (/<name>/<name>.fits and T_X). Default: {DEFAULT_ROOT}")
    ap.add_argument("--z-csv", type=Path, default=DEFAULT_Z_CSV,
                    help=f"CSV file with redshift table (columns: slug, z). Default: {DEFAULT_Z_CSV}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output directory for PNG montages. Default: {DEFAULT_OUT}")
    ap.add_argument("--crop", type=parse_tuple3, default="512,512",
                    help="Crop size H,W or C,H,W in input pixels. Default: 512,512")
    ap.add_argument("--down", type=parse_tuple3, default="128,128",
                    help="Downsample size H,W or C,H,W. Default: 128,128")
    ap.add_argument("--scales", type=str, default="25, 50, 100",
                    help="Comma-separated RT scales in kpc (controls circular kernel FWHM). E.g., '25,50,100' creates RT25kpc, RT50kpc, RT100kpc with corresponding circular kernels.")
    ap.add_argument("--fov-arcmin", type=float, default=50.0,
                    help="Square FOV (arcmin) for the formatted column; crop is on RAW grid.")
    ap.add_argument("--cheat-rt", action="store_true", default=False,
                    help="Use T_X header beam for RT (elliptical, old method). Default: False (use circular kernel from redshift).")
    ap.add_argument("--force", action="store_true", default=False,
                    help="Force regeneration of outputs even if they already exist. Default: False (skip existing outputs like rsync).")
    ap.add_argument("--only-offsets", action="store_true",
                    help="Process only sources listed in OFFSETS_PX.")
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated source names to include exclusively.")
    ap.add_argument("--save-fits", action="store_true", default=True,
                    help="Also write formatted RAW/RT/T_X FITS for each montage.")
    ap.add_argument("--fits-out", type=Path, default=Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits"),
                    help="Directory for formatted FITS (defaults to the montage folder).")

    args = ap.parse_args()
    
    # Load redshift table
    print(f"[init] Loading redshift table from {args.z_csv}")
    slug_to_z = load_z_table(args.z_csv)
    print(f"[init] Loaded {len(slug_to_z)} redshifts")
    
    # Store global n_beams for consistent cropping
    make_montage.GLOBAL_NBEAMS = compute_global_nbeams_min(args.root) * 1.85

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    only_names = set(s.strip() for s in args.only.split(",") if s.strip())
    
    out_fits_dir = args.fits_out if args.fits_out else args.out               

    n_ok = 0
    n_skip = 0
    for scale in scales:
        try:
            x_req = float(scale)
        except Exception:
            print(f"[SKIP] invalid scale {scale!r}")
            continue
        rt_label = f"RT{int(x_req) if x_req.is_integer() else x_req}kpc"
        for source_name, raw_path, t_path, sub_path, y_chosen in find_pairs_in_tree(args.root, x_req):
            if args.only_offsets and source_name not in OFFSETS_PX:
                continue
            if only_names and source_name not in only_names:
                continue
            
            # Look up redshift (only required if not using cheat_rt)
            z = slug_to_z.get(source_name, np.nan)
            if not args.cheat_rt:
                # Redshift required for circular kernel mode
                if not np.isfinite(z) or z <= 0:
                    print(f"[SKIP] {source_name}: no valid redshift (z={z}) required for circular mode")
                    n_skip += 1
                    continue
            else:
                # For cheat_rt mode, redshift not strictly required but pass NaN if unavailable
                if not np.isfinite(z) or z <= 0:
                    z = np.nan  # Will be ignored in cheat_rt mode
                
            try:
                out_png = args.out / f"{source_name}_montage_{suffix}.png"
                make_montage(source_name, raw_path, t_path, sub_path, z, rt_label, t_label,
                            downsample_size=args.down,
                            save_fits=args.save_fits,
                            out_fits_dir=args.fits_out,
                            fov_arcmin=args.fov_arcmin,
                            fwhm_kpc=x_req,  # Use the requested scale (25, 50, 100, etc.)
                            cheat_rt=args.cheat_rt,
                            y_chosen=y_chosen,
                            force=args.force,
                            out_png=out_png,
                            out_fits_dir=out_fits_dir
                            )
    
            except Exception as e:
                print(f"[SKIP] {name} {rt_label}: {e}")
                import traceback
                traceback.print_exc()
                n_skip += 1

    print(f"Done. Wrote {n_ok} montages. Skipped {n_skip}.")
    
    # === VERIFICATION REPORT ===
    print("\n" + "="*80)
    print("VERIFICATION REPORT: SUB File Coverage and Processing")
    print("="*80)
    
    # Count sources with and without SUB files for each scale
    for scale in scales:
        try:
            x_req = float(scale)
        except Exception:
            continue
        
        suffix = "cheat" if args.cheat_rt else "circular"
        Ho, Wo = _canon_size(args.down)[-2:]
        
        sources_with_sub_input = []
        sources_without_sub_input = []
        sources_with_sub_output = []
        sources_missing_sub_output = []
        
        for name, raw_path, t_path, sub_path, y_chosen in find_pairs_in_tree(args.root, x_req):
            if args.only_offsets and name not in OFFSETS_PX:
                continue
            if only_names and name not in only_names:
                continue
            
            t_label = f"T{int(y_chosen) if float(y_chosen).is_integer() else y_chosen}kpc"
            sub_label = f"{t_label}SUB"
            
            # Check input SUB file
            if sub_path is not None and sub_path.exists():
                sources_with_sub_input.append(name)
                
                # Check if output SUB file was created
                output_sub_file = out_fits_dir / f"{name}_{sub_label}_fmt_{Ho}x{Wo}_{suffix}.fits"
                if output_sub_file.exists():
                    sources_with_sub_output.append(name)
                else:
                    sources_missing_sub_output.append(name)
            else:
                sources_without_sub_input.append(name)
        
        total_sources = len(sources_with_sub_input) + len(sources_without_sub_input)
        print(f"\nScale: {x_req:.0f} kpc")
        print(f"  Total sources: {total_sources}")
        print(f"  Input SUB files available: {len(sources_with_sub_input)}")
        print(f"  Output SUB files created: {len(sources_with_sub_output)}")
        print(f"  Sources missing input SUB: {len(sources_without_sub_input)}")
        
        if sources_missing_sub_output:
            print(f"\n  ⚠️  ERROR: {len(sources_missing_sub_output)} sources have input SUB but output NOT created:")
            for name in sorted(sources_missing_sub_output)[:10]:
                print(f"      - {name}")
            if len(sources_missing_sub_output) > 10:
                print(f"      ... and {len(sources_missing_sub_output) - 10} more")
        
        if sources_without_sub_input:
            print(f"\n  ⚠️  WARNING: {len(sources_without_sub_input)} sources don't have input SUB files:")
            for name in sorted(sources_without_sub_input)[:10]:
                print(f"      - {name}")
            if len(sources_without_sub_input) > 10:
                print(f"      ... and {len(sources_without_sub_input) - 10} more")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()