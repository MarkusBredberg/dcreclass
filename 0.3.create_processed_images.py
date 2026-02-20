#!/usr/bin/env python3
"""
Merged batch script with NaN-sensitive cropping:
- For each source, load RAW, T_X, and T_XSUB (X in {25,50,100,...} kpc).
- Build RT_X by convolving RAW with a CIRCULAR X kpc kernel based on redshift.
- Crop to common number of beams for each version (RAW, RT_X, T_X, T_XSUB) using the RAW WCS and beam.
- Make a per-source montage (rows: RAW, RT_X, T_X, T_XSUB; cols: original vs beam-cropped).
- Save beam-cropped RAW/RT_X/T_X/T_XSUB FITS with updated WCS.
- Optionally produce a multi-source comparison plot (RAW | T_X | RT_X per scale).

The circular kernel size is determined by the scale parameter (--scales), not from T_X header.
"""

# Standard library
import argparse, csv, io, os, random, re, sys, warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as COSMO
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

warnings.filterwarnings("ignore", category=FITSFixedWarning)

print("Running 0.3.1.create_processed_images.py")

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
    """Extract 2x2 pixel->world Jacobian in radians/pixel from FITS header."""
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
    """Compute effective pixel scales (|dx|, |dy|) in arcseconds from header WCS."""
    J = _cd_matrix_rad(h)
    dx = np.hypot(J[0,0], J[1,0])
    dy = np.hypot(J[0,1], J[1,1])
    return dx*ARCSEC_PER_RAD, dy*ARCSEC_PER_RAD

def fwhm_major_as(h):
    """Return the major axis FWHM from the FITS header beam in arcseconds."""
    return max(float(h["BMAJ"]), float(h["BMIN"])) * 3600.0

def _fwhm_as_to_sigma_rad(fwhm_as: float) -> float:
    """Convert FWHM [arcsec] -> Gaussian sigma [radians]."""
    return (float(fwhm_as)/(2.0*np.sqrt(2.0*np.log(2.0)))) * (np.pi/(180.0*3600.0))

def beam_cov_world(h):
    """Return 2x2 covariance in world radians for the header beam."""
    bmaj_as = abs(float(h['BMAJ']))*3600.0
    bmin_as = abs(float(h['BMIN']))*3600.0
    pa_deg  = float(h.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th      = np.deg2rad(pa_deg)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], float)
    S = np.diag([sx*sx, sy*sy])
    return R @ S @ R.T

def beam_solid_angle_sr(h):
    """Gaussian beam solid angle in steradians from BMAJ/BMIN [deg] in FITS header."""
    bmaj = abs(float(h['BMAJ'])) * np.pi/180.0
    bmin = abs(float(h['BMIN'])) * np.pi/180.0
    return (np.pi/(4.0*np.log(2.0))) * bmaj * bmin

def kernel_from_beams(raw_hdr, tgt_hdr):
    """Elliptical Gaussian kernel that maps RAW beam -> TARGET beam (cheat_rt method)."""
    C_raw = beam_cov_world(raw_hdr)
    C_tgt = beam_cov_world(tgt_hdr)
    C_ker_world = C_tgt - C_raw
    w, V = np.linalg.eigh(C_ker_world)
    w = np.clip(w, 0.0, None)
    C_ker_world = (V * w) @ V.T
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
    """Return circular 2x2 covariance in world coords for a Gaussian with FWHM=fwhm_kpc at redshift z."""
    if z is None or not np.isfinite(z) or z <= 0:
        return None
    fwhm_kpc = float(fwhm_kpc)
    if fwhm_kpc <= 0:
        raise ValueError("fwhm_kpc must be positive")
    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_rad = (fwhm_kpc / DA_kpc)
    sigma = theta_rad / (2.0*np.sqrt(2.0*np.log(2.0)))
    sigma2 = float(sigma**2)
    return np.array([[sigma2, 0.0],[0.0, sigma2]], float)

def circular_kernel_from_z(z, raw_hdr, fwhm_kpc=50.0):
    """Build a circular Gaussian convolution kernel on the RAW pixel grid."""
    C_raw = beam_cov_world(raw_hdr)
    C_circ = circular_cov_kpc(z, fwhm_kpc=fwhm_kpc)
    if C_circ is None:
        raise ValueError(f"Invalid redshift z={z} for circular kernel")
    C_ker_world = C_circ - C_raw
    w, V = np.linalg.eigh(C_ker_world)
    w = np.clip(w, 0.0, None)
    C_ker_world = (V * w) @ V.T
    J = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker_world @ Jinv.T
    wp, Vp = np.linalg.eigh(Cpix)
    wp = np.clip(wp, 1e-18, None)
    s_minor = float(np.sqrt(wp[0]))
    s_major = float(np.sqrt(wp[1]))
    theta   = float(np.arctan2(Vp[1,1], Vp[0,1]))
    nker = int(np.ceil(8.0*max(s_major, s_minor))) | 1
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

def load_z_table(csv_path):
    """Load redshift table from CSV. Returns dict slug -> z."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Redshift table not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()
    if not raw.strip():
        raise ValueError(f"Redshift table appears empty: {csv_path}")
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("Redshift table has headers but no data rows.")
    rdr = csv.DictReader(io.StringIO("\n".join(lines)))
    if rdr.fieldnames is None or "slug" not in rdr.fieldnames or "z" not in rdr.fieldnames:
        raise ValueError(f"CSV missing required headers 'slug,z'; got {rdr.fieldnames!r}")
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

def read_fits_array_header_wcs(fpath: Path):
    """Read FITS file and return (2D array, header, 2D WCS)."""
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
    if arr.ndim == 3: arr = np.nanmean(arr, axis=0)
    if arr.ndim != 2: raise RuntimeError(f"Expected 2D image; got {arr.shape}")
    return arr.astype(np.float32), header, wcs2d

def reproject_like(arr: np.ndarray, src_hdr, dst_hdr) -> np.ndarray:
    """Reproject array from source WCS to destination WCS."""
    try:
        from reproject import reproject_interp
        w_src = (WCS(src_hdr).celestial if hasattr(WCS(src_hdr), "celestial") else WCS(src_hdr, naxis=2))
        w_dst = (WCS(dst_hdr).celestial if hasattr(WCS(dst_hdr), "celestial") else WCS(dst_hdr, naxis=2))
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        out, _ = reproject_interp((arr, w_src), w_dst, shape_out=(ny_out, nx_out), order='bilinear')
        return out.astype(np.float32)
    except Exception:
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
    """Extract cluster sky coordinates from FITS header."""
    if header.get('OBJCTRA') and header.get('OBJCTDEC'):
        return SkyCoord(header['OBJCTRA'], header['OBJCTDEC'], unit=(u.hourangle, u.deg))
    if header.get('RA_TARG') and header.get('DEC_TARG'):
        return SkyCoord(header['RA_TARG']*u.deg, header['DEC_TARG']*u.deg)
    if 'CRVAL1' in header and 'CRVAL2' in header:
        return SkyCoord(header['CRVAL1']*u.deg, header['CRVAL2']*u.deg)
    return None

def wcs_after_center_crop_and_resize(header, H0, W0, Hc, Wc, Ho, Wo, y0, x0):
    """Update WCS header after center crop and resize operations."""
    y1, y2 = max(0, y0 - Hc // 2), min(H0, y0 + Hc // 2)
    x1, x2 = max(0, x0 - Wc // 2), min(W0, x0 + Wc // 2)
    width  = x2 - x1
    height = y2 - y1
    sx = width  / float(Wo)
    sy = height / float(Ho)
    new = header.copy()
    if "CRPIX1" in new and "CRPIX2" in new:
        new["CRPIX1"] = (new["CRPIX1"] - x1) / sx
        new["CRPIX2"] = (new["CRPIX2"] - y1) / sy
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
    """Compute robust min/max from percentiles for display scaling."""
    finite = np.isfinite(arr)
    if not finite.any(): return 0.0, 1.0
    vals = arr[finite]
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if vmin == vmax: vmax = vmin + 1.0
    return float(vmin), float(vmax)

# ------------------------------- formatting ----------------------------------
def _canon_size(sz):
    """Canonicalize size specification to (C, H, W) tuple."""
    if isinstance(sz, (tuple, list)):
        if len(sz) == 2:  return (1, sz[0], sz[1])
        if len(sz) == 3:  return (sz[0], sz[1], sz[2])
    raise ValueError("size must be H,W or C,H,W")

def crop_to_side_arcsec_on_raw(I, Hraw, side_arcsec, *arrs, center=None):
    """Square crop on RAW grid with side length in arcsec."""
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
    """Yield (name, raw_path, t_path, sub_path, chosen_kpc) for each source directory."""
    pat_t = re.compile(r"T([0-9]+(?:\.[0-9]+)?)kpc\.fits$", re.IGNORECASE)
    for src_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        name = src_dir.name
        raw_path = src_dir / f"{name}.fits"
        if not raw_path.exists():
            continue
        candidates = []
        for fp in src_dir.glob(f"{name}T*kpc.fits"):
            if "SUB" in fp.name.upper():
                continue
            m = pat_t.search(fp.name)
            if m:
                y = float(m.group(1))
                candidates.append((abs(y - desired_kpc), y, fp))
        if candidates:
            _, ybest, fbest = sorted(candidates, key=lambda t: (t[0], t[1]))[0]
            sub_path = src_dir / f"{name}T{int(ybest) if ybest.is_integer() else ybest}kpcSUB.fits"
            if not sub_path.exists():
                sub_path = None
            yield name, raw_path, fbest, sub_path, ybest

def report_nans(path: Path):
    """Report the number of NaN pixels in a FITS file."""
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=float)
    n = np.isnan(arr).sum()
    if n > 0:
        print(f"[nancheck] {path}: {n} NaNs")

# ======================== CROPPING LOGIC ========================

def _nan_free_centred_square_side_as(arr: np.ndarray, header) -> float:
    """
    Find largest NaN-free centred square in arcsec via binary search.
    Returns side length in arcsec, or 0.0 if image is all-NaN.
    """
    if not np.any(np.isfinite(arr)):
        return 0.0
    H, W = arr.shape
    asx, asy = arcsec_per_pix(header)
    cy, cx = H / 2.0, W / 2.0
    min_side_px = 1
    max_side_px = min(H, W)
    best_side_px = 0
    while min_side_px <= max_side_px:
        mid_side_px = (min_side_px + max_side_px) // 2
        y0 = int(round(cy - mid_side_px / 2))
        x0 = int(round(cx - mid_side_px / 2))
        y0 = max(0, min(y0, H - mid_side_px))
        x0 = max(0, min(x0, W - mid_side_px))
        crop = arr[y0:y0+mid_side_px, x0:x0+mid_side_px]
        if np.all(np.isfinite(crop)):
            best_side_px = mid_side_px
            min_side_px = mid_side_px + 1
        else:
            max_side_px = mid_side_px - 1
    return best_side_px * min(asx, asy)

def compute_global_nbeams_per_version(root_dir: Path, scales: List[float]) -> Dict[float, float]:
    """
    For each version (scale), compute the global minimum n_beams across all sources.
    Returns: Dict mapping scale -> global_nbeams for that scale.
    Used by the per-source montage pipeline.
    """
    print("[compute_global_nbeams] Computing per-version n_beams...")
    global_nbeams = {}
    for scale in scales:
        scale_str = f"{int(scale)}" if scale == int(scale) else f"{scale}"
        max_nbeams_list = []
        for src_dir in sorted(p for p in root_dir.glob("*") if p.is_dir()):
            name = src_dir.name
            t_path = src_dir / f"{name}T{scale_str}kpc.fits"
            if not t_path.exists():
                continue
            try:
                arr, hdr, _ = read_fits_array_header_wcs(t_path)
                max_side_as = _nan_free_centred_square_side_as(arr, hdr)
                if max_side_as <= 0:
                    continue
                beam_fwhm_as = fwhm_major_as(hdr)
                max_nbeams_list.append(max_side_as / beam_fwhm_as)
            except Exception:
                continue
        if max_nbeams_list:
            global_nbeams[scale] = min(max_nbeams_list)
            print(f"[scan] T{scale_str}kpc: n_beams={global_nbeams[scale]:.1f} "
                  f"(min={min(max_nbeams_list):.1f}, max={max(max_nbeams_list):.1f})")
        else:
            global_nbeams[scale] = 100.0
    return global_nbeams

def compute_global_nbeams_min_t50(root_dir: Path) -> Optional[float]:
    """
    Scan all T50kpc.fits files and return the global minimum n_beams (FOV/FWHM).
    Used by the comparison plot pipeline as a simpler single-value reference.
    """
    n_beams = []
    for tfile in Path(root_dir).rglob("*T50kpc.fits"):
        try:
            with fits.open(tfile) as hdul:
                h = hdul[0].header
            fwhm = max(float(h["BMAJ"]), float(h["BMIN"])) * 3600.0
            if "CD1_1" in h:
                dx = np.hypot(h["CD1_1"], h.get("CD2_1", 0)) * 206264.806
                dy = np.hypot(h.get("CD1_2", 0), h.get("CD2_2", 0)) * 206264.806
            else:
                dx = abs(h.get("CDELT1", 1)) * 3600.0
                dy = abs(h.get("CDELT2", 1)) * 3600.0
            fovx = int(h["NAXIS1"]) * dx
            fovy = int(h["NAXIS2"]) * dy
            n_beams.append(min(fovx, fovy) / max(fwhm, 1e-9))
        except Exception as e:
            print(f"[scan] skip {tfile}: {e}")
    if not n_beams:
        print("[scan] No T50kpc files found")
        return None
    nmin = min(n_beams)
    print(f"[scan] global_nbeams_min (T50) = {nmin:.2f} across {len(n_beams)} files")
    return nmin

def check_nan_fraction(arr: np.ndarray, name: str = "") -> float:
    """Check what fraction of pixels are NaN. Returns fraction [0,1]."""
    if arr.size == 0:
        return 0.0
    n_nan = np.isnan(arr).sum()
    frac = n_nan / arr.size
    if frac > 0:
        print(f"[NaN WARNING] {name}: {frac*100:.2f}% NaN pixels ({n_nan}/{arr.size})")
    return frac

# ------------------------------ image processing core ------------------------
def process_images_for_scale(source_name: str,
                             raw_path: Path,
                             t_path: Path,
                             sub_path: Optional[Path],
                             z: float,
                             fwhm_kpc: float,
                             target_nbeams: float,
                             downsample_size=(1, 128, 128),
                             cheat_rt: bool = False):
    """Process RAW + T_X (+ T_XSUB) for one source at one scale. Returns dict of arrays."""
    if not cheat_rt:
        if not np.isfinite(z) or z <= 0:
            raise ValueError(f"Invalid redshift z={z} for {source_name}")

    I_raw,  H_raw,  W_raw  = read_fits_array_header_wcs(raw_path)
    T_nat,  H_tgt,  W_tgt  = read_fits_array_header_wcs(t_path)

    if sub_path is not None and sub_path.exists():
        SUB_nat, H_sub, W_sub = read_fits_array_header_wcs(sub_path)
        has_sub = True
    else:
        SUB_nat, H_sub, W_sub = None, None, None
        has_sub = False

    # Common grid = T_X grid (smallest FOV)
    T_common = T_nat
    H_common = H_tgt
    W_common = W_tgt

    I_on_common = reproject_like(I_raw, H_raw, H_common)
    SUB_on_common = reproject_like(SUB_nat, H_sub, H_common) if has_sub else None

    # Build RT on RAW grid then reproject to common
    if cheat_rt:
        ker = kernel_from_beams(H_raw, H_tgt)
    else:
        ker = circular_kernel_from_z(z, H_raw, fwhm_kpc=fwhm_kpc)
    I_smt = convolve_fft(I_raw, ker, boundary="fill", fill_value=np.nan,
                         nan_treatment="interpolate", normalize_kernel=True,
                         psf_pad=True, fft_pad=True, allow_huge=True)
    scale = beam_solid_angle_sr(H_tgt) / beam_solid_angle_sr(H_raw)
    RT_rawgrid = I_smt * scale
    RT_on_common = reproject_like(RT_rawgrid, H_raw, H_common)

    # Centre from header sky coords + manual offset
    header_sky = header_cluster_coord(H_raw) or header_cluster_coord(H_tgt)
    if header_sky is None:
        H0_i, W0_i = I_raw.shape
        yc_i, xc_i = H0_i // 2, W0_i // 2
        center_note = "No header sky coords; used image centres."
    else:
        x_i, y_i = W_raw.world_to_pixel(header_sky)
        yc_i, xc_i = float(y_i), float(x_i)
        center_note = f"Centered on RA={header_sky.ra.deg:.6f}, Dec={header_sky.dec.deg:.6f} deg."
    dy_px, dx_px = OFFSETS_PX.get(source_name, (0.0, 0.0))
    if dy_px or dx_px:
        yc_i += dy_px; xc_i += dx_px
        center_note += f" | manual offset (dy,dx)=({dy_px:.1f},{dx_px:.1f}) px"

    # Dynamic crop: find NaN-free side for each array, use the minimum
    beam_fwhm_as = fwhm_major_as(H_common)
    desired_side_as = target_nbeams * beam_fwhm_as
    max_side_as_T  = _nan_free_centred_square_side_as(T_common, H_common)
    max_side_as_I  = _nan_free_centred_square_side_as(I_on_common, H_common)
    max_side_as_RT = _nan_free_centred_square_side_as(RT_on_common, H_common)
    max_side_as = min(max_side_as_T, max_side_as_I, max_side_as_RT)
    if has_sub and SUB_on_common is not None:
        max_side_as = min(max_side_as, _nan_free_centred_square_side_as(SUB_on_common, H_common))
    actual_side_as = min(desired_side_as, max_side_as)
    actual_nbeams  = actual_side_as / beam_fwhm_as
    if target_nbeams is not None and target_nbeams != actual_nbeams:
        print(f"[crop] WARNING: for source {source_name}.")
        print(f"[crop] {source_name}/{fwhm_kpc:.5f}kpc: target={target_nbeams:.5f} beams, "
              f"actual={actual_nbeams:.5f} beams ({actual_side_as:.5f}\" | "
              f"NaN-free: T={max_side_as_T:.5f}\" I={max_side_as_I:.5f}\" RT={max_side_as_RT:.5f}\")")

    # Re-derive centre on common grid (with scaled manual offset)
    header_sky_common = header_cluster_coord(H_common)
    if header_sky_common is None:
        yc_common = T_common.shape[0] // 2
        xc_common = T_common.shape[1] // 2
    else:
        x_c, y_c = W_common.world_to_pixel(header_sky_common)
        yc_common, xc_common = float(y_c), float(x_c)
    dy_px_common, dx_px_common = OFFSETS_PX.get(source_name, (0.0, 0.0))
    asx_raw, asy_raw = arcsec_per_pix(H_raw)
    asx_common, asy_common = arcsec_per_pix(H_common)
    dy_px_common *= (asy_raw / asy_common)
    dx_px_common *= (asx_raw / asx_common)
    if dy_px_common or dx_px_common:
        yc_common += dy_px_common
        xc_common += dx_px_common

    if has_sub:
        (I_crop, RT_crop, T_crop, SUB_crop), (nyc, nxc), (cy_eff, cx_eff) = \
            crop_to_side_arcsec_on_raw(I_on_common, H_common, actual_side_as,
                                       RT_on_common, T_common, SUB_on_common,
                                       center=(yc_common, xc_common))
    else:
        (I_crop, RT_crop, T_crop), (nyc, nxc), (cy_eff, cx_eff) = \
            crop_to_side_arcsec_on_raw(I_on_common, H_common, actual_side_as,
                                       RT_on_common, T_common,
                                       center=(yc_common, xc_common))
        SUB_crop = None

    check_nan_fraction(I_crop,  f"{source_name} I_crop")
    check_nan_fraction(RT_crop, f"{source_name} RT_crop")
    check_nan_fraction(T_crop,  f"{source_name} T_crop")
    if has_sub:
        check_nan_fraction(SUB_crop, f"{source_name} SUB_crop")

    # Downsample to target size
    Ho, Wo = _canon_size(downsample_size)[-2:]
    def _maybe_downsample(arr, Ho, Wo):
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y = torch.nn.functional.interpolate(t, size=(Ho, Wo), mode="bilinear", align_corners=False)
        return y.squeeze(0).squeeze(0).cpu().numpy()

    I_fmt_np   = _maybe_downsample(I_crop,   Ho, Wo)
    RT_fmt_np  = _maybe_downsample(RT_crop,  Ho, Wo)
    T_fmt_np   = _maybe_downsample(T_crop,   Ho, Wo)
    SUB_fmt_np = _maybe_downsample(SUB_crop, Ho, Wo) if has_sub else None

    H0_common, W0_common = T_common.shape
    Hc, Wc = nyc, nxc
    W_i_fmt, H_i_fmt = wcs_after_center_crop_and_resize(
        H_common, H0_common, W0_common, Hc, Wc, Ho, Wo,
        int(round(cy_eff)), int(round(cx_eff)))

    return {
        'I_raw': I_raw, 'RT_rawgrid': RT_rawgrid,
        'T_on_common': T_common, 'SUB_on_common': SUB_on_common,
        'I_fmt_np': I_fmt_np, 'RT_fmt_np': RT_fmt_np,
        'T_fmt_np': T_fmt_np, 'SUB_fmt_np': SUB_fmt_np,
        'H_raw': H_raw, 'H_tgt': H_tgt, 'H_i_fmt': H_i_fmt,
        'W_raw': W_raw, 'W_i_fmt': W_i_fmt,
        'has_sub': has_sub, 'center_note': center_note,
        'actual_side_as': actual_side_as,
    }

# ========================= annotation helpers ================================

def add_beam_patch(ax, header, color='white', alpha=0.8, loc='lower left'):
    """Add beam ellipse to a WCS-projection axis (used by per-source montage)."""
    from matplotlib.patches import Ellipse
    bmaj_deg = float(header['BMAJ']); bmin_deg = float(header['BMIN'])
    bpa_deg  = float(header.get('BPA', 0.0))
    bmaj_as  = bmaj_deg * 3600.0;  bmin_as = bmin_deg * 3600.0
    ny, nx   = int(header['NAXIS2']), int(header['NAXIS1'])
    asx, asy = arcsec_per_pix(header)
    bmaj_pix = bmaj_as / asx;  bmin_pix = bmin_as / asy
    margin   = 0.08
    y_center = (ny * margin + bmaj_pix / 2 if 'lower' in loc
                else ny * (1 - margin) - bmaj_pix / 2)
    x_center = (nx * margin + bmaj_pix / 2 if 'left' in loc
                else nx * (1 - margin) - bmaj_pix / 2)
    beam = Ellipse(xy=(x_center, y_center), width=bmaj_pix, height=bmin_pix,
                   angle=bpa_deg, transform=ax.get_transform('pixel'),
                   facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5)
    ax.add_patch(beam)
    label = f"{bmaj_as:.1f}\"x{bmin_as:.1f}\""
    y_text = (ny * margin + bmaj_pix + 5 if 'lower' in loc
              else ny * (1 - margin) - bmaj_pix - 5)
    ax.text(x_center, y_text, label, transform=ax.get_transform('pixel'),
            fontsize=9, color=color, weight='bold',
            ha='center', va='bottom' if 'lower' in loc else 'top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none'))

def add_scalebar_kpc(ax, header, z, length_kpc=100.0, color='white', loc='lower right'):
    """Add physical scale bar to a WCS-projection axis (used by per-source montage)."""
    from matplotlib.lines import Line2D
    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_as = (length_kpc / DA_kpc) * ARCSEC_PER_RAD
    asx, asy = arcsec_per_pix(header)
    bar_px = theta_as / asx
    ny, nx  = int(header['NAXIS2']), int(header['NAXIS1'])
    margin  = 0.08;  bar_thickness = 3
    y_bar   = (ny * margin if 'lower' in loc else ny * (1 - margin) - bar_thickness)
    x_start = (nx * margin if 'left' in loc else nx * (1 - margin) - bar_px)
    x_end   = x_start + bar_px
    ax.add_line(Line2D([x_start, x_end], [y_bar, y_bar],
                       transform=ax.get_transform('pixel'),
                       color=color, linewidth=bar_thickness, solid_capstyle='butt'))
    label  = f"{int(length_kpc)} kpc"
    x_text = (x_start + x_end) / 2
    y_text = y_bar - 8 if 'lower' in loc else y_bar + bar_thickness + 8
    ax.text(x_text, y_text, label, transform=ax.get_transform('pixel'),
            fontsize=10, color=color, weight='bold',
            ha='center', va='top' if 'lower' in loc else 'bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.5, edgecolor='none'))

def add_beam_patch_simple(ax, header, color='white', alpha=0.8,
                          loc='lower left', fontsize=6):
    """
    Add beam ellipse to a plain (non-WCS) axis using data coordinates.
    Used by the comparison plot where axes are simple imshow panels.
    """
    from matplotlib.patches import Ellipse
    bmaj_as  = float(header['BMAJ']) * 3600.0
    bmin_as  = float(header['BMIN']) * 3600.0
    bpa_deg  = float(header.get('BPA', 0.0))
    ny, nx   = int(header['NAXIS2']), int(header['NAXIS1'])
    asx, asy = arcsec_per_pix(header)
    bmaj_pix = bmaj_as / asx
    bmin_pix = bmin_as / asy
    margin   = 0.15
    bmaj_f   = bmaj_pix / ny
    bmin_f   = bmin_pix / nx
    yc_n = margin + bmaj_f / 2 if 'lower' in loc else 1 - margin - bmaj_f / 2
    xc_n = margin + bmin_f / 2 if 'left'  in loc else 1 - margin - bmin_f / 2
    yc_n = np.clip(yc_n, bmaj_f / 2 + 0.05, 1 - bmaj_f / 2 - 0.05)
    xc_n = np.clip(xc_n, bmin_f / 2 + 0.05, 1 - bmin_f / 2 - 0.05)
    ax.add_patch(Ellipse(xy=(xc_n * nx, yc_n * ny), width=bmaj_pix, height=bmin_pix,
                         angle=bpa_deg, transform=ax.transData,
                         facecolor=color, edgecolor='black', alpha=alpha, linewidth=0.8))
    yt_n = yc_n + bmaj_f / 2 + 0.03 if 'lower' in loc else yc_n - bmaj_f / 2 - 0.03
    ax.text(xc_n, yt_n, f"{bmaj_as:.1f}″×{bmin_as:.1f}″",
            transform=ax.transAxes, fontsize=fontsize, color=color, weight='bold',
            ha='center', va='bottom' if 'lower' in loc else 'top',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7, edgecolor='none'))

def add_scalebar_kpc_simple(ax, header, z, length_kpc=1000.0,
                            color='white', loc='lower right', fontsize=6):
    """
    Add physical scale bar to a plain (non-WCS) axis using data coordinates.
    Used by the comparison plot where axes are simple imshow panels.
    """
    from matplotlib.lines import Line2D
    DA_kpc   = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_as = (length_kpc / DA_kpc) * ARCSEC_PER_RAD
    asx, asy = arcsec_per_pix(header)
    bar_px   = theta_as / asx
    ny, nx   = int(header['NAXIS2']), int(header['NAXIS1'])
    margin   = 0.10
    y_n  = margin if 'lower' in loc else 1 - margin
    xs_n = margin if 'left' in loc else 1 - margin - bar_px / nx
    xe_n = xs_n + bar_px / nx
    ax.add_line(Line2D([xs_n * nx, xe_n * nx], [y_n * ny, y_n * ny],
                       color=color, linewidth=2, solid_capstyle='butt'))
    label  = f"{length_kpc/1000:.0f} Mpc" if length_kpc >= 1000 else f"{int(length_kpc)} kpc"
    yt_n   = y_n - 0.04 if 'lower' in loc else y_n + 0.04
    ax.text((xs_n + xe_n) / 2, yt_n, label,
            transform=ax.transAxes, fontsize=fontsize, color=color, weight='bold',
            ha='center', va='top' if 'lower' in loc else 'bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7, edgecolor='none'))

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
                             suffix: str = ""):
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
                downsample_size=downsample_size, cheat_rt=cheat_rt)
            processed['scale'] = scale
            processed['t_path'] = t_path
            processed['sub_path'] = sub_path
            processed_scales.append(processed)
        except Exception as e:
            print(f"[ERROR] {source_name} @ {scale}kpc: {e}")
            continue
    if not processed_scales:
        raise RuntimeError(f"No scales could be processed for {source_name}")

    Ho, Wo = _canon_size(downsample_size)[-2:]

    # Skip if outputs already exist (unless --force)
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

    has_sub = any(d['has_sub'] for d in processed_scales)
    nrows   = 4 if has_sub else 3
    n_scales = len(processed_scales)
    ncols = n_scales * 2                 # 2 cols per scale (orig | crop)
    row_heights = [2.0] + [1.0] * (nrows - 1)
    fig = plt.figure(figsize=(4*ncols, sum(row_heights)*4.3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows, ncols, figure=fig, height_ratios=row_heights,
                  left=0.03, right=0.98, top=0.93, bottom=0.04,
                  wspace=0.15, hspace=0.12)
    
    # Place scale headers just below the RAW row (top of row 1)
    # Row 0 occupies from top=0.93 down by row_heights[0]/total_height of usable space
    usable_h = 0.93 - 0.04   # top - bottom margins
    y_below_raw = 0.93 - (row_heights[0] / sum(row_heights)) * usable_h + 0.005

    usable_w = 0.98 - 0.03
    for scale_idx, data in enumerate(processed_scales):
        sc_str = int(data['scale']) if data['scale'] == int(data['scale']) else data['scale']
        col_centre = (scale_idx * 2 + 1) / ncols
        x = 0.03 + col_centre * usable_w
        fig.text(x, y_below_raw, f"{sc_str} kpc",
                fontsize=11, fontweight='bold', ha='center', va='bottom',
                transform=fig.transFigure)
        
    # Row labels on the left margin (RT / T / SUB)
    row_label_names = ['RT', 'T', 'SUB'] if has_sub else ['RT', 'T']
    total_height = sum(row_heights)
    for row_idx, label in enumerate(row_label_names, start=1):
        # y = centre of that row in figure coords
        y_top = 1.0 - sum(row_heights[:row_idx]) / total_height
        y_bot = 1.0 - sum(row_heights[:row_idx+1]) / total_height
        y = (y_top + y_bot) / 2
        fig.text(0.005, y, label, fontsize=13, fontweight='bold',
                ha='left', va='center', transform=fig.transFigure)

    first_data = processed_scales[0]
    I_raw      = first_data['I_raw']
    I_fmt_np   = first_data['I_fmt_np']
    W_i_fmt    = first_data['W_i_fmt']
    W_common   = (WCS(first_data['H_tgt']).celestial
                  if hasattr(WCS(first_data['H_tgt']), "celestial")
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
                          + (f" ⚠️ {nan_frac*100:.1f}% NaN" if nan_frac > 0 else ""),
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
                            if hasattr(WCS(data['H_tgt']), "celestial")
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
                ax_crop.set_title(f"⚠ {nf*100:.0f}% NaN" if nf > 0 else "", fontsize=8)

    mode_str   = "header" if cheat_rt else "circular"
    nbeams_str = ", ".join([f"{int(s) if s==int(s) else s}kpc: {global_nbeams.get(s, 20.0):.1f}b"
                            for s in [d['scale'] for d in processed_scales]])
    center_note = processed_scales[0]['center_note'] if processed_scales else ""
    fig.suptitle(f"{source_name} — Multi-scale ({mode_str}) — z={z:.4f}\n"
                 f"Crop: {nbeams_str}\n{center_note}", fontsize=11, y=0.995)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    scales_str = ", ".join([f"{int(s) if s==int(s) else s}kpc" for s in [d['scale'] for d in processed_scales]])
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
                fits.writeto(raw_fits_path, data['I_fmt_np'].astype(np.float32), H_i_fmt, overwrite=True)
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
# Functions for the multi-source RAW | T_X | RT_X grid comparison plot.

def _validate_source_has_scales(source_name: str, root: Path, scales: List[float]) -> bool:
    """Return True only if source has RAW and all T_Xkpc files."""
    src_dir  = root / source_name
    if not (src_dir / f"{source_name}.fits").exists():
        return False
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        if not (src_dir / f"{source_name}T{scale_int}kpc.fits").exists():
            print(f"  Missing T{scale_int}kpc.fits for {source_name}")
            return False
    return True

def get_classified_sources_from_loader(root: Path, scales: List[float]) -> Tuple[List[str], List[str]]:
    """
    Use the load_galaxies data loader to discover DE (class 50) and NDE (class 51) sources
    that have all required scale files present.
    """
    from utils.data_loader import load_galaxies
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
    """Look up redshift from a pre-loaded redshift dict; raise if missing or invalid."""
    z = slug_to_z.get(source_name, np.nan)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"Invalid redshift z={z} for {source_name}")
    return z

def _get_annotation_header(source_name: str, root: Path,
                            scale: Optional[float],
                            crop_fov_arcsec: float,
                            downsample_size: Tuple[int, int]) -> fits.Header:
    """
    Build a synthetic FITS header with the correct pixel scale for annotation
    (beam patch + scale bar) on a downsampled comparison-plot panel.
    Beam keywords (BMAJ/BMIN/BPA) are taken from the original file; pixel scale
    is derived from the actual crop FOV and the output image size.
    """
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
    # Overwrite pixel scale to match the actual cropped FOV
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
    """
    Add beam patch and scale bar to one panel of the comparison plot.
    Uses add_beam_patch_simple / add_scalebar_kpc_simple (plain-axis versions).
    `image_type` is 'RAW', 'T', or 'RT'.
    """
    try:
        # Determine FOV by looking up the relevant beam FWHM
        ref_scale = 50 if scale is None else scale
        ref_scale_int = int(ref_scale) if float(ref_scale).is_integer() else ref_scale
        t_path = root / source_name / f"{source_name}T{ref_scale_int}kpc.fits"
        _, H_ref, _ = read_fits_array_header_wcs(t_path)
        fwhm_as  = fwhm_major_as(H_ref)
        side_as  = global_nbeams * fwhm_as  # zoom_factor = 1.0

        ann_hdr = _get_annotation_header(source_name, root, scale, side_as, downsample_size)

        if image_type == 'RT':
            # Circular beam from redshift
            DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
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
                                   downsample_size: Tuple[int, int]
                                   ) -> Dict[str, np.ndarray]:
    """
    Load and process a single source for the comparison plot.
    Returns dict with keys 'RAW', 'T{X}', 'RT{X}' for each scale X.
    """
    z = _load_redshift(source_name, slug_to_z)
    print(f"  z={z:.4f} for {source_name}")

    I_raw, H_raw, W_raw = read_fits_array_header_wcs(root / source_name / f"{source_name}.fits")

    # Crop centre with manual offset
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
        """Bilinear downsample; restores NaN where majority of contributing pixels were NaN."""
        nan_mask = np.isnan(arr)
        t = torch.from_numpy(np.nan_to_num(arr, nan=0.0)).float().unsqueeze(0).unsqueeze(0)
        m = torch.from_numpy(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            result   = torch.nn.functional.interpolate(t, size=(Ho, Wo), mode="bilinear", align_corners=False)
            mask_d   = torch.nn.functional.interpolate(m, size=(Ho, Wo), mode="bilinear", align_corners=False)
        out = result.squeeze(0).squeeze(0).cpu().numpy()
        out[mask_d.squeeze(0).squeeze(0).cpu().numpy() > 0.5] = np.nan
        return out

    results = {}

    # Process each scale: T_X and RT_X
    for scale in scales:
        scale_int = int(scale) if float(scale).is_integer() else scale
        key_t = f'T{scale_int}'; key_rt = f'RT{scale_int}'
        T_nat, H_tgt, _ = read_fits_array_header_wcs(root / source_name / f"{source_name}T{scale_int}kpc.fits")
        side_as = global_nbeams * fwhm_major_as(H_tgt)  # zoom_factor = 1.0

        # T: reproject to RAW grid, crop, downsample
        T_on_raw = reproject_like(T_nat, H_tgt, H_raw)
        (T_crop,), _, _ = crop_to_side_arcsec_on_raw(T_on_raw, H_raw, side_as, center=(yc, xc))
        results[key_t] = _downsample_nan_safe(T_crop)

        # RT: convolve RAW, scale flux, crop, downsample
        ker    = circular_kernel_from_z(z, H_raw, fwhm_kpc=scale)
        I_smt  = convolve_fft(I_raw, ker, boundary="fill", fill_value=np.nan,
                              nan_treatment="interpolate", normalize_kernel=True,
                              psf_pad=True, fft_pad=True, allow_huge=True)
        DA_kpc   = COSMO.angular_diameter_distance(z).to_value(u.kpc)
        sigma_r  = (scale / DA_kpc) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        omega_t  = 2.0 * np.pi * sigma_r**2
        RT_raw   = I_smt * (omega_t / beam_solid_angle_sr(H_raw))
        (RT_crop,), _, _ = crop_to_side_arcsec_on_raw(RT_raw, H_raw, side_as, center=(yc, xc))
        results[key_rt] = _downsample_nan_safe(RT_crop)

        print(f"  {key_t} side={side_as:.0f}\" -> {results[key_t].shape}")

    # RAW: use T50 beam FWHM for crop size
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
    """
    Render one source row in the comparison plot.
    Columns: RAW | T25 | RT25 | T50 | RT50 | T100 | RT100
    Returns next grid_row index.
    """
    try:
        images = _process_source_for_comparison(
            source_name, root, scales, slug_to_z, global_nbeams, downsample_size)
        try:
            z = _load_redshift(source_name, slug_to_z)
        except Exception:
            z = None

        col_idx = 0
        cmap = plt.cm.viridis.copy(); cmap.set_bad('white', 1.0)

        # RAW column
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

        # T_X | RT_X pairs
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
    """
    Produce a multi-source comparison grid:
      columns: RAW | T25kpc | RT25kpc | T50kpc | RT50kpc | T100kpc | RT100kpc
      rows: DE sources (top group) then NDE sources (bottom group), separated by a gap.
    `annotate` toggles beam patches and scale bars on every panel.
    """
    de_indices  = [i for i, s in enumerate(sources) if s in de_sources_all]
    nde_indices = [i for i, s in enumerate(sources) if s in nde_sources_all]

    n_cols = 1 + len(scales) * 2  # RAW + (T + RT) per scale
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
        grid_row += 1  # skip the gap row
    for i in nde_indices:
        is_first = (grid_row == len(de_indices) + (1 if de_indices and nde_indices else 0))
        grid_row = _plot_comparison_row(
            sources[i], root, scales, slug_to_z, global_nbeams, downsample_size,
            gs, grid_row, n_cols, is_first_row=is_first, fig=fig, annotate=annotate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f"[comparison] Saved -> {output_path}")

# ================================= parallel helpers ==========================

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
            out_fits_dir=args_dict['out_fits_dir'], suffix=args_dict['suffix'])
        return (source_name, True, None)
    except Exception as e:
        import traceback
        return (source_name, False, traceback.format_exc())

def generate_diagnostic_histograms(root_dir: Path, scales: List[float],
                                   global_nbeams: Dict[float, float],
                                   output_path: Path):
    """Generate histograms showing FOV distribution per version."""
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
                    label=f'T{sc_str}kpc (μ={np.mean(data[scale]["fov"]):.0f}")',
                    edgecolor='black', linewidth=1.2)
            ax.axvline(np.mean(data[scale]['fov']), color=color, linestyle='--',
                       linewidth=2.5, alpha=0.9)
    ax.set_xlabel('Field of View (arcsec)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Sources',      fontsize=13, fontweight='bold')
    ax.set_title('FOV Distribution Per Version', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[diagnostics] ✓ Saved to {output_path}")
    for scale in scales:
        if data[scale]['fov']:
            sc_str = int(scale) if scale == int(scale) else scale
            print(f"  T{sc_str}kpc: target={global_nbeams.get(scale, 20.0):.1f}, "
                  f"mean FOV={np.mean(data[scale]['fov']):.1f}\"")

# --------------------------------- CLI ---------------------------------------

def parse_tuple3(txt: str) -> Tuple[int,int,int]:
    vals = [int(v) for v in str(txt).strip().split(",")]
    if len(vals) == 2: return (1, vals[0], vals[1])
    if len(vals) == 3: return (vals[0], vals[1], vals[2])
    raise argparse.ArgumentTypeError("Use H,W or C,H,W")

def main():
    ap = argparse.ArgumentParser(
        description="Multi-scale per-source montages + optional multi-source comparison plot.")
    DEFAULT_ROOT  = Path("/users/mbredber/scratch/data/PSZ2/fits")
    DEFAULT_OUT   = Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/montages")
    DEFAULT_Z_CSV = Path("/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv")

    # --- core arguments ---
    ap.add_argument("--root",     type=Path,  default=DEFAULT_ROOT)
    ap.add_argument("--z-csv",    type=Path,  default=DEFAULT_Z_CSV)
    ap.add_argument("--out",      type=Path,  default=DEFAULT_OUT)
    ap.add_argument("--crop",     type=parse_tuple3, default="512,512")
    ap.add_argument("--down",     type=parse_tuple3, default="128,128")
    ap.add_argument("--scales",   type=str,   default="25, 50, 100")
    ap.add_argument("--fov-arcmin", type=float, default=50.0)
    ap.add_argument("--cheat-rt", action="store_true", default=False)
    ap.add_argument("--force",    action="store_true", default=False)
    ap.add_argument("--only-offsets", action="store_true")
    ap.add_argument("--only",     type=str,   default="")
    ap.add_argument("--save-fits", action="store_true", default=True)
    ap.add_argument("--fits-out", type=Path,
                    default=Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits"))
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Parallel workers for montage (default: all CPUs)")
    ap.add_argument("--only-one", type=str, default=None,
                    help="Debug: process only this single source (montage + FITS, forces --n-workers 1).")

    # --- comparison plot arguments ---
    ap.add_argument("--no-montage", action="store_true", default=False,
                    help="Skip the per-source montage pipeline.")
    ap.add_argument("--comparison-plot", action="store_true", default=False,
                    help="Produce the multi-source comparison plot.")
    ap.add_argument("--no-annotate", action="store_true", default=False,
                    help="Omit beam patches and scale bars from the comparison plot.")
    ap.add_argument("--comp-out", type=Path,
                    default=Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/comparison_plot.pdf"),
                    help="Output path for the comparison plot.")
    ap.add_argument("--comp-sources", type=str, default=None,
                    help="Comma-separated source names for comparison plot. "
                         "If omitted, randomly selects --n-de + --n-nde sources.")
    ap.add_argument("--n-de",  type=int, default=3,
                    help="Number of DE sources for comparison plot (default: 3).")
    ap.add_argument("--n-nde", type=int, default=3,
                    help="Number of NDE sources for comparison plot (default: 3).")
    ap.add_argument("--comp-seed", type=int, default=10,
                    help="Random seed for comparison source selection (default: 10).")
    ap.add_argument("--comp-figsize", type=str, default="10,9",
                    help="Comparison plot figure size as W,H in inches (default: 10,9).")

    args = ap.parse_args()

    # --- shared setup ---
    print(f"[init] Loading redshift table from {args.z_csv}")
    slug_to_z = load_z_table(args.z_csv)
    print(f"[init] Loaded {len(slug_to_z)} redshifts")
    
    # When --only-one is given, restrict to that single source for debugging.
    if args.only_one:
        args.only = args.only_one  # reuses existing --only filter
        args.n_workers = 1         # single worker keeps debug output readable
        args.force = True          # always regenerate
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
    suffix     = "cheat" if args.cheat_rt else "circ"
    Ho, Wo     = _canon_size(args.down)[-2:]
    out_fits_dir = args.fits_out if args.fits_out else args.out

    print(f"[init] Processing scales: {scale_values}")
    print("\n" + "="*80)
    print("STEP 1: Computing global crop size (NaN-free across all files)")
    print("="*80)
    global_nbeams = compute_global_nbeams_per_version(args.root, scale_values)
    print(f"[GLOBAL N_BEAMS] Per-version targets:")
    for scale in scale_values:
        sc_str = int(scale) if scale == int(scale) else scale
        print(f"  T{sc_str}kpc: {global_nbeams[scale]:.1f} beams")

    # --- diagnostic histograms ---
    diag_path = args.out.parent / "diagnostics_nbeams_distribution.png"
    generate_diagnostic_histograms(args.root, scale_values, global_nbeams, diag_path)

    # ========================= PER-SOURCE MONTAGE =========================
    if not args.no_montage:
        print("\n" + "="*80)
        print("STEP 2: Per-source multi-scale montages")
        print("="*80)
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
                         force=args.force, out_fits_dir=out_fits_dir)
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

        # Verification report
        print("\n" + "="*80)
        print("VERIFICATION REPORT: SUB File Coverage")
        print("="*80)
        for scale in scale_values:
            with_sub_in, without_sub_in, with_sub_out, missing_sub_out = [], [], [], []
            for name, raw_path, t_path, sub_path, y_chosen in find_pairs_in_tree(args.root, scale):
                if args.only_offsets and name not in OFFSETS_PX: continue
                if only_names and name not in only_names: continue
                t_label  = f"T{int(y_chosen) if float(y_chosen).is_integer() else y_chosen}kpc"
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

    # ========================= COMPARISON PLOT =========================
    if args.comparison_plot and not args.only_one:
        print("\n" + "="*80)
        print("STEP 3: Multi-source comparison plot")
        print("="*80)
        annotate = not args.no_annotate
        figsize  = tuple(float(x) for x in args.comp_figsize.split(','))

        # Compute a single global nbeams value for the comparison plot (T50 reference)
        comp_nbeams = compute_global_nbeams_min_t50(args.root)
        if comp_nbeams is None:
            print("[ERROR] Could not compute global nbeams for comparison plot — skipping.")
        else:
            if args.comp_sources:
                # User-specified sources
                sources = [s.strip() for s in args.comp_sources.split(',') if s.strip()]
                print(f"Discovering all sources for classification...")
                de_sources_all, nde_sources_all = get_classified_sources_from_loader(args.root, scale_values)
                sources = [s for s in sources if s in de_sources_all or s in nde_sources_all
                           or print(f"Warning: {s} not in classified sources, skipping") is None]
                if not sources:
                    print("[ERROR] None of the provided comp-sources are valid — skipping.")
                    return
            else:
                # Random selection with optional seed
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

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()