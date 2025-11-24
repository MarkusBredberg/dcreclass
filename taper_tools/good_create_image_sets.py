#!/usr/bin/env python3
"""
Merged batch script:
- For each source, load RAW and T_X (X in {25,50,100,...} kpc).
- Build RT by convolving RAW to the T_X beam (elliptical Gaussian) and rescaling Jy/beam.
- Make a 3×2 montage (rows: RAW, RT, T_X ; cols: original vs formatted).
- Optional: save formatted RAW/RT/T_X FITS with updated WCS.
"""

import argparse, torch
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
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
ARCSEC_PER_RAD = 206264.80624709636

def _cd_matrix_rad(h):
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
    J = _cd_matrix_rad(h)
    dx = np.hypot(J[0,0], J[1,0])
    dy = np.hypot(J[0,1], J[1,1])
    return dx*ARCSEC_PER_RAD, dy*ARCSEC_PER_RAD

def fwhm_major_as(h):
    return max(float(h["BMAJ"]), float(h["BMIN"])) * 3600.0

def _fwhm_as_to_sigma_rad(fwhm_as: float) -> float:
    return (float(fwhm_as)/(2.0*np.sqrt(2.0*np.log(2.0)))) * (np.pi/(180.0*3600.0))

def beam_cov_world(h):
    """Return 2×2 covariance (σ^2) in world radians for the header beam."""
    bmaj_as = abs(float(h['BMAJ']))*3600.0
    bmin_as = abs(float(h['BMIN']))*3600.0
    pa_deg  = float(h.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th      = np.deg2rad(pa_deg)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], float)
    S = np.diag([sx*sx, sy*sy])
    return R @ S @ R.T

def beam_solid_angle_sr(h):
    bmaj = abs(float(h['BMAJ'])) * np.pi/180.0
    bmin = abs(float(h['BMIN'])) * np.pi/180.0
    return (np.pi/(4.0*np.log(2.0))) * bmaj * bmin

def kernel_from_beams(raw_hdr, tgt_hdr):
    """
    Elliptical Gaussian kernel that maps RAW beam -> TARGET beam.
    In world coords: C_ker = C_tgt - C_raw (clip to PSD), then map to pixel coords.
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

def read_fits_array_header_wcs(fpath: Path):
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
    if header.get('OBJCTRA') and header.get('OBJCTDEC'):
        return SkyCoord(header['OBJCTRA'], header['OBJCTDEC'], unit=(u.hourangle, u.deg))
    if header.get('RA_TARG') and header.get('DEC_TARG'):
        return SkyCoord(header['RA_TARG']*u.deg, header['DEC_TARG']*u.deg)
    if 'CRVAL1' in header and 'CRVAL2' in header:
        return SkyCoord(header['CRVAL1']*u.deg, header['CRVAL2']*u.deg)
    return None

def wcs_after_center_crop_and_resize(header, H0, W0, Hc, Wc, Ho, Wo, y0, x0):
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
    finite = np.isfinite(arr)
    if not finite.any(): return 0.0, 1.0
    vals = arr[finite]
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if vmin == vmax: vmax = vmin + 1.0
    return float(vmin), float(vmax)

# ------------------------------- formatting ----------------------------------
def _canon_size(sz):
    if isinstance(sz, (tuple, list)):
        if len(sz) == 2:  return (1, sz[0], sz[1])
        if len(sz) == 3:  return (sz[0], sz[1], sz[2])
    raise ValueError("size must be H,W or C,H,W")

def center_crop_at(arr, ny_target, nx_target, cy, cx):
    ny, nx = arr.shape
    y0 = int(round(cy - ny_target/2)); x0 = int(round(cx - nx_target/2))
    y0 = max(0, min(y0, ny - ny_target)); x0 = max(0, min(x0, nx - nx_target))
    return arr[y0:y0+ny_target, x0:x0+nx_target]

def crop_to_fov_on_raw(I, Hraw, fov_arcmin, *arrs, center=None):
    """Square crop on RAW grid with side=fov_arcmin (arcmin). If `center` is given,
    it must be (cy,cx) in INPUT pixels on RAW; otherwise use the image centre."""
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
    """Square crop on RAW grid with side length in arcsec. If `center` is given,
    it must be (cy,cx) in INPUT pixels on RAW; otherwise use the image centre."""
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


def apply_formatting(image: torch.Tensor,
                     crop_size: Tuple[int, int, int] = (1, 128, 128),
                     downsample_size: Tuple[int, int, int] = (1, 128, 128),
                     center_px: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Center-crop and resize a single-channel tensor (no PIL).
    Args:
      image: [C,H0,W0] or [H0,W0]
      crop_size:      (C,Hc,Wc) or (Hc,Wc)
      downsample_size:(C,Ho,Wo) or (Ho,Wo)
      center_px: (y,x) in pixels in the *input* image to center the crop on.
    Returns:
      [C,Ho,Wo]
    """
    crop_size = _canon_size(crop_size)
    downsample_size = _canon_size(downsample_size)

    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)
    if image.dim() == 3:
        H0, W0 = image.shape[-2], image.shape[-1]
        img = image
    elif image.dim() == 2:
        H0, W0 = image.shape
        img = image.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image dims: {image.shape}")

    if crop_size[0] == 1 or downsample_size[0] == 1:
        img = img.mean(dim=0, keepdim=True)

    _, Hc, Wc = crop_size
    _, Ho, Wo = downsample_size

    if center_px is None:
        y0, x0 = H0 // 2, W0 // 2
    else:
        y0, x0 = int(round(center_px[0])), int(round(center_px[1]))

    y1, y2 = y0 - Hc // 2, y0 + Hc // 2
    x1, x2 = x0 - Wc // 2, x0 + Wc // 2
    y1, y2 = max(0, y1), min(H0, y2)
    x1, x2 = max(0, x1), min(W0, x2)

    crop = img[:, y1:y2, x1:x2].unsqueeze(0)   # [1,C,*,*]
    resized = F.interpolate(crop, size=(Ho, Wo), mode='bilinear', align_corners=False)
    return resized.squeeze(0)                   # [C,Ho,Wo]

# ------------------------------ IO helpers -----------------------------------
def t_product_path(src_dir: Path, name: str, scale_kpc: str) -> Path:
    return src_dir / f"{name}T{scale_kpc}kpc.fits"

def find_pairs_in_tree(root: Path, desired_kpc: float) -> Iterable[Tuple[str, Path, Path, float]]:
    """
    Yield (name, raw_path, t_path, chosen_kpc) where chosen_kpc is the available T scale
    nearest to desired_kpc among files like <name>T{Y}kpc.fits.
    """
    import re
    pat = re.compile(r"T([0-9]+(?:\.[0-9]+)?)kpc\.fits$", re.IGNORECASE)
    for src_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        name = src_dir.name
        raw_path = src_dir / f"{name}.fits"
        if not raw_path.exists():
            continue
        candidates = []
        for fp in src_dir.glob(f"{name}T*kpc.fits"):
            m = pat.search(fp.name)
            if m:
                y = float(m.group(1))
                candidates.append((abs(y - desired_kpc), y, fp))
        if candidates:
            _, ybest, fbest = sorted(candidates, key=lambda t: (t[0], t[1]))[0]
            yield name, raw_path, fbest, ybest
            

def compute_global_nbeams_min(root_dir):
    """
    Scan all subdirs under root_dir, find every *Tkpc.fits,
    compute n_beams = min(FOV_x, FOV_y)/FWHM, return the smallest.
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
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=float)
    n = np.isnan(arr).sum()
    if n > 0:
        print(f"[nancheck] {path}: {n} NaNs")

# ------------------------------ montage per source ---------------------------
def make_montage(source_name: str,
                 raw_path: Path,
                 t_path: Path,
                 rt_label: str,
                 t_label: str,
                 crop_size=(1, 512, 512),
                 downsample_size=(1, 128, 128),
                 out_png: Path = None,
                 save_fits: bool = False,
                 out_fits_dir: Optional[Path] = None,
                 fov_arcmin: float = 50.0,
                 cheat_rt: bool = False):
    """
    Build RT on RAW grid, then 3×2 WCS montage:
      rows: RAW, RT, T_X
      cols: original | formatted
    Optionally save formatted RAW/RT/T_X as FITS with updated WCS.
    """
    # --- load RAW and T_X
    I_raw,  H_raw,  W_raw  = read_fits_array_header_wcs(raw_path)
    T_nat,  H_tgt,  W_tgt  = read_fits_array_header_wcs(t_path)
    
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

    # --- reproject T to RAW grid for consistent multiplication and later alignment
    T_on_raw = reproject_like(T_nat, H_tgt, H_raw)

    # --- RT kernel on RAW grid (to map RAW beam -> T beam), then convolve & rescale
    if cheat_rt:
        ker = kernel_from_beams(H_raw, H_tgt)
        I_smt = convolve_fft(
            I_raw, ker, boundary="fill", fill_value=np.nan,
            nan_treatment="interpolate", normalize_kernel=True,
            psf_pad=True, fft_pad=True, allow_huge=True
        )
        scale = beam_solid_angle_sr(H_tgt) / beam_solid_angle_sr(H_raw)
        RT_rawgrid = I_smt * scale  # Jy/beam_tgt
    else:
        # Circular path (isotropic target with same area as T):
        # --- RT kernel on RAW grid (circular path): circularize TARGET beam by area ---
        C_raw = beam_cov_world(H_raw)           # RAW beam covariance (world, rad^2)
        C_tgt = beam_cov_world(H_tgt)           # TARGET (from T_X header)

        # Isotropic target with SAME area: sigma^2 = sqrt(det(C_tgt))
        sigma2 = float(np.sqrt(max(0.0, np.linalg.det(C_tgt))))
        C_tgt_circ = np.array([[sigma2, 0.0], [0.0, sigma2]], float)

        # Kernel covariance in world coords (PSD-clipped): C_ker = C_tgt_circ - C_raw
        C_ker_w = C_tgt_circ - C_raw
        w, V = np.linalg.eigh(C_ker_w); w = np.clip(w, 0.0, None)
        C_ker_w = (V * w) @ V.T

        # Map world→RAW-pixel, then build Gaussian2DKernel on RAW grid
        J = _cd_matrix_rad(H_raw); Jinv = np.linalg.inv(J)
        Cpix = Jinv @ C_ker_w @ Jinv.T
        wp, Vp = np.linalg.eigh(Cpix); wp = np.clip(wp, 1e-18, None)
        s_minor = float(np.sqrt(wp[0])); s_major = float(np.sqrt(wp[1]))
        theta   = float(np.arctan2(Vp[1,1], Vp[0,1]))
        nker    = int(np.ceil(8.0*max(s_major, s_minor))) | 1
        ker = Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

        I_smt = convolve_fft(
            I_raw, ker, boundary="fill", fill_value=np.nan,
            nan_treatment="interpolate", normalize_kernel=True,
            psf_pad=True, fft_pad=True, allow_huge=True
        )

        # Units: preserve Jy/beam; rescale by solid-angle ratio (area is unchanged by circularization)
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
    import re
    kpc_match = re.search(r'T(\d+(?:\.\d+)?)kpc', t_label)
    if kpc_match:
        kpc_value = float(kpc_match.group(1))
        # Apply scale-dependent zoom factor:
        zoom_25 = 1.42 # - 25kpc: zoom OUT
        zoom_100 = 0.49 # - 100kpc: zoom IN 
        if kpc_value <= 30:  # 25kpc and similar
            zoom_factor = zoom_25
        elif kpc_value >= 90:  # 100kpc and similar
            zoom_factor = zoom_100
        else:  # intermediate scales
            # Linear interpolation between 30-90 kpc
            #zoom_factor = zoom_25 - (kpc_value - 30) * (zoom_25 - zoom_100) / (90 - 30)
            #print(f"[zoom] kpc={kpc_value:.1f} → zoom_factor={zoom_factor:.3f}")
            zoom_factor = 1.0  # neutral zoom for intermediate scales
    else:
        zoom_factor = 1.0  # default if no match
    
    # --- equal-beams crop: side = global_min_beams * FWHM_T50 * zoom_factor ---
    fwhm_t50_as = fwhm_major_as(H_tgt)
    side_as = make_montage._global_nbeams * fwhm_t50_as * zoom_factor
    if getattr(make_montage, "GLOBAL_NBEAMS", None):
        # equal-beams crop using global min beam count with zoom adjustment
        fwhm_t50_as = fwhm_major_as(H_tgt)
        side_as = make_montage.GLOBAL_NBEAMS * fwhm_t50_as * zoom_factor
        (I_crop, RT_crop, T_crop), (nyc, nxc), (cy_raw, cx_raw) = crop_to_side_arcsec_on_raw(
            I_raw, H_raw, side_as, RT_rawgrid, T_on_raw, center=(yc_i, xc_i)
        )
    else:
        # fallback: just use your default FOV crop with zoom adjustment
        adjusted_fov = fov_arcmin * zoom_factor
        (I_crop, RT_crop, T_crop), (nyc, nxc), (cy_raw, cx_raw) = crop_to_fov_on_raw(
            I_raw, H_raw, adjusted_fov, RT_rawgrid, T_on_raw, center=(yc_i, xc_i)
        )
    # Optional downsample to a fixed display size (keeps the FOV content)
    Ho, Wo = _canon_size(downsample_size)[-2:]
    def _maybe_downsample(arr, Ho, Wo):
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        with torch.no_grad():
            y = torch.nn.functional.interpolate(t, size=(Ho, Wo), mode="bilinear", align_corners=False)
        return y.squeeze(0).squeeze(0).cpu().numpy()

    I_fmt_np  = _maybe_downsample(I_crop,  Ho, Wo)
    RT_fmt_np = _maybe_downsample(RT_crop, Ho, Wo)
    T_fmt_np  = _maybe_downsample(T_crop,  Ho, Wo)

    # --- WCS for formatted panels (FOV crop on RAW grid, then resize Ho×Wo)
    H0_i, W0_i = I_raw.shape
    Hc, Wc = nyc, nxc
    W_i_fmt, H_i_fmt = wcs_after_center_crop_and_resize(
        H_raw, H0_i, W0_i, Hc, Wc, Ho, Wo, int(round(cy_raw)), int(round(cx_raw))
    )
    W_rt_fmt, H_rt_fmt = W_i_fmt, H_i_fmt
    W_t_fmt,  H_t_fmt  = W_i_fmt, H_i_fmt   # T was reprojected to RAW, so use RAW WCS too

    # --- plotting ranges. Only used for plotting. Not for FITS outputs.
    # --- plotting ranges (only for display; FITS already use *_np arrays above)
    vmin_I, vmax_I   = robust_vmin_vmax(I_raw)
    vmin_RT, vmax_RT = robust_vmin_vmax(RT_rawgrid)
    vmin_T, vmax_T   = robust_vmin_vmax(T_nat)

    I_orig_np  = I_raw
    RT_orig_np = RT_rawgrid

    # --- figure: 3×2 with WCS axes
    fig = plt.figure(figsize=(12, 13), constrained_layout=True)
    ax00 = fig.add_subplot(3,2,1, projection=W_raw)      # RAW original
    ax01 = fig.add_subplot(3,2,2, projection=W_i_fmt)    # RAW formatted
    ax10 = fig.add_subplot(3,2,3, projection=W_raw)      # RT original (on RAW grid)
    ax11 = fig.add_subplot(3,2,4, projection=W_rt_fmt)   # RT formatted
    ax20 = fig.add_subplot(3,2,5, projection=W_raw)      # T original (native grid)
    ax21 = fig.add_subplot(3,2,6, projection=W_t_fmt)    # T formatted (native grid)

    im00 = ax00.imshow(I_orig_np,  origin="lower", vmin=vmin_I,  vmax=vmax_I);   ax00.set_title("RAW (original)")
    im01 = ax01.imshow(I_fmt_np,   origin="lower", vmin=vmin_I,  vmax=vmax_I);   ax01.set_title(f"RAW formatted (FOV {fov_arcmin:.1f}′ → {Ho}×{Wo}, zoom={zoom_factor:.2f})")
    im10 = ax10.imshow(RT_orig_np, origin="lower", vmin=vmin_RT, vmax=vmax_RT);  ax10.set_title(f"{rt_label}=RAW⊗G (original, RAW grid)")
    im11 = ax11.imshow(RT_fmt_np,  origin="lower", vmin=vmin_RT, vmax=vmax_RT);  ax11.set_title(f"{rt_label} formatted (FOV {fov_arcmin:.1f}′ → {Ho}×{Wo}, zoom={zoom_factor:.2f})")
    im20 = ax20.imshow(T_on_raw,   origin="lower", vmin=vmin_T,  vmax=vmax_T);   ax20.set_title(f"{t_label} (original, RAW grid)")
    im21 = ax21.imshow(T_fmt_np,   origin="lower", vmin=vmin_T,  vmax=vmax_T);   ax21.set_title(f"{t_label} formatted (FOV {fov_arcmin:.1f}′ → {Ho}×{Wo}, zoom={zoom_factor:.2f})")

    fig.colorbar(im00, ax=[ax00, ax01], shrink=0.85, label="RAW [Jy/beam_raw]")
    fig.colorbar(im10, ax=[ax10, ax11], shrink=0.85, label=f"{rt_label} [Jy/beam_tgt]")
    fig.colorbar(im20, ax=[ax20, ax21], shrink=0.85, label=t_label)

    fig.suptitle(f"{source_name} — {rt_label} using {t_label} — {center_note}", fontsize=13)
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- optional FITS outputs for the formatted panels
    if save_fits:
        out_fits_dir = (out_fits_dir or out_png.parent)
        out_fits_dir.mkdir(parents=True, exist_ok=True)
        Ho, Wo = _canon_size(downsample_size)[-2:]
        fits.writeto(out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_old_{zoom_factor}.fits",
                     I_fmt_np.astype(np.float32), H_i_fmt, overwrite=True)
        fits.writeto(out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_old_{zoom_factor}.fits",
                    RT_fmt_np.astype(np.float32), H_rt_fmt, overwrite=True)
        fits.writeto(out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}_old_{zoom_factor}.fits",
                    T_fmt_np.astype(np.float32), H_t_fmt, overwrite=True)
        
        report_nans(out_fits_dir / f"{source_name}_RAW_fmt_{Ho}x{Wo}_old_{zoom_factor}.fits")
        report_nans(out_fits_dir / f"{source_name}_{rt_label}_fmt_{Ho}x{Wo}_old_{zoom_factor}.fits")
        report_nans(out_fits_dir / f"{source_name}_{t_label}_fmt_{Ho}x{Wo}_old_{zoom_factor}.fits")


# --------------------------------- CLI ---------------------------------------
def parse_tuple3(txt: str) -> Tuple[int,int,int]:
    vals = [int(v) for v in str(txt).strip().split(",")]
    if len(vals) == 2: return (1, vals[0], vals[1])
    if len(vals) == 3: return (vals[0], vals[1], vals[2])
    raise argparse.ArgumentTypeError("Use H,W or C,H,W")

def main():
    ap = argparse.ArgumentParser(description="3×2 montages per source: RAW, RT=RAW⊗G, and T_X (X in kpc).")
    DEFAULT_ROOT = Path("/users/mbredber/scratch/data/PSZ2/fits")
    DEFAULT_OUT  = Path("/users/mbredber/scratch/create_image_sets_outputs/montages")

    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Root directory with per-source subfolders (/<name>/<name>.fits and T_X).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output directory for PNG montages. Default: {DEFAULT_OUT}")
    ap.add_argument("--crop", type=parse_tuple3, default="512,512",
                    help="Crop size H,W or C,H,W in input pixels. Default: 512,512")
    ap.add_argument("--down", type=parse_tuple3, default="128,128",
                    help="Downsample size H,W or C,H,W. Default: 128,128")
    ap.add_argument("--scales", type=str, default="100",
                    help="Comma-separated requested RT scales in kpc (any values; nearest available T_Y is used).")
    ap.add_argument("--fov-arcmin", type=float, default=50.0,
                    help="Square FOV (arcmin) for the formatted column; crop is on RAW grid.")
    ap.add_argument("--only-offsets", action="store_true",
                    help="Process only sources listed in OFFSETS_PX.")
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated source names to include exclusively.")
    ap.add_argument("--save-fits", action="store_true", default=True,
                    help="Also write formatted RAW/RT/T_X FITS for each montage.")
    ap.add_argument("--fits-out", type=Path, default=Path("/users/mbredber/scratch/create_image_sets_outputs/processed_psz2_fits"),
                    help="Directory for formatted FITS (defaults to the montage folder).")

    args = ap.parse_args()
    
    # store it in a global so make_montage can read it
    make_montage.GLOBAL_NBEAMS = compute_global_nbeams_min(args.root) * 1.85 # Larger factor than 1.85 makes PSZ2G048.10+57.16 too small

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    only_names = set(s.strip() for s in args.only.split(",") if s.strip())

    n_ok = 0
    n_skip = 0
    for scale in scales:
        try:
            x_req = float(scale)
        except Exception:
            print(f"[SKIP] invalid scale {scale!r}")
            continue
        rt_label = f"RT{int(x_req) if x_req.is_integer() else x_req}kpc"
        for name, raw_path, t_path, y_chosen in find_pairs_in_tree(args.root, x_req):
            if args.only_offsets and name not in OFFSETS_PX:
                continue
            if only_names and name not in only_names:
                continue
            try:
                t_label = f"T{int(y_chosen) if float(y_chosen).is_integer() else y_chosen}kpc"
                out_png = args.out / f"{name}_montage_{rt_label}_{t_label}_old.png"
                make_montage(name, raw_path, t_path, rt_label, t_label,
                            crop_size=args.crop,
                            downsample_size=args.down,
                            out_png=out_png,
                            save_fits=args.save_fits,
                            out_fits_dir=args.fits_out,
                            fov_arcmin=args.fov_arcmin)
                #print(f"[OK] {name} {rt_label} using {t_label} → {out_png}")
                n_ok += 1
            except Exception as e:
                print(f"[SKIP] {name} {rt_label}: {e}")
                n_skip += 1

    print(f"Done. Wrote {n_ok} montages. Skipped {n_skip}.")

if __name__ == "__main__":
    main()