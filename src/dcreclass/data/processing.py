"""PSZ2 image processing: kernels, cropping, NaN analysis, per-source pipeline."""

import csv, io, os, re
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.cosmology import Planck18 as COSMO
from astropy.io import fits

from dcreclass.utils.fits import (
    ARCSEC_PER_RAD,
    _cd_matrix_rad, arcsec_per_pix, fwhm_major_as,
    beam_cov_world, beam_solid_angle_sr, kernel_from_beams,
    read_fits_array_header_wcs, reproject_like,
    header_cluster_coord, wcs_after_center_crop_and_resize,
)


def circular_cov_kpc(z, fwhm_kpc=50.0):
    """Circular 2x2 covariance in world coords for FWHM=fwhm_kpc at redshift z."""
    if z is None or not np.isfinite(z) or z <= 0:
        return None
    fwhm_kpc = float(fwhm_kpc)
    if fwhm_kpc <= 0:
        raise ValueError("fwhm_kpc must be positive")
    DA_kpc    = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_rad = fwhm_kpc / DA_kpc
    sigma     = theta_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma2    = float(sigma ** 2)
    return np.array([[sigma2, 0.0], [0.0, sigma2]], float)


def circular_kernel_from_z(z, raw_hdr, fwhm_kpc=50.0):
    """Circular Gaussian convolution kernel on the RAW pixel grid."""
    C_raw  = beam_cov_world(raw_hdr)
    C_circ = circular_cov_kpc(z, fwhm_kpc=fwhm_kpc)
    if C_circ is None:
        raise ValueError(f"Invalid redshift z={z} for circular kernel")
    C_ker_world = C_circ - C_raw
    w, V = np.linalg.eigh(C_ker_world)
    w    = np.clip(w, 0.0, None)
    C_ker_world = (V * w) @ V.T
    J    = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker_world @ Jinv.T
    wp, Vp  = np.linalg.eigh(Cpix)
    wp      = np.clip(wp, 1e-18, None)
    s_minor = float(np.sqrt(wp[0]))
    s_major = float(np.sqrt(wp[1]))
    theta   = float(np.arctan2(Vp[1, 1], Vp[0, 1]))
    nker    = int(np.ceil(8.0 * max(s_major, s_minor))) | 1
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)


def load_z_table(csv_path) -> Dict[str, float]:
    """Load redshift table from CSV. Returns {slug: z}."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Redshift table not found: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        raw = f.read()
    if not raw.strip():
        raise ValueError(f"Redshift table appears empty: {csv_path}")
    lines = [ln for ln in raw.splitlines()
             if ln.strip() and not ln.lstrip().startswith('#')]
    if len(lines) < 2:
        raise ValueError("Redshift table has headers but no data rows.")
    rdr = csv.DictReader(io.StringIO('\n'.join(lines)))
    if rdr.fieldnames is None or 'slug' not in rdr.fieldnames or 'z' not in rdr.fieldnames:
        raise ValueError(f"CSV missing required headers 'slug,z'; got {rdr.fieldnames!r}")
    out: Dict[str, float] = {}
    for row in rdr:
        slug = (row.get('slug') or '').strip()
        zstr = (row.get('z') or '').strip()
        if not slug:
            continue
        if zstr.lower() in ('', 'nan', 'none'):
            out[slug] = np.nan
        else:
            try:
                out[slug] = float(zstr)
            except Exception:
                out[slug] = np.nan
    if not out:
        raise ValueError(f"No rows parsed from {csv_path}")
    return out


def _canon_size(sz):
    """Canonicalize size to (C, H, W)."""
    if isinstance(sz, (tuple, list)):
        if len(sz) == 2: return (1, sz[0], sz[1])
        if len(sz) == 3: return (sz[0], sz[1], sz[2])
    raise ValueError("size must be H,W or C,H,W")


def crop_to_side_arcsec_on_raw(I, Hraw, side_arcsec, *arrs, center=None):
    """Square crop on RAW grid with side length in arcsec."""
    asx, asy = arcsec_per_pix(Hraw)
    nx = int(round(side_arcsec / asx))
    ny = int(round(side_arcsec / asy))
    m  = max(1, min(nx, ny))
    nx = min(m, I.shape[1]); ny = min(m, I.shape[0])
    if center is None:
        cy, cx = (I.shape[0] - 1) / 2.0, (I.shape[1] - 1) / 2.0
    else:
        cy, cx = float(center[0]), float(center[1])
    y0 = int(round(cy - ny / 2)); x0 = int(round(cx - nx / 2))
    y0 = max(0, min(y0, I.shape[0] - ny))
    x0 = max(0, min(x0, I.shape[1] - nx))
    cy_eff, cx_eff = y0 + ny / 2.0, x0 + nx / 2.0
    out = [a[y0:y0+ny, x0:x0+nx] for a in (I,) + arrs]
    return out, (ny, nx), (cy_eff, cx_eff)


def find_pairs_in_tree(root: Path,
                       desired_kpc: float) -> Iterable[Tuple[str, Path, Path, Path, float]]:
    """Yield (name, raw_path, t_path, sub_path, chosen_kpc) for each source directory."""
    pat_t = re.compile(r"T([0-9]+(?:\.[0-9]+)?)kpc\.fits$", re.IGNORECASE)
    for src_dir in sorted(p for p in root.glob('*') if p.is_dir()):
        name     = src_dir.name
        raw_path = src_dir / f"{name}.fits"
        if not raw_path.exists():
            continue
        candidates = []
        for fp in src_dir.glob(f"{name}T*kpc.fits"):
            if 'SUB' in fp.name.upper():
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
    """Report number of NaN pixels in a FITS file."""
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=float)
    n = np.isnan(arr).sum()
    if n > 0:
        print(f"[nancheck] {path}: {n} NaNs")


def _nan_free_centred_square_side_as(arr: np.ndarray, header) -> float:
    """Largest NaN-free centred square side in arcsec (binary search)."""
    if not np.any(np.isfinite(arr)):
        return 0.0
    H, W = arr.shape
    asx, asy = arcsec_per_pix(header)
    cy, cx   = H / 2.0, W / 2.0
    lo, hi   = 1, min(H, W)
    best     = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        y0  = max(0, min(int(round(cy - mid / 2)), H - mid))
        x0  = max(0, min(int(round(cx - mid / 2)), W - mid))
        if np.all(np.isfinite(arr[y0:y0+mid, x0:x0+mid])):
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return best * min(asx, asy)


def compute_global_nbeams_per_version(root_dir: Path,
                                      scales: List[float]) -> Dict[float, float]:
    """Per-version global minimum n_beams across all sources."""
    print('[compute_global_nbeams] Computing per-version n_beams...')
    global_nbeams: Dict[float, float] = {}
    for scale in scales:
        scale_str   = f"{int(scale)}" if scale == int(scale) else f"{scale}"
        nbeams_list = []
        for src_dir in sorted(p for p in root_dir.glob('*') if p.is_dir()):
            t_path = src_dir / f"{src_dir.name}T{scale_str}kpc.fits"
            if not t_path.exists():
                continue
            try:
                arr, hdr, _ = read_fits_array_header_wcs(t_path)
                max_side_as = _nan_free_centred_square_side_as(arr, hdr)
                if max_side_as <= 0:
                    continue
                nbeams_list.append(max_side_as / fwhm_major_as(hdr))
            except Exception:
                continue
        if nbeams_list:
            global_nbeams[scale] = min(nbeams_list)
            print(f"[scan] T{scale_str}kpc: n_beams={global_nbeams[scale]:.1f} "
                  f"(min={min(nbeams_list):.1f}, max={max(nbeams_list):.1f})")
        else:
            global_nbeams[scale] = 100.0
    return global_nbeams


def compute_global_nbeams_min_t50(root_dir: Path) -> Optional[float]:
    """Global minimum n_beams from T50kpc files (comparison plot reference)."""
    n_beams = []
    for tfile in Path(root_dir).rglob('*T50kpc.fits'):
        try:
            with fits.open(tfile) as hdul:
                h = hdul[0].header
            fwhm = max(float(h['BMAJ']), float(h['BMIN'])) * 3600.0
            if 'CD1_1' in h:
                dx = np.hypot(h['CD1_1'], h.get('CD2_1', 0)) * ARCSEC_PER_RAD
                dy = np.hypot(h.get('CD1_2', 0), h.get('CD2_2', 0)) * ARCSEC_PER_RAD
            else:
                dx = abs(h.get('CDELT1', 1)) * 3600.0
                dy = abs(h.get('CDELT2', 1)) * 3600.0
            fovx = int(h['NAXIS1']) * dx
            fovy = int(h['NAXIS2']) * dy
            n_beams.append(min(fovx, fovy) / max(fwhm, 1e-9))
        except Exception as e:
            print(f"[scan] skip {tfile}: {e}")
    if not n_beams:
        print('[scan] No T50kpc files found')
        return None
    nmin = min(n_beams)
    print(f"[scan] global_nbeams_min (T50) = {nmin:.2f} across {len(n_beams)} files")
    return nmin


def check_nan_fraction(arr: np.ndarray, name: str = '') -> float:
    """Fraction of NaN pixels [0,1]. Prints warning if > 0."""
    if arr.size == 0:
        return 0.0
    n_nan = np.isnan(arr).sum()
    frac  = n_nan / arr.size
    if frac > 0:
        print(f"[NaN WARNING] {name}: {frac*100:.2f}% NaN pixels ({n_nan}/{arr.size})")
    return frac


def process_images_for_scale(source_name: str,
                             raw_path: Path,
                             t_path: Path,
                             sub_path: Optional[Path],
                             z: float,
                             fwhm_kpc: float,
                             target_nbeams: float,
                             downsample_size=(1, 128, 128),
                             cheat_rt: bool = False,
                             offsets_px: Optional[Dict[str, Tuple[float, float]]] = None,
                             fov_arcsec: Optional[float] = None):
    """Process RAW + T_X (+ T_XSUB) for one source at one scale. Returns dict of arrays."""
    if offsets_px is None:
        offsets_px = {}
    if not cheat_rt:
        if not np.isfinite(z) or z <= 0:
            raise ValueError(f"Invalid redshift z={z} for {source_name}")

    I_raw,  H_raw,  W_raw = read_fits_array_header_wcs(raw_path)
    T_nat,  H_tgt,  W_tgt = read_fits_array_header_wcs(t_path)

    if sub_path is not None and sub_path.exists():
        SUB_nat, H_sub, W_sub = read_fits_array_header_wcs(sub_path)
        has_sub = True
    else:
        SUB_nat, H_sub, W_sub = None, None, None
        has_sub = False

    T_common  = T_nat
    H_common  = H_tgt
    W_common  = W_tgt
    I_on_common   = reproject_like(I_raw,   H_raw, H_common)
    SUB_on_common = reproject_like(SUB_nat, H_sub, H_common) if has_sub else None

    if cheat_rt:
        ker = kernel_from_beams(H_raw, H_tgt)
    else:
        ker = circular_kernel_from_z(z, H_raw, fwhm_kpc=fwhm_kpc)
    I_smt        = convolve_fft(I_raw, ker, boundary='fill', fill_value=np.nan,
                                nan_treatment='interpolate', normalize_kernel=True,
                                psf_pad=True, fft_pad=True, allow_huge=True)
    scale_factor = beam_solid_angle_sr(H_tgt) / beam_solid_angle_sr(H_raw)
    RT_rawgrid   = I_smt * scale_factor
    RT_on_common = reproject_like(RT_rawgrid, H_raw, H_common)

    header_sky = header_cluster_coord(H_raw) or header_cluster_coord(H_tgt)
    if header_sky is None:
        H0_i, W0_i = I_raw.shape
        yc_i, xc_i = H0_i // 2, W0_i // 2
        center_note = 'No header sky coords; used image centres.'
    else:
        x_i, y_i = W_raw.world_to_pixel(header_sky)
        yc_i, xc_i = float(y_i), float(x_i)
        center_note = f"Centered on RA={header_sky.ra.deg:.6f}, Dec={header_sky.dec.deg:.6f} deg."
    dy_px, dx_px = offsets_px.get(source_name, (0.0, 0.0))
    if dy_px or dx_px:
        yc_i += dy_px; xc_i += dx_px
        center_note += f" | manual offset (dy,dx)=({dy_px:.1f},{dx_px:.1f}) px"

    beam_fwhm_as = fwhm_major_as(H_common)
    max_side_as_T   = _nan_free_centred_square_side_as(T_common,     H_common)
    max_side_as_I   = _nan_free_centred_square_side_as(I_on_common,  H_common)
    max_side_as_RT  = _nan_free_centred_square_side_as(RT_on_common, H_common)
    max_side_as     = min(max_side_as_T, max_side_as_I, max_side_as_RT)
    if has_sub and SUB_on_common is not None:
        max_side_as = min(max_side_as,
                          _nan_free_centred_square_side_as(SUB_on_common, H_common))
    if fov_arcsec is not None:
        desired_side_as = float(fov_arcsec)
    else:
        desired_side_as = target_nbeams * beam_fwhm_as
    actual_side_as = min(desired_side_as, max_side_as)
    actual_nbeams  = actual_side_as / beam_fwhm_as
    if actual_side_as < desired_side_as:
        if fov_arcsec is not None:
            print(f"[crop] WARNING: {source_name}/{fwhm_kpc:.1f}kpc: "
                  f"requested FOV={fov_arcsec:.1f}\" but NaN-free limit={max_side_as:.1f}\" "
                  f"(T={max_side_as_T:.1f}\" I={max_side_as_I:.1f}\" RT={max_side_as_RT:.1f}\")")
        else:
            print(f"[crop] WARNING: {source_name}/{fwhm_kpc:.1f}kpc: "
                  f"target={target_nbeams:.2f} beams, actual={actual_nbeams:.2f} beams "
                  f"({actual_side_as:.1f}\" | NaN-free: T={max_side_as_T:.1f}\" "
                  f"I={max_side_as_I:.1f}\" RT={max_side_as_RT:.1f}\")")

    header_sky_common = header_cluster_coord(H_common)
    if header_sky_common is None:
        yc_common = T_common.shape[0] // 2
        xc_common = T_common.shape[1] // 2
    else:
        x_c, y_c  = W_common.world_to_pixel(header_sky_common)
        yc_common, xc_common = float(y_c), float(x_c)
    dy_px_common, dx_px_common = offsets_px.get(source_name, (0.0, 0.0))
    asx_raw, asy_raw       = arcsec_per_pix(H_raw)
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

    Ho, Wo = _canon_size(downsample_size)[-2:]

    def _maybe_downsample(a, H, W):
        t = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y = torch.nn.functional.interpolate(t, size=(H, W), mode='bilinear',
                                                align_corners=False)
        return y.squeeze(0).squeeze(0).cpu().numpy()

    I_fmt_np   = _maybe_downsample(I_crop,   Ho, Wo)
    RT_fmt_np  = _maybe_downsample(RT_crop,  Ho, Wo)
    T_fmt_np   = _maybe_downsample(T_crop,   Ho, Wo)
    SUB_fmt_np = _maybe_downsample(SUB_crop, Ho, Wo) if has_sub else None

    H0_common, W0_common = T_common.shape
    W_i_fmt, H_i_fmt = wcs_after_center_crop_and_resize(
        H_common, H0_common, W0_common, nyc, nxc, Ho, Wo,
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
