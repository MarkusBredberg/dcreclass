"""FITS/WCS utility functions for radio astronomy image processing."""

import warnings
import numpy as np
from pathlib import Path
from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

warnings.filterwarnings("ignore", category=FITSFixedWarning)

ARCSEC_PER_RAD = 206264.80624709636  # 180.0 * 3600.0 / pi


def _cd_matrix_rad(h):
    """Extract 2x2 pixel->world Jacobian in radians/pixel from FITS header."""
    if 'CD1_1' in h:
        M = np.array([[h['CD1_1'], h.get('CD1_2', 0.0)],
                      [h.get('CD2_1', 0.0), h['CD2_2']]], float)
    else:
        pc11 = h.get('PC1_1', 1.0); pc12 = h.get('PC1_2', 0.0)
        pc21 = h.get('PC2_1', 0.0); pc22 = h.get('PC2_2', 1.0)
        cd1  = h.get('CDELT1', 1.0); cd2  = h.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12], [pc21, pc22]], float) @ np.diag([cd1, cd2])
    return M * (np.pi / 180.0)


def arcsec_per_pix(h):
    """Effective pixel scales (|dx|, |dy|) in arcseconds."""
    J = _cd_matrix_rad(h)
    dx = np.hypot(J[0, 0], J[1, 0])
    dy = np.hypot(J[0, 1], J[1, 1])
    return dx * ARCSEC_PER_RAD, dy * ARCSEC_PER_RAD


def fwhm_major_as(h):
    """Major axis FWHM from FITS header beam in arcseconds."""
    return max(float(h['BMAJ']), float(h['BMIN'])) * 3600.0


def _fwhm_as_to_sigma_rad(fwhm_as: float) -> float:
    """Convert FWHM [arcsec] to Gaussian sigma [radians]."""
    return (float(fwhm_as) / (2.0 * np.sqrt(2.0 * np.log(2.0)))) * (np.pi / (180.0 * 3600.0))


def beam_cov_world(h):
    """2x2 beam covariance in world radians from FITS header."""
    bmaj_as = abs(float(h['BMAJ'])) * 3600.0
    bmin_as = abs(float(h['BMIN'])) * 3600.0
    pa_deg  = float(h.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th = np.deg2rad(pa_deg)
    R  = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], float)
    S  = np.diag([sx * sx, sy * sy])
    return R @ S @ R.T


def beam_solid_angle_sr(h):
    """Gaussian beam solid angle in steradians from BMAJ/BMIN [deg]."""
    bmaj = abs(float(h['BMAJ'])) * np.pi / 180.0
    bmin = abs(float(h['BMIN'])) * np.pi / 180.0
    return (np.pi / (4.0 * np.log(2.0))) * bmaj * bmin


def kernel_from_beams(raw_hdr, tgt_hdr):
    """Elliptical Gaussian kernel mapping RAW beam -> TARGET beam."""
    C_raw = beam_cov_world(raw_hdr)
    C_tgt = beam_cov_world(tgt_hdr)
    C_ker_world = C_tgt - C_raw
    w, V = np.linalg.eigh(C_ker_world)
    w = np.clip(w, 0.0, None)
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


def read_fits_array_header_wcs(fpath: Path):
    """Read FITS file; return (2D float32 array, header, 2D WCS)."""
    with fits.open(fpath, memmap=False) as hdul:
        header   = hdul[0].header
        wcs_full = WCS(header)
        wcs2d    = wcs_full.celestial if hasattr(wcs_full, 'celestial') else WCS(header, naxis=2)
        arr = None
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None:
                arr = np.asarray(hdu.data)
                break
    if arr is None:
        raise RuntimeError(f"No data-containing HDU in {fpath}")
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=0)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D image; got {arr.shape}")
    return arr.astype(np.float32), header, wcs2d


def reproject_like(arr: np.ndarray, src_hdr, dst_hdr) -> np.ndarray:
    """Reproject array from source WCS to destination WCS."""
    try:
        from reproject import reproject_interp
        w_src = (WCS(src_hdr).celestial if hasattr(WCS(src_hdr), 'celestial')
                 else WCS(src_hdr, naxis=2))
        w_dst = (WCS(dst_hdr).celestial if hasattr(WCS(dst_hdr), 'celestial')
                 else WCS(dst_hdr, naxis=2))
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        out, _ = reproject_interp((arr, w_src), w_dst, shape_out=(ny_out, nx_out),
                                  order='bilinear')
        return out.astype(np.float32)
    except Exception:
        from scipy.ndimage import zoom as _zoom
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        ny_in, nx_in = arr.shape
        zy = ny_out / max(ny_in, 1); zx = nx_out / max(nx_in, 1)
        y  = _zoom(arr, zoom=(zy, zx), order=1)
        y  = y[:ny_out, :nx_out]
        if y.shape != (ny_out, nx_out):
            pad_y = ny_out - y.shape[0]; pad_x = nx_out - y.shape[1]
            y = np.pad(y, ((0, max(0, pad_y)), (0, max(0, pad_x))),
                       mode='edge')[:ny_out, :nx_out]
        return y.astype(np.float32)


def header_cluster_coord(header) -> Optional[SkyCoord]:
    """Extract cluster sky coordinates from FITS header."""
    if header.get('OBJCTRA') and header.get('OBJCTDEC'):
        return SkyCoord(header['OBJCTRA'], header['OBJCTDEC'],
                        unit=(u.hourangle, u.deg))
    if header.get('RA_TARG') and header.get('DEC_TARG'):
        return SkyCoord(header['RA_TARG'] * u.deg, header['DEC_TARG'] * u.deg)
    if 'CRVAL1' in header and 'CRVAL2' in header:
        return SkyCoord(header['CRVAL1'] * u.deg, header['CRVAL2'] * u.deg)
    return None


def wcs_after_center_crop_and_resize(header, H0, W0, Hc, Wc, Ho, Wo, y0, x0):
    """Update WCS header after centre-crop and resize."""
    y1, y2 = max(0, y0 - Hc // 2), min(H0, y0 + Hc // 2)
    x1, x2 = max(0, x0 - Wc // 2), min(W0, x0 + Wc // 2)
    sx = (x2 - x1) / float(Wo)
    sy = (y2 - y1) / float(Ho)
    new = header.copy()
    if 'CRPIX1' in new and 'CRPIX2' in new:
        new['CRPIX1'] = (new['CRPIX1'] - x1) / sx
        new['CRPIX2'] = (new['CRPIX2'] - y1) / sy
    if all(k in new for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')):
        new['CD1_1'] *= sx; new['CD1_2'] *= sy
        new['CD2_1'] *= sx; new['CD2_2'] *= sy
    else:
        if 'CDELT1' in new: new['CDELT1'] *= sx
        if 'CDELT2' in new: new['CDELT2'] *= sy
    new['NAXIS1'] = Wo; new['NAXIS2'] = Ho
    wcs_new = (WCS(new).celestial if hasattr(WCS(new), 'celestial') else WCS(new, naxis=2))
    return wcs_new, new


def robust_vmin_vmax(arr: np.ndarray, lo: int = 30, hi: int = 99):
    """Robust display min/max from percentiles."""
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0, 1.0
    vals = arr[finite]
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)
