#!/usr/bin/env python3

from utils.data_loader import load_galaxies
from utils.data_loader import apply_formatting
from utils.calc_tools import check_tensor
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as _imgshift
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.convolution import convolve_fft, Gaussian2DKernel
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob, re
try:
    # best case
    from skimage.registration import phase_cross_correlation as _pcc
    HAVE_PCC = True
except Exception:
    HAVE_PCC = False
try:
    from reproject import reproject_interp
    HAVE_REPROJECT = True
except Exception:
    HAVE_REPROJECT = False
print(f"HAVE_REPROJECT={HAVE_REPROJECT}")

    
print("Loading scatter_galaxies/plot_gauss_blur_grid_editing.py...")

REQUIRE_REDSHIFT_ONLY = False # ignore sources without z
ROOT = "/users/mbredber/scratch/data/PSZ2"
APPLY_UV_TAPER = os.getenv("RT_USE_UV_TAPER","0") == "1"  # default: off
UV_TAPER_FRAC  = float(os.getenv("RT_UV_TAPER_FRAC","0.0"))  # e.g. 0.2 for mild
AUTO_FUDGE = os.getenv("RT_AUTO_FUDGE", "1") == "1"   # default on
FUDGE_GLOBAL   = float(os.getenv("RT_FUDGE_SCALE","1.00"))
ARCSEC = np.deg2rad(1/3600)
crop_size = (1, 512, 512)  # (T, C, H, W) for 3 channels, 512x512 images
CLUSTER_METADATA_CSV = "/users/mbredber/scratch/data/PSZ2/cluster_metadata.csv" # Metadata CSV with redshifts

_SLUG_RX = re.compile(
    r'(PSZ2G)\s*(\d{3}\.\d{2})\s*([+\-])\s*(\d{2})(?:\.(\d{1,2}))?$',
    re.IGNORECASE
)

def _norm_slug(s: str) -> str:
    """Normalize spaces/underscores and collapse 'PSZ2 G' → 'PSZ2G'."""
    s = str(s).strip()
    s = s.replace('_', '').replace(' ', '')
    s = s.replace('PSZ2G', 'PSZ2G').replace('PSZ2G', 'PSZ2G')  # idempotent
    return s.upper()

def _slug_variants(s: str):
    """
    Return normalized slug variants:
      exact (with decimals if present) and a 'truncated-lat' version.
    """
    s = _norm_slug(s)
    m = _SLUG_RX.match(s)
    if not m:
        return {s}
    pfx, lon, sign, lat2, latdec = m.groups()
    exact = f"{pfx}{lon}{sign}{lat2}" + (f".{latdec}" if latdec else "")
    short = f"{pfx}{lon}{sign}{lat2}"
    return {exact, short}

def _load_cluster_meta(csv_path):
    import csv, os, math
    d = {}
    if not os.path.exists(csv_path):
        print(f"[meta] CSV not found: {csv_path}")
        return d

    def _try_float(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    # Try headered first
    with open(csv_path, newline="") as f:
        R = csv.DictReader(f)
        if R.fieldnames and {'slug','z'} <= {h.strip().lower() for h in R.fieldnames}:
            n = 0
            for row in R:
                slug = row.get('slug', '')
                z = _try_float(row.get('z', ''))
                if not slug or z is None or not (0.0 < z < 5.0):
                    continue
                for v in _slug_variants(slug):
                    d[v] = z
                n += 1
            print(f"[meta] loaded {len(d)} redshifts from headered CSV ({n} rows)")
            return d

    # Fallback: headerless (first col = slug, first numeric 0<z<5 in the rest)
    with open(csv_path, newline="") as f:
        R = csv.reader(f)
        n = 0
        for row in R:
            if not row or str(row[0]).lstrip().startswith('#'):
                continue
            slug = row[0]
            z = None
            for cell in row[1:]:
                z = _try_float(cell)
                if z is not None and 0.0 < z < 5.0:
                    break
            if not slug or z is None:
                continue
            for v in _slug_variants(slug):
                d[v] = z
            n += 1
        print(f"[meta] loaded {len(d)} redshifts from headerless CSV ({n} rows)")
    return d

CLUSTER_META = _load_cluster_meta(CLUSTER_METADATA_CSV)

def _z_from_meta(name: str):
    """
    Robust CSV lookup: normalize the query name and try all slug variants.
    Also tries a prefix match as a last resort (for even-shorter names).
    """
    name = _norm_slug(_name_base_from_fn(name))
    for v in _slug_variants(name):
        if v in CLUSTER_META:
            return CLUSTER_META[v]
    # last resort: prefix match (e.g., 'PSZ2G192.18+56' vs 'PSZ2G192.18+56.23')
    for k, z in CLUSTER_META.items():
        if k.startswith(name):
            return z
    return None

def radial_tukey(ny, nx, alpha=0.35):
    y = np.linspace(-1, 1, ny); x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X*X + Y*Y)
    w = np.ones_like(r)
    r0 = 1.0 - alpha
    m  = (r > r0) & (r < 1.0)
    w[m]  = 0.5 * (1.0 + np.cos(np.pi*(r[m]-r0)/max(alpha,1e-9)))
    w[r>=1.0] = 0.0
    return w

def radial_hann(ny, nx):
    y = np.linspace(-1, 1, ny); x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, y); r = np.sqrt(X*X + Y*Y)
    w = np.zeros_like(r); m = (r <= 1.0)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * r[m]))
    return w


def reproject_like(arr, src_hdr, dst_hdr):
    """
    Reproject a 2-D image from src_hdr WCS to dst_hdr WCS.

    If the 'reproject' package is available and both headers contain a valid
    2-D celestial WCS, use bilinear interpolation. Otherwise fall back to a
    center-alignment translation (keeps shape, best-effort alignment).
    """
    if arr is None or src_hdr is None or dst_hdr is None:
        return None

    # FAST PATH: identical grid → return as-is
    if _same_pixel_grid(src_hdr, dst_hdr):
        return np.asarray(arr, float)

    # Otherwise do the real thing (but with a cached WCS)
    try:
        w_src = _wcs_celestial_cached(src_hdr)
        w_dst = _wcs_celestial_cached(dst_hdr)
    except Exception:
        w_src = w_dst = None

    if HAVE_REPROJECT and (w_src is not None) and (w_dst is not None):
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        reproj, _ = reproject_interp((arr, w_src), w_dst,
                                     shape_out=(ny_out, nx_out),
                                     order='bilinear')
        return reproj.astype(float)

    # Fallback: simple center alignment with subpixel shift
    if (w_src is None) or (w_dst is None):
        return arr.astype(float)

    ny, nx = arr.shape
    (ra, dec) = w_src.wcs_pix2world([[nx/2.0, ny/2.0]], 0)[0]
    (x_dst, y_dst) = w_dst.wcs_world2pix([[ra, dec]], 0)[0]
    dx = (float(dst_hdr['NAXIS1'])/2.0) - x_dst
    dy = (float(dst_hdr['NAXIS2'])/2.0) - y_dst
    return _imgshift(arr, shift=(dy, dx), order=1, mode="nearest").astype(float)


def kpc_to_arcsec(z, L_kpc):
    """Physical size → angle (arcsec) using θ ≈ L / D_A."""
    D_A = COSMO.angular_diameter_distance(float(z)).to(u.kpc)
    theta = (L_kpc * u.kpc / D_A) * u.rad  # mark the dimensionless ratio as radians
    return theta.to(u.arcsec).value

def synth_taper_header_from_kpc(raw_hdr, z, L_kpc,
                                mode="keep_ratio"):  # or "circular"
    """
    Build a *synthetic* target header with only beam keywords filled in,
    derived from redshift and the desired physical FWHM.

    mode="keep_ratio": keep RAW beam PA and axis ratio, scale to the requested
                       geometric-mean FWHM (best match to a uv-taper product).
    mode="circular":   force a circular target beam of FWHM = X kpc.
    """
    # desired FWHM on sky
    phi_as = float(kpc_to_arcsec(z, L_kpc))

    # RAW beam (arcsec)
    bmaj_r = abs(float(raw_hdr['BMAJ'])) * 3600.0
    bmin_r = abs(float(raw_hdr['BMIN'])) * 3600.0
    pa_r   = float(raw_hdr.get('BPA', 0.0))

    # never try to sharpen: target geom. mean ≥ RAW geom. mean
    phi_as = max(phi_as, np.sqrt(bmaj_r * bmin_r))

    if mode == "circular":
        bmaj_t = bmin_t = phi_as
        pa_t   = 0.0
    else:  # keep_ratio
        r = bmaj_r / bmin_r  # RAW axis ratio
        bmin_t = phi_as / np.sqrt(r)
        bmaj_t = phi_as * np.sqrt(r)
        pa_t   = pa_r

    # synth header: only beam terms matter for our kernel & Jy/beam conversion
    thdr = fits.Header()
    thdr['BMAJ'] = bmaj_t / 3600.0   # deg
    thdr['BMIN'] = bmin_t / 3600.0   # deg
    thdr['BPA']  = pa_t
    # copy the RAW WCS so any "reproject_like(..., raw_hdr, thdr)" is a no-op
    for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
              'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
              'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
        if k in raw_hdr: thdr[k] = raw_hdr[k]
    return thdr

def _sigma_from_fwhm_arcsec(theta_fwhm_arcsec: float) -> float:
    """Convert angular FWHM [arcsec] → Gaussian sigma [radians]."""
    return (theta_fwhm_arcsec * ARCSEC) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def _make_uv_gaussian_weight(nx, ny, dx, dy, theta_fwhm_arcsec):
    """
    Return W(u,v) = exp[-2*pi^2*sigma_theta^2 * (u^2+v^2)]
    where u,v are in cycles/radian (= wavelengths).
    """
    sigma_th = _sigma_from_fwhm_arcsec(theta_fwhm_arcsec)
    
    # ---------- uv-space weight W (same shape as F(I)) ----------
    u = np.fft.fftshift(np.fft.fftfreq(W, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(H, d=dy))
    U, V = np.meshgrid(u, v)

    # world-space covariance of the RAW→TARGET kernel
    C_world = _kernel_world_covariance_from_headers(raw_hdr, H_targ,
                                                    fudge_scale=fudge_scale_demo)

    # analytic W(u,v); DC = 1 at (u,v)=(0,0)
    Wuv = _make_W_from_sigma_theta(U, V, sigma_theta_rad=None, anisotropic=C_world)

    return np.exp(-2.0 * (np.pi**2) * (sigma_th**2) * (U*U + V*V))

def apply_uv_gaussian_taper(img, hdr_img, theta_fwhm_arcsec, pad_factor=1):
    """
    Multiply the image's Fourier plane by a Gaussian uv-taper matching the
    requested angular FWHM (from the kpc target at this z), then inverse FFT
    back to image space.  Units are preserved (Jy/beam_*whatever you passed in*).

    We intentionally *do not* do any apodization here to avoid amplitude
    biases; the taper is already smooth.  Optional zero-padding reduces
    wrap-around (pad_factor=2 is safer, 1 is fastest).
    """
    A = np.asarray(img, float)
    A = np.where(np.isfinite(A), A, 0.0)

    ny0, nx0 = A.shape
    if pad_factor > 1:
        ny, nx = int(ny0*pad_factor), int(nx0*pad_factor)
        py = (ny - ny0)//2; px = (nx - nx0)//2
        A_pad = np.pad(A, ((py, ny - ny0 - py), (px, nx - nx0 - px)),
                       mode='constant', constant_values=0.0)
        crop = (slice(py, py+ny0), slice(px, px+nx0))
    else:
        ny, nx = ny0, nx0
        A_pad = A
        crop = (slice(0, ny0), slice(0, nx0))

    # pixel sizes [radian]
    dx = abs(_cdelt_deg(hdr_img, 1)) * np.pi/180.0
    dy = abs(_cdelt_deg(hdr_img, 2)) * np.pi/180.0

    # forward FFT with continuous normalisation
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A_pad))) * (dx * dy)

    # uv Gaussian weight
    W = _make_uv_gaussian_weight(nx, ny, dx, dy, theta_fwhm_arcsec)

    # apply taper and invert
    Fw = F * W
    A_tap = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fw))).real / (dx * dy)

    return A_tap[crop]


def get_z(name, hdr_primary):
    """Return a usable redshift for this source (CSV → header → siblings)."""
    import re, glob, os
    from astropy.io import fits

    def _coerce_float(val):
        try:
            z = float(val)
            return z if np.isfinite(z) else None
        except Exception:
            pass
        if isinstance(val, (bytes, bytearray)):
            try: val = val.decode('utf-8', 'ignore')
            except Exception: return None
        if isinstance(val, str):
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if m:
                try: return float(m.group(0))
                except Exception: return None
        return None

    CAND_KEYS = (
        'REDSHIFT','Z','Z_CL','ZCL','ZSPEC','Z_SPEC','ZPHOT','Z_PHOT',
        'Z_BEST','ZBEST','REDSHIFT_CL','REDSHIFT_SPEC','REDSHIFT_PHOT',
        'SIM_Z','Z_MEAN','Z_AVG','Z_LAMBDA','Z_LIT','ZPLANCK','Z_PSZ2'
    )
    BAD_EXACT = {'BZERO','BSCALE','TIMEZERO','MJDZERO','TZERO','ZERO','ZEROPOINT','DZ','DZ_AVG','Z_ERR','ZERR'}

    def _parse_from_keys(hdr):
        for k in CAND_KEYS:
            if k in hdr:
                z = _coerce_float(hdr[k])
                if z is not None and 0.0 < z < 5.0:
                    return z
        return None

    def _parse_by_sweep(hdr):
        for k, v in hdr.items():
            ku = str(k).upper().replace('HIERARCH ', '')
            if ku in BAD_EXACT: 
                continue
            if ('REDSHIFT' in ku) or re.search(r'(^|_)Z($|_|[A-Z0-9])', ku):
                if 'ZERO' in ku or ku.startswith('DZ'):
                    continue
                z = _coerce_float(v)
                if z is not None and 0.0 < z < 5.0:
                    return z
        return None

    # 0) CSV first
    name_base = _name_base_from_fn(name) if ("/" in str(name) or name.endswith(".fits")) else name
    z_meta = _z_from_meta(name_base)
    if z_meta is not None:
        print(f"[z] {name_base}: using z={z_meta:.4f} from CSV")
        return float(z_meta)

    # 1) header we were handed
    z = _parse_from_keys(hdr_primary) or _parse_by_sweep(hdr_primary)
    if z is not None:
        return z

    # 2) sibling headers on disk (prefer CHANDRA if present)
    base_dir = _find_base_dir(name_base)
    if base_dir:
        pats = (sorted(glob.glob(f"{base_dir}/*CHANDRA*.fits")) +
                sorted(glob.glob(f"{base_dir}/*.fits")))
        for p in pats:
            try:
                hdr = fits.getheader(p)
            except Exception:
                continue
            z = _parse_from_keys(hdr) or _parse_by_sweep(hdr)
            if z is not None:
                print(f"[z] {name_base}: using z={z:.4f} from {os.path.basename(p)}")
                return z

    raise KeyError(f"No redshift for {name_base}; not in CSV and no valid header keys.")

# --- add near reproject_like() ---
_WCS_CACHE = {}
def _wcs_celestial_cached(hdr):
    key = id(hdr)  # stable within this run
    w = _WCS_CACHE.get(key)
    if w is None:
        w = WCS(hdr).celestial
        _WCS_CACHE[key] = w
    return w

def _same_pixel_grid(h1, h2, atol=1e-12):
    # compare essential geometry; beams may differ
    for k in ('NAXIS1','NAXIS2','CTYPE1','CTYPE2'):
        if (h1.get(k) != h2.get(k)):
            return False
    # CD/PC + CDELT
    def _cd(h):
        if 'CD1_1' in h:
            return (float(h['CD1_1']), float(h.get('CD1_2',0.0)),
                    float(h.get('CD2_1',0.0)), float(h['CD2_2']))
        # PC*CDELT fallback
        pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
        pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
        c1=h.get('CDELT1',-1.0); c2=h.get('CDELT2', 1.0)
        m = np.array([[pc11,pc12],[pc21,pc22]],float) @ np.diag([c1,c2])
        return (float(m[0,0]), float(m[0,1]), float(m[1,0]), float(m[1,1]))
    return np.allclose(_cd(h1), _cd(h2), atol=atol) and \
           np.allclose([h1.get('CRPIX1'),h1.get('CRPIX2')],
                       [h2.get('CRPIX1'),h2.get('CRPIX2')], atol=1e-9) and \
           np.allclose([h1.get('CRVAL1'),h1.get('CRVAL2')],
                       [h2.get('CRVAL1'),h2.get('CRVAL2')], atol=1e-9)

def _stretch_from_p(arr, p_lo, p_hi, clip=False):
    if arr is None:
        return None
    y = (arr - p_lo) / (p_hi - p_lo + 1e-6)
    return np.clip(y, 0, 1) if clip else y

def _summ(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    return dict(min=float(np.min(a)), max=float(np.max(a)),
                mean=float(np.mean(a)), std=float(np.std(a)))

def _nanpct(arr, lo=60, hi=95):
    """Percentiles that ignore NaNs."""
    return tuple(np.nanpercentile(arr, [lo, hi]))

def _nanrms(a, b):
    """RMS difference between two arrays, ignoring NaNs."""
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d*d)))

def xcorr_shift(a, b):
    """Integer-pixel shift (dy, dx) that best aligns a to b via FFT x-corr."""
    A = np.fft.fftn(np.nan_to_num(a))
    B = np.fft.fftn(np.nan_to_num(b))
    cc = np.fft.ifftn(A * np.conj(B))
    ij = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    dy = ij[0] if ij[0] <= a.shape[0]//2 else ij[0] - a.shape[0]
    dx = ij[1] if ij[1] <= a.shape[1]//2 else ij[1] - a.shape[1]
    return float(dy), float(dx)

def _find_raw_path(base_dir, base_name):
    pats = sorted(glob.glob(f"{base_dir}/{base_name}*.fits"))
    if not pats:
        return None
    # Prefer files that don't look like derivatives
    BAD_SUBSTR = ("T25KPC","T50KPC","T100KPC","RMS","NOISE","MASK","BEAM","PSF","PB","WEIGHT","WT","UV")
    good = [p for p in pats if not any(s.lower() in p.lower() for s in BAD_SUBSTR)]
    return (good[0] if good else pats[0])

def _find_base_dir(name):
    variants = [
        name,
        name.replace("PSZ2G", "PSZ2 G"),
        name.replace("PSZ2G", "PSZ2_G"),
    ]
    for v in variants:
        d = _first(f"{ROOT}/fits/{v}*")
        if d: return d
    return None

def _fits_path_triplet(base_dir, real_base):
    raw_path  = f"{base_dir}/{real_base}.fits"
    t25_path  = _first(f"{base_dir}/{real_base}T25kpc*.fits")
    t50_path  = _first(f"{base_dir}/{real_base}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{real_base}T100kpc*.fits")
    return raw_path, t25_path, t50_path, t100_path

def _has_redshift_by_name(name: str) -> bool:
    # CSV wins — this avoids the expensive FITS scan for most sources
    if _z_from_meta(_name_base_from_fn(name)) is not None:
        return True

    base_dir = _find_base_dir(name)
    if base_dir is None:
        return False
    base = os.path.basename(base_dir)
    cand = []
    cand.extend(sorted(glob.glob(f"{base_dir}/*CHANDRA*.fits")))
    raw_guess = _find_raw_path(base_dir, base)
    if raw_guess: cand.append(raw_guess)
    cand.extend(sorted(glob.glob(f"{base_dir}/{base}T25kpc*.fits")))
    cand.extend(sorted(glob.glob(f"{base_dir}/{base}T50kpc*.fits")))
    cand.extend(sorted(glob.glob(f"{base_dir}/{base}T100kpc*.fits")))
    if not cand:
        cand = sorted(glob.glob(f"{base_dir}/*.fits"))
    for p in cand:
        try:
            hdr = fits.getheader(p)
            z = get_z(name, hdr)   # will also try CSV again
            if 0.0 < float(z) < 5.0:
                return True
        except Exception:
            continue
    return False
    
    
def _eq_fwhm_as(h):  # geometric-mean FWHM in arcsec
    return np.sqrt((abs(float(h['BMAJ']))*3600.0) * (abs(float(h['BMIN']))*3600.0))

def synth_taper_header_from_ref(raw_hdr, ref_hdr, kpc_target, kpc_ref=50.0, mode="keep_ratio"):
    """Make a target header for kpc_target by scaling a known 'ref' taper header (e.g., T50)."""
    scale = float(kpc_target) / float(kpc_ref)

    # Read reference beam (arcsec)
    bmaj_ref_as = abs(float(ref_hdr['BMAJ'])) * 3600.0
    bmin_ref_as = abs(float(ref_hdr['BMIN'])) * 3600.0
    pa_ref      = float(ref_hdr.get('BPA', raw_hdr.get('BPA', 0.0)))

    # Target geom-mean in arcsec, scaled from the ref beam
    phi_ref_as = np.sqrt(bmaj_ref_as * bmin_ref_as)
    phi_tgt_as = scale * phi_ref_as

    if mode == "circular":
        bmaj_t = bmin_t = phi_tgt_as
        pa_t   = 0.0
    else:  # keep_ratio of the REF beam
        r = bmaj_ref_as / bmin_ref_as
        bmin_t = phi_tgt_as / np.sqrt(r)
        bmaj_t = phi_tgt_as * np.sqrt(r)
        pa_t   = pa_ref

    # Never sharpen beyond RAW geom-mean
    phi_raw_as = np.sqrt((abs(float(raw_hdr['BMAJ']))*3600.0) * (abs(float(raw_hdr['BMIN']))*3600.0))
    if np.sqrt(bmaj_t * bmin_t) < phi_raw_as:
        fac = phi_raw_as / np.sqrt(bmaj_t * bmin_t)
        bmaj_t *= fac; bmin_t *= fac
        print(f"[warn] Requested {kpc_target}kpc < RAW resolution → clamped to RAW beam.")

    # Build header on RAW WCS/grid
    thdr = fits.Header()
    thdr['BMAJ'] = bmaj_t / 3600.0
    thdr['BMIN'] = bmin_t / 3600.0
    thdr['BPA']  = pa_t
    for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
              'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
              'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
        if k in raw_hdr: thdr[k] = raw_hdr[k]
    return thdr


def _name_base_from_fn(fn):
    # robust base name; strips any trailing "T25kpc" etc. if present
    stem = Path(str(fn)).stem
    # split only once to avoid clobbering names that legitimately contain 'T'
    return stem.split('T', 1)[0]

def _has_redshift_for_index(idx, filenames) -> bool:
    return _has_redshift_by_name(_name_base_from_fn(filenames[idx]))

def _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(128,128)):
    """
    Load RAW + taper images, reproject each taper to the RAW grid, convolve RAW
    → each target on the RAW grid (and rescale to Jy/beam_target), then downsample
    EVERYTHING from the same grid to the display size 'out_hw'.

    Returns:
      (raw_cut, t25_cut, t50_cut, t100_cut,
       rt25_cut, rt50_cut, rt100_cut,
       raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff_arcsec)
    """
    base_dir = _find_base_dir(name)
    if base_dir is None:
        raise FileNotFoundError(f"Could not locate directory for {name} under {ROOT}/fits")
    base = os.path.basename(base_dir)

    raw_path = _find_raw_path(base_dir, base)
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir} for base '{base}'")

    raw_hdr = fits.getheader(raw_path)
    raw_arr = np.squeeze(fits.getdata(raw_path)).astype(float)
    pix_native = _pixscale_arcsec(raw_hdr)

    real_base = os.path.splitext(os.path.basename(raw_path))[0]
    raw_path, t25_path, t50_path, t100_path = _fits_path_triplet(base_dir, real_base)

    # try to read TXkpc; if missing, synthesize from redshift
    def _hdr_or_synth(tpath, Xkpc, ref_hdr=None, kpc_ref=None):
        """
        Prefer on-disk header; otherwise build synthetic beam.
        Try z→kpc; if z is missing but a reference taper header is available
        (e.g. T50), fall back to scaling that header to the requested kpc.
        """
        if tpath and os.path.exists(tpath):
            return fits.getheader(tpath)
        try:
            z_local = get_z(name, raw_hdr)
            return synth_taper_header_from_kpc(raw_hdr, z_local, Xkpc, mode="keep_ratio")
        except Exception as e:
            if ref_hdr is not None and kpc_ref is not None:
                print(f"[z-miss] {name}: {e}. Falling back to REF {int(kpc_ref)}kpc header scaling.")
                return synth_taper_header_from_ref(raw_hdr, ref_hdr, Xkpc, kpc_ref, mode="keep_ratio")
            print(f"[z-miss] {name}: {e}. Proceeding with RAW→target-beam only on RAW grid.")
            # Final fallback: copy RAW beam so code does not crash (no broadening).
            th = fits.Header()
            for k in ('BMAJ','BMIN','BPA','CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
                      'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
                      'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
                if k in raw_hdr: th[k] = raw_hdr[k]
            return th


    # Build in an order that allows ref-scaling fallbacks (prefer T50 as ref)
    t50_hdr  = _hdr_or_synth(t50_path,  50)
    t25_hdr  = _hdr_or_synth(t25_path,  25, ref_hdr=t50_hdr,  kpc_ref=50.0)
    t100_hdr = _hdr_or_synth(t100_path, 100, ref_hdr=t50_hdr, kpc_ref=50.0)

    # data arrays only if files exist
    t25_arr  = np.squeeze(fits.getdata(t25_path)).astype(float)   if t25_path  else None
    t50_arr  = np.squeeze(fits.getdata(t50_path)).astype(float)   if t50_path  else None
    t100_arr = np.squeeze(fits.getdata(t100_path)).astype(float)  if t100_path else None


    ch, Hc_raw, Wc_raw = crop_ch, crop_size[-2], crop_size[-1]
    outH, outW = out_hw

    # 1) Keep the *tapered* images on their own grids
    t25_on_t  = t25_arr
    t50_on_t  = t50_arr
    t100_on_t = t100_arr

    # 2) Convolve RAW on its native grid to the target beam
    # Anti-alias only when we actually shrink the image
    ds = int(round(crop_size[-1] / out_hw[1]))  # e.g. 512→128 ⇒ ds≈4
    if ds > 1:
        raw_arr_prefiltered = gaussian_filter(raw_arr, sigma=0.5*ds, mode='nearest')
    else:
        raw_arr_prefiltered = raw_arr
    
    # Try redshift ➜ kpc→arcsec. If missing, skip uv-taper (still make rtX via image-domain PSF match).
    theta25_as = theta50_as = theta100_as = None
    try:
        z = get_z(name, raw_hdr)
        theta25_as  = kpc_to_arcsec(z,  25.0)
        theta50_as  = kpc_to_arcsec(z,  50.0)
        theta100_as = kpc_to_arcsec(z, 100.0)
    except Exception as e:
        print(f"[z-miss] {name}: {e}. Proceeding without uv-taper; convolving RAW→target beam only.")
        
    # --- per-taper fudge to fix the long-baseline mismatch (esp. T25) ---
    s25 = s50 = s100 = FUDGE_GLOBAL
    if AUTO_FUDGE:
        if t25_arr is not None:
            s25 = auto_fudge_scale(raw_arr_prefiltered, raw_hdr, t25_hdr, t25_arr,
                                s_grid=np.linspace(1.00, 1.35, 15), nbins=48)
            print(f"[fudge] {name} T25: using s={s25:.3f}")
        if t50_arr is not None:
            s50 = auto_fudge_scale(raw_arr_prefiltered, raw_hdr, t50_hdr, t50_arr,
                                s_grid=np.linspace(1.00, 1.20, 11), nbins=48)
            print(f"[fudge] {name} T50: using s={s50:.3f}")
        if t100_arr is not None:
            s100 = auto_fudge_scale(raw_arr_prefiltered, raw_hdr, t100_hdr, t100_arr,
                                    s_grid=np.linspace(1.00, 1.15, 8), nbins=48)
            print(f"[fudge] {name} T100: using s={s100:.3f}")


    # 2a) convolve RAW → target restoring beam (always)
    r2_25_native  = convolve_to_target(raw_arr_prefiltered, raw_hdr, t25_hdr,  fudge_scale=s25)
    r2_50_native  = convolve_to_target(raw_arr_prefiltered, raw_hdr, t50_hdr,  fudge_scale=s50)
    r2_100_native = convolve_to_target(raw_arr_prefiltered, raw_hdr, t100_hdr, fudge_scale=s100)

    # 2b) uv-taper (disable by default to avoid double-broadening)
    if APPLY_UV_TAPER and UV_TAPER_FRAC > 0:
        r2_25_native  = apply_uv_gaussian_taper(r2_25_native,  raw_hdr, UV_TAPER_FRAC*theta25_as,  pad_factor=2)
        r2_50_native  = apply_uv_gaussian_taper(r2_50_native,  raw_hdr, UV_TAPER_FRAC*theta50_as,  pad_factor=2)
        r2_100_native = apply_uv_gaussian_taper(r2_100_native, raw_hdr, UV_TAPER_FRAC*theta100_as, pad_factor=2)

    # 3) ➜ Reproject **convolved RAW** onto the *tapered* grids
    rt25_on_t  = reproject_like(r2_25_native,  raw_hdr, t25_hdr)  if t25_hdr  is not None else None
    rt50_on_t  = reproject_like(r2_50_native,  raw_hdr, t50_hdr)  if t50_hdr  is not None else None
    rt100_on_t = reproject_like(r2_100_native, raw_hdr, t100_hdr) if t100_hdr is not None else None

    # 4) Downsample everything from its own tapered grid → display size
    def _fmt(arr, ver_hdr):
        if ver_hdr is None or arr is None:
            return None
        # use robust pixel size along x
        s_raw = abs(_cdelt_deg(raw_hdr, 1))
        s_ver = abs(_cdelt_deg(ver_hdr, 1))
        scale = s_raw / s_ver
        Hc = int(round(Hc_raw * scale)); Wc = int(round(Wc_raw * scale))
        ten = torch.from_numpy(arr).unsqueeze(0).float()
        ten = apply_formatting(ten, (ch, Hc, Wc), (ch, outH, outW)).squeeze(0).numpy()
        return ten


    t25_cut   = _fmt(t25_on_t,  t25_hdr)
    t50_cut   = _fmt(t50_on_t,  t50_hdr)
    t100_cut  = _fmt(t100_on_t, t100_hdr)
    rt25_cut  = _fmt(rt25_on_t, t25_hdr)
    rt50_cut  = _fmt(rt50_on_t, t50_hdr)
    rt100_cut = _fmt(rt100_on_t, t100_hdr)
    
    if t25_cut  is not None: check_tensor(f"t25_cut {name}",  torch.tensor(t25_cut))
    if rt25_cut is not None: check_tensor(f"rt25kpc {name}",   torch.tensor(rt25_cut))
    if t50_cut  is not None: check_tensor(f"t50_cut {name}",  torch.tensor(t50_cut))
    if rt50_cut is not None: check_tensor(f"rt50kpc {name}",   torch.tensor(rt50_cut))
    if t100_cut is not None: check_tensor(f"t100_cut {name}", torch.tensor(t100_cut))
    if rt100_cut is not None: check_tensor(f"rt100kpc {name}", torch.tensor(rt100_cut))

    raw_cut = _fmt(raw_arr, raw_hdr)

    ds_factor = (int(round(Hc_raw)) / outH)
    pix_eff_arcsec = pix_native * ds_factor
    hdr_fft = make_fft_header(raw_hdr, pix_eff_arcsec, (outH, outW))

    return (raw_cut, t25_cut, t50_cut, t100_cut,
            rt25_cut, rt50_cut, rt100_cut,
            raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff_arcsec, hdr_fft)
    
# ----------------------------- NEW: one-source figure -----------------------------

def _gaussian2d_isotropic(shape, sigma_px):
    """Return a unit-integral isotropic Gaussian kernel on a given pixel grid."""
    ny, nx = shape
    y = np.arange(ny) - (ny-1)/2.0
    x = np.arange(nx) - (nx-1)/2.0
    X, Y = np.meshgrid(x, y)
    G = np.exp(-(X*X + Y*Y) / (2.0*sigma_px**2))
    s = np.nansum(G)
    if s <= 0 or not np.isfinite(s):  # guard
        return np.zeros_like(G)
    return G / s

def _fft_forward(A, dx, dy):
    """Continuous-norm forward FFT used elsewhere in this file: F = FFT(A) * (dx*dy)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))) * (dx * dy)

def _ifft_inverse(F, dx, dy):
    """Inverse for the above convention: a = IFFT(F) / (dx*dy)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F))).real / (dx * dy)

def _make_W_from_sigma_theta(U, V, sigma_theta_rad, anisotropic=None):
    """
    Analytic W(u,v). If 'anisotropic' is a 2x2 covariance matrix (rad^2),
    use exp[-2π^2 k^T C k]. Otherwise assume isotropic sigma_theta_rad.
    """
    if anisotropic is not None:
        # U,V are in cycles/radian. Build quadratic form per pixel.
        # k = [U, V]; exponent = -2π^2 * (k^T C k)
        a, b, c = float(anisotropic[0,0]), float(anisotropic[0,1]), float(anisotropic[1,1])
        # k^T C k = a U^2 + 2b U V + c V^2
        Q = a*(U*U) + 2.0*b*(U*V) + c*(V*V)
        return np.exp(-2.0 * (np.pi**2) * Q)
    else:
        R2 = (U*U + V*V)
        return np.exp(-2.0 * (np.pi**2) * (sigma_theta_rad**2) * R2)

def _kernel_world_covariance_from_headers(raw_like_hdr, targ_hdr, fudge_scale=1.0):
    """
    Return the kernel's world-space covariance C_ker (rad^2) that turns RAW beam → target beam.
    This mirrors the inner math of kernel_from_beams(...), but returns the *world* C_ker.
    """
    def _beam_cov_radians(bmaj_as, bmin_as, pa_deg):
        sx = _sigma_from_fwhm_arcsec(bmaj_as)
        sy = _sigma_from_fwhm_arcsec(bmin_as)
        th = np.deg2rad(pa_deg)
        R  = np.array([[np.cos(th), -np.sin(th)],
                       [np.sin(th),  np.cos(th)]], dtype=float)
        S  = np.diag([sx**2, sy**2])
        return R @ S @ R.T  # world (rad^2)

    # RAW-like and TARGET beam (arcsec)
    bmaj_r = float(raw_like_hdr['BMAJ'])  * 3600.0
    bmin_r = float(raw_like_hdr['BMIN'])  * 3600.0
    pa_r   = float(raw_like_hdr.get('BPA', 0.0))
    bmaj_t = float(targ_hdr['BMAJ']) * 3600.0
    bmin_t = float(targ_hdr['BMIN']) * 3600.0
    pa_t   = float(targ_hdr.get('BPA', pa_r))

    C_raw = _beam_cov_radians(bmaj_r, bmin_r, pa_r)
    C_tgt = _beam_cov_radians(bmaj_t, bmin_t, pa_t) * (fudge_scale**2)
    C_ker = C_tgt - C_raw

    # clip tiny negatives
    w, V = np.linalg.eigh(C_ker)
    w = np.clip(w, 0.0, None)
    return (V * w) @ V.T  # world (rad^2)


def plot_one_source_full(
    name,
    target_kpc=50,                   # 25, 50, or 100
    out_hw=(128, 128),
    fudge_scale_demo=1.0,
    save_dir="uvmaps",
    blur_fwhm_px=None,               # legacy, ignored (kernel comes from beams)
    use_anisotropic_beam_demo=False, # legacy, ignored (anisotropy handled via beams)
    **kwargs                          # absorb any other legacy kwargs safely
):
    """
    Make a single 2x4 figure for `name` showing both routes:

      TOP (image plane; shared colorbar for first 3):
        [ I ,  I ⊗ G  (image-space RT) ,  𝔉⁻¹{ 𝔉(I) · W } (uv-space RT) ,  G ]

      BOTTOM (uv plane; shared colorbar for first 3, log10 amplitude):
        [ |𝔉(I)| ,  |𝔉(I ⊗ G)| ,  |𝔉(I) · W| ,  W(u,v) ]

    Notes
    -----
    • G is the Gaussian kernel that turns the RAW restoring beam into the TARGET beam
      (from `kernel_from_beams_cached` on the RAW grid), normalized to unit integral.
    • W(u,v) is the continuous-norm FFT of G, rescaled so DC=1. This ensures that
      𝔉(I ⊗ G) = 𝔉(I) · W holds numerically on the same grid.
    • The uv-route image 𝔉⁻¹{𝔉(I)·W} is multiplied by Ω_tgt/Ω_raw to match the
      Jy/beam_target units returned by `convolve_to_target`.
    """


    os.makedirs(save_dir, exist_ok=True)

    # ---------- helpers (continuous-normalized FFT/IFFT) ----------
    def _fft_forward(A, dx, dy):
        return _np.fft.fftshift(_np.fft.fft2(_np.fft.ifftshift(_np.asarray(A, float)))) * (dx * dy)

    def _ifft_inverse(F, dx, dy):
        return _np.fft.fftshift(_np.fft.ifft2(_np.fft.ifftshift(_np.asarray(F, complex)))).real / (dx * dy)

    # ---------- load arrays on a common, FFT-friendly grid ----------
    (raw_cut, t25_cut, t50_cut, t100_cut,
     rt25_cut, rt50_cut, rt100_cut,
     raw_hdr, t25_hdr, t50_hdr, t100_hdr,
     pix_eff_arcsec, hdr_fft) = _load_fits_arrays_scaled(name, crop_ch=1, out_hw=out_hw)

    # choose target
    if int(target_kpc) == 25:
        T, RT, H_targ = t25_cut, rt25_cut, t25_hdr
        t_tag, rt_tag = "T25", "RT25"
    elif int(target_kpc) == 50:
        T, RT, H_targ = t50_cut, rt50_cut, t50_hdr
        t_tag, rt_tag = "T50", "RT50"
    else:
        T, RT, H_targ = t100_cut, rt100_cut, t100_hdr
        t_tag, rt_tag = "T100", "RT100"


    I = raw_cut
    if I is None or RT is None or H_targ is None:
        raise RuntimeError(f"Missing planes for {name}: I={I is not None}, RT={RT is not None}, H_targ={H_targ is not None}")

    H, W = I.shape
    dx = abs(_cdelt_deg(hdr_fft, 1)) * _np.pi/180.0
    dy = abs(_cdelt_deg(hdr_fft, 2)) * _np.pi/180.0

    # ---------- image-space kernel G (RAW → TARGET) ----------
    ker = kernel_from_beams_cached(raw_hdr, H_targ, fudge_scale=fudge_scale_demo)

    # small, native kernel for DISPLAY
    G_small = np.asarray(ker.array, float)
    G_small /= (np.nansum(G_small) + 1e-12)  # unit-integral

    RT_img = convolve_to_target(I, raw_hdr, H_targ, fudge_scale=fudge_scale_demo)
    
    # ---------- uv-space weight W from analytic Gaussian (DC=1) ----------
    # uv grid in cycles/radian
    u = _np.fft.fftshift(_np.fft.fftfreq(W, d=dx))
    v = _np.fft.fftshift(_np.fft.fftfreq(H, d=dy))
    U, V = _np.meshgrid(u, v)

    # world-space covariance of RAW→TARGET kernel (rad^2)
    Cw = _kernel_world_covariance_from_headers(raw_hdr, H_targ,
                                            fudge_scale=fudge_scale_demo)
    a, b, c = float(Cw[0,0]), float(Cw[0,1]), float(Cw[1,1])

    # W(u,v) = exp(-2π² · [a U² + 2b UV + c V²]); DC=1 by construction
    Wuv = _np.exp(-2.0 * (_np.pi**2) * (a*(U**2) + 2.0*b*(U*V) + c*(V**2)))


    # uv-route: multiply 𝔉(I) by W, inverse FFT, then convert units to Jy/beam_target
    F_I_raw = _fft_forward(I, dx, dy)
    F_uv    = F_I_raw * Wuv
    I_uv    = _ifft_inverse(F_uv, dx, dy)
    # match convolve_to_target() output units:
    unit_scale = (_beam_solid_angle_sr(H_targ) / _beam_solid_angle_sr(raw_hdr))
    RT_uv = I_uv * unit_scale

    # For the uv-row we also need |𝔉(I ⊗ G)|
    F_img = _fft_forward(RT_img, dx, dy)            # this already includes unit_scale
    F_uv_scaled = F_uv * unit_scale                 # matches units of F_img
    
    T_true = T  # the real uv-tapered map loaded from disk and resampled
    F_true = _fft_forward(T_true, dx, dy) if T_true is not None else None


    # ---------- shared scales ----------
    # top-row shared (images)
    top_stack = _np.concatenate([
        I[_np.isfinite(I)].ravel(),
        RT_img[_np.isfinite(RT_img)].ravel(),
        RT_uv[_np.isfinite(RT_uv)].ravel(),
        (T_true[_np.isfinite(T_true)].ravel() if T_true is not None else _np.array([]))
    ])

    vmin_top = float(_np.nanpercentile(top_stack, 1.0))
    vmax_top = float(_np.nanpercentile(top_stack, 99.5))
    norm_top = mcolors.Normalize(vmin=vmin_top, vmax=vmax_top)

    # bottom-row shared (log |F| )
    A_raw  = _np.log10(_np.abs(F_I_raw)         + 1e-12)
    A_img  = _np.log10(_np.abs(F_img)           + 1e-12)
    A_uv   = _np.log10(_np.abs(F_uv_scaled)     + 1e-12)
    A_true = _np.log10(_np.abs(F_true) + 1e-12) if F_true is not None else None

    bot_vals = _np.concatenate([
        A_raw[_np.isfinite(A_raw)].ravel(),
        A_img[_np.isfinite(A_img)].ravel(),
        A_uv [_np.isfinite(A_uv )].ravel(),
        (A_true[_np.isfinite(A_true)].ravel() if A_true is not None else _np.array([]))
    ])

    vmin_bot = float(_np.nanpercentile(bot_vals, 1.0))
    vmax_bot = float(_np.nanpercentile(bot_vals, 99.5))
    norm_bot = mcolors.Normalize(vmin=vmin_bot, vmax=vmax_bot)

    # W(u,v) display (log)
    W_show = _np.log10(_np.abs(Wuv) + 1e-12)
    vmin_W = float(_np.nanpercentile(W_show, 2.0))
    vmax_W = float(_np.nanpercentile(W_show, 98.0))
    norm_W = mcolors.Normalize(vmin=vmin_W, vmax=vmax_W)

    # ---------- figure ----------
    fig = plt.figure(figsize=(16.0, 7.8))
    gs  = fig.add_gridspec(2, 5, wspace=0.10, hspace=0.16)

    # ---- TOP ROW (image plane) ----
    axI   = fig.add_subplot(gs[0,0]); im0 = axI.imshow(I,       origin='lower', cmap='viridis', norm=norm_top); axI.set_axis_off(); axI.set_title('RAW  $I(\\ell,m)$')
    axRTi = fig.add_subplot(gs[0,1]); im1 = axRTi.imshow(RT_img,origin='lower', cmap='viridis', norm=norm_top); axRTi.set_axis_off(); axRTi.set_title(f'{rt_tag}: $I\\,*\\,G$  (image space)')
    axRTu = fig.add_subplot(gs[0,2]); im2 = axRTu.imshow(RT_uv, origin='lower', cmap='viridis', norm=norm_top); axRTu.set_axis_off(); axRTu.set_title(f'{rt_tag}: $\\mathcal{{F}}^{{-1}}[\\mathcal{{F}}(I)\\,W]$  (uv space)')
    axT   = fig.add_subplot(gs[0,3]); im3 = axT.imshow(T_true,  origin='lower', cmap='viridis', norm=norm_top); axT.set_axis_off();  axT.set_title(f'{t_tag}  (from FITS)')
    axG   = fig.add_subplot(gs[0,4]); axG.imshow(G_small,       origin='lower', cmap='magma');                  axG.set_axis_off();  axG.set_title('Gaussian kernel  $G$')

    # one shared colorbar for the four image maps (exclude G)
    fig.colorbar(ScalarMappable(norm=norm_top, cmap='viridis'),
                ax=[axI, axRTi, axRTu, axT], fraction=0.046, pad=0.02,
                label='Jy/beam')

    fig.text(0.015, 0.92, 'Image plane: RAW, two RT constructions, and the true tapered map',
            fontsize=11, weight='bold')

    # ---- BOTTOM ROW (uv plane) ----
    axF0 = fig.add_subplot(gs[1,0]); imF0 = axF0.imshow(A_raw,  origin='lower', cmap='viridis', norm=norm_bot); axF0.set_axis_off(); axF0.set_title(r'RAW  $|\mathcal{F}(I)|$  (log)')
    axF1 = fig.add_subplot(gs[1,1]); imF1 = axF1.imshow(A_img,  origin='lower', cmap='viridis', norm=norm_bot); axF1.set_axis_off(); axF1.set_title(rf'{rt_tag}: $|\mathcal{{F}}(I*G)|$  (log)')
    axF2 = fig.add_subplot(gs[1,2]); imF2 = axF2.imshow(A_uv,   origin='lower', cmap='viridis', norm=norm_bot); axF2.set_axis_off(); axF2.set_title(rf'{rt_tag}: $|\mathcal{{F}}(I)\,W|$  (log)')
    axF3 = fig.add_subplot(gs[1,3]); imF3 = axF3.imshow(A_true, origin='lower', cmap='viridis', norm=norm_bot); axF3.set_axis_off(); axF3.set_title(rf'{t_tag}: $|\mathcal{{F}}(T)|$  (log)')
    axWv = fig.add_subplot(gs[1,4]); axWv.imshow(_np.log10(_np.abs(Wuv)+1e-12), origin='lower', cmap='viridis', norm=norm_W); axWv.set_axis_off(); axWv.set_title('$W(u,v)$  (log)')

    # one shared colorbar for the four |F| maps (exclude W)
    fig.colorbar(ScalarMappable(norm=norm_bot, cmap='viridis'),
                ax=[axF0, axF1, axF2, axF3], fraction=0.046, pad=0.02,
                label=r'$\log_{10}$ amplitude (Jy)')

    fig.text(0.015, 0.50, 'UV plane: amplitudes for RAW, both RT routes, and the true tapered map',
            fontsize=11, weight='bold')

    # overall title (use RTxx)
    fig.suptitle(f"{name}  —  {rt_tag}  —  RAW, RT, visibilities, and convolution theorem (two paths)",
                y=0.995, fontsize=12)


    out = os.path.join(save_dir, f"one_source_{name}_{targ_tag}_two_paths.pdf")
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"[one-source] wrote {out}")


def kernel_from_beams(raw_hdr, targ_hdr, fudge_scale=1.0):
    """
    Build a Gaussian2DKernel that turns the RAW restoring beam into the TARGET
    restoring beam on the RAW pixel grid.

    Steps:
      • form beam covariances in world coords (radians),
      • kernel covariance C_ker = C_tgt - C_raw (with optional broadening),
      • map to pixel coords using the full 2×2 WCS Jacobian (CD/PC),
      • make Gaussian2DKernel with those pixel stddevs/orientation.

    Notes
    -----
    - Small negative eigenvalues (numerical noise) are clipped to zero.
    """

    def _beam_cov_radians(bmaj_as, bmin_as, pa_deg):
        sx = _sigma_from_fwhm_arcsec(bmaj_as)
        sy = _sigma_from_fwhm_arcsec(bmin_as)
        th = np.deg2rad(pa_deg)
        R  = np.array([[np.cos(th), -np.sin(th)],
                       [np.sin(th),  np.cos(th)]], dtype=float)
        S  = np.diag([sx**2, sy**2])
        return R @ S @ R.T  # world (rad^2)

    def _cd_matrix_rad(hdr):
        # 2×2 Jacobian: d(world)/d(pixel) in radians/pixel (handles rotation/anisotropy)
        if 'CD1_1' in hdr:
            CD = np.array([[hdr['CD1_1'], hdr.get('CD1_2', 0.0)],
                           [hdr.get('CD2_1', 0.0), hdr['CD2_2']]], dtype=float)
        else:
            pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
            pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
            cd1  = hdr.get('CDELT1', 1.0); cd2  = hdr.get('CDELT2', 1.0)
            CD   = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
        return CD * (np.pi / 180.0)  # deg/pix → rad/pix

    # Pull beam params (arcsec + PA)
    bmaj_r = float(raw_hdr['BMAJ'])  * 3600.0
    bmin_r = float(raw_hdr['BMIN'])  * 3600.0
    pa_r   = float(raw_hdr.get('BPA', 0.0))
    bmaj_t = float(targ_hdr['BMAJ']) * 3600.0
    bmin_t = float(targ_hdr['BMIN']) * 3600.0
    pa_t   = float(targ_hdr.get('BPA', pa_r))  # fall back to RAW PA if missing

    # World-space covariances and kernel covariance
    C_raw = _beam_cov_radians(bmaj_r, bmin_r, pa_r)
    C_tgt = _beam_cov_radians(bmaj_t, bmin_t, pa_t) * (fudge_scale**2)
    C_ker = C_tgt - C_raw

    # Numerical guard
    w, V = np.linalg.eigh(C_ker)
    w = np.clip(w, 0.0, None)
    C_ker = (V * w) @ V.T

    # Map kernel covariance into pixel coordinates
    J    = _cd_matrix_rad(raw_hdr)       # d(world)/d(pixel)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker @ Jinv.T         # pixel covariance

    # Eigen-decompose in pixel coords
    w_pix, V_pix = np.linalg.eigh(Cpix)
    w_pix = np.clip(w_pix, 0.0, None)
    s_major = float(np.sqrt(w_pix[1]))
    s_minor = float(np.sqrt(w_pix[0]))
    theta   = float(np.arctan2(V_pix[1, 1], V_pix[0, 1]))  # angle of major axis

    eps = 1e-9  # avoid zero-width kernels when beams are almost identical
    s_major = max(s_major, eps)
    s_minor = max(s_minor, eps)
    
    # inside kernel_from_beams(), after s_major/s_minor/theta are computed
    nker = int(np.ceil(8.0 * max(s_major, s_minor))) | 1  # make it odd
    ker  = Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)
    #ker = Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta)
    return ker

_KER_CACHE = {}
def kernel_from_beams_cached(raw_hdr, targ_hdr, fudge_scale=1.0):
    key = (
        round(float(raw_hdr['BMAJ']),9), round(float(raw_hdr['BMIN']),9), round(float(raw_hdr.get('BPA',0.0)),6),
        round(float(targ_hdr['BMAJ']),9), round(float(targ_hdr['BMIN']),9), round(float(targ_hdr.get('BPA', raw_hdr.get('BPA',0.0))),6),
        tuple(round(float(x),12) for x in (
            raw_hdr.get('CD1_1', raw_hdr.get('CDELT1', 0.0)),
            raw_hdr.get('CD1_2', 0.0),
            raw_hdr.get('CD2_1', 0.0),
            raw_hdr.get('CD2_2', raw_hdr.get('CDELT2', 0.0)),
        )),
        round(float(fudge_scale),6),
    )
    ker = _KER_CACHE.get(key)
    if ker is None:
        ker = kernel_from_beams(raw_hdr, targ_hdr, fudge_scale=fudge_scale)
        _KER_CACHE[key] = ker
    return ker

def make_fft_header(raw_hdr, pix_eff_arcsec, out_hw):
    h = fits.Header()

    # --- orientation matrix from RAW ---
    if 'CD1_1' in raw_hdr:
        M = np.array([[raw_hdr['CD1_1'], raw_hdr.get('CD1_2', 0.0)],
                      [raw_hdr.get('CD2_1', 0.0), raw_hdr['CD2_2']]], float)
    else:
        pc11 = raw_hdr.get('PC1_1', 1.0); pc12 = raw_hdr.get('PC1_2', 0.0)
        pc21 = raw_hdr.get('PC2_1', 0.0); pc22 = raw_hdr.get('PC2_2', 1.0)
        cd1  = raw_hdr.get('CDELT1', -1.0); cd2 = raw_hdr.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])  # deg/pix

    # unit-column orientation (preserve rotation/handedness), new isotropic step
    sx = abs(_cdelt_deg(raw_hdr, 1))
    sy = abs(_cdelt_deg(raw_hdr, 2))
    if sx <= 0 or sy <= 0 or not np.isfinite(sx*sy):
        s1 = np.sign(raw_hdr.get('CDELT1', raw_hdr.get('CD1_1', -1.0)) or -1.0)
        s2 = np.sign(raw_hdr.get('CDELT2', raw_hdr.get('CD2_2',  1.0)) or  1.0)
        step = pix_eff_arcsec/3600.0
        CD_new = np.array([[s1*step, 0.0],[0.0, s2*step]], float)
    else:
        R = M @ np.diag([1.0/sx, 1.0/sy])
        CD_new = R * (pix_eff_arcsec/3600.0)

    outH, outW = out_hw
    h['CD1_1'] = float(CD_new[0,0]); h['CD1_2'] = float(CD_new[0,1])
    h['CD2_1'] = float(CD_new[1,0]); h['CD2_2'] = float(CD_new[1,1])
    h['NAXIS1'] = int(outW); h['NAXIS2'] = int(outH)

    # put the reference pixel at the center of the new cutout
    h['CRPIX1'] = 0.5*(outW+1); h['CRPIX2'] = 0.5*(outH+1)

    # anchor CRVAL to the sky position that was at the RAW center
    try:
        w_raw = WCS(raw_hdr).celestial
        ra_c, dec_c = w_raw.wcs_pix2world([[raw_hdr['NAXIS1']/2.0, raw_hdr['NAXIS2']/2.0]], 0)[0]
        h['CRVAL1'] = float(ra_c); h['CRVAL2'] = float(dec_c)
    except Exception:
        for k in ('CRVAL1','CRVAL2'):
            if k in raw_hdr: h[k] = raw_hdr[k]

    # carry frame type strings (safe to copy verbatim)
    for k in ('CTYPE1','CTYPE2'):
        if k in raw_hdr: h[k] = raw_hdr[k]

    return h


def _maybe_draw_beam(ax, j, r):
    try:
        hdr = r.get('hdr25') if j in (0,1) else r.get('hdr50') if j in (3,4) else r.get('hdr100') if j in (6,7) else None
        if hdr is None or r.get('pix_eff') is None:
            return
        bmaj_as = float(hdr['BMAJ']) * 3600.0
        bmin_as = float(hdr['BMIN']) * 3600.0
        pa_deg  = float(hdr.get('BPA', 0.0))
        maj_px  = bmaj_as / float(r['pix_eff'])
        min_px  = bmin_as / float(r['pix_eff'])
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        cx = x0 + 0.12*(x1 - x0); cy = y0 + 0.12*(y1 - y0)
        e1 = Ellipse((cx, cy), width=maj_px, height=min_px, angle=-pa_deg, fill=False, lw=1.4, ec='w', alpha=0.9)
        e2 = Ellipse((cx, cy), width=maj_px, height=min_px, angle=-pa_deg, fill=False, lw=2.2, ec='k', alpha=0.3)
        ax.add_patch(e2); ax.add_patch(e1)
    except Exception:
        pass

def _beam_solid_angle_sr(hdr):
    """Gaussian beam solid angle in steradians from FITS header (BMAJ/BMIN in deg)."""
    bmaj_deg = float(hdr['BMAJ'])
    bmin_deg = float(hdr['BMIN'])
    bmaj_rad = abs(bmaj_deg) * np.pi / 180.0
    bmin_rad = abs(bmin_deg) * np.pi / 180.0
    return (np.pi / (4.0 * np.log(2.0))) * bmaj_rad * bmin_rad

def convolve_to_target(raw_arr, raw_hdr, target_hdr, fudge_scale=1.0):
    """
    Same as convolve_to_target(); provided for convenience when the image is
    already on the RAW grid. Units in = Jy/beam_native; units out = Jy/beam_target.
    """
    ker = kernel_from_beams_cached(raw_hdr, target_hdr, fudge_scale=fudge_scale)
    out = convolve_fft(raw_arr, ker, boundary='fill', fill_value=np.nan,
                   nan_treatment='interpolate', normalize_kernel=True,
                   psf_pad=True, fft_pad=True, allow_huge=True)    
    try:
        out *= (_beam_solid_angle_sr(target_hdr) / _beam_solid_angle_sr(raw_hdr))
    except Exception:
        pass
    return out

def _first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def _pixscale_arcsec(hdr):
    if 'CDELT1' in hdr:
        return abs(hdr['CDELT1']) * 3600.0
    # fallback if using CD matrix
    cd11 = hdr.get('CD1_1'); cd12 = hdr.get('CD1_2', 0.0)
    if cd11 is not None:
        pix_deg = np.hypot(cd11, cd12)
        return abs(pix_deg) * 3600.0
    raise KeyError("No CDELT* or CD* keywords in FITS header")


def auto_fudge_scale(raw_img, raw_hdr, targ_hdr, T_img,
                     s_grid=np.linspace(1.00, 1.40, 41), nbins=36):
    """
    Automatically determine a fudge_scale to improve the match between
    the tapered image T_img and the convolved raw_img → targ_hdr.
    A fudge_scale is a multiplicative factor on the target beam size
    used in the convolution kernel (kernel_from_beams). 
    This is to compensate for imperfect weighting of long baselines
    in the original imaging, which leads to a mismatch in the
    radial brightness profiles at high spatial frequencies.
    The fudge_scale should be ≥1.0 (1.0 = no change).
    
    The best fudge_scale is the one that minimizes the median absolute
    deviation of the radial brightness ratio from 1.0, over the range
    where the coherence is good (>0.6).
    
    Parameters
    ----------
    raw_img : 2D array
        The raw image on its native grid (Jy/beam_native).
    raw_hdr : FITS header
        The FITS header for raw_img.
    targ_hdr : FITS header
        The FITS header for the tapered image T_img.
    T_img : 2D array
        The tapered image on its own grid (Jy/beam_target).
    s_grid : 1D array
        The grid of fudge_scale values to try.
    nbins : int
        Number of radial bins for the comparison.
        
    Returns
    -------
    best_s : float
        The best fudge_scale found (within s_grid).
    """


    # Work entirely on the tapered grid
    best_s, best_cost = 1.0, np.inf
    U,V,FT,AT = image_to_vis(T_img, targ_hdr, beam_hdr=targ_hdr)

    for s in s_grid:
        RT_native = convolve_to_target(raw_img, raw_hdr, targ_hdr, fudge_scale=s)
        RT_on_t   = reproject_like(RT_native, raw_hdr, targ_hdr)
        _,_,FR,AR = image_to_vis(RT_on_t, targ_hdr, beam_hdr=targ_hdr)

        # radial medians + coherence (same as your vis_compare_quicklook)
        r, aT = _radial_bin(U,V,AT, nbins=nbins, stat='median')
        _, aR = _radial_bin(U,V,AR, nbins=nbins, stat='median')

        # crude coherence by annulus
        Rgrid = np.sqrt(U*U + V*V)
        edges = np.geomspace(np.nanpercentile(Rgrid, 1.0), np.nanmax(Rgrid), nbins+1)
        coh = []
        for i in range(nbins):
            m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
            if not np.any(m): coh.append(np.nan); continue
            num  = np.nanmean(FT[m]*np.conj(FR[m]))
            den1 = np.nanmean(np.abs(FT[m])**2)
            den2 = np.nanmean(np.abs(FR[m])**2)
            coh.append(np.abs(num)/np.sqrt(den1*den2))
        coh = np.asarray(coh)
        
        good = np.isfinite(aT) & np.isfinite(aR) & (coh > 0.6)

        # focus on the central 20–80% baselines where taper matters
        qlo, qhi = np.nanpercentile(r[good], [20, 80])
        band = good & (r >= qlo) & (r <= qhi)
        cost = np.inf
        if np.any(band):
            # joint gain+fudge in log-space (gain closed-form)
            d = np.log(aT[band]) - np.log(aR[band])
            w = (coh[band]**2)
            g = np.exp(np.nansum(w * d) / np.nansum(w))   # best gain
            cost = np.nanmedian(np.abs(np.log(aT[band]) - np.log(g*aR[band])))
        if cost < best_cost:
            best_cost, best_s = cost, float(s)
    return best_s

def _cdelt_deg(hdr, axis):
    """Robust pixel step [deg] for axis=1 (x≈RA) or axis=2 (y≈Dec)."""
    key = f'CDELT{axis}'
    if key in hdr:
        return float(hdr[key])
    if 'CD1_1' in hdr:  # fall back to CD
        if axis == 1:
            return float(np.hypot(hdr['CD1_1'], hdr.get('CD1_2', 0.0)))
        else:
            return float(np.hypot(hdr.get('CD2_1', 0.0), hdr['CD2_2']))
    pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
    pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
    cd1  = hdr.get('CDELT1', 1.0); cd2  = hdr.get('CDELT2', 1.0)
    M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    return float(np.hypot(M[0, axis-1], M[1, axis-1]))

def image_to_vis(img, img_hdr, beam_hdr=None, divide_by_beam=True,
                 window=None, alpha=0.35, pad_factor=2, roi=0.9,
                 subtract_mean=False):
    """
    Convert a 2D sky image to its discrete Fourier transform (visibilities).

    • Optional conversion to Jy/sr (divide by Ω_beam) before the FFT.
    • Robust DC removal: subtract median offset within a central ellipse
      (fraction 'roi' of image size; set roi=0 to disable).
    • Apodization window: 'tukey' (default) or 'hann'
    • Strong Tukey apodization to tame edge/aliasing with alpha=0.35 (default);
    • Optional zero-padding (×2 by default) to reduce wrap-around.
    """
    B = np.array(img, dtype=float)                  # keep NaNs for stats
    A = np.where(np.isfinite(B), B, 0.0)            # NaNs → 0 for FFT path

    # Jy/beam → Jy/sr if requested
    if divide_by_beam and (beam_hdr is not None):
        try:
            A = A / _beam_solid_angle_sr(beam_hdr)
        except Exception:
            pass

    ny, nx = A.shape

    # DC removal using central ellipse
    if subtract_mean and (0.0 < roi <= 1.0):
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        ry, rx = (roi * ny) / 2.0, (roi * nx) / 2.0
        yy, xx = np.ogrid[:ny, :nx]
        mask_roi = (((yy - cy) / ry)**2 + ((xx - cx) / rx)**2) <= 1.0
        if np.any(mask_roi):
            dc = np.nanmedian(np.where(mask_roi, B, np.nan))
            if np.isfinite(dc):
                A = A - dc

    # Apodize
    if window == 'tukey':
        A *= radial_tukey(ny, nx, alpha=alpha)
    elif window == 'hann':
        A *= radial_hann(ny, nx)
        
    # Zero-padding
    if pad_factor and pad_factor > 1:
        ny2, nx2 = int(ny * pad_factor), int(nx * pad_factor)
        py = (ny2 - ny) // 2; px = (nx2 - nx) // 2
        A = np.pad(A, ((py, ny2 - ny - py), (px, nx2 - nx - px)),
                   mode='constant', constant_values=0.0)
        ny, nx = A.shape

    # Pixel size [rad]
    dx = abs(_cdelt_deg(img_hdr, 1)) * np.pi / 180.0
    dy = abs(_cdelt_deg(img_hdr, 2)) * np.pi / 180.0

    # Continuous-norm FFT ⇒ |F| in Jy
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))) * (dx * dy) # centralise 0-freq of a 2D FT of inverse shifted A

    # Frequency axes (cycles/radian = wavelengths)
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    U, V = np.meshgrid(u, v)
    return U, V, F, np.abs(F)


def _radial_bin(u, v, values, nbins=36, rmin=None, rmax=None, logspace=True, stat='median'):
    """Radial bin a 2D field in uv. values can be real/abs/complex; returns r_centers, stat(values)."""
    R = np.sqrt(u*u + v*v)
    m = np.isfinite(values)
    if rmin is None: rmin = np.nanpercentile(R[m], 1.0)
    if rmax is None: rmax = np.nanmax(R[m])
    edges = (np.geomspace(rmin, rmax, nbins+1) if logspace else np.linspace(rmin, rmax, nbins+1))
    idx = np.digitize(R.ravel(), edges) - 1
    out = []
    for i in range(nbins):
        sel = (idx == i) & m.ravel()
        if not np.any(sel):
            out.append(np.nan); continue
        vals = values.ravel()[sel]
        if stat == 'median':
            out.append(np.nanmedian(vals))
        elif stat == 'mean':
            out.append(np.nanmean(vals))
        elif stat == 'abs-mean':
            out.append(np.nanmean(np.abs(vals)))
        else:
            out.append(np.nanmedian(vals))
    rc = 0.5*(edges[:-1] + edges[1:])
    return rc, np.asarray(out)

def vis_compare_quicklook(name, T, RT, RAW, hdr_img, hdr_beam_T, hdr_beam_RAW,
                          tag, outdir='.', nbins=36):
    """
    Compact figure comparing visibilities of RAW, RT (= RAW→T beam), and T.

    Left block (2×4 small panels):
      top:   |F_RAW|, |F_RT|, |F_T|, ||F_T|-|F_RT||   (all log10 with shared norm)
      bottom:arg(F_RAW), arg(F_RT), arg(F_T), Δϕ(T−RT) (deg, shared norm)

    Right block: radial medians for |F_T| and |F_RT| + ratio/coherence (as before).
    """

    # --- FFTs (all on the same img grid, dividing by the appropriate beam) ---
    U,V,Fraw,Araw = image_to_vis(RAW, hdr_img, beam_hdr=hdr_beam_RAW)
    _,_,FR,AR     = image_to_vis(RT,  hdr_img, beam_hdr=hdr_beam_T)
    _,_,FT,AT     = image_to_vis(T,   hdr_img, beam_hdr=hdr_beam_T)

    # Differences
    dphi = np.angle(FT * np.conj(FR))            # phase(T) − phase(RT)
    dA   = np.abs(AT - AR)                        # ||F_T|-|F_RT||

    # Radial stats for the right-hand plot (unchanged)
    r, aT = _radial_bin(U,V,AT, nbins=nbins, stat='median')
    _, aR = _radial_bin(U,V,AR, nbins=nbins, stat='median')
    ratio = aT / (aR + 1e-12)

    Rgrid = np.sqrt(U*U + V*V)
    edges = np.geomspace(np.nanpercentile(Rgrid,1.0), np.nanmax(Rgrid), nbins+1)
    num_r, den1_r, den2_r = [], [], []
    for i in range(nbins):
        m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
        if not np.any(m):
            num_r.append(np.nan); den1_r.append(np.nan); den2_r.append(np.nan); continue
        num  = np.nanmean(FT[m] * np.conj(FR[m]))
        den1 = np.nanmean(np.abs(FT[m])**2)
        den2 = np.nanmean(np.abs(FR[m])**2)
        num_r.append(np.abs(num)); den1_r.append(den1); den2_r.append(den2)
    coh = np.asarray(num_r) / np.sqrt(np.asarray(den1_r) * np.asarray(den2_r))

    # ---- figure ----
    fig = plt.figure(figsize=(13.4, 6.6))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.55, 1.55, 1.35],  # a bit wider to fit 4 small columns
        height_ratios=[1.0, 0.44],
        wspace=0.22, hspace=0.22
    )

    # 2×4 grid for small UV panels
    gs_maps = gs[0, 0:2].subgridspec(2, 4, wspace=0.08, hspace=0.10)

    # ---------- Amplitudes (log10) with one shared normalisation ----------
    amp_log_raw = np.log10(Araw + 1e-12)
    amp_log_rt  = np.log10(AR   + 1e-12)
    amp_log_t   = np.log10(AT   + 1e-12)
    both = np.concatenate([
        amp_log_raw[np.isfinite(amp_log_raw)].ravel(),
        amp_log_rt [np.isfinite(amp_log_rt )].ravel(),
        amp_log_t  [np.isfinite(amp_log_t  )].ravel()
    ])
    vmin = np.nanpercentile(both, 1.0)
    vmax = np.nanpercentile(both, 99.5)
    shared_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    axA0 = fig.add_subplot(gs_maps[0, 0]); imA0 = axA0.imshow(amp_log_raw, origin='lower', aspect='equal', norm=shared_norm, cmap='viridis'); axA0.set_title('|F_RAW|'); axA0.set_axis_off()
    axA1 = fig.add_subplot(gs_maps[0, 1]); imA1 = axA1.imshow(amp_log_rt,  origin='lower', aspect='equal', norm=shared_norm, cmap='viridis'); axA1.set_title('|F_RT|');  axA1.set_axis_off()
    axA2 = fig.add_subplot(gs_maps[0, 2]); imA2 = axA2.imshow(amp_log_t,   origin='lower', aspect='equal', norm=shared_norm, cmap='viridis'); axA2.set_title('|F_T|');   axA2.set_axis_off()

    # ||F_T|-|F_RT|| uses the SAME scale as amplitudes
    axA3 = fig.add_subplot(gs_maps[0, 3])
    dA_log = np.log10(dA + 1e-12)
    dA_log = np.clip(dA_log, vmin, vmax)
    imA3 = axA3.imshow(dA_log, origin='lower', aspect='equal', cmap='viridis', norm=shared_norm)
    axA3.set_title('||F_T|-|F_RT||'); axA3.set_axis_off()

    # One shared colour bar for the 4 amplitude panels
    fig.colorbar(imA2, ax=[axA0, axA1, axA2, axA3], location='right',
                 fraction=0.046, pad=0.04, label='log10 amplitude (Jy)')

    # ---------- Phases with one shared colour bar ----------
    phase_norm = mcolors.Normalize(-180, 180)
    cmap_phase = 'twilight'

    axP0 = fig.add_subplot(gs_maps[1, 0]); imP0 = axP0.imshow(np.rad2deg(np.angle(Fraw)), origin='lower', aspect='equal', cmap=cmap_phase, norm=phase_norm); axP0.set_axis_off()
    axP1 = fig.add_subplot(gs_maps[1, 1]); imP1 = axP1.imshow(np.rad2deg(np.angle(FR)),   origin='lower', aspect='equal', cmap=cmap_phase, norm=phase_norm); axP1.set_axis_off()
    axP2 = fig.add_subplot(gs_maps[1, 2]); imP2 = axP2.imshow(np.rad2deg(np.angle(FT)),   origin='lower', aspect='equal', cmap=cmap_phase, norm=phase_norm); axP2.set_axis_off()
    axP3 = fig.add_subplot(gs_maps[1, 3]); imP3 = axP3.imshow(np.rad2deg(dphi),           origin='lower', aspect='equal', cmap=cmap_phase, norm=phase_norm); axP3.set_axis_off()

    sm_phase = ScalarMappable(norm=phase_norm, cmap=cmap_phase); sm_phase.set_array([])
    fig.colorbar(sm_phase, ax=[axP0, axP1, axP2, axP3], location='right',
                 fraction=0.046, pad=0.04, label='phase (deg)', ticks=[-180,-90,0,90,180])

    # ---------- Right column: |F| medians, ratio & coherence (unchanged) ----------
    axR = fig.add_subplot(gs[:, 2])
    kλ = r/1e3
    axR.plot(kλ, aT, lw=1.5, color='C0', label='|F_T| (median)')
    axR.plot(kλ, aR, lw=1.5, color='r',  label='|F_RT| (median)')
    axR.set_xscale('log'); axR.set_yscale('log')
    axR.set_xlabel(r'baseline $r=\sqrt{u^2+v^2}$ (kλ)'); axR.set_ylabel('median |F| (Jy)')
    axR.grid(True, which='both', ls=':', alpha=0.4)

    good = (coh > 0.6)
    thrT = np.nanpercentile(aT, 5.0)
    thrR = np.nanpercentile(aR, 5.0)
    good &= (aT > thrT) & (aR > thrR)
    ratio_plot = np.where(good, ratio, np.nan)

    axR2 = axR.twinx()
    axR2.plot(kλ, ratio_plot, lw=1.4, color='k', label='|F_T| / |F_RT|')
    axR2.plot(kλ, coh,        lw=1.2, color='k', ls=':', label='coherence')
    axR2.set_ylabel('ratio / coherence')
    ymax = np.nanmax([np.nanmax(coh), np.nanmax(ratio_plot), 1.0])
    axR2.set_ylim(0, max(1.05, ymax))

    lines, labels = axR.get_legend_handles_labels()
    lines2, labels2 = axR2.get_legend_handles_labels()
    axR.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=9)

    # ---------- Bottom: phase stats vs baseline (as before) ----------
    axPstat = fig.add_subplot(gs[1, 0:2])
    phi_bar_deg, sigma_circ_deg, rphi = [], [], []
    for i in range(nbins):
        m = (Rgrid >= edges[i]) & (Rgrid < edges[i+1]) & np.isfinite(dphi)
        if not np.any(m):
            phi_bar_deg.append(np.nan); sigma_circ_deg.append(np.nan); rphi.append(np.nan); continue
        w = AT[m] * AR[m]
        z = np.exp(1j * dphi[m])
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if np.sum(w) <= 0: w = np.ones_like(w)
        C = np.sum(w * z) / np.sum(w)
        Rlen = np.abs(C)
        phi_bar_deg.append(np.rad2deg(np.angle(C)))
        sigma_circ_deg.append(np.rad2deg(np.sqrt(np.maximum(0.0, -2.0*np.log(Rlen)))))
        rphi.append(0.5*(edges[i] + edges[i+1]))
    rphi = np.asarray(rphi); phi_bar_deg = np.asarray(phi_bar_deg); sigma_circ_deg = np.asarray(sigma_circ_deg)
    axPstat.plot(rphi/1e3, np.abs(phi_bar_deg), lw=1.2, label='|mean Δϕ|')
    axPstat.plot(rphi/1e3, sigma_circ_deg, lw=1.2, ls='--', label=r'$\sigma(\Delta\phi)=\sqrt{-2\ln R}$')
    axPstat.set_xscale('log'); axPstat.set_xlabel('baseline (kλ)'); axPstat.set_ylabel('phase (deg)')
    axPstat.grid(True, which='both', ls=':', alpha=0.4); axPstat.legend(loc='upper left', fontsize=9)

    fig.suptitle(f'{name}  –  {tag}', y=0.98, fontsize=12)
    save_dir = os.path.join(outdir, 'uvmaps'); os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'uvcmp_{name}_{tag}.pdf')
    fig.savefig(out, dpi=250, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {out}')
    
hdr = fits.getheader("/users/mbredber/scratch/data/PSZ2/fits/PSZ2G192.18+56.12/PSZ2G192.18+56.12CHANDRA.fits")
print("SIM_Z raw:", repr(hdr.get("SIM_Z")))

def plot_galaxy_grid(images, filenames, labels,
                     lo=60, hi=95, RES_PCT=99,
                     RES_CMAP     = "RdBu_r",
                     SKIP_CLIP_NORM=False,
                     SCALE_SEPARATE=False,
                     DO_VIS=False):
    """
    Layout (9 columns):
      T25, RAW→25, res25,  T50, RAW→50, res50,  T100, RAW→100, res100

    When SKIP_CLIP_NORM=True:
      • No percentile clipping or normalisation anywhere.
      • Residuals are abs(T - RAW→T) in native units, each panel uses Matplotlib autoscale.

    When SKIP_CLIP_NORM=False (default):
      • Per-row percentile clip (lo,hi) shared by that row's T and RAW→T panels.
      • Residuals use a per-row max given by RES_PCT of res.
    """
    # --- keep arrays aligned ---
    n_imgs = images.shape[0]
    n = min(n_imgs, len(filenames), len(labels))
    if (n_imgs != n) or (len(filenames) != n) or (len(labels) != n):
        print(f"[warn] aligning arrays: images={n_imgs}, filenames={len(filenames)}, labels={len(labels)} → n={n}")
    images    = images[:n]
    filenames = filenames[:n]
    labels    = labels[:n]

    # gather all candidates per class
    de_all  = [i for i, y in enumerate(labels) if int(y) == 50]
    nde_all = [i for i, y in enumerate(labels) if int(y) == 51]

    # keep only those that have a usable redshift (fast check)
    if REQUIRE_REDSHIFT_ONLY:
        de_idx, nde_idx = [], []
        for i in de_all:
            if _has_redshift_for_index(i, filenames):
                de_idx.append(i)
            if len(de_idx) == 3:
                break
        for i in nde_all:
            if _has_redshift_for_index(i, filenames):
                nde_idx.append(i)
            if len(nde_idx) == 3:
                break
    else:
        de_idx  = de_all[:3]
        nde_idx = nde_all[:3]
        
    if len(de_idx) == 0 and len(nde_idx) == 0:
        sample = [ _name_base_from_fn(f) for f in filenames[:6] ]
        for nm in sample:
            bd = _find_base_dir(nm)
            print(f"[z-scan] {nm}: base_dir={bd}")
            if bd and os.path.isdir(bd):
                for p in sorted(glob.glob(f"{bd}/*.fits"))[:6]:
                    try:
                        hdr = fits.getheader(p)
                        keys = [k for k in hdr.keys() if any(x in k.upper() for x in ("Z","RED"))]
                        print("   ", os.path.basename(p), "keys:", keys)
                    except Exception as e:
                        print("   ", os.path.basename(p), "err:", e)
        raise RuntimeError("No sources with redshift found by header check. Verify filenames/dirs and header keys.")

    # If nothing survives, fall back: take the first few per class and (later) skip uv-taper
    if len(de_idx) == 0 and len(nde_idx) == 0:
        raise RuntimeError("No sources with redshift found by header check. Verify filenames/dirs and header keys.")

    order = de_idx + nde_idx
    if len(order) == 0:
        print("[error] Nothing to plot (no candidates at all).")
        return
    
    # Make sure indices are valid for the (now aligned) arrays
    order = [i for i in order if 0 <= i < len(images)]
    if not order:
        print("[error] selection produced no valid indices after bounds-check.")
        return


    images    = images[order]
    filenames = [filenames[i] for i in order]

    # dynamic grid
    top    = min(3, len(de_idx))
    bot    = min(3, len(nde_idx))
    spacer = 1 if (top > 0 and bot > 0) else 0

    n_cols = 9
    n_rows = top + spacer + bot
    cell   = 1.6
    height_ratios = [1]*top + ([0.12] if spacer else []) + [1]*bot

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*cell, n_rows*cell*0.98),
        gridspec_kw=dict(
            left=0.04, right=0.995, top=0.92, bottom=0.05,
            wspace=0.04, hspace=0.04,
            height_ratios=height_ratios),
        constrained_layout=False
    )
    # ensure axes is 2D even when n_rows==1
    import numpy as _np
    if n_rows == 1:
        axes = _np.array([axes])

    col_titles = [
        "T25 kpc", "RAW → 25 kpc", "res 25 kpc",
        "T50 kpc", "RAW → 50 kpc", "res 50 kpc",
        "T100 kpc", "RAW → 100 kpc", "res 100 kpc",
    ]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=12, pad=6)

    # row positions for the chosen sources (skip the spacer index if present)
    row_map = list(range(top)) + list(range(top + spacer, top + spacer + bot)) 
    print("Row map:", row_map)
    outH, outW = images.shape[-2], images.shape[-1]

    # ---------- PASS 1: load, convolve, and paired debug ----------
    rows = []
    for i_src, grid_row in enumerate(row_map):
        name = _name_base_from_fn(filenames[i_src])
        print("Name:", name)
        (raw_cut, t25_cut, t50_cut, t100_cut,
        rt25_cut, rt50_cut, rt100_cut,
        raw_hdr, t25_hdr, t50_hdr, t100_hdr,
        pix_eff, hdr_fft) = _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(outH, outW))
        print("Name:", name, "raw_cut shape:", None if raw_cut is None else raw_cut.shape,
              "t25_cut shape:", None if t25_cut is None else t25_cut.shape,
              "t50_cut shape:", None if t50_cut is None else t50_cut.shape,
              "t100_cut shape:", None if t100_cut is None else t100_cut.shape)
        
        rows.append(dict(
            name=name, grid_row=grid_row,
            t25=t25_cut, t50=t50_cut, t100=t100_cut,
            rt25=rt25_cut, rt50=rt50_cut, rt100=rt100_cut,
            hdr25=t25_hdr, hdr50=t50_hdr, hdr100=t100_hdr,
            hdr_fft=hdr_fft, pix_eff=pix_eff
        ))
        
        # -------- Controls and checks --------
        if DO_VIS:
            if (t25_cut is not None) and (rt25_cut is not None):
                vis_compare_quicklook(name,
                                    T=t25_cut, RT=rt25_cut, RAW=raw_cut,
                                    hdr_img=hdr_fft,
                                    hdr_beam_T=t25_hdr, hdr_beam_RAW=raw_hdr,
                                    tag='T25')
            if (t50_cut is not None) and (rt50_cut is not None):
                vis_compare_quicklook(name,
                                    T=t50_cut, RT=rt50_cut, RAW=raw_cut,
                                    hdr_img=hdr_fft,
                                    hdr_beam_T=t50_hdr, hdr_beam_RAW=raw_hdr,
                                    tag='T50')
            if (t100_cut is not None) and (rt100_cut is not None):
                vis_compare_quicklook(name,
                                    T=t100_cut, RT=rt100_cut, RAW=raw_cut,
                                    hdr_img=hdr_fft,
                                    hdr_beam_T=t100_hdr, hdr_beam_RAW=raw_hdr,
                                    tag='T100')

        # --- Registration sanity + optional correction on the display grid ---
        for tgt, T, RT in [(25, t25_cut, rt25_cut),
                        (50, t50_cut, rt50_cut),
                        (100, t100_cut, rt100_cut)]:
            if T is not None and RT is not None:
                dy, dx = xcorr_shift(T, RT)
                print(f"{name}  T{tgt} vs RT{tgt}: estimated shift dy={dy:.2f}, dx={dx:.2f} px")
                
    # Print the summary stats for each row once                
    for r in rows:
        name = r['name']
        for tgt in (25, 50, 100):
            T  = r.get(f"t{tgt}")
            RT = r.get(f"rt{tgt}")
            if T is None or RT is None:
                continue
            sT, sR = _summ(T), _summ(RT)
            rms = _nanrms(T, RT)
            print(f"Row {name}  T{tgt}kpc vs RT{tgt}kpc  |  "
                f"T[min={sT['min']:.3g}, max={sT['max']:.3g}, mean={sT['mean']:.3g}, std={sT['std']:.3g}]  ||  "
                f"RT[min={sR['min']:.3g}, max={sR['max']:.3g}, mean={sR['mean']:.3g}, std={sR['std']:.3g}]  |  "
                f"RMSΔ={rms:.3g}")

    def describe(h):
        return f"{h['BMAJ']*3600:.2f}\"×{h['BMIN']*3600:.2f}\" @ PA={h.get('BPA',0):.1f}°"
    try:
        if (raw_hdr is not None) and (t50_hdr is not None):
            print("RAW:", describe(raw_hdr), "  T50:", describe(t50_hdr))
    except NameError:
        pass
            
    # ---------- FAST PATH: no clipping/normalisation ----------
    if SKIP_CLIP_NORM:
        # Build planes with SIGNED residuals
        for r in rows:
            res25  = (r['t25']  - r['rt25'])  if (r['t25']  is not None and r['rt25']  is not None) else None
            res50  = (r['t50']  - r['rt50'])  if (r['t50']  is not None and r['rt50']  is not None) else None
            res100 = (r['t100'] - r['rt100']) if (r['t100'] is not None and r['rt100'] is not None) else None
            r['planes'] = [r['t25'], r['rt25'], res25,
                        r['t50'], r['rt50'], res50,
                        r['t100'], r['rt100'], res100]

        # one shared scale for T/RT (positive images)
        NONRES_COLS = {0,1,3,4,6,7}
        nonres_vals = [a for r in rows for j,a in enumerate(r['planes'])
                    if a is not None and j in NONRES_COLS]
        vmin = min(float(np.nanmin(a)) for a in nonres_vals) if nonres_vals else 0.0
        vmax = max(float(np.nanmax(a)) for a in nonres_vals) if nonres_vals else 1.0

        # symmetric robust scale for signed residuals
        RES_COLS = {2,5,8}
        resid_vals = [a for r in rows for j,a in enumerate(r['planes'])
                    if a is not None and j in RES_COLS]
        if resid_vals:
            rr = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in resid_vals])
            R = float(np.nanpercentile(rr, RES_PCT)) or 1.0
        else:
            R = 1.0
        resid_norm = TwoSlopeNorm(vmin=-R, vcenter=0.0, vmax=+R)

        # draw
        img_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # <-- add this
        for r in rows:
            gr = r['grid_row']
            for j, arr in enumerate(r['planes']):
                ax = axes[gr, j]
                if arr is not None:
                    if j in RES_COLS:
                        # was: norm=res_norm_global  (doesn't exist here)
                        ax.imshow(arr, cmap=RES_CMAP, norm=resid_norm,
                                origin="lower", interpolation="nearest")
                    else:
                        # was: norm=img_norm  (but img_norm wasn't defined)
                        ax.imshow(arr, cmap="viridis", norm=img_norm,
                                origin="lower", interpolation="nearest")
                        _maybe_draw_beam(ax, j, r)
                ax.set_axis_off()

        # two stacked colorbars (non-resid, resid)
        sm_nonres = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
        sm_resid  = ScalarMappable(norm=resid_norm, cmap=RES_CMAP)
        cax_nonres = fig.add_axes([0.996, 0.55, 0.012, 0.38])
        cax_resid  = fig.add_axes([0.996, 0.07, 0.012, 0.38])
        fig.colorbar(sm_nonres, cax=cax_nonres, label="Jy/beam")
        fig.colorbar(sm_resid,  cax=cax_resid,  label="Δ (Jy/beam)")

        # spacer strip + group labels (dynamic)
        if spacer:
            s = top
            for j in range(n_cols):
                axes[s, j].axis('off')
                axes[s, j].set_facecolor('white')
                if s-1 >= 0:
                    axes[s-1, j].spines['bottom'].set_color('white')
                    axes[s-1, j].spines['bottom'].set_linewidth(3)
                if s+1 < n_rows:
                    axes[s+1, j].spines['top'].set_color('white')
                    axes[s+1, j].spines['top'].set_linewidth(3)

        # place "DE" and "NDE" at the center row of each group
        if top > 0:
            r_mid_de = (top-1)//2
            axes[r_mid_de, 0].text(-0.20, 0.5, "DE",
                                transform=axes[r_mid_de,0].transAxes,
                                va='center', ha='right', fontsize=13, fontweight='bold')
        if bot > 0:
            r0_bot   = top + spacer
            r_mid_nd = r0_bot + (bot-1)//2
            axes[r_mid_nd, 0].text(-0.20, 0.5, "NDE",
                                transform=axes[r_mid_nd,0].transAxes,
                                va='center', ha='right', fontsize=13, fontweight='bold')

        outname = "psz2_psfmatch_pairs25_50_100_signedres_TWO_CBARS.pdf"
        plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Wrote {outname}")
        print("Used the short-circuit path with SKIP_CLIP_NORM=True.")
        return
    
    # ---------- Optional: show global clip stats (for reference only) ----------
    all_vals = []
    for r in rows:
        for k in ("t25","t50","t100","rt25","rt50","rt100"):
            a = r[k]
            if a is not None:
                a = np.asarray(a, float)
                a = a[np.isfinite(a)]
                if a.size:
                    all_vals.append(a.ravel())
    if all_vals:
        v = np.concatenate(all_vals)
        g_lo, g_hi = np.nanpercentile(v, [lo, hi])
        if not np.isfinite(g_hi - g_lo) or g_hi <= g_lo:
            g_lo, g_hi = np.nanmin(v), np.nanmax(v)
        print(f"Global clip percentiles: {lo}→{hi} gives {g_lo:.3e} to {g_hi:.3e}")

    # ---------- Build planes per row (SIGNED residuals, real clipping) ----------
    rows_planes = []
    R_all = []  # collect residuals to make a single residual colorbar later

    for r in rows:
        # Per-row percentiles, shared by that row's T and RT panels
        vals = []
        for k in ("t25","t50","t100","rt25","rt50","rt100"):
            a = r[k]
            if a is None: 
                continue
            a = np.asarray(a, float)
            a = a[np.isfinite(a)]
            if a.size: vals.append(a.ravel())
        if vals:
            v = np.concatenate(vals)
            lo_i, hi_i = np.nanpercentile(v, [lo, hi])
            if not np.isfinite(hi_i - lo_i) or hi_i <= lo_i:
                lo_i, hi_i = np.nanmin(v), np.nanmax(v)
        else:
            lo_i, hi_i = 0.0, 1.0

        def _stretch_panel(arr):
            if arr is None:
                return None
            plo, phi = _nanpct(arr, lo=lo, hi=hi)
            if not np.isfinite(phi - plo) or phi <= plo:
                plo, phi = np.nanmin(arr), np.nanmax(arr)
            return _stretch_from_p(arr, plo, phi, clip=True)

        if not SCALE_SEPARATE:
            t25_s   = _stretch_from_p(r['t25'],   lo_i, hi_i, clip=True)  if r['t25']   is not None else None
            rt25_s  = _stretch_from_p(r['rt25'],  lo_i, hi_i, clip=True)  if r['rt25']  is not None else None
            t50_s   = _stretch_from_p(r['t50'],   lo_i, hi_i, clip=True)  if r['t50']   is not None else None
            rt50_s  = _stretch_from_p(r['rt50'],  lo_i, hi_i, clip=True)  if r['rt50']  is not None else None
            t100_s  = _stretch_from_p(r['t100'],  lo_i, hi_i, clip=True)  if r['t100']  is not None else None
            rt100_s = _stretch_from_p(r['rt100'], lo_i, hi_i, clip=True)  if r['rt100'] is not None else None
        else:
            t25_s   = _stretch_panel(r['t25'])
            rt25_s  = _stretch_panel(r['rt25'])
            t50_s   = _stretch_panel(r['t50'])
            rt50_s  = _stretch_panel(r['rt50'])
            t100_s  = _stretch_panel(r['t100'])
            rt100_s = _stretch_panel(r['rt100'])


        # SIGNED residuals, in the same stretched space
        res25   = (t25_s  - rt25_s)   if (t25_s  is not None and rt25_s  is not None) else None
        res50   = (t50_s  - rt50_s)   if (t50_s  is not None and rt50_s  is not None) else None
        res100  = (t100_s - rt100_s)  if (t100_s is not None and rt100_s is not None) else None

        # Robust symmetric limit for residuals (per row)
        rr = [np.abs(x[np.isfinite(x)]).ravel() for x in (res25,res50,res100) if x is not None]
        R_i = float(np.nanpercentile(np.concatenate(rr), 99.5)) if rr else 1.0
        if not np.isfinite(R_i) or R_i == 0: R_i = 1.0

        r['planes']   = [t25_s, rt25_s, res25,  t50_s, rt50_s, res50,  t100_s, rt100_s, res100]
        r['res_norm'] = TwoSlopeNorm(vmin=-R_i, vcenter=0.0, vmax=+R_i)
        rows_planes.append(r)
        if rr: R_all.append(R_i)

    # norms/which columns
    img_norm  = mcolors.Normalize(vmin=0.0, vmax=1.0)  # stretched images live in [0,1]
    RES_COLS  = {2,5,8}

    # ---------- draw ----------
    for r in rows_planes:
        gr = r['grid_row']
        for j, arr in enumerate(r['planes']):
            ax = axes[gr, j]
            if arr is not None:
                if j in RES_COLS:
                    ax.imshow(arr, cmap=RES_CMAP, norm=r['res_norm'], origin="lower", interpolation="nearest")
                else:
                    ax.imshow(arr, cmap="viridis", norm=img_norm, origin="lower", interpolation="nearest")
                    _maybe_draw_beam(ax, j, r)

            ax.set_axis_off()
        axes[gr, 0].text(-0.02, 0.5, r['name'], transform=axes[gr,0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=8)

    # two stacked colorbars (shared for all rows)
    cax_nonres = fig.add_axes([0.996, 0.55, 0.012, 0.38])
    cax_resid  = fig.add_axes([0.996, 0.07, 0.012, 0.38])
    fig.colorbar(ScalarMappable(norm=img_norm, cmap="viridis"), cax=cax_nonres,
                label="stretched intensity (0–1)")
    valid_R = [float(r) for r in R_all if np.isfinite(r) and r > 0]
    R_global = max(valid_R) if valid_R else 1.0
    res_norm_global = TwoSlopeNorm(vmin=-R_global, vcenter=0.0, vmax=+R_global)
    fig.colorbar(ScalarMappable(norm=res_norm_global, cmap=RES_CMAP),
                cax=cax_resid, label="Δ (stretched units)")

    # spacer strip + group labels (dynamic)
    if spacer:
        s = top
        for j in range(n_cols):
            axes[s, j].axis('off')
            axes[s, j].set_facecolor('white')
            if s-1 >= 0:
                axes[s-1, j].spines['bottom'].set_color('white')
                axes[s-1, j].spines['bottom'].set_linewidth(3)
            if s+1 < n_rows:
                axes[s+1, j].spines['top'].set_color('white')
                axes[s+1, j].spines['top'].set_linewidth(3)

    if top > 0:
        r_mid_de = (top-1)//2
        axes[r_mid_de, 0].text(-0.20, 0.5, "DE",
                               transform=axes[r_mid_de,0].transAxes,
                               va='center', ha='right', fontsize=13, fontweight='bold')
    if bot > 0:
        r0_bot   = top + spacer
        r_mid_nd = r0_bot + (bot-1)//2
        axes[r_mid_nd, 0].text(-0.20, 0.5, "NDE",
                               transform=axes[r_mid_nd,0].transAxes,
                               va='center', ha='right', fontsize=13, fontweight='bold')

    outname = f"psz2_psfmatch_pairs25_50_100_with_residuals_ROWCLIP.pdf"
    plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {outname}")


if __name__ == "__main__": 
    result = load_galaxies(
        galaxy_classes=[50, 51],
        versions=['T25kpc', 'T50kpc', 'T100kpc'],   # or include 'RAW' if you want it in the cube
        fold=5,
        crop_size=(1, 512, 512),                 # same physical FOV for all versions
        downsample_size=(1, 128, 128),           # common output size
        sample_size=500,
        USE_GLOBAL_NORMALISATION=False,             
        NORMALISE=False, STRETCH=False,           
        NORMALISETOPM=False, AUGMENT=False,
        EXTRADATA=False, PRINTFILENAMES=True,
        train=False, DEBUG=True
    )
    eval_imgs   = result[2]
    eval_labels = result[3]  # eval_labels lives at index 3
    eval_fns    = result[5]
    
    for probe in ["PSZ2G120.08-44", "PSZ2G150.56+58", "PSZ2G031.93+78"]:
        z = _z_from_meta(probe)
        print(f"[meta-test] {probe} → z={z}")
        
    # --- Example: make the one-source figure for the first eval example ---
    example_name = _name_base_from_fn(eval_fns[0])
    print("[one-source] building figure for:", example_name)
    plot_one_source_full(
        name=example_name,
        target_kpc=50,             # choose 25 / 50 / 100
        out_hw=(128, 128),         # should match the loader’s downsample size
        blur_fwhm_px=5.0,          # size of the demo Gaussian in image pixels
        use_anisotropic_beam_demo=False  # set True to base W(u,v) on the beam difference
    )
    
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels, lo=60, hi=100, SKIP_CLIP_NORM=False, SCALE_SEPARATE=True, DO_VIS=True)