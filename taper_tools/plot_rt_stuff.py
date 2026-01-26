"""
Used to figure out how to build RT50kpc images from RAW FITS.

One-source quicklook (compact + well-annotated)

What you get (4 columns × 4 rows):
Row 1 (image plane, all share one colorbar/scale):
  [ I * G ,  F^{-1}{F(I) · W} ,  T50kpc(from FITS) ,  kernel G ]
Row 2 (uv plane, log10 amplitude, all share one colorbar/scale):
  [ log10|F(I*G)| , log10|F(I)·W| , log10|F(T50)| , log10 W(u,v) ]
Row 3 (image residuals, zero-centered diverging):
  [ (I*G)-(uv route) , (I*G)-T50 , (uv route)-T50 , (blank) ]
Row 4 (uv residuals, log10 |Δ amplitude|):
  [ | |F(I*G)| - |F(I)·W| | , | |F(I*G)| - |F(T50)| | , | |F(I)·W| - |F(T50)| | , (blank) ]

Notes
-----
• The *kernel G* is unit-integral and dimensionless; we STILL plot it with the *same
  colormap AND the same numeric scale* as the images in Row 1, exactly as requested.
  This is purely for visual comparison; do not interpret the absolute values physically.
• Likewise, *log10 W(u,v)* uses the SAME colormap + scale as the three uv amplitude
  maps in Row 2 to make their relative shapes directly comparable.

Dependencies: numpy, matplotlib, astropy (fits, WCS), astropy.convolution (Gaussian2DKernel)
"""

import os, re, glob, random, numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.coordinates import Angle
import astropy.units as u
from astropy.wcs import WCS
from scipy.ndimage import zoom as _zoom, shift as _imgshift
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
try:
    from reproject import reproject_interp
    HAVE_REPROJECT = True
except Exception:
    HAVE_REPROJECT = False
mpl.rcParams.update({
    "font.size": 12,          # base font
    "axes.titlesize": 12,     # subplot titles
    "axes.labelsize": 12,     # axis labels (for colorbars)
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 13,   # suptitle
    "legend.fontsize": 11,
})

ARCSEC_PER_RAD = 60**2*180/np.pi

def summarize(Y):
    # Y: (nsrc, nbins)
    mu = np.nanmedian(Y, axis=0)
    sd = np.nanstd(Y, axis=0)
    n  = np.sum(np.isfinite(Y), axis=0)
    sd[n < 2] = np.nan     # avoid DoF warnings / meaningless bands
    return mu, sd

# --- UV radial profiles and coherence ---
def _radial_profiles(U, V, A_T, A_IG, A_IW, F_T, F_IG, nbins=28):
    """
    Build radial (baseline r) profiles:
      - medians of |F_T|, |F_RT| (take RT≡IW), and their ratio
      - binned complex coherence R in [0,1]
      - phase metrics: <|Δφ|> (direct) and σ(Δφ) (radians)
    Returns dict with arrays per bin (centers).
    """
    r = np.hypot(U, V)                  # wavelengths (cycles/radian)
    rk = r / 1e3                        # kλ
    # log-spaced bins across occupied uv-range
    rmin = np.nanmax([np.nanmin(rk[rk>0]),  rk.max()*1e-3])
    rmax = rk.max()
    edges = np.geomspace(rmin, rmax, nbins+1)
    rc = np.sqrt(edges[:-1]*edges[1:])  # bin centers
    out = {"r_kla": rc}

    def _bin_stat(X, fn=np.nanmedian):
        y = np.empty(nbins); y[:] = np.nan
        for i in range(nbins):
            m = (rk>=edges[i]) & (rk<edges[i+1]) & np.isfinite(X)
            if m.any(): y[i] = fn(X[m])
        return y

    # amplitude medians
    med_T  = _bin_stat(A_T)
    med_RT = _bin_stat(A_IW)            # RT route ≡ I·W
    med_IG = _bin_stat(A_IG)
    # ratios (median of ratio is noisier → ratio of medians is stabler)
    ratio_T_IG = med_T / (med_IG + 1e-30)

    # coherence per bin: |<F_T F_RT*>| / sqrt(<|F_T|^2><|F_RT|^2>)
    R = np.empty(nbins); R[:] = np.nan
    mean_abs_dphi = np.empty(nbins); mean_abs_dphi[:] = np.nan
    for i in range(nbins):
        m = (rk>=edges[i]) & (rk<edges[i+1])
        if not m.any(): continue
        Ft = F_T[m]; Fr = F_IG[m]
        num = np.nanmean(Ft * np.conj(Fr))
        den = np.sqrt(np.nanmean(np.abs(Ft)**2) * np.nanmean(np.abs(Fr)**2)) + 1e-30
        R[i] = np.abs(num) / den
        dphi = np.angle(Ft) - np.angle(Fr)
        dphi = (dphi + np.pi) % (2*np.pi) - np.pi   # wrap to [-π,π]
        mean_abs_dphi[i] = np.nanmean(np.abs(dphi))

    # phase sigma from coherence
    sigma_phi = np.sqrt(np.maximum(0.0, -2.0*np.log(np.clip(R, 1e-12, 1.0))))
    out.update({
        "med_T": med_T, "med_RT": med_RT, "med_IG": med_IG,
        "ratio_T_IG": ratio_T_IG, "R": R,
        "mean_abs_dphi": mean_abs_dphi, "sigma_phi": sigma_phi
    })
    return out



def center_crop_at(arr, ny_target, nx_target, cy, cx):
    ny, nx = arr.shape
    y0 = int(round(cy - ny_target/2)); x0 = int(round(cx - nx_target/2))
    y0 = max(0, min(y0, ny - ny_target)); x0 = max(0, min(x0, nx - nx_target))
    return arr[y0:y0+ny_target, x0:x0+nx_target]

def _nanrms(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m): return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d*d)))

def robust_affine_match_with_offset(T, X, nsig=3.0, roi=0.85, max_iter=5):
    """
    Solve for g, b so that g*X + b ≈ T (robust, sigma-clipped, within ROI).
    Returns (g, b, stats).
    """
    T = np.asarray(T, float); X = np.asarray(X, float)
    ny, nx = T.shape
    cy, cx = (ny-1)/2., (nx-1)/2.
    ry, rx = roi*ny/2., roi*nx/2.
    yy, xx = np.ogrid[:ny, :nx]
    m = (((yy-cy)/ry)**2 + ((xx-cx)/rx)**2) <= 1.0
    m &= np.isfinite(T) & np.isfinite(X)

    g, b = 1.0, 0.0
    for _ in range(max_iter):
        A = np.column_stack([X[m], np.ones(m.sum())])
        y = T[m]
        g_new, b_new = np.linalg.lstsq(A, y, rcond=None)[0]
        R = T - (g_new*X + b_new)
        med = np.nanmedian(R[m])
        mad = np.nanmedian(np.abs(R[m] - med))
        sig = 1.4826*mad if (mad > 0 and np.isfinite(mad)) else np.nan
        if np.isfinite(sig):
            m_new = m & (np.abs(R - med) < nsig*sig)
            if m_new.sum() == m.sum():
                g, b = float(g_new), float(b_new)
                break
            m = m_new
        g, b = float(g_new), float(b_new)

    rms = _nanrms(T, g*X + b)
    return g, b, {"Npix": int(m.sum()), "rms": rms}

def micro_blur_to_match(T, X, max_sigma_pix=1.5, ngrid=12):
    """
    Try a small *additional* circular Gaussian blur on X to minimize RMS vs T.
    Returns (sigma_pix_opt, X_blurred).
    """
    from astropy.convolution import Gaussian2DKernel, convolve_fft
    best_s, best_y, best_r = 0.0, X, _nanrms(T, X)
    if max_sigma_pix <= 0: 
        return 0.0, X
    for s in np.linspace(0.0, max_sigma_pix, ngrid):
        if s <= 1e-6:
            Y = X
        else:
            ker = Gaussian2DKernel(s)
            Y = convolve_fft(X, ker, normalize_kernel=True, nan_treatment='interpolate',
                             boundary='fill', fill_value=0.0, psf_pad=True, fft_pad=True,
                             allow_huge=True)
        r = _nanrms(T, Y)
        if r < best_r:
            best_s, best_y, best_r = float(s), Y, r
    return best_s, best_y


def center_crop_like_fov(arr, ny_target, nx_target):
    """Center crop a 2D array to (ny_target, nx_target)."""
    ny, nx = arr.shape
    y0 = max(0, (ny - ny_target) // 2)
    x0 = max(0, (nx - nx_target) // 2)
    return arr[y0:y0+ny_target, x0:x0+nx_target]

def pick_random_raws(root_dir, n=5, seed=0, pattern="**/*.fits"):
    """
    Return up to n random RAW FITS under root_dir, excluding any file
    that already looks like a T50kpc product.
    """
    # recursive glob
    files = glob.glob(os.path.join(root_dir, pattern), recursive=True)
    raws  = [p for p in files if "T50kpc" not in Path(p).name]
    if not raws:
        raise FileNotFoundError(f"No RAW fits found under {root_dir}")
    rng = random.Random(seed)
    if len(raws) <= n:
        return raws
    return rng.sample(raws, n)

def find_source_pairs(root_dir):
    """
    Find (RAW, T50) pairs where both live in the same leaf directory and
    share the directory's basename:
      <dir>/<dir>.fits                 ← RAW
      <dir>/<dir>T50kpc.fits           ← T50
    Ignores any other FITS variants (SUB, XMM, CHANDRA, compact-model, etc.).
    """
    pairs = []
    # depth-1 leaf directories under root_dir
    for d in glob.glob(os.path.join(root_dir, "*")):
        if not os.path.isdir(d):
            continue
        base = Path(d).name
        raw = os.path.join(d, f"{base}.fits")
        t50 = os.path.join(d, f"{base}T50kpc.fits")
        if os.path.exists(raw) and os.path.exists(t50):
            pairs.append((raw, t50))
    return pairs

def pick_random_pairs(root_dir, n=5, seed=42):
    pairs = find_source_pairs(root_dir)
    if not pairs:
        raise FileNotFoundError(f"No (RAW, T50kpc) pairs found under {root_dir}")
    rng = random.Random(seed)
    if len(pairs) <= n:
        return pairs
    return rng.sample(pairs, n)


def _rotation_deg(hdr):
    """Approx orientation (deg) of +x pixel axis on-sky from the CD/PC matrix."""
    J = _cd_matrix_rad(hdr)
    ang = np.degrees(np.arctan2(J[1,0], J[0,0]))  # direction of column 1
    return float(ang)

def _wcs_center(hdr):
    """Return (RA, Dec) at the geometric center of the image (deg)."""
    try:
        w = WCS(hdr).celestial
        nx, ny = float(hdr['NAXIS1']), float(hdr['NAXIS2'])
        ra, dec = w.wcs_pix2world([[nx/2.0, ny/2.0]], 0)[0]
        return float(ra), float(dec)
    except Exception:
        return np.nan, np.nan

def _print_wcs_info(tag, hdr):
    dx, dy = pix_scales_rad(hdr)
    asx, asy = dx*206265.0, dy*206265.0
    ang = _rotation_deg(hdr)
    ra, dec = _wcs_center(hdr)
    print(f"[WCS] {tag}: {int(hdr['NAXIS1'])}×{int(hdr['NAXIS2'])} px | "
          f"{asx:.6f}×{asy:.6f} arcsec/px | rot~{ang:+.3f} deg | "
          f"center RA={ra:.6f} deg, Dec={dec:.6f} deg")

def _phase_xcorr_shift(a, b):
    """
    Phase correlation shift (dy, dx) between 2D arrays (float, can contain NaNs).
    Returns sub-pixel estimate using a simple 1D quadratic peak fit.
    """
    A = np.nan_to_num(a - np.nanmedian(a), nan=0.0)
    B = np.nan_to_num(b - np.nanmedian(b), nan=0.0)
    FA = np.fft.fftn(A); FB = np.fft.fftn(B)
    R = FA * np.conj(FB); R /= (np.abs(R) + 1e-12)
    c = np.fft.ifftn(R).real
    ij = np.unravel_index(np.argmax(c), c.shape)
    ny, nx = a.shape
    dy = ij[0]; dx = ij[1]
    if dy > ny//2: dy -= ny
    if dx > nx//2: dx -= nx

    # subpixel refinement along each axis
    def _subpix(axis_vals, pos):
        m = int(pos)
        if m <= 0 or m >= len(axis_vals)-1:
            return float(pos)
        a0, a1, a2 = axis_vals[m-1], axis_vals[m], axis_vals[m+1]
        denom = (a0 - 2*a1 + a2)
        if abs(denom) < 1e-12:
            return float(pos)
        return m + 0.5*(a0 - a2)/denom

    # refine
    dyf = _subpix(c[:, (dx % nx)], dy)
    dxf = _subpix(c[(dy % ny), :], dx)
    return float(dyf), float(dxf)


# --- tiny WCS cache & equality checks ---
_WCS_CACHE = {}
def _wcs_celestial_cached(h):
    key = id(h)
    w = _WCS_CACHE.get(key)
    if w is None:
        w = WCS(h).celestial
        _WCS_CACHE[key] = w
    return w

def _same_pixel_grid(h1, h2, atol=1e-12):
    for k in ('NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2'):
        if h1.get(k) != h2.get(k):
            return False
    def _cd(h):
        if 'CD1_1' in h:
            return (float(h['CD1_1']), float(h.get('CD1_2', 0.0)),
                    float(h.get('CD2_1', 0.0)), float(h['CD2_2']))
        # PC * CDELT fallback
        pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
        pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
        c1=h.get('CDELT1',-1.0); c2=h.get('CDELT2', 1.0)
        M = np.array([[pc11,pc12],[pc21,pc22]],float) @ np.diag([c1,c2])
        return (float(M[0,0]), float(M[0,1]), float(M[1,0]), float(M[1,1]))
    return (np.allclose(_cd(h1), _cd(h2), atol=atol) and
            np.allclose([h1.get('CRPIX1'),h1.get('CRPIX2')],
                        [h2.get('CRPIX1'),h2.get('CRPIX2')], atol=1e-9) and
            np.allclose([h1.get('CRVAL1'),h1.get('CRVAL2')],
                        [h2.get('CRVAL1'),h2.get('CRVAL2')], atol=1e-9))

DEBUG_REPROJECT = True  # set False to silence

def robust_gain_match(ref_T, X, nsig=3.0, roi=0.85):
    """
    Solve for g* so that g*·X ≈ ref_T in a robust sense.
    - ref_T: target image (T50 on RAW grid)
    - X:     candidate image (IG or IU)
    Returns (g*, stats dict)
    """
    T = np.asarray(ref_T, float); Y = np.asarray(X, float)
    ny, nx = T.shape
    cy, cx = (ny-1)/2., (nx-1)/2.
    ry, rx = roi*ny/2., roi*nx/2.
    yy, xx = np.ogrid[:ny, :nx]
    in_roi = (((yy-cy)/ry)**2 + ((xx-cx)/rx)**2) <= 1.0

    # estimate noise via MAD on differences in the ROI
    D = T - Y
    mad = np.nanmedian(np.abs(D[in_roi] - np.nanmedian(D[in_roi])))
    sigma = 1.4826*mad if np.isfinite(mad) and mad>0 else np.nan

    # high-S/N mask
    m = in_roi & np.isfinite(T) & np.isfinite(Y)
    if np.isfinite(sigma):
        m &= (np.abs(T) > nsig*sigma) | (np.abs(Y) > nsig*sigma)

    if not np.any(m):
        # fall back to all finite pixels
        m = np.isfinite(T) & np.isfinite(Y)

    # weighted least squares with simple weights (could also use |T| or |Y|)
    t = T[m]; x = Y[m]
    # closed-form slope (no intercept): minimize ||t - g x||
    num = np.sum(t*x)
    den = np.sum(x*x) + 1e-12
    g = float(num/den)

    # diagnostics
    res = t - g*x
    rms = float(np.sqrt(np.mean(res*res)))
    return g, {"Npix": int(m.sum()), "sigma_est": float(sigma) if np.isfinite(sigma) else None, "rms": rms}


def reproject_like(arr, src_hdr, dst_hdr):
    """
    Put a 2-D image on the pixel grid defined by dst_hdr.

    Priority:
      1) If 'reproject' is available and both headers have valid celestial WCS,
         use bilinear reproject_interp to the exact output shape.
      2) Otherwise: resize to dst shape with scipy.ndimage.zoom, then (if WCS
         exists) align image centers via a subpixel shift. This guarantees the
         returned array has shape (dst_hdr['NAXIS2'], dst_hdr['NAXIS1']).
    """
    if arr is None or src_hdr is None or dst_hdr is None:
        return None

    if _same_pixel_grid(src_hdr, dst_hdr):
        if DEBUG_REPROJECT:
            print("[reproject_like] grids identical → return as-is")
        return np.asarray(arr, float)

    # prepare WCS
    w_src = w_dst = None
    try:
        w_src = _wcs_celestial_cached(src_hdr)
        w_dst = _wcs_celestial_cached(dst_hdr)
    except Exception:
        w_src = w_dst = None

    ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])

    if HAVE_REPROJECT and (w_src is not None) and (w_dst is not None):
        if DEBUG_REPROJECT:
            print("[reproject_like] using reproject_interp "
                  f"→ shape_out=({ny_out},{nx_out}), order=bilinear")
        reproj, _ = reproject_interp((arr, w_src), w_dst,
                                     shape_out=(ny_out, nx_out),
                                     order='bilinear')
        return reproj.astype(float)

    # Fallback: zoom + (optional) center alignment
    arr = np.asarray(arr, float)
    ny_in, nx_in = arr.shape
    zy = ny_out / max(ny_in, 1); zx = nx_out / max(nx_in, 1)
    arr_rs = _zoom(arr, zoom=(zy, zx), order=1)

    if DEBUG_REPROJECT:
        print(f"[reproject_like] FALLBACK zoom: in=({ny_in},{nx_in}) out=({ny_out},{nx_out}) "
              f"zoom=({zy:.6f},{zx:.6f})")

    try:
        if (w_src is not None) and (w_dst is not None):
            ra, dec = w_src.wcs_pix2world([[nx_in/2.0, ny_in/2.0]], 0)[0]
            x_dst, y_dst = w_dst.wcs_world2pix([[ra, dec]], 0)[0]
            dx = (nx_out/2.0) - x_dst; dy = (ny_out/2.0) - y_dst
            if DEBUG_REPROJECT:
                print(f"[reproject_like] center-align shift: dx={dx:.3f} px, dy={dy:.3f} px")
            arr_rs = _imgshift(arr_rs, shift=(dy, dx), order=1, mode="nearest")
    except Exception as e:
        if DEBUG_REPROJECT:
            print(f"[reproject_like] center-align failed: {e!r}")

    # exact shape
    arr_rs = arr_rs[:ny_out, :nx_out]
    if arr_rs.shape != (ny_out, nx_out):
        pad_y = ny_out - arr_rs.shape[0]; pad_x = nx_out - arr_rs.shape[1]
        arr_rs = np.pad(arr_rs, ((0, max(0, pad_y)), (0, max(0, pad_x))),
                        mode='edge')[:ny_out, :nx_out]
    return arr_rs.astype(float)


# ------------------------------ FITS helpers ------------------------------

def load_fits(path):
    """Read a FITS file → (2D array as float, header). Squeezes any singleton axes."""
    data = np.squeeze(fits.getdata(path)).astype(float)
    hdr  = fits.getheader(path)
    return data, hdr

def _cd_matrix_rad(hdr):
    """
    2×2 pixel→world Jacobian in radians/pixel, robust to PC/CDELT or CD keywords.
    This captures pixel scale anisotropy and rotation (handedness) of the image grid.
    """
    if 'CD1_1' in hdr:
        M = np.array([[hdr['CD1_1'], hdr.get('CD1_2', 0.0)],
                      [hdr.get('CD2_1', 0.0), hdr['CD2_2']]], float)
    else:
        pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
        pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
        cd1  = hdr.get('CDELT1', 1.0); cd2  = hdr.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    return M * (np.pi/180.0)

def pix_scales_rad(hdr):
    """
    Effective pixel steps (|dx|,|dy|) in radians from the Jacobian columns.
    Only the magnitudes matter for the FFT sampling.
    """
    J = _cd_matrix_rad(hdr)
    dx = np.hypot(J[0,0], J[1,0])  # length of column 1 (x-direction)
    dy = np.hypot(J[0,1], J[1,1])  # length of column 2 (y-direction)
    return abs(dx), abs(dy)

# -------------------------- Beam → covariance utils --------------------------

def _fwhm_as_to_sigma_rad(fwhm_as):
    """
    Convert FWHM [arcsec] → Gaussian sigma [radians].
    FWHM = 2*sqrt(2*ln 2)*sigma.
    """
    return (float(fwhm_as) / (2.0*np.sqrt(2.0*np.log(2.0)))) * (np.pi/(180.0*3600.0))

def beam_cov_world(hdr):
    """
    Beam covariance in *world coordinates* (radians^2) from FITS BMAJ/BMIN/BPA.
    """
    bmaj_as = float(hdr['BMAJ']) * 3600.0
    bmin_as = float(hdr['BMIN']) * 3600.0
    pa_deg  = float(hdr.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th      = np.deg2rad(pa_deg)
    R       = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], float)
    S       = np.diag([sx**2, sy**2])
    return R @ S @ R.T

def beam_solid_angle_sr(hdr):
    """Gaussian beam solid angle in steradians from BMAJ/BMIN [deg] in FITS header."""
    bmaj = abs(float(hdr['BMAJ'])) * np.pi/180.0
    bmin = abs(float(hdr['BMIN'])) * np.pi/180.0
    return (np.pi/(4.0*np.log(2.0))) * bmaj * bmin

def kernel_from_beams(raw_hdr, tgt_hdr):
    """
    Build a Gaussian2DKernel (in *pixel coordinates of RAW grid*) that turns the RAW
    restoring beam into the TARGET restoring beam.
      1) compute world covariances C_raw, C_tgt,
      2) kernel covariance C_ker = C_tgt - C_raw (clip tiny negatives),
      3) map to pixel coords via full 2×2 Jacobian, then
      4) create a finite, odd-sized kernel (≈ ±4σ support).
    """
    # world covariances
    C_raw = beam_cov_world(raw_hdr)
    C_tgt = beam_cov_world(tgt_hdr)
    C_ker = C_tgt - C_raw

    # clean numerics: no negative variances
    w, V  = np.linalg.eigh(C_ker)
    w = np.clip(w, 0.0, None)
    C_ker = (V * w) @ V.T

    # map to pixel coords
    J    = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker @ Jinv.T

    # eigen-decompose in pixel coords → principal sigmas and orientation
    wp, Vp = np.linalg.eigh(Cpix)
    wp = np.clip(wp, 1e-18, None)
    s_minor = float(np.sqrt(wp[0]))
    s_major = float(np.sqrt(wp[1]))
    theta   = float(np.arctan2(Vp[1,1], Vp[0,1]))
    nker    = int(np.ceil(8.0*max(s_major, s_minor))) | 1  # odd-sized window
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

# ------------------------------ FFT utilities ------------------------------

def fft2c(A, dx, dy):
    """
    Continuous-norm FFT (so amplitudes are in the same physical units as the image):
      F = FFT(A) * (dx * dy)
    We zero-fill NaNs to avoid contaminating the FFT.
    """
    A0 = np.nan_to_num(np.asarray(A, float), nan=0.0)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A0))) * (dx * dy)

def ifft2c(F, dx, dy):
    """Inverse of fft2c: a = IFFT(F) / (dx * dy). Real part is the sky image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F))).real / (dx * dy)

def uv_grid(ny, nx, dx, dy):
    """Regular uv grid (cycles/radian) matching an ny×nx image with pixel steps dx,dy [rad]."""
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    return np.meshgrid(u, v)

def W_from_Cker(U, V, C_world):
    """
    Analytic taper in uv-plane for a Gaussian kernel with world covariance C_world:
      W(u,v) = exp(-2π² · kᵀ C_world k)  with k=[u,v] in cycles/radian (wavelengths).
    DC gain is 1 by construction.
    """
    a = float(C_world[0,0]); b = float(C_world[0,1]); c = float(C_world[1,1])
    Q = a*(U*U) + 2.0*b*(U*V) + c*(V*V)
    return np.exp(-2.0 * (np.pi**2) * Q)


def make_rt_quicklook(raw_fits, t50_fits=None, out_pdf="one_source_quicklook_with_residuals.pdf",
                      fov_arcmin=None, force_square=True,
                      per_image_stretch=False, percentile_lo=1.0, percentile_hi=99.5):
    # --- small helpers (scoped to this function) ---
    ARCSEC_PER_RAD = 206264.80624709636

    def _arcsec_per_pix(hdr):
        dx, dy = pix_scales_rad(hdr)            # rad/px
        return dx * ARCSEC_PER_RAD, dy * ARCSEC_PER_RAD

    
    def per_image_percentile_stretch(x, lo=1.0, hi=99.5):
        # x: 2D array → returns 2D array scaled into [0,1] by per-image percentiles
        a = np.asarray(x, float)
        lo_v = np.nanpercentile(a, lo)
        hi_v = np.nanpercentile(a, hi)
        y = (a - lo_v) / (hi_v - lo_v + 1e-12)
        return np.clip(y, 0.0, 1.0)


    # ------------------------------------------------

    # ---- 1) Load RAW ----
    I, Hraw = load_fits(raw_fits)
    dx, dy  = pix_scales_rad(Hraw)                     # [rad/px]
    asx, asy = dx * ARCSEC_PER_RAD, dy * ARCSEC_PER_RAD

    _print_wcs_info("RAW", Hraw)

    # ---- 2) Find/load T50kpc (native) and reproject to RAW grid ----
    if not t50_fits:
        d = os.path.dirname(raw_fits)
        base = Path(raw_fits).stem
        cands = sorted(glob.glob(os.path.join(d, f"{base}T50kpc*.fits"))) \
              + sorted(glob.glob(os.path.join(d, "*T50kpc*.fits")))
        t50_fits = cands[0] if cands else None
    assert t50_fits and os.path.exists(t50_fits), "Need a T50kpc FITS for the comparisons."

    T_native, Htgt = load_fits(t50_fits)
    _print_wcs_info("T50 native", Htgt)

    T = reproject_like(T_native, Htgt, Hraw)  # on RAW grid now
    print(f"[shapes] RAW={I.shape}  T50_native={T_native.shape}  T50_on_RAW={T.shape}")

    # Center difference (native centers; just for log)
    ra_r, de_r = _wcs_center(Hraw); ra_t, de_t = _wcs_center(Htgt)
    dRA = (ra_t - ra_r) * 3600.0 * np.cos(np.radians(de_r))
    dDE = (de_t - de_r) * 3600.0
    print(f"[centers] (T50 native) - (RAW): dRA≈{dRA:+.3f}\"  dDec≈{dDE:+.3f}\"")

    # ---- 3) Image-space route: I * G  (→ Jy/beam_target) ----
    ker = kernel_from_beams(Hraw, Htgt)

    # embed kernel into a canvas so its *size* is comparable in Row 1
    G_small = np.asarray(ker.array, float); G_small /= (np.nansum(G_small) + 1e-12)
    G_canvas_full = np.zeros_like(I, dtype=float)
    gy, gx = G_small.shape; cy, cx = np.array(I.shape)//2
    ys, xs = cy - gy//2, cx - gx//2
    G_canvas_full[ys:ys+gy, xs:xs+gx] = G_small

    IG_full = convolve_fft(I, ker, boundary='fill', fill_value=np.nan,
                           nan_treatment='interpolate', normalize_kernel=True,
                           psf_pad=True, fft_pad=True, allow_huge=True)
    IG_full *= (beam_solid_angle_sr(Htgt) / beam_solid_angle_sr(Hraw))  # to Jy/beam_target

    # ---- 4) uv-space route on the full frame (we'll crop later) ----
    F_I_full = fft2c(I, dx, dy)
    Cker_w   = beam_cov_world(Htgt) - beam_cov_world(Hraw)
    U_full, V_full = uv_grid(*I.shape, dx, dy)
    Wuv_full = W_from_Cker(U_full, V_full, Cker_w)     # DC=1
    FIW_full = F_I_full * Wuv_full
    IU_full  = ifft2c(FIW_full, dx, dy) * (beam_solid_angle_sr(Htgt) / beam_solid_angle_sr(Hraw))

    # ===================== CROP TO REQUESTED FOV (or T50 native FOV) =====================
    # Pixel scale of RAW image (arcsec/pixel)
    dxT_as, dyT_as = _arcsec_per_pix(Htgt)    # arcsec/px (native T)
    nyT, nxT       = T_native.shape

    if fov_arcmin is not None:
        # User override: crop to a fixed FOV (arcmin) on the RAW grid
        # Convert desired FOV to arcsec
        fov_as = float(fov_arcmin) * 60.0
        # Desired crop in RAW pixels
        nx_crop = int(round(fov_as / asx))
        ny_crop = int(round(fov_as / asy))
        if force_square:
            m = min(nx_crop, ny_crop)
            nx_crop, ny_crop = m, m
        # Do not exceed RAW frame
        nx_crop = min(nx_crop, I.shape[1])
        ny_crop = min(ny_crop, I.shape[0])
        print(f"[crop] using user FOV: {fov_arcmin:.3f} arcmin → RAW crop {nx_crop}×{ny_crop} px "
            f"(scale {asx:.4f}×{asy:.4f} ″/px)")
    else:
        # Default: crop to T50 native FOV (as before)
        fov_x_as = nxT * dxT_as
        fov_y_as = nyT * dyT_as
        nx_crop = int(round(fov_x_as / asx))
        ny_crop = int(round(fov_y_as / asy))
        nx_crop = min(nx_crop, I.shape[1])
        ny_crop = min(ny_crop, I.shape[0])
        print(f"[crop] target FOV from T50 native: {fov_x_as/60:.3f}×{fov_y_as/60:.3f} arcmin "
            f"→ RAW crop {nx_crop}×{ny_crop} px (at {asx:.4f}×{asy:.4f}″/px)")

    # ----- CENTER: geometric centre of RAW only -----
    cy, cx = (I.shape[0] - 1) / 2.0, (I.shape[1] - 1) / 2.0

    # Apply the header-centred crop to all planes
    I_crop   = center_crop_at(I,        ny_crop, nx_crop, cy, cx)
    IG_crop  = center_crop_at(IG_full,  ny_crop, nx_crop, cy, cx)
    IU_crop  = center_crop_at(IU_full,  ny_crop, nx_crop, cy, cx)
    T_crop   = center_crop_at(T,        ny_crop, nx_crop, cy, cx)
    G_canvas = center_crop_at(G_canvas_full, ny_crop, nx_crop, cy, cx)

    # Rebuild UV quantities on the CROPPED shape (important for rows 2 and 4)
    F_I   = fft2c(I_crop, dx, dy)
    U, V  = uv_grid(ny_crop, nx_crop, dx, dy)
    Wuv   = W_from_Cker(U, V, Cker_w)
    FIW   = F_I * Wuv

    F_IG  = fft2c(IG_crop, dx, dy)
    F_T   = fft2c(T_crop,  dx, dy)

    A_I   = np.abs(F_I)
    A_IG  = np.abs(F_IG)
    A_IW  = np.abs(FIW)
    A_T   = np.abs(F_T)
    
    # --- radial UV profiles & phase/coherence ---
    prof = _radial_profiles(U, V, A_T, A_IG, A_IW, F_T, F_IG, nbins=28)
    
    # ===== Separate figure: UV radial profiles (amplitudes only) =====
    fig_uv, axr = plt.subplots(figsize=(5.8, 6.0))
    x = prof["r_kla"]
    axr.plot(x, prof["med_T"], linestyle="--", linewidth=1.8, color="C0", zorder=3, label=r"$|\mathcal{F}(T)|$ (median)")
    axr.plot(x, prof["med_RT"], linewidth=1.4, color="C1", zorder=1, label=r"$|\mathcal{F}(I)\,W|$ (median)")
    axr.plot(x, prof["med_IG"], linewidth=1.4, color="C2", zorder=2, label=r"$|\mathcal{F}(I*G)|$ (median)")
    axr.set_xscale("log"); axr.set_yscale("log")
    axr.set_xlabel(r"baseline $r=\sqrt{u^2+v^2}$ (k$\lambda$)")
    axr.set_ylabel("amplitude (Jy)")
    y_top = 1.15 * np.nanmax([np.nanmax(prof["med_T"]),
                            np.nanmax(prof["med_RT"]),
                            np.nanmax(prof["med_IG"])])
    axr.set_ylim(1e-15, y_top)
    axr.grid(True, which="both", alpha=0.25)
    axr.legend(frameon=False)
    uv_pdf = out_pdf.replace(".pdf", "_uv_profiles.pdf")
    fig_uv.savefig(uv_pdf, dpi=250, bbox_inches="tight"); plt.close(fig_uv)
    print(f"[ok] wrote {uv_pdf}")
    
    # ===== Separate figure: phase stats =====
    fig_ph, axb = plt.subplots(figsize=(6.8, 3.6))
    deg = 180/np.pi
    axb.plot(x, prof["mean_abs_dphi"]*deg,
            label=r"$\langle|\Delta\phi(T,\;I*G)|\rangle$")
    axb.plot(x, prof["sigma_phi"]*deg, linestyle="--",
            label=r"$\sigma\!\left(\Delta\phi(T,\;I*G)\right)$")
    axb.set_xscale("log")
    axb.set_xlabel("baseline (kλ)")
    axb.set_ylabel("phase (deg)")
    y_top = 1.15*np.nanmax([
        np.nanmax(prof["mean_abs_dphi"]*deg),
        np.nanmax(prof["sigma_phi"]*deg)
    ])
    axb.set_ylim(0, y_top)
    axb.grid(True, which="both", alpha=0.25)
    axb.legend(frameon=False)
    ph_pdf = out_pdf.replace(".pdf", "_phase_stats.pdf")
    fig_ph.savefig(ph_pdf, dpi=250, bbox_inches="tight"); plt.close(fig_ph)
    print(f"[ok] wrote {ph_pdf}")


    # ---------------- diagnostics: alignment & residuals (on CROPPED arrays) ----------------
    dy_IG_T, dx_IG_T   = _phase_xcorr_shift(IG_crop, T_crop)
    dy_IU_T, dx_IU_T   = _phase_xcorr_shift(IU_crop, T_crop)
    dy_IG_IU, dx_IG_IU = _phase_xcorr_shift(IG_crop, IU_crop)

    def _nanrms(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if not np.any(m): return np.nan
        d = a[m] - b[m]
        return float(np.sqrt(np.mean(d*d)))

    rms_IG_T  = _nanrms(IG_crop, T_crop)
    rms_IU_T  = _nanrms(IU_crop, T_crop)
    rms_IG_IU = _nanrms(IG_crop, IU_crop)

    print("[align] phase-XCorr shifts (dy, dx) in pixels (CROPPED):")
    print(f"        IG vs T  : dy={dy_IG_T:+.3f}, dx={dx_IG_T:+.3f}")
    print(f"        IU vs T  : dy={dy_IU_T:+.3f}, dx={dx_IU_T:+.3f}")
    print(f"        IG vs IU : dy={dy_IG_IU:+.3f}, dx={dx_IG_IU:+.3f}")
    print("[resid] RMS (Jy/beam) without any shift (CROPPED):")
    print(f"        IG − T   : {rms_IG_T:.6g}")
    print(f"        IU − T   : {rms_IU_T:.6g}")
    print(f"        IG − IU  : {rms_IG_IU:.6g}")
    
    prof.update({
        "rms_IG_T":  float(rms_IG_T),
        "rms_IU_T":  float(rms_IU_T),
        "rms_IG_IU": float(rms_IG_IU),
    })


    # ---------------- shared normalisations (per row) ----------------
    # Row 1 (image plane)
    omega_raw = beam_solid_angle_sr(Hraw)
    omega_tgt = beam_solid_angle_sr(Htgt)

    I_disp = I_crop * (omega_raw / omega_tgt)   # RAW → Jy/beam_tgt for display
    row1_images = [I_disp, IG_crop, IU_crop, T_crop, G_canvas]

    if per_image_stretch:
        norm_row1 = mcolors.Normalize(vmin=0.0, vmax=1.0)   # each panel stretched into [0,1]
    else:
        row1_stack = np.concatenate([x[np.isfinite(x)].ravel() for x in row1_images])
        norm_row1  = mcolors.Normalize(
            vmin=float(np.nanpercentile(row1_stack, 1.0)),
            vmax=float(np.nanpercentile(row1_stack, 99.5)),
        )

    # Row 2 (UV products, log scale); scale W to match the UV range for visual comparison
    logA_I  = np.log10(A_I  + 1e-12)
    logA_IG = np.log10(A_IG + 1e-12)
    logA_IW = np.log10(A_IW + 1e-12)
    logA_T  = np.log10(A_T  + 1e-12)
    uv_noW_stack = np.concatenate([x[np.isfinite(x)].ravel()
                                for x in (logA_I, logA_IG, logA_IW, logA_T)])
    if per_image_stretch:
        norm_row2 = mcolors.Normalize(vmin=0.0, vmax=1.0)
    else:
        vmin_uv = float(np.nanpercentile(uv_noW_stack, 1.0))
        vmax_uv = float(np.nanpercentile(uv_noW_stack, 99.5))
        norm_row2 = mcolors.Normalize(vmin=vmin_uv, vmax=vmax_uv)


    p99W   = float(np.nanpercentile(np.log10(Wuv + 1e-12), 99.5))
    sW     = 10.0**(vmax_uv - p99W)            # scale W so its 99.5% ≈ UV vmax
    logW_scaled = np.log10(sW * Wuv + 1e-12)
    print(f"[uv scale] W scaled by sW={sW:.6g} so its 99.5% matches UV vmax")

    # Row 3: (optional) gain-match IG/IU to T for visual residuals
    gIG, sIG = robust_gain_match(T_crop, IG_crop, nsig=3.0, roi=0.85)
    gIU, sIU = robust_gain_match(T_crop, IU_crop, nsig=3.0, roi=0.85)
    print(f"[gain] g* (IG→T) = {gIG:.6f}  (N={sIG['Npix']}, rms after={sIG['rms']:.6g})")
    print(f"[gain] g* (IU→T) = {gIU:.6f}  (N={sIU['Npix']}, rms after={sIU['rms']:.6g})")
    IGm, IUm = gIG * IG_crop, gIU * IU_crop

    R12 = IGm - IUm
    R13 = IGm - T_crop
    R23 = IUm - T_crop
    resid_abs_stack = np.concatenate([np.abs(r)[np.isfinite(r)].ravel() for r in (R12, R13, R23)])
    rmax = float(np.nanpercentile(resid_abs_stack, 99.0))
    norm_row3 = TwoSlopeNorm(vmin=-rmax, vcenter=0.0, vmax=rmax)

    # Row 4: UV residuals (log|Δ amplitude|)
    D12 = np.abs(A_IG - A_IW)
    D13 = np.abs(A_IG - A_T)
    D23 = np.abs(A_IW - A_T)
    logD12, logD13, logD23 = [np.log10(x + 1e-12) for x in (D12, D13, D23)]
    if per_image_stretch:
        norm_row4 = mcolors.Normalize(vmin=0.0, vmax=1.0)
    else:
        row4_stack = np.concatenate([x[np.isfinite(x)].ravel() for x in (logD12, logD13, logD23)])
        norm_row4  = mcolors.Normalize(
            vmin=float(np.nanpercentile(row4_stack, 1.0)),
            vmax=float(np.nanpercentile(row4_stack, 99.5)),
        )
    
    # Wrapper: apply per-image percentile stretch only to non-negative maps we display.
    def _maybe_stretch(arr):
        if per_image_stretch:
            return per_image_percentile_stretch(arr, percentile_lo, percentile_hi)
        return arr

    # ------------------------------- FIGURE A: TOP (Rows 1–2) -------------------------------

    stem    = Path(out_pdf).with_suffix("")
    out_top = f"{stem}_top12.pdf"
    ncols   = 5

    # Shorter and tighter
    fig_top = plt.figure(figsize=(10.4, 5.0), constrained_layout=False)
    gs_top  = fig_top.add_gridspec(
        2, ncols,
        width_ratios=[1,1,1,1,1],
        height_ratios=[1,1],
        wspace=0.06,
        hspace=0.06,            # tighter gap between the two rows
    )
    plt.subplots_adjust(left=0.035, right=0.955, top=0.955, bottom=0.055)  # leave space at right for cbar axes

    # ----- Row 1
    row1_titles = [
        "I (RAW)",
        r"$I * G$",
        r"$\mathcal{F}^{-1}\{\mathcal{F}(I)\,W\}$",
        "T50kpc",
        r"Kernel $G$",
    ]
    row1_axes = []
    for j, img in enumerate([I_crop, IG_crop, IU_crop, T_crop, G_canvas]):
        ax = fig_top.add_subplot(gs_top[0, j]); row1_axes.append(ax)
        im = ax.imshow(_maybe_stretch(img), origin='lower', cmap='viridis', norm=norm_row1)
        ax.set_axis_off()
        ax.set_title(row1_titles[j], fontsize=10)

    # Top-row colorbar in its own axis (prevents overlap)
    cax1 = fig_top.add_axes([0.962, 0.55, 0.022, 0.36])   # [left, bottom, width, height] in figure coords
    cbar1 = fig_top.colorbar(ScalarMappable(norm=norm_row1, cmap='viridis'), cax=cax1)
    cbar1.set_label('brightness [Jy/beam$_{\\mathrm{tgt}}$]\n(kernel $G$: unitless)', labelpad=10)
    cbar1.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    sf1 = ScalarFormatter(useMathText=True); sf1.set_powerlimits((-2, 3))
    cbar1.ax.yaxis.set_major_formatter(sf1)
    cbar1.ax.tick_params(labelsize=9, length=3, width=0.8)

    # ----- Row 2
    row2_titles = [
        r'$\log_{10}\,|\mathcal{F}(I)|$',
        r'$\log_{10}\,|\mathcal{F}(I*G)|$',
        r'$\log_{10}\,|\mathcal{F}(I)\,W|$',
        r'$\log_{10}\,|\mathcal{F}(T_{50})|$',
        r'$\log_{10}\,\tilde{W}(u,v)$ (scaled)',
    ]
    row2_axes = []
    for j, arr in enumerate([logA_I, logA_IG, logA_IW, logA_T, logW_scaled]):
        ax = fig_top.add_subplot(gs_top[1, j]); row2_axes.append(ax)
        ax.imshow(_maybe_stretch(arr), origin='lower', cmap='viridis', norm=norm_row2)
        ax.set_axis_off()
        ax.set_title(row2_titles[j], fontsize=10)

    # Bottom-row colorbar in its own axis (separate from the top one)
    cax2 = fig_top.add_axes([0.962, 0.10, 0.022, 0.36])
    cbar2 = fig_top.colorbar(ScalarMappable(norm=norm_row2, cmap='viridis'), cax=cax2)
    cbar2.set_label(r'shared $\log_{10}$ scale (|$\mathcal{F}$|)'
                    '\n' r'[Jy·sr / beam$_{\mathrm{tgt}}$]', labelpad=10)
    cbar2.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    sf2 = ScalarFormatter(useMathText=True); sf2.set_powerlimits((-2, 3))
    cbar2.ax.yaxis.set_major_formatter(sf2)
    cbar2.ax.tick_params(labelsize=9, length=3, width=0.8)

    fig_top.savefig(out_top, dpi=250, bbox_inches='tight')
    plt.close(fig_top)
    print(f"[ok] wrote {out_top}")




    # ------------------------------- FIGURE B: BOTTOM (Rows 3–4) -------------------------------
    out_bot = f"{stem}_bot34.pdf"
    fig_bot = plt.figure(figsize=(10.8, 7.9), constrained_layout=False)
    gs_bot  = fig_bot.add_gridspec(2, 3, width_ratios=[1,1,1],
                                height_ratios=[1,1], wspace=0.06, hspace=0.14)
    plt.subplots_adjust(left=0.055, right=0.995, top=0.96, bottom=0.078)

    # Row 3 (image residuals) — use Fourier notation, not "uv route"
    row3_titles = [
        r"$(I*G) - \mathcal{F}^{-1}\{\mathcal{F}(I)\,W\}$",
        r"$(I*G) - T_{50}$",
        r"$\mathcal{F}^{-1}\{\mathcal{F}(I)\,W\} - T_{50}$",
    ]
    row3_axes = []
    for j, R in enumerate([R12, R13, R23]):
        ax = fig_bot.add_subplot(gs_bot[0, j]); row3_axes.append(ax)
        ax.imshow(R, origin='lower', cmap='coolwarm', norm=norm_row3)
        ax.set_axis_off(); ax.set_title(row3_titles[j], fontsize=11)
    fig_bot.colorbar(ScalarMappable(norm=norm_row3, cmap='coolwarm'),
                    ax=row3_axes, fraction=0.03, pad=0.01,
                    label=r'residual [Jy / beam$_{\rm tgt}$]')


    # Row 4
    row4_titles = [
        r'$\log_{10}\,\left|\,|\mathcal{F}(I*G)|-|\mathcal{F}(I)\,W|\,\right|$',
        r'$\log_{10}\,\left|\,|\mathcal{F}(I*G)|-|\mathcal{F}(T_{50})|\,\right|$',
        r'$\log_{10}\,\left|\,|\mathcal{F}(I)\,W|-|\mathcal{F}(T_{50})|\,\right|$',
    ]
    row4_axes = []
    for j, arr in enumerate([logD12, logD13, logD23]):
        ax = fig_bot.add_subplot(gs_bot[1, j]); row4_axes.append(ax)
        ax.imshow(_maybe_stretch(arr), origin='lower', cmap='viridis', norm=norm_row4)
        ax.set_axis_off(); ax.set_title(row4_titles[j], fontsize=11)
        ax.set_aspect('equal')
    fig_bot.colorbar(ScalarMappable(norm=norm_row4, cmap='viridis'),
                    ax=row4_axes, fraction=0.03, pad=0.01,
                    label=r'$\log_{10}$ |Δ amplitude|  [Jy·sr / beam$_{\rm tgt}$]')


    fig_bot.savefig(out_bot, dpi=250, bbox_inches='tight')
    plt.close(fig_bot)
    print(f"[ok] wrote {out_bot}")


    # ---- Row 1 meta (cropped) ----
    fovx_arcmin = nx_crop * asx / 60.0
    fovy_arcmin = ny_crop * asy / 60.0
    print("[Row1 meta — CROPPED to T50 FOV]")
    print(f"  I (RAW):                 {ny_crop}×{nx_crop} px — {asx:.4f}×{asy:.4f} arcsec/px — "
          f"FOV≈{fovx_arcmin:.3f}×{fovy_arcmin:.3f} arcmin")
    print(f"  I * G (image space):     {ny_crop}×{nx_crop} px — {asx:.4f}×{asy:.4f} arcsec/px")
    print(f"  $\\mathcal{{F}}^{{-1}}[\\mathcal{{F}}(I)\\,W]$: {ny_crop}×{nx_crop} px — {asx:.4f}×{asy:.4f} arcsec/px")
    print(f"  T50kpc (from FITS):      {ny_crop}×{nx_crop} px — {asx:.4f}×{asy:.4f} arcsec/px")
    return prof


# ------------------------------ CLI example ------------------------------
if __name__ == "__main__":
    ROOT = "/home/sysadmin/Scripts/scatter_galaxies/data/PSZ2/fits"
    pairs = pick_random_pairs(ROOT, n=3, seed=42)

    profiles = []
    for i, (RAW, T50) in enumerate(pairs, 1):
        stem = Path(RAW).stem
        out  = f"quicklook_{i:02d}_{stem}.pdf"
        prof = make_rt_quicklook(
            RAW, t50_fits=T50, out_pdf=out,
            fov_arcmin=20.0, per_image_stretch=False,
            percentile_lo=1.0, percentile_hi=99.5,
        )
        profiles.append(prof)
        print(f"[batch] wrote {out}  ←  {RAW}  |  {T50}")

    # If multiple, make stacked summary plots (UV amplitudes + phase statistics)
    if len(profiles) > 1:
        rms_IG_T  = np.array([p["rms_IG_T"]  for p in profiles], float)
        rms_IU_T  = np.array([p["rms_IU_T"]  for p in profiles], float)
        rms_IG_IU = np.array([p["rms_IG_IU"] for p in profiles], float)

        print("[residuals across sources] RMS (Jy/beam):")
        print(f"  IG−T   : mean={np.nanmean(rms_IG_T):.3g}  std={np.nanstd(rms_IG_T):.3g}")
        print(f"  IU−T   : mean={np.nanmean(rms_IU_T):.3g}  std={np.nanstd(rms_IU_T):.3g}")
        print(f"  IG−IU  : mean={np.nanmean(rms_IG_IU):.3g} std={np.nanstd(rms_IG_IU):.3g}")

        
        r = profiles[0]["r_kla"]
        def stack(key):
            return np.vstack([p[key] for p in profiles])

        # prep stats
        Y_T  = stack("med_T");   mu_T,  sd_T  = summarize(Y_T)
        Y_RT = stack("med_RT");  mu_RT, sd_RT = summarize(Y_RT)
        Y_IG = stack("med_IG");  mu_IG, sd_IG = summarize(Y_IG)

        # --- (A) UV amplitudes: medians with ±1σ bands ---
        fig, ax = plt.subplots(figsize=(6.6, 4.6))

        # draw fills first (behind lines)
        ax.fill_between(r, mu_RT - sd_RT, mu_RT + sd_RT, alpha=0.18, color="C1", zorder=1)
        ax.fill_between(r, mu_IG - sd_IG, mu_IG + sd_IG, alpha=0.18, color="C2", zorder=1)
        ax.fill_between(r, mu_T  - sd_T,  mu_T  + sd_T,  alpha=0.12, color="C0", zorder=1)

        # draw RT & IG lines
        ax.plot(r, mu_RT, color="C1", linewidth=1.6, label=r"$|\mathcal{F}(I)\,W|$ (median)", zorder=5)
        ax.plot(r, mu_IG, color="C2", linewidth=1.6, label=r"$|\mathcal{F}(I*G)|$ (median)", zorder=6)

        # draw T last, blue dashed on top
        ax.plot(r, mu_T,  color="C0", linewidth=1.8, linestyle="--",
                label=r"$|\mathcal{F}(T)|$ (median)", zorder=10)

        # axes/format
        ymax = np.nanmax([np.nanmax(mu_T+sd_T), np.nanmax(mu_RT+sd_RT), np.nanmax(mu_IG+sd_IG)])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(np.nanmin(r), np.nanmax(r))
        ax.set_ylim(1e-15, 1.15*ymax)
        ax.set_xlabel(r"baseline $r=\sqrt{u^2+v^2}$ (k$\lambda$)")
        ax.set_ylabel("amplitude (Jy)")
        ax.grid(True, which="both", alpha=0.25)
        # legend with T first
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(r"$|\mathcal{F}(T)|$ (median)"),
                labels.index(r"$|\mathcal{F}(I)\,W|$ (median)"),
                labels.index(r"$|\mathcal{F}(I*G)|$ (median)")]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False)

        fig.tight_layout()
        fig.savefig("quicklook_uv_profiles_stack.pdf", dpi=250, bbox_inches="tight")
        plt.close(fig)
        print("[ok] wrote quicklook_uv_profiles_stack.pdf (medians with ±1σ bands)")


        # --- (B) Phase statistics: medians with ±1σ bands (in degrees) ---
        deg = 180.0/np.pi
        fig, ax = plt.subplots(figsize=(6.6, 4.0))

        # ⟨|Δφ|⟩ : blue solid
        Y_mean = stack("mean_abs_dphi") * deg
        mu_m, sd_m = summarize(Y_mean)
        ax.plot(r, mu_m, linewidth=1.8, label=r"$\langle|\Delta\phi(T,\;I*G)|\rangle$ (median)", zorder=2)
        ax.fill_between(r, mu_m - sd_m, mu_m + sd_m, alpha=0.18, zorder=1)

        # σ(Δφ): orange dashed
        Y_sig = stack("sigma_phi") * deg
        mu_s, sd_s = summarize(Y_sig)
        ax.plot(r, mu_s, linestyle="--", linewidth=1.8,
                label=r"$\sigma\!\left(\Delta\phi(T,\;I*G)\right)$ (median)", zorder=3)
        ax.fill_between(r, mu_s - sd_s, mu_s + sd_s, alpha=0.18, zorder=1)

        ax.set_xscale("log"); ax.set_xlim(np.nanmin(r), np.nanmax(r))
        y_candidates = np.hstack([
            (mu_m + sd_m)[np.isfinite(mu_m + sd_m)],
            (mu_s + sd_s)[np.isfinite(mu_s + sd_s)],
        ])
        y_top = 1.10 * np.nanpercentile(y_candidates, 98)   # robust cap
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel("baseline (kλ)")
        ax.set_ylabel("phase (deg)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig("quicklook_phase_stats_stack.pdf", dpi=250, bbox_inches="tight")
        plt.close(fig)
        print("[ok] wrote quicklook_phase_stats_stack.pdf (medians with ±1σ bands)")



# Smallest redshift
# PSZ2G136.64-25.03: 0.016

# Largest redshift
# PSZ2G152.47+42.11: 0.9