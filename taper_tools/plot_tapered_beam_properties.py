import os, io, csv, glob, random, numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import zoom as _zoom, shift as _imgshift
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for servers
import matplotlib.pyplot as plt

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

DEBUG_REPROJECT = True  # set False to silence

def pick_random_pairs(root_dir, n=5, seed=42):
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
        T50 = os.path.join(d, f"{base}T50kpc.fits")
        if os.path.exists(raw) and os.path.exists(T50):
            pairs.append((raw, T50))
    if not pairs:
        raise FileNotFoundError(f"No (RAW, T50kpc) pairs found under {root_dir}")
    rng = random.Random(seed)
    if len(pairs) <= n:
        return pairs
    return rng.sample(pairs, n)

# ------------------------------ WCS helpers ------------------------------

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
    J = _cd_matrix_rad(hdr)
    ang = float(np.degrees(np.arctan2(J[1,0], J[0,0])))  # direction of column 1
    ra, dec = _wcs_center(hdr)
    print(f"[WCS] {tag}: {int(hdr['NAXIS1'])}×{int(hdr['NAXIS2'])} px | "
          f"{asx:.6f}×{asy:.6f} arcsec/px | rot~{ang:+.3f} deg | "
          f"center RA={ra:.6f} deg, Dec={dec:.6f} deg")
    
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


def reproject_like(arr, src_hdr, dst_hdr):
    """
    Put a 2-D image on the pixel grid defined by dst_hdr.

    Priority:
      1) If 'reproject' is available and both headers have valid celestial WCS,
         use bilinear reproject_interp to the exact output shape.
      2) Otherwise: resize to dst shape with scipy.ndimage.zoom, then (if WCS
         exists) align image centers via a subpixel shift. This guarantees the
         returned array has shape (dst_hdr['NAXIS2'], dst_hdr['NAXIS1']).
         
    Returns reprojected array as float, or None if inputs are invalid.
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

def circular_cov_kpc(z, fwhm_kpc=50.0):
    """ Return a circular 2×2 covariance matrix in world coords (radians^2)
    corresponding to a Gaussian with FWHM=fwhm_kpc at redshift z (kpc physical). """
    if z is None or not np.isfinite(z) or z <= 0:
        return None
    fwhm_kpc = float(fwhm_kpc)
    if fwhm_kpc <= 0:
        raise ValueError("fwhm_kpc must be positive")
    # angular diameter distance DA in kpc
    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_rad = (fwhm_kpc / DA_kpc)                    # small-angle in radians (unitless number)
    sigma = theta_rad / (2.0*np.sqrt(2.0*np.log(2.0)))
    #sigma = theta_rad
    sigma2 = float(sigma**2)
    return np.array([[sigma2, 0.0],[0.0, sigma2]], float)

def load_z_table(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Redshift table not found: {csv_path}")

    # Read the file and keep only non-empty lines; also handle UTF-8 BOM.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()

    # If the filesystem lies about size, raw may still be empty; show a helpful preview.
    if not raw.strip():
        raise ValueError(
            "Redshift table appears empty after reading: "
            f"{csv_path} (first 80 bytes repr={raw[:80]!r})"
        )

    # Drop blank lines and comment lines if any
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

# ------------------------------ Main function ------------------------------

def get_beam_info(raw_fits, T50_fits=None, out_pdf="one_source_quicklook_with_residuals.png",
                      z=None, fwhm_kpc=50.0):
    
    # - Load RAW ----
    I, Hraw = load_fits(raw_fits)
    bmaj_RAW = float(Hraw['BMAJ'])
    bmin_RAW = float(Hraw['BMIN'])
    q_tgt_RAW = min(bmaj_RAW, bmin_RAW) / max(bmaj_RAW, bmin_RAW) if (bmaj_RAW > 0 and bmin_RAW > 0) else np.nan
    e_tgt_RAW = 1.0 - q_tgt_RAW
    pa_tgt_RAW = float(Hraw.get('BPA', np.nan))
    _print_wcs_info("RAW", Hraw)
    
    T_native, Htgt = load_fits(T50_fits)
    _print_wcs_info("T50 native", Htgt)
    # beam ellipticity from header (units cancel)
    bmaj_T50 = float(Htgt['BMAJ'])
    bmin_T50 = float(Htgt['BMIN'])
    q_tgt_T50 = min(bmaj_T50, bmin_T50) / max(bmaj_T50, bmin_T50) if (bmaj_T50 > 0 and bmin_T50 > 0) else np.nan
    e_tgt_T50 = 1.0 - q_tgt_T50
    pa_tgt_T50 = float(Htgt.get('BPA', np.nan))
    print(f"[beam] T50 native: BMAJ={bmaj_T50:.3f} deg, BMIN={bmin_T50:.3f} deg, q={q_tgt_T50:.3f}, e={e_tgt_T50:.3f}, PA={pa_tgt_T50:.1f} deg")

    # Find blurring kernels
    C_raw_w   = beam_cov_world(Hraw)  # RAW beam covariance in world coords
    Cker_circ = circular_cov_kpc(z, fwhm_kpc=fwhm_kpc)           # world-cov of the *kernel*, circular
    omega_tgt_raw = beam_solid_angle_sr(Hraw)  # float(2.0*np.pi*np.sqrt(max(0.0, np.linalg.det(C_raw_w))))  # sr
    omega_rt_circ = float(2.0*np.pi*np.sqrt(max(0.0, np.linalg.det(Cker_circ))))  # sr
    omega_tgt_circ = float(2.0*np.pi*np.sqrt(max(0.0, np.linalg.det(C_raw_w + Cker_circ)))) # solid angle of RAW + circular 50 kpc

    C_tgt_hdr = beam_cov_world(Htgt)  # T50 beam covariance in world coords
    Cker_cheat = C_tgt_hdr - C_raw_w  # "cheat" kernel covariance in world coords
    omega_rt_cheat = float(2.0*np.pi*np.sqrt(max(0.0, np.linalg.det(Cker_cheat))))  # Solid angle of "cheat" kernel
    omega_tgt_T50 = float(2.0*np.pi*np.sqrt(max(0.0, np.linalg.det(C_tgt_hdr))))  # solid angle of T50 kernel

    omega_tgt_T50_minus_omega_raw = omega_tgt_T50 - omega_tgt_raw # Should be close to zero
    omega_circ_diff = omega_tgt_circ - omega_rt_circ - omega_tgt_raw
    omega_cheat_diff = omega_tgt_T50 - omega_rt_cheat - omega_tgt_raw

    # --- Major/minor axis ratios: cheat kernel vs. rt (circular) kernel ---
    beam_major_ratio = float(bmaj_T50 / (bmaj_RAW + 1e-30))
    beam_minor_ratio = float(bmin_T50 / (bmin_RAW + 1e-30))
    beam_major_diff = float(bmaj_T50 - bmaj_RAW)
    beam_minor_diff = float(bmin_T50 - bmin_RAW)

    # Scaled with angular distance
    bmaj_T50_scaled = bmaj_T50 * (COSMO.angular_diameter_distance(z).to_value(u.Mpc) if (z is not None and np.isfinite(z) and z > 0) else np.nan)
    bmin_T50_scaled = bmin_T50 * (COSMO.angular_diameter_distance(z).to_value(u.Mpc) if (z is not None and np.isfinite(z) and z > 0) else np.nan)
    bmaj_RAW_scaled = bmaj_RAW * (COSMO.angular_diameter_distance(z).to_value(u.Mpc) if (z is not None and np.isfinite(z) and z > 0) else np.nan)
    bmin_RAW_scaled = bmin_RAW * (COSMO.angular_diameter_distance(z).to_value(u.Mpc) if (z is not None and np.isfinite(z) and z > 0) else np.nan)

    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    bmaj_RAW_phys_kpc = DA_kpc * np.deg2rad(bmaj_RAW) 
    bmaj_T50_phys_kpc = DA_kpc * np.deg2rad(bmaj_T50)
    bmin_RAW_phys_kpc = DA_kpc * np.deg2rad(bmin_RAW)
    bmin_T50_phys_kpc = DA_kpc * np.deg2rad(bmin_T50)

    # Physical scale for the omegas
    scale_phys = DA_kpc**2  # kpc^2 per steradian
    omega_tgt_raw_phys_kpc2 = omega_tgt_raw * scale_phys
    omega_rt_circ_phys_kpc2 = omega_rt_circ * scale_phys
    omega_tgt_circ_phys_kpc2 = omega_tgt_circ * scale_phys
    omega_rt_cheat_phys_kpc2 = omega_rt_cheat * scale_phys    

    # eigenvalues → sigmas in image plane [radians]
    lam_cheat = np.maximum(np.linalg.eigvalsh(Cker_cheat), 0.0)
    lam_circ  = np.maximum(np.linalg.eigvalsh(Cker_circ),  0.0)
    sig_cheat = np.sqrt(lam_cheat)  # [rad]
    sig_circ  = np.sqrt(lam_circ)   # [rad] (degenerate, but compute anyway)

    # sort so [minor, major]
    sig_cheat.sort()
    sig_circ.sort()
    ker_major_ratio_img = float(sig_cheat[1] / (sig_circ[1] + 1e-30))
    ker_minor_ratio_img = float(sig_cheat[0] / (sig_circ[0] + 1e-30))

    # OPTIONAL: same in uv space (sigma_uv = 1/(2π sigma_img))
    sig_uv_cheat = 1.0 / (2.0*np.pi*np.maximum(sig_cheat, 1e-30))
    sig_uv_circ  = 1.0 / (2.0*np.pi*np.maximum(sig_circ,  1e-30))
    ker_major_ratio_uv = float(sig_uv_cheat[1] / (sig_uv_circ[1] + 1e-30))
    ker_minor_ratio_uv = float(sig_uv_cheat[0] / (sig_uv_circ[0] + 1e-30))

    
    C_pred = C_raw_w + Cker_circ               # modelled target from RAW + circular 50 kpc
    Delta = C_tgt_hdr - C_pred                 # 2x2 residual
    # Frobenius norm normalized by target size (scale-free)
    cov_mismatch_frob = float(np.linalg.norm(Delta, ord='fro') /
                            (np.linalg.norm(C_tgt_hdr, ord='fro') + 1e-30))
    # Signed trace mismatch (area-ish) as a fraction of target trace
    cov_mismatch_trace = float(np.trace(Delta) / (np.trace(C_tgt_hdr) + 1e-30)) 
    
    # Decompose ΔC in the target's principal-axis frame
    wt, Rt = np.linalg.eigh(C_tgt_hdr)
    RtT = Rt.T
    Delta_t = RtT @ Delta @ Rt  # rotate into tgt eigenbasis

    # components: area (trace), ellipticity (eigenvalue mismatch), PA (off-diagonal)
    frac_area_mis = float(np.trace(Delta) / (np.trace(C_tgt_hdr) + 1e-30))
    ellip_mis = float((Delta_t[0,0] - Delta_t[1,1]) /
                    ( (wt[0] + wt[1]) + 1e-30 ))
    pa_mis = float(Delta_t[0,1] / (0.5*(wt[0] + wt[1]) + 1e-30))

    
    # --- RAW/T50 beam PAs and misalignment (0–90 deg) ---
    pa_raw = float(Hraw.get('BPA', np.nan))
    pa_tgt = float(Htgt.get('BPA', np.nan))
    def _dpa(a, b):
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.nan
        d = (a - b + 90.0) % 180.0 - 90.0   # wrap to [-90, +90)
        return abs(d)
    dpa_deg = _dpa(pa_tgt, pa_raw)
    
    # after computing Cker_circ, C_raw_w, C_tgt_hdr, Delta, etc., add:
    taper_leverage = float(np.linalg.norm(Cker_circ, ord='fro') / (np.linalg.norm(C_raw_w, ord='fro') + 1e-30))
    pred_size = float(np.linalg.norm(C_raw_w + Cker_circ, ord='fro'))

    # ---- infer uv-kernel from covariance difference ----
    # eigen-decomposition of image-plane "cheat" kernel (radians^2)
    w, V = np.linalg.eigh(Cker_cheat)
    w = np.maximum(w, 0.0)  # clip tiny negatives from noise/roundoff
    sig_img = np.sqrt(w)     # [radians] along principal axes

    # image ↔ uv Fourier relation: sigma_uv = 1 / (2π * sigma_img)
    sig_uv = 1.0 / (2.0*np.pi*np.maximum(sig_img, 1e-30))  # [wavelengths]

    # FWHM in uv (wavelengths) along major/minor; convert to kλ
    fwhm_uv = (2.0*np.sqrt(2.0*np.log(2.0))) * sig_uv / 1e3  # [kλ]
    fwhm_uv.sort()  # ascending: [minor, major]
    uvker_fwhm_minor_kl, uvker_fwhm_major_kl = float(fwhm_uv[0]), float(fwhm_uv[1]) 
    uvker_fwhm_geo_kl = float(np.sqrt(uvker_fwhm_minor_kl * uvker_fwhm_major_kl))
    uvker_axis_ratio = float(uvker_fwhm_minor_kl / max(uvker_fwhm_major_kl, 1e-30))

    # theoretical circular 50 kpc kernel in uv (for comparison)
    w_circ, _ = np.linalg.eigh(Cker_circ)
    sig_img_circ = np.sqrt(np.maximum(w_circ, 0.0))
    sig_uv_circ   = 1.0 / (2.0*np.pi*np.maximum(sig_img_circ, 1e-30))
    fwhm_uv_circ_kl = float((2.0*np.sqrt(2.0*np.log(2.0))) * np.mean(sig_uv_circ) / 1e3)  # scalar

    # size mismatch: geometric-mean FWHM vs. circular expectation
    uvker_size_mismatch = float(uvker_fwhm_geo_kl / (fwhm_uv_circ_kl + 1e-30))
    
    # observed restoring beam (geometric-mean FWHM) in radians
    theta_obs_geo_rad = np.sqrt(bmaj_T50 * bmin_T50) * (np.pi / 180.0)

    # predicted 50 kpc target beam from model C_pred = C_raw_w + C_50kpc
    w_pred = np.maximum(np.linalg.eigvalsh(C_pred), 0.0)        # eigenvalues [rad^2]
    sigma_pred = np.sqrt(w_pred)                                # [rad]
    fwhm_pred = (2.0 * np.sqrt(2.0 * np.log(2.0))) * sigma_pred # [rad]
    theta_pred_geo_rad = float(np.sqrt(fwhm_pred[0] * fwhm_pred[1]))

    # convert to physical kpc
    obs_phys_kpc  = float(DA_kpc * theta_obs_geo_rad)
    pred_phys_kpc = float(DA_kpc * theta_pred_geo_rad)
    resid_phys_kpc = obs_phys_kpc - pred_phys_kpc

    # Pixel sizes in radians
    dx_rad, dy_rad = pix_scales_rad(Hraw)
    pix_area_sr = float(dx_rad * dy_rad)  # pixel area in steradians
    npix = float(I.size)
  
    return {
        "z": float(z),
        "cosmo_distance_Mpc": DA_kpc / 1e3 if (z is not None and np.isfinite(z) and z > 0) else np.nan,
        "omega_rt_circ": float(omega_rt_circ),
        "omega_rt_cheat": float(omega_rt_cheat),
        "omega_tgt_circ": float(omega_tgt_circ),
        "omega_tgt_raw": float(omega_tgt_raw),
        "omega_tgt_T50": float(omega_tgt_T50),
        "omega_circ_diff": float(omega_circ_diff),
        "omega_cheat_diff": float(omega_cheat_diff),
        "omega_tgt_T50_minus_omega_raw": float(omega_tgt_T50_minus_omega_raw),
        "omega_rt_circ_phys_kpc2": float(omega_rt_circ_phys_kpc2),
        "omega_rt_cheat_phys_kpc2": float(omega_rt_cheat_phys_kpc2),
        "omega_tgt_circ_phys_kpc2": float(omega_tgt_circ_phys_kpc2),
        "omega_tgt_T50_phys_kpc2": float(omega_tgt_T50 * scale_phys),
        "omega_tgt_raw_phys_kpc2": float(omega_tgt_raw_phys_kpc2),
        "ratio_rt_circ_over_rt_cheat": float(omega_rt_circ / (omega_rt_cheat + 1e-30)),
        "ratio_rt_circ_over_tgt_raw": float(omega_rt_circ / (omega_tgt_raw + 1e-30)),
        "ratio_rt_circ_over_tgt_T50": float(omega_rt_circ / (omega_tgt_T50 + 1e-30)),
        "ratio_rt_cheat_over_tgt_raw": float(omega_rt_cheat / (omega_tgt_raw + 1e-30)),
        "ratio_tgt_circ_over_tgt_raw": float(omega_tgt_circ / (omega_tgt_raw + 1e-30)),
        "ratio_tgt_circ_over_tgt_T50": float(omega_tgt_circ / (omega_tgt_T50 + 1e-30)),
        "ratio_tgt_circ_over_rt_cheat": float(omega_rt_cheat / (omega_tgt_T50 + 1e-30)),
        "ratio_tgt_T50_over_tgt_raw": float(omega_tgt_T50 / (omega_tgt_raw + 1e-30)),
        "scaled_omega_rt_circ_over_rt_cheat": float(omega_tgt_T50/(omega_tgt_circ + 1e-30)*omega_rt_circ/(omega_rt_cheat + 1e-30)),
        "e_tgt": float(e_tgt_T50),                      # Ellipticity of the T50 beam
        "q_tgt": float(q_tgt_T50),                      # Axis ratio of the T50 beam
        "pa_tgt": float(pa_tgt_T50),                    # Position angle of the T50 beam (deg E of N)
        "dpa_deg": float(dpa_deg),                      # |PA_T50 - PA_RAW| in [0, 90] deg
        "pa_raw": float(pa_raw),                        # Position angle of the RAW beam (deg E of N)
        "pa_tgt": float(pa_tgt),                        # Position angle of the T50 beam (deg E of N)
        "uvker_fwhm_major_kl": uvker_fwhm_major_kl, # FWHM [kλ] along major/minor axes
        "uvker_fwhm_minor_kl": uvker_fwhm_minor_kl, # FWHM [kλ] along major/minor axes
        "uvker_fwhm_geo_kl": uvker_fwhm_geo_kl,     # geometric mean FWHM [kλ]
        "uvker_fwhm_phys_kpc": float(DA_kpc * theta_obs_geo_rad), # physical scale of observed beam [kpc]
        "beam_T50_phys_kpc": obs_phys_kpc,
        "beam_model50kpc_phys_kpc": pred_phys_kpc,
        "beam_T50_minus_model50kpc_phys_kpc": resid_phys_kpc,
        "uvker_axis_ratio": uvker_axis_ratio,       # 0–1; 1=circular uv taper
        "uvker_size_mismatch": uvker_size_mismatch, # ≈1 if matches 50 kpc circular model
        "ker_major_ratio_img": ker_major_ratio_img, # σ_major(cheat) / σ_major(rt-circ)
        "ker_minor_ratio_img": ker_minor_ratio_img, # σ_minor(cheat) / σ_minor(rt-circ)
        "ker_major_ratio_uv":  ker_major_ratio_uv,  # optional, uv domain
        "ker_minor_ratio_uv":  ker_minor_ratio_uv,  # optional, uv domain

        "beam_major_T50": bmaj_T50,                 # BMAJ of T50 beam [deg]
        "beam_minor_T50": bmin_T50,                 # BMIN of T50 beam [deg]
        "beam_major_RAW": bmaj_RAW,                 # BMAJ of RAW beam [deg]
        "beam_minor_RAW": bmin_RAW,                 # BMIN of RAW beam [deg]
        "beam_major_T50_scaled": bmaj_T50_scaled,   # BMAJ of T50 beam [deg], scaled
        "beam_minor_T50_scaled": bmin_T50_scaled,   # BMIN of T50 beam [deg], scaled
        "beam_major_RAW_scaled": bmaj_RAW_scaled,   # BMAJ of RAW beam [deg], scaled
        "beam_minor_RAW_scaled": bmin_RAW_scaled,   # BMIN of RAW beam [deg], scaled
        "beam_major_RAW_phys_kpc": bmaj_RAW_phys_kpc, # BMAJ of RAW beam [kpc physical]
        "beam_major_T50_phys_kpc": bmaj_T50_phys_kpc, # BMAJ of T50 beam [kpc physical]
        "beam_minor_RAW_phys_kpc": bmin_RAW_phys_kpc, # BMIN of RAW beam [kpc physical]
        "beam_minor_T50_phys_kpc": bmin_T50_phys_kpc, # BMIN of T50 beam [kpc physical]
        "beam_major_ratio": beam_major_ratio,       # BMAJ(T50) / BMAJ(RAW)
        "beam_minor_ratio": beam_minor_ratio,       # BMIN(T50) / BMIN(RAW)
        "beam_major_diff": beam_major_diff,         # BMAJ(T50) - BMAJ(RAW) [deg]
        "beam_minor_diff": beam_minor_diff,         # BMIN(T50) - BMIN(RAW) [deg]

        "taper_leverage": taper_leverage,           # ‖C₅₀‖₍F₎ / ‖C_raw‖₍F₎
        "pred_size_frob": pred_size,                # ‖C_raw + C₅₀‖₍F₎
        "pix_area_sr": pix_area_sr,                 # pixel area in steradians
        "npix": npix,                                # number of pixels in image
        "cov_mismatch_frob": cov_mismatch_frob,     # 0 = perfect match; larger = worse
        "cov_mismatch_trace": cov_mismatch_trace,   # signed fractional area mismatch
        "frac_area_mis": frac_area_mis,             # signed fractional area mismatch
        "ellip_mis": ellip_mis,                     # signed ellipticity mismatch
        "pa_mis": pa_mis,                           # signed PA mismatch in target's frame
    }


# ------------------------------ CLI example ------------------------------
if __name__ == "__main__":
    
    # Configuration
    ROOT = "/users/mbredber/scratch/data/PSZ2/fits"
    Z_CSV = "/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv"
    slug_to_z = load_z_table(Z_CSV)
    pairs = pick_random_pairs(ROOT, n=10**6, seed=42)  # effectively "all"
    y_axis = "omega_tgt_circ"  # (ΔΩ): (Ω_rt_cheat) / (Ω_tgt_raw)
    x_axis = "z"
    c_axis = "ellip_mis"           # (ΔC in tgt eigenframe): (ΔC_00-ΔC_11)/(w0+w1)

    
    # ---- interactive figure (shows after first point) ----
    #plt.ion()
    fig = plt.figure(figsize=(6.6, 5.2), dpi=130)
    gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 0.15], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    ax       = fig.add_subplot(gs[1, 0])                 # main scatter (bottom-left)
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)      # top histogram
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)      # right histogram
    cax      = fig.add_subplot(gs[1, 2])                 # dedicated colorbar axis (far right)

    # scatter (coloured by c_axis)
    sc = ax.scatter([], [], s=24, alpha=0.9, c=[], cmap="coolwarm")  # diverging
    cbar = fig.colorbar(sc, cax=cax)

    label_map = {
        "uvker_axis_ratio":        "uv-kernel axis ratio (minor/major)",
        "uvker_fwhm_geo_kl":       "uv-kernel geometric FWHM [kλ]",
        "uvker_size_mismatch":     "uv-kernel size / 50 kpc circular",
        "uvker_fwhm_major_kl":      "uv-kernel FWHM major axis [kλ]",
        "uvker_fwhm_minor_kl":      "uv-kernel FWHM minor axis [kλ]",
        "cov_mismatch_frob":       "Covariance mismatch ‖ΔC‖₍F₎ / ‖C_tgt‖₍F₎",
        "cov_mismatch_trace":      "Covariance trace mismatch (fraction)",
        "ker_major_ratio_img":     "kernel major-axis ratio (image): cheat / rt-circ",
        "ker_minor_ratio_img":     "kernel minor-axis ratio (image): cheat / rt-circ",
        "ker_major_ratio_uv":      "kernel major-axis ratio (uv): cheat / rt-circ",
        "ker_minor_ratio_uv":      "kernel minor-axis ratio (uv): cheat / rt-circ",
        "beam_major_T50":          "beam major-axis T50 [deg]",
        "beam_minor_T50":          "beam minor-axis T50 [deg]",
        "beam_major_RAW":          "beam major-axis RAW [deg]",
        "beam_minor_RAW":          "beam minor-axis RAW [deg]",
        "beam_major_T50_scaled":   "beam major-axis T50 scaled [Mpc]",
        "beam_minor_T50_scaled":   "beam minor-axis T50 scaled [Mpc]",
        "beam_major_RAW_scaled":   "beam major-axis RAW scaled [Mpc]",
        "beam_minor_RAW_scaled":   "beam minor-axis RAW scaled [Mpc]",
        "beam_major_RAW_phys_kpc": "beam major-axis RAW [kpc physical]",
        "beam_major_T50_phys_kpc": "beam major-axis T50 [kpc physical]",
        "beam_minor_RAW_phys_kpc": "beam minor-axis RAW [kpc physical]",
        "beam_minor_T50_phys_kpc": "beam minor-axis T50 [kpc physical]",
        "beam_major_ratio":        "beam major-axis ratio: T50 / RAW",
        "beam_minor_ratio":        "beam minor-axis ratio: T50 / RAW",
        "beam_major_diff":         "beam major-axis difference: T50 − RAW [deg]",
        "beam_minor_diff":         "beam minor-axis difference: T50 − RAW [deg]",
        "beam_T50_phys_kpc":        "T50 beam size [kpc physical]",
        "beam_model50kpc_phys_kpc": "Model 50 kpc beam size [kpc physical]",
        "beam_T50_minus_model50kpc_phys_kpc": "T50 − model 50 kpc beam size [kpc physical]",
        "omega_rt_circ":           "rt-circ kernel area [sr]",
        "omega_rt_cheat":          "rt-cheat kernel area [sr]",
        "omega_tgt_circ":          "tgt-circ beam area [sr]",
        "omega_tgt_T50":           "tgt-hdr beam area [sr]",
        "omega_tgt_raw":           "tgt-raw beam area [sr]",
        "omega_tgt_raw2":          "tgt-raw beam area (from C_raw) [sr]",
        "omega_tgt_T50":           "tgt-T50 beam area [sr]",
        "omega_circ_diff":         "omega_tgt_circ − (omega_rt_circ + omega_tgt_raw) [sr]",
        "omega_cheat_diff":        "omega_tgt_T50 − (omega_rt_cheat + omega_tgt_raw) [sr]",
        "omega_tgt_T50_minus_omega_raw": "omega_tgt_T50 − omega_tgt_raw [sr]",
        "omega_rt_circ_phys_kpc2":  "rt-circ kernel area [kpc² physical]",
        "omega_rt_cheat_phys_kpc2": "rt-cheat kernel area [kpc² physical]",
        "omega_tgt_circ_phys_kpc2": "tgt-circ beam area [kpc² physical]",
        "omega_tgt_raw_phys_kpc2":  "tgt-raw beam area [kpc² physical]",
        "omega_tgt_T50_phys_kpc2": "tgt-T50 beam area [kpc² physical]",
        "ratio_rt_circ_over_rt_cheat": "ratio (rt-circ / rt-cheat)",
        "ratio_tgt_circ_over_tgt_T50": "ratio (tgt-circ / tgt-hdr)",
        "ratio_rt_circ_over_tgt_T50":  "ratio (rt-circ / tgt-hdr)",
        "ratio_tgt_circ_over_rt_cheat":"ratio (tgt-circ / rt-cheat)",
        "taper_leverage":           "taper leverage ‖C₅₀‖₍F₎ / ‖C_raw‖₍F₎",
        "frac_area_mis":           "fractional area mismatch (trace)",
        "ellip_mis":               "ellipticity mismatch",
        "pa_mis":                  "PA mismatch (target frame)",
        "q_tgt":                   "Axis ratio q = BMIN/BMAJ",
        "e_tgt":                   "Ellipticity e = 1 − BMIN/BMAJ",
        "dpa_deg":                 "PA misalignment ΔPA [deg]",
        "z":                        "Redshift z",
        "cosmo_distance_Mpc":       "Angular diameter distance D_A [Mpc]",
        "npix":                    "Number of image pixels",
        "pix_area_sr":              "Pixel area [sr]",
    }

    # tidy the marginal axes
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    ax_histx.tick_params(axis='x', which='both', length=0)
    ax_histx.tick_params(axis='y', which='both', labelsize=9)
    ax_histy.tick_params(axis='y', which='both', length=0)
    ax_histy.tick_params(axis='x', which='both', labelsize=9)

    line_lin, = ax.plot([], [], lw=1.6, label="linear fit")
    line_exp, = ax.plot([], [], lw=1.6, ls="--", label="exp. decay fit")
    line_mean, = ax.plot([], [], lw=1.2, ls=":", label="mean(y)")
    ax.set_xlabel(label_map.get(x_axis, x_axis))
    ax.set_ylabel(label_map.get(y_axis, y_axis))
    cbar.set_label(label_map.get(c_axis, c_axis))
    #ax.grid(True, which="both", linestyle=":")
    legend_shown = False

    # click a point to print the cluster name
    xs, inv_g, names, col = [], [], [], []   # x collects res[x_axis]
    def _onpick(event):
        if event.artist is sc and names:
            ind = int(event.ind[0])
            print(f"[click] {names[ind]}: z={xs[ind]:.4f}, 1/g={inv_g[ind]:.4f}, col={col[ind]:.4f}")
    sc.set_picker(True)
    fig.canvas.mpl_connect("pick_event", _onpick)
    fig.canvas.draw_idle()

    def _exp_decay(z, A, b, C):
        return A * np.exp(-b * z) + C

    def _update_plot():
        # update scatter points
        X = np.asarray(xs, float)
        Y = np.asarray(inv_g, float)
        sc.set_offsets(np.c_[X, Y])
        if col:
            C = np.asarray(col, float)
            sc.set_array(C)
            finite = np.isfinite(C)
            if finite.any():
                # symmetric limits around 0 based on robust percentile of |C|
                c_abs = np.abs(C[finite])
                hi = np.nanpercentile(c_abs, 98)
                v = float(hi if np.isfinite(hi) and hi > 0 else np.nanmax(c_abs))
                if not np.isfinite(v) or v == 0:
                    v = 1.0
                sc.set_clim(-v, +v)
                cbar.update_normal(sc)

        if X.size:
            xpad = 0.05 * max(1e-9, (np.nanmax(X) - np.nanmin(X)))
            ax.set_xlim(np.nanmin(X) - xpad, np.nanmax(X) + xpad)

            Ypos = Y[Y > 0]
            if Ypos.size:
                ymin = np.nanmin(Ypos)
                ymax = np.nanmax(Ypos)
                ax.set_ylim(ymin * 0.9, ymax * 1.1)

        # fits (only when we have enough finite points)
        mask = np.isfinite(X) & np.isfinite(Y)
        Xf, Yf = X[mask], Y[mask]
        if Xf.size >= 2:
            xx = np.linspace(Xf.min(), Xf.max(), 200)

            # linear fit on Y:  Y = m*x + c
            m, c = np.polyfit(Xf, Yf, 1)
            y_lin = m * xx + c
            line_lin.set_data(xx, y_lin)
            line_lin.set_label(f"linear: m={m:.3g}, c={c:.3g}")

            # exponential decay with offset on linear Y: Y = A*exp(-b*z) + C
            if Xf.size >= 3:
                try:
                    p0 = [Yf.max() - Yf.min(), 1.0, Yf.min()]
                    (A, b, C), _ = curve_fit(_exp_decay, Xf, Yf, p0=p0, maxfev=10000)
                    line_exp.set_data(xx, _exp_decay(xx, A, b, C))
                    line_exp.set_label(f"exp: A={A:.3g}, b={b:.3g}, C={C:.3g}")
                except Exception:
                    pass

            # mean horizontal line over finite Y
            ybar = float(np.mean(Yf))
            line_mean.set_data([Xf.min(), Xf.max()], [ybar, ybar])
            line_mean.set_label(f"mean(y)={ybar:.3g}")

            ax.legend(frameon=False)

        # autoscale (use finite Y, not only positive)
        if X.size:
            xpad = 0.05 * max(1e-9, (np.nanmax(X) - np.nanmin(X)))
            ax.set_xlim(np.nanmin(X) - xpad, np.nanmax(X) + xpad)
            Yfin = Y[np.isfinite(Y)]
            if Yfin.size:
                ymin = np.nanmin(Yfin); ymax = np.nanmax(Yfin)
                ax.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))
        
        # --- marginal histograms (rebuild each update) ---
        ax_histx.cla()
        ax_histy.cla()
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)
        ax_histx.grid(False); ax_histy.grid(False)

        if Xf.size:
            nbx = max(10, int(np.sqrt(Xf.size)))
            ax_histx.hist(Xf, bins=nbx, color='0.5', alpha=0.6, edgecolor='none')
            ax_histx.set_ylabel("N", fontsize=9)

        if Yf.size:
            Yf = Yf[np.isfinite(Yf)]
            if Yf.size:
                y_min = float(np.nanmin(Yf))
                y_max = float(np.nanmax(Yf))
                # treat "almost equal" as equal (NumPy 2.x histogram is strict)
                if np.isclose(y_max, y_min, rtol=0.0, atol=1e-12*max(1.0, abs(y_max))):
                    y0 = y_min
                    ax_histy.hist(Yf, bins=1, range=(y0-0.5, y0+0.5),
                                  orientation='horizontal',
                                  color='0.5', alpha=0.6, edgecolor='none')
                else:
                    span = y_max - y_min
                    pad  = max(1e-12, 1e-3*span)  # small padding to ensure finite-sized bins
                    lo   = y_min - pad
                    hi   = y_max + pad
                    nby  = max(10, int(np.sqrt(Yf.size)))
                    ax_histy.hist(Yf, bins=nby, range=(lo, hi),  # explicit, padded range
                                  orientation='horizontal',
                                  color='0.5', alpha=0.6, edgecolor='none')
                ax_histy.set_xlabel("N", fontsize=9)


        # match axis limits tightly to main plot
        if Xf.size:
            ax_histx.set_xlim(ax.get_xlim())
        if Yf.size:
            ax_histy.set_ylim(ax.get_ylim())
            
        #ax.set_title(f"Beam-area scaling vs. redshift  (N={len(X)})")
        fig.canvas.draw_idle()
        #plt.pause(0.01)  # GUI heartbeat


    # ---- process all sources, updating the plot each time ----
    UPDATE_EVERY = 10
    for i, (RAW, T50) in enumerate(pairs, 1):
        slug = Path(RAW).stem
        z = slug_to_z.get(slug, np.nan)
        if not np.isfinite(z):
            print(f"[skip] no finite z for {slug}")
            continue

        res = get_beam_info(
            RAW, T50_fits=T50, out_pdf=None,
            z=z, fwhm_kpc=50.0,
        )

        xs.append(float(res[x_axis]))
        inv_g.append(float(res[y_axis]))
        names.append(slug)
        col.append(float(res[c_axis]))
        if i % UPDATE_EVERY == 0:
            _update_plot()

    #plt.ioff()
    _update_plot()
    plt.tight_layout()
    out = "omega_circ_over_omega_T50_vs_z_live.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out} with {len(xs)} clusters")
