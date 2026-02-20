#!/usr/bin/env python3
"""
Simplified debugging script to show all versions WITHOUT reprojection.
This will help identify if reproject_like is introducing NaNs.

ENHANCED with beam size annotations and 100 kpc scale bars.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u

print("Running 0.1.2.plot_versions.py (debugging NaNs without reprojection)")

def read_fits_simple(fpath):
    """Read FITS file and return (2D array, header, WCS)."""
    with fits.open(fpath, memmap=False) as hdul:
        header = hdul[0].header
        wcs_full = WCS(header)
        wcs2d = wcs_full.celestial if hasattr(wcs_full, "celestial") else WCS(header, naxis=2)
        arr = np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))
        if arr.ndim == 3:
            arr = np.nanmean(arr, axis=0)
    return arr, header, wcs2d

def robust_vmin_vmax(arr, lo=1, hi=99):
    """Compute robust min/max from percentiles."""
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0, 1.0
    vals = arr[finite]
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)

def arcsec_per_pix(h):
    """Compute effective pixel scales (|dx|, |dy|) in arcseconds from header WCS."""
    ARCSEC_PER_RAD = 206264.80624709636
    if 'CD1_1' in h:
        M = np.array([[h['CD1_1'], h.get('CD1_2', 0.0)],
                      [h.get('CD2_1', 0.0), h['CD2_2']]], float)
    else:
        pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
        pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
        cd1 =h.get('CDELT1', 1.0); cd2 =h.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    M = M * (np.pi/180.0)  # Convert to radians
    dx = np.hypot(M[0,0], M[1,0])
    dy = np.hypot(M[0,1], M[1,1])
    return dx*ARCSEC_PER_RAD, dy*ARCSEC_PER_RAD

def add_beam_patch(ax, header, color='white', alpha=0.8, loc='lower left'):
    """
    Add beam ellipse to corner of WCS axis.
    
    Parameters:
    - ax: WCS axis (matplotlib axis with projection)
    - header: FITS header containing beam info (BMAJ, BMIN, BPA)
    - color: Color of beam ellipse
    - alpha: Transparency of beam ellipse
    - loc: Location string ('lower left', 'lower right', 'upper left', 'upper right')
    """
    from matplotlib.patches import Ellipse
    
    # Extract beam parameters from header
    bmaj_deg = float(header['BMAJ'])  # degrees
    bmin_deg = float(header['BMIN'])  # degrees
    bpa_deg = float(header.get('BPA', 0.0))  # degrees
    
    # Convert to arcseconds for display
    bmaj_as = bmaj_deg * 3600.0
    bmin_as = bmin_deg * 3600.0
    
    # Get image dimensions and pixel scale
    ny, nx = int(header['NAXIS2']), int(header['NAXIS1'])
    asx, asy = arcsec_per_pix(header)
    
    # Convert beam size to pixels
    bmaj_pix = bmaj_as / asx
    bmin_pix = bmin_as / asy
    
    # Position beam patch in corner (in pixel coordinates)
    margin_frac = 0.08  # 8% margin from edge
    if 'lower' in loc:
        y_center = ny * margin_frac + bmaj_pix / 2
    else:  # upper
        y_center = ny * (1 - margin_frac) - bmaj_pix / 2
        
    if 'left' in loc:
        x_center = nx * margin_frac + bmaj_pix / 2
    else:  # right
        x_center = nx * (1 - margin_frac) - bmaj_pix / 2
    
    # Create ellipse patch in pixel coordinates
    beam = Ellipse(
        xy=(x_center, y_center),
        width=bmaj_pix,
        height=bmin_pix,
        angle=bpa_deg,  # Position angle
        transform=ax.get_transform('pixel'),
        facecolor=color,
        edgecolor='black',
        alpha=alpha,
        linewidth=1.5
    )
    ax.add_patch(beam)
    
    # Add text label with beam size
    label = f"{bmaj_as:.1f}″×{bmin_as:.1f}″"
    if 'lower' in loc:
        y_text = ny * margin_frac + bmaj_pix + 5
    else:
        y_text = ny * (1 - margin_frac) - bmaj_pix - 5
    
    ax.text(
        x_center, y_text, label,
        transform=ax.get_transform('pixel'),
        fontsize=9, color=color, weight='bold',
        ha='center', va='bottom' if 'lower' in loc else 'top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none')
    )

def add_scalebar_kpc(ax, header, z, length_kpc=5000.0, color='white', loc='lower right'):
    """
    Add physical scale bar to WCS axis.
    
    Parameters:
    - ax: WCS axis (matplotlib axis with projection)
    - header: FITS header with WCS info
    - z: Redshift
    - length_kpc: Physical length of scale bar in kpc
    - color: Color of scale bar
    - loc: Location string ('lower left', 'lower right', 'upper left', 'upper right')
    """
    from matplotlib.lines import Line2D
    
    ARCSEC_PER_RAD = 206264.80624709636
    
    # Calculate angular size of scale bar
    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_rad = length_kpc / DA_kpc
    theta_as = theta_rad * ARCSEC_PER_RAD
    
    # Convert to pixels
    asx, asy = arcsec_per_pix(header)
    bar_length_pix = theta_as / asx
    
    # Get image dimensions
    ny, nx = int(header['NAXIS2']), int(header['NAXIS1'])
    
    # Position scale bar in corner
    margin_frac = 0.08
    bar_thickness = 3  # pixels
    
    if 'lower' in loc:
        y_bar = ny * margin_frac
    else:  # upper
        y_bar = ny * (1 - margin_frac) - bar_thickness
        
    if 'left' in loc:
        x_start = nx * margin_frac
    else:  # right
        x_start = nx * (1 - margin_frac) - bar_length_pix
    
    x_end = x_start + bar_length_pix
    
    # Draw scale bar as thick line
    line = Line2D(
        [x_start, x_end], [y_bar, y_bar],
        transform=ax.get_transform('pixel'),
        color=color, linewidth=bar_thickness,
        solid_capstyle='butt'
    )
    ax.add_line(line)
    
    # Add text label
    label = f"{int(length_kpc)} kpc"
    x_text = (x_start + x_end) / 2
    
    if 'lower' in loc:
        y_text = y_bar - 8
        va = 'top'
    else:
        y_text = y_bar + bar_thickness + 8
        va = 'bottom'
    
    ax.text(
        x_text, y_text, label,
        transform=ax.get_transform('pixel'),
        fontsize=10, color=color, weight='bold',
        ha='center', va=va,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.5, edgecolor='none')
    )

def check_nans(arr, label):
    """Report NaN statistics."""
    n_nan = np.isnan(arr).sum()
    n_tot = arr.size
    pct = 100 * n_nan / n_tot if n_tot > 0 else 0
    print(f"  {label:15s}: {n_nan:8d} NaNs ({pct:5.2f}%) of {n_tot} pixels")
    return n_nan

# Source to debug
source_name = "PSZ2G048.10+57.16"
root_dir = Path("/users/mbredber/scratch/data/PSZ2/fits")
source_dir = root_dir / source_name

# Redshift for scale bar (load from your redshift table)
z = 0.549  # PSZ2G048.10+57.16 redshift

print("="*80)
print(f"DEBUGGING NaN SOURCE: {source_name}")
print("="*80)

# Load all files
scales = [25, 50, 100]
data = {}

for scale in scales:
    print(f"\n[SCALE {scale}kpc]")
    
    # Load T file
    t_path = source_dir / f"{source_name}T{scale}kpc.fits"
    if not t_path.exists():
        print(f"  SKIP: T{scale}kpc.fits not found")
        continue
    
    t_arr, t_hdr, t_wcs = read_fits_simple(t_path)
    print(f"  T{scale}kpc loaded: shape={t_arr.shape}")
    check_nans(t_arr, f"T{scale}kpc")
    
    # Load SUB file
    sub_path = source_dir / f"{source_name}T{scale}kpcSUB.fits"
    if sub_path.exists():
        sub_arr, sub_hdr, sub_wcs = read_fits_simple(sub_path)
        print(f"  T{scale}kpcSUB loaded: shape={sub_arr.shape}")
        check_nans(sub_arr, f"T{scale}kpcSUB")
    else:
        sub_arr, sub_hdr, sub_wcs = None, None, None
        print(f"  T{scale}kpcSUB not found")
    
    data[scale] = {
        't_arr': t_arr, 't_hdr': t_hdr, 't_wcs': t_wcs,
        'sub_arr': sub_arr, 'sub_hdr': sub_hdr, 'sub_wcs': sub_wcs
    }

print("\n" + "="*80)
print("CREATING VISUALIZATION (NO REPROJECTION)")
print("="*80)

# Create figure: 2 rows (T, SUB) × 3 columns (25, 50, 100 kpc)
fig = plt.figure(figsize=(15, 10), constrained_layout=True)

for i, scale in enumerate(scales):
    if scale not in data:
        continue
    
    d = data[scale]
    
    # Calculate what the bar SHOULD be
    DA_kpc = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_rad = 5000.0 / DA_kpc  # 500 kpc
    theta_as = theta_rad * 206264.80624709636
    asx, asy = arcsec_per_pix(d['t_hdr'])
    bar_length_pix = theta_as / asx
    
    ny, nx = int(d['t_hdr']['NAXIS2']), int(d['t_hdr']['NAXIS1'])
    
    print(f"\nT{scale}kpc:")
    print(f"  Pixel scale: {asx:.3f} arcsec/pix")
    print(f"  500 kpc = {theta_as:.1f} arcsec = {bar_length_pix:.1f} pixels")
    print(f"  Fraction of image width: {100*bar_length_pix/nx:.1f}%")
    
    # Row 1: T images (native grid, no reprojection)
    ax_t = fig.add_subplot(2, 3, i+1, projection=d['t_wcs'])
    vmin_t, vmax_t = robust_vmin_vmax(d['t_arr'])
    im_t = ax_t.imshow(d['t_arr'], origin="lower", vmin=vmin_t, vmax=vmax_t)
    ax_t.set_title(f"T{scale}kpc (native grid)\nShape: {d['t_arr'].shape}")
    fig.colorbar(im_t, ax=ax_t, shrink=0.8)
    
    # Add beam annotation and scale bar
    add_beam_patch(ax_t, d['t_hdr'], color='yellow', loc='lower left')
    add_scalebar_kpc(ax_t, d['t_hdr'], z, length_kpc=5000.0, color='white', loc='lower right')
    
    # Row 2: SUB images (native grid, no reprojection)
    if d['sub_arr'] is not None and d['sub_wcs'] is not None:
        ax_sub = fig.add_subplot(2, 3, 3 + i + 1, projection=d['sub_wcs'])
        vmin_sub, vmax_sub = robust_vmin_vmax(d['sub_arr'])
        im_sub = ax_sub.imshow(d['sub_arr'], origin="lower", vmin=vmin_sub, vmax=vmax_sub)
        ax_sub.set_title(f"T{scale}kpcSUB (native grid)\nShape: {d['sub_arr'].shape}")
        
        # Add beam annotation and scale bar
        add_beam_patch(ax_sub, d['sub_hdr'], color='yellow', loc='lower left')
        add_scalebar_kpc(ax_sub, d['sub_hdr'], z, length_kpc=5000.0, color='white', loc='lower right')
    else:
        ax_sub = fig.add_subplot(2, 3, 3 + i + 1)
        im_sub = ax_sub.imshow(np.zeros((100, 100)), origin="lower", cmap='gray')
        ax_sub.set_title(f"T{scale}kpcSUB\n(not available)")
        ax_sub.axis('off')
    
    fig.colorbar(im_sub, ax=ax_sub, shrink=0.8)

fig.suptitle(f"{source_name} — All scales on NATIVE grids (NO reprojection) — z={z:.4f}", fontsize=16)
plt.savefig(f"{source_name}_native_grids_no_reproject.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\n[OK] Saved: {source_name}_native_grids_no_reproject.png")
print("\n" + "="*80)
print("ANALYSIS:")
print("- If NaNs appear in this figure → problem is in ORIGINAL files")
print("- If NO NaNs in this figure → problem is in reproject_like()")
print("="*80)