"""matplotlib annotation helpers for FITS radio images (beam patch, scale bar)."""

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as COSMO

from dcreclass.utils.fits import ARCSEC_PER_RAD, arcsec_per_pix


def add_beam_patch(ax, header, color='white', alpha=0.8, loc='lower left'):
    """Add beam ellipse to a WCS-projection axis."""
    from matplotlib.patches import Ellipse
    bmaj_as  = float(header['BMAJ']) * 3600.0
    bmin_as  = float(header['BMIN']) * 3600.0
    bpa_deg  = float(header.get('BPA', 0.0))
    ny, nx   = int(header['NAXIS2']), int(header['NAXIS1'])
    asx, asy = arcsec_per_pix(header)
    bmaj_pix = bmaj_as / asx; bmin_pix = bmin_as / asy
    margin   = 0.08
    y_center = (ny * margin + bmaj_pix / 2 if 'lower' in loc
                else ny * (1 - margin) - bmaj_pix / 2)
    x_center = (nx * margin + bmaj_pix / 2 if 'left' in loc
                else nx * (1 - margin) - bmaj_pix / 2)
    beam = Ellipse(xy=(x_center, y_center), width=bmaj_pix, height=bmin_pix,
                   angle=bpa_deg, transform=ax.get_transform('pixel'),
                   facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5)
    ax.add_patch(beam)
    label  = f"{bmaj_as:.1f}\"x{bmin_as:.1f}\""
    y_text = (ny * margin + bmaj_pix + 5 if 'lower' in loc
              else ny * (1 - margin) - bmaj_pix - 5)
    ax.text(x_center, y_text, label, transform=ax.get_transform('pixel'),
            fontsize=9, color=color, weight='bold',
            ha='center', va='bottom' if 'lower' in loc else 'top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none'))


def add_scalebar_kpc(ax, header, z, length_kpc=100.0, color='white', loc='lower right'):
    """Add physical scale bar to a WCS-projection axis."""
    from matplotlib.lines import Line2D
    DA_kpc   = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_as = (length_kpc / DA_kpc) * ARCSEC_PER_RAD
    asx, _   = arcsec_per_pix(header)
    bar_px   = theta_as / asx
    ny, nx   = int(header['NAXIS2']), int(header['NAXIS1'])
    margin   = 0.08; bar_thickness = 3
    y_bar    = (ny * margin if 'lower' in loc else ny * (1 - margin) - bar_thickness)
    x_start  = (nx * margin if 'left' in loc else nx * (1 - margin) - bar_px)
    x_end    = x_start + bar_px
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
                          loc='lower left', fontsize=6, x_offset=0.0):
    """Add beam ellipse to a plain (non-WCS) imshow axis."""
    from matplotlib.patches import Ellipse
    bmaj_as  = float(header['BMAJ']) * 3600.0
    bmin_as  = float(header['BMIN']) * 3600.0
    bpa_deg  = float(header.get('BPA', 0.0))
    ny, nx   = int(header['NAXIS2']), int(header['NAXIS1'])
    asx, asy = arcsec_per_pix(header)
    bmaj_pix = bmaj_as / asx; bmin_pix = bmin_as / asy
    margin   = 0.10
    bmaj_f   = bmaj_pix / ny; bmin_f = bmin_pix / nx
    yc_n = margin + bmaj_f / 2 if 'lower' in loc else 1 - margin - bmaj_f / 2
    xc_n = margin + bmin_f / 2 if 'left'  in loc else 1 - margin - bmin_f / 2
    xc_n += x_offset
    yc_n = np.clip(yc_n, bmaj_f / 2 + 0.05, 1 - bmaj_f / 2 - 0.05)
    xc_n = np.clip(xc_n, bmin_f / 2 + 0.05, 1 - bmin_f / 2 - 0.05)
    ax.add_patch(Ellipse(xy=(xc_n * nx, yc_n * ny), width=bmaj_pix, height=bmin_pix,
                         angle=bpa_deg, transform=ax.transData,
                         facecolor=color, edgecolor='black', alpha=alpha, linewidth=0.8))
    yt_n = yc_n - bmaj_f / 2 - 0.03 if 'lower' in loc else yc_n + bmaj_f / 2 + 0.03
    ax.text(xc_n, yt_n, f"{bmaj_as:.1f}\u2033\u00d7{bmin_as:.1f}\u2033",
            transform=ax.transAxes, fontsize=fontsize, color=color, weight='bold',
            ha='center', va='top' if 'lower' in loc else 'bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7, edgecolor='none'))


def add_scalebar_kpc_simple(ax, header, z, length_kpc=1000.0,
                            color='white', loc='lower right', fontsize=6):
    """Add physical scale bar to a plain (non-WCS) imshow axis."""
    from matplotlib.lines import Line2D
    DA_kpc   = COSMO.angular_diameter_distance(z).to_value(u.kpc)
    theta_as = (length_kpc / DA_kpc) * ARCSEC_PER_RAD
    asx, _   = arcsec_per_pix(header)
    bar_px   = theta_as / asx
    ny, nx   = int(header['NAXIS2']), int(header['NAXIS1'])
    margin   = 0.10
    y_n      = margin if 'lower' in loc else 1 - margin
    xs_n     = margin if 'left' in loc else 1 - margin - bar_px / nx
    xe_n     = xs_n + bar_px / nx
    ax.add_line(Line2D([xs_n * nx, xe_n * nx], [y_n * ny, y_n * ny],
                       color=color, linewidth=2, solid_capstyle='butt'))
    label = f"{length_kpc/1000:.0f} Mpc" if length_kpc >= 1000 else f"{int(length_kpc)} kpc"
    yt_n  = y_n - 0.04 if 'lower' in loc else y_n + 0.04
    ax.text((xs_n + xe_n) / 2, yt_n, label,
            transform=ax.transAxes, fontsize=fontsize, color=color, weight='bold',
            ha='center', va='top' if 'lower' in loc else 'bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7, edgecolor='none'))
