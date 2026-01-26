#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata.utils import Cutout2D

# the one image we’re experimenting on:
FITS_FILE = "data/PSZ2/fits/PSZ2G023.17+86.71/PSZ2G023.17+86.71.fits"

# output dir for all your trials:
OUT_DIR = "data/PSZ2/hyperparam_sweep/"
os.makedirs(OUT_DIR, exist_ok=True)

CROP_SIZE = (1000, 1000) # fixed cutout settings
BG_SIGMAS = [0, 1, 20, 100, 1000] # set of background‐filter sigmas to try
FLOOR_SIGMAS = [0, 0.1, 0.5, 1.0, 1.5] # set of detection‐floor sigmas to try

# read your FITS, do the “force 2D” dance
hdul = fits.open(FITS_FILE)
hdr  = hdul[0].header
data = np.squeeze(np.array(hdul[0].data))
if data.ndim == 3:
    data = data[0]
hdul.close()

# grab WCS center to cut out around
wcs2d = WCS(hdr).celestial
cen   = SkyCoord(hdr["CRVAL1"], hdr["CRVAL2"], unit="deg")
xpix, ypix = wcs2d.world_to_pixel(cen)
cut       = Cutout2D(data, (xpix, ypix), CROP_SIZE, wcs=wcs2d).data

# now sweep!
for bg_s in BG_SIGMAS:
    # background = big‐sigma Gaussian
    background = gaussian_filter(cut, sigma=bg_s)

    # flat = cut minus background
    flat = cut - background

    for floor_s in FLOOR_SIGMAS:
        # compute detection floor from SMALLER smooth (3px)
        local_rms    = np.nanstd(gaussian_filter(cut, sigma=3))
        detect_floor = floor_s*local_rms

        # clamp at floor
        floored = np.clip(flat, detect_floor, None)

        # asinh stretch + normalize
        stretched = np.arcsinh(floored / np.nanstd(floored))
        vmin, vmax = np.nanpercentile(stretched, [1, 99])
        normed = np.clip((stretched - vmin) / (vmax - vmin), 0, 1)

        # write it
        fname = f"cut_bg{bg_s:03d}_floor{floor_s:.1f}.png"
        outp  = os.path.join(OUT_DIR, fname)
        plt.imsave(outp, normed, cmap="gray", origin="lower")
        print("→", outp)
