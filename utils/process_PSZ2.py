#!/usr/bin/env python3

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd


# ─── CONFIG ────────────────────────────────────────────────────────────────
FITS_ROOT       = "/users/mbredber/scratch/data/PSZ2/fits"  # Path to downloaded FITS files
CROP_ROOT       = "/users/mbredber/scratch/data/PSZ2/crops" # Path to save cropped PNGs
META_CSV        = "/users/mbredber/scratch/data/PSZ2/cluster_metadata.csv" # Metadata CSV file
CROP_SIZE       = None  # Set to None to use full size

FILENAME_SUFFIXES = ["T15arcsec"] # List of suffixes to use
# or to grab _all_ .fits variants in the first cluster folder:
if FILENAME_SUFFIXES == ["ALL"]:
    import glob
    suffixes = set()
    for slug in os.listdir(FITS_ROOT):
        dpath = os.path.join(FITS_ROOT, slug)
        if not os.path.isdir(dpath):
            continue
        pattern = os.path.join(dpath, f"{slug}*.fits")
        for fpath in glob.glob(pattern):
            base = os.path.splitext(os.path.basename(fpath))[0]
            suffixes.add(base.replace(slug, ""))

    FILENAME_SUFFIXES = sorted(suffixes)
    print("FILENAME_SUFFIXES:", FILENAME_SUFFIXES)


# ─── SETUP ─────────────────────────────────────────────────────────────────

BUCKETS = {
    'RH':   ['RH', 'DE'],
    'RR':   ['RR', 'DE'],
    'NDE':  ['NDE'],
    'cRH':  ['cRH', 'cDE'],
    'cRR':  ['cRR', 'cDE'],
    'U':    ['U'],
    'N/A':  ['unclassified']
}


os.makedirs(CROP_ROOT, exist_ok=True)

# ─── LOAD METADATA ─────────────────────────────────────────────────────────
meta = pd.read_csv(META_CSV)
meta['slug'] = meta['slug'].str.replace(" ", "")
meta = meta.set_index('slug')
rows = []

for suffix in FILENAME_SUFFIXES:
    suffix_dir = suffix or "RAW"
    OUT_BASE    = os.path.join("data/PSZ2/classified", suffix_dir)
    os.makedirs(OUT_BASE, exist_ok=True)
    for dirs in BUCKETS.values():
        for d in dirs:
            os.makedirs(os.path.join(OUT_BASE, d), exist_ok=True)

    # ─── LOOP OVER CLUSTERS ────────────────────────────────────────────────────
    for slug in os.listdir(FITS_ROOT):
        dpath = os.path.join(FITS_ROOT, slug)
        fits_path = os.path.join(dpath, f"{slug}{suffix}.fits")
        if not os.path.exists(fits_path):
            print(f"❌ No {suffix} FITS for {slug}")
            continue

        fits_path = os.path.join(dpath, f"{slug}{suffix}.fits")
        print(f"  {fits_path}")
        if not os.path.exists(fits_path):
            print(f"❌ No FITS file for {slug}")
            continue

        try:
            hdu = fits.open(fits_path)[0]
            data = hdu.data
            #data = data[0,0,:,:]
            data = data.squeeze()
            if data.ndim != 2:
                raise ValueError(f"Expected a 2D image but got {data.ndim}D")
            
            plt.imshow(data, origin='lower', cmap='afmhot')
            plt.axis('off')
            plt.savefig(os.path.join(CROP_ROOT, f"{slug}.png"), bbox_inches='tight', pad_inches=0)
            plt.savefig("Last.png", bbox_inches='tight', pad_inches=0)
                
            # Crop the image if needed
            if CROP_SIZE is not None:
                center = np.array(data.shape) // 2
                #half_size = CROP_SIZE // 2
                #half_size = np.array(CROP_SIZE) // 2
                half_x, half_y = CROP_SIZE[0] // 2, CROP_SIZE[1] // 2
                #data = data[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size]
                data = data[center[0]-half_x:center[0]+half_x, center[1]-half_y:center[1]+half_y]
                
            out_png = os.path.join(CROP_ROOT, f"{slug}.png")
            plt.imshow(data, origin='lower', cmap='afmhot')
            plt.axis('off')
            plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Classification
            if slug in meta.index:
                cls = meta.loc[slug]['Classification']
                if isinstance(cls, (bytes, bytearray)):
                    cls = cls.decode('utf-8')
            else:
                cls = 'N/A'

            # if there are multiple labels (comma-separated), copy into each bucket
            labels = [s.strip() for s in cls.split(',')]
            for lab in labels:
                buckets = BUCKETS[lab] if lab in BUCKETS else ['unclassified']
                for bucket in buckets:
                    dest_dir = os.path.join(OUT_BASE, bucket)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(out_png, os.path.join(dest_dir, os.path.basename(out_png)))

            rows.append({
                'slug': slug,
                'image': os.path.basename(out_png),
                'class': cls
            })


            print(f"✅ Saved {slug}.png to {cls}. It has size {data.shape}.")

        except Exception as e:
            print(f"⚠️ Error processing {slug}: {e}")

    # ─── WRITE CSV SUMMARY ─────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    suffix_csv = os.path.join(OUT_BASE, f"train_labels_{suffix_dir}.csv")
    df.to_csv(suffix_csv, index=False)
    print(f"📄 Written {len(df)} entries to {suffix_csv}")

