#!/usr/bin/env python3
import os
import shutil
from astropy.table import Table

# ─── CONFIG ────────────────────────────────────────────────────────────────
FITS_ROOT         = "/users/mbredber/scratch/data/PSZ2/fits"
CLASS_ROOT        = "/users/mbredber/scratch/data/PSZ2/classified"  # where to place copies
META_FITS         = "/users/mbredber/scratch/data/PSZ2/planck_dr2_metadata.fits"
FILENAME_SUFFIXES = ["ALL"]   # or list of suffixes, e.g. ["_T50kpcSUB", "_T100kpcSUB"]

BUCKETS = {
    'RH':   ['RH', 'DE'],
    'RR':   ['RR', 'DE'],
    'NDE':  ['NDE'],
    'cRH':  ['cRH', 'cDE'],
    'cRR':  ['cRR', 'cDE'],
    'U':    ['U'],
    'N/A':  ['unclassified']
}

# ─── PREP ─────────────────────────────────────────────────────────────────
os.makedirs(CLASS_ROOT, exist_ok=True)

# read metadata table (must have column “Name” and “Classification”)
meta = Table.read(META_FITS)
meta = meta.to_pandas()
meta['slug'] = meta['Name'].apply(lambda n: n.decode('utf-8') if isinstance(n, (bytes,bytearray)) else str(n))
meta['slug'] = meta['slug'].str.replace(" ", "")
meta = meta.set_index('slug')

# if you asked for “ALL” suffixes, discover them automatically:
if FILENAME_SUFFIXES == ["ALL"]:
    import glob
    suffixes = set()
    for slug in os.listdir(FITS_ROOT):
        d = os.path.join(FITS_ROOT, slug)
        if not os.path.isdir(d): continue
        for p in glob.glob(os.path.join(d, f"{slug}*.fits")):
            base = os.path.splitext(os.path.basename(p))[0]
            suffixes.add(base.replace(slug, ""))
    FILENAME_SUFFIXES = sorted(suffixes)

# ─── COPY LOOP ─────────────────────────────────────────────────────────────
for suffix in FILENAME_SUFFIXES:
    # create top-level dir for this suffix
    suffix_name = suffix or "RAW"
    out_base = os.path.join(CLASS_ROOT, suffix_name)
    os.makedirs(out_base, exist_ok=True)
    # make bucket subdirs
    for bucket_list in BUCKETS.values():
        for b in bucket_list:
            os.makedirs(os.path.join(out_base, b), exist_ok=True)

    for slug in os.listdir(FITS_ROOT):
        dpath   = os.path.join(FITS_ROOT, slug)
        fits_fn = f"{slug}{suffix}.fits"
        src     = os.path.join(dpath, fits_fn)
        if not os.path.isfile(src):
            print(f"⚠️  missing {fits_fn} for {slug}")
            continue

        # classification string (could be comma-separated)
        if slug in meta.index:
            cls_str = meta.loc[slug, 'Classification']
            if isinstance(cls_str, (bytes,bytearray)):
                cls_str = cls_str.decode('utf-8')
        else:
            cls_str = 'N/A'

        # split multi-labels, then map to buckets
        for lab in (s.strip() for s in cls_str.split(',')):
            buckets = BUCKETS.get(lab, ['unclassified'])
            for b in buckets:
                dest_dir = os.path.join(out_base, b)
                dest_fn = f"{slug}.fits"
                dst     = os.path.join(dest_dir, dest_fn)
                shutil.copy2(src, dst)

        print(f"✅  {fits_fn} → {cls_str}")

print("Done.")
