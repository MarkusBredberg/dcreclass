#!/usr/bin/env python3
"""
audit_dataset_composition.py
Audits PSZ2 dataset composition from first principles.
Reproduces split counts and verifies tab:dataset_composition figures.
No PyTorch dependency — stdlib + numpy + sklearn only.
"""

import csv
import os
import re
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSIFIED_DIR = '/users/mbredber/scratch/data/PSZ2/classified'
BEAM_CROP_DIR  = '/users/mbredber/scratch/data/PSZ2/beam_crop/circular/fits_files'
CSV_PATH       = '/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv'

DE_SUBDIR  = 'DE'
NDE_SUBDIR = 'NDE'

VERSIONS = ['RAW', 'T25kpc', 'T50kpc', 'T100kpc']

# Split parameters (from loaders.py)
SGKF_N_SPLITS  = 5   # train/test
SGKF_SEED_TEST = 41
SGKF_CV_SPLITS = 10  # train/val
SGKF_SEED_CV   = 42


# ---------------------------------------------------------------------------
# 1. Load redshift CSV
# ---------------------------------------------------------------------------
def load_redshift_csv(path=CSV_PATH):
    """Return dict {slug: z_float}.  z=0.0 means unknown/zero."""
    z_map = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row['slug'].strip()
            try:
                z = float(row['z'])
            except (ValueError, KeyError):
                z = None
            z_map[slug] = z
    return z_map


# ---------------------------------------------------------------------------
# 2. List slugs from classified/<version>/<class>/ (strip .fits)
# ---------------------------------------------------------------------------
def list_classified_slugs(version, class_dir):
    """
    Return sorted list of slugs (basename without version suffix and .fits).
    RAW files:    PSZ2G023.17+86.71.fits              → PSZ2G023.17+86.71
    Taper files:  PSZ2G023.17+86.71T25kpc.fits        → PSZ2G023.17+86.71
    """
    path = os.path.join(CLASSIFIED_DIR, version, class_dir)
    if not os.path.isdir(path):
        return []
    slugs = []
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        name = f[:-5] if f.endswith('.fits') else f
        # Strip trailing version tag if present (e.g. "T25kpc", "T50kpc", "T100kpc")
        if version != 'RAW' and name.endswith(version):
            name = name[: -len(version)]
        slugs.append(name)
    return sorted(slugs)


# ---------------------------------------------------------------------------
# 3. List slugs from beam_crop formatted FITS for a given version tag
# ---------------------------------------------------------------------------
def list_beamcrop_slugs(version_tag):
    """
    Parse filenames like:
      PSZ2GXXX_<version_tag>_fmt_128x128_circular.fits
    and return set of slugs.
    """
    pattern = re.compile(
        r'^(.+?)_' + re.escape(version_tag) + r'_fmt_\d+x\d+_circular\.fits$'
    )
    slugs = set()
    if not os.path.isdir(BEAM_CROP_DIR):
        return slugs
    for fname in os.listdir(BEAM_CROP_DIR):
        m = pattern.match(fname)
        if m:
            slugs.add(m.group(1))
    return slugs


# ---------------------------------------------------------------------------
# 4. Run StratifiedGroupKFold split and report counts
# ---------------------------------------------------------------------------
def run_split(slugs_de, slugs_nde):
    """
    Reproduce the exact train/test split from loaders.py.
    Returns a dict with total, test, and val/train ranges across CV folds.
    """
    all_slugs  = list(slugs_de) + list(slugs_nde)
    all_labels = [1] * len(slugs_de) + [0] * len(slugs_nde)
    y      = np.array(all_labels)
    groups = np.array(all_slugs)
    n      = len(y)

    # ---- train/test split ----
    sgkf_test = StratifiedGroupKFold(
        n_splits=SGKF_N_SPLITS, shuffle=True, random_state=SGKF_SEED_TEST
    )
    trainval_idx, test_idx = next(
        sgkf_test.split(np.zeros(n), y, groups)
    )

    test_de  = int(np.sum(y[test_idx] == 1))
    test_nde = int(np.sum(y[test_idx] == 0))
    tv_de    = int(np.sum(y[trainval_idx] == 1))
    tv_nde   = int(np.sum(y[trainval_idx] == 0))

    # ---- inner CV for val counts ----
    y_tv  = y[trainval_idx]
    g_tv  = groups[trainval_idx]
    sgkf_cv = StratifiedGroupKFold(
        n_splits=SGKF_CV_SPLITS, shuffle=True, random_state=SGKF_SEED_CV
    )
    val_de_counts  = []
    val_nde_counts = []
    train_de_counts  = []
    train_nde_counts = []
    for tr_rel, va_rel in sgkf_cv.split(np.zeros(len(y_tv)), y_tv, g_tv):
        val_de_counts.append(int(np.sum(y_tv[va_rel] == 1)))
        val_nde_counts.append(int(np.sum(y_tv[va_rel] == 0)))
        train_de_counts.append(int(np.sum(y_tv[tr_rel] == 1)))
        train_nde_counts.append(int(np.sum(y_tv[tr_rel] == 0)))

    return {
        'n_total': n,
        'n_de': len(slugs_de),
        'n_nde': len(slugs_nde),
        'test_n': len(test_idx),
        'test_de': test_de,
        'test_nde': test_nde,
        'trainval_n': len(trainval_idx),
        'trainval_de': tv_de,
        'trainval_nde': tv_nde,
        'val_de_range':  (min(val_de_counts),   max(val_de_counts)),
        'val_nde_range': (min(val_nde_counts),  max(val_nde_counts)),
        'val_n_range':   (min(v+n for v,n in zip(val_de_counts, val_nde_counts)),
                          max(v+n for v,n in zip(val_de_counts, val_nde_counts))),
        'train_de_range':  (min(train_de_counts),  max(train_de_counts)),
        'train_nde_range': (min(train_nde_counts), max(train_nde_counts)),
        'train_n_range':   (min(t+n for t,n in zip(train_de_counts, train_nde_counts)),
                            max(t+n for t,n in zip(train_de_counts, train_nde_counts))),
    }


def fmt_range(lo, hi):
    if lo == hi:
        return str(lo)
    return f"{lo}–{hi}"


# ---------------------------------------------------------------------------
# 5. Main audit
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("PSZ2 DATASET COMPOSITION AUDIT")
    print("=" * 72)

    z_map = load_redshift_csv()
    print(f"\nRedshift CSV loaded: {len(z_map)} entries from {CSV_PATH}\n")

    # Collect per-version results for summary table
    summary = {}

    raw_de_slugs  = None
    raw_nde_slugs = None

    for version in VERSIONS:
        ver_dir = os.path.join(CLASSIFIED_DIR, version)
        if not os.path.isdir(ver_dir):
            print(f"[{version}] directory not found — skipping.\n")
            continue

        de_slugs  = list_classified_slugs(version, DE_SUBDIR)
        nde_slugs = list_classified_slugs(version, NDE_SUBDIR)

        print("-" * 72)
        print(f"VERSION: {version}")
        print(f"  Classified  DE: {len(de_slugs):3d}   NDE: {len(nde_slugs):3d}")

        # --- Redshift coverage ---
        def z_breakdown(slugs, label):
            has_z  = []
            zero_z = []
            nan_z  = []
            absent = []
            for s in slugs:
                if s not in z_map:
                    absent.append(s)
                else:
                    z = z_map[s]
                    if z is None or (isinstance(z, float) and np.isnan(z)):
                        nan_z.append(s)
                    elif z == 0.0:
                        zero_z.append(s)
                    elif z > 0:
                        has_z.append(s)
                    else:
                        zero_z.append(s)
            print(f"  {label} redshift: z>0={len(has_z)}, z=NaN={len(nan_z)}, "
                  f"z=0={len(zero_z)}, absent={len(absent)}")
            if nan_z:
                print(f"    z=NaN slugs: {nan_z}")
            if zero_z:
                print(f"    z=0 slugs:   {zero_z}")
            if absent:
                print(f"    absent slugs:{absent}")
            return has_z, zero_z, absent

        de_has_z,  de_zero_z,  de_absent  = z_breakdown(de_slugs,  "  DE")
        nde_has_z, nde_zero_z, nde_absent = z_breakdown(nde_slugs, " NDE")

        # --- Missing vs RAW ---
        if version == 'RAW':
            raw_de_slugs  = set(de_slugs)
            raw_nde_slugs = set(nde_slugs)
        else:
            if raw_de_slugs is not None:
                missing_de  = raw_de_slugs  - set(de_slugs)
                missing_nde = raw_nde_slugs - set(nde_slugs)
                extra_de    = set(de_slugs)  - raw_de_slugs
                extra_nde   = set(nde_slugs) - raw_nde_slugs
                if missing_de:
                    print(f"  DE  missing vs RAW ({len(missing_de)}): {sorted(missing_de)}")
                if missing_nde:
                    print(f"  NDE missing vs RAW ({len(missing_nde)}):")
                    for s in sorted(missing_nde):
                        z_val = z_map.get(s, 'absent')
                        if z_val == 'absent':
                            z_str = "absent from CSV"
                        elif z_val is None or (isinstance(z_val, float) and np.isnan(z_val)):
                            z_str = "z=NaN"
                        else:
                            z_str = f"z={z_val:.4f}"
                        print(f"    {s}  ({z_str})")
                if extra_de:
                    print(f"  DE  EXTRA vs RAW  ({len(extra_de)}): {sorted(extra_de)}")
                if extra_nde:
                    print(f"  NDE EXTRA vs RAW  ({len(extra_nde)}): {sorted(extra_nde)}")
                if not missing_de and not missing_nde and not extra_de and not extra_nde:
                    print("  Same source set as RAW.")

        # --- Beam-crop formatted FITS ---
        bc_slugs = list_beamcrop_slugs(version)
        if bc_slugs:
            bc_de_slugs  = bc_slugs & set(de_slugs)
            bc_nde_slugs = bc_slugs & set(nde_slugs)
            missing_bc_de  = set(de_slugs)  - bc_slugs
            missing_bc_nde = set(nde_slugs) - bc_slugs
            extra_bc       = bc_slugs - set(de_slugs) - set(nde_slugs)
            print(f"  Beam-crop FITS  DE: {len(bc_de_slugs):3d}   NDE: {len(bc_nde_slugs):3d}"
                  f"  (unclassified in BC: {len(extra_bc)})")
            if missing_bc_de:
                print(f"  Classified DE missing from beam_crop ({len(missing_bc_de)}): "
                      f"{sorted(missing_bc_de)}")
            if missing_bc_nde:
                print(f"  Classified NDE missing from beam_crop ({len(missing_bc_nde)}): "
                      f"{sorted(missing_bc_nde)}")
        else:
            bc_de_slugs  = set()
            bc_nde_slugs = set()

        # --- Split from classified listing (all slugs) ---
        if de_slugs and nde_slugs:
            split_cls = run_split(de_slugs, nde_slugs)
            print(f"\n  Split (classified listing):")
            print(f"    Total         : {split_cls['n_total']}  "
                  f"(DE={split_cls['n_de']}, NDE={split_cls['n_nde']})")
            print(f"    Test          : {split_cls['test_n']}  "
                  f"(DE={split_cls['test_de']}, NDE={split_cls['test_nde']})")
            print(f"    Train+val     : {split_cls['trainval_n']}  "
                  f"(DE={split_cls['trainval_de']}, NDE={split_cls['trainval_nde']})")
            print(f"    Val  range    : {fmt_range(*split_cls['val_n_range'])}  "
                  f"(DE={fmt_range(*split_cls['val_de_range'])}, "
                  f"NDE={fmt_range(*split_cls['val_nde_range'])})")
            print(f"    Train range   : {fmt_range(*split_cls['train_n_range'])}  "
                  f"(DE={fmt_range(*split_cls['train_de_range'])}, "
                  f"NDE={fmt_range(*split_cls['train_nde_range'])})")
        else:
            split_cls = None

        # --- Split from beam_crop listing ---
        if bc_de_slugs and bc_nde_slugs:
            split_bc = run_split(sorted(bc_de_slugs), sorted(bc_nde_slugs))
            print(f"\n  Split (beam_crop listing):")
            print(f"    Total         : {split_bc['n_total']}  "
                  f"(DE={split_bc['n_de']}, NDE={split_bc['n_nde']})")
            print(f"    Test          : {split_bc['test_n']}  "
                  f"(DE={split_bc['test_de']}, NDE={split_bc['test_nde']})")
            print(f"    Train+val     : {split_bc['trainval_n']}  "
                  f"(DE={split_bc['trainval_de']}, NDE={split_bc['trainval_nde']})")
            print(f"    Val  range    : {fmt_range(*split_bc['val_n_range'])}  "
                  f"(DE={fmt_range(*split_bc['val_de_range'])}, "
                  f"NDE={fmt_range(*split_bc['val_nde_range'])})")
            print(f"    Train range   : {fmt_range(*split_bc['train_n_range'])}  "
                  f"(DE={fmt_range(*split_bc['train_de_range'])}, "
                  f"NDE={fmt_range(*split_bc['train_nde_range'])})")
        else:
            split_bc = None

        print()
        summary[version] = {
            'de_cls': len(de_slugs),
            'nde_cls': len(nde_slugs),
            'de_bc': len(bc_de_slugs),
            'nde_bc': len(bc_nde_slugs),
            'split_cls': split_cls,
            'split_bc': split_bc,
        }

    # -----------------------------------------------------------------------
    # 6. Summary table
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SUMMARY TABLE (tab:dataset_composition equivalent)")
    print("=" * 72)
    print(f"{'Version':<12} {'DE(cls)':>7} {'NDE(cls)':>8} {'Total(cls)':>10} "
          f"{'DE(bc)':>7} {'NDE(bc)':>7} {'Total(bc)':>9} "
          f"{'Test':>12} {'Train+val':>12}")
    print("-" * 90)
    for ver, s in summary.items():
        total_cls = s['de_cls'] + s['nde_cls']
        total_bc  = s['de_bc']  + s['nde_bc']
        sp = s['split_cls']
        sp_bc = s['split_bc']
        test_str    = (f"DE={sp['test_de']},NDE={sp['test_nde']}"     if sp    else "—")
        test_bc_str = (f"DE={sp_bc['test_de']},NDE={sp_bc['test_nde']}" if sp_bc else "—")
        tv_str    = (f"DE={sp['trainval_de']},NDE={sp['trainval_nde']}" if sp    else "—")
        tv_bc_str = (f"DE={sp_bc['trainval_de']},NDE={sp_bc['trainval_nde']}" if sp_bc else "—")
        print(f"{ver:<12} {s['de_cls']:>7} {s['nde_cls']:>8} {total_cls:>10} "
              f"{s['de_bc']:>7} {s['nde_bc']:>7} {total_bc:>9}")
        print(f"{'':12}   classified split test: {test_str}   trainval: {tv_str}")
        if sp_bc:
            print(f"{'':12}   beam_crop  split test: {test_bc_str}   trainval: {tv_bc_str}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 7. Redshift-filtered counts (for "other versions" row)
    # -----------------------------------------------------------------------
    print("\nREDSHIFT-FILTERED COUNTS (only sources with z > 0)")
    print("-" * 50)
    for ver, s in summary.items():
        ver_dir = os.path.join(CLASSIFIED_DIR, ver)
        if not os.path.isdir(ver_dir):
            continue
        de_slugs  = list_classified_slugs(ver, DE_SUBDIR)
        nde_slugs = list_classified_slugs(ver, NDE_SUBDIR)
        def has_z(slug):
            z = z_map.get(slug)
            return z is not None and not (isinstance(z, float) and np.isnan(z)) and z > 0
        de_z  = [s2 for s2 in de_slugs  if has_z(s2)]
        nde_z = [s2 for s2 in nde_slugs if has_z(s2)]
        print(f"  {ver:<10}  DE(z>0)={len(de_z):3d}   NDE(z>0)={len(nde_z):3d}   "
              f"Total={len(de_z)+len(nde_z):3d}")
        if len(de_z) + len(nde_z) > 0:
            sp_z = run_split(de_z, nde_z)
            print(f"             test: DE={sp_z['test_de']}, NDE={sp_z['test_nde']}, "
                  f"total={sp_z['test_n']}")
            print(f"             trainval: DE={sp_z['trainval_de']}, NDE={sp_z['trainval_nde']}, "
                  f"total={sp_z['trainval_n']}")
    print()


if __name__ == '__main__':
    main()
