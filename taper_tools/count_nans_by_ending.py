#!/usr/bin/env python3
"""
Count NaN pixels grouped by SUFFIX TYPE (not individual files).
Shows aggregate statistics for each file type ending.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def count_nans_in_fits(fits_path):
    """Count NaN pixels in a FITS file."""
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                return None, None, None
            
            arr = np.asarray(data, dtype=float)
            arr = np.squeeze(arr)
            
            if arr.ndim == 3:
                arr = np.nanmean(arr, axis=0)
            
            if arr.ndim != 2:
                return None, None, None
            
            total = arr.size
            nan_count = np.isnan(arr).sum()
            nan_pct = (nan_count / total * 100) if total > 0 else 0.0
            
            return total, nan_count, nan_pct
    except Exception as e:
        print(f"Error reading {fits_path}: {e}")
        return None, None, None

def extract_suffix_type(fits_path):
    """
    Extract the generic suffix pattern from filename.
    
    Example transformations:
    PSZ2G023.17+86.71_T25kpcSUB_fmt_128x128_circular.fits -> _T25kpcSUB_fmt_128x128_circular.fits
    PSZ2G091.79-27.00_T100kpc_fmt_128x128_old.fits -> _T100kpc_fmt_128x128_old.fits
    PSZ2G092.71+73.46_RAW_fmt_128x128_circular.fits -> _RAW_fmt_128x128_circular.fits
    
    Strategy: Find the first underscore that starts a known pattern (RAW, RT, T)
    """
    filename = fits_path.name
    
    # Known patterns that start a suffix
    patterns = ['_RAW_', '_RT', '_T']
    
    # Find the first occurrence of any pattern
    for pattern in patterns:
        idx = filename.find(pattern)
        if idx != -1:
            # Return everything from this pattern onward
            return filename[idx:]
    
    # Fallback: if no pattern found, return the whole filename
    # (this handles edge cases)
    return filename

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate NaN statistics by file suffix type"
    )
    
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits"),
        help="Directory containing FITS files"
    )
    
    args = parser.parse_args()
    
    if not args.dir.exists():
        print(f"Error: Directory not found: {args.dir}")
        return
    
    print(f"Scanning {args.dir}...")
    fits_files = list(args.dir.glob("**/*.fits"))
    print(f"Found {len(fits_files)} FITS files\n")
    
    # Group by suffix type
    suffix_stats = defaultdict(lambda: {
        'count': 0,
        'files_with_nans': 0,
        'total_pixels': 0,
        'total_nans': 0,
        'max_nan_pct': 0.0,
        'worst_file': None,
        'sample_files': []  # Store a few example filenames
    })
    
    print("Analyzing files...")
    for fits_path in tqdm(fits_files):
        suffix = extract_suffix_type(fits_path)
        total, nan_count, nan_pct = count_nans_in_fits(fits_path)
        
        if total is None:
            continue
        
        stats = suffix_stats[suffix]
        stats['count'] += 1
        stats['total_pixels'] += total
        stats['total_nans'] += nan_count
        
        # Store up to 3 sample filenames for reference
        if len(stats['sample_files']) < 3:
            stats['sample_files'].append(fits_path.name)
        
        if nan_count > 0:
            stats['files_with_nans'] += 1
            
            if nan_pct > stats['max_nan_pct']:
                stats['max_nan_pct'] = nan_pct
                stats['worst_file'] = fits_path.name
    
    # Print report grouped by suffix
    print("\n" + "="*80)
    print("NaN STATISTICS BY FILE SUFFIX TYPE")
    print("="*80)
    
    # Sort by total NaN count (descending)
    sorted_suffixes = sorted(
        suffix_stats.items(),
        key=lambda x: x[1]['total_nans'],
        reverse=True
    )
    
    for suffix, stats in sorted_suffixes:
        if stats['total_nans'] == 0:
            continue  # Skip clean file types
        
        avg_nan_pct = (stats['total_nans'] / stats['total_pixels'] * 100) if stats['total_pixels'] > 0 else 0.0
        affected_pct = (stats['files_with_nans'] / stats['count'] * 100) if stats['count'] > 0 else 0.0
        
        print(f"\nSuffix: {suffix}")
        print(f"  Total files: {stats['count']}")
        print(f"  Files with NaNs: {stats['files_with_nans']} ({affected_pct:.1f}%)")
        print(f"  Total NaNs: {stats['total_nans']:,}")
        print(f"  Average NaN %: {avg_nan_pct:.2f}%")
        print(f"  Max NaN %: {stats['max_nan_pct']:.2f}%")
        if stats['worst_file']:
            print(f"  Worst file: {stats['worst_file']}")
        print(f"  Sample files: {', '.join(stats['sample_files'][:2])}")
    
    # Summary
    print("\n" + "="*80)
    print("PROBLEMATIC FILE TYPES (>0.1% NaN rate)")
    print("="*80)
    
    problematic = []
    for suffix, stats in suffix_stats.items():
        if stats['total_pixels'] > 0:
            nan_rate = stats['total_nans'] / stats['total_pixels']
            if nan_rate > 0.001:  # >0.1%
                problematic.append((suffix, stats, nan_rate))
    
    if problematic:
        for suffix, stats, nan_rate in sorted(problematic, key=lambda x: x[2], reverse=True):
            print(f"\n  {suffix}")
            print(f"    NaN rate: {nan_rate*100:.2f}%")
            print(f"    Affected: {stats['files_with_nans']}/{stats['count']} files ({stats['files_with_nans']/stats['count']*100:.1f}%)")
    else:
        print("\n  No problematic file types found!")
    
    # Print all suffix types summary
    print("\n" + "="*80)
    print("ALL FILE TYPES SUMMARY (sorted by count)")
    print("="*80)
    
    all_suffixes = sorted(suffix_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    for suffix, stats in all_suffixes:
        avg_nan_pct = (stats['total_nans'] / stats['total_pixels'] * 100) if stats['total_pixels'] > 0 else 0.0
        status = "✓ CLEAN" if stats['total_nans'] == 0 else f"⚠ {avg_nan_pct:.2f}% NaNs"
        print(f"  {suffix:50s} | {stats['count']:4d} files | {status}")

if __name__ == "__main__":
    main()