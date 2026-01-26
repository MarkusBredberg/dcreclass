#!/usr/bin/env python3
"""
Script to count FITS images with NaN values, grouped by filename ending.
Specifically looks at the suffix after '128x128' in the filename.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from astropy.io import fits
import numpy as np

def check_for_nan(filepath):
    """
    Check if a FITS file contains any NaN values.
    
    Args:
        filepath: Path to the FITS file
        
    Returns:
        bool: True if file contains at least one NaN, False otherwise
    """
    try:
        with fits.open(filepath) as hdul:
            # Get the primary data array
            data = hdul[0].data
            
            # Check if data exists and contains NaN
            if data is not None:
                return np.isnan(data).any()
            return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return False

def extract_ending(filename, marker="128x128"):
    """
    Extract the suffix after a specific marker in the filename.
    
    Args:
        filename: Name of the file
        marker: String marker to split on (default: "128x128")
        
    Returns:
        str: The ending after the marker, or empty string if marker not found
    """
    if marker in filename:
        return filename.split(marker, 1)[1]
    return ""

def main():
    # Define the directory to search
    search_dir = "/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits"
    
    # Check if directory exists
    if not os.path.exists(search_dir):
        print(f"Error: Directory {search_dir} does not exist!", file=sys.stderr)
        sys.exit(1)
    
    # Dictionary to store counts: {ending: count_with_nan}
    nan_counts = defaultdict(int)
    
    # Dictionary to store total counts: {ending: total_count}
    total_counts = defaultdict(int)
    
    print(f"Scanning directory: {search_dir}")
    print("This may take a while...\n")
    
    # Find all FITS files with "128x128" in the name
    files_to_check = list(Path(search_dir).glob("*128x128*.fits"))
    
    if not files_to_check:
        print("No files matching pattern '*128x128*.fits' found!")
        sys.exit(0)
    
    print(f"Found {len(files_to_check)} files to check\n")
    
    # Process each file
    for i, filepath in enumerate(files_to_check, 1):
        # Show progress every 100 files
        if i % 100 == 0:
            print(f"Processed {i}/{len(files_to_check)} files...")
        
        filename = filepath.name
        
        # Extract the ending after "128x128"
        ending = extract_ending(filename, "128x128")
        
        # Count this file type
        total_counts[ending] += 1
        
        # Check if file has NaN
        if check_for_nan(filepath):
            nan_counts[ending] += 1
    
    print(f"\nFinished processing {len(files_to_check)} files\n")
    print("=" * 60)
    print("RESULTS: Files with NaN values by ending")
    print("=" * 60)
    
    # Sort endings alphabetically for consistent output
    sorted_endings = sorted(nan_counts.keys())
    
    if not sorted_endings:
        print("No files with NaN values found!")
    else:
        # Print results with both NaN count and total count
        for ending in sorted_endings:
            nan_count = nan_counts[ending]
            total = total_counts[ending]
            percentage = (nan_count / total * 100) if total > 0 else 0
            print(f"{ending:30s}: {nan_count:4d} / {total:4d} ({percentage:5.1f}%)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()