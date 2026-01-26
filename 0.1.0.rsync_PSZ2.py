#!/usr/bin/env python3
"""
Fixed PSZ2 downloader that re-downloads sources with incomplete FITS file sets.

Changes from original:
1. Checks for expected FITS files (RAW, T25kpc, T50kpc, T100kpc, and SUB variants)
2. Re-downloads if any expected files are missing
3. Option to force re-download all sources
"""

import requests
from bs4          import BeautifulSoup
from urllib.parse import urljoin
import tarfile, os, csv, argparse

BASE     = "https://lofar-surveys.org"
INDEX    = BASE + "/planck_dr2.html"
OUT_ROOT = "data/PSZ2"
FITS_ROOT= os.path.join(OUT_ROOT, "fits")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(FITS_ROOT, exist_ok=True)

# Expected FITS file patterns for a complete download
# Note: Not all sources have all versions, but we check for common ones
EXPECTED_SUFFIXES = [
    '.fits',              # RAW
    'T25kpc.fits',
    'T25kpcSUB.fits',     # Not all sources have this
    'T50kpc.fits',
    'T50kpcSUB.fits',     # Not all sources have this
    'T100kpc.fits',
    'T100kpcSUB.fits',    # Not all sources have this
]

# Minimum expected files (RAW + at least one T version)
MIN_EXPECTED_FILES = 2


def check_complete_download(dest_dir, slug):
    """
    Check if a source directory has a reasonably complete set of FITS files.
    
    A source is considered complete if it has:
    1. RAW file ({slug}.fits)
    2. All three T*kpc files (T25kpc, T50kpc, T100kpc)
    3. If it has ANY SUB file, it should have ALL three SUB files
    
    This stricter check ensures we catch sources missing SUB files when they should have them.
    
    Returns:
        (is_complete, existing_count, missing_files)
    """
    if not os.path.exists(dest_dir):
        return False, 0, EXPECTED_SUFFIXES
    
    existing_fits = [f for f in os.listdir(dest_dir) if f.endswith('.fits')]
    
    if len(existing_fits) < MIN_EXPECTED_FILES:
        return False, len(existing_fits), []
    
    existing_set = set(existing_fits)
    
    # Required files (all sources should have these)
    required_files = [
        f"{slug}.fits",           # RAW
        f"{slug}T25kpc.fits",     # T25kpc
        f"{slug}T50kpc.fits",     # T50kpc
        f"{slug}T100kpc.fits",    # T100kpc
        f"{slug}T25kpcSUB.fits",
        f"{slug}T50kpcSUB.fits",
        f"{slug}T100kpcSUB.fits",
    ]
    
    # Check if required files are present
    has_all_required = all(f in existing_set for f in required_files)
    

    
    # Determine if complete
    if not has_all_required:
        # Missing required files
        missing = [f for f in required_files if f not in existing_set]
        return False, len(existing_fits), missing
    
    # Complete: has all required, and either no SUB or all SUB
    return True, len(existing_fits), []


def download_and_extract(page_url, slug, force=False):
    """
    Download and extract tarball for a source.
    
    Args:
        page_url: URL of the cluster detail page
        slug: Source name (e.g., PSZ2G023.17+86.71)
        force: If True, download even if directory exists
    
    Returns:
        ('downloaded', count) or ('skipped', count) or ('failed', 0)
    """
    dest = os.path.join(FITS_ROOT, slug)
    
    # Check if already complete (unless force=True)
    if not force:
        is_complete, existing_count, missing = check_complete_download(dest, slug)
        if is_complete:
            print(f"✓ {slug} (complete with {existing_count} FITS files)")
            return ('skipped', existing_count)
        elif existing_count > 0:
            print(f"⚠️  {slug} (incomplete: {existing_count} files, missing {len(missing)} expected files) - re-downloading")
    
    # Fetch the cluster detail page
    r = requests.get(page_url)
    if r.status_code != 200:
        print(f"✖️  {slug}: couldn't fetch {page_url}")
        return ('failed', 0)
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Find tarball link
    tar_a = soup.find("a", href=lambda h: h and (h.endswith(".tar") or h.endswith(".tar.gz")))
    if not tar_a:
        print(f"⚠️  {slug}: no tarball link on {page_url}")
        return ('failed', 0)
    
    tar_url = urljoin(page_url, tar_a["href"])
    out_tar = os.path.join(OUT_ROOT, f"{slug}.tar.gz")
    
    print(f"⏬  {slug} (downloading tarball...)")
    
    # Download tarball
    try:
        tr = requests.get(tar_url)
        tr.raise_for_status()
    except Exception as e:
        print(f"❌ {slug}: failed to download tarball: {e}")
        return ('failed', 0)
    
    # Write tarball to disk
    with open(out_tar, "wb") as fd:
        fd.write(tr.content)
    
    # Remove old directory if it exists
    if os.path.exists(dest):
        import shutil
        shutil.rmtree(dest)
    
    # Extract tarball
    os.makedirs(dest, exist_ok=True)
    try:
        with tarfile.open(out_tar, "r:gz") as tf:
            tf.extractall(dest)
    except Exception as e:
        print(f"❌ {slug}: failed to extract tarball: {e}")
        os.remove(out_tar)
        return ('failed', 0)
    
    # Count extracted FITS files
    extracted = os.listdir(dest)
    fits_files = [f for f in extracted if f.endswith('.fits')]
    
    print(f"   → Extracted {len(fits_files)} FITS files: {', '.join(sorted(fits_files))}")
    
    # Clean up tarball
    os.remove(out_tar)
    
    return ('downloaded', len(fits_files))


def main():
    parser = argparse.ArgumentParser(
        description="Download PSZ2 cluster FITS files with completeness checking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download all sources (even if complete)'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check for incomplete downloads, do not download'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("PSZ2 CLUSTER FITS DOWNLOADER")
    print("="*80)
    if args.force:
        print("⚠️  FORCE MODE: Will re-download all sources")
    if args.check_only:
        print("ℹ️  CHECK-ONLY MODE: Will only report incomplete sources")
    print()
    
    # 1) Download + parse main table
    print("Fetching cluster list from LOFAR surveys...")
    r = requests.get(INDEX)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    
    table = soup.find("table")
    header_cells = [th.text.strip() for th in table.find("tr").find_all("th")]
    rows = table.find_all("tr")[1:]  # skip header
    
    cluster_pages = []
    cluster_info = {}
    
    for tr in rows:
        tds = tr.find_all("td")
        a = tr.find("a", href=lambda h: h and "clusters/" in h)
        if not a:
            continue
        
        slug = os.path.basename(a["href"]).replace(".html", "")
        cluster_pages.append((urljoin(BASE, a["href"]), slug))
        
        # Extract metadata
        z = float(tds[4].text.strip()) if tds[4].text.strip() else None
        M500 = float(tds[5].text.strip()) if tds[5].text.strip() else None
        M500_err = float(tds[6].text.strip()) if tds[6].text.strip() else None
        r500 = float(tds[7].text.strip()) if tds[7].text.strip() else None
        r500_err = float(tds[8].text.strip()) if tds[8].text.strip() else None
        classification = tds[11].text.strip() if len(tds) > 11 and tds[11].text.strip() else ""
        
        cluster_info[slug] = {
            'z': z,
            'M500': M500,
            'M500_err': M500_err,
            'r500': r500,
            'r500_err': r500_err,
            'classification': classification,
        }
    
    print(f"Found {len(cluster_pages)} cluster pages\n")
    
    # Check-only mode: just report incomplete sources
    if args.check_only:
        print("="*80)
        print("CHECKING FOR INCOMPLETE DOWNLOADS")
        print("="*80)
        incomplete = []
        for page_url, slug in cluster_pages:
            dest = os.path.join(FITS_ROOT, slug)
            is_complete, count, missing = check_complete_download(dest, slug)
            if not is_complete:
                incomplete.append((slug, count, missing))
                if count == 0:
                    print(f"✗ {slug}: NOT DOWNLOADED")
                else:
                    print(f"⚠️  {slug}: INCOMPLETE ({count} files, missing ~{len(missing)} expected)")
        
        print(f"\n{'='*80}")
        print(f"Found {len(incomplete)} incomplete sources out of {len(cluster_pages)} total")
        print("="*80)
        if incomplete:
            print("\nRun without --check-only to download missing sources")
        return
    
    # 2) Download/extract each source
    stats = {
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'total_fits': 0
    }
    
    for i, (page_url, slug) in enumerate(cluster_pages, 1):
        if i % 50 == 0:
            print(f"\n--- Progress: {i}/{len(cluster_pages)} sources processed ---\n")
        
        status, count = download_and_extract(page_url, slug, force=args.force)
        stats[status] += 1
        if status == 'downloaded':
            stats['total_fits'] += count
    
    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"  ✅ Downloaded: {stats['downloaded']} sources ({stats['total_fits']} FITS files)")
    print(f"  ✓ Skipped (complete): {stats['skipped']}")
    print(f"  ❌ Failed: {stats['failed']}")
    print(f"  📊 Total processed: {len(cluster_pages)}")
    print("="*80)
    
    # Write metadata CSV
    metadata_csv = os.path.join(OUT_ROOT, "cluster_metadata.csv")
    with open(metadata_csv, "w", newline="") as fd:
        w = csv.DictWriter(fd, fieldnames=["slug", "z", "M500", "M500_err", "r500", "r500_err", "classification"])
        w.writeheader()
        for slug, info in cluster_info.items():
            w.writerow({"slug": slug, **info})
    
    print(f"🗒️  Wrote {len(cluster_info)} rows to {metadata_csv}")


if __name__ == "__main__":
    main()