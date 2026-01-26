# Quick diagnostic script
from pathlib import Path
import numpy as np
from astropy.io import fits

sub_files = list(Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits").glob("**/*_T25kpcSUB_fmt_128x128_circular.fits"))

for f in sub_files[:10]:  # Check first 10
    with fits.open(f) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        print(f"{f.name}:")
        print(f"  min={np.nanmin(data):.6e}, max={np.nanmax(data):.6e}")
        print(f"  mean={np.nanmean(data):.6e}, std={np.nanstd(data):.6e}")
        print(f"  zeros: {(data==0).sum()}, finite: {np.isfinite(data).sum()}/{data.size}")
        
sub_files = list(Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits").glob("**/*_T100kpcSUB_fmt_128x128_circular.fits"))

for f in sub_files[:10]:  # Check first 10
    with fits.open(f) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        print(f"{f.name}:")
        print(f"  min={np.nanmin(data):.6e}, max={np.nanmax(data):.6e}")
        print(f"  mean={np.nanmean(data):.6e}, std={np.nanstd(data):.6e}")
        print(f"  zeros: {(data==0).sum()}, finite: {np.isfinite(data).sum()}/{data.size}")
        
sub_files = list(Path("/users/mbredber/scratch/data/PSZ2/create_image_sets_outputs/processed_psz2_fits").glob("**/*_T100kpcSUB_fmt_128x128_circular_new.fits"))

for f in sub_files[:10]:  # Check first 10
    with fits.open(f) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        print(f"{f.name}:")
        print(f"  min={np.nanmin(data):.6e}, max={np.nanmax(data):.6e}")
        print(f"  mean={np.nanmean(data):.6e}, std={np.nanstd(data):.6e}")
        print(f"  zeros: {(data==0).sum()}, finite: {np.isfinite(data).sum()}/{data.size}")