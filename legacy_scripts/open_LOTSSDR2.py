import os
from astropy.io import fits
import matplotlib.pyplot as plt

# — WITH THE ACTUAL PATH ——
folder = '/users/mbredber/scratch/data/LOTSSDR2/PSZ2G124.20-36.48_FITS'

for fname in os.listdir(folder):
    if fname.endswith('50kpcSUB.fits'):
        fits_path = os.path.join(folder, fname)
        hdu = fits.open(fits_path)[0]
        data = hdu.data
        data = data.squeeze()   # drops singleton dimensions, e.g. (1,1,752,752) → (752,752)

        plt.imshow(data, origin='lower', cmap='gray')
        plt.axis('off')

        outname = fname.replace('.fits', '.png')
        plt.savefig(os.path.join(folder, outname),
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {outname}")
