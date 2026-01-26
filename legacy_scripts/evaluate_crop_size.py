import os, glob, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ——— REPLACE this with your image directory ———
image_dir = '/users/mbredber/data/MGCLS/classified_crops_1600/DE'

# ——— (Optional) change this to your preferred PDF name ———
output_pdf = 'all_images_figure.pdf'

# gather all .jpeg/.jpg
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jp*g')))
n = len(image_paths)
if n == 0:
    raise FileNotFoundError(f"No JPEG images found in {image_dir}")

# arrange panels in a near-square grid
cols = max(1, math.ceil(math.sqrt(n)))
rows = math.ceil(n / cols)

# create high-res figure: 4″ per panel at 300 dpi
fig = plt.figure(figsize=(cols*4, rows*4), dpi=300)

for i, path in enumerate(image_paths, start=1):
    # load as grayscale array
    img = np.array(Image.open(path).convert('L'))

    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(img, cmap='gray', origin='lower')
    ax.set_title(os.path.basename(path), fontsize=8)
    ax.axis('off')

    h, w = img.shape
    x0 = (w - 1024) / 2
    y0 = (h - 1024) / 2
    rect = Rectangle((x0, y0), 1024, 1024,
                     edgecolor='white', facecolor='none', linewidth=1)
    rect2 = Rectangle((x0, y0), 128, 128,
                        edgecolor='red', facecolor='none', linewidth=1)
    ax.add_patch(rect)
    ax.add_patch(rect2)

plt.tight_layout()
plt.savefig(output_pdf, format='pdf', dpi=300)
plt.close(fig)
print(f"Saved figure to {output_pdf}")