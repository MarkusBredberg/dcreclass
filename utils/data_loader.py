import skimage, cv2, collections, random, math, hashlib, glob, os, re, torch, json
import numpy as np, pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from utils.GAN_models import load_gan_generator
from utils.calc_tools import normalise_images, generate_from_noise, load_model, check_tensor
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpecFromSubplotSpec
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
from collections import Counter, defaultdict
from PIL import Image

# For reproducibility
SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

root_path =  '/users/mbredber/scratch/data/' # '/home/markusbredberg/Scripts/data/'  #

######################################################################################################
################################### GENERAL STUFF ####################################################
######################################################################################################


def get_classes():
    return [
        # GALAXY10 size: 3x256x256
        {"tag": 0, "length": 1081, "description": "Disturbed Galaxies"},
        {"tag": 1, "length": 1853, "description": "Merging Galaxies"},
        {"tag": 2, "length": 2645, "description": "Round Smooth Galaxies"},
        {"tag": 3, "length": 2027, "description": "In-between Round Smooth Galaxies"},
        {"tag": 4, "length": 334, "description": "Cigar Shaped Smooth Galaxies"},
        {"tag": 5, "length": 2043, "description": "Barred Spiral Galaxies"},
        {"tag": 6, "length": 1829, "description": "Unbarred Tight Spiral Galaxies"},
        {"tag": 7, "length": 2628, "description": "Unbarred Loose Spiral Galaxies"},
        {"tag": 8, "length": 1423, "description": "Edge-on Galaxies without Bulge"},
        {"tag": 9, "length": 1873, "description": "Edge-on Galaxies with Bulge"},
        # FIRST size: 300x300
        {"tag": 10, "length": 395, "description": "FRI"},
        {"tag": 11, "length": 824, "description": "FRII"},
        {"tag": 12, "length": 291, "description": "Compact"},
        {"tag": 13, "length": 248, "description": "Bent"},
        # MNIST size: 1x28x28
        {"tag": 14, "length": 60000, "description": "All Digits"},
        {"tag": 15, "length": 60000, "description": "All Digits"},
        {"tag": 16, "length": 60000, "description": "All Digits"},
        {"tag": 17, "length": 60000, "description": "All Digits"},
        {"tag": 18, "length": 60000, "description": "All Digits"},
        {"tag": 19, "length": 60000, "description": "All Digits"},
        {"tag": 20, "length": 6000, "description": "Digit Zero"},
        {"tag": 21, "length": 6000, "description": "Digit One"},
        {"tag": 22, "length": 6000, "description": "Digit Two"},
        {"tag": 23, "length": 6000, "description": "Digit Three"},
        {"tag": 24, "length": 6000, "description": "Digit Four"},
        {"tag": 25, "length": 6000, "description": "Digit Five"},
        {"tag": 26, "length": 6000, "description": "Digit Six"},
        {"tag": 27, "length": 6000, "description": "Digit Seven"},
        {"tag": 28, "length": 6000, "description": "Digit Eight"},
        {"tag": 29, "length": 6000, "description": "Digit Nine"},
        # Radio Galaxy Zoo size: 1x132x132
        {"tag": 31, "length": 10, "description": "1_1"},
        {"tag": 32, "length": 15, "description": "1_2"},
        {"tag": 33, "length": 20, "description": "1_3"},
        {"tag": 34, "length": 12, "description": "2_2"},
        {"tag": 35, "length": 18, "description": "2_3"},
        {"tag": 36, "length": 25, "description": "3_3"},
        # MGCLS 1x1600x1600
        {"tag": 40, "length": 122, "description": "DE"}, # Diffuse Emission (only 52 unique sources)
        {"tag": 41, "length": 90, "description": "NDE"}, # Only 46 unique sources
        {"tag": 42, "length": 13, "description": "RH"}, # Radio Halo
        {"tag": 43, "length": 14, "description": "RR"}, # Radio Relic
        {"tag": 44, "length": 1, "description": "mRH"}, # Mini Radio Halo
        {"tag": 45, "length": 1, "description": "Ph"}, # Phoenix
        {"tag": 46, "length": 4, "description": "cRH"}, # Candidate Radio Halo
        {"tag": 47, "length": 16, "description": "cRR"}, # Candidate Radio Relic
        {"tag": 48, "length": 7, "description": "cmRH"}, # Candidate Mini Radio Halo
        {"tag": 49, "length": 2, "description": "cPh"}, # Candidate Phoenix
        # PSZ2 4x369x369
        {"tag": 50, "length": 62, "description": "DE"}, # RR + RH
        {"tag": 51, "length": 114, "description": "NDE"}, # No Diffuse Emission
        {"tag": 52, "length": 53, "description": "RH"}, # Radio Halo
        {"tag": 53, "length": 20, "description": "RR"}, # Radio Relic (Only 8 unique sources)
        {"tag": 54, "length": 19, "description": "cRH"}, # Candidate Radio Halo
        {"tag": 55, "length": 6, "description": "cRR"}, # Candidate Radio Relic
        {"tag": 56, "length": 24, "description": "cDE"}, # candidate Diffuse Emission
        {"tag": 57, "length": 47, "description": "U"}, # Uncertain
        {"tag": 58, "length": 40, "description": "unclassified"} # Unclassified
    ]


########################################################################################################
####################################### DEBUGGING PLOTTING #############################################
########################################################################################################

def plot_pixel_overlaps_side_by_side(
    train_images, eval_images,
    train_filenames=None, eval_filenames=None,
    max_hashes=20, outdir="./overlap_debug"
):
    """
    For each pixel-identical hash shared by train/test, save a side-by-side figure.
    Title of each panel: 'train — <name>' or 'test — <name>'.
    Works with 2D, 3D (C,H,W), or 4D (T,C,H,W) tensors per image.
    """
    os.makedirs(outdir, exist_ok=True)

    # fallbacks if filenames aren't available
    if not train_filenames: train_filenames = [f"idx {i}" for i in range(len(train_images))]
    if not eval_filenames:  eval_filenames  = [f"idx {i}" for i in range(len(eval_images))]

    # build hash -> indices maps
    train_map, eval_map = {}, {}
    for i, img in enumerate(train_images):
        h = img_hash(img)
        train_map.setdefault(h, []).append(i)
    for j, img in enumerate(eval_images):
        h = img_hash(img)
        eval_map.setdefault(h, []).append(j)

    commons = list(set(train_map) & set(eval_map))
    if not commons:
        print("[overlap-debug] No pixel-identical images between train and test.")
        return 0

    for k, h in enumerate(commons[:max_hashes]):
        t_idxs = train_map[h]
        e_idxs = eval_map[h]
        nrows  = max(len(t_idxs), len(e_idxs))

        fig, axs = plt.subplots(nrows, 2, figsize=(6, 3*nrows))
        if nrows == 1:
            axs = np.array([axs])  # normalize shape

        for r in range(nrows):
            # left column: train
            if r < len(t_idxs):
                ti = t_idxs[r]
                arr = _to_2d_for_imshow(train_images[ti], how="first")
                axs[r, 0].imshow(arr, cmap='viridis', origin='lower')
                axs[r, 0].set_title(f"train — {train_filenames[ti]}", fontsize=10)
            axs[r, 0].axis('off')

            # right column: test
            if r < len(e_idxs):
                ej = e_idxs[r]
                arr = _to_2d_for_imshow(eval_images[ej], how="first")
                axs[r, 1].imshow(arr, cmap='viridis', origin='lower')
                axs[r, 1].set_title(f"test — {eval_filenames[ej]}", fontsize=10)
            axs[r, 1].axis('off')

        fig.suptitle(f"Pixel-identical hash: {h[:12]}…  (train {t_idxs}  |  test {e_idxs})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(outdir, f"overlap_{k:03d}_{h}.png"), dpi=200)
        plt.close(fig)
        print("Plotted overlap at ", os.path.join(outdir, f"overlap_{k:03d}_{h}.png"))

    print(f"[overlap-debug] Wrote {min(len(commons), max_hashes)} figure(s) to {outdir}")
    return len(commons)

def plot_class_images_old(images, labels, filenames=None, set_name='train'):
    # ensure labels are a plain list of ints
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    # if images have more than one channel (C, H, W), only use the first channel
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        images = [img[0] for img in images]
    
    desc_map = {c['tag']: c['description'] for c in get_classes()}
    
    for cls in sorted(set(labels)):
        # collect up to 10 examples of this class
        idxs = [i for i,l in enumerate(labels) if l == cls][:10]
        if not idxs:
            continue
        
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        fig.suptitle(f"{set_name} images for class {cls} – {desc_map.get(cls, '')}", fontsize=12)
        
        for ax, idx in zip(axes.flat, idxs):
            img = images[idx]
            arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
            
            # If multiple channels take the first channel
            if arr.ndim == 3 and arr.shape[0] > 1:
                arr = arr[0]
            
            ax.imshow(arr, cmap='viridis', origin='lower')
            ax.axis('off')
            if filenames and idx < len(filenames):
                ax.set_title(filenames[idx], fontsize=8)
        
        # blank out any unused subplots
        for ax in axes.flat[len(idxs):]:
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"./classifier/processing_step/{cls}_{set_name}_images.png", dpi=300)
        plt.close(fig)
        
# Create the same function as above, but it plots both train and eval sets and both classes side by side for comparison
# I want the top left quadrant to be the first class of train images, top right to be the first class of eval images, bottom left to be the second class of train images, bottom right to be the second class of eval images
def plot_class_images(train_images, eval_images, train_labels, eval_labels, train_filenames=None, eval_filenames=None, set_name='comparison'):
    # ensure labels are a plain list of ints
    if isinstance(train_labels, torch.Tensor):
        print("Converting train_labels tensor to list")
        train_labels = train_labels.tolist()
    if isinstance(eval_labels, torch.Tensor):
        eval_labels = eval_labels.tolist()

    desc_map = {c['tag']: c['description'] for c in get_classes()}
    
    unique_classes = sorted(set(train_labels) | set(eval_labels))
    if len(unique_classes) < 2:
        print("Not enough unique classes to compare.")
        return
    
    class1, class2 = unique_classes[:2]    
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Top-left: train class1 
    axes[0, 0].set_title("Train", fontsize=12)
    axes[0, 0].set_ylabel(f"Class {class1}", fontsize=12)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_frame_on(False)

    idxs1_train = [i for i, l in enumerate(train_labels) if l == class1][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[0, 0].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs1_train):
        img = train_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    # Top-right: eval class1
    axes[0, 1].set_title("Eval", fontsize=12)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 1].set_frame_on(False)

    idxs1_eval = [i for i, l in enumerate(eval_labels) if l == class1][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[0, 1].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs1_eval):
        img = eval_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    # Bottom-left: train class2
    axes[1, 0].set_ylabel(f"Class {class2}", fontsize=12)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_frame_on(False)

    idxs2_train = [i for i, l in enumerate(train_labels) if l == class2][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[1, 0].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs2_train):
        img = train_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    # Bottom-right: eval class2
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_frame_on(False)

    idxs2_eval = [i for i, l in enumerate(eval_labels) if l == class2][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[1, 1].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs2_eval):
        img = eval_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.savefig(f"./classifier/processing_step/{class1}_{class2}_{set_name}_comparison.png", dpi=300)
    plt.close()


#######################################################################################################
################################### DATA AUGMENTATION FUNCTIONS #######################################
#######################################################################################################


def _pix_scales_arcsec(hdr):
    """
    Return pixel scales (px, py) in arcsec/pixel from a FITS header.
    Handles CDELT, CD (rotation/shear), and PC*CDELT conventions.
    """
    def _has(*keys): return all(k in hdr for k in keys)

    # Case 1: CD matrix in deg/pix (rotation/shear allowed)
    if _has('CD1_1','CD1_2','CD2_1','CD2_2'):
        cd11 = float(hdr['CD1_1']); cd12 = float(hdr['CD1_2'])
        cd21 = float(hdr['CD2_1']); cd22 = float(hdr['CD2_2'])
        # scale along x = sqrt( CD1_1^2 + CD2_1^2 ); along y = sqrt( CD1_2^2 + CD2_2^2 )
        # (columns are axis vectors in world units per pixel)
        sx = math.sqrt(cd11*cd11 + cd21*cd21) * 3600.0
        sy = math.sqrt(cd12*cd12 + cd22*cd22) * 3600.0
        return abs(sx), abs(sy)

    # Case 2: PC matrix (unitless) + CDELT in deg/pix
    if _has('PC1_1','PC1_2','PC2_1','PC2_2') and _has('CDELT1','CDELT2'):
        cdelt1 = float(hdr['CDELT1']); cdelt2 = float(hdr['CDELT2'])
        pc11 = float(hdr['PC1_1']); pc12 = float(hdr['PC1_2'])
        pc21 = float(hdr['PC2_1']); pc22 = float(hdr['PC2_2'])
        cd11 = pc11 * cdelt1; cd12 = pc12 * cdelt1
        cd21 = pc21 * cdelt2; cd22 = pc22 * cdelt2
        sx = math.sqrt(cd11*cd11 + cd21*cd21) * 3600.0
        sy = math.sqrt(cd12*cd12 + cd22*cd22) * 3600.0
        return abs(sx), abs(sy)

    # Case 3: plain CDELT in deg/pix (no rotation)
    if _has('CDELT1','CDELT2'):
        return abs(float(hdr['CDELT1'])) * 3600.0, abs(float(hdr['CDELT2'])) * 3600.0

    # Occasional alternative keywords (non-standard but seen in the wild)
    for kx, ky in [('PIXSCAL1','PIXSCAL2'), ('XPIXSCAL','YPIXSCAL')]:
        if _has(kx, ky):
            return abs(float(hdr[kx])), abs(float(hdr[ky]))

    raise KeyError("Cannot determine pixel scale from FITS header (no CD/PC+CDELT/CDELT).")

def _pixdeg(hdr):
    """
    Return a single representative pixel scale in deg/pix,
    using the geometric mean of the x/y scales.
    """
    px_arcsec, py_arcsec = _pix_scales_arcsec(hdr)
    # geometric mean in arcsec/pix → deg/pix
    return math.sqrt(px_arcsec * py_arcsec) / 3600.0

# --- version normalization (+ rtXXkpc single-version mode) ---
def _to_int_if_close(x, tol=1e-6):
    """Return int if x is (nearly) integer; else a compact float string."""
    if abs(x - round(x)) < tol:
        return str(int(round(x)))
    # avoid trailing zeros and scientific unless needed
    s = f"{x:.6f}".rstrip('0').rstrip('.')
    return s

def _canon_ver(v):
    """
    Generalize to:
    RAW / raw / i
    T{num}[unit][SUB]
    RT{num}[unit]
    {num}[unit]  -> T{num}[unit]
    • Accepts punctuation and any case: 'Rt50', 'rt-50 kpc', 't0.2mpc', '25', '25kpc', 't25kpcsub'
    • Units: kpc (default if omitted), mpc (converted to kpc). Others → left as-is.
    • Output normalized to dataset folders: T{N}kpc, RT{N}kpc, T{N}kpcSUB
    """
    s_raw = str(v).strip()
    # strip spaces/underscores/dashes and lower for parsing
    s = re.sub(r'[^0-9a-zA-Z\.]', '', s_raw).lower()

    # RAW aliases
    if s in {'raw', 'i', 'image'}:
        return 'RAW'

    # Try patterns: (rt|t) + number + optional unit + optional SUB
    m = re.match(r'^(rt|t)(\d+(?:\.\d+)?)([a-z]*)?(sub)?$', s)
    if m:
        pref, val_str, unit, sub = m.groups()
        unit = unit or 'kpc'
        try:
            val = float(val_str)
        except ValueError:
            return v  # give up gracefully

        # normalize to kpc for folder names
        if unit == 'mpc':
            val *= 1000.0
            unit = 'kpc'

        # If unit not kpc/mpc, keep it (but your folders are kpc—so we only standardize these)
        if unit in {'kpc'}:
            norm_num = _to_int_if_close(val)
            out = f"{pref.upper()}{norm_num}kpc"
            if sub:
                out += "SUB"
            return out
        else:
            # Unknown unit → keep original semantics but uppercase prefix
            norm_num = _to_int_if_close(val)
            out = f"{pref.upper()}{norm_num}{unit}"
            if sub:
                out += "SUB"
            return out

    # Plain number (w/ optional unit) means T-version
    m2 = re.match(r'^(\d+(?:\.\d+)?)([a-z]*)$', s)
    if m2:
        val_str, unit = m2.groups()
        unit = unit or 'kpc'
        try:
            val = float(val_str)
        except ValueError:
            return v

        if unit == 'mpc':
            val *= 1000.0
            unit = 'kpc'

        if unit in {'kpc'}:
            norm_num = _to_int_if_close(val)
            return f"T{norm_num}kpc"
        else:
            norm_num = _to_int_if_close(val)
            return f"T{norm_num}{unit}"

    # Fall back unchanged if nothing matched
    return v

def _pick_equal_taper_from(versions):
    # versions may be string or list/tuple; return a T* token if any, else default "T50kpc"
    vlist = versions if isinstance(versions, (list, tuple)) else [versions]
    norm  = [_canon_ver(v) for v in vlist]
    for t in norm:
        if str(t).upper().startswith('T'):
            return t
    return "T50kpc"

def _scan_min_beams(base_path, classes, taper):
    nmin = None
    hdrs = {}
    for cls in classes:
        sub = get_classes()[cls]["description"]
        folder = os.path.join(base_path, taper, sub)
        if not os.path.isdir(folder): 
            continue
        for f in os.listdir(folder):
            if not f.lower().endswith(".fits"): 
                continue
            path = os.path.join(folder, f)
            try:
                h = fits.getheader(path)
                fwhm_as = max(float(h["BMAJ"]), float(h["BMIN"])) * 3600.0
                ax, ay = _pix_scales_arcsec(h)
                fovx = int(h["NAXIS1"]) * ax
                fovy = int(h["NAXIS2"]) * ay
                nb = min(fovx, fovy) / max(fwhm_as, 1e-9)
                nmin = nb if (nmin is None) else min(nmin, nb)
                hdrs[os.path.splitext(f)[0]] = h  # cache by basename
            except Exception:
                pass
    return nmin, hdrs

def per_image_percentile_stretch(x, lo=30, hi=99):
    # x: [B, C, H, W]; returns same shape
    B = x.shape[0] # batch size
    out = x.clone() # avoid modifying in place
    for i in range(B):
        flat = out[i].reshape(-1)
        p_low  = flat.quantile(lo/100)
        p_high = flat.quantile(hi/100)
        out[i] = ((out[i] - p_low) / (p_high - p_low + 1e-6)).clamp(0, 1)
    return out


# --- RT (I*G) helpers: world<->pixel, beams, and kernel construction ---
def _cd_matrix_rad(h):
    if 'CD1_1' in h:
        M = np.array([[h['CD1_1'], h.get('CD1_2', 0.0)],
                      [h.get('CD2_1', 0.0), h['CD2_2']]], float)
    else:
        pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
        pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
        cd1 =h.get('CDELT1', 1.0); cd2 =h.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    return M * (np.pi/180.0)

def _fwhm_as_to_sigma_rad(fwhm_as):
    return (float(fwhm_as)/(2.0*np.sqrt(2.0*np.log(2.0)))) * (np.pi/(180.0*3600.0))

def _beam_cov_world(h):
    # requires BMAJ/BMIN in deg; BPA in deg (optional)
    bmaj_as = float(h['BMAJ']) * 3600.0
    bmin_as = float(h['BMIN']) * 3600.0
    pa_deg  = float(h.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th      = np.deg2rad(pa_deg)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], float)
    S = np.diag([sx*sx, sy*sy])
    return R @ S @ R.T

def _beam_solid_angle_sr(h):
    bmaj = abs(float(h['BMAJ'])) * np.pi/180.0
    bmin = abs(float(h['BMIN'])) * np.pi/180.0
    return (np.pi/(4.0*np.log(2.0))) * bmaj * bmin

def _kernel_from_beams(raw_hdr, tgt_hdr):
    # world covariance difference
    C_raw = _beam_cov_world(raw_hdr)
    C_tgt = _beam_cov_world(tgt_hdr)
    C_ker = C_tgt - C_raw
    w, V  = np.linalg.eigh(C_ker); w = np.clip(w, 0.0, None)     # clip tiny negatives
    C_ker = (V * w) @ V.T
    # world → RAW-pixel
    J    = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker @ Jinv.T
    wp, Vp = np.linalg.eigh(Cpix); wp = np.clip(wp, 1e-18, None)
    s_minor = float(np.sqrt(wp[0])); s_major = float(np.sqrt(wp[1]))
    theta   = float(np.arctan2(Vp[1,1], Vp[0,1]))
    nker    = int(np.ceil(8.0*max(s_major, s_minor))) | 1
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

def _to_2d_for_imshow(x, how="first"):
    """
    Return a (H, W) numpy array suitable for plt.imshow from a tensor/ndarray.

    Accepts shapes like:
      (H, W)
      (C, H, W)            or (H, W, C)
      (B, C, H, W)         or (T, C, H, W)
      (B, T, C, H, W)

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Image-like object.
    how : {"first","mean","max"}
        How to reduce non-spatial/extra axes (channels, time, batch).
    """

    def _reduce(a, axis=0):
        if how == "mean":
            return a.mean(axis=axis)
        if how == "max":
            return a.max(axis=axis)
        # "first"
        return np.take(a, 0, axis=axis)

    # ---- convert to numpy float32 without altering values ----
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().float().numpy()
    else:
        a = np.asarray(x, dtype=np.float32)

    # ---- peel dimensions until we have (H, W) ----
    if a.ndim == 2:
        img = a

    elif a.ndim == 3:
        # Heuristic: channels-first if first dim is small (<=4) and last isn't;
        # channels-last if last dim is small (<=4) and first isn't.
        c_first = (a.shape[0] in (1, 2, 3, 4)) and (a.shape[-1] not in (1, 2, 3, 4))
        c_last  = (a.shape[-1] in (1, 2, 3, 4)) and (a.shape[0]  not in (1, 2, 3, 4))

        if c_first:
            # (C, H, W)
            img = a[0] if a.shape[0] == 1 else _reduce(a, axis=0)
        elif c_last:
            # (H, W, C)
            img = a[..., 0] if a.shape[-1] == 1 else _reduce(a, axis=-1)
        else:
            # Ambiguous; take first plane along the leading axis.
            img = _reduce(a, axis=0)

    elif a.ndim == 4:
        # Assume leading axis is batch/time → reduce then recurse.
        img = _to_2d_for_imshow(_reduce(a, axis=0), how=how)

    elif a.ndim == 5:
        # (B, T, C, H, W) → reduce B and T, then recurse.
        a = _reduce(a, axis=0)
        a = _reduce(a, axis=0)
        img = _to_2d_for_imshow(a, how=how)

    else:
        # Fallback: keep reducing the first axis until 2D.
        while a.ndim > 2:
            a = _reduce(a, axis=0)
        img = a

    # Ensure float32 ndarray
    return np.asarray(img, dtype=np.float32)

# Introduce a function that takes the excess images from the evaluation set and adds them to the training set to balance the classes
def move_excess_eval_to_train(train_images, train_labels, eval_images, eval_labels, function_names=None):
    """
    Move excess samples from eval set to train set to balance classes.
    """
    # Count samples per class in both sets
    train_counter = Counter(train_labels)
    eval_counter = Counter(eval_labels)
    
    class_idxs_eval = defaultdict(list)
    for i, lbl in enumerate(eval_labels): # Collect indices per class in eval set
        class_idxs_eval[lbl].append(i)
    
    selected_eval = []
    for cls in train_counter.keys():
        n_train = train_counter[cls]
        n_eval = eval_counter.get(cls, 0)
        if n_eval > n_train:
            n_to_move = n_eval - n_train
            idxs = class_idxs_eval[cls]
            selected_eval.extend(random.sample(idxs, n_to_move))
    
    # Move selected samples from eval to train
    for idx in sorted(selected_eval, reverse=True):
        train_images.append(eval_images[idx])
        train_labels.append(eval_labels[idx])
        if function_names is not None:
            function_names.append(function_names[idx])
        del eval_images[idx]
        del eval_labels[idx]
        if function_names is not None:
            del function_names[idx]
    
    print(f"Moved {len(selected_eval)} samples from eval to train to balance classes.")
    
    if function_names is not None:
        return train_images, train_labels, eval_images, eval_labels, function_names
    
    return train_images, train_labels, eval_images, eval_labels

def balance_classes(images, labels, function_names=None):
    """
    Randomly down‐sample each class so they all have the same number of samples
    equal to the size of the smallest class.
    """
    # Print the number of samples per class before balancing
    counter = Counter(labels)
    print("Class distribution before balancing:", dict(counter))
    
    class_idxs = defaultdict(list)
    for i, lbl in enumerate(labels): # Collect indices per class
        class_idxs[lbl].append(i)
    min_n = min(len(idxs) for idxs in class_idxs.values())  # find smallest class size
    selected = []
    for idxs in class_idxs.values():
        selected.extend(random.sample(idxs, min_n))
    random.shuffle(selected)
    
    counter_after = Counter([labels[i] for i in selected])
    print("Class distribution after balancing:", dict(counter_after))
    
    if function_names is not None:
        return [images[i] for i in selected], [labels[i] for i in selected], [function_names[i] for i in selected]
    
    return [images[i] for i in selected], [labels[i] for i in selected]


def redistribute_excess(train_images, eval_images, train_labels, eval_labels,
                        train_filenames=None, eval_filenames=None):
    """
    Balance the evaluation set by moving excess images from the larger class(es) 
    to the training set, ensuring equal representation of each class in eval.
    """
    
    # Work with tensors directly
    if isinstance(train_labels, list):
        train_labels = torch.tensor(train_labels, dtype=torch.long)
    if isinstance(eval_labels, list):
        eval_labels = torch.tensor(eval_labels, dtype=torch.long)
    
    train_fnames = train_filenames if train_filenames else []
    eval_fnames = eval_filenames if eval_filenames else []

    # Group eval samples by class using tensor operations
    eval_bins = defaultdict(list)
    for i in range(len(eval_images)):
        lbl = int(eval_labels[i].item())
        eval_bins[lbl].append(i)
    
    target_classes = sorted(set(eval_labels.tolist()))
    
    # Print before
    #print(f"Samples per each unique class in the evaluation set before redistribution: "
    #      f"{Counter(eval_labels.tolist())}")
    
    # Find minimum class size
    min_count = min(len(eval_bins[cls]) for cls in target_classes if cls in eval_bins)
    keep_idxs, move_idxs = [], []
    for cls in target_classes:
        if cls not in eval_bins:
            continue
        
        idxs = eval_bins[cls]
        
        # Sort by content hash for deterministic selection
        items_sorted = sorted(idxs, key=lambda i: img_hash(eval_images[i]))
        
        # Debug: Print which samples are being kept/moved
        if eval_fnames:
            kept_names = [eval_fnames[items_sorted[j]] for j in range(min(min_count, len(items_sorted)))]
            moved_names = [eval_fnames[items_sorted[j]] for j in range(min_count, len(items_sorted))]
            
            # Sort images according to filenames for easier readability
            kept_names.sort()
            moved_names.sort()
            
            print(f"Class {cls}: Keeping {len(kept_names)} samples: {kept_names[:15]}{'...' if len(kept_names) > 15 else ''}")
            #if moved_names:
            #    print(f"Class {cls}: Moving {len(moved_names)} samples to train: {moved_names[:15]}{'...' if len(moved_names) > 15 else ''}")
            
        
        keep_idxs.extend(items_sorted[:min_count])
        move_idxs.extend(items_sorted[min_count:])
    
    # Use tensor indexing - much faster than list operations
    new_eval_imgs = eval_images[keep_idxs]
    new_eval_lbls = eval_labels[keep_idxs]
    new_eval_fnames = [eval_fnames[i] for i in keep_idxs] if eval_fnames else []
    
    # Move excess to train using cat
    if move_idxs:
        to_move_imgs = eval_images[move_idxs]
        to_move_lbls = eval_labels[move_idxs]
        to_move_fnames = [eval_fnames[i] for i in move_idxs] if eval_fnames else []
        
        final_train_imgs = torch.cat([train_images, to_move_imgs], dim=0)
        final_train_lbls = torch.cat([train_labels, to_move_lbls], dim=0)
        final_train_fnames = (list(train_fnames) if train_fnames else []) + to_move_fnames
    else:
        final_train_imgs = train_images
        final_train_lbls = train_labels
        final_train_fnames = train_fnames if train_fnames else []
    
    # Print after
    print(f"Samples per each unique class in evaluation set after redistribution: "
          f"{Counter(new_eval_lbls.tolist())}")
    
    return (
        final_train_imgs, new_eval_imgs, 
        final_train_lbls, new_eval_lbls,
        final_train_fnames, new_eval_fnames
    )

# Using systematic transformations instead of random choices in augmentation
def apply_transforms_with_config(image, config):
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: transforms.functional.hflip(x) if config['flip_h'] else x),  # Horizontal flip
        transforms.Lambda(lambda x: transforms.functional.vflip(x) if config['flip_v'] else x),  # Vertical flip
        transforms.Lambda(lambda x: transforms.functional.rotate(x, config['rotation']))  # Rotation by specific angle
    ])
    
    if image.dim() == 2:  # Ensure image is 3D
        image = image.unsqueeze(0)
    transformed_image = preprocess(image) 
    return transformed_image

def apply_formatting(image: torch.Tensor,
                     crop_size: tuple = (1, 128, 128),
                     downsample_size: tuple = (1, 128, 128)
                    ) -> torch.Tensor:
    """
    Center-crop and resize a single-channel tensor without PIL.

    Args:
      image: Tensor of shape [C, H0, W0] or [1, H0, W0].
      crop_size:      (C,Hc,Wc) or (Hc,Wc) or (T,C,Hc,Wc) → will be canonicalized.
      downsample_size:(C,Ho,Wo) or (Ho,Wo) or (T,C,Ho,Wo) → will be canonicalized.

    Returns:
      Tensor of shape [C, Ho, Wo].
    """

    # Canonicalize sizes to (C,H,W)
    def _canon_size(sz):
        if len(sz) == 2:
            return (1, sz[0], sz[1])
        if len(sz) == 3:
            return sz
        if len(sz) == 4:
            return (sz[-3], sz[-2], sz[-1])
        raise ValueError(f"crop/downsample size must have 2, 3 or 4 dims, got {sz}")

    crop_size = _canon_size(crop_size)
    downsample_size = _canon_size(downsample_size)

    # Normalize image dims
    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)               # [1,H0,W0]
    if image.dim() == 3:
        C, H0, W0 = image.shape
        img = image
    elif image.dim() == 2:
        H0, W0 = image.shape
        img = image.unsqueeze(0)
        C = 1
    else:
        raise ValueError(f"Unexpected image dims: {image.shape}")

    # Grayscale handling based on canonicalized channel dim
    if crop_size[0] == 1 or downsample_size[0] == 1:
        img = img.mean(dim=0, keepdim=True)

    # Unpack sizes
    _, Hc, Wc = crop_size
    _, Ho, Wo = downsample_size

    # Center crop and resize
    y0, x0 = H0 // 2, W0 // 2
    y1, y2 = y0 - Hc // 2, y0 + Hc // 2
    x1, x2 = x0 - Wc // 2, x0 + Wc // 2
    y1, y2 = max(0, y1), min(H0, y2)
    x1, x2 = max(0, x1), min(W0, x2)

    crop = img[:, y1:y2, x1:x2].unsqueeze(0)   # [1,C,Hc,Wc]
    resized = F.interpolate(crop, size=(Ho, Wo), mode='bilinear') # bilinear or area
    return resized.squeeze(0)                   # [C,Ho,Wo]

def img_hash(img: torch.Tensor) -> str:
    arr = img.cpu().contiguous().numpy()
    returnval = hashlib.sha1(arr.tobytes()).hexdigest()
    return returnval
        
def augment_images(
    images, labels, rotations = [0, 90, 180, 270],    #rotations = np.arange(0, 360, 20).tolist(),
    flips = [(False, False), (True, False)], mem_threshold=1000,
    #translations = [(10, 0), (-10, 0), (0, 10), (0, -10)], #[(5, 0), (-5, 0), (0, 5), (0, -5)],
    translations = [(0, 0)], 
    ST_augmentation=False, n_gen = 1):
    """
    General function to augment images in chunks with memory optimization.

    Args:
        images (list or tensor): List or tensor of input images.
        labels (list or tensor): Corresponding labels for the images.
        img_shape (tuple): Shape of the input images.
        rotations (list): List of rotation angles in degrees.
        flips (list of tuples): List of tuples specifying horizontal and vertical flips.
        brightness_adjustments (list, optional): List of brightness adjustment factors. Default is None.

    Returns:
        tuple: Augmented images and labels as tensors.
    """
    
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)  # handles list/ndarray of ints or 1-hot rows

    label_dtype  = labels.dtype
    label_device = labels.device

    # — normalize all inputs to exactly 3D (C=1, H, W) —
    normed = []
    for img in images:
        if isinstance(img, torch.Tensor):
            # if someone passed a “batch” dim of 1, remove it
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)
            # if they somehow gave you a plain 2D H×W, make it (1,H,W)
            if img.dim() == 2:
                img = img.unsqueeze(0)
        normed.append(img)
    images = normed

    # Initialize empty lists for results
    augmented_images, augmented_labels = [], []
    cumulative_augmented_images, cumulative_augmented_labels = [], []
    
    if ST_augmentation:
        # labels may be a tensor; make them plain ints
        lbl_list = [int(x) for x in (labels.tolist() if torch.is_tensor(labels) else labels)]
        for cls in sorted(set(lbl_list)):
            # don't depend on exact count in the filename; use a pattern
            pattern = f"/users/mbredber/scratch/ST_generation/1to{n_gen}_*_{cls}.npy"
            candidates = sorted(glob.glob(pattern))
            if not candidates:
                print(f"[augment_images] No ST file matching {pattern}; skipping for class {cls}.")
                continue
            st_images = np.load(candidates[0])
            st_images = torch.tensor(st_images).float().unsqueeze(1)
            images.extend(st_images)
            lbl_list.extend([cls]*len(st_images))
        labels = lbl_list
    
    for idx, image in enumerate(images):
        for rot in rotations:
            for flip_h, flip_v in flips:
                for translation in translations:
                    if translation != (0, 0):
                        image = transforms.functional.affine(
                            image, angle=0, translate=translation, scale=1.0, shear=0, fill=0
                        )
                    config = {'rotation': rot, 'flip_h': flip_h, 'flip_v': flip_v}
                    augmented_image = apply_transforms_with_config(image.clone().detach(), config)
                    augmented_images.append(augmented_image)
                    augmented_labels.append(labels[idx])  # Append corresponding label

                    # Memory check: Save and clear if too many augmentations are in memory
                    if len(augmented_images) >= mem_threshold:  # Threshold for saving (adjustable)
                        cumulative_augmented_images.extend(augmented_images)
                        cumulative_augmented_labels.extend(augmented_labels)
                        augmented_images, augmented_labels = [], []  # Reset batch

    # Extend cumulative lists with remaining augmented images from the chunk
    cumulative_augmented_images.extend(augmented_images)
    cumulative_augmented_labels.extend(augmented_labels)
    
    # Convert cumulative lists to tensors
    augmented_images_tensor = torch.stack(cumulative_augmented_images)
    augmented_labels_tensor = torch.tensor(cumulative_augmented_labels)
    if len(cumulative_augmented_labels) == 0:
            augmented_labels_tensor = torch.empty((0,) + labels.shape[1:], 
                                                dtype=label_dtype, device=label_device)
    else:
        first = cumulative_augmented_labels[0]
        if isinstance(first, torch.Tensor):
            augmented_labels_tensor = torch.stack(
                [x.to(dtype=label_dtype, device=label_device) 
                for x in cumulative_augmented_labels], dim=0)
        else:
            augmented_labels_tensor = torch.tensor(
                cumulative_augmented_labels, dtype=label_dtype, device=label_device)

    return augmented_images_tensor, augmented_labels_tensor


##########################################################################################
################################# CACHE FUNCTIONS ########################################
##########################################################################################



def _build_cache_key(galaxy_classes, versions, fold, crop_size, downsample_size, 
                     sample_size, REMOVEOUTLIERS, BALANCE, AUGMENT, STRETCH,
                     percentile_lo, percentile_hi, NORMALISE, NORMALISETOPM,
                     USE_GLOBAL_NORMALISATION, GLOBAL_NORM_MODE, PREFER_PROCESSED, train):
    """
    Create a unique cache key based on all parameters that affect the loaded data.
    
    Returns:
        str: A hash-based cache key
    """
    # Normalize versions to a consistent string representation
    if isinstance(versions, (list, tuple)):
        ver_str = "_".join(sorted([_canon_ver(v) for v in versions]))
    else:
        ver_str = _canon_ver(versions)
    
    # Build parameter dictionary with all relevant settings
    params = {
        'galaxy_classes': sorted(galaxy_classes) if isinstance(galaxy_classes, list) else [galaxy_classes],
        'versions': ver_str,
        'fold': fold,
        'crop_size': crop_size,
        'downsample_size': downsample_size,
        'sample_size': sample_size,
        'REMOVEOUTLIERS': REMOVEOUTLIERS,
        'BALANCE': BALANCE,
        'AUGMENT': AUGMENT,
        'STRETCH': STRETCH,
        'percentile_lo': percentile_lo,
        'percentile_hi': percentile_hi,
        'NORMALISE': NORMALISE,
        'NORMALISETOPM': NORMALISETOPM,
        'USE_GLOBAL_NORMALISATION': USE_GLOBAL_NORMALISATION,
        'GLOBAL_NORM_MODE': GLOBAL_NORM_MODE,
        'PREFER_PROCESSED': PREFER_PROCESSED,
        'train': train
    }
    
    # Create a deterministic string from parameters and hash it
    param_str = json.dumps(params, sort_keys=True)
    cache_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    # Build readable prefix
    classes_str = "_".join(map(str, params['galaxy_classes']))
    prefix = f"cache_cls{classes_str}_ver{ver_str}_f{fold}"
    
    return f"{prefix}_{cache_hash}"


def _save_cache(cache_key, data_tuple, cache_dir="./.cache/data"):
    """
    Save processed data to cache.
    
    Args:
        cache_key: Unique identifier for this cache entry
        data_tuple: Tuple of (train_images, train_labels, eval_images, eval_labels, [train_fns, eval_fns])
        cache_dir: Directory to store cache files
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
    
    # Convert data to saveable format
    save_dict = {
        'train_images': data_tuple[0],
        'train_labels': data_tuple[1],
        'eval_images': data_tuple[2],
        'eval_labels': data_tuple[3]
    }
    
    # Add filenames if present (6-tuple)
    if len(data_tuple) == 6:
        save_dict['train_filenames'] = data_tuple[4]
        save_dict['eval_filenames'] = data_tuple[5]
    
    try:
        torch.save(save_dict, cache_path)
        print(f"✓ Saved data cache to {cache_path}")
    except Exception as e:
        print(f"⚠ Failed to save cache: {e}")


def _load_cache(cache_key, cache_dir="./.cache/data"):
    """
    Load processed data from cache if it exists.
    
    Args:
        cache_key: Unique identifier for this cache entry
        cache_dir: Directory where cache files are stored
    
    Returns:
        Tuple of data if cache exists, None otherwise
    """
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
    
    if not os.path.isfile(cache_path):
        return None
    
    try:
        print(f"✓ Loading data from cache: {cache_path}")
        save_dict = torch.load(cache_path)
        
        # Check if filenames are present
        if 'train_filenames' in save_dict and 'eval_filenames' in save_dict:
            return (
                save_dict['train_images'],
                save_dict['train_labels'],
                save_dict['eval_images'],
                save_dict['eval_labels'],
                save_dict['train_filenames'],
                save_dict['eval_filenames']
            )
        else:
            return (
                save_dict['train_images'],
                save_dict['train_labels'],
                save_dict['eval_images'],
                save_dict['eval_labels']
            )
    except Exception as e:
        print(f"⚠ Failed to load cache (will regenerate): {e}")
        return None    
    
    
##########################################################################################
################################## SPECIFIC DATASET LOADER ###############################
##########################################################################################

def load_PSZ2(
    path = root_path + "PSZ2/classified/",
    sample_size = 300,              # per class in training set; eval uses sample_size*0.2
    target_classes = [50, 51],
    versions = "T100kpcSUB",          # string or list/tuple; list => Multiple versions
    crop_size = (1, 512, 512),        # (C,Hc,Wc) — angular FoV is taken from the ref version
    downsample_size = (1, 128, 128),  # (C,Ho,Wo) — output per frame
    fold = 0,                         # 0..4 = CV folds, 5 = last split
    train = False,                    # Not implemented
    processed_dir = "/users/mbredber/scratch/create_image_sets_outputs/processed_psz2_fits", # directory for preformatted images
    prefer_processed = True,   # whether to prefer processed images when available
    gate_with = None  # None | "auto" | "T50kpc" | 50 (number). If set, gate RAW/T/RT against this T directory.
):
    # Data available at: https://lofar-surveys.org/planck_dr2.html
    # 1. Fetch data with utils.download_PSZ2.py
    # 2. Categorise data with utils.process_PSZ2.py
    # 3. Format and taper data with taper_tools.create_image_sets.py
    # This is the data loader for any version of the processed data

    print("Parameters:")
    print("  path:", path)
    print("  versions:", versions)
    print("  crop_size:", crop_size)
    print("  downsample_size:", downsample_size)
    print("  target_classes:", target_classes)
    print("  processed_dir:", processed_dir)
    print("  prefer_processed:", prefer_processed)
    print("  gate_with:", gate_with)
    print("  train:", train)
    print("  fold:", fold)

    def _kpc_tag(v):
        # Find the canonical kpc tag for a version string
        vU = str(v).upper()
        if vU.startswith("RT"):
            num = ''.join(c for c in str(v) if c.isdigit())
            return f"RT{num}kpc"
        if vU.startswith("T"):
            # allow inputs like T50kpc or T50kpcSUB → T50kpc
            m = re.search(r'T(\d+)kpc', vU)
            if m:
                return f"T{m.group(1)}kpc"
        return str(v)
    
    def _nearest_T_dir(root_path, subfolder, target_num):
        cand = []
        for d in os.listdir(root_path):
            if not d.upper().startswith("T"):
                continue
            m = re.search(r"T(\d+(?:\.\d+)?)KPC", d.upper())
            if not m:
                continue
            try:
                y = float(m.group(1))
            except Exception:
                continue
            dir_sub = os.path.join(root_path, d, subfolder)
            if os.path.isdir(dir_sub):
                cand.append((abs(y - target_num), y, d))
        if not cand:
            return None
        cand.sort(key=lambda t: (t[0], t[1]))
        return cand[0][2]  # folder name like "T50kpc"
    
    # --- shapes ---
    def _canon(size):
        if len(size) == 2: return (1, size[0], size[1])
        if len(size) == 3: return size
        raise ValueError("crop_size/downsample_size must be (H,W) or (C,H,W)")
    ch_c, Hc_ref, Wc_ref = _canon(crop_size)
    ch_d, Ho, Wo         = _canon(downsample_size)
            
    # ----- equal-beams pre-scan to make images similar in beam counts -----
    # Disable equal-beam logic when matching loader-2
    if prefer_processed:
        EQUAL_TAPER = _pick_equal_taper_from(versions)  # T50kpc by default
        n_beams_min, _T_header_cache = _scan_min_beams(path, target_classes, taper=EQUAL_TAPER)
    else:
        n_beams_min, _T_header_cache = None, {}

    # --- class → folder map ---
    classes_map = {c["tag"]: c["description"] for c in get_classes()}

    images, labels, basenames = [], [], []
    def _source_id(base):
        # strip any tail that starts with TXXkpc, e.g. T50kpc, T50kpcSUB, T50kpc_resid
        return re.sub(r'T\d+kpc.*$', '', base)
    _seen_sources = set()
    
    # ============= multi-version stack =============
    if isinstance(versions, (list, tuple)) and len(versions) > 1:
        # prefer a taper present in versions; else RAW
        versions = list(versions) if isinstance(versions, (list, tuple)) else [versions]
        tapers = [v for v in versions if str(v).upper().startswith("T")]
        ref_version = tapers[0] if tapers else ("RAW" if any(str(v).upper() == "RAW" for v in versions) else str(versions[0]))

        def _list_bases(ver, sub):
            folder = os.path.join(path, ver, sub)
            if not os.path.isdir(folder):
                return folder, set()
            files = os.listdir(folder)
            bases = {os.path.splitext(f)[0] for f in files if f.upper().endswith(".fits")}
            return folder, bases

        for cls in target_classes:
            sub = classes_map.get(cls)
            if not sub:
                continue

            folder_map, base_sets = {}, []
            for vf in versions:
                vfU = str(vf).upper()
                if vfU.startswith("RT"):
                    # derive gate version, e.g. RT50kpc -> T50kpc
                    num = ''.join([c for c in str(vf) if c.isdigit()])
                    gate = f"RT{num}kpc"

                    # list bases from RAW and gate; RT cubes are built from RAW convolved to gate beam
                    folder_raw,  bases_raw  = _list_bases("RAW",  sub)
                    folder_gate, bases_gate = _list_bases(gate,   sub)
                    folder_map["RAW"] = folder_raw
                    folder_map[gate]  = folder_gate
                    base_sets.append(bases_raw & bases_gate)     # require presence in both RAW and gate to be eligible

                    # optional: point vf to RAW for ref-only lookups (we never read vf directly for RT)
                    folder_map[vf] = folder_raw
                else:
                    folder, bases = _list_bases(vf, sub)
                    folder_map[vf] = folder
                    base_sets.append(bases)

            # intersection over all version requirements (for RT entries this was RAW∩gate)
            common = sorted(set.intersection(*base_sets)) if base_sets else []
            
            for base in common:
                # ref header/pixscale defines the angular FoV for cropping other versions
                ref_path = os.path.join(folder_map[ref_version], f"{base}.fits")
                ref_hdr = fits.getheader(ref_path)
                ref_pix = _pixdeg(ref_hdr)   # deg/px

                frames, ok = [], True
                for vf in versions:
                    vfU = str(vf).upper()

                    # === T/RT: prefer processed file; else generate ===
                    if vfU.startswith("T") or vfU.startswith("RT"):
                        src_name = _source_id(base)
                        tag = _kpc_tag(vfU)  # e.g. RT50kpc or T50kpc
                        Hwant, Wwant = Ho, Wo  # preformatted target size
                        proc_path = os.path.join(
                            processed_dir,
                            f"{src_name}_{tag}_fmt_{Hwant}x{Wwant}.fits"
                        )

                        if os.path.isfile(proc_path):
                            arr = np.squeeze(fits.getdata(proc_path)).astype(np.float32)
                            if arr.ndim == 3:
                                arr = arr.mean(axis=0)
                            if arr.ndim != 2:
                                ok = False
                                break
                            ten = torch.from_numpy(arr).unsqueeze(0).float()  # already formatted
                            frames.append(ten)
                            continue

                        # --- processed file missing: generate like the montage script ---
                        if vfU.startswith("T"):
                            # read native T image and format to ref FoV
                            fpath = os.path.join(folder_map[vf], f"{base}.fits")
                            arr = np.squeeze(fits.getdata(fpath)).astype(np.float32)
                            if arr.ndim == 3:
                                arr = arr.mean(axis=0)
                            if arr.ndim != 2:
                                ok = False
                                break
                            hdr = fits.getheader(fpath)

                            if n_beams_min is not None:
                                fwhm_as = max(float(hdr["BMAJ"]), float(hdr["BMIN"])) * 3600.0
                                side_as = n_beams_min * fwhm_as
                                px, py = _pix_scales_arcsec(hdr)
                                Hc_eff = max(1, int(round(side_as / py)))
                                Wc_eff = max(1, int(round(side_as / px)))
                            else:
                                ref_hdr = fits.getheader(ref_path)
                                ref_pix = _pixdeg(ref_hdr)
                                pix = _pixdeg(hdr)
                                Hc_eff = max(1, int(round(Hc_ref * (ref_pix / pix))))
                                Wc_eff = max(1, int(round(Wc_ref * (ref_pix / pix))))

                            ten = torch.from_numpy(arr).unsqueeze(0).float()
                            frm = apply_formatting(ten, crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))
                            frames.append(frm)
                            continue

                        # RT fallback: convolve RAW to T-beam and format
                        num = ''.join([c for c in vfU if c.isdigit()])
                        gate_T = f"T{num}kpc"
                        raw_path = os.path.join(folder_map["RAW"], f"{base}.fits")
                        txx_path = os.path.join(folder_map[gate_T], f"{base}.fits")
                        if not (os.path.isfile(raw_path) and os.path.isfile(txx_path)):
                            ok = False
                            break

                        raw_arr = np.squeeze(fits.getdata(raw_path)).astype(np.float32)
                        if raw_arr.ndim == 3:
                            raw_arr = raw_arr.mean(axis=0)
                        txx_hdr = fits.getheader(txx_path)
                        raw_hdr = fits.getheader(raw_path)

                        ker = _kernel_from_beams(raw_hdr, txx_hdr)
                        rt_arr = convolve_fft(
                            raw_arr, ker, boundary="fill", fill_value=np.nan,
                            nan_treatment="interpolate", normalize_kernel=True,
                            psf_pad=True, fft_pad=True, allow_huge=True
                        )
                        rt_arr *= (_beam_solid_angle_sr(txx_hdr) / _beam_solid_angle_sr(raw_hdr))

                        if n_beams_min is not None:
                            fwhm_as = max(float(txx_hdr["BMAJ"]), float(txx_hdr["BMIN"])) * 3600.0
                            side_as = n_beams_min * fwhm_as
                            px, py = _pix_scales_arcsec(raw_hdr)
                            Hc_eff = max(1, int(round(side_as / py)))
                            Wc_eff = max(1, int(round(side_as / px)))
                        else:
                            ref_hdr = fits.getheader(ref_path)
                            ref_pix = _pixdeg(ref_hdr)
                            raw_pix = _pixdeg(raw_hdr)
                            Hc_eff = max(1, int(round(Hc_ref * (ref_pix / raw_pix))))
                            Wc_eff = max(1, int(round(Wc_ref * (ref_pix / raw_pix))))

                        ten = torch.from_numpy(rt_arr).unsqueeze(0).float()
                        frm = apply_formatting(ten, crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))
                        frames.append(frm)

                        if not ok or not frames:
                            continue

                        cube = torch.stack(frames, dim=0)  # [T,1,Ho,Wo]
                        images.append(cube)
                        labels.append(cls)
                        basenames.append(_source_id(base))

    # ============= SINGLE-VERSION PATH (optionally rtXXkpc) =============
    else:
        vU = versions[0].upper() if isinstance(versions, (list, tuple)) else str(versions).upper()
        tag = _kpc_tag(versions)
        for cls in target_classes:
            sub = classes_map.get(cls)
            #print("Processing class:", cls, "subfolder:", sub)
            if not sub:
                continue

            # verify RAW folder exists
            raw_dir = os.path.join(path, "RAW", sub)
            if not os.path.isdir(raw_dir):
                print(f"[SKIP] RAW folder missing: {raw_dir}")
                continue

            # --- Unified gating: applies to RAW/T/RT if gate_with is set ---
            gate_keys = None
            gate_dirname = None
            if gate_with is not None:
                # resolve desired gate kpc
                desired_num = None
                if isinstance(gate_with, str) and gate_with.lower() == "auto":
                    # derive from versions if T/RT, default to 50 for RAW
                    m_rt = re.search(r"RT(\d+(?:\.\d+)?)", vU)
                    m_t  = re.search(r"T(\d+(?:\.\d+)?)KPC", vU)
                    if m_rt:
                        desired_num = float(m_rt.group(1))
                    elif m_t:
                        desired_num = float(m_t.group(1))
                    else:
                        desired_num = 50.0  # RAW default
                elif isinstance(gate_with, (int, float)):
                    desired_num = float(gate_with)
                elif isinstance(gate_with, str):
                    m = re.search(r"T(\d+(?:\.\d+)?)KPC", gate_with.upper())
                    if m: desired_num = float(m.group(1))

                if desired_num is not None:
                    preferred_gate = f"T{int(desired_num) if desired_num.is_integer() else desired_num}kpc"
                    # pick exact or nearest available TXXkpc/<sub>
                    if os.path.isdir(os.path.join(path, preferred_gate, sub)):
                        gate_dirname = preferred_gate
                    else:
                        nearest = _nearest_T_dir(path, sub, desired_num)
                        gate_dirname = nearest

                if gate_dirname is None:
                    print(f"[GATE] No suitable TXXkpc found for gating with '{gate_with}' in sub='{sub}'. Proceeding without gating.")
                else:
                    gate_dir = os.path.join(path, gate_dirname, sub)
                    raw_map = {os.path.splitext(f)[0].lower(): os.path.splitext(f)[0] for f in os.listdir(raw_dir) if f.lower().endswith(".fits")}            
                    txx_map = {os.path.splitext(f)[0].lower(): os.path.splitext(f)[0] for f in os.listdir(gate_dir) if f.lower().endswith(".fits")}
                    gate_keys = set(raw_map) & set(txx_map)  # lowercase intersection only
                    print(f"[GATE] Using {gate_dirname} for sub='{sub}' ({len(gate_keys)} sources intersect).")

            for fname in sorted(os.listdir(raw_dir)):
                if not fname.lower().endswith(".fits"):
                    print("Skipping non-FITS file:", fname)
                    continue
                base = os.path.splitext(fname)[0]
                src  = _source_id(base)
                if src in _seen_sources:
                    print("Skipping already seen source:", src)
                    continue
                if gate_keys is not None and (base.lower() not in gate_keys):
                    print("Skipping (not in gated intersection):", src)
                    continue
                
                # === Prefer processed T/RT; else generate ===
                use_processed = bool(prefer_processed) and (vU.startswith("T") or vU.startswith("RT") or vU == "RAW")
                if use_processed:
                    src_name = _source_id(base)
                    Hwant, Wwant = Ho, Wo
                    proc_path = os.path.join(
                        processed_dir,
                        f"{src_name}_{tag}_fmt_{Hwant}x{Wwant}_old.fits" # This is the used file name
                    )
                    if os.path.isfile(proc_path):
                        arr = np.squeeze(fits.getdata(proc_path)).astype(np.float32)
                        if arr.ndim == 3: arr = arr.mean(axis=0)
                        if arr.ndim == 2:
                            ten = torch.from_numpy(arr).unsqueeze(0).float()
                            images.append(ten); labels.append(cls); basenames.append(src)
                            _seen_sources.add(src)
                            continue
                    else:
                        #print(f"[MISS] processed not found for class={sub}: {proc_path}")
                        continue

                # === FALLBACK when processed file is missing or not preferred ===
                fpath = os.path.join(raw_dir, fname)
                if vU.startswith("T"):
                    # Load tapered image directly from TXXkpc/<sub>/<base>TXXkpc.fits and format
                    t_dir = os.path.join(path, tag, sub)  # tag is e.g. "T50kpc"
                    t_path = os.path.join(t_dir, f"{base}.fits")
                    if not os.path.isfile(t_path):
                        print(f"[MISS] tapered FITS not found for class={sub}: {t_path}")
                        continue

                    #print(f"[T-FALLBACK] Using tapered image: {t_path}")
                    arr = np.squeeze(fits.getdata(t_path)).astype(np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=0)
                    if arr.ndim != 2:
                        print("  [SKIP] T image not 2D after squeeze/mean.")
                        print("  arr.shape:", arr.shape)
                        continue
                    hdr = fits.getheader(t_path)

                    # Crop based on beam count if requested; else use requested crop
                    Hc_eff, Wc_eff = Hc_ref, Wc_ref
                    if n_beams_min is not None:
                        fwhm_as = max(float(hdr["BMAJ"]), float(hdr["BMIN"])) * 3600.0
                        side_as = n_beams_min * fwhm_as
                        px, py = _pix_scales_arcsec(hdr)
                        Hc_eff = max(1, int(round(side_as / py)))
                        Wc_eff = max(1, int(round(side_as / px)))

                    ten = torch.from_numpy(arr).unsqueeze(0).float()
                    frm = apply_formatting(ten, crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))

                elif vU.startswith("RT"):
                    # RT fallback: convolve RAW to a circularized T-beam at the requested scale, then format
                    # 1) Load RAW frame
                    arr = np.squeeze(fits.getdata(fpath)).astype(np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=0)
                    if arr.ndim != 2:
                        print("  [SKIP] RAW data not 2D after squeeze/mean.")
                        print("  arr.shape:", arr.shape)
                        continue
                    raw_hdr = fits.getheader(fpath)
                    Hc_eff, Wc_eff = Hc_ref, Wc_ref

                    # 2) Resolve the gate T directory and FITS filename: .../TXXkpc/<sub>/<base>TXXkpc.fits
                    num = ''.join([c for c in vU if c.isdigit()]) or "50"
                    preferred_gate = f"T{int(float(num)) if float(num).is_integer() else num}kpc"
                    if os.path.isdir(os.path.join(path, preferred_gate, sub)):
                        gate_dirname = preferred_gate
                    else:
                        gate_dirname = _nearest_T_dir(path, sub, float(num))
                    if gate_dirname is None:
                        print(f"[GATE] No TXXkpc dir available for RT{num}kpc in sub='{sub}'. Skipping {src}.")
                        continue

                    txx_path = os.path.join(path, gate_dirname, sub, f"{base}{gate_dirname}.fits")
                    if not os.path.isfile(txx_path):
                        #print(f"  [SKIP] missing gate T image for RT convolution: {txx_path}")
                        continue
                    txx_hdr = fits.getheader(txx_path)

                    # 3) Build a circular (area-preserving) target covariance in world coords, then kernel on RAW pixels
                    C_raw_w = _beam_cov_world(raw_hdr)
                    C_tgt_w = _beam_cov_world(txx_hdr)

                    # Circularize target: isotropic with same area as target (sigma^2 = sqrt(det(C_tgt)))
                    sigma2 = float(np.sqrt(max(0.0, np.linalg.det(C_tgt_w))))
                    C_tgt_circ_w = np.array([[sigma2, 0.0],[0.0, sigma2]], float)

                    # Kernel covariance in world coords (PSD-clipped): C_ker = C_tgt_circ - C_raw
                    C_ker_w = C_tgt_circ_w - C_raw_w
                    w, V = np.linalg.eigh(C_ker_w)
                    w = np.clip(w, 0.0, None)
                    C_ker_w = (V * w) @ V.T

                    # Map to RAW pixel coords
                    J = _cd_matrix_rad(raw_hdr)
                    Jinv = np.linalg.inv(J)
                    Cpix = Jinv @ C_ker_w @ Jinv.T

                    # Build Gaussian kernel on a pixel grid (no Gaussian2DKernel dependency)
                    evals, evecs = np.linalg.eigh(Cpix)
                    evals = np.clip(evals, 1e-18, None)
                    s1, s2 = float(np.sqrt(evals[0])), float(np.sqrt(evals[1]))  # pixel stddevs
                    nker = int(np.ceil(8.0 * max(s1, s2))) | 1
                    k = (nker - 1) // 2
                    yy, xx = np.mgrid[-k:k+1, -k:k+1].astype(np.float32)
                    X = np.stack([xx, yy], axis=-1)  # [...,2]
                    Cinv = evecs @ np.diag(1.0/np.array([s1*s1, s2*s2], dtype=np.float32)) @ evecs.T
                    # exp(-0.5 * x^T Cinv x)
                    quad = (X @ Cinv * X).sum(axis=-1)
                    ker = np.exp(-0.5 * quad)
                    s = float(ker.sum())
                    if not np.isfinite(s) or s <= 0:
                        print("  [SKIP] degenerate kernel for", src)
                        continue
                    ker /= s

                    # 4) Convolve RAW and rescale to Jy/beam_tgt
                    arr = convolve_fft(
                        arr, ker, boundary="fill", fill_value=np.nan,
                        nan_treatment="interpolate", normalize_kernel=True,
                        psf_pad=True, fft_pad=True, allow_huge=True
                    )
                    arr *= (_beam_solid_angle_sr(txx_hdr) / _beam_solid_angle_sr(raw_hdr))

                    # 5) Equal-beams cropping on RAW grid using target FWHM, if requested
                    if n_beams_min is not None:
                        fwhm_as = max(float(txx_hdr["BMAJ"]), float(txx_hdr["BMIN"])) * 3600.0
                        side_as = n_beams_min * fwhm_as
                        px, py = _pix_scales_arcsec(raw_hdr)
                        Hc_eff = max(1, int(round(side_as / py)))
                        Wc_eff = max(1, int(round(side_as / px)))

                    # 6) Format to network size
                    ten = torch.from_numpy(arr).unsqueeze(0).float()
                    frm = apply_formatting(ten, crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))

                elif vU == "RAW":
                    # NEW: fallback for RAW — read native RAW and format
                    arr = np.squeeze(fits.getdata(fpath)).astype(np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=0)
                    if arr.ndim != 2:
                        print("  [SKIP] RAW data not 2D after squeeze/mean.")
                        print("  arr.shape:", arr.shape)
                        continue
                    raw_hdr = fits.getheader(fpath)

                    Hc_eff, Wc_eff = Hc_ref, Wc_ref
                    if n_beams_min is not None:
                        fwhm_as = max(float(raw_hdr["BMAJ"]), float(raw_hdr["BMIN"])) * 3600.0
                        side_as = n_beams_min * fwhm_as
                        px, py = _pix_scales_arcsec(raw_hdr)
                        Hc_eff = max(1, int(round(side_as / py)))
                        Wc_eff = max(1, int(round(side_as / px)))

                    ten = torch.from_numpy(arr).unsqueeze(0).float()
                    frm = apply_formatting(ten, crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))
                else:
                    print(f"Skipping source {src} as processed files are preferred and available.")
                    continue

                # common append
                images.append(frm)
                labels.append(cls)
                basenames.append(src)
                _seen_sources.add(src)


    # --- split by basename (stratified + grouped) ---
    y = np.array(labels)

    if len(y) == 0:
        # Helpful diagnostics for T/RT loads
        try:
            v_tag = _kpc_tag(versions[0] if isinstance(versions, (list, tuple)) else versions)
        except Exception:
            v_tag = str(versions)
        raise ValueError(
            f"[PSZ2] No samples collected. "
            f"Looked for version '{v_tag}' in processed_dir={processed_dir} "
            f"with fmt_{Ho}x{Wo}. "
            f"Tip: ensure crop_size matches available *_fmt_HxW.fits, "
            f"or use the fallback glob below."
            f"First location tried:\n {os.path.join(processed_dir, f'*_{v_tag}_fmt_{Ho}x{Wo}.fits')} "
        )

    # Separate test from training and validation data. 
    # The exact split is not important here, as it will be corrected later.
    # Eval images will be moved into training set. Never the other way around.
    print("Number of sources in total:", len(basenames))
    groups = np.array(basenames) # When mutliple versions exist,  this ensures all versions of a source stay together
    initial_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=41) # Fixed seed separates test from train+val
    trainval_idx, test_idx = next(initial_sgkf.split(np.zeros(len(y)), y, groups)) # 80% train+val, last 20% test

    if train: # Use train+val split. This is used for hyperparameter tuning
        print("Using train+val split for training/validation with number of source = ", len(trainval_idx))
        if fold is None or fold < 0 or fold >= 10: 
            raise ValueError("For train=True, fold must be an integer between 0 and 9 for 10-fold CV")
        
        # Extract train+val subset
        y_trainval = y[trainval_idx] # Select images for train and validation
        groups_trainval = groups[trainval_idx] # Select groups for train and validation
        
        sgkf_cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED) # 10-fold CV with random seed
        cv_splits = list(sgkf_cv.split(np.zeros(len(y_trainval)), y_trainval, groups_trainval)) # All combinations of train/val splits
        
        # Choose the specific fold or train/val split
        tr_idx_rel, va_idx_rel = cv_splits[fold] 
        tr_idx = trainval_idx[tr_idx_rel]
        va_idx = trainval_idx[va_idx_rel]
            
    else: # Return all trainval as training, test as evalF
        print("Using full train+val as training set, test set as evaluation set with number of source = ", len(test_idx)+len(trainval_idx))
        tr_idx = trainval_idx
        va_idx = test_idx
        
    def _take(idxs): # helper to select by indices
        return [images[i] for i in idxs], [labels[i] for i in idxs], [basenames[i] for i in idxs]

    train_images, train_labels, train_fns = _take(tr_idx)
    eval_images,  eval_labels,  eval_fns  = _take(va_idx)
    
    # Compare the data in train and eval sets with check_tensor to ensure similarity
    print("Initial dataset sizes — train:", len(train_images), "eval:", len(eval_images))
    for cls in set(train_labels):
        cls_imgs = [img for img, lbl in zip(train_images, train_labels) if lbl == cls]
        check_tensor(f"train_images class {cls} just after splitting", cls_imgs)
    for cls in set(eval_labels):
        cls_imgs = [img for img, lbl in zip(eval_images, eval_labels) if lbl == cls]
        check_tensor(f"eval_images class {cls} just after splitting", cls_imgs)
    
    print("Final dataset sizes — train:", len(train_images), "eval:", len(eval_images))

    return train_images, train_labels, eval_images, eval_labels, train_fns, eval_fns


def load_galaxies(galaxy_classes, path=None, versions=None, fold=None, island=None, crop_size=None, downsample_size=None, 
                  sample_size=None, REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False, 
                  USE_GLOBAL_NORMALISATION=False, GLOBAL_NORM_MODE="percentile", STRETCH=False, alpha=10.0,
                  percentile_lo=30, percentile_hi=99, NORMALISE=True, NORMALISETOPM=False, PREFER_PROCESSED=True, 
                  EXTRADATA=False, PRINTFILENAMES=False, SAVE_IMAGES=False, train=None, USE_CACHE=False, DEBUG=True):
    """
    Master loader that delegates to specific dataset loaders and returns zero-based labels.
    """
    
    # Try to load from cache first if enabled
    if USE_CACHE:
        cache_key = _build_cache_key(
            galaxy_classes, versions, fold, crop_size, downsample_size,
            sample_size, REMOVEOUTLIERS, BALANCE, AUGMENT, STRETCH,
            percentile_lo, percentile_hi, NORMALISE, NORMALISETOPM,
            USE_GLOBAL_NORMALISATION, GLOBAL_NORM_MODE, PREFER_PROCESSED, train
        )
        
        cached_data = _load_cache(cache_key)
        if cached_data is not None:
            # Return cached data - it already has the correct format
            if PRINTFILENAMES and len(cached_data) == 6:
                return cached_data
            elif not PRINTFILENAMES and len(cached_data) == 4:
                return cached_data
            elif PRINTFILENAMES and len(cached_data) == 4:
                # Cache doesn't have filenames but we need them - regenerate
                print("⚠ Cache exists but doesn't include filenames - regenerating")
            else:
                # Cache has filenames but we don't need them - just return first 4
                return cached_data[:4]
    
    def get_max_class(galaxy_classes):
        if isinstance(galaxy_classes, list):
            return max(galaxy_classes)
        return galaxy_classes
    
    # Clean up kwargs to remove None values
    kwargs = {'path': path, 'versions': versions, 'sample_size': sample_size, 'fold':fold, 'train': train,
              'island': island, 'crop_size': crop_size, 'downsample_size': downsample_size, 'prefer_processed': PREFER_PROCESSED}
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    max_class = get_max_class(galaxy_classes)

    # Delegate to specific loaders based on class range
    if max_class <= 59 and max_class >= 50:
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_PSZ2(target_classes=target_classes, **clean_kwargs)
    else:
        raise ValueError("Invalid galaxy class provided.")
    
    if len(data) == 4:
        train_images, train_labels, eval_images, eval_labels = data
        train_filenames = eval_filenames = None  # No filenames returned
    elif len(data) == 6:
        train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames = data
        overlap = set(train_filenames) & set(eval_filenames)
        assert not overlap, f"PSZ2 split error — these IDs are in both sets: {overlap}"
    else:
        raise ValueError("Data loader did not return the expected number of outputs.")
    

    if NORMALISE:
        if DEBUG:
            plot_class_images(train_images, eval_images, train_labels, eval_labels, train_filenames, eval_filenames, set_name='1.before_normalisation')
            for cls in set(train_labels):
                cls_imgs = [img for img, lbl in zip(train_images, train_labels) if lbl == cls]
                check_tensor(f"train_images class {cls} before normalisation", cls_imgs)
            for cls in set(eval_labels):
                cls_imgs = [img for img, lbl in zip(eval_images, eval_labels) if lbl == cls]
                check_tensor(f"eval_images class {cls} before normalisation", cls_imgs)
                
        if isinstance(train_images, list):
            train_images = torch.stack(train_images)
        if isinstance(eval_images, list):
            eval_images = torch.stack(eval_images)
        all_images = torch.cat([train_images, eval_images], dim=0)
        if USE_GLOBAL_NORMALISATION: # Regular normalisation of all images to [0,1]
            if DEBUG:
                print("Applying global normalisation to [0,1]")
            all_images = normalise_images(all_images, out_min=0, out_max=1)
        else:  # Percentile stretch to [0,1]
            if DEBUG:
                print(f"Applying percentile stretch to [{percentile_lo},{percentile_hi}]%")
            if all_images.ndim == 5:   # [B, T, C, H, W]
                for t in range(all_images.shape[1]):
                    all_images[:, t] = per_image_percentile_stretch(all_images[:, t], percentile_lo, percentile_hi)
            else:                      # [B, C, H, W]
                all_images = per_image_percentile_stretch(all_images, percentile_lo, percentile_hi)
        train_images = all_images[:len(train_images)]
        eval_images  = all_images[len(train_images):]
       
        if NORMALISETOPM:
            all_images = torch.cat([train_images, eval_images], dim=0)
            all_images = normalise_images(all_images, out_min=-1, out_max=1)
            train_images = all_images[:len(train_images)]
            eval_images  = all_images[len(train_images):]
            
        if DEBUG:
            plot_class_images(train_images, eval_images, train_labels, eval_labels, train_filenames, eval_filenames, set_name='2.after_normalisation')
            for cls in set(train_labels):
                cls_imgs = [img for img, lbl in zip(train_images, train_labels) if lbl == cls]
                check_tensor(f"train_images class {cls} after normalisation", cls_imgs)
            for cls in set(eval_labels):
                cls_imgs = [img for img, lbl in zip(eval_images, eval_labels) if lbl == cls]
                check_tensor(f"eval_images class {cls} after normalisation", cls_imgs)
            
    if STRETCH: # If this is after the redistribution accuracy drops significantly
        all_images = torch.cat([train_images, eval_images], dim=0) # Concatenate along batch dimension

        # Asinh stretch (elementwise), preserves shape/device/dtype
        stretched = torch.asinh(all_images * alpha) / math.asinh(alpha)

        # Split back
        n_tr = train_images.shape[0]
        train_images = stretched[:n_tr]
        eval_images  = stretched[n_tr:]
        
        if DEBUG:
            print("Applied asinh stretch with alpha =", alpha)
            plot_class_images(train_images, eval_images, train_labels, eval_labels, train_filenames, eval_filenames, set_name='3.after_stretching')
            for cls in set(train_labels):
                cls_imgs = [img for img, lbl in zip(train_images, train_labels) if lbl == cls]
                check_tensor(f"train_images class {cls} after stretching", cls_imgs)
            for cls in set(eval_labels):
                cls_imgs = [img for img, lbl in zip(eval_images, eval_labels) if lbl == cls]
                check_tensor(f"eval_images class {cls} after stretching", cls_imgs)
                
    train_images, eval_images, train_labels, eval_labels, train_filenames, eval_filenames = \
        redistribute_excess(
            train_images, eval_images,
            train_labels, eval_labels,
            train_filenames, eval_filenames,
        )
        
    # Check for overlap between train and test sets
    train_hashes = {img_hash(img) for img in train_images}
    eval_hashes  = {img_hash(img) for img in eval_images}
    if len(eval_images) != 0:
        common = train_hashes & eval_hashes
        if common:
            # Find *all* overlaps and plot side by side with names
            print(f"🔍 Found {len(common)} pixel-identical hash(es) between train and test.")
            # train_filenames / eval_filenames might be [] if not available; the helper handles that.
            _ = plot_pixel_overlaps_side_by_side(
                train_images, eval_images,
                train_filenames=train_filenames if 'train_filenames' in locals() else None,
                eval_filenames=eval_filenames     if 'eval_filenames' in locals()  else None,
                max_hashes=min(50, len(common)),  # show up to 50 hashes; tweak as you like
                outdir="./overlap_debug"
            )

            # Keep your original single example printout (useful in logs)
            overlap_hash = next(iter(common))
            train_idxs   = [i for i, img in enumerate(train_images) if img_hash(img) == overlap_hash]
            test_idxs    = [i for i, img in enumerate(eval_images)  if img_hash(img) == overlap_hash]
            print(f"🔍 Example overlap hash {overlap_hash!r} at train {train_idxs} and test {test_idxs}")

            # Now raise, so you notice it—but only *after* writing the figures.
            raise AssertionError(f"Overlap detected: {len(common)} images appear in both train and test validation! "
                                f"See './overlap_debug/' for side-by-side plots.")

    # Convert lists to tensors if needed
    if isinstance(train_images, list):
        print("Converting train_images list to tensor.")
        train_images = torch.stack(train_images)
    if isinstance(eval_images, list):
        print("Converting eval_images list to tensor.")
        eval_images  = torch.stack(eval_images)

    if BALANCE:
        if DEBUG:
            print("Balancing classes in training set by downsampling majority classes…")
        train_images, train_labels = balance_classes(train_images, train_labels) # Remove excess images from the largest class
    
    # Convert labels to a list if they are tensors
    sample_indices = {}
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.tolist()
    for cls in sorted(set(train_labels)):
        idxs = [i for i, lbl in enumerate(train_labels) if lbl == cls]
        sample_indices[cls] = idxs[:10]
        
    if isinstance(eval_labels, torch.Tensor):
        eval_labels = eval_labels.tolist()
    for cls in sorted(set(eval_labels)):
        idxs = [i for i, lbl in enumerate(eval_labels) if lbl == cls]
        sample_indices[cls] = sample_indices.get(cls, []) + idxs[:10]
        
    # Save images per class if enabled   
    if SAVE_IMAGES:
        for kind, imgs, lbls in (
            ('train', train_images, train_labels),
            ('eval',  eval_images,  eval_labels),
        ):
            for cls in target_classes:
                cls_imgs = [img for img, lbl in zip(imgs, lbls) if lbl == (cls-min(target_classes))]
                print(f"Length of {kind} images for class {cls}: {len(cls_imgs)}")
                np.save(f"{path}_{kind}_{cls}_{len(cls_imgs)}.npy", cls_imgs)
        
    # Load extra metadata if requested
    if EXTRADATA and not PRINTFILENAMES:
        if DEBUG:
            print("Loading PSZ2 metadata")
        meta_df = pd.read_csv(os.path.join(root_path, "PSZ2/cluster_source_data.csv"))
        print("PSZ2 metadata columns:", meta_df.columns.tolist())
        meta_df.rename(columns={"slug": "base"}, inplace=True)
        meta_df.set_index("base", inplace=True)

        # build a list of metadata rows in the same order as your `filenames` list:
        train_data = [meta_df.loc[base].values for base in train_filenames]
        eval_data  = [meta_df.loc[base].values for base in eval_filenames]
        
    # Data augmentation (with flips and rotations)
    if AUGMENT:
        if DEBUG:
            print("Applying data augmentation…")
        train_images, train_labels = augment_images(train_images, train_labels, ST_augmentation=False)
        if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53): 
            eval_images, eval_labels = augment_images(eval_images, eval_labels, ST_augmentation=False) # Only augment if not RR and RH
        else:
            if len(eval_images.shape) == 3:
                eval_images = eval_images.unsqueeze(1)
            if isinstance(eval_images, (list, tuple)):
                eval_images = torch.stack(eval_images)
            if isinstance(eval_labels, (list, tuple)):
                eval_labels = torch.tensor(eval_labels, dtype=torch.long)
        if EXTRADATA and not PRINTFILENAMES:
                n_aug = 8  # default is 4*2 = 8
                train_data = [row for row in train_data for _ in range(n_aug)]
                if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53):
                    eval_data  = [row for row in eval_data  for _ in range(n_aug)]
        if PRINTFILENAMES:
            n_aug = 8  # default is 4*2 = 8
            train_filenames = [fname for fname in train_filenames for _ in range(n_aug)]
            if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53): 
                eval_filenames  = [fname for fname in eval_filenames  for _ in range(n_aug)]

    else:
        # Unsqueeze if the images are of shape (B, H, W) 
        if len(train_images.shape) == 3:
            train_images = train_images.unsqueeze(1)
        if len(eval_images.shape) == 3:
            eval_images = eval_images.unsqueeze(1)
        if isinstance(train_images, (list, tuple)):
            train_images = torch.stack(train_images)
        if isinstance(train_labels, (list, tuple)):
            train_labels = torch.tensor(train_labels, dtype=torch.long)
        if isinstance(eval_images, (list, tuple)):
            eval_images = torch.stack(eval_images)
        if isinstance(eval_labels, (list, tuple)):
            eval_labels = torch.tensor(eval_labels, dtype=torch.long)
       
    # Save to cache if enabled     
    if USE_CACHE:
        if PRINTFILENAMES:
            cache_data = (train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames)
        elif EXTRADATA:
            # Don't cache EXTRADATA - it's metadata that can be loaded separately
            cache_data = (train_images, train_labels, eval_images, eval_labels)
        else:
            cache_data = (train_images, train_labels, eval_images, eval_labels)
        
        _save_cache(cache_key, cache_data)
            
    # Return statements
    if PRINTFILENAMES:
        return train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames
    elif EXTRADATA:
        train_data = torch.tensor(np.stack(train_data), dtype=torch.float32)
        eval_data  = torch.tensor(np.stack(eval_data),  dtype=torch.float32)
        return train_images, train_labels, eval_images, eval_labels, train_data, eval_data
    
    return train_images, train_labels, eval_images, eval_labels


def load_halos_and_relics(
    galaxy_classes,
    versions=('RAW',),
    fold=5,
    crop_size=(1, 128, 128),
    downsample_size=(1, 128, 128),
    sample_size=1_000_000,
    REMOVEOUTLIERS=True,
    BALANCE=False,
    STRETCH=False,
    percentile_lo=1,
    percentile_hi=99,
    AUGMENT=False,
    NORMALISE=True,
    NORMALISETOPM=False,
    USE_GLOBAL_NORMALISATION=False,
    GLOBAL_NORM_MODE='percentile',
    PRINTFILENAMES=False,
    train=True,
):
    """
    Unifies dataset loading so the training driver can call a single entry point.
    Returns:
      train=True:
        4-tuple: (train_images, train_labels, valid_images, valid_labels)
        6-tuple if PRINTFILENAMES: (..., train_fns, valid_fns)
      train=False:
        4-tuple: (empty_images, empty_labels, test_images, test_labels)
        6-tuple if PRINTFILENAMES: (..., test_fns)
    Notes:
      * Labels are left as original class tags (e.g. 52, 53). The driver relabels later.
      * If multiple `versions` are provided (e.g. ['RAW','T50kpc']), PSZ2 returns a tesseract [T,1,H,W] per sample.
    """

    # -------- 1) Load raw images/labels (+ optional filenames) ----------
    # Decide dataset family from class tags
    is_psz2  = any(50 <= int(c) <= 58 for c in galaxy_classes)
    is_first = any(10 <= int(c) <= 13 for c in galaxy_classes)

    images = labels = filenames = None

    if is_psz2:
        raw = load_PSZ2(
            path = root_path + "PSZ2/classified/",
            sample_size = sample_size,              # per class in training set; eval uses sample_size*0.2
            target_classes = galaxy_classes,  # list of int class tags to load
            versions = versions,          # string or list/tuple; list => Multiple versions
            crop_size = crop_size,        # (C,Hc,Wc) — angular FoV is taken from the ref version
            downsample_size = downsample_size,  # (C,Ho,Wo) — output per frame
        )
        
        if len(raw) == 6:
            tr_imgs, tr_lbls, ev_imgs, ev_lbls, tr_fns, ev_fns = raw
        elif len(raw) == 4:
            tr_imgs, tr_lbls, ev_imgs, ev_lbls = raw
            tr_fns, ev_fns = [], []
        else:
            raise RuntimeError(f"load_PSZ2 returned unexpected shape (len={len(raw)})")
        # combine so we can stratify/split here
        def _as_list(x):
            return [x[i] for i in range(len(x))] if torch.is_tensor(x) else list(x)
        images    = _as_list(tr_imgs) + _as_list(ev_imgs)
        labels    = (tr_lbls.cpu().tolist() if torch.is_tensor(tr_lbls) else list(tr_lbls)) + \
                    (ev_lbls.cpu().tolist() if torch.is_tensor(ev_lbls) else list(ev_lbls))
        filenames = list(tr_fns) + list(ev_fns)
    else:
        raise ValueError("load_galaxies: unsupported class set; add a branch for your dataset.")

    # Ensure list types
    if isinstance(images, torch.Tensor):
        images = [img for img in images]
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().tolist()

    # -------- 2) Optional image-level transforms (percentile stretch, asinh, normalise) ----------
    def _maybe_proc(img):
        x = img
        if STRETCH:
            # per-image percentile stretch then asinh, like the driver expects
            x = per_image_percentile_stretch(x, lo=percentile_lo, hi=percentile_hi)
            x = torch.asinh(10 * x) / math.asinh(10)
        if NORMALISE:
            if NORMALISETOPM:
                x = normalise_images(x, -1, 1)
            else:
                x = normalise_images(x, 0, 1)
        return x

    images = [_maybe_proc(x) for x in images]

    # -------- 3) Optional class balancing (undersample to min class size) ----------
    if BALANCE:
        from collections import defaultdict
        byc = defaultdict(list)
        for i, lbl in enumerate(labels):
            byc[int(lbl)].append(i)
        min_n = min(len(v) for v in byc.values())
        keep = []
        for v in byc.values():
            keep.extend(v[:min_n])
        keep = sorted(keep)
        images   = [images[i] for i in keep]
        labels   = [labels[i] for i in keep]
        filenames = [filenames[i] for i in keep] if filenames else []

    # -------- 4) Optional augmentation (pre-split, like your current driver when LATE_AUG=False) ----------
    if AUGMENT:
        imgs_t = torch.stack(images)
        lbls_t = torch.tensor(labels, dtype=torch.long)
        imgs_t, lbls_t = augment_images(imgs_t, lbls_t)
        images = [imgs_t[i] for i in range(len(imgs_t))]
        labels = lbls_t.cpu().tolist()
        if filenames:
            # replicate filenames n_aug times
            n_aug = len(images) // max(1, len(filenames))
            filenames = [fn for fn in filenames for _ in range(n_aug)]

    # -------- 5) Stratified split into train/valid (or build test only) ----------
    y = np.array(labels)
    idx_all = np.arange(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(idx_all, y))
    # Map fold==5 to "last split" for your driver’s convention
    split_idx = fold if fold in [0,1,2,3,4] else 4
    tr_idx, va_idx = splits[split_idx]

    def _take(idxs):
        ims = [images[i] for i in idxs]
        lbs = torch.tensor([labels[i] for i in idxs], dtype=torch.long)
        fns = [filenames[i] for i in idxs] if filenames else []
        # stack if 3D or 4D tensors, else leave as-is
        try:
            ims = torch.stack(ims)
        except Exception:
            pass
        return ims, lbs, fns

    train_images, train_labels, train_fns = _take(tr_idx)
    valid_images, valid_labels, valid_fns = _take(va_idx)

    # Optionally bound per-class sample sizes
    if isinstance(sample_size, int) and sample_size > 0:
        # limit train set per class
        cls_counts = {c:0 for c in sorted(set(labels))}
        keep = []
        for i, lbl in enumerate(train_labels.tolist()):
            if cls_counts[lbl] < sample_size:
                keep.append(i); cls_counts[lbl] += 1
        train_images = train_images[keep]
        train_labels = train_labels[keep]
        if train_fns:
            train_fns = [train_fns[i] for i in keep]

    if not train:
        empty_imgs = torch.empty((0,)+tuple(train_images.shape[1:])) if isinstance(train_images, torch.Tensor) else []
        empty_lbls = torch.empty((0,), dtype=torch.long)
        if PRINTFILENAMES:
            return empty_imgs, empty_lbls, valid_images, valid_labels, [], valid_fns
        return empty_imgs, empty_lbls, valid_images, valid_labels

    if PRINTFILENAMES:
        return train_images, train_labels, valid_images, valid_labels, train_fns, valid_fns
    return train_images, train_labels, valid_images, valid_labels
