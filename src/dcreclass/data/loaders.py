# loaders.py
# Data loading pipeline for PSZ2 galaxy cluster FITS images.
# Handles single- and multi-version loading, caching, augmentation,
# normalisation, and stratified train/eval splitting.

import random, math, hashlib, glob, os, re, torch, json
import numpy as np, pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dcreclass.utils import normalise_images, check_tensor
from dcreclass.utils import plot_class_images, plot_pixel_overlaps_side_by_side
from sklearn.model_selection import StratifiedGroupKFold
from astropy.io import fits
from astropy.convolution import convolve_fft
from collections import Counter, defaultdict

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Default data root — override by passing `path` explicitly to loaders
ROOT_PATH = '/users/mbredber/scratch/data/'

######################################################################################################
################################### GENERAL STUFF ####################################################
######################################################################################################

# Minimum FITS files expected per source (RAW + at least one T version)

def check_complete_download(dest_dir: Union[str, Path], slug: str) -> Tuple[bool, int, List[str]]:
    """
    Check whether a source directory has a complete set of FITS files.

    A source is complete if it has the RAW file, all three T*kpc files,
    and all three T*kpcSUB files.

    Args:
        dest_dir: Path to the source directory.
        slug:     Source name, e.g. PSZ2G023.17+86.71.

    Returns:
        Tuple (is_complete, n_existing, missing_files)
    """
    dest_dir = Path(dest_dir)

    if not dest_dir.exists():
        return False, 0, []

    existing = {f for f in os.listdir(dest_dir) if f.endswith('.fits')}

    if len(existing) < 2:  # At least RAW + one T version expected for a valid source
        return False, len(existing), []

    # All files a complete source should have
    required = [
        f"{slug}.fits",
        f"{slug}T25kpc.fits",
        f"{slug}T50kpc.fits",
        f"{slug}T100kpc.fits",
        f"{slug}T25kpcSUB.fits",
        f"{slug}T50kpcSUB.fits",
        f"{slug}T100kpcSUB.fits",
    ]

    missing = [f for f in required if f not in existing]
    return (len(missing) == 0), len(existing), missing


def get_classes() -> List[Dict]:
    """Return PSZ2 class definitions (tag, length, description)."""
    return [
        {"tag": 50, "length": 62,  "description": "DE"},           # RH + RR
        {"tag": 51, "length": 114, "description": "NDE"},          # No Diffuse Emission
        {"tag": 52, "length": 53,  "description": "RH"},           # Radio Halo
        {"tag": 53, "length": 20,  "description": "RR"},           # Radio Relic (only 8 unique sources)
        {"tag": 54, "length": 19,  "description": "cRH"},          # Candidate Radio Halo
        {"tag": 55, "length": 6,   "description": "cRR"},          # Candidate Radio Relic
        {"tag": 56, "length": 24,  "description": "cDE"},          # Candidate Diffuse Emission
        {"tag": 57, "length": 47,  "description": "U"},            # Uncertain
        {"tag": 58, "length": 40,  "description": "unclassified"},
    ]


#######################################################################################################
################################### PIXEL SCALE / BEAM HELPERS ########################################
#######################################################################################################


def _pix_scales_arcsec(hdr: Any) -> Tuple[float, float]:
    """
    Return pixel scales (px, py) in arcsec/pixel from a FITS header.
    Handles CDELT, CD (rotation/shear), and PC*CDELT conventions.
    """
    def _has(*keys: str) -> bool:
        return all(k in hdr for k in keys)

    # Case 1: CD matrix in deg/pix (rotation/shear allowed)
    if _has('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
        cd11 = float(hdr['CD1_1']); cd12 = float(hdr['CD1_2'])
        cd21 = float(hdr['CD2_1']); cd22 = float(hdr['CD2_2'])
        sx = math.sqrt(cd11*cd11 + cd21*cd21) * 3600.0
        sy = math.sqrt(cd12*cd12 + cd22*cd22) * 3600.0
        return abs(sx), abs(sy)

    # Case 2: PC matrix (unitless) + CDELT in deg/pix
    if _has('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2') and _has('CDELT1', 'CDELT2'):
        cdelt1 = float(hdr['CDELT1']); cdelt2 = float(hdr['CDELT2'])
        pc11 = float(hdr['PC1_1']); pc12 = float(hdr['PC1_2'])
        pc21 = float(hdr['PC2_1']); pc22 = float(hdr['PC2_2'])
        cd11 = pc11 * cdelt1; cd12 = pc12 * cdelt1
        cd21 = pc21 * cdelt2; cd22 = pc22 * cdelt2
        sx = math.sqrt(cd11*cd11 + cd21*cd21) * 3600.0
        sy = math.sqrt(cd12*cd12 + cd22*cd22) * 3600.0
        return abs(sx), abs(sy)

    # Case 3: plain CDELT in deg/pix (no rotation)
    if _has('CDELT1', 'CDELT2'):
        return abs(float(hdr['CDELT1'])) * 3600.0, abs(float(hdr['CDELT2'])) * 3600.0

    # Non-standard but occasionally encountered keywords
    for kx, ky in [('PIXSCAL1', 'PIXSCAL2'), ('XPIXSCAL', 'YPIXSCAL')]:
        if _has(kx, ky):
            return abs(float(hdr[kx])), abs(float(hdr[ky]))

    raise KeyError("Cannot determine pixel scale from FITS header (no CD/PC+CDELT/CDELT).")


def _pixdeg(hdr: Any) -> float:
    """
    Return a single representative pixel scale in deg/pix,
    using the geometric mean of the x/y scales.
    """
    px_arcsec, py_arcsec = _pix_scales_arcsec(hdr)
    return math.sqrt(px_arcsec * py_arcsec) / 3600.0


#######################################################################################################
################################### VERSION NORMALISATION #############################################
#######################################################################################################


def _to_int_if_close(x: float, tol: float = 1e-6) -> str:
    """Return int string if x is nearly integer, else a compact float string."""
    if abs(x - round(x)) < tol:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip('0').rstrip('.')


def _canon_ver(v: Any) -> str:
    """
    Normalise a version token to a canonical folder name.

    Accepts: 'RAW'/'raw'/'i', 'T50kpc', 't50', '50', 'Blur50kpc', 'blur50', etc.
    Returns: 'RAW', 'T{N}kpc', 'T{N}kpcSUB', or 'Blur{N}kpc'.
    Units: kpc (default if omitted), mpc (converted to kpc).
    """
    s_raw = str(v).strip()
    s = re.sub(r'[^0-9a-zA-Z\.]', '', s_raw).lower()

    if s in {'raw', 'i', 'image'}:
        return 'RAW'

    # Pattern: (rt|t) + number + optional unit + optional SUB
    m = re.match(r'^(rt|t)(\d+(?:\.\d+)?)([a-z]*)?(sub)?$', s)
    if m:
        pref, val_str, unit, sub = m.groups()
        unit = unit or 'kpc'
        try:
            val = float(val_str)
        except ValueError:
            return str(v)
        if unit == 'mpc':
            val *= 1000.0
            unit = 'kpc'
        norm_num = _to_int_if_close(val)
        out = f"{pref.upper()}{norm_num}{unit}"
        if sub:
            out += "SUB"
        return out

    # Plain number (with optional unit) → T-version
    m2 = re.match(r'^(\d+(?:\.\d+)?)([a-z]*)$', s)
    if m2:
        val_str, unit = m2.groups()
        unit = unit or 'kpc'
        try:
            val = float(val_str)
        except ValueError:
            return str(v)
        if unit == 'mpc':
            val *= 1000.0
            unit = 'kpc'
        norm_num = _to_int_if_close(val)
        return f"T{norm_num}{unit}"

    return str(v)


def _pick_equal_taper_from(versions: Union[str, List, Tuple]) -> str:
    """Return the first T* token from versions, or 'T50kpc' as default."""
    vlist = versions if isinstance(versions, (list, tuple)) else [versions]
    norm  = [_canon_ver(v) for v in vlist]
    for t in norm:
        if str(t).upper().startswith('T'):
            return t
    return "T50kpc"


def _scan_min_beams(
        base_path: str,
        classes: List[int],
        taper: str) -> Tuple[Optional[float], Dict]:
    """
    Scan all FITS files for a given taper version and return the minimum
    number of beams across-field, plus a header cache keyed by basename.

    Args:
        base_path: Root directory containing taper subdirectories.
        classes:   List of class tags (e.g. [50, 51]).
        taper:     Taper folder name (e.g. 'T50kpc').

    Returns:
        (n_beams_min, header_cache) — min is None if no files found.
    """
    # Build a tag → description map from get_classes() for safe lookup
    tag_to_desc = {c["tag"]: c["description"] for c in get_classes()}

    nmin: Optional[float] = None
    hdrs: Dict[str, Any] = {}

    for cls in classes:
        sub = tag_to_desc.get(cls)
        if sub is None:
            print(f"[WARN] _scan_min_beams: unknown class tag {cls}, skipping.")
            continue
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
                hdrs[os.path.splitext(f)[0]] = h
            except Exception:
                pass
    return nmin, hdrs


#######################################################################################################
################################### DATA AUGMENTATION #################################################
#######################################################################################################


def per_image_percentile_stretch(
        x: torch.Tensor,
        lo: float = 30,
        hi: float = 99) -> torch.Tensor:
    """
    Stretch each image in a batch to [0, 1] using per-image percentile clipping.

    Args:
        x:  Tensor of shape [B, C, H, W].
        lo: Lower percentile (0–100).
        hi: Upper percentile (0–100).

    Returns:
        Stretched tensor of the same shape.
    """
    B = x.shape[0]
    out = x.clone()
    for i in range(B):
        flat   = out[i].reshape(-1)
        p_low  = flat.quantile(lo / 100)
        p_high = flat.quantile(hi / 100)
        out[i] = ((out[i] - p_low) / (p_high - p_low + 1e-6)).clamp(0, 1)
    return out


# --- RT (I*G) helpers: world↔pixel, beam covariance, convolution kernel ---

def _cd_matrix_rad(h: Any) -> np.ndarray:
    """Return the 2×2 CD matrix in radians/pixel from a FITS header."""
    if 'CD1_1' in h:
        M = np.array([[h['CD1_1'], h.get('CD1_2', 0.0)],
                      [h.get('CD2_1', 0.0), h['CD2_2']]], float)
    else:
        pc11 = h.get('PC1_1', 1.0); pc12 = h.get('PC1_2', 0.0)
        pc21 = h.get('PC2_1', 0.0); pc22 = h.get('PC2_2', 1.0)
        cd1  = h.get('CDELT1', 1.0); cd2  = h.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12], [pc21, pc22]], float) @ np.diag([cd1, cd2])
    return M * (np.pi / 180.0)


def _fwhm_as_to_sigma_rad(fwhm_as: float) -> float:
    """Convert FWHM in arcsec to Gaussian sigma in radians."""
    return (float(fwhm_as) / (2.0 * np.sqrt(2.0 * np.log(2.0)))) * (np.pi / (180.0 * 3600.0))


def _beam_cov_world(h: Any) -> np.ndarray:
    """
    Return the 2×2 beam covariance matrix in world (radian) coordinates.
    Requires BMAJ/BMIN in degrees and optionally BPA in degrees.
    """
    bmaj_as = float(h['BMAJ']) * 3600.0
    bmin_as = float(h['BMIN']) * 3600.0
    pa_deg  = float(h.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th = np.deg2rad(pa_deg)
    R  = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th),  np.cos(th)]], float)
    S  = np.diag([sx * sx, sy * sy])
    return R @ S @ R.T


def _beam_solid_angle_sr(h: Any) -> float:
    """Return the beam solid angle in steradians from BMAJ/BMIN (degrees)."""
    bmaj = abs(float(h['BMAJ'])) * np.pi / 180.0
    bmin = abs(float(h['BMIN'])) * np.pi / 180.0
    return (np.pi / (4.0 * np.log(2.0))) * bmaj * bmin


def balance_classes(
        images: Union[List, torch.Tensor],
        labels: Union[List, torch.Tensor],
        function_names: Optional[List[str]] = None
) -> Union[Tuple[List, List], Tuple[List, List, List]]:
    """
    Randomly down-sample each class to the size of the smallest class.

    Args:
        images:         List or tensor of images.
        labels:         Corresponding class labels.
        function_names: Optional list of source names (same length as images).

    Returns:
        (images, labels) or (images, labels, function_names) if names provided.
    """
    counter = Counter(labels)
    print("Class distribution before balancing:", dict(counter))

    class_idxs: Dict[Any, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_idxs[lbl].append(i)

    min_n = min(len(idxs) for idxs in class_idxs.values())
    selected: List[int] = []
    for idxs in class_idxs.values():
        selected.extend(random.sample(idxs, min_n))
    random.shuffle(selected)

    counter_after = Counter([labels[i] for i in selected])
    print("Class distribution after balancing:", dict(counter_after))

    if function_names is not None:
        return ([images[i] for i in selected],
                [labels[i] for i in selected],
                [function_names[i] for i in selected])

    return [images[i] for i in selected], [labels[i] for i in selected]


def apply_formatting(
        image: torch.Tensor,
        crop_size: Tuple = (1, 512, 512),
        downsample_size: Tuple = (1, 128, 128)) -> torch.Tensor:
    """
    Centre-crop and bilinear-resize a single-channel tensor without PIL.

    Args:
        image:          Tensor of shape [C, H, W] (or [1, C, H, W]).
        crop_size:      (C, Hc, Wc) or (Hc, Wc) — crop window size.
        downsample_size:(C, Ho, Wo) or (Ho, Wo) — output size.

    Returns:
        Tensor of shape [C, Ho, Wo].
    """
    def _canon_size(sz: Tuple) -> Tuple[int, int, int]:
        if len(sz) == 2: return (1, sz[0], sz[1])
        if len(sz) == 3: return sz
        if len(sz) == 4: return (sz[-3], sz[-2], sz[-1])
        raise ValueError(f"crop/downsample size must have 2–4 dims, got {sz}")

    crop_size       = _canon_size(crop_size)
    downsample_size = _canon_size(downsample_size)

    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)
    if image.dim() == 3:
        _, H0, W0 = image.shape
        img = image
    elif image.dim() == 2:
        H0, W0 = image.shape
        img = image.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image dims: {image.shape}")

    if crop_size[0] == 1 or downsample_size[0] == 1:
        img = img.mean(dim=0, keepdim=True)

    _, Hc, Wc = crop_size
    _, Ho, Wo = downsample_size

    y0, x0 = H0 // 2, W0 // 2
    y1 = max(0, y0 - Hc // 2);  y2 = min(H0, y0 + Hc // 2)
    x1 = max(0, x0 - Wc // 2);  x2 = min(W0, x0 + Wc // 2)

    crop    = img[:, y1:y2, x1:x2].unsqueeze(0)           # [1, C, Hc, Wc]
    resized = F.interpolate(crop, size=(Ho, Wo), mode='bilinear')
    return resized.squeeze(0)                               # [C, Ho, Wo]


def img_hash(img: torch.Tensor) -> str:
    """Return a SHA-1 hex digest of the raw tensor bytes (for duplicate detection)."""
    arr = img.cpu().contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()


def apply_transforms_with_config(
        image: torch.Tensor,
        config: Dict[str, Any]) -> torch.Tensor:
    """Apply deterministic flip + rotation from a config dict."""
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: transforms.functional.hflip(x) if config['flip_h'] else x),
        transforms.Lambda(lambda x: transforms.functional.vflip(x) if config['flip_v'] else x),
        transforms.Lambda(lambda x: transforms.functional.rotate(x, config['rotation'])),
    ])
    if image.dim() == 2:
        image = image.unsqueeze(0)
    return preprocess(image)


def augment_images(
        images: Union[List[torch.Tensor], torch.Tensor],
        labels: Union[List, torch.Tensor],
        rotations: List[int] = list(range(0, 360, 30)),
        flips: List[Tuple[bool, bool]] = [(False, False), (True, False)],
        mem_threshold: int = 1000,
        translations: List[Tuple[int, int]] = [(0, 0)],
        ST_augmentation: bool = False,
        n_gen: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment images with systematic rotations, flips, and optional translations.

    Args:
        images:           List or tensor of input images [C, H, W].
        labels:           Corresponding labels.
        rotations:        List of rotation angles in degrees.
        flips:            List of (flip_h, flip_v) boolean pairs.
        mem_threshold:    Flush to cumulative list after this many augmentations.
        translations:     List of (dx, dy) pixel offsets.
        ST_augmentation:  Load style-transfer generated images from scratch dir.
        n_gen:            Number of generated images per class (for ST loading).

    Returns:
        (augmented_images, augmented_labels) as tensors.
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    label_dtype  = labels.dtype
    label_device = labels.device

    # Normalise all inputs to 3D [C, H, W]
    normed: List[torch.Tensor] = []
    for img in images:
        if isinstance(img, torch.Tensor):
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)
            if img.dim() == 2:
                img = img.unsqueeze(0)
        normed.append(img)
    images = normed

    augmented_images:    List[torch.Tensor] = []
    augmented_labels:    List               = []
    cumulative_images:   List[torch.Tensor] = []
    cumulative_labels:   List               = []

    if ST_augmentation:
        lbl_list = [int(x) for x in (labels.tolist() if torch.is_tensor(labels) else labels)]
        for cls in sorted(set(lbl_list)):
            pattern    = f"/users/mbredber/scratch/ST_generation/1to{n_gen}_*_{cls}.npy"
            candidates = sorted(glob.glob(pattern))
            if not candidates:
                print(f"[augment_images] No ST file matching {pattern}; skipping class {cls}.")
                continue
            st_images = torch.tensor(np.load(candidates[0])).float().unsqueeze(1)
            images.extend(st_images)
            lbl_list.extend([cls] * len(st_images))
        labels = lbl_list

    for idx, image in enumerate(images):
        for rot in rotations:
            for flip_h, flip_v in flips:
                for translation in translations:
                    if translation != (0, 0):
                        image = transforms.functional.affine(
                            image, angle=0, translate=translation,
                            scale=1.0, shear=0, fill=0)
                    config  = {'rotation': rot, 'flip_h': flip_h, 'flip_v': flip_v}
                    aug_img = apply_transforms_with_config(image.clone().detach(), config)
                    augmented_images.append(aug_img)
                    augmented_labels.append(labels[idx])

                    if len(augmented_images) >= mem_threshold:
                        cumulative_images.extend(augmented_images)
                        cumulative_labels.extend(augmented_labels)
                        augmented_images, augmented_labels = [], []

    cumulative_images.extend(augmented_images)
    cumulative_labels.extend(augmented_labels)

    augmented_images_tensor = torch.stack(cumulative_images)

    if len(cumulative_labels) == 0:
        augmented_labels_tensor = torch.empty(
            (0,) + labels.shape[1:], dtype=label_dtype, device=label_device)
    else:
        first = cumulative_labels[0]
        if isinstance(first, torch.Tensor):
            augmented_labels_tensor = torch.stack(
                [x.to(dtype=label_dtype, device=label_device)
                 for x in cumulative_labels], dim=0)
        else:
            augmented_labels_tensor = torch.tensor(
                cumulative_labels, dtype=label_dtype, device=label_device)

    return augmented_images_tensor, augmented_labels_tensor


##########################################################################################
################################# CACHE FUNCTIONS ########################################
##########################################################################################


def _build_cache_key(
        galaxy_classes: List[int],
        versions: Union[str, List[str]],
        fold: Optional[int],
        crop_size: Optional[Tuple],
        downsample_size: Optional[Tuple],
        sample_size: Optional[int],
        REMOVEOUTLIERS: bool,
        BALANCE: bool,
        AUGMENT: bool,
        STRETCH: bool,
        percentile_lo: float,
        percentile_hi: float,
        NORMALISE: bool,
        NORMALISETOPM: bool,
        USE_GLOBAL_NORMALISATION: bool,
        GLOBAL_NORM_MODE: str,
        train: Optional[bool],
        crop_mode: str) -> str:
    """
    Build a unique, human-readable cache key from all parameters that affect loaded data.

    Returns:
        str: A short hash-prefixed key string.
    """
    if isinstance(versions, (list, tuple)):
        ver_str = "_".join(sorted([_canon_ver(v) for v in versions]))
    else:
        ver_str = _canon_ver(versions)

    params = {
        'galaxy_classes':          sorted(galaxy_classes) if isinstance(galaxy_classes, list) else [galaxy_classes],
        'versions':                ver_str,
        'fold':                    fold,
        'crop_size':               crop_size,
        'downsample_size':         downsample_size,
        'sample_size':             sample_size,
        'REMOVEOUTLIERS':          REMOVEOUTLIERS,
        'BALANCE':                 BALANCE,
        'AUGMENT':                 AUGMENT,
        'STRETCH':                 STRETCH,
        'percentile_lo':           percentile_lo,
        'percentile_hi':           percentile_hi,
        'NORMALISE':               NORMALISE,
        'NORMALISETOPM':           NORMALISETOPM,
        'USE_GLOBAL_NORMALISATION':USE_GLOBAL_NORMALISATION,
        'GLOBAL_NORM_MODE':        GLOBAL_NORM_MODE,
        'train':                   train,
        'crop_mode':               crop_mode,
    }

    param_str  = json.dumps(params, sort_keys=True)
    cache_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]

    classes_str = "_".join(map(str, params['galaxy_classes']))
    if USE_GLOBAL_NORMALISATION:
        prefix = f"cache_cls{classes_str}_ver{ver_str}_globnorm{GLOBAL_NORM_MODE}_f{fold}"
    else:
        prefix = f"cache_cls{classes_str}_ver{ver_str}_f{fold}"
    return f"{prefix}_{cache_hash}"


def _save_cache(
        cache_key: str,
        data_tuple: Tuple,
        cache_dir: str = "./.cache/images") -> None:
    """
    Save processed data to a .pt cache file.

    Args:
        cache_key:  Unique identifier for this entry.
        data_tuple: 4-tuple or 6-tuple of (train_images, train_labels,
                    eval_images, eval_labels [, train_fns, eval_fns]).
        cache_dir:  Directory to store cache files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    save_dict = {
        'train_images': data_tuple[0],
        'train_labels': data_tuple[1],
        'eval_images':  data_tuple[2],
        'eval_labels':  data_tuple[3],
    }
    if len(data_tuple) == 6:
        save_dict['train_filenames'] = data_tuple[4]
        save_dict['eval_filenames']  = data_tuple[5]

    try:
        torch.save(save_dict, cache_path)
        print(f"✓ Saved data cache to {cache_path}")
    except Exception as e:
        print(f"⚠ Failed to save cache: {e}")


def _load_cache(
        cache_key: str,
        cache_dir: str = "./.cache/images") -> Optional[Tuple]:
    """
    Load processed data from cache if it exists.

    Args:
        cache_key: Unique identifier for this entry.
        cache_dir: Directory where cache files are stored.

    Returns:
        4- or 6-tuple of tensors/lists, or None if no cache found.
    """
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if not os.path.isfile(cache_path):
        return None

    try:
        print(f"✓ Loading data from cache: {cache_path}")
        save_dict = torch.load(cache_path)

        if 'train_filenames' in save_dict and 'eval_filenames' in save_dict:
            return (save_dict['train_images'], save_dict['train_labels'],
                    save_dict['eval_images'],  save_dict['eval_labels'],
                    save_dict['train_filenames'], save_dict['eval_filenames'])
        else:
            return (save_dict['train_images'], save_dict['train_labels'],
                    save_dict['eval_images'],  save_dict['eval_labels'])
    except Exception as e:
        print(f"⚠ Failed to load cache (will regenerate): {e}")
        return None


##########################################################################################
################################## SPECIFIC DATASET LOADER ###############################
##########################################################################################


_PSZ2_DATA_DIR = "/users/mbredber/scratch/data/PSZ2"
# crop_mode → whether a fixed arcsec FOV is used (beam_crop uses beam-equalised FOV)
_CROP_MODE_MAP = {
    'beam_crop':  dict(fov_based=False),
    'fov_crop':   dict(fov_based=True),
    'pixel_crop': dict(fov_based=False),   # raw on-the-fly; no processed files
}

# blur_method → convolution behaviour flags
_BLUR_METHOD_MAP = {
    'circular':        dict(subtract_beam=True,  cheat_rt=False),
    'circular_no_sub': dict(subtract_beam=False, cheat_rt=False),
    'cheat':           dict(subtract_beam=True,  cheat_rt=True),
}

def _processed_subdir(crop_mode: str, blur_method: str) -> Optional[str]:
    """Return the PSZ2-relative fits_files path, or None for pixel_crop."""
    if crop_mode == 'pixel_crop':
        return None
    return f"{crop_mode}/{blur_method}/fits_files"

def load_PSZ2(
        path: str = ROOT_PATH + "PSZ2/classified/",
        sample_size: int = 300,
        target_classes: Optional[List[int]] = None,
        versions: Union[str, List[str]] = "RAW",
        crop_size: Tuple = (1, 512, 512),
        downsample_size: Tuple = (1, 128, 128),
        fold: int = 0,
        train: bool = False,
        processed_dir: Optional[str] = None,
        gate_with: Optional[Union[str, int, float]] = None,
        crop_mode: str = 'pixel_crop',
        blur_method: str = 'circular',
) -> Tuple[List, List, List, List, List[str], List[str]]:
    """
    Load PSZ2 galaxy cluster images for one cross-validation fold.

    Data pipeline:
        1. Download with scripts/01.rsync_PSZ2.py
        2. Categorise with scripts/02.categorise_PSZ2.py
        3. Process with scripts/03.create_processed_images.py

    Args:
        path:           Root directory containing per-source FITS folders.
        sample_size:    Max samples per class in the training set.
        target_classes: List of class tags to load (e.g. [50, 51]).
                        Defaults to [50, 51] if None.
        versions:       Version string or list (e.g. 'T50kpc', ['T25kpc','T50kpc']).
        crop_size:      (C, H, W) centre-crop size (used by pixel_crop only).
        downsample_size:(C, H, W) output size after resizing.
        fold:           Cross-validation fold index (0–9).
        train:          If True, use 10-fold CV on the train+val set.
                        If False, use full train+val as training, test as eval.
        processed_dir:  Override the processed FITS directory (optional).
        gate_with:      Restrict sources to those also present in a given
                        T version dir. 'auto' derives from versions; int/str
                        specify a kpc scale explicitly.
        crop_mode:      Cropping strategy: 'pixel_crop' | 'beam_crop' | 'fov_crop'.
                        Default: 'pixel_crop'.
        blur_method:    Blurring kernel: 'circular' | 'circular_no_sub' | 'cheat'.
                        Ignored for pixel_crop. Default: 'circular'.

    Returns:
        (train_images, train_labels, eval_images, eval_labels,
         train_filenames, eval_filenames)
    """
    # Avoid mutable default argument
    if target_classes is None:
        target_classes = [50, 51]

    # Validate crop_mode and blur_method
    if crop_mode not in _CROP_MODE_MAP:
        raise ValueError(f"Unknown crop_mode {crop_mode!r}. "
                         f"Choose from: {list(_CROP_MODE_MAP)}")
    if crop_mode != 'pixel_crop' and blur_method not in _BLUR_METHOD_MAP:
        raise ValueError(f"Unknown blur_method {blur_method!r}. "
                         f"Choose from: {list(_BLUR_METHOD_MAP)}")

    # Resolve processed_dir and filename suffix
    _subdir = _processed_subdir(crop_mode, blur_method)
    filename_suffix = blur_method if _subdir is not None else None
    if _subdir is not None and processed_dir is None:
        processed_dir = os.path.join(_PSZ2_DATA_DIR, _subdir)

    print("Parameters:")
    print("  path:", path)
    print("  versions:", versions)
    print("  crop_mode:", crop_mode)
    print("  blur_method:", blur_method)
    print("  filename_suffix:", filename_suffix)
    print("  crop_size:", crop_size)
    print("  downsample_size:", downsample_size)
    print("  target_classes:", target_classes)
    print("  processed_dir:", processed_dir)
    print("  gate_with:", gate_with)
    print("  train:", train)
    print("  fold:", fold)

    def _kpc_tag(v: Any) -> str:
        """Canonical kpc tag for a version string, e.g. 'T50kpc', 'Blur25kpc'."""
        vU = str(v).upper()
        if vU.startswith("BLUR"):
            num = ''.join(c for c in str(v) if c.isdigit())
            return f"Blur{num}kpc"
        if vU.startswith("T"):
            m = re.search(r'T(\d+)kpc', vU)
            if m:
                return f"T{m.group(1)}kpc"
        return str(v)

    def _nearest_T_dir(root_path: str, subfolder: str, target_num: float) -> Optional[str]:
        """Find the TXXkpc folder whose kpc value is nearest to target_num."""
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
            if os.path.isdir(os.path.join(root_path, d, subfolder)):
                cand.append((abs(y - target_num), y, d))
        if not cand:
            return None
        cand.sort(key=lambda t: (t[0], t[1]))
        return cand[0][2]

    def _canon(size: Tuple) -> Tuple[int, int, int]:
        if len(size) == 2: return (1, size[0], size[1])
        if len(size) == 3: return size
        raise ValueError("crop_size/downsample_size must be (H,W) or (C,H,W)")

    _, Hc_ref, Wc_ref = _canon(crop_size)
    _, Ho, Wo         = _canon(downsample_size)

    # Equal-beams pre-scan (only when using processed files)
    if processed_dir is not None:
        EQUAL_TAPER = _pick_equal_taper_from(versions)
        n_beams_min, _T_header_cache = _scan_min_beams(path, target_classes, taper=EQUAL_TAPER)
    else:
        n_beams_min, _T_header_cache = None, {}

    classes_map = {c["tag"]: c["description"] for c in get_classes()}

    images: List[torch.Tensor] = []
    labels: List[int]          = []
    basenames: List[str]       = []

    def _source_id(base: str) -> str:
        """Strip TXXkpc suffix from a basename to get the bare source ID."""
        return re.sub(r'T\d+kpc.*$', '', base)

    _seen_sources: set = set()

    # =========================================================================
    # MULTI-VERSION STACK
    # =========================================================================
    if isinstance(versions, (list, tuple)) and len(versions) > 1:

        norm_versions = [_canon_ver(v) for v in versions]

        # Find a reference T version to enumerate sources
        ref_T_version: Optional[str] = None
        for v in norm_versions:
            if str(v).upper().startswith('T') and not str(v).upper().startswith('BLUR'):
                ref_T_version = v
                print(f"[DEBUG] Reference T version from list: {ref_T_version}")
                break

        if ref_T_version is None:
            for test_v in ['T50kpc', 'T100kpc', 'T25kpc']:
                if os.path.isdir(os.path.join(path, test_v)):
                    ref_T_version = test_v
                    print(f"[DEBUG] Default reference T version: {ref_T_version}")
                    break

        if ref_T_version is not None:
            for cls in target_classes:
                sub = classes_map.get(cls)
                if not sub:
                    continue

                ref_dir = os.path.join(path, ref_T_version, sub)
                if not os.path.isdir(ref_dir):
                    print(f"[DEBUG] Reference directory missing: {ref_dir}")
                    continue

                source_ids: set = set()
                for f in os.listdir(ref_dir):
                    if f.lower().endswith('.fits'):
                        src_id = _source_id(os.path.splitext(f)[0])
                        source_ids.add(src_id)

                if processed_dir is None:
                    raise ValueError(
                        f"Multi-version loading requires processed files. "
                        f"crop_mode={crop_mode!r} has no processed_subdir.")

                # Debug: check first 5 sources
                for src_id in sorted(source_ids)[:5]:
                    has_all = all(
                        os.path.isfile(os.path.join(
                            processed_dir,
                            f"{src_id}_{_kpc_tag(nv)}_fmt_{Ho}x{Wo}_{filename_suffix}.fits"))
                        for nv in norm_versions)
                    if has_all:
                        print(f"[DEBUG]   ✓ {src_id} has all versions")

                # Full valid-source check
                valid_sources = [
                    src_id for src_id in source_ids
                    if all(os.path.isfile(os.path.join(
                               processed_dir,
                               f"{src_id}_{_kpc_tag(nv)}_fmt_{Ho}x{Wo}_{filename_suffix}.fits"))
                           for nv in norm_versions)
                ]

                for src_id in valid_sources:
                    frames: List[torch.Tensor] = []
                    ok = True
                    for nv in norm_versions:
                        proc_path = os.path.join(
                            processed_dir,
                            f"{src_id}_{_kpc_tag(nv)}_fmt_{Ho}x{Wo}_{filename_suffix}.fits")
                        try:
                            arr = np.squeeze(fits.getdata(proc_path)).astype(np.float32)
                            if arr.ndim == 3: arr = arr.mean(axis=0)
                            if arr.ndim != 2: ok = False; break
                            frames.append(torch.from_numpy(arr).unsqueeze(0).float())
                        except Exception as e:
                            print(f"[DEBUG] Failed to load {proc_path}: {e}")
                            ok = False; break

                    if not ok or len(frames) != len(norm_versions):
                        continue

                    images.append(torch.stack(frames, dim=0))
                    labels.append(cls)
                    basenames.append(src_id)

    # =========================================================================
    # SINGLE-VERSION PATH
    # =========================================================================
    else:
        vU  = versions[0].upper() if isinstance(versions, (list, tuple)) else str(versions).upper()
        tag = _kpc_tag(versions[0] if isinstance(versions, (list, tuple)) else versions)
        print("Processing single version:", vU, "tag:", tag)

        for cls in target_classes:
            sub = classes_map.get(cls)
            if not sub:
                continue

            raw_dir = os.path.join(path, "RAW", sub)
            if not os.path.isdir(raw_dir):
                print(f"[SKIP] RAW folder missing: {raw_dir}")
                continue

            # Gating: restrict to sources also present in a TXXkpc directory
            gate_keys:    Optional[set] = None
            gate_dirname: Optional[str] = None

            if gate_with is not None:
                desired_num: Optional[float] = None
                if isinstance(gate_with, str) and gate_with.lower() == "auto":
                    m_rt = re.search(r"BLUR(\d+(?:\.\d+)?)", vU)
                    m_t  = re.search(r"T(\d+(?:\.\d+)?)KPC", vU)
                    if m_rt:   desired_num = float(m_rt.group(1))
                    elif m_t:  desired_num = float(m_t.group(1))
                    else:      desired_num = 50.0
                elif isinstance(gate_with, (int, float)):
                    desired_num = float(gate_with)
                elif isinstance(gate_with, str):
                    m = re.search(r"T(\d+(?:\.\d+)?)KPC", gate_with.upper())
                    if m: desired_num = float(m.group(1))

                if desired_num is not None:
                    preferred_gate = f"T{int(desired_num) if float(desired_num).is_integer() else desired_num}kpc"
                    gate_dirname = (preferred_gate
                                    if os.path.isdir(os.path.join(path, preferred_gate, sub))
                                    else _nearest_T_dir(path, sub, desired_num))

                if gate_dirname is None:
                    print(f"[GATE] No TXXkpc found for gate_with='{gate_with}', sub='{sub}'. No gating applied.")
                else:
                    gate_dir = os.path.join(path, gate_dirname, sub)
                    raw_map  = {os.path.splitext(f)[0].lower() for f in os.listdir(raw_dir) if f.lower().endswith(".fits")}
                    txx_map  = {os.path.splitext(f)[0].lower() for f in os.listdir(gate_dir) if f.lower().endswith(".fits")}
                    gate_keys = raw_map & txx_map
                    print(f"[GATE] Using {gate_dirname} for sub='{sub}' ({len(gate_keys)} sources).")

            for fname in sorted(os.listdir(raw_dir)):
                if not fname.lower().endswith(".fits"):
                    continue
                base = os.path.splitext(fname)[0]
                src  = _source_id(base)
                if src in _seen_sources:
                    continue
                if gate_keys is not None and base.lower() not in gate_keys:
                    continue

                if processed_dir is not None and (
                        vU.startswith("T") or vU.startswith("BLUR") or vU == "RAW"):
                    proc_path = os.path.join(
                        processed_dir,
                        f"{src}_{tag}_fmt_{Ho}x{Wo}_{filename_suffix}.fits")
                    if os.path.isfile(proc_path):
                        arr = np.squeeze(fits.getdata(proc_path)).astype(np.float32)
                        if arr.ndim == 3: arr = arr.mean(axis=0)
                        if arr.ndim == 2:
                            images.append(torch.from_numpy(arr).unsqueeze(0).float())
                            labels.append(cls)
                            basenames.append(src)
                            _seen_sources.add(src)
                            continue
                    else:
                        print(f"[MISS] processed not found for class={sub}: {proc_path}")
                        continue

                # Fallback: generate on-the-fly from raw FITS
                fpath = os.path.join(raw_dir, fname)

                if vU.startswith("T"):
                    t_path = os.path.join(path, tag, sub, f"{base}.fits")
                    if not os.path.isfile(t_path):
                        print(f"[MISS] tapered FITS missing: {t_path}")
                        continue
                    arr = np.squeeze(fits.getdata(t_path)).astype(np.float32)
                    if arr.ndim == 3: arr = arr.mean(axis=0)
                    if arr.ndim != 2: continue
                    hdr = fits.getheader(t_path)
                    Hc_eff, Wc_eff = Hc_ref, Wc_ref
                    if n_beams_min is not None:
                        fwhm_as = max(float(hdr["BMAJ"]), float(hdr["BMIN"])) * 3600.0
                        side_as = n_beams_min * fwhm_as
                        px, py  = _pix_scales_arcsec(hdr)
                        Hc_eff  = max(1, int(round(side_as / py)))
                        Wc_eff  = max(1, int(round(side_as / px)))
                    frm = apply_formatting(
                        torch.from_numpy(arr).unsqueeze(0).float(),
                        crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))

                elif vU.startswith("BLUR"):
                    arr = np.squeeze(fits.getdata(fpath)).astype(np.float32)
                    if arr.ndim == 3: arr = arr.mean(axis=0)
                    if arr.ndim != 2: continue
                    raw_hdr = fits.getheader(fpath)
                    Hc_eff, Wc_eff = Hc_ref, Wc_ref

                    num = ''.join([c for c in vU if c.isdigit()]) or "50"
                    preferred_gate = f"T{int(float(num)) if float(num).is_integer() else num}kpc"
                    gate_dirname = (preferred_gate
                                    if os.path.isdir(os.path.join(path, preferred_gate, sub))
                                    else _nearest_T_dir(path, sub, float(num)))
                    if gate_dirname is None:
                        print(f"[GATE] No TXXkpc for Blur{num}kpc, sub='{sub}'. Skipping {src}.")
                        continue

                    txx_path = os.path.join(path, gate_dirname, sub, f"{base}{gate_dirname}.fits")
                    if not os.path.isfile(txx_path):
                        continue
                    txx_hdr = fits.getheader(txx_path)

                    # Build circularised convolution kernel in pixel space
                    C_raw_w  = _beam_cov_world(raw_hdr)
                    C_tgt_w  = _beam_cov_world(txx_hdr)
                    sigma2   = float(np.sqrt(max(0.0, np.linalg.det(C_tgt_w))))
                    C_circ_w = np.array([[sigma2, 0.0], [0.0, sigma2]], float)
                    if vU.startswith("BLURNOSUB"):
                        # No beam subtraction: C_ker = C_target (final PSF = C_beam + C_target)
                        C_ker_w = C_circ_w
                    else:
                        C_ker_w  = C_circ_w - C_raw_w
                        w, V     = np.linalg.eigh(C_ker_w)
                        C_ker_w  = (V * np.clip(w, 0.0, None)) @ V.T

                    J    = _cd_matrix_rad(raw_hdr)
                    Cpix = np.linalg.inv(J) @ C_ker_w @ np.linalg.inv(J).T

                    evals, evecs = np.linalg.eigh(Cpix)
                    evals = np.clip(evals, 1e-18, None)
                    s1, s2 = float(np.sqrt(evals[0])), float(np.sqrt(evals[1]))
                    nker   = int(np.ceil(8.0 * max(s1, s2))) | 1
                    k      = (nker - 1) // 2
                    yy, xx = np.mgrid[-k:k+1, -k:k+1].astype(np.float32)
                    X      = np.stack([xx, yy], axis=-1)
                    Cinv   = evecs @ np.diag(1.0 / np.array([s1*s1, s2*s2])) @ evecs.T
                    ker    = np.exp(-0.5 * (X @ Cinv * X).sum(axis=-1))
                    s      = float(ker.sum())
                    if not np.isfinite(s) or s <= 0:
                        print(f"  [SKIP] degenerate kernel for {src}")
                        continue
                    ker /= s

                    arr = convolve_fft(
                        arr, ker, boundary="fill", fill_value=np.nan,
                        nan_treatment="interpolate", normalize_kernel=True,
                        psf_pad=True, fft_pad=True, allow_huge=True)
                    arr *= _beam_solid_angle_sr(txx_hdr) / _beam_solid_angle_sr(raw_hdr)

                    if n_beams_min is not None:
                        fwhm_as = max(float(txx_hdr["BMAJ"]), float(txx_hdr["BMIN"])) * 3600.0
                        side_as = n_beams_min * fwhm_as
                        px, py  = _pix_scales_arcsec(raw_hdr)
                        Hc_eff  = max(1, int(round(side_as / py)))
                        Wc_eff  = max(1, int(round(side_as / px)))

                    frm = apply_formatting(
                        torch.from_numpy(arr).unsqueeze(0).float(),
                        crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))

                elif vU == "RAW":
                    arr = np.squeeze(fits.getdata(fpath)).astype(np.float32)
                    if arr.ndim == 3: arr = arr.mean(axis=0)
                    if arr.ndim != 2: continue
                    raw_hdr = fits.getheader(fpath)
                    Hc_eff, Wc_eff = Hc_ref, Wc_ref
                    if n_beams_min is not None:
                        fwhm_as = max(float(raw_hdr["BMAJ"]), float(raw_hdr["BMIN"])) * 3600.0
                        side_as = n_beams_min * fwhm_as
                        px, py  = _pix_scales_arcsec(raw_hdr)
                        Hc_eff  = max(1, int(round(side_as / py)))
                        Wc_eff  = max(1, int(round(side_as / px)))
                    frm = apply_formatting(
                        torch.from_numpy(arr).unsqueeze(0).float(),
                        crop_size=(1, Hc_eff, Wc_eff), downsample_size=(1, Ho, Wo))
                else:
                    continue

                images.append(frm)
                labels.append(cls)
                basenames.append(src)
                _seen_sources.add(src)

    # =========================================================================
    # STRATIFIED TRAIN / EVAL SPLIT
    # =========================================================================
    y = np.array(labels)

    if len(y) == 0:
        try:
            v_tag = _kpc_tag(versions[0] if isinstance(versions, (list, tuple)) else versions)
        except Exception:
            v_tag = str(versions)
        if processed_dir and filename_suffix:
            detail = f"Looked for: {os.path.join(processed_dir, f'*_{v_tag}_fmt_{Ho}x{Wo}_{filename_suffix}.fits')}"
        else:
            detail = f"crop_mode={crop_mode!r}, path={path!r}, versions={versions!r}"
        raise ValueError(f"[PSZ2] No samples collected. {detail}")

    print("Total sources loaded:", len(basenames))
    groups = np.array(basenames)

    # Fixed-seed split: 80% train+val, 20% test
    initial_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=41)
    trainval_idx, test_idx = next(initial_sgkf.split(np.zeros(len(y)), y, groups))

    if train:
        # 10-fold CV on the train+val subset (for hyperparameter search)
        print(f"Using train+val split (n={len(trainval_idx)}) with 10-fold CV")
        if fold is None or fold < 0 or fold >= 10:
            raise ValueError("For train=True, fold must be an integer 0–9.")
        y_trainval      = y[trainval_idx]
        groups_trainval = groups[trainval_idx]
        sgkf_cv   = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
        cv_splits = list(sgkf_cv.split(np.zeros(len(y_trainval)), y_trainval, groups_trainval))
        tr_idx_rel, va_idx_rel = cv_splits[fold]
        tr_idx = trainval_idx[tr_idx_rel]
        va_idx = trainval_idx[va_idx_rel]
    else:
        # Full train+val as training, held-out test as eval
        print(f"Using full train+val (n={len(trainval_idx)}) as training; test (n={len(test_idx)}) as eval.")
        tr_idx = trainval_idx
        va_idx = test_idx

    def _take(idxs: np.ndarray) -> Tuple[List, List, List[str]]:
        return ([images[i] for i in idxs],
                [labels[i] for i in idxs],
                [basenames[i] for i in idxs])

    train_images, train_labels, train_fns = _take(tr_idx)
    eval_images,  eval_labels,  eval_fns  = _take(va_idx)

    return train_images, train_labels, eval_images, eval_labels, train_fns, eval_fns


##########################################################################################
################################## MASTER LOADER #########################################
##########################################################################################


def load_galaxies(
        galaxy_classes: List[int],
        path: Optional[str] = None,
        versions: Optional[Union[str, List[str]]] = None,
        fold: Optional[int] = None,
        island: Optional[Any] = None,
        crop_size: Optional[Tuple] = None,
        downsample_size: Optional[Tuple] = None,
        sample_size: Optional[int] = None,
        REMOVEOUTLIERS: bool = True,
        BALANCE: bool = False,
        AUGMENT: bool = False,
        USE_GLOBAL_NORMALISATION: bool = False,
        GLOBAL_NORM_MODE: str = "percentile",
        STRETCH: bool = False,
        alpha: float = 10.0,
        percentile_lo: float = 30,
        percentile_hi: float = 99,
        NORMALISE: bool = True,
        NORMALISETOPM: bool = False,
        EXTRADATA: bool = False,
        PRINTFILENAMES: bool = False,
        SAVE_IMAGES: bool = False,
        train: Optional[bool] = None,
        USE_CACHE: bool = True,
        DEBUG: bool = True,
        crop_mode: str = 'pixel_crop',
        blur_method: str = 'circular',
) -> Tuple:
    """
    Master loader: delegates to the appropriate dataset loader, applies
    normalisation, augmentation, and caching.

    Args:
        galaxy_classes:           List of class tags to load (e.g. [50, 51]).
        path:                     Override default data root path.
        versions:                 Image version(s) to load (e.g. 'T50kpc').
        fold:                     Cross-validation fold index.
        island:                   Unused legacy parameter.
        crop_size:                (C, H, W) centre-crop size.
        downsample_size:          (C, H, W) output size.
        sample_size:              Max samples per class.
        REMOVEOUTLIERS:           Remove statistical outliers (not yet implemented).
        BALANCE:                  Down-sample majority class to minority size.
        AUGMENT:                  Apply rotation/flip augmentation.
        USE_GLOBAL_NORMALISATION: Normalise all images jointly to [0, 1].
        GLOBAL_NORM_MODE:         'percentile' or 'minmax' (used with global norm).
        STRETCH:                  Apply asinh stretch after normalisation.
        alpha:                    Asinh stretch scale factor.
        percentile_lo:            Lower percentile for per-image stretch.
        percentile_hi:            Upper percentile for per-image stretch.
        NORMALISE:                Apply normalisation (global or per-image).
        NORMALISETOPM:            Re-normalise to [-1, 1] after main normalisation.
        crop_mode:                Cropping strategy passed to load_PSZ2.
        EXTRADATA:                Also return PSZ2 metadata tensors.
        PRINTFILENAMES:           Also return source filename lists.
        SAVE_IMAGES:              Save per-class image arrays to .npy files.
        train:                    Passed to load_PSZ2 to select CV split mode.
        USE_CACHE:                Load from / save to disk cache.
        DEBUG:                    Print diagnostic plots and tensor stats.

    Returns:
        (train_images, train_labels, eval_images, eval_labels)
        or with filenames/metadata appended depending on flags.
    """
    if USE_CACHE:
        cache_key = _build_cache_key(
            galaxy_classes, versions, fold, crop_size, downsample_size,
            sample_size, REMOVEOUTLIERS, BALANCE, AUGMENT, STRETCH,
            percentile_lo, percentile_hi, NORMALISE, NORMALISETOPM,
            USE_GLOBAL_NORMALISATION, GLOBAL_NORM_MODE, train, crop_mode)

        cached_data = _load_cache(cache_key)
        if cached_data is not None:
            if PRINTFILENAMES and len(cached_data) == 6:
                return cached_data
            elif not PRINTFILENAMES and len(cached_data) == 4:
                return cached_data
            elif PRINTFILENAMES and len(cached_data) == 4:
                print("⚠ Cache exists but lacks filenames — regenerating.")
            else:
                return cached_data[:4]

    # Build kwargs, omitting None values (so loader defaults take effect)
    kwargs = {
        'path': path, 'versions': versions, 'sample_size': sample_size,
        'fold': fold, 'train': train,
        'crop_size': crop_size, 'downsample_size': downsample_size,
        'crop_mode': crop_mode, 'blur_method': blur_method,
    }
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    max_class = max(galaxy_classes) if isinstance(galaxy_classes, list) else galaxy_classes

    if 50 <= max_class <= 59:
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_PSZ2(target_classes=target_classes, **clean_kwargs)
    else:
        raise ValueError(f"Invalid galaxy class: {max_class}. Only PSZ2 classes (50–59) are supported.")

    if len(data) == 4:
        train_images, train_labels, eval_images, eval_labels = data
        train_filenames = eval_filenames = None
    elif len(data) == 6:
        train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames = data
        overlap = set(train_filenames) & set(eval_filenames)
        assert not overlap, f"PSZ2 split error — source(s) in both sets: {overlap}"
    else:
        raise ValueError(f"Data loader returned unexpected number of outputs: {len(data)}")

    # ── Normalisation ──────────────────────────────────────────────────────────
    if NORMALISE:
        if DEBUG:
            plot_class_images(get_classes(), train_images, eval_images,
                              train_labels, eval_labels,
                              train_filenames, eval_filenames,
                              set_name='1.before_normalisation')
            for cls in set(train_labels):
                check_tensor(f"train class {cls} before norm",
                             [img for img, lbl in zip(train_images, train_labels) if lbl == cls])
            for cls in set(eval_labels):
                check_tensor(f"eval class {cls} before norm",
                             [img for img, lbl in zip(eval_images, eval_labels) if lbl == cls])

        if isinstance(train_images, list): train_images = torch.stack(train_images)
        if isinstance(eval_images,  list): eval_images  = torch.stack(eval_images)
        all_images = torch.cat([train_images, eval_images], dim=0)

        if USE_GLOBAL_NORMALISATION:
            if DEBUG: print("Applying global normalisation to [0, 1]")
            all_images = normalise_images(all_images, out_min=0, out_max=1)
        else:
            if DEBUG: print(f"Applying per-image percentile stretch [{percentile_lo}, {percentile_hi}]%")
            if all_images.ndim == 5:        # [B, T, C, H, W]
                for t in range(all_images.shape[1]):
                    all_images[:, t] = per_image_percentile_stretch(
                        all_images[:, t], percentile_lo, percentile_hi)
            else:                           # [B, C, H, W]
                all_images = per_image_percentile_stretch(all_images, percentile_lo, percentile_hi)

        n_tr = len(train_images)
        train_images = all_images[:n_tr]
        eval_images  = all_images[n_tr:]

        if NORMALISETOPM:
            all_images   = torch.cat([train_images, eval_images], dim=0)
            all_images   = normalise_images(all_images, out_min=-1, out_max=1)
            train_images = all_images[:n_tr]
            eval_images  = all_images[n_tr:]

        if DEBUG:
            plot_class_images(get_classes(), train_images, eval_images,
                              train_labels, eval_labels,
                              train_filenames, eval_filenames,
                              set_name='2.after_normalisation')

    # ── Asinh stretch ──────────────────────────────────────────────────────────
    if STRETCH:
        all_images   = torch.cat([train_images, eval_images], dim=0)
        stretched    = torch.asinh(all_images * alpha) / math.asinh(alpha)
        n_tr         = train_images.shape[0]
        train_images = stretched[:n_tr]
        eval_images  = stretched[n_tr:]
        if DEBUG:
            print(f"Applied asinh stretch (alpha={alpha})")

    # ── Overlap check ──────────────────────────────────────────────────────────
    if len(eval_images) > 0:
        train_hashes = {img_hash(img) for img in train_images}
        eval_hashes  = {img_hash(img) for img in eval_images}
        common       = train_hashes & eval_hashes
        if common:
            print(f"🔍 {len(common)} pixel-identical image(s) in train and eval.")
            plot_pixel_overlaps_side_by_side(
                train_images, eval_images,
                train_filenames=train_filenames,
                eval_filenames=eval_filenames,
                max_hashes=min(50, len(common)),
                outdir="./overlap_debug")
            raise AssertionError(
                f"Overlap: {len(common)} image(s) in both train and eval. "
                f"See ./overlap_debug/")

    # ── Tensor conversion ──────────────────────────────────────────────────────
    if isinstance(train_images, list): train_images = torch.stack(train_images)
    if isinstance(eval_images,  list): eval_images  = torch.stack(eval_images)

    # ── Class balancing ────────────────────────────────────────────────────────
    if BALANCE:
        if DEBUG: print("Balancing classes in training set…")
        train_images, train_labels = balance_classes(train_images, train_labels)

    # ── Label conversion ───────────────────────────────────────────────────────
    if isinstance(train_labels, torch.Tensor): train_labels = train_labels.tolist()
    if isinstance(eval_labels,  torch.Tensor): eval_labels  = eval_labels.tolist()

    # ── Optional metadata load ────────────────────────────────────────────────
    if EXTRADATA and not PRINTFILENAMES:
        if DEBUG: print("Loading PSZ2 metadata")
        meta_df = pd.read_csv(os.path.join(ROOT_PATH, "PSZ2/cluster_source_data.csv"))
        meta_df.rename(columns={"slug": "base"}, inplace=True)
        meta_df.set_index("base", inplace=True)
        train_data = [meta_df.loc[base].values for base in train_filenames]
        eval_data  = [meta_df.loc[base].values for base in eval_filenames]

    # ── Augmentation ──────────────────────────────────────────────────────────
    if AUGMENT:
        if DEBUG: print("Applying data augmentation…")
        train_images, train_labels = augment_images(train_images, train_labels)
        if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53):
            eval_images, eval_labels = augment_images(eval_images, eval_labels)
        else:
            if eval_images.dim() == 3: eval_images = eval_images.unsqueeze(1)
            if isinstance(eval_images, (list, tuple)): eval_images = torch.stack(eval_images)
            if isinstance(eval_labels, (list, tuple)):
                eval_labels = torch.tensor(eval_labels, dtype=torch.long)
        if PRINTFILENAMES:
            n_aug = 24
            train_filenames = [fn for fn in train_filenames for _ in range(n_aug)]
            if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53):
                eval_filenames = [fn for fn in eval_filenames for _ in range(n_aug)]
    else:
        if train_images.dim() == 3: train_images = train_images.unsqueeze(1)
        if eval_images.dim()  == 3: eval_images  = eval_images.unsqueeze(1)
        if isinstance(train_images, (list, tuple)): train_images = torch.stack(train_images)
        if isinstance(eval_images,  (list, tuple)): eval_images  = torch.stack(eval_images)
        if isinstance(train_labels, (list, tuple)):
            train_labels = torch.tensor(train_labels, dtype=torch.long)
        if isinstance(eval_labels,  (list, tuple)):
            eval_labels  = torch.tensor(eval_labels,  dtype=torch.long)

    # ── Cache save ────────────────────────────────────────────────────────────
    if USE_CACHE:
        cache_data = (train_images, train_labels, eval_images, eval_labels,
                      *(([train_filenames, eval_filenames]) if PRINTFILENAMES else []))
        _save_cache(cache_key, cache_data)

    # ── Return ────────────────────────────────────────────────────────────────
    if PRINTFILENAMES:
        return train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames
    if EXTRADATA:
        return (train_images, train_labels, eval_images, eval_labels,
                torch.tensor(np.stack(train_data), dtype=torch.float32),
                torch.tensor(np.stack(eval_data),  dtype=torch.float32))
    return train_images, train_labels, eval_images, eval_labels