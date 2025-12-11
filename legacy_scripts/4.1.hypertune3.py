import sys, os, glob, re, csv, torch, os, math, hashlib, time, random
from utils.data_loader import load_galaxies, get_classes,  get_synthetic, augment_images, apply_formatting
from utils.classifiers import (
    RustigeClassifier, TinyCNN, MLPClassifier, SCNN, CNNSqueezeNet, ScatterResNet,
    BinaryClassifier, ScatterSqueezeNet, ScatterSqueezeNet2,
    DualCNNSqueezeNet, DualInputConvolutionalSqueezeNet, DISSN)
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram
from pathlib import Path
from astropy.cosmology import Planck18 as COSMO
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim import AdamW
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from kymatio.torch import Scattering2D
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm
import itertools
from itertools import product
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt


SEED = 42  # Set a seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make cuDNN deterministic (may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)
matplotlib.use('Agg')
tqdm.pandas(disable=True)

print("Running ht1 with seed", SEED)

###############################################
################ CONFIGURATION ################
###############################################

PSZ2_ROOT = "/users/mbredber/scratch/data/PSZ2"  # FITS root used below

galaxy_classes    = [50, 51]
max_num_galaxies  = 1000000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions  = [1]
J, L, order       = 2, 12, 2
num_epochs_cuda = 200
num_epochs_cpu = 100
folds = [5]  # e.g., [1, 2, 3, 4, 5] for five-fold cross-validation
lambda_values = [0]  # Ratio between generated images and original images per class. 8 is reserfved for TRAINONGENERATED
num_experiments = 3
gen_model_name = 'DDPM'  # Name of the generative model used to create generated images

# pick exactly one classifier
classifier        = ["TinyCNN",       # Very Simple CNN
                     "Rustige",       # from Rustige et al. 2023
                     "SCNN",          # simple CNN variant
                     "CNNSqueezeNet", # with SE blocks
                     "DualCNNSqueezeNet", # Dual mode Convolutional SqueezeNet
                     "DICSN",          # Dual-Input Convolutional SqueezeNet
                     "DISSN",          # Dual-Input Scatter SqueezeNet
                     "ScatterNet",
                     "ScatterSqueezeNet",
                     "ScatterSqueezeNet2",
                     "Binary",
                     "ScatterResNet"][-4]

# Define every value you want to try
param_grid = {
    'lr':            [1e-3],
    'reg':           [1e-3],
    'label_smoothing':[0.2],
    'J':             [2],          # Only used for scattering classifiers
    'L':             [12],         # Only used for scattering classifiers
    'order':         [2],          # Only used for scattering classifiers
    'percentile_lo': [1, 30, 60],  # Only used for data normalisation
    'percentile_hi': [80, 90, 99], # Only used for data normalisation
    'crop_size':     [(512,512)],  # Not used for preformatted data
    'downsample_size':[(128,128)], # Not used for preformatted data
    'versions':       ['RAW']  # 'raw', 'T50kpc', ad hoc tapering: e.g. 'rt50'  strings in list → product() iterates them individually
} #'versions': [('raw', 'rt50')]  # tuple signals “stack these”

STRETCH = True  # Arcsinh stretch
USE_GLOBAL_NORMALISATION = False           # single on/off switch . False - image-by-image normalisation 
GLOBAL_NORM_MODE = "percentile"           # "percentile" or "flux". Becomes "none" if USE_GLOBAL_NORMALISATION is 
NORMALISEIMGS = True  # Normalise images to [0, 1]
NORMALISEIMGSTOPM = False  # Normalise images to [-1, 1] 
NORMALISESCS = False  # Normalise scattering coefficients to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients to [-1, 1]
FILTERED = True  # Remove in training, validation and test data for the classifier
FILTERGEN = False  # Remove generated images that are too similar to other generated images
AUGMENT = True  # Use classical data augmentation (flips, rotations)
PRINTFILENAMES = True
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights
TRAINONGENERATED = False  # Use generated data as testdata
ES, patience = True, 10  # Use early stopping
SCHEDULER = False  # Use a learning rate scheduler
SHOWIMGS = True  # Show some generated images for each class (Tool for control)


########################################################################
##################### AUTOMATIC CONFIGURATION ##########################
########################################################################

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple silicon fallback if relevant
    num_epochs = num_epochs_cpu
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
print(f"{DEVICE.upper()} is available. Setting epochs to {num_epochs}.")

# --- optional RT knobs via env (keep default off to remain reproducible) ---
APPLY_UV_TAPER = os.getenv("RT_USE_UV_TAPER", "0") == "1"   # default off
UV_TAPER_FRAC  = float(os.getenv("RT_UV_TAPER_FRAC", "0.0"))  # e.g. 0.2 (20% of target FWHM)

if TRAINONGENERATED:
    lambda_values = [8]  # To identify and distinguish TRAINONGENERATED from other runs
    print("Using generated data for testing.")
    
# —— MULTI-LABEL SWITCH for RH/RR ——
if galaxy_classes == [52, 53]:
    from utils.data_loader import load_halos_and_relics
    import seaborn as sns
    MULTILABEL = True            # predict RH and RR independently
    LABEL_INDEX = {"RH": 0, "RR": 1}
    THRESHOLD = 0.5
else:
    MULTILABEL = False

_loader = load_halos_and_relics if galaxy_classes == [52, 53] else load_galaxies
BALANCE = True if galaxy_classes == [52, 53] else False  # Balance the dataset by undersampling the majority class


########################################################################
##################### HELPER FUNCTIONS #################################
########################################################################

def plot_first_rows_by_source(images, filenames, versions, out_path, n_show=10):
    """
    Plot the first N rows titled by source name. If images are [B,T,1,H,W] or [B,T,H,W],
    show 2 columns (left/right = first/second plane). If single version, plot 1 column.
    """
    if isinstance(images, (list, tuple)):
        images = torch.stack([torch.as_tensor(x) for x in images], dim=0)

    # Normalize shape to convenient form
    if images.dim() == 5:                 # [B, T, 1, H, W]
        images = images.flatten(2, 3)     # [B, T, H, W]
    elif images.dim() == 4:               # [B, C, H, W]
        pass
    else:
        raise ValueError(f"Unsupported images ndim={images.ndim}")

    B = images.shape[0]
    n_show = min(n_show, B)

    # Row titles from filenames (strip trailing T*kpc or T*kpcSUB)
    names = (filenames[:n_show] if filenames else [f"idx_{i}" for i in range(n_show)])
    def _src_name(s):
        b = os.path.splitext(os.path.basename(str(s)))[0]
        return re.sub(r'(?:T\d+kpc(?:SUB)?)$', '', b)
    row_titles = [_src_name(s) for s in names]

    is_two_cols = images.shape[1] >= 2      # have at least two planes (e.g., RT/T)
    if is_two_cols:
        fig, axes = plt.subplots(n_show, 2, figsize=(5.4, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = np.array([axes])
        for i in range(n_show):
            left  = images[i, 0].detach().cpu().numpy()
            right = images[i, 1].detach().cpu().numpy()
            axes[i, 0].imshow(left, cmap="viridis", origin="lower")
            axes[i, 0].set_title(f"{row_titles[i]} — {versions[0] if isinstance(versions,(list,tuple)) else 'v0'}")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(right, cmap="viridis", origin="lower")
            axes[i, 1].set_title(f"{row_titles[i]} — {versions[1] if isinstance(versions,(list,tuple)) else 'v1'}")
            axes[i, 1].axis('off')
    else:
        fig, axes = plt.subplots(n_show, 1, figsize=(2.7, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = [axes]
        for i in range(n_show):
            img = images[i, 0].detach().cpu().numpy()   # first/only channel
            ax = axes[i]
            ax.imshow(img, cmap="viridis", origin="lower")
            ax.set_title(f"{row_titles[i]} — {versions if not isinstance(versions,(list,tuple)) else versions[0]}")
            ax.axis('off')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[quicklook] wrote {out_path}")

def compute_classification_metrics(y_true, y_pred, multilabel, num_classes):
    acc = accuracy_score(y_true, y_pred)
    if multilabel:
        avg = 'macro'
        return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                     recall_score(y_true, y_pred, average=avg, zero_division=0), \
                     f1_score(y_true, y_pred, average=avg, zero_division=0)
    if num_classes == 2:
        return acc, precision_score(y_true, y_pred, average='binary', zero_division=0), \
                     recall_score(y_true, y_pred, average='binary', zero_division=0), \
                     f1_score(y_true, y_pred, average='binary', zero_division=0)
    avg = 'macro'
    return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                 recall_score(y_true, y_pred, average=avg, zero_division=0), \
                 f1_score(y_true, y_pred, average=avg, zero_division=0)

def permute_like(x, perm):
    if x is None:
        return None

    # Torch tensors: deterministic gather via index_select and device-safe indices
    if isinstance(x, torch.Tensor):
        idx = perm.to(device=x.device, dtype=torch.long)
        return x.index_select(0, idx)

    # Prepare a plain integer index array once for non-tensor cases
    if isinstance(perm, torch.Tensor):
        idx = perm.detach().cpu().to(torch.long).tolist()
    elif isinstance(perm, np.ndarray):
        idx = [int(i) for i in perm.tolist()]
    else:
        idx = [int(i) for i in perm]

    if isinstance(x, np.ndarray):
        return x[np.asarray(idx, dtype=np.int64)]
    if isinstance(x, list):
        return [x[i] for i in idx]
    if isinstance(x, tuple):
        return tuple(x[i] for i in idx)
    return x

base_cls = min(galaxy_classes) # Remove?
def relabel(y):
    """
    Convert raw single-class ids to 2-bit multi-label targets [RH, RR].
    RH (52) -> [1,0]
    RR (53) -> [0,1]
    If you ever have 'both', set both bits to 1 *upstream*.
    """
    if MULTILABEL:
        y = y.long()
        out = torch.zeros((y.shape[0], 2), dtype=torch.float32, device=y.device)
        out[:, 0] = (y == 52).float()   # RH
        out[:, 1] = (y == 53).float()   # RR
        return out
    return (y - base_cls).long()

def as_index_labels(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=1) if y.ndim > 1 else y

def _as_5d(x):
    return x if x.dim() == 5 else x.unsqueeze(1)  # [B,1,H,W] -> [B,1,1,H,W]

def collapse_logits(logits, num_classes, multilabel):
    if logits.ndim == 4:
        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
    elif logits.ndim == 3:
        logits = logits.mean(dim=-1)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if not multilabel and logits.shape[1] == 1 and num_classes == 2:
        logits = torch.cat([-logits, logits], dim=1)
    return logits

# put near your other helpers
def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """Make images [B,C,H,W]. If [B,T,1,H,W], fold T into C."""
    if x is None:
        return x
    if x.dim() == 5:
        # [B, T, 1, H, W]  ->  [B, T, H, W] -> treat T as channels
        return x.flatten(1, 2)  # fold_T_axis does the same; this is inline & fast
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x  # already [B,C,H,W]

###############################################
########### DATA STORING FUNCTIONS ###############
###############################################

def initialize_metrics(metrics,
                    model_name, subset_size, fold, experiment,
                    lr, reg, lam,
                    crop, down, ver):
    cs = f"{crop[0]}x{crop[1]}"
    ds = f"{down[0]}x{down[1]}"
    key_base = (
        f"{model_name}_ss{subset_size}_f{fold}_lr{lr}_reg{reg}_lam{lam}_cs{cs}_ds{ds}_ver{ver}"
    )
    metrics.setdefault(f"{key_base}_accuracy", [])
    metrics.setdefault(f"{key_base}_precision", [])
    metrics.setdefault(f"{key_base}_recall", [])
    metrics.setdefault(f"{key_base}_f1_score", [])

def update_metrics(metrics,
                model_name, subset_size, fold, experiment,
                lr, reg, accuracy, precision, recall, f1, lam,
                crop, down, ver):
    cs = f"{crop[0]}x{crop[1]}"
    ds = f"{down[0]}x{down[1]}"
    
    key_base = (
    f"{model_name}"
    f"_ss{subset_size}"
    f"_f{fold}"
    f"_lr{lr}"
    f"_reg{reg}"
    f"_lam{lam}"
    f"_cs{cs}"
    f"_ds{ds}"
    f"_ver{ver}"
    )

    metrics[f"{key_base}_accuracy"].append(accuracy)
    metrics[f"{key_base}_precision"].append(precision)
    metrics[f"{key_base}_recall"].append(recall)
    metrics[f"{key_base}_f1_score"].append(f1)
    
def initialize_history(history, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    if model_name not in history:
        history[model_name] = {}

    loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"

    if loss_key not in history[model_name]:
        history[model_name][loss_key] = []
    if val_loss_key not in history[model_name]:
        history[model_name][val_loss_key] = []

def initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    all_true_labels[key] = []
    all_pred_labels[key] = []


###########################################################################
################# MAIN LOOP FOR PLOTTING GAUSSIAN BLUR GRID #################
###########################################################################

for lr, reg, ls, J_val, L_val, order_val, lo_val, hi_val, crop, down, vers in product(
        param_grid['lr'],
        param_grid['reg'],
        param_grid['label_smoothing'],
        param_grid['J'],
        param_grid['L'],
        param_grid['order'],
        param_grid['percentile_lo'],
        param_grid['percentile_hi'],
        param_grid['crop_size'],
        param_grid['downsample_size'],
        param_grid['versions']
    ):
    percentile_lo = lo_val
    percentile_hi = hi_val

    # Assign into your existing variables
    learning_rates       = [lr]
    regularization_params = [reg]
    label_smoothing      = ls
    J, L, order          = J_val, L_val, order_val
    crop_size            = crop
    downsample_size      = down
    versions              = vers
    
    print(f"\n▶ Experiment: g_classes={galaxy_classes}, lr={lr}, reg={reg}, ls={ls}, "
        f"J={J}, L={L}, crop={crop_size}, down={downsample_size}, ver={versions}, "
        f"lo={percentile_lo}, hi={percentile_hi}, classifier={classifier}, "
        f"global_norm={USE_GLOBAL_NORMALISATION}, norm_mode={GLOBAL_NORM_MODE} ◀\n")

    if any(cls in galaxy_classes for cls in [10, 11, 12, 13]):
        batch_size = 128
    else:
        batch_size = 16

    img_shape = downsample_size
    print("IMG SHAPE:", img_shape)

    FIXED_ANCHOR = False          # <— key: do not require fixed T*kpc headers
    _anchor_versions = []         # no anchor gating

    # —— MULTI-LABEL mode for RH/RR ——
    if galaxy_classes == [52, 53]:
        MULTILABEL = True
        LABEL_INDEX = {"RH": 0, "RR": 1}
        THRESHOLD = 0.5
        def relabel(y):
            y = y.long()
            out = torch.zeros((y.shape[0], 2), dtype=torch.float32, device=y.device)
            out[:, 0] = (y == 52).float()
            out[:, 1] = (y == 53).float()
            return out
    else:
        MULTILABEL = False
        def relabel(y):
            return (y - base_cls).long()

    if set(galaxy_classes) & {18} or set(galaxy_classes) & {19}:
        galaxy_classes = [20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Include all digits if 18 or 19 is in target_classes
    else:
        galaxy_classes = galaxy_classes
    num_classes = len(galaxy_classes)
    
    def _verkey(v):
        if isinstance(v, (list, tuple)):
            return "+".join(map(str, v))
        return str(v)
    ver_key = _verkey(versions)

    ###############################################
    ########## INITIALIZE DICTIONARIES ############
    ###############################################

    metric_colors = {
        "accuracy": 'blue',
        "precision": 'green',
        "recall": 'red',
        "f1_score": 'orange'
    }

    dataset_sizes   = defaultdict(list)
    metrics         = defaultdict(list)
    all_true_labels = defaultdict(list)
    all_pred_labels = defaultdict(list)
    all_pred_probs  = defaultdict(list)
    training_times = {}  # dict of dicts of lists
    history        = {}  # dict of dicts of lists



    ###############################################
    ########### LOOP OVER DATA FOLD ###############
    ###############################################            

    FIRSTTIME = True  # Set to True to print model summaries only once
    param_combinations = list(itertools.product(folds, learning_rates, regularization_params, lambda_values))
    for fold, lr, reg, lambda_generate in param_combinations:
        torch.cuda.empty_cache()
        runname = f"{galaxy_classes}_{gen_model_name}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size[0]}x{crop_size[1]}"

        log_path = f"./classifier/log_{runname}.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        _out = _loader(
            galaxy_classes=galaxy_classes,
            versions=versions or ['raw'],
            fold=max(folds),
            crop_size=crop_size,
            downsample_size=downsample_size,
            sample_size=max_num_galaxies, 
            REMOVEOUTLIERS=FILTERED,
            BALANCE=BALANCE,
            STRETCH=STRETCH,
            percentile_lo=percentile_lo,
            percentile_hi=percentile_hi,
            AUGMENT=AUGMENT,
            NORMALISE=NORMALISEIMGS,
            NORMALISETOPM=NORMALISEIMGSTOPM,
            USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
            GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
            PRINTFILENAMES=PRINTFILENAMES,
            train=True)

        if len(_out) == 4:
            train_images, train_labels, valid_images, valid_labels = _out
            train_fns = test_fns = None

            perm_train = torch.randperm(train_images.size(0))
            train_images, train_labels = train_images[perm_train], train_labels[perm_train]
            
            perm_valid = torch.randperm(valid_images.size(0))
            valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]

        elif len(_out) == 6:
            train_images, train_labels, valid_images, valid_labels, train_fns, valid_fns = _out                

            # shuffle AFTER late augmentation
            perm_train = torch.randperm(train_images.size(0))
            train_images, train_labels = train_images[perm_train], train_labels[perm_train]
            if PRINTFILENAMES and train_fns is not None:
                train_fns = permute_like(train_fns, perm_train)

            perm_valid = torch.randperm(valid_images.size(0))
            valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]
            if PRINTFILENAMES and valid_fns is not None:
                valid_fns = permute_like(valid_fns, perm_valid)
        else:
            raise ValueError(f"loader(train=True) returned {len(_out)} values, expected 4 or 6")

        def _desc(name, x):
            print(f"[{name}] shape={tuple(x.shape)}, min={float(x.min()):.3g}, max={float(x.max()):.3g}")

        _desc("TRAIN", train_images)
        _desc("VALID", valid_images)

        dataset_sizes[fold] = [max(2, int(len(train_images) * p)) for p in dataset_portions]

        train_labels, valid_labels = relabel(train_labels), relabel(valid_labels)                

        
        check_tensor(f"train_images for version: {ver_key}:", train_images)


        if dataset_sizes == {}:
            dataset_sizes[fold] = [int(len(train_images) * p) for p in dataset_portions]

                
        ##########################################################
        ############ NORMALISE AND PACKAGE THE INPUT #############
        ##########################################################

        if classifier in ['Rustige', 'CNNSqueezeNet', 'SCNN', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
        
            a = _as_5d(train_images).to(DEVICE)
            b = _as_5d(valid_images).to(DEVICE)

            img_splits = [a, b]
            img_lengths = [len(t) for t in img_splits]
            all_images = torch.cat(img_splits, dim=0)

            lbl_splits = [
                train_labels.to(DEVICE),
                valid_labels.to(DEVICE)
            ]
            all_labels = torch.cat(lbl_splits, dim=0)

            boundaries = [0] + list(torch.cumsum(torch.tensor(img_lengths), dim=0).numpy())
            chunked_imgs = [
                all_images[boundaries[i]:boundaries[i+1]]
                for i in range(len(img_lengths))
            ]
            chunked_lbls = [
                all_labels[boundaries[i]:boundaries[i+1]]
                for i in range(len(img_lengths))
            ]

            train_images, valid_images = chunked_imgs
            train_labels, valid_labels = chunked_lbls            
        
        if MULTILABEL:
            # labels are 2-hot; compute per-label pos_weight for BCE
            pos_counts = train_labels.sum(dim=0)                       # [2]
            neg_counts = train_labels.shape[0] - pos_counts            # [2]
            pos_counts = torch.clamp(pos_counts, min=1.0)
            pos_weight = (neg_counts / pos_counts).to(DEVICE)          # [2]
            print(f"[pos_weight] RH={pos_weight[0].item():.2f}, RR={pos_weight[1].item():.2f}")
            weights = None  # not used in BCE branch
        else:
            if USE_CLASS_WEIGHTS:
                unique, counts = np.unique(train_labels.cpu().numpy(), return_counts=True)
                total_count = sum(counts)
                class_weights = {i: total_count / count for i, count in zip(unique, counts)}
                weights = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)],
                                    dtype=torch.float, device=DEVICE)
            else:
                weights = None

        if fold in [0, 5] and SHOWIMGS:
            imgs = train_images.detach().cpu().numpy()
            # Make labels 1-D: 0 for first class, 1 for second; then shift to 52/53
            lbls = (as_index_labels(train_labels) + min(galaxy_classes)).detach().cpu().numpy()
            plot_images_by_class(
                imgs,
                labels=lbls,
                num_images=5,
                save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_example_train_data.pdf"
            )
        # Prepare input data
        mock_tensor = torch.zeros_like(train_images)
        valid_mock_tensor = torch.zeros_like(valid_images)
        if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2']:
            scattering = Scattering2D(J=J, L=L, shape=img_shape).to(DEVICE)
            # Define cache paths (you can adjust these names as needed)
            train_cache_path = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{TRAINONGENERATED}.npy"
            valid_cache_path = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{TRAINONGENERATED}.npy"
        
            # fold T into C on both real & scattering inputs
            train_images = fold_T_axis(train_images) # Merges the image version into the channel dimension
            valid_images = fold_T_axis(valid_images)
            mock_train = torch.zeros_like(train_images)
            mock_valid = torch.zeros_like(valid_images)

            train_cache = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
            valid_cache = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
            train_scat_coeffs = compute_scattering_coeffs(train_images, scattering, batch_size=128, device=DEVICE, cache_file=train_cache)
            valid_scat_coeffs = compute_scattering_coeffs(valid_images, scattering, batch_size=128, device=DEVICE, cache_file=valid_cache)

            if train_scat_coeffs.dim() == 5:
                # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
                print("Shape of train_scat_coeffs before flattening: ", train_scat_coeffs.shape)
                train_scat_coeffs = train_scat_coeffs.flatten(start_dim=1, end_dim=2)
                valid_scat_coeffs = valid_scat_coeffs.flatten(start_dim=1, end_dim=2)
                print("Shape of train_scat_coeffs after flattening: ", train_scat_coeffs.shape)

            all_scat = torch.cat([train_scat_coeffs, valid_scat_coeffs], dim=0)
            if NORMALISESCS or NORMALISESCSTOPM:
                if NORMALISESCSTOPM:
                    all_scat = normalise_images(all_scat, -1, 1)
                else:
                    all_scat = normalise_images(all_scat, 0, 1)
            train_scat_coeffs, valid_scat_coeffs = all_scat[:len(train_scat_coeffs)], all_scat[len(train_scat_coeffs):]

            scatdim = train_scat_coeffs.shape[1:]   # tuple(C, H, W)

            if classifier in ['ScatterNet', 'ScatterResNet']:
                train_dataset = TensorDataset(mock_train, train_scat_coeffs, train_labels)
                valid_dataset = TensorDataset(mock_valid, valid_scat_coeffs, valid_labels)
            else: # if classifier in ['ScatterSqueezeNet', 'ScatterSqueezeNet2']:
                train_dataset = TensorDataset(train_images, train_scat_coeffs, train_labels)
                valid_dataset = TensorDataset(valid_images, valid_scat_coeffs, valid_labels)
        else:
            if train_images.dim() == 5:
                train_images = fold_T_axis(train_images)   # [B,T,1,H,W] -> [B,T,H,W]
                valid_images = fold_T_axis(valid_images)
                # test_images was folded earlier
            for x,name in [(train_images,"train"), (valid_images,"valid")]:
                assert x.dim() == 4, f"{name}_images should be [B,C,H,W], got {tuple(x.shape)}"
            mock_train = torch.zeros_like(train_images)
            mock_valid = torch.zeros_like(valid_images)
            train_dataset = TensorDataset(train_images, mock_train, train_labels)
            valid_dataset = TensorDataset(valid_images, mock_valid, valid_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

        if SHOWIMGS and lambda_generate not in [0, 8]: 
            if classifier in ['TinyCNN', 'SCNN', 'CNNSqueezeNet', 'Rustige', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
                #save_images_tensorboard(generated_images[:36], save_path=f"./{gen_model_name}_{galaxy_classes}_generated.png", nrow=6)
                plot_histograms(pristine_train_images, valid_images, title1="Train images", title2="Valid images", imgs3=generated_images, imgs4=test_images, title3='Generated images', title4='Test images', save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{ver_key}_histogram.png")

        ###############################################
        ############# SANITY CHECKS ###################
        ###############################################
        
        if fold == folds[0] and SHOWIMGS and downsample_size[-1] == 128:               
            if len(galaxy_classes) == 2:
                # Plot histograms for the two classes
                if MULTILABEL:
                    train_images_cls1 = train_images[train_labels[:, 0] > 0.5]
                    train_images_cls2 = train_images[train_labels[:, 1] > 0.5]
                else:
                    train_images_cls1 = train_images[train_labels == galaxy_classes[0] - min(galaxy_classes)]
                    train_images_cls2 = train_images[train_labels == galaxy_classes[1] - min(galaxy_classes)]

                #Make sure the images are not tupples
                if isinstance(train_images_cls1, tuple):
                    train_images_cls1 = train_images_cls1[0]
                if isinstance(train_images_cls2, tuple):
                    train_images_cls2 = train_images_cls2[0]
                
                plot_histograms(
                    train_images_cls1.cpu(),
                    train_images_cls2.cpu(),
                    title1=f"Class {galaxy_classes[0]}",
                    title2=f"Class {galaxy_classes[1]}",
                    save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_histogram_{ver_key}.png"
                )
                
                # Histogram for summed pixel values. One image is one point in the histogram.
                sums_cls1 = train_images_cls1.view(train_images_cls1.size(0), -1).sum(dim=1).cpu()
                sums_cls2 = train_images_cls2.view(train_images_cls2.size(0), -1).sum(dim=1).cpu()
                plt.figure(figsize=(8,6))
                plt.hist(sums_cls1.numpy(), bins=30, alpha=0.5, label=f"Class {galaxy_classes[0]}")
                plt.hist(sums_cls2.numpy(), bins=30, alpha=0.5, label=f"Class {galaxy_classes[1]}")
                plt.xlabel("Sum of pixel values")
                plt.ylabel("Number of images")
                plt.title(f"Histogram of summed pixel values for classes {galaxy_classes[0]} and {galaxy_classes[1]}")
                plt.legend()
                plt.savefig(f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_sum_histogram_{ver_key}.png")
                plt.close()

                for cls in galaxy_classes:
                    cls_idx = cls - min(galaxy_classes)

                    sel_train = torch.where(as_index_labels(train_labels) == cls_idx)[0][:36]
                    titles_train = [train_fns[i] for i in sel_train.tolist()] if train_fns is not None else None
                    sel_test = torch.where(as_index_labels(test_labels) == cls_idx)[0][:36]
                    titles_test = [test_fns[i] for i in sel_test.tolist()] if test_fns is not None else None

                    imgs4plot = _ensure_4d(train_images)[sel_train].cpu()
                    plot_image_grid(
                        imgs4plot, num_images=36, titles=titles_train,
                        save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_train_grid_{ver_key}.png"
                    )

                    imgs4plot = _ensure_4d(test_images)[sel_test].cpu()
                    plot_image_grid(
                        imgs4plot, num_images=36, titles=titles_test,
                        save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_test_grid_{ver_key}.png"
                    )

                    if lambda_generate not in [0, 8]:
                        gen_imgs = _ensure_4d(generated_by_class[cls])[:36].cpu()
                        plot_image_grid(
                            gen_imgs, num_images=36,
                            save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_generated_grid_{ver_key}.png"
                        )

                        plot_histograms(
                            gen_imgs,
                            orig_imgs.cpu(),
                            title1="Generated Images",
                            title2="Train Images",
                            save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_histogram_{ver_key}.png"
                        )
                        plot_background_histogram(
                            orig_imgs,
                            gen_imgs,
                            img_shape=(1, 128, 128),
                            title="Background histograms",
                            save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_background_hist_{ver_key}.png")

        
        ###############################################
        ############# DEFINE MODEL ####################
        ###############################################
        
        if classifier == "Rustige":
            models = {"RustigeClassifier": {"model": RustigeClassifier(n_output_nodes=num_classes).to(DEVICE)}} 
        elif classifier == "SCNN":
            models = {"SCNN": {"model": SCNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "CNNSqueezeNet":
            models = {"CNNSqueezeNet": {"model": CNNSqueezeNet(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "DualCNNSqueezeNet":
            models = {"DualCNNSqueezeNet": {"model": DualCNNSqueezeNet(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "DICSN":
            H, W = valid_images.shape[-2], valid_images.shape[-1]
            models = {"DICSN": {"model": DualInputConvolutionalSqueezeNet(input_shape=(1, H, W), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "DISSN":
            models = {"DISSN": {"model": DISSN(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        elif classifier == "TinyCNN":
            models = {"TinyCNN": {"model": TinyCNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
        elif classifier == "DANN":
            models = {"DANN": {"model": DANNClassifier(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterNet":
            models = {"ScatterNet": {"model": MLPClassifier(input_dim=int(np.prod(scatdim)), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterResNet":
            models = {"ScatterResNet": {"model": ScatterResNet(scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterSqueezeNet":
            models = {"ScatterSqueezeNet": {"model": ScatterSqueezeNet(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterSqueezeNet2":
            models = {"ScatterSqueezeNet2": {"model": ScatterSqueezeNet2(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        elif classifier == 'Binary':
            models = {"BinaryClassifier": {"model": BinaryClassifier(input_shape=tuple(valid_images.shape[1:])).to(DEVICE)}}
        else:
            raise ValueError("Model not found. Please select one of 'scatterMLP', 'smallSTMLP', or 'normalCNN'.")

        classifier_name, model_details = next(iter(models.items()))


        ###############################################
        ############### TRAINING LOOP #################
        ###############################################
        
        if MULTILABEL:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if weights is not None:
                print(f"Using class weights: {weights}")
                criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
            else:
                print("No class weighting")
                criterion = nn.BCEWithLogitsLoss() if len(galaxy_classes)==2 else nn.CrossEntropyLoss()

            

        optimizer = AdamW(models[classifier_name]["model"].parameters(), lr=lr, weight_decay=reg)
        if SCHEDULER:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10*lr, 
                                    steps_per_epoch=len(train_loader), epochs=num_epochs)

        for classifier_name, model_details in models.items():
            model = model_details["model"].to(DEVICE)

            for subset_size in dataset_sizes[fold]:
                if subset_size <= 0:
                    print(f"Skipping invalid subset size: {subset_size}")
                    continue
                if subset_size not in training_times:
                    training_times[subset_size] = {}
                if fold not in training_times[subset_size]:
                    training_times[subset_size][fold] = []

                for experiment in range(num_experiments):
                    initialize_history(history, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                    initialize_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate, crop_size, downsample_size, ver_key)
                    initialize_labels(all_true_labels, all_pred_labels, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)

                    start_time = time.time()
                    model.apply(reset_weights)

                    subset_indices = torch.randperm(len(train_dataset))[:subset_size].tolist() # Randomly select indices to include generated samples
                    subset_train_dataset = Subset(train_dataset, subset_indices)
                    eff_bs = max(2, min(batch_size, len(subset_train_dataset)))
                    subset_train_loader = DataLoader(subset_train_dataset, batch_size=eff_bs, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

                    early_stopping = EarlyStopping(patience=patience, verbose=False) if ES else None

                    for epoch in tqdm(range(num_epochs), desc=f'Training {classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_{classifier}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size}'):
                        model.train()
                        total_loss = 0
                        total_images = 0

                        for images, scat, _rest in subset_train_loader:
                            labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE) # Send to device
                            optimizer.zero_grad()
                            if classifier == "DANN":
                                # 1) forward pass: two heads
                                class_logits, domain_logits = model(images, alpha=1.0)

                                # 2) classification loss
                                class_loss = criterion(class_logits, labels)

                                # 3) domain loss (0=real, 1=fake)
                                B = labels.size(0)
                                domain_labels = torch.zeros(B, dtype=torch.long, device=DEVICE)
                                # if you interleave generated samples in the same loader you need a flag per-sample;
                                # for now this will assume your loader is real‐only, so all zeros
                                domain_loss = nn.CrossEntropyLoss()(domain_logits, domain_labels)

                                # 4) total loss
                                loss = class_loss + 0.5 * domain_loss

                                loss.backward()
                                optimizer.step()
                                if SCHEDULER:
                                    scheduler.step()

                                total_loss += float(loss.item() * images.size(0))
                                total_images += float(images.size(0))
                            else:
                                if classifier in ["ScatterNet", "ScatterResNet"]:
                                    logits = model(scat)
                                elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
                                    logits = model(images, scat)
                                else:
                                    logits = model(images)

                                logits = collapse_logits(logits, num_classes, MULTILABEL)
                                labels = labels.float() if MULTILABEL else labels.long()
                                loss = criterion(logits, labels)

                                loss.backward()
                                optimizer.step()
                                total_loss += float(loss.item() * images.size(0))
                                total_images += float(images.size(0))

                        average_loss = total_loss / total_images
                        loss_key = f"{gen_model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        history[gen_model_name][loss_key].append(average_loss)

                        model.eval()
                        val_total_loss = 0
                        val_total_images = 0

                        with torch.inference_mode(): # Validate on validation data
                            for i, (images, scat, _rest) in enumerate(valid_loader):
                                if images is None or len(images) == 0:
                                    print(f"Empty batch at index {i}. Skipping...")
                                    continue
                                labels = _rest
                                images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                                if classifier == "DANN":
                                    logits, _ = model(images, alpha=1.0)
                                else:
                                    if classifier in ["ScatterNet", "ScatterResNet"]:
                                        logits = model(scat)
                                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
                                        logits = model(images, scat)
                                    else:
                                        logits = model(images)

                                logits = collapse_logits(logits, num_classes, MULTILABEL)
                                labels = labels.float() if MULTILABEL else labels.long()
                                if not MULTILABEL:
                                    assert labels.dtype == torch.long, f"labels dtype {labels.dtype} must be long"
                                loss = criterion(logits, labels)
                                mn, mx = int(labels.min()), int(labels.max())
                                assert 0 <= mn and mx < num_classes, f"label range [{mn},{mx}] not in [0,{num_classes-1}]"

                                val_total_loss += float(loss.item() * images.size(0))
                                val_total_images += float(images.size(0))

                        val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                        val_loss_key = f"{gen_model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        history[gen_model_name][val_loss_key].append(val_average_loss)
                        
                        if ES:
                            early_stopping(val_average_loss, model, f'./classifier/trained_models/{gen_model_name}_best_model.pth')
                            if early_stopping.early_stop:
                                break

                    model.eval()
                    with torch.inference_mode(): # Evaluate on test data
                        key = f"{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        all_pred_probs[key] = []
                        all_pred_labels[key] = []
                        all_true_labels[key] = []
                        mis_images = []
                        mis_trues  = []
                        mis_preds  = []
                        
                        for images, scat, _rest in test_loader:
                            labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            if classifier == "DANN":
                                logits, _ = model(images, alpha=1.0)
                            else:
                                if classifier in ["ScatterNet", "ScatterResNet"]:
                                    logits = model(scat)
                                elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
                                    logits = model(images, scat)
                                else:
                                    logits = model(images)
                                    
                            logits = collapse_logits(logits, num_classes, MULTILABEL)
                            if MULTILABEL:
                                probs = torch.sigmoid(logits).cpu().numpy()           # [B,2]
                                preds = (probs >= THRESHOLD).astype(int)              # [B,2]
                                trues = labels.cpu().numpy().astype(int)              # [B,2]
                                all_pred_probs[key].extend(probs)
                                all_pred_labels[key].extend(preds)
                                all_true_labels[key].extend(trues)
                            else:
                                pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
                                true_labels = labels.cpu().numpy()
                                pred_labels = np.argmax(pred_probs, axis=1)
                                all_pred_probs[key].extend(pred_probs)
                                all_pred_labels[key].extend(pred_labels)
                                all_true_labels[key].extend(true_labels)
                                
                            if SHOWIMGS and experiment == num_experiments - 1:
                                if MULTILABEL:
                                    batch_pred = preds          # shape [B, 2]
                                    batch_true = trues          # shape [B, 2]
                                    mask = (batch_pred != batch_true).any(axis=1)
                                else:
                                    batch_pred = pred_labels    # shape [B]
                                    batch_true = true_labels    # shape [B]
                                    mask = batch_pred != batch_true

                                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=images.device)
                                mis_images.append(images.detach().cpu()[mask_t.cpu()])
                                mis_trues.append(batch_true[mask])
                                mis_preds.append(batch_pred[mask])
                                        
                        # --- metrics ---
                        y_true = np.array(all_true_labels[key])
                        y_pred = np.array(all_pred_labels[key])
                        accuracy, precision, recall, f1 = compute_classification_metrics(y_true, y_pred, multilabel=MULTILABEL, num_classes=num_classes)
                        update_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate, crop_size, downsample_size, ver_key)
                        print(f"Fold {fold}, Experiment {experiment}, Subset Size {subset_size}, Classifier {classifier_name}, "
                            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

                        if SHOWIMGS and mis_images and experiment == num_experiments - 1:
                            mis_images = torch.cat(mis_images, dim=0)[:36]
                            mis_trues  = np.concatenate(mis_trues)[:36]
                            mis_preds  = np.concatenate(mis_preds)[:36]
                            
                            fig, axes = plt.subplots(6, 6, figsize=(12, 12))
                            axes = axes.flatten()

                            for i, ax in enumerate(axes[:len(mis_images)]):
                                img_tensor = mis_images[i]                           # shape is either (1,128,128) or (2,128,128)
                                # pick the first channel if there are two, else drop the singleton channel
                                img = img_tensor[0] if img_tensor.shape[0] > 1 else img_tensor.squeeze(0)
                                ax.imshow(img.numpy(), cmap='viridis')
                                ax.set_title(f"T={mis_trues[i]}, P={mis_preds[i]}")
                                ax.axis('off')

                            for ax in axes[len(mis_images):]:
                                ax.axis('off')

                            out_path = f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_misclassified_{ver_key}.png"
                            fig.savefig(out_path, dpi=150, bbox_inches='tight')
                            plt.close(fig)

                base = (
                    f"{gen_model_name}"
                    f"_ss{subset_size}"
                    f"_f{fold}"
                    f"_lr{lr}"
                    f"_reg{reg}"
                    f"_lam{lambda_generate}"
                    f"_cs{crop_size[0]}x{crop_size[1]}"
                    f"_ds{downsample_size[0]}x{downsample_size[1]}"
                    f"_ver{ver_key}"
                )
                mean_acc = float(np.mean(metrics[f"{base}_accuracy"])) if metrics[f"{base}_accuracy"] else float('nan')
                mean_prec = float(np.mean(metrics[f"{base}_precision"])) if metrics[f"{base}_precision"] else float('nan')
                mean_rec = float(np.mean(metrics[f"{base}_recall"])) if metrics[f"{base}_recall"] else float('nan')
                mean_f1 = float(np.mean(metrics[f"{base}_f1_score"])) if metrics[f"{base}_f1_score"] else float('nan')
                print(f"AVERAGE over {num_experiments} experiments — Accuracy: {mean_acc:.4f}, Precision: {mean_prec:.4f}, Recall: {mean_rec:.4f}, F1 Score: {mean_f1:.4f}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                #training_times[subset_size][fold].append(elapsed_time)
                training_times.setdefault(fold, {}).setdefault(subset_size, []).append(elapsed_time)

                if fold == folds[-1] and experiment == num_experiments - 1:
                    with open(log_path, 'w') as file:
                        file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")
            
            generated_features = []
            with torch.inference_mode():
                for images, scat, _rest in validation_loader:
                    labels = _rest
                    images = images.to(DEVICE, non_blocking=True)
                    scat   = scat.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

                    if classifier == "DANN":
                        class_logits, _ = model(images, alpha=1.0)
                        outputs = class_logits.cpu().detach().numpy()
                    elif classifier in ["ScatterNet", "ScatterResNet"]:
                        outputs = model(scat).cpu().detach().numpy()
                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
                        outputs = model(images, scat).cpu().detach().numpy()
                    else:
                        outputs = model(images).cpu().detach().numpy()

                    generated_features.append(outputs)

            generated_features = np.concatenate(generated_features, axis=0)
            cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(generated_features, n_clusters=num_classes)
            with open(log_path, 'w') as file:
                file.write(f"Results for fold {fold}, Classifier {classifier_name}, lr={lr}, reg={reg}, lambda_generate={lambda_generate}, percentile_lo={percentile_lo}, percentile_hi={percentile_hi}, crop_size={crop_size}, downsample_size={downsample_size}, STRETCH={STRETCH}, FILTERED={FILTERED}, TRAINONGENERATED={TRAINONGENERATED} \n")
                file.write(f"Cluster Error: {cluster_error} \n")
                file.write(f"Cluster Distance: {cluster_distance} \n")
                file.write(f"Cluster Standard Deviation: {cluster_std_dev} \n")

            model_save_path = f'./classifier/trained_models/{gen_model_name}_model.pth'
            torch.save(model.state_dict(), model_save_path)

