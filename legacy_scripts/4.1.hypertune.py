import os, re, time, random, hashlib, itertools, torch
from utils.data_loader2 import load_galaxies # This only contains two rotation angles rather than 12
from utils.classifiers import (
    RustigeClassifier, TinyCNN, ScatterNet, ScatterResNet, 
    DualScatterSqueezeNet, DualCNNSqueezeNet)
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, Subset
from kymatio.torch import Scattering2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42  # Set a seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

print("Running ht1 with seed", SEED)

###############################################
################ CONFIGURATION ################
###############################################

galaxy_classes    = [50, 51]
max_num_galaxies  = 1000000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions  = [1]
J, L, order       = 2, 12, 2
num_epochs_cuda = 200
num_epochs_cpu = 100
folds = [0, 1, 2, 3, 4]
num_experiments = 1  # Number of runs per fold and hyperparameter combo

classifier = ["CNN",         # 0.Very Simple CNN
              "Rustige",     # 1.Simple CNN from Rustige et al. 2023, https://github.com/floriangriese/wGAN-supported-augmentation/blob/main/src/Classifiers/SimpleClassifiers/Classifiers.py
              "DualCSN",     # 2.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              "ScatterNet",  # 3.Scattering coefficients as input to MLP
              "DualSSN",     # 4.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              "ScatterResNet"][4] # 5. Scatter + ResNet18

# Define every value you want to try
param_grid = {
    'lr':            [5e-6], #[5e-6, 5e-5, 5e-4],
    'reg':           [1e-1], #[1e-1, 1e-2, 1e-3],
    'label_smoothing':[0.1], #[0.0, 0.1],
    'J':             [2],          # Only used for scattering classifiers
    'L':             [12],         # Only used for scattering classifiers
    'order':         [2],          # Only used for scattering classifiers
    'percentile_lo': [30], #[1, 30, 60],  # Only used for data normalisation
    'percentile_hi': [99], #[80, 90, 99], # Only used for data normalisation
    'crop_size':     [(512,512)],  # Not used for preformatted data
    'downsample_size':[(128,128)], # Not used for preformatted data
    'versions':       ['RAW']  # 'raw', 'T50kpc', ad hoc tapering: e.g. 'rt50'  strings in list → product() iterates them individually
} #'versions': [('raw', 'rt50')]  # tuple signals “stack these”

#PREFER_PROCESSED = True if param_grid['versions'][0] != 'RAW' else False  # Prefer processed images if available (constant n_beams)
PREFER_PROCESSED = True
STRETCH = True  # Arcsinh stretch
USE_GLOBAL_NORMALISATION = False           # single on/off switch . False - image-by-image normalisation 
GLOBAL_NORM_MODE = "percentile"           # "percentile" or "flux". Becomes "none" if USE_GLOBAL_NORMALISATION is 
NORMALISEIMGS = True  # Normalise images to [0, 1]
NORMALISEIMGSTOPM = False  # Normalise images to [-1, 1] 
NORMALISESCS = False  # Normalise scattering coefficients to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients to [-1, 1]
FILTERED = True  # Remove in training and validation for the classifier
FILTERGEN = False  # Remove generated images that are too similar to other generated images
AUGMENT = True  # Use classical data augmentation (flips, rotations)
MIXUP = True  # Use MixUp augmentation as a means to reduce overfitting
PRINTFILENAMES = True
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights
ES, patience = True, 50  # Use early stopping
SCHEDULER = True  # Use a learning rate scheduler
SHOWIMGS = False  # Show some generated images for each class (Tool for control)
DEBUG = False  # Perform data overlap checks
USE_CACHE = True  # Use cached scattering coefficients if available


########################################################################
##################### AUTOMATIC CONFIGURATION ##########################
########################################################################

os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)
os.makedirs('./hypertuning_outputs', exist_ok=True)


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

# —— MULTI-LABEL SWITCH for RH/RR ——
if galaxy_classes == [52, 53]:
    from utils.data_loader import load_halos_and_relics
    MULTILABEL = True            # predict RH and RR independently
    LABEL_INDEX = {"RH": 0, "RR": 1}
    THRESHOLD = 0.5
    USE_CLASS_WEIGHTS = False  # Disable class weights for multi-label
else:
    MULTILABEL = False

_loader = load_halos_and_relics if galaxy_classes == [52, 53] else load_galaxies
BALANCE = True if galaxy_classes == [52, 53] else False  # Balance the dataset by undersampling the majority class


########################################################################
##################### HELPER FUNCTIONS #################################
########################################################################

#-------------------------------------------------------------
# Evaluation
#-------------------------------------------------------------
    

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
    
def img_hash(img: torch.Tensor) -> str:
    arr = img.cpu().contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()

def _desc(name, x):
    print(f"[{name}] shape={tuple(x.shape)}, min={float(x.min()):.3g}, max={float(x.max()):.3g}")
                
# -------------------------------------------------------------
# Data processing helpers
# -------------------------------------------------------------

def mixup_data(x1, x2, y, alpha=0.4):
    """
    Apply MixUp augmentation: create convex combinations of pairs of examples.
    
    Args:
        x1: First input (images)
        x2: Second input (scattering coefficients)
        y: Labels
        alpha: MixUp hyperparameter
    
    Returns:
        Mixed inputs and targets
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
    return (y - min(galaxy_classes)).long()

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

def initialise_metrics(metrics, key_base):
    metrics.setdefault(f"{key_base}_accuracy", [])
    metrics.setdefault(f"{key_base}_precision", [])
    metrics.setdefault(f"{key_base}_recall", [])
    metrics.setdefault(f"{key_base}_f1_score", [])

def update_metrics(metrics, key_base, accuracy, precision, recall, f1):
    metrics[f"{key_base}_accuracy"].append(accuracy)
    metrics[f"{key_base}_precision"].append(precision)
    metrics[f"{key_base}_recall"].append(recall)
    metrics[f"{key_base}_f1_score"].append(f1)
    
def initialise_history(history, base, experiment):
    train_loss_key = f"{base}_{experiment}_train_loss"
    val_loss_key = f"{base}_{experiment}_val_loss"
    for key in [train_loss_key, val_loss_key]:
        if key not in history:
            history[key] = []

def initialise_labels(key, all_true_labels, all_pred_labels):
    all_true_labels[key] = []
    all_pred_labels[key] = []


###########################################################################
################# MAIN LOOP FOR PLOTTING GAUSSIAN BLUR GRID #################
###########################################################################

for lr, reg, ls, J_val, L_val, order_val, lo_val, hi_val, crop, down, vers in itertools.product(
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
        f"global_norm={USE_GLOBAL_NORMALISATION}, norm_mode={GLOBAL_NORM_MODE}, "
        f"PREFER_PROCESSED={PREFER_PROCESSED} ◀\n")
    
    cs = f"{crop_size[-2]}x{crop_size[-1]}"
    ds = f"{downsample_size[-2]}x{downsample_size[-1]}"

    if any(cls in galaxy_classes for cls in [10, 11, 12, 13]):
        batch_size = 128
    else:
        batch_size = 16

    img_shape = downsample_size
    scattering = Scattering2D(J=J, L=L, shape=img_shape[-2:], max_order=order)      
    print("IMG SHAPE:", img_shape)

    # —— MULTI-LABEL mode for RH/RR ——
    if galaxy_classes == [52, 53]:
        MULTILABEL = True
        LABEL_INDEX = {"RH": 0, "RR": 1}
        THRESHOLD = 0.5

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
    training_times  = defaultdict(dict) 
    history         = defaultdict(list)

    ###############################################
    ########### LOOP OVER DATA FOLD ###############
    ###############################################            
    for fold in folds:
        torch.cuda.empty_cache()
        runname = f"{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size[0]}x{crop_size[1]}"

        log_path = f"./classifier/log_{runname}.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # real train + valid
        _out = _loader(
            galaxy_classes=galaxy_classes,
            versions=versions or ['raw'],
            fold=fold,
            crop_size=crop_size,
            downsample_size=downsample_size,
            sample_size=max_num_galaxies, 
            REMOVEOUTLIERS=FILTERED,
            BALANCE=BALANCE,           # Reduce the larger classes to the size of the smallest class
            STRETCH=STRETCH,
            percentile_lo=percentile_lo,  # Percentile stretch lower bound
            percentile_hi=percentile_hi,  # Percentile stretch upper bound
            AUGMENT=AUGMENT,
            NORMALISE=NORMALISEIMGS,
            NORMALISETOPM=NORMALISEIMGSTOPM,
            USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
            GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
            PRINTFILENAMES=PRINTFILENAMES,
            PREFER_PROCESSED=PREFER_PROCESSED,
            USE_CACHE=USE_CACHE,
            DEBUG=DEBUG,
            train=True) # Obtain the train and validation set. Not test set

        if len(_out) == 4:
            train_images, train_labels, valid_images, valid_labels = _out
            train_fns = valid_fns = None

        elif len(_out) == 6:
            train_images, train_labels, valid_images, valid_labels, train_fns, valid_fns = _out                

        else:
            raise ValueError(f"loader returned {len(_out)} values, expected 4 or 6")
        
        perm_train, perm_valid = torch.randperm(train_images.size(0)), torch.randperm(valid_images.size(0))
        train_images, valid_images = train_images[perm_train], valid_images[perm_valid]
        train_labels, valid_labels = train_labels[perm_train], valid_labels[perm_valid]
        
        if PRINTFILENAMES and train_fns and valid_fns is not None:
            train_fns = permute_like(train_fns, perm_train)
            valid_fns = permute_like(valid_fns, perm_valid)

        train_labels, valid_labels = relabel(train_labels), relabel(valid_labels)       

        if dataset_sizes == {}:
            dataset_sizes[fold] = [max(2, int(len(train_images) * p)) for p in dataset_portions]

                
        ##########################################################
        ############ NORMALISE AND PACKAGE THE INPUT #############
        ##########################################################
        if classifier in ['Rustige', 'DualCSN', 'DualSSN']:
            train_images = _as_5d(train_images).to(DEVICE)
            valid_images = _as_5d(valid_images).to(DEVICE)
        
        if classifier in ['ScatterNet', 'ScatterResNet', 'DualSSN']:
    
            # fold T into C on both real & scattering inputs
            train_images = fold_T_axis(train_images) # Merges the image version into the channel dimension
            valid_images = fold_T_axis(valid_images)
            mock_train = torch.zeros_like(train_images)
            mock_valid = torch.zeros_like(valid_images)

            # Define cache paths
            train_cache = f"./.cache/hypertune_train_scat_{galaxy_classes}_{fold}_{dataset_portions[0]}_{FILTERED}.pt"
            valid_cache = f"./.cache/hypertune_valid_scat_{galaxy_classes}_{fold}_{dataset_portions[0]}_{FILTERED}.pt"

            # Load or compute train scattering coefficients
            if os.path.exists(train_cache) and USE_CACHE:
                print(f"✓ Loading train scattering coefficients from cache: {train_cache}")
                train_scat_coeffs = torch.load(train_cache)
            else:
                print(f"Computing train scattering coefficients...")
                # Take time to compute scattering coefficients
                start_time = time.time()
                train_scat_coeffs = compute_scattering_coeffs(train_images, scattering, batch_size=128, device="cpu")
                end_time = time.time()
                print(f"Time taken to compute train scattering coefficients: {end_time - start_time:.2f} seconds")
                os.makedirs(os.path.dirname(train_cache), exist_ok=True)
                torch.save(train_scat_coeffs, train_cache)
                print(f"✓ Saved train scattering coefficients to cache: {train_cache}")

            # Load or compute validation scattering coefficients
            if os.path.exists(valid_cache) and USE_CACHE:
                print(f"✓ Loading valid scattering coefficients from cache: {valid_cache}")
                valid_scat_coeffs = torch.load(valid_cache)
            else:
                print(f"Computing valid scattering coefficients...")
                valid_scat_coeffs = compute_scattering_coeffs(valid_images, scattering, batch_size=128, device="cpu")
                os.makedirs(os.path.dirname(valid_cache), exist_ok=True)
                torch.save(valid_scat_coeffs, valid_cache)
                print(f"✓ Saved valid scattering coefficients to cache: {valid_cache}")

            if train_scat_coeffs.dim() == 5:
                # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
                train_scat_coeffs = train_scat_coeffs.flatten(start_dim=1, end_dim=2)
                valid_scat_coeffs = valid_scat_coeffs.flatten(start_dim=1, end_dim=2)

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
            else: # if classifier == 'DualSSN':
                train_dataset = TensorDataset(train_images, train_scat_coeffs, train_labels)
                valid_dataset = TensorDataset(valid_images, valid_scat_coeffs, valid_labels)
        else:
            if train_images.dim() == 5:
                train_images = fold_T_axis(train_images)   # [B,T,1,H,W] -> [B,T,H,W]
                valid_images = fold_T_axis(valid_images)
            for x,name in [(train_images,"train"), (valid_images,"valid")]:
                assert x.dim() == 4, f"{name}_images should be [B,C,H,W], got {tuple(x.shape)}"
            mock_train = torch.zeros_like(train_images)
            mock_valid = torch.zeros_like(valid_images)
            train_dataset = TensorDataset(train_images, mock_train, train_labels)
            valid_dataset = TensorDataset(valid_images, mock_valid, valid_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

            
        ###############################################
        ############# SANITY CHECKS ###################
        ###############################################
        
        if fold == folds[0] and SHOWIMGS and downsample_size[-1] == 128:     
            # Plot some example training images with their labels
            imgs = train_images.detach().cpu().numpy()
            lbls = (as_index_labels(train_labels) + min(galaxy_classes)).detach().cpu().numpy()
            plot_images_by_class(
                imgs,
                labels=lbls,
                num_images=5,
                save_path=f"./hypertuning_outputs/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_example_train_data.pdf"
            )     
            
            if len(galaxy_classes) == 2:
                # Plot histograms for the two classes
                if MULTILABEL:
                    train_images_cls1 = train_images[train_labels[:, 0] > 0.5]
                    train_images_cls2 = train_images[train_labels[:, 1] > 0.5]
                else:
                    train_images_cls1 = train_images[train_labels == galaxy_classes[0] - min(galaxy_classes)]
                    train_images_cls2 = train_images[train_labels == galaxy_classes[1] - min(galaxy_classes)]

                #Make sure the images are not tupples
                if isinstance(train_images_cls1, tuple): train_images_cls1 = train_images_cls1[0]
                if isinstance(train_images_cls2, tuple): train_images_cls2 = train_images_cls2[0]

                plot_histograms(
                    train_images_cls1.cpu(),
                    train_images_cls2.cpu(),
                    title1=f"Class {galaxy_classes[0]}",
                    title2=f"Class {galaxy_classes[1]}",
                    save_path=f"./hypertuning_outputs/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_histogram_{ver_key}.png"
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
                plt.savefig(f"./hypertuning_outputs/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_sum_histogram_{ver_key}.png")
                plt.close()

                for cls in galaxy_classes:
                    cls_idx = cls - min(galaxy_classes)

                    sel_train = torch.where(as_index_labels(train_labels) == cls_idx)[0][:36]
                    titles_train = [train_fns[i] for i in sel_train.tolist()] if train_fns is not None else None
                    sel_valid = torch.where(as_index_labels(valid_labels) == cls_idx)[0][:36]
                    titles_valid = [valid_fns[i] for i in sel_valid.tolist()] if valid_fns is not None else None

                    imgs4plot = _ensure_4d(train_images)[sel_train].cpu()
                    plot_image_grid(
                        imgs4plot, num_images=36, titles=titles_train,
                        save_path=f"./hypertuning_outputs/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_train_grid_{ver_key}.png"
                    )

                    imgs4plot = _ensure_4d(valid_images)[sel_valid].cpu()
                    plot_image_grid(
                        imgs4plot, num_images=36, titles=titles_valid,
                        save_path=f"./hypertuning_outputs/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_valid_grid_{ver_key}.png"
                    )

        
        ###############################################
        ############# DEFINE MODEL ####################
        ###############################################
        
        if classifier == "Rustige":
            models = {"RustigeClassifier": {"model": RustigeClassifier(n_output_nodes=num_classes).to(DEVICE)}} 
        elif classifier == "DualCSN":
            models = {"DualCSN": {"model": DualCNNSqueezeNet(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "CNN":
            models = {"CNN": {"model": TinyCNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
        elif classifier == "ScatterNet":
            models = {"ScatterNet": {"model": ScatterNet(input_dim=int(np.prod(scatdim)), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterResNet":
            models = {"ScatterResNet": {"model": ScatterResNet(scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        elif classifier == "DualSSN":
            models = {"DualSSN": {"model": DualScatterSqueezeNet(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        classifier_name, model_details = next(iter(models.items()))


        ###############################################
        ############### TRAINING LOOP #################
        ###############################################
        
        if MULTILABEL:
            # labels are 2-hot; compute per-label pos_weight for BCE
            pos_counts = train_labels.sum(dim=0)                       # [2]
            neg_counts = train_labels.shape[0] - pos_counts            # [2]
            pos_counts = torch.clamp(pos_counts, min=1.0)
            pos_weight = (neg_counts / pos_counts).to(DEVICE)          # [2]
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if USE_CLASS_WEIGHTS:
                unique, counts = np.unique(train_labels.cpu().numpy(), return_counts=True)
                total_count = sum(counts)
                class_weights = {i: total_count / count for i, count in zip(unique, counts)}
                weights = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)],
                                    dtype=torch.float, device=DEVICE)
                print("Using class weighting:", weights)
                criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
            else:
                print("No class weighting")
                criterion = nn.BCEWithLogitsLoss() if len(galaxy_classes)==2 else nn.CrossEntropyLoss()

        optimizer = AdamW(models[classifier_name]["model"].parameters(), lr=lr, weight_decay=reg)

        for classifier_name, model_details in models.items():
            model = model_details["model"].to(DEVICE)

            for subset_size in dataset_sizes[fold]:
                if subset_size <= 0:
                    print(f"Skipping invalid subset size: {subset_size}")
                    continue
                if fold not in training_times:
                    training_times[fold] = {}
                if subset_size not in training_times[fold]:
                    training_times[fold][subset_size] = []

                for experiment in range(num_experiments):
                    base = f"cl{classifier_name}_ss{subset_size}_f{fold}_lr{lr}_reg{reg}_ls{label_smoothing}_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{ver_key}"
                    initialise_history(history, base, experiment)
                    initialise_metrics(metrics, base)
                    initialise_labels(base, all_true_labels, all_pred_labels)

                    start_time = time.time()
                    model.apply(reset_weights)

                    subset_indices = torch.randperm(len(train_dataset))[:subset_size].tolist() # Randomly select indices to include generated samples
                    subset_train_dataset = Subset(train_dataset, subset_indices)
                    eff_bs = max(2, min(batch_size, len(subset_train_dataset))) # effective batch size cannot be larger than dataset
                    subset_train_loader = DataLoader(subset_train_dataset, batch_size=eff_bs, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
                    if SCHEDULER:
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            max_lr=lr * 3,
                            steps_per_epoch=len(subset_train_loader),
                            epochs=num_epochs,
                            pct_start=0.3,  # Spend 30% of training warming up
                            anneal_strategy='cos'
                        )

                    early_stopping = EarlyStopping(patience=patience, verbose=False) if ES else None

                    for epoch in tqdm(range(num_epochs), desc=f'Training {classifier}_{galaxy_classes}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{classifier}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size}'):
                        model.train()
                        train_total_loss = 0
                        train_total_images = 0

                        for images, scat, _rest in subset_train_loader:
                            labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            optimizer.zero_grad()
                            # ADD MixUp here:
                            if np.random.rand() > 0.5 and MIXUP:  # Apply MixUp 50% of the time
                                images, scat, labels_a, labels_b, lam = mixup_data(images, scat, labels, alpha=0.4)
                                
                                if classifier in ["ScatterNet", "ScatterResNet"]:
                                    logits = model(scat)
                                elif classifier == 'DualSSN':
                                    logits = model(images, scat)
                                else:
                                    logits = model(images)

                                logits = collapse_logits(logits, num_classes, MULTILABEL)
                                labels_a = labels_a.float() if MULTILABEL else labels_a.long()
                                labels_b = labels_b.float() if MULTILABEL else labels_b.long()
                                
                                # CHANGE: Use MixUp criterion
                                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                            else:
                                # Normal forward pass without MixUp
                                if classifier in ["ScatterNet", "ScatterResNet"]:
                                    logits = model(scat)
                                elif classifier == 'DualSSN':
                                    logits = model(images, scat)
                                else:
                                    logits = model(images)

                                logits = collapse_logits(logits, num_classes, MULTILABEL)
                                labels = labels.float() if MULTILABEL else labels.long()
                                loss = criterion(logits, labels)
                                
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to prevent gradients that are too large
                            optimizer.step()
                            if SCHEDULER:
                                scheduler.step()
                            train_total_loss += float(loss.item() * images.size(0))
                            train_total_images += float(images.size(0))

                        # Calculate epoch training metrics
                        train_average_loss = train_total_loss / train_total_images
                        train_loss_key = f"{base}_{experiment}_train_loss"
                        history[train_loss_key].append(train_average_loss)

                        model.eval()
                        val_total_loss = 0
                        val_total_images = 0
                        with torch.inference_mode():
                            for i, (images, scat, _rest) in enumerate(valid_loader):
                                if images is None or len(images) == 0:
                                    print(f"Empty batch at index {i}. Skipping...")
                                    continue
                                labels = _rest
                                images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                                if classifier in ["ScatterNet", "ScatterResNet"]:
                                    logits = model(scat)
                                elif classifier == 'DualSSN':
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
                        val_loss_key = f"{base}_{experiment}_val_loss"
                        history[val_loss_key].append(val_average_loss)
                        if ES:
                            early_stopping(val_average_loss, model, f'./classifier/trained_models/{base}_best_model.pth')
                            if early_stopping.early_stop:
                                break

                    if ES: # Load the best model saved by early stopping
                        checkpoint_path = f'./classifier/trained_models/{base}_best_model.pth'
                        if os.path.exists(checkpoint_path):
                            try:
                                model.load_state_dict(torch.load(checkpoint_path))
                            except RuntimeError as e:
                                print(f"Warning: Could not load checkpoint due to architecture mismatch: {e}")
                                print(f"Continuing with current model state (last trained epoch)")
                        else:
                            print(f"No checkpoint found at {checkpoint_path}, using current model state")
                        

                    model.eval()                  
                    with torch.inference_mode(): 
                        all_pred_probs[base] = []
                        all_pred_labels[base] = []
                        all_true_labels[base] = []
                        mis_images = []
                        mis_trues  = []
                        mis_preds  = []
                        all_logits = []
                        
                        for images, scat, _rest in valid_loader: # Evaluate on validation data
                            labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            if classifier in ["ScatterNet", "ScatterResNet"]:
                                logits = model(scat)
                            elif classifier == 'DualSSN':
                                logits = model(images, scat)
                            else:
                                logits = model(images)
                                
                            logits = collapse_logits(logits, num_classes, MULTILABEL)
                            if MULTILABEL:
                                probs = torch.sigmoid(logits).cpu().numpy()           # [B,2]
                                preds = (probs >= THRESHOLD).astype(int)              # [B,2]
                                trues = labels.cpu().numpy().astype(int)              # [B,2]
                                all_pred_probs[base].extend(probs)
                                all_pred_labels[base].extend(preds)
                                all_true_labels[base].extend(trues)
                            else:
                                pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
                                true_labels = labels.cpu().numpy()
                                pred_labels = np.argmax(pred_probs, axis=1)
                                all_pred_probs[base].extend(pred_probs)
                                all_pred_labels[base].extend(pred_labels)
                                all_true_labels[base].extend(true_labels)
                                                    
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
                        y_true = np.array(all_true_labels[base])
                        y_pred = np.array(all_pred_labels[base])
                        accuracy, precision, recall, f1 = compute_classification_metrics(y_true, y_pred, multilabel=MULTILABEL, num_classes=num_classes)
                        update_metrics(metrics, base, accuracy, precision, recall, f1)
                        print(f"Fold {fold}, Experiment {experiment}, Subset Size {subset_size}, Classifier {classifier_name}, "
                            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

                        # Plot misclassified images from the last experiment
                        if SHOWIMGS and mis_images and experiment == num_experiments - 1:
                            mis_images = torch.cat(mis_images, dim=0)[:36]
                            mis_trues  = np.concatenate(mis_trues)[:36]
                            mis_preds  = np.concatenate(mis_preds)[:36]
                            
                            fig, axes = plt.subplots(6, 6, figsize=(12, 12))
                            axes = axes.flatten()

                            for i, ax in enumerate(axes[:len(mis_images)]):
                                img_tensor = mis_images[i]
                                # pick the first channel if there are two, else drop the singleton channel
                                img = img_tensor[0] if img_tensor.shape[0] > 1 else img_tensor.squeeze(0)
                                ax.imshow(img.numpy(), cmap='viridis')
                                ax.set_title(f"T={mis_trues[i]}, P={mis_preds[i]}")
                                ax.axis('off')

                            for ax in axes[len(mis_images):]:
                                ax.axis('off')

                            out_path = f"./hypertuning_outputs/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_misclassified_{ver_key}.png"
                            fig.savefig(out_path, dpi=150, bbox_inches='tight')
                            plt.close(fig)

                end_time = time.time()
                elapsed_time = end_time - start_time
                training_times.setdefault(fold, {}).setdefault(subset_size, []).append(elapsed_time)

                if fold == folds[-1] and experiment == num_experiments - 1:
                    with open(log_path, 'a') as file:
                        file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

            model_save_path = f'./classifier/trained_models/{base}_model.pth'
            torch.save(model.state_dict(), model_save_path)
            
           
    # Calculate and print grand average over ALL folds and experiments
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    for fold in folds:
        for subset_size in dataset_sizes[fold]:
            base = f"cl{classifier_name}_ss{subset_size}_f{fold}_lr{lr}_reg{reg}_ls{label_smoothing}_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{ver_key}"
            
            # Collect metrics from this fold/subset combination
            if f"{base}_accuracy" in metrics:
                all_accuracies.extend(metrics[f"{base}_accuracy"])
                all_precisions.extend(metrics[f"{base}_precision"])
                all_recalls.extend(metrics[f"{base}_recall"])
                all_f1_scores.extend(metrics[f"{base}_f1_score"])

    # Calculate grand averages
    grand_mean_acc = float(np.mean(all_accuracies)) if all_accuracies else float('nan')
    grand_mean_prec = float(np.mean(all_precisions)) if all_precisions else float('nan')
    grand_mean_rec = float(np.mean(all_recalls)) if all_recalls else float('nan')
    grand_mean_f1 = float(np.mean(all_f1_scores)) if all_f1_scores else float('nan')

    print("\n" + "="*80)
    print(f"GRAND AVERAGE over ALL {len(folds)} folds and {num_experiments} experiments:")
    print(f"  Accuracy:  {grand_mean_acc:.4f} ± {float(np.std(all_accuracies)):.4f}")
    print(f"  Precision: {grand_mean_prec:.4f} ± {float(np.std(all_precisions)):.4f}")
    print(f"  Recall:    {grand_mean_rec:.4f} ± {float(np.std(all_recalls)):.4f}")
    print(f"  F1 Score:  {grand_mean_f1:.4f} ± {float(np.std(all_f1_scores)):.4f}")
    print(f"  Total experiments: {len(all_accuracies)}")
    print("="*80 + "\n")