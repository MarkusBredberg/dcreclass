import os, re, time, random, pickle, hashlib, itertools, torch
from utils.data_loader import load_galaxies, get_classes
from utils.classifiers import (
    RustigeClassifier, TinyCNN, MLPClassifier, SCNN, CNNSqueezeNet, ScatterResNet,
    BinaryClassifier, ScatterSqueezeNet, ScatterSqueezeNet2, DualCNNSqueezeNet)
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchsummary import summary
from kymatio.torch import Scattering2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42  # Set a seed for reproducibility # Original: 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

print("Running script 4.1 with dl1 Latest version with seed", SEED)


###############################################
################ CONFIGURATION ################
###############################################
galaxy_classes    = [50, 51]
max_num_galaxies  = 1000000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions  = [1]
J, L, order       = 2, 12, 2
num_epochs_cuda = 100
num_epochs_cpu = 100
learning_rates = [1e-3]
regularization_params = [1e-3]  
label_smoothing = 0.2
num_experiments = 5
folds = [0] # 0-9 for 10-fold cross validation, 10 for only one training
percentile_lo = 30 # Percentile stretch lower bound
percentile_hi = 90  # Percentile stretch upper bound
versions = 'RT50kpc' # any mix of loadable and runtime-tapered planes. 'rt50' or 'rt100' for tapering. Square brackets for stacking

classifier = ["TinyCNN", # Very Simple CNN
              "Rustige", # Simple CNN from Rustige et al. 2023, https://github.com/floriangriese/wGAN-supported-augmentation/blob/main/src/Classifiers/SimpleClassifiers/Classifiers.py
              "SCNN", # Simple CNN similar to Rustige's
              "CNNSqueezeNet", # SCNN with Squeeze-and-Excitation blocks
              "DualCNNSqueezeNet", # Dual CNN with Squeeze-and-Excitation blocks
              "ScatterNet", "ScatterSqueezeNet", "ScatterSqueezeNet2",
              "Binary", "ScatterResNet"][-4]

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
PRINTFILENAMES = True
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights
ES, patience = True, 30  # Use early stopping
SCHEDULER = True  # Use a learning rate scheduler
SHOWIMGS = True  # Show some generated images for each class (Tool for control)


########################################################################
##################### AUTOMATIC CONFIGURATION ##########################
########################################################################

os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)
 
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
GLOBAL_NORM_MODE = "none" if not USE_GLOBAL_NORMALISATION else GLOBAL_NORM_MODE

if galaxy_classes[0] in list(range(50, 60)):
    crop_size = (512, 512)  # Crop size for the images
    downsample_size = (128, 128)  # Downsample size for the images
    batch_size = 16 
else:
    print("Unknown galaxy class specified. Exiting.")
    exit(1)

img_shape = downsample_size
cs = f"{crop_size[-2]}x{crop_size[-1]}"
ds = f"{downsample_size[-2]}x{downsample_size[-1]}"

num_classes = len(galaxy_classes)

def _verkey(v):
    if isinstance(v, (list, tuple)):
        return "+".join(map(str, v))
    return str(v)
ver_key = _verkey(versions)

scattering = Scattering2D(J=J, L=L, shape=img_shape[-2:], max_order=order)      
hidden_dim1 = 256
hidden_dim2 = 128
vae_latent_dim = 64

########################################################################
##################### HELPER FUNCTIONS #################################
########################################################################

#-------------------------------------------------------------
# Evaluation
#-------------------------------------------------------------

def check_overfitting_indicators(metrics, history, base, num_experiments):
    """Print comprehensive overfitting diagnostics."""
        
    print("\n" + "="*60)
    print("OVERFITTING DIAGNOSTICS")
    print("="*60)
    
    # 1. Accuracy levels
    print(f"📊 Train Accuracy: {np.mean(metrics[f'{base}_train_acc']):.4f} ± {np.std(metrics[f'{base}_train_acc']):.4f}")
    print(f"📊 Validation Accuracy: {np.mean(metrics[f'{base}_val_acc']):.4f} ± {np.std(metrics[f'{base}_val_acc']):.4f}")
    print(f"📊 Test Accuracy: {np.mean(metrics[f'{base}_accuracy']):.4f} ± {np.std(metrics[f'{base}_accuracy']):.4f}")
    
    # 2. Check if early stopping triggered
    for experiment in range(num_experiments):
        loss_key = f"{base}_{experiment}_train_loss"
        if loss_key in history:
            epochs_trained = len(history[loss_key])
            print(f"📈 Epochs trained for experiment {experiment}: {epochs_trained-patience}/{num_epochs}")
    
    print("="*60 + "\n")
    
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
                 
def plot_intensity_histogram(tensor1, tensor2, label1, label2, save_path, bins=30):
    vals1 = tensor1.sum(dim=tuple(range(1, tensor1.ndim))).cpu().numpy()
    vals2 = tensor2.sum(dim=tuple(range(1, tensor2.ndim))).cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.hist(vals1, bins=bins, alpha=0.6, label=label1, color='C1')
    plt.hist(vals2, bins=bins, alpha=0.6, label=label2, color='C0')
    plt.xlabel('Total Intensity'); plt.ylabel('Count')
    plt.title(f'Total Intensity per Image: {label1} vs {label2}')
    plt.legend(); plt.savefig(save_path); plt.close()

def plot_training_history(history, base, experiment, save_dir='./classifier/test'):
    """
    Plot training, validation, and test loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history
        base: Base key for this experiment
        experiment: Experiment number
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    train_loss_key = f"{base}_{experiment}_train_loss"
    val_loss_key = f"{base}_{experiment}_val_loss"
    test_loss_key = f"{base}_{experiment}_test_loss"
    train_acc_key = f"{base}_{experiment}_train_acc"
    val_acc_key = f"{base}_{experiment}_val_acc"
    test_acc_key = f"{base}_{experiment}_test_acc"
    
    # Check if keys exist
    if train_loss_key not in history or val_loss_key not in history:
        print(f"Warning: Loss keys not found for {base}_{experiment}")
        return
    
    train_losses = history[train_loss_key]
    val_losses = history[val_loss_key]
    test_losses = history.get(test_loss_key, [])
    train_accs = history.get(train_acc_key, [])
    val_accs = history.get(val_acc_key, [])
    test_accs = history.get(test_acc_key, [])
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    if test_losses:
        ax1.plot(epochs[:len(test_losses)], test_losses, 'g-', label='Test Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training, Validation, and Test Loss\n{base}_exp{experiment}', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for best validation loss
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax1.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.plot(best_epoch, best_val_loss, 'o', color='orange', markersize=8, 
                label=f'Best Val (epoch {best_epoch})')
        ax1.legend(fontsize=11)
    
    # Plot accuracy (if available)
    if train_accs and val_accs:
        ax2.plot(epochs[:len(train_accs)], train_accs, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs[:len(val_accs)], val_accs, 'r-', label='Val Acc', linewidth=2)
        if test_accs:
            ax2.plot(epochs[:len(test_accs)], test_accs, 'g-', label='Test Acc', linewidth=2)
        
        # Add vertical line at best validation epoch
        if val_losses:
            best_epoch = np.argmin(val_losses) + 1
            ax2.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Mark the accuracy values at best validation epoch
            if best_epoch <= len(val_accs):
                ax2.plot(best_epoch, val_accs[best_epoch-1], 'o', color='red', markersize=8)
            if test_accs and best_epoch <= len(test_accs):
                ax2.plot(best_epoch, test_accs[best_epoch-1], 'o', color='green', markersize=8)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title(f'Training, Validation, and Test Accuracy\n{base}_exp{experiment}', fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    else:
        ax2.text(0.5, 0.5, 'Accuracy data not available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    save_path = f"{save_dir}/{base}_exp{experiment}_training_curves_rs=41.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
def img_hash(img: torch.Tensor) -> str:
    arr = img.cpu().contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()

def _desc(name, x):
    print(f"[{name}] shape={tuple(x.shape)}, min={float(x.min()):.3g}, max={float(x.max()):.3g}")
                
# -------------------------------------------------------------
# Data processing helpers
# -------------------------------------------------------------

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
    train_acc_key = f"{base}_{experiment}_train_acc"
    val_loss_key = f"{base}_{experiment}_val_loss"
    val_acc_key = f"{base}_{experiment}_val_acc"
    test_loss_key = f"{base}_{experiment}_test_loss"
    test_acc_key = f"{base}_{experiment}_test_acc"
    for key in [train_loss_key, val_loss_key, val_acc_key, train_acc_key, test_loss_key, test_acc_key]:
        if key not in history:
            history[key] = []

def initialise_labels(key, all_true_labels, all_pred_labels):
    all_true_labels[key] = []
    all_pred_labels[key] = []


###############################################
########## INITIALIZE DICTIONARIES ############
###############################################

metrics = {
    "accuracy": {},
    "precision": {},
    "recall": {},
    "f1_score": {}
}

metric_colors = {
    "accuracy": 'blue',
    "precision": 'green',
    "recall": 'red',
    "f1_score": 'orange'
}

all_true_labels = {}
all_pred_labels = {}
training_times = {}
all_pred_probs = {}
history = {} 
dataset_sizes = {}


###############################################
########### READ IN TEST DATA #################
######## Needs only be done once ##############
###############################################

_out  = _loader(galaxy_classes=galaxy_classes,
            versions=versions, 
            fold=max(folds), #Any fold other than 5 gives me the test data for the five fold cross validation
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
            train=False)  # Obtain the test set

if len(_out) == 4:
    train_val_images, train_val_labels, test_images, test_labels = _out
    train_val_fns = test_fns = None
elif len(_out) == 6:
    train_val_images, train_val_labels, test_images, test_labels, train_val_fns, test_fns = _out
else:
    raise ValueError(f"load_galaxies returned {len(_out)} values, expected 4 or 6")

perm_test = torch.randperm(test_images.size(0))
test_images, test_labels = test_images[perm_test], test_labels[perm_test]
if PRINTFILENAMES and test_fns is not None:
    test_fns = permute_like(test_fns, perm_test)

test_labels = relabel(test_labels)
print("Labels of the test set after relabelling:", torch.unique(test_labels, return_counts=True))


##############################################################################
################# NORMALISE AND PACKAGE TEST DATA ############################
##############################################################################

if classifier in ['Rustige', 'CNNSqueezeNet', 'SCNN', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
    test_images = _as_5d(test_images).to(DEVICE)

# Prepare input data
if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2']:
    # Define cache paths (you can adjust these names as needed)
    test_cache_path = f"./.cache/test_scat_{galaxy_classes}_{dataset_portions[0]}_{FILTERED}.npy"

    # fold T into C on both real & scattering inputs
    test_images = fold_T_axis(test_images) # Merges the image version into the channel dimension
    mock_test = torch.zeros_like(test_images)
    test_cache = f"./.cache/test_scat_{galaxy_classes}_{dataset_portions[0]}_{FILTERED}.pt"
    test_scat_coeffs = compute_scattering_coeffs(test_images, scattering, batch_size=128, device="cpu")

    if test_scat_coeffs.dim() == 5:
        # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
        print("Shape of test_scat_coeffs before flattening: ", test_scat_coeffs.shape)
        test_scat_coeffs = test_scat_coeffs.flatten(start_dim=1, end_dim=2)
        print("Shape of test_scat_coeffs after flattening: ", test_scat_coeffs.shape)

    if NORMALISESCS or NORMALISESCSTOPM:
        # Now we need to normalise the scattering coefficients globally, and thus compute the scattering coeffs for all data
        trainval_cache_path = f"./.cache/trainval_scat_{galaxy_classes}_{dataset_portions[0]}_{FILTERED}.npy"
        trainval_images = fold_T_axis(train_val_images)
        trainval_cache = f"./.cache/trainval_scat_{galaxy_classes}_{dataset_portions[0]}_{FILTERED}.pt"
        trainval_scat_coeffs = compute_scattering_coeffs(trainval_images, scattering, batch_size=128, device="cpu")
        if trainval_scat_coeffs.dim() == 5:
            # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
            trainval_scat_coeffs = trainval_scat_coeffs.flatten(start_dim=1, end_dim=2)
        all_scat = torch.cat([trainval_scat_coeffs, test_scat_coeffs], dim=0)
        if NORMALISESCSTOPM:
            all_scat = normalise_images(all_scat, -1, 1)
        else:
            all_scat = normalise_images(all_scat, 0, 1)
        trainval_scat_coeffs, test_scat_coeffs = all_scat[:len(trainval_scat_coeffs)], all_scat[len(trainval_scat_coeffs):]
    if classifier in ['ScatterNet', 'ScatterResNet']:
        test_dataset = TensorDataset(mock_test, test_scat_coeffs, test_labels)
    else: # if classifier in ['ScatterSqueezeNet', 'ScatterSqueezeNet2']:
        test_dataset = TensorDataset(test_images, test_scat_coeffs, test_labels)
else:
    if test_images.dim() == 5:
        test_images = fold_T_axis(test_images)  # [B,T,1,H,W] -> [B,T,H,W]
    assert test_images.dim() == 4, f"test_images should be [B,C,H,W], got {tuple(test_images.shape)}"
    mock_test = torch.zeros_like(test_images)
    test_dataset = TensorDataset(test_images, mock_test, test_labels)
                        
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)


###############################################
########### LOOP OVER DATA FOLD ###############
###############################################            

FIRSTTIME = True  # Set to True to print model summaries only once
param_combinations = list(itertools.product(folds, learning_rates, regularization_params))
for fold, lr, reg in param_combinations:
    torch.cuda.empty_cache()
    runname = f"{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{ver_key}_f{fold}"

    print(f"\n▶ Experiment: g_classes={galaxy_classes}, lr={lr}, reg={reg}, ls={label_smoothing}, "
    f"J={J}, L={L}, crop={crop_size}, down={downsample_size}, ver={versions}, "
    f"lo={percentile_lo}, hi={percentile_hi}, classifier={classifier}, "
    f"global_norm={USE_GLOBAL_NORMALISATION}, norm_mode={GLOBAL_NORM_MODE}, "
    f"PREFER_PROCESSED={PREFER_PROCESSED} ◀\n")

    log_path = f"./classifier/log_{runname}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _out = _loader(
            galaxy_classes=galaxy_classes,
            versions=versions, 
            fold=fold, #Any fold other than 5 gives me the test data for the five fold cross validation
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

    # ——— Data sanity checks ———
    
    # after loading train_images, test_images:
    train_hashes = {img_hash(img) for img in train_images}
    valid_hashes = {img_hash(img) for img in valid_images}
    test_hashes  = {img_hash(img) for img in test_images}

    train_val_common, train_test_common, val_test_common = train_hashes & valid_hashes, train_hashes & test_hashes, valid_hashes & test_hashes
    assert not train_val_common, f"Overlap detected: {len(train_val_common)} images appear in both train and validation!"
    assert not train_test_common, f"Overlap detected: {len(train_test_common)} images appear in both train and test validation!"
    assert not val_test_common, f"Overlap detected: {len(val_test_common)} images appear in both validation and test validation!"
    
    for i, cls in enumerate(galaxy_classes):
        if MULTILABEL:
            train_mask = train_labels[:, i] > 0.5
            valid_mask = valid_labels[:, i] > 0.5
            test_mask  = test_labels[:,  i] > 0.5
        else:
            train_mask = as_index_labels(train_labels) == i
            valid_mask = as_index_labels(valid_labels) == i
            test_mask  = as_index_labels(test_labels)  == i

        check_tensor(f"Train images for class {cls} (idx={i})", train_images[train_mask])
        check_tensor(f"Valid images for class {cls} (idx={i})", valid_images[valid_mask])
        check_tensor(f"Test images for class {cls} (idx={i})",  test_images[test_mask])

    _desc("TRAIN", train_images)
    _desc("VALID", valid_images)
    _desc("TEST", test_images)

    print("First 10 training labels after relabelling:", train_labels[:10], "Label distribution:", torch.unique(train_labels, return_counts=True))         
    print("First 10 validation labels after relabelling:", valid_labels[:10], "Label distribution:", torch.unique(valid_labels, return_counts=True))
    print("First 10 test labels after relabelling:", test_labels[:10], "Label distribution:", torch.unique(test_labels, return_counts=True))
    # ————————————————————————

    if dataset_sizes == {}:
        dataset_sizes[fold] = [max(2, int(len(train_images) * p)) for p in dataset_portions]
        print(f"Dataset sizes for fold {fold}: {dataset_sizes[fold]}")

    ##########################################################
    ############ NORMALISE AND PACKAGE THE INPUT #############
    ##########################################################

    if classifier in ['Rustige', 'CNNSqueezeNet', 'SCNN', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
        train_images = _as_5d(train_images).to(DEVICE)
        valid_images = _as_5d(valid_images).to(DEVICE)
    
    if MULTILABEL:
        # labels are 2-hot; compute per-label pos_weight for BCE
        pos_counts = train_labels.sum(dim=0)                       # [2]
        neg_counts = train_labels.shape[0] - pos_counts            # [2]
        pos_counts = torch.clamp(pos_counts, min=1.0)
        pos_weight = (neg_counts / pos_counts).to(DEVICE)          # [2]
        print(f"[pos_weight] RH={pos_weight[0].item():.2f}, RR={pos_weight[1].item():.2f}")

    # Prepare input data
    if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2']:
        # Define cache paths (you can adjust these names as needed)
        train_cache_path = f"./.cache/train_scat_{galaxy_classes}_{fold}_{dataset_portions[0]}_{FILTERED}.npy"
        valid_cache_path = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{dataset_portions[0]}_{FILTERED}.npy"
    
        # fold T into C on both real & scattering inputs
        train_images = fold_T_axis(train_images) # Merges the image version into the channel dimension
        valid_images = fold_T_axis(valid_images)
        mock_train = torch.zeros_like(train_images)
        mock_valid = torch.zeros_like(valid_images)

        train_cache = f"./.cache/train_scat_{galaxy_classes}_{fold}_{dataset_portions[0]}_{FILTERED}.pt"
        valid_cache = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{dataset_portions[0]}_{FILTERED}.pt"
        train_scat_coeffs = compute_scattering_coeffs(train_images, scattering, batch_size=128, device="cpu")
        valid_scat_coeffs = compute_scattering_coeffs(valid_images, scattering, batch_size=128, device="cpu")

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
        for x,name in [(train_images,"train"), (valid_images,"valid")]:
            assert x.dim() == 4, f"{name}_images should be [B,C,H,W], got {tuple(x.shape)}"
        train_dataset = TensorDataset(train_images, mock_train, train_labels)
        valid_dataset = TensorDataset(valid_images, mock_valid, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

        
    ###############################################
    ############# SANITY CHECKS ###################
    ###############################################
    
    if fold == folds[0] and SHOWIMGS and downsample_size[-1] == 128 and FIRSTTIME:     
        # Plot some example training images with their labels
        imgs = train_images.detach().cpu().numpy()
        lbls = (as_index_labels(train_labels) + min(galaxy_classes)).detach().cpu().numpy()
        plot_images_by_class(
            imgs,
            labels=lbls,
            num_images=5,
            save_path=f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_example_train_data.pdf"
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
                save_path=f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_histogram_{ver_key}.pdf"
            )
            
            plot_background_histogram(
                train_images_cls1.cpu(),
                train_images_cls2.cpu(),
                img_shape=(1, 128, 128),
                title="Background histograms",
                save_path=f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_background_hist.pdf"
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
            plt.savefig(f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_sum_histogram_{ver_key}.pdf")
            plt.close()

            for cls in galaxy_classes:
                cls_idx = cls - min(galaxy_classes)

                sel_train = torch.where(as_index_labels(train_labels) == cls_idx)[0][:36]
                titles_train = [train_fns[i] for i in sel_train.tolist()] if train_fns is not None else None
                sel_valid = torch.where(as_index_labels(valid_labels) == cls_idx)[0][:36]
                titles_valid = [valid_fns[i] for i in sel_valid.tolist()] if valid_fns is not None else None
                sel_test = torch.where(as_index_labels(test_labels) == cls_idx)[0][:36]
                titles_test = [test_fns[i] for i in sel_test.tolist()] if test_fns is not None else None

                imgs4plot = _ensure_4d(train_images)[sel_train].cpu()
                plot_image_grid(
                    imgs4plot, num_images=36, titles=titles_train,
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_train_grid_{ver_key}.pdf"
                )

                imgs4plot = _ensure_4d(valid_images)[sel_valid].cpu()
                plot_image_grid(
                    imgs4plot, num_images=36, titles=titles_valid,
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_valid_grid_{ver_key}.pdf"
                )
                
                imgs4plot = _ensure_4d(test_images)[sel_test].cpu()
                plot_image_grid(
                    imgs4plot, num_images=36, titles=titles_test,
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_test_grid_{ver_key}.pdf"
                )

                # summed-intensity histogram helper unchanged...
                tag_to_desc = { d["tag"]: d["description"] for d in get_classes() }

                plot_intensity_histogram(
                    train_images_cls1.cpu(),
                    train_images_cls2.cpu(),
                    label1=tag_to_desc[get_classes()[galaxy_classes[0]]['tag']],
                    label2=tag_to_desc[get_classes()[galaxy_classes[1]]['tag']],
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_summed_intensity_histogram.pdf"
                )
                
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
    elif classifier == "TinyCNN":
        models = {"TinyCNN": {"model": TinyCNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
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

    for classifier_name, model_details in models.items():
        if FIRSTTIME:
            print(f"Summary for {classifier_name}:")
            if classifier == "ScatterNet":
                summary(model_details["model"], input_size=(int(np.prod(scatdim)),), device=DEVICE)
            elif classifier == "ScatterResNet":
                summary(model_details["model"], input_size=scatdim, device=DEVICE)
            elif classifier == "ScatterSqueezeNet":
                summary(model_details["model"], input_size=[valid_images.shape[1:], scatdim])
            elif classifier == "ScatterSqueezeNet2":
                summary(model_details["model"], input_size=[valid_images.shape[1:], scatdim])
            else:
                summary(model_details["model"], input_size=tuple(valid_images.shape[1:]), device=DEVICE)
        FIRSTTIME = False
        
        
    ###############################################
    ############### TRAINING LOOP #################
    ###############################################
    
    if USE_CLASS_WEIGHTS:
        unique, counts = np.unique(train_labels.cpu().numpy(), return_counts=True)
        total_count = sum(counts)
        class_weights = {i: total_count / count for i, count in zip(unique, counts)}
        weights = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)],
                            dtype=torch.float, device=DEVICE)
    else:
        weights = None
    
    if MULTILABEL:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        if weights is not None:
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
                    train_correct = 0  # Add this to track training accuracy per epoch

                    for images, scat, _rest in subset_train_loader:
                        labels = _rest
                        images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        if classifier in ["ScatterNet", "ScatterResNet"]:
                            logits = model(scat)
                        elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
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
                        
                        # Track training accuracy during training
                        with torch.no_grad():
                            if MULTILABEL:
                                preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                                trues = labels.cpu().numpy()
                                train_correct += (preds == trues).all(axis=1).sum()
                            else:
                                preds = logits.argmax(dim=1)
                                train_correct += (preds == labels).sum().item()

                    # Calculate epoch training metrics
                    train_average_loss = train_total_loss / train_total_images
                    train_epoch_acc = train_correct / train_total_images if train_total_images > 0 else 0.0
                    train_loss_key = f"{base}_{experiment}_train_loss"
                    train_acc_key = f"{base}_{experiment}_train_acc"
                    history[train_loss_key].append(train_average_loss)
                    history[train_acc_key].append(train_epoch_acc)  # Store per-epoch train accuracy


                    model.eval()
                    val_total_loss = 0
                    val_total_images = 0
                    val_correct = 0  # Add this to track validation accuracy per epoch
                    with torch.inference_mode():
                        for i, (images, scat, _rest) in enumerate(valid_loader): 
                            if images is None or len(images) == 0:
                                print(f"Empty batch at index {i}. Skipping...")
                                continue
                            labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            if classifier in ["ScatterNet", "ScatterResNet"]:
                                logits = model(scat)
                            elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
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

                            # Track validation accuracy per epoch
                            if MULTILABEL:
                                preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                                trues = labels.cpu().numpy()
                                val_correct += (preds == trues).all(axis=1).sum()
                            else:
                                preds = logits.argmax(dim=1)
                                val_correct += (preds == labels).sum().item()

                    val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                    val_epoch_acc = val_correct / val_total_images if val_total_images > 0 else 0.0
                    val_loss_key = f"{base}_{experiment}_val_loss"
                    val_acc_key = f"{base}_{experiment}_val_acc"
                    history[val_loss_key].append(val_average_loss)
                    history[val_acc_key].append(val_epoch_acc)  # Store per-epoch val accuracy
                    
                    # ADD TEST EVALUATION DURING TRAINING
                    test_total_loss = 0
                    test_total_images = 0
                    test_correct = 0
                    with torch.inference_mode():
                        for i, (images, scat, _rest) in enumerate(test_loader): 
                            if images is None or len(images) == 0:
                                continue
                            labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            
                            if classifier in ["ScatterNet", "ScatterResNet"]:
                                logits = model(scat)
                            elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                                logits = model(images, scat)
                            else:
                                logits = model(images)

                            logits = collapse_logits(logits, num_classes, MULTILABEL)
                            labels = labels.float() if MULTILABEL else labels.long()
                            
                            loss = criterion(logits, labels)
                            batch_size = images.size(0)
                            
                            test_total_loss += float(loss.item() * batch_size)
                            
                            if MULTILABEL:
                                preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                                trues = labels.cpu().numpy()
                                test_correct += (preds == trues).all(axis=1).sum()
                            else:
                                preds = logits.argmax(dim=1)
                                test_correct += (preds == labels).sum().item()
                            
                            test_total_images += batch_size

                    test_average_loss = test_total_loss / test_total_images if test_total_images > 0 else float('inf')
                    test_epoch_acc = test_correct / test_total_images if test_total_images > 0 else 0.0
                    test_loss_key = f"{base}_{experiment}_test_loss"
                    test_acc_key = f"{base}_{experiment}_test_acc"
                    history[test_loss_key].append(test_average_loss)
                    history[test_acc_key].append(test_epoch_acc)
                    
                    # Update print statement to include test metrics
                    print(f"Epoch [{epoch+1}/{num_epochs}] - "
                        f"Train Loss: {train_average_loss:.4f}, Train Acc: {train_epoch_acc:.4f} - "
                        f"Val Loss: {val_average_loss:.4f}, Val Acc: {val_epoch_acc:.4f} - "
                        f"Test Loss: {test_average_loss:.4f}, Test Acc: {test_epoch_acc:.4f}")
                    
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
                    
                plot_training_history(history, base, experiment, save_dir='./classifier/test')

                model.eval()                  
                with torch.inference_mode():
                    train_correct = 0
                    train_total = 0
                    for images, scat, _rest in subset_train_loader: # Evaluate on training data
                        labels = _rest
                        images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                        
                        if classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                            logits = model(images, scat)
                        else:
                            logits = model(images)
                            
                        logits = collapse_logits(logits, num_classes, MULTILABEL)
                        
                        if MULTILABEL:
                            preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                            trues = labels.cpu().numpy()
                            train_correct += (preds == trues).all(axis=1).sum()
                        else:
                            preds = logits.argmax(dim=1)
                            train_correct += (preds == labels).sum().item()
                        
                        train_total += images.size(0)

                final_train_acc = train_correct / train_total if train_total > 0 else 0.0
                metrics.setdefault(f'{base}_train_acc', []).append(final_train_acc)
                
                with torch.inference_mode():
                    val_correct = 0
                    val_total = 0
                    for images, scat, _rest in valid_loader: # Evaluate on validation data 
                        labels = _rest
                        images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                        if classifier in ["ScatterNet", "ScatterResNet"]:
                            logits = model(scat)
                        elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                            logits = model(images, scat)
                        else:
                            logits = model(images)
                            
                        logits = collapse_logits(logits, num_classes, MULTILABEL)
                        if MULTILABEL:
                            preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                            trues = labels.cpu().numpy()
                            val_correct += (preds == trues).all(axis=1).sum()
                        else:
                            preds = logits.argmax(dim=1)
                            val_correct += (preds == labels).sum().item()
                        
                        val_total += images.size(0)
                
                final_val_acc = val_correct / val_total if val_total > 0 else 0.0
                print(f"Final validation accuracy for experiment {experiment}: {final_val_acc:.4f}")
                metrics.setdefault(f'{base}_val_acc', []).append(final_val_acc)
                    
                with torch.inference_mode(): 
                    all_pred_probs[base] = []
                    all_pred_labels[base] = []
                    all_true_labels[base] = []
                    mis_images = []
                    mis_trues  = []
                    mis_preds  = []
                    all_logits = []
                    
                    for images, scat, _rest in test_loader: # Evaluate on test data
                        labels = _rest
                        images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                        if classifier in ["ScatterNet", "ScatterResNet"]:
                            logits = model(scat)
                        elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                            logits = model(images, scat)
                        else:
                            logits = model(images)
                            
                        all_logits.append(logits.cpu().numpy())
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
                            #ax.set_title(f"T={mis_trues[i]}, P={mis_preds[i]}")
                            # Instead of seeing T=0, P=1. I want to see the actual class names (T=DE, P+NDE)
                            if MULTILABEL:
                                true_labels = [str(min(galaxy_classes) + idx) for idx, val in enumerate(mis_trues[i]) if val == 1]
                                pred_labels = [str(min(galaxy_classes) + idx) for idx, val in enumerate(mis_preds[i]) if val == 1]
                                ax.set_title(f"T={','.join(true_labels)}\nP={','.join(pred_labels)}", fontsize=8)
                            else:
                                ax.set_title(f"T={mis_trues[i]}, P={mis_preds[i]}", fontsize=8)
                            
                            ax.axis('off')

                        for ax in axes[len(mis_images):]:
                            ax.axis('off')

                        out_path = f"./classifier/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_misclassified_{ver_key}.pdf"
                        fig.savefig(out_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)

            mean_acc = float(np.mean(metrics[f"{base}_accuracy"])) if metrics[f"{base}_accuracy"] else float('nan')
            mean_prec = float(np.mean(metrics[f"{base}_precision"])) if metrics[f"{base}_precision"] else float('nan')
            mean_rec = float(np.mean(metrics[f"{base}_recall"])) if metrics[f"{base}_recall"] else float('nan')
            mean_f1 = float(np.mean(metrics[f"{base}_f1_score"])) if metrics[f"{base}_f1_score"] else float('nan')
            print(f"AVERAGE over {num_experiments} experiments — Accuracy: {mean_acc:.4f}, Precision: {mean_prec:.4f}, Recall: {mean_rec:.4f}, F1 Score: {mean_f1:.4f}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            training_times.setdefault(fold, {}).setdefault(subset_size, []).append(elapsed_time)

            if fold == folds[-1] and experiment == num_experiments - 1:
                with open(log_path, 'a') as file:
                    file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

        model_save_path = f'./classifier/trained_models/{base}_model.pth'
        torch.save(model.state_dict(), model_save_path)
        
        check_overfitting_indicators(metrics, history, base, num_experiments)

        all_logits = np.concatenate(all_logits, axis=0)
        cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(all_logits, n_clusters=num_classes)
        with open(log_path, 'a') as file:
            file.write(f"Results for fold {fold}, Classifier {classifier_name}, lr={lr}, reg={reg}, percentile_lo={percentile_lo}, percentile_hi={percentile_hi}, crop_size={crop_size}, downsample_size={downsample_size}, STRETCH={STRETCH}, FILTERED={FILTERED} \n")
            file.write(f"Cluster Error: {cluster_error} \n")
            file.write(f"Cluster Distance: {cluster_distance} \n")
            file.write(f"Cluster Standard Deviation: {cluster_std_dev} \n")

        model_save_path = f'./classifier/trained_models/{base}_model.pth'
        torch.save(model.state_dict(), model_save_path)
        
# Create aggregate plot showing all experiments
if num_experiments > 1:
    plot_dir = './classifier/test'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Created directory: {plot_dir}")
    
    print("Creating aggregate plots for all experiments...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 grid
    
    for experiment in range(num_experiments):
        train_loss_key = f"{base}_{experiment}_train_loss"
        val_loss_key = f"{base}_{experiment}_val_loss"
        test_loss_key = f"{base}_{experiment}_test_loss"
        train_acc_key = f"{base}_{experiment}_train_acc"
        val_acc_key = f"{base}_{experiment}_val_acc"
        test_acc_key = f"{base}_{experiment}_test_acc"
        
        if train_loss_key in history and val_loss_key in history:
            epochs = range(1, len(history[train_loss_key]) + 1)
            
            # Plot losses
            axes[0, 0].plot(epochs, history[train_loss_key], alpha=0.6, label=f'Exp {experiment}')
            axes[0, 1].plot(epochs, history[val_loss_key], alpha=0.6, label=f'Exp {experiment}')
            if test_loss_key in history:
                axes[0, 2].plot(epochs[:len(history[test_loss_key])], history[test_loss_key], 
                               alpha=0.6, label=f'Exp {experiment}')
            
            # Plot accuracies
            if train_acc_key in history and val_acc_key in history:
                axes[1, 0].plot(epochs[:len(history[train_acc_key])], history[train_acc_key], 
                               alpha=0.6, label=f'Exp {experiment}')
                axes[1, 1].plot(epochs[:len(history[val_acc_key])], history[val_acc_key], 
                               alpha=0.6, label=f'Exp {experiment}')
                if test_acc_key in history:
                    axes[1, 2].plot(epochs[:len(history[test_acc_key])], history[test_acc_key], 
                                   alpha=0.6, label=f'Exp {experiment}')
    
    # Configure subplots
    axes[0, 0].set_title('Training Loss (All Experiments)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss (All Experiments)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Test Loss (All Experiments)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy (All Experiments)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].set_title('Validation Accuracy (All Experiments)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    axes[1, 2].set_title('Test Accuracy (All Experiments)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = f"{plot_dir}/{base}_all_experiments.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregate plot to {save_path}")

directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'
for fold, lr, reg in param_combinations:
    for subset_size in dataset_sizes[fold]:
        for experiment in range(num_experiments):
                            
            metrics_save_path = f'./classifier/4.1.runs/{runname}_sz{subset_size}_e{experiment}_metrics_data.pkl'
            # Build robust, per-setting summaries using empirical percentiles
            robust_summary = {}   # { base_key: {metric: {'n', 'p16','p50','p84','sigma68'} } }
            skip_keys = {"accuracy", "precision", "recall", "f1_score"}
            rows = []
            for key, values in metrics.items():
                if key in skip_keys:
                    continue
                if not isinstance(values, (list, tuple)) or len(values) == 0:
                    continue

                vals = np.asarray(values, dtype=float)
                p16, p50, p84 = np.percentile(vals, [16, 50, 84])
                sigma68 = 0.5 * (p84 - p16)

                base, metric_name = key.rsplit('_', 1)  # split "..._accuracy" → ("...", "accuracy")
                robust_summary.setdefault(base, {})[metric_name] = {
                    "n": int(vals.size),
                    "p16": float(p16),
                    "p50": float(p50),   # median
                    "p84": float(p84),
                    "sigma68": float(sigma68)  # half-width of the central 68% interval
                }
                
                # Histogram with percentile markers (bins span data range)
                vmin, vmax = float(np.min(vals)), float(np.max(vals))
                if vmin == vmax:  # guard against a degenerate run where all values are identical
                    eps = 1e-6
                    vmin, vmax = vmin - eps, vmax + eps
                edges = np.linspace(vmin, vmax, 21)  # 20 bins across [min, max]

                import matplotlib.patches as mpatches

                # compute stats
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                p16, p50, p84 = float(p16), float(p50), float(p84)

                plt.figure(figsize=(5,3.2))
                counts, bins, patches = plt.hist(vals, bins=edges, edgecolor='black', color='green', alpha=0.45)

                # vertical lines: mean (solid), median (dashed), p16/p84 (dotted)
                mean_line, = plt.plot([mean, mean], [0, counts.max()], linestyle='-', color='blue', linewidth=2, label=f"Mean = {mean:.3f}, σ = {std:.3f}")
                median_line, = plt.plot([p50, p50], [0, counts.max()], linestyle='--', color='orange', linewidth=2, label=f"Median = {p50:.3f}")
                p16_line = plt.plot([p16, p16], [0, counts.max()], color='green', linestyle=':', linewidth=1)
                p84_line = plt.plot([p84, p84], [0, counts.max()], color='green', linestyle=':', linewidth=1)
                mean_plus_std_line = plt.plot([mean + std, mean + std], [0, counts.max()], color='blue', linestyle='--', linewidth=1)
                mean_minus_std_line = plt.plot([mean - std, mean - std], [0, counts.max()], color='blue', linestyle='--', linewidth=1)

                # shaded central 68% credibility region (p16--p84)
                ymax = counts.max()
                region_patch = plt.fill_betweenx([0, ymax], p16, p84, alpha=0.12, facecolor='green')

                plt.xlim(vmin, vmax)
                plt.xlabel(metric_name.capitalize())
                plt.ylabel("Count")

                # assemble legend: use created artists and the shaded patch
                legend_handles = [median_line, mean_line, mpatches.Patch(facecolor='green', alpha=0.12, label=f"68% interval [{p16:.3f}, {p84:.3f}]")]
                plt.legend(handles=legend_handles, loc='best', frameon=False, fontsize='small')

                plt.tight_layout()

                save_path_hist = (
                    f"./classifier/{galaxy_classes}_{classifier}_"
                    f"{dataset_sizes[folds[0]][-1]}_{metric_name}_histogram.pdf"
                )
                plt.savefig(save_path_hist, dpi=150)
                plt.close()

                rows.append({
                    "setting": base,
                    "metric": metric_name,
                    "n": int(vals.size),
                    "p16": float(p16),
                    "p50": float(p50),
                    "p84": float(p84),
                    "sigma68": float(sigma68)
                })

            # Also write a tidy CSV so you can scan summaries quickly
            summary_csv = f'{directory}{classifier}_{galaxy_classes}_percentile_summary.csv'
            pd.DataFrame(rows).to_csv(summary_csv, index=False)
            
            with open(metrics_save_path, 'wb') as f:
                pickle.dump({
                    "models": models,
                    "history": history,
                    "metrics": metrics,
                    "metric_colors": metric_colors,
                    "all_true_labels": all_true_labels,
                    "all_pred_labels": all_pred_labels,
                    "training_times": training_times,
                    "all_pred_probs": all_pred_probs,
                    "percentile_summary": robust_summary
                }, f)
            print(f"Saved metrics PKL to {os.path.abspath(metrics_save_path)}")
                