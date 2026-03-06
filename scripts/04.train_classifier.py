import os, time, random, pickle, hashlib, itertools, torch
from dcreclass.data import load_galaxies, get_classes
from dcreclass.models import CNN, ScatterNet, DualCNNSqueezeNet, DualScatterSqueezeNet
from dcreclass.training import (EarlyStopping, reset_weights,
                                relabel, permute_like,
                                mixup_data, mixup_criterion,
                                config_already_exists,
                                initialise_history, initialise_labels, initialise_metrics,
                                compute_classification_metrics, update_metrics,
                                plot_training_history, plot_intensity_histogram,
                                check_overfitting, img_hash)
from dcreclass.utils import (cluster_metrics, normalise_images, check_tensor, fold_T_axis,
                             compute_scattering_coeffs, custom_collate, round_to_1,
                             plot_histograms, plot_images_by_class, plot_image_grid,
                             plot_background_histogram)
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchsummary import summary
from kymatio.torch import Scattering2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from math import log10, floor
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
num_epochs_cuda = 200
num_epochs_cpu = 100
lr = 4e-5 # Learning rate #prefer processed has 6e-5 for RAW
reg = 1e-1 # Weight decay (L2 regularization)
label_smoothing = 0.1
num_experiments = 1
folds = [0] # 0-9 for 10-fold cross validation, 10 for only one training
percentile_lo = 30 # Percentile stretch lower bound
percentile_hi = 99  # Percentile stretch upper bound
versions = ['RAW'] # any mix of loadable and runtime-tapered planes. 'rt50' or 'rt100' for tapering. Square brackets for stacking
classifier = ["CNN",         # 0.Very Simple CNN
              "ScatterNet",  # 1.Scattering coefficients as input to MLP
              "DualCSN",     # 2.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              "DualSSN"      # 3.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              ][3]

# Override configuration from environment variables set by the SLURM job script
if os.environ.get('CLASSIFIER'):
    classifier = os.environ['CLASSIFIER']
if os.environ.get('VERSIONS'):
    _ver_str = os.environ['VERSIONS']
    versions = _ver_str.split('+') if '+' in _ver_str else [_ver_str]
if os.environ.get('LR'):
    lr = float(os.environ['LR'])
if os.environ.get('REG'):
    reg = float(os.environ['REG'])

print(f"Using classifier: {classifier}")

OUTDIR_BASE = "/users/mbredber/p2_DCRECLASS/outputs/"

# Run-specific output directories, set via env vars from the SLURM job script.
# Falls back to legacy layout if not running inside a managed job.
_RUN_DIR = os.environ.get('RUN_DIR')
_DATA_RUN_DIR = os.environ.get('DATA_RUN_DIR')

if _RUN_DIR and _DATA_RUN_DIR:
    FIGURES_DIR = os.path.join(_RUN_DIR, 'figures')
    LOGS_DIR    = os.path.join(_RUN_DIR, 'logs')
    MODELS_DIR  = os.path.join(_DATA_RUN_DIR, 'models')
    METRICS_DIR = os.path.join(_DATA_RUN_DIR, 'metrics')
else:
    FIGURES_DIR = os.path.join(OUTDIR_BASE, 'classifier', 'figures')
    LOGS_DIR    = os.path.join(OUTDIR_BASE, 'classifier', 'logfiles')
    MODELS_DIR  = os.path.join(OUTDIR_BASE, 'classifier', 'trained_models')
    METRICS_DIR = os.path.join(OUTDIR_BASE, 'classifier', 'runs')

CROP_MODE = 'beam_crop'  # 'beam_crop' | 'fov_crop' | 'cheat_crop' | 'pixel_crop'
if os.environ.get('CROP_MODE'):
    CROP_MODE = os.environ['CROP_MODE']
STRETCH = True  # Arcsinh stretch
USE_GLOBAL_NORMALISATION = False          # single on/off switch . False - image-by-image normalisation 
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
USE_CACHE = False  # Use cached images, scattering coefficients, and metrics where available

########################################################################
##################### AUTOMATIC CONFIGURATION ##########################
########################################################################

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(f'{OUTDIR_BASE}/.cache/scattering_coefficients', exist_ok=True)

# Append full Python-side hyperparameters to the shared config file
_config_file = os.environ.get('CONFIG_FILE', os.path.join(LOGS_DIR, 'config.txt'))
with open(_config_file, 'a') as _cf:
    _cf.write("\n========================================\n")
    _cf.write("HYPERPARAMETERS (Python script)\n")
    _cf.write("========================================\n")
    _cf.write(f"SEED:                    {SEED}\n")
    _cf.write(f"classifier:              {classifier}\n")
    _cf.write(f"versions:                {versions}\n")
    _cf.write(f"CROP_MODE:               {CROP_MODE}\n")
    _cf.write(f"galaxy_classes:          {galaxy_classes}\n")
    _cf.write(f"folds:                   {folds}\n")
    _cf.write(f"num_experiments:         {num_experiments}\n")
    _cf.write(f"dataset_portions:        {dataset_portions}\n")
    _cf.write(f"max_num_galaxies:        {max_num_galaxies}\n")
    _cf.write(f"lr:                      {lr}\n")
    _cf.write(f"reg:                     {reg}\n")
    _cf.write(f"label_smoothing:         {label_smoothing}\n")
    _cf.write(f"num_epochs_cuda:         {num_epochs_cuda}\n")
    _cf.write(f"num_epochs_cpu:          {num_epochs_cpu}\n")
    _cf.write(f"J, L, order:             {J}, {L}, {order}\n")
    _cf.write(f"percentile_lo:           {percentile_lo}\n")
    _cf.write(f"percentile_hi:           {percentile_hi}\n")
    _cf.write(f"STRETCH:                 {STRETCH}\n")
    _cf.write(f"USE_GLOBAL_NORMALISATION:{USE_GLOBAL_NORMALISATION}\n")
    _cf.write(f"GLOBAL_NORM_MODE:        {GLOBAL_NORM_MODE}\n")
    _cf.write(f"NORMALISEIMGS:           {NORMALISEIMGS}\n")
    _cf.write(f"NORMALISEIMGSTOPM:       {NORMALISEIMGSTOPM}\n")
    _cf.write(f"NORMALISESCS:            {NORMALISESCS}\n")
    _cf.write(f"NORMALISESCSTOPM:        {NORMALISESCSTOPM}\n")
    _cf.write(f"FILTERED:                {FILTERED}\n")
    _cf.write(f"FILTERGEN:               {FILTERGEN}\n")
    _cf.write(f"AUGMENT:                 {AUGMENT}\n")
    _cf.write(f"MIXUP:                   {MIXUP}\n")
    _cf.write(f"USE_CLASS_WEIGHTS:       {USE_CLASS_WEIGHTS}\n")
    _cf.write(f"ES:                      {ES}\n")
    _cf.write(f"patience:                {patience}\n")
    _cf.write(f"SCHEDULER:               {SCHEDULER}\n")
    _cf.write(f"USE_CACHE:               {USE_CACHE}\n")

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

scattering = Scattering2D(J=J, L=L, shape=img_shape[-2:], max_order=order)      

########################################################################
############## LOCAL HELPER FUNCTIONS ##################################
########################################################################

def _verkey(v):
    if isinstance(v, (list, tuple)):
        return "+".join(map(str, v))
    return str(v)
ver_key = _verkey(versions)

def _as_index_labels(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=1) if y.ndim > 1 else y

def _as_5d(x):
    return x if x.dim() == 5 else x.unsqueeze(1)  # [B,1,H,W] -> [B,1,1,H,W]

def _collapse_logits(logits, num_classes, multilabel):
    if logits.ndim == 4:
        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
    elif logits.ndim == 3:
        logits = logits.mean(dim=-1)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if not multilabel and logits.shape[1] == 1 and num_classes == 2:
        logits = torch.cat([-logits, logits], dim=1)
    return logits

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

def _desc(name, x):
    print(f"[{name}] shape={tuple(x.shape)}, min={float(x.min()):.3g}, max={float(x.max()):.3g}")


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

_out  = load_galaxies(galaxy_classes=galaxy_classes,
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

            USE_CACHE=USE_CACHE,
            DEBUG=DEBUG,
            crop_mode=CROP_MODE,
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

test_labels = relabel(test_labels, galaxy_classes, multilabel=MULTILABEL)
print("Labels of the test set after relabelling:", torch.unique(test_labels, return_counts=True))


##############################################################################
################# NORMALISE AND PACKAGE TEST DATA ############################
##############################################################################

if classifier in ['CNN', 'DualCSN', 'DualSSN']: # When images are used
    test_images = _as_5d(test_images).to(DEVICE)

if classifier in ['ScatterNet', 'DualSSN']: # When scattering is used

    # fold T into C on both real & scattering inputs
    test_images = fold_T_axis(test_images) # Merges the image version into the channel dimension
    mock_test = torch.zeros_like(test_images)
    
    # Define cache paths
    if USE_GLOBAL_NORMALISATION:
        test_cache = f"./.cache/scattering_coefficients/test_scat_{galaxy_classes}_{versions}_{GLOBAL_NORM_MODE}_{CROP_MODE}.pt"
    else:
        test_cache = f"./.cache/scattering_coefficients/test_scat_{galaxy_classes}_{versions}_{CROP_MODE}.pt"

    # Load or compute test scattering coefficients
    if os.path.exists(test_cache) and USE_CACHE:
        print(f"✓ Loading test scattering coefficients from cache: {test_cache}")
        test_scat_coeffs = torch.load(test_cache)
    else:
        test_scat_coeffs = compute_scattering_coeffs(test_images, scattering, batch_size=128, device="cpu")
        os.makedirs(os.path.dirname(test_cache), exist_ok=True)
        torch.save(test_scat_coeffs, test_cache)
        
    if test_scat_coeffs.dim() == 5:
        # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
        test_scat_coeffs = test_scat_coeffs.flatten(start_dim=1, end_dim=2)

    if NORMALISESCS or NORMALISESCSTOPM:
        # Now we need to normalise the scattering coefficients globally, and thus compute the scattering coeffs for all data
        trainval_images = fold_T_axis(train_val_images)
        if USE_GLOBAL_NORMALISATION:
            trainval_cache = f"./.cache/scattering_coefficients/trainval_scat_{galaxy_classes}_{versions}_{GLOBAL_NORM_MODE}_{CROP_MODE}.pt"
        else:
            trainval_cache = f"./.cache/scattering_coefficients/trainval_scat_{galaxy_classes}_{versions}_{CROP_MODE}.pt"
        if os.path.exists(trainval_cache) and USE_CACHE:
            print(f"✓ Loading trainval scattering coefficients from cache: {trainval_cache}")
            trainval_scat_coeffs = torch.load(trainval_cache)
        else:
            trainval_scat_coeffs = compute_scattering_coeffs(trainval_images, scattering, batch_size=128, device="cpu")
            os.makedirs(os.path.dirname(trainval_cache), exist_ok=True)
            torch.save(trainval_scat_coeffs, trainval_cache)
        if trainval_scat_coeffs.dim() == 5:
            trainval_scat_coeffs = trainval_scat_coeffs.flatten(start_dim=1, end_dim=2)
        all_scat = torch.cat([trainval_scat_coeffs, test_scat_coeffs], dim=0)
        if NORMALISESCSTOPM:
            all_scat = normalise_images(all_scat, -1, 1)
        else:
            all_scat = normalise_images(all_scat, 0, 1)
        trainval_scat_coeffs, test_scat_coeffs = all_scat[:len(trainval_scat_coeffs)], all_scat[len(trainval_scat_coeffs):]
    if classifier in ['ScatterNet']:
        test_dataset = TensorDataset(mock_test, test_scat_coeffs, test_labels)
    else: # if classifier == 'DualSSN':
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

experiments_run = set()

FIRSTTIME = True  # Set to True to print model summaries only once
for fold in folds:
    torch.cuda.empty_cache()
    runname = f"{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{ver_key}_f{fold}"

    print(f"\n▶ Experiment: g_classes={galaxy_classes}, lr={lr}, reg={reg}, ls={label_smoothing}, "
    f"J={J}, L={L}, crop={crop_size}, down={downsample_size}, ver={versions}, "
    f"lo={percentile_lo}, hi={percentile_hi}, classifier={classifier}, "
    f"global_norm={USE_GLOBAL_NORMALISATION}, norm_mode={GLOBAL_NORM_MODE}, "
    f"CROP_MODE={CROP_MODE} ◀")

    log_path = f"{LOGS_DIR}/log_{classifier}_{runname}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    if USE_CACHE:
        fold_subset_sizes = [round_to_1(max(2, int(len(train_val_images) * p))) for p in dataset_portions]
        all_configs_exist = all(
            config_already_exists(classifier, galaxy_classes, lr, reg, percentile_lo, percentile_hi,
                                cs, ds, ver_key, fold, ss, exp,
                                metrics_dir=METRICS_DIR, use_global_norm=USE_GLOBAL_NORMALISATION,
                                global_norm_mode=GLOBAL_NORM_MODE)
            for ss in fold_subset_sizes
            for exp in range(num_experiments)
        )
        
        if all_configs_exist:
            print(f"⏭️  Skipping fold {fold} - all configurations already trained")
            dataset_sizes[fold] = fold_subset_sizes
            continue
    
    _out = load_galaxies(
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

            USE_CACHE=USE_CACHE,
            DEBUG=DEBUG,
            crop_mode=CROP_MODE,
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
        
    train_labels, valid_labels = (relabel(train_labels, galaxy_classes, multilabel=MULTILABEL),
                                  relabel(valid_labels, galaxy_classes, multilabel=MULTILABEL))

    # ——— Data sanity checks ———
    if DEBUG:
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
                train_mask = _as_index_labels(train_labels) == i
                valid_mask = _as_index_labels(valid_labels) == i
                test_mask  = _as_index_labels(test_labels)  == i

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

    dataset_sizes[fold] = [max(2, int(len(train_images) * p)) for p in dataset_portions]
    
    
    ##########################################################
    ############ NORMALISE AND PACKAGE THE INPUT #############
    ##########################################################

    if classifier in ['CNN', 'DualCSN', 'DualSSN']:
        train_images = _as_5d(train_images).to(DEVICE)
        valid_images = _as_5d(valid_images).to(DEVICE)

    if classifier in ['ScatterNet', 'DualSSN']:
    
        # fold T into C on both real & scattering inputs
        train_images = fold_T_axis(train_images) # Merges the image version into the channel dimension
        valid_images = fold_T_axis(valid_images)
        mock_train = torch.zeros_like(train_images)
        mock_valid = torch.zeros_like(valid_images)

        # Define cache paths
        train_cache = f"./.cache/scattering_coefficients/train_scat_{galaxy_classes}_{versions}_{fold}_{FILTERED}_{CROP_MODE}.pt"
        valid_cache = f"./.cache/scattering_coefficients/valid_scat_{galaxy_classes}_{versions}_{fold}_{FILTERED}_{CROP_MODE}.pt"

        # Load or compute train scattering coefficients
        if os.path.exists(train_cache) and USE_CACHE:
            print(f"✓ Loading train scattering coefficients from cache: {train_cache}")
            train_scat_coeffs = torch.load(train_cache)
        else:
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
            valid_scat_coeffs = compute_scattering_coeffs(valid_images, scattering, batch_size=128, device="cpu")
            os.makedirs(os.path.dirname(valid_cache), exist_ok=True)
            torch.save(valid_scat_coeffs, valid_cache)

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

        if classifier in ['ScatterNet']:
            train_dataset = TensorDataset(mock_train, train_scat_coeffs, train_labels)
            valid_dataset = TensorDataset(mock_valid, valid_scat_coeffs, valid_labels)
        else: # if classifier == 'DualSSN':
            print("Shape of train images:", train_images.shape)
            print("Shape of train scattering coefficients:", train_scat_coeffs.shape)
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
    
    if fold == folds[0] and SHOWIMGS and downsample_size[-1] == 128 and FIRSTTIME:     
        # Plot some example training images with their labels
        imgs = train_images.detach().cpu().numpy()
        lbls = (_as_index_labels(train_labels) + min(galaxy_classes)).detach().cpu().numpy()
        plot_images_by_class(
            imgs,
            labels=lbls,
            classes=get_classes(),
            num_images=5,
            save_path=f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_example_train_data.pdf"
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
                save_path=f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_histogram_{ver_key}.pdf"
            )
            
            plot_background_histogram(
                train_images_cls1.cpu(),
                train_images_cls2.cpu(),
                img_shape=(1, 128, 128),
                title="Background histograms",
                save_path=f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_background_hist.pdf"
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
            plt.savefig(f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_sum_histogram_{ver_key}.pdf")
            plt.close()

            for cls in galaxy_classes:
                cls_idx = cls - min(galaxy_classes)

                sel_train = torch.where(_as_index_labels(train_labels) == cls_idx)[0][:36]
                titles_train = [train_fns[i] for i in sel_train.tolist()] if train_fns is not None else None
                sel_valid = torch.where(_as_index_labels(valid_labels) == cls_idx)[0][:36]
                titles_valid = [valid_fns[i] for i in sel_valid.tolist()] if valid_fns is not None else None
                sel_test = torch.where(_as_index_labels(test_labels) == cls_idx)[0][:36]
                titles_test = [test_fns[i] for i in sel_test.tolist()] if test_fns is not None else None

                imgs4plot = _ensure_4d(train_images)[sel_train].cpu()
                plot_image_grid(
                    imgs4plot, num_images=36, titles=titles_train,
                    save_path=f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_train_grid_{ver_key}.pdf"
                )

                imgs4plot = _ensure_4d(valid_images)[sel_valid].cpu()
                plot_image_grid(
                    imgs4plot, num_images=36, titles=titles_valid,
                    save_path=f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_valid_grid_{ver_key}.pdf"
                )
                
                imgs4plot = _ensure_4d(test_images)[sel_test].cpu()
                plot_image_grid(
                    imgs4plot, num_images=36, titles=titles_test,
                    save_path=f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_{cls}_test_grid_{ver_key}.pdf"
                )

                # summed-intensity histogram helper unchanged...
                tag_to_desc = { d["tag"]: d["description"] for d in get_classes() }

                plot_intensity_histogram(
                    train_images_cls1.cpu(),
                    train_images_cls2.cpu(),
                    label1=tag_to_desc[get_classes()[galaxy_classes[0]]['tag']],
                    label2=tag_to_desc[get_classes()[galaxy_classes[1]]['tag']],
                    save_path=f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_summed_intensity_histogram.pdf"
                )
       
                
    ###############################################
    ############# DEFINE MODEL ####################
    ###############################################

    # Directly create the model for the selected classifier
    if classifier == "CNN":
        model = CNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)
    elif classifier == "ScatterNet":
        model = ScatterNet(input_dim=int(np.prod(scatdim)), num_classes=num_classes).to(DEVICE)
    elif classifier == "DualCSN":
        model = DualCNNSqueezeNet(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)
    elif classifier == "DualSSN":
        model = DualScatterSqueezeNet(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    # Print model summary once
    if FIRSTTIME:
        print(f"Summary for {classifier}:")
        if classifier == 'ScatterNet':  # Only scattering input
            summary(model, input_size=scatdim, device=DEVICE)
        elif classifier == "DualSSN":  # Both image and scattering input
            summary(model, input_size=[valid_images.shape[1:], scatdim])
        else:  # Only image input
            summary(model, input_size=tuple(valid_images.shape[1:]), device=DEVICE)
        FIRSTTIME = False
        
        
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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=reg)

    for subset_size in dataset_sizes[fold]:
        if subset_size <= 0:
            print(f"Skipping invalid subset size: {subset_size}")
            continue
        
        all_experiments_cached = all(
            config_already_exists(classifier, galaxy_classes, lr, reg, percentile_lo, percentile_hi,
                                cs, ds, ver_key, fold, subset_size, exp,
                                metrics_dir=METRICS_DIR, use_global_norm=USE_GLOBAL_NORMALISATION,
                                global_norm_mode=GLOBAL_NORM_MODE)
            for exp in range(num_experiments)
        )
        if USE_CACHE and all_experiments_cached:
            print(f"⏭️  Skipping subset_size {subset_size} - all experiments already cached")
            continue
        
        if fold not in training_times:
            training_times[fold] = {}
        if subset_size not in training_times[fold]:
            training_times[fold][subset_size] = []

        for experiment in range(num_experiments):
            
            if USE_CACHE and config_already_exists(classifier, galaxy_classes, lr, reg, percentile_lo, percentile_hi, cs, ds, ver_key, fold, subset_size, experiment,
                                                    metrics_dir=METRICS_DIR, use_global_norm=USE_GLOBAL_NORMALISATION, global_norm_mode=GLOBAL_NORM_MODE):
                print(f"⏭️  Skipping: fold={fold}, subset_size={subset_size}, experiment={experiment} - configuration already trained")
                continue
            
            all_logits = []

            base = f"cl{classifier}_ss{round_to_1(subset_size)}_f{fold}_lr{lr}_reg{reg}_ls{label_smoothing}_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{ver_key}"
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
                if DEBUG:
                    train_correct = 0  # Add this to track training accuracy per epoch

                for images, scat, _rest in subset_train_loader:
                    labels = _rest
                    images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                                            
                    # ADD MixUp here:
                    if np.random.rand() > 0.5 and MIXUP:  # Apply MixUp 50% of the time
                        images, scat, labels_a, labels_b, lam = mixup_data(images, scat, labels, alpha=0.4)
                        
                        if classifier == 'ScatterNet':
                            logits = model(scat)
                        elif classifier == 'DualSSN':
                            logits = model(images, scat)
                        else:
                            logits = model(images)

                        logits = _collapse_logits(logits, num_classes, MULTILABEL)
                        labels_a = labels_a.float() if MULTILABEL else labels_a.long()
                        labels_b = labels_b.float() if MULTILABEL else labels_b.long()
                        
                        # CHANGE: Use MixUp criterion
                        loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    else:
                        # Normal forward pass without MixUp
                        if classifier == 'ScatterNet':
                            logits = model(scat)
                        elif classifier == 'DualSSN':
                            logits = model(images, scat)
                        else:
                            logits = model(images)

                        logits = _collapse_logits(logits, num_classes, MULTILABEL)
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
                    if DEBUG:
                        with torch.no_grad():
                            if MULTILABEL:
                                preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                                trues = labels.cpu().numpy()
                                train_correct += (preds == trues).all(axis=1).sum()
                            else:
                                preds = logits.argmax(dim=1)
                                train_correct += (preds == labels).sum().item()
                                
                if DEBUG:
                    train_epoch_acc = train_correct / train_total_images if train_total_images > 0 else 0.0
                    train_acc_key = f"{base}_{experiment}_train_acc"
                    history[train_acc_key].append(train_epoch_acc)  # Store per-epoch train accuracy

                # Calculate epoch training metrics
                train_average_loss = train_total_loss / train_total_images
                train_loss_key = f"{base}_{experiment}_train_loss"
                history[train_loss_key].append(train_average_loss)


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
                        if classifier == 'ScatterNet':
                            logits = model(scat)
                        elif classifier == 'DualSSN':
                            logits = model(images, scat)
                        else:
                            logits = model(images)

                        logits = _collapse_logits(logits, num_classes, MULTILABEL)
                        labels = labels.float() if MULTILABEL else labels.long()
                        if not MULTILABEL:
                            assert labels.dtype == torch.long, f"labels dtype {labels.dtype} must be long"
                        loss = criterion(logits, labels)
                        mn, mx = int(labels.min()), int(labels.max())
                        assert 0 <= mn and mx < num_classes, f"label range [{mn},{mx}] not in [0,{num_classes-1}]"

                        val_total_loss += float(loss.item() * images.size(0))
                        val_total_images += float(images.size(0))

                        # Track validation accuracy per epoch
                        if DEBUG:
                            if MULTILABEL:
                                preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                                trues = labels.cpu().numpy()
                                val_correct += (preds == trues).all(axis=1).sum()
                            else:
                                preds = logits.argmax(dim=1)
                                val_correct += (preds == labels).sum().item()
                    
                if DEBUG:
                    val_epoch_acc = val_correct / val_total_images if val_total_images > 0 else 0.0
                    val_acc_key = f"{base}_{experiment}_val_acc"
                    history[val_acc_key].append(val_epoch_acc)  # Store per-epoch val accuracy

                val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                val_loss_key = f"{base}_{experiment}_val_loss"
                history[val_loss_key].append(val_average_loss)
                
                # ADD TEST EVALUATION DURING TRAINING
                if DEBUG:
                    test_total_loss = 0
                    test_total_images = 0
                    test_correct = 0  # Initialize test_correct counter
                    with torch.inference_mode():
                        for i, (images, scat, _rest) in enumerate(test_loader): 
                            if images is None or len(images) == 0:
                                continue
                            labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            
                            if classifier == 'ScatterNet':
                                logits = model(scat)
                            elif classifier == 'DualSSN':
                                logits = model(images, scat)
                            else:
                                logits = model(images)

                            logits = _collapse_logits(logits, num_classes, MULTILABEL)
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
                
                    # Stop if overfitting is detected
                    train_val_gap = train_epoch_acc - val_epoch_acc
                    train_test_gap = train_epoch_acc - test_epoch_acc

                    # Stop if overfitting becomes severe
                    if train_epoch_acc > 0.95 and (train_val_gap > 0.25 or train_test_gap > 0.25):  # 25% gap threshold
                        print(f"⚠️  Severe overfitting detected at epoch {epoch+1}")
                        print(f"   Train-Val gap: {train_val_gap:.4f}, Train-Test gap: {train_test_gap:.4f}")
                        print(f"   Stopping early to prevent further overfitting")
                        break
                
                if ES:
                    early_stopping(val_average_loss, model, f'{MODELS_DIR}/{base}_best_model.pth')
                    if early_stopping.early_stop:
                        break

            if ES: # Load the best model saved by early stopping
                checkpoint_path = f'{MODELS_DIR}/{base}_best_model.pth'
                if os.path.exists(checkpoint_path):
                    try:
                        model.load_state_dict(torch.load(checkpoint_path))
                    except RuntimeError as e:
                        print(f"Warning: Could not load checkpoint due to architecture mismatch: {e}")
                        print(f"Continuing with current model state (last trained epoch)")
                else:
                    print(f"No checkpoint found at {checkpoint_path}, using current model state")
                
            if DEBUG:
                plot_training_history(history, base, experiment, save_dir=FIGURES_DIR)

            model.eval()                  
            with torch.inference_mode():
                train_correct = 0
                train_total = 0
                for images, scat, _rest in subset_train_loader: # Evaluate on training data
                    labels = _rest
                    images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                    
                    if classifier == 'ScatterNet':
                        logits = model(scat)
                    elif classifier == 'DualSSN':
                        logits = model(images, scat)
                    else:
                        logits = model(images)
                        
                    logits = _collapse_logits(logits, num_classes, MULTILABEL)
                    
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
                    if classifier == 'ScatterNet':
                        logits = model(scat)
                    elif classifier == 'DualSSN':
                        logits = model(images, scat)
                    else:
                        logits = model(images)
                        
                    logits = _collapse_logits(logits, num_classes, MULTILABEL)
                    if MULTILABEL:
                        preds = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                        trues = labels.cpu().numpy()
                        val_correct += (preds == trues).all(axis=1).sum()
                    else:
                        preds = logits.argmax(dim=1)
                        val_correct += (preds == labels).sum().item()
                    
                    val_total += images.size(0)
            
            final_val_acc = val_correct / val_total if val_total > 0 else 0.0
            if DEBUG:
                print(f"Final validation accuracy for experiment {experiment}: {final_val_acc:.4f}")
            metrics.setdefault(f'{base}_val_acc', []).append(final_val_acc)
                
            with torch.inference_mode(): 
                all_pred_probs[base] = []
                all_pred_labels[base] = []
                all_true_labels[base] = []
                mis_images = []
                mis_trues  = []
                mis_preds  = []
                
                for images, scat, _rest in test_loader: # Evaluate on test data
                    labels = _rest
                    images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                    if classifier == 'ScatterNet':
                        logits = model(scat)
                    elif classifier == 'DualSSN':
                        logits = model(images, scat)
                    else:
                        logits = model(images)
                        
                    all_logits.append(logits.cpu().numpy())
                    logits = _collapse_logits(logits, num_classes, MULTILABEL)
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
                print(f"Fold {fold}, Experiment {experiment}, Subset Size {subset_size}, Classifier {classifier}, "
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

                    out_path = f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_{dataset_sizes[folds[0]][-1]}_{percentile_lo}_{percentile_hi}_misclassified_{ver_key}.pdf"
                    fig.savefig(out_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

            end_time = time.time()
            elapsed_time = end_time - start_time
            training_times.setdefault(fold, {}).setdefault(subset_size, []).append(elapsed_time)

            if fold == folds[-1] and experiment == num_experiments - 1:
                with open(log_path, 'a') as file:
                    file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

            # Save model and compute cluster metrics if we trained something
            if all_logits:
                model_save_path = f'{MODELS_DIR}/{base}_model.pth'
                torch.save(model.state_dict(), model_save_path)
                
                all_logits_concat = np.concatenate(all_logits, axis=0)
                cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(all_logits_concat, n_clusters=num_classes)
                
                print(f"✓ Saved model for fold {fold}, subset_size {subset_size}: {model_save_path}")
                experiments_run.add((fold, subset_size, experiment)) # Mark that we ran at least one experiment for this fold and subset size
                
                # Write to log file (keep this)
                with open(log_path, 'a') as file:
                    file.write(f"Results for fold {fold}, Classifier {classifier}, lr={lr}, reg={reg}, percentile_lo={percentile_lo}, percentile_hi={percentile_hi}, crop_size={crop_size}, downsample_size={downsample_size}, STRETCH={STRETCH}, FILTERED={FILTERED} \n")
                    file.write(f"Cluster Error: {cluster_error} \n")
                    file.write(f"Cluster Distance: {cluster_distance} \n")
                    file.write(f"Cluster Standard Deviation: {cluster_std_dev} \n")
            else:
                print(f"⏭️  No new experiments trained for fold {fold}, subset_size {subset_size} - skipping model save and cluster metrics")
            
###############################################
########### SAVE ALL EXPERIMENT METRICS #######
###############################################

# Save metrics for each experiment that was actually run
print("\n" + "="*80)
print("SAVING METRICS FOR ALL EXPERIMENTS")
print("="*80)


for fold in folds:
    for subset_size in dataset_sizes.get(fold, []):
        for experiment in range(num_experiments):
            print("Experiment: ", experiment)
            # Skip if this experiment was not run (was cached)
            if (fold, subset_size, experiment) not in experiments_run:
                print(f"⏭️  Skipping metrics save for fold={fold}, subset_size={subset_size}, exp={experiment} - experiment was not run")
                continue
                
            # Build the configuration key
            base = f"cl{classifier}_ss{round_to_1(subset_size)}_f{fold}_lr{lr}_reg{reg}_ls{label_smoothing}_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{ver_key}"
            
            # Extract ONLY the metrics for this specific configuration
            config_metrics = {
                "accuracy": metrics.get(f"{base}_accuracy", []),
                "precision": metrics.get(f"{base}_precision", []),
                "recall": metrics.get(f"{base}_recall", []),
                "f1_score": metrics.get(f"{base}_f1_score", []),
                "train_acc": metrics.get(f"{base}_train_acc", []),
                "val_acc": metrics.get(f"{base}_val_acc", []),
            }
            
            # Extract history for this configuration
            config_history = {}
            for key in history.keys():
                if key.startswith(f"{base}_{experiment}"):
                    config_history[key] = history[key]
            
            # Extract labels and predictions for this configuration
            config_true_labels = {base: all_true_labels.get(base, [])}
            config_pred_labels = {base: all_pred_labels.get(base, [])}
            config_pred_probs = {base: all_pred_probs.get(base, [])}
            
            # Extract training time for this configuration
            config_training_times = {
                fold: {subset_size: training_times.get(fold, {}).get(subset_size, [])}
            }
            
            # Save to file
            if USE_GLOBAL_NORMALISATION:
                metrics_save_path = f'{METRICS_DIR}/{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{ver_key}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_{GLOBAL_NORM_MODE}_metrics_data.pkl'
            else:
                metrics_save_path = f'{METRICS_DIR}/{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{ver_key}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_metrics_data.pkl'
            
            with open(metrics_save_path, 'wb') as f:
                pickle.dump({
                    "metrics": config_metrics,
                    "history": config_history,
                    "all_true_labels": config_true_labels,
                    "all_pred_labels": config_pred_labels,
                    "all_pred_probs": config_pred_probs,
                    "training_times": config_training_times,
                    "classifier_name": classifier,
                    "model_architecture": str(model) if 'model' in locals() else f"Model {classifier} not initialized",
                    "cluster_error": cluster_error if 'cluster_error' in locals() else None,
                    "cluster_distance": cluster_distance if 'cluster_distance' in locals() else None,
                    "cluster_std_dev": cluster_std_dev if 'cluster_std_dev' in locals() else None,
                }, f)
            print(f"✓ Saved metrics PKL for fold={fold}, subset_size={subset_size}, exp={experiment}: {os.path.basename(metrics_save_path)}")
            
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
                    f"{FIGURES_DIR}/{galaxy_classes}_{classifier}_"
                    f"{round_to_1(dataset_sizes[folds[0]][-1])}_{metric_name}_histogram.pdf"
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
            
# Calculate and print grand average over ALL folds and experiments
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []

for fold in folds:
    for subset_size in dataset_sizes[fold]:
        base = f"cl{classifier}_ss{round_to_1(subset_size)}_f{fold}_lr{lr}_reg{reg}_ls{label_smoothing}_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{ver_key}"
        
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
  
if DEBUG:          
    check_overfitting(metrics, history, classifier, dataset_sizes, folds, lr, reg, label_smoothing, crop_size, downsample_size, percentile_lo, percentile_hi, ver_key)
            
    # Create aggregate plot showing all experiments
    if num_experiments > 1:
        plot_dir = FIGURES_DIR
        os.makedirs(plot_dir, exist_ok=True)
        
        print("Creating aggregate plots for all experiments...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 grid
        
        for fold, experiment in list(itertools.product(folds, range(num_experiments))):
            base = f"cl{classifier}_f{fold}_lr{lr}_reg{reg}_ls{label_smoothing}_cs{crop_size}_ds{downsample_size}_pl{percentile_lo}_ph{percentile_hi}_ver{ver_key}"
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

# Print summary
if experiments_run:
    print(f"\n✓ Saved metrics for {len(experiments_run)} newly trained experiments")
else:
    print(f"\n⏭️  No new experiments were trained - all were cached")