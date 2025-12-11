import os, re, time, random, pickle, hashlib, itertools, torch
from utils.data_loader import load_galaxies, get_classes, get_synthetic, augment_images
from utils.classifiers import RustigeClassifier, TinyCNN, MLPClassifier, SCNN, CNNSqueezeNet, ScatterResNet, DANNClassifier, BinaryClassifier, ScatterSqueezeNet, ScatterSqueezeNet2, DualCNNSqueezeNet
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    # make cuDNN deterministic (may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

print("Running script 4.1 with dl1 Latest version with seed", SEED)

#EDIT: Allow for max_num_galaxies as None

###############################################
################ CONFIGURATION ################
###############################################
galaxy_classes    = [50, 51]
max_num_galaxies  = 1000000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions  = [1]
J, L, order       = 2, 12, 2
classifier = ["TinyCNN", # Very Simple CNN
              "Rustige", # Simple CNN from Rustige et al. 2023, https://github.com/floriangriese/wGAN-supported-augmentation/blob/main/src/Classifiers/SimpleClassifiers/Classifiers.py
              "SCNN", # Simple CNN similar to Rustige's
              "CNNSqueezeNet", # SCNN with Squeeze-and-Excitation blocks
              "DualCNNSqueezeNet", # Dual CNN with Squeeze-and-Excitation blocks
              "DANN", # Domain-Adversarial Neural Network
              "ScatterNet", "ScatterSqueezeNet", "ScatterSqueezeNet2",
              "Binary", "ScatterResNet"][-4]
gen_model_names = ['DDPM'] #['ST', 'DDPM', 'wGAN', 'GAN', 'Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP'] # Specify the generative model_name
num_epochs_cuda = 100
num_epochs_cpu = 100
learning_rates = [1e-3]
regularization_params = [1e-3]  
label_smoothing = 0.2  
num_experiments = 10
folds = [0] # 0-4 for 5-fold cross validation, 5 for only one training
#folds = [0]
lambda_values = [0]  # Ratio between generated images and original images per class. 8 is reserfved for TRAINONGENERATED
percentile_lo = 30 # Percentile stretch lower bound
percentile_hi = 99  # Percentile stretch upper bound
versions = 'RAW' # any mix of loadable and runtime-tapered planes. 'rt50' or 'rt100' for tapering. Square brackets for stacking

PREFER_PROCESSED = False  
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
TRAINONGENERATED = False  # Train only on generated data, validate on real data
ES, patience = True, 5  # Use early stopping
SCHEDULER = True  # Use a learning rate scheduler
SHOWIMGS = True  # Show some generated images for each class (Tool for control)
USE_MEMMAP = False  # Use memory-mapped files for large datasets (slower but less RAM)


#PREFER_PROCESSED = True if versions != 'RAW' else False  # Prefer processed images if available (constant n_beams)


#####################################################################################
########################### AUTOMATIC CONFIGURATION #################################
#####################################################################################

os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)
    
if any (cls in galaxy_classes for cls in [10, 11, 12, 13]):
    crop_size = (128, 128)  # Crop size for the images
    downsample_size = (128, 128)  # Downsample size for the images
    batch_size = 128
elif galaxy_classes[0] in list(range(40, 49)):
    crop_size = (1600, 1600)  # Crop size for the images
    downsample_size = (128, 128)  # Downsample size for the images
    batch_size = 16
elif galaxy_classes[0] in list(range(50, 60)):
    crop_size = (512, 512)  # Crop size for the images
    downsample_size = (128, 128)  # Downsample size for the images
    batch_size = 16 

img_shape = downsample_size

if TRAINONGENERATED:
    lambda_values = [8]  # To identify and distinguish TRAINONGENERATED from other runs
    print("Using generated data for testing.")

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

if set(galaxy_classes) & {18} or set(galaxy_classes) & {19}:
    galaxy_classes = [20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Include all digits if 18 or 19 is in target_classes
else:
    galaxy_classes = galaxy_classes
num_classes = len(galaxy_classes)

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
EXTRAVARS = False  # Use extra features (redshift, mass, size) for the classifier. Will automatically be true if test_meta is not None.
GLOBAL_NORM_MODE = "none" if not USE_GLOBAL_NORMALISATION else GLOBAL_NORM_MODE
    
if EXTRAVARS:
    try:
        from utils.classifiers import MetaWrapper  # only used if EXTRAVARS=True
    except Exception:
        MetaWrapper = None

def _verkey(v):
    if isinstance(v, (list, tuple)):
        return "+".join(map(str, v))
    return str(v)
ver_key = _verkey(versions)

########################################################################
##################### HELPER FUNCTIONS #################################
########################################################################

def as_index_labels(y: torch.Tensor) -> torch.Tensor:
    # returns 1-D class indices regardless of encoding
    return y.argmax(dim=1) if y.ndim > 1 else y

def _as_5d(x):
    return x if x.dim() == 5 else x.unsqueeze(1)  # [B,1,H,W] -> [B,1,1,H,W]

def permute_like(x, perm):
    if x is None: return None
    idx = perm.cpu().tolist()
    if isinstance(x, torch.Tensor): return x[perm]
    if isinstance(x, np.ndarray):   return x[idx]
    if isinstance(x, (list, tuple)): return [x[i] for i in idx]
    return x

base_cls = min(galaxy_classes)
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

def collapse_logits(logits, num_classes, multilabel):
    if logits.ndim == 4:
        logits = F.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
    elif logits.ndim == 3:
        logits = logits.mean(dim=-1)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if not multilabel and logits.shape[1] == 1 and num_classes == 2:
        logits = torch.cat([-logits, logits], dim=1)
    return logits

def shuffle_with_filenames(images, labels, filenames=None):
    perm = torch.randperm(images.size(0))
    images, labels = images[perm], labels[perm]
    if filenames is not None:
        filenames = [filenames[i] for i in perm.tolist()]
    return images, labels, filenames


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
    metrics.setdefault(f"{key_base}_accuracy", []).append(accuracy)
    metrics.setdefault(f"{key_base}_precision", []).append(precision)
    metrics.setdefault(f"{key_base}_recall", []).append(recall)
    metrics.setdefault(f"{key_base}_f1_score", []).append(f1)

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

scattering = Scattering2D(J=J, L=L, shape=img_shape[-2:], max_order=order)      
for gen_model_name in gen_model_names:

    with torch.inference_mode():
        dummy = torch.zeros((1, *img_shape)).cpu()
        scat_dummy = scattering(dummy)
        if scat_dummy.dim()==5:
            # fold T into channels
            scat_dummy = scat_dummy.flatten(1,2)
        # now scat_dummy.shape == [1, C, H, W]
        scatshape = tuple(scat_dummy.shape[1:])   # (C, H, W)
    hidden_dim1 = 256
    hidden_dim2 = 128
    vae_latent_dim = 64
    
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
        train_images, train_labels, test_images, test_labels = _out
        train_fns = test_fns = None
        perm_test = torch.randperm(test_images.size(0))
        test_images, test_labels = test_images[perm_test], test_labels[perm_test]
    elif len(_out) == 6:
        train_images, train_labels, test_images, test_labels, train_fns, test_fns = _out
        test_images, test_labels, test_fns = shuffle_with_filenames(test_images, test_labels, test_fns)

    else:
        raise ValueError(f"load_galaxies returned {len(_out)} values, expected 4 or 6")
    
    # Print all the source names in the TEST set:
    #if test_fns is not None:
    #    print("\nTEST filenames:")
    #    for fn in test_fns:
    #        print(" -", fn)
    #    print("")
    #else:
    #    print("No test filenames available to print.\n")

    train_labels = relabel(train_labels)  # Will be used for the sanity check below
    test_labels  = relabel(test_labels)   # [0,1,...] aligning with galaxy_classes
    if not MULTILABEL:
        unique_labels, counts = torch.unique(as_index_labels(test_labels), return_counts=True)

    # ——— Data sanity checks ———
    for i, cls in enumerate(galaxy_classes):
        if MULTILABEL:
            train_mask = train_labels[:, i] > 0.5
            test_mask  = test_labels[:,  i] > 0.5
        else:
            train_mask = as_index_labels(train_labels) == i
            test_mask  = as_index_labels(test_labels)  == i

        check_tensor(f"Trainval images for class {cls} (idx={i})", train_images[train_mask])
        check_tensor(f"Test images for class {cls} (idx={i})",  test_images[test_mask])

    def img_hash(img: torch.Tensor) -> str:
        # ensure CPU & contiguous
        arr = img.cpu().contiguous().numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()

    # after loading train_images, test_images:
    train_hss = {img_hash(img) for img in train_images}
    test_hashes   = {img_hash(img) for img in test_images}

    common = train_hss & test_hashes
    assert not common, f"Overlap detected: {len(common)} images appear in both train and test validation!"
    # ————————————————————————

    # Produce an empty tensor to occupy the not used component of the datasets. 
    mock_tensor = torch.zeros_like(test_images)
    if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2']:
        test_images = fold_T_axis(test_images)
        test_scat_coeffs = compute_scattering_coeffs(test_images, scattering, batch_size=128, device='cpu')
        if test_scat_coeffs.dim() == 5:
            # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
            test_scat_coeffs = test_scat_coeffs.flatten(start_dim=1, end_dim=2)
                        
        if NORMALISESCS or NORMALISESCSTOPM:
            if NORMALISESCSTOPM:
                test_scat_coeffs = normalise_images(test_scat_coeffs, -1, 1)
            else:
                test_scat_coeffs = normalise_images(test_scat_coeffs, 0, 1)

        if classifier in ['ScatterNet', 'ScatterResNet']:
            mock_test = torch.zeros_like(test_images)  # images are ignored
            test_dataset = TensorDataset(mock_test, test_scat_coeffs, test_labels)
        elif classifier in ['ScatterSqueezeNet', 'ScatterSqueezeNet2']:
            if EXTRAVARS:
                print("Not implemented yet for ScatterSqueezeNet with extra variables")
                exit(1)
            else:
                test_dataset = TensorDataset(test_images, test_scat_coeffs, test_labels)

    else:
        if test_images.dim() == 5:
            test_images = fold_T_axis(test_images)  # [B,T,1,H,W] -> [B,T,H,W]
        assert test_images.dim() == 4, f"test_images should be [B,C,H,W], got {tuple(test_images.shape)}"
        test_dataset = TensorDataset(test_images, mock_tensor, test_labels)

                            
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)

    ###############################################
    ########### LOOP OVER DATA FOLD ###############
    ###############################################            

    FIRSTTIME = True  # Set to True to print model summaries only once
    param_combinations = list(itertools.product(folds, learning_rates, regularization_params, lambda_values))
    for fold, lr, reg, lambda_generate in param_combinations:
        torch.cuda.empty_cache()
        runname = f"{galaxy_classes}_{gen_model_name}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size[0]}x{crop_size[1]}"

        print(f"\n▶ Experiment: g_classes={galaxy_classes}, lr={lr}, reg={reg}, ls={label_smoothing}, "
        f"J={J}, L={L}, crop={crop_size}, down={downsample_size}, ver={versions}, "
        f"lo={percentile_lo}, hi={percentile_hi}, classifier={classifier}, "
        f"global_norm={USE_GLOBAL_NORMALISATION}, norm_mode={GLOBAL_NORM_MODE}, "
        f"PREFER_PROCESSED={PREFER_PROCESSED} ◀\n")

        log_path = f"./classifier/log_{runname}.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        #file = open(log_path, 'w')
            
        # ——— Data loading for training/validation ———
        if TRAINONGENERATED:
            # only synthetic for training; reserve real for validation
            train_images = torch.empty((0, *img_shape), device=DEVICE)
            train_labels = torch.empty((0,), dtype=torch.long, device=DEVICE)
            _out = _loader(
                galaxy_classes=galaxy_classes,
                versions=versions,
                fold=fold,
                crop_size=crop_size,
                downsample_size=downsample_size,
                sample_size=max_num_galaxies,
                REMOVEOUTLIERS=FILTERED,
                BALANCE=BALANCE,
                AUGMENT=False,
                NORMALISE=NORMALISEIMGS,
                NORMALISETOPM=NORMALISEIMGSTOPM,
                USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
                GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
                train=True,
                PRINTFILENAMES=PRINTFILENAMES,
                PREFER_PROCESSED=PREFER_PROCESSED
            )

            if len(_out) == 4:
                _, _, valid_images, valid_labels = _out
                train_fns = valid_fns = None
            elif len(_out) == 6:
                _, _, valid_images, valid_labels, train_fns, valid_fns = _out
            else:
                raise ValueError(f"load_galaxies returned {len(_out)} values, expected 4 or 6")

            # Relabel and shuffle validation data
            valid_labels = relabel(valid_labels)
            perm_valid = torch.randperm(valid_images.size(0))
            valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]
            if PRINTFILENAMES and (valid_fns is not None):
                valid_fns = permute_like(valid_fns, perm_valid)

        else:
            _out  = _loader(galaxy_classes=galaxy_classes,
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
                        train=True)

            if len(_out) == 4:
                train_images, train_labels, valid_images, valid_labels = _out
                train_fns = test_fns = None

                perm_train = torch.randperm(train_images.size(0))
                train_images, train_labels = train_images[perm_train], train_labels[perm_train]
                
                perm_valid = torch.randperm(valid_images.size(0))
                valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]

            elif len(_out) == 6:
                if PRINTFILENAMES:
                    train_images, train_labels, valid_images, valid_labels, train_fns, valid_fns = _out
                else:
                    print("Not implemented extradata yet")
                
                if EXTRAVARS:
                    train_data = train_fns
                    valid_data = valid_fns
            dataset_sizes[fold] = [max(2, int(len(train_images) * p)) for p in dataset_portions]
            
            train_labels = relabel(train_labels)  # [0,1,2,...] for classes [50,51,...]
            valid_labels = relabel(valid_labels)  # [0,1,2,...] for classes [50,51,...]
            
        ##########################################################
        ############# READ IN GENERATED DATA #####################
        ##########################################################
                    
        # Generate more training data if requested
        if lambda_generate not in [0]:
            batch_size_generate = 500 
            if train_images.numel() > 0:
                num_generate = int(lambda_generate / len(galaxy_classes) * len(train_images))
            else:
                num_generate = int(lambda_generate * 125)

            
            for cls in galaxy_classes:
                # use your shared helper to do all generation + filtering
                gen_imgs_list, gen_lbls_list = get_synthetic(
                    num_generate=num_generate,
                    gen_model_name=gen_model_name,
                    cls=cls,
                    galaxy_classes=galaxy_classes,
                    img_shape=img_shape,
                    FILTERGEN=FILTERGEN,
                    CLIPDDPM=True,
                    model_kwargs={
                        'gan_type': gan_type,
                        'gan_latent_dim': gan_latent_dim,
                        'gan_sample_size': gan_sample_size,
                        'lr_gen': lr_gen,
                        'lr_disc': lr_disc,
                        'gan_gen_loss': gan_gen_loss,
                        'gan_disc_loss': gan_disc_loss,
                        'gan_adam_beta': gan_adam_beta,
                        'gan_weight_decay': gan_weight_decay,
                        'gan_label_smoothing': gan_label_smoothing,
                        'gan_lambda_div': gan_lambda_div,
                        'gan_data_version': gan_data_version,
                        'gan_epoch': gan_epoch,
                        'NORMALISEIMGS': NORMALISEIMGS,
                        'NORMALISETOPM': NORMALISEIMGSTOPM,
                        'VAE_train_size': VAE_train_size,
                        'scatshape': scatshape,
                        'hidden_dim1': hidden_dim1,
                        'hidden_dim2': hidden_dim2,
                        'vae_latent_dim': vae_latent_dim
                    },
                    fold=fold,
                    device=DEVICE
                )
                # flatten and append
                if isinstance(gen_imgs_list, torch.Tensor):
                    generated_images = gen_imgs_list
                    print("Generated images shape: ", generated_images.shape)
                else:
                    generated_images = torch.cat(gen_imgs_list, dim=0)
                    
                if isinstance(gen_lbls_list, torch.Tensor):
                    generated_labels = gen_lbls_list
                else:
                    generated_labels = torch.cat(gen_lbls_list, dim=0)

                if train_images.numel() > 0:     
                    train_images = torch.cat([train_images, generated_images.to(train_images.device)])
                    train_labels = torch.cat([train_labels, generated_labels])
                else:
                    train_images = generated_images.to(DEVICE)
                    train_labels = generated_labels
                    
                # store per-class generated images for sanity checking later
                if 'generated_by_class' not in locals() and SHOWIMGS:
                    generated_by_class = {}
                generated_by_class[cls] = generated_images.cpu()

                # Append the filtered images and labels to your training data:
                if SHOWIMGS and lambda_generate not in [0, 8]:
                    pristine_train_images = train_images
                    pristine_train_labels = train_labels
                    
            # Check the tensor for generated images
            for i, cls in enumerate(galaxy_classes):
                cls_mask = (as_index_labels(train_labels) == i)
                cls_images = train_images[cls_mask]
                check_tensor(f"Generated images for class {cls} with model {gen_model_name}", cls_images)
            train_labels = relabel(train_labels)  # [0,1,2,...] for classes [50,51,...]
                    
        if TRAINONGENERATED:
            # For each class, select the correct slice of (images, labels), then concatenate all.
            class_imgs, class_lbls = [], []
            offset = min(galaxy_classes)
            for cls in galaxy_classes:
                idx = cls - offset
                imgs = train_images[train_labels == idx]
                lbls = train_labels[train_labels == idx]
                if  cls == 10:
                    imgs, lbls = imgs[:389], lbls[:389]
                elif cls == 11:
                    imgs, lbls = imgs[:816], lbls[:816]
                elif cls == 12:
                    imgs, lbls = imgs[:292], lbls[:292]
                elif cls == 13:
                    imgs, lbls = imgs[:242], lbls[:242]
                class_imgs.append(imgs)
                class_lbls.append(lbls)
            train_images = torch.cat(class_imgs, dim=0)
            train_labels = torch.cat(class_lbls, dim=0)
                    
            # Now apply augmentation to both images and labels
            train_images, train_labels = augment_images(train_images, train_labels)
            USE_CLASS_WEIGHTS = True
            
            # Shuffle the training data
            perm = torch.randperm(train_images.size(0))
            train_images = train_images[perm]
            train_labels = train_labels[perm]
            if EXTRAVARS:
                print("Cannot use extra features with TRAINONGENERATED, setting EXTRAVARS to False")
                EXTRAVARS = False
            
        if not MULTILABEL:
            unique_labels, counts = torch.unique(as_index_labels(train_labels), return_counts=True)
            #labels_1d = train_labels.argmax(dim=1) if train_labels.ndim > 1 else train_labels
            #unique_labels, counts = torch.unique(labels_1d, return_counts=True)

        if dataset_sizes == {}:
            dataset_sizes[fold] = [int(len(train_images) * p) for p in dataset_portions]

                
        ##########################################################
        ############ NORMALISE AND PACKAGE THE INPUT #############
        ##########################################################

        if classifier in ['Rustige', 'CNNSqueezeNet', 'SCNN', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
            if lambda_generate not in [0, 8]:
                # 1. concatenate images
                p = _as_5d(pristine_train_images).to(DEVICE)
                g = _as_5d(generated_images).to(DEVICE)
                v = _as_5d(valid_images).to(DEVICE)

                img_splits = [p, g, v]
                img_lengths = [len(t) for t in img_splits]
                all_images = torch.cat(img_splits, dim=0)

                # 2. concatenate labels in the same order
                lbl_splits = [
                    pristine_train_labels.to(DEVICE),
                    generated_labels.to(DEVICE),
                    valid_labels.to(DEVICE)
                ]
                all_labels = torch.cat(lbl_splits, dim=0)

                # 3. compute split boundaries once
                boundaries = [0] + list(torch.cumsum(torch.tensor(img_lengths), dim=0).numpy())

                # 4. slice images back into (pristine, generated, valid)
                chunked_imgs = [
                    all_images[boundaries[i]:boundaries[i+1]]
                    for i in range(len(img_lengths))
                ]
                # 5. slice labels back in exactly the same way
                chunked_lbls = [
                    all_labels[boundaries[i]:boundaries[i+1]]
                    for i in range(len(img_lengths))
                ]

                # 6. reassign train/valid splits
                pristine_train_images, generated_images, valid_images = chunked_imgs
                pristine_train_labels, generated_labels, valid_labels = chunked_lbls

                # 7. rebuild train_images and train_labels without pulling validation back in
                train_images = torch.cat([pristine_train_images, generated_images], dim=0)
                train_labels = torch.cat([pristine_train_labels, generated_labels], dim=0)

            else:
                # same idea if lambda_generate == 0 (i.e. only real train+valid)
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
        
        # ── SANITY-CHECK PLOTS ON FIRST FOLD ONLY ──
        if fold == folds[0] and SHOWIMGS and downsample_size == (1, 128, 128):

            def _mask_for_class(labels: torch.Tensor, cls_val: int) -> torch.Tensor:
                """
                Return a 1-D boolean mask of shape [B] selecting samples of class `cls_val`.
                In multi-label mode (RH/RR), use the corresponding column; otherwise use indices.
                """
                if MULTILABEL:
                    # Map class value (e.g. 52 or 53) to column 0 (RH) or 1 (RR)
                    col = 0 if cls_val == galaxy_classes[0] else 1
                    return labels[:, col] > 0.5
                else:
                    return as_index_labels(labels) == (cls_val - min(galaxy_classes))

            def _to_4d(x: torch.Tensor) -> torch.Tensor:
                # For plotting, make sure images are [N, C, H, W]
                return fold_T_axis(x) if x.dim() == 5 else x

            if len(galaxy_classes) == 2:
                # Build per-class views
                train_images_cls1 = _to_4d(train_images[_mask_for_class(train_labels, galaxy_classes[0])])
                train_images_cls2 = _to_4d(train_images[_mask_for_class(train_labels, galaxy_classes[1])])

                # Make sure the images are not tuples
                if isinstance(train_images_cls1, tuple): train_images_cls1 = train_images_cls1[0]
                if isinstance(train_images_cls2, tuple): train_images_cls2 = train_images_cls2[0]

                plot_histograms(
                    train_images_cls1.cpu(),
                    train_images_cls2.cpu(),
                    title1=f"Class {galaxy_classes[0]}",
                    title2=f"Class {galaxy_classes[1]}",
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_histogram.pdf"
                )

                plot_background_histogram(
                    train_images_cls1.cpu(),
                    train_images_cls2.cpu(),
                    img_shape=(1, 128, 128),
                    title="Background histograms",
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_background_hist.pdf"
                )

                for cls in galaxy_classes:
                    orig_imgs = _to_4d(train_images[_mask_for_class(train_labels, cls)])[:36]
                    test_imgs = _to_4d(test_images[_mask_for_class(test_labels, cls)])[:36]

                    plot_image_grid(
                        orig_imgs.cpu(),
                        num_images=36,
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_{cls}_train_grid.pdf"
                    )
                    plot_image_grid(
                        test_imgs.cpu(),
                        num_images=36,
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_{cls}_test_grid.pdf"
                    )

                    # summed-intensity histogram helper unchanged...
                    tag_to_desc = { d["tag"]: d["description"] for d in get_classes() }
                    def plot_intensity_histogram(tensor1, tensor2, label1, label2, save_path, bins=30):
                        vals1 = tensor1.sum(dim=tuple(range(1, tensor1.ndim))).cpu().numpy()
                        vals2 = tensor2.sum(dim=tuple(range(1, tensor2.ndim))).cpu().numpy()
                        plt.figure(figsize=(10,5))
                        plt.hist(vals1, bins=bins, alpha=0.6, label=label1, color='C1')
                        plt.hist(vals2, bins=bins, alpha=0.6, label=label2, color='C0')
                        plt.xlabel('Total Intensity'); plt.ylabel('Count')
                        plt.title(f'Total Intensity per Image: {label1} vs {label2}')
                        plt.legend(); plt.savefig(save_path); plt.close()

                    plot_intensity_histogram(
                        train_images_cls1.cpu(),
                        train_images_cls2.cpu(),
                        label1=tag_to_desc[get_classes()[galaxy_classes[0]]['tag']],
                        label2=tag_to_desc[get_classes()[galaxy_classes[1]]['tag']],
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_summed_intensity_histogram.pdf"
                    )

                    if lambda_generate not in [0, 8]:
                        gen_imgs = generated_by_class[cls][:36]
                        plot_image_grid(
                            gen_imgs,
                            num_images=36,
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_{cls}_generated_grid.pdf"
                        )
                        plot_histograms(
                            gen_imgs,
                            orig_imgs.cpu(),
                            title1="Generated Images",
                            title2="Train Images",
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_{cls}_histogram.pdf"
                        )
                        plot_background_histogram(
                            orig_imgs,
                            gen_imgs,
                            img_shape=(1, 128, 128),
                            title="Background histograms",
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_{cls}_background_hist.pdf"
                        )

        
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
                save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_example_train_data.pdf"
            )
        # Prepare input data
        mock_tensor = torch.zeros_like(train_images)
        valid_mock_tensor = torch.zeros_like(valid_images)
        if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2']:
            # Define cache paths (you can adjust these names as needed)
            train_cache_path = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{TRAINONGENERATED}.npy"
            valid_cache_path = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{TRAINONGENERATED}.npy"
            
            if USE_MEMMAP:
                # Use memmap-based caching (from script 1)
                train_cache_file, train_full_shape = cache_scattering_memmap(train_images, scattering, train_cache_path, batch_size=128, device="cpu")
                valid_cache_file, valid_full_shape = cache_scattering_memmap(valid_images, scattering, valid_cache_path, batch_size=128, device="cpu")
                # Create dataset using the memmap cache
                train_dataset = CachedScatterDataset(train_images, train_labels, train_cache_file, train_full_shape)
                valid_dataset = CachedScatterDataset(valid_images, valid_labels, valid_cache_file, valid_full_shape)
                scatdim = train_full_shape[1:]  # e.g., (C, H, W)
                
            else:
                # fold T into C on both real & scattering inputs
                train_images = fold_T_axis(train_images) # Merges the image version into the channel dimension
                valid_images = fold_T_axis(valid_images)
                mock_train = torch.zeros_like(train_images)
                mock_valid = torch.zeros_like(valid_images)


                train_cache = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
                valid_cache = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
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
                    if EXTRAVARS:
                        print(train_images.shape)
                        print(train_scat_coeffs.shape)
                        print(train_data.shape)
                        print(train_labels.shape)
                        train_dataset = TensorDataset(train_images, train_scat_coeffs, train_data, train_labels)
                        valid_dataset = TensorDataset(valid_images, valid_scat_coeffs, valid_data, valid_labels)
                    else:
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)
        
        print("Length of train dataset: ", len(train_dataset))

        if SHOWIMGS and lambda_generate not in [0, 8]: 
            if classifier in ['TinyCNN', 'SCNN', 'CNNSqueezeNet', 'Rustige', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
                #save_images_tensorboard(generated_images[:36], save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_generated.pdf", nrow=6)
                plot_histograms(pristine_train_images, valid_images, title1="Train images", title2="Valid images", imgs3=generated_images, imgs4=test_images, title3='Generated images', title4='Test images', save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_histograms.pdf")

        
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
        elif classifier == "DANN":
            models = {"DANN": {"model": DANNClassifier(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterNet":
            models = {"ScatterNet": {"model": MLPClassifier(input_dim=int(np.prod(scatdim)), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterResNet":
            models = {"ScatterResNet": {"model": ScatterResNet(scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterSqueezeNet":
            if EXTRAVARS:
                models = {"ScatterSqueezeNet": {"model": MetaWrapper(ScatterSqueezeNet(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes), meta_dim=test_meta.shape[1]).to(DEVICE)}}
            else:
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
        
        if MULTILABEL:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if weights is not None:
                print(f"Using class weights: {weights}")
                criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
            else:
                print("No class weighting")
                criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss() 

        optimizer = AdamW(models[classifier_name]["model"].parameters(), lr=lr, weight_decay=reg)

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
                    subset_train_loader = DataLoader(subset_train_dataset, batch_size=eff_bs, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=True)
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

                    for epoch in tqdm(range(num_epochs), desc=f'Training {classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_{classifier}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size}'):
                        model.train()
                        total_loss = 0
                        total_images = 0

                        for images, scat, _rest in subset_train_loader:
                            if EXTRAVARS:
                                meta, labels = _rest
                                meta = meta.to(DEVICE)  # Send metadata to device
                            else:
                                labels = _rest
                            images  = images.to(DEVICE, non_blocking=True)
                            scat    = scat.to(DEVICE, non_blocking=True)
                            labels  = labels.to(DEVICE, non_blocking=True)

                            optimizer.zero_grad()
                            if classifier == "DANN":
                                class_logits, domain_logits = model(images, alpha=1.0)
                                logits = class_logits
                            else:
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
                                if EXTRAVARS:
                                    meta, labels = _rest
                                    meta = meta.to(DEVICE)  # Send metadata to device
                                else:
                                    labels = _rest
                                #images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                                images  = images.to(DEVICE, non_blocking=True)
                                scat    = scat.to(DEVICE, non_blocking=True)
                                labels  = labels.to(DEVICE, non_blocking=True)

                                if classifier == "DANN":
                                    logits, _ = model(images, alpha=1.0)
                                else:
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

                        val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                        val_loss_key = f"{gen_model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        history[gen_model_name][val_loss_key].append(val_average_loss)
                        
                        if ES:
                            early_stopping(val_average_loss, model, f'./classifier/trained_models/best_model.pth')
                            if early_stopping.early_stop:
                                break 

                    if ES: # Load the best model saved by early stopping
                        model.load_state_dict(torch.load(f'./classifier/trained_models/best_model.pth'))

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
                            if EXTRAVARS:
                                meta, labels = _rest
                                meta = meta.to(DEVICE)  # Send metadata to device
                            else:
                                labels = _rest
                            #images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            images  = images.to(DEVICE, non_blocking=True)
                            scat    = scat.to(DEVICE, non_blocking=True)
                            labels  = labels.to(DEVICE, non_blocking=True)

                            if classifier == "DANN":
                                logits, _ = model(images, alpha=1.0)
                            else:
                                if classifier in ["ScatterNet", "ScatterResNet"]:
                                    logits = model(scat)
                                elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
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
                                probs = torch.softmax(logits, dim=1).cpu().numpy()
                                trues = labels.cpu().numpy()
                                preds = np.argmax(probs, axis=1)
                                all_pred_probs[key].extend(probs)
                                all_pred_labels[key].extend(preds)
                                all_true_labels[key].extend(trues)

                            # collect misclassified examples (final experiment only)
                            if SHOWIMGS and experiment == num_experiments - 1:
                                wrong_mask = (preds != trues).any(axis=1) if MULTILABEL else (preds != trues)
                                if np.any(wrong_mask):
                                    mask_t = torch.as_tensor(wrong_mask, dtype=torch.bool)
                                    mis_images.append(images.detach().cpu()[mask_t])
                                    mis_trues.append(trues[wrong_mask])
                                    mis_preds.append(preds[wrong_mask])


                        y_true = np.array(all_true_labels[key])
                        y_pred = np.array(all_pred_labels[key])

                        if MULTILABEL:
                            accuracy = accuracy_score(y_true, y_pred)
                            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                            recall    = recall_score(y_true, y_pred,    average='macro', zero_division=0)
                            f1        = f1_score(y_true, y_pred,        average='macro', zero_division=0)
                        else:
                            accuracy = accuracy_score(y_true, y_pred)
                            precision = precision_score(y_true, y_pred, average='macro', zero_division=0) if num_classes > 2 \
                                        else precision_score(y_true, y_pred, average='binary', zero_division=0)
                            recall    = recall_score(y_true, y_pred,    average='macro', zero_division=0) if num_classes > 2 \
                                        else recall_score(y_true, y_pred,    average='binary', zero_division=0)
                            f1        = f1_score(y_true, y_pred,        average='macro', zero_division=0) if num_classes > 2 \
                                        else f1_score(y_true, y_pred,        average='binary', zero_division=0)


                        update_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate, crop_size, downsample_size, ver_key)

                        
                        # Print accuracy and other metrics
                        print(f"Fold {fold}, Experiment {experiment}, Subset Size {subset_size}, Classifier {classifier_name}, "
                            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ")
    

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

                            out_path = f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[0]][-1]}_misclassified.pdf"
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
                print(f"Fold {fold}, Subset Size {subset_size}, Classifier {classifier_name}, AVERAGE over {num_experiments} experiments — Accuracy: {mean_acc:.4f}, Precision: {mean_prec:.4f}, Recall: {mean_rec:.4f}, F1 Score: {mean_f1:.4f}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                #training_times[subset_size][fold].append(elapsed_time)
                training_times.setdefault(fold, {}).setdefault(subset_size, []).append(elapsed_time)

                if fold == folds[-1] and experiment == num_experiments - 1:
                    with open(log_path, 'w') as file:
                        file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")
            
            generated_features = []
            with torch.inference_mode():
                for images, scat, _rest in test_loader:
                    if EXTRAVARS:
                        meta, labels = _rest
                    else:
                        labels = _rest
                    images  = images.to(DEVICE, non_blocking=True)
                    scat    = scat.to(DEVICE, non_blocking=True)
                    labels  = labels.to(DEVICE, non_blocking=True)
                    #images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)

                    if classifier == "DANN":
                        class_logits, _ = model(images, alpha=1.0)
                        outputs = class_logits.cpu().detach().numpy()
                    elif classifier in ["ScatterNet", "ScatterResNet"]:
                        outputs = model(scat).cpu().detach().numpy()
                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
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
    
    directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'
    for fold, lr, reg, lambda_generate in param_combinations:
        for subset_size in dataset_sizes[fold]:
            for experiment in range(num_experiments):
                                
                metrics_save_path = f'./classifier/4.1.runs/{runname}_sz{subset_size}_f{fold}_e{experiment}_lam{lambda_generate}_metrics_data.pkl'

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
                        f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_"
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
                summary_csv = f'{directory}{classifier}_{galaxy_classes}_{gen_model_name}_percentile_summary.csv'
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
                    