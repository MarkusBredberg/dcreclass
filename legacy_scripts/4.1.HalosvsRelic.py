import numpy as np
import torch, math, time, random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from utils.data_loader import load_galaxies, get_classes,  get_synthetic, augment_images
from utils.classifiers import RustigeClassifier, TinyCNN, MLPClassifier, SCNN, CNNSqueezeNet, ScatterResNet, DANNClassifier, BinaryClassifier, ScatterSqueezeNet, ScatterSqueezeNet2, DualCNNSqueezeNet
#from utils.Cloud_Net import CloudNet
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram
from torchvision.utils import make_grid, save_image
from kymatio.torch import Scattering2D
import pandas as pd
import pickle
from tqdm import tqdm
from torch.optim import AdamW
import itertools
import matplotlib 
import sys, os

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

print("Running script 4.1 Halo vs Relic", SEED)

#Update the classifiers so that they are okay for just two classes
#EDIT: Allow for max_num_galaxies as None

###############################################
################ CONFIGURATION ################
###############################################

classes = get_classes()
galaxy_classes = [52, 53]  # Classes to classify
max_num_galaxies = 1000000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions = [0.1, 1]  # Portions of complete dataset for the accuracy vs dataset size
J, L, order = 2, 12, 2  # Scatter transform parameters
classifier = ["TinyCNN", # Very Simple CNN
              "Rustige", # Simple CNN from Rustige et al. 2023, https://github.com/floriangriese/wGAN-supported-augmentation/blob/main/src/Classifiers/SimpleClassifiers/Classifiers.py
              "SCNN", # Simple CNN similar to Rustige's
              "CNNSqueezeNet", # SCNN with Squeeze-and-Excitation blocks
              "DualCNNSqueezeNet", # Dual CNN with Squeeze-and-Excitation blocks
              "CloudNet", # From https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection/tree/master
              "DANN", # Domain-Adversarial Neural Network
              "ScatterNet", "ScatterSqueezeNet", "ScatterSqueezeNet2",
              "Binary", "ScatterResNet"][-4]
gen_model_names = ['DDPM'] #['ST', 'DDPM', 'wGAN', 'GAN', 'Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP'] # Specify the generative model_name
num_epochs_cuda = 200
num_epochs_cpu = 100
learning_rates = [1e-3]  # Learning rates
regularization_params = [1e-3]  # Regularisation parameters
label_smoothing = 0  # Label smoothing for the classifier
num_experiments = 10
folds = [5] # 0-4 for 5-fold cross validation, 5 for only one training
lambda_values = [0]  # Ratio between generated images and original images per class. 8 is reserfved for TRAINONGENERATED

STRETCH = True  # Stretch the images with mathematical morphology
ES, patience = True, 10  # Use early stopping
SCHEDULER = False  # Use a learning rate scheduler
SHOWIMGS = True  # Show some generated images for each class (Tool for control)

NORMALISEIMGS = False  # Normalise images to [0, 1]
NORMALISEIMGSTOPM = False  # Normalise images to [-1, 1] 
NORMALISESCS = False  # Normalise scattering coefficients to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients to [-1, 1]

BALANCE = False  # Balance the dataset by undersampling the majority class
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights
TRAINONGENERATED = False  # Use generated data as testdata
FILTERED = True  # Remove in training, validation and test data for the classifier
FILTERGEN = False  # Remove generated images that are too similar to other generated images

HISTOGRAMMATCHING = False  # Histogram matching for generated images
USE_MEMMAP = False  # Use memmap for scattering coefficients
SIGMACLIPPONDDPM = False  # Apply sigma clipping to DDPM data

# -------------------------- GAN CONFIGURATION --------------------------
gan_epoch = 5000           # e.g., epoch number to load from
gan_gen_loss = 'MSE'       # e.g., generator loss value (as used in filename)
gan_disc_loss = 'BCE'      # e.g., discriminator loss value (as used in filename)
gan_latent_dim = 128
gan_sample_size = 100
lr_gen = 1e-3
lr_disc = 1e-4
gan_adam_beta = (0.5, 0.999)
gan_weight_decay = 0
gan_label_smoothing = 0.7
gan_lambda_div = 0
gan_type = ['Simple', 'Advanced'][0]
gan_data_version = 'clean' if FILTERED else 'full'  # 'full' or 'clean'


# ---------------------------- VAE CONFIGURATION -----------------------------
VAE_train_size = 1101128
forbidden_classes = 12  # Generated bent sources look awful
#########################################################################################################################
    
if any (cls in galaxy_classes for cls in [10, 11, 12, 13]):
    crop_size = (128, 128)  # Crop size for the images
    downsample_size = (128, 128)  # Downsample size for the images
    batch_size = 128
elif galaxy_classes[0] in list(range(40, 49)):
    crop_size = (1600, 1600)  # Crop size for the images
    downsample_size = (128, 128)  # Downsample size for the images
    batch_size = 16
elif galaxy_classes[0] in list(range(50, 60)):
    crop_size = (1, 2000, 2000)  # Crop size for the images
    downsample_size = (1, 2000, 2000)  # Downsample size for the images
    batch_size = 16 

img_shape = downsample_size

if TRAINONGENERATED:
    lambda_values = [8]  # To identify and distinguish TRAINONGENERATED from other runs
    print("Using generated data for testing.")

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")
    

if set(galaxy_classes) & {18} or set(galaxy_classes) & {19}:
    galaxy_classes = [20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Include all digits if 18 or 19 is in target_classes
else:
    galaxy_classes = galaxy_classes
num_classes = len(galaxy_classes)

EXTRAVARS = False  # Use extra features (redshift, mass, size) for the classifier. Will automatically be true if test_meta is not None.

###############################################
########### DATA STORING FUNCTIONS ###############
###############################################


def initialize_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    
    
def update_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate):
    subset_size_str = str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(accuracy)
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(precision)
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(recall)
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(f1)
    
    
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

    with torch.no_grad():
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
    
    print("crop_size: ", crop_size)
    _out  = load_galaxies(galaxy_class=galaxy_classes, 
                fold=max(folds), #Any fold other than 5 gives me the test data for the five fold cross validation
                crop_size=crop_size,
                downsample_size=downsample_size,
                sample_size=max_num_galaxies, 
                REMOVEOUTLIERS=FILTERED,
                BALANCE=BALANCE,           # Reduce the larger classes to the size of the smallest class
                STRETCH=STRETCH,
                AUGMENT=True,
                train=False)
    

    if len(_out) == 4:
        train_images, train_labels, test_images, test_labels = _out
        train_data = test_data = None
    elif len(_out) == 6:
        print("Received 6 outputs from load_galaxies, including extra features.")
        train_images, train_labels, test_images, test_labels, train_data, test_data = _out

        EXTRAVARS = True  # Use extra features (redshift, mass, size) for the classifier

        class MetaWrapper(nn.Module):
            def __init__(self, base_model: nn.Module, meta_dim: int, hidden: int = 64, num_classes: int = 2):
                super().__init__()
                self.base = base_model
                # assume `base_model` ends by outputting a feature vector of size `feat_dim`
                # you may need to probe this or hard‐code it based on model summaries
                feat_dim = self._find_feat_dim(meta_dim, num_classes)
                self.meta_fc = nn.Sequential(
                    nn.Linear(meta_dim, hidden),
                    nn.ReLU(),
                )
                self.classifier = nn.Linear(feat_dim + hidden, num_classes)

            def _find_feat_dim(self, meta_dim, num_classes):
                # dummy pass to get the feature‐vector size
                self.base.eval()
                with torch.no_grad():
                    # pass a single dummy through the “body” of base_model up to before its classifier
                    # here we assume you can do `base_model.feature_extractor(...)`
                    # if you don’t have such a method, you’ll need to monkey‐patch your models or
                    # rework this; for now we’ll hard‐code:
                    return 128

            def forward(self, img, scat, meta):
                # route through whatever your base expects
                if hasattr(self.base, "forward_scatter"):
                    img_feat = self.base.forward_scatter(scat)
                elif hasattr(self.base, "forward_both"):
                    img_feat = self.base.forward_both(img, scat)
                else:
                    img_feat = self.base(img)
                meta_feat = self.meta_fc(meta)
                return self.classifier(torch.cat([img_feat, meta_feat], dim=1))

    else:
        raise ValueError(f"load_galaxies returned {len(_out)} values, expected 4 or 6")
    
    # ——— Data sanity checks ———
    print("Shape of test_images: ", np.shape(test_images))
            
    check_tensor("Test images", test_images)
    check_tensor("Train images", train_images)
    
    # Check the tensor for each class
    for cls in galaxy_classes:
        cls_mask = (train_labels == cls)
        cls_images = train_images[cls_mask]
        check_tensor(f"Train images for class {cls}", cls_images)

    import hashlib

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
    
    test_labels = test_labels - test_labels.min() 
    perm = torch.randperm(test_images.size(0))
    test_images = test_images[perm]
    test_labels = test_labels[perm]
    if EXTRAVARS:
        test_data = test_data[perm]


    # Print the distribution of raw test labels
    unique_labels, counts = torch.unique(test_labels, return_counts=True)
    print("Test labels distribution (raw):", dict(zip(unique_labels.tolist(), counts.tolist())))
    
    if NORMALISEIMGS or NORMALISEIMGSTOPM:
        if NORMALISEIMGSTOPM:
            test_images = normalise_images(test_images, -1, 1)
        else:
            test_images = normalise_images(test_images, 0, 1)

    # Prepare input data
    # Produce an empty tensor to occupy the not used component of the datasets. 
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
        if EXTRAVARS:
            test_dataset = TensorDataset(test_images, test_scat_coeffs, test_data, test_labels)
        else:
            test_dataset = TensorDataset(test_images, test_scat_coeffs, test_labels)
    else: 
        mock_tensor = torch.zeros_like(test_images)
        test_dataset = TensorDataset(test_images, mock_tensor, test_labels)
                            
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

    print(f"Test dataset size: {len(test_dataset)}")

    ###############################################
    ########### LOOP OVER DATA FOLD ###############
    ###############################################            

    FIRSTTIME = True  # Set to True to print model summaries only once
    param_combinations = list(itertools.product(folds, learning_rates, regularization_params, lambda_values))
    for fold, lr, reg, lambda_generate in param_combinations:
        torch.cuda.empty_cache()
        print(f"\n Training with fold: {fold}, lr: {lr}, reg: {reg}, lambda: {lambda_generate}")
        runname = f'{galaxy_classes}_{gen_model_name}_lr{lr}_reg{reg}'
        log_path = f"./classifier/log_{runname}.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        #file = open(log_path, 'w')
            
        # ——— Data loading for training/validation ———
        if TRAINONGENERATED:
            # only synthetic for training; reserve real for validation
            train_images = torch.empty((0, *img_shape), device=DEVICE)
            train_labels = torch.empty((0,), dtype=torch.long, device=DEVICE)
            _out = load_galaxies(
                galaxy_class=galaxy_classes,
                fold=fold,
                crop_size=crop_size,
                downsample_size=downsample_size,
                sample_size=max_num_galaxies,
                REMOVEOUTLIERS=FILTERED,
                BALANCE=BALANCE,           # Reduce the larger classes to the size of the smallest class
                AUGMENT=False,   
                train=True
            )
            
            if len(_out) == 4:
                _, _, valid_images, valid_labels = _out
                valid_data = None
            elif len(_out) == 6:
                _, _, valid_images, valid_labels, _, valid_data = _out

            valid_labels = valid_labels - valid_labels.min()
            perm = torch.randperm(valid_images.size(0))
            valid_images, valid_labels = valid_images[perm], valid_labels[perm]
            if EXTRAVARS:
                valid_data = valid_data[perm]
        else:
            # real train + valid
            _out = load_galaxies(
                galaxy_class=galaxy_classes,
                fold=fold,
                crop_size=crop_size,
                downsample_size=downsample_size,
                sample_size=max_num_galaxies,
                REMOVEOUTLIERS=FILTERED,
                BALANCE=BALANCE,           # Reduce the larger classes to the size of the smallest class
                STRETCH=STRETCH,
                AUGMENT=True,
                train=True
            )

            if len(_out) == 4:
                train_images, train_labels, valid_images, valid_labels = _out
                train_data = test_data = None
            elif len(_out) == 6:
                train_images, train_labels, valid_images, valid_labels, train_data, valid_data = _out

            assert train_images.size(0) == train_labels.size(0), \
                f"train_images ({len(train_images)}) and train_labels ({len(train_labels)}) must match!"

            valid_labels = valid_labels - valid_labels.min()
            perm = torch.randperm(train_images.size(0))
            train_images, train_labels = train_images[perm], train_labels[perm]
            perm = torch.randperm(valid_images.size(0))
            valid_images, valid_labels = valid_images[perm], valid_labels[perm]
            if EXTRAVARS:
                train_data = train_data[perm]
                valid_data = valid_data[perm]
            dataset_sizes[fold] = [int(len(train_images) * p) for p in dataset_portions]
            
            
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
            print("Old training data size: ", train_images.size())
            print('Number of images to generate for each class: ', num_generate)
            
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
            for cls in galaxy_classes:
                cls_mask = (train_labels == cls)
                cls_images = train_images[cls_mask]
                check_tensor(f"Generated images for class {cls} with model {gen_model_name}", cls_images)
        train_labels = train_labels - train_labels.min()
        
        unique_labels, counts = torch.unique(train_labels, return_counts=True)
        print("Train labels distribution (raw):", dict(zip(unique_labels.tolist(), counts.tolist())))
        
        if TRAINONGENERATED:
            if EXTRAVARS:
                print("Cannot use extra features with TRAINONGENERATED, setting EXTRAVARS to False")
                EXTRAVARS = False
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

            
        unique_labels, counts = torch.unique(train_labels, return_counts=True)
        print("Train labels distribution (raw) after possible filtering and augmentation:", dict(zip(unique_labels.tolist(), counts.tolist())))

        if dataset_sizes == {}:
            dataset_sizes[fold] = [int(len(train_images) * p) for p in dataset_portions]

                
        ##########################################################
        ############ NORMALISE AND PACKAGE THE INPUT #############
        ##########################################################

        if classifier in ['Rustige', 'CNNSqueezeNet', 'SCNN', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
            if lambda_generate not in [0, 8]:
                # 1. concatenate images
                img_splits = [
                    pristine_train_images.to(DEVICE),
                    generated_images.to(DEVICE),
                    valid_images.to(DEVICE)
                ]
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
                img_splits = [
                    train_images.to(DEVICE),
                    valid_images.to(DEVICE)
                ]
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
                        
        
        
        # Normalise images if requested
        if NORMALISEIMGS or NORMALISEIMGSTOPM:
            check_tensor("Train images before normalisation", train_images)
            if NORMALISEIMGSTOPM:
                train_images = normalise_images(train_images, -1, 1)
            else:
                train_images = normalise_images(train_images, 0, 1)
            check_tensor("Train images after normalisation", train_images)
            
        
        # ── SANITY-CHECK PLOTS ON FIRST FOLD ONLY ──
        if fold == folds[0] and SHOWIMGS and downsample_size == (1, 128, 128):
            
            if len(galaxy_classes) == 2:
                # Plot histograms for the two classes
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
                    save_path=f"./classifier/{classifier}_{gen_model_name}_{galaxy_classes[0]}_{galaxy_classes[1]}_f{fold}_histogram.png"
                )

                plot_background_histogram(
                    train_images_cls1.cpu(),        # shape (936, 1, 128, 128)
                    train_images_cls2.cpu(),        # shape (720, 1, 128, 128)
                    img_shape=(1, 128, 128),
                    title="Background histograms",
                    save_path=f"./classifier/{classifier}_{gen_model_name}_{galaxy_classes[0]}_{galaxy_classes[1]}_f{fold}_background_hist.png"
                )

                for cls in galaxy_classes:
                    orig_imgs = train_images[train_labels == (cls - min(galaxy_classes))][:36]
                    test_imgs = test_images[test_labels == (cls - min(galaxy_classes))][:36]
                                    
                    plot_image_grid(
                        orig_imgs.cpu(),
                        num_images=36,
                        save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_f{fold}_real_grid.png"
                    )
                    plot_image_grid(
                        test_imgs.cpu(),
                        num_images=36,
                        save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_f{fold}_test_grid.png"
                    )

                    if lambda_generate not in [0, 8]:
                        gen_imgs = generated_by_class[cls][:36]

                        plot_image_grid(
                            gen_imgs,
                            num_images=36,
                            save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_f{fold}_generated_grid.png"
                        )
                        plot_histograms(
                            gen_imgs,
                            orig_imgs.cpu(),
                            title1="Generated Images",
                            title2="Train Images",
                            save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_f{fold}_histogram.png"
                        )
                        plot_background_histogram(
                            orig_imgs,
                            gen_imgs,
                            img_shape=(1, 128, 128),
                            title="Background histograms",
                            save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_f{fold}_background_hist.png")
        
        if USE_CLASS_WEIGHTS:
            unique, counts = np.unique(train_labels.cpu().numpy(), return_counts=True)
            total_count = sum(counts)
            class_weights = {i: total_count / count for i, count in zip(unique, counts)}
            weights = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)],
                                dtype=torch.float, device=DEVICE)
            missing_classes = [cls for cls in unique if cls not in class_weights]
            if missing_classes:
                print(f"Warning: Missing classes in dataset: {missing_classes}")
                class_weights.update({int(cls): 1.0 for cls in missing_classes})
                
            unique_valid, counts_valid = np.unique(valid_labels.cpu().numpy(), return_counts=True)
            unique_test, counts_test = np.unique(test_labels.cpu().numpy(), return_counts=True)
            class_counts_valid = dict(zip(map(int, unique_valid), map(int, counts_valid)))
            class_counts_test = dict(zip(map(int, unique_test), map(int, counts_test)))
        else:
            weights = None

        if fold in [0, 5] and SHOWIMGS:
            imgs = train_images.detach().cpu().numpy()
            lbls = (train_labels + min(galaxy_classes)).detach().cpu().numpy()
            plot_images_by_class(imgs, labels=lbls, num_images=5, save_path=f"./classifier/{classifier}_{gen_model_name}_{galaxy_classes}_example_train_data.pdf")
        
        # Prepare input data
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

                # fold T into channels on both real & scattering inputs
                print("Shape of train_images before folding T axis: ", train_images.shape)

                train_images = fold_T_axis(train_images)
                valid_images = fold_T_axis(valid_images)
                mock_train = torch.zeros_like(train_images)
                mock_valid = torch.zeros_like(valid_images)

                print("Shape of train_images after folding T axis: ", train_images.shape)

                train_cache = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
                valid_cache = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
                train_scat_coeffs = compute_scattering_coeffs(train_images, scattering, batch_size=128, device="cpu")
                valid_scat_coeffs = compute_scattering_coeffs(valid_images, scattering, batch_size=128, device="cpu")
                print("Shape of train_scat_coeffs directly after computation: ", train_scat_coeffs.shape)

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
                print("scatdim ", scatdim)

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
            mock_train = torch.zeros_like(train_images)
            mock_valid = torch.zeros_like(valid_images)
            train_dataset = TensorDataset(train_images, mock_train, train_labels)
            valid_dataset = TensorDataset(valid_images, mock_valid, valid_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

        if SHOWIMGS and lambda_generate not in [0, 8]: 
            if classifier in ['TinyCNN', 'SCNN', 'CNNSqueezeNet', 'Rustige', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
                #save_images_tensorboard(generated_images[:36], save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_generated.png", nrow=6)
                plot_histograms(pristine_train_images, valid_images, title1="Train images", title2="Valid images", imgs3=generated_images, imgs4=test_images, title3='Generated images', title4='Test images', save_path=f"./classifier/{classifier}_{gen_model_name}_{galaxy_classes}_histograms.png")

        
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
        elif classifier == 'CloudNet':
            cloud_model = CloudNet()  
            cloud_model.load_weights(os.path.join(sys.path[0],
                'Cloud-Net_trained_on_38-Cloud_training_patches.h5'))
            cloud_model.trainable = True   # now you can fine-tune it on your radio maps
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
                    summary(model_details["model"], input_size=img_shape, device=DEVICE)
            FIRSTTIME = False
            # Compute scattering coefficients for one sample (ensure the model is in the right mode)


        ###############################################
        ############### TRAINING LOOP #################
        ###############################################
        
        
        if False:
            import torch.nn.functional as F

            class FocalLoss(nn.Module):
                def __init__(self, weight=None, gamma=2, reduction='mean'):
                    super().__init__()
                    self.weight = weight
                    self.gamma = gamma
                    self.reduction = reduction

                def forward(self, inputs, targets):
                    logp = F.log_softmax(inputs, dim=1)
                    ce   = F.nll_loss(logp, targets, weight=self.weight, reduction='none')
                    p    = torch.exp(-ce)
                    loss = (1 - p)**self.gamma * ce
                    if self.reduction == 'mean':
                        return loss.mean()
                    if self.reduction == 'sum':
                        return loss.sum()
                    return loss

            print(f"Using FocalLoss(gamma=2) with weight={weights}")
            criterion = FocalLoss(weight=weights if weights is not None else None, gamma=2, reduction='mean')

        
        if weights is not None:
            print(f"Using class weights: {weights}")
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        else:
            print("No class weighting")
            if len(galaxy_classes) == 2:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            

        optimizer = AdamW(models[classifier_name]["model"].parameters(), lr=lr, weight_decay=reg)
        if SCHEDULER:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10*lr, 
                                    steps_per_epoch=len(train_loader), epochs=num_epochs)

        for classifier_name, model_details in models.items():
            print(f"Training {classifier_name} model...")
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
                    initialize_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                    initialize_labels(all_true_labels, all_pred_labels, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)

                    start_time = time.time()
                    model.apply(reset_weights)

                    subset_indices = torch.randperm(len(train_dataset))[:subset_size].tolist() # Randomly select indices to include generated samples
                    subset_train_dataset = Subset(train_dataset, subset_indices)
                    subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

                    early_stopping = EarlyStopping(patience=patience, verbose=False) if ES else None

                    for epoch in tqdm(range(num_epochs), desc=f'Training {classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}'):
                        model.train()
                        total_loss = 0
                        total_images = 0

                        for images, scat, _rest in subset_train_loader:
                            if EXTRAVARS:
                                meta, labels = _rest
                                meta = meta.to(DEVICE)  # Send metadata to device
                            else:
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
                                    outputs = model(scat)
                                elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                                    outputs = model(images, scat)
                                else:
                                    outputs = model(images)
                                loss = criterion(outputs, labels)
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

                        with torch.no_grad(): # Validate on validation data
                            for i, (images, scat, _rest) in enumerate(valid_loader):
                                if images is None or len(images) == 0:
                                    print(f"Empty batch at index {i}. Skipping...")
                                    continue
                                if EXTRAVARS:
                                    meta, labels = _rest
                                    meta = meta.to(DEVICE)  # Send metadata to device
                                else:
                                    labels = _rest
                                images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                                if classifier == "DANN":
                                    class_logits, _ = model(images, alpha=1.0)
                                    loss = criterion(class_logits, labels)
                                else: 
                                    if classifier in ["ScatterNet", "ScatterResNet"]:
                                        outputs = model(scat)
                                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                                        outputs = model(images, scat)
                                    else:
                                        outputs = model(images)
                                    loss = criterion(outputs, labels)
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
                    with torch.no_grad(): # Evaluate on test data
                        key = f"{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        all_pred_probs[key] = []
                        all_pred_labels[key] = []
                        all_true_labels[key] = []

                        for images, scat, _rest in test_loader:
                            if EXTRAVARS:
                                meta, labels = _rest
                                meta = meta.to(DEVICE)  # Send metadata to device
                            else:
                                labels = _rest
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            if classifier == "DANN":
                                class_logits, _ = model(images, alpha=1.0)
                                pred_probs = torch.softmax(class_logits, dim=1).cpu().numpy()
                            else:
                                if classifier in ["ScatterNet", "ScatterResNet"]:
                                    pred_probs = model(scat).cpu().detach().numpy()

                                elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                                    pred_probs = torch.softmax(model(images, scat), dim=1).cpu().detach().numpy()
                                else:
                                    pred_probs = model(images).cpu().detach().numpy()
                            true_labels = labels.cpu().numpy()
                            #true_labels = torch.argmax(labels, dim=1).cpu().numpy()
                            pred_labels = np.argmax(pred_probs, axis=1)
                            all_pred_probs[key].extend(pred_probs)
                            all_pred_labels[key].extend(pred_labels)
                            all_true_labels[key].extend(true_labels)

                        accuracy = accuracy_score(all_true_labels[key], all_pred_labels[key])
                        precision = precision_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                        recall = recall_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                        f1 = f1_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)

                        update_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate)

                end_time = time.time()
                elapsed_time = end_time - start_time
                training_times[subset_size][fold].append(elapsed_time)
                if subset_size > 20000:
                    with open(log_path, 'w') as file:
                        file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

            generated_features = []
            with torch.no_grad():
                for images, scat, _ in test_loader:
                    if EXTRAVARS:
                        meta, labels = _rest
                        meta = meta.to(DEVICE)  # Send metadata to device
                    else:
                        labels = _rest
                    images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
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
                file.write(f"Cluster results for fold {fold}, classifier {classifier}, lr {lr}, reg {reg}, lambda_generate {lambda_generate} \n")
                file.write(f"Cluster Error: {cluster_error} \n")
                file.write(f"Cluster Distance: {cluster_distance} \n")
                file.write(f"Cluster Standard Deviation: {cluster_std_dev} \n")

            model_save_path = f'./classifier/trained_models/{gen_model_name}_model.pth'
            torch.save(model.state_dict(), model_save_path)
            
    directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

    for fold, lr, reg, lambda_generate in param_combinations:
        for subset_size in dataset_sizes[fold]:
            for experiment in range(num_experiments):
                metrics_save_path = f'{directory}{classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl'
                with open(metrics_save_path, 'wb') as f:
                    pickle.dump({
                        "models": models,
                        "history": history,
                        "metrics": metrics,
                        "metric_colors": metric_colors,
                        "all_true_labels": all_true_labels,
                        "all_pred_labels": all_pred_labels,
                        "training_times": training_times,
                        "all_pred_probs": all_pred_probs
                    }, f)
                print(f"Metrics and related data saved to {metrics_save_path}") 