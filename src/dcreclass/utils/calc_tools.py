import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from lpips import LPIPS
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.ndimage import label
from sklearn.metrics import mean_squared_error
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from kymatio.torch import Scattering2D

# cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate

######################################
###### GENERAL HELPER FUNCTIONS ######
######################################

def round_to_1(x):
    return round(x, -int(math.floor(math.log10(abs(x)))))

#################################
###### METRICS FUNCTIONS ########
#################################

def recalculate_metrics_with_correct_positive_class(y_true, y_pred, pos_label=0):
    """
    Recalculate precision, recall, and F1 with the correct positive class.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    
    return acc, precision, recall, f1

def initialize_metrics(metrics, subset_size, fold, experiment, lr, reg):
    """Initialize metric storage dictionaries"""
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []

def update_metrics(metrics, subset_size, fold, experiment, lr, reg,
                   accuracy, precision, recall, f1,
                   history_val, all_true_labels, all_pred_labels, training_times, all_pred_probs):
    """Update metrics dictionaries"""
    subset_size_str = str(subset_size)
    metrics[f"accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(accuracy)
    metrics[f"precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(precision)
    metrics[f"recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(recall)
    metrics[f"f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(f1)
    metrics[f"history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(history_val)
    metrics[f"all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(all_true_labels)
    metrics[f"all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(all_pred_labels)
    metrics[f"training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(training_times)
    metrics[f"all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(all_pred_probs)

def cluster_metrics(features, n_clusters=13):
    """
    Calculate cluster error, cluster distance, and cluster standard deviation.
    
    Args:
        features (np.ndarray): The feature vectors of the generated images.
        n_clusters (int): The number of clusters for KMeans.
    
    Returns:
        cluster_error (float): The sum of squared distances of samples to their closest cluster center.
        cluster_distance (float): The average distance between cluster centers.
        cluster_std_dev (float): The standard deviation of distances to cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    # Cluster error: Sum of squared distances of samples to their closest cluster center
    cluster_error = kmeans.inertia_
    
    # Cluster distance: Average pairwise distance between cluster centers
    centers = kmeans.cluster_centers_
    cluster_distance = np.mean(pairwise_distances(centers))
    
    # Cluster standard deviation: Standard deviation of distances to cluster centers
    distances = np.min(pairwise_distances(features, centers), axis=1)
    cluster_std_dev = np.std(distances)
    
    return cluster_error, cluster_distance, cluster_std_dev

###############################################
###### IMAGE PROCESSING FUNCTIONS #############
###############################################

def check_tensor(name, tensor):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    if tensor.numel() == 0:
        print(f"Warning: {name} is empty, skipping stats.")
        return  # skip completely empty tensors
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaNs")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Infs")
    if tensor.is_floating_point():
        print(f"{name} stats: min={tensor.min().item():.3f}, "
              f"max={tensor.max().item():.3f}, "
              f"mean={tensor.mean().item():.3f}, "
              f"std={tensor.std().item():.3f}")
    else:
        vals, counts = torch.unique(tensor, return_counts=True)
        print(f"{name} unique values: {vals.tolist()}, counts: {counts.tolist()}")


def normalise_images(images, out_min=-1, out_max=1):
    global_min = images.min()
    global_max = images.max()
    images = (images - global_min) / (global_max - global_min)   # now in [0,1]
    return out_min + images * (out_max - out_min)                # now in [out_min,out_max]l


def compute_scattering_coeffs(images, scattering=Scattering2D(J=3, L=8, shape=(128, 128), max_order=2), batch_size=128, device="cpu"):
    scat_coeffs_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            if batch.ndim == 4 and batch.size(1) == 1:
                batch_scat = scattering(batch).detach()
            else:
                #scat_channels = [scattering(batch[:, i:i+1, :, :]).detach() for i in range(batch.shape[1])]
                # ensure the whole batch is contiguous
                batch = batch.contiguous()
                # and each channel‐slice, too
                scat_channels = [
                    scattering(batch[:, i:i+1, :, :].contiguous()).detach()
                    for i in range(batch.shape[1])
                ]

                batch_scat = torch.cat(scat_channels, dim=1)

            # Squeeze out the singleton dimension at index 1 if present.
            if batch_scat.shape[1] == 1:
                batch_scat = batch_scat.squeeze(1)  # becomes [B, C, H, W]
            if not batch_scat.is_contiguous():
                batch_scat = batch_scat.contiguous()
            scat_coeffs_list.append(batch_scat.cpu())
        scat_coeffs = torch.cat(scat_coeffs_list, dim=0)
    return scat_coeffs

def fold_T_axis(imgs: torch.Tensor) -> torch.Tensor:
    """
    If imgs is 5-D (N, T, C, H, W), reshape to (N, T*C, H, W);
    otherwise return unchanged.
    """
    if imgs.dim() == 5:
        N, T, C, H, W = imgs.shape
        return imgs.view(N, T * C, H, W)
    return imgs

#################################################
##### MODEL AND DATA LOADING FUNCTIONS ##########
#################################################


def custom_collate(batch):
    """
    Custom collate function to handle different batch structures.
    This function checks the structure of the first item in the batch
    and collates accordingly."""
    if not batch:
        return None
    first = batch[0]
    # 3-tuple: (img, scat, label)
    if isinstance(first, (tuple, list)) and len(first) == 3:
        imgs, scats, labels = zip(*batch)
        return (
            torch.utils.data.dataloader.default_collate(imgs),
            torch.utils.data.dataloader.default_collate(scats),
            torch.utils.data.dataloader.default_collate(labels),
        )
    # 4-tuple: (img, scat, meta, label)
    elif isinstance(first, (tuple, list)) and len(first) == 4:
        imgs, scats, metas, labels = zip(*batch)
        return (
            torch.utils.data.dataloader.default_collate(imgs),
            torch.utils.data.dataloader.default_collate(scats),
            torch.utils.data.dataloader.default_collate(metas),
            torch.utils.data.dataloader.default_collate(labels),
        )
    # fallback
    return torch.utils.data.dataloader.default_collate(batch)