import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils.GAN_models import load_gan_generator
from utils.calc_tools import normalise_images, generate_from_noise, load_model, check_tensor
from firstgalaxydata import FIRSTGalaxyData
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.ndimage import label
import skimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collections
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image
from astropy.io import fits
import random
import math
import hashlib
import glob
import os

# For reproducibility
SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

root_path =  '/users/mbredber/scratch/data/' #'/home/sysadmin/Scripts/data/'  #

######################################################################################################
################################### GENERAL STUFF ####################################################
######################################################################################################

import sys
sys.path.append(root_path)
#from MiraBest.MiraBest_F import MBFRConfident, MBFRUncertain, MBFRFull, MBRandom





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
        # MIRABEST size: 150x150
        {"tag": 14, "length": 397, "description": "Confidently classified FRIs"},
        {"tag": 15, "length": 436, "description": "Confidently classified FRIIs"},
        {"tag": 16, "length": 591, "description": "FRI"},
        {"tag": 17, "length": 633, "description": "FRII"},
        # MNIST size: 1x28x28
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


######################################################################################################
################################### DATA SELECTION FUNCTIONS #########################################
######################################################################################################

def percentile_stretch(x, lo=80, hi=99):
    """
    x: Tensor of shape (B, C, H, W)
    Returns: same shape, linearly rescaled so that the lo-th percentile → 0 and hi-th → 1
    """
    if len(x.shape) == 2:  # If x is 2D, add a channel dimension twice
        x = x.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3:  # If x is 3D, add a channel dimension
        x = x.unsqueeze(0)
    elif len(x.shape) != 4:  # If x is not 4D, raise an error
        raise ValueError("Input tensor x must be of shape (B, C, H, W) or (H, W). Now it is: ", x.shape)

    #flat_all = x.view(-1)
    flat_all = x.reshape(-1)
    p_low  = flat_all.quantile(lo/100)
    p_high = flat_all.quantile(hi/100)
    
    # reshape to broadcast over (B,C,H,W)
    p_low  = p_low.view(1,1,1,1)
    p_high = p_high.view(1,1,1,1)
    
    y = (x - p_low) / (p_high - p_low + 1e-6)
    return y.clamp(0, 1)

def asinh_stretch(x, alpha=10):
    return torch.asinh(alpha * x) / math.asinh(alpha)

def log_stretch(x, alpha=10):
    return torch.log1p(alpha * x) / math.log1p(alpha)
    
def add_highpass(x, alpha=100):
    # asinh stretch
    y = torch.asinh(alpha * x) / math.asinh(alpha)
    # low-pass + subtract
    lp = torch.nn.functional.avg_pool2d(y, kernel_size=15, stride=1, padding=7)
    hp = torch.clamp(y - lp, min=0.)
    # normalize hp
    mn = hp.amin(dim=(2,3), keepdim=True)
    mx = hp.amax(dim=(2,3), keepdim=True)
    hp = (hp - mn) / (mx - mn + 1e-6)
    # stack channels

    return torch.cat([y, hp], dim=1)

def gamma_stretch(x, γ=0.5): # x in [0,1]
    return x.pow(γ)

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
    import numpy as np
    import torch

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


def scale_weaker_region(image, adjustment, initial_threshold=0.9, step=0.01):
    """
    Identify two distinct peaks in the image by thresholding, continue until they merge into one,
    and apply a scaling adjustment to the weaker peak. The image is then normalized so that the
    maximum pixel intensity is 1.
    
    Args:
        image (torch.Tensor or np.ndarray): Input image to be processed.
        adjustment (float): Scaling factor to adjust the weaker peak region.
        initial_threshold (float): Starting threshold value to identify two separate regions.
        step (float): Step size to decrease the threshold until two regions merge.
        
    Returns:
        np.ndarray: Processed image with the weaker peak region scaled and normalized.
    """
    # Convert the image to NumPy if it's a torch tensor
    image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image.copy()
    
    # Normalize the image to have a peak intensity of 1
    image_np = image_np / np.max(image_np)
    
    # Start with an initial threshold and gradually decrease it to find two regions
    threshold = initial_threshold
    labeled_image = None
    num_features = 0
    prev_labeled_image = None
    found_two_regions = False
    drop_below_threshold_count = 0  # Counter to avoid premature break

    while threshold > 0:
        binary_image = image_np > threshold
        labeled_image, num_features = skimage.measure.label(binary_image, return_num=True)
        
        # If we've detected two regions, mark that as found and save the state
        if num_features >= 2:
            found_two_regions = True
            prev_labeled_image = labeled_image  # Store the last state with two distinct regions
            drop_below_threshold_count = 0  # Reset the drop counter when two regions are found
        # If we've previously found two regions but now drop to one, start counting to ensure stability before breaking
        elif num_features < 2 and found_two_regions:
            drop_below_threshold_count += 1
            if drop_below_threshold_count >= 3:  # Allow a few checks to avoid breaking too early
                break
        
        threshold -= step

    # If we never found two distinct regions, exit and print a warning
    if not found_two_regions or prev_labeled_image is None:
        print("Unable to find two distinct regions; no adjustment applied.")
        return image_np

    
    # Identify the two regions based on the peak intensities in the last two-region state
    region_intensities = []
    for region_label in range(1, np.max(prev_labeled_image) + 1):
        region_mask = (prev_labeled_image == region_label)
        region_intensity = np.max(image_np[region_mask])
        region_intensities.append((region_intensity, region_label))
    
    # Sort the regions by intensity and identify the second brightest
    region_intensities.sort(reverse=True, key=lambda x: x[0])
    brightest_region_label = region_intensities[0][1]
    second_brightest_region_label = region_intensities[1][1]
    weaker_region_mask = (prev_labeled_image == second_brightest_region_label)
    
    image_np[weaker_region_mask] *= (1 + adjustment) # Adjust brightness weaker peak region
    image_np = image_np / np.max(image_np) # Normalise image
    
    return image_np


def filter_by_peak_intensity(images, labels, threshold=0.6, region_size=(64, 64)):
    removed_by_peak = []
    filtered_images_by_peak = []
    filtered_labels_by_peak = []
    
    for image, lbl in zip(images, labels):
        # Convert image to numpy for processing, removing channel dimension if present
        image_np = image.squeeze().numpy() if image.dim() == 3 else image.numpy()
        
        # Determine the center of the image and extract the central region
        center_x, center_y = image_np.shape[1] // 2, image_np.shape[0] // 2
        central_region = image_np[center_y - region_size[0] // 2:center_y + region_size[0] // 2,
                                  center_x - region_size[1] // 2:center_x + region_size[1] // 2]
                
        # Convert central region back to a torch tensor before using torch.max
        central_region_tensor = torch.tensor(central_region)
        max_intensity = torch.max(central_region_tensor).item()
    
        # Filter images based on peak intensity
        if max_intensity <= threshold:
            removed_by_peak.append((image, lbl))
        else:
            filtered_images_by_peak.append(image)
            filtered_labels_by_peak.append(lbl)
        
    return filtered_images_by_peak, filtered_labels_by_peak, removed_by_peak


def filter_images_with_edge_emission(images, crop_size=128, threshold=0.2):
    """
    Filter out images based on the max and sum of edge pixel values from a hypothetical 128x128 central crop.
    :param images: List of image tensors
    :return: List of filtered images and a list of removed images based on edge emission
    """
    filtered_images = []
    removed_by_edge = []
        
    for image in images:
        
        if image.dim() == 3 and image.shape[0] == 1:  # If the first dimension is 1 (single-channel grayscale)
            image = image.squeeze(0)
            
        height, width = image.shape[-2], image.shape[-1]
        start_y = (height - crop_size) // 2
        start_x = (width - crop_size) // 2
        end_y = start_y + crop_size
        end_x = start_x + crop_size
                
        # Extract the edges of the hypothetical cropped 128x128 region
        top_edge = image[start_y, start_x:end_x]
        bottom_edge = image[end_y - 1, start_x:end_x]
        left_edge = image[start_y:end_y, start_x]
        right_edge = image[start_y:end_y, end_x - 1]
        
        # Find the max pixel value along these edges
        top_edge_max = torch.max(top_edge)
        bottom_edge_max = torch.max(bottom_edge)
        left_edge_max = torch.max(left_edge)
        right_edge_max = torch.max(right_edge)
        total_max_edge_value = torch.max(torch.stack([top_edge_max, bottom_edge_max, left_edge_max, right_edge_max])).item()
        
        # Calculate the sum of edge pixel values
        top_edge_sum = torch.sum(top_edge.abs()).item()
        bottom_edge_sum = torch.sum(bottom_edge.abs()).item()
        left_edge_sum = torch.sum(left_edge.abs()).item()
        right_edge_sum = torch.sum(right_edge.abs()).item()
        total_edge_sum = top_edge_sum + bottom_edge_sum + left_edge_sum + right_edge_sum
        
        if total_max_edge_value > threshold:
            removed_by_edge.append(image)
        else:
            filtered_images.append(image)
        
    return filtered_images, removed_by_edge


def calculate_outside_emission(image, region_size=(64, 64)):
    """Calculate the fraction of emission outside a central region and the total emission."""
    image_np = image.squeeze().numpy() if image.dim() == 3 else image.numpy()  # Squeeze to remove channel dimension if present
    image_np = image_np / np.max(image_np) if np.max(image_np) != 0 else image_np
    
    center_x, center_y = image_np.shape[1] // 2, image_np.shape[0] // 2
    central_region = image_np[center_y - region_size[0] // 2:center_y + region_size[0] // 2,
                                center_x - region_size[1] // 2:center_x + region_size[1] // 2]
    
    total_emission = np.sum(image_np)
    if total_emission == 0:
        return 0, total_emission
    
    central_emission = np.sum(central_region)
    outside_emission_fraction = (total_emission - central_emission) / total_emission        
    return outside_emission_fraction, total_emission


def count_emission_regions(image, threshold=0.1, region_size=(128, 128)):
    """Count the number of distinct emission regions using connected component analysis."""
    image_np = image.squeeze().numpy() if image.dim() == 3 else image.numpy()
    center_y, center_x = image_np.shape[0] // 2, image_np.shape[1] // 2
    half_height, half_width = region_size[0] // 2, region_size[1] // 2
    start_y, end_y = center_y - half_height, center_y + half_height
    start_x, end_x = center_x - half_width, center_x + half_width
    central_region = image_np[start_y:end_y, start_x:end_x]
    binary_image = central_region > threshold
    labeled_image, num_features = label(binary_image)
    
    return num_features

def plot_class_images(images, labels, filenames=None, set_name='train'):
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
            plt.savefig(f"./classifier/{cls}_{set_name}_images.png", dpi=300)
            plt.close(fig)

def plot_cut_flow_for_all_filters(images, labels, num_thresholds=11, region_size=(64, 64), save_path_prefix="cut_flow"):
    """
    Saves four cut-flow graphs showing the proportion of images removed depending on the threshold value for
    peak intensity, outside emission, total intensity, and emission regions.
    
    Args:
        images: List or tensor of images to filter.
        labels: List or tensor of corresponding labels.
        num_thresholds: Number of threshold values to test (will generate linearly spaced values between 0 and 1).
        region_size: Size of the central region for filtering functions.
        save_path_prefix: Prefix for the saved image file names.
    """
    thresholds = np.linspace(0, 1, num=num_thresholds)
    thresholds = [0, 0.005, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995, 1]
    total_images = len(images)
    
    # Initialize lists to hold the proportion of images removed for each filtering method
    removed_proportion_peak = []
    removed_proportion_outside_emission = []
    removed_proportion_intensity = []
    removed_proportion_regions = []
    
    for threshold in thresholds:
        # Apply peak intensity filtering
        _, _, removed_by_peak = filter_by_peak_intensity(images, labels, threshold, region_size)
        removed_proportion_peak.append(len(removed_by_peak) / total_images)
        
        # Apply outside emission filtering
        removed_by_emission = []
        for image, lbl in zip(images, labels):
            outside_emission_fraction, _ = calculate_outside_emission(image, region_size)
            if outside_emission_fraction > threshold:
                removed_by_emission.append((image, lbl))
        removed_proportion_outside_emission.append(len(removed_by_emission) / total_images)
        
        # Apply intensity filtering
        removed_by_intensity = []
        for image, lbl in zip(images, labels):
            _, total_emission = calculate_outside_emission(image, region_size)
            if total_emission > threshold*1000:  # Assuming the threshold is for intensity
                removed_by_intensity.append((image, lbl))
        removed_proportion_intensity.append(len(removed_by_intensity) / total_images)
        
        # Apply emission regions filtering
        removed_by_regions = []
        for image, lbl in zip(images, labels):
            num_regions = count_emission_regions(image, threshold, region_size)
            if num_regions > 3:  # Assuming the required number of regions is 2
                removed_by_regions.append((image, lbl))
        removed_proportion_regions.append(len(removed_by_regions) / total_images)
    
    # Create and save cut-flow plots for each filtering method
    def save_cut_flow_plot(thresholds, removed_proportions, title, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, removed_proportions, marker='o', linestyle='-', color='b')
        plt.xlabel('Threshold Value')
        plt.ylabel('Proportion of Images Removed')
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    # Save the plot for each filter type
    save_cut_flow_plot(thresholds, removed_proportion_peak, 'Cut-Flow: Peak Intensity', f"{save_path_prefix}peak_intensity.png")
    save_cut_flow_plot(thresholds, removed_proportion_outside_emission, 'Cut-Flow: Outside Emission', f"{save_path_prefix}outside_emission.png")
    save_cut_flow_plot(thresholds, removed_proportion_intensity, 'Cut-Flow: Intensity Threshold', f"{save_path_prefix}intensity.png")
    save_cut_flow_plot(thresholds, removed_proportion_regions, 'Cut-Flow: Emission Regions', f"{save_path_prefix}regions.png")


def filter_away_faintest(images, labels, region=(128, 128), threshold=0.1, save_path="./bright_pixel_filtering.png"):
    """
    Filters out images (and their associated labels) where the brightest pixel in the specified
    central region is below the given threshold.

    Parameters:
        images (list of np.ndarray or torch.Tensor): List of 2D image arrays/tensors.
        labels (list): List of labels corresponding to each image.
        region (tuple): Size of the central region (height, width) to check for brightness.
        threshold (float): Minimum required brightness in the region.
        save_path (str): File path to save the plot of removed images.

    Returns:
        tuple: (filtered_images, filtered_labels), where both lists contain only the images
               and labels that meet or exceed the brightness threshold.
    """
    filtered_images = []
    filtered_labels = []
    
    # We store (image, label, max_val) in removed_images
    removed_images = []

    for img, label in zip(images, labels):
        # Compute central region coordinates
        x_start, x_end = (img.shape[1] - region[1]) // 2, (img.shape[1] + region[1]) // 2
        y_start, y_end = (img.shape[0] - region[0]) // 2, (img.shape[0] + region[0]) // 2
        roi = img[y_start:y_end, x_start:x_end]

        # Convert ROI to NumPy if it's a torch.Tensor
        if isinstance(roi, torch.Tensor):
            roi = roi.detach().cpu().numpy()

        # Determine max brightness in the ROI
        max_val = np.max(roi)

        # Filter logic
        if max_val < threshold:
            # Store (img, label, max_val) for plotting
            removed_images.append((img, label, max_val))
        else:
            filtered_images.append(img)
            filtered_labels.append(label)

    total_per_class = Counter(labels)
    removed_per_class = Counter(lbl for _, lbl, _ in removed_images)

    for cls, total_cnt in total_per_class.items():
        removed_cnt = removed_per_class.get(cls, 0)
        fraction = removed_cnt / total_cnt * 100 if total_cnt else 0
        #print(f"Class {cls}: removed {removed_cnt}/{total_cnt} images ({fraction:.2f}%)")

    removed_count = len(removed_images)

    # Plot removed images
    if removed_count > 0:
        plt.figure(figsize=(5 * removed_count, 5))
        for i, (img, label, max_val) in enumerate(removed_images):
            plt.subplot(1, removed_count, i + 1)
            arr = _to_2d_for_imshow(img, how="first")
            im = plt.imshow(arr, cmap='viridis')
            plt.colorbar(im, ax=plt.gca())
            plt.title(f"Removed Image {i+1}\nLabel: {label}, Max: {max_val:.3f}")
            plt.axis('off')
        plt.savefig(save_path)
        plt.close()
    else:
        print("No images were filtered away.")

    return filtered_images, filtered_labels


def remove_outliers(images, labels, threshold=0.1, peak_threshold=0.6, intensity_threshold=200.0, max_regions=3, region_size=(64, 64), v="training", PLOTFILTERED=False):
    """
    Remove images and their corresponding labels if:
    - The images have more than the threshold fraction of total emission outside a central region (size specified by region_size),
    - Their summed intensities exceed the specified threshold,
    - They have more than a certain number of emission regions (specified by region_threshold).
    
    :param images: List of image tensors
    :param labels: List of corresponding labels
    :param threshold: Fraction of emission allowed outside the central region
    :param peak_threshold: Threshold for peak intensity filtering
    :param intensity_threshold: Threshold for total intensity filtering
    :param region_threshold: Maximum number of distinct emission regions allowed
    :param region_size: Size of the central region (tuple of two ints, e.g., (64, 64) for a 64x64 region)
    :return: Filtered list of image tensors, corresponding labels, and the fraction of images removed
    """
    filtered_labels, removed_by_emission, removed_by_intensity, removed_by_peak_intensity, removed_by_regions = [], [], [], [], []
    filtered_images, removed_by_edge = filter_images_with_edge_emission(images) # Remove images with bright pixels at the edge
    
    for image, lbl in zip(filtered_images, labels):  # Renamed 'label' to 'lbl' to avoid conflict
        outside_emission_fraction, total_emission = calculate_outside_emission(image)
        num_regions = count_emission_regions(image, threshold=0.3)
        
        if outside_emission_fraction > threshold and total_emission <= intensity_threshold:
            removed_by_emission.append((image, lbl))  # Remove images with emission outside central region
        elif total_emission > intensity_threshold:
            removed_by_intensity.append((image, lbl))  # Remove images with total intensity above threshold
        elif num_regions > max_regions:
            removed_by_regions.append((image, lbl))  # Remove images with too many regions
        else:
            filtered_labels.append(lbl)
    
    # Remove images with peak intensity
    filtered_images, filtered_labels, removed_by_peak_intensity = filter_by_peak_intensity(filtered_images, filtered_labels, peak_threshold, region_size)

    # Compile the final filtered list of images and labels
    final_filtered_images = [img for img in filtered_images 
                            if not any(torch.equal(img, removed_img[0]) for removed_img in removed_by_peak_intensity)]
    final_filtered_labels = [lbl for img, lbl in zip(filtered_images, filtered_labels) 
                            if not any(torch.equal(img, removed_img[0]) for removed_img in removed_by_peak_intensity)]
    fraction_final_removed = 1 - len(final_filtered_images) / len(images)

    # Print statistics
    #print(f"Images removed by edge pixels: {len(removed_by_edge)}")
    #print(f"Images removed by intensity (> {intensity_threshold}): {len(removed_by_intensity)}")
    #print(f"Images removed by outside emission (> {threshold} fraction): {len(removed_by_emission)}")
    #print(f"Images removed by peak intensity (< {peak_threshold}): {len(removed_by_peak_intensity)}")
    #print(f"Images removed by emission regions (> {max_regions} regions): {len(removed_by_regions)}")
    print(f"Fraction of images removed from {v} set: {fraction_final_removed}")
    #print("Number of images removed in total:", len(images) - len(final_filtered_images))
    
    def plot_removed_images(removed_by_emission, removed_by_intensity, removed_by_peak_intensity, removed_by_regions, removed_by_edge):
        """Plot examples of removed images for each filtering mechanism with a tighter layout and no empty frames."""

        # Determine the number of columns per row based on the maximum number of images to display (up to 5)
        max_images_per_row = 5
        fig, axs = plt.subplots(5, max_images_per_row, figsize=(10, 10), 
                        gridspec_kw={'wspace': 0.05, 'hspace': 0.1}, 
                        constrained_layout=True)
        fig.suptitle("Images Removed by Various Filtering Criteria")

        # Helper function to display a set of images
        def display_images(removed_images, ax_row, title, add_green_square=False):
            if isinstance(removed_images, tuple):
                print("Unexpected tuple format in image:", removed_images)

            # Add a single title to the left of the row
            axs[ax_row, 0].text(-0.3, 0.5, title, va='center', ha='center', rotation=90, fontsize=12, transform=axs[ax_row, 0].transAxes)

            for i in range(max_images_per_row):
                if i < len(removed_images):
                    image_data = removed_images[i]
                    if isinstance(image_data, tuple):
                        image_data = image_data[0]

                    if isinstance(image_data, (float, int)):
                        print(f"Skipping invalid data: {image_data}")
                        continue

                    if isinstance(image_data, torch.Tensor):
                        image_data = image_data.numpy()

                    if isinstance(image_data, np.ndarray):
                        if image_data.ndim == 3 and image_data.shape[0] == 1:
                            image_data = image_data.squeeze(0)
                    else:
                        if image_data.dim() == 3 and image_data.shape[0] == 1:
                            image_data = image_data.squeeze(0)

                    arr = _to_2d_for_imshow(image_data, how="first")
                    axs[ax_row, i].imshow(arr, cmap='viridis')
                    axs[ax_row, i].axis('off')

                    # Add a red semi-transparent rectangle (128x128 central crop)
                    img_height, img_width = image_data.shape
                    crop_size = 128
                    red_rect = patches.Rectangle(
                        ((img_width - crop_size) / 2, (img_height - crop_size) / 2),
                        crop_size, crop_size,
                        linewidth=2, edgecolor='r', facecolor='none', alpha=0.5
                    )
                    axs[ax_row, i].add_patch(red_rect)

                    if add_green_square:
                        green_crop_size = 64
                        green_rect = patches.Rectangle(
                            ((img_width - green_crop_size) / 2, (img_height - green_crop_size) / 2),
                            green_crop_size, green_crop_size,
                            linewidth=2, edgecolor='g', facecolor='none', alpha=0.5
                        )
                        axs[ax_row, i].add_patch(green_rect)
                else:
                    # Remove the axes entirely if there's no data to show
                    fig.delaxes(axs[ax_row, i])

        # Display images removed by various filtering criteria
        display_images(removed_by_edge, 0, f"Edge Pixels \n {len(removed_by_edge)}")
        display_images(removed_by_emission, 1, f"Outside Emission \n {len(removed_by_emission)}", add_green_square=True)
        display_images(removed_by_intensity, 2, f"Intensity \n {len(removed_by_intensity)}")
        display_images(removed_by_peak_intensity, 3, f"Peak Intensity \n {len(removed_by_peak_intensity)}", add_green_square=True)
        display_images(removed_by_regions, 4, f"Region Count \n {len(removed_by_regions)}")

        plt.savefig('./generator/filtering/removed_by_filtering.png')
        plt.close()



    if PLOTFILTERED:
        plot_removed_images(removed_by_emission, removed_by_intensity, removed_by_peak_intensity, removed_by_regions, removed_by_edge)
    
    return final_filtered_images, final_filtered_labels

#######################################################################################################
################################### DATA AUGMENTATION FUNCTIONS #######################################
#######################################################################################################


def balance_classes(images, labels):
    """
    Randomly down‐sample each class so they all have the same number of samples
    equal to the size of the smallest class.
    """
    # collect indices per class
    class_idxs = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_idxs[lbl].append(i)
    # find smallest class size
    min_n = min(len(idxs) for idxs in class_idxs.values())
    # sample
    selected = []
    for idxs in class_idxs.values():
        selected.extend(random.sample(idxs, min_n))
    random.shuffle(selected)
    # return balanced lists
    return [images[i] for i in selected], [labels[i] for i in selected]


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


def complex_apply_transforms_with_config(image, config, img_shape=(128, 128), initial_threshold=0.9, step=0.01, brightness_adjustment=0.0):
    # Apply weaker peak region scaling first
    image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image
    scaled_image_np = scale_weaker_region(image_np, brightness_adjustment, initial_threshold=initial_threshold, step=step)
    scaled_image = torch.tensor(scaled_image_np).unsqueeze(0)  # Convert back to tensor with correct dimensions
    
    # Existing transformation pipeline
    preprocess = transforms.Compose([
        transforms.CenterCrop((img_shape[-2], img_shape[-1])),
        transforms.Resize((img_shape[-2], img_shape[-1])),  # Resize images to the desired size
        transforms.Lambda(lambda x: transforms.functional.hflip(x) if config['flip_h'] else x),  # Horizontal flip
        transforms.Lambda(lambda x: transforms.functional.vflip(x) if config['flip_v'] else x),  # Vertical flip
        transforms.Lambda(lambda x: transforms.functional.rotate(x, config['rotation']))  # Rotation by specific angle
    ])
    
    if image.dim() == 2:  # Ensure image is 3D
        scaled_image = scaled_image.unsqueeze(0)
    transformed_image = preprocess(scaled_image)
    transformed_image = transformed_image.squeeze(1)
    
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
    resized = F.interpolate(crop, size=(Ho, Wo), mode='bilinear', align_corners=False)
    return resized.squeeze(0)                   # [C,Ho,Wo]

def img_hash(img: torch.Tensor) -> str:
    arr = img.cpu().contiguous().numpy()
    returnval = hashlib.sha1(arr.tobytes()).hexdigest()
    return returnval

######################################################################################################
################################### DATA LOADING FUNCTION ############################################
######################################################################################################


def isolate_galaxy_batch(images, upper_intensity_threshold=500000, lower_intensity_threshold=1000):
    total_images = len(images)
    removed_images = 0
    accepted_images = []

    for image in images:
        # Assuming the input is a NumPy array and already in grayscale format (2D array)
        if image.shape[0] == 3:  # If it's a 3-channel image, convert it to grayscale
            gray_image = image.mean(axis=0)  # Mean over the channel dimension
        else:
            gray_image = image.squeeze()  # Already grayscale, just squeeze if needed

        gray_image = (gray_image * 255).astype(np.uint8)  # Scale to 8-bit image
        thresh_val = 60  # Threshold value
        _, binary_image = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY)

        # Find the connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8, ltype=cv2.CV_32S)

        # Find the label of the component that includes the center pixel
        center_x, center_y = gray_image.shape[1] // 2, gray_image.shape[0] // 2
        center_label = labels[center_y, center_x]

        # Create a mask for the component that includes the center pixel
        mask = np.zeros_like(gray_image)
        mask[labels == center_label] = 255

        # Set all pixels outside the central source to black
        black_background_image = np.where(mask == 255, gray_image, 0)

        # Add channel dimension back if necessary
        isolated_image = np.expand_dims(black_background_image, axis=0)  

        # Check the sum of the intensity of the isolated image
        intensity_sum = np.sum(black_background_image)
        
        if intensity_sum > upper_intensity_threshold or intensity_sum < lower_intensity_threshold:
            # If intensity is below threshold or above, discard image
            removed_images += 1
        else:
            accepted_images.append(isolated_image / 255)  # Normalize back to [0, 1]

    # Calculate the fraction of images removed
    removed_fraction = removed_images / total_images

    # Print the results
    print(f"Images removed in isolate_galaxy_batch: {removed_images}")
    print(f"Fraction removed in isolate_galaxy_batch: {removed_fraction:.2f}")

    # Return the accepted images
    return accepted_images
        
def augment_images(
    images, labels, rotations=[0, 90, 180, 270],
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

    return augmented_images_tensor, augmented_labels_tensor

        
def reduce_by_class(images, labels, sample_size, num_classes):
    class_data = collections.defaultdict(list)
    
    # Group images by class
    for img, lbl in zip(images, labels):
        class_data[lbl].append(img)
    
    reduced_images, reduced_labels = [], []
    for lbl, imgs in class_data.items():
        # Limit samples to sample_size or the available number of samples
        selected_imgs = imgs[:sample_size]
        reduced_images.extend(selected_imgs)
        reduced_labels.extend([lbl] * len(selected_imgs))
    
    return reduced_images, reduced_labels

import math
import hashlib
from collections import defaultdict

def redistribute_excess(train_images, train_labels,
                        eval_images,  eval_labels,
                        target_classes,
                        train_filenames=None, eval_filenames=None):

    # 1) Pool everything
    all_imgs = list(train_images) + list(eval_images)
    all_lbls = list(train_labels) + list(eval_labels)
    # only build filenames if provided, otherwise pad with None
    all_fnames = (list(train_filenames) if train_filenames else []) \
               + (list(eval_filenames)  if eval_filenames  else [])
    if not all_fnames:
        all_fnames = [None] * len(all_imgs)

    # 2) Group by class, keeping filenames together
    bins = defaultdict(list)
    for img, lbl, fname in zip(all_imgs, all_lbls, all_fnames):
        bins[int(lbl)].append((img, fname))
        
    # 3) Compute how many per class to put in eval:
    total = len(all_imgs)
    n_cls = len(target_classes)
    per_class = math.ceil((total * 0.10) / n_cls)
    per_class = min(per_class, min(len(bins[c]) for c in target_classes))
    
    # 4) Deterministically split each bin by hash, preserving both img+fname
    new_eval_imgs, new_eval_lbls, new_eval_fnames = [], [], []
    new_train_imgs, new_train_lbls, new_train_fnames = [], [], []
    for cls in target_classes:
        items = bins[cls]
        items_sorted = sorted(items, key=lambda x: img_hash(x[0]))
        ev = items_sorted[:per_class]
        tr = items_sorted[per_class:]
        new_eval_imgs   += [x[0] for x in ev]
        new_eval_lbls   += [cls]*len(ev)
        new_eval_fnames += [x[1] for x in ev]
        new_train_imgs   += [x[0] for x in tr]
        new_train_lbls   += [cls]*len(tr)
        new_train_fnames += [x[1] for x in tr]
        
    
    # 5) Convert back to original types
    # — Images —
    if isinstance(train_images, torch.Tensor):
        # preserve shape if no examples
        train_imgs2 = (torch.stack(new_train_imgs)
                       if new_train_imgs
                       else torch.empty((0,)+train_images.shape[1:]))
    else:
        train_imgs2 = new_train_imgs

    if isinstance(eval_images, torch.Tensor):
        eval_imgs2 = (torch.stack(new_eval_imgs)
                      if new_eval_imgs
                      else torch.empty((0,)+eval_images.shape[1:]))
    else:
        eval_imgs2 = new_eval_imgs

    # — Labels —
    if isinstance(train_labels, torch.Tensor):
        train_lbls2 = torch.tensor(new_train_lbls, dtype=train_labels.dtype)
    else:
        train_lbls2 = new_train_lbls

    if isinstance(eval_labels, torch.Tensor):
        eval_lbls2 = torch.tensor(new_eval_lbls, dtype=eval_labels.dtype)
    else:
        eval_lbls2 = new_eval_lbls
        
    return (
        train_imgs2, train_lbls2,
        new_train_fnames if train_filenames else [],
        eval_imgs2,  eval_lbls2,
        new_eval_fnames  if eval_filenames  else [],
    )

##########################################################################################
################################## SPECIFIC DATASET LOADER ###############################
##########################################################################################

def load_PSZ2(path=root_path + "PSZ2/classified/",
              fold=5,
              sample_size=300,
              target_classes=[53],
              crop_size=(1, 256, 256),
              downsample_size=(1, 256, 256),
              versions='T100kpcSUB',      # <-- replaced 'version' with 'versions'
              FLUX_CLIPPING=False,
              train=False):
    """
    PSZ2 loader with strict version alignment for tesseracts.

    • If T>1 (i.e., crop_size/downsample_size provided as 4-tuples), we:
      - choose the versions from `versions` (e.g., 'T50kpc', 'T100kpc', 'RAW').
      - build the set intersection of base names across the chosen versions
      - if 'RAW' is present, we additionally remove any RAW-only bases not present in T50kpc or T100kpc
      - stack frames per base into a cube [T, C, H, W]

    • If T==1 we keep the legacy single-version path (FITS -> tensor via apply_formatting).

    NOTE: We now **override** the CUBE selection using `versions`:
      - If `versions` is a list with length > 1 → CUBE=True (tesseract, PNGs).
      - Else → CUBE=False (single version, FITS).
    """
    # --- unpack crop/downsample (kept as-is for compatibility) ---
    if len(crop_size) == 4:
        num_versions, ch_c, h_c, w_c = crop_size
        crop_size = (ch_c, h_c, w_c)
    elif len(crop_size) == 3:
        num_versions = None
        ch_c, h_c, w_c = crop_size
    elif len(crop_size) == 2:
        num_versions = None
        ch_c, h_c, w_c = 1, crop_size[0], crop_size[1]
    else:
        raise ValueError("crop_size must be 2, 3 or 4 dims")

    if len(downsample_size) == 4:
        num_versions, ch_d, h_d, w_d = downsample_size
        downsample_size = (ch_d, h_d, w_d)
    elif len(downsample_size) == 3:
        ch_d, h_d, w_d = downsample_size
    elif len(downsample_size) == 2:
        ch_d, h_d, w_d = 1, downsample_size[0], downsample_size[1]
    else:
        raise ValueError("downsample_size must be 2, 3 or 4 dims")

    def _canon_one(v):
        s  = str(v).strip()
        sl = s.lower()

        # Canonical folders, ordered for readability
        canonical = {
            # RAW
            'raw':          'RAW',

            # 25 kpc
            't25kpcsub':    'T25kpcSUB',
            '25kpcsub':     'T25kpcSUB',
            't25kpc':       'T25kpc',
            't25':          'T25kpc',
            '25kpc':        'T25kpc',

            # 50 kpc
            't50kpcsub':    'T50kpcSUB',
            '50kpcsub':     'T50kpcSUB',
            't50kpc':       'T50kpc',
            't50':          'T50kpc',
            '50kpc':        'T50kpc',

            # 100 kpc
            't100kpcsub':   'T100kpcSUB',
            '100kpcsub':    'T100kpcSUB',
            't100kpc':      'T100kpc',
            't100':         'T100kpc',
            '100kpc':       'T100kpc',
        }
        if sl in canonical:
            return canonical[sl]

        # Shorthand like 'rt25' or 'rt50kpc' → 'T25kpc' / 'T50kpc'
        if sl.startswith('rt'):
            num = sl[2:].rstrip('kpc')
            if num.isdigit():
                return f"T{num}kpc"

        # Fall through untouched (e.g. custom folder names)
        return s


    if isinstance(versions, (list, tuple)):
        _vf_list = [_canon_one(v) for v in versions]
    else:
        _vf_list = [_canon_one(versions)]

    CUBE = len(_vf_list) > 1   # <-- override: CUBE determined by `versions`
    # For single-version FITS branch below, reuse local name `version`
    version = _vf_list[0]

    images, labels, filenames = [], [], []

    # --- tag -> folder name map ---
    classes_map = {c["tag"]: c["description"] for c in get_classes()}

    if CUBE:
        # pick the requested version folders from `versions` (kept rest of logic identical)
        vf_list = _vf_list
        #print(f"[load_PSZ2] CUBE mode: using versions {vf_list} (T={len(vf_list)})")

        def _list_fits_bases(version, class_folder):
            folder = os.path.join(root_path, "PSZ2/classified", version, class_folder)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Version folder missing: {folder}")
            bases = {
                os.path.splitext(fn)[0]
                for fn in os.listdir(folder)
                if fn.lower().endswith(".fits")
            }
            if not bases:
                raise FileNotFoundError(f"No FITS files found in {folder}")
            return bases

        for cls in target_classes:
            class_folder = classes_map.get(cls)
            if class_folder is None:
                continue
            label = cls

            # Require a common basename across *all* requested versions
            base_sets = {vf: _list_fits_bases(vf, class_folder) for vf in vf_list}
            base_names = sorted(set.intersection(*(s for s in base_sets.values())))
            if not base_names:
                raise FileNotFoundError(
                    f"No common FITS basenames across versions {vf_list} for class {class_folder}"
                )

            # Load each base across all versions; error if any file is missing
            for base in base_names:
                frames = []
                for vf in vf_list:
                    fits_path = os.path.join(root_path, "PSZ2/classified", vf, class_folder, f"{base}.fits")
                    if not os.path.isfile(fits_path):
                        raise FileNotFoundError(f"Missing FITS: {fits_path}")

                    arr = fits.getdata(fits_path).astype(np.float32)
                    arr = np.squeeze(arr)
                    if arr.ndim == 3:          # collapse cubes if necessary
                        arr = arr.mean(axis=0)
                    elif arr.ndim != 2:
                        raise ValueError(f"Unexpected FITS shape {arr.shape} in {fits_path}")

                    frame = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

                    # If you want FLUX_CLIPPING here, insert your existing block before formatting.
                    frame = apply_formatting(frame, crop_size, downsample_size)
                    frames.append(frame)

                cube = torch.stack(frames, dim=0)  # [T,1,H,W]
                images.append(cube)
                labels.append(label)
                filenames.append(base)

    else:
        # ---------- legacy single-version path (FITS-based) ----------
        #print(f"[load_PSZ2] Single-version path: version='{version}' (FITS)")
        for cls in target_classes:
            class_folder = classes_map.get(cls, None)
            if class_folder is None:
                continue
            label = cls

            folder_path = os.path.join(path, version, class_folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in os.listdir(folder_path):
                if not fname.lower().endswith(".fits"):
                    continue
                base, _ = os.path.splitext(fname)
                # ignore any file that already embeds the version suffix in its name
                if base.endswith(version):
                    continue

                fits_path = os.path.join(folder_path, fname)
                arr = fits.getdata(fits_path).astype(float)
                arr2 = np.squeeze(arr)
                if arr2.ndim == 3:
                    arr2 = np.mean(arr2, axis=0)
                elif arr2.ndim != 2:
                    raise ValueError(f"Expected 2-D or 3-D stack, got {arr2.shape!r}")

                if FLUX_CLIPPING:
                    hdr = fits.getheader(fits_path)
                    fluxconv = hdr.get('FLUXCONV', 1.0)
                    flux = arr * fluxconv
                    flux_t = torch.from_numpy(np.squeeze(flux)).float().unsqueeze(0)
                    flux_clipped = flux_t.clamp(1e-10, 1e-5)
                    frame = (flux_clipped - 1e-10) / (1e-5 - 1e-10)
                else:
                    frame = torch.from_numpy(arr2).unsqueeze(0).float()

                frame = apply_formatting(frame, crop_size=crop_size, downsample_size=downsample_size)
                images.append(frame)           # [C,H,W]
                labels.append(label)
                filenames.append(base)

    # --------- safety checks ---------
    if len(images) == 0:
        raise ValueError("No images loaded. Check the PSZ2 path/versions/classes.")

    assert len(images) == len(labels) == len(filenames), \
        f"mismatch: {len(images)} imgs, {len(labels)} labels, {len(filenames)} files"

    # --------- class-conflict handling (RH vs RR) ----------
    # For {52 (RH), 53 (RR)} first drop *cluster basenames* that appear in both classes,
    # independent of pixel identity. Then (optionally) do hash de-dup on the remainder.
    if set(target_classes) == {52, 53}:
        # 1) basename-level conflict filter
        base_to_labels = {}
        for lbl, base in zip(labels, filenames):
            base_to_labels.setdefault(base, set()).add(lbl)
        conflict_bases = {b for b, labs in base_to_labels.items() if len(labs) > 1}
        if conflict_bases:
            print(f"Excluding {len(conflict_bases)} ambiguous clusters present in both RH and RR.")
        filtered_triplets = [
            (img, lbl, base)
            for img, lbl, base in zip(images, labels, filenames)
            if base not in conflict_bases
        ]
        if filtered_triplets:
            images, labels, filenames = map(list, zip(*filtered_triplets))
        else:
            images, labels, filenames = [], [], []

        # 2) (optional) hash de-dup (keeps only one copy of identical frames)
        from collections import Counter
        hashes = []
        for img in images:
            arr = img.squeeze().numpy()
            hashes.append(hashlib.sha1(arr.tobytes()).hexdigest())
        counts = Counter(hashes)
        dup_hashes = {h for h, c in counts.items() if c > 1}
        if dup_hashes:
            keep = set()
            seen = set()
            for i, h in enumerate(hashes):
                if h not in seen:
                    keep.add(i)
                    seen.add(h)
            images  = [images[i]  for i in sorted(keep)]
            labels  = [labels[i]  for i in sorted(keep)]
            filenames = [filenames[i] for i in sorted(keep)]


    # --------- stratified split ----------
    all_idx = list(range(len(images)))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    eval_images  = [images[i] for i in test_idx]
    eval_labels  = [labels[i] for i in test_idx]
    train_filenames = [filenames[i] for i in train_idx]
    eval_filenames  = [filenames[i] for i in test_idx]

    return train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames

def load_galaxies(galaxy_classes, path=None, versions=None, fold=None, island=None, crop_size=None, downsample_size=None, sample_size=None, 
                  REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False, FLUX_CLIPPING=False, STRETCH=False, percentile_lo=80, percentile_hi=99,
                  EXTRADATA=False, PRINTFILENAMES=False, NORMALISE=True, NORMALISETOPM=False, SAVE_IMAGES=False, train=None):
    """
    Master loader that delegates to specific dataset loaders and returns zero-based labels.
    """
    def get_max_class(galaxy_classes):
        if isinstance(galaxy_classes, list):
            return max(galaxy_classes)
        return galaxy_classes

    # Clean up kwargs to remove None values
    kwargs = {'path': path, 'versions': versions, 'sample_size': sample_size, 'fold':fold, 'train': train,
              'island': island, 'crop_size': crop_size, 'downsample_size': downsample_size, 'FLUX_CLIPPING': FLUX_CLIPPING}
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    max_class = get_max_class(galaxy_classes)

    # Delegate to specific loaders based on class range
    if 50 < max_class <= 60:
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_PSZ2(target_classes=target_classes, **clean_kwargs)
    else:
        raise ValueError("Invalid galaxy class provided.")
    
    if len(data) == 4:
        train_images, train_labels, eval_images, eval_labels = data
        train_filenames = eval_filenames = None  # No filenames returned
    elif len(data) == 6:
        train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames = data
        # DEBUG PSZ2 split
        #print(f"→ load_PSZ2 returned {len(data)} items")
        
        ## inspect a few IDs and the corresponding labels
        #print("  train IDs:", train_filenames[:5])
        #print("  train labels:", train_labels[:5])
        #print("  eval IDs:", eval_filenames[:5])
        #print("  eval labels:", eval_labels[:5])

        # ensure no cluster appears in both
        overlap = set(train_filenames) & set(eval_filenames)
        assert not overlap, f"PSZ2 split error — these IDs are in both sets: {overlap}"
        
    else:
        raise ValueError("Data loader did not return the expected number of outputs.")
    
    
    # If images are list, converge them to tensors
    #if isinstance(train_images, list):
    #    train_images = torch.stack(train_images).clone().detach()
    #    train_labels = torch.tensor(train_labels, dtype=torch.long)
    #    eval_images  = torch.stack(eval_images).clone().detach()
    #    eval_labels  = torch.tensor(eval_labels,  dtype=torch.long)
    
    # Check for overlap between train and test sets
    train_hashes = {img_hash(img) for img in train_images}
    eval_hashes = {img_hash(img) for img in eval_images}
    if len(eval_images) != 0:
        common = train_hashes & eval_hashes
        if common:
            overlap_hash = next(iter(common))
            train_idxs = [i for i,img in enumerate(train_images) if img_hash(img) == overlap_hash]
            test_idxs  = [i for i,img in enumerate(eval_images) if img_hash(img) == overlap_hash]
            print(f"🔍 Overlap hash {overlap_hash!r} found at train indices {train_idxs} and test indices {test_idxs}")
            assert False, f"Overlap detected: {len(common)} images appear in both train and test validation!"
            
    if isinstance(train_images, list):
        train_images = torch.stack(train_images)
    if isinstance(eval_images, list):
        eval_images  = torch.stack(eval_images)
        
    if galaxy_classes[0] == 52 and galaxy_classes[1] == 53: # Plot some train and validation images for each class
        # Plotting the training and evaluation images for each class
        from matplotlib import gridspec
        
        unique_train_labels = sorted(set(train_labels))
        unique_eval_labels  = sorted(set(eval_labels))
        n_train_classes = len(unique_train_labels)
        n_eval_classes  = len(unique_eval_labels)
        n_cols = 4  # Number of columns in the grid
        n_rows = max(n_train_classes, n_eval_classes) // n_cols + 1
        # Plot 5 examples per class for train and eval, with row labels
        n_examples = 5
        train_classes = sorted(set(train_labels))
        eval_classes  = sorted(set(eval_labels))
        n_train = len(train_classes)
        n_eval  = len(eval_classes)

        fig, axes = plt.subplots(n_train + n_eval, n_examples,
                                figsize=(n_examples * 3, (n_train + n_eval) * 2),
                                constrained_layout=True)

        # Plot train rows
        for i, cls in enumerate(train_classes):
            imgs = [img for img, lbl in zip(train_images, train_labels) if lbl == cls][:n_examples]
            for j, img in enumerate(imgs):
                ax = axes[i, j]
                ax.imshow(img.squeeze(), cmap='gray', origin='lower')
                ax.axis('off')
            # Left-hand label for this row
            axes[i, 0].text(-0.2, 0.5, f'Train\nClass {cls}',
                            transform=axes[i, 0].transAxes,
                            va='center', ha='right', fontsize=12)

        # Plot eval rows
        for k, cls in enumerate(eval_classes):
            i = n_train + k
            imgs = [img for img, lbl in zip(eval_images, eval_labels) if lbl == cls][:n_examples]
            for j, img in enumerate(imgs):
                ax = axes[i, j]
                ax.imshow(img.squeeze(), cmap='gray', origin='lower')
                ax.axis('off')
            axes[i, 0].text(-0.2, 0.5, f'Eval\nClass {cls}',
                            transform=axes[i, 0].transAxes,
                            va='center', ha='right', fontsize=12)

        plt.savefig(f"./classifier/{galaxy_classes[0]}_{galaxy_classes[1]}_5perclass_labeled.png", dpi=300)
        plt.close(fig)
        print(f"Saved training and evaluation images for classes {galaxy_classes} to ./classifier/{'_'.join(map(str, galaxy_classes))}_train_eval_images.png")

    if BALANCE:
        train_images, train_labels = balance_classes(train_images, train_labels) # Remove excess images from the largest class
        
    if STRETCH and not FLUX_CLIPPING and False:
        train_class_ids = sorted(set(train_labels))
        for class_id in train_class_ids:
            train_idx   = next(i for i, lbl in enumerate(train_labels) if lbl == class_id)
            source_name = train_filenames[train_idx]
            cls_img     = train_images[train_idx]
            #print(f"🔍 Using training source {source_name} for class {class_id} at index {train_idx}")
            
            # now load its FITS
            fits_path = f"/users/mbredber/scratch/data/PSZ2/fits/{source_name}/{source_name}.fits"
            fits_data = fits.getdata(fits_path).astype(float)
            fits_img  = torch.from_numpy(fits_data)
            
            # Crop the FITS image to match the crop size
            fits_img = apply_formatting(fits_img, crop_size=(crop_size[-3], crop_size[-2], crop_size[-1]), downsample_size=(downsample_size[-3], downsample_size[-2], downsample_size[-1]))
            #check_tensor('CUTOUT', cls_img)
            #check_tensor('FITS',   fits_img)

            # 4) define your stretches and clipping ranges
            clip_ranges = [(60,99.9), (70,99.5), (80,99)]
            clip_thresholds = [0.02, 0.025, 0.03, 0.35]  # adjust or extend as you like
            funcs = [
                ('original', lambda x: x),
                ('asinh',     lambda x: asinh_stretch(x, alpha=10)),
                ('log',       lambda x: log_stretch(x, alpha=10)),
            ]

                # 5) loop over the *same* source for CUTOUT vs FITS
            for img, suffix in [
                (cls_img,  "CUTOUT"),
                (fits_img, "FITS"),
            ]:
                # define rows and columns
                clip_ranges_plot = [(0, 100)] + clip_ranges[:3]  # top row is no cutoff
                funcs = [
                    ('original', lambda x: x),
                    ('log',      lambda x: log_stretch(x,  alpha=10)),
                    ('asinh',    lambda x: asinh_stretch(x, alpha=10)),
                ]
                n_rows = len(clip_ranges_plot)
                n_cols = len(funcs)

                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(4 * n_cols, 4 * n_rows),
                    gridspec_kw={ 'hspace': 0.3,  # shrink vertical gap
                                  'wspace': 0.2 },# tighten horizontal gap
                    constrained_layout=False
                )
                # make room on left for our y‑labels
                fig.subplots_adjust(left=0.18)

                # set column headers
                for j, (name, _) in enumerate(funcs):
                    axes[0, j].set_title(name)
                    
                #check_tensor(f'Properties of the {suffix} image before stretching', img)

                # fill grid
                for i, (lo, hi) in enumerate(clip_ranges_plot):
                    for j, (_, fn) in enumerate(funcs):
                        P   = percentile_stretch(img, lo=lo, hi=hi)
                        out = fn(P)
                        #check_tensor(f'Properties of the {suffix} image after stretching with {lo}-{hi} percentile, and {fn.__name__} function', out)
                        arr = out.squeeze().cpu().numpy()

                        ax = axes[i, j]
                        # ensure arr is 2D (H,W) or color (H,W,3/4) before imshow
                        if isinstance(arr, torch.Tensor):
                            arr = arr.detach().cpu().numpy()

                        if arr.ndim == 3:
                            # Case: (C,H,W) or (T,H,W)
                            if arr.shape[0] in (3, 4):
                                # interpret as channels-first RGB(A) → move to (H,W,3/4)
                                arr = np.moveaxis(arr, 0, -1)
                            else:
                                # e.g. (2,H,W): pick the first plane (or use arr.mean(0) if you prefer)
                                arr = arr[0]

                        im = ax.imshow(arr, cmap='viridis', origin='lower')

                        if j == 0:
                            ax.set_ylabel(f'pctile [{lo},{hi}]',
                                        rotation=0, labelpad=40)
                        # hide only ticks and spines, keep the ylabel
                        ax.set_xticks([])  
                        ax.set_yticks([])  
                        for spine in ax.spines.values():  
                            spine.set_visible(False)
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        

                fig.suptitle(source_name)
                plt.savefig(f"./classifier/{class_id}_train_{source_name}_{suffix}_stretching.png", dpi=300)
                plt.close(fig)
                
        # now do exactly the same thing for one example per class in the EVAL set
        eval_class_ids = sorted(set(eval_labels))
        for class_id in eval_class_ids:
            # pick the first eval image of this class
            eval_idx    = next(i for i, lbl in enumerate(eval_labels) if lbl == class_id)
            source_name = eval_filenames[eval_idx]
            cls_img     = eval_images[eval_idx]
            #print(f"🔍 Using eval source {source_name} for class {class_id} at index {eval_idx}")

            # load its FITS just like above
            fits_path = f"/users/mbredber/scratch/data/PSZ2/fits/{source_name}/{source_name}.fits"
            fits_data = fits.getdata(fits_path).astype(float)
            fits_img  = torch.from_numpy(fits_data)
            fits_img  = apply_formatting(
                fits_img,
                crop_size=(crop_size[-3], crop_size[-2], crop_size[-1]),
                downsample_size=(downsample_size[-3], downsample_size[-2], downsample_size[-1])
            )
            #check_tensor('CUTOUT_eval', cls_img)
            #check_tensor('FITS_eval',   fits_img)

            # same clip‐ranges & funcs
            clip_ranges_plot = [(0,100), (60,99.9), (70,99.8), (80,99.7)]
            funcs = [
                ('original', lambda x: x),
                ('log',      lambda x: log_stretch(x,  alpha=10)),
                ('asinh',    lambda x: asinh_stretch(x, alpha=10)),
            ]
            n_rows = len(clip_ranges_plot)
            n_cols = len(funcs)

            # loop over CUTOUT vs FITS just like for train
            for img, suffix in [(cls_img, "CUTOUT"), (fits_img, "FITS")]:
                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(4 * n_cols, 4 * n_rows),
                    gridspec_kw={ 'hspace': 0.3,
                                'wspace': 0.2 },
                    constrained_layout=False
                )
                fig.subplots_adjust(left=0.18)

                # set column headers
                for j, (name, _) in enumerate(funcs):
                    axes[0, j].set_title(name)

                # fill grid
                for i, (lo, hi) in enumerate(clip_ranges_plot):
                    for j, (_, fn) in enumerate(funcs):
                        P   = percentile_stretch(img, lo=lo, hi=hi)
                        out = fn(P)
                        arr = out.squeeze().cpu().numpy()

                        ax = axes[i, j]
                        
                        if isinstance(arr, torch.Tensor):
                            arr = arr.detach().cpu().numpy()

                        if arr.ndim == 3:
                            # Case: (C,H,W) or (T,H,W)
                            if arr.shape[0] in (3, 4):
                                # interpret as channels-first RGB(A) → move to (H,W,3/4)
                                arr = np.moveaxis(arr, 0, -1)
                            else:
                                # e.g. (2,H,W): pick the first plane (or use arr.mean(0) if you prefer)
                                arr = arr[0]

                        im = ax.imshow(arr, cmap='viridis', origin='lower')

                        if j == 0:
                            ax.set_ylabel(
                                f"pctile [{lo},{hi}]",
                                rotation=0,
                                va='center',
                                ha='right',
                                labelpad=30,
                                fontsize=12
                            )

                        ax.set_xticks([])
                        ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # include suffix in the title & filename
                fig.suptitle(f"eval {class_id} — {source_name} ({suffix})")
                plt.savefig(f"./classifier/{class_id}_eval_{source_name}_{suffix}_stretching.png", dpi=300)
                plt.close(fig)
            
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
        
    plot_class_images(train_images, train_labels, train_filenames, set_name='train_1.before_normalisation')
    plot_class_images(eval_images,  eval_labels,  eval_filenames,  set_name='eval_1.before_normalisation')

    if NORMALISE:
        if isinstance(train_images, list):
            train_images = torch.stack(train_images)
        if isinstance(eval_images, list):
            eval_images = torch.stack(eval_images)
        all_images = torch.cat([train_images, eval_images], dim=0)
        #all_images = normalise_images(all_images, out_min=0, out_max=1)  
        if FLUX_CLIPPING: # Regular normalisation of all images to [0,1]
            all_images = normalise_images(all_images, out_min=0, out_max=1)
        else:  # Percentile stretch to [0,1]
            
            def per_image_percentile_stretch(x, lo=80, hi=99):
                # x: [B, C, H, W]; returns same shape
                B = x.shape[0]
                out = x.clone()
                for i in range(B):
                    flat = out[i].reshape(-1)
                    p_low  = flat.quantile(lo/100)
                    p_high = flat.quantile(hi/100)
                    out[i] = ((out[i] - p_low) / (p_high - p_low + 1e-6)).clamp(0, 1)
                return out

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
        
    plot_class_images(train_images, train_labels, train_filenames, set_name='train_2.after_normalisation')
    plot_class_images(eval_images,  eval_labels,  eval_filenames,  set_name='eval_2.after_normalisation')

    if STRETCH:
        # Asinh stretch after percentile stretch
        all_images = torch.cat([train_images, eval_images], dim=0)
        images = np.stack([img.squeeze().numpy() for img in all_images], axis=0)
        pct = torch.from_numpy(images).float()     
        images = asinh_stretch(pct, alpha=10)                                # apply asinh
        #images = log_stretch(pct, alpha=10)                                  # apply log stretch
        train_images = images[:len(train_images)]
        eval_images  = images[len(train_images):]
        
    plot_class_images(train_images, train_labels, train_filenames, set_name='train_3.after_stretching')
    plot_class_images(eval_images,  eval_labels,  eval_filenames,  set_name='eval_3.after_stretching')

        
    classes_present = torch.unique(torch.cat([torch.tensor(train_labels), torch.tensor(eval_labels)])).tolist()
    train_images, train_labels, train_filenames, eval_images, eval_labels, eval_filenames = redistribute_excess(train_images, train_labels, eval_images, eval_labels, classes_present, train_filenames, eval_filenames)
    
    if SAVE_IMAGES:
        for kind, imgs, lbls in (
            ('train', train_images, train_labels),
            ('eval',  eval_images,  eval_labels),
        ):
            for cls in target_classes:
                cls_imgs = [img for img, lbl in zip(imgs, lbls) if lbl == (cls-min(target_classes))]
                print(f"Length of {kind} images for class {cls}: {len(cls_imgs)}")
                np.save(f"{path}_{kind}_{cls}_{len(cls_imgs)}.npy", cls_imgs)
        
    if EXTRADATA and not PRINTFILENAMES:
        meta_df = pd.read_csv(os.path.join(root_path, "PSZ2/cluster_source_data.csv"))
        print("PSZ2 metadata columns:", meta_df.columns.tolist())
        meta_df.rename(columns={"slug": "base"}, inplace=True)
        meta_df.set_index("base", inplace=True)

        # build a list of metadata rows in the same order as your `filenames` list:
        train_data = [meta_df.loc[base].values for base in train_filenames]
        eval_data  = [meta_df.loc[base].values for base in eval_filenames]
        
    if AUGMENT:
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
        
    print("Shape of train_images returned from PSZ2:", train_images.shape)
    
    if PRINTFILENAMES:
        return train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames
    elif EXTRADATA:
        train_data = torch.tensor(np.stack(train_data), dtype=torch.float32)
        eval_data  = torch.tensor(np.stack(eval_data),  dtype=torch.float32)
        return train_images, train_labels, eval_images, eval_labels, train_data, eval_data
    
    
    return train_images, train_labels, eval_images, eval_labels