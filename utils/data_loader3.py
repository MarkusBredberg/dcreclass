import skimage, cv2, collections, random, math, hashlib, glob, os, re, sys, h5py, torch
import numpy as np, pandas as pd
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils.GAN_models import load_gan_generator
from utils.calc_tools import normalise_images, generate_from_noise, load_model, check_tensor
from firstgalaxydata import FIRSTGalaxyData
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
####################################### LOAD ARTIFICIAL DATA ###########################################
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

def get_synthetic(
    gen_model_name,
    cls,
    num_generate = 1000,
    batch_size_generate = 500,
    galaxy_classes = [10, 11, 12, 13],
    img_shape=(1, 128, 128),
    FILTERGEN=False,
    CLIPDDPM=True,
    model_kwargs = {},
    fold=0,
    device='cpu'
):
    """
    Returns two lists: generated_images_list and generated_labels_list
    following the GAN, DDPM or VAE branch & optional duplicate‑filtering.
    """
    generated_images_list = []
    generated_labels_list = []
    if gen_model_name == 'DDPM':
        ddpm_file_map = {
            #10: "new_generated_fr1.npy",  # Generated by starting with noisy image
            #11: "new_generated_fr2.npy",
            #12: "new_generated_compact.npy",
            #13: "new_generated_bent.npy"
            10: "generated_fr1_8k.npy",   # Generated by starting with pure noise
            11: "generated_fr2_8k.npy",
            12: "generated_compact_8k.npy",
            13: "generated_bent_8k.npy"
        }
        ddpm_path = f"/users/mbredber/data/DDPM/{ddpm_file_map[cls]}"
        print(f"Loading DDPM data for class {cls} from {ddpm_path}")
        ddpm_data = np.load(ddpm_path)  # shape assumed (N, 128, 128)
        ddpm_data = torch.from_numpy(ddpm_data).unsqueeze(1).float()  # (N, 1, 128, 128)
        n_here = min(ddpm_data.shape[0], num_generate)  # Use as many samples as num_generate, or all if fewer exist
        ddpm_data = ddpm_data[:n_here]
        if CLIPDDPM: # Clip to [0,1] range
            ddpm_data = np.clip(ddpm_data, 0, 1)

        # Append to the lists (do not overwrite!)
        generated_images_list.append(ddpm_data)
        generated_labels_list.append(torch.ones(n_here, dtype=torch.long) * (cls - min(galaxy_classes)))
        
    elif gen_model_name in ['ST', 'wGAN', 'WGAN']:
        if gen_model_name == 'ST':
            #pattern = f"/users/mbredber/scratch/ST_generation/*100_{cls}.npy"
            pattern = f"/users/mbredber/scratch/ST_generation/1to1_*_{cls}.npy"
        else:
            pattern = f"/users/mbredber/data/wGAN_FIRST/generated_images_class_{cls}.npy"
        candidates = sorted(glob.glob(pattern))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No ST file found matching pattern {pattern}")
        st_path = candidates[0]
        print(f"Loading data for class {cls} from {st_path}")
        st_data = np.load(st_path)  # shape assumed (N, 128, 128)
        st_data = torch.from_numpy(st_data).unsqueeze(1).float()  # (N, 1, 128, 128)
        n_here = min(st_data.shape[0], num_generate)  # Use as many samples as num_generate, or all if fewer exist
        st_data = st_data[:n_here]
        if gen_model_name in ['wGAN', 'WGAN']: # Normalise from 0, 255 to [0, 1]
            st_data = st_data / 255.0
    
        generated_images_list.append(st_data)
        generated_labels_list.append(torch.ones(n_here, dtype=torch.long) * (cls - min(galaxy_classes)))
        
    elif gen_model_name == 'NOISE': # To troubleshoot, add noise as generated data
        noise = torch.randn(num_generate, *img_shape)
        noise = noise.to(device)
        generated_images_list.append(noise)
        generated_labels_list.append(torch.ones(num_generate, dtype=torch.long) * (cls - min(galaxy_classes)))

    else:
        if gen_model_name == 'GAN':
            gan_path = (
                f"./GAN/generators/generator_"
                f"{model_kwargs['gan_type']}_ss{model_kwargs['gan_sample_size']}_cl{cls}_lrgen{model_kwargs['lr_gen']}_"
                f"lrdisc{model_kwargs['lr_disc']}_gl{model_kwargs['gan_gen_loss']}_dl{model_kwargs['gan_disc_loss']}_"
                f"ls{model_kwargs['gan_label_smoothing']}_ld{model_kwargs['gan_lambda_div']}_dv{model_kwargs['gan_data_version']}_"
                f"ep{model_kwargs['gan_epoch']}_f{fold}.pth"
            )
            print(f"Loading GAN data for class {cls} from {gan_path}")
            model = load_gan_generator(
                gan_path,
                latent_dim=model_kwargs['gan_latent_dim']
            ).to(device)
            latent_dim = model_kwargs['gan_latent_dim']
        else: # VAE
            subname = gen_model_name.split('-',1)[1]             # drop the "VAE-" prefix
            model_dir = f"/users/mbredber/scratch/generator/VAE_{cls}_{model_kwargs['VAE_train_size']}_{subname}_{fold}"
            path      = os.path.join(model_dir, "model.pth")
            model     = load_model(
                path,
                scatshape=model_kwargs['scatshape'],
                hidden_dim1=model_kwargs['hidden_dim1'],
                hidden_dim2=model_kwargs['hidden_dim2'],
                latent_dim=model_kwargs['vae_latent_dim'],
                num_classes=len(galaxy_classes)
            )
            latent_dim = model_kwargs['vae_latent_dim']
            
        model.eval()
        num_batches = (num_generate + batch_size_generate - 1) // batch_size_generate
        total_generated = 0

        for batch_idx in range(num_batches):
            current_batch = min(batch_size_generate, num_generate - batch_idx * batch_size_generate)
            total_generated += current_batch

            generated_labels = torch.ones(current_batch, dtype=torch.long) * (cls - min(galaxy_classes))
            with torch.no_grad():
                generated_batch = generate_from_noise(
                    model,
                    latent_dim=latent_dim,
                    num_samples=current_batch,
                    DEVICE=device)
            # Ensure the tensor is [batch_size, channels, H, W]
            if generated_batch.shape[0] != current_batch:
                generated_batch = generated_batch.permute(1, 0, 2, 3)
                
            if gen_model_name == 'GAN': 
                generated_batch = normalise_images(generated_batch, 0, 1) # Normalise from [-1, 1] to [0, 1]

            if FILTERGEN and cls != 12:
                total_unique = 0
                tol = 1e-3
                # --------------------------
                # (1) Remove duplicates *within* the new batch
                # --------------------------
                flat = generated_batch.view(current_batch, -1)  # flatten each image
                dists = torch.cdist(flat, flat, p=2)
                dists.fill_diagonal_(1e6)  # ignore self-comparisons
                unique_mask = torch.ones(current_batch, dtype=torch.bool, device=generated_batch.device)
                for i in range(current_batch):
                    if unique_mask[i]:
                        # Mark subsequent images that are too close as duplicates
                        unique_mask[i+1:] &= ~(dists[i, i+1:] < tol)

                batch_unique_images = generated_batch[unique_mask]
                batch_unique_labels = generated_labels[unique_mask]
                n_unique = len(batch_unique_images)

                # --------------------------
                # (2) Remove duplicates *across* previously accepted images
                # --------------------------
                if len(generated_images_list) > 0:
                    # Flatten the newly filtered batch
                    new_flat = batch_unique_images.view(n_unique, -1)

                    # Flatten everything we've already accepted
                    prev_accepted = torch.cat(generated_images_list, dim=0)  # shape: [N_accepted, C, H, W]
                    prev_flat = prev_accepted.view(prev_accepted.size(0), -1)

                    # Compare new vs. all accepted
                    cross_dists = torch.cdist(new_flat, prev_flat, p=2)
                    cross_unique_mask = torch.ones(n_unique, dtype=torch.bool, device=batch_unique_images.device)
                    for i in range(n_unique):
                        # If the new image is within tol of *any* accepted image, exclude it
                        if torch.any(cross_dists[i] < tol):
                            cross_unique_mask[i] = False

                    batch_unique_images = batch_unique_images[cross_unique_mask]
                    batch_unique_labels = batch_unique_labels[cross_unique_mask]
                    n_unique = len(batch_unique_images)

                # Now everything in batch_unique_* is unique vs. each other & prior batches
                total_unique += n_unique
                
                # --------------------------
                # (3) Augment the unique images
                # --------------------------
                
                print("Shape of batch_unique_images: ", batch_unique_images.shape)
                generated_batch, generated_labels = augment_images(batch_unique_images, batch_unique_labels) # Produces 8 versions of each image
                
                removal_fraction = (total_generated - total_unique) / total_generated * 100
                print(f"Class {cls}: Removed {removal_fraction:.2f}% of images "
                f"(Generated: {total_generated}, Unique: {total_unique})")

            generated_images_list.append(generated_batch)
            generated_labels_list.append(generated_labels)
        

    generated_images = torch.cat(generated_images_list, dim=0)
    generated_labels = torch.cat(generated_labels_list, dim=0)

    return generated_images, generated_labels


######################################################################################################
################################### DATA SELECTION FUNCTIONS #########################################
######################################################################################################


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

def percentile_stretch(x, lo=30, hi=99):
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

def asinh_stretch(x, alpha=10):
    return torch.asinh(alpha * x) / math.asinh(alpha)

def log_stretch(x, alpha=10):
    return torch.log1p(alpha * x) / math.log1p(alpha)

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
    resized = F.interpolate(crop, size=(Ho, Wo), mode='bilinear') # bilinear or area
    return resized.squeeze(0)                   # [C,Ho,Wo]

def img_hash(img: torch.Tensor) -> str:
    #print("Shape of image before hashing:", img.shape)
    arr = img.cpu().contiguous().numpy()
    returnval = hashlib.sha1(arr.tobytes()).hexdigest()
    #print("Shape of image after hashing:", arr.shape)
    return returnval

######################################################################################################
################################### DATA LOADING FUNCTION ############################################
######################################################################################################

def load_vae_data(runname, sample_size, existing_images, existing_labels, data_type='generated'):
    print(f"Loading VAE {data_type} data...")

    def load_single_run(run):
        vae_file_path = root_path + f"VAE/VAE_{run}_{data_type}_images.pt"
        try:
            vae_data = torch.load(vae_file_path)
            vae_images = vae_data['images']
            vae_labels = vae_data['labels']
            return vae_images, vae_labels
        except FileNotFoundError:
            print(f"File {vae_file_path} not found. Proceeding without VAE images.")
            return None, None

    if isinstance(runname, list):
        vae_images_list, vae_labels_list = [], []
        for run in runname:
            vae_images, vae_labels = load_single_run(run)
            if vae_images is not None and vae_labels is not None:
                vae_images_list.append(vae_images)
                vae_labels_list.append(vae_labels)

        if vae_images_list and vae_labels_list:
            vae_images = torch.cat(vae_images_list, dim=0)
            vae_labels = torch.cat(vae_labels_list, dim=0)
        else:
            return existing_images, existing_labels
    else:
        vae_images, vae_labels = load_single_run(runname)
        if vae_images is None or vae_labels is None:
            print("No VAE images found.")
            return existing_images, existing_labels

    combined_images = torch.cat((existing_images, vae_images[:sample_size - len(existing_images)]), dim=0)
    combined_labels = torch.cat((existing_labels, vae_labels[:sample_size - len(existing_labels)]), dim=0)
    print("Combined images shape:", combined_images.shape)
    return combined_images.tolist(), combined_labels.tolist()


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
    print(f"Samples per each unique class in the evaluation set before redistribution: "
          f"{Counter(eval_labels.tolist())}")
    
    # Find minimum class size
    min_count = min(len(eval_bins[cls]) for cls in target_classes if cls in eval_bins)
    keep_idxs, move_idxs = [], []
    for cls in target_classes:
        if cls not in eval_bins:
            continue
        
        idxs = eval_bins[cls]
        
        # CRITICAL: Sort by filename hash for DETERMINISTIC selection
        # This ensures the SAME samples are kept across all folds/modes
        items_sorted = sorted(idxs, key=lambda i: hashlib.sha1(
            str(eval_fnames[i] if eval_fnames else f"idx_{i}").encode("utf-8")).hexdigest())
        
        # Debug: Print which samples are being kept/moved
        if eval_fnames:
            kept_names = [eval_fnames[items_sorted[j]] for j in range(min(min_count, len(items_sorted)))]
            moved_names = [eval_fnames[items_sorted[j]] for j in range(min_count, len(items_sorted))]
            print(f"Class {cls}: Keeping {len(kept_names)} samples: {kept_names[:5]}{'...' if len(kept_names) > 5 else ''}")
            if moved_names:
                print(f"Class {cls}: Moving {len(moved_names)} samples to train: {moved_names[:5]}{'...' if len(moved_names) > 5 else ''}")
        
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
    fold = 0,                         # 0..9 = CV folds
    train = False,                    # Not implemented
    processed_dir = "/users/mbredber/scratch/create_image_sets_outputs/processed_psz2_fits", # directory for preformatted images
    prefer_processed = True,   # whether to prefer processed images when available
    gate_with = None  # None | "auto" | "T50kpc" | 50 (number). If set, gate RAW/T/RT against this T directory.
):
    # Data available at: https://lofar-surveys.org/planck_dr2.html
    # Fetch data with utils.download_PSZ2.py
    # Categorise data with utils.process_PSZ2.py
    # Format and taper data with taper_tools.create_image_sets.py
    # This is the data loader for any version of the processed data

    if fold == 0 and train == True:
        print("Parameters:")
        print("  path:", path)
        print("  versions:", versions)
        print("  crop_size:", crop_size)
        print("  downsample_size:", downsample_size)
        print("  target_classes:", target_classes)
        print("  processed_dir:", processed_dir)
        print("  prefer_processed:", prefer_processed)
        print("  gate_with:", gate_with)

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

                    def _beam_cov_world(h):
                        bmaj_as = abs(float(h['BMAJ']))*3600.0
                        bmin_as = abs(float(h['BMIN']))*3600.0
                        pa_deg  = float(h.get('BPA', 0.0))
                        to_sig  = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))
                        sx = (bmaj_as * to_sig) * (np.pi/(180.0*3600.0))
                        sy = (bmin_as * to_sig) * (np.pi/(180.0*3600.0))
                        th = np.deg2rad(pa_deg)
                        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], float)
                        S = np.diag([sx*sx, sy*sy])
                        return R @ S @ R.T

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

    print("Number of sources in total:", len(basenames))
    groups = np.array(basenames) # When mutliple versions exist,  this ensures all versions of a source stay together
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        initial_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) # Fixed seed for initial split
        initial_splits = list(initial_sgkf.split(np.zeros(len(y)), y, groups))
        
        # Use the first split: fold 0 becomes test set, rest becomes train+val
        trainval_idx, test_idx = initial_splits[0] # Trainval_idx is a numpy array of indices for train+val
        
    except Exception:
        from sklearn.model_selection import GroupShuffleSplit
        print("⚠️ Warning: StratifiedGroupKFold not available, falling back to GroupShuffleSplit")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        (trainval_idx, test_idx), = gss.split(np.zeros(len(y)), groups=groups)

    if train: # Use train+val split. This is used for hyperparameter tuning
        print("Using train+val split for training/validation.")
        if fold is None or fold < 0 or fold >= 10: 
            raise ValueError("For train=True, fold must be an integer between 0 and 9 for 10-fold CV")
        
        # Extract train+val subset
        print("Trainval_idx length:", len(trainval_idx)); print("Test_idx length:", len(test_idx))
        y_trainval = y[trainval_idx] # Select images for train and validation
        groups_trainval = groups[trainval_idx] # Select groups for train and validation
        
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf_cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED) # 10-fold CV with random seed
            cv_splits = list(sgkf_cv.split(np.zeros(len(y_trainval)), y_trainval, groups_trainval)) # All combinations of train/val splits
            
            # Choose the specific fold or train/val split
            tr_idx_rel, va_idx_rel = cv_splits[fold] 
            tr_idx = trainval_idx[tr_idx_rel]
            va_idx = trainval_idx[va_idx_rel]
            
        except Exception:
            from sklearn.model_selection import GroupShuffleSplit
            print("⚠️ Warning: StratifiedGroupKFold not available for CV, falling back to GroupShuffleSplit")
            gss_cv = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + fold)
            (tr_idx_rel, va_idx_rel), = gss_cv.split(np.zeros(len(y_trainval)), groups=groups_trainval)
            tr_idx = trainval_idx[tr_idx_rel]
            va_idx = trainval_idx[va_idx_rel]
            
    else: # Return all trainval as training, test as eval
        print("Using full train+val as training set, test set as evaluation set.")
        tr_idx = trainval_idx
        va_idx = test_idx

    def _take(idxs):
        return [images[i] for i in idxs], [labels[i] for i in idxs], [basenames[i] for i in idxs]

    train_images, train_labels, train_fns = _take(tr_idx)
    eval_images,  eval_labels,  eval_fns  = _take(va_idx)
    
    print("Number of training samples:", len(train_images))
    print("Number of evaluation samples:", len(eval_images))

    return train_images, train_labels, eval_images, eval_labels, train_fns, eval_fns


def load_galaxies(galaxy_classes, path=None, versions=None, fold=None, island=None, crop_size=None, downsample_size=None, sample_size=None, DEBUG=False,
                  REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False, USE_GLOBAL_NORMALISATION=False, GLOBAL_NORM_MODE="percentile", STRETCH=False, alpha=10.0,
                  percentile_lo=30, percentile_hi=99, NORMALISE=True, NORMALISETOPM=False, PREFER_PROCESSED=True, EXTRADATA=False, PRINTFILENAMES=False, SAVE_IMAGES=False, train=None):
    """
    Master loader that delegates to specific dataset loaders and returns zero-based labels.
    """
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

            
    if isinstance(train_images, list):
        train_images = torch.stack(train_images)
    if isinstance(eval_images, list):
        eval_images  = torch.stack(eval_images)

    if BALANCE:
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
        
    if DEBUG:
        plot_class_images(train_images, train_labels, train_filenames, set_name='train_1.before_normalisation')
        plot_class_images(eval_images,  eval_labels,  eval_filenames,  set_name='eval_1.before_normalisation')
        check_tensor("train_images", train_images)
        check_tensor("eval_images",  eval_images)

    if NORMALISE:
        if isinstance(train_images, list):
            print("Converting train_images list to tensor for normalisation.")
            train_images = torch.stack(train_images)
        if isinstance(eval_images, list):
            print("Converting eval_images list to tensor for normalisation.")
            eval_images = torch.stack(eval_images)
        all_images = torch.cat([train_images, eval_images], dim=0)
        #all_images = normalise_images(all_images, out_min=0, out_max=1)  
        if USE_GLOBAL_NORMALISATION: # Regular normalisation of all images to [0,1]
            if DEBUG:
                print("Applying global normalisation to [0,1]")
            all_images = normalise_images(all_images, out_min=0, out_max=1)
        else:  # Percentile stretch to [0,1]
            if DEBUG:
                print(f"Applying percentile stretch to [{percentile_lo},{percentile_hi}]%")
                
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
        if DEBUG:
            print("Applying normalise to ±1")
            plot_class_images(train_images, train_labels, train_filenames, set_name='train_2.after_normalisation')
            plot_class_images(eval_images,  eval_labels,  eval_filenames,  set_name='eval_2.after_normalisation')

    if STRETCH:
        # Concatenate so train/eval get the exact same mapping
        all_images = torch.cat([train_images, eval_images], dim=0)
                
        # Asinh stretch (elementwise), preserves shape/device/dtype
        stretched = torch.asinh(all_images * alpha) / math.asinh(alpha)

        # Split back
        n_tr = train_images.shape[0]
        train_images = stretched[:n_tr]
        eval_images  = stretched[n_tr:]
        
        if DEBUG:
            print("Applying asinh stretch")
            plot_class_images(train_images, train_labels, train_filenames, set_name='train_3.after_stretching')
            plot_class_images(eval_images,  eval_labels,  eval_filenames,  set_name='eval_3.after_stretching')

    # Always redistribute excess images for the test data, for fair evaluation
    train_images,  eval_images, train_labels, eval_labels, train_filenames, eval_filenames = redistribute_excess(train_images, eval_images, train_labels, eval_labels, train_filenames, eval_filenames)
    
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
        if DEBUG:
            print("Loading PSZ2 metadata")
        meta_df = pd.read_csv(os.path.join(root_path, "PSZ2/cluster_source_data.csv"))
        print("PSZ2 metadata columns:", meta_df.columns.tolist())
        meta_df.rename(columns={"slug": "base"}, inplace=True)
        meta_df.set_index("base", inplace=True)

        # build a list of metadata rows in the same order as your `filenames` list:
        train_data = [meta_df.loc[base].values for base in train_filenames]
        eval_data  = [meta_df.loc[base].values for base in eval_filenames]
        
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
            x = percentile_stretch(x, lo=percentile_lo, hi=percentile_hi)
            x = asinh_stretch(x, alpha=10)
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
