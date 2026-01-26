import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from lpips import LPIPS
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.ndimage import label
from utils.VAE_models import get_VAE_model
from sklearn.metrics import mean_squared_error
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from kymatio.torch import Scattering2D

# cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate, normalise_images, check


#################################
###### METRICS FUNCTIONS ########
#################################


def calculate_rmae(real, gen):
    return np.mean(np.abs(real - gen) / (np.abs(real) + np.abs(gen) + 1e-10), axis=None)

def bhattacharyya_coefficient(data1, data2, bins=50):
    hist1, bin_edges = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    # Normalize the histograms so they sum to 1
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    return np.sum(np.sqrt(hist1 * hist2))

def total_variation_distance(data1, data2, bins=50):
    hist1, bin_edges = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    # Normalize the histograms so they sum to 1
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    return 0.5 * np.sum(np.abs(hist1 - hist2))

def histogram_intersection(data1, data2, bins=50):
    hist1, bin_edges = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    # Normalize the histograms so they sum to 1
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    return np.sum(np.minimum(hist1, hist2))

def calculate_lpips_diversity(images, model_name='alex', device='cpu'):
    loss_fn = LPIPS(net=model_name).to(device)
    indices = random.sample(range(len(images)), k=min(100, len(images)))  # sample 100 images
    pairs = [(i, j) for i in indices for j in indices if i < j]
    total_score = 0.0
    for i, j in pairs[:200]:  # compute on at most 200 pairs for speed
        d = loss_fn(images[i].unsqueeze(0).to(device), images[j].unsqueeze(0).to(device))
        total_score += d.item()
    return total_score / len(pairs[:200])

def calculate_cmmd(real_images, gen_images, sigma=1.0):
    # Flatten images to vectors
    real = real_images.view(real_images.size(0), -1)
    gen = gen_images.view(gen_images.size(0), -1)
    
    def gaussian_kernel(x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        dist_sq = (diff ** 2).sum(dim=2)
        return torch.exp(-dist_sq / (2 * sigma**2))
    
    K_xx = gaussian_kernel(real, real, sigma)
    K_yy = gaussian_kernel(gen, gen, sigma)
    K_xy = gaussian_kernel(real, gen, sigma)
    
    n = real.size(0)
    m = gen.size(0)
    # Exclude diagonal for same-set kernel sums
    sum_K_xx = K_xx.sum() - torch.diag(K_xx).sum()
    sum_K_yy = K_yy.sum() - torch.diag(K_yy).sum()
    
    mmd = sum_K_xx / (n * (n - 1)) + sum_K_yy / (m * (m - 1)) - 2 * K_xy.mean()
    return mmd.item()


def get_inception_features(images, model, batch_size=32, device='cpu'):
    model.eval()
    n_images = images.shape[0]
    activations = []
    with torch.no_grad():
        for i in range(0, n_images, batch_size):
            batch = images[i:i+batch_size]
            # Resize to 299x299 and convert to 3 channels if needed
            batch = F.interpolate(batch, size=(299,299), mode='bilinear', align_corners=False)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            # Convert from [-1,1] to [0,1]
            batch = (batch + 1) / 2
            batch = batch.to(device)
            pred = model(batch)
            # If output is 2D (logits), use as is; if 4D, pool to get a flat vector.
            if len(pred.shape) == 2:
                features = pred
            else:
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1,1))
                features = pred.view(batch.size(0), -1)
            activations.append(features.cpu().numpy())
    return np.concatenate(activations, axis=0)

def calculate_fid(real_images, gen_images, device='cpu'):
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception.eval()
    real_act = get_inception_features(real_images, inception, device=device)
    gen_act = get_inception_features(gen_images, inception, device=device)
    mu_real = np.mean(real_act, axis=0)
    mu_gen = np.mean(gen_act, axis=0)
    sigma_real = np.cov(real_act, rowvar=False)
    sigma_gen = np.cov(gen_act, rowvar=False)
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


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



def change_images(images):
    # Apply random flipping 
    if np.random.rand() > 0.5:
        images = images.flip(3)  # Flip horizontally
    if np.random.rand() > 0.5:
        images = images.flip(2)  # Flip vertically
    
    # Apply random rotation 90, 180, 270 degrees
    if np.random.rand() > 0.5:
        images = torch.rot90(images, 1, [2, 3])  # Rotate 90 degrees
    if np.random.rand() > 0.5:
        images = torch.rot90(images, 2, [2, 3])  # Rotate 180 degrees
    if np.random.rand() > 0.5:
        images = torch.rot90(images, 3, [2, 3])  # Rotate 270 degrees
    return images

def get_main_and_secondary_peaks(images, threshold=0.1):
    peak_intensity_stats_all = []
    
    for image in images:
        image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image
        binary_image = image_np > threshold
        labeled_image, num_features = label(binary_image)
        region_max_intensities = []
        
        # Loop over each labeled region (distinct summit)
        for region_label in range(1, num_features + 1):
            region_mask = labeled_image == region_label
            region_pixel_values = image_np[region_mask]
            if region_pixel_values.size > 0:
                region_max_intensities.append(np.max(region_pixel_values))
        
        sorted_intensities = np.sort(region_max_intensities)[::-1]
        
        if len(sorted_intensities) > 1:
            main_peak = sorted_intensities[0]
            second_peak = sorted_intensities[1]
        elif len(sorted_intensities) == 1:
            main_peak = sorted_intensities[0]
            second_peak = 0  
        else:
            main_peak = 0
            second_peak = 0
        
        # Append the main and secondary peaks for this image
        peak_intensity_stats_all.append([main_peak, second_peak])

    # Convert the list to a torch tensor for consistent output
    peak_intensity_stats_all = torch.tensor(peak_intensity_stats_all)
    
    return peak_intensity_stats_all


def get_main_and_secondary_peaks_with_locations(images, threshold=0.1):
    peak_info_all = []
    
    for image in images:
        # Convert image to numpy array
        image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image
        
        # Apply threshold to create binary image
        binary_image = image_np > threshold
        
        # Label connected regions (summits)
        labeled_image, num_features = label(binary_image)
        
        # Store peak intensities and locations of distinct regions
        region_max_intensities = []
        region_max_locations = []
        
        # Loop over each labeled region (distinct summit)
        for region_label in range(1, num_features + 1):
            region_mask = labeled_image == region_label
            region_pixel_values = image_np[region_mask]
            region_coordinates = np.argwhere(region_mask)
            
            if region_pixel_values.size > 0:
                # Find the maximum intensity and its location in this region
                max_intensity = np.max(region_pixel_values)
                max_index = np.argmax(region_pixel_values)
                max_location = region_coordinates[max_index]
                
                region_max_intensities.append(max_intensity)
                region_max_locations.append(max_location)
        
        # Sort the distinct region maxima in descending order and get the corresponding locations
        if len(region_max_intensities) > 1:
            sorted_indices = np.argsort(region_max_intensities)[::-1]
            main_peak_value = region_max_intensities[sorted_indices[0]]
            main_peak_location = region_max_locations[sorted_indices[0]]
            second_peak_value = region_max_intensities[sorted_indices[1]]
            second_peak_location = region_max_locations[sorted_indices[1]]
        elif len(region_max_intensities) == 1:
            main_peak_value = region_max_intensities[0]
            main_peak_location = region_max_locations[0]
            second_peak_value = 0
            second_peak_location = None
        else:
            main_peak_value = 0
            main_peak_location = None
            second_peak_value = 0
            second_peak_location = None

        # Append the main and secondary peaks with their locations for this image
        peak_info_all.append({
            'main_peak_value': main_peak_value,
            'main_peak_location': main_peak_location,
            'second_peak_value': second_peak_value,
            'second_peak_location': second_peak_location
        })

    return peak_info_all



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

def filter_generated_images(generated_images_list, generated_labels_list):
    # Pairwise comparison of generated images
    filtered_images_list, filtered_labels_list = [], []
    for i, img1 in enumerate(generated_images_list):
        is_unique = True
        for j, img2 in enumerate(generated_images_list):
            if i == j:
                continue
            if torch.allclose(img1, img2, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            filtered_images_list.append(img1)
            filtered_labels_list.append(generated_labels_list[i])
            
    # Plot some example of images next to their similar images
    if SHOWIMGS:
        for i in range(min(18, len(filtered_images_list))):
            save_images_tensorboard(torch.cat([filtered_images_list[i], generated_images_list[i]], dim=0), save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_filtered_grid.png", nrow=6)

    return filtered_images_list, filtered_labels_list

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


def cache_scattering(images, scattering, cache_path, batch_size=128, device='cpu'):
    if os.path.exists(cache_path):
        print("Loading cached scattering coefficients from", cache_path)
        scat_coeffs = torch.load(cache_path)
    else:
        print("Computing scattering coefficients and caching to", cache_path)
        scat_coeffs_list = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(device)
                batch_scat = scattering(batch).detach()
                if batch_scat.shape[1] == 1:
                    batch_scat = batch_scat.squeeze(1)
                scat_coeffs_list.append(batch_scat.cpu())
        scat_coeffs = torch.cat(scat_coeffs_list, dim=0)
        torch.save(scat_coeffs, cache_path)
    return scat_coeffs


# First, when you compute and save the scattering coefficients,
# write them to a memmap file instead of saving a full tensor.
def cache_scattering_memmap(images, scattering, cache_path, batch_size=128, device="cpu"):
    num_images = len(images)
    # Determine the shape for one sample
    with torch.no_grad():
        sample = scattering(images[:batch_size].to(device)).detach()
        if sample.shape[1] == 1:
            sample = sample.squeeze(1)
        sample_shape = sample.shape[1:]  # e.g., (C, H, W)
    full_shape = (num_images,) + sample_shape
    
    if not os.path.exists(cache_path):
        print("Computing and writing scattering coefficients to memmap at", cache_path)
        # Create a memmap array in write mode
        memmap_array = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=full_shape)
        with torch.no_grad():
            for i in range(0, num_images, batch_size):
                batch = images[i:i+batch_size].to(device)
                batch_scat = scattering(batch).detach()
                if batch_scat.shape[1] == 1:
                    batch_scat = batch_scat.squeeze(1)
                memmap_array[i:i+batch_size] = batch_scat.cpu().numpy()
        # Flush changes to disk
        memmap_array.flush()
    else:
        print("Memmap cache found at", cache_path)
    return cache_path, full_shape

# Then, create a custom dataset that loads scattering coefficients from the memmap file.
class CachedScatterDataset(Dataset):
    def __init__(self, images, labels, cache_file, scatter_shape):
        self.images = images  # still in memory
        self.labels = labels
        self.cache_file = cache_file
        self.scatter_shape = scatter_shape
        # Open the memmap in read mode; this does not load everything into RAM.
        self.scat_memmap = np.memmap(cache_file, dtype=np.float32, mode='r', shape=scatter_shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # Only load the scattering coefficient for the current index.
        scat = torch.tensor(self.scat_memmap[idx]).float()
        if NORMALISESCS:
            scat = normalise_images(scat, 0, 1) 
        if NORMALISESCSTOPM:
            scat = normalise_images(scat, -1, 1)

        return image, scat, label

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


def save_images_tensorboard(images, save_path, nrow=4):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, save_path)
    

def get_model_name(galaxy_classes, num_galaxies, encoder, fold):
    if isinstance(galaxy_classes, int):
        runname = f'{galaxy_classes}_{num_galaxies}_{encoder}_{fold}'
        return f'./generator/VAE_{runname}/model.pth'
        #return f'./generator/model_{runname}.pth'
    else:
        galaxy_classes_str = ", ".join(map(str, galaxy_classes))
        runname = f'[{galaxy_classes_str}]_{num_galaxies}_{encoder}_{fold}'
        return f'./generator/VAE_{runname}/model.pth'


def load_model(path, scatshape, hidden_dim1=256, hidden_dim2=128, latent_dim=64, num_classes=1, J=2, DEVICE='cuda'):
    print(f"Loading model from {path}")
    name = path.split('_')[-2].split('.')[0]
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model = get_VAE_model(name=name, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes, J=J)

    # Adapt the state_dict to match model's state_dict
    model_state_dict = model.state_dict()
    adapted_state_dict = {}
    for key in model_state_dict.keys():
        if key in state_dict and model_state_dict[key].shape == state_dict[key].shape:
            adapted_state_dict[key] = state_dict[key]
        else:
            print(f"Skipping {key} due to size mismatch or missing key")

    # Load the adapted state dict
    model.load_state_dict(adapted_state_dict, strict=False)
    model.to(DEVICE)
    return model

def get_model_directory(galaxy_classes=11, num_galaxies=1001028, encoder='Dual', fold=0):
    if isinstance(galaxy_classes, int):
        runname = f'{galaxy_classes}_{num_galaxies}_{encoder}_{fold}'
        return f'./generator/VAE_{runname}/model.pth'
        #return f'./generator/model_{runname}.pth'
    else:
        galaxy_classes_str = ", ".join(map(str, galaxy_classes))
        runname = f'[{galaxy_classes_str}]_{num_galaxies}_{encoder}_{fold}'
        return f'./generator/VAE_{runname}/model.pth'


def generate_from_noise(model, img_shape=(1, 128, 128), latent_dim=128,
                        num_samples=1000, DEVICE=None):
    device = next(model.parameters()).device if DEVICE is None else torch.device(DEVICE)
    model.eval()
    with torch.no_grad():
        # Create noise as a 2D tensor. Uncomment the next line if your decoder expects a 4D tensor.
        if hasattr(model, 'decoder'):
            noise =  torch.randn(num_samples, latent_dim, device=device)
            generated_images = model.decoder(noise)
        else:
            noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
            generated_images = model(noise)
            
        if not isinstance(generated_images, torch.Tensor):
            print("Converting generated images to tensor...")
            generated_images = torch.from_numpy(generated_images)
        
        # Check if the output shape already matches img_shape; if not, attempt to reshape.
        if generated_images.shape[1:] != img_shape:
            try:
                print("Reshaping generated images...")
                generated_images = generated_images.view(-1, *img_shape)
            except Exception as e:
                print("Error in reshaping generated images:", e)
                raise
            
        return generated_images
 


            
def find_most_similar_images(real_images, generated_images):
    most_similar_images = []
    for real_img in real_images:
        mse_list = [mean_squared_error(real_img.cpu().numpy().flatten(), gen_img.cpu().numpy().flatten()) for gen_img in generated_images]
        mse_tensor = torch.tensor(mse_list)
        best_match = generated_images[torch.argmin(mse_tensor)]
        most_similar_images.append(best_match)
    return torch.stack(most_similar_images)


def center_max_intensity(images):
    batch_size, height, width = images.shape
    center_x, center_y = height // 2, width // 2

    for i in range(batch_size):
        image = images[i]
        max_pos = torch.nonzero(image == image.max(), as_tuple=False)[0]  # Find the position of the max intensity pixel
        max_y, max_x = max_pos[0], max_pos[1]
        shift_y, shift_x = center_y - max_y, center_x - max_x

        # Roll the image to center the max intensity pixel
        images[i] = torch.roll(image, shifts=(shift_y, shift_x), dims=(0, 1))

    return images


def compute_radial_intensity(images, img_shape):
    center = (img_shape[1] // 2, img_shape[2] // 2)
    max_distance = int(np.sqrt(center[0]**2 + center[1]**2))
    radial_intensity = np.zeros(max_distance)
    radial_counts = np.zeros(max_distance)
    for img in images:
        img = img.cpu().numpy().squeeze()
        for i in range(img_shape[1]):
            for j in range(img_shape[2]):
                distance = int(np.sqrt((i - center[0])**2 + (j - center[1])**2))
                if distance < max_distance:
                    radial_intensity[distance] += img[i, j]
                    radial_counts[distance] += 1

    radial_intensity /= radial_counts
    return radial_intensity

