import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cv2
from utils.plotting import vae_multmod
from utils.GAN_models import load_gan_generator
import torch
from utils.calc_tools import get_main_and_secondary_peaks, get_main_and_secondary_peaks_with_locations
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression 
import time

print("Running 3.2")


######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

galaxy_class, num_galaxies = 11, 1101128
num_display, num_generate = 5, 1000 # Plotted images per model, generated images for statistics
include_two_point_correlation = False
folds = [0, 1, 2, 3, 4]
model_names = ['Dual', 'CNN', 'STMLP', 'ldiffSTMLP', 'lavgSTMLP', 'GAN']

# -------------------------- NEW GAN CONFIGURATION --------------------------
# Set USE_GAN to True to use a pre-trained GAN generator instead of a VAE.
gan_epoch = 500           # e.g., epoch number to load from
gan_gen_loss = 'MSE'       # e.g., generator loss value (as used in filename)
gan_disc_loss = 'BCE'      # e.g., discriminator loss value (as used in filename)
gan_latent_dim = 128
lr_gen = 1e-3
lr_disc = 1e-4
gan_adam_beta = (0.5, 0.999)
gan_weight_decay = 0.0
gan_label_smoothing = 0.9
gan_type = ['Simple', 'Advanced'][0]
# -------------------------------------------------------------------------

# Define consistent color mapping
colors = {
    'Real':      '#0072B2',  # blue
    'STMLP':     '#E69F00',  # orange
    'lavgSTMLP': '#CC79A7',  # reddish purple
    'ldiffSTMLP':'#F0E442',  # yellow
    'Dual':      '#009E73',  # bluish green
    'CNN':       '#56B4E9',  # sky blue
    'GAN':       '#FF0000'   # bright red
}



#######################################################################################################

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(galaxy_class) if isinstance(galaxy_class, list) else 1
USE_GAN = True if model_names == ['GAN'] else False
print("USE_GAN is ", USE_GAN)

# --------------------------- LOAD AND SORT DATA -----------------------------------------

# Reassemble so that 'GAN' is last
vae_model_names = [name for name in model_names if name != 'GAN']
gans = [name for name in model_names if name == 'GAN']
model_names = vae_model_names + gans

# Load the saved data (unchanged)
load_dir = f'./generator/eval/{galaxy_class}_evaluations'
loaded_data = torch.load(os.path.join(load_dir, f'compressed_{galaxy_class}_{num_galaxies}_data.pt'))


# Initialize empty lists for later use
generated_images_list = []
reconstructed_images_list = []
latent_representations = []

# --- Load GAN images (if present) ---
aggregated_gan = None
if 'GAN' in model_names:
    gan_images = []
    for fold in folds:
        gan_path = f"./GAN/generators/generator_{gan_type}_latent{gan_latent_dim}_lrgen{lr_gen}_lrdisc{lr_disc}_gl{gan_gen_loss}_dl{gan_disc_loss}_ab{gan_adam_beta}_wd{gan_weight_decay}_ls{gan_label_smoothing}_ep{gan_epoch}_f{fold}.pth"
        generator = load_gan_generator(gan_path, latent_dim=gan_latent_dim).to(DEVICE)
        latent_vectors = torch.randn(num_generate, gan_latent_dim, 1, 1).to(DEVICE)
        gan_generated_images = generator(latent_vectors).detach().cpu()
        gan_generated_images = (gan_generated_images + 1) * 0.5  # Renormalize to [0, 1]
        gan_images.append(gan_generated_images)
    # Concatenate along the batch dimension for GAN
    aggregated_gan = torch.cat(gan_images, dim=0)

# --- Load VAE-generated and reconstructed images ---
if any(m in model_names for m in ['Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP']):
    print("Reading in VAEs")
    # Load raw lists from the compressed file
    raw_vae_generated = [img.float().cpu() for img in loaded_data['generated_images_list']]
    raw_reconstructed = [img.float().cpu() for img in loaded_data['reconstructed_images_list']]
    raw_latents = [rep.float().cpu() for rep in loaded_data['latent_representations']]
    latent_dim = loaded_data['latent_dim']
    models = loaded_data['models']
    # We want only the VAE models (non-GAN)
    vae_model_names = [name for name in model_names if name != "GAN"]

    # Group the VAE images by model name using the metadata in 'models'
    grouped_vae_generated = {name: [] for name in vae_model_names}
    grouped_reconstructed = {name: [] for name in vae_model_names}
    grouped_latents = {name: [] for name in vae_model_names}
    for img, m in zip(raw_vae_generated, models):
        if m['name'] in vae_model_names:
            grouped_vae_generated[m['name']].append(img)
    for img, m in zip(raw_reconstructed, models):
        if m['name'] in vae_model_names:
            grouped_reconstructed[m['name']].append(img)
    for rep, m in zip(raw_latents, models):
        if m['name'] in vae_model_names:
            grouped_latents[m['name']].append(rep)
    # Aggregate (concatenate) images for each model
    aggregated_vae_generated = {name: torch.cat(grouped_vae_generated[name], dim=0) 
                                for name in grouped_vae_generated}
    aggregated_reconstructed = {name: torch.cat(grouped_reconstructed[name], dim=0) 
                                for name in grouped_reconstructed}
    aggregated_latents = {name: torch.cat(grouped_latents[name], dim=0) 
                          for name in grouped_latents}

    # --- Reassemble the final lists in the order of model_names ---
    final_generated_images_list = []
    final_reconstructed_images_list = []
    final_latent_representations = []
    for name in model_names:
        if name == "GAN":
            final_generated_images_list.append(aggregated_gan)
            # For GAN, we do not have reconstructions or latent representations.
        else:
            final_generated_images_list.append(aggregated_vae_generated[name])
            final_reconstructed_images_list.append(aggregated_reconstructed[name])
            final_latent_representations.append(aggregated_latents[name])
    # Overwrite our lists with the aggregated versions in the correct order
    generated_images_list = final_generated_images_list
    if not USE_GAN:
        reconstructed_images_list = final_reconstructed_images_list
        latent_representations = final_latent_representations

# Now, 'generated_images_list' is a list with one tensor per model (in the order of model_names),
# and similarly for 'reconstructed_images_list' and 'latent_representations' (for non-GAN models).

print("Length of generated images (should equal len(model_names)): ", len(generated_images_list))
if generated_images_list is not None and len(generated_images_list) > 0:
    print(f"Generated {len(generated_images_list[0]) * len(generated_images_list)} images.")
else:
    print("No images were generated.")

# Create the output directories
save_dir = f'./generator/eval/{galaxy_class}_{model_names}_{num_galaxies}'
for folder in ["latent", "peak_locations", "peak_statistics", "asymmetry"]:
    os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

# Load test images and labels (unchanged)
test_images = loaded_data['test_images'].float()
test_labels = loaded_data['test_labels']
img_shape = test_images[0].shape

print("Models: ", models)
print("Model names: ", model_names)

###########################################################
################# EVALUATION FUNCTIONS #####################
###########################################################


def old_bhattacharyya_coefficient(hist1, hist2):
    return np.sum(np.sqrt(hist1 * hist2))

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


def find_most_similar_images(real_images, generated_images):
    most_similar_images = []
    for real_img in real_images:
        real_img_flat = real_img.flatten().to(generated_images.device)
        mse_list = [(real_img_flat - gen_img.flatten()).pow(2).mean() for gen_img in generated_images]
        mse_tensor = torch.tensor(mse_list, device=generated_images.device)
        best_match = generated_images[torch.argmin(mse_tensor)]
        most_similar_images.append(best_match)
    return torch.stack(most_similar_images)    

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


def calculate_rmae(real, gen):  # Relative mean absolute error
    return np.mean(np.abs(real - gen) / (real + 1e-8))


def two_point_correlation(image, max_distance):
    image = image.squeeze().cpu().numpy()
    height, width = image.shape
    correlation = np.zeros(max_distance)
    counts = np.zeros(max_distance)

    for i in range(height):
        for j in range(width):
            for di in range(-max_distance, max_distance + 1):
                for dj in range(-max_distance, max_distance + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        distance = int(np.sqrt(di**2 + dj**2))
                        if distance < max_distance:
                            correlation[distance] += image[i, j] * image[ni, nj]
                            counts[distance] += 1

    correlation /= counts
    return correlation


def compute_average_two_point_correlation(images, max_distance):
    correlation_sum = np.zeros(max_distance)
    num_images = len(images)

    for image in images:
        correlation = two_point_correlation(image, max_distance)
        correlation_sum += correlation

    average_correlation = correlation_sum / num_images
    return average_correlation

def calculate_summed_intensities(real_images, generated_images_list, model_names):
    real_intensity_sum = real_images.view(real_images.size(0), -1).sum(dim=1).cpu().numpy()
    all_generated_sums = {encoder: [] for encoder in set(model_names)}

    for i, generated_images in enumerate(generated_images_list):
        encoder = model_names[i]
        generated_intensity_sum = generated_images.view(generated_images.size(0), -1).sum(dim=1).cpu().numpy()
        all_generated_sums[encoder].append(generated_intensity_sum)

    all_intensities = [real_intensity_sum]
    for encoder in all_generated_sums:
        concatenated_sums = np.concatenate(all_generated_sums[encoder], axis=0)
        all_intensities.append(concatenated_sums)

    combined_intensities = np.concatenate(all_intensities)

    return real_intensity_sum, {k: np.concatenate(v, axis=0) for k, v in all_generated_sums.items()}, combined_intensities


def calculate_total_intensities(real_images, generated_images_list, model_names):
    real_intensities = real_images.view(-1).cpu().numpy()
    all_generated_intensities = {encoder: [] for encoder in set(model_names)}

    for i, generated_images in enumerate(generated_images_list):
        encoder = model_names[i]
        generated_intensities = generated_images.view(-1).cpu().numpy()
        all_generated_intensities[encoder].append(generated_intensities)

    all_intensities = [real_intensities]
    for encoder in all_generated_intensities:
        concatenated_intensities = np.concatenate(all_generated_intensities[encoder], axis=0)
        all_intensities.append(concatenated_intensities)

    combined_intensities = np.concatenate(all_intensities)

    return real_intensities, {k: np.concatenate(v, axis=0) for k, v in all_generated_intensities.items()}, combined_intensities


def calculate_peak_intensities(real_images, generated_images_list, model_names):
    real_peak_intensity = real_images.view(real_images.size(0), -1).max(dim=1).values.cpu().numpy()
    all_generated_peaks = {encoder: [] for encoder in set(model_names)}

    for i, generated_images in enumerate(generated_images_list):
        encoder = model_names[i]
        generated_peak_intensity = generated_images.view(generated_images.size(0), -1).max(dim=1).values.cpu().numpy()
        all_generated_peaks[encoder].append(generated_peak_intensity)

    all_intensities = [real_peak_intensity]
    for encoder in all_generated_peaks:
        concatenated_peaks = np.concatenate(all_generated_peaks[encoder], axis=0)
        all_intensities.append(concatenated_peaks)

    combined_intensities = np.concatenate(all_intensities)

    return real_peak_intensity, {k: np.concatenate(v, axis=0) for k, v in all_generated_peaks.items()}, combined_intensities


def calculate_two_point_correlation_score(original_corr, generated_corr):
    return np.mean(np.abs(original_corr - generated_corr) / (original_corr + generated_corr + 1e-10))


def calculate_pca_average_intensity(pca_components):
    return np.mean(np.abs(pca_components), axis=1)

    
def calculate_ssim(real_images, gen_images):
    return np.mean([ssim(real_img.squeeze(), gen_img.squeeze(), data_range=gen_img.max() - gen_img.min()) for real_img, gen_img in zip(real_images.cpu().numpy(), gen_images.cpu().numpy())])


def interpolate_images(model, img_shape, num_steps=20):
    # Generate two random latent vectors
    z1 = torch.randn(1, latent_dim).to(DEVICE)
    z2 = torch.randn(1, latent_dim).to(DEVICE)

    # Interpolation steps
    z_steps = [(1 - alpha) * z1 + alpha * z2 for alpha in np.linspace(0, 1, num_steps)]

    interpolated_images = []

    # Check if the model is conditional
    if hasattr(model, 'decoder') and 'conditional' in str(type(model.decoder)).lower():
        # Generate images from interpolated latent vectors for conditional model
        labels = torch.ones(1, dtype=torch.long).to(DEVICE)  # Assuming label '0'
        labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # One-hot encode the labels
        for z in z_steps:
            img = model.decoder(z, labels).view(-1, *img_shape).cpu().detach().numpy()
            interpolated_images.append(img)
    else:
        # Generate images from interpolated latent vectors for non-conditional model
        for z in z_steps:
            img = model.decoder(z).view(-1, *img_shape).cpu().detach().numpy()
            interpolated_images.append(img)

    return interpolated_images

def extract_first_number(line):
    """Extract the first token that can be converted to a float."""
    parts = line.split(":")
    if len(parts) < 2:
        return None
    tokens = parts[1].strip().split()
    for token in tokens:
        token_clean = token.replace(',', '')
        try:
            return float(token_clean)
        except ValueError:
            continue
    return None


def parse_log_file(file_path):
    log_file_path = file_path.replace("model", "log").replace(".pth", ".txt")
    if not os.path.exists(log_file_path):
        print(f"Log file {log_file_path} does not exist.")
        return 0, 0, 0, 0  # Default values

    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    if len(lines) < 3:
        print(f"Log file {log_file_path} does not have enough lines.")
        return 0, 0, 0, 0  # Default values

    try:
        if "ST" in file_path or 'Dual' in file_path:  # Scattering transform models
            scatter_time_1 = extract_first_number(lines[0])
            scatter_time_2 = extract_first_number(lines[1])
            train_time = extract_first_number(lines[2])
            epochs = extract_first_number(lines[3])
        else:  # Non-scattering models
            scatter_time_1 = 0
            scatter_time_2 = 0
            train_time = extract_first_number(lines[0])
            epochs = extract_first_number(lines[1])
    except Exception as e:
        print(f"Error parsing log file {log_file_path}: {e}")
        return 0, 0, 0, 0  # Default values

    # Fallback to 0 if any value wasn't found
    scatter_time_1 = scatter_time_1 if scatter_time_1 is not None else 0
    scatter_time_2 = scatter_time_2 if scatter_time_2 is not None else 0
    train_time = train_time if train_time is not None else 0
    epochs = epochs if epochs is not None else 0

    return scatter_time_1, scatter_time_2, train_time, epochs


def old_evaluate_symmetry(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert to NumPy if it's a tensor
    
    gray_image = np.squeeze(image, axis=0) #Reduce grayscale image dimension
    height, width = gray_image.shape
    
    # Vertical symmetry (left vs. right)
    left_half = gray_image[:, :width // 2]
    right_half = gray_image[:, width // 2:]
    right_half_flipped = np.flip(right_half, axis=1)
    if left_half.shape[1] != right_half_flipped.shape[1]:
        right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))
    vertical_diff = np.sum(np.abs(left_half - right_half_flipped))
    
    # Horizontal symmetry (top vs. bottom)
    top_half = gray_image[:height // 2, :]
    bottom_half = gray_image[height // 2:, :]
    bottom_half_flipped = np.flip(bottom_half, axis=0)
    if top_half.shape[0] != bottom_half_flipped.shape[0]:
        bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))
    horizontal_diff = np.sum(np.abs(top_half - bottom_half_flipped))
    
    # Overall asymmetry score is the average of vertical and horizontal scores
    vertical_score = vertical_diff / (width * height)
    horizontal_score = horizontal_diff / (width * height)
    overall_asymmetry_score = (vertical_score + horizontal_score) / 2
    
    return vertical_score, horizontal_score, overall_asymmetry_score

def evaluate_symmetry(image):
    """
    Calculate a symmetry score by folding the image twice (first vertically, then horizontally),
    and comparing how well the regions overlap.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert to NumPy if it's a tensor
    
    gray_image = np.squeeze(image, axis=0)  # Reduce grayscale image dimension
    height, width = gray_image.shape
    
    # Vertical fold: Left half folded over the right half
    left_half = gray_image[:, :width // 2]
    right_half = gray_image[:, width // 2:]
    right_half_flipped = np.flip(right_half, axis=1)
    
    if left_half.shape[1] != right_half_flipped.shape[1]:
        right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))
    
    # Combine the left and right halves (after folding) to get a vertically folded image
    vertical_fold = (left_half + right_half_flipped) / 2
    
    # Horizontal fold: Top half folded over the bottom half of the vertically folded image
    top_half = vertical_fold[:height // 2, :]
    bottom_half = vertical_fold[height // 2:, :]
    bottom_half_flipped = np.flip(bottom_half, axis=0)
    
    if top_half.shape[0] != bottom_half_flipped.shape[0]:
        bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))
    
    # Calculate symmetry difference after both folds
    double_fold_diff = np.sum(np.abs(top_half - bottom_half_flipped))
    double_fold_score = double_fold_diff / (width * height) # Normalise 
    
    return double_fold_score


def calculate_mse(original, constructed):
    # Convert PyTorch tensor to NumPy array and ensure it's in the correct format
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()  # Convert to NumPy if it's a tensor
    if isinstance(constructed, torch.Tensor):
        constructed = constructed.cpu().numpy()

    # Ensure grayscale images are properly formatted
    if len(original.shape) == 3 and original.shape[0] == 1:
        original = np.squeeze(original, axis=0)
    if len(constructed.shape) == 3 and constructed.shape[0] == 1:
        constructed = np.squeeze(constructed, axis=0)
    
    # Convert to grayscale if necessary
    if len(original.shape) == 3:  # If the image has color channels
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
    
    if len(constructed.shape) == 3:  # If the image has color channels
        constructed_gray = cv2.cvtColor(constructed, cv2.COLOR_BGR2GRAY)
    else:
        constructed_gray = constructed
    
    # Resize constructed image if necessary
    if original_gray.shape != constructed_gray.shape:
        constructed_gray = cv2.resize(constructed_gray, (original_gray.shape[1], original_gray.shape[0]))
    
    # Calculate Mean Squared Error
    mse_value = np.mean((original_gray - constructed_gray) ** 2)
    return mse_value


#######################################################
################# PLOTTING FUNCTIONS ##################
#######################################################


def plot_asymmetry_vs_mse(original_images, constructed_images, encoder, save_path='./generator/eval/asymmetry_vs_mse_with_regression.pdf'):
    start_time = time.time()
    asymmetry_scores = []
    mse_values = []
    
    for original, constructed in zip(original_images, constructed_images):
        # Calculate asymmetry score for the original image
        asymmetry_score = evaluate_symmetry(original)
        asymmetry_scores.append(asymmetry_score)
        
        # Calculate MSE between original and constructed image
        mse_value = calculate_mse(original, constructed)
        mse_values.append(mse_value)
    
    # Convert lists to numpy arrays for regression
    asymmetry_scores = np.array(asymmetry_scores).reshape(-1, 1)
    mse_values = np.array(mse_values)
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(asymmetry_scores, mse_values)
    
    # Get the regression line equation
    slope = model.coef_[0]
    intercept = model.intercept_
    regression_equation = f'{slope:.2f}x + {intercept:.2f}'
    
    # Predict MSE values using the linear regression model
    predicted_mse_values = model.predict(asymmetry_scores)
    
    # Create a scatter plot and plot the linear regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(asymmetry_scores, mse_values, color='blue', label='MSE vs Asymmetry')
    plt.plot(asymmetry_scores, predicted_mse_values, color='red', label=f'Linear Fit: {regression_equation}')
    plt.xlabel('Asymmetry Score')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'MSE vs Asymmetry Score for {encoder}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Function plot_asymmetry executed in {time.time() - start_time:.2f} seconds")


def plot_asymmetry_score_histogram(images, num_bins=20, save_path='./generator/eval/asymmetry_histogram.pdf'):
    """
    Calculate asymmetry scores for a set of images and plot a histogram showing the distribution of the scores.

    Parameters:
    - images: list or tensor of images to calculate the asymmetry scores for.
    - num_bins: Number of bins for the histogram (default is 20).
    """
    # Step 1: Calculate asymmetry scores for each image
    asymmetry_scores = [evaluate_symmetry(image) for image in images]
    
    # Step 2: Extract the overall asymmetry score
    overall_asymmetry_scores = asymmetry_scores

    # Step 3: Plot histogram of asymmetry scores
    plt.figure(figsize=(8, 6))
    plt.hist(overall_asymmetry_scores, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Asymmetry Score')
    plt.ylabel('Number of Images')
    plt.title('Histogram of Asymmetry Scores')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
def plot_images_by_asymmetry_bins(images, constructed_images, save_path='./generator/eval/asymmetry_bins.pdf'):
    start_time = time.time()
    # Step 1: Calculate asymmetry scores for each image
    asymmetry_scores = []
    
    for original, constructed in zip(images, constructed_images):
        asymmetry_score = evaluate_symmetry(original)
        asymmetry_scores.append(asymmetry_score)
    
    # Step 2: Sort the asymmetry scores and corresponding images
    asymmetry_scores = np.array(asymmetry_scores)
    sorted_indices = np.argsort(asymmetry_scores)
    sorted_scores = asymmetry_scores[sorted_indices]
    sorted_images = [images[i] for i in sorted_indices]
    sorted_constructed = [constructed_images[i] for i in sorted_indices]
    
    # Step 3: Divide into nine bins and select images from first, third, fifth, seventh, and ninth bins
    num_images = len(sorted_images)
    bin_size = num_images // 9
    
    # Ensure bin indices are within bounds
    first_bin_idx = bin_size - 1 if bin_size > 0 else 0
    third_bin_idx = 3 * bin_size - 1
    fifth_bin_idx = 5 * bin_size - 1
    seventh_bin_idx = 7 * bin_size - 1
    ninth_bin_idx = 9 * bin_size - 1
    
    # Select images and constructed images from the specified bins
    selected_bins = [first_bin_idx, third_bin_idx, fifth_bin_idx, seventh_bin_idx, ninth_bin_idx]
    
    selected_images = [sorted_images[i] for i in selected_bins]
    selected_constructed_images = [sorted_constructed[i] for i in selected_bins]
    selected_scores = [sorted_scores[i] for i in selected_bins]
    
    # Step 4: Plot the images in five columns: one for each bin (first, third, fifth, seventh, ninth)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    
    # Row 1: Original images
    for i, (image, score) in enumerate(zip(selected_images, selected_scores)):
        axs[0, i].imshow(image.squeeze(), cmap='gray')
        axs[0, i].set_title(f'Asymmetry Score: {score:.4f} (Bin {i*2 + 1})')
        axs[0, i].axis('off')
    
    # Row 2: constructed images
    for i, constructed_image in enumerate(selected_constructed_images):
        axs[1, i].imshow(constructed_image.squeeze(), cmap='gray')
        axs[1, i].set_title(f'constructed Image (Bin {i*2 + 1})')
        axs[1, i].axis('off')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Function plot_asymmetry_bins executed in {time.time() - start_time:.2f} seconds")
    

def plot_images_with_peaks(images, peaks_info_all, save_path='./generator/eval/peaks.pdf'):
    start_time = time.time()
    # Ensure we only plot the first 3 images and their peaks
    images = images[:3]
    peaks_info_all = peaks_info_all[:3]
    
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))  # Set up a 1x3 grid for images
    
    for i, (image, peaks) in enumerate(zip(images, peaks_info_all)):
        axs[i].imshow(image.cpu().numpy().squeeze(), cmap='viridis')

        main_peak_location = peaks.get('main_peak_location')
        second_peak_location = peaks.get('second_peak_location')

        if main_peak_location is not None and len(main_peak_location) == 2:
            axs[i].scatter(main_peak_location[1], main_peak_location[0], facecolor='none', edgecolor='gold', s=100, marker='o')

        if second_peak_location is not None and len(second_peak_location) == 2:
            axs[i].scatter(second_peak_location[1], second_peak_location[0], facecolor='none', edgecolor='silver', s=100, marker='o')

        axs[i].set_title(f"Image {i+1}")
        axs[i].axis('off')   
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Function plot_images_with_peaks executed in {time.time() - start_time:.2f} seconds")


def plot_peak_statistics(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, title, save_path, show_secondary=True):
    start_time = time.time()
    categories_produced = ['Real'] + model_names

    # Convert PyTorch tensors to NumPy arrays if they are not already
    if isinstance(peak_intensity_stats_real, torch.Tensor):
        peak_intensity_stats_real = peak_intensity_stats_real.cpu().numpy()
    if isinstance(peak_intensity_stats_produced, torch.Tensor):
        peak_intensity_stats_produced = [prod.cpu().numpy() for prod in peak_intensity_stats_produced]

    # Split peak intensity stats for real data
    real_primary_peaks = np.array([item['main_peak_value'] for item in peak_intensity_stats_real])
    real_secondary_peaks = np.array([item['second_peak_value'] for item in peak_intensity_stats_real])

    # Calculate mean and std for real peaks
    real_primary_mean, real_primary_std = np.mean(real_primary_peaks), np.std(real_primary_peaks)
    real_secondary_mean, real_secondary_std = np.mean(real_secondary_peaks), np.std(real_secondary_peaks)

    prod_primary_means, prod_primary_stds, prod_secondary_means, prod_secondary_stds = [], [], [], []

    for i in range(len(peak_intensity_stats_produced)):  # Loop over all the different model_names
        # Get the dictionary from the list for each produced set of peaks
        prod_primary_peaks = np.array([item['main_peak_value'] for item in peak_intensity_stats_produced[i]])
        prod_secondary_peaks = np.array([item['second_peak_value'] for item in peak_intensity_stats_produced[i]])

        # Calculate mean and std for constructed peaks
        prod_primary_means.append(np.mean(prod_primary_peaks))
        prod_primary_stds.append(np.std(prod_primary_peaks))
        prod_secondary_means.append(np.mean(prod_secondary_peaks))
        prod_secondary_stds.append(np.std(prod_secondary_peaks))

    # Function to plot bars with values
    def plot_with_values(ax, index, primary_means, primary_stds, secondary_means, secondary_stds, categories, title, show_secondary):
        bar_width = 0.35

        # Plot primary peak data
        colors_used = [colors.get(cat, 'gray') for cat in categories]  # Use your defined color mapping
        bars_primary = ax.bar(index, primary_means, bar_width, yerr=primary_stds, label='Primary Peak', color=colors_used, capsize=5)
        
        # Plot secondary peak data only if show_secondary is True
        if show_secondary:
            bars_secondary = ax.bar(index + bar_width, secondary_means, bar_width, yerr=secondary_stds, label='Secondary Peak', color=colors_used, alpha=0.7, capsize=5)

        # Add text labels above the bars
        for bar in bars_primary:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

        if show_secondary:
            for bar in bars_secondary:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

        # Customize plot labels
        ax.set_xlabel('Model')
        ax.set_ylabel('Peak Intensity')
        ax.set_title(title)
        ax.set_xticks(index + (bar_width / 2) if show_secondary else bar_width / 2)
        ax.set_xticklabels(categories)
        ax.legend()

    # Plot produced vs real
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(len(categories_produced))

    # Add real data as the first entry
    real_primary_mean = [real_primary_mean]
    real_primary_std = [real_primary_std]
    real_secondary_mean = [real_secondary_mean]
    real_secondary_std = [real_secondary_std]

    plot_with_values(ax, index, real_primary_mean + prod_primary_means, real_primary_std + prod_primary_stds,
                     real_secondary_mean + prod_secondary_means, real_secondary_std + prod_secondary_stds,
                     categories_produced, title, show_secondary)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Function plot_peaks_statistics executed in {time.time() - start_time:.2f} seconds")


def plot_peak_ratio_distribution(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, save_path):
    # Calculate ratios for real data
    real_ratios = np.array([item['second_peak_value'] for item in peak_intensity_stats_real]) / \
                  np.clip(np.array([item['main_peak_value'] for item in peak_intensity_stats_real]), 1e-8, None)

    # Calculate ratios for produced data using the correct keys from the dictionaries
    prod_ratios = [np.array([prod_stats['second_peak_value'] for prod_stats in prod_stats_list])/ \
                   np.clip(np.array([prod_stats['main_peak_value'] for prod_stats in prod_stats_list]), 1e-8, None) \
                   for prod_stats_list in peak_intensity_stats_produced]

    # Combine data for plotting
    data = [real_ratios] + prod_ratios
    labels = ['Real'] + model_names
    
    # Plot
    plt.figure(figsize=(10, 6))
    boxplot_colors = [colors.get(label, 'gray') for label in labels]  # Use your defined color mapping
    box = plt.boxplot(data, patch_artist=True, tick_labels=labels)

    
    # Apply colors to the box plots
    for patch, color in zip(box['boxes'], boxplot_colors):
        patch.set_facecolor(color)
    
    plt.title('Distribution of Primary-to-Secondary Peak Intensity Ratios')
    plt.ylabel('Ratio of secondary to primary peak')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_peak_histograms_combined(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, prodtype, save_path):
    # Extract real primary and secondary peak intensities
    real_primary_peaks = np.array([item['main_peak_value'] for item in peak_intensity_stats_real]).flatten()
    real_secondary_peaks = np.array([item['second_peak_value'] for item in peak_intensity_stats_real]).flatten()

    # Store produced primary and secondary peak intensities per model, using dictionary keys instead of indices
    produced_primary_peaks = {}
    produced_secondary_peaks = {}
    
    for i, model_name in enumerate(model_names):
        prod_primary_peaks = np.array([item['main_peak_value'] for item in peak_intensity_stats_produced[i]]).flatten()  # Access using the 'main_peak_value' key
        prod_secondary_peaks = np.array([item['second_peak_value'] for item in peak_intensity_stats_produced[i]]).flatten()  # Access using the 'second_peak_value' key

        produced_primary_peaks[model_name] = prod_primary_peaks
        produced_secondary_peaks[model_name] = prod_secondary_peaks

    # Combine the intensities for bin calculation
    combined_primary_intensities = np.concatenate([real_primary_peaks] + list(produced_primary_peaks.values()))
    combined_secondary_intensities = np.concatenate([real_secondary_peaks] + list(produced_secondary_peaks.values()))

    # Call the provided plotting function for both primary and secondary peaks
    plot_intensity_histograms_individually(real_primary_peaks, produced_primary_peaks, combined_primary_intensities, model_names, 
                                           title='Histogram of Primary Peak Intensities with RMAE', 
                                           xlabel='Primary Peak Intensity', 
                                           save_path=f'{save_path}_primary.pdf')

    plot_intensity_histograms_individually(real_secondary_peaks, produced_secondary_peaks, combined_secondary_intensities, model_names, 
                                           title='Histogram of Secondary Peak Intensities with RMAE', 
                                           xlabel='Secondary Peak Intensity', 
                                           save_path=f'{save_path}_secondary.pdf')



def plot_peak_ratio_heatmap(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, prodtype, save_path):
    # Calculate the ratio of primary to secondary peaks for real data
    real_ratios = np.array([item['main_peak_value'] for item in peak_intensity_stats_real]) / \
                  np.clip([item['second_peak_value'] for item in peak_intensity_stats_real], 1e-8, None)
    all_ratios = [real_ratios]

    # Calculate the ratios for produced data, assuming dictionary structure
    prod_ratios = []
    for prod_stats in peak_intensity_stats_produced:
        prod_ratio = np.array([item['main_peak_value'] for item in prod_stats]) / np.clip(
            [item['second_peak_value'] for item in prod_stats], 1e-8, None)
        prod_ratios.append(prod_ratio)
        all_ratios.append(prod_ratio)

    # Ensure all ratios have the same length for consistency in plotting
    min_len = min(len(ratio) for ratio in all_ratios)
    all_ratios = [ratio[:min_len] for ratio in all_ratios]
    data = np.column_stack(all_ratios)
    log_data = np.log10(data + 1)
    labels = ['Real'] + [f'{name}_{prodtype}' for name in model_names]  # Adjust labels

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(log_data, annot=False, cmap='coolwarm', cbar=True, xticklabels=labels, yticklabels=False)
    plt.title('Secondary-to-Primary Peak Intensity Ratios')
    plt.xlabel('Model')
    plt.ylabel('Image Index')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_radial_intensity(
    original_radial_intensity,
    rec_radial_intensity_list,
    models_radial_intensity,
    model_names,
    title="Radial Intensity",
    save=True,
    save_path_generated='./generator/eval/radial_generated.pdf',
    save_path_reconstructed='./generator/eval/radial_reconstructed.pdf'
):
    print("Plotting radial intensity")

    # Initialize storage for radial intensities by encoder
    all_radial_intensities = {}
    all_reconstructed_intensities = {}

    # Collect and group radial intensities by encoder for generated images
    for model_name, radial_intensity in zip(model_names, models_radial_intensity):
        encoder = model_name.split('_')[0]  # Use the first part of the model name as the encoder identifier
        if encoder not in all_radial_intensities:
            all_radial_intensities[encoder] = []
        all_radial_intensities[encoder].append(radial_intensity)

    # Only collect and group reconstructed intensities if the list is non-empty
    if rec_radial_intensity_list is not None and len(rec_radial_intensity_list) > 0:
        for model_name, rec_radial_intensity in zip(model_names, rec_radial_intensity_list):
            encoder = model_name.split('_')[0]  # Use the first part of the model name as the encoder identifier
            if encoder not in all_reconstructed_intensities:
                all_reconstructed_intensities[encoder] = []
            all_reconstructed_intensities[encoder].append(rec_radial_intensity)
    else:
        print("No reconstructed images provided; skipping reconstructed plot.")

    # Plot 1: Generated Images
    plt.figure(figsize=(10, 6))
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.plot(original_radial_intensity, label='Real (solid black)', color='black', linestyle='-', linewidth=2)

    for encoder, radial_intensities in all_radial_intensities.items():
        mean_radial_intensity = np.mean(radial_intensities, axis=0)
        std_radial_intensity = np.std(radial_intensities, axis=0)
        rmae = calculate_rmae(original_radial_intensity, mean_radial_intensity)
        color = colors.get(encoder, 'gray')
        plt.plot(mean_radial_intensity, label=f'{encoder} Generated (RMAE={rmae:.3f})', linestyle=':', color=color, alpha=0.8)
        plt.fill_between(range(len(mean_radial_intensity)), 
                         mean_radial_intensity - std_radial_intensity, 
                         mean_radial_intensity + std_radial_intensity, 
                         color=color, alpha=0.2)

    plt.xlabel('Distance from Center (log scale)')
    plt.ylabel('Average Pixel Intensity')
    plt.title(f'{title} - Generated Images')
    plt.legend(loc='upper right')
    plt.grid(True)
    if save:
        plt.savefig(save_path_generated)
    else:
        plt.show()
    plt.close()

    # Plot 2: Reconstructed Images (only if data exists)
    if all_reconstructed_intensities:
        plt.figure(figsize=(10, 6))
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.plot(original_radial_intensity, label='Real (solid black)', color='black', linestyle='-', linewidth=2)

        for encoder, rec_radial_intensities in all_reconstructed_intensities.items():
            mean_rec_radial_intensity = np.mean(rec_radial_intensities, axis=0)
            std_rec_radial_intensity = np.std(rec_radial_intensities, axis=0)
            rmae = calculate_rmae(original_radial_intensity, mean_rec_radial_intensity)
            color = colors.get(encoder, 'gray')
            plt.plot(mean_rec_radial_intensity, label=f'{encoder} Reconstructed (RMAE={rmae:.3f})', linestyle='--', color=color, alpha=0.8)
            plt.fill_between(range(len(mean_rec_radial_intensity)), 
                             mean_rec_radial_intensity - std_rec_radial_intensity, 
                             mean_rec_radial_intensity + std_rec_radial_intensity, 
                             color=color, alpha=0.2)

        plt.xlabel('Distance from Center (log scale)')
        plt.ylabel('Average Pixel Intensity')
        plt.title(f'{title} - Reconstructed Images')
        plt.legend(loc='upper right')
        plt.grid(True)
        if save:
            plt.savefig(save_path_reconstructed)
        else:
            plt.show()
        plt.close()
    

def plot_stacked_images(real_images, reconstructed_images_list, generated_images_list, model_names, save=True, save_path='./generator/eval/stacked_images.pdf'):
    print("Plotting stacked images")
    
    # Only keep the first occurrence of each model
    seen_models = set()
    unique_indices = []
    for i, name in enumerate(model_names):
        if name not in seen_models:
            seen_models.add(name)
            unique_indices.append(i)
    
    # Update model names and generated images list; handle reconstructed_images_list if provided
    model_names = [model_names[i] for i in unique_indices]
    generated_images_list = [generated_images_list[i] for i in unique_indices]
    if reconstructed_images_list is not None:
        reconstructed_images_list = [reconstructed_images_list[i] for i in unique_indices]
    
    num_models = len(generated_images_list)
    
    # Determine the number of images to use across available lists
    if reconstructed_images_list is not None:
        num_images_to_use = min(len(real_images), *[len(gen_imgs) for gen_imgs in generated_images_list], *[len(rec_imgs) for rec_imgs in reconstructed_images_list])
    else:
        num_images_to_use = min(len(real_images), *[len(gen_imgs) for gen_imgs in generated_images_list])
    
    # Subset the images
    real_images = real_images[:num_images_to_use]
    generated_images_list = [gen_imgs[:num_images_to_use] for gen_imgs in generated_images_list]
    if reconstructed_images_list is not None:
        reconstructed_images_list = [rec_imgs[:num_images_to_use] for rec_imgs in reconstructed_images_list]
    
    # Calculate average pixel intensities
    real_avg_intensity = np.mean([img.cpu().numpy().squeeze() for img in real_images], axis=0)
    generated_avg_intensities = [np.mean([img.cpu().numpy().squeeze() for img in gen_imgs], axis=0) for gen_imgs in generated_images_list]
    if reconstructed_images_list is not None:
        reconstructed_avg_intensities = [np.mean([img.cpu().numpy().squeeze() for img in rec_imgs], axis=0) for rec_imgs in reconstructed_images_list]
    
    # Determine vmin and vmax based on available data
    if reconstructed_images_list is not None:
        all_intensities = [real_avg_intensity] + generated_avg_intensities + reconstructed_avg_intensities
    else:
        all_intensities = [real_avg_intensity] + generated_avg_intensities
    vmin = min(np.min(intensity) for intensity in all_intensities)
    vmax = max(np.max(intensity) for intensity in all_intensities)
    
    # Set up the figure layout: use three rows if reconstructions exist; otherwise two rows
    if reconstructed_images_list is not None:
        nrows = 3
        row_titles = ['Original', 'Reconstructed', 'Generated']
        fig, axes = plt.subplots(nrows, num_models, figsize=(num_models * 2.3, 6), 
                                 gridspec_kw={'height_ratios': [2, 2, 2], 'wspace': 0, 'hspace': 0})
    else:
        nrows = 2
        row_titles = ['Original', 'Generated']
        fig, axes = plt.subplots(nrows, num_models, figsize=(num_models * 2.3, 4), 
                                 gridspec_kw={'wspace': 0, 'hspace': 0})
    
    # If only one model, ensure axes is 2D
    if num_models == 1:
        axes = np.expand_dims(axes, axis=1)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the real images' average intensity in the top row
    im = axes[0, 0].imshow(real_avg_intensity, cmap='viridis', norm=norm)
    axes[0, 0].axis('off')
    # Remove extra plots in the first row if there are more columns
    for i in range(1, num_models):
        axes[0, i].remove()
    
    # Plot the remaining rows based on availability
    if reconstructed_images_list is not None:
        for i, (rec_avg, gen_avg, name) in enumerate(zip(reconstructed_avg_intensities, generated_avg_intensities, model_names)):
            axes[1, i].imshow(rec_avg, cmap='viridis', norm=norm)
            axes[1, i].axis('off')
            axes[2, i].imshow(gen_avg, cmap='viridis', norm=norm)
            axes[2, i].axis('off')
            axes[2, i].text(0.5, -0.15, name, va='center', ha='center', fontsize=10, transform=axes[2, i].transAxes)
        # Add row titles
        axes[0, 0].text(-0.4, 0.5, row_titles[0], va='center', ha='center', fontsize=12, rotation=90, transform=axes[0, 0].transAxes)
        axes[1, 0].text(-0.4, 0.5, row_titles[1], va='center', ha='center', fontsize=12, rotation=90, transform=axes[1, 0].transAxes)
        axes[2, 0].text(-0.4, 0.5, row_titles[2], va='center', ha='center', fontsize=12, rotation=90, transform=axes[2, 0].transAxes)
    else:
        for i, (gen_avg, name) in enumerate(zip(generated_avg_intensities, model_names)):
            axes[1, i].imshow(gen_avg, cmap='viridis', norm=norm)
            axes[1, i].axis('off')
            axes[1, i].text(0.5, -0.15, name, va='center', ha='center', fontsize=10, transform=axes[1, i].transAxes)
        # Add row titles for the two rows
        axes[0, 0].text(-0.4, 0.5, row_titles[0], va='center', ha='center', fontsize=12, rotation=90, transform=axes[0, 0].transAxes)
        axes[1, 0].text(-0.4, 0.5, row_titles[1], va='center', ha='center', fontsize=12, rotation=90, transform=axes[1, 0].transAxes)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Average Pixel Intensity')
    
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_intensity_histograms_individually(
    real_intensity_sum, all_produced_sums, combined_intensities, model_names, title,
    bottomx=9e-4, xlabel='Intensity sum', bins=None, bin_split_value=None, RMAE=True,
    save=True, save_path='./generator/eval/summedintensityhist.pdf', 
    FIXED_SCALE=False, fixed_xlim=(0, 1), fixed_ylim=(1e-6, 1), 
    universallinestyle="-", models_to_plot = model_names, 
    legend_location = 'upper right'):
    # Initialize the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Define custom histogram bins if bin_split_value is provided
    if bins is None:
        if bin_split_value is not None:
            bin_edges_before = np.histogram_bin_edges(combined_intensities[combined_intensities <= bin_split_value], bins=15)
            bin_edges_after = np.histogram_bin_edges(combined_intensities[combined_intensities > bin_split_value], bins=7)
            bin_edges_combined = np.unique(np.concatenate([bin_edges_before, bin_edges_after]))
            bins = bin_edges_combined
        else:
            bins = np.histogram_bin_edges(combined_intensities, bins=30)

    # Calculate histogram and plot for real data
    real_hist, _ = np.histogram(real_intensity_sum, bins=bins, density=True)
    real_hist /= np.sum(real_hist)
    real_std = np.sqrt(real_hist / len(real_intensity_sum))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    capsize = 6

    ax1.step(bins, np.append(real_hist, real_hist[-1]), where='post', color='black', linestyle=universallinestyle, linewidth=3, label=f'Real (n={len(real_intensity_sum)})')
    ax1.errorbar(bin_centers, real_hist, yerr=real_std, fmt=' ', capsize=capsize, color='black', alpha=0.7, elinewidth=1.5)

    # Plot histograms for generated/reconstructed images
    for encoder in models_to_plot:
        if encoder not in all_produced_sums:
            continue

        concatenated_sums = all_produced_sums[encoder]
        gen_hist, _ = np.histogram(concatenated_sums, bins=bins, density=True)
        gen_hist /= np.sum(gen_hist)
        gen_std = np.sqrt(gen_hist / len(concatenated_sums))
        if RMAE:
            rmae_gen = calculate_rmae(real_hist, gen_hist)
        
        ax1.step(bins, np.append(gen_hist, gen_hist[-1]), where='post', color=colors[encoder],
                 linestyle=universallinestyle,
                 label=f'{encoder} (n={len(concatenated_sums)}, RMAE={rmae_gen:.3g})' if RMAE else f'{encoder} (n={len(concatenated_sums)})',
                 alpha=0.7, linewidth=3)
        ax1.errorbar(bin_centers, gen_hist, yerr=gen_std, fmt=' ', capsize=capsize, color=colors[encoder], alpha=0.7, elinewidth=1.5)

        # Calculate relative error for all model_names and plot
        relative_error = 2 * (real_hist - gen_hist) / (real_hist + gen_hist + 1e-10)
        ax2.step(bins, np.append(relative_error, relative_error[-1]), where='post', color=colors[encoder],
                 linestyle=universallinestyle, alpha=0.7, linewidth=2)

    # Apply fixed scale if enabled
    if FIXED_SCALE:
        ax1.set_xlim(fixed_xlim)
        ax1.set_ylim(fixed_ylim)
        ax2.set_xlim(fixed_xlim)
        ax2.set_ylim(-1, 1)  # Relative error fixed range

    # Set axis labels, scales, and legend
    ax1.set_yscale('log')
    ax1.set_ylabel('Frequency', fontsize=18)
    ax1.legend(loc=legend_location, fontsize=16)
    ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=16)

    ax2.set_ylabel(r'$ 2 \times \frac{(\mathrm{Real} - \mathrm{Gen})}{\mathrm{Real} + \mathrm{Gen}}$', fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.grid(True)
    ax2.tick_params(axis='both', labelsize=16)

    # Set figure title and save/show
    #fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()



def plot_latent_spaces(latent_representations, model_names, default_perplexity=30, learning_rate=200, n_iter=1000, save_path=None):
    # Only keep the first occurrence of each model
    seen_models = set()
    unique_indices = []
    for i, name in enumerate(model_names):
        if name not in seen_models:
            seen_models.add(name)
            unique_indices.append(i)
    
    # Update the model names and latent representations
    model_names = [model_names[i] for i in unique_indices]
    
    # Filter latent representations to ensure they exist for the unique indices
    latent_representations = [latent_representations[i] for i in unique_indices if i < len(latent_representations)]
    
    num_models = len(latent_representations)
    
    if num_models == 1:
        fig, axes = plt.subplots(1, num_models, figsize=(15, 5))
        axes = [axes]  # Convert single Axes object to a list
    else:
        fig, axes = plt.subplots(1, num_models, figsize=(15, 5))

    sc = None  # Initialize `sc` to None

    for i, (latents, name) in enumerate(zip(latent_representations, model_names)):
        # Check for NaN or Inf values in latents
        if not np.all(np.isfinite(latents.cpu().numpy())):
            print(f"Skipping {name} due to invalid latent representation values (NaN or Inf found).")
            continue

        # Ensure latents is a 2D array
        if len(latents.shape) == 1:
            latents = latents.reshape(1, -1)
        elif len(latents.shape) > 2:
            latents = latents.reshape(latents.shape[0], -1)

        # Adjust perplexity to avoid issues
        perplexity = min(default_perplexity, latents.shape[0] - 1)
        
        tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, max_iter=n_iter, random_state=42)
        tsne_results = tsne.fit_transform(latents)
        
        sc = axes[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=latents[:, 0], cmap='viridis', alpha=0.6)
        axes[i].set_title(name)
        axes[i].axis('off')

    # Only add a colorbar if `sc` has been assigned (i.e., at least one plot was made)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Latent Space Value')

    plt.savefig(save_path, dpi=300, bbox_inches='tight') if save_path else plt.show()
    
def plot_latent_distributions_for_each_model(latent_representations, model_names, save=True, save_dir='./latent_distributions'):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for i, latents in enumerate(latent_representations):
        # Convert to numpy if it's a tensor
        latents = latents.cpu().numpy()
        latent_dim = latents.shape[1]
        
        fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(20, 20))
        
        for j, ax in enumerate(axes.flat):
            if j < latent_dim:
                latent_values = latents[:, j]
                latent_values = latent_values[np.isfinite(latent_values)]
                if len(latent_values) == 0:
                    ax.axis('off')
                    continue
                ax.hist(latent_values, bins=30, color='blue', alpha=0.7)
                ax.set_title(f'Latent Variable {j+1}')
                if np.isfinite(latents).all():
                    ax.set_xlim(min(latents.min(), -3), max(latents.max(), 3))
                else:
                    ax.set_xlim(-3, 3)
            else:
                ax.axis('off')
        
        plt.tight_layout()
        if save:
            model_save_path = os.path.join(save_dir, f'{model_names[i]}.pdf')
            plt.savefig(model_save_path)
        else:
            plt.show()
        plt.close()

    
def plot_latent_distributions(latent_representations, save=True, save_path='./latent_distributions.pdf'):
    latent_representations = latent_representations.cpu().numpy()
    latent_dim = latent_representations.shape[1]
    
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < latent_dim:
            latent_values = latent_representations[:, i]
            latent_values = latent_values[np.isfinite(latent_values)]
            if len(latent_values) == 0:
                ax.axis('off')
                continue
            ax.hist(latent_values, bins=30, color='blue', alpha=0.7)
            ax.set_title(f'Latent Variable {i+1}')
            if np.isfinite(latent_representations).all():
                ax.set_xlim(min(latent_representations.min(), -3), max(latent_representations.max(), 3))
            else:
                ax.set_xlim(-3, 3)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_latent_cross_correlation(latent_representations, model_names, save_dir=f'./latent/'):
    start_time = time.time()
    for latents, model_name in zip(latent_representations, model_names):
            # Calculate the correlation matrix
            correlation_matrix = np.corrcoef(latents, rowvar=False)
            df = pd.DataFrame(correlation_matrix)
            
            # Create a mask to show only the lower triangle
            mask = np.triu(np.ones_like(df, dtype=bool))

            plt.figure(figsize=(20, 16))  # Increase the figure size for readability
            sns.heatmap(
                df, mask=mask, cmap="coolwarm", cbar=True, vmin=-1, vmax=1,
                square=True, xticklabels=10, yticklabels=10,  # Display every 10th label only
                annot=False  # Disable cell annotations to reduce clutter
            )
            plt.title(f"Cross-Correlation between Latent Dimensions for {model_name}", fontsize=18)
            plt.xlabel("Latent Dimensions", fontsize=14)
            plt.ylabel("Latent Dimensions", fontsize=14)
            
            # Save the plot
            plot_save_path = os.path.join(save_dir, f'full_cross_correlation_{model_name}.pdf')
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Function plot_latent_cross_correlation executed in {time.time() - start_time:.2f} seconds")




def plot_covariance(latent_vectors, num_dimensions=5, save=True, save_path='./covariance.pdf'):
    """
    Plots the covariance matrix for the first `num_dimensions` latent dimensions.

    Args:
        latent_vectors (np.ndarray): A 2D array of latent vectors (samples x dimensions).
        num_dimensions (int): The number of latent dimensions to include in the covariance matrix.
    """
    if latent_vectors.shape[1] < num_dimensions:
        raise ValueError(f"The latent vectors have only {latent_vectors.shape[1]} dimensions, "
                         f"but you requested {num_dimensions} dimensions.")

    # Select the first `num_dimensions` dimensions
    latent_subset = latent_vectors[:, :num_dimensions]

    # Compute the covariance matrix
    covariance_matrix = np.cov(latent_subset, rowvar=False)

    # Plot the covariance matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(covariance_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Covariance')
    plt.title(f"Covariance Matrix for First {num_dimensions} Latent Dimensions", fontsize=16)
    plt.xlabel('Latent Dimension', fontsize=14)
    plt.ylabel('Latent Dimension', fontsize=14)
    plt.xticks(ticks=np.arange(num_dimensions), labels=[f"Dim {i+1}" for i in range(num_dimensions)])
    plt.yticks(ticks=np.arange(num_dimensions), labels=[f"Dim {i+1}" for i in range(num_dimensions)])
    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()


def plot_interpolated_images(interpolated_images, save=True, save_path='./interpolation_graph.pdf'):
    num_steps = len(interpolated_images)
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()
    
def calculate_statistics(speed_data):
    means = speed_data.mean(axis=0)
    stds = speed_data.std(axis=0)
    return means, stds


def plot_generation_speed(models, save=True, save_path='./generator/eval/speedhist.pdf'):
    start_time = time.time()
    model_dict = {}

    # Parse the log files and calculate training speeds
    for model_info in models:
        if os.path.exists(model_info["path"]):
            scatter_time_1, scatter_time_2, train_time, epochs = parse_log_file(model_info["path"])
            model_name = model_info["name"]
            num_galaxies = model_info.get("num_galaxies", 1)  # Default to 1 if not provided
            
            if epochs > 0 and train_time > 0:  # Prevent division by zero
                train_speed = num_galaxies / (train_time * epochs)
                
                # Add the train speed to the respective model name
                if model_name not in model_dict:
                    model_dict[model_name] = []
                model_dict[model_name].append(train_speed)

    # Prepare data for plotting
    model_names = list(model_dict.keys())
    mean_speeds = [np.mean(model_dict[name]) for name in model_names]
    std_speeds = [np.std(model_dict[name]) for name in model_names]

    # Plotting
    plt.figure(figsize=(10, 6))
    index = np.arange(len(model_names))

    # Plot bars with error bars
    plt.bar(index, mean_speeds, yerr=std_speeds, capsize=5, label='Training Speed')

    plt.xlabel('Model')
    plt.ylabel('Images per Second per Epoch')
    plt.title('Model Training Speeds with Mean and Standard Deviation')
    plt.xticks(index, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    print(f"Function plot_generation_speed executed in {time.time() - start_time:.2f} seconds")
    

def old_plot_model_times_with_error_bars(models, save=True, save_path='./generator/eval/time_with_errors.pdf'):
    # Dictionaries to store times by model name
    scatter_times_1 = {}
    scatter_times_2 = {}
    train_times = {}
    
    # Collecting data for each model, grouping by model name
    for model_info in models:
        if os.path.exists(model_info["path"]):
            scatter_time_1, scatter_time_2, train_time, epochs = parse_log_file(model_info["path"])            
            model_name = model_info["name"]
            
            # Initialize lists for each model name if not already present
            if model_name not in scatter_times_1:
                scatter_times_1[model_name] = []
                scatter_times_2[model_name] = []
                train_times[model_name] = []
            
            # Append the times for each run to the corresponding model
            scatter_times_1[model_name].append(scatter_time_1)
            scatter_times_2[model_name].append(scatter_time_2)
            train_times[model_name].append(train_time)
    
    # Prepare data for plotting: calculate means and standard deviations
    avg_scatter_times_1 = []
    std_scatter_times_1 = []
    avg_scatter_times_2 = []
    std_scatter_times_2 = []
    avg_train_times = []
    std_train_times = []
    model_names = list(scatter_times_1.keys())

    for model_name in model_names:
        avg_scatter_times_1.append(np.mean(scatter_times_1[model_name]))
        std_scatter_times_1.append(np.std(scatter_times_1[model_name]))
        
        avg_scatter_times_2.append(np.mean(scatter_times_2[model_name]))
        std_scatter_times_2.append(np.std(scatter_times_2[model_name]))
        
        avg_train_times.append(np.mean(train_times[model_name]))
        std_train_times.append(np.std(train_times[model_name]))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    index = np.arange(len(model_names))

    # Plot with error bars
    ax.bar(index - bar_width, avg_scatter_times_1, bar_width, yerr=std_scatter_times_1, label='Train Scattering Time', capsize=5, color='blue')
    ax.bar(index, avg_scatter_times_2, bar_width, yerr=std_scatter_times_2, label='Test Scattering Time', capsize=5, color='orange')
    ax.bar(index + bar_width, avg_train_times, bar_width, yerr=std_train_times, label='Training Time', capsize=5, color='green')

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Model Times')
    ax.set_xticks(index)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    
def plot_model_times_with_error_bars(models, save=True, save_path='./generator/eval/time_with_errors.pdf'):
    # Dictionaries to store times by model name
    scatter_times_1 = {}
    scatter_times_2 = {}
    train_times = {}
    
    # Collecting data for each model, grouping by model name
    for model_info in models:
        if os.path.exists(model_info["path"]):
            scatter_time_1, scatter_time_2, train_time, epochs = parse_log_file(model_info["path"])            
            model_name = model_info["name"]
            
            # Initialize lists for each model name if not already present
            if model_name not in scatter_times_1:
                scatter_times_1[model_name] = []
                scatter_times_2[model_name] = []
                train_times[model_name] = []
            
            # Append the times for each run to the corresponding model
            scatter_times_1[model_name].append(scatter_time_1)
            scatter_times_2[model_name].append(scatter_time_2)
            train_times[model_name].append(train_time)
    
    # Prepare data for plotting: calculate means and standard deviations
    avg_scatter_times_1 = []
    std_scatter_times_1 = []
    avg_scatter_times_2 = []
    std_scatter_times_2 = []
    avg_train_times = []
    std_train_times = []
    model_names = list(scatter_times_1.keys())

    for model_name in model_names:
        avg_scatter_times_1.append(np.mean(scatter_times_1[model_name]))
        std_scatter_times_1.append(np.std(scatter_times_1[model_name]))
        
        avg_scatter_times_2.append(np.mean(scatter_times_2[model_name]))
        std_scatter_times_2.append(np.std(scatter_times_2[model_name]))
        
        avg_train_times.append(np.mean(train_times[model_name]))
        std_train_times.append(np.std(train_times[model_name]))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    index = np.arange(len(model_names))

    # Plot with error bars
    ax.bar(index - bar_width, avg_scatter_times_1, bar_width, yerr=std_scatter_times_1, label='Train Scattering Time', capsize=10, color='blue')
    ax.bar(index, avg_scatter_times_2, bar_width, yerr=std_scatter_times_2, label='Test Scattering Time', capsize=10, color='orange')
    ax.bar(index + bar_width, avg_train_times, bar_width, yerr=std_train_times, label='Training Time', capsize=10, color='green')

    # Set font sizes for maximum readability
    ax.set_xlabel('Model', fontsize=18)
    ax.set_ylabel('Time (seconds)', fontsize=18)
    ax.set_title('Model Times', fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(model_names, rotation=45, fontsize=16)
    
    # Update legend font size
    ax.legend(fontsize=16)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()


def plot_image_grid(images, img_shape, nrow=10, ncol=10, title='Image grid', savepath=f'{save_dir}/image_grid.pdf', cmap='viridis'):
    # Plot a grid of images with nrow x ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    axs = axs.ravel()
    
    for i in range(nrow * ncol):
        image = images[i].view(*img_shape).cpu().numpy().squeeze()
        axs[i].imshow(image, cmap=cmap)
        axs[i].axis('off')
        axs[i].set_aspect('auto')  # Ensure the image fits tightly in each subplot

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title, fontsize=16)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



# Plot generated vs original images
if not USE_GAN:
    vae_multmod(
        test_images[:num_display],
        reconstructed_images_list,
        vae_model_names,  # Use filtered names here
        save=True,
        save_path=f"{save_dir}/reconstructed.pdf",
        show_title=False,
        show_originals=True
    )
    # Ensure tensors are moved to CPU and then convert to numpy for shape printing
vae_multmod(old_images=None, generated_images=generated_images_list, model_names=model_names,
            save=True, save_path=f"{save_dir}/generated.pdf", show_title=False, show_originals=False)

# SIMILAR GENERATED IMAGES AND ORIGINAL IMAGES
most_similar_images_list = []
for generated_images in generated_images_list:
    most_similar_images = find_most_similar_images(test_images[:num_display], generated_images)
    most_similar_images_list.append(most_similar_images)
vae_multmod(test_images[:num_display], most_similar_images_list, model_names,
            save=True, save_path=f"{save_dir}/generated_similar.pdf", show_title=False, show_originals=True)

#COMPONENT PEAK STATISTICS
# Initialize empty lists to store peak intensity statistics for all model_names
all_peak_intensity_stats_rec = []
all_peak_intensity_stats_gen = []

#plot_asymmetry_vs_mse(test_images, reconstructed_images, save_path=f"{save_dir}/asymmetry_{encoder_name}.pdf")

# Loop through each encoder and accumulate the peak statistics
peaks_info_real = get_main_and_secondary_peaks_with_locations(test_images, threshold=0.1)
plot_images_with_peaks(test_images, peaks_info_real, save_path=f'{save_dir}/peak_locations/real.pdf')
for i, encoder_name in enumerate(model_names):
    generated_images = generated_images_list[i]
    peaks_info_gen = get_main_and_secondary_peaks_with_locations(generated_images)
    all_peak_intensity_stats_gen.append(peaks_info_gen)     # Accumulate the statistics for the reconstructed and generated images for all model_names
    plot_asymmetry_score_histogram(test_images, num_bins=21, save_path=f"{save_dir}/asymmetry/histogram_{encoder_name}.pdf")
    plot_images_with_peaks(generated_images, peaks_info_gen, save_path=f'{save_dir}/peak_locations/generations_{encoder_name}.pdf')
    plot_image_grid(generated_images, img_shape, 10, 10, title="Generations", savepath=f"{save_dir}/generations/{encoder_name}_generations.pdf")
    #plot_covariance(latent_representations[i], num_dimensions=5, save_path=f"{save_dir}/latent/{encoder_name}_covariance.pdf")

    if not USE_GAN and encoder_name not in ['GAN']:
        reconstructed_images = reconstructed_images_list[i]
        peaks_info_rec = get_main_and_secondary_peaks_with_locations(reconstructed_images)
        all_peak_intensity_stats_rec.append(peaks_info_rec)
        plot_asymmetry_vs_mse(test_images, reconstructed_images, encoder_name, save_path=f"{save_dir}/asymmetry/MSE_vs_asym_{encoder_name}.pdf")
        plot_images_by_asymmetry_bins(test_images, reconstructed_images, save_path=f'{save_dir}/asymmetry/reconstructions_{encoder_name}.pdf')
        plot_images_with_peaks(reconstructed_images, peaks_info_rec, save_path=f'{save_dir}/peak_locations/reconstructions_{encoder_name}.pdf')


# Plot primary and secondary peak statistics for all model_names together
if not USE_GAN:
    plot_peak_statistics(peaks_info_real, all_peak_intensity_stats_rec, vae_model_names, 
                     title='Primary and Secondary Peak Intensities (Reconstructed vs Real)',
                     save_path=f'{save_dir}/peak_statistics/values_recon.pdf')
plot_peak_statistics(peaks_info_real, all_peak_intensity_stats_gen, model_names, 
                     title='Primary and Secondary Peak Intensities (Generated vs Real)',
                     save_path=f'{save_dir}/peak_statistics/values_gen.pdf')

# Plot distribution of primary-to-secondary peak intensity ratios for all model_names
if not USE_GAN:
    plot_peak_ratio_distribution(peaks_info_real, all_peak_intensity_stats_rec, vae_model_names, 
                                save_path=f'{save_dir}/peak_statistics/ratio_dist_recon.pdf')
plot_peak_ratio_distribution(peaks_info_real, all_peak_intensity_stats_gen, model_names, 
                             save_path=f'{save_dir}/peak_statistics/ratio_dist_gen.pdf')

# Plot histograms for primary and secondary peaks for all model_names together
if not USE_GAN:
    plot_peak_histograms_combined(peaks_info_real, all_peak_intensity_stats_rec, vae_model_names, prodtype='Reconstructed',
                        save_path=f'{save_dir}/peak_statistics/histograms_recon.pdf')
plot_peak_histograms_combined(peaks_info_real, all_peak_intensity_stats_gen, model_names, prodtype='Generated',
                     save_path=f'{save_dir}/peak_statistics/histograms_gen.pdf')

# Plot heatmap for primary-to-secondary peak intensity ratios for all model_names together
if not USE_GAN: 
    plot_peak_ratio_heatmap(peaks_info_real, all_peak_intensity_stats_rec, vae_model_names, prodtype='Reconstructed',
                            save_path=f'{save_dir}/peak_statistics/ratio_rec_heatmap')
plot_peak_ratio_heatmap(peaks_info_real, all_peak_intensity_stats_gen, model_names, prodtype='Generated',
                        save_path=f'{save_dir}/peak_statistics/ratio_gen_heatmap')


# PIXEL DISTRIBUTIONS
# Calculate summed intensities and plot
if not USE_GAN: 
    real_intensity_sum, all_reconstructed_sums, real_rec_combinations = calculate_summed_intensities(test_images, reconstructed_images_list, vae_model_names)
    real_intensity_sum = np.array(real_intensity_sum, dtype=np.float32)
    max_real = np.max(real_intensity_sum)
    print("Shape of real_intensity_sum:", np.shape(real_intensity_sum))
    print("Length of all_reconstructed_sums:", len(all_reconstructed_sums))
    print("Lengths of individual arrays in all_reconstructed_sums:", [len(arr) for arr in all_reconstructed_sums])
    print("Max of real_intensity_sum: ", max(real_intensity_sum))
    plot_intensity_histograms_individually(
        real_intensity_sum,
        all_reconstructed_sums,
        real_rec_combinations,
        vae_model_names,
        title=f"Reconstructions of {galaxy_class} coded {num_galaxies}",
        xlabel='Accumulated pixel intensity per image',
        FIXED_SCALE=True,
        fixed_xlim=(0, max_real+30),
        fixed_ylim=(9e-4, 1),
        save_path=f"{save_dir}/summedintensitydist_recon.pdf"
    )
real_intensity_sum, all_generated_sums, real_gen_combinations = calculate_summed_intensities(test_images, generated_images_list, model_names)
real_intensity_sum = np.array(real_intensity_sum, dtype=np.float32)
max_real = np.max(real_intensity_sum)
plot_intensity_histograms_individually(
    real_intensity_sum,
    all_generated_sums,
    real_gen_combinations,
    model_names,
    title=f"Generations of {galaxy_class} coded {num_galaxies}",
    xlabel='Accumulated pixel intensity per image',
    FIXED_SCALE=True,
    fixed_xlim=(0, max_real+30),
    fixed_ylim=(9e-4, 1),
    save_path=f"{save_dir}/summedintensitydist_gen.pdf"
)

# For each encoder, calculate Bhattacharyya coefficient, TVD, and histogram intersection
print("Calculating Bhattacharyya coefficient, TVD, and histogram intersection for summed intensities...")
for i, encoder in enumerate(model_names):
    #generated_intensity_sum = all_generated_sums[i] if len(all_generated_sums) > 1 else all_generated_sums[encoder]
    generated_intensity_sum = all_generated_sums[encoder]
    bhattacharyya = bhattacharyya_coefficient(real_intensity_sum, generated_intensity_sum)
    tvd = total_variation_distance(real_intensity_sum, generated_intensity_sum)
    intersection = histogram_intersection(real_intensity_sum, generated_intensity_sum)
    print(f"Model: {encoder}, Bhattacharyya: {bhattacharyya:.4f}, TVD: {tvd:.4f}, Intersection: {intersection:.4f}")


# Calculate peak intensities and plot
if not USE_GAN:
    real_peak_intensity, all_reconstructed_peaks, real_rec_combinations = calculate_peak_intensities(test_images, reconstructed_images_list, vae_model_names)
    plot_intensity_histograms_individually(
        real_peak_intensity,
        all_reconstructed_peaks,
        real_rec_combinations,
        vae_model_names,
        title=f"Reconstructions of {galaxy_class} coded {num_galaxies}",
        xlabel='Intensity for the brightest pixel in each image',
        FIXED_SCALE=True,
        fixed_xlim=(0, 1),
        fixed_ylim=(9e-4, 1),
        save_path=f"{save_dir}/peakintensitydist_recon.pdf",
        legend_location='upper left'
    )
real_peak_intensity, all_generated_peaks, real_gen_combinations = calculate_peak_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms_individually(
    real_peak_intensity,
    all_generated_peaks,
    real_gen_combinations,
    model_names,
    title=f"Generations of {galaxy_class} coded {num_galaxies}",
    xlabel='Intensity for the brightest pixel in each image',
    FIXED_SCALE=True,
    fixed_xlim=(0, 1),
    fixed_ylim=(9e-4, 1),
    save_path=f"{save_dir}/peakintensitydist_gen.pdf",
    legend_location='upper left'
)

# For each encoder, calculate Bhattacharyya coefficient, TVD, and histogram intersection
print("Calculating Bhattacharyya coefficient, TVD, and histogram intersection for peak intensities...")
for i, encoder in enumerate(model_names):
    generated_peak_intensity = all_generated_peaks[encoder]
    bhattacharyya = bhattacharyya_coefficient(real_peak_intensity, generated_peak_intensity)
    tvd = total_variation_distance(real_peak_intensity, generated_peak_intensity)
    intersection = histogram_intersection(real_peak_intensity, generated_peak_intensity)
    print(f"Model: {encoder}, Bhattacharyya: {bhattacharyya:.4f}, TVD: {tvd:.4f}, Intersection: {intersection:.4f}")


# Calculate total intensities and plot
if not USE_GAN:
    real_total_intensity, all_reconstructed_total_intensities, real_rec_combinations = calculate_total_intensities(test_images, reconstructed_images_list, vae_model_names)
    plot_intensity_histograms_individually(
        real_total_intensity,
        all_reconstructed_total_intensities,
        real_rec_combinations,
        vae_model_names,
        title=f"Reconstructions of {galaxy_class} coded {num_galaxies}",
        xlabel='Intensity for each generated pixel',
        FIXED_SCALE=True,
        fixed_xlim=(0, 1),
        fixed_ylim=(9e-9, 1),
        save_path=f"{save_dir}/totalintensitydist_recon.pdf",
        legend_location='lower left'
    )
real_total_intensity, all_generated_total_intensities, real_gen_combinations = calculate_total_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms_individually(
    real_total_intensity,
    all_generated_total_intensities,
    real_gen_combinations,
    model_names,
    title=f"Generations of {galaxy_class} coded {num_galaxies}",
    xlabel='Intensity for each generated pixel',
    FIXED_SCALE=True,
    fixed_xlim=(0, 1),
    fixed_ylim=(9e-9, 1),
    save_path=f"{save_dir}/totalintensitydist_gen.pdf",
    legend_location='lower left'
)

# For each encoder, calculate Bhattacharyya coefficient, TVD, and histogram intersection
print("Calculating Bhattacharyya coefficient, TVD, and histogram intersection for total intensities...")
for i, encoder in enumerate(model_names):
    generated_total_intensity = all_generated_total_intensities[encoder]
    bhattacharyya = bhattacharyya_coefficient(real_total_intensity, generated_total_intensity)
    tvd = total_variation_distance(real_total_intensity, generated_total_intensity)
    intersection = histogram_intersection(real_total_intensity, generated_total_intensity)
    print(f"Model: {encoder}, Bhattacharyya: {bhattacharyya:.4f}, TVD: {tvd:.4f}, Intersection: {intersection:.4f}")



# RADIAL INTENSITY COMPARISON
original_radial_intensity = compute_radial_intensity(test_images[:num_display], img_shape)
models_gen_radial_intensity, models_rec_radial_intensity = [], []
for i, generated_images in enumerate(generated_images_list):
    radial_intensity = compute_radial_intensity(generated_images, img_shape)
    models_gen_radial_intensity.append(radial_intensity)
if not USE_GAN:
    for i, reconstructed_images in enumerate(reconstructed_images_list):
        radial_intensity = compute_radial_intensity(reconstructed_images, img_shape)
        models_rec_radial_intensity.append(radial_intensity)    
title = f'Radial Intensity of {galaxy_class}'
plot_radial_intensity(original_radial_intensity, models_rec_radial_intensity, models_gen_radial_intensity, model_names, title, save_path_generated=f"{save_dir}/radial_generated.pdf", save_path_reconstructed=f"{save_dir}/radial_reconstructed.pdf")
plot_stacked_images(test_images, reconstructed_images_list=None, generated_images_list=generated_images_list, model_names=model_names, save_path=f"{save_dir}/stacked_images.pdf")
plot_generation_speed(models, save_path=f'{save_dir}/speedhist.pdf')
plot_model_times_with_error_bars(models, save_path=f'{save_dir}/timehist.pdf')

# For each encoder, calculate Bhattacharyya coefficient, TVD, and histogram intersection for radial intensities
print("Calculating Bhattacharyya coefficient, TVD, and histogram intersection for radial intensities...")
for i, encoder in enumerate(model_names):
    generated_radial_intensity = models_gen_radial_intensity[i]
    bhattacharyya = bhattacharyya_coefficient(original_radial_intensity, generated_radial_intensity)
    tvd = total_variation_distance(original_radial_intensity, generated_radial_intensity)
    intersection = histogram_intersection(original_radial_intensity, generated_radial_intensity)
    print(f"Model: {encoder}, Bhattacharyya: {bhattacharyya:.4f}, TVD: {tvd:.4f}, Intersection: {intersection:.4f}")

# LATENT SPACE
if not USE_GAN:
    latent_representations_combined = torch.cat(latent_representations, dim=0)
    plot_latent_spaces(latent_representations, vae_model_names, save_path=f'{save_dir}/latent_spaces.pdf')
    plot_latent_distributions_for_each_model(latent_representations, vae_model_names, save=True, save_dir=f'{save_dir}/latent')
    plot_latent_cross_correlation(latent_representations, vae_model_names, save_dir=f'{save_dir}/latent')
else:
    print("Skipping latent space plots because GAN is used.")

print("Plots generated and saved.")
