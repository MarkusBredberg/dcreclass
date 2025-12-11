#!/usr/bin/env python3
"""
Script to load a small batch of galaxy images, compute scattering metrics,
optimize a random image to match the original image’s scattering coefficients,
and then plot a 2x3 figure for each image showing:
    1. Original image.
    2. Pixel-by-pixel recreated image.
    3. Raw scattering coefficients.
    4. lavg scattering coefficients.
    5. ldiff scattering coefficients.
    6. PCA reduced scattering coefficients.
"""
import torch
import matplotlib.pyplot as plt
from kymatio.torch import Scattering2D
from utils.scatter_reduction import lavg, ldiff, pca
from utils.data_loader import load_galaxies, get_classes

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Scattering parameters (should match your reduction function defaults)
J = 3          # number of scales
L = 8          # number of orientations
max_order = 2  # scattering order

# Image shape and sample configuration
img_shape = (1, 128, 128)   # (channels, height, width)
num_sample_images = 4       # number of images to ultimately process/display

# IMPORTANT: Use a larger sample_size for the data loader so that after
# augmentation/outlier removal at least one image remains.
sample_size = 10

REMOVEOUTLIERS = True
fold = 0
galaxy_class = 11

# --- DATA LOADING ---
print("Loading galaxy data ...")
try:
    data = load_galaxies(galaxy_class=galaxy_class,
                         fold=fold,
                         img_shape=img_shape,
                         sample_size=sample_size,
                         REMOVEOUTLIERS=REMOVEOUTLIERS,
                         train=True)
except Exception as e:
    print("Error loading galaxies:", e)
    exit(1)

train_images, train_labels, test_images, test_labels = data
print(f"Loaded {train_images.size(0)} training images.")

if train_images.size(0) == 0:
    print("No images were returned by load_galaxies. "
          "Try increasing the sample_size or setting REMOVEOUTLIERS=False.")
    exit(1)

# Use the first few images for demonstration
images = train_images[:num_sample_images].to(device)

# --- SETUP SCATTERING TRANSFORM ---
# Create a scattering transform matching the chosen parameters.
scattering = Scattering2D(J=J, L=L, shape=img_shape[-2:], max_order=max_order)
scattering.to(device)
scattering.eval()  # Set to evaluation mode

# --- MAIN LOOP OVER IMAGES ---
for idx, original_img in enumerate(images):
    print(f"\nProcessing image {idx} ...")
    # Ensure the image has a batch dimension: (1, 1, H, W)
    original_img = original_img.unsqueeze(0)
    
    # Compute target scattering coefficients (detach since these are fixed)
    with torch.no_grad():
        target_scattering = scattering(original_img)
    
    # Initialize a random candidate image (learnable tensor) with the same shape
    candidate_img = torch.nn.Parameter(torch.randn_like(original_img, device=device))
    
    # Setup optimizer and loss function (MSE on scattering coefficients)
    optimizer = torch.optim.Adam([candidate_img], lr=0.01)
    mse_loss = torch.nn.MSELoss()
    
    # Optimize the candidate image so its scattering coefficients match the target's
    num_opt_iterations = 200
    for it in range(num_opt_iterations):
        optimizer.zero_grad()
        candidate_scattering = scattering(candidate_img)
        loss = mse_loss(candidate_scattering, target_scattering)
        loss.backward()
        optimizer.step()
        # Optionally clamp candidate image to [0, 1] to keep pixel values realistic
        with torch.no_grad():
            candidate_img.clamp_(0, 1)
        if (it + 1) % 50 == 0:
            print(f" Iteration {it+1}/{num_opt_iterations}, loss: {loss.item():.6f}")
    
    # --- Compute all metrics from the candidate image ---
    with torch.no_grad():
        cand_scattering = scattering(candidate_img)
        # Force the scattering tensor to have a batch dimension.
        if cand_scattering.dim() == 3:
            cand_scattering = cand_scattering.unsqueeze(0)
        elif cand_scattering.dim() == 4 and cand_scattering.shape[0] != 1:
            cand_scattering = cand_scattering[0].unsqueeze(0)
        cand_scattering_cpu = cand_scattering.cpu()

        print("Shape of scattering coefficients", cand_scattering_cpu.shape)
        
        # Raw scattering coefficients (display the 0th channel for visualization)
        raw_scat = cand_scattering_cpu[0, 0].detach()
        print("Shape of raw_scat: ", raw_scat.shape)
        
        # Compute lavg scattering coefficients (display the first channel)
        lavg_coeffs, _ = lavg(raw_scat, J=J, L=L, m=max_order)
        lavg_img = lavg_coeffs[0].detach().numpy()
        
        # Compute ldiff scattering coefficients (display the first channel)
        ldiff_coeffs, _ = ldiff(cand_scattering_cpu, J=J, L=L, m=max_order)
        ldiff_img = ldiff_coeffs[0].detach().numpy()
        
        # Compute PCA reduced scattering coefficients (display the first component)
        pca_mean, _ = pca(cand_scattering_cpu, J=J, L=L, m=max_order, n_components=5)
        pca_img = pca_mean[0].detach().numpy()
        
        # Candidate image pixel values (squeeze out batch and channel dimensions)
        cand_img_np = candidate_img[0, 0].detach().cpu().numpy()
        # Original image (for comparison)
        orig_img_np = original_img[0, 0].detach().cpu().numpy()
    
    # --- Create a 2x3 figure with subplots ---
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    
    # 1. Original image
    axs[0].imshow(orig_img_np, cmap='gray')
    axs[0].set_title("Original Image")
    
    # 2. Recreated image (pixel-level candidate)
    axs[1].imshow(cand_img_np, cmap='gray')
    axs[1].set_title("Recreated Image")
    
    # 3. Raw scattering coefficients (first channel)
    axs[2].imshow(raw_scat, cmap='viridis')
    axs[2].set_title("Scattering Coeffs")
    
    # 4. lavg scattering coefficients (first channel)
    axs[3].imshow(lavg_img, cmap='viridis')
    axs[3].set_title("lavg Scattering")
    
    # 5. ldiff scattering coefficients (first channel)
    axs[4].imshow(ldiff_img, cmap='viridis')
    axs[4].set_title("ldiff Scattering")
    
    # 6. PCA reduced scattering coefficients (first component)
    axs[5].imshow(pca_img.reshape(1, -1), aspect='auto', cmap='viridis')
    axs[5].set_title("PCA Reduced Scattering")
    
    # Remove axis ticks for a cleaner display
    for ax in axs:
        ax.axis("off")
    
    plt.tight_layout()
    
    # Save the figure (or you can display with plt.show())
    out_filename = f"output_image_{idx}.pdf"
    plt.savefig(out_filename)
    print(f"Saved figure to {out_filename}")
    plt.close(fig)

print("Script complete.")
