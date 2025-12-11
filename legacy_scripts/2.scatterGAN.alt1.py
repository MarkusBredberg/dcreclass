import os
import numpy as np
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from utils.data_loader import load_galaxies
from utils.GAN_models import DCGANDiscriminator, DCGANGenerator, GANDiscriminator, DualInputDiscriminator, load_gan_generator
from utils.plotting import plot_GAN_losses, plot_histograms
from utils.GAN_loss_functions import (
    d_loss_bce, d_loss_hinge, d_loss_lsgan, d_loss_mse, d_loss_wgan,
    g_loss_bce, g_loss_hinge, g_loss_lsgan, g_loss_mse, g_loss_wgan
)
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import random
from lpips import LPIPS
import pandas as pd
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy import linalg
from kymatio.torch import Scattering2D


# -------------------------------
# Fixed parameters for hyperparameter optimization
# -------------------------------
REMOVEOUTLIERS = True  # Sets dv to 'clean' if True and 'full' otherwise
galaxy_class = 10
folds = [5]
batch_size = 256
epochs = 500
image_size = 128
save_interval = 100
output_dir = "GAN"
gan_type = ["Simple", "ScatterDualGAN"][1]
J, L, order = 3, 8, 2 # Scattering parameters


# Grid of hyperparameters to search over
grid_latent_dim = [128]
grid_lr_gen = [1e-3]       # Separate learning rate for generator
grid_lr_disc = [1e-4]      # Separate learning rate for discriminator
grid_gen_loss = ["MSE"]
grid_disc_loss = ["BCE"]   # Discriminator loss
grid_adam_betas = [(0.5, 0.999)]  # Beta values for Adam optimizer
grid_weight_decay = [0.0]         # Weight decay options
grid_label_smoothing = [0.7]        # Label smoothing values for BCE loss
grid_lambda_div = [0.01]        # Weight for diversity loss

#######################

# Set random seed for reproducibility
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

if REMOVEOUTLIERS:
    print("Removing outliers. Dataset will be cleaned")
    data_version = 'clean'
else:
    data_version = 'full'

print("Running ScatterGAN with gan type", gan_type)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "generators"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "discriminators"), exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

fixed_noise = torch.randn(36, grid_latent_dim[0], 1, 1, device=device)

scattering = Scattering2D(J=J, L=L, shape=(image_size, image_size), max_order=order)       
def compute_scattering_coeffs(images, scattering=scattering, batch_size=256, device="cpu"):
    print("Computing scattering coefficients in batches on CPU...")
    scat_coeffs_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            # Force the batch to CPU for scattering transform
            batch = images[i:i + batch_size].to("cpu")
            batch_scat = scattering(batch).detach()  # Compute scattering on CPU
            # Squeeze out the singleton dimension at index 1 if present.
            if batch_scat.shape[1] == 1:
                batch_scat = batch_scat.squeeze(1)  # becomes [B, C, H, W]
            # Move scattering coefficients to target device
            scat_coeffs_list.append(batch_scat.to(device))
        scat_coeffs = torch.cat(scat_coeffs_list, dim=0)
    return scat_coeffs




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
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
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

def save_images_tensorboard(images, file_path, nrow=4):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, file_path)

def compute_lpips_diversity(images, model_name='alex'):
    loss_fn = LPIPS(net=model_name).to(device)
    indices = random.sample(range(len(images)), k=min(100, len(images)))  # sample 100 images
    pairs = [(i, j) for i in indices for j in indices if i < j]
    total_score = 0.0
    for i, j in pairs[:200]:  # compute on at most 200 pairs for speed
        d = loss_fn(images[i].unsqueeze(0).to(device), images[j].unsqueeze(0).to(device))
        total_score += d.item()
    return total_score / len(pairs[:200])

def run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div):
    # Create a unique experiment name from the hyperparameters
    exp_name = f"latent{latent_dim}_cl{galaxy_class}_lrgen{lr_gen}_lrdisc{lr_disc}_gl{gen_loss}_dl{disc_loss}_ab{adam_betas}_wd{weight_decay}_ls{label_smoothing}_ld{lambda_div}_dv{data_version}"
    exp_dir = os.path.join(output_dir, gan_type, f"{exp_name}_f{folds[-1]}")
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Starting experiment: {exp_name}")

    # Load data for the fixed galaxy class and fold
    data = load_galaxies(
        galaxy_class=galaxy_class,
        fold=fold,
        img_shape=(1, image_size, image_size),
        sample_size=1000,
        REMOVEOUTLIERS=REMOVEOUTLIERS,
        train=True
    )
    train_images, _, _, _ = data
    
    # Shuffle the data
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm].float()
    
    # Normalize to [-1, 1]
    train_images = 2 * (train_images - train_images.min()) / (train_images.max() - train_images.min()) - 1
    dataset = TensorDataset(train_images)

    # Save example real images for reference (using 36 images arranged in 6x6 grid)
    real_examples = train_images[:36]
    save_images_tensorboard(real_examples, os.path.join(exp_dir, "ScatterGAN_example_real_images.png"), nrow=6)

    # Initialize models and datasets
    if gan_type == 'Simple':
        generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = DCGANDiscriminator(ndf=64, nc=1).to(device)
        
        mock_tensor = torch.zeros_like(train_images)
        train_dataset = TensorDataset(train_images, mock_tensor)
    elif gan_type == 'ScatterDualGAN':
        # Build a new dataset with images and precomputed scattering coefficients
        train_scat_coeffs = compute_scattering_coeffs(train_images, scattering=scattering, batch_size=batch_size, device=device)
        train_dataset = TensorDataset(train_images, train_scat_coeffs)
        
        img_shape = (1, image_size, image_size)
        scat_shape = train_scat_coeffs.shape[1:]  # e.g. (C, H, W)
        
        # Use a standard generator and our dual-input discriminator.
        generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = DualInputDiscriminator(img_shape, scat_shape, output_dim=1, dropout_rate=0.3, J=J).to(device)

    else:
        raise ValueError("No chosen gan_type")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen, betas=adam_betas, weight_decay=weight_decay)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=adam_betas, weight_decay=weight_decay)

    # Choose loss functions
    real_label_val = label_smoothing
    fake_label_val = 0.0

    if disc_loss == "BCE":
        disc_loss_fn = lambda real, fake: d_loss_bce(real, fake, real_label_val, fake_label_val)
    elif disc_loss == "MSE":
        disc_loss_fn = d_loss_mse
    elif disc_loss == "LSGAN":
        disc_loss_fn = d_loss_lsgan
    elif disc_loss == "HINGE":
        disc_loss_fn = d_loss_hinge
    elif disc_loss == "WGAN":
        disc_loss_fn = d_loss_wgan
    else:
        raise ValueError(f"Unknown discriminator loss type: {disc_loss}")

    if gen_loss == "BCE":
        gen_loss_fn = lambda fake: g_loss_bce(fake, real_label_val)
    elif gen_loss == "MSE":
        gen_loss_fn = g_loss_mse
    elif gen_loss == "LSGAN":
        gen_loss_fn = g_loss_lsgan
    elif gen_loss == "HINGE":
        gen_loss_fn = g_loss_hinge
    elif gen_loss == "WGAN":
        gen_loss_fn = g_loss_wgan
    else:
        raise ValueError(f"Unknown generator loss type: {gen_loss}")

    # Initialize loss tracking
    g_loss_history, d_loss_history = [], []
    log_interval = max(1, epochs // 10)
    with logging_redirect_tqdm():
        for epoch in tqdm(range(1, epochs + 1), miniters=log_interval):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                images, scat_coeffs = batch
                images = images.to(device)
                scat_coeffs = scat_coeffs.to(device) if gan_type == 'ScatterDualGAN' else None
                b_size = images.size(0)
                if b_size != batch_size:
                    print(f"Batch size {b_size} is not equal to expected batch size {batch_size}. Skipping this batch.")
                    continue
                # Apply random transformations to the real images
                

                # Train Discriminator / Critic
                optimizer_D.zero_grad()
                if gan_type == 'ScatterDualGAN':
                    real_pred = discriminator(images, scat_coeffs)
                else:
                    real_pred = discriminator(images)

                # Generate fake images.
                z = torch.randn(b_size, latent_dim, 1, 1, device=device)                
                fake_images = generator(z)
                
                # For fake images:
                if gan_type == 'ScatterDualGAN':
                    fake_scat = compute_scattering_coeffs(fake_images, scattering=scattering, device=device)
                    fake_pred = discriminator(fake_images.detach(), fake_scat)
                else:
                    fake_pred = discriminator(fake_images.detach())

                fake_flat = fake_images.reshape(b_size, -1)
                pairwise_dists = torch.pdist(fake_flat, p=2)
                diversity_loss = -torch.mean(pairwise_dists)

                d_loss = disc_loss_fn(real_pred, fake_pred)

                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                if gan_type == 'ScatterDualGAN':
                    fake_scat_for_g = compute_scattering_coeffs(fake_images, scattering=scattering, device=device)
                    fake_pred_for_g = discriminator(fake_images, fake_scat_for_g)
                else:
                    fake_pred_for_g = discriminator(fake_images)
                g_loss = gen_loss_fn(fake_pred_for_g) + lambda_div * diversity_loss
                g_loss.backward()
                optimizer_G.step()

                # Track losses
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                num_batches += 1

            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            d_loss_history.append(avg_d_loss)
            g_loss_history.append(avg_g_loss)

            if epoch % log_interval == 0 or epoch == epochs:
                logging.info(f"Exp {exp_name} | Epoch [{epoch}/{epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

            if epoch % save_interval == 0 or epoch == epochs:
                generator.eval()
                with torch.no_grad():
                    test_z = torch.randn(16, latent_dim, 1, 1, device=device)
                    gen_sample = generator(test_z).cpu()
                generator.train()

                img_file = os.path.join(exp_dir, f"ScatterGAN_generated_epoch_{epoch}.png")
                save_images_tensorboard(gen_sample, img_file, nrow=4)
                logging.info(f"Saved generated images to {img_file}")

                if len(grid_adam_betas) == 1:  # Save model checkpoints only if there is a single beta value.
                    torch.save(generator.state_dict(), f"./GAN/generators/generator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                    torch.save(discriminator.state_dict(), f"./GAN/discriminators/discriminator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                logging.info(f"Saved model checkpoints for epoch {epoch}")

    plot_GAN_losses(exp_dir, galaxy_class, g_loss_history, d_loss_history, gen_loss, disc_loss)
    logging.info(f"Training complete for {exp_name}. Loss plot saved.")
    
    # Plot histograms of the generated images
    gen_sample = generator(fixed_noise).cpu()    
    plot_histograms(
    gen_sample,
    train_images,
    title1="Generated Images",
    title2="Train Images",
    save_path=os.path.join(exp_dir, "histograms.png"))

    generator.eval()
    with torch.no_grad():
        rep_generated = generator(fixed_noise).cpu()
    rep_img_path = os.path.join(exp_dir, "rep_generated.png")
    save_images_tensorboard(rep_generated, rep_img_path, nrow=6)

    # Compute FID using a subset of real images (up to 100 images) and the generated images
    real_subset = train_images[:min(100, len(train_images))]
    fid_score = calculate_fid(real_subset, rep_generated, device=device)

    # --- Evaluate diversity using LPIPS ---
    with torch.no_grad():
        div_noise = torch.randn(100, latent_dim, 1, 1, device=device)
        generated_div_images = generator(div_noise).cpu()
    diversity_score = compute_lpips_diversity(generated_div_images)

    # --- Compute MSE between generated and real images ---
    # Here we compare the representative generated images to the 36 real example images
    mse_score = F.mse_loss(rep_generated.to(device), real_examples.to(device)).item()

    return {
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "rep_image_path": rep_img_path,
        "final_generator_loss": g_loss_history[-1],
        "lpips_diversity": diversity_score,
        "mse_score": mse_score,
        "fid_score": fid_score,
        "latent_dim": latent_dim,
        "lr_gen": lr_gen,
        "lr_disc": lr_disc,
        "gen_loss": gen_loss,
        "disc_loss": disc_loss
    }


results = []
for latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div in itertools.product(
    grid_latent_dim, folds, grid_lr_gen, grid_lr_disc, grid_gen_loss, grid_disc_loss, grid_adam_betas, grid_weight_decay, grid_label_smoothing, grid_lambda_div
):
    res = run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div)
    results.append(res)

# Print out the hyperparameter combination that yielded the most diverse generated images
best_experiment = max(results, key=lambda x: x["lpips_diversity"])
print(f"Best hyperparameters based on diversity: {best_experiment['exp_name']} with LPIPS diversity score: {best_experiment['lpips_diversity']:.4f}")

# Create and print a summary table of all experiments
df = pd.DataFrame(results)
df = df.sort_values(by='fid_score', ascending=True)
cols = ['exp_name', 'final_generator_loss', 'lpips_diversity', 'mse_score', 'fid_score', 'latent_dim', 'lr_gen', 'lr_disc', 'gen_loss', 'disc_loss']
print("\nSummary of all experiments (sorted by increasing FID):")
print(df[cols].to_string(index=False))


# Save that table as a log file
summary_path = os.path.join(output_dir, "hyperparam_summary.csv")
df.to_csv(summary_path, index=False)
logging.info(f"Summary table saved to {summary_path}")

n_experiments = len(results)
cols_plot = min(6, n_experiments)
rows_plot = math.ceil(n_experiments / cols_plot)

fig, axs = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot * 3, rows_plot * 3))
axs = axs.flatten() if n_experiments > 1 else [axs]

for ax in axs[n_experiments:]:
    fig.delaxes(ax)

for idx, res in enumerate(results):
    img = mpimg.imread(res["rep_image_path"])
    ax = axs[idx]
    ax.imshow(img)
    ax.axis('off')
    title = (f"{res['exp_name']}\nG_loss: {res['final_generator_loss']:.4f}\n"
             f"Diversity: {res['lpips_diversity']:.4f}\nMSE: {res['mse_score']:.4f}")
    ax.set_title(title, fontsize=8)

fig.suptitle("Hyperparameter Optimization Summary", fontsize=16)
plt.tight_layout()
summary_path = os.path.join(output_dir, "hyperparam_summary.png")
plt.savefig(summary_path)
plt.close()
logging.info(f"Summary grid saved to {summary_path}")
