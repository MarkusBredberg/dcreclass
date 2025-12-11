import os
import math
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from utils.data_loader import load_galaxies  
# ------------------------------------------------
# Example: Simple U-Net for single-channel diffusion
# ------------------------------------------------
class SimpleUNet(nn.Module):
    """
    Minimal U-Net for 1-channel input/output with time embedding.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        # 1) Still embed time to some dimension (128)...
        # 2) ...but map it down to base_channels (64) so shapes match
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, base_channels),  # Map 128 -> 64
        )

        # Down
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.down = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)

        # Middle
        self.mid_conv1 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.mid_conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # Up
        self.up = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        # Embed time t to shape (B, base_channels)
        t_embed = self.time_mlp(t.unsqueeze(-1).float())  # (B, base_channels)
        # Reshape to broadcast over spatial dims => (B, base_channels, 1, 1)
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)

        # Down
        h1 = torch.relu(self.conv1(x))
        h1 = torch.relu(self.conv2(h1) + t_embed)
        h2 = self.down(h1)

        # Middle
        h3 = torch.relu(self.mid_conv1(h2) + t_embed)
        h3 = torch.relu(self.mid_conv2(h3))

        # Up
        h4 = self.up(h3)
        h4 = torch.relu(self.conv3(h4) + t_embed)
        # Skip connection with h1
        h4 = torch.relu(self.conv4(h4) + h1)
        return self.final_conv(h4)


# ------------------------------------------------
# Diffusion utilities
# ------------------------------------------------
def get_beta_schedule(num_timesteps, start=1e-4, end=0.02):
    """
    Linear schedule for betas from start to end.
    """
    return torch.linspace(start, end, num_timesteps)

def forward_diffusion_sample(x0, t, alphas_cumprod):
    """
    Add noise to x0 at time step t according to the forward diffusion process:
    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
    """
    # Gather alpha_bar[t] for each sample in the batch
    alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise, noise

@torch.no_grad()
def sample(model, x_shape, alphas, alphas_cumprod, betas, num_steps=1000, device="cpu"):
    """
    Reverse diffusion sampling:
    Start from x_T ~ N(0,I) and iteratively sample x_{t-1}.
    """
    x = torch.randn(x_shape, device=device)
    for t in reversed(range(num_steps)):
        t_tensor = torch.tensor([t] * x.shape[0], device=device).long()
        # Predict noise
        eps = model(x, t_tensor)
        alpha_bar_t = alphas_cumprod[t]
        alpha_t = alphas[t]

        # Estimate x0: x0 = (x_t - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t)
        x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        if t > 0:
            beta_t = betas[t]
            # Sample x_{t-1} ~ N(mu, sigma^2 I), with:
            # mu = sqrt(alpha_bar_{t-1}) * x0
            # sigma^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
            alpha_bar_t_prev = alphas_cumprod[t-1]
            sigma_t = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            sigma_t = torch.sqrt(sigma_t)
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha_bar_t_prev) * x0 + sigma_t * noise
        else:
            x = x0
    return x

# ------------------------------------------------
# Training script
# ------------------------------------------------
def save_images_tensorboard(images, file_path, nrow=4):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, file_path)

def plot_diffusion_losses(out_dir, loss_history):
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Diffusion Training Loss")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "diffusion_loss.png")
    plt.savefig(plot_path)
    plt.close()

def run_diffusion_experiment(
    galaxy_class=11,
    fold=5,
    image_size=128,
    sample_size=100,
    num_timesteps=100,
    epochs=2000,
    batch_size=64,
    lr=1e-4,
    save_interval=10,
    output_dir="DiffusionOutputs",
    device=None
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Running diffusion experiment on device: {device}")

    # 1) Load data
    data = load_galaxies(
        galaxy_class=galaxy_class,
        fold=fold,
        img_shape=(1, image_size, image_size),
        sample_size=sample_size,
        REMOVEOUTLIERS=False,
        train=True
    )
    train_images, _, _, _ = data    
    perm = torch.randperm(len(train_images))     # Shuffle
    train_images = train_images[perm].float()
    # Normalize to [-1, 1]
    train_images = 2 * (train_images - train_images.min()) / (train_images.max() - train_images.min()) - 1

    dataset = TensorDataset(train_images)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2) Model & optimizer
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3) Beta schedule
    betas = get_beta_schedule(num_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    mse_loss = nn.MSELoss()

    # 4) Training loop
    global_step = 0
    loss_history = []
    model.train()
    with logging_redirect_tqdm():
        for epoch in tqdm(range(1, epochs + 1)):
            epoch_loss = 0.0
            for batch in train_loader:
                x0 = batch[0].to(device)
                b_size = x0.size(0)

                # Sample a random t in [0, num_timesteps)
                t = torch.randint(0, num_timesteps, (b_size,), device=device).long()

                # Forward diffusion
                x_noisy, noise = forward_diffusion_sample(x0, t, alphas_cumprod)

                # Predict the noise with the model
                noise_pred = model(x_noisy, t)

                # Compute loss
                loss = mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_epoch_loss)


            if epoch % 10 == 0:
                logging.info(f"Epoch [{epoch}/{epochs}] | Loss: {avg_epoch_loss:.4f}")

            # Save sample images periodically
            if epoch % save_interval == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    # Generate samples via reverse diffusion
                    samples = sample(
                        model,
                        x_shape=(16, 1, image_size, image_size),
                        alphas=alphas,
                        alphas_cumprod=alphas_cumprod,
                        betas=betas,
                        num_steps=num_timesteps,
                        device=device
                    ).cpu()
                img_path = os.path.join(output_dir, f"diffusion_samples_epoch_{epoch}.png")
                save_images_tensorboard(samples, img_path, nrow=4)
                logging.info(f"Saved sample images to {img_path}")
                model.train()

    # Plot final loss
    plot_diffusion_losses(output_dir, loss_history)

    # Save final model
    model_path = os.path.join(output_dir, "diffusion_model_final.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Final model saved to {model_path}")

    return loss_history

if __name__ == "__main__":
    # ---------------------------------------
    # Setup logging
    # ---------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Example single run
    run_diffusion_experiment(
        galaxy_class=11,
        fold=5,
        image_size=128,
        sample_size=70,
        num_timesteps=1000,
        epochs=1000,
        batch_size=64,
        lr=1e-4,
        save_interval=100,
        output_dir="DiffusionOutputs",
        device=device
    )
