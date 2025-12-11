import os
import numpy as np
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.data_loader import load_galaxies
from utils.GAN_models import DCGANDiscriminator, DCGANGenerator, GANDiscriminator, DualInputDiscriminator, AdvancedGenerator
from utils.plotting import plot_GAN_losses, plot_histograms, save_images_tensorboard, plot_variance_histograms
from utils.GAN_loss_functions import pick_gan_loss_functions
from utils.calc_tools import (
    bhattacharyya_coefficient, total_variation_distance, histogram_intersection,
    calculate_fid, calculate_lpips_diversity, calculate_cmmd, change_images, cluster_metrics, normalise_images    
)
import itertools
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from kymatio.torch import Scattering2D

"""
This script trains a GAN model that outputs images in the range [-1, 1].
The model is trained on galaxy images, and the training process includes hyperparameter optimization.
The script also evaluates the model using various metrics and saves the results in a LaTeX format.
"""

# -------------------------------
# Fixed parameters for hyperparameter optimization
# -------------------------------
galaxy_class = 40
folds = [0, 1, 2, 3, 4]
batch_size = 256
epochs = 500
image_size = 128
sample_size = 100 # Number of images to sample from the dataset
save_interval = 100
output_dir = "GAN"
gan_type = ["Simple", "ScatterDualGAN", "Advanced"][0]
if gan_type == "ScatterDualGAN":
    J, L, order = 4, 6, 2
    batch_size = 1024  # The bigger the faster

# Hyperparameter grid (grid_filter controls REMOVEOUTLIERS)
grid_latent_dim = [128]
grid_lr_gen = [1e-3]       # Separate learning rate for generator
grid_lr_disc = [1e-3]      # Separate learning rate for discriminator
grid_gen_loss = ["MSE", "BCE"]
grid_disc_loss = ["BCE"]
grid_adam_betas = [(0.5, 0.999)]
grid_weight_decay = [0.0]
grid_label_smoothing = [0.9]  # Label smoothing for generator
grid_lambda_div = [0]      
grid_filter = [False]   # REMOVEOUTLIERS option

EVALUATE = True  # Set to True to evaluate the model after training

############################################################################################

# Set random seed for reproducibility
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

print("Running ScatterGAN with gan type", gan_type)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "generators"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "discriminators"), exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# If using ScatterDualGAN, initialize the scattering transform.
if gan_type == "ScatterDualGAN":
    scattering = Scattering2D(J=J, L=L, shape=(image_size, image_size), max_order=order)
    def compute_scattering_coeffs(images, scattering, batch_size, grad_mode=False, output_device=None):
        # Always compute on CPU.
        scat_coeffs_list = []
        context = torch.enable_grad() if grad_mode else torch.no_grad()
        with context:
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].cpu().contiguous()
                batch_scat = scattering(batch)
                if not grad_mode:
                    batch_scat = batch_scat.detach()
                if batch_scat.shape[1] == 1:
                    batch_scat = batch_scat.squeeze(1)
                scat_coeffs_list.append(batch_scat)
        scat_coeffs = torch.cat(scat_coeffs_list, dim=0)
        if output_device is not None:
            scat_coeffs = scat_coeffs.to(output_device)
        return scat_coeffs


def run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss,
                   adam_betas, weight_decay, label_smoothing, lambda_div, REMOVEOUTLIERS):
    if REMOVEOUTLIERS:
        print("Removing outliers. Dataset will be cleaned")
        data_version = 'clean'
    else:
        data_version = 'full'
    exp_name = (f"ss{sample_size}_cl{galaxy_class}_lrgen{lr_gen}_lrdisc{lr_disc}_gl{gen_loss}_dl{disc_loss}_"
                f"ls{label_smoothing}_ld{lambda_div}_dv{data_version}")
    exp_dir = os.path.join(output_dir, gan_type, f"{exp_name}_f{folds[-1]}")
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Starting experiment: {exp_name}")

    # Load data for the fixed galaxy class and fold
    data = load_galaxies(
        galaxy_class=galaxy_class,
        fold=fold,
        downsample_size=(image_size, image_size),
        sample_size=sample_size,
        REMOVEOUTLIERS=REMOVEOUTLIERS,
        train=True
    )
    
    train_images, _, _, _ = data
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm].float()
    train_images = normalise_images(train_images, -1, 1) # Normalize images to [-1, 1] because of the tanh activation in the generator
    
    print("Loaded data for galaxy class", galaxy_class, "with shape", train_images.shape)

    # Choose dataset and initialize models depending on gan_type
    if gan_type == "ScatterDualGAN":
        train_scat = compute_scattering_coeffs(train_images, scattering=scattering, batch_size=1024, grad_mode=False, output_device=device)
        train_dataset = TensorDataset(train_images, train_scat)
        img_shape = (1, image_size, image_size)
        scat_shape = train_scat.shape[1:]
        generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = DualInputDiscriminator(img_shape, scat_shape, output_dim=1, dropout_rate=0.3, J=J).to(device)
    elif gan_type == "Advanced":
        train_dataset = TensorDataset(train_images)
        generator = AdvancedGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = GANDiscriminator(ndf=64, nc=1).to(device)
    elif gan_type == "Simple":
        train_dataset = TensorDataset(train_images)
        generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = DCGANDiscriminator(ndf=64, nc=1).to(device)
    else:
        raise ValueError("Unknown gan_type")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    save_images_tensorboard(train_images[:36], os.path.join(exp_dir, "ScatterGAN_example_real_images.png"), nrow=6)     # Save example real images (6x6 grid)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen, betas=adam_betas, weight_decay=weight_decay)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=adam_betas, weight_decay=weight_decay)

    gen_loss_fn, disc_loss_fn = pick_gan_loss_functions(gen_loss=gen_loss, disc_loss=disc_loss, real_label_val=label_smoothing, fake_label_val=0.0)

    g_loss_history, d_loss_history = [], []
    log_interval = max(1, epochs // 10)
    with logging_redirect_tqdm():
        for epoch in tqdm(range(1, epochs + 1), miniters=log_interval):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                if gan_type == "ScatterDualGAN":
                    images, scat_coeffs = batch
                    images = images.to(device)
                    scat_coeffs = scat_coeffs.to(device)
                else:
                    images = batch[0].to(device)
                b_size = images.size(0)
                optimizer_D.zero_grad()
                if gan_type == "ScatterDualGAN":
                    real_pred = discriminator(images, scat_coeffs)
                else:
                    real_pred = discriminator(images)
                z = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake_images = generator(z)
                fake_images = change_images(fake_images) # Apply random transformation to stabilize training
                if gan_type == "ScatterDualGAN":
                    fake_scat_disc = compute_scattering_coeffs(
                        fake_images, scattering=scattering, batch_size=batch_size, grad_mode=False, output_device=device)
                    fake_pred = discriminator(fake_images.detach(), fake_scat_disc)
                else:
                    fake_pred = discriminator(fake_images.detach()) 
                fake_flat = fake_images.reshape(b_size, -1)
                pairwise_dists = torch.pdist(fake_flat, p=2)
                diversity_loss = -torch.mean(pairwise_dists)
                d_loss = disc_loss_fn(real_pred, fake_pred)
                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()
                if gan_type == "ScatterDualGAN":
                    fake_scat_gen = compute_scattering_coeffs(
                        fake_images, scattering=scattering, batch_size=batch_size, grad_mode=True, output_device=device)
                    fake_pred_for_g = discriminator(fake_images, fake_scat_gen)
                else:
                    fake_pred_for_g = discriminator(fake_images)
                g_loss = gen_loss_fn(fake_pred_for_g) + lambda_div * diversity_loss
                g_loss.backward()
                optimizer_G.step()

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
                if len(grid_adam_betas) == 1:
                    torch.save(generator.state_dict(), f"./GAN/generators/generator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                    torch.save(discriminator.state_dict(), f"./GAN/discriminators/discriminator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                logging.info(f"Saved model checkpoints for epoch {epoch}")

    plot_GAN_losses(exp_dir, galaxy_class, g_loss_history, d_loss_history, gen_loss, disc_loss)
    logging.info(f"Training complete for {exp_name}. Loss plot saved.")

    generator.eval()
    with torch.no_grad():
        noise = torch.randn(36, grid_latent_dim[0], 1, 1, device=device)  # Specific random input for generating images
        rep_generated = generator(noise).cpu()
    rep_img_path = os.path.join(exp_dir, "rep_generated.png")
    save_images_tensorboard(rep_generated, rep_img_path, nrow=6)
    
    # Evaluation metrics
    
    if EVALUATE:
    
        seeds = [0, 1, 2, 3, 4]
        all_metrics = []

        for seed in seeds:
            torch.manual_seed(seed)
            
            with torch.no_grad():
                eval_noise = torch.randn(len(train_images), latent_dim, 1, 1, device=device)
                gen_sample = generator(eval_noise).detach().cpu()   
                
            if seed == 0:
                plot_histograms(gen_sample, train_images, title1="Generated Images", title2="Train Images",
                    save_path=os.path.join(exp_dir, "histograms.png"))
                plot_variance_histograms(gen_sample, train_images,
                    save_path=os.path.join(exp_dir, "variance_histograms.png"))   
            
            subset_size = 100
            real_subset = train_images[:subset_size]
            gen_subset = gen_sample[:subset_size]
            
            flattened_features = gen_sample.view(gen_sample.size(0), -1)
            cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(flattened_features, n_clusters=galaxy_class)
            bhatta = bhattacharyya_coefficient(real_subset.cpu().numpy().flatten(), gen_subset.cpu().numpy().flatten())
            tv_dist = total_variation_distance(real_subset.cpu().numpy().flatten(), gen_subset.cpu().numpy().flatten())
            hist_int = histogram_intersection(real_subset.cpu().numpy().flatten(), gen_subset.cpu().numpy().flatten())
            fid_score = calculate_fid(real_subset, gen_subset, device=device)
            mse_score = F.mse_loss(gen_subset.to(device), real_subset.to(device)).item()
            cmmd_score = calculate_cmmd(real_subset.to(device), gen_subset.to(device))
            diversity_score = calculate_lpips_diversity(gen_subset)
            
            # --- NEW: Evaluate on variance histograms ---
            # Compute the per-image standard deviation (variance) for real and generated images.
            with torch.no_grad():
                real_variances = train_images.view(train_images.size(0), -1).std(dim=1, unbiased=False)
                gen_variances = gen_sample.view(gen_sample.size(0), -1).std(dim=1, unbiased=False)
            subset_size_var = 100
            real_var_subset = real_variances[:subset_size_var]
            gen_var_subset = gen_variances[:subset_size_var]
            var_mse = F.mse_loss(gen_var_subset.to(device), real_var_subset.to(device)).item()
            var_bhattacharyya = bhattacharyya_coefficient(real_var_subset.cpu().numpy(), gen_var_subset.cpu().numpy())
            var_tv_distance = total_variation_distance(real_var_subset.cpu().numpy(), gen_var_subset.cpu().numpy())
            var_hist_int = histogram_intersection(real_var_subset.cpu().numpy(), gen_var_subset.cpu().numpy())
    
            
            metrics = {
                "fid_score": fid_score,  # calculated in your snippet
                "mse_score": mse_score,
                "lpips_diversity": diversity_score,
                "bhattacharyya": bhatta,
                "tv_distance": tv_dist,
                "hist_intersection": hist_int,
                "cmmd_score": cmmd_score,
                "cluster_error": cluster_error,
                "cluster_distance": cluster_distance,
                "cluster_std_dev": cluster_std_dev,
                "var_mse": var_mse,
                "var_bhattacharyya": var_bhattacharyya,
                "var_tv_distance": var_tv_distance,
                "var_hist_intersection": var_hist_int,
            }
            all_metrics.append(metrics)
            
        # Now, compute mean and std for each metric:
        aggregated_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            aggregated_metrics[key + '_mean'] = np.mean(values)
            aggregated_metrics[key + '_std'] = np.std(values)

    else:
        # If EVALUATE is False, we skip the evaluation metrics
        logging.info("Skipping evaluation metrics as EVALUATE is set to False.")
        aggregated_metrics = {
            "fid_score_mean": None,
            "mse_score_mean": None,
            "lpips_diversity_mean": None,
            "cmmd_score_mean": None,
            "cluster_error_mean": None,
            "cluster_distance_mean": None,
            "cluster_std_dev_mean": None,
            "bhattacharyya_mean": None,
            "tv_distance_mean": None,
            "hist_intersection_mean": None,
            "var_mse_mean": None,
            "var_bhattacharyya_mean": None,
            "var_tv_distance_mean": None,
            "var_hist_intersection_mean": None
        }
    
    return {
        "exp_name": exp_name,  # Kept for logging but will be dropped from the table
        "exp_dir": exp_dir,
        "rep_image_path": rep_img_path,
        "final_generator_loss": g_loss_history[-1],
        "final_discriminator_loss": d_loss_history[-1],
        "aggregated_metrics": aggregated_metrics,  # Metrics
        "latent_dim": latent_dim,                # Hyperparameters
        "lr_gen": lr_gen,
        "lr_disc": lr_disc,
        "gen_loss": gen_loss,
        "disc_loss": disc_loss,
        "label_smoothing": label_smoothing,
        "lambda_div": lambda_div,
        "REMOVEOUTLIERS": REMOVEOUTLIERS
    }

print(list(itertools.product(grid_latent_dim, folds, grid_lr_gen, grid_lr_disc, grid_gen_loss, grid_disc_loss, grid_adam_betas, grid_weight_decay, grid_label_smoothing, grid_lambda_div, grid_filter)))

results = []
for latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div, REMOVEOUTLIERS in itertools.product(
    grid_latent_dim, folds, grid_lr_gen, grid_lr_disc, grid_gen_loss, grid_disc_loss, grid_adam_betas, grid_weight_decay, grid_label_smoothing, grid_lambda_div, grid_filter
):
    res = run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div, REMOVEOUTLIERS)
    results.append(res)

if EVALUATE:
    # Create the summary table:
    df = pd.DataFrame(results)
    ## Flatten the aggregated_metrics column into individual columns
    df = pd.concat([df.drop(columns=['aggregated_metrics']), df['aggregated_metrics'].apply(pd.Series)], axis=1)

    # Drop the exp_name column so it does not appear in the table.
    if 'exp_name' in df.columns:
        df = df.drop(columns=['exp_name'])
    # Define the order of columns, including label_smoothing, lambda_div, and the new variance metrics.

    # Suppose df is your DataFrame with aggregated metrics for each experiment.
    # First, define which metrics are "lower is better"
    lower_is_better = [
        'fid_score_mean', 'mse_score_mean', 'cmmd_score_mean', 'cluster_error_mean', 
        'cluster_distance_mean', 'cluster_std_dev_mean', 'bhattacharyya_mean', 
        'tv_distance_mean', 'hist_intersection_mean', 'var_mse_mean', 
        'var_bhattacharyya_mean', 'var_tv_distance_mean', 'var_hist_intersection_mean'
    ]

    # Normalize each metric column across experiments.
    df_norm = df.copy()
    for col in lower_is_better:
        # Invert these metrics so higher is better.
        df_norm[col] = -df_norm[col]

    # Then, for each metric, compute a z-score:
    for col in df_norm.columns:
        if col not in ['exp_dir', 'rep_image_path', 'REMOVEOUTLIERS']:
            if pd.api.types.is_numeric_dtype(df_norm[col]):
                df_norm[col] = (df_norm[col] - df_norm[col].mean()) / (df_norm[col].std() + 1e-8)

    # Exclude losses from composite score computation:
    exclude_cols = ['exp_dir', 'rep_image_path', 'REMOVEOUTLIERS', 'label_smoothing', 'lambda_div', 'latent_dim', 'lr_gen', 'lr_disc', 'gen_loss', 'disc_loss',
                    'final_generator_loss', 'final_discriminator_loss']
    metric_cols = [col for col in df_norm.columns if col not in exclude_cols]
    numeric_metric_cols = [col for col in metric_cols if pd.api.types.is_numeric_dtype(df_norm[col])]
    df['composite_score'] = df_norm[numeric_metric_cols].mean(axis=1)

    # Now sort the DataFrame by the composite score (assuming higher is better)
    df = df.sort_values(by='composite_score', ascending=False)

    print("Best overall experiment based on composite score:")
    print(df.iloc[0])

    # Optionally, update your columns list to include the composite score
    cols = [
        'final_generator_loss', 'final_discriminator_loss', 'mse_score_mean', 'lpips_diversity_mean',
        'fid_score_mean', 'cmmd_score_mean', 'cluster_error_mean', 'cluster_distance_mean', 'cluster_std_dev_mean',
        'bhattacharyya_mean', 'tv_distance_mean', 'hist_intersection_mean',
        'var_mse_mean', 'var_bhattacharyya_mean', 'var_tv_distance_mean', 'var_hist_intersection_mean',
        'latent_dim', 'lr_gen', 'lr_disc', 'gen_loss', 'disc_loss', 'label_smoothing',
        'lambda_div', 'REMOVEOUTLIERS', 'composite_score'
    ]

    print("\nSummary of all experiments (sorted by composite score):")
    print(df[cols].to_string())

    # Group metrics into "mean ± std" format for LaTeX
    mean_std_pairs = [
        ("fid_score_mean", "fid_score_std"),
        ("mse_score_mean", "mse_score_std"),
        ("lpips_diversity_mean", "lpips_diversity_std"),
        ("cmmd_score_mean", "cmmd_score_std"),
        ("cluster_error_mean", "cluster_error_std"),
        ("cluster_distance_mean", "cluster_distance_std"),
        ("cluster_std_dev_mean", "cluster_std_dev_std"),
        ("bhattacharyya_mean", "bhattacharyya_std"),
        ("tv_distance_mean", "tv_distance_std"),
        ("hist_intersection_mean", "hist_intersection_std"),
        ("var_mse_mean", "var_mse_std"),
        ("var_bhattacharyya_mean", "var_bhattacharyya_std"),
        ("var_tv_distance_mean", "var_tv_distance_std"),
        ("var_hist_intersection_mean", "var_hist_intersection_std"),
    ]

    # Create a new DataFrame with the formatted metrics
    latex_df = df.copy()

    for mean_col, std_col in mean_std_pairs:
        if mean_col in latex_df and std_col in latex_df:
            latex_df[mean_col.replace("_mean", "")] = latex_df.apply(
                lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}", axis=1
            )

    # Keep only relevant columns (remove stds, keep formatted means)
    drop_cols = [std for _, std in mean_std_pairs]
    latex_df = latex_df.drop(columns=drop_cols)

    # Define final columns to keep and their order (add any you want)
    final_cols = [
        'latent_dim', 'lr_gen', 'lr_disc', 'label_smoothing', 'lambda_div', 'REMOVEOUTLIERS',
        'FID', 'MSE', 'LPIPS', 'CMMD', 'Cluster Error', 'Cluster Dist', 'Cluster Std',
        'Bhattacharyya', 'TV Dist', 'Hist Int',
        'Var MSE', 'Var Bhatt', 'Var TV', 'Var Hist',
        'composite_score'
    ]


    # Some renaming for nice LaTeX headers (optional)
    latex_df = latex_df.rename(columns={
        'fid_score': 'FID',
        'mse_score': 'MSE',
        'lpips_diversity': 'LPIPS',
        'cmmd_score': 'CMMD',
        'cluster_error': 'Cluster Error',
        'cluster_distance': 'Cluster Dist',
        'cluster_std_dev': 'Cluster Std',
        'bhattacharyya': 'Bhattacharyya',
        'tv_distance': 'TV Dist',
        'hist_intersection': 'Hist Int',
        'var_mse': 'Var MSE',
        'var_bhattacharyya': 'Var Bhatt',
        'var_tv_distance': 'Var TV',
        'var_hist_intersection': 'Var Hist',
    })

    # Output LaTeX-ready table
    latex_output = latex_df[final_cols].to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, f"summary_table_latex_{gan_type}_{galaxy_class}_{epochs}.tex"), "w") as f:
        f.write(latex_output)
    print("Saved LaTeX-formatted summary table.")

    # Create a vertical version of the best experiment row for LaTeX
    best_row = latex_df[final_cols].iloc[[0]].T.reset_index()
    best_row.columns = ['Metric', 'Value']
    vertical_latex = best_row.to_latex(index=False, escape=False)

    with open(os.path.join(output_dir, f"summary_table_vertical_{gan_type}_{galaxy_class}_{epochs}.tex"), "w") as f:
        f.write(vertical_latex)
    print("Saved vertical LaTeX-formatted table.")


    # Now, the experiment with the highest composite score could be considered the best overall.
    best_experiment = df.loc[df['composite_score'].idxmax()]
    print("Best overall experiment based on composite score:")
    print(best_experiment[['latent_dim', 'lr_gen', 'lr_disc', 'gen_loss', 'disc_loss', 'label_smoothing', 'lambda_div', 'REMOVEOUTLIERS', 'composite_score']])