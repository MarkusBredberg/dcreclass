import itertools
import os
from utils.data_loader import load_galaxies, get_classes
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from kymatio.torch import Scattering2D
from utils.models import get_model
from utils.calc_tools import normalize_to_minus1_1, normalize_to_0_1
from utils.custom_mse import NormalisedWeightedMSELoss, CustomMSELoss, WeightedMSELoss, RadialWeightedMSELoss, CustomIntensityWeightedMSELoss, MaxIntensityMSELoss, StandardMSELoss, CombinedMSELoss, ExperimentalMSELoss, BasicMSELoss
from utils.scatter_reduction import lavg, ldiff
from utils.training_tools import EarlyStopping 
from torchsummary import summary
from tqdm import tqdm
import time
from utils.plotting import vae_plot_comparison, loss_vs_epoch, plot_original_images, plot_weight_and_loss_maps, plot_images, plot_histograms, plot_reconstructions
import matplotlib.pyplot as plt
import cv2


######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

#encoder_list = ['CDual', 'CCNN', 'CSTMLP', 'CldiffSTMLP', 'ClavgSTMLP']
#encoder_list = ['Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP']
#encoder_list = ['CCNN']
#encoder_list = ['Dual', 'CNN']
encoder_list = ['STMLP']

ind = 8 #Choose the loss function to use

#galaxy_classes = [[10, 11, 12, 13]] #Use double square parenthesis for conditional VAEs
galaxy_classes = [11]
num_galaxies_list = [1100+ind]
hidden_dim1 = [256]
hidden_dim2 = [128]
latent_dims = [64]
learning_rates = [1e-4]
reg_params = [1e-4] 
initial_final_betas = [(1e-1, 1)]
num_epochs_cuda = 500; num_epochs_cpu = 10
batch_size = 128 
img_shape = (1, 128, 128)
J, L, order = 2, 12, 2

FFCV = True # Use five-fold cross-validation
ES = True # Use early stopping
IMGCHECK = False # Check the input images (Tool for control)
SAVEIMGS = False # Save the reconstructed images in tensor format
NORMALISETOPM = False # Normalise to [-1, 1]
PLOTFAILED = True # Plot the top 5 images with the largest MSE loss and weakest peak intensities

#########################################################################################################################

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Running script 2.alt_asym.py")


if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")

classes = get_classes()

train_loader, test_loader, batch_scat_coeffs, all_scat_coeffs = None, None, None, None
for galaxy_class, num_galaxies, encoder_choice, hidden_dim1, hidden_dim2, latent_dim, lr, reg, (initial_beta, final_beta), fold in itertools.product(
        galaxy_classes, num_galaxies_list, encoder_list, hidden_dim1, hidden_dim2, latent_dims, reg_params, learning_rates, initial_final_betas, [0] if FFCV else [6]): #change [0] for range(5)
    print(f"Training {encoder_choice} on {num_galaxies} galaxies of type {galaxy_class} with hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, latent_dim={latent_dim}, lr={lr}, beta={initial_beta}-{final_beta}, fold={fold}")

    # Create a log file and figure directory for each run
    runname = f'{galaxy_class}_{num_galaxies}_{encoder_choice}_{fold}'
    fig_path = f"./generator/VAE_{runname}"
    log_path = f"{fig_path}/log.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    file = open(log_path, 'w')

    # Free memory explicitly
    if 'train_loader' in locals() and train_loader is not None:
        del train_loader
    if 'test_loader' in locals() and test_loader is not None:
        del test_loader
    if 'train_images' in locals() and train_images is not None and torch.is_tensor(train_images):
        del train_images
    if 'test_images' in locals() and test_images is not None and torch.is_tensor(test_images):
        del test_images
    if 'batch_scat_coeffs' in locals() and batch_scat_coeffs is not None and torch.is_tensor(batch_scat_coeffs):
        del batch_scat_coeffs
    if 'all_scat_coeffs' in locals() and all_scat_coeffs is not None and torch.is_tensor(all_scat_coeffs):
        del all_scat_coeffs
    torch.cuda.empty_cache()
    
    data = load_galaxies(galaxy_class=galaxy_class, 
                        fold=fold,
                        img_shape=img_shape, 
                        sample_size=num_galaxies, 
                        process=True, 
                        train=True, 
                        runname=None, 
                        generated=False, 
                        reconstructed=False)
    train_images, train_labels, test_images, test_labels = data
    
    perm = torch.randperm(train_images.size(0))
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    
    perm = torch.randperm(test_images.size(0))
    test_images = test_images[perm]
    test_labels = test_labels[perm]
    
    print("Shape of train_imgs: ", np.shape(train_images))
    print("Shape of test_imgs: ", np.shape(test_images))
    
    # Check the input data
    print("Train images shape as read in:", np.shape(train_images))
    if IMGCHECK:
        plot_images(train_images, train_labels)
        plot_histograms(train_images, test_images)


    ##########################################################
    ############### CALCULATE INPUT DATA #####################
    ##########################################################

    def evaluate_symmetry(image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        gray_image = np.squeeze(image, axis=0)
        height, width = gray_image.shape
        left_half = gray_image[:, :width // 2]
        right_half = gray_image[:, width // 2:]
        right_half_flipped = np.flip(right_half, axis=1)
        
        if left_half.shape[1] != right_half_flipped.shape[1]:
            right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))
        
        vertical_fold = (left_half + right_half_flipped) / 2
        top_half = vertical_fold[:height // 2, :]
        bottom_half = vertical_fold[height // 2:, :]
        bottom_half_flipped = np.flip(bottom_half, axis=0)
        
        if top_half.shape[0] != bottom_half_flipped.shape[0]:
            bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))
        
        double_fold_diff = np.sum(np.abs(top_half - bottom_half_flipped))
        double_fold_score = double_fold_diff / (width * height)
        
        return double_fold_score
    
    asymmetry_scores_train = [evaluate_symmetry(image) for image in train_images]
    asymmetry_scores_test = [evaluate_symmetry(image) for image in test_images]
    
    asymmetry_scores_train = torch.tensor(asymmetry_scores_train, dtype=torch.float32)
    asymmetry_scores_test = torch.tensor(asymmetry_scores_test, dtype=torch.float32)


    if 'ST' in encoder_choice or 'Dual' in encoder_choice:
        scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order)        
        def compute_scattering_coeffs(images):
            print("Computing scattering coefficients...")
            with torch.no_grad():  # Disable gradient calculation
                start_time = time.time()
                scat_coeffs = scattering(images).detach()
                if scat_coeffs.dim() == 3:
                    scat_coeffs = scat_coeffs.unsqueeze(0)
                scat_coeffs = torch.squeeze(scat_coeffs)
                elapsed_time = time.time() - start_time
                print(f"Time taken to compute scattering coefficients: {elapsed_time:.2f} seconds")
                file.write(f"Time taken to compute scattering coefficients: {elapsed_time:.2f} seconds \n")

            return scat_coeffs

        torch.cuda.empty_cache()
        train_scat_coeffs = compute_scattering_coeffs(train_images).float()
        test_scat_coeffs = compute_scattering_coeffs(test_images).float()
        print("Shape of scat coeffs", train_scat_coeffs.shape)

    if 'lavg' in encoder_choice or 'ldiff' in encoder_choice:
        if 'lavg' in encoder_choice:
            train_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
        else:
            train_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
                
    ##########################################################
    ############ NORMALISE AND FILTER THE INPUT ##############
    ##########################################################
                
    # Normalize train and test images to [0, 1]
    train_images = normalize_to_0_1(train_images)
    test_images = normalize_to_0_1(test_images)

    if NORMALISETOPM:
        # If NORMALISETOPM is True, normalize to [-1, 1]
        train_images = normalize_to_minus1_1(train_images)
        test_images = normalize_to_minus1_1(test_images)

    # Handle scattering coefficients normalization in a similar way
    if 'ST' in encoder_choice or 'Dual' in encoder_choice:
        #train_scat_coeffs = normalize_to_0_1(train_scat_coeffs)
        #test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)

        if NORMALISETOPM:
            train_scat_coeffs = normalize_to_minus1_1(train_scat_coeffs)
            test_scat_coeffs = normalize_to_minus1_1(test_scat_coeffs)
    
    #Check input after renormalisation and filtering  
    if IMGCHECK: 
        plot_images(train_images, train_labels)
        plot_histograms(train_images, test_images)
        
    ##########################################################
    ############ CREATE DATA LOADERS #########################
    ##########################################################

    if 'ST' in encoder_choice or 'Dual' in encoder_choice: #Double dataset for convenience for dual model in training loop
        train_dataset = TensorDataset(train_scat_coeffs, train_images, asymmetry_scores_train)
        test_dataset = TensorDataset(test_scat_coeffs, test_images, asymmetry_scores_test)
    else: 
        train_dataset = TensorDataset(train_images, train_images, asymmetry_scores_train) 
        test_dataset = TensorDataset(test_images, test_images, asymmetry_scores_test) 

    # Create the data loaders
    def custom_collate(batch):
        if len(batch) < 3: # Drop the batch if it contains fewer than 3 images
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

    ##########################################################
    ############ TRAINING ####################################
    ##########################################################

    # Define the model
    num_classes = train_labels.max().item() 
    print(f"train_labels min: {train_labels.min()}, max: {train_labels.max()}, num_classes: {num_classes}")
    if 'C' in encoder_choice and encoder_choice != 'CNN':
        train_labels = torch.nn.functional.one_hot(train_labels.squeeze(), num_classes=num_classes).float()
        test_labels = torch.nn.functional.one_hot(test_labels.squeeze(), num_classes=num_classes).float()
    scatshape = np.shape(train_scat_coeffs)[1:] if "ST" in encoder_choice or "Dual" in encoder_choice else np.array([1, 1, 1])    
    model = get_model(encoder_choice, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes, J=J)
    model.to(DEVICE)


    # Print model summary
    if 'C' in encoder_choice and encoder_choice != 'CNN':
        print("Summary not available for conditional VAEs")
    elif 'ST' in encoder_choice:
        summary(model, input_size=train_scat_coeffs[0].shape, device=DEVICE)
    elif 'Dual' in encoder_choice:
        summary(model, input_size=[img_shape, train_scat_coeffs[0].shape], device=DEVICE)      
    else:
        summary(model, input_size=img_shape, device=DEVICE)


    # Define the loss function and optimizer
    def lr_lambda(epoch):
        if epoch < num_epochs * 0.05:
            return epoch / (num_epochs * 0.05)
        return 0.5 * (1 + np.cos(np.pi * (epoch - num_epochs * 0.05) / (num_epochs * 0.95)))
    model.apply(lambda m: nn.init.calculate_gain('leaky_relu', 0.2) if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) else None)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=reg)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    beta_increment = (final_beta - initial_beta) / num_epochs
    
    MSEfun = ['blablabla', 'normalised', 'radial', 'weighted', 'intensity', 'maxintensity', 'custom', 'combined', 'experimental', 'basic'][ind]
    print("Ind: ", ind, " and MSEfun: ", MSEfun)
    if MSEfun == 'radial': mse_loss = RadialWeightedMSELoss(threshold=0.1, intensity_weight=0.001, radial_weight=0.001)
    elif MSEfun == 'normalised': mse_loss = NormalisedWeightedMSELoss(threshold=0.1, weight=0.5) 
    elif MSEfun == 'weighted': mse_loss = WeightedMSELoss(threshold=0.1, weight=0.001)
    elif MSEfun == 'intensity': mse_loss = CustomIntensityWeightedMSELoss(intensity_threshold=0.07, intensity_weight=0.0001, log_weight=0.0001)
    elif MSEfun == 'maxintensity': mse_loss = MaxIntensityMSELoss(intensity_weight=0.001)
    elif MSEfun == 'standard': mse_loss = StandardMSELoss()
    elif MSEfun == 'custom': mse_loss = CustomMSELoss(intensity_weight=0.001, sum_weight=0.001)
    elif MSEfun == 'combined': mse_loss = CombinedMSELoss(intensity_weight=0.001, sum_weight=0.001, threshold=0.1, high_intensity_weight=0.001)
    elif MSEfun == 'experimental': mse_loss = ExperimentalMSELoss(intensity_weight=0.001, sum_weight=0.001, histogram_weight=0.001, threshold=0.5, high_intensity_weight=0.001)
    elif MSEfun == 'basic': mse_loss = BasicMSELoss(threshold=0)
    else: mse_loss = nn.MSELoss(reduction='sum')
    
    
    def asymmetry_weight(score, scale_factor=100, offset=1.0):
        #return scale_factor * score + offset
        return scale_factor * (score ** 2) + offset
        #return scale_factor * torch.log1p(score) + offset
    
    def vae_loss_function(x, x_hat, mean, log_var, asymmetry_scores, beta=1.0):
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        if x_hat.shape != x.shape:
            x_hat = x_hat.view(x.shape)
        # Calculate weighted MSE loss
        weights = asymmetry_weight(asymmetry_scores).to(x.device)
        RecLoss = mse_loss(x_hat, x) * weights
        return RecLoss.mean() + beta * KLD

    # Train the model
    model.train()
    train_losses, val_losses = [], []
    train_mse, train_kl = [], []
    val_mse, val_kl = [], []
    best_loss, best_model_state = float('inf'), None
    early_stopping = EarlyStopping(patience=25)
    model_path = f'{fig_path}/model.pth'
    start_time = time.time()
    with tqdm(total=num_epochs, desc=f"Training {runname}", position=0) as pbar:
        for epoch in range(num_epochs):
            
            # Training
            beta = initial_beta + epoch * beta_increment
            overall_loss = 0
            epoch_mse, epoch_kl = 0, 0
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", leave=False, position=1) as epoch_bar:
                for batch_idx, batch in epoch_bar:
                    if batch is None:
                        continue  
                    scat_coeffs, x, asymmetry_scores = batch  
                    x = x.to(DEVICE)
                    asymmetry_scores = asymmetry_scores.to(DEVICE)
                    
                    optimizer.zero_grad()
                    if 'ST' in encoder_choice or 'Dual' in encoder_choice: # Scattering transform
                        scat_coeffs = scat_coeffs.to(DEVICE)
                        if 'C' in encoder_choice: # Conditional VAE
                            labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                            if 'MLP' in encoder_choice:
                                x_hat, mean, log_var = model(scat_coeffs, labels)
                            elif 'CDual' in encoder_choice:
                                x_hat, mean, log_var = model(x, scat_coeffs, labels)
                        elif 'Dual' in encoder_choice:
                            x_hat, mean, log_var = model(x, scat_coeffs)
                        else:
                            x_hat, mean, log_var = model(scat_coeffs)
                    elif 'C' in encoder_choice and encoder_choice != 'CNN': # Conditional VAE - No scattering transform
                        labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                        x_hat, mean, log_var = model(x, labels)
                    else:
                        x_hat, mean, log_var = model(x)
                        
                    if x_hat.shape != x.shape:
                        x_hat = x_hat.view(x.shape)

                    loss = vae_loss_function(x, x_hat, mean, log_var, asymmetry_scores, beta=beta)
                    mse = mse_loss(x_hat, x).item()
                    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()).item()
                    overall_loss += float(loss.item())
                    epoch_mse += float(mse)
                    epoch_kl += float(kl)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    #scheduler.step()
                    epoch_bar.set_description(f"Epoch {epoch + 1} Batch {batch_idx + 1} Loss: {loss.item():.4f}")

                average_loss = overall_loss / len(train_loader.dataset)
                epoch_mse /= len(train_loader.dataset)
                epoch_kl /= len(train_loader.dataset)
                train_losses.append(average_loss)
                train_mse.append(epoch_mse)
                train_kl.append(epoch_kl)

                if average_loss < best_loss or epoch == 0:
                    best_loss = average_loss
                    best_model_state = model.state_dict()
                    
                # Validation
                model.eval()
                val_loss = 0
                val_mse_epoch, val_kl_epoch = 0, 0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_loader):
                        if batch is None:
                            continue 
                        scat_coeffs, x, asymmetry_scores = batch  
                        x = x.to(DEVICE)
                        asymmetry_scores = asymmetry_scores.to(DEVICE)
                        
                        if 'ST' in encoder_choice or 'Dual' in encoder_choice: # Scattering transform
                            scat_coeffs = scat_coeffs.to(DEVICE)
                            if 'C' in encoder_choice: # Conditional VAE
                                labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                                if 'MLP' in encoder_choice:
                                    x_hat, mean, log_var = model(scat_coeffs, labels)
                                elif 'CDual' in encoder_choice:
                                    x_hat, mean, log_var = model(x, scat_coeffs, labels)
                            elif 'Dual' in encoder_choice:
                                x_hat, mean, log_var = model(x, scat_coeffs)
                            else:
                                x_hat, mean, log_var = model(scat_coeffs)
                        elif 'C' in encoder_choice and encoder_choice != 'CNN': # Conditional VAE - No scattering transform
                            labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                            x_hat, mean, log_var = model(x, labels)
                        else:
                            x_hat, mean, log_var = model(x)

                        if x_hat.shape != x.shape:
                            x_hat = x_hat.view(x.shape)
                            
                        loss = vae_loss_function(x, x_hat, mean, log_var, asymmetry_scores, beta=beta)
                        mse = mse_loss(x_hat, x)
                        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()).item()
                        val_loss += float(loss.item())
                        val_mse_epoch += float(mse)
                        val_kl_epoch += float(kl)
                    val_loss /= len(test_loader.dataset)
                    val_mse_epoch /= len(test_loader.dataset)
                    val_kl_epoch /= len(test_loader.dataset)
                    val_losses.append(val_loss)
                    val_mse.append(val_mse_epoch)
                    val_kl.append(val_kl_epoch)
                    
                model.train()
                pbar.update(1)
                pbar.set_postfix_str(f"Average Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}")
                early_stopping(val_loss, model, model_path) # Save the model if validation loss decreases
                if ES:
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                else:
                    if average_loss < best_loss:
                        best_loss = average_loss
                        best_model_state = model.state_dict()
                
    elapsed_time = time.time() - start_time
    file.write(f"Time taken to train the model: {elapsed_time:.2f} seconds \n")
    file.write(f"Total epochs run before early stopping: {epoch}\n")
    file.write(f"Training Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}, ")
    file.write(f"Training MSE: {epoch_mse:.4f}, Training KL: {epoch_kl:.4f}, ")
    file.write(f"Validation MSE: {val_mse_epoch:.4f}, Validation KL: {val_kl_epoch:.4f}\n")

    ##########################################################
    ############ EVALUATION ##################################
    ##########################################################

    # Evaluate the model
    model.load_state_dict(best_model_state) # Load the best model
    model.eval()
    reconstructed_images_list, original_images_list = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if batch is None:
                continue 
            scat_coeffs, x, asymmetry_scores = batch  
            x = x.to(DEVICE)
            if 'ST' in encoder_choice or 'Dual' in encoder_choice: # Scattering transform
                scat_coeffs = scat_coeffs.to(DEVICE)
                if 'C' in encoder_choice: # Conditional VAE
                    labels = test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                    if 'MLP' in encoder_choice:
                        x_hat, mean, log_var = model(scat_coeffs, labels)
                    elif 'CDual' in encoder_choice:
                        x_hat, mean, log_var = model(x, scat_coeffs, labels)
                elif 'Dual' in encoder_choice:
                    x_hat, mean, log_var = model(x, scat_coeffs)
                else:
                    x_hat, mean, log_var = model(scat_coeffs)
            elif 'C' in encoder_choice and encoder_choice != 'CNN': # Conditional VAE - No scattering transform
                labels = test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                x_hat, mean, log_var = model(x, labels)
            else:
                x_hat, mean, log_var = model(x)

            # Verify and append only if shapes match
            if x_hat.shape == x.shape:
                reconstructed_images_list.append(x_hat.cpu())
                original_images_list.append(x.cpu())
            else:
                print(f"Shape mismatch: x_hat shape: {x_hat.shape}, x shape: {x.shape}")
            if len(reconstructed_images_list) >= 100:
                break
            
            # Save the loss map
            if batch_idx == 0:
                if MSEfun in ['radial', 'normalised', 'intensity', 'weighted', 'maxintensity', 'standard', 'custom', 'combined']:
                    loss_map, weight_map = mse_loss(x_hat, x, return_map=True)
                    loss_map = loss_map.cpu().numpy()
                    loss_map[loss_map == 0] = np.nan
                    plot_weight_and_loss_maps(weight_map, loss_map, x, x_hat, savepath=f'{fig_path}/lossmap.png') 
                
    all_reconstructed_images = torch.cat(reconstructed_images_list)
    all_original_images = torch.cat(original_images_list)
    supertitle = f"VAE with {encoder_choice} encoder, hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, latent_dim={latent_dim}, \n for {num_galaxies} galaxies of type {galaxy_class} and {num_epochs} epochs."
    
    
    def plot_image_grid(images, img_shape, nrow=10, ncol=10, title='Image grid', savepath=f'{fig_path}/top_100_faint_grid.png', cmap='viridis'):
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
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    if PLOTFAILED:
        def plot_top_mse_images(all_original_images, all_reconstructed_images, num_images=5, supertitle="Failed reconstructions"):
            mse_losses = []
            for orig, recon in zip(all_original_images, all_reconstructed_images):
                mse_loss_per_image = torch.mean((orig - recon) ** 2).item()
                mse_losses.append(mse_loss_per_image)
                
            top_5_mse_indices = np.argsort(mse_losses)[-5:]
            originals = all_original_images[top_5_mse_indices]
            reconstructions = all_reconstructed_images[top_5_mse_indices]
            
            fig, axs = plt.subplots(2, num_images, figsize=(num_images * 4, 10))  # 2 rows, num_images columns
            fig.suptitle(supertitle, fontsize=16)
            vmin, vmax = 0, 1
            for i in range(num_images):
                axs[0, i].imshow(originals[i].squeeze().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
                axs[0, i].set_title(f'Original Image {i+1}')
                axs[0, i].axis('off')

                axs[1, i].imshow(reconstructions[i].squeeze().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
                axs[1, i].set_title(f'Reconstructed Image {i+1}\nMSE Loss: {mse_losses[i]:.4f}')
                axs[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(f'{fig_path}/failed_reconstructions.png', bbox_inches='tight')
            plt.close()

            
        def plot_top_low_intensity_images(originals, reconstructions, num_images=5, supertitle="Faint reconstructions"):
            peak_intensities_recon = [torch.max(recon).item() for recon in reconstructions]
            peak_intensities_orig = [torch.max(orig).item() for orig in originals]
            
            top_low_intensity_indices = np.argsort(peak_intensities_recon)[:num_images]
            top_low_intensity_originals = originals[top_low_intensity_indices]
            top_low_intensity_reconstructions = reconstructions[top_low_intensity_indices]
            
            fig, axs = plt.subplots(2, num_images, figsize=(num_images * 4, 10))  # 2 rows, num_images columns
            fig, axs = plt.subplots(2, num_images, figsize=(14, 7))
            fig.suptitle(supertitle, fontsize=16)
            vmin, vmax = 0, 1
            for i in range(num_images):
                axs[0, i].imshow(top_low_intensity_originals[i].squeeze().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
                axs[0, i].axis('off')

                axs[1, i].imshow(top_low_intensity_reconstructions[i].squeeze().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
                axs[1, i].axis('off')

            axs[0, 0].text(-0.1, 0.5, "Original images", va='center', ha='center', fontsize=12, rotation=90, transform=axs[0, 0].transAxes)
            axs[1, 0].text(-0.1, 0.5, "Reconstructed images", va='center', ha='center', fontsize=12, rotation=90, transform=axs[1, 0].transAxes)

            plt.tight_layout()
            plt.savefig(f'{fig_path}/low_peak_intensity_comparisons.png', bbox_inches='tight')
            plt.close()
            
        def get_top_100_mse_images(original_images, reconstructed_images):
            mse_losses = []
            for orig, recon in zip(original_images, reconstructed_images):
                mse_loss_per_image = torch.mean((orig - recon) ** 2).item()
                mse_losses.append(mse_loss_per_image)

            # Get the indices of the 100 images with the highest MSE losses
            top_100_mse_indices = np.argsort(mse_losses)[-100:]
            top_100_mse_originals = original_images[top_100_mse_indices]
            top_100_mse_reconstructions = reconstructed_images[top_100_mse_indices]
            
            return top_100_mse_originals, top_100_mse_reconstructions

        # Function to calculate the top 100 images with the faintest emissions (lowest peak intensity)
        def get_top_100_faint_images(original_images, reconstructed_images):
            peak_intensities_recon = [torch.max(recon).item() for recon in reconstructed_images]
            
            # Get the indices of the 100 images with the lowest peak intensities
            top_100_faint_indices = np.argsort(peak_intensities_recon)[:100]
            top_100_faint_originals = original_images[top_100_faint_indices]
            top_100_faint_reconstructions = reconstructed_images[top_100_faint_indices]
            
            return top_100_faint_originals, top_100_faint_reconstructions

        plot_top_low_intensity_images(all_original_images, all_reconstructed_images, num_images=5, supertitle=supertitle)
        plot_top_mse_images(all_original_images, all_reconstructed_images, num_images=5, supertitle=supertitle)
         
        # Plot the 100 reconstructions with the highest MSE loss
        top_100_mse_originals, top_100_mse_reconstructions = get_top_100_mse_images(all_original_images, all_reconstructed_images)
        plot_image_grid(top_100_mse_originals, img_shape, 10, 10, 
                        title="Originals of top 100 reconstructions with highest MSE loss", 
                        savepath=f'{fig_path}/top_100_mse_grid.png')
        
        # Plot the 100 reconstructions with the faintest emission
        top_100_faint_originals, top_100_faint_reconstructions = get_top_100_faint_images(all_original_images, all_reconstructed_images)
        plot_image_grid(top_100_faint_originals, img_shape, 10, 10,  
                        title="Originals of top 100 reconstructions with faintest emission", 
                        savepath=f'{fig_path}/top_100_faint_grid.png')



    if SAVEIMGS:
        torch.save(all_reconstructed_images, f'{fig_path}/reconstructed_images.pt')
        print(f'All reconstructed images saved in tensor format with shape: {all_reconstructed_images.shape}')

    # Quick evaluation
    file.write(f"Supertitle: {supertitle}\n")
    file.write(f"img_shape: {img_shape}\n")
    file.write(f"J: {J}\n")
    file.write(f"L: {L}\n")
    file.write(f"order: {order}\n")
    file.write(f"batch_size: {batch_size}\n")
    file.write(f"learning rate: {lr}\n")
    file.close()
        
    loss_vs_epoch(train_losses, val_losses, save=True, save_path=f"{fig_path}/loss.png")
    
    # Reconstruction comparison
    x_reshaped = all_original_images.view(-1, img_shape[-2], img_shape[-1]).detach().cpu()
    x_hat_reshaped = all_reconstructed_images.view(-1, img_shape[-2], img_shape[-1]).detach().cpu()
    vae_plot_comparison(x_reshaped[:min(len(x_reshaped), 5)], x_hat_reshaped[:min(len(x_hat_reshaped), 5)], supertitle=supertitle, num_images=min(len(x_reshaped), 5), save=True, save_path=f"{fig_path}/comparison.png")

    # Generate new images and plot
    with torch.no_grad():
        noise = torch.randn((100, latent_dim)).to(DEVICE)
        if 'C' in encoder_choice and encoder_choice != 'CNN':
            sample_labels = torch.arange(0, 100) % train_labels.size(1)
            sample_labels = sample_labels.to(DEVICE)
            sample_labels_one_hot = torch.nn.functional.one_hot(sample_labels, num_classes=train_labels.size(1)).float()
            generated_images = model.decoder(noise, sample_labels_one_hot)
        else:
            generated_images = model.decoder(noise)

        plot_image_grid(generated_images, img_shape, 10, 10, title="Generations", savepath=f"{fig_path}/generation.png")
        plot_image_grid(all_reconstructed_images, img_shape, 10, 10, title="Reconstructions", savepath= f"{fig_path}/reconstructions.png")

    train_loader = DataLoader(TensorDataset(train_images, train_images), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    plot_original_images(train_loader, img_shape[-1], 10, save_path=f'{fig_path}/originals.png')
    print(f"Number of images in train_loader: {len(train_loader.dataset)}")
    for images, _ in train_loader:
        print(f"Loaded batch of images with shape: {images.shape}")
        break
    plot_reconstructions(test_loader, test_labels, batch_size, model, runname, DEVICE=DEVICE, num_images=5, cmap='viridis')
