import itertools
import os
from utils.data_loader import load_galaxies, get_classes
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from kymatio.torch import Scattering2D
from utils.VAE_models import get_VAE_model
from utils.custom_mse import NormalisedWeightedMSELoss, CustomMSELoss, WeightedMSELoss, RadialWeightedMSELoss, CustomIntensityWeightedMSELoss, MaxIntensityMSELoss, StandardMSELoss, CombinedMSELoss, ExperimentalMSELoss, BasicMSELoss
from utils.scatter_reduction import lavg, ldiff
from utils.training_tools import EarlyStopping 
from utils.calc_tools import normalise_images
from torchsummary import summary
from tqdm import tqdm
import heapq
import time
from utils.plotting import vae_plot_comparison, loss_vs_epoch, plot_original_images, plot_weight_and_loss_maps, plot_histograms, plot_reconstructions
import matplotlib.pyplot as plt


######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

#encoder_list = ['CDual', 'CCNN', 'CSTMLP', 'CldiffSTMLP', 'ClavgSTMLP']
#encoder_list = ['CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP']
encoder_list = ['Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP']
#encoder_list = ['ldiffSTMLP']

FFCV = True               # Use five-fold cross-validation
NORMALISEIMGS = True      # Normalise imgs with themselves to [0, 1]
NORMALISEIMGSTOPM = False # Normalise imgs with themselves to [-1, 1]
NORAMLISESCS = False      # Normalise scattering coefficients with themselves to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients with themselves to [-1, 1]

IMGCHECK = False          # Check the input images (Tool for control)
SAVEIMGS = False          # Save the reconstructed images in tensor format
PLOTFAILED = True         # Plot the top 5 images with the largest MSE loss and weakest peak intensities
REMOVEOUTLIERS = True     # Filter away problematic images

F = 0 #Choose the loss function to use

#galaxy_classes = [[10, 11, 12, 13]] #Use double square parenthesis for conditional VAEs
galaxy_classes = [40]
hidden_dim1 = [256]
hidden_dim2 = [128]
latent_dims = [64]
learning_rates = [1e-4]
reg_params = [1e-4] 
initial_final_betas = [(1e-1, 1)]
num_epochs_cuda = 200; num_epochs_cpu = 200
batch_size = 32
J, L, order = 2, 12, 2

#1001028

######################################################################################################
################################### CODE THE RUN ####################################################
######################################################################################################

if NORMALISEIMGSTOPM:
    NORMALISEIMGS = True 
    A = 2
elif NORMALISEIMGS:
    A = 1
else:
    A = 0

if NORMALISESCSTOPM:
    NORAMLISESCS = True
    B = 2
elif NORAMLISESCS:
    B = 1
else:
    B = 0

C = 1 # 0: No normalisation, 1: Batch normalisation, 2: Groupnorm 
D = 1 if REMOVEOUTLIERS else 0 # 0: No filter, 1: Filter
E = J # Number of scales
# F: Loss function

if galaxy_classes[0] in [40, 41, 42, 43, 44, 45, 46, 47, 48]:
    crop_size = (1600, 1600)
    downsample_size = (128, 128)
else:
    crop_size = (128, 128)
    downsample_size = (128, 128)
img_shape = (1, downsample_size[0], downsample_size[1]) # (1, 128, 128) for 128x128 images
    

num_galaxies_list = [int(1000000+A*1e5+B*1e4+C*1e3+D*1e2+E*10+F)]
print("Num_galaxies: ", num_galaxies_list[0])



#########################################################################################################################

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Running script 2.scattervae_training.py")


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
        galaxy_classes, num_galaxies_list, encoder_list, hidden_dim1, hidden_dim2, latent_dims, reg_params, learning_rates, initial_final_betas, range(5) if FFCV else [0]): 
    print(f"Training {encoder_choice} on {num_galaxies} galaxies of type {galaxy_class} with hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, latent_dim={latent_dim}, lr={lr}, beta={initial_beta}-{final_beta}, fold={fold}")

    # Create a log file and figure directory for each run
    runname = f'{galaxy_class}_{num_galaxies}_{encoder_choice}_{fold}'
    fig_path = f"./generator/VAE_{runname}"
    log_path = f"{fig_path}/log.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    file = open(log_path, 'w')
    
    if 'train_loader' in locals(): del train_loader
    if 'test_loader' in locals(): del test_loader
    if 'train_images' in locals(): del train_images
    if 'test_images' in locals(): del test_images
    if 'train_scat_coeffs' in locals(): del train_scat_coeffs
    if 'test_scat_coeffs' in locals(): del test_scat_coeffs
    torch.cuda.empty_cache() 

    if galaxy_class == 100:
        # Directly load the patches data from your .npz file
        data = np.load('/users/mbredber/scratch/Scripts/radio_galaxies_patches_128.npz')['patches']
        print("Type, ", type(data))
        train_images = torch.tensor(data, dtype=torch.float32)  # Assuming 'images' is the correct key
        train_labels = torch.zeros(len(train_images), dtype=torch.long)

        # Split the data into train and test sets
        train_size = int(0.8 * len(train_images))
        test_size = len(train_images) - train_size
        train_dataset, test_dataset = random_split(
            TensorDataset(train_images, train_labels),
            [train_size, test_size]
        )

        # Convert datasets to tensors
        train_images, train_labels = zip(*train_dataset)
        test_images, test_labels = zip(*test_dataset)

        # Convert to tensors
        train_images = torch.stack(train_images)
        train_labels = torch.stack(train_labels)
        test_images = torch.stack(test_images)
        test_labels = torch.stack(test_labels)

        # Add the channel dimension
        train_images = train_images.unsqueeze(1)  # Shape becomes (320, 1, 64, 64)
        test_images = test_images.unsqueeze(1)

    else:
        data = load_galaxies(galaxy_class=galaxy_class, 
                            fold=fold,
                            crop_size=crop_size,
                            downsample_size=downsample_size,
                            sample_size=1000000, 
                            REMOVEOUTLIERS=REMOVEOUTLIERS,
                            train=True)
        train_images, train_labels, test_images, test_labels = data

    print("Shape of train_imgs: ", np.shape(train_images))
    print("Shape of test_imgs: ", np.shape(test_images))
    
    # Shuffle images
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm].float()
    train_labels = train_labels[perm].float()
    perm = torch.randperm(len(test_images))
    test_images = test_images[perm].float()
    test_labels = test_labels[perm].float()
    
    # Check the input data
    print("Train images shape as read in:", np.shape(train_images))
    if IMGCHECK:
        plot_histograms(train_images, test_images)

    # Prepare input data
    if 'ST' in encoder_choice or 'Dual' in encoder_choice:
        scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order)       
        def compute_scattering_coeffs_batched(images, scattering, batch_size=128, device="cpu"):
            print("Computing scattering coefficients in batches on CPU...")
            scat_coeffs_list = []
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size].to(device)
                    batch_scat = scattering(batch).detach()  # e.g. shape [B, 1, C, H, W]
                    # Squeeze out the singleton dimension at index 1 if present.
                    if batch_scat.shape[1] == 1:
                        batch_scat = batch_scat.squeeze(1)  # becomes [B, C, H, W]
                    scat_coeffs_list.append(batch_scat.cpu())
                scat_coeffs = torch.cat(scat_coeffs_list, dim=0)
            return scat_coeffs

        torch.cuda.empty_cache()
        train_scat_coeffs = compute_scattering_coeffs_batched(train_images, scattering, batch_size=128, device="cpu").float()
        test_scat_coeffs = compute_scattering_coeffs_batched(test_images, scattering, batch_size=128, device="cpu").float()
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
                
    ##########################################################
    ############ NORMALISE AND FILTER THE INPUT ##############
    ##########################################################

    # Bring train + test together, normalise globally, then split back
    all_imgs = torch.cat([train_images, test_images], dim=0)
    if NORMALISEIMGS or NORMALISEIMGSTOPM:
        if NORMALISEIMGSTOPM:
            all_imgs = normalise_images(all_imgs, -1, 1)
        else:
            all_imgs = normalise_images(all_imgs, 0, 1)
    train_images, test_images = all_imgs[:len(train_images)], all_imgs[len(train_images):]

    # If using scattering, do the same for those coefficients
    if 'ST' in encoder_choice or 'Dual' in encoder_choice:
        all_scat = torch.cat([train_scat_coeffs, test_scat_coeffs], dim=0)
        if NORAMLISESCS or NORMALISESCSTOPM:
            if NORMALISESCSTOPM:
                all_scat = normalise_images(all_scat, -1, 1)
            else:
                all_scat = normalise_images(all_scat, 0, 1)
        train_scat_coeffs, test_scat_coeffs = all_scat[:len(train_scat_coeffs)], all_scat[len(train_scat_coeffs):]

    
    #Check input after renormalisation and filtering  
    if IMGCHECK: 
        plot_histograms(train_images, test_images)
        
    ##########################################################
    ############ CREATE DATA LOADERS #########################
    ##########################################################

    if 'ST' in encoder_choice or 'Dual' in encoder_choice: #Double dataset for convenience for dual model in training loop
        train_dataset = TensorDataset(train_scat_coeffs, train_images)
        test_dataset = TensorDataset(test_scat_coeffs, test_images)
    else: 
        train_dataset = TensorDataset(train_images, train_images) 
        test_dataset = TensorDataset(test_images, test_images) 

    # Create the data loaders
    def custom_collate(batch):
        if len(batch) < 3: # Drop the batch if it contains fewer than 3 images
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)

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
    model = get_VAE_model(encoder_choice, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes, J=J)
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
        
    #tf.keras.utils.plot_model(vae,"model.png", show_shapes = True,)


    # Define the loss function and optimizer
    def lr_lambda(epoch):
        if epoch < num_epochs * 0.05:
            return epoch / (num_epochs * 0.05)
        return 0.5 * (1 + np.cos(np.pi * (epoch - num_epochs * 0.05) / (num_epochs * 0.95)))
    model.apply(lambda m: nn.init.calculate_gain('leaky_relu', 0.2) if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) else None)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=reg)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    beta_increment = (final_beta - initial_beta) / num_epochs
    
    MSEfun = ['standard', 'normalised', 'radial', 'weighted', 'intensity', 'maxintensity', 'custom', 'combined', 'experimental', 'basic'][F]
    print("F: ", F, " and MSEfun: ", MSEfun)
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
    
    def vae_loss_function(x, x_hat, mean, log_var, beta=1.0):
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        if x_hat.shape != x.shape:
            x_hat = x_hat.view(x.shape)
        RecLoss = mse_loss(x_hat, x)
        return RecLoss + beta * KLD

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
                    scat_coeffs, x = batch  
                    x = x.to(DEVICE)
                    
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

                    loss = vae_loss_function(x, x_hat, mean, log_var, beta=beta)
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
                        scat_coeffs, x = batch  
                        x = x.to(DEVICE)
                        
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
                            
                        loss = vae_loss_function(x, x_hat, mean, log_var, beta=beta)
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
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                elif average_loss < best_loss:
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
            scat_coeffs, x = batch
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
            #if x_hat.shape == x.shape:
            #    reconstructed_images_list.append(x_hat.cpu())
            #    original_images_list.append(x.cpu())
            #else:
            #    print(f"Shape mismatch: x_hat shape: {x_hat.shape}, x shape: {x.shape}")
            
            # Ensure x_hat matches x shape before appending
            if x_hat.shape != x.shape:
                x_hat = x_hat.view(x.shape)
            reconstructed_images_list.append(x_hat.cpu())
            original_images_list.append(x.cpu())
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
            
        def old_get_top_100_mse_images(original_images, reconstructed_images):
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
        def old_get_top_100_faint_images(original_images, reconstructed_images):
            peak_intensities_recon = [torch.max(recon).item() for recon in reconstructed_images]
            
            # Get the indices of the 100 images with the lowest peak intensities
            top_100_faint_indices = np.argsort(peak_intensities_recon)[:100]
            top_100_faint_originals = original_images[top_100_faint_indices]
            top_100_faint_reconstructions = reconstructed_images[top_100_faint_indices]
            
            return top_100_faint_originals, top_100_faint_reconstructions
        
        
        
        def get_top_n_images_by_metric_batch(original_images, reconstructed_images, metric_fn, top_n=100, batch_size=128):
            """
            Optimized function to extract the top N images based on a given metric, reducing memory usage.
            """
            top_heap = []  # Min-heap to store top N values
            indices = []

            for i in range(0, len(original_images), batch_size):
                batch_orig = original_images[i:i + batch_size]
                batch_recon = reconstructed_images[i:i + batch_size]
                for idx, (orig, recon) in enumerate(zip(batch_orig, batch_recon)):
                    metric = metric_fn(orig, recon)
                    if len(top_heap) < top_n:
                        heapq.heappush(top_heap, (metric, i + idx))
                    else:
                        heapq.heappushpop(top_heap, (metric, i + idx))

            top_indices = [idx for _, idx in sorted(top_heap, reverse=True)]
            top_originals = original_images[top_indices]
            top_reconstructions = reconstructed_images[top_indices]

            return top_originals, top_reconstructions


        # Function to get top 100 highest MSE images
        def get_top_100_mse_images(original_images, reconstructed_images, batch_size=128):
            return get_top_n_images_by_metric_batch(
                original_images, reconstructed_images, 
                metric_fn=lambda orig, recon: torch.mean((orig - recon) ** 2).item(), 
                top_n=100, 
                batch_size=batch_size
            )

        # Function to get top 100 faintest images (lowest peak intensity)
        def get_top_100_faint_images(original_images, reconstructed_images, batch_size=128):
            return get_top_n_images_by_metric_batch(
                original_images, reconstructed_images, 
                metric_fn=lambda _, recon: torch.max(recon).item(), 
                top_n=100, 
                batch_size=batch_size
            )



        # plot_top_low_intensity_images(all_original_images, all_reconstructed_images, num_images=5, supertitle=supertitle)
        # plot_top_mse_images(all_original_images, all_reconstructed_images, num_images=5, supertitle=supertitle)
         
        ## Plot the 100 reconstructions with the highest MSE loss
        top_100_mse_originals, top_100_mse_reconstructions = get_top_100_mse_images(all_original_images, all_reconstructed_images)
        plot_image_grid(top_100_mse_originals, img_shape, 4, 4, 
                        title="Originals of top 100 reconstructions with highest MSE loss", 
                        savepath=f'{fig_path}/top_100_mse_grid.png')
        
        # Plot the 100 reconstructions with the faintest emission
        top_100_faint_originals, top_100_faint_reconstructions = get_top_100_faint_images(all_original_images, all_reconstructed_images)
        plot_image_grid(top_100_faint_originals, img_shape, 4, 4,  
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

        plot_image_grid(generated_images, img_shape, 4, 4, title="Generations", savepath=f"{fig_path}/generation.png")
        plot_image_grid(all_reconstructed_images, img_shape, 4, 4, title="Reconstructions", savepath= f"{fig_path}/reconstructions.png")

    train_loader = DataLoader(TensorDataset(train_images, train_images), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    plot_original_images(train_loader, img_shape[-1], 10, save_path=f'{fig_path}/originals.png')
    print(f"Number of images in train_loader: {len(train_loader.dataset)}")
    for images, _ in train_loader:
        print(f"Loaded batch of images with shape: {images.shape}")
        break
    plot_reconstructions(test_loader, test_labels, batch_size, model, runname, DEVICE=DEVICE, num_images=5, cmap='viridis')
