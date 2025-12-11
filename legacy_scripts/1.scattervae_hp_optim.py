import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from kymatio.torch import Scattering2D
from tqdm import tqdm
from utils.data_loader import load_galaxies
from utils.models import get_model
from utils.custom_mse import ExperimentalMSELoss
from utils.scatter_reduction import lavg, ldiff
from utils.training_tools import EarlyStopping
import random

initial_params = {
    'num_galaxies': 500,
    'encoder_choice': 'STMLP',
    'latent_dim': 64,
    'learning_rate': 1e-4,
    'initial_beta': 1e-2,
    'final_beta': 0.5,
    'th': 0.5,
    'mseweight': 5,
    'J': 1,  
    'L': 12,
    'reg_param': 1e-4  
}

param_bounds = {
    'num_galaxies': (500, 500),
    'latent_dim': (8, 128),
    'learning_rate': (1e-5, 5e-1),
    'initial_beta': (1e-4, 1e-1),
    'final_beta': (1e-2, 2),
    'th': (0.01, 0.99),
    'mseweight': (1, 10),
    'J': (1, 4),
    'L': (4, 16),
    'reg_param': (1e-10, 1e-3)  
}

param_step_sizes = {
    'num_galaxies': 1000,
    'latent_dim': 8,
    'learning_rate': 1e-3,
    'initial_beta': 1e-3,
    'final_beta': 0.1,
    'J': 2,
    'L': 8,
    'reg_param': 1e-5  
}

encoder_choices = ['STMLP']

galaxy_class = 11
img_shape = (1, 128, 128)
batch_size = 128
order = 2 

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = 150
else:
    DEVICE = "cpu"
    num_epochs = 20

# Implements a random walk over the hyperparameter space to explore new configurations
def random_walk(params, bounds, step_sizes, choices):
    new_params = params.copy()
    for key in params:
        if key in step_sizes:
            step = random.uniform(-step_sizes[key], step_sizes[key])
            new_params[key] += float(step)
            new_params[key] = max(min(new_params[key], bounds[key][1]), bounds[key][0])
            
            # Ensure certain parameters are cast to integers
            if key in ['num_galaxies', 'J', 'L', 'latent_dim']:
                new_params[key] = int(new_params[key])
                
        elif key == 'encoder_choice':
            new_params[key] = random.choice(choices)
    return new_params


# Train the VAE model with the current set of hyperparameters
def train_vae(params):
    runname = (f"{params['num_galaxies']}_{params['encoder_choice']}_{params['latent_dim']}_"
            f"{params['learning_rate']:.1e}_{params['initial_beta']:.2f}-{params['final_beta']:.2f}_"
            f"{params['J']}_{params['L']}")
    print("Training ", runname)

    train_images, train_labels, test_images, test_labels = load_galaxies(galaxy_class, img_shape=img_shape, sample_size=params['num_galaxies'], fold=5, train=True)
    
    train_labels, test_labels = train_labels.to(DEVICE), test_labels.to(DEVICE)
    train_images, test_images = train_images.to(DEVICE), test_images.to(DEVICE)

    scattering = Scattering2D(J=params['J'], L=params['L'], shape=img_shape[1:], max_order=order).to(DEVICE)
    train_scat_coeffs = scattering(train_images.contiguous()).squeeze()
    test_scat_coeffs = scattering(test_images.contiguous()).squeeze()
    print('Shape of train scat coeffs: ', train_scat_coeffs.shape)

    scat_shape = train_scat_coeffs.shape[1:]

    if params['encoder_choice'] in ["lavgSTMLP", "ldiffSTMLP"]:
        if params['encoder_choice'] == "lavgSTMLP":
            train_scat_coeffs = torch.stack([lavg(coeff, J=params['J'], L=params['L'], m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([lavg(coeff, J=params['J'], L=params['L'], m=order)[0] for coeff in test_scat_coeffs])
        else:
            train_scat_coeffs = torch.stack([ldiff(coeff, J=params['J'], L=params['L'], m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([ldiff(coeff, J=params['J'], L=params['L'], m=order)[0] for coeff in test_scat_coeffs])
            
    if params['encoder_choice'] in ["STCNN", "STMLP", "ST", "lavgSTMLP", "ldiffSTMLP", "primSTMLP"]:
        train_dataset = TensorDataset(train_scat_coeffs, train_images)
        test_dataset = TensorDataset(test_scat_coeffs, test_images)
    else:
        train_dataset = TensorDataset(train_images, train_images)
        test_dataset = TensorDataset(test_images, test_images)
    
    print("Test dataset:", test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    print(f"Length of test dataset: {len(test_loader.dataset)}")
    print(f"Length of test loader: {len(test_loader)}")

    model = get_model(params['encoder_choice'], scatshape=scat_shape, hidden_dim1=256, hidden_dim2=128, latent_dim=int(params['latent_dim']), J=params['J']).to(DEVICE)    
    model.apply(lambda m: nn.init.xavier_uniform_(m.weight.data) if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) else None)

    optimizer = AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['reg_param'])  # AdamW with regularization
    beta_increment = (params['final_beta'] - params['initial_beta']) / num_epochs
    
    """mse_loss = ExperimentalMSELoss(
        intensity_weight=params['intensity_weight'],
        sum_weight=params['sum_weight'],
        histogram_weight=params['histogram_weight'],
        threshold=params['th'],
        high_intensity_weight=params['high_intensity_weight']
    )"""
    
    def vae_loss_function(x, x_hat, mean, log_var, beta=1.0):
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        if x_hat.shape != x.shape:
            x_hat = x_hat.view(x.shape)
        RecLoss = nn.functional.mse_loss(x_hat, x)
        return RecLoss + beta * KLD

    model.train()    
    train_losses, val_losses = [], []
    best_loss, best_model_state = float('inf'), None
    early_stopping = EarlyStopping(patience=20)

    with tqdm(total=num_epochs, desc=f"Training {runname}", position=1) as pbar:
        for epoch in range(num_epochs):
            beta = params['initial_beta'] + epoch * beta_increment
            overall_loss = 0
            epoch_mse, epoch_kl = 0, 0

            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", leave=False, position=1) as epoch_bar:
                for batch_idx, batch in epoch_bar:
                    if batch is None:
                        continue  
                    scat_coeffs, x = batch  
                    x = x.to(DEVICE)
                    optimizer.zero_grad()
                    if 'ST' in params['encoder_choice'] or 'Dual' in params['encoder_choice']:
                        scat_coeffs = scat_coeffs.to(DEVICE)
                        if 'C' in params['encoder_choice'] and 'MLP' in params['encoder_choice']:
                            labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                            x_hat, mean, log_var = model(scat_coeffs, labels)
                        elif 'Dual' in params['encoder_choice']:
                            x_hat, mean, log_var = model(x, scat_coeffs)
                        else:
                            x_hat, mean, log_var = model(scat_coeffs)
                    elif 'C' in params['encoder_choice'] and params['encoder_choice'] != 'CNN':
                        labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                        x_hat, mean, log_var = model(x, labels)
                    elif params['encoder_choice'] == "CNN":
                        x_hat, mean, log_var = model(x)
                    else:
                        print("Encoder choice not recognized.")
                        
                    if x_hat.shape != x.shape:
                        x_hat = x_hat.view(x.shape)         

                    loss = vae_loss_function(x, x_hat, mean, log_var, beta=beta)
                    mse = nn.functional.mse_loss(x_hat, x).item()
                    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()).item()
                    overall_loss += float(loss.item())
                    epoch_mse += float(mse)
                    epoch_kl += float(kl)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    epoch_bar.set_description(f"Epoch {epoch + 1} Batch {batch_idx + 1} Loss: {loss.item():.4f}")

            average_loss = overall_loss / len(train_loader.dataset)
            epoch_mse /= len(train_loader.dataset)
            epoch_kl /= len(train_loader.dataset)
            train_losses.append(average_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if batch is None:
                        continue  
                    scat_coeffs, x = batch  
                    x = x.to(DEVICE)
                    if 'ST' in params['encoder_choice'] or 'Dual' in params['encoder_choice']:
                        scat_coeffs = scat_coeffs.to(DEVICE)
                        if 'C' in params['encoder_choice'] and 'MLP' in params['encoder_choice']:
                            labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                            x_hat, mean, log_var = model(scat_coeffs, labels)
                        elif 'Dual' in params['encoder_choice']:
                            x_hat, mean, log_var = model(x, scat_coeffs)
                        else:
                            x_hat, mean, log_var = model(scat_coeffs)
                    elif 'C' in params['encoder_choice'] and params['encoder_choice'] != 'CNN':
                        labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                        x_hat, mean, log_var = model(x, labels)
                    elif params['encoder_choice'] == "CNN":
                        x_hat, mean, log_var = model(x)
                    else:
                        print("Encoder choice not recognized.")
                    
                    if x_hat.shape != x.shape:
                        x_hat = x_hat.view(x.shape)
                    
                    loss = vae_loss_function(x, x_hat, mean, log_var, beta=beta)
                    val_loss += float(loss.item())
            val_loss /= len(test_loader)  
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()

            model.train()
            pbar.update(1)
            pbar.set_postfix_str(f"Average Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if early_stopping.early_stop:
                print("Early stopping")
                break

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict()

    return best_model_state, best_loss

num_iterations = 300
best_params = initial_params
best_model_state, best_loss = train_vae(best_params)

with tqdm(total=num_iterations, desc="Finding optimal hyperparameters", position=0) as pbar1:
    for _ in tqdm(range(num_iterations), desc='Optimizing', leave=True):
        new_params = random_walk(best_params, param_bounds, param_step_sizes, encoder_choices)
        new_model_state, new_loss = train_vae(new_params)
        formatted_params = {k: f"{v:.2e}" if isinstance(v, (float, int)) and abs(v) < 1 else v for k, v in new_params.items()}
        
        if new_loss < best_loss:
            best_loss = new_loss
            best_params = new_params
            best_model_state = new_model_state
            formatted_best_params = {k: f"{v:.2e}" if isinstance(v, (float, int)) and abs(v) < 1 else v for k, v in best_params.items()}
            print(f"New best params found: {formatted_best_params} with loss: {best_loss:.2e}")
        
        pbar1.update(1)
        formatted_best_params = {k: f"{v:.2e}" if isinstance(v, (float, int)) and abs(v) < 1 else v for k, v in best_params.items()}
        pbar1.set_postfix_str(f"Current best parameters: {formatted_best_params}, Loss: {best_loss:.2e}")

# Format the final print statement for best parameters and loss
formatted_best_params = {k: f"{v:.2e}" if isinstance(v, (float, int)) and abs(v) < 1 else v for k, v in best_params.items()}
print(f"Best parameters: {formatted_best_params} with loss: {best_loss:.2e}")


