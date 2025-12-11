import torch
import numpy as np
from kymatio.torch import Scattering2D
from utils.data_loader import load_galaxies, get_classes
from utils.scatter_reduction import lavg, ldiff
from utils.models import get_model
from utils.calc_tools import normalize_to_minus1_1, normalize_to_0_1, generate_from_noise, get_model_directory
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
import os
#from cmmd.main import compute_cmmd_from_tensors


print("Running script 3.1.scattervae_eval.py")
#The test data must be reloaded for each trained model to match the folds of their tranining data


########################################################################################################
################################### CONFIGURATION ######################################################
########################################################################################################

encoder_list = ['Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP']

# Configuration for auto-loading models
galaxy_classes = 11
hidden_dim1, hidden_dim2, latent_dim = 256, 128, 64
J, L, order = 2, 12, 2
num_display, num_generate = 5, 1000 # Plotted images per model, generated images for statistics
img_shape = (1, 128, 128)


FFCV = True              # Use five-fold cross-validation
NORMALISEIMGS = True     # Normalise imgs with themselves to [0, 1]
NORMALISEIMGSTOPM = False # Normalise imgs with themselves to [-1, 1]
NORAMLISESCS = False      # Normalise scattering coefficients with themselves to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients with themselves to [-1, 1]
REMOVEOUTLIERS = True     # If the training was done on filtered data

include_two_point_correlation = False
FFCV = True # Use five-fold cross-validation
NORMALISETOPM = False # Normalise to [-1, 1]

num_gen = num_generate * 5 if FFCV else num_generate

models = None # if autoloading
# Paths to the specific models instead of auto-loading
# models = [{"name": "ldiffSTMLP", "path": "./generator/model_15_2000_ldiffSTMLP_320_50_0.001_0.01_0.5.pth", "hidden_dim": 320, "latent_dim": 50},
#    {"name": "ldiffSTMLP", "path": "./generator/model_15_300_ldiffSTMLP_320_64_0.003_0.01_0.5.pth", "hidden_dim": 320, "latent_dim": 64}]



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
F = 8 # Loss function
    
num_galaxies_list = [int(1000000+A*1e5+B*1e4+C*1e3+D*1e2+E*10+F)]
num_galaxies = num_galaxies_list[0]
print("Num_galaxies: ", num_galaxies)

#########################################################################################################################
############################################# LOAD DATA ##################################################################
#########################################################################################################################


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = get_classes()
num_classes = len(galaxy_classes) if isinstance(galaxy_classes, list) else 1


# Create save directory for the combined classes
save_dir = f'./generator/eval/{galaxy_classes}_evaluations'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if galaxy_classes == 100:
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
    # Load the data
    data = load_galaxies(galaxy_class=galaxy_classes, fold=0, img_shape=img_shape, sample_size=num_galaxies, REMOVEOUTLIERS=True, train=True)
    train_images, train_labels, test_images, test_labels = data
    print("Shape of test images: ", test_images.shape)

    #Shuffle data
    perm = torch.randperm(test_images.size(0))
    test_images = test_images[perm]
    test_labels = test_labels[perm]

# Calculate scattering coefficients
print("Calculating scattering coefficients")
scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order).to(DEVICE)
def compute_scattering_coeffs(images, scattering, batch_size=32):
    scattering.eval()
    coeffs = []
    with torch.no_grad():
        num_batches = (len(images) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch = images[i * batch_size:(i + 1) * batch_size].to(DEVICE)
            coeffs.append(scattering(batch).cpu())
    return torch.cat(coeffs, dim=0)
scat_coeffs = compute_scattering_coeffs(test_images, scattering).squeeze()
print("Shape of scat_coeffs: ", np.shape(scat_coeffs))
lavg_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in scat_coeffs])
ldiff_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in scat_coeffs])



# Normalize train and test images to [0, 1]
if NORMALISEIMGS:
    train_images = normalize_to_0_1(train_images)
    test_images = normalize_to_0_1(test_images)

if NORMALISEIMGSTOPM: # normalize to [-1, 1]
    train_images = normalize_to_minus1_1(train_images)
    test_images = normalize_to_minus1_1(test_images)

# Handle scattering coefficients normalization in a similar way
if 'ST' in encoder_list or 'Dual' in encoder_list:
    if NORAMLISESCS:
        scat_coeffs = normalize_to_0_1(scat_coeffs)
        lavg_scat_coeffs = normalize_to_0_1(lavg_scat_coeffs)
        ldiff_scat_coeffs = normalize_to_0_1(ldiff_scat_coeffs)

        if NORMALISESCSTOPM:
            scat_coeffs = normalize_to_minus1_1(scat_coeffs)
            lavg_scat_coeffs = normalize_to_minus1_1(lavg_scat_coeffs)
            ldiff_scat_coeffs = normalize_to_minus1_1(ldiff_scat_coeffs)

    
# Calculate the input dimensions
scatshape = np.shape(scat_coeffs)
lavg_scatshape = np.shape(lavg_scat_coeffs)
ldiff_scatshape = np.shape(ldiff_scat_coeffs)


###########################################################
################# EVALUATION FUNCTIONS #####################
###########################################################


"""try:
    # For torchvision version 0.10.0 and later
    inception_weights = torchvision.models.Inception_V3_Weights.DEFAULT
    inception_model = inception_v3(weights=inception_weights)
except AttributeError:
    # For older versions of torchvision
    inception_model = inception_v3(pretrained=True)

inception_model.fc = torch.nn.Identity()  # Use the model up to the last pooling layer
inception_model.eval()
inception_model.to(DEVICE)

def extract_features(images, batch_size=32):
    features = []
    inception_model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        num_batches = (len(images) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch = images[i * batch_size:(i + 1) * batch_size].cpu()
            if batch.shape[1] == 1:  # Single-channel images
                batch = batch.repeat(1, 3, 1, 1)  # Repeat the channel to get 3 channels
            batch = batch.to(DEVICE)  # Move to GPU
            features.append(inception_model(batch).cpu().numpy())
    return np.concatenate(features, axis=0)


def calculate_fid(real_features, generated_features):
    # Ensure there are no NaNs or Infs in the features
    if np.any(np.isnan(real_features)) or np.any(np.isnan(generated_features)) or np.any(np.isinf(real_features)) or np.any(np.isinf(generated_features)):
        raise ValueError("Features contain NaN or Inf values.")
    
    mu1, sigma1 = real_features.mean(axis=0), cov(real_features, rowvar=False)f
    mu2, sigma2 = generated_features.mean(axis=0), cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid"""





def load_model(path, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes):
    print(f"Loading model from {path}")
    name = path.split('_')[-2].split('.')[0]
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model = get_model(name=name, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes, J=J)


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


def compute_silhouette_score(latent_space, n_clusters=2):
    # Replace NaN values with zeros
    latent_space = np.nan_to_num(latent_space)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(latent_space)
    
    # Check if the number of unique labels is greater than 1 and if the number of samples is sufficient
    if len(np.unique(labels)) > 1 and latent_space.shape[0] > 2:
        silhouette = silhouette_score(latent_space, labels)
    else:
        silhouette = float('nan')  # Assign NaN if the silhouette score cannot be computed

    return silhouette



def reconstruct_and_evaluate(model, images, scat_coeffs, lavg_scat_coeffs, ldiff_scat_coeffs, labels, path, batch_size=32):
    print(f"Reconstructing images using {path}")
    if 'lavg' in path:
        scat_coeffs = lavg_scat_coeffs
    elif 'ldiff' in path:
        scat_coeffs = ldiff_scat_coeffs
    
    model.eval()
    all_reconstructed_images = []
    all_latent_representations = []
    mse_list = []
    kl_divergence_list = []
    silhouette_scores = []

    with torch.no_grad():
        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(num_batches):
            x_batch = images[i*batch_size:(i+1)*batch_size].to(DEVICE)
            scat_batch = scat_coeffs[i*batch_size:(i+1)*batch_size].to(DEVICE) if scat_coeffs is not None else None
            if scat_batch.dim() == 5:
                scat_batch = scat_batch.squeeze(1)  # Remove the second dimension (channels dimension being 1)
            labels_batch = labels[i*batch_size:(i+1)*batch_size].to(DEVICE) if labels is not None else None

            if 'ST' in path or 'Dual' in path:
                if 'C' in path and 'MLP' in path:
                    x_hat, mean, log_var = model(scat_batch, labels_batch)
                elif 'Dual' in path:
                    x_hat, mean, log_var = model(x_batch, scat_batch)
                else:
                    x_hat, mean, log_var = model(scat_batch)
            elif 'C' in path and 'CNN' not in path:
                x_hat, mean, log_var = model(x_batch, labels_batch)
            else:
                x_hat, mean, log_var = model(x_batch)
            
            reconstructed_images = x_hat.view(-1, *img_shape)
            all_reconstructed_images.append(reconstructed_images.cpu())
            
            z = model.reparameterize(mean, log_var)
            all_latent_representations.append(z.cpu())

            images_flat = x_batch.view(x_batch.size(0), -1).cpu().numpy()
            reconstructed_images_flat = reconstructed_images.view(reconstructed_images.size(0), -1).cpu().numpy()
            mse = np.mean((images_flat - reconstructed_images_flat) ** 2, axis=1)
            mse_list.append(mse)

            if mean is not None and log_var is not None:
                kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).cpu().numpy()
                kl_divergence_list.append(kl_divergence)
            
            z_np = z.cpu().numpy()
            if z_np.shape[0] > 2:  # Only calculate silhouette score if there are more than 2 samples
                silhouette = compute_silhouette_score(z_np, n_clusters=2)
            else:
                silhouette = float('nan')  # Skip silhouette score calculation if not enough samples
            silhouette_scores.extend([silhouette] * x_batch.size(0))

    all_reconstructed_images = torch.cat(all_reconstructed_images, dim=0)
    all_latent_representations = torch.cat(all_latent_representations, dim=0)
    mse = np.concatenate(mse_list, axis=0)
    kl_divergence = np.concatenate(kl_divergence_list, axis=0) if kl_divergence_list else None
    silhouette_scores = np.array(silhouette_scores)
    
    mean_mse = np.mean(mse)
    mean_kl_divergence = np.mean(kl_divergence) if kl_divergence is not None else None
    mean_silhouette = np.nanmean(silhouette_scores)  # Handle NaN values appropriately

    return all_reconstructed_images, all_latent_representations, mean_mse, mean_kl_divergence, mean_silhouette


###################################################################################
########################### MAIN SCRIPT ###########################################
###################################################################################

#real_features = extract_features(test_images)

# Initialize empty lists to store metrics for each fold
summary_table, reconstructed_images_list, generated_images_list, latent_representations, model_names, log_files, models = [], [], [], [], [], [], []

for encoder in encoder_list:
    # Initialize lists to store metrics for all folds for the current model
    all_mse, all_kl_divergence, all_silhouette, all_fid, all_cmmd = [], [], [], [], []

    for fold in range(5) if FFCV else [6]:
        path = get_model_directory(galaxy_classes, num_galaxies, encoder, fold)
        
        if os.path.exists(path):
            print(f"Model {encoder} found at {path}")
            if "lavgSTMLP" in encoder:
                scatshape = lavg_scatshape
            elif "ldiffSTMLP" in encoder:
                scatshape = ldiff_scatshape
            else:
                scatshape = np.shape(scat_coeffs)

            model = load_model(path, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes)
            models.append({"name": encoder, "path": path})
            reconstructed_images, latent_rep, mse, kl_divergence, silhouette = reconstruct_and_evaluate(model, test_images, scat_coeffs, lavg_scat_coeffs, ldiff_scat_coeffs, test_labels, path)
            generated_images = generate_from_noise(model, train_labels, img_shape, latent_dim, num_generate) 
            #gen_features = extract_features(generated_images)
            #fid = calculate_fid(real_features, gen_features)
            #cmmd_value = compute_cmmd_from_tensors(test_images, generated_images, batch_size=32)
            
            # Append the metrics for this fold to the lists
            #all_mse.append(mse)
            #all_kl_divergence.append(kl_divergence)
            #all_silhouette.append(silhouette)
            #all_fid.append(fid)
            #all_cmmd.append(cmmd_value)
            
            # Append the reconstructed and generated images, as well as latent representations
            reconstructed_images_list.append(reconstructed_images)
            generated_images_list.append(generated_images)
            latent_representations.append(latent_rep)
            model_names.append(encoder)
        else: 
            print(f"Model {encoder} not found at {path}")

    # Only calculate and store the summary if there are valid results for the model
    """if all_mse:  # This checks if any metrics were appended, meaning a valid path existed
        #mse_mean = np.mean(all_mse)
        #mse_std = np.std(all_mse)
        kl_divergence_mean = np.mean(all_kl_divergence)
        kl_divergence_std = np.std(all_kl_divergence)
        silhouette_mean = np.mean(all_silhouette)
        silhouette_std = np.std(all_silhouette)
        fid_mean = np.mean(all_fid)
        fid_std = np.std(all_fid)
        #cmmd_mean = np.mean(all_cmmd)
        #cmmd_std = np.std(all_cmmd)

        # Store the summary for the current model
        summary_table.append({
            "Model": encoder,
            "MSE Mean": mse_mean,
            "MSE Std": mse_std,
            "KL Divergence Mean": kl_divergence_mean,
            "KL Divergence Std": kl_divergence_std,
            "Silhouette Score Mean": silhouette_mean,
            "Silhouette Score Std": silhouette_std,
            "FID Mean": fid_mean,
            "FID Std": fid_std
            #"CMMD Mean": cmmd_mean, 
            #"CMMD Std": cmmd_std  
        })

# After processing all encoders, you can convert the summary table to a DataFrame, etc.

# Convert to DataFrame
summary_df = pd.DataFrame(summary_table)
summary_df = summary_df.fillna(0)  # Fill NaN values with 0

# Save or print the summary
print("Summary DataFrame:\n", summary_df)
summary_df.to_csv(f"{save_dir}/summary_df.csv", index=False)

# If you want to identify the best model based on a composite score:
summary_df["Overall Score"] = (
    0.25 * (1 - summary_df["MSE Mean"] / summary_df["MSE Mean"].max()) +
    0.25 * (1 - summary_df["KL Divergence Mean"] / summary_df["KL Divergence Mean"].max()) +
    0.5 * (summary_df["Silhouette Score Mean"] / summary_df["Silhouette Score Mean"].max())
)

# Find and print the best model
best_model = summary_df.loc[summary_df["Overall Score"].idxmax()]
print("Best Model:", best_model["Model"])
with open(f"./generator/eval/{galaxy_classes}_log.txt", "w") as file:
    file.write(summary_df.to_string())
    file.write("\nBest Model: " + best_model["Model"])

#Create latex table
summary_df.to_latex(f"{save_dir}/summary_df.tex", index=False)"""

# Print the shapes of all parameters before saving
print("Shape of scat_coeffs:", scat_coeffs.shape)
print("Shape of lavg_scat_coeffs:", lavg_scat_coeffs.shape)
print("Shape of ldiff_scat_coeffs:", ldiff_scat_coeffs.shape)
print("Shape of test_images:", test_images.shape)
print("Shape of test_labels:", len(test_labels))  # Assuming this is a 1D array or list
print("Length of reconstructed_images_list:", len(reconstructed_images_list))
print("Shape of first element in reconstructed_images_list:", reconstructed_images_list[0].shape)
print("Length of generated_images_list:", len(generated_images_list))
print("Shape of first element in generated_images_list:", generated_images_list[0].shape)
print("Length of latent_representations:", len(latent_representations))
print("Shape of latent_representations:", latent_representations[0].shape)
print("Length of model_names:", len(model_names))  # Assuming model_names is a list of strings
print("Length of models:", len(models))  # Assuming models is a list of objects

# Now save the data
save_path = os.path.join(save_dir, f'compressed_{galaxy_classes}_{num_galaxies}_data.pt')
with open(save_path, 'wb') as f:
    torch.save({
        'scat_coeffs': scat_coeffs.half(),
        'lavg_scat_coeffs': lavg_scat_coeffs.half(),
        'ldiff_scat_coeffs': ldiff_scat_coeffs.half(),
        'test_images': test_images.half(),
        'test_labels': test_labels,
        'reconstructed_images_list': [img.half() for img in reconstructed_images_list],
        'generated_images_list': [img.half() for img in generated_images_list],
        'latent_representations': [rep.half() for rep in latent_representations],
        'latent_dim': latent_dim,
        'model_names': model_names,
        'models': models
    }, f, _use_new_zipfile_serialization=True)

print(f"Data saved to {save_path} using PyTorch's zipfile serialization.")
