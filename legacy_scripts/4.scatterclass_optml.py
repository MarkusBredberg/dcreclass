import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.data_loader import load_galaxies, get_classes
from utils.classifiers import Scattering_Classifier, CNN_Classifier, SimpleCNN, SimpleFCN, ProjectModel
from utils.training_tools import EarlyStopping, reset_weights
from utils.scatter_reduction import lavg, ldiff
from utils.calc_tools import normalize_to_0_1, normalize_to_minus1_1, cluster_metrics, get_model_name, generate_from_noise, load_model
from utils.plotting import plot_loss, plot_images, plot_histograms, plot_images_by_class
from kymatio.torch import Scattering2D
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.optim import AdamW
import itertools
import matplotlib
import os
os.makedirs('./classifier/trained_models', exist_ok=True)
matplotlib.use('Agg')
tqdm.pandas(disable=True)

###############################################
################ CONFIGURATION ################
###############################################

classes = get_classes()
galaxy_classes = [10, 11, 12, 13]  # Classes to classify
max_num_galaxies = 100000 # Upper limit for the training size for each class
dataset_portions = [0.01, 0.1, 1]  # Portions of complete dataset for the accuracy vs dataset size
J, L, order = 2, 12, 2  # Scatter transform parameters
classifier = ["scatterMLP", "normalCNN", "simpleCNN", "simpleFCN", "ProjectModel"][4]  # Choose one model
num_epochs_cuda = 200
num_epochs_cpu = 2
batch_size = 64
learning_rates = [1e-3, 1e-4]  # Learning rates
regularization_params = [1e-1]  # Regularisation parameters
num_experiments = 1
num_folds = 5
img_shape = (1, 128, 128)

FFCV = True # Use five-fold cross-validation
ES = True # Use early stopping
IMGCHECK = False # Check the input images (Tool for control)
SAVEIMGS = False # Save the reconstructed images in tensor format
NORMALISEIMGS = False # Normalise images to [0, 1]
NORMALISEIMGSTOPM = False # Normalise images to [-1, 1]
NORMALISESCS = False # Normalise scattering coefficients to [0, 1]
NORMALISESCSTOPM = False # Normalise scattering coefficients to [-1, 1]
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights

# Generate training data with trained VAE model
encoder =  ['Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP'][0]
scatshape = (1, 128, 128)
hidden_dim1 = 256
hidden_dim2 = 128
latent_dim = 64
lambda_generate = 2 # Ratio between generated images and original images per class
VAE_train_size = 1101028 # Code in the name of the VAE model being called for
VAE_classes = galaxy_classes

#########################################################################################################################

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")


###############################################
########### DATA STORING FUNCTIONS ###############
###############################################


# Initialize these dictionaries with empty lists for each unique combination of subset_size, galaxy_classes, and model_name
def initialize_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg):
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    
# Function to update metrics with the new values  
def update_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1):
    subset_size_str = str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(accuracy)
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(precision)
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(recall)
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(f1)
    
    
def initialize_history(history, model_name, subset_size, fold, experiment, lr, reg):
    if model_name not in history:
        history[model_name] = {}

    loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"

    if loss_key not in history[model_name]:
        history[model_name][loss_key] = []
    if val_loss_key not in history[model_name]:
        history[model_name][val_loss_key] = []


def initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment, lr, reg):
    key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    all_true_labels[key] = []
    all_pred_labels[key] = []


###############################################
########## INITIALIZE DICTIONARIES ############
###############################################

metrics = {
    "accuracy": {},
    "precision": {},
    "recall": {},
    "f1_score": {}
}

metric_colors = {
    "accuracy": 'blue',
    "precision": 'green',
    "recall": 'red',
    "f1_score": 'orange'
}

all_true_labels = {}
all_pred_labels = {}
training_times = {}
all_pred_probs = {}
history = {} 
dataset_sizes = {}

            
###############################################
########### LOOP OVER DATA FOLD ###############
###############################################            

FIRSTTIME = True # Set to True to print model summaries only once
param_combinations = list(itertools.product(range(5) if FFCV else [6], learning_rates, regularization_params))
for fold, lr, reg in param_combinations:
    torch.cuda.empty_cache()
    print(f"\n Training with fold: {fold}, learning rate: {lr}, and regularization: {reg}")
    runname = f'{galaxy_classes}_{classifier}_lr{lr}_reg{reg}'
    log_path = f"./classifier/log_{runname}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file = open(log_path, 'w')

    # Load the data
    data = load_galaxies(galaxy_class=galaxy_classes, 
                        fold=fold,
                        img_shape=img_shape, 
                        sample_size=max_num_galaxies, 
                        REMOVEOUTLIERS=False,
                        train=True)
    train_images, train_labels, test_images, test_labels = data
    
    perm = torch.randperm(train_images.size(0))
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    perm = torch.randperm(test_images.size(0))
    test_images = test_images[perm]
    test_labels = test_labels[perm]
    
    print("Train images shape: ", np.shape(train_images))
    print("Test images shape: ", np.shape(test_images))
    print("Example labels: ", train_labels[:5])
            
    num_galaxies = len(train_images)
    print("New number of galaxies is the same as the training size: ", num_galaxies)
    dataset_sizes[fold] = [int(num_galaxies * perc) for perc in dataset_portions]
    print(f"Dataset_sizes in fold f{fold}:  {dataset_sizes}")
    
    if set(galaxy_classes) & {18} & {19}:
        galaxy_classes = [20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Include all digits if 18 or 19 is in target_classes
    else:
        galaxy_classes = galaxy_classes
    num_classes = len(galaxy_classes)
    
    if USE_CLASS_WEIGHTS:
        # Compute class counts and weights
        unique, counts = np.unique(train_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        total_count = sum(counts)
        class_weights = {cls: total_count / count for cls, count in class_counts.items()}
        weights = torch.tensor([class_weights.get(cls, 1.0) for cls in galaxy_classes], dtype=torch.float).to(DEVICE)

        print("Number of images per class in the training set: ", class_counts)
        print("Class weights:", class_weights)

        # Handle missing classes in the dataset
        missing_classes = [cls for cls in galaxy_classes if cls not in class_weights]
        if missing_classes:
            print(f"Warning: Missing classes in dataset: {missing_classes}")
            class_weights.update({cls: 1.0 for cls in missing_classes})
        
        # Check test set class distribution
        unique_test, counts_test = np.unique(test_labels, return_counts=True)
        class_counts_test = dict(zip(unique_test, counts_test))
        print("Number of images per class in the test set: ", class_counts_test)
    
    else:
        weights = None  # No weights
        print("No class weighting")
        print("Classes used: ", galaxy_classes)

    if fold == 0:
        plot_images_by_class(train_images, train_labels, num_images=5, save_path=f"./classifier/{classifier}_{galaxy_classes}_{num_galaxies}_example_inputs.png")


    # Prepare input data
    if 'MLP' in classifier:
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

        train_scat_coeffs = compute_scattering_coeffs(train_images)
        test_scat_coeffs = compute_scattering_coeffs(test_images)
        
        if 'lavg' in classifier:
            train_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
        elif 'ldiff' in classifier:
            train_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
            
        scatdim = train_scat_coeffs[1:].shape
        #print("Shape of scattering coefficients:", scatdim)

                
    ##########################################################
    ############ NORMALISE AND FILTER THE INPUT ##############
    ##########################################################
                
    # Normalize train and test images to [0, 1]
    if NORMALISEIMGS:
        train_images = normalize_to_0_1(train_images)
        test_images = normalize_to_0_1(test_images)

    if NORMALISEIMGSTOPM: # normalize to [-1, 1]
        train_images = normalize_to_minus1_1(train_images)
        test_images = normalize_to_minus1_1(test_images)

    # Handle scattering coefficients normalization in a similar way
    if 'MLP' in classifier: #Double dataset for convenience for dual model in training loop
        if NORMALISESCS:
            train_scat_coeffs = normalize_to_0_1(train_scat_coeffs)
            test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)

            if NORMALISESCSTOPM:
                train_scat_coeffs = normalize_to_minus1_1(train_scat_coeffs)
                test_scat_coeffs = normalize_to_minus1_1(test_scat_coeffs)


    #Check input after renormalisation and filtering  
    if IMGCHECK: 
        plot_images(train_images, train_labels)
        plot_histograms(train_images, test_images)
        
    #Generate more trainig data
    if lambda_generate > 0:
        num_generate = int(lambda_generate / len(galaxy_classes) * len(train_images))
        print("Old training data size: ", train_images.size())
        for VAE_class in VAE_classes:
            try:
                path = get_model_name(VAE_class, VAE_train_size, encoder, fold=0) # EDIT THIS LATER
                model = load_model(path, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes)
            except:
                print(f"Model not found. Cannot Generate new data for VAE_class: {VAE_class}")
                continue
            #    name = path.split('_')[-2].split('.')[0]
            
            print(f"Before generation - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                  f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            generated_labels = torch.ones(num_generate, dtype=torch.long)*(VAE_class-min(galaxy_classes))  
            generated_images = generate_from_noise(model, train_labels=generated_labels, latent_dim=latent_dim, num_samples=num_generate, DEVICE='cpu') 
            print(f"After generation - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                  f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")                       
            print(f"Generated {num_generate} images and appended to training data.")


            
            # Ensure both tensors are on the same device
            generated_images = generated_images.to(train_images.device)
            generated_labels = generated_labels.to(train_labels.device)
                           
            # Append generated data to training data
            train_images = torch.cat([train_images, generated_images])
            train_labels = torch.cat([train_labels.clone().detach(), generated_labels])

        print("New training data size: ", generated_images.size())        
        print(f"Some generated labels: {generated_labels[:5]} for VAE_class: {VAE_class}") 
        print(f"generated_images device: {generated_images.device}")
    
    ###### RELABEL ######
    # Remap labels to start from 0
    min_label = train_labels.min()
    train_labels = train_labels - min_label
    test_labels = test_labels - min_label

    # Reshape labels to one-hot encoding
    #print(f"Max label: {train_labels.max()}, Min label: {train_labels.min()}, Num classes: {num_classes}")
    train_labels = torch.nn.functional.one_hot(train_labels, num_classes=num_classes).float()
    test_labels = torch.nn.functional.one_hot(test_labels, num_classes=num_classes).float()

    # Ensure labels are of shape [batch_size, num_classes]
    train_labels = train_labels.view(-1, num_classes)
    test_labels = test_labels.view(-1, num_classes)

    ####### Create the data loaders #########
    if 'MLP' in classifier: #Double dataset for convenience for dual model in training loop
        train_dataset = TensorDataset(train_scat_coeffs, train_labels)
        test_dataset = TensorDataset(test_scat_coeffs, test_labels)
    else: 
        train_dataset = TensorDataset(train_images, train_labels) 
        test_dataset = TensorDataset(test_images, test_labels) 
        
    def custom_collate(batch):
        if len(batch) < 3: # Drop the batch if it contains fewer than 3 images
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
    
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    #print(f"Train loader batches: {len(train_loader)}, Test loader batches: {len(test_loader)}")

            
    ###############################################
    ############# DEFINE MODEL ####################
    ###############################################

    # Selection of model
    if classifier == "scatterMLP":
        #models = {"ProjectModel": {"model": ProjectModel(input_shape=tuple(test_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
        models = {"scatterMLP": {"model": Scattering_Classifier(input_channels=scatdim[-3], num_classes=num_classes, J=J).to(DEVICE)}}
    elif classifier == "normalCNN":
        models = {"normalCNN": {"model": CNN_Classifier(input_shape=tuple(test_images.shape[1:]) , num_classes=num_classes).to(DEVICE)}}
    elif classifier == "simpleCNN":
        models = {"simpleCNN": {"model": SimpleCNN(input_shape=tuple(test_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
    elif classifier == "simpleFCN":
        models = {"simpleFCN": {"model": SimpleFCN(input_shape=tuple(test_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
    elif classifier == "ProjectModel" :
        models = {"ProjectModel": {"model": ProjectModel(input_shape=tuple(test_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
   
    else:
        raise ValueError("Model not found. Please select one of 'scatterMLP', 'smallSTMLP', or 'normalCNN'.")

    # Apply summary to each model individually
    for model_name, model_details in models.items():
        if FIRSTTIME:
            print(f"Summary for {model_name}:")
            if model_name in ["scatterMLP", "smallSTMLP"]:
                summary(model_details["model"], input_size=train_scat_coeffs[0].shape, device=DEVICE)
            else:
                summary(model_details["model"], input_size=img_shape, device=DEVICE)
        FIRSTTIME = False

    ###############################################
    ############### TRAINING LOOP #################
    ###############################################
    
    if weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(models[model_name]["model"].parameters(), lr=lr, weight_decay=reg)
    # Initialize an empty dictionary for training times


    # Loop over models and dataset sizes
    for model_name, model_details in models.items():
        print(f"Training {model_name} model...")
        model = model_details["model"].to(DEVICE)

        for subset_size in dataset_sizes[fold]:
            if subset_size <= 0:
                print(f"Skipping invalid subset size: {subset_size}")
                continue
            if subset_size not in training_times:
                training_times[subset_size] = {}  # Initialize the times for this subset size
            if fold not in training_times[subset_size]:
                training_times[subset_size][fold] = []

            for experiment in range(num_experiments):
                initialize_history(history, model_name, subset_size, fold, experiment, lr, reg)
                initialize_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg)
                initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment, lr, reg)


                start_time = time.time()
                model.apply(reset_weights)

                # Create subset data loader
                subset_indices = list(range(subset_size))
                subset_train_dataset = Subset(train_dataset, subset_indices)
                subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)
                
                print(f"Total dataset size: {len(train_dataset)}")
                print(f"Subset size requested: {subset_size}")

                early_stopping = EarlyStopping(patience=10, verbose=False) if ES else None

                for epoch in tqdm(range(num_epochs), desc=f'Training with dataset size {subset_size}'):
                    model.train()
                    total_loss = 0
                    total_images = 0

                    for images, labels in subset_train_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += float(loss.item() * images.size(0))
                        total_images += float(images.size(0))

                    average_loss = total_loss / total_images
                    loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                    history[model_name][loss_key].append(average_loss)

                    # Validation loop
                    model.eval()
                    val_total_loss = 0
                    val_total_images = 0
                    #correct = 0

                    with torch.no_grad():
                        for i, (images, labels) in enumerate(test_loader):
                            if images is None or len(images) == 0:  # Handle empty batches
                                print(f"Empty batch at index {i}. Skipping...")
                                continue
                            images, labels = images.to(DEVICE), labels.to(DEVICE)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_total_loss += float(loss.item() * images.size(0))
                            val_total_images += float(images.size(0))
                            #_, predicted = torch.max(outputs, 1)
                            #true_labels = torch.argmax(labels, dim=1)
                            #correct += float((predicted == true_labels).sum().item())

                    val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                    val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                    history[model_name][val_loss_key].append(val_average_loss)
                    
                    if ES: # Early stopping
                        early_stopping(val_average_loss, model, f'./classifier/trained_models/{model_name}_best_model.pth')
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break

                # Final evaluation after training
                with torch.no_grad():
                    key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                    all_pred_probs[key] = []  # Initialize list under the specific key
                    all_pred_labels[key] = []
                    all_true_labels[key] = []

                    for images, labels in test_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = model(images)
                        
                        # Apply sigmoid or softmax if needed
                        pred_probs = torch.sigmoid(outputs).cpu().numpy()  # Adjust for multi-class with softmax if necessary
                        true_labels = torch.argmax(labels, dim=1).cpu().numpy()
                        pred_labels = np.argmax(pred_probs, axis=1)

                        # Collect predictions and true labels
                        all_pred_probs[key].extend(pred_probs)  # Use .extend() on the list for each key
                        all_pred_labels[key].extend(pred_labels)
                        all_true_labels[key].extend(true_labels)

                    # Calculate and store metrics
                    accuracy = accuracy_score(all_true_labels[key], all_pred_labels[key])
                    precision = precision_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                    recall = recall_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                    f1 = f1_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)

                    update_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1)

            # Track training time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken to train the model for fold {fold}: {elapsed_time:.2f} seconds")
            training_times[subset_size][fold].append(elapsed_time)
            file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

        # Calculate cluster metrics in batches to avoid memory issues
        generated_features = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(DEVICE)
                features = model(images).cpu().detach().numpy()
                generated_features.append(features)

        # Concatenate all generated features
        generated_features = np.concatenate(generated_features, axis=0)

        # Calculate cluster metrics
        cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(generated_features, n_clusters=13)

        print(f"\n Cluster Error: {cluster_error}")
        print(f"Cluster Distance: {cluster_distance}")
        print(f"Cluster Standard Deviation: {cluster_std_dev}")

        # Save the trained model
        model_save_path = f'./classifier/trained_models/{model_name}_model.pth'
        torch.save(model.state_dict(), model_save_path)
        

# Save metrics to a pickle file
metrics_save_path = f'./classifier/trained_models/{galaxy_classes}_{learning_rates}_{regularization_params}_metrics_data.pkl'
with open(metrics_save_path, 'wb') as f:
    pickle.dump({
        "Dataset_sizes": dataset_sizes,
        "num_folds": num_folds,
        "num_experiments": num_experiments,
        "num_galaxies": num_galaxies,
        "learning_rates": learning_rates,
        "regularization_params": regularization_params,
        "classifier": classifier,
        "classes": galaxy_classes,
        "models": models,
        "model_name": model_name,
        "history": history,
        "metrics": metrics,
        "metric_colors": metric_colors,
        "all_true_labels": all_true_labels,
        "all_pred_labels": all_pred_labels,
        "training_times": training_times,
        "all_pred_probs": all_pred_probs
    }, f)
print(f"Metrics and related data saved to {metrics_save_path}")



###############################################
########### PLOTTING FUNCTIONS ################
###############################################

def plot_metrics(metrics, model_name, dataset_sizes=dataset_sizes, num_folds=num_folds, num_experiments=num_experiments, learning_rates=learning_rates, regularization_params=regularization_params, save_dir='./classifier'):
    metric_colors = {
        "accuracy": "blue",
        "precision": "green",
        "recall": "red",
        "f1_score": "orange"
    }
    for fold in range(num_folds):
        for lr in learning_rates:
            for reg in regularization_params:
                fig, ax = plt.subplots(figsize=(10, 6))

                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    # Collect metric values for each dataset size
                    metric_values = {size: [] for size in dataset_sizes[fold]}
                    for subset_size in dataset_sizes[fold]:
                        for experiment in range(num_experiments):
                            key = f"{model_name}_{metric}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                            if key in metrics and metrics[key]:  # Ensure the key exists and is not empty
                                metric_values[subset_size].extend(metrics[key])

                # Calculate means and standard deviations
                mean_values = [
                    np.mean(metric_values[size]) if metric_values[size] else np.nan
                    for size in dataset_sizes[fold]
                ]
                std_values = [
                    np.std(metric_values[size]) if metric_values[size] else 0
                    for size in dataset_sizes[fold]
                ]

                # Only plot if data exists for at least one dataset size
                if any(np.isfinite(mean_values)):
                    ax.errorbar(
                        dataset_sizes[fold],
                        mean_values,
                        yerr=std_values,
                        marker='o',
                        linestyle='-',
                        label=f"{metric.capitalize()}",
                        color=metric_colors[metric],
                    )

            ax.set_title(f"Performance for {model_name} (LR={lr}, Reg={reg})", fontsize=16)
            ax.set_xlabel('Dataset Size', fontsize=14)
            ax.set_ylabel('Metric Value', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True)
            ax.set_xscale('log')
            plt.tight_layout()

            # Save the figure with appropriate naming
            plt.savefig(f'{save_dir}/{model_name}_{galaxy_classes}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_all_metrics.png')
            plt.close(fig)

    

def plot_all_metrics_vs_dataset_size(metrics, model_name, dataset_sizes=dataset_sizes, num_folds=num_folds, num_experiments=num_experiments, learning_rates=learning_rates, regularization_params=regularization_params, save_dir='./classifier'):
    """
    Plots accuracy, precision, recall, and F1 score as a function of dataset size.
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots

    # Metric configurations
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ['blue', 'green', 'red', 'orange']

    # Loop over each metric for the subplot
    for i, metric in enumerate(metric_names):
        row, col = i // 2, i % 2  # Determine subplot position

        # Loop through dataset sizes
        for reg in regularization_params:
            for lr in learning_rates:
                for experiment in range(num_experiments):                   
                    for fold in range(num_folds):
                        for subset_size in dataset_sizes[fold]:
                            for keyword in ["accuracy", "precision", "recall", "f1_score"]:
                                key = f"{model_name}_{keyword}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                                if key in metrics and len(metrics[key]) > 0:
                                    ax[row, col].plot(metrics[key], label=f"{keyword.capitalize()}_{subset_size}_{fold}", color=colors[i])
                                    ax[row, col].set_title(f"{metric_titles[i]} vs Dataset Size", fontsize=14)
                                    ax[row, col].set_xlabel('Dataset Size', fontsize=12)
                                    ax[row, col].set_ylabel(metric_titles[i], fontsize=12)
                                    ax[row, col].grid(True)
                                    ax[row, col].set_xscale('log')  # Log scale for dataset sizes
                                    ax[row, col].legend(fontsize=10)

                        # Adjust layout and save
                        plt.tight_layout()
                        plt.savefig(f"{save_dir}/{model_name}_{fold}_metrics_vs_dataset_size.png")
                        plt.close()


def plot_roc_curves(all_true_labels, all_pred_probs, model_name, dataset_sizes=dataset_sizes, num_folds=num_folds, num_experiments=num_experiments, learning_rates=learning_rates, regularization_params=regularization_params, 
                    galaxy_classes=galaxy_classes, save_dir='./classifier'):
    """
    Plots the ROC curve for each class in a multiclass setting and saves it.
    """
    # Convert min_label to an integer
    min_label = min(galaxy_classes)
    min_label = int(min_label) if isinstance(min_label, torch.Tensor) else min_label
    adjusted_classes = [cls - min_label for cls in galaxy_classes]  # Subtract min_label from each class

    for fold in range(num_folds):
        for subset_size in dataset_sizes[fold]:
            if subset_size <= 0:
                continue
            for experiment in range(num_experiments):
                for lr in learning_rates:
                    for reg in regularization_params:
                        key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        true_labels = np.array(all_true_labels[key])
                        pred_probs = np.array(all_pred_probs[key])

                        # Binarize the labels for one-vs-rest ROC computation
                        true_labels_bin = label_binarize(true_labels, classes=np.arange(len(adjusted_classes)))

                        # Plot ROC curve for each class
                        fig, ax = plt.subplots(figsize=(6, 5))
                        for i, class_label in enumerate(adjusted_classes):
                            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, lw=2, label=f'Class {class_label} ROC (area = {roc_auc:.2f})')

                        # Plot chance line
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate', fontsize=16)
                        ax.set_ylabel('True Positive Rate', fontsize=16)
                        ax.set_title(f'ROC Curve - {model_name} \n {subset_size}, Fold {fold}, Experiment {experiment}', fontsize=14)
                        ax.legend(loc="lower right")

                        # Save the ROC plot
                        plt.savefig(f'{save_dir}/{model_name}_{galaxy_classes}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_roc_curve.png')
                        plt.close(fig)



def plot_confusion_matrix(all_true_labels=all_true_labels, all_pred_labels=all_pred_labels, model_name=model_name, 
                          dataset_sizes=dataset_sizes, num_folds=num_folds, num_experiments=num_experiments, 
                          learning_rates=learning_rates, regularization_params=regularization_params, 
                          galaxy_classes=galaxy_classes, 
                          class_descriptions=[cls['description'] for cls in classes if cls['tag'] in galaxy_classes], 
                          save_dir='./classifier'):
    """
    Plots the confusion matrix and saves it for the combined results across folds and experiments,
    while skipping invalid keys.
    """
    print("Class_descriptions for CM:", class_descriptions)
    print("Dataset sizes in CM plotting: ", dataset_sizes)

    for fold in range(num_folds):
        for subset_size in dataset_sizes[fold]:
            if subset_size <= 0:
                print(f"Skipping invalid subset size: {subset_size}")
                continue
            for experiment in range(num_experiments):
                for lr in learning_rates:
                    for reg in regularization_params:
                        key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"

                        # Check if the key exists in both true_labels and pred_labels
                        if key not in all_true_labels or key not in all_pred_labels:
                            print(f"Skipping missing key: {key}")
                            continue

                        true_labels = all_true_labels[key]
                        pred_labels = all_pred_labels[key]

                        # Validate that the label arrays are not empty
                        if not true_labels or not pred_labels:
                            print(f"Skipping key with empty labels: {key}")
                            continue

                        # Calculate confusion matrix
                        cm = confusion_matrix(true_labels, pred_labels, normalize='true')

                        # Plot the confusion matrix
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt=".1%", linewidths=.5, square=True, cmap='Blues', ax=ax,
                                    xticklabels=class_descriptions, yticklabels=class_descriptions, annot_kws={"size": 16})
                        colorbar = ax.collections[0].colorbar
                        colorbar.ax.tick_params(labelsize=16)

                        # Set labels and title
                        accuracy = accuracy_score(true_labels, pred_labels)
                        ax.set_title(f'Model: {model_name} \n Total accuracy: {accuracy*100:.2f}%', fontsize=14)
                        ax.set_ylabel('True label', fontsize=16)
                        ax.set_xlabel('Predicted label', fontsize=16)

                        # Save Confusion Matrix
                        save_path = f'{save_dir}/{model_name}_{galaxy_classes}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_confusion_matrix.png'
                        plt.savefig(save_path)
                        plt.close()
    

def plot_loss(models=models, history=history, dataset_sizes=dataset_sizes, num_folds=num_folds, num_experiments=num_experiments, galaxy_classes=galaxy_classes, 
              learning_rates=learning_rates, regularization_params=regularization_params, num_galaxies=num_galaxies, classifier=classifier):

    for fold in range(num_folds):
        for subset_size in dataset_sizes[fold]:            
            fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure for clarity
            colors = itertools.cycle(['blue', 'orange', 'green', 'red', 'magenta'])  # Specifies fold

            for model_name in models.keys():
                color = next(colors)  # Each model gets its own color
                for experiment in range(num_experiments):
                    if subset_size <= 0:
                        print(f"Skipping invalid subset size: {subset_size}")
                        continue
                    for lr in learning_rates:
                        for reg in regularization_params:
                            # Generate keys for training and validation loss
                            loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                            val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                            
                            if loss_key not in history.get(model_name, {}):
                                print(f"Skipping missing loss_key: {loss_key}")
                                continue
                            if val_loss_key not in history.get(model_name, {}):
                                print(f"Skipping missing val_loss_key: {val_loss_key}")
                                continue
                            
                            marker_size = np.log(subset_size / max(dataset_sizes[fold]) * 1000)  # Scale marker size by dataset size

                            # Plot training loss
                            ax.plot(history[model_name][loss_key], color=color, linestyle='-', 
                                    marker='o', markersize=marker_size, label=f"{model_name} train (fold {fold})")

                            # Plot validation loss
                            ax.plot(history[model_name][val_loss_key], color=color, linestyle='--', 
                                    marker='x', markersize=marker_size, label=f"{model_name} val (fold {fold})")

        # Add title and labels
        ax.set_title(f'Training and Validation Loss for Subset Size {subset_size}', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)

        # Show legend and grid
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Remove duplicate labels
        
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')  # Adjust the fontsize and remove duplicates
        ax.grid(True)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"./classifier/{classifier}_{galaxy_classes}_{subset_size}_loss.png")
        plt.close()


###############################################
############ PLOTS AFTER ALL FOLDS ############
###############################################

class_descriptions = [cls['description'] for cls in classes if cls['tag'] in galaxy_classes]

plot_confusion_matrix(all_true_labels, all_pred_labels, model_name)
plot_roc_curves(all_true_labels, all_pred_probs, model_name)
plot_loss(models)
plot_all_metrics_vs_dataset_size(metrics, model_name)
plot_metrics(metrics, model_name)

# Print metrics summary for each subset size
for metric in ["accuracy", "precision", "recall", "f1_score"]:
    print(f"{metric.capitalize()}:")
    for fold in range(num_folds):
        for subset_size in dataset_sizes[fold]:
            metric_values = []
            for experiment in range(num_experiments):
                for lr in learning_rates:
                    for reg in regularization_params:
                        key = f"{classifier}_{metric}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        metric_values.append(metrics[key])

                        mean = np.mean(metric_values)
                        std = np.std(metric_values)

                        print(f"Subset size {subset_size} learning rate {lr} and regularisation parameters {reg}: {mean:.4f} ± {std:.4f}")

# Print the training times for each fold and subset size with mean and std
print("\nTraining Times:")
for subset_size, folds in training_times.items():
    times = []
    print(f"Subset size {subset_size}:")
    for fold, elapsed_times in folds.items():
        if not elapsed_times:  # Skip if no times recorded
            print(f"  Fold {fold}: No training times recorded.")
            continue
        for elapsed_time in elapsed_times:
            times.append(elapsed_time)
            print(f"  Fold {fold}: {elapsed_time:.2f} seconds")

    if times:
        # Calculate mean and standard deviation
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Mean training time: {mean_time:.2f} seconds ± {std_time:.2f} seconds\n")
    else:
        print("  No training times recorded for this subset size.\n")