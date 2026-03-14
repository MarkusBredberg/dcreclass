import os
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dcreclass.utils.calc_tools import round_to_1

###############################################
########### DATA STORING FUNCTIONS ###############
###############################################

def initialise_metrics(metrics, key_base):
    metrics.setdefault(f"{key_base}_accuracy", [])
    metrics.setdefault(f"{key_base}_precision", [])
    metrics.setdefault(f"{key_base}_recall", [])
    metrics.setdefault(f"{key_base}_f1_score", [])

def update_metrics(metrics, key_base, accuracy, precision, recall, f1):
    metrics[f"{key_base}_accuracy"].append(accuracy)
    metrics[f"{key_base}_precision"].append(precision)
    metrics[f"{key_base}_recall"].append(recall)
    metrics[f"{key_base}_f1_score"].append(f1)
    
def initialise_history(history, base, experiment):
    train_loss_key = f"{base}_{experiment}_train_loss"
    train_acc_key = f"{base}_{experiment}_train_acc"
    val_loss_key = f"{base}_{experiment}_val_loss"
    val_acc_key = f"{base}_{experiment}_val_acc"
    test_loss_key = f"{base}_{experiment}_test_loss"
    test_acc_key = f"{base}_{experiment}_test_acc"
    for key in [train_loss_key, val_loss_key, val_acc_key, train_acc_key, test_loss_key, test_acc_key]:
        if key not in history:
            history[key] = []

def initialise_labels(key, all_true_labels, all_pred_labels):
    all_true_labels[key] = []
    all_pred_labels[key] = []
    
####################################################
############ OVERFITTING DIAGNOSTICS ################
####################################################

def check_overfitting(metrics, history, classifier_name, dataset_sizes, folds, lr, reg, label_smoothing, crop_mode, percentile_lo, percentile_hi, ver_key):
    """ Same as check_overfitting_indicators but includes all folds and parameter combinations """
    print("\n" + "="*60)
    print("COMPREHENSIVE OVERFITTING DIAGNOSTICS ACROSS ALL EXPERIMENTS")
    print("="*60)

    # Calculate the average and std for everything combined
    all_train_acc = []
    all_val_acc = []
    all_test_acc = []
    for fold in folds:
        for subset_size in dataset_sizes.get(fold, []):
            base = (f"{classifier_name}_ver{ver_key}_cm{crop_mode}"
                    f"_lr{lr}_reg{reg}_ls{label_smoothing}"
                    f"_lo{percentile_lo}_hi{percentile_hi}"
                    f"_f{fold}_ss{round_to_1(subset_size)}")

            # Only extend if the metrics actually exist for this fold/subset
            if f"{base}_train_acc" in metrics and len(metrics[f"{base}_train_acc"]) > 0:
                all_train_acc.extend(metrics.get(f"{base}_train_acc", []))
                all_val_acc.extend(metrics.get(f"{base}_val_acc", []))
                all_test_acc.extend(metrics.get(f"{base}_accuracy", []))
    
    # Only compute stats if we have data
    if all_train_acc:
        print(f"📊 Overall Train Accuracy: {np.mean(all_train_acc):.4f} ± {np.std(all_train_acc):.4f}")
        print(f"📊 Overall Validation Accuracy: {np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}")
        print(f"📊 Overall Test Accuracy: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}")
    else:
        print("⚠️  No accuracy metrics found for the specified folds")
        
##################################################
############### METRICS CALCULATION ################
##################################################
    

def compute_classification_metrics(y_true, y_pred, num_classes, multilabel=False):
    acc = accuracy_score(y_true, y_pred)
    if multilabel:
        avg = 'macro'
        return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                     recall_score(y_true, y_pred, average=avg, zero_division=0), \
                     f1_score(y_true, y_pred, average=avg, zero_division=0)
    if num_classes == 2:
        return acc, precision_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0), \
                     recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0), \
                     f1_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
    avg = 'macro'
    return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                 recall_score(y_true, y_pred, average=avg, zero_division=0), \
                 f1_score(y_true, y_pred, average=avg, zero_division=0)
                 
def plot_intensity_histogram(tensor1, tensor2, label1, label2, save_path, bins=30):
    vals1 = tensor1.sum(dim=tuple(range(1, tensor1.ndim))).cpu().numpy()
    vals2 = tensor2.sum(dim=tuple(range(1, tensor2.ndim))).cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.hist(vals1, bins=bins, alpha=0.6, label=label1, color='C1')
    plt.hist(vals2, bins=bins, alpha=0.6, label=label2, color='C0')
    plt.xlabel('Total Intensity'); plt.ylabel('Count')
    plt.title(f'Total Intensity per Image: {label1} vs {label2}')
    plt.legend(); plt.savefig(save_path); plt.close()

def plot_training_history(history, base, experiment, save_dir='./classifier/figures/loss_curves'):
    """
    Plot training, validation, and test loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history
        base: Base key for this experiment
        experiment: Experiment number
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    train_loss_key = f"{base}_{experiment}_train_loss"
    val_loss_key = f"{base}_{experiment}_val_loss"
    test_loss_key = f"{base}_{experiment}_test_loss"
    train_acc_key = f"{base}_{experiment}_train_acc"
    val_acc_key = f"{base}_{experiment}_val_acc"
    test_acc_key = f"{base}_{experiment}_test_acc"
    
    # Check if keys exist
    if train_loss_key not in history or val_loss_key not in history:
        print(f"Warning: Loss keys not found for {base}_{experiment}")
        return
    
    train_losses = history[train_loss_key]
    val_losses = history[val_loss_key]
    test_losses = history.get(test_loss_key, [])
    train_accs = history.get(train_acc_key, [])
    val_accs = history.get(val_acc_key, [])
    test_accs = history.get(test_acc_key, [])
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    if test_losses:
        ax1.plot(epochs[:len(test_losses)], test_losses, 'g-', label='Test Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training, Validation, and Test Loss', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for best validation loss
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax1.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.plot(best_epoch, best_val_loss, 'o', color='orange', markersize=8, 
                label=f'Best Val (epoch {best_epoch})')
        ax1.legend(fontsize=11)
    
    # Plot accuracy (if available)
    if train_accs and val_accs:
        ax2.plot(epochs[:len(train_accs)], train_accs, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs[:len(val_accs)], val_accs, 'r-', label='Val Acc', linewidth=2)
        if test_accs:
            ax2.plot(epochs[:len(test_accs)], test_accs, 'g-', label='Test Acc', linewidth=2)
        
        # Add vertical line at best validation epoch
        if val_losses:
            best_epoch = np.argmin(val_losses) + 1
            ax2.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Mark the accuracy values at best validation epoch
            if best_epoch <= len(val_accs):
                ax2.plot(best_epoch, val_accs[best_epoch-1], 'o', color='red', markersize=8)
            if test_accs and best_epoch <= len(test_accs):
                ax2.plot(best_epoch, test_accs[best_epoch-1], 'o', color='green', markersize=8)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title(f'Training, Validation, and Test Accuracy', fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    else:
        ax2.text(0.5, 0.5, 'Accuracy data not available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    # Add a supertitle
    fig.suptitle(f'Training History for {base} Experiment {experiment}', fontsize=16)
    plt.tight_layout()
    save_path = f"{save_dir}/{base}_exp{experiment}_training_curves_rs=41.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
def img_hash(img: torch.Tensor) -> str:
    arr = img.cpu().contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()

def config_already_exists(classifier, galaxy_classes, lr, reg, percentile_lo, percentile_hi,
                          cs, ds, ver_key, fold, subset_size, experiment,
                          metrics_dir, use_global_norm=False, global_norm_mode='none'):
    """
    Check if a configuration has already been trained and saved.

    Returns:
        bool: True if the PKL file exists, False otherwise
    """
    base = (f"{classifier}_{galaxy_classes}_lr{lr}_reg{reg}"
            f"_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}"
            f"_ver{ver_key}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}")
    if use_global_norm:
        fname = f"{base}_{global_norm_mode}_metrics_data.pkl"
    else:
        fname = f"{base}_metrics_data.pkl"
    metrics_path = os.path.join(metrics_dir, fname)
    exists = os.path.exists(metrics_path)
    if exists:
        print(f"✓ Configuration already exists: fold={fold}, subset_size={subset_size}, experiment={experiment}")
    return exists

""" These functions are used to train a model and evaluate its performance."""
def reset_weights(m):
    '''plt.close()

    This function will reset model weights to a specified initialization.
    Works for most common types of layers.
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def display_examples(X, y_true, y_pred, indices, title):
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices[:3]):  # Display first three examples
        plt.subplot(1, 3, i + 1)
        plt.imshow(X[idx].squeeze(), cmap='gray', interpolation='none')  # Make sure to squeeze in case there's an extra singleton dimension
        plt.title(f"{title}\nTrue: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()

def pad_sequences(sequences, pad_value=0.0):
    '''
    This function will pad the lists to ensure they all have the same length.
    '''
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_sequences.append(seq + [pad_value] * (max_length - len(seq)))
        else:
            padded_sequences.append(seq)
    return padded_sequences


def mixup_data(x1, x2, y, alpha=0.4):
    """
    Apply MixUp augmentation: create convex combinations of pairs of examples.
    
    Args:
        x1: First input (images)
        x2: Second input (scattering coefficients)
        y: Labels
        alpha: MixUp hyperparameter
    
    Returns:
        Mixed inputs and targets
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) # Sample lambda from Beta distribution because 
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def permute_like(x, perm):
    if x is None:
        return None

    # Torch tensors: deterministic gather via index_select and device-safe indices
    if isinstance(x, torch.Tensor):
        idx = perm.to(device=x.device, dtype=torch.long)
        return x.index_select(0, idx)

    # Prepare a plain integer index array once for non-tensor cases
    if isinstance(perm, torch.Tensor):
        idx = perm.detach().cpu().to(torch.long).tolist()
    elif isinstance(perm, np.ndarray):
        idx = [int(i) for i in perm.tolist()]
    else:
        idx = [int(i) for i in perm]

    if isinstance(x, np.ndarray):
        return x[np.asarray(idx, dtype=np.int64)]
    if isinstance(x, list):
        return [x[i] for i in idx]
    if isinstance(x, tuple):
        return tuple(x[i] for i in idx)
    return x

def relabel(y, galaxy_classes, multilabel=False):
    """
    Convert raw single-class ids to zero-based indices or 2-bit multi-label targets.
    multilabel=True: RH (52) -> [1,0],  RR (53) -> [0,1]
    multilabel=False: subtract min(galaxy_classes) to get 0-based index labels.
    """
    if multilabel:
        y = y.long()
        out = torch.zeros((y.shape[0], 2), dtype=torch.float32, device=y.device)
        out[:, 0] = (y == 52).float()   # RH
        out[:, 1] = (y == 53).float()   # RR
        return out
    return (y - min(galaxy_classes)).long()


# Updated process_data function to handle padding
def process_data(data, type="Loss"):
    if len(data) == 0 or all(len(seq) == 0 for seq in data):
        return np.array([]), np.array([])  # Handle case where all sequences are empty

    data = np.array(pad_sequences(data))  # Pad sequences before converting to NumPy array
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    if type == "Loss":
        data_mean = np.mean(data, axis=0)  # Average over num_experiments
        if data.shape[0] > 1:
            data_std = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
        else:
            data_std = np.zeros_like(data_mean)
        return data_mean, data_std
    elif type == "Accuracy":
        data_mean_vec = np.mean(data, axis=0)
        data_mean = np.max(data_mean_vec)  # Average over num_experiments
        maxind = np.where(data_mean == data_mean_vec)[0]
        if maxind.size > 1:
            maxind = maxind[0]
        if data.shape[0] > 1:
            data_std_vec = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
            data_std = data_std_vec[maxind]
        else:
            data_std = np.zeros_like(data_mean)
        return data_mean, data_std

    def save_checkpoint(self, val_loss, model, model_path):
        '''Save model when validation loss decreases.'''
        #if self.verbose:
            #print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), model_path)
        self.best_loss = val_loss
        
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, save_model=True):
        """
        Initializes the EarlyStopping class.
        
        Parameters:
        - patience: Number of epochs to wait for improvement before stopping.
        - verbose: Whether to print messages when the model is saved.
        - save_model: Whether to save the model when validation loss improves.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_model = save_model  # New flag to control model saving

    def __call__(self, val_loss, model=None, model_path=None):
        """
        Check if validation loss has improved, and handle early stopping or saving the model.

        Parameters:
        - val_loss: The current validation loss.
        - model: The model to save (if saving is enabled).
        - model_path: The path where the model should be saved (if saving is enabled).
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_model:
                self.save_checkpoint(val_loss, model, model_path)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            if self.save_model:
                self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), model_path)
        