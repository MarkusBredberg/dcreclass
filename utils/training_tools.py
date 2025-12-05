import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

""" These functions are used to train a model and evaluate its performance."""

def vae_loss_function(x, x_hat, mean, log_var, batch_size=32, beta=1.0):
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    x = x.view(batch_size, -1)
    x_hat = x_hat.view(batch_size, -1)    
    RecLoss = F.mse_loss(x_hat, x, reduction='sum')
    #RecLoss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    return RecLoss + beta * KLD

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

def matrix_square_sum(matrix):
    return sum([pow(num, 2) for row in matrix for num in row])

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