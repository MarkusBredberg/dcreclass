import numpy as np, pickle, torch, os, itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.data_loader import get_classes, load_galaxies
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Import your classifiers
from utils.classifiers import CNN, ScatterNet, DualCNNSqueezeNet, DualScatterSqueezeNet
from utils.calc_tools import fold_T_axis, custom_collate

# Create output directories
os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)
os.makedirs('./classifier/attention_maps', exist_ok=True)

print("Running enhanced evaluation script 2.0 with attention visualization")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

###############################################
################ CONFIGURATION ################
###############################################

FILTERED = True       
USE_GLOBAL_NORMALISATION = False # False for per-image norm
GLOBAL_NORM_MODE = "percentile"
ADJUST_POSITIVE_CLASS = True
GENERATE_ATTENTION_MAPS = True  # Toggle attention map generation

classes = get_classes()
galaxy_classes = [50, 51]
learning_rates = [5e-5]
regularization_params = [1e-1]
num_experiments = 3
percentile_lo, percentile_hi = 30, 99
folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
classifier = ["CNN",         # 0.Very Simple CNN
              "ScatterNet",  # 1.Scattering coefficients as input to MLP
              "DualCSN",     # 2.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              "DualSSN"      # 3.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              ][3]
crop_size = (512, 512)
downsample_size = (128, 128)
version = 'T25kpc'
J, L, order = 2, 12, 2

# NEW: Attention visualization settings
ATTENTION_METHODS = ['saliency', 'gradcam', 'integrated_gradients']  # Methods to use

# Define colormap for visualization
cmap_green = LinearSegmentedColormap.from_list( 
    'white_to_green',
    ['white', '#006400']
)

###############################################
######### ATTENTION VISUALIZATION #############
###############################################

class AttentionVisualizer:
    """
    Class to generate various attention/saliency visualizations for trained models.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the attention visualizer.
        
        Args:
            model: Trained PyTorch model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for intermediate activations (needed for Grad-CAM)
        self.activations = []
        self.gradients = []
        
    def _register_hooks(self):
        """
        Register forward and backward hooks to capture activations and gradients.
        This is needed for Grad-CAM.
        """
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        # Register hooks on the last convolutional layer
        # This varies by architecture - adjust as needed
        target_layer = None
        
        # Find the last Conv2d layer in the model
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
            return True
        return False
    
    # Around line 125: Modify generate_saliency_map
    def generate_saliency_map(self, image, scat, target_class, branch='image'):
        """
        Generate saliency map for specified branch.
        
        Args:
            branch: 'image' or 'scattering' - which input to compute gradients for
        """
        # Enable gradient computation for specified input
        if branch == 'image':
            image.requires_grad = True
            target_input = image
        elif branch == 'scattering':
            if scat is None:
                return None  # No scattering branch
            scat.requires_grad = True
            target_input = scat
        else:
            return None
        
        # Forward pass
        if scat is not None:
            output = self.model(image, scat)
        else:
            output = self.model(image)
        
        # Handle different output shapes
        if output.ndim > 2:
            output = F.adaptive_avg_pool2d(output, (1, 1)).squeeze()
        if output.ndim == 1:
            output = output.unsqueeze(0)
        
        # Get score for target class
        score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients with respect to target input
        gradients = target_input.grad.data.abs()
        
        # Aggregate across channels
        saliency = gradients.squeeze().cpu().numpy()
        if saliency.ndim > 2:
            saliency = np.max(saliency, axis=0)
        elif saliency.ndim == 1:
            # Scattering coefficients are 1D, visualize as bar chart or heatmap
            # For simplicity, reshape to 2D square
            size = int(np.ceil(np.sqrt(len(saliency))))
            padded = np.zeros(size * size)
            padded[:len(saliency)] = saliency
            saliency = padded.reshape(size, size)
        
        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def generate_gradcam(self, image, scat, target_class, branch='image'):
        """
        Generate Grad-CAM (Gradient-weighted Class Activation Mapping) for specified branch.
        Shows which spatial regions are important for the prediction.
        
        Args:
            image: Input image tensor [1, C, H, W]
            scat: Scattering coefficients (can be None)
            target_class: Target class index
            branch: 'image' or 'scattering' - which input to analyze
            
        Returns:
            cam: Class activation map [H, W]
        """
        # Check if branch is valid
        if branch == 'scattering' and scat is None:
            return None
        
        # Clear previous activations and gradients
        self.activations = []
        self.gradients = []
        
        # Register hooks
        hooks_registered = self._register_hooks()
        if not hooks_registered:
            print("Warning: Could not find Conv2d layer for Grad-CAM")
            return None
        
        # Forward pass
        if scat is not None:
            output = self.model(image, scat)
        else:
            output = self.model(image)
        
        # Handle different output shapes
        if output.ndim > 2:
            output = F.adaptive_avg_pool2d(output, (1, 1)).squeeze()
        if output.ndim == 1:
            output = output.unsqueeze(0)
        
        # Get score for target class
        score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Check if we captured activations and gradients
        if not self.activations or not self.gradients:
            print("Warning: No activations or gradients captured")
            return None
        
        # Get the last activation and gradient
        activation = self.activations[-1]  # [1, C, H', W']
        gradient = self.gradients[-1]      # [1, C, H', W']
        
        # Global average pooling of gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activation).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Determine target size for upsampling based on branch
        if branch == 'image':
            target_size = image.shape[-2:]
        elif branch == 'scattering':
            # For scattering coefficients, create a square visualization
            # Map to image space for visualization
            target_size = image.shape[-2:]
        else:
            target_size = image.shape[-2:]
        
        # Upsample to target size
        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
        
        # Convert to numpy and normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def generate_integrated_gradients(self, image, scat, target_class, branch='image', steps=50):
        """
        Generate Integrated Gradients attribution map for specified branch.
        More robust than vanilla saliency by averaging gradients along a path.
        
        Args:
            image: Input image tensor [1, C, H, W]
            scat: Scattering coefficients (can be None)
            target_class: Target class index
            branch: 'image' or 'scattering' - which input to analyze
            steps: Number of interpolation steps
            
        Returns:
            attribution: Attribution map [H, W]
        """
        # Check if branch is valid
        if branch == 'scattering' and scat is None:
            return None
        
        # Create baselines (zeros)
        baseline_image = torch.zeros_like(image)
        baseline_scat = torch.zeros_like(scat) if scat is not None else None
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        integrated_grads_image = None
        integrated_grads_scat = None
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated_image = (baseline_image + alpha * (image - baseline_image)).clone().detach()
            interpolated_image.requires_grad = True

            if scat is not None:
                interpolated_scat = (baseline_scat + alpha * (scat - baseline_scat)).clone().detach()
                interpolated_scat.requires_grad = True
            else:
                interpolated_scat = None
            
            # Forward pass
            if interpolated_scat is not None:
                output = self.model(interpolated_image, interpolated_scat)
            else:
                output = self.model(interpolated_image)
            
            # Handle output shapes
            if output.ndim > 2:
                output = F.adaptive_avg_pool2d(output, (1, 1)).squeeze()
            if output.ndim == 1:
                output = output.unsqueeze(0)
            
            # Get score for target class
            score = output[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            score.backward()
            
            # Accumulate gradients for the specified branch
            if branch == 'image':
                grads = interpolated_image.grad.data
                if integrated_grads_image is None:
                    integrated_grads_image = grads
                else:
                    integrated_grads_image += grads
            elif branch == 'scattering':
                if interpolated_scat is not None:
                    grads = interpolated_scat.grad.data
                    if integrated_grads_scat is None:
                        integrated_grads_scat = grads
                    else:
                        integrated_grads_scat += grads
        
        # Select the appropriate gradients based on branch
        if branch == 'image':
            integrated_grads = integrated_grads_image
            input_tensor = image
            baseline = baseline_image
        elif branch == 'scattering':
            integrated_grads = integrated_grads_scat
            input_tensor = scat
            baseline = baseline_scat
        else:
            return None
        
        if integrated_grads is None:
            return None
        
        # Average the gradients
        integrated_grads /= steps
        
        # Multiply by (input - baseline)
        attribution = (input_tensor - baseline) * integrated_grads
        
        # Aggregate across channels
        attribution = attribution.squeeze().detach().cpu().numpy()
        if attribution.ndim > 2:
            attribution = np.sum(np.abs(attribution), axis=0)
        elif attribution.ndim == 1:
            # Scattering coefficients are 1D, reshape to 2D for visualization
            size = int(np.ceil(np.sqrt(len(attribution))))
            padded = np.zeros(size * size)
            padded[:len(attribution)] = attribution
            attribution = padded.reshape(size, size)
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution
    
    def visualize_attention(self, image, scat, true_label, pred_label, pred_label_idx,
                        source_name=None,
                        methods=['saliency', 'gradcam', 'integrated_gradients'],
                        save_path=None):
        """
        Generate and visualize multiple attention maps for a single example.
        
        Args:
            image: Input image tensor [1, C, H, W]
            scat: Scattering coefficients (can be None)
            true_label: Ground truth class (original label like 50/51) - for display
            pred_label: Predicted class (original label like 50/51) - for display
            pred_label_idx: Predicted class index (0 or 1) - for model operations
            source_name: Optional source name to display
            methods: List of methods to use
            save_path: Path to save the visualization
        """
        # Prepare the original image for display
        img_display = image.squeeze().cpu().numpy()
        if img_display.ndim > 2:
            img_display = img_display[0]
        
        # Number of subplots needed
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))  # 4x4 per subplot
        
        # Plot original image
        title_text = f'Original\nTrue: {true_label}\nPred: {pred_label}'
        if source_name:
            title_text = f'{source_name}\n' + title_text
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title(title_text, fontsize=10)  # Smaller font for more text
        axes[0].axis('off')
        
        # Generate and plot each attention map
        for idx, method in enumerate(methods, 1):
            if method == 'saliency':
                attention_map = self.generate_saliency_map(image, scat, pred_label_idx)  # USE INDEX
                title = 'Saliency Map'
            elif method == 'gradcam':
                attention_map = self.generate_gradcam(image, scat, pred_label_idx)  # USE INDEX
                title = 'Grad-CAM'
            elif method == 'integrated_gradients':
                attention_map = self.generate_integrated_gradients(image, scat, pred_label_idx)  # USE INDEX
                title = 'Integrated Gradients'
            else:
                continue
            
            if attention_map is None:
                axes[idx].text(0.5, 0.5, f'{method}\nNot Available', 
                             ha='center', va='center', fontsize=12)
                axes[idx].axis('off')
                continue
            
            # Overlay attention map on original image
            axes[idx].imshow(img_display, cmap='gray', alpha=0.6)
            im = axes[idx].imshow(attention_map, cmap='jet', alpha=0.4)
            axes[idx].set_title(title, fontsize=12)
            axes[idx].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
        
        plt.close(fig)


def generate_attention_visualizations(model, test_loader, galaxy_classes, source_names,
                                      save_dir='./classifier/attention_maps',
                                      methods=None, classifier_name=classifier):
    """
    Generate attention visualizations for test samples.
    For multi-branch models (DualSSN, DualCSN), shows attention from both branches.
    """
    if methods is None:
        methods = ['saliency', 'gradcam', 'integrated_gradients']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Detect multi-branch architecture
    is_multi_branch = classifier_name in ['DualSSN', 'DualCSN']
    sources_per_class = 3 if is_multi_branch else 6
    
    print(f"Classifier: {classifier_name}")
    print(f"Multi-branch: {is_multi_branch}")
    print(f"Will collect {sources_per_class} sources per class")
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(model, device=device)
    
    # Storage for examples from each class
    class_examples = {cls: {'images': [], 'scats': [], 'true_labels': [], 
                            'pred_labels': [], 'probs': [], 'indices': [], 'source_names': []} 
                    for cls in galaxy_classes}

    # Collect examples
    print("Collecting test examples for attention visualization...")
    model.eval()
    sample_idx_global = 0
    with torch.no_grad():
        for images, scat, labels in test_loader:
            images = images.to(device)
            scat = scat.to(device) if scat is not None else None
            labels = labels.to(device)
            
            # Get predictions
            if scat is not None:
                outputs = model(images, scat)
            else:
                outputs = model(images)
            
            # Handle output shapes
            if outputs.ndim > 2:
                outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).squeeze()
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            
            # Store examples for each class
            for i in range(len(labels)):
                true_label = int(labels[i].item())
                pred_label = int(preds[i].item())
                
                if len(class_examples[true_label]['images']) < sources_per_class:
                    class_examples[true_label]['images'].append(images[i:i+1].clone())
                    class_examples[true_label]['scats'].append(
                        scat[i:i+1].clone() if scat is not None else None)
                    class_examples[true_label]['true_labels'].append(true_label)
                    class_examples[true_label]['pred_labels'].append(pred_label)
                    class_examples[true_label]['probs'].append(probs[i].cpu().numpy())
                    class_examples[true_label]['indices'].append(sample_idx_global + i)
                    class_examples[true_label]['source_names'].append(
                        source_names[sample_idx_global + i] if sample_idx_global + i < len(source_names) else f"Test_{sample_idx_global + i}")
            
            sample_idx_global += len(labels)
            
            # Check if we have enough examples
            if all(len(v['images']) >= sources_per_class for v in class_examples.values()):
                break
    
    # Generate visualizations
    print(f"Generating attention maps using methods: {methods}")

    # Group samples by (true_label, pred_label) pairs
    class_pred_groups = defaultdict(lambda: {'images': [], 'scats': [], 'source_names': [], 
                                             'pred_idx': [], 'true_labels': []})

    for class_idx, examples in class_examples.items():
        for sample_idx in range(len(examples['images'])):
            pred_label_idx = examples['pred_labels'][sample_idx]
            pred_label = galaxy_classes[pred_label_idx]
            
            key = (class_idx, pred_label)
            class_pred_groups[key]['images'].append(examples['images'][sample_idx])
            class_pred_groups[key]['scats'].append(examples['scats'][sample_idx])
            class_pred_groups[key]['source_names'].append(examples['source_names'][sample_idx])
            class_pred_groups[key]['pred_idx'].append(pred_label_idx)
            class_pred_groups[key]['true_labels'].append(class_idx)

    # Create one figure per (true, pred) combination
    for (true_label, pred_label), group_data in class_pred_groups.items():
        if len(group_data['images']) == 0:
            continue
        
        n_sources = len(group_data['images'])
        n_methods = len(methods)
        
        # Calculate total rows: sources × branches
        if is_multi_branch:
            n_rows = n_sources * 2  # Each source gets 2 rows (image branch + scat branch)
        else:
            n_rows = n_sources  # Each source gets 1 row
        
        print(f"\nProcessing true={true_label}, pred={pred_label}")
        print(f"  Sources: {n_sources}, Total rows: {n_rows}")
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_methods + 1, 
                                figsize=(4 * (n_methods + 1), 4 * n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Process each source
        row_idx = 0
        for source_idx in range(n_sources):
            image = group_data['images'][source_idx]
            scat_coeff = group_data['scats'][source_idx]
            pred_label_idx = group_data['pred_idx'][source_idx]
            true_label_val = group_data['true_labels'][source_idx]
            source_name = group_data['source_names'][source_idx]
            
            # Extract PSZ2 name
            psz2_name = source_name.split('/')[-1].replace('.fits', '') if '/' in source_name else source_name
            
            # Prepare image for display
            img_display = image.squeeze().cpu().numpy()
            if img_display.ndim > 2:
                img_display = img_display[0]
            
            # Branch configurations
            if is_multi_branch:
                branches = [
                    ('image', 'Image Branch'),
                    ('scattering', 'Scattering Branch')
                ]
            else:
                branches = [('image', '')]
            
            # Generate rows for each branch
            for branch_type, branch_label in branches:
                # Plot original image
                axes[row_idx, 0].imshow(img_display, cmap='gray')
                axes[row_idx, 0].set_title('Original' if row_idx == 0 else '', fontsize=12)
                axes[row_idx, 0].axis('off')

                # Y-axis label with source info
                label_parts = [psz2_name, f"True:{true_label_val}", f"Pred:{galaxy_classes[pred_label_idx]}"]
                if is_multi_branch:
                    label_parts.append(f"[{branch_label}]")
                
                y_label = '\n'.join(label_parts)
                fig.text(0.01, 1 - (row_idx + 0.5) / n_rows, y_label, 
                        fontsize=7, va='center', ha='left', 
                        transform=fig.transFigure, rotation=0)
                
                # Generate and plot attention maps for this branch
                for method_idx, method in enumerate(methods, 1):
                    if method == 'saliency':
                        attention_map = visualizer.generate_saliency_map(
                            image, scat_coeff, pred_label_idx, branch=branch_type)
                        title = 'Saliency Map' if row_idx == 0 else ''
                    elif method == 'gradcam':
                        attention_map = visualizer.generate_gradcam(
                            image, scat_coeff, pred_label_idx, branch=branch_type)
                        title = 'Grad-CAM' if row_idx == 0 else ''
                    elif method == 'integrated_gradients':
                        attention_map = visualizer.generate_integrated_gradients(
                            image, scat_coeff, pred_label_idx, branch=branch_type)
                        title = 'Integrated Gradients' if row_idx == 0 else ''
                    else:
                        continue
                    
                    if attention_map is None:
                        axes[row_idx, method_idx].text(0.5, 0.5, f'{method}\nNot Available',
                                                    ha='center', va='center', fontsize=10)
                        axes[row_idx, method_idx].axis('off')
                        continue
                    
                    # Overlay attention map
                    axes[row_idx, method_idx].imshow(img_display, cmap='gray', alpha=0.6)
                    im = axes[row_idx, method_idx].imshow(attention_map, cmap='jet', alpha=0.4)
                    axes[row_idx, method_idx].set_title(title, fontsize=12)
                    axes[row_idx, method_idx].axis('off')
                    
                    # Add colorbar only on first row
                    if row_idx == 0:
                        plt.colorbar(im, ax=axes[row_idx, method_idx], 
                                fraction=0.046, pad=0.04)
                
                row_idx += 1
        
        # Add overall title with classifier name
        fig.suptitle(f'{classifier_name} — True: {true_label}, Predicted: {pred_label}', 
                    fontsize=16, y=0.995)
        
        plt.tight_layout(rect=[0.12, 0, 1, 0.99])  # More left margin for longer labels
        
        # Save as PNG
        save_path = os.path.join(save_dir, f"attention_maps.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid visualization to {save_path}")
        plt.close(fig)

    print(f"\nAttention visualizations saved to {save_dir}")


###############################################
############# HELPER FUNCTIONS ################
###############################################

from math import log10, floor
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

def recalculate_metrics_with_correct_positive_class(y_true, y_pred, pos_label=0):
    """
    Recalculate precision, recall, and F1 with the correct positive class.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    
    return acc, precision, recall, f1

def initialize_metrics(metrics, subset_size, fold, experiment, lr, reg):
    """Initialize metric storage dictionaries"""
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []
    metrics[f"all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"] = []

def update_metrics(metrics, subset_size, fold, experiment, lr, reg,
                   accuracy, precision, recall, f1,
                   history_val, all_true_labels, all_pred_labels, training_times, all_pred_probs):
    """Update metrics dictionaries"""
    subset_size_str = str(subset_size)
    metrics[f"accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(accuracy)
    metrics[f"precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(precision)
    metrics[f"recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(recall)
    metrics[f"f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(f1)
    metrics[f"history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(history_val)
    metrics[f"all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(all_true_labels)
    metrics[f"all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(all_pred_labels)
    metrics[f"training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(training_times)
    metrics[f"all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}"].append(all_pred_probs)


###############################################
######### SETTING THE RIGHT PARAMETERS ########
###############################################

directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

# Define dataset sizes (must match training)
if galaxy_classes == [50, 51]:
    dataset_sizes = {fold: [3000] for fold in range(10)}
elif galaxy_classes == [52, 53]:
    dataset_sizes = {fold: [2, 16, 168] for fold in range(10)}
else:
    print("Please specify the dataset sizes for the given galaxy classes.")
    exit(1)

largest_sz = max([sz for sizes in dataset_sizes.values() for sz in sizes])

# Merge map generation
merge_map = {}
all_sizes = {s for fs in dataset_sizes.values() for s in fs}
for size in all_sizes:
    nd = len(str(size))
    factor = 10 ** max(nd - 2, 0)
    new_rep = int(round(size / factor) * factor)
    merge_map[size] = str(new_rep)

all_cluster_metrics = {
    'errors': [],
    'distances': [],
    'std_devs': []
}

###############################################
############# READ IN PICKLE DATA #############
###############################################

tot_metrics = {}

print("Loading metrics from pickle files...")
loaded_count = 0
failed_count = 0

# Load all saved metrics
for lr, reg, experiment, fold in itertools.product(
    learning_rates, regularization_params, range(num_experiments), folds
):
    for subset_size in dataset_sizes[fold]:
        cs = f"{crop_size[0]}x{crop_size[1]}"
        ds = f"{downsample_size[0]}x{downsample_size[1]}"
        if USE_GLOBAL_NORMALISATION:
            metrics_read_path = f"./classifier/4.1.runs/global_norm/{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{version}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_{GLOBAL_NORM_MODE}_metrics_data.pkl"
        else:
            metrics_read_path = f"./classifier/4.1.runs/{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{version}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_metrics_data.pkl"
        
        try:
            with open(metrics_read_path, 'rb') as f:
                data = pickle.load(f)
            
            loaded_metrics = data["metrics"]
            history = data["history"]
            all_true_labels_dict = data["all_true_labels"]
            all_pred_labels_dict = data["all_pred_labels"]
            all_pred_probs_dict = data["all_pred_probs"]
            training_times_dict = data["training_times"]
            
            if data.get("cluster_error") is not None:
                all_cluster_metrics['errors'].append(data["cluster_error"])
            if data.get("cluster_distance") is not None:
                all_cluster_metrics['distances'].append(data["cluster_distance"])
            if data.get("cluster_std_dev") is not None:
                all_cluster_metrics['std_devs'].append(data["cluster_std_dev"])
            
            initialize_metrics(tot_metrics, subset_size, fold, experiment, lr, reg)
            
            base = f"cl{classifier}_ss{subset_size}_f{fold}_lr{lr}_reg{reg}_ls0.1_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{version}"
            
            acc = loaded_metrics.get("accuracy", [])
            prec = loaded_metrics.get("precision", [])
            rec = loaded_metrics.get("recall", [])
            f1 = loaded_metrics.get("f1_score", [])
            
            y_true = all_true_labels_dict.get(base, [])
            y_pred = all_pred_labels_dict.get(base, [])
            y_probs = all_pred_probs_dict.get(base, [])
            
            if not y_true or not y_pred:
                failed_count += 1
                continue
            
            if ADJUST_POSITIVE_CLASS:
                acc_val, prec_val, rec_val, f1_val = recalculate_metrics_with_correct_positive_class(
                    y_true, y_pred, pos_label=0
                )
            else:
                acc_val = acc[0] if isinstance(acc, list) and acc else 0.0
                prec_val = prec[0] if isinstance(prec, list) and prec else 0.0
                rec_val = rec[0] if isinstance(rec, list) and rec else 0.0
                f1_val = f1[0] if isinstance(f1, list) and f1 else 0.0
            
            update_metrics(
                tot_metrics, subset_size, fold, experiment, lr, reg,
                acc_val, prec_val, rec_val, f1_val,
                history,
                all_true_labels_dict,
                all_pred_labels_dict,
                training_times_dict,
                all_pred_probs_dict
            )
            loaded_count += 1
            
        except FileNotFoundError:
            print(f"Metrics file not found at {metrics_read_path}")
            failed_count += 1
            continue
        except Exception as e:
            print(f"Error loading {metrics_read_path}: {e}")
            failed_count += 1
            continue

print(f"Done loading. Successfully loaded: {loaded_count}, Failed: {failed_count}")
metrics = tot_metrics


###############################################
########### PLOTTING FUNCTIONS ################
###############################################

def robust_metric_histograms(
    metrics,
    galaxy_classes=galaxy_classes,
    classifier=classifier,
    dataset_sizes=dataset_sizes,
    folds=folds,
    learning_rates=learning_rates,
    regularization_params=regularization_params,
    save_dir="./classifier/figures"
):
    """
    For each metric series (accuracy/precision/recall/f1_score), make a histogram
    with 16/50/84 percentile markers and write a compact percentile summary CSV.
    
    This function properly aggregates across ALL experiments (folds × num_experiments).
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    wanted = {"accuracy", "precision", "recall", "f1_score"}

    # Aggregate metrics across all experiments
    # Key format in metrics dict: "metric_subset_fold_experiment_lr_reg"
    # We want to group by: "metric_subset_lr_reg" (aggregate across ALL folds and experiments)
    grouped = defaultdict(list)

    print("\nAggregating metrics across experiments...")
    for key, values in metrics.items():
        parts = key.split("_")
        
        # Expected format: metric_subset_fold_experiment_lr_reg
        # Example: "accuracy_4000_0_0_5e-05_0.1"
        if len(parts) < 6:
            continue
            
        metric_name = parts[0]  # "accuracy", "precision", etc.
        
        if metric_name not in wanted:
            continue
            
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            continue
            
        try:
            vals = np.asarray(values, dtype=float).ravel()
        except Exception:
            continue
            
        if vals.size == 0 or not np.isfinite(vals).any():
            continue

        # Group by metric_subset_lr_reg (dropping fold and experiment indices)
        # This aggregates across ALL folds AND experiments
        subset_size = parts[1]
        lr = parts[4]
        reg = parts[5]
        group_key = f"{metric_name}_{subset_size}_{lr}_{reg}"
        
        grouped[group_key].extend(vals.tolist())
        
    print(f"Created {len(grouped)} metric groups")

    # Create histograms and calculate statistics for each group
    for group_key, all_vals in grouped.items():
        vals = np.asarray(all_vals, dtype=float)
        
        if vals.size == 0 or not np.isfinite(vals).any():
            continue

        print(f"Processing {group_key}: {len(vals)} values")

        # Calculate percentiles for robust statistics
        p16, p50, p84 = np.percentile(vals, [16, 50, 84])
        mean = np.mean(vals)
        std = np.std(vals)
        sigma68 = 0.5 * (p84 - p16)  # Half-width of central 68% interval

        # Parse the group key for filenames and titles
        gparts = group_key.split("_")
        metric_name = gparts[0]

        # Set up histogram bins
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            eps = max(1e-6, abs(vmin) * 1e-3)
            vmin, vmax = vmin - eps, vmax + eps
        edges = np.linspace(vmin, vmax, 21)

        # Create histogram plot
        plt.figure(figsize=(6, 4))
        counts, bins, patches = plt.hist(vals, bins=edges, color='#77dd77', 
                                        edgecolor="black", alpha=0.7)
        
        # Add vertical lines for statistics
        plt.axvline(mean, linestyle='-', color='blue', linewidth=2, 
                   label=f'Mean: {mean:.3f}')
        plt.axvline(p50, linestyle='--', color='orange', linewidth=2, 
                   label=f'Median: {p50:.3f}')
        plt.axvline(p16, linestyle=':', color='green', linewidth=1.5)
        plt.axvline(p84, linestyle=':', color='green', linewidth=1.5)
        
        # Shade the 68% confidence interval
        plt.fill_betweenx([0, counts.max()], p16, p84, alpha=0.2, 
                         color='green', label=f'68% interval')

        plt.xlim(vmin, vmax)
        plt.xlabel(metric_name.capitalize(), fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f"{metric_name.capitalize()} Distribution (n={len(vals)} runs)", 
                 fontsize=13)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the histogram
        save_path_hist = f"{save_dir}/{galaxy_classes}_{classifier}_{version}_{largest_sz}_{learning_rates[0]}_{regularization_params[0]}_{metric_name}_histogram.pdf"
        plt.savefig(save_path_hist, dpi=150)
        plt.close()
        print(f"  Saved histogram: {os.path.basename(save_path_hist)}")

        # Store statistics for CSV output
        rows.append({
            "metric": metric_name,
            "n": int(vals.size),
            "setting": group_key,
            "mean": float(mean),
            "std": float(std),
            "p16": float(p16),
            "p50": float(p50),
            "p84": float(p84),
            "sigma68": float(sigma68)
        })

    # Write CSV summary
    if rows:
        import csv
        csv_path = f"{save_dir}/{galaxy_classes}_{classifier}_robust_summary.csv"
        
        # Check if file exists to determine if we should append
        file_exists = os.path.exists(csv_path)
        
        fieldnames = ["metric", "n", "setting", "mean", "std", "p16", "p50", "p84", "sigma68"]
        
        # Read existing data if file exists
        existing_data = []
        if file_exists:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
        
        # Create a set of existing settings to avoid duplicates
        existing_settings = {row['setting'] for row in existing_data}
        
        # Filter out rows that already exist
        new_rows = [row for row in rows if row['setting'] not in existing_settings]
        
        if new_rows:
            # Combine existing and new data
            all_rows = existing_data + new_rows
            
            # Write all data (overwrite file)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in all_rows:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
            
            print(f"\n✓ Updated CSV with {len(new_rows)} new entries: {csv_path}")
            print(f"  Total entries: {len(all_rows)}")
        else:
            print(f"\n⏭️  No new entries to add to CSV (all settings already present)")
    else:
        print("\n⚠️  No data to write to CSV")
        

def plot_cluster_metrics(cluster_metrics_dict, save_dir='./classifier/figures'):
    """
    Plot cluster metrics across experiments.
    
    Args:
        cluster_metrics_dict: Dictionary with keys 'errors', 'distances', 'std_devs'
        save_dir: Directory to save the plot
    """
    cluster_errors = cluster_metrics_dict.get('errors', [])
    cluster_distances = cluster_metrics_dict.get('distances', [])
    cluster_stds = cluster_metrics_dict.get('std_devs', [])
    
    # Check if we have any data
    if not cluster_errors and not cluster_distances and not cluster_stds:
        print("⚠️  No cluster metrics data found - skipping cluster metrics plot")
        return
    
    # Create histograms for each cluster metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_data = [
        (cluster_errors, 'Cluster Error', axes[0]),
        (cluster_distances, 'Cluster Distance', axes[1]),
        (cluster_stds, 'Cluster Std Dev', axes[2])
    ]
    
    for data, title, ax in metrics_data:
        if data:
            ax.hist(data, bins=20, color='#77dd77', edgecolor='black', alpha=0.7)
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}±{std_val:.3f}')
            ax.set_xlabel(title)
            ax.set_ylabel('Count')
            ax.set_title(f'{title} Distribution (n={len(data)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {title} data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    save_path = f'{save_dir}/{galaxy_classes}_{classifier}_{version}_{largest_sz}_{lr}_{reg}_cluster_metrics_distribution.pdf'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved cluster metrics plot to {save_path}")

def plot_avg_roc_curves(metrics, merge_map=merge_map, 
                         folds=folds, num_experiments=num_experiments, 
                        learning_rates=learning_rates, regularization_params=regularization_params, 
                        galaxy_classes=galaxy_classes, 
                        class_descriptions={cls['tag']: cls['description'] for cls in classes}, 
                        save_dir='./classifier/figures'):
    """
    Plot average ROC curves with confidence intervals across all experiments.
    """
    from scipy.interpolate import make_interp_spline
    
    # Calculate adjusted class labels (0-indexed)
    min_label = min(galaxy_classes)
    adjusted_classes = [cls - min_label for cls in galaxy_classes]
    fpr_grid = np.linspace(0, 1, 1000)  # Grid for interpolation
    
    for lr in learning_rates:
        for reg in regularization_params:
            # Get unique subset sizes across all folds
            all_subset_sizes = sorted(set([sz for fold in folds for sz in dataset_sizes[fold]]))
            
            for subset_size in all_subset_sizes:
                # Storage for ROC curves from all experiments
                roc_values = {class_label: [] for class_label in adjusted_classes}
                
                for experiment in range(num_experiments):
                    for fold in folds:
                        if subset_size not in dataset_sizes[fold]:
                            continue
                            
                        # Build key to access stored predictions
                        cs = f"{crop_size[0]}x{crop_size[1]}"
                        ds = f"{downsample_size[0]}x{downsample_size[1]}"
                        
                        # Get true labels and predicted probabilities
                        true_labels_dict = metrics.get(f"all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}", [])
                        pred_probs_dict = metrics.get(f"all_pred_probs_{subset_size}_{fold}_{experiment}_{lr}_{reg}", [])
                        
                        if not true_labels_dict or not pred_probs_dict:
                            continue
                            
                        # Extract the actual arrays from the dictionaries
                        base = f"cl{classifier}_ss{subset_size}_f{fold}_lr{lr}_reg{reg}_ls0.1_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{version}"
                        true_labels = true_labels_dict[0].get(base) if isinstance(true_labels_dict, list) else true_labels_dict.get(base)
                        pred_probs = pred_probs_dict[0].get(base) if isinstance(pred_probs_dict, list) else pred_probs_dict.get(base)
                        
                        if true_labels is None or pred_probs is None or len(true_labels) == 0 or len(pred_probs) == 0:
                            continue
                            
                        pred_probs = np.asarray(pred_probs)
                        y = np.asarray(true_labels)

                        # Map tags to indices if needed
                        if y.max() > len(galaxy_classes) - 1:
                            tag_to_idx = {tag: i for i, tag in enumerate(sorted(galaxy_classes))}
                            y = np.vectorize(tag_to_idx.get)(y)

                        # Skip if only one class present (ROC undefined)
                        if np.unique(y).size < 2:
                            print(f"Skipping fold {fold} due to only one class present")
                            continue

                        # Calculate ROC curves
                        if len(adjusted_classes) == 2:
                            # Binary classification
                            scores = pred_probs[:, 1] if pred_probs.ndim == 2 and pred_probs.shape[1] > 1 else pred_probs.ravel()
                            fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
                            interp_tpr = np.interp(fpr_grid, fpr, tpr)
                            class_label = adjusted_classes[1]  # True positive class
                            roc_values[class_label].append(interp_tpr)
                        else:
                            # Multi-class classification
                            y_bin = label_binarize(y, classes=np.arange(len(adjusted_classes)))
                            for i, class_label in enumerate(adjusted_classes):
                                fpr, tpr, _ = roc_curve(y_bin[:, i], pred_probs[:, i])
                                interp_tpr = np.interp(fpr_grid, fpr, tpr)
                                roc_values[class_label].append(interp_tpr)

                # Create ROC plot
                fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
                ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                
                # Plot ROC curve for each class
                for class_label, galaxy_class in zip(adjusted_classes, galaxy_classes):
                    if not roc_values[class_label]:
                        continue
                        
                    tpr_values = np.array(roc_values[class_label])
                    mean_tpr = np.mean(tpr_values, axis=0)
                    std_tpr = np.std(tpr_values, axis=0)
                    tpr_p16, tpr_p84 = np.percentile(tpr_values, [16, 84], axis=0)
                    
                    # Calculate AUC statistics
                    mean_auc = auc(fpr_grid, mean_tpr)
                    n = tpr_values.shape[0]
                    auc_values = [auc(fpr_grid, tpr_values[i, :]) for i in range(n)]
                    std_auc = np.std(auc_values)
                    auc_p16, auc_p84 = np.percentile(auc_values, [16, 84])
                    
                    print(f"Class {class_descriptions.get(galaxy_class, str(galaxy_class))}: "
                          f"AUC={mean_auc:.3f}±{std_auc:.3f}, 16th={auc_p16:.3f}, 84th={auc_p84:.3f}, runs={n}")
                    
                    # Plot mean ROC curve
                    ax.plot(fpr_grid, mean_tpr, lw=3.5, color='#77dd77',
                            label=f'Mean ROC (AUC={mean_auc:.3f}, 16th={auc_p16:.3f}, 84th={auc_p84:.3f})')
                    
                    # Add shaded confidence interval
                    ax.fill_between(fpr_grid,
                                    np.clip(tpr_p16, 0, 1),
                                    np.clip(tpr_p84, 0, 1),
                                    color='#77dd77', alpha=0.3,
                                    label=f'68% confidence interval (runs={n})')

                # Plot diagonal reference line (random classifier)
                ax.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=22)
                ax.set_ylabel('True Positive Rate', fontsize=22)
                ax.legend(loc="lower right", fontsize=18)
                
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f'{save_dir}/{galaxy_classes}_{classifier}_{version}_{largest_sz}_{lr}_{reg}_avg_roc_curve.pdf')
                plt.close(fig)
                
    print(f"Saved average ROC curve at {save_dir}/{galaxy_classes}_{classifier}_{version}_{largest_sz}_{learning_rates[0]}_{regularization_params[0]}_avg_roc_curve.pdf")


def plot_avg_std_confusion_matrix(metrics, metric_stats, merge_map=merge_map, save_dir='./classifier/figures'):
    """
    Plot average confusion matrix with standard deviations across all experiments.
    """
    
    for lr, reg in itertools.product(learning_rates, regularization_params):
        subset_conf_matrices = {}
        
        # Collect confusion matrices from all experiments
        for experiment in range(num_experiments):
            for fold in folds:
                for subset_size in dataset_sizes[fold]:
                    cs = f"{crop_size[0]}x{crop_size[1]}"
                    ds = f"{downsample_size[0]}x{downsample_size[1]}"
                    
                    # Get true and predicted labels
                    true_labels_dict = metrics.get(f"all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}", [])
                    pred_labels_dict = metrics.get(f"all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}", [])
                    
                    if not true_labels_dict or not pred_labels_dict:
                        continue
                    
                    base = f"cl{classifier}_ss{subset_size}_f{fold}_lr{lr}_reg{reg}_ls0.1_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{version}"
                    true_labels = true_labels_dict[0].get(base) if isinstance(true_labels_dict, list) else true_labels_dict.get(base)
                    pred_labels = pred_labels_dict[0].get(base) if isinstance(pred_labels_dict, list) else pred_labels_dict.get(base)
                    
                    if true_labels is None or pred_labels is None or len(true_labels) == 0 or len(pred_labels) == 0:
                        continue
                    
                    # Calculate normalized confusion matrix
                    pred_labels = np.array(pred_labels)
                    num_classes = len(galaxy_classes)
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true', labels=list(range(num_classes)))
                    
                    if cm.size == 0 or cm.shape[0] != cm.shape[1]:
                        print(f"Skipping invalid confusion matrix with shape {cm.shape}")
                        continue
                    
                    # Group by merged subset size
                    merged_key = merge_map.get(subset_size, subset_size)
                    if merged_key not in subset_conf_matrices:
                        subset_conf_matrices[merged_key] = []
                    subset_conf_matrices[merged_key].append(cm)
        
        # Create confusion matrix plots for each subset size
        for subset_size, cm_list in subset_conf_matrices.items():
            if not cm_list:
                print(f"No valid confusion matrices for subset size {subset_size}")
                continue
            
            # Calculate mean and std confusion matrices
            cms = np.array(cm_list)
            avg_cm = np.mean(cms, axis=0)
            std_cm = np.std(cms, axis=0)
            
            # Get class descriptions
            desc_by_tag = {cls['tag']: cls['description'] for cls in classes}
            present_descriptions = [desc_by_tag[tag] for tag in galaxy_classes]
            
            # Create annotation array with mean ± std
            ann = np.empty(avg_cm.shape, dtype=object)
            for i in range(avg_cm.shape[0]):
                for j in range(avg_cm.shape[1]):
                    ann[i, j] = f"{avg_cm[i, j]:.2f}\n±{std_cm[i, j]:.2f}"
            
            # Calculate overall accuracy statistics
            values = metric_stats.get('accuracy', [])
            mean_value = np.mean(values) if values else 0.0
            std_dev = np.std(values) if values else 0.0
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                avg_cm,
                annot=ann,
                fmt="",
                cmap=cmap_green,
                xticklabels=present_descriptions,
                yticklabels=present_descriptions,
                annot_kws={"fontsize": 40},
                ax=ax
            )
            
            # Increase tick label font size
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=40, rotation=0, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=40, rotation=90)
            ax.set_xlabel("Predicted Label", fontsize=40)
            ax.set_ylabel("True Label", fontsize=40)
            ax.set_title(f"Average Accuracy: {mean_value:.2f} ± {std_dev:.2f}", fontsize=40)
            
            # Adjust colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=40)

            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{galaxy_classes}_{classifier}_{version}_{largest_sz}_{lr}_{reg}_avg_confusion_matrix.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved confusion matrix to {save_path}")


def plot_training_history(history, base, experiment, save_dir='./classifier/test'):
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
    save_path = f"{save_dir}/{base}_exp{experiment}_training_curves.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


###############################################recall
############ PLOTS AFTER ALL FOLDS ############
###############################################

class_descriptions = [cls['description'] for cls in classes if cls['tag'] in galaxy_classes]

# ---------------------- Rankings & summary ----------------------
# Summarize for the last (largest) subset in the selected fold(s)
metrics_last = defaultdict(list)

for fold in folds:
    if fold not in dataset_sizes:
        continue
    subset = max(dataset_sizes[fold])
    for exp, lr, reg in itertools.product(range(num_experiments), learning_rates, regularization_params):
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            k = f"{metric}_{subset}_{fold}_{exp}_{lr}_{reg}"
            if k in tot_metrics and tot_metrics[k]:
                metrics_last[metric].extend(tot_metrics[k])

# Calculate and print mean ± std for each metric
print("\n" + "="*60)
print("OVERALL PERFORMANCE SUMMARY")
print("="*60)
for metric in ["accuracy", "precision", "recall", "f1_score"]:
    vals = np.array(metrics_last.get(metric, []), dtype=float)
    if vals.size:
        print(f"{metric.capitalize()}: Mean = {vals.mean():.4f}, Std = {vals.std():.4f}")
print("="*60 + "\n")

# ---------------------- Training Times Summary ----------------------
print("\nTraining Times (aggregated):")
all_subset_sizes = {s for fs in dataset_sizes.values() for s in fs}
merged_times = defaultdict(list)

# Pull training-times dicts from metrics
for k, v in metrics.items():
    if "training_times" not in k or not v:
        continue
    tt = v[0] if isinstance(v, list) else v
    if not isinstance(tt, dict):
        continue

    # Navigate the nested dictionary structure to extract times
    for k1, v1 in tt.items():
        if not isinstance(v1, dict):
            continue

        # Check if this is Layout A: tt[subset_size][fold] -> list
        if isinstance(k1, (int, np.integer)) and k1 in all_subset_sizes:
            subset_size = int(k1)
            cat = merge_map.get(subset_size, str(subset_size))
            for times in v1.values():
                if isinstance(times, (list, tuple, np.ndarray)):
                    merged_times[cat].extend(list(times))
        else:
            # Layout B: tt[fold][subset_size] -> list
            for sub_k, times in v1.items():
                if isinstance(sub_k, (int, np.integer)) and sub_k in all_subset_sizes:
                    cat = merge_map.get(sub_k, str(sub_k))
                    if isinstance(times, (list, tuple, np.ndarray)):
                        merged_times[cat].extend(list(times))

if not merged_times:
    print("No training times recorded.")
else:
    for cat, times in sorted(merged_times.items(), key=lambda x: int(x[0])):
        times = np.asarray(times, dtype=float)
        print(f"Dataset size {cat}: {times.mean():.2f} ± {times.std():.2f} seconds (n={len(times)})")

# ——— Generate plots ———
print("\nGenerating plots...")
# Plot training history for each experiment (if history data is available)
for fold in folds:
    if fold not in dataset_sizes:
        continue
    for subset_size in dataset_sizes[fold]:
        for experiment in range(num_experiments):
            cs = f"{crop_size[0]}x{crop_size[1]}"
            ds = f"{downsample_size[0]}x{downsample_size[1]}"
            base = f"cl{classifier}_ss{subset_size}_f{fold}_lr{learning_rates[0]}_reg{regularization_params[0]}_ls0.1_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{version}"
            
            # Check if we have history data for this configuration
            history_key = f"history_{subset_size}_{fold}_{experiment}_{learning_rates[0]}_{regularization_params[0]}"
            if history_key in metrics and metrics[history_key]:
                history_data = metrics[history_key][0] if isinstance(metrics[history_key], list) else metrics[history_key]
                if isinstance(history_data, dict):
                    plot_training_history(history_data, base, experiment)
robust_metric_histograms(metrics)
plot_avg_roc_curves(metrics, merge_map=merge_map)
plot_avg_std_confusion_matrix(metrics, metric_stats=metrics_last, merge_map=merge_map)
plot_cluster_metrics(all_cluster_metrics)

print("\n" + "="*60)
print("EVALUATION SCRIPT FINISHED SUCCESSFULLY!")
print("="*60)
print(f"\nPlots and summaries saved to: ./classifier/")
print(f"- Histograms: ./classifier/figures/")
print(f"- ROC curves: ./classifier/figures/")
print(f"- Confusion matrices: ./classifier/figures/")
print(f"- Summary CSV: ./classifier/figures/{galaxy_classes}_{classifier}_robust_summary.csv")


###############################################
########## LOAD TEST DATA & MODEL #############
###############################################

if GENERATE_ATTENTION_MAPS:
    print("\n" + "="*60)
    print("GENERATING ATTENTION VISUALIZATIONS")
    print("="*60)
    
    # Load test data
    print("Loading test data...")
    _out = load_galaxies(
        galaxy_classes=galaxy_classes,
        versions=[version],
        fold=0,  # Use fold 0 for test data
        crop_size=crop_size,
        downsample_size=downsample_size,
        sample_size=1000000,
        REMOVEOUTLIERS=FILTERED,
        BALANCE=False,
        STRETCH=True,
        percentile_lo=percentile_lo,
        percentile_hi=percentile_hi,
        AUGMENT=False,
        NORMALISE=True,
        NORMALISETOPM=False,
        USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
        GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
        PRINTFILENAMES=True,  # CHANGED: Get filenames!
        PREFER_PROCESSED=True,
        USE_CACHE=True,
        DEBUG=False,
        train=False
    )

    # Now capture filenames from the 6-value return
    if len(_out) == 6:
        _, _, test_images, test_labels, test_train_fns, test_eval_fns = _out
        # Combine and get unique source names
        source_names = sorted(set(test_train_fns + test_eval_fns))
        print(f"Loaded {len(source_names)} source names from load_galaxies")
        if source_names:
            print(f"  Sample names: {source_names[:3]}")
    elif len(_out) == 4:
        _, _, test_images, test_labels = _out
        # Fallback: generate placeholder names
        source_names = [f"Test_{i}" for i in range(len(test_labels))]
        print("Warning: PRINTFILENAMES=False, using placeholder names")
    else:
        raise ValueError(f"load_galaxies returned {len(_out)} values, expected 4 or 6")
    
    # Prepare test data based on classifier type
    if classifier in ['ScatterNet', 'DualSSN']:
        from kymatio.torch import Scattering2D
        from utils.calc_tools import compute_scattering_coeffs
        
        test_images_folded = fold_T_axis(test_images)
        
        # Compute scattering coefficients
        scattering = Scattering2D(J=J, L=L, shape=downsample_size, max_order=order)
        test_scat = compute_scattering_coeffs(test_images_folded, scattering, batch_size=128, device="cpu")
        
        if test_scat.dim() == 5:
            test_scat = test_scat.flatten(start_dim=1, end_dim=2)
        
        if classifier == 'ScatterNet':
            mock_test = torch.zeros_like(test_images_folded)
            test_dataset = TensorDataset(mock_test, test_scat, test_labels)
        else:  # DualSSN
            test_dataset = TensorDataset(test_images_folded, test_scat, test_labels)
    else:  # CNN or DualCSN
        if test_images.dim() == 5:
            test_images = fold_T_axis(test_images)
        mock_scat = torch.zeros_like(test_images)
        test_dataset = TensorDataset(test_images, mock_scat, test_labels)
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                            num_workers=0, collate_fn=custom_collate, drop_last=False)
    
    # Load the best model
    print("Loading trained model...")
    fold_to_use = folds[0]
    subset_size_to_use = dataset_sizes[fold_to_use][0]
    cs = f"{crop_size[0]}x{crop_size[1]}"
    ds = f"{downsample_size[0]}x{downsample_size[1]}"
    
    model_path = f"./classifier/trained_models/cl{classifier}_ss{round_to_1(subset_size_to_use)}_f{fold_to_use}_lr{learning_rates[0]}_reg{regularization_params[0]}_ls0.1_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{version}_model.pth"
    
    # Initialize model architecture
    img_shape = tuple(test_images.shape[1:]) if test_images.dim() == 4 else tuple(test_images.shape[2:])
    num_classes = len(galaxy_classes)
    
    if classifier == "CNN":
        model = CNN(input_shape=img_shape, num_classes=num_classes).to(device)
    elif classifier == "ScatterNet":
        scatdim = test_scat.shape[1:]
        model = ScatterNet(input_dim=int(np.prod(scatdim)), num_classes=num_classes).to(device)
    elif classifier == "DualCSN":
        model = DualCNNSqueezeNet(input_shape=img_shape, num_classes=num_classes).to(device)
    elif classifier == "DualSSN":
        scatdim = test_scat.shape[1:]
        model = DualScatterSqueezeNet(img_shape=img_shape, scat_shape=scatdim, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    # Load trained weights
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            GENERATE_ATTENTION_MAPS = False
    else:
        print(f"Model file not found: {model_path}")
        GENERATE_ATTENTION_MAPS = False
    
    # Generate attention visualizations
    if GENERATE_ATTENTION_MAPS:
        generate_attention_visualizations(
            model=model,
            test_loader=test_loader,
            galaxy_classes=galaxy_classes,
            source_names=source_names,
            save_dir='./classifier',
            methods=ATTENTION_METHODS
        )

###############################################
########### EXISTING PLOTTING CODE ############
###############################################

# [Keep all your existing plotting functions here - they're already good]
# Just add them back from your original script 4.2

print("\n" + "="*60)
print("EVALUATION SCRIPT FINISHED SUCCESSFULLY!")
print("="*60)
if GENERATE_ATTENTION_MAPS:
    print(f"\nAttention maps saved to: ./classifier/attention_maps/")