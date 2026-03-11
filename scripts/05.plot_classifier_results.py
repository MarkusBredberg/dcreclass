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
from utils.classifiers import CNN, ImageCNN, ScatterNet, DualCNNSqueezeNet, DualScatterSqueezeNet
from utils.calc_tools import fold_T_axis, custom_collate

_SCRATCH     = "/users/mbredber/scratch"
MODELS_DIR   = os.path.join(_SCRATCH, 'data', 'models')
METRICS_DIR  = os.path.join(_SCRATCH, 'data', 'metrics')
FIGURES_DIR  = os.path.join(_SCRATCH, 'figures', 'classifying')
ATTN_DIR     = os.path.join(_SCRATCH, 'figures', 'classifying', 'attention_maps')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'filtered'), exist_ok=True)
os.makedirs(ATTN_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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
              "DualSSN",     # 3.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              "ImageCNN",    # 4.Single image-encoder branch from DualCSN/DualSSN
              ][3]
crop_size = (512, 512)
downsample_size = (128, 128)
version     = 'T25kpc'
crop_mode   = 'beam_crop'   # 'beam_crop' | 'fov_crop' | 'pixel_crop'
blur_method = 'circular'    # 'circular'  | 'circular_no_sub' | 'cheat'
J, L, order = 2, 12, 2

# NEW: Attention visualization settings
ATTENTION_METHODS = ['saliency', 'gradcam', 'integrated_gradients']  # Methods to use

# Define colormap for visualization
cmap_green = LinearSegmentedColormap.from_list( 
    'white_to_green',
    ['white', '#006400']
)

###############################################
######### SETTING THE RIGHT PARAMETERS ########
###############################################

directory = os.path.join(MODELS_DIR, 'filtered') + os.sep if FILTERED else MODELS_DIR + os.sep

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
            metrics_read_path = os.path.join(METRICS_DIR, f"{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{version}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_{GLOBAL_NORM_MODE}_metrics_data.pkl")
        else:
            metrics_read_path = os.path.join(METRICS_DIR, f"{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{version}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_metrics_data.pkl")
        
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
print(f"\nPlots and summaries saved to: {FIGURES_DIR}")
print(f"- Histograms: {FIGURES_DIR}")
print(f"- ROC curves: {FIGURES_DIR}")
print(f"- Confusion matrices: {FIGURES_DIR}")
print(f"- Summary CSV: {os.path.join(FIGURES_DIR, f'{galaxy_classes}_{classifier}_robust_summary.csv')}")


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
    
    model_path = os.path.join(MODELS_DIR, f"cl{classifier}_ss{round_to_1(subset_size_to_use)}_f{fold_to_use}_lr{learning_rates[0]}_reg{regularization_params[0]}_ls0.1_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{version}_model.pth")
    
    # Initialize model architecture
    img_shape = tuple(test_images.shape[1:]) if test_images.dim() == 4 else tuple(test_images.shape[2:])
    num_classes = len(galaxy_classes)
    
    if classifier == "CNN":
        model = CNN(input_shape=img_shape, num_classes=num_classes).to(device)
    elif classifier == "ImageCNN":
        model = ImageCNN(input_shape=img_shape, num_classes=num_classes).to(device)
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
            save_dir=ATTN_DIR,
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
    print(f"\nAttention maps saved to: {ATTN_DIR}")