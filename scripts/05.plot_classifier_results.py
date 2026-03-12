import numpy as np, pickle, torch, os, itertools, argparse
from sklearn.metrics import roc_curve, auc
from dcreclass.data import get_classes, load_galaxies
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import TensorDataset, DataLoader

# Import your classifiers
from dcreclass.models import CNN, ImageCNN, ScatterNet, DualCNNSqueezeNet, DualScatterSqueezeNet
from dcreclass.utils import fold_T_axis, custom_collate, compute_scattering_coeffs, round_to_1
from dcreclass.utils.calc_tools import initialize_metrics, update_metrics, recalculate_metrics_with_correct_positive_class
from dcreclass.utils.plotting import (robust_metric_histograms, plot_avg_roc_curves,
                                      plot_avg_std_confusion_matrix, plot_cluster_metrics,
                                      plot_training_history, generate_attention_visualizations)

def _parse_args():
    p = argparse.ArgumentParser(description="Plot classifier results")
    p.add_argument('--classifier',    default='ImageCNN',
                   choices=['CNN','ScatterNet','DualCSN','DualSSN','ImageCNN'])
    p.add_argument('--version',       default='RAW',
                   help="Single image version, e.g. RAW or T25kpc")
    p.add_argument('--crop-mode',     default='beam_crop',
                   choices=['beam_crop','beam_crop_no_sub','fov_crop','cheat_crop'])
    p.add_argument('--blur-method',   default='circular',
                   choices=['circular','circular_no_sub','cheat'])
    p.add_argument('--lr',            type=float, nargs='+', default=[5e-5])
    p.add_argument('--reg',           type=float, nargs='+', default=[1e-1])
    p.add_argument('--folds',         type=int, nargs='+', default=list(range(10)))
    p.add_argument('--num-experiments', type=int, default=3)
    p.add_argument('--percentile-lo', type=int, default=30)
    p.add_argument('--percentile-hi', type=int, default=99)
    p.add_argument('--run-dir',       default=None)
    p.add_argument('--data-run-dir',  default=None)
    p.add_argument('--no-attention',  dest='generate_attention', action='store_false')
    p.set_defaults(generate_attention=True)
    return p.parse_args()

args = _parse_args()
classifier            = args.classifier
version               = args.version
crop_mode             = args.crop_mode
blur_method           = args.blur_method
learning_rates        = args.lr
regularization_params = args.reg
folds                 = args.folds
num_experiments       = args.num_experiments
percentile_lo         = args.percentile_lo
percentile_hi         = args.percentile_hi
GENERATE_ATTENTION_MAPS = args.generate_attention

OUTDIR_BASE = "/users/mbredber/p2_DCRECLASS/outputs/scratch/"
_run_dir      = args.run_dir
_data_run_dir = args.data_run_dir or args.run_dir
if _run_dir and _data_run_dir:
    FIGURES_DIR = os.path.join(_run_dir, 'figures/classifying')
    MODELS_DIR  = os.path.join(_data_run_dir, 'data', 'models')
    METRICS_DIR = os.path.join(_data_run_dir, 'data', 'metrics')
else:
    _SCRATCH    = "/users/mbredber/scratch"
    FIGURES_DIR = os.path.join(_SCRATCH, 'figures', 'classifying')
    MODELS_DIR  = os.path.join(_SCRATCH, 'data', 'models')
    METRICS_DIR = os.path.join(_SCRATCH, 'data', 'metrics')

# Constants not exposed as args
galaxy_classes = [50, 51]
crop_size = (512, 512)
downsample_size = (128, 128)
J, L, order = 2, 12, 2
FILTERED = True
USE_GLOBAL_NORMALISATION = False
global_norm_mode = "percentile"
ADJUST_POSITIVE_CLASS = True

classes = get_classes()

ATTN_DIR = os.path.join(FIGURES_DIR, 'attention_maps')


os.makedirs(ATTN_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("Running enhanced evaluation script 2.0 with attention visualization")
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

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

label_smoothing = 0.1  # must match script 04 default

# Narrow METRICS_DIR and MODELS_DIR to the per-run subdirectory — mirrors script 04
_sub = (f"{classifier}_{crop_mode}_{blur_method}_{learning_rates[0]}_"
        f"{regularization_params[0]}_{percentile_lo}_{percentile_hi}_{label_smoothing}")
METRICS_DIR = os.path.join(METRICS_DIR, _sub)
MODELS_DIR  = os.path.join(MODELS_DIR,  _sub)

run_version = (f"{galaxy_classes}_{classifier}_{[version]}_{crop_mode}_{blur_method}_"
               f"{percentile_lo}_{percentile_hi}_{global_norm_mode}_"
               f"{learning_rates[0]}_{regularization_params[0]}_{label_smoothing}")
save_dir = os.path.join(FIGURES_DIR, run_version)
os.makedirs(save_dir, exist_ok=True)

def _base(fold, subset_size, lr, reg):
    """Canonical key — must mirror script 04's _base()."""
    return (f"{classifier}_ver{version}_cm{crop_mode}"
            f"_lr{lr}_reg{reg}_ls{label_smoothing}"
            f"_lo{percentile_lo}_hi{percentile_hi}"
            f"_f{fold}_ss{round_to_1(subset_size)}")

def _pkl_path(fold, subset_size, experiment, lr, reg):
    return os.path.join(METRICS_DIR, f"{_base(fold, subset_size, lr, reg)}_e{experiment}.pkl")

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
        metrics_read_path = _pkl_path(fold, subset_size, experiment, lr, reg)
        
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
            
            base = _base(fold, subset_size, lr, reg)
            
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

            # Compute AUC from raw probabilities
            auc_val = float('nan')
            if y_true and y_probs:
                try:
                    y_probs_arr = np.asarray(y_probs)
                    scores = (y_probs_arr[:, 1]
                              if y_probs_arr.ndim == 2 and y_probs_arr.shape[1] > 1
                              else y_probs_arr.ravel())
                    if np.unique(np.asarray(y_true)).size >= 2:
                        fpr, tpr, _ = roc_curve(y_true, scores)
                        auc_val = auc(fpr, tpr)
                except Exception:
                    pass
            auc_key = f"auc_{subset_size}_{fold}_{experiment}_{lr}_{reg}"
            tot_metrics.setdefault(auc_key, []).append(auc_val)

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
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            k = f"{metric}_{subset}_{fold}_{exp}_{lr}_{reg}"
            if k in tot_metrics and tot_metrics[k]:
                metrics_last[metric].extend(tot_metrics[k])

# Calculate and print mean ± std for each metric
print("\n" + "="*60)
print("OVERALL PERFORMANCE SUMMARY")
print("="*60)
for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
    vals = np.array(metrics_last.get(metric, []), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        print(f"{metric.upper() if metric == 'auc' else metric.capitalize()}: Mean = {vals.mean():.4f}, Std = {vals.std():.4f}")
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
            base = _base(fold, subset_size, learning_rates[0], regularization_params[0])
            
            # Check if we have history data for this configuration
            history_key = f"history_{subset_size}_{fold}_{experiment}_{learning_rates[0]}_{regularization_params[0]}"
            if history_key in metrics and metrics[history_key]:
                history_data = metrics[history_key][0] if isinstance(metrics[history_key], list) else metrics[history_key]
                if isinstance(history_data, dict):
                    plot_training_history(history_data, base, experiment, save_dir=save_dir)
robust_metric_histograms(metrics, galaxy_classes, classifier, dataset_sizes, folds,
                         learning_rates, regularization_params, save_dir=save_dir)
plot_avg_roc_curves(metrics, classifier,
                    version=version, dataset_sizes=dataset_sizes,
                    crop_size=crop_size, downsample_size=downsample_size,
                    percentile_lo=percentile_lo, percentile_hi=percentile_hi,
                    merge_map=merge_map, folds=folds,
                    num_experiments=num_experiments,
                    learning_rates=learning_rates,
                    regularization_params=regularization_params,
                    galaxy_classes=galaxy_classes,
                    save_dir=save_dir)
plot_avg_std_confusion_matrix(metrics, metrics_last, galaxy_classes, classifier,
                              version, largest_sz, learning_rates, regularization_params,
                              folds, dataset_sizes, crop_size, downsample_size,
                              percentile_lo, percentile_hi,
                              merge_map=merge_map, num_experiments=num_experiments,
                              save_dir=save_dir)
plot_cluster_metrics(all_cluster_metrics, save_dir=save_dir)

print("\n" + "="*60)
print("EVALUATION SCRIPT FINISHED SUCCESSFULLY!")
print("="*60)
print(f"\nPlots and summaries saved to: {FIGURES_DIR}")
print(f"- Histograms: {FIGURES_DIR}")
print(f"- ROC curves: {FIGURES_DIR}")
print(f"- Confusion matrices: {FIGURES_DIR}")
print(f"- Summary CSV: {os.path.join(FIGURES_DIR, f'{galaxy_classes}_{classifier}_robust_summary.csv')}")


if GENERATE_ATTENTION_MAPS:
    # Check model exists before doing any expensive data loading
    _fold_check = folds[0]
    _ss_check   = dataset_sizes[_fold_check][0]
    _model_path_check = os.path.join(MODELS_DIR,
        f"{_base(_fold_check, _ss_check, learning_rates[0], regularization_params[0])}_model.pth")
    if not os.path.exists(_model_path_check):
        print(f"\nSkipping attention visualizations — model not found: {_model_path_check}")
        GENERATE_ATTENTION_MAPS = False

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
        global_norm_mode=global_norm_mode,
        PRINTFILENAMES=True,  # CHANGED: Get filenames!
        USE_CACHE=True,
        DEBUG=False,
        crop_mode=crop_mode,
        blur_method=blur_method,
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
    model_path = os.path.join(MODELS_DIR,
        f"{_base(fold_to_use, subset_size_to_use, learning_rates[0], regularization_params[0])}_model.pth")
    
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