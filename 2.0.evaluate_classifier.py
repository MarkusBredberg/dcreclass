import numpy as np, pickle, torch, os, itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.data_loader import get_classes
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Create output directories
os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)

print("Running evaluation script 4.2")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

###############################################
################ CONFIGURATION ################
###############################################

# IMPORTANT: These must match your 4.1 training configuration
FILTERED = True       # Evaluation with filtered data (REMOVEOUTLIERS = True)
USE_GLOBAL_NORMALISATION = False          # single on/off switch . False - image-by-image normalisation 
GLOBAL_NORM_MODE = "percentile"
ADJUST_POSITIVE_CLASS = True  # Whether to relabel classes so that DE is positive class

classes = get_classes()
galaxy_classes = [50, 51]  # Must match training: e.g., [50, 51] for your binary classification
learning_rates = [5e-5]    # Must match training lr
regularization_params = [1e-1]  # Must match training reg
num_experiments = 3   # Must match training num_experiments
percentile_lo, percentile_hi = 30, 99  # Must match training percentiles
folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Must match training folds (0-9 for 10-fold CV)
classifier = ["CNN",         # 0.Very Simple CNN
              "ScatterNet",  # 1.Scattering coefficients as input to MLP
              "DualCSN",     # 2.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              "DualSSN"      # 3.Dual input CNN with scattering coefficients as one input branch and Squeeze-and-Excitation blocks
              ][3]
crop_size = (512, 512)  # Must match training crop size
downsample_size = (128, 128)  # Must match training downsample size
version = 'RAW+RT25kpc+RT50kpc+RT100kpc'  # Must match training version

# Define colormap for visualization
cmap_green = LinearSegmentedColormap.from_list( 
    'white_to_green',
    ['white', '#006400']
)

###############################################
######### SETTING THE RIGHT PARAMETERS ########
###############################################

directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

# Define dataset sizes for each fold (this should match what 4.1 actually used)
if galaxy_classes == [50, 51]:
    # For 10-fold CV, each fold should have approximately the same size
    dataset_sizes = {fold: [3000] for fold in range(10)} # 3000 for prefer processed=True
elif galaxy_classes == [52, 53]:  # RH vs RR
    dataset_sizes = {fold: [2, 16, 168] for fold in range(10)}
else:
    print("Please specify the dataset sizes for the given galaxy classes.")
    exit(1)

largest_sz = max([sz for sizes in dataset_sizes.values() for sz in sizes])

############################################################
################# MERGE MAP GENERATION #####################
############################################################

# This creates a mapping to group similar dataset sizes together for plotting
merge_map = {}
all_sizes = {s for fs in dataset_sizes.values() for s in fs}
for size in all_sizes:
    nd = len(str(size))
    factor = 10 ** max(nd - 2, 0)
    new_rep = int(round(size / factor) * factor)
    merge_map[size] = str(new_rep)

print("merge_map =", merge_map)

all_cluster_metrics = {
    'errors': [],
    'distances': [],
    'std_devs': []
}

###############################################
############# READ IN PICKLE DUMP #############
###############################################

from math import log10, floor
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

# If the positive class was mislabeled, recalculate metrics
if ADJUST_POSITIVE_CLASS:
    def recalculate_metrics_with_correct_positive_class(y_true, y_pred, pos_label=0):
        """
        Recalculate precision, recall, and F1 with the correct positive class.
        
        Args:
            y_true: True labels (0 or 1 after relabeling)
            y_pred: Predicted labels (0 or 1 after relabeling)
            pos_label: Which label to treat as positive (0 for DE, 1 for NDE)
        
        Returns:
            accuracy, precision, recall, f1_score
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Accuracy is unaffected by which class is positive
        acc = accuracy_score(y_true, y_pred)
        
        # Recalculate with correct positive label
        precision = precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
        
        return acc, precision, recall, f1

def initialize_metrics(metrics, subset_size, fold, experiment, lr, reg):
    """Initialize metric storage dictionaries for a given experiment configuration"""
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
    """Update metrics dictionaries with values from a single experiment"""
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

# Initialize storage for aggregated metrics
tot_metrics = {}

print("Loading metrics from pickle files...")
loaded_count = 0
failed_count = 0

# Iterate over all experiment configurations to load saved metrics
for lr, reg, experiment, fold in itertools.product(
    learning_rates, regularization_params, range(num_experiments), folds
):
    for subset_size in dataset_sizes[fold]:
        # Construct the file path
        cs = f"{crop_size[0]}x{crop_size[1]}"
        ds = f"{downsample_size[0]}x{downsample_size[1]}"
        if USE_GLOBAL_NORMALISATION:
            metrics_read_path = f"./classifier/4.1.runs/global_norm/{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{version}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_{GLOBAL_NORM_MODE}_metrics_data.pkl"
        else:
            metrics_read_path = f"./classifier/4.1.runs/{classifier}_{galaxy_classes}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{cs}_ds{ds}_ver{version}_f{fold}_ss{round_to_1(subset_size)}_e{experiment}_metrics_data.pkl"
        
        try:
            with open(metrics_read_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract loaded data
            loaded_metrics = data["metrics"]
            history = data["history"]
            all_true_labels_dict = data["all_true_labels"]
            all_pred_labels_dict = data["all_pred_labels"]
            all_pred_probs_dict = data["all_pred_probs"]
            training_times_dict = data["training_times"]
            
            # *** ADD THIS: Extract cluster metrics ***
            if data.get("cluster_error") is not None:
                all_cluster_metrics['errors'].append(data["cluster_error"])
            if data.get("cluster_distance") is not None:
                all_cluster_metrics['distances'].append(data["cluster_distance"])
            if data.get("cluster_std_dev") is not None:
                all_cluster_metrics['std_devs'].append(data["cluster_std_dev"])
            
            
            # Initialize storage for this experiment configuration
            initialize_metrics(tot_metrics, subset_size, fold, experiment, lr, reg)
            
            # Build the key (should match what's in the dictionaries)
            base = f"cl{classifier}_ss{subset_size}_f{fold}_lr{lr}_reg{reg}_ls0.1_cs{cs}_ds{ds}_pl{percentile_lo}_ph{percentile_hi}_ver{version}"
            
            # Get metrics directly (they're already aggregated in the file)
            acc = loaded_metrics.get("accuracy", [])
            prec = loaded_metrics.get("precision", [])
            rec = loaded_metrics.get("recall", [])
            f1 = loaded_metrics.get("f1_score", [])
            
            # Get labels
            y_true = all_true_labels_dict.get(base, [])
            y_pred = all_pred_labels_dict.get(base, [])
            y_probs = all_pred_probs_dict.get(base, [])
            
            if not y_true or not y_pred:
                print(f"Warning: No labels found for {base}")
                failed_count += 1
                continue
            
            # RECALCULATE METRICS WITH CORRECT POSITIVE LABEL
            if ADJUST_POSITIVE_CLASS:
                # pos_label=0 means DE (class 50, which becomes 0 after relabeling) is positive
                acc_val, prec_val, rec_val, f1_val = recalculate_metrics_with_correct_positive_class(
                    y_true, y_pred, pos_label=0
                )
            
            else:
                # Use the first values (since they're already calculated)
                acc_val = acc[0] if isinstance(acc, list) and acc else 0.0
                prec_val = prec[0] if isinstance(prec, list) and prec else 0.0
                rec_val = rec[0] if isinstance(rec, list) and rec else 0.0
                f1_val = f1[0] if isinstance(f1, list) and f1 else 0.0
            
            # Update the aggregated metrics
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

