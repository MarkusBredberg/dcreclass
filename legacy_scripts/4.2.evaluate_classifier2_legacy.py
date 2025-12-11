import numpy as np, pickle, torch, os, itertools
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from utils.data_loader import get_classes, load_galaxies, get_synthetic
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # <-- For creating custom line legend entries
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps  # For newer Matplotlib versions
import seaborn as sns
os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)

print("Running script 4.2")
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

###############################################
################ CONFIGURATION ################
###############################################

# Make the file names longer and more descriptive

FILTERED = True       # Evaluation with filtered data (REMOVEOUTLIERS = True)
TRAINONGENERATED = False # Use generated data as testdata

classes = get_classes()
galaxy_classes = [50, 51] # e.g., [10, 11, 12, 13] for FRI, FRII, Compact, Bent
learning_rates = [1e-3]
regularization_params = [1e-3]
lambda_values = [0]
num_experiments = 10
percentile_lo, percentile_hi = 30, 99
folds = [0] # Number of folds for cross-validation
generators = ['DDPM']
classifier = ["TinyCNN", "Rustige", "SCNN", "CNNSqueezeNet", "DualCNNSqueezeNet", "CloudNet", "DANN", "ScatterNet", "ScatterDual", "ScatterSqueezeNet", "Binary", "ScatterResNet"][-3]
print("Classifier:", classifier)
crop_size = (512, 512)  # Crop size for the images

#Saved metrics PKL to /capstor/scratch/cscs/mbredber/classifier/4.1.runs/[50, 51]_lr0.001_reg0.001_lo30_hi99_cs512x512_ds128x128_verRAW_f0_sz1160_f0_e1_metrics_data.pkl
#Saved metrics PKL to /capstor/scratch/cscs/mbredber/classifier/4.1.runs/[50, 51]_DDPM_lr0.001_reg0.001_lo30_hi99_cs512x512_sz1160_f0_e0_lam0_metrics_data.pkl


# -------------------------- NEW GAN CONFIGURATION --------------------------
gan_epoch = 500           # e.g., epoch number to load from
gan_gen_loss = 'MSE'       # e.g., generator loss value (as used in filename)
gan_disc_loss = 'BCE'      # e.g., discriminator loss value (as used in filename)
gan_latent_dim = 128
lr_gen = 1e-3
lr_disc = 1e-4
gan_sample_size = 100 # Number of images sampled to train the GAN
gan_adam_beta = (0.5, 0.999)
gan_weight_decay = 0.0
gan_label_smoothing = 0.9
gan_lambda_div = 0.1
gan_type = ['Simple', 'Advanced'][0]
gan_data_version = 'full' if FILTERED else 'clean'  # 'full' or 'clean'
# -------------------------- END GAN CONFIGURATION --------------------------

# Define consistent color mapping
colors = {
    'Real': '#0072B2',       # blue
    'STMLP': '#E69F00',      # orange
    'lavgSTMLP': '#CC79A7',   # reddish purple
    'ldiffSTMLP': '#F0E442',  # yellow
    'Dual': '#009E73',       # bluish green
    'CNN': '#56B4E9',        # sky blue
    'GAN': '#D55E00'         # vermillion
}

# Define colormaps for visualization
cmap_green = LinearSegmentedColormap.from_list( 
    'white_to_green',
    ['white', '#006400']
)

###############################################
######### SETTING THE RIGHT PARAMETERS ########
###############################################

directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

if galaxy_classes == [31, 32, 33, 34, 35, 36]:
    dataset_sizes = [[1088, 10883, 54416, 108832]]
    num_experiments = 1
    FILTERED = False
elif galaxy_classes == [40, 41]:
    dataset_sizes = [[140, 704, 1408], [140, 704, 1408], [140, 704, 1408], [140, 704, 1408], [140, 704, 1408], [140, 704, 1408]] # For 1024, 512
elif galaxy_classes == [50, 51]:
    if TRAINONGENERATED:
        dataset_sizes = [[1600, 8000, 16000], [1600, 8000, 16000], [1600, 8000, 16000], [1600, 8000, 16000], [1600, 8000, 16000],  [1600, 8000, 16000]]
    else:
        #dataset_sizes = [[12, 127, 1272], [12, 127, 1272], [12, 127, 1272], [12, 127, 1272], [12, 127, 1272], [12, 127, 1272]]
        #dataset_sizes = [[12, 124, 1248], [12, 124, 1248], [12, 124, 1248], [12, 124, 1248], [12, 124, 1248], [12, 124, 1248]]
        #dataset_sizes = [[1264], [1264], [1264], [1264], [1264], [1264]]
        dataset_sizes = [[1160]]
        #dataset_sizes = [[12, 126, 1264], [12, 126, 1264], [12, 126, 1264], [12, 126, 1264], [12, 126, 1264], [12, 126, 1264]]
        #dataset_sizes = [[14, 144, 1440], [14, 144, 1440], [14, 144, 1440], [14, 144, 1440], [14, 144, 1440], [14, 144, 1440]] 
        #dataset_sizes = [[1440], [1440], [1440], [1440], [1440], [1440]]
elif galaxy_classes == [52, 53]: # RH vs RR
    dataset_sizes = [[2, 16, 168], [2, 16, 168], [2, 16, 168], [2, 16, 168], [2, 16, 168], [2, 16, 168]]
elif galaxy_classes == [11, 12]:
    dataset_sizes = [[885, 4428, 8856], [885, 4428, 8856], [885, 4428, 8856], [885, 4428, 8856], [885, 4428, 8856], [885, 4428, 8856]]
elif galaxy_classes == [10, 11, 12, 13]:
    if FILTERED:
        if max(folds) == 5:
            if TRAINONGENERATED:
                dataset_sizes = [[1391, 6956, 13912], [1391, 6956, 13912], [1391, 6956, 13912], [1391, 6956, 13912], [1391, 6956, 13912], [1391, 6956, 13912]]
            else:
                #dataset_sizes = [[1393, 6968, 13936], [1393, 6968, 13936], [1393, 6968, 13936], [1393, 6968, 13936], [1393, 6968, 13936], [1393, 6968, 13936]]
                dataset_sizes = [[1393, 13936], [1393, 13936], [1393, 13936], [1393, 13936], [1393, 13936], [1393, 13936]]
                #dataset_sizes = [[5574, 55744], [5574, 55744], [5574, 55744], [5574, 55744], [5574, 55744], [5574, 55744]] # When translations are used
            #dataset_sizes = [[123, 1236, 12368], [123, 1233, 12336], [123, 1236, 12368], [123, 1236, 12368], [123, 1236, 12368]] 
            #dataset_sizes = [[200], [200], [200], [200], [200], [200]]
        else:
            print("Using default dataset sizes for 5 folds.")
            dataset_sizes = [[1236, 6184, 12368], [1233, 6168, 12336], [1236, 6184, 12368], [1236, 6184, 12368], [1236, 6184, 12368], [1236, 6184, 12368]]
            #dataset_sizes = [[140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064]] # Used for faster trouble shooting
            #dataset_sizes = [[139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936]]
            #dataset_sizes = [[1389, 6948, 13896], [1389, 6948, 13896], [1389, 6948, 13896], [1389, 6948, 13896], [1389, 6948, 13896], [1389, 6948, 13896]]
            #dataset_sizes = [[281, 2812, 14064, 28128], [281, 2812, 14064, 28128], [281, 2812, 14064, 28128], 
    else:
        #dataset_sizes = [[249, 2492, 12464, 24928], [249, 2492, 12464, 24928], [249, 2492, 12464, 24928], [249, 2492, 12464, 24928], 250, 2505, 12528, 25056]]    
        dataset_sizes = [[1236, 6184, 12368], [1236, 6184, 12368], [1236, 6184, 12368], [1236, 6184, 12368], [1236, 6184, 12368], [1236, 6184, 12368]]
elif galaxy_classes == [10, 11]:
   dataset_sizes = [[9656], [9656], [9656], [9656], [9656], [9656]]
else:
    print("Please specify the dataset sizes for the given galaxy classes.")

if 'GAN' in generators:
    latent_dim = 128 

if TRAINONGENERATED:
    lambda_values = [8]  # To identify and distinguish TESTONGENERATED from other runs

largest_sz = dataset_sizes[folds[-1]][-1]
    

############################################################
################# MERGE MAP GENERATION #####################
############################################################

merge_map = {}
for sizes in dataset_sizes:
    for size in sizes:
        nd = len(str(size))
        factor = 10 ** max(nd - 2, 0)
        new_rep = int(round(size / factor) * factor)
        merge_map[size] = str(new_rep)

print("merge_map =", merge_map)

###############################################
############# READ IN PICKLE DUMP #############
###############################################

def initialize_metrics(metrics, generator, subset_size, fold, experiment, lr, reg, lambda_generate):
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"{generator}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []

def update_metrics(metrics, generator, subset_size, fold, experiment, lr, reg,
                   accuracy, precision, recall, f1, lambda_generate,
                   history_val, all_true_labels, all_pred_labels, training_times, all_pred_probs):
    subset_size_str = str(subset_size)
    metrics[f"{generator}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(accuracy)
    metrics[f"{generator}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(precision)
    metrics[f"{generator}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(recall)
    metrics[f"{generator}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(f1)
    metrics[f"{generator}_history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(history_val)
    metrics[f"{generator}_all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_true_labels)
    metrics[f"{generator}_all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_pred_labels)
    metrics[f"{generator}_training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(training_times)
    metrics[f"{generator}_all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_pred_probs)

tot_metrics = {}
valid_lambda_values = []

print("Loading metrics from pickle files...")
for lambda_generate in lambda_values:
    lambda_loaded = False
    for lr, reg, experiment, generator, fold in itertools.product(
        learning_rates, regularization_params, range(num_experiments), generators, folds
    ):
        for subset_size in dataset_sizes[fold]:
            #metrics_read_path = (f"{directory}{classifier}_{galaxy_classes}_{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl")
            metrics_read_path = f"./classifier/4.1.runs/{galaxy_classes}_{generator}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_"\
                f"cs{crop_size[0]}x{crop_size[1]}_sz{subset_size}_f{fold}_e{experiment}_lam{lambda_generate}_metrics_data.pkl"
            try:
                with open(metrics_read_path, 'rb') as f:
                    data = pickle.load(f)
                clf_models    = data["models"]
                history = data["history"]
                loaded_metrics =  data["metrics"]
                metric_colors =   data["metric_colors"]
                all_true_labels = data["all_true_labels"]
                all_pred_labels = data["all_pred_labels"]
                training_times =  data["training_times"]
                all_pred_probs =  data["all_pred_probs"]
                initialize_metrics(tot_metrics, generator, subset_size, fold, experiment, lr, reg, lambda_generate)
                eval_key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                y_true = all_true_labels.get(eval_key, [])
                y_pred = all_pred_labels.get(eval_key, [])
                if not y_true or not y_pred:
                    continue  # nothing to aggregate for this combo

                acc  = accuracy_score(y_true, y_pred)
                if len(galaxy_classes) == 2:
                    # for binary classification, pos_label=1 means the second class in sorted order
                    positive_class = 1 if galaxy_classes[1] > galaxy_classes[0] else 0
                    prec = precision_score(y_true, y_pred, pos_label=positive_class, average='binary', zero_division=0)
                    rec  = recall_score(y_true, y_pred, pos_label=positive_class, average='binary', zero_division=0)
                    f1   = f1_score(y_true, y_pred, pos_label=positive_class, average='binary', zero_division=0)
                else:
                    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)


                update_metrics(
                    tot_metrics, generator, subset_size, fold, experiment, lr, reg,
                    acc, prec, rec, f1, lambda_generate,
                    history,
                    all_true_labels,
                    all_pred_labels,
                    training_times,
                    all_pred_probs
                )
                lambda_loaded = True
            except FileNotFoundError:
                print(f"Metrics file not found at {metrics_read_path}. Ignoring this λ. Try to check dataset sizes for each fold.")
                continue

    if lambda_loaded:
        valid_lambda_values.append(lambda_generate)
    else:
        print(f"No metrics at all for λ={lambda_generate}, skipping.")
print("Done loading. Valid λ:", valid_lambda_values)
metrics = tot_metrics
lambda_values = valid_lambda_values

###############################################
########### PLOTTING FUNCTIONS ################
###############################################

def robust_metric_histograms(
    metrics,
    galaxy_classes=galaxy_classes,
    classifier=classifier,
    dataset_sizes=dataset_sizes,
    folds=folds,
    save_dir="./classifier"
):
    """
    For each metric series (accuracy/precision/recall/f1_score), make a histogram
    with 16/50/84 percentile markers and write a compact percentile summary CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    wanted = {"accuracy", "precision", "recall", "f1_score"}

    grouped = defaultdict(list)

    for key, values in metrics.items():
        # "{generator}_{metric}_{subset}_{fold}_{experiment}_{lr}_{reg}_{lambda}"
        parts = key.split("_")
        if len(parts) < 8:
            continue
        metric_name = parts[1]
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

        # drop the 'experiment' token so we aggregate across experiments
        # group key keeps generator, metric, subset, fold, lr, reg, lambda
        group_key = "_".join([parts[0], parts[1], parts[2], parts[3], parts[5], parts[6], parts[7]])
        grouped[group_key].extend(vals.tolist())

    for group_key, all_vals in grouped.items():
        vals = np.asarray(all_vals, dtype=float)
        if vals.size == 0 or not np.isfinite(vals).any():
            continue

        p16, p84 = np.percentile(vals, [16, 84])
        mean = np.mean(vals)
        sigma68 = 0.5 * (p84 - p16)

        # parse for filenames/titles
        gparts = group_key.split("_")  # [generator, metric, subset, fold, lr, reg, lambda]
        generator, metric_name = gparts[0], gparts[1]

        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            eps = max(1e-6, abs(vmin) * 1e-3)
            vmin, vmax = vmin - eps, vmax + eps
        edges = np.linspace(vmin, vmax, 21)

        plt.figure(figsize=(5, 3))
        plt.hist(vals, bins=edges, color='#77dd77', edgecolor="none", alpha=0.8)
        for x, style in [(p16, ":"), (mean, "-"), (p84, ":")]:
            plt.axvline(x, linestyle=style, color='black')

            # Add the actual x-values in text at the lower part (but not fully bottom to avoid overlap)
            plt.text(x, plt.ylim()[0] + 0.2 * (plt.ylim()[1] - plt.ylim()[0]), f"{x:.3f}", rotation=90, verticalalignment='center', horizontalalignment='right' if style == ':' else 'left')

        plt.xlim(vmin, vmax)
        plt.xlabel(metric_name.capitalize())
        plt.ylabel("Count")
        plt.tight_layout()

        # Add legend that explains the lines are 68% intervals and mean
        p16_line = mlines.Line2D([], [], color='black', linestyle=':', label='16th/84th percentiles')
        mean_line = mlines.Line2D([], [], color='black', linestyle='-', label='Mean')
        plt.legend(handles=[p16_line, mean_line], loc='upper left', fontsize=10)

        save_path_hist = f"{save_dir}/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_{metric_name}_histogram.pdf"
        plt.savefig(save_path_hist, dpi=150)
        plt.close()

        rows.append({
            "setting": group_key,
            "metric": metric_name,
            "n": int(vals.size),
            "p16": float(p16),
            "mean": float(mean),
            "p84": float(p84),
            "sigma68": float(sigma68)
        })

    # write a compact CSV summary
    if rows:
        import csv
        csv_path = f"{save_dir}/{galaxy_classes}_{classifier}_robust_summary.csv"
        fieldnames = list({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"Wrote robust summary to {csv_path}")



def extract_feats(loader):
        feats = []
        # pick the penultimate feature‐vector layer automatically
        if hasattr(model, 'feature_extractor'):
            penult = model.feature_extractor
        elif hasattr(model, 'fc1'):
            penult = model.fc1
        elif hasattr(model, 'fc2'):
            penult = model.fc2
        elif hasattr(model, 'classifier'):
            penult = model.classifier[1]
        elif hasattr(model, 'fc'):
            penult = model.fc
        else:
            raise RuntimeError(f"Can't find a penultimate layer to hook in {model.__class__.__name__}")
        handle = penult.register_forward_hook(lambda m, inp, out: feats.append(out.detach().cpu()))

        for x, _, y in loader:
            _ = model(x.to(device))
        handle.remove()
        return torch.cat(feats,0).numpy()

def visualize_feature_tsne(model, real_loader, gen_loader, wrappername, perplexity=30, n_iter=2000, device='cpu', save_path="./classifier/tsne_by_class.pdf"):
    model = model.to(device).eval()
    model_name = model.__class__.__name__
    print(f"Model name: {model_name}")

    try:
        real_feats = extract_feats(real_loader)
        real_labels = real_loader.dataset.tensors[2].numpy()   # assuming your DataLoader stores labels

        gen_feats  = extract_feats(gen_loader)
        gen_labels = gen_loader.dataset.tensors[2].numpy()     # same here
    except Exception as e:
        # Continue the code but skip this visualization if there's an error
        print(f"Error extracting features: {e}")
        return
    
    # stack & fit TSNE
    y   = np.concatenate([real_labels, gen_labels])
    X     = np.vstack([real_feats, gen_feats])
    X_emb = TSNE(perplexity=perplexity, max_iter=n_iter, random_state=42).fit_transform(X)

    # split back into real vs gen
    n_real = real_feats.shape[0]
    real_2d, gen_2d = X_emb[:n_real], X_emb[n_real:]

    # 1) scatter as before
    plt.figure(figsize=(6,6))
    plt.scatter(
        real_2d[:,0], real_2d[:,1],
        facecolors='none', edgecolors='tab:blue', marker='o',
        s=40, alpha=0.6, label='Real'
    )
    plt.scatter(
        gen_2d[:,0], gen_2d[:,1],
        facecolors='none', edgecolors='tab:orange', marker='o',
        s=50, alpha=0.6, label='Generated'
    )
    plt.legend()
    plt.title(f't-SNE of penultimate-layer features ({wrappername})')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # 2) now heatmap density
    plot_tsne_density(
        real_2d, gen_2d,
        bins=100,
        save_path=save_path.replace(".pdf","_density.pdf")
    )

    print(f"Saved t-SNE plots to {save_path} and density to {save_path.replace('.pdf','_density.pdf')}")

def visualize_tsne_by_class(model, real_loader, gen_loader, wrappername, device='cpu', save_path="./classifier/tsne_by_class.pdf"):
    model = model.to(device).eval()
    
    try:
        real_feats = extract_feats(real_loader)
        real_labels = real_loader.dataset.tensors[2].numpy()   # assuming your DataLoader stores labels

        gen_feats  = extract_feats(gen_loader)
        gen_labels = gen_loader.dataset.tensors[2].numpy()     # same here
    except Exception as e:
        # Continue the code but skip this visualization if there's an error
        print(f"Error extracting features: {e}")
        return

    # combine
    X   = np.vstack([real_feats, gen_feats])
    y   = np.concatenate([real_labels, gen_labels])
    X2d = TSNE(perplexity=30, max_iter=2000, random_state=42).fit_transform(X)

    plt.figure(figsize=(6,6))
    plt.scatter(X2d[:,0], X2d[:,1], c=y, cmap='tab10', alpha=0.7)

    # define names in index‐order
    # names by index
    class_names = ['FRI', 'FRII', 'Compact', 'Bent']
    n_real = real_feats.shape[0]
    is_real = np.concatenate([np.ones(n_real, bool), np.zeros(len(gen_feats), bool)])
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(vmin=0, vmax=len(class_names)-1)

    plt.figure(figsize=(6,6))
    for cls in np.unique(y):
        for flag, marker in [(True,'o'), (False,'x')]:
            idx = np.where((y==cls) & (is_real==flag))
            plt.scatter(
                X2d[idx,0], X2d[idx,1],
                c=[cmap(norm(cls))],
                marker=marker,
                label=f"{class_names[int(cls)]} {'Real' if flag else 'Fake'}",
                alpha=0.6
            )
    plt.legend(ncol=2, title="Class + Source")
    plt.title(f"t-SNE of penultimate-layer features ({wrappername}), colored by class")
    plt.savefig(save_path)


    
def plot_tsne_density(real_coords, gen_coords, bins=100, save_path="./classifier/tsne_density.pdf"):
    """
    real_coords / gen_coords: np.array of shape (N,2) from your TSNE embedding
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    h1 = ax1.hist2d(real_coords[:,0], real_coords[:,1],
                    bins=bins, cmap='viridis')
    ax1.set_title("Real t-SNE density")
    fig.colorbar(h1[3], ax=ax1)
    
    h2 = ax2.hist2d(gen_coords[:,0], gen_coords[:,1],
                    bins=bins, cmap='viridis')
    ax2.set_title("Generated t-SNE density")
    fig.colorbar(h2[3], ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    

def plot_overlap_image_grids(model, real_loader, gen_loader,
                             device='cpu',
                             top_k=25,
                             save_dir="./classifier"):
    """
    Finds the top_k generated images whose penultimate‐layer features
    are closest / furthest to any real feature, and plots each as a 5×5 grid.
    """
    model = model.to(device).eval()

    def extract_feats_and_imgs(loader):
        feats, imgs = [], []
        # hook on penult
        if hasattr(model, 'feature_extractor'):
            penult = model.feature_extractor
        elif hasattr(model, 'fc1'):
            penult = model.fc1
        elif hasattr(model, 'classifier'):
            penult = model.classifier[1]
        elif hasattr(model, 'fc'):
            penult = model.fc
        else:
            raise RuntimeError("No penultimate layer found")

        handle = penult.register_forward_hook(lambda m, inp, out: feats.append(out.detach().cpu()))
        for x, _, _ in loader:
            imgs.append(x.cpu())
            _ = model(x.to(device))
        handle.remove()
        X = torch.cat(imgs, dim=0).numpy()      # (N,1,H,W)
        F = torch.cat(feats, dim=0).numpy()     # (N,D)
        return F, X

    try: 
        real_feats, real_imgs = extract_feats_and_imgs(real_loader)
        gen_feats, gen_imgs = extract_feats_and_imgs(gen_loader)
    except Exception as e:
        # Continue the code but skip this visualization if there's an error
        print(f"Error extracting features or images: {e}")
        return

    # build NN on real feats
    nn = NearestNeighbors(n_neighbors=1).fit(real_feats)
    dists, _ = nn.kneighbors(gen_feats)
    dists = dists.ravel()
    order = np.argsort(dists)

    def plot_grid(imgs, title, fname):
        fig, axes = plt.subplots(5, 5, figsize=(5, 5))
        for ax, im in zip(axes.flat, imgs):
            ax.imshow(im.squeeze(), cmap='gray')
            ax.axis('off')
        fig.suptitle(title)
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{fname}")
        plt.close(fig)

    # closest
    closest_idx = order[:top_k]
    plot_grid(gen_imgs[closest_idx],
              "Top 25 Generated ➞ Real‐like",
              f"{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_gen_top25_real_like.pdf")

    # furthest
    furthest_idx = order[-top_k:]
    plot_grid(gen_imgs[furthest_idx],
              "Top 25 Generated ➞ Outliers",
              f"{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_gen_top25_outliers.pdf")

def plot_accuracy_vs_lambda(lambda_values, metrics, generators, dataset_sizes=dataset_sizes,
                            folds=folds, num_experiments=num_experiments,
                            learning_rates=learning_rates, regularization_params=regularization_params,
                            save_dir='./classifier'):
    # Check if lambda = 0 is present. Otherwise skip this function
    if 0 not in lambda_values:
        print("Skipping accuracy vs. lambda plot as λ=0 is not present.")
        return
    
    # Create one figure for all models
    plt.figure(figsize=(8, 5))    
    
    # Loop over each model in your generators list
    for mname in generators:
        # Calculate the "classically augmented" baseline (λ=0)
        classical_accs = []
        for fold, lr, reg, experiment in itertools.product(folds, learning_rates, regularization_params, range(num_experiments)):
            key = f'{mname}_accuracy_{dataset_sizes[fold][-1]}_{fold}_{experiment}_{lr}_{reg}_{0}'
            classical_accs.append(metrics[key][0])
        classical_acc = np.mean(classical_accs)
        classical_std = np.std(classical_accs)
        
        # Compute mean accuracies and standard deviations for each λ
        gan_accs_means = []
        gan_stds = []
        for lambda_generate in lambda_values:
            gan_accs_for_one_lambda = []
            for fold, lr, reg, experiment in itertools.product(folds, learning_rates, regularization_params, range(num_experiments)):
                key = f'{mname}_accuracy_{dataset_sizes[fold][-1]}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}'
                gan_accs_for_one_lambda.append(metrics[key][0])
            gan_accs_means.append(np.mean(gan_accs_for_one_lambda))
            gan_stds.append(np.std(gan_accs_for_one_lambda))
        gan_accs_means = np.array(gan_accs_means).flatten()
        gan_stds = np.array(gan_stds).flatten()
        
        # Pick the colour for this model
        col = colors.get(mname, 'black')
        
        # Plot the baseline for λ=0 (classically augmented) as a dashed horizontal line
        plt.axhline(y=classical_acc, color=col, linestyle='--', label=f'Classically augmented')
        plt.fill_between(lambda_values, classical_acc - classical_std, classical_acc + classical_std,
                         color=col, alpha=0.1)
        
        # Plot each point for the augmented results at each λ with error bars
        for i, lambda_val in enumerate(lambda_values):
            # Only label the first point per model to avoid duplicate legend entries
            label = f'Classically + {mname} augmented' if i == 0 else None
            plt.errorbar(lambda_val, gan_accs_means[i], yerr=gan_stds[i], fmt='o', capsize=5,
                         color=col, ecolor=col, label=label)
    
    # Add labels, legend, grid and save the figure
    plt.xlabel(r'$\lambda_{gen}$')
    plt.ylabel('Accuracy')
    plt.xticks(lambda_values, [str(l) for l in lambda_values])
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = f'{save_dir}/{galaxy_classes}_{classifier}_{generators}_{max(dataset_sizes[-1])}_{lr}_{reg}_accuracy_vs_lambda.pdf'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved overlayed accuracy vs. lambda plot to {plot_path}")
    

def plot_all_metrics_vs_dataset_size(
    metrics,
    generators,
    merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"},
    dataset_sizes=dataset_sizes,
    folds=folds,
    num_experiments=num_experiments,
    learning_rates=learning_rates,
    regularization_params=regularization_params,
    lambda_values=lambda_values,
    save_dir='./classifier'):
    metric_names  = ["accuracy", "precision", "recall", "f1_score"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for generator in generators:
        for metric, title in zip(metric_names, metric_titles):
            plt.figure(figsize=(10, 6))

            # one line per λ
            for λ in lambda_values:
                # collect values per merged category
                metric_values_per_category = defaultdict(list)

                # iterate over all combinations
                for lr, reg, exp, fold in itertools.product(
                    learning_rates, regularization_params,
                    range(num_experiments), folds
                ):
                    for subset_size in dataset_sizes[fold]:
                        if subset_size not in merge_map:
                            continue
                        key = f"{generator}_{metric}_{subset_size}_{fold}_{exp}_{lr}_{reg}_{λ}"
                        if key in metrics and metrics[key]:
                            bucket = str(merge_map.get(subset_size, subset_size))  # don’t drop sizes missing from merge_map
                            metric_values_per_category[bucket].append(metrics[key][0])  # use the single recorded value

                # sort categories and compute mean±std
                categories = sorted(metric_values_per_category.keys(), key=lambda x: int(x))
                means = [ np.mean(metric_values_per_category[c]) for c in categories ]
                stds  = [ np.std (metric_values_per_category[c]) for c in categories ]

                # plot
                plt.plot(categories, means, marker='o', linestyle='-', label=f"λ={λ}")
                plt.fill_between(categories,
                                 np.array(means) - np.array(stds),
                                 np.array(means) + np.array(stds),
                                 alpha=0.2)

            # now that all λ-lines are drawn, add legend
            plt.legend(title='Lambda Values', fontsize=14, loc='best')
            plt.title(f'{title} vs Dataset Size ({generator})', fontsize=18)
            plt.xlabel('Training Dataset Size', fontsize=14)
            plt.ylabel(title, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            os.makedirs(save_dir, exist_ok=True)
            fname = f'{save_dir}/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_{metric}_vs_dataset_size.pdf'
            plt.savefig(fname)
            plt.close()
            print(f"Saved {title} vs Dataset Size plot to {fname}")


def plot_avg_roc_curves(metrics, generators, dataset_sizes=dataset_sizes, merge_map=merge_map, 
                         folds= folds, num_experiments=num_experiments, 
                        learning_rates=learning_rates, regularization_params=regularization_params, 
                        galaxy_classes=galaxy_classes, 
                        class_descriptions={cls['tag']: cls['description'] for cls in classes}, 
                        save_dir='./classifier'):
    from scipy.interpolate import make_interp_spline # To make the artificiallyjagged curve more smooth
    
    min_label = min(galaxy_classes)
    adjusted_classes = [cls - min_label for cls in galaxy_classes]
    fpr_grid = np.linspace(0, 1, 1000)
    for generator in generators:
        for lr in learning_rates:
            for reg in regularization_params:
                for lambda_generate in lambda_values:
                    for s in range(len(dataset_sizes[0])):
                        roc_values = {class_label: [] for class_label in adjusted_classes}
                        for experiment in range(num_experiments):
                            for fold in folds:
                                subset_size = dataset_sizes[fold][s]
                                true_labels_dict = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                                pred_probs_dict = metrics[f"{generator}_all_pred_probs_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]         
                                key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                                true_labels = true_labels_dict.get(key)
                                pred_probs = pred_probs_dict.get(key)
                                if (true_labels is None) or (pred_probs is None) or (len(true_labels) == 0) or (len(pred_probs) == 0):
                                    continue
                                pred_probs = np.asarray(pred_probs)
                                y = np.asarray(true_labels)

                                # Map tags -> indices only if labels aren't already 0..C-1
                                if y.max() > len(galaxy_classes) - 1:
                                    tag_to_idx = {tag: i for i, tag in enumerate(sorted(galaxy_classes))}
                                    y = np.vectorize(tag_to_idx.get)(y)

                                # ROC is undefined if only one class is present in this fold; skip it.
                                if np.unique(y).size < 2:
                                    print(f"Skipping fold {fold} due to only one class present")
                                    continue

                                if len(adjusted_classes) == 2:
                                    scores = pred_probs[:, 1] if pred_probs.ndim == 2 and pred_probs.shape[1] > 1 else pred_probs.ravel()
                                    fpr, tpr, _ = roc_curve(true_labels, scores, pos_label=1)
                                    interp_tpr = np.interp(fpr_grid, fpr, tpr)
                                    class_label = adjusted_classes[0] # True positive class label is the first in adjusted_classes
                                    roc_values[class_label].append(interp_tpr)
                                else:
                                    y_bin = label_binarize(y, classes=np.arange(len(adjusted_classes)))
                                    for i, class_label in enumerate(adjusted_classes):
                                        fpr, tpr, _ = roc_curve(y_bin[:, i], pred_probs[:, i])
                                        interp_tpr = np.interp(fpr_grid, fpr, tpr)
                                        roc_values[class_label].append(interp_tpr)

                        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
                        ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
                        for spine in ax.spines.values():
                            spine.set_linewidth(2)
                        for class_label, galaxy_class in zip(adjusted_classes, galaxy_classes):
                            if not roc_values[class_label]:
                                continue
                            tpr_values = np.array(roc_values[class_label])
                            mean_tpr = np.mean(tpr_values, axis=0)
                            std_tpr = np.std(tpr_values, axis=0)
                            tpr_p16, tpr_p84 = np.percentile(tpr_values, [16, 84], axis=0)                            
                            mean_auc = auc(fpr_grid, mean_tpr)
                            n = tpr_values.shape[0] # number of runs
                            auc_values = [auc(fpr_grid, tpr_values[i, :]) for i in range(n)]
                            std_auc = np.std(auc_values)
                            auc_p16, auc_p84 = np.percentile(auc_values, [16, 84])
                            print(f"Class {class_descriptions.get(galaxy_class, str(galaxy_class))}: AUC={mean_auc:.3f}=±{std_auc:.3f}, 16th={auc_p16:.3f}, 84th={auc_p84:.3f}, runs={n}")
                            ax.plot(fpr_grid, mean_tpr, lw=3.5, color='#77dd77',
                                    label=f'Mean ROC (AUC={mean_auc:.3f}, 16th={auc_p16:.3f}, 84th={auc_p84:.3f})')
                            # Add shaded percentile region
                            ax.fill_between(fpr_grid,
                                            np.clip(tpr_p16, 0, 1),
                                            np.clip(tpr_p84, 0, 1),
                                            color='#77dd77', alpha=0.3,
                                            label=f'68% confidence interval (runs={n})')

                        ax.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate', fontsize=22)
                        ax.set_ylabel('True Positive Rate', fontsize=22)
                        merged_subset_key = merge_map.get(subset_size, str(subset_size))
                        ax.legend(loc="lower right", fontsize=18)
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(f'{save_dir}/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_avg_roc_curve.pdf')
                        plt.close(fig)
                print(f"Saved average ROC curve at {save_dir}/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_avg_roc_curve.pdf")


def plot_diff_avg_std_confusion_matrix(metrics, generators, metric_stats,
    merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"},
    lambda_vals=(0, 1), save_dir='./classifier'):
    
    # Check if lambda = 0 and lambda = 1 are present. Otherwise skip this function
    if any(lam not in lambda_values for lam in lambda_vals):
        print("Skipping difference in average confusion matrix plot as one or both lambda values are not present.")
        return
    
    custom_cmap = LinearSegmentedColormap.from_list("baby_red_green", ["#ff6961", "white", "#77dd77"])
    
    # Loop over learning rate and regularization (remove lambda from outer loop)
    for lr, reg in itertools.product(learning_rates, regularization_params):
        subset_conf_matrices = {}
        # Collect confusion matrices for both lambda values concurrently
        for experiment in range(num_experiments):
            for fold in folds:
                for subset_size in dataset_sizes[fold]:
                    for lam in lambda_vals:
                        key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"
                        true_labels_dict = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"][0]
                        pred_labels_dict = metrics[f"{generator}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"][0]
                        true_labels = true_labels_dict.get(key)
                        pred_labels = pred_labels_dict.get(key)
                        if (true_labels is None) or (pred_labels is None) or (len(true_labels) == 0) or (len(pred_labels) == 0):
                            continue
                        cm = confusion_matrix(true_labels, pred_labels, normalize='true')
                        merged_key = merge_map.get(subset_size, subset_size)
                        if merged_key not in subset_conf_matrices:
                            subset_conf_matrices[merged_key] = {lam_val: [] for lam_val in lambda_vals}
                        subset_conf_matrices[merged_key][lam].append(cm)
                        
        # Now, for each merged key, check that we have data for both lambda values.
        for merged_size, lam_dict in subset_conf_matrices.items():
            if any(len(lam_dict[lam_val]) == 0 for lam_val in lambda_vals):
                print(f"Not enough data for merged size {merged_size} for lambda values {lambda_vals}")
                continue
            cms_lam0 = np.array(lam_dict[lambda_vals[0]])
            cms_lam1 = np.array(lam_dict[lambda_vals[1]])
            avg_cm_lam0 = np.mean(cms_lam0, axis=0)
            std_cm_lam0 = np.std(cms_lam0, axis=0)
            avg_cm_lam1 = np.mean(cms_lam1, axis=0)
            std_cm_lam1 = np.std(cms_lam1, axis=0)
            diff_avg = avg_cm_lam1 - avg_cm_lam0
            diff_std = np.sqrt(std_cm_lam1**2 + std_cm_lam0**2)
            ann = np.empty(diff_avg.shape, dtype=object)
            for i in range(diff_avg.shape[0]):
                for j in range(diff_avg.shape[1]):
                    ann[i, j] = f"{diff_avg[i, j]:.2f}\n±{diff_std[i, j]:.2f}"
            
            desc_by_tag = {cls['tag']: cls['description'] for cls in classes}
            present_descriptions = [desc_by_tag[tag] for tag in galaxy_classes]
            lambda0_values = metric_stats[lambda_vals[0]]['accuracy']
            lambda1_values = metric_stats[lambda_vals[1]]['accuracy']
            mean_diff = np.mean(lambda1_values) - np.mean(lambda0_values)
            std_diff = np.sqrt(np.std(lambda1_values)**2 + np.std(lambda0_values)**2)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(diff_avg, annot=ann, fmt="", cmap=custom_cmap, center=0,
                        xticklabels=present_descriptions, yticklabels=present_descriptions, ax=ax)
            ax.set_xlabel("Predicted Label", fontsize=14)
            ax.set_ylabel("True Label", fontsize=14)
            ax.set_title(f"Difference in Average Accuracy: {mean_diff:.2f} ± {std_diff:.2f}", fontsize=16)
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_vals[1]}-{lambda_vals[0]}_diff_confusion_matrix.pdf"
            plt.savefig(save_path)
            plt.close(fig)


def plot_avg_std_confusion_matrix(metrics, generators, metric_stats, merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"}, save_dir='./classifier'):
    
    for generator in generators:
        for lr, reg, lambda_generate in itertools.product(learning_rates, regularization_params, lambda_values):
            subset_conf_matrices = {}
            for experiment in range(num_experiments):
                for fold in folds:
                    for subset_size in dataset_sizes[fold]:
                        true_labels_dict = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                        pred_labels_dict = metrics[f"{generator}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]         
                        key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        true_labels = true_labels_dict.get(key)
                        pred_labels = pred_labels_dict.get(key)
                        if (true_labels is None) or (pred_labels is None) or (len(true_labels) == 0) or (len(pred_labels) == 0):
                            continue
                        pred_labels = np.array(pred_labels)
                        num_classes = len(galaxy_classes)
                        cm = confusion_matrix(true_labels, pred_labels, normalize='true', labels=list(range(num_classes)))
                        if cm.size == 0 or cm.shape[0] != cm.shape[1]:
                            print(f"Skipping invalid confusion matrix with shape {cm.shape}")
                            continue
                        merged_key = merge_map.get(subset_size, subset_size)
                        if merged_key not in subset_conf_matrices:
                            subset_conf_matrices[merged_key] = []
                        subset_conf_matrices[merged_key].append(cm)
                        
            for subset_size, cm_list in subset_conf_matrices.items():
                if not cm_list:
                    print(f"No valid confusion matrices for subset size {subset_size}")
                    continue
                cms = np.array(cm_list)
                avg_cm = np.mean(cms, axis=0)
                std_cm = np.std(cms, axis=0)
                desc_by_tag = {cls['tag']: cls['description'] for cls in classes}
                present_descriptions = [desc_by_tag[tag] for tag in galaxy_classes]
                ann = np.empty(avg_cm.shape, dtype=object)
                for i in range(avg_cm.shape[0]):
                    for j in range(avg_cm.shape[1]):
                        ann[i, j] = f"{avg_cm[i, j]:.2f}\n±{std_cm[i, j]:.2f}"
                values = metric_stats[lambda_generate]['accuracy']
                mean_value = np.mean(values)
                std_dev = np.std(values)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    avg_cm,
                    annot=ann,
                    fmt="",
                    #cmap="Blues",
                    cmap=cmap_green,
                    xticklabels=present_descriptions,
                    yticklabels=present_descriptions,
                    annot_kws={"fontsize": 40},
                    ax=ax
                )
                # increase tick‐label font size
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=40, rotation=0, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=40, rotation=90)
                ax.set_xlabel("Predicted Label", fontsize=40)
                ax.set_ylabel("True Label", fontsize=40)
                ax.set_title(f"Average Accuracy: {mean_value:.2f} ± {std_dev:.2f}", fontsize=40)
                
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=40)

                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_avg_confusion_matrix.pdf"
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

def plot_loss(
    generators,
    history=history,
    dataset_sizes=dataset_sizes,
    folds=folds,
    num_experiments=num_experiments,
    galaxy_classes=galaxy_classes,
    learning_rates=learning_rates,
    regularization_params=regularization_params,
    classifier=classifier,
    lambda_values=lambda_values,
    save_dir='./classifier'
):
    sorted_lambdas = sorted(lambda_values)
    n_lambdas = len(sorted_lambdas)

    # Create discrete colors for each lambda using a continuous colormap
    base_cmap = colormaps["viridis"]
    if n_lambdas > 1:
        color_list = [base_cmap(i / (n_lambdas - 1)) for i in range(n_lambdas)]
    else:
        color_list = [base_cmap(0.5)]

    lambda_to_color = {
        lam: color_list[i] for i, lam in enumerate(sorted_lambdas)
    }

    for generator in generators:
        fig, ax = plt.subplots(figsize=(10, 8))

        # We'll store two special line handles for Train/Val,
        # plus a handle for each λ color.
        # Then we combine them in a single legend with 2 columns.
        train_line = mlines.Line2D([], [], color='black', linestyle='-',
                                   label='Train')
        val_line   = mlines.Line2D([], [], color='black', linestyle='--',
                                   label='Validation')

        # Keep track of which λ's we actually plotted (in case some are skipped)
        plotted_lambdas = set()

        for lr, reg in itertools.product(learning_rates, regularization_params):
            for lambda_generate in sorted_lambdas:
                all_train_losses = []
                all_val_losses   = []

                for experiment in range(num_experiments):
                    for fold in folds:
                        largest_subset_size = max(dataset_sizes[fold])
                        if largest_subset_size <= 0:
                            continue

                        loss_key = (
                            f"{generator}_loss_{largest_subset_size}_{fold}_"
                            f"{experiment}_{lr}_{reg}_{lambda_generate}"
                        )
                        val_loss_key = (
                            f"{generator}_val_loss_{largest_subset_size}_{fold}_"
                            f"{experiment}_{lr}_{reg}_{lambda_generate}"
                        )

                        if loss_key not in history.get(generator, {}):
                            continue
                        if val_loss_key not in history[generator]:
                            continue

                        train_loss = history[generator][loss_key]
                        val_loss   = history[generator][val_loss_key]

                        all_train_losses.append(train_loss)
                        all_val_losses.append(val_loss)

                if not all_train_losses or not all_val_losses:
                    continue

                # At least one valid set found for this λ
                plotted_lambdas.add(lambda_generate)

                # Ensure consistent epoch lengths by truncation
                min_len_train = min(len(t) for t in all_train_losses)
                min_len_val   = min(len(t) for t in all_val_losses)
                min_len       = min(min_len_train, min_len_val)

                all_train_losses = [arr[:min_len] for arr in all_train_losses]
                all_val_losses   = [arr[:min_len] for arr in all_val_losses]

                # Convert to arrays, compute mean ± std
                all_train_losses = np.array(all_train_losses)
                all_val_losses   = np.array(all_val_losses)
                mean_train = np.mean(all_train_losses, axis=0)
                std_train  = np.std(all_train_losses, axis=0)
                mean_val   = np.mean(all_val_losses, axis=0)
                std_val    = np.std(all_val_losses, axis=0)

                epochs = range(len(mean_train))
                color = lambda_to_color[lambda_generate]

                # Plot training (solid)
                ax.plot(
                    epochs, mean_train, color=color, linestyle='-'
                )
                ax.fill_between(
                    epochs,
                    mean_train - std_train,
                    mean_train + std_train,
                    color=color, alpha=0.2
                )

                # Plot validation (dashed)
                ax.plot(
                    epochs, mean_val, color=color, linestyle='--'
                )
                ax.fill_between(
                    epochs,
                    mean_val - std_val,
                    mean_val + std_val,
                    color=color, alpha=0.2
                )

        ax.set_title(f"Training and Validation Loss for Generator={generator}", fontsize=18)
        ax.set_xlabel("Epochs", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        ax.grid(True)

        # Build the color legend for each λ that was actually plotted
        color_handles = []
        color_labels = []
        for lam in sorted_lambdas:
            if lam not in plotted_lambdas:
                continue
            # Create a dummy line with the correct color
            line = mlines.Line2D([], [], color=lambda_to_color[lam], linestyle='-',
                                 label=f"λ={lam}")
            color_handles.append(line)
            color_labels.append(f"λ={lam}")

        # Combine color-based handles + line-style handles
        # So we get a 2-column legend: one for λ colors, one for Train/Val styles
        combined_handles = color_handles + [train_line, val_line]
        combined_labels  = color_labels  + ["Train", "Validation"]

        # Create a single legend with 2 columns
        ax.legend(combined_handles, combined_labels, fontsize=16, loc="upper right",
                  ncol=2, columnspacing=1.2, handletextpad=0.7)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_{lr}_{reg}_{lambda_generate}_loss.pdf")
        plt.close(fig)

###############################################
############ PLOTS AFTER ALL FOLDS ############
###############################################

class_descriptions = [cls['description'] for cls in classes if cls['tag'] in galaxy_classes]

# ---------------------- Rankings & summary ----------------------
# We summarize for the last (largest) subset in the selected fold(s).
metrics_last = defaultdict(lambda: defaultdict(list))

for lam in valid_lambda_values:
    for fold in folds:
        subset = max(dataset_sizes[fold])
        for exp, lr, reg in itertools.product(range(num_experiments), learning_rates, regularization_params):
            for metric in ["accuracy", "precision", "recall", "f1_score"]:
                k = f"{generators[0]}_{metric}_{subset}_{fold}_{exp}_{lr}_{reg}_{lam}"
                if k in tot_metrics and tot_metrics[k]:
                    metrics_last[lam][metric].append(tot_metrics[k][0])

# Pretty print rankings per λ
#for lam, stats in metrics_last.items():
#    print(f"\nRankings for λ = {lam}:\n")
#    for metric in ["accuracy", "precision", "recall", "f1_score"]:
#        vals = stats.get(metric, [])
#        print(f"{metric.capitalize()} Rankings (sorted by highest value):")
#        for rank, v in enumerate(sorted(vals, reverse=True), start=1):
#            print(f"{rank}: {metric.capitalize()}: {v:.4f}")
#        print()

# Mean ± Std per λ
for lam, stats in metrics_last.items():
    print(f"Lambda = {lam}")
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        vals = np.array(stats.get(metric, []), dtype=float)
        if vals.size:
            print(f"{metric.capitalize()}: Mean = {vals.mean():.4f}, Std = {vals.std():.4f}")
    print()

print("\nTraining Times (aggregated):")
all_subset_sizes = {s for fs in dataset_sizes for s in fs}
merged_times = defaultdict(list)

# Pull training-times dicts we stored in `metrics` during the load step
for k, v in metrics.items():
    if "_training_times_" not in k or not v:
        continue
    tt = v[0]
    if not isinstance(tt, dict):
        continue

    # Detect layout per top-level key
    for k1, v1 in tt.items():
        if not isinstance(v1, dict):
            continue

        if isinstance(k1, (int, np.integer)) and k1 in all_subset_sizes:
            # Layout A: tt[subset_size][fold] -> list
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
        print(f"{cat}: {times.mean():.2f} ± {times.std():.2f} seconds (n={len(times)})")


# If there exist a lambda>0
if any(lam > 0 for lam in lambda_values):
    for generator in generators:

        # --- build real‐data loader (test set) ---
        _, _, test_images, test_labels = load_galaxies(
            galaxy_class=galaxy_classes, fold=5,
            crop_size=(128,128), downsample_size=(128,128), sample_size=None,
            REMOVEOUTLIERS=FILTERED, train=False
        )
        # ensure we have a channel dimension: (N, 1, H, W)
        if test_images.dim() == 3:
            test_images = test_images.unsqueeze(1)
        real_ds = TensorDataset(test_images, torch.zeros_like(test_images), test_labels)
        real_loader = DataLoader(real_ds, batch_size=256, shuffle=False)


        # --- build generated‐data loader using get_synthetic() ---
        all_imgs, all_labels = [], []

        if generator == 'GAN':
            model_kwargs = {
                'NORMALISEIMGS': False,
                'NORMALISEIMGSTOPM': False,
                'gan_type':        gan_type,
                'gan_latent_dim':  gan_latent_dim,
                'lr_gen':          lr_gen,
                'lr_disc':         lr_disc,
                'sample_size':     gan_sample_size,
                'gen_loss':        gan_gen_loss,
                'disc_loss':       gan_disc_loss,
                'label_smoothing': gan_label_smoothing,
                'lambda_div':      gan_lambda_div,
                'data_version':    gan_data_version,
                'epoch':           gan_epoch,
            }


        elif generator == 'Dual':
            from kymatio.torch import Scattering2D
            scattering = Scattering2D(J=2, L=12, shape=(128, 128), max_order=2)   
            scat_coeffs = scattering(test_images[:5])    
            model_kwargs = {
                'NORMALISEIMGS': False,  # Normalise images to [0, 1]
                'NORMALISEIMGSTOPM': False,  # Normalise images to [-1, 1] 
                'scatshape': np.shape(scat_coeffs)[1:],
                'hidden_dim1': 128,
                'hidden_dim2': 128,
                'latent_dim': 64,
                'vae_latent_dim': 64}

        else:
            model_kwargs = {}
        
            
        for cls in galaxy_classes:
            fold = 5 if generator == 'DDPM' else 0
            imgs_list, labels_list = get_synthetic(
                gen_model_name=generator,
                cls=cls,
                num_generate=len(test_images),
                img_shape=(1, 128, 128), 
                batch_size_generate=500,
                galaxy_classes=galaxy_classes,
                FILTERGEN=FILTERED,
                model_kwargs=model_kwargs,
                fold=fold,
                device=device
            )
            all_imgs   .extend(imgs_list)
            all_labels .extend(labels_list)

        # if a batch has shape [1, N, H, W], swap to [N, 1, H, W]
        all_imgs = [
            img.permute(1, 0, 2, 3) if img.dim()==4 and img.shape[0]==1 else img
            for img in all_imgs
        ]

        fixed = []
        for img in all_imgs:
            if img.dim()==4 and img.shape[1] > img.shape[0]:
                # swap batch and channel dims
                img = img.permute(1, 0, 2, 3)
            fixed.append(img)
        gen_images = torch.cat(fixed, dim=0)

        all_labels = [lab.unsqueeze(0) if lab.dim() == 0 else lab for lab in all_labels]
        gen_labels = torch.cat(all_labels, dim=0)

        # ensure we have a channel dimension (N, 1, H, W)
        if gen_images.dim() == 3:
            gen_images = gen_images.unsqueeze(1)
        gen_ds = TensorDataset(gen_images, torch.zeros_like(gen_images), gen_labels)
        gen_loader = DataLoader(gen_ds, batch_size=256, shuffle=False)


        print("clf_models type:", type(clf_models))
        if isinstance(clf_models, dict):
            print("clf_models keys:", list(clf_models.keys()))
        else:
            print(clf_models)

        print("clf_models", clf_models)

        # RustigeClassifier, ProjectModel, MLPClassifier, DualClassifier, DANNClassifier
        if classifier == 'Rustige':
            wrappername = 'RustigeClassifier'
        elif classifier == 'ProjectModel':
            wrappername = 'ProjectModel'
        elif classifier == 'DANN':
            wrappername = 'DANNClassifier'

        # pick the one and only model wrapper you saved
        wrappername = next(iter(clf_models))
        wrapper     = clf_models[wrappername]
        model   = wrapper['model']                 # the actual nn.Module you trained
        model   = model.to(device).eval()

        visualize_feature_tsne(
            model,
            real_loader,
            gen_loader,
            wrappername,
            device=device,
            perplexity=30,
            n_iter=1000,
            save_path=f"./classifier/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_tsne.pdf"
        )
        visualize_tsne_by_class(model, real_loader, gen_loader, wrappername, device=device,
                                save_path=f"./classifier/{galaxy_classes}_{classifier}_{generator}_{largest_sz}_tsne_by_class.pdf"
        )

        plot_overlap_image_grids(model, real_loader, gen_loader, device=device)


# ——— Make plots ———
plot_loss(generators, history=history)
robust_metric_histograms(metrics)
plot_avg_roc_curves(metrics, generators, merge_map=merge_map)
#plot_all_metrics_vs_dataset_size(metrics, generators, merge_map=merge_map)
#plot_accuracy_vs_lambda(lambda_values, metrics, generators)
plot_avg_std_confusion_matrix(metrics, generators, metric_stats=metrics_last, merge_map=merge_map)
#plot_diff_avg_std_confusion_matrix(metrics, generators, metric_stats=metrics_last, merge_map=merge_map)

print("Script finished successfully!")