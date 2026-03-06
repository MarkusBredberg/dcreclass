import math, torch, os, hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from typing import Dict, List, Optional, Any
from pathlib import Path


def img_hash(img: torch.Tensor) -> str:
    arr = img.cpu().contiguous().numpy()
    returnval = hashlib.sha1(arr.tobytes()).hexdigest()
    return returnval

def _to_2d_for_imshow(x: torch.Tensor | np.ndarray,
                      how: str = "first") -> np.ndarray:
    """
    Return a (H, W) numpy array suitable for plt.imshow from a tensor/ndarray.

    Accepts shapes like:
      (H, W)
      (C, H, W)            or (H, W, C)
      (B, C, H, W)         or (T, C, H, W)
      (B, T, C, H, W)

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Image-like object.
    how : {"first","mean","max"}
        How to reduce non-spatial/extra axes (channels, time, batch).
    """

    def _reduce(a, axis=0):
        if how == "mean":
            return a.mean(axis=axis)
        if how == "max":
            return a.max(axis=axis)
        # "first"
        return np.take(a, 0, axis=axis)

    # ---- convert to numpy float32 without altering values ----
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().float().numpy()
    else:
        a = np.asarray(x, dtype=np.float32)

    # ---- peel dimensions until we have (H, W) ----
    if a.ndim == 2:
        img = a

    elif a.ndim == 3:
        # Heuristic: channels-first if first dim is small (<=4) and last isn't;
        # channels-last if last dim is small (<=4) and first isn't.
        c_first = (a.shape[0] in (1, 2, 3, 4)) and (a.shape[-1] not in (1, 2, 3, 4))
        c_last  = (a.shape[-1] in (1, 2, 3, 4)) and (a.shape[0]  not in (1, 2, 3, 4))

        if c_first:
            # (C, H, W)
            img = a[0] if a.shape[0] == 1 else _reduce(a, axis=0)
        elif c_last:
            # (H, W, C)
            img = a[..., 0] if a.shape[-1] == 1 else _reduce(a, axis=-1)
        else:
            # Ambiguous; take first plane along the leading axis.
            img = _reduce(a, axis=0)

    elif a.ndim == 4:
        # Assume leading axis is batch/time → reduce then recurse.
        img = _to_2d_for_imshow(_reduce(a, axis=0), how=how)

    elif a.ndim == 5:
        # (B, T, C, H, W) → reduce B and T, then recurse.
        a = _reduce(a, axis=0)
        a = _reduce(a, axis=0)
        img = _to_2d_for_imshow(a, how=how)

    else:
        # Fallback: keep reducing the first axis until 2D.
        while a.ndim > 2:
            a = _reduce(a, axis=0)
        img = a

    # Ensure float32 ndarray
    return np.asarray(img, dtype=np.float32)


def plot_image_grid(images: torch.Tensor | np.ndarray,
                    num_images: int = 36,
                    nrow: int = 6,
                    save_path: Optional[str] = None,
                    titles: Optional[List[str]] = None,
                    cmap: str = "viridis") -> Optional[plt.Figure]:
    """
    Show up to `num_images` in an nrow x ncol grid.
    Accepts images shaped [N,C,H,W] or [N,H,W] (torch.Tensor or np.ndarray).
    If `titles` is provided, it should be a list of strings (one per shown image).
    """

    # to numpy
    if isinstance(images, torch.Tensor):
        imgs = images.detach().cpu().numpy()
    else:
        imgs = np.asarray(images)

    # keep only the first num_images
    imgs = imgs[:num_images]

    # shape handling
    if imgs.ndim == 4:          # [N, C, H, W]
        if imgs.shape[1] == 1:  # single channel
            imgs = imgs[:, 0]   # -> [N, H, W]
        elif imgs.shape[1] == 3:  # RGB
            imgs = np.transpose(imgs, (0, 2, 3, 1))  # -> [N, H, W, 3]
        else:
            # fall back to first channel
            imgs = imgs[:, 0]
    elif imgs.ndim != 3:        # expect [N, H, W]
        raise ValueError(f"Expected images with ndim 3 or 4, got {imgs.ndim}")

    n = imgs.shape[0]
    ncols = max(1, int(nrow))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2*ncols, 2.2*nrows))
    axes = np.atleast_1d(axes).ravel()

    # pad/trim titles safely
    if titles is None:
        titles_list = [None]*n
    else:
        titles_list = list(titles)[:n] + [None]*max(0, n - len(titles))

    for i, ax in enumerate(axes):
        if i < n:
            im = imgs[i]
            if im.ndim == 2:           # grayscale
                ax.imshow(im, origin="lower", cmap=cmap)
            else:                       # RGB
                ax.imshow(im, origin="lower")
            if titles_list[i]:
                ax.set_title(str(titles_list[i]), fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig

def plot_pixel_overlaps_side_by_side(
        train_images: List[torch.Tensor],
        eval_images: List[torch.Tensor],
        train_filenames: Optional[List[str]] = None,
        eval_filenames: Optional[List[str]] = None,
        max_hashes: int = 20,
        outdir: str = "./overlap_debug") -> int:
    """
    For each pixel-identical hash shared by train/test, save a side-by-side figure.
    Title of each panel: 'train — <name>' or 'test — <name>'.
    Works with 2D, 3D (C,H,W), or 4D (T,C,H,W) tensors per image.
    """
    os.makedirs(outdir, exist_ok=True)

    # fallbacks if filenames aren't available
    if not train_filenames: train_filenames = [f"idx {i}" for i in range(len(train_images))]
    if not eval_filenames:  eval_filenames  = [f"idx {i}" for i in range(len(eval_images))]

    # build hash -> indices maps
    train_map, eval_map = {}, {}
    for i, img in enumerate(train_images):
        h = img_hash(img)
        train_map.setdefault(h, []).append(i)
    for j, img in enumerate(eval_images):
        h = img_hash(img)
        eval_map.setdefault(h, []).append(j)

    commons = list(set(train_map) & set(eval_map))
    if not commons:
        print("[overlap-debug] No pixel-identical images between train and test.")
        return 0

    for k, h in enumerate(commons[:max_hashes]):
        t_idxs = train_map[h]
        e_idxs = eval_map[h]
        nrows  = max(len(t_idxs), len(e_idxs))

        fig, axs = plt.subplots(nrows, 2, figsize=(6, 3*nrows))
        if nrows == 1:
            axs = np.array([axs])  # normalize shape

        for r in range(nrows):
            # left column: train
            if r < len(t_idxs):
                ti = t_idxs[r]
                arr = _to_2d_for_imshow(train_images[ti], how="first")
                axs[r, 0].imshow(arr, cmap='viridis', origin='lower')
                axs[r, 0].set_title(f"train — {train_filenames[ti]}", fontsize=10)
            axs[r, 0].axis('off')

            # right column: test
            if r < len(e_idxs):
                ej = e_idxs[r]
                arr = _to_2d_for_imshow(eval_images[ej], how="first")
                axs[r, 1].imshow(arr, cmap='viridis', origin='lower')
                axs[r, 1].set_title(f"test — {eval_filenames[ej]}", fontsize=10)
            axs[r, 1].axis('off')

        fig.suptitle(f"Pixel-identical hash: {h[:12]}…  (train {t_idxs}  |  test {e_idxs})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(outdir, f"overlap_{k:03d}_{h}.png"), dpi=200)
        plt.close(fig)
        print("Plotted overlap at ", os.path.join(outdir, f"overlap_{k:03d}_{h}.png"))

    print(f"[overlap-debug] Wrote {min(len(commons), max_hashes)} figure(s) to {outdir}")
    return len(commons)


def plot_class_images(classes: List[Dict],
                      train_images: torch.Tensor | List,
                      eval_images: torch.Tensor | List,
                      train_labels: List[int] | torch.Tensor,
                      eval_labels: List[int] | torch.Tensor,
                      train_filenames: Optional[List[str]] = None,
                      eval_filenames: Optional[List[str]] = None,
                      set_name: str = "comparison") -> None:    # ensure labels are a plain list of ints
    if isinstance(train_labels, torch.Tensor):
        print("Converting train_labels tensor to list")
        train_labels = train_labels.tolist()
    if isinstance(eval_labels, torch.Tensor):
        eval_labels = eval_labels.tolist()

    desc_map = {c['tag']: c['description'] for c in classes}
    
    unique_classes = sorted(set(train_labels) | set(eval_labels))
    if len(unique_classes) < 2:
        print("Not enough unique classes to compare.")
        return
    
    class1, class2 = unique_classes[:2]    
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Top-left: train class1 
    axes[0, 0].set_title("Train", fontsize=12)
    axes[0, 0].set_ylabel(f"Class {class1}", fontsize=12)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_frame_on(False)

    idxs1_train = [i for i, l in enumerate(train_labels) if l == class1][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[0, 0].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs1_train):
        img = train_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    # Top-right: eval class1
    axes[0, 1].set_title("Eval", fontsize=12)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 1].set_frame_on(False)

    idxs1_eval = [i for i, l in enumerate(eval_labels) if l == class1][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[0, 1].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs1_eval):
        img = eval_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    # Bottom-left: train class2
    axes[1, 0].set_ylabel(f"Class {class2}", fontsize=12)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_frame_on(False)

    idxs2_train = [i for i, l in enumerate(train_labels) if l == class2][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[1, 0].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs2_train):
        img = train_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    # Bottom-right: eval class2
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_frame_on(False)

    idxs2_eval = [i for i, l in enumerate(eval_labels) if l == class2][:9]
    gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[1, 1].get_subplotspec(), wspace=0.02, hspace=0.02)
    subaxes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, idx in zip(subaxes, idxs2_eval):
        img = eval_images[idx]
        arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]
        ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.savefig(f"./classifier/processing_step/{class1}_{class2}_{set_name}_comparison.png", dpi=300)
    plt.close()
    
def plot_images_by_class(images: torch.Tensor | np.ndarray,
                         labels: np.ndarray | List,
                         classes: List[Dict],
                         num_images: int = 5,
                         save_path: str = "./classifier/unknown_model_example_inputs.png") -> None:
    """
    Plots a specified number of input images in a row for each class with the class label as a title.
    """
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = labels.argmax(axis=1)  # collapse multi-label to a single index

    unique_labels = np.unique(labels)
    fig, axes = plt.subplots(len(unique_labels), num_images,
                      figsize=(num_images * 3, len(unique_labels) * 3))


    # make room on the left for the row labels
    fig.subplots_adjust(left=0.2)

    # right after fig, build a tag→description map
    class_map = {c["tag"]: c["description"] for c in classes}

    for i, label in enumerate(unique_labels):
        label_images = images[labels == label][:num_images]

        # use the description instead of the tag number
        axes[i, 0].set_ylabel(
            class_map[int(label)],
            fontsize=20,
            rotation=0,
            ha="right",
            va="center"
        )

        for j in range(num_images):
            ax = axes[i, j]
            arr = label_images[j]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().detach().numpy()

            # If shape is [T, C, H, W], average channels and show each T slice
            if arr.ndim == 4:
                for t in range(arr.shape[0]):
                    t_img = arr[t].mean(axis=0)  # average over channels
                    if j + t * num_images >= axes.shape[1]:
                        continue  # skip overflow
                    ax = axes[i, j + t * num_images // arr.shape[0]]
                    ax.imshow(t_img, cmap="viridis")
                    ax.set_xticks([]); ax.set_yticks([])
                    if j > 0:
                        ax.axis("off")

            elif arr.ndim == 3:
                img2d = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
                img2d = img2d.squeeze()
                ax.imshow(img2d, cmap="viridis")
                ax.set_xticks([]); ax.set_yticks([])
                if j > 0:
                    ax.axis("off")

            ax.set_xticks([]); ax.set_yticks([])
            if j > 0:
                ax.axis("off")

    fig.subplots_adjust(left=0.2, top=0.95, bottom=0.05, hspace=0.15)
    plt.savefig(save_path)
    plt.close()


def plot_background_histogram(orig_imgs: torch.Tensor,
                              gen_imgs: torch.Tensor,
                              img_shape: tuple = (1, 128, 128),
                              title: str = "Background pixels",
                              save_path: str = "background_histogram.png") -> None:
    # define a circular mask to exclude the central source;
    # adjust `radius` (in pixels) to match your source size
    radius = 30  
    h, w = img_shape[1], img_shape[2]
    Y, X = np.ogrid[:h, :w]
    cy, cx = h//2, w//2
    bkg_mask = (Y - cy)**2 + (X - cx)**2 > radius**2  # True for background pixels

    # helper to compute per-image sum over background
    def total_background(images):
        sums = []
        for im in images.cpu().numpy():
            im = np.asarray(im)  # ensure it's a numpy array
            if im.ndim == 3:  # e.g. (T_or_C, H, W)
                im = im[0] if im.shape[0] == 1 else im.mean(axis=0)
            elif im.ndim == 4:  # e.g. (T, C, H, W) or (C, T, H, W)
                im = im.reshape(im.shape[0]*im.shape[1], im.shape[2], im.shape[3]).mean(axis=0)
            elif im.ndim == 1 and im.size == h * w:
                pass
            elif im.ndim != 2:
                raise ValueError(f"Expected 2D image; got {im.shape}")
            im = im.reshape(h, w)
        return sums

    # compute for one class (you can loop over classes as needed)
    real_sums = total_background(orig_imgs)  # orig_cls from your sanity-check loop
    gen_sums  = total_background(gen_imgs)

    # plot histograms
    plt.figure()
    plt.hist(real_sums, bins=50, alpha=0.7, label='Real')
    plt.hist(gen_sums,  bins=50, alpha=0.7, label='Generated')
    plt.xlabel('Total background intensity')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    

def plot_histograms(imgs1: torch.Tensor,
                    imgs2: torch.Tensor,
                    title1: str = "First input",
                    title2: str = "Second input",
                    imgs3: Optional[torch.Tensor] = None,
                    imgs4: Optional[torch.Tensor] = None,
                    title3: str = "Third input",
                    title4: str = "Fourth input",
                    bins: int = 50,
                    main_title: str = "Pixel Value Distribution",
                    save_path: str = "./figures/histogram.png") -> None:
    """
    Plots up to four histograms (two required, two optional) in a single figure.
    
    Args:
        imgs1 (torch.Tensor): First batch of images.
        imgs2 (torch.Tensor): Second batch of images.
        title1 (str): Label for the first histogram.
        title2 (str): Label for the second histogram.
        imgs3 (torch.Tensor or None): Third batch of images (optional).
        imgs4 (torch.Tensor or None): Fourth batch of images (optional).
        title3 (str): Label for the third histogram (used only if imgs3 is not None).
        title4 (str): Label for the fourth histogram (used only if imgs4 is not None).
        bins (int): Number of histogram bins.
        main_title (str): Main title for the plot.
        save_path (str): File path to save the resulting plot.
    """
    plt.figure(figsize=(10, 6))

    # Convert and flatten the first two required image tensors
    imgs1_np = imgs1.cpu().detach().numpy().flatten()
    imgs2_np = imgs2.cpu().detach().numpy().flatten()
    plt.hist(imgs1_np, bins=bins, density=True, histtype='step', linewidth=2, label=title1)
    plt.hist(imgs2_np, bins=bins, density=True, histtype='step', linewidth=2, label=title2)

    # Optionally handle imgs3
    if imgs3 is not None:
        imgs3_np = imgs3.cpu().detach().numpy().flatten()
        plt.hist(imgs3_np, bins=bins, density=True, histtype='step', linewidth=2, label=title3)

    # Optionally handle imgs4
    if imgs4 is not None:
        imgs4_np = imgs4.cpu().detach().numpy().flatten()
        plt.hist(imgs4_np, bins=bins, density=True, histtype='step', linewidth=2, label=title4)

    # Add titles and labels
    plt.title(main_title)
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.legend(loc='upper right')

    plt.savefig(save_path)
    plt.close()
    

###############################################
########### PLOTTING FUNCTIONS ################
###############################################

def robust_metric_histograms(metrics: Dict[str, Any],
                             galaxy_classes: List[int],
                             classifier: str,
                             dataset_sizes: Dict[int, List[int]],
                             folds: List[int],
                             learning_rates: List[float],
                             regularization_params: List[float],
                             save_dir: str = "./classifier/figures") -> None:
    """
    For each metric series (accuracy/precision/recall/f1_score), make a histogram
    with 16/50/84 percentile markers and write a compact percentile summary CSV.

    Histogram style matches plot_redshift_hist.py:
      - stepfilled histtype for clean bar outlines
      - axvspan for shaded 68% interval
      - steelblue mean, goldenrod dashed median, grey dotted percentile boundaries
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    wanted = {"accuracy", "precision", "recall", "f1_score"}

    # ── Shared style config (mirrors plot_redshift_hist.py) ───────────────────
    FONT_SIZE = 8

    plt.rcParams.update({"font.size": FONT_SIZE})

    # ── Aggregate metrics across all folds and experiments ────────────────────
    # Key format in metrics dict: "metric_subset_fold_experiment_lr_reg"
    # Group by "metric_subset_lr_reg" to collapse across folds and experiments
    grouped = defaultdict(list)

    print("\nAggregating metrics across experiments...")
    for key, values in metrics.items():
        parts = key.split("_")

        # Minimum expected parts: metric, subset, fold, experiment, lr, reg
        if len(parts) < 6:
            continue

        metric_name = parts[0]  # e.g. "accuracy", "precision"

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

        # Build group key: drop fold and experiment indices (parts[2] and parts[3])
        subset_size = parts[1]
        lr          = parts[4]
        reg         = parts[5]
        group_key   = f"{metric_name}_{subset_size}_{lr}_{reg}"

        grouped[group_key].extend(vals.tolist())

    print(f"Created {len(grouped)} metric groups")

    # ── Plot one histogram per group ──────────────────────────────────────────
    for group_key, all_vals in grouped.items():
        vals = np.asarray(all_vals, dtype=float)

        if vals.size == 0 or not np.isfinite(vals).any():
            continue

        print(f"Processing {group_key}: {len(vals)} values")

        # ── Robust statistics ─────────────────────────────────────────────────
        p16, p50, p84 = np.percentile(vals, [16, 50, 84])
        mean     = np.mean(vals)
        std      = np.std(vals)
        sigma68  = 0.5 * (p84 - p16)  # Half-width of central 68% interval

        # Parse metric name from group key for axis label
        metric_name = group_key.split("_")[0]

        # ── Bin edges: 20 equal-width bins across the value range ─────────────
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            # Avoid degenerate bins if all values are identical
            eps  = max(1e-6, abs(vmin) * 1e-3)
            vmin, vmax = vmin - eps, vmax + eps
        edges = np.linspace(vmin, vmax, 21)  # 21 edges → 20 bins

        # ── Figure and axes ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(4.5, 3.0))

        # stepfilled: clean outline, no inner vertical bar lines
        ax.hist(
            vals, bins=edges,
            histtype="stepfilled",
            color="gray",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85
        )

        # Shaded 68% interval as a vertical band (drawn behind the histogram)
        ax.axvspan(p16, p84, color="lightgreen", alpha=0.35,
                   label="68% interval", zorder=0)

        # Mean: solid steelblue vertical line
        ax.axvline(mean, color="steelblue", linewidth=1.2,
                   label=f"Mean: {mean:.3f}")

        # Median: dashed goldenrod vertical line
        ax.axvline(p50, color="goldenrod", linewidth=1.2, linestyle="--",
                   label=f"Median: {p50:.3f}")

        # 68% boundary markers: dotted grey lines
        ax.axvline(p16, color="grey", linewidth=0.7, linestyle=":")
        ax.axvline(p84, color="grey", linewidth=0.7, linestyle=":")

        # ── Axis labels and ticks ─────────────────────────────────────────────
        ax.set_xlabel(metric_name.capitalize(), fontsize=FONT_SIZE)
        ax.set_ylabel("Count", fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE - 1)
        ax.set_xlim(vmin, vmax)

        # ── Legend ────────────────────────────────────────────────────────────
        ax.legend(fontsize=FONT_SIZE - 1, framealpha=0.8, loc="upper left")

        # ── Save ──────────────────────────────────────────────────────────────
        fig.tight_layout()
        save_path_hist = (
            f"{save_dir}/{galaxy_classes}_{classifier}_{version}_{largest_sz}_"
            f"{learning_rates[0]}_{regularization_params[0]}_{metric_name}_histogram.pdf"
        )
        fig.savefig(save_path_hist, format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved histogram: {os.path.basename(save_path_hist)}")

        # ── Collect statistics for CSV ────────────────────────────────────────
        rows.append({
            "metric":   metric_name,
            "n":        int(vals.size),
            "setting":  group_key,
            "mean":     float(mean),
            "std":      float(std),
            "p16":      float(p16),
            "p50":      float(p50),
            "p84":      float(p84),
            "sigma68":  float(sigma68)
        })

    # ── Write CSV summary ─────────────────────────────────────────────────────
    if rows:
        import csv
        csv_path = f"{save_dir}/{galaxy_classes}_{classifier}_robust_summary.csv"

        fieldnames = ["metric", "n", "setting", "mean", "std",
                      "p16", "p50", "p84", "sigma68"]

        # Load existing rows to avoid duplicates
        existing_data = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                existing_data = list(csv.DictReader(f))

        existing_settings = {row["setting"] for row in existing_data}
        new_rows = [row for row in rows if row["setting"] not in existing_settings]

        if new_rows:
            all_rows = existing_data + new_rows
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
        

def plot_cluster_metrics(cluster_metrics_dict: Dict[str, List[float]],
                         galaxy_classes: List[int],
                         classifier: str,
                         version: str,
                         largest_sz: int,
                         lr: float,
                         reg: float,
                         save_dir: str = "./classifier/figures") -> None:    
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
    
def plot_avg_roc_curves(metrics: Dict[str, Any],
                        classifier: str,
                        configs: Optional[List[Dict]] = None,
                        merge_map: Optional[Dict] = None,
                        folds: Optional[List[int]] = None,
                        num_experiments: int = 1,
                        learning_rates: Optional[List[float]] = None,
                        regularization_params: Optional[List[float]] = None,
                        galaxy_classes: Optional[List[int]] = None,
                        class_descriptions: Optional[Dict[int, str]] = None,
                        save_dir: str = "./classifier/figures") -> None:
    """
    Plot average ROC curves with 68% confidence intervals.

    Supports overlaying multiple experiment configurations on one figure.
    Each configuration is a dict specifying ALL parameters that may vary:

        configs = [
            {
                'label':                  'T25kpc v1',   # shown in legend
                'learning_rates':         [5e-5],
                'regularization_params':  [1e-1],
                'percentile_lo':          30,
                'percentile_hi':          99,
                'galaxy_classes':         [50, 51],
                'folds':                  list(range(10)),
                'dataset_sizes':          {fold: [3000] for fold in range(10)},
                'num_experiments':        3,
                'version':                'T25kpc',
                'crop_size':              (512, 512),
                'downsample_size':        (128, 128),
            },
            { ... },
        ]

    If configs=None, falls back to a single config built from the keyword arguments.

    Style matches robust_metric_histograms / plot_redshift_hist:
      - Steelblue, goldenrod, tomato, ... colour cycle per config
      - Lightgreen-style shaded 68% CI (colour-matched per config)
      - Black dashed diagonal (random classifier)
      - FONT_SIZE = 8, compact figure
    """
    # ── Style ─────────────────────────────────────────────────────────────────
    FONT_SIZE = 8
    plt.rcParams.update({"font.size": FONT_SIZE})

    # Colour cycle: first entry is steelblue to match histogram style
    CONFIG_COLOURS = ["steelblue", "goldenrod", "tomato", "mediumpurple", "darkcyan"]

    # ── Build config list ─────────────────────────────────────────────────────
    if configs is None:
        # Single config from keyword arguments
        configs = [
            {
                'label':                 f"lr={learning_rates[0]}, reg={regularization_params[0]}",
                'learning_rates':        learning_rates,
                'regularization_params': regularization_params,
                'percentile_lo':         percentile_lo,
                'percentile_hi':         percentile_hi,
                'galaxy_classes':        galaxy_classes,
                'folds':                 folds,
                'dataset_sizes':         dataset_sizes,
                'num_experiments':       num_experiments,
                'version':               version,
                'crop_size':             crop_size,
                'downsample_size':       downsample_size,
            }
        ]

    # ── Collect all subset sizes across all configs ───────────────────────────
    all_subset_sizes = sorted(set(
        sz
        for cfg in configs
        for fold in cfg['folds']
        for sz in cfg['dataset_sizes'][fold]
    ))

    os.makedirs(save_dir, exist_ok=True)

    # ── One figure per subset size ────────────────────────────────────────────
    for subset_size in all_subset_sizes:

        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        # ── One curve group per config ────────────────────────────────────────
        for cfg_idx, cfg in enumerate(configs):
            colour = CONFIG_COLOURS[cfg_idx % len(CONFIG_COLOURS)]

            # Unpack all config parameters
            cfg_lr_list   = cfg['learning_rates']
            cfg_reg_list  = cfg['regularization_params']
            cfg_plo       = cfg['percentile_lo']
            cfg_phi       = cfg['percentile_hi']
            cfg_classes   = cfg['galaxy_classes']
            cfg_folds     = cfg['folds']
            cfg_dsizes    = cfg['dataset_sizes']
            cfg_nexps     = cfg['num_experiments']
            cfg_version   = cfg['version']
            cfg_cs        = f"{cfg['crop_size'][0]}x{cfg['crop_size'][1]}"
            cfg_ds        = f"{cfg['downsample_size'][0]}x{cfg['downsample_size'][1]}"
            cfg_label     = cfg.get('label', f"Config {cfg_idx}")

            # Adjusted (0-indexed) class labels for sklearn
            min_label        = min(cfg_classes)
            adjusted_classes = [c - min_label for c in cfg_classes]
            fpr_grid         = np.linspace(0, 1, 1000)

            # Accumulate interpolated TPR curves across all folds, experiments, lr, reg
            roc_values = {cls: [] for cls in adjusted_classes}

            for lr, reg, experiment, fold in itertools.product(
                cfg_lr_list, cfg_reg_list, range(cfg_nexps), cfg_folds
            ):
                # Skip if this subset size doesn't exist in this fold/config
                if subset_size not in cfg_dsizes.get(fold, []):
                    continue

                # Retrieve stored predictions using the full compound key
                true_labels_dict = metrics.get(
                    f"all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}", [])
                pred_probs_dict  = metrics.get(
                    f"all_pred_probs_{subset_size}_{fold}_{experiment}_{lr}_{reg}", [])

                if not true_labels_dict or not pred_probs_dict:
                    continue

                # The base key encodes ALL config parameters
                base = (
                    f"cl{classifier}_ss{subset_size}_f{fold}"
                    f"_lr{lr}_reg{reg}_ls0.1"
                    f"_cs{cfg_cs}_ds{cfg_ds}"
                    f"_pl{cfg_plo}_ph{cfg_phi}"
                    f"_ver{cfg_version}"
                )

                true_labels = (true_labels_dict[0].get(base)
                               if isinstance(true_labels_dict, list)
                               else true_labels_dict.get(base))
                pred_probs  = (pred_probs_dict[0].get(base)
                               if isinstance(pred_probs_dict, list)
                               else pred_probs_dict.get(base))

                if (true_labels is None or pred_probs is None
                        or len(true_labels) == 0 or len(pred_probs) == 0):
                    continue

                pred_probs = np.asarray(pred_probs)
                y          = np.asarray(true_labels)

                # Remap original class tags to 0-based indices if needed
                if y.max() > len(cfg_classes) - 1:
                    tag_to_idx = {tag: i for i, tag in enumerate(sorted(cfg_classes))}
                    y = np.vectorize(tag_to_idx.get)(y)

                # ROC is undefined with only one class present
                if np.unique(y).size < 2:
                    print(f"  Skipping fold {fold}, experiment {experiment}: "
                          f"only one class present")
                    continue

                # ── Compute interpolated ROC ───────────────────────────────────
                if len(adjusted_classes) == 2:
                    # Binary: score = probability of positive class
                    scores = (pred_probs[:, 1]
                              if pred_probs.ndim == 2 and pred_probs.shape[1] > 1
                              else pred_probs.ravel())
                    fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
                    roc_values[adjusted_classes[1]].append(
                        np.interp(fpr_grid, fpr, tpr))
                else:
                    # Multi-class: one-vs-rest per class
                    y_bin = label_binarize(y, classes=np.arange(len(adjusted_classes)))
                    for i, cls in enumerate(adjusted_classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], pred_probs[:, i])
                        roc_values[cls].append(np.interp(fpr_grid, fpr, tpr))

            # ── Plot mean curve + 68% CI for each class in this config ─────────
            for cls, galaxy_class in zip(adjusted_classes, cfg_classes):
                if not roc_values[cls]:
                    print(f"  No ROC data for class {galaxy_class} in config '{cfg_label}'")
                    continue

                tpr_arr  = np.array(roc_values[cls])   # (n_runs, 1000)
                mean_tpr = np.mean(tpr_arr, axis=0)
                tpr_p16, tpr_p84 = np.percentile(tpr_arr, [16, 84], axis=0)

                # AUC statistics
                n        = tpr_arr.shape[0]
                mean_auc = auc(fpr_grid, mean_tpr)
                auc_vals = [auc(fpr_grid, tpr_arr[i]) for i in range(n)]
                auc_p16, auc_p84 = np.percentile(auc_vals, [16, 84])

                # Legend label: include class name only for multi-class problems
                class_desc = class_descriptions.get(galaxy_class, str(galaxy_class))
                suffix = (f"{class_desc} | " if len(adjusted_classes) > 2 else "")
                legend_label = (
                    f"{suffix}{cfg_label} "
                    f"(AUC={mean_auc:.3f}, [{auc_p16:.3f}, {auc_p84:.3f}], n={n})"
                )

                print(f"  [{cfg_label}] {class_desc}: AUC={mean_auc:.3f}, "
                      f"16th={auc_p16:.3f}, 84th={auc_p84:.3f}, runs={n}")

                # Mean curve
                ax.plot(fpr_grid, mean_tpr,
                        color=colour, linewidth=1.2, label=legend_label)

                # Shaded 68% band
                ax.fill_between(
                    fpr_grid,
                    np.clip(tpr_p16, 0, 1),
                    np.clip(tpr_p84, 0, 1),
                    color=colour, alpha=0.15)

                # Dotted 16th/84th boundary lines
                ax.plot(fpr_grid, np.clip(tpr_p16, 0, 1),
                        color=colour, linewidth=0.5, linestyle=":")
                ax.plot(fpr_grid, np.clip(tpr_p84, 0, 1),
                        color=colour, linewidth=0.5, linestyle=":")

        # ── Random classifier diagonal ─────────────────────────────────────────
        ax.plot([0, 1], [0, 1],
                color="black", linewidth=0.8, linestyle="--",
                label="Random classifier")

        # ── Axes and legend ───────────────────────────────────────────────────
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZE)
        ax.set_ylabel("True Positive Rate",  fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE - 1)
        ax.legend(loc="lower right", fontsize=FONT_SIZE - 1, framealpha=0.8)

        # ── Save ──────────────────────────────────────────────────────────────
        fig.tight_layout()
        save_path = (
            f"{save_dir}/{galaxy_classes}_{classifier}_{version}_"
            f"{largest_sz}_ss{subset_size}_avg_roc_curve.pdf"
        )
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.basename(save_path)}")

    print(f"\nROC curves saved to {save_dir}")


def plot_avg_std_confusion_matrix(metrics: Dict[str, Any],
                                  metric_stats: Dict[str, List[float]],
                                  galaxy_classes: List[int],
                                  classifier: str,
                                  version: str,
                                  largest_sz: int,
                                  learning_rates: List[float],
                                  regularization_params: List[float],
                                  folds: List[int],
                                  dataset_sizes: Dict[int, List[int]],
                                  crop_size: tuple,
                                  downsample_size: tuple,
                                  percentile_lo: int,
                                  percentile_hi: int,
                                  merge_map: Optional[Dict] = None,
                                  save_dir: str = "./classifier/figures") -> None:
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
                annot_kws={"fontsize": 50},
                ax=ax
            )
            
            # Increase tick label font size
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=50, rotation=0, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=50, rotation=90)
            ax.set_xlabel("Predicted Label", fontsize=50)
            ax.set_ylabel("True Label", fontsize=50)
            ax.set_title(f"Avg. Accuracy: {mean_value:.2f} ± {std_dev:.2f}", fontsize=50)
            
            # Adjust colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=50)

            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{galaxy_classes}_{classifier}_{version}_{largest_sz}_{lr}_{reg}_avg_confusion_matrix.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved confusion matrix to {save_path}")


def plot_training_history(history: Dict[str, List[float]],
                          base: str,
                          experiment: int,
                          save_dir: str = "./classifier/test") -> None:
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


def generate_attention_visualizations(model: torch.nn.Module,
                                      test_loader: Any,
                                      galaxy_classes: List[int],
                                      source_names: List[str],
                                      save_dir: str = "./classifier/attention_maps",
                                      methods: Optional[List[str]] = None,
                                      classifier_name: str = "CNN") -> None:
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
