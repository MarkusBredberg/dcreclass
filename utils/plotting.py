import math, torch, os, hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

def img_hash(img: torch.Tensor) -> str:
    arr = img.cpu().contiguous().numpy()
    returnval = hashlib.sha1(arr.tobytes()).hexdigest()
    return returnval

def _to_2d_for_imshow(x, how="first"):
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


def plot_image_grid(
    images,
    num_images=36,
    nrow=6,
    save_path=None,
    titles=None,          # <-- new, optional
    cmap="viridis",
    **_
):
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
    train_images, eval_images,
    train_filenames=None, eval_filenames=None,
    max_hashes=20, outdir="./overlap_debug"
):
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

        
# Create the same function as above, but it plots both train and eval sets and both classes side by side for comparison
# I want the top left quadrant to be the first class of train images, top right to be the first class of eval images, bottom left to be the second class of train images, bottom right to be the second class of eval images
def plot_class_images(classes, train_images, eval_images, train_labels, eval_labels, train_filenames=None, eval_filenames=None, set_name='comparison'):
    # ensure labels are a plain list of ints
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
    
def plot_images_by_class(images, labels, classes, num_images=5, save_path="./classifier/unknown_omdel_example_inputs.png"):
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


def plot_background_histogram(orig_imgs, gen_imgs, img_shape=(1, 128, 128), title="Background pixels", save_path="backgound_histogram.png"):

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
    

def plot_histograms(
    imgs1, imgs2,
    title1='First input', title2='Second input',
    imgs3=None, imgs4=None,
    title3='Third input', title4='Fourth input',
    bins=50,
    main_title="Pixel Value Distribution",
    save_path='./figures/histogram.png'
):
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