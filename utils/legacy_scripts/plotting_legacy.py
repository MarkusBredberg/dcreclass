from utils.data_loader import get_classes
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torchvision.utils import make_grid, save_image
import os

#plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram


###############################################
########## SIMPLE PLOTS OF IMAGES #############
###############################################

def first_channel(arr):
    # take Tensor or ndarray → numpy array
    a = arr.cpu().detach().numpy() if isinstance(arr, torch.Tensor) else arr

    # If three channels plot the average map of them. Otherwise only plot the first channel.
    if a.ndim == 3:
        if a.shape[0] == 3:
            return np.mean(a, axis=0)
        else:
            return a[0]
    return a.squeeze()

def save_images_tensorboard(images, file_path, nrow=4):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, file_path)

def show_image(x, idx, save=True, title=None, save_path=None):
    # Assume x is a batch tensor where each item is a flattened image
    image = x[idx]  # Select the image at index 'idx'

    # Check if the image needs to be reshaped
    if len(image.shape) == 1:  # Image is flattened
        # Assuming the image is square, calculate the size of each side
        side_length = int(np.sqrt(image.shape[0]))
        if side_length * side_length != image.shape[0]:
            raise ValueError("Image data does not form a perfect square, cannot reshape automatically")
        image = image.view(side_length, side_length).cpu().detach().numpy().squeeze()
    else:
        # Handle case for non-flattened image if necessary
        image = image.cpu().detach().numpy().squeeze()

    print("shape of image: ", image.shape)
    # Create figure and axes
    plt.figure(figsize=(6, 6))
    plt.imshow(first_channel(image), cmap='viridis')
    plt.title(title)
    plt.axis('off')
    if not save:
        plt.show()
    else:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.savefig('./loss_vs_epoch.png')
        plt.close()
        
        
def plot_GAN_losses(output_dir, galaxy_class, g_loss_history, d_loss_history, gen_loss_name, disc_loss_name): #Used in 2.scatterGAN
    
    plt.figure(figsize=(10, 5))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Generator loss
    ax1.plot(g_loss_history, label="Generator Loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Generator Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create a second y-axis for the discriminator loss
    ax2 = ax1.twinx()
    ax2.plot(d_loss_history, label="Discriminator Loss", color="tab:orange")
    ax2.set_ylabel("Discriminator Loss", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("GAN Training Losses")
    fig.tight_layout()
    filename = f"ScatterGAN_loss_plot_{galaxy_class}_gen-{gen_loss_name}_disc-{disc_loss_name}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()



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


    
def plot_images_by_class(images, labels, num_images=5, save_path="./classifier/unknown_omdel_example_inputs.png"):
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
    classes = get_classes()
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

def plot_original_images(data_loader, image_size, grid_size, save_path=None, save=True):
    # Initialize an empty container to hold the grid of images
    container = np.zeros((image_size * grid_size, image_size * grid_size))
    
    # Populate the container with images from the data loader
    index = 0
    for images, _ in data_loader:
        for img in images:
            row = index // grid_size
            col = index % grid_size
            container[row * image_size: (row + 1) * image_size, col * image_size: (col + 1) * image_size] = img.squeeze().cpu().numpy()
            index += 1
            if index >= grid_size * grid_size:
                break
        if index >= grid_size * grid_size:
            break
    
    # Plot the grid of images
    plt.figure(figsize=(10, 10))
    plt.imshow(first_channel(container), cmap='viridis')
    plt.axis('off')
    
    if save:
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.savefig('./loss_vs_epoch.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


###############################################
########## PLOTS COMPARING NEW AND OLD ########
###############################################


def new_images_epoch(src_imgs, supertitle, generated_images, num_galaxies, 
                     epochs_to_plot=[1, 5, 10], save=False, save_path='./figures/new_img_ep.png'):
    """
    Plots generated images at different epochs, comparing them with one or more source images.

    Parameters:
    - src_imgs: Array of original images.
    - generated_images: Array containing generated images with dimensions (epochs, channels, height, width) for num_galaxies=1
                        or (num_samples, epochs, channels, height, width) for num_galaxies > 1.
    - num_galaxies: The number of galaxy images to plot.
    - epochs_to_plot: List of epochs at which the generated images are to be plotted.
    """
    
    # Convert TensorFlow tensors to numpy arrays if not already in that format
    src_imgs = src_imgs.detach().cpu().numpy()

    # Set up the figure layout based on the number of galaxies
    fig = plt.figure(figsize=(15, 20 if num_galaxies > 1 else 5))
    fig.suptitle(supertitle, fontsize=20)

    # Adjust the grid layout based on the number of galaxies
    if num_galaxies > 1:
        gs = fig.add_gridspec(4, len(epochs_to_plot))  # 4 rows, columns equal to the number of epochs
    else:
        gs = fig.add_gridspec(1, len(epochs_to_plot) + 1)  # 1 row, first for original image, next for epochs

    print("shape of generated_images in function: ", np.shape(generated_images))
    # Adding images to subplots
    if num_galaxies > 1:
        # Plotting the original images in the top row
        for i in range(len(epochs_to_plot)):
            ax = fig.add_subplot(gs[0, i])
            if i < len(src_imgs):
                original_img = np.moveaxis(src_imgs[i], 0, -1)  # Adjust channel dimension for plotting
                ax.imshow(first_channel(original_img))
                ax.set_title(f"Input Image {i+1}")
            ax.axis('off')

        # Plotting generated images in the remaining four rows
        for row in range(1, 4):  # Adjusted to 4 rows including original images
            for col, epoch in enumerate(epochs_to_plot):
                ax = fig.add_subplot(gs[row, col])
                if row-1 < generated_images.shape[0]:  # Ensure there are enough generated images for the row
                    generated_img = np.moveaxis(generated_images[row-1, col], 0, -1)
                    ax.imshow(first_channel(generated_img))
                    ax.set_title(f"Epoch {epoch + 1} - Image {row}")
                ax.axis('off')

    else:
        # Plotting the single original image
        ax = fig.add_subplot(gs[0, 0])
        original_img = np.moveaxis(src_imgs[0], 0, -1)
        ax.imshow(first_channel(original_img))
        ax.set_title("Original Image")
        ax.axis('off')

        # Plotting generated images for each specified epoch
        for i, epoch in enumerate(epochs_to_plot):
            ax = fig.add_subplot(gs[0, i + 1])
            generated_img = np.moveaxis(generated_images[i], 0, -1)
            ax.imshow(first_channel(generated_img))
            ax.set_title(f"Epoch {epoch + 1}")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for supertitle
    
    if save:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    
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
    
    
def plot_variance_histograms(gen_images, real_images, save_path):
    """
    Compute per-image variance (over pixels) for generated and real images,
    and plot histograms comparing the two distributions.
    """
    # Flatten each image and compute variance over the pixels.
    gen_variances = gen_images.view(gen_images.size(0), -1).var(dim=1).detach().cpu().numpy()
    real_variances = real_images.view(real_images.size(0), -1).var(dim=1).detach().cpu().numpy()
    
    plt.figure()
    plt.hist(gen_variances, bins=30, alpha=0.5, label="Generated Image Variance")
    plt.hist(real_variances, bins=30, alpha=0.5, label="Real Image Variance")
    plt.xlabel("Variance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Image Variances")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def show_two_rows(src_images, gen_images, title, savepath=None):
    num_images = len(src_images)
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4), gridspec_kw={'wspace': 0, 'hspace': 0})

    # Ensure axs is treated as a 2D array, even if num_images is 1
    if num_images == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    plt.suptitle(title)

    for i in range(num_images):
        axs[0, i].imshow(first_channel(src_images[i].squeeze()), cmap='viridis')
        axs[0, i].axis('off')
        axs[1, i].imshow(first_channel(gen_images[i].squeeze()), cmap='viridis')
        axs[1, i].axis('off')

    # Set titles to the left of each row
    fig.text(0.04, 0.75, "Original", va='center', ha='center', rotation='vertical', fontsize=12)
    fig.text(0.04, 0.25, "Synthesized", va='center', ha='center', rotation='vertical', fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #plt.show()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.savefig('./loss_vs_epoch.png')
        
def plot_reconstructions(test_loader, test_labels, batch_size, model, runname, DEVICE="cuda", num_images=5, cmap='viridis'):
    """
    Function to plot the reconstructions of test images.
    :param test_loader: DataLoader containing the test data.
    :param model: Trained VAE model.
    :param img_shape: Shape of the input images.
    :param batch_size: Batch size used during testing.
    :param runname: Name of the run (for saving output).
    :param device: Device (e.g., "cuda" or "cpu").
    :param num_images: Number of images to plot and compare.
    :param cmap: Color map for plotting (default is 'viridis').
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                continue
            scat_coeffs, x = batch
            x = x.to(DEVICE)
            
            #Runname is of the form {galaxy_class}_{num_galaxies}_{encoder_choice}_{fold}'
            encoder_choice = runname.split('_')[2]

            if 'ST' in encoder_choice or 'Dual' in encoder_choice: # Scattering transform
                scat_coeffs = scat_coeffs.to(DEVICE)
                if 'C' in encoder_choice: # Conditional VAE
                    labels = test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                    if 'MLP' in encoder_choice:
                        x_hat, mean, log_var = model(scat_coeffs, labels)
                    elif 'CDual' in encoder_choice:
                        x_hat, mean, log_var = model(x, scat_coeffs, labels)
                elif 'Dual' in encoder_choice:
                    x_hat, mean, log_var = model(x, scat_coeffs)
                else:
                    x_hat, mean, log_var = model(scat_coeffs)
            elif 'C' in encoder_choice and encoder_choice != 'CNN': # Conditional VAE - No scattering transform
                labels = test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                x_hat, mean, log_var = model(x, labels)
            else:
                x_hat, mean, log_var = model(x)
                
            if x_hat.shape != x.shape:
                x_hat = x_hat.view(x.shape)
            
            # Select only the number of images to plot (num_images)
            original_images = x[:num_images].cpu()
            reconstructed_images = x_hat[:num_images].cpu()
            
            # Plot the images in two rows
            fig, axs = plt.subplots(2, num_images, figsize=(num_images * 5, 10))  # 2 rows, num_images columns
            for i in range(num_images):
                axs[0, i].imshow(first_channel(original_images[i].squeeze().numpy()), cmap=cmap, vmin=0, vmax=1)
                axs[0, i].set_title(f'Original Image {i+1}')
                axs[0, i].axis('off')
                
                axs[1, i].imshow(first_channel(reconstructed_images[i].squeeze().numpy()), cmap=cmap, vmin=0, vmax=1)
                axs[1, i].set_title(f'Reconstructed Image {i+1}')
                axs[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'./generator/VAE_{runname}/reconstructions.png', bbox_inches='tight')
            plt.close(fig)

            # Break after plotting the first batch
            break

        
            
def plot_weight_and_loss_maps(weight_map, loss_map, x, xhat, savepath=None):
    """
    Saves the weight map, loss map, and original image for the first batch.

    Parameters:
    weight_map (numpy.ndarray or torch.Tensor): The weight map.
    loss_map (numpy.ndarray or torch.Tensor): The loss map.
    x (numpy.ndarray or torch.Tensor): The original image.
    batch_idx (int): The index of the batch.
    runname (str): The name of the run for saving the file.
    """
    # Ensure the input is on the CPU and detached if it is a torch.Tensor
    if hasattr(weight_map, 'cpu'):
        weight_map = weight_map.cpu().detach().numpy()
    if hasattr(loss_map, 'cpu'):
        loss_map = loss_map.cpu().detach().numpy()
    if hasattr(x, 'cpu'):
        x = x.cpu().detach().numpy()
        xhat = xhat.cpu().detach().numpy()
    
    # Create the figure
    plt.figure(figsize=(12, 12))

    # Plot the weight map
    plt.subplot(2, 2, 1)
    plt.imshow(first_channel(weight_map[0].squeeze()), cmap='magma')  # Remove the first dimension
    plt.colorbar(label='Weight')
    plt.title('Weight Map')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')

    # Plot the loss map
    plt.subplot(2, 2, 2)
    plt.imshow(first_channel(loss_map[0].squeeze()), cmap='inferno')  # Remove the first dimension
    plt.colorbar(label='Loss')
    plt.title('Loss Map')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')

    # Plot the original image
    plt.subplot(2, 2, 3)
    plt.imshow(first_channel(x[0].squeeze()), cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('Original Image')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    
    # Plot the reconstructed image
    plt.subplot(2, 2, 4)
    plt.imshow(first_channel(xhat[0].squeeze()), cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('Reconstructed Image')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')

    # Finalize and save the figure
    plt.suptitle(f'Weight Map and Reconstructed Image')
    plt.savefig(savepath) if savepath is not None else plt.show()
    plt.close()

def vae_plot_comparison(old_images, generated_images, supertitle, num_images=5, save=False, save_path=None):
    """
    Plots two rows of images, where the first row displays the old images and the second row displays the corresponding generated images.
    Adds a supertitle to the plot and specific titles to the rows.

    Parameters:
    - old_images: a batch of old images in the shape (batch_size, channels, height, width) or (batch_size, height * width)
    - generated_images: a batch of generated images in the same shape as old_images
    - supertitle: title for the top of the plot
    - num_images: number of images to display from the batch
    """
    assert old_images.shape[0] >= num_images, "Not enough old images in the batch"
    assert generated_images.shape[0] >= num_images, "Not enough generated images in the batch"

    # Convert to numpy if images are tensors
    if isinstance(old_images, torch.Tensor):
        old_images = old_images.cpu().detach().numpy()
    if isinstance(generated_images, torch.Tensor):
        generated_images = generated_images.cpu().detach().numpy()

    # Handle different shapes for old_images
    if old_images.ndim == 2:  # Flattened images
        image_size = int(old_images.shape[1]**0.5)
        old_images = old_images.reshape(-1, image_size, image_size)
    elif old_images.ndim == 4 and old_images.shape[1] == 1:  # Images with channel dimension
        old_images = old_images.squeeze(1)

    # Handle different shapes for generated_images
    if generated_images.ndim == 2:  # Flattened images
        image_size = int(generated_images.shape[1]**0.5)
        generated_images = generated_images.reshape(-1, image_size, image_size)
    elif generated_images.ndim == 4 and generated_images.shape[1] == 1:  # Images with channel dimension
        generated_images = generated_images.squeeze(1)


    # Create figure with subplots
    fig, axs = plt.subplots(2, num_images, figsize=(14, 7))
    fig.suptitle(supertitle, fontsize=16)

    # Display images
    for i in range(num_images):
        axs[0, i].imshow(first_channel(old_images[i]), cmap='viridis')
        axs[0, i].axis('off')
        axs[1, i].imshow(first_channel(generated_images[i]), cmap='viridis')
        axs[1, i].axis('off')

    axs[0, 0].text(-0.1, 0.5, "Original images", va='center', ha='center', fontsize=12, rotation=90, transform=axs[0, 0].transAxes)
    axs[1, 0].text(-0.1, 0.5, "Reconstructed images", va='center', ha='center', fontsize=12, rotation=90, transform=axs[1, 0].transAxes)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between subplots
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for supertitle
    if not save:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
 
 
def vae_multmod(old_images, generated_images, model_names, save=False, save_path=None, show_title=False, show_originals=True):
    """
    Plots comparison of original and generated images for multiple models.
    """
    # Check if the generated images list is empty
    if not generated_images:
        print("Generated images list is empty.")
        return

    # Only keep the first occurrence of each model
    seen_models = set()
    unique_indices = []
    for i, name in enumerate(model_names):
        if name not in seen_models:
            seen_models.add(name)
            unique_indices.append(i)
    
    # Update the model names and generated images list
    model_names = [model_names[i] for i in unique_indices]
    generated_images = [generated_images[i] for i in unique_indices]

    # when showing originals, drop “Real” from the generated‐images list
    if show_originals and old_images is not None and "Real" in model_names:
        rm = model_names.index("Real")
        model_names.pop(rm)
        generated_images.pop(rm)

    if old_images is not None:
        num_images = len(old_images)
    else:
        num_images = 5

    num_models = len(model_names)
    num_rows = num_models + (1 if show_originals and old_images is not None else 0)

    fig, axs = plt.subplots(num_rows, num_images, figsize=(num_images * 2, num_rows * 2), constrained_layout=True)

    # Normalize the images to [0, 1] range for consistent color mapping
    norm = plt.Normalize(vmin=0, vmax=1)

    # If axs is 1D (when num_rows or num_images is 1), convert it to a 2D array
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_images == 1:
        axs = np.expand_dims(axs, axis=1)

    # Display original images if provided and show_originals is True
    if show_originals and old_images is not None:
        if old_images.ndim == 4:
            old_images = old_images.squeeze(1)  # Squeeze the channel dimension if it exists
        for i in range(num_images):
            img = old_images[i].cpu().numpy().squeeze()  # Move to CPU and convert to numpy
            axs[0, i].imshow(first_channel(img), cmap='RdYlBu_r', norm=norm)
            axs[0, i].axis('off')
        axs[0, 0].text(-0.1, 0.5, "Original", va='center', ha='center', fontsize=12, rotation=90, transform=axs[0, 0].transAxes)

    # Display generated images for each model
    for j, (model_name, gen_images) in enumerate(zip(model_names, generated_images)):
        if gen_images.ndim == 4:
            gen_images = gen_images.squeeze(1)  # Squeeze the channel dimension if it exists
        for i in range(num_images):
            img = gen_images[i].cpu().numpy().squeeze()  # Move to CPU and convert to numpy
            if show_originals and old_images is not None:
                axs[j + 1, i].imshow(first_channel(img), cmap='RdYlBu_r', norm=norm)
                axs[j + 1, i].axis('off')
            else:
                axs[j, i].imshow(first_channel(img), cmap='RdYlBu_r', norm=norm)
                axs[j, i].axis('off')
        if show_originals and old_images is not None:
            axs[j + 1, 0].text(-0.1, 0.5, model_name, va='center', ha='center', fontsize=12, rotation=90, transform=axs[j + 1, 0].transAxes)
        else:
            axs[j, 0].text(-0.1, 0.5, model_name, va='center', ha='center', fontsize=12, rotation=90, transform=axs[j, 0].transAxes)

    # Optionally display the title
    if show_title:
        plt.suptitle("Generated images most similar to original", fontsize=16)

    # Save or show the plot
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0) if save else plt.show()
    plt.close()


###############################################
################ LOSS PLOTS ##################
###############################################


def loss_vs_epoch(train_losses, val_losses=None, epochs_to_plot=None, save=False, save_path='./figures/loss.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the training losses
    ax.plot(train_losses, label='Train Loss')
    
    # Plot validation losses only if they are provided
    if val_losses is not None:
        ax.plot(val_losses, label='Validation Loss')

    # Plot specific epochs if provided
    if epochs_to_plot is not None:
        epochs_to_plot = [epoch for epoch in epochs_to_plot if epoch < len(train_losses)]
        if epochs_to_plot:
            ax.plot(epochs_to_plot, [train_losses[epoch] for epoch in epochs_to_plot], 'ro', markersize=12)

    # Set plot labels and scales
    ax.set_title('Loss vs. Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc='upper right')
    
    # Save or show the plot
    if save:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    
def plot_loss(epochs, loss_mean, loss_error, val_loss_mean, val_loss_error, 
              model_name, subset_size, save_path='./classifier/last_lost.png'):
    # Ensure the number of epochs matches the length of loss_mean and loss_error
    if len(epochs) != len(loss_mean):
        print(f"Length mismatch: epochs ({len(epochs)}) and loss_mean ({len(loss_mean)})")
        epochs = range(1, len(loss_mean) + 1)  # Adjust epochs to match the length of loss_mean

    plt.errorbar(epochs, loss_mean, yerr=loss_error.squeeze(), fmt='-', label=f'{model_name} Train Loss')
    plt.errorbar(epochs, val_loss_mean, yerr=val_loss_error.squeeze(), fmt='--', label=f'{model_name} Validation Loss')
    plt.title(f"Training size {subset_size}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()