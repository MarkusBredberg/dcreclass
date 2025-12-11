# mount google drive
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# import packages
import numpy as np
import torch 
from tqdm import tqdm
from utils.plotting import plot_image_grid, plot_histograms
from utils.calc_tools import normalise_images
import io
from contextlib import redirect_stdout
import re


# change the path below to where you download the scattering package
import sys
sys.path.append('/content/scattering_transform/')
import scattering
# import os
# os.chdir('/content/scattering')

# import scattering package

#################################################
########### PARAMETERS ############################
#################################################

cls = 10
type = 'train' # 'train' or 'eval'

n_singleloop = 389 # How many images to do 1-to-{n_gen} generation for
n_gen = 1 # How many images to generate per input image

n_multloop = 10 # How many loops to run for the generation
n_multin = n_singleloop #How ny images to generate per input image in multi-generation
n_multgen = 100 # How many images to generate per input image in multi-generation

M, N = 128, 128 # Image size

mode = 'single' # 'single', 'multi' or 'both'

"""
Length of train images for class 10 : 389
Length of train images for class 11 : 816
Length of train images for class 12 : 292
Length of train images for class 13 : 242"""

#################################################
########### SET UP ##############################
#################################################

print("Running ST generation script based on the notebook...")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
input_file = f"/users/mbredber/data/FIRST/FIRST_{type}_{cls}_f5_{n_singleloop}.npy"
#input_file =  f'/users/mbredber/data/MGCLS/classified_crops_1600/MGCLS_{type}_{cls}_{n_singleloop}.npy'   
#input_file = f"/users/mbredber/scratch/data/PSZ2/classified/T100kpcSUB/DE/PSZ2_{type}_{cls}_{n_singleloop}.npy"


###################################################
############ SCATTERING SETUP #####################
###################################################

input_data = np.load(input_file) 

if mode != 'multi':
    raw = input_data[:n_singleloop, 0, :M, :N]
    image_target = scattering.whiten(raw)

    image_syn_list = []
    total_max_residual = 0.0
    total_mean_residual = 0.0

    for img in tqdm(image_target, desc=f'1-to-{n_gen} generation', total=n_singleloop):
        f = io.StringIO()
        with redirect_stdout(f):
            #syn = scattering.synthesis('s_cov', img[None, ...], seed=0)
            syns = scattering.synthesis('s_cov', img[None, ...], seed=0, ensemble=True, N_ensemble=n_gen)
        output = f.getvalue()
        image_syn_list.extend(syns if isinstance(syns, (tuple, list, np.ndarray)) else [syns])

    image_syn = np.stack(image_syn_list)
    if image_syn.shape[0] == 1:
        image_syn = image_syn  # keep shape (1, 128, 128)
    else:
        image_syn = image_syn.squeeze()

    print("Number of images generated:", len(image_syn_list))

    np.save(f'ST_generation/1to{n_gen}_{n_singleloop}_{cls}.npy', image_syn)
    scattering.show(image_target[:5], image_syn[:5], (-5, 5), savepath=f'ST_generation/1to{n_gen}_{n_singleloop}_{cls}.png')


if mode != 'single':
    image_multisyn = None
    for loop_idx in tqdm(range(n_multloop), desc=f'{n_multin}-to-{n_multgen} loops'):
        # randomly pick n_multin distinct inputs
        idx = np.random.choice(n_singleloop, n_multin, replace=False)
        raw_multi = input_data[idx, 0, :M, :N]
        
        # whiten and average their scattering coefficients
        image_targets = scattering.whiten(raw_multi, overall=True)
        syns = scattering.synthesis('s_cov_iso', image_targets, seed=0, ensemble=True, N_ensemble=n_multgen)

        # Visualize each loop’s batch
        scattering.show(image_targets[:1], syns[:1], (-5, 5), 
                        savepath=f'ST_generation/{n_multin}to{n_multgen}_loop{loop_idx}_{cls}.png'
        )
        
        # Append the generated images to the list
        if image_multisyn is None:
            image_multisyn = syns
        else:
            image_multisyn = np.concatenate((image_multisyn, syns), axis=0)
    np.save(f'ST_generation/{n_multin}to{n_multgen}_{cls}.npy',image_multisyn)
"""
input_size:  (104, 128, 128)
use torch backend
Running ST generation script based on the notebook...
Shape of image_targets: (104, 128, 128)
Shape of image_target: (20, 128, 128)
input_size:  (104, 128, 128)
# of estimators:  993
max residual:  2.5006247 , mean residual:  0.03837063
max residual:  0.12988472 , mean residual:  0.0053988565
time used:  25.184828996658325 s
input_size:  (20, 128, 128)
# of estimators:  993
max residual:  2.313126 , mean residual:  0.037668213
max residual:  0.004366532 , mean residual:  0.0005614061
time used:  27.022812366485596 s
 Shape of syns: (1, 128, 128)
 Shape of image_syn: (10, 128, 128)
Traceback (most recent call last):
  File "/users/mbredber/scatter_galaxies/2.ST_generation.py", line 117, in <module>
    scattering.show(image_targets[:5], syns[:5], (-5, 5), 
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/mbredber/scatter_galaxies/scattering/__init__.py", line 641, in show
    plt.imshow(image_syn[i], vmin=hist_range[0], vmax=hist_range[1])
               ~~~~~~~~~^^^
IndexError: index 1 is out of bounds for axis 0 with size 1
"""