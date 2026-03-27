#!/bin/bash -l

#SBATCH --job-name=process_data
#SBATCH --account=sk036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16

SCRIPT_PATH="/users/mbredber/p2_DCRECLASS/scripts/03.create_processed_images.py"
mkdir -p "/users/mbredber/scratch/figures"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

# Restore system defaults
source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# Load Python
module load cray/23.12
module load cray-python/3.11.5

export PYTHONPATH=/users/mbredber/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

# Outputs go automatically to:
#   /users/mbredber/scratch/data/PSZ2/<crop_mode>/<blur_method>/fits_files/
#   /users/mbredber/scratch/data/PSZ2/<crop_mode>/<blur_method>/montages/
#   /users/mbredber/scratch/figures/processing/comparison_<crop_mode>_<blur_method>.pdf
COMMON_ARGS="--comparison-plot --no-montage --force --n-workers 16 --scales 25,50,100 --blur-method circular --no-annotate"

echo "--- beam_crop / circular (no annotations) ---"
python "$SCRIPT_PATH" $COMMON_ARGS --crop-mode beam_crop

echo "--- fov_crop / circular (no annotations) ---"
python "$SCRIPT_PATH" $COMMON_ARGS --crop-mode fov_crop --fov-arcsec 800

echo "--- pixel_crop / circular (no annotations) ---"
python "$SCRIPT_PATH" $COMMON_ARGS --crop-mode pixel_crop

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
