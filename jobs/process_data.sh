#!/bin/bash -l

#SBATCH --job-name=process_data
#SBATCH --account=sk032
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
COMMON_ARGS="--comparison-plot --force --n-workers 16 --scales 25,50,100"

#echo "--- beam_crop / circular ---"
#python "$SCRIPT_PATH" $COMMON_ARGS --crop-mode beam_crop --blur-method circular
#
#echo "--- beam_crop / circular_no_sub ---"
#python "$SCRIPT_PATH" $COMMON_ARGS --crop-mode beam_crop --blur-method circular_no_sub

echo "--- beam_crop / cheat ---"
python "$SCRIPT_PATH" $COMMON_ARGS --crop-mode beam_crop --blur-method cheat

#echo "--- fov_crop / circular ---"
#python "$SCRIPT_PATH" $COMMON_ARGS --crop-mode fov_crop --blur-method circular --fov-arcsec 800

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
