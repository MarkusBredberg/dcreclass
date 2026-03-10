#!/bin/bash -l

#SBATCH --job-name=comp_plots
#SBATCH --account=sk032
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

SCRIPT_PATH="/users/mbredber/p2_DCRECLASS/scripts/03.create_processed_images.py"
OUT_DIR="/users/mbredber/p2_DCRECLASS/outputs/comparison_plots"
mkdir -p "$OUT_DIR"
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

COMMON_ARGS="--comparison-plot --no-montage --force --n-workers 4 --scales 25,50,100"

echo "--- beam_crop ---"
python "$SCRIPT_PATH" $COMMON_ARGS \
    --comp-out "${OUT_DIR}/comparison_beam_crop.pdf"

echo "--- beam_crop_no_sub ---"
python "$SCRIPT_PATH" $COMMON_ARGS \
    --no-beam-sub \
    --comp-out "${OUT_DIR}/comparison_beam_crop_no_sub.pdf"

echo "--- cheat_crop ---"
python "$SCRIPT_PATH" $COMMON_ARGS \
    --cheat-rt \
    --comp-out "${OUT_DIR}/comparison_cheat_crop.pdf"

echo "--- fov_crop ---"
python "$SCRIPT_PATH" $COMMON_ARGS \
    --fov-crop --fov-arcsec 800 \
    --comp-out "${OUT_DIR}/comparison_fov_crop.pdf"

echo "--- pixel_crop ---"
python "$SCRIPT_PATH" $COMMON_ARGS \
    --comp-out "${OUT_DIR}/comparison_pixel_crop.pdf"

echo "========================================"
echo "Job finished at: $(date)"
echo "Outputs written to: $OUT_DIR"
echo "========================================"
