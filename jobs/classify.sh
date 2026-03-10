#!/bin/bash -l

#SBATCH --job-name=classify
#SBATCH --account=sk032
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

# ── Run configuration ─────────────────────────────────────────────────────────
export CLASSIFIER="ImageCNN"
export CROP_MODE="beam_crop"

# Output directories for this run
RUN_LABEL="${CLASSIFIER}_${CROP_MODE}"
export RUN_DIR="/users/mbredber/p2_DCRECLASS/outputs/classifier/${RUN_LABEL}"
export DATA_RUN_DIR="/users/mbredber/p2_DCRECLASS/outputs/classifier/${RUN_LABEL}"
export CONFIG_FILE="${RUN_DIR}/logs/config.txt"

mkdir -p "${RUN_DIR}/figures"
mkdir -p "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/models"
mkdir -p "${RUN_DIR}/metrics"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

# ── Environment ───────────────────────────────────────────────────────────────
source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
module load cray/23.12
module load cray-python/3.11.5

export PYTHONPATH=/users/mbredber/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Classifier:  ${CLASSIFIER}"
echo "Crop mode:   ${CROP_MODE}"
echo "Output dir:  ${RUN_DIR}"
echo "========================================"

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo "--- Training ${CLASSIFIER} on ${CROP_MODE} ---"
python /users/mbredber/p2_DCRECLASS/scripts/04.train_classifier.py

# ── Step 2: Evaluate ──────────────────────────────────────────────────────────
echo "--- Evaluating ${CLASSIFIER} on ${CROP_MODE} ---"
python /users/mbredber/p2_DCRECLASS/scripts/05.plot_classifier_results.py

echo "========================================"
echo "Job finished at: $(date)"
echo "Results written to: ${RUN_DIR}"
echo "========================================"
