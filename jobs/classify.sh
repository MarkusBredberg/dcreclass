#!/bin/bash -l

#SBATCH --job-name=classify
#SBATCH --account=sk032
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

# ── Run configuration ─────────────────────────────────────────────────────────
CLASSIFIER="ImageCNN"
CROP_MODE="beam_crop"
VERSIONS="T25kpc"
FOLDS="0 1 2 3 4 5 6 7 8 9"
NUM_EXPERIMENTS=3

# Output directories for this run
RUN_DIR="/users/mbredber/p2_DCRECLASS/outputs/scratch"
mkdir -p "${RUN_DIR}/figures/"
mkdir -p "${RUN_DIR}/data/logs/"
mkdir -p "${RUN_DIR}/data/models/"
mkdir -p "${RUN_DIR}/data/metrics/"}
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
echo "Versions:    ${VERSIONS}"
echo "Run dir:     ${RUN_DIR}"
echo "Data dir:    ${DATA_RUN_DIR}"
echo "========================================"

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo "--- Training ${CLASSIFIER} on ${CROP_MODE} ---"
python /users/mbredber/p2_DCRECLASS/scripts/04.train_classifier.py \
    --classifier "${CLASSIFIER}" \
    --versions "${VERSIONS}" \
    --crop-mode "${CROP_MODE}" \
    --run-dir "${RUN_DIR}" \
    --data-run-dir "${DATA_RUN_DIR}" \
    --folds ${FOLDS} \
    --num-experiments ${NUM_EXPERIMENTS}

# ── Step 2: Evaluate ──────────────────────────────────────────────────────────
echo "--- Evaluating ${CLASSIFIER} on ${CROP_MODE} ---"
python /users/mbredber/p2_DCRECLASS/scripts/05.plot_classifier_results.py \
    --classifier "${CLASSIFIER}" \
    --version "${VERSIONS}" \
    --crop-mode "${CROP_MODE}" \
    --run-dir "${RUN_DIR}" \
    --data-run-dir "${DATA_RUN_DIR}"

echo "========================================"
echo "Job finished at: $(date)"
echo "Results written to: ${RUN_DIR}"
echo "========================================"
