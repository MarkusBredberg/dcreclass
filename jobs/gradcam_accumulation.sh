#!/bin/bash -l

#SBATCH --job-name=gradcam_accum
#SBATCH --account=sk036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ── Run configuration ─────────────────────────────────────────────────────────
# Pass extra arguments to the script, e.g.:
#   EXTRA_ARGS="--force"          overwrite all cached runs
#   EXTRA_ARGS="--force 3 7"      overwrite only runs 3 and 7
#   EXTRA_ARGS="--runs 0 1 2"     only process these runs
EXTRA_ARGS=""

CACHE_DIR="/users/mbredber/p2_DCRECLASS/outputs/scratch/figures/classifying/explore_classification_results_outputs/gradcam_cache"
mkdir -p "${CACHE_DIR}"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

# ── Environment ───────────────────────────────────────────────────────────────
source /users/mbredber/p2_DCRECLASS/.venv/bin/activate

export PYTHONPATH=/users/mbredber/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Cache dir:   ${CACHE_DIR}"
echo "Extra args:  ${EXTRA_ARGS}"
echo "========================================"

python /users/mbredber/p2_DCRECLASS/scripts/run_gradcam_accumulation.py ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "Done at: $(date)"
echo "Cache contents:"
ls -lh "${CACHE_DIR}"/*.npz 2>/dev/null || echo "  (no cache files found)"
echo "========================================"
