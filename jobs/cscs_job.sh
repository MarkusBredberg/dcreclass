#!/bin/bash -l

#SBATCH --job-name=skapa
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
#SBATCH --cpus-per-task=12
#SBATCH -C gpu

# ============================================================
# Key hyperparameters — must match 04.train_classifier.py
# ============================================================
export CLASSIFIER="DualSSN"
export VERSIONS="RAW"
export LR="4e-5"
export REG="1e-1"

# ============================================================
# Generate a unique run ID and create output directories
# ============================================================
DATE=$(date +%Y%m%d)
HHMM=$(date +%H%M)
RUN_ID="${CLASSIFIER}_${VERSIONS}_lr${LR}_reg${REG}_${DATE}_${HHMM}_${SLURM_JOB_ID}"
SCRIPT_PATH="/users/mbredber/p2_DCRECLASS/scripts/04.train_classifier.py"

export RUN_DIR="/users/mbredber/p2_DCRECLASS/outputs/${RUN_ID}"
export DATA_RUN_DIR="/users/mbredber/p2_DCRECLASS/data/runs/${RUN_ID}"

mkdir -p "${RUN_DIR}/figures"
mkdir -p "${RUN_DIR}/logs"
mkdir -p "${DATA_RUN_DIR}/models"
mkdir -p "${DATA_RUN_DIR}/metrics"

# Write initial configuration file
CONFIG_FILE="${RUN_DIR}/logs/config.txt"
export CONFIG_FILE
cat > "$CONFIG_FILE" << EOF
========================================
INFRASTRUCTURE
========================================
Run ID:        ${RUN_ID}
Timestamp:     $(date)
SLURM Job ID:  ${SLURM_JOB_ID}
Node:          $(hostname)
CPUs:          ${SLURM_CPUS_PER_TASK}
Memory:        120G
Script:        ${SCRIPT_PATH}
Script MD5:    $(md5sum "$SCRIPT_PATH" | cut -d' ' -f1)
RUN_DIR:       ${RUN_DIR}
DATA_RUN_DIR:  ${DATA_RUN_DIR}
SLURM out:     /users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-${SLURM_JOB_ID}.out
SLURM err:     /users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-${SLURM_JOB_ID}.err

========================================
HYPERPARAMETERS (job script)
========================================
CLASSIFIER:    ${CLASSIFIER}
VERSIONS:      ${VERSIONS}
LR:            ${LR}
REG:           ${REG}
EOF

# Restore system defaults
source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# Load Python
module load cray/23.12
module load cray-python/3.11.5

# Explicitly tell Python where to find user-installed packages
# This is needed because compute nodes may not inherit ~/.local
export PYTHONPATH=/users/mbredber/.local/lib/python3.11/site-packages:$PYTHONPATH

# Also point to the editable install's source directly
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

# Verify
python -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Run directory: $RUN_DIR"
echo "Data directory: $DATA_RUN_DIR"
echo "========================================"

python "$SCRIPT_PATH"

echo "Job finished at: $(date)"

# Symlink SLURM logs into the run directory for easy access
ln -sf "/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-${SLURM_JOB_ID}.out" "${RUN_DIR}/logs/sbatchrun.out"
ln -sf "/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-${SLURM_JOB_ID}.err" "${RUN_DIR}/logs/sbatchrun.err"
