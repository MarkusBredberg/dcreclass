#!/bin/bash -l

#SBATCH --job-name=fetch_sort
#SBATCH --account=sk032
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

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

echo "--- Step 1: Download missing PSZ2 sources ---"
python /users/mbredber/p2_DCRECLASS/scripts/01.rsync_PSZ2.py

echo "--- Step 2: Categorise missing sources ---"
python /users/mbredber/p2_DCRECLASS/scripts/02.categorise_PSZ2.py --symlink

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
