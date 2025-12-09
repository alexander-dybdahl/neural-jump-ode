#!/bin/bash
#SBATCH --job-name=njode_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --array=0-8

# Array job to test multiple hyperparameter combinations
# This will run 9 jobs in parallel (3 hidden_dims × 3 n_hidden_layers)

# Load modules
module load gcc/13.2.0 python/3.11.6

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Define parameter combinations
HIDDEN_DIMS=(50 100 150)
N_LAYERS=(1 2 3)

# Calculate indices
DIM_IDX=$((SLURM_ARRAY_TASK_ID / 3))
LAYER_IDX=$((SLURM_ARRAY_TASK_ID % 3))

HIDDEN_DIM=${HIDDEN_DIMS[$DIM_IDX]}
N_HIDDEN_LAYERS=${N_LAYERS[$LAYER_IDX]}

echo "Running job ${SLURM_ARRAY_TASK_ID}: hidden_dim=${HIDDEN_DIM}, n_hidden_layers=${N_HIDDEN_LAYERS}"

# Run experiment with specific hyperparameters
python experiments/experiment_heston.py \
    --n-train 10000 \
    --n-val 2000 \
    --n-epochs 200 \
    --hidden-dim ${HIDDEN_DIM} \
    --n-hidden-layers ${N_HIDDEN_LAYERS} \
    --activation relu \
    --learning-rate 1e-3 \
    --batch-size 128 \
    --print-every 10 \
    --device cpu

echo "Job ${SLURM_ARRAY_TASK_ID} completed!"
