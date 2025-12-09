#!/bin/bash
#SBATCH --job-name=njode_ou
#SBATCH --output=logs/ou_%j.out
#SBATCH --error=logs/ou_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=4096

# Load modules
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6

# Optimize PyTorch for multi-core CPU usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Run experiment
echo "Running Ornstein-Uhlenbeck experiment..."
python -u experiments/experiment_ou.py \
    --n-train 1000 \
    --n-val 200 \
    --n-epochs 200 \
    --hidden-dim 50 \
    --n-hidden-layers 1 \
    --activation relu \
    --learning-rate 1e-3 \
    --batch-size 128 \
    --print-every 10 \
    --device cpu \
    --shared-network

echo "Job completed!"
