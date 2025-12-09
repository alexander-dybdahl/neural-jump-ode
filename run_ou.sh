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
module load gcc/13.2.0 python/3.11.6

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Run experiment
echo "Running Ornstein-Uhlenbeck experiment..."
python experiments/experiment_ou.py \
    --n-train 10000 \
    --n-val 2000 \
    --n-epochs 200 \
    --hidden-dim 50 \
    --n-hidden-layers 1 \
    --activation relu \
    --learning-rate 1e-3 \
    --batch-size 128 \
    --print-every 10 \
    --device cpu

echo "Job completed!"
