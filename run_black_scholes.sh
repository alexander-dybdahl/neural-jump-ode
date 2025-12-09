#!/bin/bash
#SBATCH --job-name=njode_bs
#SBATCH --output=logs/bs_%j.out
#SBATCH --error=logs/bs_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=4096

# Load modules
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Run experiment
echo "Running Black-Scholes experiment..."
python experiments/experiment_black_scholes.py \
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
