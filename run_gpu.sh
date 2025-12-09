#!/bin/bash
#SBATCH --job-name=njode_gpu
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8g

# Load modules for GPU
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6_cuda

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Run experiment with GPU
echo "Running experiment on GPU..."
python experiments/experiment_heston.py \
    --n-train 5000 \
    --n-val 500 \
    --n-epochs 200 \
    --hidden-dim 100 \
    --n-hidden-layers 2 \
    --activation relu \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --print-every 10 \
    --device cuda

echo "Job completed!"
