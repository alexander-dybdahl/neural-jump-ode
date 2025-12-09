#!/bin/bash
#SBATCH --job-name=njode_heston
#SBATCH --output=logs/heston_%j.out
#SBATCH --error=logs/heston_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=4096

# Load modules
module load gcc/13.2.0 python/3.11.6

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up virtual environment (first time only)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run experiment with default parameters
echo "Running Heston experiment..."
python experiments/experiment_heston.py \
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
