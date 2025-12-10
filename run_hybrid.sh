#!/bin/bash
#SBATCH --job-name=njode_hybrid
#SBATCH --output=logs/hybrid_%j.out
#SBATCH --error=logs/hybrid_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=4096

# Load modules
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6

# Optimize PyTorch for multi-core CPU usage
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

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
echo "Running Hybrid OU-BS experiment..."
python -u experiments/experiment_hybrid.py \
    --n-train 10000 \
    --n-val 2000 \
    --n-epochs 200 \
    --hidden-dim 50 \
    --n-hidden-layers 1 \
    --activation relu \
    --dt-ode-step 0.01 \
    --learning-rate 1e-3 \
    --batch-size 128 \
    --print-every 10 \
    --device cpu \
    --num-moments 2 \
    --moment-weights 1.0 10.0 \
    --theta-ou 1.0 \
    --mu-ou 0.0 \
    --sigma-ou 0.3 \
    --mu-bs 0.1 \
    --sigma-bs 0.5 \
    --T 1.0 \
    --n-steps 100 \
    --x0 1.0

echo "Job completed!"
