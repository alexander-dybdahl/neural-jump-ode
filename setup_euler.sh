#!/bin/bash
# Setup script for first-time setup on Euler cluster
# Run this interactively: bash setup_euler.sh

echo "=== Setting up Neural Jump ODE on Euler ==="

# Load modules
echo "Loading modules..."
module load gcc/13.2.0 python/3.11.6

# Show loaded modules
module list

# Create necessary directories
mkdir -p logs
mkdir -p runs

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To submit jobs, use:"
echo "  sbatch run_heston.sh          # Run Heston experiment"
echo "  sbatch run_black_scholes.sh   # Run Black-Scholes experiment"
echo "  sbatch run_ou.sh               # Run Ornstein-Uhlenbeck experiment"
echo "  sbatch run_gpu.sh              # Run with GPU (larger problem)"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To view logs:"
echo "  tail -f logs/heston_JOBID.out"
