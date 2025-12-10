# Neural Jump ODE

A PyTorch implementation of Neural Jump ODEs for modeling irregular time series with jumps, specifically designed for Black Scholes, Ornstein-Uhlenbeck, and Heston stochastic differential equations.

## Overview

This project implements a neural network approach to modeling continuous-time processes observed at irregular intervals. The model consists of three main components:

1. **Jump Network**: Maps observations to hidden states
2. **ODE Function**: Models continuous evolution between observations
3. **Output Network**: Maps hidden states to predictions

The implementation focuses on three key stochastic processes:

- **Black Scholes**: Geometric Brownian motion with constant volatility
- **Ornstein-Uhlenbeck**: Mean-reverting process
- **Heston**: Stochastic volatility model with correlated Brownian motions

## Project Structure

```
neural-jump-ode/
├── neural_jump_ode/          # Main package
│   ├── models/               # Model definitions
│   ├── simulation/           # SDE simulation and data generation
│   └── utils/                # Training utilities and plotting
├── experiments/              # Experiment scripts for each process
├── tests/                    # Unit tests
└── runs/                     # Training outputs and results
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/alexander-dybdahl/neural-jump-ode.git
cd neural-jump-ode

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Running Tests

```bash
python tests/test_basic.py
```

### Basic Example

```python
from neural_jump_ode import NeuralJumpODE, nj_ode_loss
from neural_jump_ode.simulation import create_trajectory_batch

# Create model
model = NeuralJumpODE(
    input_dim=1, 
    hidden_dim=32, 
    output_dim=1,
    n_steps_between=5
)

# Generate synthetic Black Scholes data
batch_times, batch_values = create_trajectory_batch(
    n_trajectories=10,
    process_type="black_scholes",
    obs_fraction=0.1,
    mu=0.0,
    sigma=0.2,
    x0=1.0
)

# Forward pass
preds, preds_before = model(batch_times, batch_values)
loss = nj_ode_loss(batch_times, batch_values, preds, preds_before)
```

### Running Experiments

```bash
# Run individual experiments with default parameters
python experiments/experiment_black_scholes.py
python experiments/experiment_ou.py
python experiments/experiment_heston.py

# Compare all experiments
python experiments/compare_experiments.py
```

### Command-Line Arguments

All experiments support the following arguments with their default values:

#### Model Architecture

```bash
--hidden-dim 32              # Hidden layer dimension
--n-hidden-layers 1          # Number of hidden layers in each network
--activation relu            # Activation function (relu/tanh/sigmoid/elu/leaky_relu/selu)
--dropout-rate 0.1           # Dropout probability for regularization
--input-scaling tanh         # Input scaling for ODE network (identity/tanh/sigmoid)
--n-steps-between 5          # Euler steps between observations
```

#### Training Parameters

```bash
--learning-rate 1e-3         # Adam learning rate
--weight-decay 5e-4          # L2 regularization weight
--n-epochs 200               # Number of training epochs
--batch-size 128             # Mini-batch size
--no-shuffle                 # Disable trajectory shuffling (default: shuffle enabled)
--print-every 5              # Print progress every N epochs
--device auto                # Device to use (auto/cpu/cuda)
```

#### Moment Learning

```bash
--num-moments 2              # Number of moments to learn (1=mean only, 2=mean+variance)
--moment-weights 1.0 3.0     # Loss weights for each moment [mean_weight, var_weight]
--shared-network             # Use single network for all moments (default: separate networks)
```

#### Data Generation

```bash
--cache-data                 # Cache data (reuse paths each epoch, default: generate fresh)
--n-train 1000               # Number of training trajectories
--n-val 200                  # Number of validation trajectories
--obs-fraction 0.1           # Fraction of time points observed (e.g., 0.1 = 10%)
--T 1.0                      # Time horizon
--n-steps 100                # Number of time steps in simulation grid
```

#### Black-Scholes Specific (dX_t = μX_t dt + σX_t dW_t)

```bash
--mu 0.1                     # Drift parameter
--sigma 0.5                  # Volatility parameter
--x0 1.0                     # Initial value
```

#### Ornstein-Uhlenbeck Specific (dX_t = θ(μ - X_t) dt + σ dW_t)

```bash
--theta 1.0                  # Mean reversion speed
--mu 0.5                     # Long-term mean
--sigma 0.3                  # Volatility parameter
--x0 0.0                     # Initial value
```

#### Heston Specific (dX_t = μX_t dt + √V_t X_t dW1_t, dV_t = κ(θ - V_t) dt + ξ√V_t dW2_t)

```bash
--mu 0.5                     # Drift parameter
--kappa 2.0                  # Volatility mean reversion speed
--theta 0.04                 # Long-term variance
--xi 0.5                     # Volatility of volatility
--rho -0.5                   # Correlation between Brownian motions
--x0 1.0                     # Initial stock price
--v0 0.04                    # Initial variance
```

#### Example Commands

```bash
# High volatility Black-Scholes with sparse observations
python experiments/experiment_black_scholes.py \
    --sigma 0.8 --obs-fraction 0.05 --n-epochs 100

# Mean-reverting OU process with shared network
python experiments/experiment_ou.py \
    --theta 2.0 --shared-network --cache-data

# Heston with strong correlation and custom moment weights
python experiments/experiment_heston.py \
    --rho -0.8 --moment-weights 1.0 5.0 --n-train 2000
```

## Model Architecture

The Neural Jump ODE handles irregularly-sampled time series by modeling:

- **Discrete jumps** at observation times via the Jump Network
- **Continuous evolution** between observations via the ODE Function
- **Predictions** at any time via the Output Network

The loss function uses L2 norms and encourages the model to:

1. Predict the observed values after jumps: ||x_i - y_i||
2. Maintain smooth evolution between observations: ||y_i - y_i^-||

## Stochastic Differential Equations

The project implements three key SDEs with grid-based simulation:

### Black Scholes

```
dX_t = μ X_t dt + σ X_t dW_t
```

- Simulated using log Euler scheme
- ~10% of fixed grid points used as observations

### Ornstein-Uhlenbeck

```
dX_t = θ(μ - X_t) dt + σ dW_t
```

- Mean-reverting process with Euler scheme
- Analytical conditional expectation for validation

### Heston

```
dX_t = μ X_t dt + √V_t X_t dW1_t
dV_t = κ(θ - V_t) dt + ξ √V_t dW2_t
```

- Stochastic volatility with correlated Brownian motions
- Euler scheme with volatility clamping

## Features

- **Multiple Moments**: Learn mean and variance simultaneously
- **Configurable Architecture**: Adjustable hidden layers, activation functions, and network dimensions
- **Relative Loss Tracking**: Compares model performance to theoretical conditional expectations
- **Training History Plots**: Track loss curves across epochs
- **Trajectory Comparison Plots**: Visualize true paths, model predictions, and conditional expectations
- **Relative Loss Plots**: Compare model performance across different processes
- **GPU Support**: Automatic CUDA detection and acceleration
- **Checkpoint Resuming**: Automatic training continuation from interruptions
- **Command-Line Interface**: Configure experiments via argparse for all parameters

## Running on HPC Clusters

The project includes SLURM job scripts for running on ETH Euler cluster:

```bash
# Initial setup on cluster
bash setup_euler.sh

# Submit single job
sbatch run_heston.sh
sbatch run_black_scholes.sh
sbatch run_ou.sh

# Submit GPU job for larger experiments
sbatch run_gpu.sh

# Run hyperparameter search (array job)
sbatch run_array_job.sh

# Monitor job progress
squeue -u $USER
tail -f logs/heston_JOBID.out

# Download results to local machine
scp -r [your_nethz]@euler.ethz.ch:~/neural-jump-ode/runs ./euler_results
```

See `EULER_GUIDE.md` for detailed instructions on cluster usage.

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.7.0
- Seaborn >= 0.11.0

## Development

This project was developed as part of a Seminar in Computational Finance course at ETH Zurich.
