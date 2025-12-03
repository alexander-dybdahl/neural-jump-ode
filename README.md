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
# Run individual experiments
python experiments/experiment_black_scholes.py
python experiments/experiment_ou.py
python experiments/experiment_heston.py

# Compare all experiments
python experiments/compare_experiments.py
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

- **Relative Loss Tracking**: Compares model performance to theoretical conditional expectations
- **Figure 1 Style Plots**: Trajectory comparisons with true paths, model predictions, and conditional expectations  
- **Figure 2 Style Plots**: Relative loss curves across different processes
- **GPU Support**: Automatic CUDA detection and acceleration
- **Checkpoint Resuming**: Automatic training continuation from interruptions

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.7.0

## Development

This project was developed as part of the Seminar in Computational Finance for CSE at ETH Zurich.

## License

[Add license information]
