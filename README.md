# Neural Jump ODE

A PyTorch implementation of Neural Jump ODEs for modeling irregular time series with jumps, particularly suited for financial data and jump-diffusion processes.

## Overview

This project implements a neural network approach to modeling continuous-time processes observed at irregular intervals, with explicit handling of jumps. The model consists of three main components:

1. **Jump Network**: Maps observations to hidden states
2. **ODE Function**: Models continuous evolution between observations 
3. **Output Network**: Maps hidden states to predictions

## Project Structure

```
neural-jump-ode/
├── neural_jump_ode/          # Main package
│   ├── models/               # Model definitions
│   ├── simulation/           # Synthetic data generation
│   └── utils/                # Training utilities
├── experiments/              # Experiment scripts
├── tests/                    # Unit tests
├── runs/                     # Training outputs
└── data/                     # Data files
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

# Generate synthetic data
batch_times, batch_values = create_trajectory_batch(
    n_trajectories=10,
    process_type="jump_diffusion",
    obs_rate=8.0
)

# Forward pass
preds, preds_before = model(batch_times, batch_values)
loss = nj_ode_loss(batch_times, batch_values, preds, preds_before)
```

### Running Experiments

```bash
python experiments/basic_experiment.py
```

## Model Architecture

The Neural Jump ODE handles irregularly-sampled time series by modeling:

- **Discrete jumps** at observation times via the Jump Network
- **Continuous evolution** between observations via the ODE Function
- **Predictions** at any time via the Output Network

The loss function encourages the model to:
1. Predict the observed values after jumps
2. Maintain smooth evolution between observations

## Synthetic Data

The project includes utilities for generating various jump processes:

- **Jump-diffusion processes**: Geometric Brownian motion with Poisson jumps
- **Ornstein-Uhlenbeck with jumps**: Mean-reverting processes with jumps
- **Irregular observation sampling**: Realistic observation patterns

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
