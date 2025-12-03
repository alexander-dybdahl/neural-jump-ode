"""
Plotting utilities for Neural Jump ODE experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json
from pathlib import Path

def plot_training_history(history_path: str, save_path: Optional[str] = None):
    """Plot training and validation loss over epochs."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', alpha=0.7)
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch_times'], alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_single_trajectory_with_condexp(
    model,
    process_type: str,
    process_params: dict,
    obs_fraction: float = 0.1,
    seed: int = 123,
    save_path: Optional[str] = None,
):
    """
    Reproduce Figure 1 style plot.
    Steps:
      1. Simulate one full path on grid for chosen process.
      2. Subsample observation times (about 10 percent).
      3. Build true conditional expectation path on grid.
      4. Build NJ ODE prediction path on grid.
      5. Plot:
         - true path (solid)
         - model path (solid)
         - true conditional expectation (dotted)
         - observed values (dots)
    """
    from ..simulation.data_generation import (
        generate_black_scholes, generate_ou, generate_heston,
        subsample_random_grid_points, condexp_black_scholes_on_grid,
        condexp_ou_on_grid, condexp_heston_on_grid
    )
    
    # Generate one full path
    if process_type == "black_scholes":
        times_full, X_full = generate_black_scholes(seed=seed, **process_params)
    elif process_type == "ornstein_uhlenbeck":
        times_full, X_full = generate_ou(seed=seed, **process_params)
    elif process_type == "heston":
        times_full, X_full, V_full = generate_heston(seed=seed, **process_params)
    else:
        raise ValueError(f"Unknown process type: {process_type}")
    
    # Subsample observation points
    obs_times, obs_values = subsample_random_grid_points(
        times_full, X_full, obs_fraction, seed=seed
    )
    
    # Build true conditional expectation on full grid
    if process_type == "black_scholes":
        ce_full = condexp_black_scholes_on_grid(
            times_full, X_full, obs_times, process_params.get("mu", 0.0)
        )
    elif process_type == "ornstein_uhlenbeck":
        ce_full = condexp_ou_on_grid(
            times_full, X_full, obs_times, 
            process_params.get("theta", 1.0), process_params.get("mu", 0.0)
        )
    elif process_type == "heston":
        ce_full = condexp_heston_on_grid(
            times_full, X_full, obs_times, process_params.get("mu", 0.0)
        )
    
    # Build model prediction on full grid
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare observation data
    obs_times_tensor = obs_times.to(device)
    obs_values_tensor = obs_values.unsqueeze(-1).to(device)  # Add feature dimension
    
    # Get model predictions at observation times
    with torch.no_grad():
        preds, _ = model([obs_times_tensor], [obs_values_tensor])
        model_obs = preds[0].squeeze(-1).cpu()  # Remove feature dimension
    
    # Build model path on full grid without seeing the future:
    # use the last model prediction at or before each time
    model_full = torch.zeros_like(times_full)
    for i, t in enumerate(times_full):
        # index of latest observation time <= t
        idx = torch.searchsorted(obs_times, t, right=True) - 1
        idx = torch.clamp(idx, min=0, max=len(obs_times) - 1)
        model_full[i] = model_obs[idx]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy for plotting
    times_np = times_full.numpy()
    X_np = X_full.numpy()
    ce_np = ce_full.numpy()
    model_np = model_full.numpy()
    obs_times_np = obs_times.numpy()
    obs_values_np = obs_values.numpy()
    
    plt.plot(times_np, X_np, 'b-', label='True Path', linewidth=1.5)
    plt.plot(times_np, model_np, 'r-', label='Model Path', linewidth=1.5)
    plt.plot(times_np, ce_np, 'g:', label='True Conditional Expectation', linewidth=2)
    plt.scatter(obs_times_np, obs_values_np, c='black', s=30, label='Observations', zorder=5)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'{process_type.replace("_", " ").title()} Process - Model vs True Conditional Expectation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_relative_loss(history_paths: List[str], labels: List[str], 
                      save_path: Optional[str] = None):
    """
    Load multiple history.json files and plot their relative_loss vs epoch.
    For Figure 2 style visualization.
    
    Args:
        history_paths: List of paths to history.json files
        labels: List of labels for each experiment (e.g., ["Black Scholes", "Heston", "OU"])
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for history_path, label in zip(history_paths, labels):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if 'relative_loss' in history:
                epochs = range(len(history['relative_loss']))
                plt.plot(epochs, history['relative_loss'], label=label, linewidth=2)
            else:
                print(f"Warning: 'relative_loss' not found in {history_path}")
                
        except FileNotFoundError:
            print(f"Warning: History file {history_path} not found")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from {history_path}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Relative Loss (L_model - L_true) / L_true')
    plt.title('Relative Loss: Model vs True Conditional Expectation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_relative_loss_single(history_path: str, save_path: Optional[str] = None):
    """
    Load history.json and plot history["relative_loss"] vs epoch.
    Single experiment version of plot_relative_loss.
    """
    plot_relative_loss([history_path], ["Relative Loss"], save_path)