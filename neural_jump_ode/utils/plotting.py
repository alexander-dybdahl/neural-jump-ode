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
        condexp_ou_on_grid, condexp_heston_on_grid,
        condvar_black_scholes_on_grid, condvar_ou_on_grid, condvar_heston_on_grid
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
    
    # Build true conditional variance on full grid if model uses multiple moments
    cv_full = None
    num_moments_model = getattr(model, 'num_moments', 1)
    if num_moments_model > 1:
        if process_type == "black_scholes":
            cv_full = condvar_black_scholes_on_grid(
                times_full, X_full, obs_times, 
                process_params.get("mu", 0.0), process_params.get("sigma", 0.2)
            )
        elif process_type == "ornstein_uhlenbeck":
            cv_full = condvar_ou_on_grid(
                times_full, X_full, obs_times,
                process_params.get("theta", 1.0), process_params.get("sigma", 0.2)
            )
        elif process_type == "heston":
            cv_full = condvar_heston_on_grid(
                times_full, X_full, obs_times,
                process_params.get("mu", 0.0), process_params.get("sigma", 0.2)
            )
    
    # Build model prediction on full grid by simulating the NJ-ODE dynamics
    model.eval()
    device = next(model.parameters()).device
    num_moments = getattr(model, 'num_moments', 1)
    shared_network = getattr(model, 'shared_network', False)
    
    # We will build model_full by explicitly simulating the NJ-ODE on times_full
    model_full_mean = torch.zeros_like(times_full)
    model_full_var = torch.zeros_like(times_full) if num_moments > 1 else None
    times_full_device = times_full.to(device)
    
    with torch.no_grad():
        # Loop over intervals between observations
        for i in range(len(obs_times) - 1):
            T_i = obs_times[i]
            T_next = obs_times[i + 1]
            x_i = obs_values[i].unsqueeze(0).unsqueeze(-1).to(device)  # (1, 1)
            
            # Jump at T_i - handle multiple moments and shared/separate networks
            if shared_network:
                h = model.jump_nn(x_i)  # (1, d_h)
                h_list = [h for _ in range(num_moments)]
            else:
                h_list = [model.jump_nns[m](x_i) for m in range(num_moments)]
            t_cur = T_i.to(device)
            
            # Indices of fine times in [T_i, T_next)
            mask = (times_full >= T_i) & (times_full <= T_next)
            ts_interval = times_full_device[mask]
            idx_interval = torch.where(mask)[0]
            
            for j, t_target in enumerate(ts_interval):
                # Integrate from t_cur to t_target using Euler (with n_steps_between substeps)
                n_sub = max(1, model.n_steps_between)
                dt = (t_target - t_cur) / float(n_sub)
                for _ in range(n_sub):
                    t_new = t_cur + dt
                    h_list = model.euler_step(h_list, x_i, t_cur, t_new)
                    t_cur = t_new
                
                # Extract outputs for each moment
                if shared_network:
                    y_all = model.output_nn(h_list[0])  # (1, num_moments)
                    y_mean = y_all[:, 0:1]  # (1, 1)
                    model_full_mean[idx_interval[j]] = y_mean.squeeze().cpu()
                    if num_moments > 1:
                        y_w = y_all[:, 1:2]  # (1, 1) - raw W output
                        y_var = y_w ** 2  # V = W² to get actual variance
                        model_full_var[idx_interval[j]] = y_var.squeeze().cpu()
                else:
                    y_mean = model.output_nns[0](h_list[0])  # (1, 1)
                    model_full_mean[idx_interval[j]] = y_mean.squeeze().cpu()
                    if num_moments > 1:
                        y_w = model.output_nns[1](h_list[1])  # (1, 1) - raw W output
                        y_var = y_w ** 2  # V = W² to get actual variance
                        model_full_var[idx_interval[j]] = y_var.squeeze().cpu()        # Handle times after the last observation
        if len(obs_times) > 0:
            T_last = obs_times[-1]
            x_last = obs_values[-1].unsqueeze(0).unsqueeze(-1).to(device)  # (1, 1)
            
            # Jump at T_last - handle shared/separate networks
            if shared_network:
                h = model.jump_nn(x_last)  # (1, d_h)
                h_list = [h for _ in range(num_moments)]
            else:
                h_list = [model.jump_nns[m](x_last) for m in range(num_moments)]
            t_cur = T_last.to(device)
            
            # Indices of fine times > T_last
            mask = times_full > T_last
            ts_interval = times_full_device[mask]
            idx_interval = torch.where(mask)[0]
            
            for j, t_target in enumerate(ts_interval):
                # Integrate from t_cur to t_target
                n_sub = max(1, model.n_steps_between)
                dt = (t_target - t_cur) / float(n_sub)
                for _ in range(n_sub):
                    t_new = t_cur + dt
                    h_list = model.euler_step(h_list, x_last, t_cur, t_new)
                    t_cur = t_new
                
                # Extract outputs for each moment
                if shared_network:
                    y_all = model.output_nn(h_list[0])  # (1, num_moments)
                    y_mean = y_all[:, 0:1]  # (1, 1)
                    model_full_mean[idx_interval[j]] = y_mean.squeeze().cpu()
                    if num_moments > 1:
                        y_w = y_all[:, 1:2]  # (1, 1) - raw W output
                        y_var = y_w ** 2  # V = W² to get actual variance
                        model_full_var[idx_interval[j]] = y_var.squeeze().cpu()
                else:
                    y_mean = model.output_nns[0](h_list[0])  # (1, 1)
                    model_full_mean[idx_interval[j]] = y_mean.squeeze().cpu()
                    if num_moments > 1:
                        y_w = model.output_nns[1](h_list[1])  # (1, 1) - raw W output
                        y_var = y_w ** 2  # V = W² to get actual variance
                        model_full_var[idx_interval[j]] = y_var.squeeze().cpu()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy for plotting
    times_np = times_full.numpy()
    X_np = X_full.numpy()
    ce_np = ce_full.numpy()
    model_mean_np = model_full_mean.numpy()
    obs_times_np = obs_times.numpy()
    obs_values_np = obs_values.numpy()
    
    # Plot basic paths
    plt.plot(times_np, X_np, 'b-', label='True Path', linewidth=1.5)
    plt.plot(times_np, model_mean_np, 'r-', label='Model Mean', linewidth=1.5)
    plt.plot(times_np, ce_np, 'g:', label='True Conditional Expectation', linewidth=2)
    plt.scatter(obs_times_np, obs_values_np, c='black', s=30, label='Observations', zorder=5)
    
    # Add variance bands if available
    if model_full_var is not None:
        model_var_np = model_full_var.numpy()
        model_std_np = np.sqrt(np.maximum(model_var_np, 0))  # Ensure non-negative
        
        upper_band = model_mean_np + 2 * model_std_np
        lower_band = model_mean_np - 2 * model_std_np
        
        plt.fill_between(times_np, lower_band, upper_band, 
                        color='red', alpha=0.2, label='Model ±2σ')
        
        # Add true conditional variance bands if available
        if cv_full is not None:
            cv_np = cv_full.numpy()
            true_std_np = np.sqrt(np.maximum(cv_np, 0))  # Ensure non-negative
            
            true_upper_band = ce_np + 2 * true_std_np
            true_lower_band = ce_np - 2 * true_std_np
            
            plt.fill_between(times_np, true_lower_band, true_upper_band,
                            color='green', alpha=0.15, label='True ±2σ')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    title = f'{process_type.replace("_", " ").title()} Process - Model vs True Conditional Expectation'
    if model_full_var is not None:
        title += ' (with Variance)'
    plt.title(title)
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