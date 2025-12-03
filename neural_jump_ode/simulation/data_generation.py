"""
Simulation utilities for generating synthetic time series data 
to test Neural Jump ODE models.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def generate_poisson_jumps(rate: float, T: float, seed: Optional[int] = None) -> torch.Tensor:
    """Generate jump times from a Poisson process using vectorized operations."""
    if seed is not None:
        np.random.seed(seed)
    
    # Estimate number of jumps (with some buffer) for vectorized generation
    expected_jumps = int(rate * T * 1.5 + 10)  # Buffer to avoid resampling
    
    # Generate all inter-arrival times at once
    inter_arrivals = np.random.exponential(1.0 / rate, expected_jumps)
    
    # Compute cumulative arrival times
    arrival_times = np.cumsum(inter_arrivals)
    
    # Keep only times within [0, T]
    valid_times = arrival_times[arrival_times < T]
    
    return torch.tensor(valid_times, dtype=torch.float32)


def generate_brownian_motion(n_steps: int, dt: float, d: int = 1, 
                           seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Brownian motion path."""
    if seed is not None:
        torch.manual_seed(seed)
    
    dW = torch.randn(n_steps, d) * np.sqrt(dt)
    W = torch.cumsum(dW, dim=0)
    times = torch.arange(n_steps, dtype=torch.float32) * dt
    
    return times, W


def generate_jump_diffusion(jump_rate: float = 2.0, drift: float = 0.05, 
                          vol: float = 0.2, jump_mean: float = 0.0, 
                          jump_std: float = 0.1, T: float = 1.0, 
                          dt: float = 0.01, x0: float = 1.0,
                          seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a jump-diffusion process:
    dX_t = drift * X_t * dt + vol * X_t * dW_t + X_{t-} * J * dN_t
    
    where N_t is Poisson(jump_rate) and J ~ N(jump_mean, jump_std^2)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate fine grid for simulation
    times_fine = torch.arange(0, T + dt, dt)
    n_steps = len(times_fine)
    
    # Initialize process
    X = torch.zeros(n_steps)
    X[0] = x0
    
    # Generate Brownian motion
    dW = torch.randn(n_steps - 1) * np.sqrt(dt)
    
    # Generate jump times and sizes
    jump_times = generate_poisson_jumps(jump_rate, T, seed)
    jump_sizes = torch.normal(jump_mean, jump_std, (len(jump_times),))
    
    # Vectorized simulation of continuous part
    log_returns = drift * dt + vol * dW
    X[1:] = X[0] * torch.exp(torch.cumsum(log_returns, dim=0))
    
    # Apply jumps efficiently using vectorized operations
    if len(jump_times) > 0:
        # Find which time step each jump occurs in
        jump_indices = torch.searchsorted(times_fine[1:], jump_times, right=False) + 1  # +1 because we search in [1:]
        jump_indices = torch.clamp(jump_indices, 1, n_steps - 1)  # Ensure valid indices
        
        # Apply jumps in chronological order
        for jump_idx, jump_size in zip(jump_indices, jump_sizes):
            X[jump_idx:] *= (1 + jump_size)
    
    return times_fine, X


def generate_ou_with_jumps(theta: float = 1.0, mu: float = 0.0, sigma: float = 0.3,
                          jump_rate: float = 1.5, jump_mean: float = 0.0, 
                          jump_std: float = 0.2, T: float = 1.0, dt: float = 0.01,
                          x0: float = 0.0, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Ornstein-Uhlenbeck process with jumps:
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t + J * dN_t
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    times_fine = torch.arange(0, T + dt, dt)
    n_steps = len(times_fine)
    
    X = torch.zeros(n_steps)
    X[0] = x0
    
    # Generate Brownian increments
    dW = torch.randn(n_steps - 1) * np.sqrt(dt)
    
    # Generate jumps
    jump_times = generate_poisson_jumps(jump_rate, T, seed)
    jump_sizes = torch.normal(jump_mean, jump_std, (len(jump_times),))
    
    # Vectorized OU simulation using exact solution
    # X(t+dt) = X(t)*exp(-theta*dt) + mu*(1-exp(-theta*dt)) + integral of noise
    exp_decay = np.exp(-theta * dt)
    mean_reversion = mu * (1 - exp_decay)
    noise_factor = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta)) if theta > 0 else sigma * np.sqrt(dt)
    
    # Generate the OU process
    for i in range(1, n_steps):
        X[i] = X[i-1] * exp_decay + mean_reversion + noise_factor * torch.randn(1).item()
    
    # Apply jumps efficiently
    if len(jump_times) > 0:
        jump_indices = torch.searchsorted(times_fine[1:], jump_times, right=False) + 1
        jump_indices = torch.clamp(jump_indices, 1, n_steps - 1)
        
        for jump_idx, jump_size in zip(jump_indices, jump_sizes):
            X[jump_idx:] += jump_size
    
    return times_fine, X


def subsample_trajectory(times: torch.Tensor, values: torch.Tensor, 
                        obs_rate: float = 10.0, irregular: bool = False,
                        seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample a trajectory to create irregular observation times.
    
    Args:
        times: Full time grid
        values: Process values on full grid
        obs_rate: Average number of observations per unit time
        irregular: If True, use random observation times, else regular grid
    """
    if irregular:
        # Generate random observation times
        T = times[-1].item()
        if seed is not None:
            np.random.seed(seed)
        obs_times = generate_poisson_jumps(obs_rate, T, seed)
        obs_times = torch.cat([torch.tensor([0.0]), obs_times])  # Always include t=0
    else:
        # Regular grid
        dt_obs = 1.0 / obs_rate
        obs_times = torch.arange(0, times[-1] + dt_obs, dt_obs)
    
    # Efficient vectorized interpolation using searchsorted
    # Find indices for interpolation
    indices = torch.searchsorted(times, obs_times, right=False)
    
    # Handle boundary cases
    indices = torch.clamp(indices, 1, len(times) - 1)
    
    # Get left and right time points and values
    t_left = times[indices - 1]
    t_right = times[indices]
    v_left = values[indices - 1]
    v_right = values[indices]
    
    # Compute interpolation weights vectorized
    weights = (obs_times - t_left) / (t_right - t_left + 1e-10)  # Small epsilon to avoid division by zero
    weights = torch.clamp(weights, 0, 1)  # Ensure weights are in [0,1]
    
    # Linear interpolation
    obs_values = v_left + weights * (v_right - v_left)
    
    # Handle exact boundary matches
    obs_values[obs_times <= times[0]] = values[0]
    obs_values[obs_times >= times[-1]] = values[-1]
    
    return obs_times, obs_values


def create_trajectory_batch(n_trajectories: int, process_type: str = "jump_diffusion",
                          obs_rate: float = 10.0, irregular: bool = True,
                          **process_kwargs) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Create a batch of trajectories for training/testing.
    
    Returns:
        batch_times: List of observation time tensors
        batch_values: List of observation value tensors
    """
    batch_times = []
    batch_values = []
    
    for i in range(n_trajectories):
        # Generate full trajectory
        if process_type == "jump_diffusion":
            times, values = generate_jump_diffusion(seed=i, **process_kwargs)
        elif process_type == "ou_jumps":
            times, values = generate_ou_with_jumps(seed=i, **process_kwargs)
        else:
            raise ValueError(f"Unknown process type: {process_type}")
        
        # Subsample to create observations
        obs_times, obs_values = subsample_trajectory(
            times, values, obs_rate, irregular, seed=i
        )
        
        batch_times.append(obs_times)
        batch_values.append(obs_values.unsqueeze(-1))  # Add dimension for d_x
    
    return batch_times, batch_values