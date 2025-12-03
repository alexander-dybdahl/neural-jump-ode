"""
Simulation utilities for generating synthetic time series data 
to test Neural Jump ODE models.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def generate_poisson_jumps(rate: float, T: float, seed: Optional[int] = None) -> torch.Tensor:
    """Generate jump times from a Poisson process."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate inter-arrival times
    times = []
    t = 0.0
    while t < T:
        t += np.random.exponential(1.0 / rate)
        if t < T:
            times.append(t)
    
    return torch.tensor(times, dtype=torch.float32)


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
    
    # Simulate the process
    jump_idx = 0
    for i in range(1, n_steps):
        t = times_fine[i]
        
        # Continuous part (Geometric Brownian Motion)
        X[i] = X[i-1] * (1 + drift * dt + vol * dW[i-1])
        
        # Jump part
        while jump_idx < len(jump_times) and jump_times[jump_idx] <= t:
            X[i] *= (1 + jump_sizes[jump_idx])
            jump_idx += 1
    
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
    
    # Simulate
    jump_idx = 0
    for i in range(1, n_steps):
        t = times_fine[i]
        
        # OU evolution
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW[i-1]
        
        # Jumps
        while jump_idx < len(jump_times) and jump_times[jump_idx] <= t:
            X[i] += jump_sizes[jump_idx]
            jump_idx += 1
    
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
    
    # Interpolate values at observation times
    obs_values = torch.interp(obs_times, times, values)
    
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