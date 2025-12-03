"""
Simulation utilities for generating synthetic time series data 
to test Neural Jump ODE models.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def generate_black_scholes(
    mu: float = 0.0,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 100,
    x0: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate Black Scholes on [0, T] with n_steps using Euler on log X.
    Returns (times, X) with shape (n_steps + 1,).
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dt = T / n_steps
    times = torch.linspace(0.0, T, n_steps + 1)
    
    # Generate all random increments at once (vectorized)
    dW = torch.randn(n_steps) * torch.sqrt(torch.tensor(dt))
    
    # Vectorized computation: logX[k+1] = logX[k] + (mu - 0.5*sigma^2)*dt + sigma*dW[k]
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_terms = sigma * dW
    log_increments = drift_term + diffusion_terms
    
    # Cumulative sum to get full log process
    logX = torch.zeros(n_steps + 1)
    logX[0] = torch.log(torch.tensor(x0))
    logX[1:] = logX[0] + torch.cumsum(log_increments, dim=0)
    
    # Convert back to X
    X = torch.exp(logX)
    
    return times, X


def generate_ou(
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 0.3,
    T: float = 1.0,
    n_steps: int = 100,
    x0: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate Ornstein Uhlenbeck on [0, T] with n_steps using vectorized Euler.
    dX_t = theta (mu - X_t) dt + sigma dW_t
    Returns (times, X).
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dt = T / n_steps
    times = torch.linspace(0.0, T, n_steps + 1)
    
    # Generate all random increments at once (vectorized)
    dW = torch.randn(n_steps) * torch.sqrt(torch.tensor(dt))
    
    # Initialize process
    X = torch.zeros(n_steps + 1)
    X[0] = x0
    
    # Vectorized Euler scheme using exact solution for efficiency
    # For OU process, we can use the exact solution between time steps
    exp_decay = torch.exp(-theta * dt)
    mean_reversion = mu * (1 - exp_decay)
    noise_factor = sigma * torch.sqrt((1 - torch.exp(-2 * theta * dt)) / (2 * theta)) if theta > 0 else sigma * torch.sqrt(torch.tensor(dt))
    
    # Generate noise terms
    noise_terms = noise_factor * torch.randn(n_steps)
    
    # Recursive computation (still need loop but more efficient)
    for i in range(n_steps):
        X[i + 1] = X[i] * exp_decay + mean_reversion + noise_terms[i]
    
    return times, X


def generate_heston(
    mu: float = 0.0,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.5,
    T: float = 1.0,
    n_steps: int = 100,
    x0: float = 1.0,
    v0: float = 0.04,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate scalar Heston model on [0, T] with n_steps using vectorized Euler.
    dX_t = mu X_t dt + sqrt(V_t) X_t dW1_t
    dV_t = kappa (theta - V_t) dt + xi sqrt(V_t) dW2_t, corr(W1, W2)=rho.
    Returns (times, X, V).
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dt = T / n_steps
    times = torch.linspace(0.0, T, n_steps + 1)
    
    # Generate all correlated random increments at once (vectorized)
    z1 = torch.randn(n_steps)
    z2 = torch.randn(n_steps)
    
    # Create correlated increments
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    sqrt_1_rho2 = torch.sqrt(torch.tensor(1 - rho**2))
    
    dW1 = sqrt_dt * z1
    dW2 = sqrt_dt * (rho * z1 + sqrt_1_rho2 * z2)
    
    # Initialize processes
    X = torch.zeros(n_steps + 1)
    V = torch.zeros(n_steps + 1)
    X[0] = x0
    V[0] = v0
    
    # Euler scheme (still need loop for dependency, but more efficient)
    for i in range(n_steps):
        V_current = torch.clamp(V[i], min=1e-6)
        sqrt_V = torch.sqrt(V_current)
        
        # Update both processes
        X[i + 1] = X[i] + mu * X[i] * dt + sqrt_V * X[i] * dW1[i]
        V[i + 1] = torch.clamp(
            V[i] + kappa * (theta - V[i]) * dt + xi * sqrt_V * dW2[i],
            min=1e-6
        )
    
    return times, X, V


def subsample_random_grid_points(
    times: torch.Tensor,
    values: torch.Tensor,
    obs_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Choose about obs_fraction of grid points as observation times.
    Always include first and last grid point.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    n_grid = len(times)
    n_obs = max(2, int(obs_fraction * n_grid))
    
    # Always include first and last indices
    indices = [0, n_grid - 1]
    
    # Randomly choose remaining indices from interior
    if n_obs > 2:
        interior_indices = list(range(1, n_grid - 1))
        n_interior = min(n_obs - 2, len(interior_indices))
        selected_interior = np.random.choice(interior_indices, n_interior, replace=False)
        indices.extend(selected_interior)
    
    # Sort indices
    indices = sorted(set(indices))
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    
    return times[indices_tensor], values[indices_tensor]


def create_trajectory_batch(n_trajectories: int, process_type: str = "black_scholes",
                          obs_fraction: float = 0.1, **process_kwargs) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Create a batch of trajectories for training/testing.
    
    For process_type in {"black_scholes", "ornstein_uhlenbeck", "heston"}:
      - simulate full path on fixed grid using corresponding generator
      - subsample about obs_fraction of grid points as observations
    
    Returns:
        batch_times: List of observation time tensors
        batch_values: List of observation value tensors
    """
    batch_times = []
    batch_values = []
    
    for i in range(n_trajectories):
        if process_type == "black_scholes":
            times, values = generate_black_scholes(seed=i, **process_kwargs)
        elif process_type == "ornstein_uhlenbeck":
            times, values = generate_ou(seed=i, **process_kwargs)
        elif process_type == "heston":
            times, values, _ = generate_heston(seed=i, **process_kwargs)  # Discard V for now
        else:
            raise ValueError(f"Unknown process type: {process_type}. Supported: black_scholes, ornstein_uhlenbeck, heston")
        
        # Subsample grid points
        obs_times, obs_values = subsample_random_grid_points(
            times, values, obs_fraction, seed=i
        )
        
        batch_times.append(obs_times)
        batch_values.append(obs_values.unsqueeze(-1))  # Add dimension for d_x
    
    return batch_times, batch_values


# Conditional expectation functions for different processes

def condexp_black_scholes_on_grid(times_full: torch.Tensor, X_full: torch.Tensor, 
                                 obs_times: torch.Tensor, mu: float) -> torch.Tensor:
    """
    Piecewise conditional expectation for Black Scholes:
    For t in [T_i, T_{i+1}]:
      E[X_t | X_{T_i}] = X_{T_i} * exp(mu * (t - T_i)).
    After last obs, extend using last observation.
    """
    condexp = torch.zeros_like(times_full)
    
    for i, t in enumerate(times_full):
        # Find the latest observation time <= t
        obs_idx = torch.searchsorted(obs_times, t, right=True) - 1
        obs_idx = torch.clamp(obs_idx, min=0, max=len(obs_times) - 1)
        
        T_i = obs_times[obs_idx]
        X_i = X_full[torch.searchsorted(times_full, T_i)]
        
        # Conditional expectation formula
        condexp[i] = X_i * torch.exp(mu * (t - T_i))
    
    return condexp


def condexp_ou_on_grid(times_full: torch.Tensor, X_full: torch.Tensor, 
                       obs_times: torch.Tensor, theta: float, mu: float) -> torch.Tensor:
    """
    Piecewise conditional expectation for Ornstein-Uhlenbeck:
    For t in [T_i, T_{i+1}]:
      E[X_t | X_{T_i}] = X_{T_i} * exp(-theta * (t - T_i)) + mu * (1 - exp(-theta * (t - T_i))).
    """
    condexp = torch.zeros_like(times_full)
    
    for i, t in enumerate(times_full):
        # Find the latest observation time <= t
        obs_idx = torch.searchsorted(obs_times, t, right=True) - 1
        obs_idx = torch.clamp(obs_idx, min=0, max=len(obs_times) - 1)
        
        T_i = obs_times[obs_idx]
        X_i = X_full[torch.searchsorted(times_full, T_i)]
        
        # Conditional expectation formula for OU
        s = t - T_i
        exp_decay = torch.exp(-theta * s)
        condexp[i] = X_i * exp_decay + mu * (1 - exp_decay)
    
    return condexp


def condexp_heston_on_grid(times_full: torch.Tensor, X_full: torch.Tensor, 
                          obs_times: torch.Tensor, mu: float) -> torch.Tensor:
    """
    Piecewise conditional expectation for Heston (using simplified formula from appendix):
    For t in [T_i, T_{i+1}]:
      E[X_t | X_{T_i}] = X_{T_i} * exp(mu * (t - T_i)).
    """
    # For Heston, we use the same formula as Black-Scholes as mentioned in the appendix
    return condexp_black_scholes_on_grid(times_full, X_full, obs_times, mu)


def bs_condexp_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    mu: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    For each trajectory:
      y_i(t_k) = E[X_{t_k} | X_{T_j}] according to BS formula
      y_i^-(t_k) = limit from left, same piecewise structure as NJ ODE.
    This returns two lists with same shapes as preds and preds_before.
    """
    y_true = []
    y_true_before = []
    
    for times, values in zip(batch_times, batch_values):
        n_obs = len(times)
        y = torch.zeros_like(values)
        y_before = torch.zeros_like(values)
        
        # For each observation time
        for i in range(n_obs):
            # At observation time, conditional expectation equals observation
            y[i] = values[i]
            
            if i > 0:
                # Before observation, predict using previous observation
                dt = times[i] - times[i-1]
                y_before[i] = values[i-1] * torch.exp(mu * dt)
            else:
                # For first observation, set equal to observation
                y_before[i] = values[i]
        
        y_true.append(y)
        y_true_before.append(y_before)
    
    return y_true, y_true_before


def ou_condexp_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    theta: float,
    mu: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    For each trajectory:
      y_i(t_k) = E[X_{t_k} | X_{T_j}] according to OU formula
      y_i^-(t_k) = limit from left, same piecewise structure as NJ ODE.
    """
    y_true = []
    y_true_before = []
    
    for times, values in zip(batch_times, batch_values):
        n_obs = len(times)
        y = torch.zeros_like(values)
        y_before = torch.zeros_like(values)
        
        for i in range(n_obs):
            # At observation time, conditional expectation equals observation
            y[i] = values[i]
            
            if i > 0:
                # Before observation, predict using OU formula
                dt = times[i] - times[i-1]
                exp_decay = torch.exp(-theta * dt)
                y_before[i] = values[i-1] * exp_decay + mu * (1 - exp_decay)
            else:
                # For first observation
                y_before[i] = values[i]
        
        y_true.append(y)
        y_true_before.append(y_before)
    
    return y_true, y_true_before


def heston_condexp_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    mu: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    For each trajectory:
      y_i(t_k) = E[X_{t_k} | X_{T_j}] according to Heston formula (simplified as BS)
      y_i^-(t_k) = limit from left, same piecewise structure as NJ ODE.
    """
    # Use Black-Scholes formula as mentioned in the appendix
    return bs_condexp_at_obs(batch_times, batch_values, mu)