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
    
    # Convert parameters to tensors for consistency
    theta_tensor = torch.tensor(theta)
    mu_tensor = torch.tensor(mu)
    sigma_tensor = torch.tensor(sigma)
    
    # Vectorized Euler scheme using exact solution for efficiency
    # For OU process, we can use the exact solution between time steps
    exp_decay = torch.exp(-theta_tensor * dt)
    mean_reversion = mu_tensor * (1 - exp_decay)
    noise_factor = sigma_tensor * torch.sqrt((1 - torch.exp(-2 * theta_tensor * dt)) / (2 * theta_tensor)) if theta > 0 else sigma_tensor * torch.sqrt(dt)
    
    # Generate noise terms
    noise_terms = noise_factor * torch.randn(n_steps)
    
    # Recursive computation (still need loop but more efficient)
    for i in range(n_steps):
        X[i + 1] = X[i] * exp_decay + mean_reversion + noise_terms[i]
    
    return times, X


def generate_hybrid_ou_bs(
    theta_ou: float = 1.0,
    mu_ou: float = 0.0,
    sigma_ou: float = 0.3,
    mu_bs: float = 0.0,
    sigma_bs: float = 0.2,
    T: float = 1.0,
    n_steps: int = 100,
    x0: float = 1.0,
    switch_time: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Simulate a hybrid process: OU from [0, switch_time] and Black-Scholes from [switch_time, T].
    
    Args:
        theta_ou: mean reversion speed for OU
        mu_ou: long-term mean for OU
        sigma_ou: volatility for OU
        mu_bs: drift for Black-Scholes
        sigma_bs: volatility for Black-Scholes
        T: final time
        n_steps: number of time steps
        x0: initial value
        switch_time: time to switch from OU to BS (if None, random in [0.2*T, 0.8*T])
        seed: random seed
    
    Returns:
        (times, X, actual_switch_time) where times and X have shape (n_steps + 1,)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Determine switch time
    if switch_time is None:
        switch_time = np.random.uniform(0.2 * T, 0.8 * T)
    
    dt = T / n_steps
    times = torch.linspace(0.0, T, n_steps + 1)
    X = torch.zeros(n_steps + 1)
    X[0] = x0
    
    # Find the index where we switch
    switch_idx = int(switch_time / dt)
    
    # Phase 1: OU process [0, switch_time]
    exp_decay = torch.exp(torch.tensor(-theta_ou * dt))
    mean_reversion = mu_ou * (1 - exp_decay)
    noise_factor_ou = sigma_ou * torch.sqrt((1 - torch.exp(torch.tensor(-2 * theta_ou * dt))) / (2 * theta_ou)) if theta_ou > 0 else sigma_ou * torch.sqrt(torch.tensor(dt))
    
    for i in range(min(switch_idx, n_steps)):
        noise = noise_factor_ou * torch.randn(1).item()
        X[i + 1] = X[i] * exp_decay + mean_reversion + noise
    
    # Phase 2: Black-Scholes [switch_time, T]
    # Continue from the value at switch_time, using log-space dynamics
    if switch_idx < n_steps:
        logX = torch.log(X[switch_idx])
        drift_term = (mu_bs - 0.5 * sigma_bs**2) * dt
        
        for i in range(switch_idx, n_steps):
            dW = torch.randn(1).item() * np.sqrt(dt)
            logX = logX + drift_term + sigma_bs * dW
            X[i + 1] = torch.exp(logX)
    
    return times, X, switch_time


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
        elif process_type == "hybrid_ou_bs":
            times, values, _ = generate_hybrid_ou_bs(seed=i, **process_kwargs)  # Discard switch_time for now
        else:
            raise ValueError(f"Unknown process type: {process_type}. Supported: black_scholes, ornstein_uhlenbeck, heston, hybrid_ou_bs")
        
        # Subsample grid points
        obs_times, obs_values = subsample_random_grid_points(
            times, values, obs_fraction, seed=i
        )
        
        batch_times.append(obs_times)
        batch_values.append(obs_values.unsqueeze(-1))  # Add dimension for d_x
    
    return batch_times, batch_values


# Conditional expectation functions for different processes

def condexp_hybrid_on_grid(
    times_full: torch.Tensor,
    X_full: torch.Tensor,
    obs_times: torch.Tensor,
    switch_time: float,
    theta_ou: float,
    mu_ou: float,
    mu_bs: float,
) -> np.ndarray:
    """
    Build conditional expectation E[X_t | observations] on full grid for hybrid OU-BS process.
    
    The process switches from OU to BS at switch_time.
    The conditional expectation evolves continuously, but dynamics change at the switch.
    
    Args:
        times_full: full time grid
        X_full: full trajectory (not used for prediction, just for shape)
        obs_times: observation times
        switch_time: time when process switches from OU to BS
        theta_ou: OU mean reversion
        mu_ou: OU long-term mean
        mu_bs: BS drift
    
    Returns:
        ce: conditional expectation on full grid (numpy array)
    """
    n = len(times_full)
    ce = np.zeros(n)
    
    # Find observations and their indices
    obs_indices = []
    obs_values_dict = {}
    for obs_t in obs_times:
        idx = (torch.abs(times_full - obs_t)).argmin().item()
        obs_indices.append(idx)
        obs_values_dict[idx] = X_full[idx].item()
    
    obs_indices = sorted(obs_indices)
    
    # Find the index closest to switch_time
    switch_idx = (torch.abs(times_full - switch_time)).argmin().item()
    
    # Process each interval between observations
    for interval_idx in range(len(obs_indices)):
        start_idx = obs_indices[interval_idx]
        end_idx = obs_indices[interval_idx + 1] if interval_idx + 1 < len(obs_indices) else n
        
        # At observation point, CE equals observed value
        ce[start_idx] = obs_values_dict[start_idx]
        
        # Fill in values between observations
        for i in range(start_idx + 1, end_idx):
            t_current = times_full[i].item()
            
            # Check if we cross the switch point in this interval
            if start_idx < switch_idx <= i:
                # We need to evolve through the switch point
                # First: evolve from start to switch using OU
                t_start = times_full[start_idx].item()
                x_start = ce[start_idx]
                t_switch = times_full[switch_idx].item()
                dt_to_switch = t_switch - t_start
                
                exp_decay = np.exp(-theta_ou * dt_to_switch)
                x_at_switch = x_start * exp_decay + mu_ou * (1 - exp_decay)
                
                # Second: evolve from switch to current time using BS
                dt_from_switch = t_current - t_switch
                ce[i] = x_at_switch * np.exp(mu_bs * dt_from_switch)
            else:
                # No regime change in this step
                t_start = times_full[start_idx].item()
                x_start = ce[start_idx]
                dt = t_current - t_start
                
                if t_current < switch_time:
                    # OU regime
                    exp_decay = np.exp(-theta_ou * dt)
                    ce[i] = x_start * exp_decay + mu_ou * (1 - exp_decay)
                else:
                    # BS regime (both start and current are after switch)
                    ce[i] = x_start * np.exp(mu_bs * dt)
    
    # Handle points after last observation
    if obs_indices:
        last_obs_idx = obs_indices[-1]
        
        for i in range(last_obs_idx + 1, n):
            t_current = times_full[i].item()
            
            # Check if we cross the switch point
            if last_obs_idx < switch_idx <= i:
                # Evolve through switch point
                t_last = times_full[last_obs_idx].item()
                x_last = ce[last_obs_idx]
                t_switch = times_full[switch_idx].item()
                dt_to_switch = t_switch - t_last
                
                exp_decay = np.exp(-theta_ou * dt_to_switch)
                x_at_switch = x_last * exp_decay + mu_ou * (1 - exp_decay)
                
                dt_from_switch = t_current - t_switch
                ce[i] = x_at_switch * np.exp(mu_bs * dt_from_switch)
            else:
                # No regime change
                t_last = times_full[last_obs_idx].item()
                x_last = ce[last_obs_idx]
                dt = t_current - t_last
                
                if t_current < switch_time:
                    # OU regime
                    exp_decay = np.exp(-theta_ou * dt)
                    ce[i] = x_last * exp_decay + mu_ou * (1 - exp_decay)
                else:
                    # BS regime
                    ce[i] = x_last * np.exp(mu_bs * dt)
    
    return ce


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


def condvar_black_scholes_on_grid(times_full: torch.Tensor, X_full: torch.Tensor,
                                  obs_times: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """
    Piecewise conditional variance for Black-Scholes:
    For t in [T_i, T_{i+1}]:
      Var[X_t | X_{T_i}] = X_{T_i}^2 * (exp(sigma^2 * (t - T_i)) - 1) * exp(2 * mu * (t - T_i)).
    At observation times, variance is 0.
    """
    condvar = torch.zeros_like(times_full)
    
    for i, t in enumerate(times_full):
        # Find the latest observation time <= t
        obs_idx = torch.searchsorted(obs_times, t, right=True) - 1
        obs_idx = torch.clamp(obs_idx, min=0, max=len(obs_times) - 1)
        
        T_i = obs_times[obs_idx]
        
        # If we are exactly at an observation time, variance is 0
        if torch.isclose(t, T_i, atol=1e-6):
            condvar[i] = 0.0
        else:
            X_i = X_full[torch.searchsorted(times_full, T_i)]
            s = t - T_i
            condvar[i] = X_i**2 * (torch.exp(sigma**2 * s) - 1) * torch.exp(2 * mu * s)
    
    return condvar


def condvar_ou_on_grid(times_full: torch.Tensor, X_full: torch.Tensor,
                       obs_times: torch.Tensor, theta: float, sigma: float) -> torch.Tensor:
    """
    Piecewise conditional variance for Ornstein-Uhlenbeck:
    For t in [T_i, T_{i+1}]:
      Var[X_t | X_{T_i}] = sigma^2 / (2 * theta) * (1 - exp(-2 * theta * (t - T_i))).
    At observation times, variance is 0.
    """
    condvar = torch.zeros_like(times_full)
    
    for i, t in enumerate(times_full):
        # Find the latest observation time <= t
        obs_idx = torch.searchsorted(obs_times, t, right=True) - 1
        obs_idx = torch.clamp(obs_idx, min=0, max=len(obs_times) - 1)
        
        T_i = obs_times[obs_idx]
        
        # If we are exactly at an observation time, variance is 0
        if torch.isclose(t, T_i, atol=1e-6):
            condvar[i] = 0.0
        else:
            s = t - T_i
            condvar[i] = sigma**2 / (2 * theta) * (1 - torch.exp(-2 * theta * s))
    
    return condvar


def condvar_heston_on_grid(times_full: torch.Tensor, X_full: torch.Tensor,
                          obs_times: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """
    Piecewise conditional variance for Heston (using Black-Scholes approximation):
    For t in [T_i, T_{i+1}]:
      Var[X_t | X_{T_i}] ≈ X_{T_i}^2 * (exp(sigma^2 * (t - T_i)) - 1) * exp(2 * mu * (t - T_i)).
    """
    # For Heston, we use the same formula as Black-Scholes as an approximation
    return condvar_black_scholes_on_grid(times_full, X_full, obs_times, mu, sigma)


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


# Conditional variance functions for multiple moments
def bs_condvar_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    mu: float,
    sigma: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute conditional variance for Black-Scholes process.
    For BS: Var[X_t | X_s] = X_s^2 * (exp(sigma^2*(t-s)) - 1) * exp(2*mu*(t-s))
    """
    var_true = []
    var_true_before = []
    
    for times, values in zip(batch_times, batch_values):
        n_obs = len(times)
        var = torch.zeros_like(values)
        var_before = torch.zeros_like(values)
        
        for i in range(n_obs):
            # At observation time, variance is 0 (we know the value exactly)
            var[i] = torch.zeros_like(values[i])
            
            if i > 0:
                # Before observation, predict variance using previous observation
                dt = times[i] - times[i-1]
                X_prev = values[i-1]
                var_before[i] = X_prev**2 * (torch.exp(sigma**2 * dt) - 1) * torch.exp(2 * mu * dt)
            else:
                var_before[i] = torch.zeros_like(values[i])
        
        var_true.append(var)
        var_true_before.append(var_before)
    
    return var_true, var_true_before


def ou_condvar_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    theta: float,
    sigma: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute conditional variance for Ornstein-Uhlenbeck process.
    For OU: Var[X_t | X_s] = sigma^2/(2*theta) * (1 - exp(-2*theta*(t-s)))
    """
    var_true = []
    var_true_before = []
    
    for times, values in zip(batch_times, batch_values):
        n_obs = len(times)
        var = torch.zeros_like(values)
        var_before = torch.zeros_like(values)
        
        for i in range(n_obs):
            # At observation time, variance is 0 (we know the value exactly)
            var[i] = torch.zeros_like(values[i])
            
            if i > 0:
                # Before observation, compute conditional variance
                dt = times[i] - times[i-1]
                conditional_var = sigma**2 / (2 * theta) * (1 - torch.exp(-2 * theta * dt))
                var_before[i] = torch.full_like(values[i], conditional_var)
            else:
                var_before[i] = torch.zeros_like(values[i])
        
        var_true.append(var)
        var_true_before.append(var_before)
    
    return var_true, var_true_before


def heston_condvar_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    mu: float,
    sigma: float,  # Approximation parameter for variance
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute conditional variance for Heston process.
    Simplified approximation using Black-Scholes variance formula.
    """
    # Use Black-Scholes variance as approximation
    return bs_condvar_at_obs(batch_times, batch_values, mu, sigma)


def hybrid_condexp_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    switch_time: float,
    theta_ou: float,
    mu_ou: float,
    mu_bs: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute conditional expectation for hybrid OU-BS process at observation points.
    
    For times before switch: use OU dynamics
    For times after switch: use BS dynamics
    """
    y_true = []
    y_before = []
    
    for times, values in zip(batch_times, batch_values):
        n_obs = len(times)
        
        # Separate observations by regime
        mask_ou = times < switch_time
        mask_bs = times >= switch_time
        
        ce = torch.zeros_like(values)
        ce_before = torch.zeros_like(values)
        
        # OU regime: use OU conditional expectation
        if mask_ou.any():
            times_ou = [times[mask_ou]]
            values_ou = [values[mask_ou]]
            ce_ou, ce_ou_before = ou_condexp_at_obs(times_ou, values_ou, theta_ou, mu_ou)
            ce[mask_ou] = ce_ou[0]
            ce_before[mask_ou] = ce_ou_before[0]
        
        # BS regime: use BS conditional expectation
        if mask_bs.any():
            times_bs = [times[mask_bs]]
            values_bs = [values[mask_bs]]
            ce_bs, ce_bs_before = bs_condexp_at_obs(times_bs, values_bs, mu_bs)
            ce[mask_bs] = ce_bs[0]
            ce_before[mask_bs] = ce_bs_before[0]
        
        y_true.append(ce)
        y_before.append(ce_before)
    
    return y_true, y_before


def hybrid_condvar_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    switch_time: float,
    theta_ou: float,
    sigma_ou: float,
    mu_bs: float,
    sigma_bs: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute conditional variance for hybrid OU-BS process at observation points.
    
    Approximation: use the appropriate variance formula based on current regime.
    This is an approximation because it doesn't account for regime uncertainty.
    """
    var_true = []
    var_before = []
    
    for times, values in zip(batch_times, batch_values):
        n_obs = len(times)
        
        # Separate observations by regime
        mask_ou = times < switch_time
        mask_bs = times >= switch_time
        
        cv = torch.zeros_like(values)
        cv_before = torch.zeros_like(values)
        
        # OU regime: use OU conditional variance
        if mask_ou.any():
            times_ou = [times[mask_ou]]
            values_ou = [values[mask_ou]]
            cv_ou, cv_ou_before = ou_condvar_at_obs(times_ou, values_ou, theta_ou, sigma_ou)
            cv[mask_ou] = cv_ou[0]
            cv_before[mask_ou] = cv_ou_before[0]
        
        # BS regime: use BS conditional variance
        if mask_bs.any():
            times_bs = [times[mask_bs]]
            values_bs = [values[mask_bs]]
            cv_bs, cv_bs_before = bs_condvar_at_obs(times_bs, values_bs, mu_bs, sigma_bs)
            cv[mask_bs] = cv_bs[0]
            cv_before[mask_bs] = cv_bs_before[0]
        
        var_true.append(cv)
        var_before.append(cv_before)
    
    return var_true, var_before


def get_conditional_moments_at_obs(
    batch_times: List[torch.Tensor],
    batch_values: List[torch.Tensor],
    process_type: str,
    num_moments: int = 1,
    **process_params
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Get conditional expectations for multiple moments.
    Returns tensors of shape (n_i, d_x, num_moments).
    """
    moments_true = []
    moments_true_before = []
    
    for times, values in zip(batch_times, batch_values):
        n_obs, d_x = values.shape
        moments = torch.zeros(n_obs, d_x, num_moments)
        moments_before = torch.zeros(n_obs, d_x, num_moments)
        
        # First moment (mean)
        if process_type == "black_scholes":
            mean_true, mean_before = bs_condexp_at_obs([times], [values], process_params.get("mu", 0.0))
        elif process_type == "ornstein_uhlenbeck":
            mean_true, mean_before = ou_condexp_at_obs([times], [values], 
                                                       process_params.get("theta", 1.0), 
                                                       process_params.get("mu", 0.0))
        elif process_type == "heston":
            mean_true, mean_before = heston_condexp_at_obs([times], [values], process_params.get("mu", 0.0))
        elif process_type == "hybrid_ou_bs":
            # For hybrid with known switch time, use regime-specific formulas
            # If switch_time is None (random), we can't compute true conditional moments
            switch_time = process_params.get("switch_time")
            if switch_time is None:
                # Random switch times - return zeros (disable relative loss)
                mean_true = [torch.zeros_like(values)]
                mean_before = [torch.zeros_like(values)]
            else:
                # Fixed switch time - compute regime-specific conditional expectations
                mean_true, mean_before = hybrid_condexp_at_obs(
                    [times], [values],
                    switch_time=switch_time,
                    theta_ou=process_params.get("theta_ou", 1.0),
                    mu_ou=process_params.get("mu_ou", 0.0),
                    mu_bs=process_params.get("mu_bs", 0.0)
                )
        else:
            raise ValueError(f"Unknown process type for conditional moments: {process_type}")
        
        moments[:, :, 0] = mean_true[0]
        moments_before[:, :, 0] = mean_before[0]
        
        # Second moment (variance) if requested
        if num_moments > 1:
            if process_type == "black_scholes":
                var_true, var_before = bs_condvar_at_obs([times], [values], 
                                                         process_params.get("mu", 0.0), 
                                                         process_params.get("sigma", 0.2))
            elif process_type == "ornstein_uhlenbeck":
                var_true, var_before = ou_condvar_at_obs([times], [values], 
                                                         process_params.get("theta", 1.0),
                                                         process_params.get("sigma", 0.3))
            elif process_type == "heston":
                var_true, var_before = heston_condvar_at_obs([times], [values], 
                                                             process_params.get("mu", 0.0),
                                                             process_params.get("xi", 0.5))
            elif process_type == "hybrid_ou_bs":
                # For hybrid with known switch time, use regime-specific variance formulas
                switch_time = process_params.get("switch_time")
                if switch_time is None:
                    # Random switch times - return zeros (disable relative loss)
                    var_true = [torch.zeros_like(values)]
                    var_before = [torch.zeros_like(values)]
                else:
                    # Fixed switch time - compute regime-specific conditional variances
                    var_true, var_before = hybrid_condvar_at_obs(
                        [times], [values],
                        switch_time=switch_time,
                        theta_ou=process_params.get("theta_ou", 1.0),
                        sigma_ou=process_params.get("sigma_ou", 0.3),
                        mu_bs=process_params.get("mu_bs", 0.0),
                        sigma_bs=process_params.get("sigma_bs", 0.2)
                    )
            
            moments[:, :, 1] = var_true[0]
            moments_before[:, :, 1] = var_before[0]
        
        # Higher moments would be added here if needed
        
        moments_true.append(moments)
        moments_true_before.append(moments_before)
    
    return moments_true, moments_true_before