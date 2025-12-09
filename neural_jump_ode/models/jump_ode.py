import torch
import torch.nn as nn
from torch.nn import functional as F

# Activation function mapping
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU,
    'selu': nn.SELU,
}

class JumpNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden_layers=1, activation='relu'):
        super().__init__()
        act_fn = ACTIVATION_FUNCTIONS.get(activation.lower(), nn.ReLU)
        layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, d_x)
        return self.net(x)


class ODEFunc(nn.Module):
    """
    f_theta(h, x_last, t_last, dt_elapsed) -> dh/dt
    """
    def __init__(self, hidden_dim, input_dim, n_hidden_layers=1, activation='relu'):
        super().__init__()
        act_fn = ACTIVATION_FUNCTIONS.get(activation.lower(), nn.ReLU)
        layers = [nn.Linear(hidden_dim + input_dim + 2, hidden_dim), act_fn()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, h, x_last, t_last):
        # t, t_last: scalar tensors
        # h: (batch, d_h)
        # x_last: (batch, d_x)
        t_elapsed = (t - t_last).expand_as(h[..., :1])
        t_rel = (t_last).expand_as(h[..., :1])
        inp = torch.cat([h, x_last, t_rel, t_elapsed], dim=-1)
        dh = self.net(inp)
        return dh


class OutputNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_hidden_layers=1, activation='relu'):
        super().__init__()
        act_fn = ACTIVATION_FUNCTIONS.get(activation.lower(), nn.ReLU)
        layers = []
        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class NeuralJumpODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dt_between_obs=None, n_steps_between=0, num_moments=1, n_hidden_layers=1, activation='relu',
                 shared_network=False):
        """
        dt_between_obs: size of Euler step for interpolation between obs
        n_steps_between: number of intermediate steps between two obs times
                         if 0, only evaluate at obs times
        num_moments: number of moments to learn (1=mean only, 2=mean+variance, etc.)
        n_hidden_layers: number of hidden layers in each neural network component (default=1)
        activation: activation function to use ('relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu', 'selu')
        shared_network: if True, use single shared network for all moments; if False, separate networks (default=False)
        """
        super().__init__()
        self.num_moments = num_moments
        self.shared_network = shared_network
        
        if shared_network:
            # Single shared network for all moments
            self.jump_nn = JumpNN(input_dim, hidden_dim, n_hidden_layers, activation)
            self.ode_func = ODEFunc(hidden_dim, input_dim, n_hidden_layers, activation)
            # Output network needs to output all moments at once
            self.output_nn = OutputNN(hidden_dim, output_dim * num_moments, n_hidden_layers, activation)
            self.jump_nns = None
            self.ode_funcs = None
            self.output_nns = None
        else:
            # Create separate networks for each moment
            self.jump_nns = nn.ModuleList([JumpNN(input_dim, hidden_dim, n_hidden_layers, activation) for _ in range(num_moments)])
            self.ode_funcs = nn.ModuleList([ODEFunc(hidden_dim, input_dim, n_hidden_layers, activation) for _ in range(num_moments)])
            self.output_nns = nn.ModuleList([OutputNN(hidden_dim, output_dim, n_hidden_layers, activation) for _ in range(num_moments)])
            self.jump_nn = None
            self.ode_func = None
            self.output_nn = None
        
        self.n_steps_between = n_steps_between
        self.dt_between_obs = dt_between_obs
        self.output_dim = output_dim

    def euler_step(self, h_list, x_last, t_last, t_next):
        """
        Single Euler step from t_last to t_next for all moments.
        """
        if self.shared_network:
            # Single hidden state for shared network
            h = h_list[0]
            dh = self.ode_func(t_next, h, x_last, t_last)
            dt = (t_next - t_last)
            new_h = h + dt * dh
            return [new_h]  # Return as list for consistency
        else:
            # Separate hidden states for each moment
            new_h_list = []
            for i in range(self.num_moments):
                dh = self.ode_funcs[i](t_next, h_list[i], x_last, t_last)
                dt = (t_next - t_last)
                new_h_list.append(h_list[i] + dt * dh)
            return new_h_list

    def forward_single(self, times, values):
        """
        Forward for a single trajectory.

        times: (n_obs,) sorted tensor of observation times
        values: (n_obs, d_x) matching observations

        Returns:
            obs_pred: (n_obs, d_y, num_moments) outputs at observation times for each moment
            obs_pred_before_jump: (n_obs, d_y, num_moments) outputs just before jumps for each moment
        """
        n_obs, d_x = values.shape
        device = values.device
        d_y = self.output_dim

        obs_pred = []
        obs_pred_before = []

        # initialise before first obs as zero output for all moments
        y_before = torch.zeros(1, d_y, self.num_moments, device=device)

        for i in range(n_obs):
            t_i = times[i]
            x_i = values[i].unsqueeze(0)  # shape (1, d_x)

            if self.shared_network:
                # Shared network: single hidden state, multi-output
                h = self.jump_nn(x_i)  # (1, d_h)
                y_flat = self.output_nn(h)  # (1, d_y * num_moments)
                # Reshape to (1, d_y, num_moments)
                y_i = y_flat.view(1, d_y, self.num_moments)
                h_list = [h]  # Store as list for consistency
            else:
                # Separate networks: each moment has its own network
                h_list = [self.jump_nns[m](x_i) for m in range(self.num_moments)]  # List of (1, d_h)
                y_list = [self.output_nns[m](h_list[m]) for m in range(self.num_moments)]  # List of (1, d_y)
                # Stack moments: (1, d_y, num_moments)
                y_i = torch.stack(y_list, dim=-1)

            obs_pred.append(y_i.squeeze(0))  # (d_y, num_moments)
            obs_pred_before.append(y_before.squeeze(0))  # (d_y, num_moments)

            # propagate to next observation if there is one
            if i < n_obs - 1:
                t_next = times[i + 1]

                if self.n_steps_between <= 0:
                    # single step from t_i to t_next
                    h_next_minus_list = self.euler_step(h_list, x_i, t_i, t_next)
                else:
                    # multiple Euler substeps
                    if self.dt_between_obs is None:
                        dt = (t_next - t_i) / float(self.n_steps_between)
                    else:
                        dt = self.dt_between_obs
                    h_cur_list = h_list
                    t_cur = t_i
                    while t_cur + dt < t_next:
                        t_new = t_cur + dt
                        h_cur_list = self.euler_step(h_cur_list, x_i, t_cur, t_new)
                        t_cur = t_new
                    # final partial step to exactly t_next
                    if t_cur < t_next:
                        h_cur_list = self.euler_step(h_cur_list, x_i, t_cur, t_next)
                    h_next_minus_list = h_cur_list

                if self.shared_network:
                    # Shared network: single output
                    y_before_flat = self.output_nn(h_next_minus_list[0])  # (1, d_y * num_moments)
                    y_before = y_before_flat.view(1, d_y, self.num_moments)
                else:
                    # Separate networks: each moment separately
                    y_before_list = [self.output_nns[m](h_next_minus_list[m]) for m in range(self.num_moments)]
                    y_before = torch.stack(y_before_list, dim=-1)  # (1, d_y, num_moments)

        obs_pred = torch.stack(obs_pred, dim=0)  # (n_obs, d_y, num_moments)
        obs_pred_before = torch.stack(obs_pred_before, dim=0)  # (n_obs, d_y, num_moments)
        return obs_pred, obs_pred_before

    def forward(self, batch_times, batch_values):
        """
        batch_times: list of length B, each element (n_i,) tensor of times
        batch_values: list of length B, each element (n_i, d_x) tensor

        Returns:
            preds: list of tensors of shape (n_i, d_y, num_moments)
            preds_before: list of tensors (n_i, d_y, num_moments)
        """
        preds = []
        preds_before = []
        for times, values in zip(batch_times, batch_values):
            y, y_before = self.forward_single(times, values)
            preds.append(y)
            preds_before.append(y_before)
        return preds, preds_before

def nj_ode_loss(
    batch_times,
    batch_values,
    preds,
    preds_before,
    ignore_first_continuity: bool = False,
    moment_weights=None,
    eps: float = 1e-10,
):
    """
    Neural Jump ODE loss function for mean and variance prediction.

    The loss has two components for each moment:
    1. Jump term: How well predictions match observations after jumps
    2. Continuity term: How smooth predictions are before jumps
    
    For mean (moment 0): Predicts E[X_t | observations]
    For variance (moment 1): Predicts Var[X_t | observations] via W where V = W²

    Args:
        batch_times: List of observation time tensors for each trajectory
        batch_values: List of observation value tensors (true X values)
        preds: Model predictions at observation times (after jump)
        preds_before: Model predictions just before observation times
        ignore_first_continuity: If True, no continuity penalty at t=0 (default: False)
        moment_weights: Weights for each moment's loss [mean_weight, var_weight, ...]
        weight: Balance between jump (weight) and continuity (1-weight) terms (default: 0.5)
        eps: Small constant for numerical stability in square roots (default: 1e-10)
    
    Returns:
        Average loss across all trajectories in the batch
    """
    if moment_weights is not None and not torch.is_tensor(moment_weights):
        moment_weights = torch.tensor(moment_weights, device=preds[0].device)

    trajectory_losses = []

    for true_values, pred_after_jump, pred_before_jump in zip(batch_values, preds, preds_before):
        # true_values: (n_observations, dimension)
        # pred_after_jump, pred_before_jump: (n_observations, dimension, num_moments)
        n_obs, d_x, num_moments = pred_after_jump.shape

        total_loss_for_trajectory = 0.0

        # ========== MOMENT 0: Mean Prediction ==========
        pred_mean_after = pred_after_jump[:, :, 0]     # Y_t:  predictions after jump
        pred_mean_before = pred_before_jump[:, :, 0]   # Y_t-: predictions before jump

        # Jump term: || X_t - Y_t ||²  (how well we match observations)
        squared_error_after_jump = torch.sum((true_values - pred_mean_after) ** 2, dim=1)  # (n_obs,)
        
        # Continuity term: || Y_t- - X_t ||²  (smoothness before jump)
        squared_error_before_jump = torch.sum((pred_mean_before - true_values) ** 2, dim=1)  # (n_obs,)

        # Ignore continuity at first observation (no "before" exists)
        if ignore_first_continuity and n_obs > 0:
            squared_error_before_jump = squared_error_before_jump.clone()
            squared_error_before_jump[0] = 0.0

        # Combine with weights: inner = (2w*sqrt(jump) + 2(1-w)*sqrt(continuity))²
        mean_loss_per_obs = (torch.sqrt(squared_error_after_jump + eps) + torch.sqrt(squared_error_before_jump + eps)) ** 2  # (n_obs,)
        mean_loss = mean_loss_per_obs.mean()  # Average over observations

        # Apply moment weight for mean
        mean_weight = 1.0 if moment_weights is None else moment_weights[0]
        total_loss_for_trajectory = total_loss_for_trajectory + mean_weight * mean_loss

        # ========== MOMENT 1: Variance Prediction (if learning 2+ moments) ==========
        if num_moments > 1:
            pred_var_raw_after = pred_after_jump[:, :, 1]    # W_t:  raw variance output after jump
            pred_var_raw_before = pred_before_jump[:, :, 1]  # W_t-: raw variance output before jump

            # Actual variance predictions (squared to ensure non-negativity)
            pred_variance_after = pred_var_raw_after ** 2    # V_t  = W_t²
            pred_variance_before = pred_var_raw_before ** 2  # V_t- = W_t-²

            # True variance target: (X - mean_prediction)²
            # Detach mean predictions so variance loss doesn't affect mean learning
            pred_mean_after_detached = pred_mean_after.detach()
            pred_mean_before_detached = pred_mean_before.detach()

            true_variance_after = (true_values - pred_mean_after_detached) ** 2    # (n_obs, d_x)
            true_variance_before = (true_values - pred_mean_before_detached) ** 2  # (n_obs, d_x)

            # Jump term for variance: || true_var - predicted_var ||²
            var_squared_error_after = torch.sum((true_variance_after - pred_variance_after) ** 2, dim=1)  # (n_obs,)
            
            # Continuity term for variance: || true_var- - predicted_var- ||²
            var_squared_error_before = torch.sum((true_variance_before - pred_variance_before) ** 2, dim=1)  # (n_obs,)

            # Ignore first observation continuity
            if ignore_first_continuity and n_obs > 0:
                var_squared_error_before = var_squared_error_before.clone()
                var_squared_error_before[0] = 0.0

            # Combine with weights (same formula as mean)
            variance_loss_per_obs = (torch.sqrt(var_squared_error_after + eps) + torch.sqrt(var_squared_error_before + eps)) ** 2
            variance_loss = variance_loss_per_obs.mean()

            # Apply moment weight for variance (typically > 1 to emphasize variance learning)
            variance_weight = 1.0 if moment_weights is None else moment_weights[1]
            total_loss_for_trajectory = total_loss_for_trajectory + variance_weight * variance_loss

        trajectory_losses.append(total_loss_for_trajectory)

    # Average loss across all trajectories in batch
    return torch.stack(trajectory_losses).mean()

