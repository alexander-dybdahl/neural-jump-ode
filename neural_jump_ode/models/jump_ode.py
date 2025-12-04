import torch
import torch.nn as nn
from torch.nn import functional as F

class JumpNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (batch, d_x)
        return self.net(x)


class ODEFunc(nn.Module):
    """
    f_theta(h, x_last, t_last, dt_elapsed) -> dh/dt
    """
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h):
        return self.net(h)


class NeuralJumpODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dt_between_obs=None, n_steps_between=0, num_moments=1):
        """
        dt_between_obs: size of Euler step for interpolation between obs
        n_steps_between: number of intermediate steps between two obs times
                         if 0, only evaluate at obs times
        num_moments: number of moments to learn (1=mean only, 2=mean+variance, etc.)
        """
        super().__init__()
        self.num_moments = num_moments
        
        # Create separate networks for each moment
        self.jump_nns = nn.ModuleList([JumpNN(input_dim, hidden_dim) for _ in range(num_moments)])
        self.ode_funcs = nn.ModuleList([ODEFunc(hidden_dim, input_dim) for _ in range(num_moments)])
        self.output_nns = nn.ModuleList([OutputNN(hidden_dim, output_dim) for _ in range(num_moments)])
        
        self.n_steps_between = n_steps_between
        self.dt_between_obs = dt_between_obs

    def euler_step(self, h_list, x_last, t_last, t_next):
        """
        Single Euler step from t_last to t_next for all moments.
        """
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
        d_y = self.output_nns[0].net[-1].out_features

        obs_pred = []
        obs_pred_before = []

        # initialise before first obs as zero output for all moments
        y_before = torch.zeros(1, d_y, self.num_moments, device=device)

        for i in range(n_obs):
            t_i = times[i]
            x_i = values[i].unsqueeze(0)  # shape (1, d_x)

            # jump: set hidden state from observation for each moment
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

def nj_ode_loss(batch_times, batch_values, preds, preds_before, ignore_first_continuity=False, moment_weights=None):
    """
    Implements Phi_N from the paper for multiple moments:
    (||x_i - y_i|| + ||y_i - y_i^-||)^2 averaged over times, paths, and moments.
    
    Args:
        batch_times, batch_values: same as in forward
        preds, preds_before: outputs from NeuralJumpODE.forward
        ignore_first_continuity: if True, set continuity penalty to 0 at first observation
        moment_weights: optional tensor of shape (num_moments,) to weight different moments
    Each element i: preds[i], preds_before[i] have shape (n_i, d_y, num_moments)
    For moment k=0 (mean), we compare with x directly.
    For moment k=1 (variance), we compare with (x - mean_pred)^2.
    """
    losses = []
    for i, (times, x, y, y_before) in enumerate(zip(batch_times, batch_values, preds, preds_before)):
        # x: (n_i, d_x), y: (n_i, d_x, num_moments), y_before: (n_i, d_x, num_moments)
        n_obs, d_x, num_moments = y.shape
        
        total_loss = 0.0
        
        for moment in range(num_moments):
            y_m = y[:, :, moment]  # (n_i, d_x)
            y_before_m = y_before[:, :, moment]  # (n_i, d_x)
            
            if moment == 0:
                # First moment (mean): compare with x directly
                target = x
            elif moment == 1:
                # Second moment (variance): compare with (x - mean_pred)^2
                mean_pred = y[:, :, 0]  # Use predicted mean
                target = (x - mean_pred) ** 2
            else:
                # Higher moments: (x - mean_pred)^moment
                mean_pred = y[:, :, 0]  # Use predicted mean
                target = torch.abs(x - mean_pred) ** (moment + 1)
            
            # "jump part": target - y_m
            jump = (target - y_m)
            # "continuous part": y_m - y_before_m
            cont = (y_m - y_before_m)

            jump_norm = torch.norm(jump, dim=1)  # (n_i,)
            cont_norm = torch.norm(cont, dim=1)  # (n_i,)
            
            # Ignore first continuity if requested
            if ignore_first_continuity and len(cont_norm) > 0:
                cont_norm = cont_norm.clone()  # Avoid in-place modification
                cont_norm[0] = 0.0
            
            # Weight by moment importance
            weight = 1.0 if moment_weights is None else moment_weights[moment]
            total_loss += weight * torch.mean(jump_norm + cont_norm)
        
        losses.append(total_loss)

    return torch.stack(losses).mean()