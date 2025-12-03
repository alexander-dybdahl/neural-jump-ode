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
                 dt_between_obs=None, n_steps_between=0):
        """
        dt_between_obs: size of Euler step for interpolation between obs
        n_steps_between: number of intermediate steps between two obs times
                         if 0, only evaluate at obs times
        """
        super().__init__()
        self.jump_nn = JumpNN(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim, input_dim)
        self.output_nn = OutputNN(hidden_dim, output_dim)
        self.n_steps_between = n_steps_between
        self.dt_between_obs = dt_between_obs

    def euler_step(self, h, x_last, t_last, t_next):
        """
        Single Euler step from t_last to t_next.
        """
        dh = self.ode_func(t_next, h, x_last, t_last)
        dt = (t_next - t_last)
        return h + dt * dh

    def forward_single(self, times, values):
        """
        Forward for a single trajectory.

        times: (n_obs,) sorted tensor of observation times
        values: (n_obs, d_x) matching observations

        Returns:
            obs_pred: (n_obs, d_y) outputs at observation times
            obs_pred_before_jump: (n_obs, d_y) outputs just before jumps
        """
        n_obs, d_x = values.shape
        device = values.device

        obs_pred = []
        obs_pred_before = []

        # initialise before first obs as zero output
        y_before = torch.zeros(1, self.output_nn.net[-1].out_features,
                               device=device)

        for i in range(n_obs):
            t_i = times[i]
            x_i = values[i].unsqueeze(0)  # shape (1, d_x)

            # jump: set hidden state from observation
            h_i = self.jump_nn(x_i)  # (1, d_h)
            y_i = self.output_nn(h_i)  # (1, d_y)

            obs_pred.append(y_i.squeeze(0))
            obs_pred_before.append(y_before.squeeze(0))

            # propagate to next observation if there is one
            if i < n_obs - 1:
                t_next = times[i + 1]

                if self.n_steps_between <= 0:
                    # single step from t_i to t_next
                    h_next_minus = self.euler_step(h_i, x_i, t_i, t_next)
                else:
                    # multiple Euler substeps
                    if self.dt_between_obs is None:
                        dt = (t_next - t_i) / float(self.n_steps_between)
                    else:
                        dt = self.dt_between_obs
                    h = h_i
                    t_cur = t_i
                    while t_cur + dt < t_next:
                        t_new = t_cur + dt
                        h = self.euler_step(h, x_i, t_cur, t_new)
                        t_cur = t_new
                    # final partial step to exactly t_next
                    if t_cur < t_next:
                        h = self.euler_step(h, x_i, t_cur, t_next)
                    h_next_minus = h

                y_before = self.output_nn(h_next_minus)

        obs_pred = torch.stack(obs_pred, dim=0)
        obs_pred_before = torch.stack(obs_pred_before, dim=0)
        return obs_pred, obs_pred_before

    def forward(self, batch_times, batch_values):
        """
        batch_times: list of length B, each element (n_i,) tensor of times
        batch_values: list of length B, each element (n_i, d_x) tensor

        Returns:
            preds: list of tensors of shape (n_i, d_y)
            preds_before: list of tensors (n_i, d_y)
        """
        preds = []
        preds_before = []
        for times, values in zip(batch_times, batch_values):
            y, y_before = self.forward_single(times, values)
            preds.append(y)
            preds_before.append(y_before)
        return preds, preds_before

def nj_ode_loss(batch_times, batch_values, preds, preds_before, ignore_first_continuity=False):
    """
    Implements Phi_N from the paper:
    (||x_i - y_i|| + ||y_i - y_i^-||)^2 averaged over times and paths.
    
    Args:
        batch_times, batch_values: same as in forward
        preds, preds_before: outputs from NeuralJumpODE.forward
        ignore_first_continuity: if True, set continuity penalty to 0 at first observation
    Each element i: preds[i], preds_before[i] have shape (n_i, d_y)
    and we assume d_y = d_x (we predict the process itself).
    """
    losses = []
    for x, y, y_before in zip(batch_values, preds, preds_before):
        # x: (n_i, d_x), y: (n_i, d_x), y_before: (n_i, d_x)
        # "jump part": x - y
        # "continuous part": y - y_before

        jump = (x - y)
        cont = (y - y_before)

        jump_norm = torch.linalg.norm(jump, dim=-1)        # shape (n_i,)
        cont_norm = torch.linalg.norm(cont, dim=-1)        # shape (n_i,)

        # Optional: no continuity penalty at first point
        if ignore_first_continuity and len(cont_norm) > 0:
            cont_norm = cont_norm.clone()  # Ensure we can modify it
            cont_norm[0] = 0.0

        err = (jump_norm + cont_norm).pow(2).mean()

        losses.append(err)
    return torch.stack(losses).mean()