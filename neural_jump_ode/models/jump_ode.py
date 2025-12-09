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
    weight: float = 0.5,
    eps: float = 1e-10,
):
    """
    NJODE loss in the style of compute_loss_2 / compute_var_loss (type 1)
    for mean and variance heads.

    Moments:
      - moment 0: Y  (conditional mean of X)
      - moment 1: W  (raw variance head, V = W^2 is conditional variance of X)

    Args:
        batch_times, batch_values: lists as returned by the data loader
        preds, preds_before: outputs from NeuralJumpODE.forward
            each element has shape (n_i, d_y, num_moments)
        ignore_first_continuity: if True, continuity term at first obs is set to 0
        moment_weights: optional length-num_moments tensor of weights
        weight: 'w' in the paper, usually 0.5
        eps: small constant for numerical stability in square roots
    """
    if moment_weights is not None and not torch.is_tensor(moment_weights):
        moment_weights = torch.tensor(moment_weights, device=preds[0].device)

    losses = []

    for x, y, y_before in zip(batch_values, preds, preds_before):
        # x: (n_i, d_x)
        # y, y_before: (n_i, d_x, num_moments)
        n_obs, d_x, num_moments = y.shape
        device = x.device

        total_loss = 0.0

        # ---------- Mean head (Y) ----------
        Y = y[:, :, 0]          # (n_i, d_x)
        Y_bj = y_before[:, :, 0]

        # jump term: X_t - Y_t
        sq_jump = torch.sum((x - Y) ** 2, dim=1)              # (n_i,)
        # continuity term: Y_{t-} - X_t  (same as X_t - Y_{t-} up to sign)
        sq_cont = torch.sum((Y_bj - x) ** 2, dim=1)           # (n_i,)

        if ignore_first_continuity and n_obs > 0:
            sq_cont = sq_cont.clone()
            sq_cont[0] = 0.0

        jump_term = 2.0 * weight * torch.sqrt(sq_jump + eps)
        cont_term = 2.0 * (1.0 - weight) * torch.sqrt(sq_cont + eps)

        inner_mean = (jump_term + cont_term) ** 2             # (n_i,)
        loss_mean = inner_mean.mean()                         # ≈ Σ inner / n_obs

        w_mean = 1.0 if moment_weights is None else moment_weights[0]
        total_loss = total_loss + w_mean * loss_mean

        # ---------- Variance head (W -> V = W²) ----------
        if num_moments > 1:
            W = y[:, :, 1]          # (n_i, d_x)
            W_bj = y_before[:, :, 1]

            V = W ** 2
            V_bj = W_bj ** 2

            # detach mean, as in paper: Z = (X - Y_detached)², Z_- = (X - Y_-_detached)²
            Y_det = Y.detach()
            Y_bj_det = Y_bj.detach()

            true_var = (x - Y_det) ** 2          # (n_i, d_x)
            true_var_bj = (x - Y_bj_det) ** 2   # (n_i, d_x)

            # || Z_t - V_t || and || Z_{t-} - V_{t-} ||
            sq_jump_var = torch.sum((true_var - V) ** 2, dim=1)          # (n_i,)
            sq_cont_var = torch.sum((true_var_bj - V_bj) ** 2, dim=1)    # (n_i,)

            if ignore_first_continuity and n_obs > 0:
                sq_cont_var = sq_cont_var.clone()
                sq_cont_var[0] = 0.0

            jump_var_term = 2.0 * weight * torch.sqrt(sq_jump_var + eps)
            cont_var_term = 2.0 * (1.0 - weight) * torch.sqrt(sq_cont_var + eps)

            inner_var = (jump_var_term + cont_var_term) ** 2
            loss_var = inner_var.mean()

            w_var = 1.0 if moment_weights is None else moment_weights[1]
            total_loss = total_loss + w_var * loss_var

        losses.append(total_loss)

    return torch.stack(losses).mean()
