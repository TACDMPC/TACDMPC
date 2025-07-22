
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from torch.distributions import Normal

# Importa le classi necessarie
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost



class ActorMPC(nn.Module):
    # Aggiunto observation_dim al costruttore ---
    def __init__(self, nx: int, nu: int, horizon: int, dt: float, f_dyn,
                 f_dyn_jac=None, device: str = "cpu",
                 grad_method: str = "auto_diff",
                 observation_dim: Optional[int] = None): # Nuovo parametro
        super().__init__()
        self.nx, self.nu, self.horizon, self.dt = nx, nu, horizon, dt
        self.device = torch.device(device)
        self.dtype = torch.float32

        # Se observation_dim non è specificato, usa nx per retrocompatibilità
        input_dim = observation_dim if observation_dim is not None else nx

        output_dim = horizon * (nx + nu) * 2

        self.cost_map_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), # Usa input_dim
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        ).to(self.device, dtype=self.dtype)

        C_placeholder = torch.zeros(horizon, nx + nu, nx + nu, device=device, dtype=self.dtype)
        c_placeholder = torch.zeros(horizon, nx + nu, device=device, dtype=self.dtype)
        C_final_placeholder = torch.zeros(nx + nu, nx + nu, device=device, dtype=self.dtype)
        c_final_placeholder = torch.zeros(nx + nu, device=device, dtype=self.dtype)

        cost_module = GeneralQuadCost(
            nx=nx, nu=nu,
            C=C_placeholder, c=c_placeholder,
            C_final=C_final_placeholder, c_final=c_final_placeholder,
            device=device
        )

        self.mpc = DifferentiableMPCController(
            f_dyn=f_dyn, total_time=horizon * dt, step_size=dt,
            horizon=horizon, cost_module=cost_module,
            f_dyn_jac=f_dyn_jac, device=device,
            reg_eps=1e-2,
            grad_method=grad_method
        )

        self.log_std = nn.Parameter(torch.full((nu,), 0.0, device=self.device, dtype=self.dtype))

    def _update_cost_module(self, x: Tensor):
        batch_size = x.shape[0]
        params = self.cost_map_net(x)

        q_size = self.horizon * (self.nx + self.nu)
        p_size = self.horizon * (self.nx + self.nu)
        q_params_raw, p_params_raw = params.split([q_size, p_size], dim=-1)

        q_diag_flat = F.softplus(q_params_raw) + 1e-2
        p_flat = p_params_raw
        q_diag = q_diag_flat.view(batch_size, self.horizon, self.nx + self.nu)
        p = p_flat.view(batch_size, self.horizon, self.nx + self.nu)

        C = torch.diag_embed(q_diag)
        c = p

        self.mpc.cost_module.C = C
        self.mpc.cost_module.c = c
        self.mpc.cost_module.C_final = C[:, -1].clone()
        self.mpc.cost_module.c_final = c[:, -1].clone()

        x_ref = torch.zeros(batch_size, self.horizon + 1, self.nx, device=self.device, dtype=self.dtype)
        u_ref = torch.zeros(batch_size, self.horizon, self.nu, device=self.device, dtype=self.dtype)
        self.mpc.cost_module.set_reference(x_ref, u_ref)

    def forward(self, x: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        self._update_cost_module(x)

        U_init = torch.randn(x.shape[0], self.horizon, self.nu, device=self.device, dtype=self.dtype) * 0.01

        # Passa solo la parte di stato fisico [x, y, theta] all'MPC
        predicted_states, predicted_actions = self.mpc(x[:, :self.nx], U_init)

        u_mpc_mean = predicted_actions[:, 0]

        std = self.log_std.exp()
        dist = Normal(u_mpc_mean, std)
        action = u_mpc_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, predicted_states, predicted_actions

    def evaluate_actions(self, x: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        self._update_cost_module(x)

        U_init = torch.zeros(x.shape[0], self.horizon, self.nu, device=self.device, dtype=self.dtype)
        # Passa solo la parte di stato fisico [x, y, theta] all'MPC
        _, predicted_actions = self.mpc(x[:, :self.nx], U_init)
        u_mpc_mean = predicted_actions[:, 0]

        std = self.log_std.exp()
        dist = Normal(u_mpc_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
