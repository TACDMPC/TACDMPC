
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch.distributions import Normal

# Importa le classi necessarie
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost


class ActorMPC(nn.Module):
    def __init__(self, nx: int, nu: int, horizon: int, dt: float, f_dyn, f_dyn_jac=None, device: str = "cpu"):
        super().__init__()
        self.nx, self.nu, self.horizon, self.dt = nx, nu, horizon, dt
        self.device = torch.device(device)
        self.dtype = torch.float32

        output_dim = horizon * (nx + nu) * 2

        # Rete senza Sigmoid finale
        self.cost_map_net = nn.Sequential(
            nn.Linear(nx, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        ).to(self.device, dtype=self.dtype)

        # Inizializzazione del modulo di costo
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

        # Creazione del controller con regolarizzazione aumentata
        self.mpc = DifferentiableMPCController(
            f_dyn=f_dyn, total_time=horizon * dt, step_size=dt,
            horizon=horizon, cost_module=cost_module,
            f_dyn_jac=f_dyn_jac, device=device,
            reg_eps=1e-2  # Regolarizzazione per la stabilità
        )

        self.log_std = nn.Parameter(torch.full((nu,), 0.0, device=self.device, dtype=self.dtype))

    def _update_cost_module(self, x: Tensor):
        batch_size = x.shape[0]
        params = self.cost_map_net(x)

        q_size = self.horizon * (self.nx + self.nu)
        p_size = self.horizon * (self.nx + self.nu)
        q_params_raw, p_params_raw = params.split([q_size, p_size], dim=-1)

        # Costruzione stabile dei costi con softplus
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

    def forward(self, x: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        self._update_cost_module(x)

        # Inizializzazione con rumore per il solver
        U_init = torch.randn(x.shape[0], self.horizon, self.nu, device=self.device, dtype=self.dtype) * 0.01

        predicted_states, predicted_actions = self.mpc(x, U_init)

        u_mpc_mean = predicted_actions[:, 0]

        std = self.log_std.exp()
        dist = Normal(u_mpc_mean, std)
        action = u_mpc_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, predicted_states, predicted_actions