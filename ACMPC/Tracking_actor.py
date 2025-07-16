# File: Trackingactor.py (Versione Modificata per Tracking)

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from DifferentialMPC import DifferentiableMPCController, GradMethod, GeneralQuadCost


class ActorMPC(nn.Module):
    """
    ActorMPC modificato per il TRACKING.

    La "Neural Cost Map" impara i parametri di costo (Q, p) che sono ottimali
    per far sì che il sistema segua una traiettoria di riferimento (x_ref, u_ref)
    fornita al controllore MPC.
    """

    def __init__(self, nx: int, nu: int, horizon: int, dt: float, f_dyn, device: str = "cpu"):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.horizon = horizon
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = torch.double
        state_dim = nx
        hidden_dim = 512
        output_dim = self.horizon * (self.nx + self.nu) * 2

        self.cost_map_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        ).to(self.device, dtype=self.dtype)

        cost = self._create_placeholder_cost()
        self.mpc = DifferentiableMPCController(
            f_dyn=f_dyn,
            total_time=horizon * dt,
            step_size=dt,
            horizon=horizon,
            cost_module=cost,
            grad_method=GradMethod.AUTO_DIFF,
            device=device
        )
        initial_log_std = 0.0
        self.log_std = nn.Parameter(torch.full((nu,), initial_log_std, device=self.device, dtype=self.dtype))

    def _create_placeholder_cost(self):
        """Crea un modulo di costo con tensori placeholder."""
        C = torch.zeros(self.horizon, self.nx + self.nu, self.nx + self.nu, device=self.device, dtype=self.dtype)
        c = torch.zeros(self.horizon, self.nx + self.nu, device=self.device, dtype=self.dtype)
        C_final = torch.zeros(self.nx + self.nu, self.nx + self.nu, device=self.device, dtype=self.dtype)
        c_final = torch.zeros(self.nx + self.nu, device=self.device, dtype=self.dtype)
        return GeneralQuadCost(self.nx, self.nu, C, c, C_final, c_final, device=self.device)

    def _generate_and_scale_costs(self, x: Tensor):
        """Genera i parametri di costo dalla rete e li scala."""
        q_p_lower_bound = 0.1
        q_p_upper_bound = 10000.0
        cost_params_raw = self.cost_map_net(x)
        cost_params_scaled = q_p_lower_bound + (q_p_upper_bound - q_p_lower_bound) * cost_params_raw

        split_size = self.horizon * (self.nx + self.nu)
        q_diag_flat = cost_params_scaled[:, :split_size]
        p_flat = cost_params_scaled[:, split_size:]

        batch_size = x.shape[0]
        q_diag = q_diag_flat.view(batch_size, self.horizon, self.nx + self.nu)
        p = p_flat.view(batch_size, self.horizon, self.nx + self.nu)

        C = torch.diag_embed(q_diag)
        c = p

        C_final = C[:, -1, :, :]
        c_final = c[:, -1, :]

        return C, c, C_final, c_final

    def forward(self, x: Tensor, x_ref: Tensor, u_ref: Tensor, deterministic: bool = False):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x_ref.ndim == 2:
            x_ref = x_ref.unsqueeze(0)
        if u_ref.ndim == 2:
            u_ref = u_ref.unsqueeze(0)

        # 1. Genera e aggiorna i costi nell'MPC
        C, c, C_final, c_final = self._generate_and_scale_costs(x)
        self.mpc.cost_module.C = C
        self.mpc.cost_module.c = c
        self.mpc.cost_module.C_final = C_final
        self.mpc.cost_module.c_final = c_final

        # 2. Risolvi l'MPC per ottenere l'azione ottimale
        U_init = torch.zeros(x.shape[0], self.horizon, self.nu, device=self.device, dtype=self.dtype)

        # --- MODIFICA CHIAVE PER IL TRACKING ---
        # Passiamo i riferimenti (x_ref, u_ref) al solutore MPC.
        # Il modulo di costo userà questi riferimenti per calcolare l'errore di tracking.
        u_mpc, _ = self.mpc.solve_step(
            x,
            U_init,
            x_ref_batch=x_ref,
            u_ref_batch=u_ref
        )

        # 3. Campiona l'azione per l'esplorazione
        std = self.log_std.exp()
        dist = Normal(u_mpc, std)

        action = u_mpc if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Rimuovi la dimensione del batch se l'input era unbatched
        if x.shape[0] == 1:
            return action.squeeze(0), log_prob.squeeze(0)

        return action, log_prob