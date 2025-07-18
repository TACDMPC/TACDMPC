# File: actor.py

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
# Assicurati che le importazioni relative al tuo progetto siano corrette
from DifferentialMPC import DifferentiableMPCController, GradMethod, GeneralQuadCost
from typing import Optional # Aggiungi Optional se non già presente

class ActorMPC(nn.Module):
    """
    ActorMPC

    L'attore ora consiste in:
    1. Una rete neurale (cost_map_net) che mappa stati-parametri
       della funzione di costo dell'MPC (matrici Q_k e vettori p_k).
    2. Un controllore MPC differenziabile che calcola l'azione ottimale data la funzione di costo.
    3. Una distribuzione stocastica (Normale) la cui media è l'output dell'MPC,
       per consentire l'esplorazione.
    """

    def __init__(self, nx: int, nu: int, horizon: int, dt: float, f_dyn, device: str = "cpu"):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.horizon = horizon
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = torch.double

        # --- 1. Definizione della Neural Cost Map ---
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

        # --- 2. Inizializzazione del Controllore MPC Differenziabile ---
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

        # --- 3. Parametro per l'Esplorazione ---
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
        q_lower_bound = 0.1
        q_upper_bound = 10000.0
        p_bound = 1.0

        cost_params_raw = self.cost_map_net(x)
        split_size = self.horizon * (self.nx + self.nu)
        q_params_raw = cost_params_raw[:, :split_size]
        p_params_raw = cost_params_raw[:, split_size:]

        q_diag_flat = q_lower_bound + (q_upper_bound - q_lower_bound) * q_params_raw
        p_flat = (p_params_raw * 2 - 1) * p_bound

        batch_size = x.shape[0]
        q_diag = q_diag_flat.view(batch_size, self.horizon, self.nx + self.nu)
        p = p_flat.view(batch_size, self.horizon, self.nx + self.nu)

        C = torch.diag_embed(q_diag)
        c = p
        C_final = C[:, -1, :, :]
        c_final = c[:, -1, :]

        return C, c, C_final, c_final

    def forward(self, x: Tensor, deterministic: bool = False, x_ref: Optional[Tensor] = None, u_ref: Optional[Tensor] = None):
        """
        Esegue il forward pass, accettando riferimenti opzionali per il tracciamento.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # 1. Genera e aggiorna i costi nell'MPC
        C, c, C_final, c_final = self._generate_and_scale_costs(x)
        self.mpc.cost_module.C = C
        self.mpc.cost_module.c = c
        self.mpc.cost_module.C_final = C_final
        self.mpc.cost_module.c_final = c_final

        # --- MODIFICA CHIAVE ---
        # Imposta il riferimento nel modulo di costo dell'MPC prima di risolvere.
        # Questo permette all'MPC di calcolare il costo rispetto a un target dinamico.
        self.mpc.cost_module.set_reference(x_ref=x_ref, u_ref=u_ref)
        # -----------------------

        # 2. Risolvi l'MPC
        U_init = torch.zeros(x.shape[0], self.horizon, self.nu, device=self.device, dtype=self.dtype)
        u_mpc, _ = self.mpc.solve_step(x, U_init)

        # 3. Campiona l'azione
        std = self.log_std.exp()
        dist = Normal(u_mpc, std)
        action = u_mpc if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        predicted_states = self.mpc.X_last
        predicted_actions = self.mpc.U_last

        if x.shape[0] == 1:
            return (
                action.squeeze(0),
                log_prob.squeeze(0),
                predicted_states.squeeze(0),
                predicted_actions.squeeze(0)
            )

        return action, log_prob, predicted_states, predicted_actions