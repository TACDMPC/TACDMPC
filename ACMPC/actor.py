# File: actor.py

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from DifferentialMPC import DifferentiableMPCController, GradMethod, GeneralQuadCost


class ActorMPC(nn.Module):
    def __init__(self, nx: int, nu: int, horizon: int, dt: float, f_dyn, device: str = "cpu"):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.horizon = horizon
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = torch.float32

        state_dim = nx
        hidden_dim = 512
        output_dim = self.horizon * (self.nx + self.nu) * 2

        self.cost_map_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid()
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
        C = torch.zeros(self.horizon, self.nx + self.nu, self.nx + self.nu, device=self.device, dtype=self.dtype)
        c = torch.zeros(self.horizon, self.nx + self.nu, device=self.device, dtype=self.dtype)
        C_final = torch.zeros(self.nx + self.nu, self.nx + self.nu, device=self.device, dtype=self.dtype)
        c_final = torch.zeros(self.nx + self.nu, device=self.device, dtype=self.dtype)
        return GeneralQuadCost(self.nx, self.nu, C, c, C_final, c_final, device=self.device)

    def _generate_and_scale_costs(self, x: Tensor):
        lower_bound, upper_bound = 0.1, 100000.0
        cost_params_raw = self.cost_map_net(x)
        split_size = self.horizon * (self.nx + self.nu)
        q_params_raw, p_params_raw = cost_params_raw.split(split_size, dim=-1)

        q_diag_flat = lower_bound + (upper_bound - lower_bound) * q_params_raw
        p_flat = lower_bound + (upper_bound - lower_bound) * p_params_raw

        batch_size = x.shape[0]
        q_diag = q_diag_flat.view(batch_size, self.horizon, self.nx + self.nu)
        p = p_flat.view(batch_size, self.horizon, self.nx + self.nu)

        C = torch.diag_embed(q_diag)
        c = p
        C_final = C[:, -1, :, :]
        c_final = c[:, -1, :]
        return C, c, C_final, c_final

    def forward(self, x: Tensor, deterministic: bool = False):
        """
        --- MODIFICA CHIAVE: Esegue un ciclo sul batch ---
        Questo adatta l'attore al design non-batchato del modulo di costo.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # 1. Genera i costi per l'intero batch in un colpo solo
        C_batch, c_batch, C_final_batch, c_final_batch = self._generate_and_scale_costs(x)

        # Liste per raccogliere i risultati di ogni elemento del batch
        action_list = []
        log_prob_list = []
        pred_states_list = []
        pred_actions_list = []

        # 2. Itera su ogni elemento del batch
        for i in range(batch_size):
            # Prendi la slice i-esima dello stato
            x_i = x[i:i + 1]  # Mantieni la dimensione del batch (B=1)

            # Imposta i costi NON-BATCHATI nel modulo MPC per questa iterazione
            self.mpc.cost_module.C = C_batch[i]
            self.mpc.cost_module.c = c_batch[i]
            self.mpc.cost_module.C_final = C_final_batch[i]
            self.mpc.cost_module.c_final = c_final_batch[i]

            # 3. Risolvi l'MPC per il singolo elemento
            U_init_i = torch.zeros(1, self.horizon, self.nu, device=self.device, dtype=self.dtype)
            u_mpc_i, _ = self.mpc.solve_step(x_i, U_init_i)

            # 4. Campiona l'azione e calcola il log_prob
            std = self.log_std.exp()
            dist = Normal(u_mpc_i, std)
            action_i = u_mpc_i if deterministic else dist.rsample()
            log_prob_i = dist.log_prob(action_i).sum(dim=-1)

            # Aggiungi i risultati alle liste
            action_list.append(action_i)
            log_prob_list.append(log_prob_i)
            pred_states_list.append(self.mpc.X_last)
            pred_actions_list.append(self.mpc.U_last)

        # 5. Concatena i risultati per ricreare i tensori batchati
        actions = torch.cat(action_list, dim=0)
        log_probs = torch.cat(log_prob_list, dim=0)
        predicted_states = torch.cat(pred_states_list, dim=0)
        predicted_actions = torch.cat(pred_actions_list, dim=0)

        # Gestione output per batch/unbatched
        if batch_size == 1:
            return (
                actions.squeeze(0), log_probs.squeeze(0),
                predicted_states.squeeze(0), predicted_actions.squeeze(0)
            )

        return actions, log_probs, predicted_states, predicted_actions