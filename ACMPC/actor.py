# File: actor.py

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from DifferentialMPC import DifferentiableMPCController, GradMethod, GeneralQuadCost


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
        self.dtype = torch.double  # L'MPC differenziabile spesso richiede double precision

        # --- 1. Definizione della Neural Cost Map ---
        state_dim = nx
        hidden_dim = 512
        # L'output deve generare i termini diagonali di Q e i termini lineari p
        # per ogni step dell'orizzonte. Dimensione: T * (nx+nu) per Q_diag e T * (nx+nu) per p.
        # Ref: "dimensionality of the output dimension of the Cost Map is 2T(n_state + n_input)"
        output_dim = self.horizon * (self.nx + self.nu) * 2

        self.cost_map_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Sigmoid per limitare l'output tra 0 e 1, come base per la scalatura.
        ).to(self.device, dtype=self.dtype)

        # --- 2. Inizializzazione del Controllore MPC Differenziabile ---
        # Il modulo di costo viene inizializzato con zeri, ma sarà sovrascritto
        # ad ogni forward pass con i valori generati dalla cost_map_net.
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
        # Deviazione standard per la policy stocastica.
        # È un parametro apprendibile per controllare l'esplorazione.
        initial_log_std = 0.0  # Valore iniziale, può essere tunato
        self.log_std = nn.Parameter(torch.full((nu,), initial_log_std, device=self.device, dtype=self.dtype))

    def _create_placeholder_cost(self):
        """Crea un modulo di costo con tensori placeholder."""
        C = torch.zeros(self.horizon, self.nx + self.nu, self.nx + self.nu, device=self.device, dtype=self.dtype)
        c = torch.zeros(self.horizon, self.nx + self.nu, device=self.device, dtype=self.dtype)
        C_final = torch.zeros(self.nx + self.nu, self.nx + self.nu, device=self.device, dtype=self.dtype)
        c_final = torch.zeros(self.nx + self.nu, device=self.device, dtype=self.dtype)
        return GeneralQuadCost(self.nx, self.nu, C, c, C_final, c_final, device=self.device)

    # In actor.py, replace the _generate_and_scale_costs function with this one.

    def _generate_and_scale_costs(self, x: Tensor):
        """
        Genera i parametri di costo dalla rete e li scala.
        Ref: "the last layer ... has been chosen to be a sigmoid which allows for upper and lower bounds" [cite: 226]
        """
        # Limiti per la scalatura, come menzionato nel paper [cite: 227]
        q_lower_bound = 0.1
        q_upper_bound = 10000.0
        # CORREZIONE: Definiamo un limite separato per il termine lineare 'p'
        # per permettere valori sia positivi che negativi, cruciali per il controllo.
        p_bound = 1.0  # Questo è un iperparametro che può essere tunato.

        # Passa lo stato attraverso la rete per ottenere i parametri di costo grezzi (output tra 0 e 1)
        cost_params_raw = self.cost_map_net(x)

        # Separa i parametri grezzi per i termini quadratici (Q) e lineari (p)
        split_size = self.horizon * (self.nx + self.nu)
        q_params_raw = cost_params_raw[:, :split_size]
        p_params_raw = cost_params_raw[:, split_size:]

        # --- CORREZIONE: Applica scaling differenti per Q e p ---

        # 1. Scala i parametri di Q per garantire la positività (stabilità dell'MPC)
        q_diag_flat = q_lower_bound + (q_upper_bound - q_lower_bound) * q_params_raw

        # 2. Scala i parametri di p in un range simmetrico attorno a zero.
        #    La trasformazione (x * 2 - 1) mappa l'output della sigmoide da [0, 1] a [-1, 1].
        p_flat = (p_params_raw * 2 - 1) * p_bound

        # Riformatta i tensori per l'orizzonte dell'MPC
        batch_size = x.shape[0]
        q_diag = q_diag_flat.view(batch_size, self.horizon, self.nx + self.nu)
        p = p_flat.view(batch_size, self.horizon, self.nx + self.nu)

        # Costruisce la matrice di costo quadratica C (diagonale) [cite: 223]
        C = torch.diag_embed(q_diag)
        # Il vettore di costo lineare c è semplicemente p
        c = p

        # Per semplicità, assumiamo che il costo finale sia lo stesso dell'ultimo step,
        # ma potrebbe essere gestito diversamente se necessario.
        C_final = C[:, -1, :, :]
        c_final = c[:, -1, :]

        return C, c, C_final, c_final

    def forward(self, x: Tensor, deterministic: bool = False):
        """
        Esegue il forward pass, restituendo anche le traiettorie predette dall'MPC
        per l'addestramento del critico con MPVE.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # 1. Genera e aggiorna i costi nell'MPC
        C, c, C_final, c_final = self._generate_and_scale_costs(x)
        self.mpc.cost_module.C = C
        self.mpc.cost_module.c = c
        self.mpc.cost_module.C_final = C_final
        self.mpc.cost_module.c_final = c_final

        # 2. Risolvi l'MPC
        U_init = torch.zeros(x.shape[0], self.horizon, self.nu, device=self.device, dtype=self.dtype)
        u_mpc, _ = self.mpc.solve_step(x, U_init)

        # 3. Campiona l'azione
        std = self.log_std.exp()
        dist = Normal(u_mpc, std)
        action = u_mpc if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # --- NUOVO: Restituisci le traiettorie salvate dal solutore MPC ---
        # self.mpc.X_last e self.mpc.U_last vengono salvati internamente da solve_step
        predicted_states = self.mpc.X_last
        predicted_actions = self.mpc.U_last

        # Gestione output per batch/unbatched
        if x.shape[0] == 1:
            return (
                action.squeeze(0),
                log_prob.squeeze(0),
                predicted_states.squeeze(0),
                predicted_actions.squeeze(0)
            )

        return action, log_prob, predicted_states, predicted_actions
