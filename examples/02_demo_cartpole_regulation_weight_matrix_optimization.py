import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout
from typing import Dict, Tuple

from DifferentialMPC import DifferentiableMPCController, GradMethod, GeneralQuadCost

@dataclass(frozen=True)
class CartPoleParams:
    m_c: float;
    m_p: float;
    l: float;
    g: float

    @classmethod
    def from_gym(cls):
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            env = gym.make("CartPole-v1")
        return cls(m_c=float(env.unwrapped.masscart), m_p=float(env.unwrapped.masspole),
                   l=float(env.unwrapped.length), g=float(env.unwrapped.gravity))


def f_cartpole(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    pos, vel, theta, omega = x.split(1, dim=-1)
    force = u
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass, m_p_l = p.m_c + p.m_p, p.m_p * p.l
    temp = (force + m_p_l * omega.pow(2) * sin_t) / total_mass
    theta_dd_num = (p.g * sin_t - cos_t * temp)
    theta_dd_den = (p.l * (4.0 / 3.0 - p.m_p * cos_t.pow(2) / total_mass))
    theta_dd = theta_dd_num / theta_dd_den
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass
    x_dot = torch.cat([vel, vel_dd, omega, theta_dd], dim=-1)
    return x + x_dot * dt

def f_cartpole_jac_batched(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> tuple[
    torch.Tensor, torch.Tensor]:
    B = x.shape[0]
    pos, vel, theta, omega = x.split(1, dim=-1)
    force = u
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass = p.m_c + p.m_p
    m_p_l = p.m_p * p.l
    temp = (force + m_p_l * omega.pow(2) * sin_t) / total_mass
    theta_dd_num = (p.g * sin_t - cos_t * temp)
    theta_dd_den = (p.l * (4.0 / 3.0 - p.m_p * cos_t.pow(2) / total_mass))
    theta_dd = theta_dd_num / theta_dd_den
    d_temp_d_force = torch.full_like(force, 1 / total_mass)
    d_temp_d_theta = m_p_l * omega.pow(2) * cos_t / total_mass
    d_temp_d_omega = 2 * m_p_l * omega * sin_t / total_mass

    d_num_d_theta = p.g * cos_t + sin_t * temp - cos_t * d_temp_d_theta
    d_den_d_theta = p.l * (2 * p.m_p * cos_t * sin_t / total_mass)
    d_theta_dd_d_theta = (d_num_d_theta * theta_dd_den - theta_dd_num * d_den_d_theta) / theta_dd_den.pow(2)

    d_theta_dd_d_omega = (-cos_t * d_temp_d_omega) / theta_dd_den
    d_theta_dd_d_force = (-cos_t * d_temp_d_force) / theta_dd_den

    d_vel_dd_d_theta = d_temp_d_theta - (m_p_l / total_mass) * (d_theta_dd_d_theta * cos_t - theta_dd * sin_t)
    d_vel_dd_d_omega = d_temp_d_omega - (m_p_l / total_mass) * d_theta_dd_d_omega * cos_t
    d_vel_dd_d_force = d_temp_d_force - (m_p_l / total_mass) * d_theta_dd_d_force * cos_t

    I = torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
    J_c = torch.zeros(B, 4, 4, device=x.device, dtype=x.dtype)

    J_c[:, 0, 1] = 1.0
    J_c[:, 1, 2] = d_vel_dd_d_theta.squeeze(-1)
    J_c[:, 1, 3] = d_vel_dd_d_omega.squeeze(-1)
    J_c[:, 2, 3] = 1.0
    J_c[:, 3, 2] = d_theta_dd_d_theta.squeeze(-1)
    J_c[:, 3, 3] = d_theta_dd_d_omega.squeeze(-1)

    A = I + J_c * dt

    B_mat = torch.zeros(B, 4, 1, device=x.device, dtype=x.dtype)
    B_mat[:, 1, 0] = d_vel_dd_d_force.squeeze(-1) * dt
    B_mat[:, 3, 0] = d_theta_dd_d_force.squeeze(-1) * dt

    return A, B_mat



class CostModel(nn.Module):
    """Un modello che contiene i parametri di costo Q e R come parametri allenabili."""

    def __init__(self, nx: int, nu: int, device: torch.device):
        super().__init__()
        # Inizializza i parametri per le diagonali di Q e R
        # Usiamo valori iniziali che dopo softplus diano circa i valori manuali
        q_init = torch.tensor([2.3, 0.0, 4.6, 0.0], device=device)  # ln(e^x - 1)
        r_init = torch.tensor([-2.3], device=device)
        self.q_diag_params = nn.Parameter(q_init)
        self.r_diag_params = nn.Parameter(r_init)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restituisce le matrici di costo Q e R.
        Usa softplus per garantire che i costi siano sempre positivi.
        """
        q_diag = torch.nn.functional.softplus(self.q_diag_params)
        r_diag = torch.nn.functional.softplus(self.r_diag_params)
        return torch.diag(q_diag), torch.diag(r_diag)


def run_training_phase(common_params: Dict) -> Tuple[CostModel, list]:
    """Esegue il ciclo di training per ottimizzare i parametri di costo."""
    DEVICE, BATCH_SIZE, DT, HORIZON, N_SIM, params, x0 = [common_params[k] for k in
                                                          ['DEVICE', 'BATCH_SIZE', 'DT', 'HORIZON', 'N_SIM', 'params',
                                                           'x0']]
    nx, nu = 4, 1
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)
    TRAINING_EPOCHS = 2
    LEARNING_RATE = 0.05

    print("\n" + "=" * 60)
    print(" Inizio Fase di Training per i Parametri di Costo")
    print("=" * 60)

    cost_model = CostModel(nx, nu, DEVICE).to(DEVICE)
    optimizer = optim.Adam(cost_model.parameters(), lr=LEARNING_RATE)

    # --- INIZIO CORREZIONE 1 ---
    C_placeholder = torch.zeros(HORIZON, nx + nu, nx + nu, device=DEVICE)
    c_placeholder = torch.zeros(HORIZON, nx + nu, device=DEVICE)
    C_final_placeholder = torch.zeros(BATCH_SIZE, nx + nu, nx + nu, device=DEVICE)
    c_final_placeholder = torch.zeros(BATCH_SIZE, nx + nu, device=DEVICE)
    # --- FINE CORREZIONE 1 ---

    cost_module = GeneralQuadCost(nx, nu, C_placeholder, c_placeholder, C_final_placeholder, c_final_placeholder,
                                  device=str(DEVICE))
    x_target = torch.zeros(nx, device=DEVICE)
    x_ref_b = x_target.unsqueeze(0).repeat(BATCH_SIZE, HORIZON + 1, 1)
    u_ref_b = torch.zeros(BATCH_SIZE, HORIZON, nu, device=DEVICE)
    cost_module.set_reference(x_ref=x_ref_b, u_ref=u_ref_b)

    mpc = DifferentiableMPCController(
        f_dyn=dyn, total_time=HORIZON * DT, horizon=HORIZON, step_size=DT,
        cost_module=cost_module, u_min=torch.tensor([-20.0]), u_max=torch.tensor([20.0]),
        grad_method=GradMethod.ANALYTIC, f_dyn_jac=lambda x, u, dt: f_cartpole_jac_batched(x, u, dt, params),
        device=str(DEVICE)
    )

    training_losses = []

    for epoch in range(TRAINING_EPOCHS):
        epoch_start_time = time.time()
        optimizer.zero_grad()

        Q_learn, R_learn = cost_model()

        C_new = torch.zeros(HORIZON, nx + nu, nx + nu, device=DEVICE)
        C_new[:, :nx, :nx] = Q_learn
        C_new[:, nx:, nx:] = R_learn

        # --- INIZIO CORREZIONE 2 ---
        C_final_new = torch.zeros(BATCH_SIZE, nx + nu, nx + nu, device=DEVICE)
        C_final_new[:, :nx, :nx] = Q_learn * 10
        # --- FINE CORREZIONE 2 ---

        mpc.cost_module.C = C_new
        mpc.cost_module.C_final = C_final_new
        mpc.cost_module.c.zero_()
        mpc.cost_module.c_final.zero_()

        x_current = x0
        total_cost = torch.tensor(0.0, device=DEVICE)

        for _ in range(N_SIM):
            _, U_opt = mpc.forward(x_current)
            u_apply = U_opt[:, 0, :]

            state_cost = torch.einsum('bi,ij,bj->b', x_current, Q_learn, x_current)
            control_cost = torch.einsum('bi,ij,bj->b', u_apply, R_learn, u_apply)
            total_cost += (state_cost + control_cost).mean()

            x_current = dyn(x_current, u_apply, DT)

        total_cost.backward()
        optimizer.step()

        training_losses.append(total_cost.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoca {epoch + 1}/{TRAINING_EPOCHS} | "
                  f"Loss (Costo Totale): {total_cost.item():.4f} | "
                  f"Tempo/10 epoche: {time.time() - epoch_start_time:.2f}s")

    print("\nTraining completato!")
    q_final, r_final = cost_model()
    print("Diagonale di Q appresa:", torch.diag(q_final).detach().cpu().numpy())
    print("Diagonale di R appresa:", torch.diag(r_final).detach().cpu().numpy())
    print("=" * 60)

    return cost_model, training_losses


def run_simulation(title: str, cost_model: CostModel | None, common_params: Dict) -> Dict:
    """Esegue una simulazione con costi manuali (se model=None) o appresi."""
    DEVICE, BATCH_SIZE, DT, HORIZON, N_SIM, params, x0 = [common_params[k] for k in
                                                          ['DEVICE', 'BATCH_SIZE', 'DT', 'HORIZON', 'N_SIM', 'params',
                                                           'x0']]
    nx, nu = 4, 1
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)

    print(f"\n--- Avvio Simulazione: '{title}' ---")

    if cost_model is None:
        Q = torch.diag(torch.tensor([10.0, 1.0, 100.0, 1.0], device=DEVICE))
        R = torch.diag(torch.tensor([0.1], device=DEVICE))
    else:
        with torch.no_grad():
            Q, R = cost_model()

    C = torch.zeros(HORIZON, nx + nu, nx + nu, device=DEVICE)
    c = torch.zeros(HORIZON, nx + nu, device=DEVICE)
    C[:, :nx, :nx] = Q
    C[:, nx:, nx:] = R

    C_final = torch.zeros(BATCH_SIZE, nx + nu, nx + nu, device=DEVICE)
    C_final[:, :nx, :nx] = Q * 10
    c_final = torch.zeros(BATCH_SIZE, nx + nu, device=DEVICE)

    cost_module = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=str(DEVICE))

    # ======================== INIZIO CODICE MANCANTE ========================
    # Imposta le traiettorie di riferimento con la dimensione di batch corretta.
    x_target = torch.zeros(nx, device=DEVICE)
    x_ref_b = x_target.unsqueeze(0).repeat(BATCH_SIZE, HORIZON + 1, 1)
    u_ref_b = torch.zeros(BATCH_SIZE, HORIZON, nu, device=DEVICE)
    cost_module.set_reference(x_ref=x_ref_b, u_ref=u_ref_b)
    # ========================= FINE CODICE MANCANTE =========================

    mpc = DifferentiableMPCController(
        f_dyn=dyn, total_time=HORIZON * DT, horizon=HORIZON, step_size=DT,
        cost_module=cost_module, u_min=torch.tensor([-20.0]), u_max=torch.tensor([20.0]),
        grad_method=GradMethod.ANALYTIC, f_dyn_jac=lambda x, u, dt: f_cartpole_jac_batched(x, u, dt, params),
        device=str(DEVICE)
    )

    x_current = x0
    xs_list, us_list = [x_current], []
    for i in range(N_SIM):
        with torch.no_grad():
            _, U_opt = mpc.forward(x_current)
        u_apply = U_opt[:, 0, :]
        us_list.append(u_apply)
        x_current = dyn(x_current, u_apply, DT)
        xs_list.append(x_current)

    Xs = torch.stack(xs_list, dim=1).detach()
    Us = torch.stack(us_list, dim=1).detach()

    final_angle_error = torch.mean(torch.abs(Xs[:, -1, 2])).item()
    print(f"Errore finale medio (angolo): {final_angle_error:.4f} rad")
    return {"Xs": Xs, "Us": Us, "dt": DT}


def _plot_full_comparison(results_before: Dict, results_after: Dict, losses: list):
    """Visualizza una dashboard completa che confronta prima/dopo il training."""
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Confronto Prestazioni MPC: Prima vs. Dopo Training del Costo", fontsize=18, weight='bold')

    agent_to_plot = 0
    sim_len = results_before['Xs'].shape[1] - 1
    dt = results_before['dt']
    time_ax = torch.arange(sim_len + 1) * dt

    # 1. Training Loss
    ax = axs[0, 0]
    ax.plot(losses, 'm-o', label='Loss di Training')
    ax.set_title("Curva di Apprendimento"), ax.set_xlabel("Epoca di Training"), ax.set_ylabel("Loss")
    ax.grid(True, linestyle='--'), ax.legend(), ax.set_yscale('log')

    # 2. Angolo del Pendolo
    ax = axs[0, 1]
    ax.plot(time_ax, results_before['Xs'][agent_to_plot, :, 2].cpu(), 'b--', label='Prima del Training (Manuale)')
    ax.plot(time_ax, results_after['Xs'][agent_to_plot, :, 2].cpu(), 'g-', label='Dopo il Training (Appreso)')
    ax.axhline(0, color='r', linestyle=':', label='Target Angolo')
    ax.set_title(f"Regolazione Angolo (Agente #{agent_to_plot + 1})"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel(
        "Angolo [rad]")
    ax.grid(True, linestyle='--'), ax.legend()

    # 3. Posizione del Carrello
    ax = axs[1, 0]
    ax.plot(time_ax, results_before['Xs'][agent_to_plot, :, 0].cpu(), 'b--', label='Prima del Training (Manuale)')
    ax.plot(time_ax, results_after['Xs'][agent_to_plot, :, 0].cpu(), 'g-', label='Dopo il Training (Appreso)')
    ax.axhline(0, color='r', linestyle=':', label='Target Posizione')
    ax.set_title(f"Posizione Carrello (Agente #{agent_to_plot + 1})"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel(
        "Posizione [m]")
    ax.grid(True, linestyle='--'), ax.legend()

    # 4. Input di Controllo
    ax = axs[1, 1]
    time_ax_ctrl = torch.arange(sim_len) * dt
    ax.plot(time_ax_ctrl, results_before['Us'][agent_to_plot, :, 0].cpu(), 'b--', label='Prima del Training (Manuale)')
    ax.plot(time_ax_ctrl, results_after['Us'][agent_to_plot, :, 0].cpu(), 'g-', label='Dopo il Training (Appreso)')
    ax.set_title(f"Input di Controllo (Agente #{agent_to_plot + 1})"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel(
        "Forza")
    ax.grid(True, linestyle='--'), ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ======================== SCRIPT PRINCIPALE ========================
def main():
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione Cart-Pole su dispositivo: {DEVICE}")

    common_params = {
        'DEVICE': DEVICE, 'BATCH_SIZE': 50, 'DT': 0.05,
        'HORIZON': 10, 'N_SIM': 50, 'params': CartPoleParams.from_gym()
    }
    base_state = torch.tensor([0.0, 0.0, 0.4, 0.0], device=DEVICE)  # Stato iniziale più difficile
    common_params['x0'] = base_state.unsqueeze(0).repeat(common_params['BATCH_SIZE'], 1) + \
                          torch.randn(common_params['BATCH_SIZE'], 4, device=DEVICE) * \
                          torch.tensor([0.5, 0.5, 0.2, 0.2], device=DEVICE)

    # FASE 1: TRAINING
    trained_model, training_losses = run_training_phase(common_params)

    # FASE 2: CONFRONTO
    results_before = run_simulation("Prima del Training (Costi Manuali)", None, common_params)
    results_after = run_simulation("Dopo il Training (Costi Appresi)", trained_model, common_params)

    # FASE 3: PLOTTING
    print("\nGenerazione della dashboard di confronto...")
    _plot_full_comparison(results_before, results_after, training_losses)


if __name__ == "__main__":
    main()