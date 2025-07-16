# examples/02_demo_cartpole_regulation.py
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout

# Importa le classi principali dal pacchetto
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost


# ======================== DEFINIZIONE DEL SISTEMA CART-POLE ========================
@dataclass(frozen=True)
class CartPoleParams:
    m_c: float;
    m_p: float;
    l: float;
    g: float

    @classmethod
    def from_gym(cls):
        # Sopprime l'output di benvenuto di gym
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            env = gym.make("CartPole-v1")
        return cls(
            m_c=float(env.unwrapped.masscart), m_p=float(env.unwrapped.masspole),
            l=float(env.unwrapped.length), g=float(env.unwrapped.gravity)
        )


def f_cartpole(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    """Dinamica non lineare del Cart-Pole, gestisce il batching."""
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass = p.m_c + p.m_p
    m_p_l = p.m_p * p.l

    temp = (force + m_p_l * omega ** 2 * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0 / 3.0 - p.m_p * cos_t ** 2 / total_mass))
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass

    next_state = torch.stack((
        pos + vel * dt,
        vel + vel_dd * dt,
        theta + omega * dt,
        omega + theta_dd * dt
    ), dim=-1)
    return next_state

def main():
    # --- Configurazione Globale ---
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione Cart-Pole su dispositivo: {DEVICE}")

    BATCH_SIZE = 100
    DT = 0.05
    HORIZON = 30
    N_SIM = 100

    # --- Parametri e Dinamica ---
    params = CartPoleParams.from_gym()
    nx, nu = 4, 1
    # Creiamo un handle di funzione compatibile con il controller MPC
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)

    # --- Riferimenti e Costi ---
    x_target = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)  # Obiettivo: centro, fermo, palo dritto
    x_ref_horizon = x_target.repeat(HORIZON + 1, 1)
    u_ref_horizon = torch.zeros(HORIZON, nu, device=DEVICE)

    Q = torch.diag(torch.tensor([10.0, 1.0, 100.0, 1.0], device=DEVICE))
    R = torch.diag(torch.tensor([0.1], device=DEVICE))
    C = torch.zeros(HORIZON, nx + nu, nx + nu, device=DEVICE);
    C[:, :nx, :nx] = Q;
    C[:, nx:, nx:] = R
    c = torch.zeros(HORIZON, nx + nu, device=DEVICE)
    C_final_mat = torch.zeros(nx + nu, nx + nu, device=DEVICE);
    C_final_mat[:nx, :nx] = Q * 100
    c_final = torch.zeros(nx + nu, device=DEVICE)

    cost = GeneralQuadCost(
        nx=nx, nu=nu, C=C, c=c,
        C_final=C_final_mat, c_final=c_final,
        device=str(DEVICE), x_ref=x_ref_horizon, u_ref=u_ref_horizon
    )

    # --- Setup MPC Controller ---
    mpc = DifferentiableMPCController(
        f_dyn=dyn,
        total_time=HORIZON * DT,
        step_size=DT,
        horizon=HORIZON,
        cost_module=cost,
        u_min=torch.tensor([-20.0], device=DEVICE),
        u_max=torch.tensor([20.0], device=DEVICE),
        grad_method="auto_diff",  # La Jacobiana è complessa, usiamo auto-diff
        N_sim=N_SIM,
        device=str(DEVICE)
    )

    # --- Batch di stati iniziali casuali (vicino al punto di instabilità) ---
    base_state = torch.tensor([0.0, 0.0, 0.2, 0.0], device=DEVICE)  # Partenza con palo inclinato
    x0 = base_state + torch.tensor([1.0, 1.0, 0.5, 0.5], device=DEVICE) * torch.randn(BATCH_SIZE, nx, device=DEVICE)

    # --- Esecuzione Simulazione ---
    print(f"Avvio del rollout MPC per {BATCH_SIZE} agenti Cart-Pole...")
    t_start = time.perf_counter()
    # Poiché l'obiettivo è fisso, non dobbiamo passare riferimenti dinamici a forward()
    Xs, Us = mpc.forward(x0)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_time_ms = (t_end - t_start) * 1000

    print(f"Rollout completato in {total_time_ms:.2f} ms.")
    print(f"Tempo medio per passo di simulazione: {total_time_ms / N_SIM:.2f} ms.")

    _plot_results(Xs.cpu(), Us.cpu(), DT, x_target.cpu(), BATCH_SIZE)
# ======================== PLOTTING ========================


print("Generazione dei grafici...")

# --- Dati per i grafici ---
# ======================== FUNZIONI DI PLOTTING (Completa) ========================
def _plot_results(xs_mpc: torch.Tensor, us_mpc: torch.Tensor, dt: float, target: torch.Tensor, batch_size: int):
    """Visualizza i risultati della simulazione del Cart-Pole."""

    # --- Dati per i grafici ---
    nx = xs_mpc.shape[-1]
    N_TO_PLOT = min(10, batch_size)
    labels = ["Posizione [m]", "Velocità [m/s]", "Angolo [rad]", "Vel. Angolare [rad/s]"]
    time_state = torch.arange(xs_mpc.shape[1]) * dt
    time_ctrl = torch.arange(us_mpc.shape[1]) * dt
    colors = plt.cm.viridis(np.linspace(0, 1, N_TO_PLOT))

    # --- Figura 1: Traiettorie di Stato ---
    fig, axs = plt.subplots(nx, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Traiettorie di Stato per {N_TO_PLOT} Agenti (su {batch_size})")
    for i in range(nx):
        for b in range(N_TO_PLOT):
            # Aggiungi etichetta solo al primo agente per non affollare la legenda
            label = f'Agente {b + 1}' if i == 0 else None
            axs[i].plot(time_state, xs_mpc[b, :, i], color=colors[b], alpha=0.7, label=label)
        axs[i].axhline(target[i].item(), linestyle=":", color="k", label="Target" if i == 0 else "")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    if N_TO_PLOT > 0:
        fig.legend(loc='upper right')
    axs[-1].set_xlabel("Tempo [s]")

    # --- Figura 2: Comandi di controllo ---
    plt.figure(figsize=(12, 6))
    plt.title(f"Comandi di Controllo per {N_TO_PLOT} Agenti")
    for b in range(N_TO_PLOT):
        plt.plot(time_ctrl, us_mpc[b, :, 0], color=colors[b], alpha=0.7, label=f'Agente {b + 1}')
    plt.xlabel("Tempo [s]")
    plt.ylabel("Forza [N]")
    plt.grid(True)
    if N_TO_PLOT > 0:
        plt.legend()

    # --- Mostra tutte le figure create ---
    plt.show()
# ======================== ENTRY-POINT ========================
if __name__ == "__main__":
    main()