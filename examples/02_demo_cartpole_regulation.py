# examples/02_demo_cartpole_regulation_updated.py
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout

# Importa le classi principali dalla nuova codebase
# Assicurati che i file 'controller.py' e 'cost.py' siano nella stessa directory
# o in un percorso accessibile da Python.
from DifferentialMPC import DifferentiableMPCController
from DifferentialMPC import GeneralQuadCost


# ======================== DEFINIZIONE DEL SISTEMA CART-POLE ========================
@dataclass(frozen=True)
class CartPoleParams:
    """Parametri fisici del sistema Cart-Pole."""
    m_c: float
    m_p: float
    l: float
    g: float

    @classmethod
    def from_gym(cls):
        """Estrae i parametri dall'ambiente Gym 'CartPole-v1'."""
        # Sopprime l'output di benvenuto di gym
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            env = gym.make("CartPole-v1")
        # I parametri in gym sono leggermente diversi, ma li usiamo per coerenza
        return cls(
            m_c=float(env.unwrapped.masscart),
            m_p=float(env.unwrapped.masspole),
            l=float(env.unwrapped.length),
            g=float(env.unwrapped.gravity)
        )


def f_cartpole(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    """
    Dinamica non lineare del Cart-Pole.
    Gestisce il batching automatico lungo la prima dimensione.
    Input:
        x: (B, 4) - [pos, vel, theta, omega]
        u: (B, 1) - [forza]
    Output:
        next_state: (B, 4)
    """
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass = p.m_c + p.m_p
    m_p_l = p.m_p * p.l

    temp = (force + m_p_l * omega.pow(2) * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0 / 3.0 - p.m_p * cos_t.pow(2) / total_mass))
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass

    # Integrazione di Eulero
    next_state = torch.stack((
        pos + vel * dt,
        vel + vel_dd * dt,
        theta + omega * dt,
        omega + theta_dd * dt
    ), dim=-1)
    return next_state

# ======================== FUNZIONI DI PLOTTING ========================
def _plot_results(xs_mpc: torch.Tensor, us_mpc: torch.Tensor, dt: float, target: torch.Tensor, batch_size: int):
    """Visualizza i risultati della simulazione del Cart-Pole."""
    nx = xs_mpc.shape[-1]
    N_TO_PLOT = min(10, batch_size)
    labels = ["Posizione [m]", "Velocità [m/s]", "Angolo [rad]", "Vel. Angolare [rad/s]"]
    time_state = torch.arange(xs_mpc.shape[1]) * dt
    time_ctrl = torch.arange(us_mpc.shape[1]) * dt
    colors = plt.cm.viridis(np.linspace(0, 1, N_TO_PLOT))

    # --- Figura 1: Traiettorie di Stato ---
    fig, axs = plt.subplots(nx, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Traiettorie di Stato per {N_TO_PLOT} Agenti (su {batch_size} totali)")
    for i in range(nx):
        for b in range(N_TO_PLOT):
            label = f'Agente {b + 1}' if i == 0 else None
            axs[i].plot(time_state, xs_mpc[b, :, i], color=colors[b], alpha=0.7, label=label)
        axs[i].axhline(target[i].item(), linestyle=":", color="k", label="Target" if i == 0 else "")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    if N_TO_PLOT > 0:
        fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    axs[-1].set_xlabel("Tempo [s]")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Figura 2: Comandi di controllo ---
    plt.figure(figsize=(12, 6))
    plt.title(f"Comandi di Controllo per {N_TO_PLOT} Agenti")
    for b in range(N_TO_PLOT):
        plt.plot(time_ctrl, us_mpc[b, :, 0], color=colors[b], alpha=0.7, label=f'Agente {b + 1}')
    plt.xlabel("Tempo [s]")
    plt.ylabel("Forza [N]")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if N_TO_PLOT > 0:
        plt.legend()
    plt.tight_layout()

    plt.show()

# ======================== SCRIPT PRINCIPALE ========================
def main():
    # --- Configurazione Globale ---
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione Cart-Pole su dispositivo: {DEVICE}")

    BATCH_SIZE = 100
    DT = 0.05
    HORIZON = 30  # Orizzonte di predizione dell'MPC (T)
    N_SIM = 100   # Numero di passi della simulazione in anello chiuso

    # --- Parametri e Dinamica ---
    params = CartPoleParams.from_gym()
    nx, nu = 4, 1
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)

    # --- Riferimenti e Costi ---
    x_target = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
    # I riferimenti ora vengono gestiti internamente dal modulo di costo,
    # ma li creiamo per passarglieli all'inizio.
    x_ref_horizon = x_target.repeat(1, HORIZON + 1, 1) # Shape (1, H+1, nx) per batching
    u_ref_horizon = torch.zeros(1, HORIZON, nu, device=DEVICE) # Shape (1, H, nu)

    Q = torch.diag(torch.tensor([10.0, 1.0, 100.0, 1.0], device=DEVICE))
    R = torch.diag(torch.tensor([0.1], device=DEVICE))

    # Costo di tappa (running cost)
    C = torch.zeros(HORIZON, nx + nu, nx + nu, device=DEVICE)
    C[:, :nx, :nx] = Q
    C[:, nx:, nx:] = R
    c = torch.zeros(HORIZON, nx + nu, device=DEVICE)

    # Costo finale (terminal cost)
    C_final_mat = torch.zeros(nx + nu, nx + nu, device=DEVICE)
    C_final_mat[:nx, :nx] = Q * 10
    c_final = torch.zeros(nx + nu, device=DEVICE)

    cost = GeneralQuadCost(
        nx=nx, nu=nu, C=C, c=c,
        C_final=C_final_mat, c_final=c_final,
        device=str(DEVICE),
        x_ref=x_ref_horizon, u_ref=u_ref_horizon
    )

    # --- Setup MPC Controller ---
    # NOTA: N_sim non è più un parametro del controller, perché il ciclo
    # di simulazione è ora gestito esternamente.
    mpc = DifferentiableMPCController(
        f_dyn=dyn,
        total_time=HORIZON * DT,
        step_size=DT,
        horizon=HORIZON,
        cost_module=cost,
        u_min=torch.tensor([-20.0], device=DEVICE),
        u_max=torch.tensor([20.0], device=DEVICE),
        grad_method="auto_diff",
        device=str(DEVICE)
    )

    # --- Batch di stati iniziali casuali (vicino al punto di instabilità) ---
    base_state = torch.tensor([0.0, 0.0, 0.2, 0.0], device=DEVICE)
    x0 = base_state + torch.tensor([0.5, 0.5, 0.2, 0.2], device=DEVICE) * torch.randn(BATCH_SIZE, nx, device=DEVICE)

    # --- Esecuzione Simulazione (ANELLO CHIUSO ESPLICITO) ---
    print(f"Avvio del rollout MPC in anello chiuso per {BATCH_SIZE} agenti Cart-Pole...")
    t_start = time.perf_counter()

    x_current = x0
    xs_list = [x_current]
    us_list = []

    # Ciclo di simulazione esplicito
    for i in range(N_SIM):
        if i % 20 == 0:
            print(f"  Passo di simulazione: {i+1}/{N_SIM}")

        # 1. Risolvi l'MPC per ottenere la sequenza di controlli ottimali
        #    La nuova funzione `forward` restituisce l'intera traiettoria ottimale (X*, U*)
        #    per l'orizzonte H, partendo dallo stato corrente `x_current`.
        _, U_optimal_horizon = mpc.forward(x_current)

        # 2. Applica solo il primo controllo della sequenza ottimale
        u_apply = U_optimal_horizon[:, 0, :]

        # 3. Salva lo stato corrente e il controllo applicato
        us_list.append(u_apply)

        # 4. Propaga la dinamica al passo successivo usando il controllo applicato
        x_current = dyn(x_current, u_apply, DT)
        xs_list.append(x_current)

    # Concatena i risultati in tensori per il plotting
    Xs = torch.stack(xs_list, dim=1)
    Us = torch.stack(us_list, dim=1)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_time_ms = (t_end - t_start) * 1000

    print("Rollout completato.")
    print(f"Tempo totale: {total_time_ms:.2f} ms.")
    print(f"Tempo medio per passo di simulazione: {total_time_ms / N_SIM:.2f} ms.")

    print("Generazione dei grafici...")
    _plot_results(Xs.cpu(), Us.cpu(), DT, x_target.cpu(), BATCH_SIZE)

# ======================== ENTRY-POINT ========================
if __name__ == "__main__":
    main()
