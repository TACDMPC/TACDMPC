# examples/03_demo_unicycle_tracking.py
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importa le classi principali dal pacchetto
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost


# ======================== DEFINIZIONE DEL SISTEMA UNICICLO ========================
def f_dyn_unicycle(state: torch.Tensor, control: torch.Tensor, dt: float) -> torch.Tensor:
    """Dinamica non lineare di un uniciclo, gestisce il batching."""
    x, y, theta = state[..., 0], state[..., 1], state[..., 2]
    v, omega = control[..., 0], control[..., 1]

    x_next = x + v * torch.cos(theta) * dt
    y_next = y + v * torch.sin(theta) * dt
    theta_next = theta + omega * dt

    return torch.stack([x_next, y_next, theta_next], dim=-1)


def f_dyn_jac_unicycle(state: torch.Tensor, control: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobiana analitica della dinamica dell'uniciclo."""
    # Le Jacobiane sono calcolate per un singolo stato (non batched),
    # il controller le vettorizzerà con vmap.
    nx, nu = 3, 2
    theta = state[2]
    v = control[0]

    A = torch.eye(nx, device=state.device, dtype=state.dtype)
    A[0, 2] = -v * torch.sin(theta) * dt
    A[1, 2] = v * torch.cos(theta) * dt

    B = torch.zeros(nx, nu, device=state.device, dtype=state.dtype)
    B[0, 0] = torch.cos(theta) * dt
    B[1, 0] = torch.sin(theta) * dt
    B[2, 1] = dt

    return A, B


# ======================== ROUTINE PRINCIPALE ========================
def main():
    # --- Configurazione Globale ---
    torch.set_default_dtype(torch.double)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione Uniciclo su dispositivo: {DEVICE}")

    BATCH_SIZE = 2
    dt = 0.1
    T = 15  # Orizzonte MPC
    N_sim = 250  # Passi di simulazione
    nx, nu = 3, 2

    # --- Creazione Traiettorie di Riferimento ---
    ref_len = N_sim + T + 1
    t_ref = torch.linspace(0, 4 * torch.pi, ref_len, device=DEVICE)

    # Riferimento Agente 1 (lemniscata grande)
    x_ref_A = 5.0 * torch.sin(t_ref)
    y_ref_A = 5.0 * torch.sin(t_ref) * torch.cos(t_ref)
    theta_ref_A = torch.atan2(torch.gradient(y_ref_A)[0], torch.gradient(x_ref_A)[0])
    ref_A = torch.stack([x_ref_A, y_ref_A, theta_ref_A], dim=1)

    # Riferimento Agente 2 (lemniscata piccola e traslata)
    x_ref_B = 3.0 * torch.sin(t_ref) + 2.0
    y_ref_B = 3.0 * torch.sin(t_ref) * torch.cos(t_ref) - 1.0
    theta_ref_B = torch.atan2(torch.gradient(y_ref_B)[0], torch.gradient(x_ref_B)[0])
    ref_B = torch.stack([x_ref_B, y_ref_B, theta_ref_B], dim=1)

    x_ref_full = torch.stack([ref_A, ref_B], dim=0)

    # Calcolo dei controlli di riferimento (velocità lineare e angolare)
    u_ref_list = []
    for i in range(BATCH_SIZE):
        vx_ref = torch.gradient(x_ref_full[i, :, 0])[0] / dt
        vy_ref = torch.gradient(x_ref_full[i, :, 1])[0] / dt
        v_ref = torch.sqrt(vx_ref ** 2 + vy_ref ** 2)

        # Usa np.unwrap per gestire i salti di angolo da -pi a pi
        unwrapped_theta = np.unwrap(x_ref_full[i, :, 2].cpu().numpy())
        omega_ref = torch.gradient(torch.from_numpy(unwrapped_theta).to(DEVICE))[0] / dt

        u_ref_list.append(torch.stack([v_ref, omega_ref], dim=1)[:-1])
    u_ref_full = torch.stack(u_ref_list, dim=0)

    # --- Setup Costi e MPC ---
    Q = torch.diag(torch.tensor([20.0, 20.0, 5.0], device=DEVICE))
    R = torch.diag(torch.tensor([0.1, 0.1], device=DEVICE))
    C = torch.zeros(T, nx + nu, nx + nu, device=DEVICE);
    C[:, :nx, :nx] = Q;
    C[:, nx:, nx:] = R
    c = torch.zeros(T, nx + nu, device=DEVICE)
    C_final = C[0].clone() * 10
    c_final = torch.zeros(nx + nu, device=DEVICE)
    cost_module = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=str(DEVICE))

    mpc = DifferentiableMPCController(
        f_dyn_unicycle, T * dt, dt, T, cost_module,
        u_min=torch.tensor([-10.0, -2 * np.pi], device=DEVICE),
        u_max=torch.tensor([+10.0, +2 * np.pi], device=DEVICE),
        grad_method="analytic", f_dyn_jac=f_dyn_jac_unicycle,
        reg_eps=1e-6, device=str(DEVICE), N_sim=N_sim
    )

    # --- Esecuzione Simulazione ---
    x0_A = torch.tensor([x_ref_full[0, 0, 0] + 1.0, x_ref_full[0, 0, 1] - 1.0, np.pi / 2], device=DEVICE)
    x0_B = torch.tensor([x_ref_full[1, 0, 0] - 1.0, x_ref_full[1, 0, 1] + 1.0, -np.pi / 2], device=DEVICE)
    x0_batch = torch.stack([x0_A, x0_B], dim=0)

    print(f"Avvio di {BATCH_SIZE} simulazioni MPC per agenti Uniciclo...")
    t_start = time.perf_counter()
    Xs, Us = mpc.forward(x0_batch, x_ref_full=x_ref_full, u_ref_full=u_ref_full)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time_ms = (t_end - t_start) * 1000
    print(f"Simulazione completata in {total_time_ms:.2f} ms.")

    # ======================== PLOTTING ========================
    print("Generazione dei grafici...")
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#0077BB', '#EE7733']
    time_ax_state = torch.arange(N_sim + 1, device='cpu') * dt
    time_ax_ctrl = torch.arange(N_sim, device='cpu') * dt

    # --- Figura 1: Plot 2D delle traiettorie ---
    plt.figure(figsize=(10, 8))
    plt.title("Tracking di Traiettoria per 2 Agenti Uniciclo")
    plt.plot(x_ref_full[0, :, 0].cpu(), x_ref_full[0, :, 1].cpu(), '--', color=colors[0], label='Riferimento Agente 1')
    plt.plot(x_ref_full[1, :, 0].cpu(), x_ref_full[1, :, 1].cpu(), '--', color=colors[1], label='Riferimento Agente 2')
    plt.plot(Xs[0, :, 0].cpu(), Xs[0, :, 1].cpu(), color=colors[0], label='Traiettoria Agente 1', linewidth=2)
    plt.plot(Xs[1, :, 0].cpu(), Xs[1, :, 1].cpu(), color=colors[1], label='Traiettoria Agente 2', linewidth=2)
    plt.scatter(x0_batch[:, 0].cpu(), x0_batch[:, 1].cpu(), c=colors, marker='o', s=100, edgecolors='k', zorder=5,
                label='Punti Iniziali')
    plt.xlabel("Posizione X [m]");
    plt.ylabel("Posizione Y [m]");
    plt.legend();
    plt.axis('equal');
    plt.grid(True)

    # --- Figura 2: Plot Errore Quadratico Medio (MSE) ---
    error_pos = Xs - x_ref_full[:, :N_sim + 1, :]
    error_theta_wrapped = torch.atan2(torch.sin(error_pos[:, :, 2]), torch.cos(error_pos[:, :, 2]))
    error_state = torch.stack([error_pos[:, :, 0], error_pos[:, :, 1], error_theta_wrapped], dim=2)
    mse_state = torch.mean(error_state ** 2, dim=0)

    error_vel = Us - u_ref_full[:, :N_sim, :]
    mse_vel = torch.mean(error_vel ** 2, dim=0)

    plt.figure(figsize=(14, 7))
    plt.title("Errore Quadratico Medio di Tracking (Stato e Controllo)")
    # Errori di stato
    plt.plot(time_ax_state, mse_state[:, 0].cpu(), label='MSE Errore X', color='#0077BB')
    plt.plot(time_ax_state, mse_state[:, 1].cpu(), label='MSE Errore Y', color='#33BBEE')
    plt.plot(time_ax_state, mse_state[:, 2].cpu(), label='MSE Errore Theta', color='#009988')
    # Errori di controllo
    plt.plot(time_ax_ctrl, mse_vel[:, 0].cpu(), label='MSE Errore v', linestyle=':', color='#EE7733')
    plt.plot(time_ax_ctrl, mse_vel[:, 1].cpu(), label='MSE Errore ω', linestyle=':', color='#CC3311')

    plt.xlabel("Tempo [s]");
    plt.ylabel("Errore Quadratico Medio");
    plt.axhline(0.0, color='k', linewidth=0.5, linestyle='--')
    plt.legend();
    plt.grid(True);
    plt.yscale('log')

    # --- Mostra tutte le figure create ---
    plt.tight_layout()
    plt.show()

# ======================== ENTRY-POINT ========================
if __name__ == "__main__":
    main()