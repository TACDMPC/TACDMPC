# examples/01_demo_linear_tracking_performance_analysis.py
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importa le classi principali dalla nuova codebase
# Assicurati che i file 'controller.py' e 'cost.py' siano nella stessa directory
# o in un percorso accessibile da Python.
from DifferentialMPC import DifferentiableMPCController
from DifferentialMPC import GeneralQuadCost


# ======================== DEFINIZIONE DEL SISTEMA ========================
def f_dyn_linear(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Dinamica di un semplice sistema lineare (integratore doppio).
    Gestisce il batching automatico.
    Input:
        x: (B, 2) or (2,) - [posizione, velocità]
        u: (B, 1) or (1,) - [accelerazione]
    Output:
        next_state: (B, 2) or (2,)
    """
    # Matrici di stato A e B
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)

    # Propagazione dello stato: x_k+1 = A @ x_k + B @ u_k
    # Per i tensori batchati, la formula corretta è X_k+1 = X_k @ A.T + U_k @ B.T
    # torch.matmul (@) gestisce correttamente sia input batchati (2D) che non (1D),
    # risolvendo il problema con vmap.
    return x @ A.T + u @ B.T


def f_dyn_jac_linear(x: torch.Tensor, u: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobiane analitiche della dinamica lineare."""
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    return A, B


# ======================== FUNZIONI DI PLOTTING E ANALISI ========================
def _plot_performance_analysis(
        Xs: torch.Tensor,
        Us: torch.Tensor,
        x_ref_full: torch.Tensor,
        per_step_times: np.ndarray,
        dt: float,
        batch_size: int,
        sim_len: int
):
    """
    Visualizza una dashboard completa con l'analisi delle prestazioni del solver MPC.
    """
    N_TO_PLOT = min(5, batch_size)
    colors = plt.cm.viridis(np.linspace(0, 1, N_TO_PLOT))
    time_ax_sim = torch.arange(sim_len + 1, device='cpu') * dt
    time_ax_ctrl = torch.arange(sim_len, device='cpu') * dt

    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Analisi delle Prestazioni del Solver MPC", fontsize=18, weight='bold')

    # 1. Grafico del Tracking di Posizione
    ax = axs[0, 0]
    for i in range(N_TO_PLOT):
        ax.plot(time_ax_sim, Xs[i, :, 0].cpu(), color=colors[i], label=f'Agente {i + 1}')
        ax.plot(time_ax_sim, x_ref_full[i, :sim_len + 1, 0].cpu(), '--', color=colors[i], alpha=0.7)
    ax.set_title("Tracking della Posizione")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Posizione [m]")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    # 2. Grafico degli Input di Controllo
    ax = axs[0, 1]
    for i in range(N_TO_PLOT):
        ax.plot(time_ax_ctrl, Us[i, :, 0].cpu(), color=colors[i], alpha=0.8, label=f'Agente {i + 1}')
    ax.set_title("Input di Controllo")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Forza/Accelerazione")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    # 3. Analisi Statistica dell'Errore di Tracking
    ax = axs[1, 0]
    position_error = Xs[:, :, 0] - x_ref_full[:, :sim_len + 1, 0]
    mean_error = position_error.mean(dim=0).cpu()
    std_error = position_error.std(dim=0).cpu()

    ax.plot(time_ax_sim, mean_error, 'r-', label='Errore Medio su Batch')
    ax.fill_between(
        time_ax_sim,
        mean_error - std_error,
        mean_error + std_error,
        color='r', alpha=0.2, label='Deviazione Standard Errore'
    )
    ax.set_title("Analisi Errore di Tracking (Posizione)")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Errore [m]")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    # 4. Analisi dei Tempi di Esecuzione
    ax = axs[1, 1]
    avg_time = per_step_times.mean()
    ax.plot(np.arange(sim_len), per_step_times, 'b-', alpha=0.7, label='Tempo per Passo')
    ax.axhline(avg_time, color='r', linestyle='--', label=f'Tempo Medio: {avg_time:.2f} ms')
    ax.set_title("Analisi Tempi di Esecuzione per Passo")
    ax.set_xlabel("Passo di Simulazione")
    ax.set_ylabel("Tempo di Calcolo [ms]")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ======================== SCRIPT PRINCIPALE ========================
def main():
    torch.set_default_dtype(torch.double)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su dispositivo: {DEVICE}")

    # --- PARAMETRI ---
    BATCH_SIZE = 50
    DT = 0.05
    HORIZON = 20
    N_SIM = 150

    nx, nu = 2, 1

    # --- COSTI ---
    Q = torch.diag(torch.tensor([100.0, 1.0], device=DEVICE))
    R = torch.diag(torch.tensor([0.01], device=DEVICE))
    C = torch.zeros(HORIZON, nx + nu, nx + nu, device=DEVICE)
    C[:, :nx, :nx] = Q
    C[:, nx:, nx:] = R
    c = torch.zeros(HORIZON, nx + nu, device=DEVICE)
    C_final = C[0].clone() * 10
    c_final = torch.zeros(nx + nu, device=DEVICE)
    cost_module = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=str(DEVICE))

    # --- SETUP MPC ---
    mpc = DifferentiableMPCController(
        f_dyn=f_dyn_linear,
        total_time=HORIZON * DT,
        step_size=DT,
        horizon=HORIZON,
        cost_module=cost_module,
        u_min=torch.tensor([-50.0], device=DEVICE),
        u_max=torch.tensor([50.0], device=DEVICE),
        grad_method="analytic",
        f_dyn_jac=f_dyn_jac_linear,
        device=str(DEVICE)
    )

    # --- PREPARAZIONE SIMULAZIONE ---
    x0_base = torch.tensor([0.0, 0.0], device=DEVICE)
    x0 = x0_base.repeat(BATCH_SIZE, 1) + (torch.rand(BATCH_SIZE, nx, device=DEVICE) - 0.5) * torch.tensor([2.0, 1.0],
                                                                                                          device=DEVICE)

    ref_len = N_SIM + HORIZON + 1
    t_full = torch.arange(ref_len, device=DEVICE) * DT
    amplitudes = 5.0 + torch.rand(BATCH_SIZE, 1, device=DEVICE) * 5.0
    omegas = 1.0 + torch.rand(BATCH_SIZE, 1, device=DEVICE) * 1.0
    pos_ref_batch = amplitudes * torch.sin(omegas * t_full.unsqueeze(0))
    vel_ref_batch = amplitudes * omegas * torch.cos(omegas * t_full.unsqueeze(0))
    x_ref_full = torch.stack([pos_ref_batch, vel_ref_batch], dim=2)
    u_ref_full = torch.zeros(BATCH_SIZE, ref_len - 1, nu, device=DEVICE)

    print(f"Avvio di {BATCH_SIZE} simulazioni MPC di tracking in parallelo...")

    # --- SIMULAZIONE (ANELLO CHIUSO ESPLICITO) ---
    x_current = x0
    xs_list = [x_current]
    us_list = []
    per_step_times = []

    t0_total = time.perf_counter()
    for k in range(N_SIM):
        if (k + 1) % 25 == 0:
            print(f"  Passo di simulazione: {k + 1}/{N_SIM}")

        xr_win = x_ref_full[:, k: k + HORIZON + 1]
        ur_win = u_ref_full[:, k: k + HORIZON]
        mpc.cost_module.set_reference(x_ref=xr_win, u_ref=ur_win)

        t_step_start = time.perf_counter()
        _, U_optimal_horizon = mpc.forward(x_current)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t_step_end = time.perf_counter()
        per_step_times.append((t_step_end - t_step_start) * 1000)  # in ms

        u_apply = U_optimal_horizon[:, 0, :]
        us_list.append(u_apply)
        x_current = f_dyn_linear(x_current, u_apply, DT)
        xs_list.append(x_current)

    total_sim_time_s = time.perf_counter() - t0_total

    # --- AGGREGAZIONE RISULTATI ---
    Xs = torch.stack(xs_list, dim=1)
    Us = torch.stack(us_list, dim=1)
    per_step_times_np = np.array(per_step_times)

    # --- STAMPA METRICHE DI PERFORMANCE ---
    position_error = Xs[:, :, 0] - x_ref_full[:, :N_SIM + 1, 0]
    mse = torch.mean(position_error ** 2).item()
    final_error = torch.mean(torch.abs(position_error[:, -1])).item()

    print("\n--- Simulazione Completata ---")
    print(f"Tempo totale di esecuzione (wall-clock): {total_sim_time_s:.3f} s")
    print("\n--- Statistiche sui Tempi di Calcolo (per passo) ---")
    print(f"  - Media:   {per_step_times_np.mean():.2f} ms")
    print(f"  - Mediana: {np.median(per_step_times_np):.2f} ms")
    print(f"  - Dev Std: {per_step_times_np.std():.2f} ms")
    print(f"  - Min:     {per_step_times_np.min():.2f} ms")
    print(f"  - Max:     {per_step_times_np.max():.2f} ms")
    print("\n--- Metriche di Tracking ---")
    print(f"  - Errore Quadratico Medio (MSE) di Posizione: {mse:.4f}")
    print(f"  - Errore Assoluto Medio Finale:               {final_error:.4f}")

    # --- PLOTTING ---
    print("\nGenerazione della dashboard di analisi...")
    _plot_performance_analysis(Xs.cpu(), Us.cpu(), x_ref_full.cpu(), per_step_times_np, DT, BATCH_SIZE, N_SIM)


if __name__ == "__main__":
    main()
