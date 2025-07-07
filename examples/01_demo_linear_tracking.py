# examples/01_demo_linear_tracking.py
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importa le classi principali dal tuo nuovo pacchetto
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost


# La definizione della dinamica è ora locale a questo script
def f_dyn_linear(x, u, dt):
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    return torch.einsum("...ij,...j->...i", A, x) + torch.einsum("...ij,...j->...i", B, u)


def f_dyn_jac_linear(x, u, dt):
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    return A, B


def main():
    torch.set_default_dtype(torch.double)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su dispositivo: {device}")

    # --- PARAMETRI ---
    BATCH_SIZE = 50
    dt, horizon, N_sim = 0.05, 10, 150
    nx, nu = 2, 1

    # --- COSTI ---
    Q = torch.diag(torch.tensor([100.0, 50.0], device=device))
    R = torch.diag(torch.tensor([0.001], device=device))
    C = torch.zeros(horizon, nx + nu, nx + nu, device=device);
    C[:, :nx, :nx] = Q;
    C[:, nx:, nx:] = R
    c = torch.zeros(horizon, nx + nu, device=device)
    C_final = C[0].clone() * 10
    c_final = torch.zeros(nx + nu, device=device)
    cost_module = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=str(device))

    # --- SETUP MPC ---
    mpc = DifferentiableMPCController(
        f_dyn=f_dyn_linear,
        total_time=horizon * dt, step_size=dt, horizon=horizon,
        cost_module=cost_module,
        u_min=torch.tensor([-50.0], device=device),
        u_max=torch.tensor([50.0], device=device),
        grad_method="analytic", f_dyn_jac=f_dyn_jac_linear,
        reg_eps=1e-6, device=str(device), N_sim=N_sim
    )

    # --- PREPARAZIONE SIMULAZIONE ---
    x0_base = torch.tensor([10.0, 0.0], device=device)
    x0 = x0_base.repeat(BATCH_SIZE, 1) + (torch.rand(BATCH_SIZE, nx, device=device) - 0.5) * torch.tensor([4.0, 2.0],
                                                                                                          device=device)

    ref_len = N_sim + horizon + 1
    t_full = torch.arange(ref_len, device=device) * dt
    amplitudes = 8.0 + torch.rand(BATCH_SIZE, 1, device=device) * 4.0
    omegas = 1.5 + torch.rand(BATCH_SIZE, 1, device=device) * 1.0
    pos_ref_batch = amplitudes * torch.sin(omegas * t_full.unsqueeze(0))
    vel_ref_batch = amplitudes * omegas * torch.cos(omegas * t_full.unsqueeze(0))
    x_ref_full = torch.stack([pos_ref_batch, vel_ref_batch], dim=2)
    u_ref_full = torch.zeros(BATCH_SIZE, ref_len - 1, nu, device=device)

    print(f"Avvio di {BATCH_SIZE} simulazioni MPC in parallelo...")

    # --- SIMULAZIONE ---
    t0_total = time.perf_counter()
    Xs, Us = mpc.forward(x0, x_ref_full=x_ref_full, u_ref_full=u_ref_full)
    if device.type == "cuda": torch.cuda.synchronize()
    total_time_ms = (time.perf_counter() - t0_total) * 1e3

    print("Simulazione completata.")
    print(f"Tempo totale: {total_time_ms:.2f} ms")

    # --- PLOTTING ---
    N_TO_PLOT = min(10, BATCH_SIZE)
    colors = plt.cm.viridis(np.linspace(0, 1, N_TO_PLOT))
    time_ax = torch.arange(N_sim + 1, device='cpu') * dt

    plt.figure(figsize=(14, 7))
    plt.title(f"Tracking di Posizione per {N_TO_PLOT} Agenti (su {BATCH_SIZE})")
    for i in range(N_TO_PLOT):
        plt.plot(time_ax, Xs[i, :, 0].cpu().numpy(), color=colors[i], label=f'Agente {i + 1}')
        plt.plot(time_ax, x_ref_full[i, :N_sim + 1, 0].cpu().numpy(), '--', color=colors[i], alpha=0.8)
    plt.xlabel("Tempo [s]");
    plt.ylabel("Posizione [m]");
    plt.grid(True);
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()