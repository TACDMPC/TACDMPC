import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from DifferentialMPC import DifferentiableMPCController
from DifferentialMPC import GeneralQuadCost


# ======================== DEFINIZIONE DEL SISTEMA ========================
def f_dyn_linear(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """Dinamica di un semplice sistema lineare (integratore doppio)."""
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    return x @ A.T + u @ B.T


def f_dyn_jac_linear_batched(x: torch.Tensor, u: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobiane analitiche ."""
    batch_size = x.shape[0]
    A_base = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B_base = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    A = A_base.unsqueeze(0).expand(batch_size, -1, -1)
    B = B_base.unsqueeze(0).expand(batch_size, -1, -1)
    return A, B


def _plot_performance_analysis(
        title: str, Xs: torch.Tensor, Us: torch.Tensor, x_ref_full: torch.Tensor,
        per_step_times: np.ndarray, dt: float, batch_size: int, sim_len: int):
    N_TO_PLOT = min(5, batch_size)
    colors = plt.cm.viridis(np.linspace(0, 1, N_TO_PLOT))
    time_ax_sim = torch.arange(sim_len + 1) * dt
    time_ax_ctrl = torch.arange(sim_len) * dt
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f"Analisi Prestazioni Solver MPC - {title}", fontsize=18, weight='bold')
    ax = axs[0, 0]
    for i in range(N_TO_PLOT):
        ax.plot(time_ax_sim, Xs[i, :, 0].cpu(), color=colors[i], label=f'Agente {i + 1}')
        ax.plot(time_ax_sim, x_ref_full[i, :sim_len + 1, 0].cpu(), '--', color=colors[i], alpha=0.7)
    ax.set_title("Tracking della Posizione"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel("Posizione [m]"), ax.grid(
        True), ax.legend()
    ax = axs[0, 1]
    for i in range(N_TO_PLOT):
        ax.plot(time_ax_ctrl, Us[i, :, 0].cpu(), color=colors[i], alpha=0.8, label=f'Agente {i + 1}')
    ax.set_title("Input di Controllo"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel("Forza/Accelerazione"), ax.grid(
        True), ax.legend()
    ax = axs[1, 0]
    position_error = Xs[:, :, 0] - x_ref_full[:, :sim_len + 1, 0]
    mean_error = position_error.mean(dim=0).cpu()
    std_error = position_error.std(dim=0).cpu()
    ax.plot(time_ax_sim, mean_error, 'r-', label='Errore Medio su Batch')
    ax.fill_between(time_ax_sim, mean_error - std_error, mean_error + std_error, color='r', alpha=0.2,
                    label='Dev. Std.')
    ax.set_title("Analisi Errore di Tracking"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel("Errore [m]"), ax.grid(
        True), ax.legend()
    ax = axs[1, 1]
    avg_time = per_step_times.mean()
    ax.plot(np.arange(sim_len), per_step_times, 'b-', alpha=0.7, label='Tempo per Passo')
    ax.axhline(avg_time, color='r', linestyle='--', label=f'Tempo Medio: {avg_time:.2f} ms')
    ax.set_title("Analisi Tempi di Esecuzione"), ax.set_xlabel("Passo di Simulazione"), ax.set_ylabel(
        "Tempo [ms]"), ax.grid(True), ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.96]), plt.show()


def _plot_comparison(results_analytic: Dict, results_autodiff: Dict):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Confronto Prestazioni: Jacobiano Analitico vs. Auto-Diff", fontsize=18, weight='bold')
    ax = axs[0]
    times_data = [results_analytic['per_step_times'], results_autodiff['per_step_times']]
    labels = ['Analitico', 'Auto-Diff']
    box = ax.boxplot(times_data, patch_artist=True, labels=labels)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
    ax.set_title("Distribuzione Tempi di Esecuzione per Passo"), ax.set_ylabel("Tempo di Calcolo [ms]")
    ax.grid(True, linestyle='--', linewidth=0.5, axis='y')
    ax = axs[1]
    mses = [results_analytic['mse'], results_autodiff['mse']]
    bars = ax.bar(labels, mses, color=colors, edgecolor='black')
    ax.set_title("Errore Quadratico Medio (MSE) Finale"), ax.set_ylabel("MSE Posizione")
    ax.grid(True, linestyle='--', linewidth=0.5, axis='y')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
    fig.tight_layout(rect=[0, 0, 1, 0.95]), plt.show()


# ======================== SIMULAZIONE ========================
def run_simulation(grad_method: str, f_dyn_jac, common_params: Dict) -> Dict:
    """Esegue una singola simulazione, stampa i risultati, mostra i grafici e restituisce le metriche."""
    DEVICE, BATCH_SIZE, DT, HORIZON, N_SIM = [common_params[k] for k in
                                              ['DEVICE', 'BATCH_SIZE', 'DT', 'HORIZON', 'N_SIM']]
    nx, nu = 2, 1

    title = f"Metodo: {grad_method.upper()}"
    print("\n" + "=" * 70)
    print(f"🚀 Avvio Simulazione con {grad_method.upper()} 🚀")
    print("=" * 70)

    cost_module = common_params['cost_module']
    mpc = DifferentiableMPCController(
        f_dyn=f_dyn_linear, total_time=HORIZON * DT, step_size=DT, horizon=HORIZON,
        cost_module=cost_module, u_min=torch.tensor([-50.0], device=DEVICE),
        u_max=torch.tensor([50.0], device=DEVICE), grad_method=grad_method,
        f_dyn_jac=f_dyn_jac, device=str(DEVICE)
    )

    x_current = common_params['x0']
    x_ref_full = common_params['x_ref_full']
    u_ref_full = common_params['u_ref_full']
    xs_list, us_list, per_step_times = [x_current], [], []

    for k in range(N_SIM):
        xr_win = x_ref_full[:, k: k + HORIZON + 1]
        ur_win = u_ref_full[:, k: k + HORIZON]
        mpc.cost_module.set_reference(x_ref=xr_win, u_ref=ur_win)
        t_step_start = time.perf_counter()
        _, U_optimal_horizon = mpc.forward(x_current)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        per_step_times.append((time.perf_counter() - t_step_start) * 1000)
        u_apply = U_optimal_horizon[:, 0, :]
        us_list.append(u_apply)
        x_current = f_dyn_linear(x_current, u_apply, DT)
        xs_list.append(x_current)

    Xs, Us = torch.stack(xs_list, dim=1), torch.stack(us_list, dim=1)
    per_step_times_np = np.array(per_step_times)
    position_error = Xs[:, :, 0] - x_ref_full[:, :N_SIM + 1, 0]
    mse = torch.mean(position_error ** 2).item()

    print(f"\n--- Risultati per {grad_method.upper()} ---")
    print(f"  - Media Tempo per Passo: {per_step_times_np.mean():.2f} ms")
    print(f"  - Errore Quadratico Medio (MSE): {mse:.4f}")

    print("\nGenerazione della dashboard di analisi dettagliata...")
    _plot_performance_analysis(title, Xs.cpu(), Us.cpu(), x_ref_full.cpu(), per_step_times_np, DT, BATCH_SIZE, N_SIM)

    return {"per_step_times": per_step_times_np, "mse": mse}


# ======================== SCRIPT PRINCIPALE ========================
def main():
    torch.set_default_dtype(torch.double)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su dispositivo: {DEVICE}")

    common_params = {'DEVICE': DEVICE, 'BATCH_SIZE': 50, 'DT': 0.05, 'HORIZON': 20, 'N_SIM': 150}
    nx, nu = 2, 1

    Q = torch.diag(torch.tensor([100.0, 1.0], device=DEVICE))
    R = torch.diag(torch.tensor([0.01], device=DEVICE))
    C = torch.zeros(common_params['HORIZON'], nx + nu, nx + nu, device=DEVICE)
    C[:, :nx, :nx], C[:, nx:, nx:] = Q, R
    c = torch.zeros(common_params['HORIZON'], nx + nu, device=DEVICE)
    C_final = C[0].clone() * 10
    c_final = torch.zeros(nx + nu, device=DEVICE)
    common_params['cost_module'] = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=str(DEVICE))

    x0 = torch.zeros(common_params['BATCH_SIZE'], nx, device=DEVICE)
    ref_len = common_params['N_SIM'] + common_params['HORIZON'] + 1
    t_full = torch.arange(ref_len, device=DEVICE) * common_params['DT']
    amplitudes = 5.0 + torch.rand(common_params['BATCH_SIZE'], 1, device=DEVICE) * 5.0
    omegas = 1.0 + torch.rand(common_params['BATCH_SIZE'], 1, device=DEVICE) * 1.0
    pos_ref = amplitudes * torch.sin(omegas * t_full.unsqueeze(0))
    vel_ref = amplitudes * omegas * torch.cos(omegas * t_full.unsqueeze(0))
    common_params['x0'] = x0
    common_params['x_ref_full'] = torch.stack([pos_ref, vel_ref], dim=2)
    common_params['u_ref_full'] = torch.zeros(common_params['BATCH_SIZE'], ref_len - 1, nu, device=DEVICE)

    results_analytic = run_simulation("analytic", f_dyn_jac_linear_batched, common_params)
    results_autodiff = run_simulation("auto_diff", None, common_params)

    print("\n" + "=" * 50)
    print("📊 RIEPILOGO FINALE DEL CONFRONTO 📊")
    print("=" * 50)
    avg_analytic = results_analytic['per_step_times'].mean()
    avg_autodiff = results_autodiff['per_step_times'].mean()
    print(f"Tempo Medio Passo (Analitico Vettorizzato): {avg_analytic:.2f} ms")
    print(f"Tempo Medio Passo (Auto-Diff Vettorizzato): {avg_autodiff:.2f} ms")
    if avg_analytic < avg_autodiff:
        print(f" Il metodo analitico è stato {avg_autodiff / avg_analytic:.2f} volte più veloce.")
    else:
        print(f" Il metodo auto-diff è stato {avg_analytic / avg_autodiff:.2f} volte più veloce.")
    print("-" * 50)
    print(f"MSE Finale (Analitico): {results_analytic['mse']:.4f}")
    print(f"MSE Finale (Auto-Diff): {results_autodiff['mse']:.4f}")
    print("=" * 50)

    print("\nGenerazione della dashboard di confronto...")
    _plot_comparison(results_analytic, results_autodiff)


if __name__ == "__main__":
    main()