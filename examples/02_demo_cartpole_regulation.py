# examples/02_demo_cartpole_comparison.py
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout
from typing import Dict
from DifferentialMPC import DifferentiableMPCController, GradMethod
from DifferentialMPC import GeneralQuadCost

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
    """Dinamica non lineare del Cart-Pole (supporta batch)."""
    pos, vel, theta, omega = x.split(1, dim=-1)
    force = u
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass, m_p_l = p.m_c + p.m_p, p.m_p * p.l
    temp = (force + m_p_l * omega.pow(2) * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0 / 3.0 - p.m_p * cos_t.pow(2) / total_mass))
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

def _plot_comparison(results_analytic: Dict, results_autodiff: Dict):
    fig, axs = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [2, 1, 1]})
    fig.suptitle("Confronto Prestazioni: Jacobiano Analitico vs. Auto-Diff (Cart-Pole)", fontsize=18, weight='bold')
    ax = axs[0]
    agent_to_plot = 0
    sim_len = results_analytic['Xs'].shape[1] - 1
    dt = results_analytic['dt']
    time_ax = torch.arange(sim_len + 1) * dt
    ax.plot(time_ax, results_analytic['Xs'][agent_to_plot, :, 2].cpu(), 'b-', label='Angolo (Analitico)')
    ax.plot(time_ax, results_autodiff['Xs'][agent_to_plot, :, 2].cpu(), 'g-.', label='Angolo (Auto-Diff)', alpha=0.8)
    ax.axhline(0, color='r', linestyle='--', label='Target')
    ax.set_title(f"Regolazione dell'Angolo (Agente #{agent_to_plot + 1})")
    ax.set_xlabel("Tempo [s]"), ax.set_ylabel("Angolo [rad]")
    ax.grid(True, linestyle='--', linewidth=0.5), ax.legend()

    ax = axs[1]
    times_data = [results_analytic['per_step_times'], results_autodiff['per_step_times']]
    labels = ['Analitico', 'Auto-Diff']
    box = ax.boxplot(times_data, patch_artist=True, labels=labels)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
    ax.set_title("Distribuzione Tempi di Esecuzione"), ax.set_ylabel("Tempo per Passo [ms]")
    ax.grid(True, linestyle='--', linewidth=0.5, axis='y')

    ax = axs[2]
    final_errors = [results_analytic['final_error'], results_autodiff['final_error']]
    bars = ax.bar(labels, final_errors, color=colors, edgecolor='black')
    ax.set_title("Errore Assoluto Medio Finale (Angolo)"), ax.set_ylabel("Errore [rad]")
    ax.grid(True, linestyle='--', linewidth=0.5, axis='y')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
    fig.tight_layout(rect=[0, 0, 1, 0.95]), plt.show()

def run_simulation(grad_method: str, f_dyn_jac, common_params: Dict) -> Dict:
    DEVICE, BATCH_SIZE, DT, HORIZON, N_SIM = [common_params[k] for k in
                                              ['DEVICE', 'BATCH_SIZE', 'DT', 'HORIZON', 'N_SIM']]
    nx, nu, params = 4, 1, common_params['params']
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)

    print(f"\n--- Avvio Simulazione con grad_method = '{grad_method.upper()}' ---")

    x_target = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
    x_ref = x_target.repeat(BATCH_SIZE, HORIZON + 1, 1)
    u_ref = torch.zeros(BATCH_SIZE, HORIZON, nu, device=DEVICE)
    Q = torch.diag(torch.tensor([10.0, 1.0, 100.0, 1.0], device=DEVICE))
    R = torch.diag(torch.tensor([0.1], device=DEVICE))
    C = torch.zeros(HORIZON, nx + nu, nx + nu, device=DEVICE)
    C[:, :nx, :nx], C[:, nx:, nx:] = Q, R
    c = torch.zeros(HORIZON, nx + nu, device=DEVICE)
    C_final = torch.zeros(nx + nu, nx + nu, device=DEVICE)
    C_final[:nx, :nx] = Q * 10
    c_final = torch.zeros(nx + nu, device=DEVICE)
    cost_module = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=str(DEVICE), x_ref=x_ref, u_ref=u_ref)

    mpc = DifferentiableMPCController(
        f_dyn=dyn,
        total_time=HORIZON * DT*5,
        cost_module=cost_module,
        horizon=HORIZON,
        step_size=DT,
        u_min=torch.tensor([-20.0], device=DEVICE),
        u_max=torch.tensor([20.0], device=DEVICE),
        grad_method=grad_method,
        f_dyn_jac=f_dyn_jac,
        device=str(DEVICE)
    )

    x_current = common_params['x0']
    xs_list, us_list, per_step_times = [x_current], [], []
    for i in range(N_SIM):
        t_step = time.perf_counter()
        _, U_opt = mpc.forward(x_current)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        per_step_times.append((time.perf_counter() - t_step) * 1000)
        u_apply = U_opt[:, 0, :]
        us_list.append(u_apply)
        x_current = dyn(x_current, u_apply, DT)
        xs_list.append(x_current)

    Xs, Us = torch.stack(xs_list, dim=1), torch.stack(us_list, dim=1)
    final_error = torch.mean(torch.abs(Xs[:, -1, 2])).item()
    print(f"Tempo medio per passo: {np.mean(per_step_times):.2f} ms")
    print(f"Errore finale medio (angolo): {final_error:.4f} rad")
    return {"Xs": Xs, "Us": Us, "per_step_times": np.array(per_step_times), "final_error": final_error, "dt": DT}


# ======================== SCRIPT PRINCIPALE ========================
def main():
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione Cart-Pole su dispositivo: {DEVICE}")

    common_params = {
        'DEVICE': DEVICE, 'BATCH_SIZE': 100, 'DT': 0.05,
        'HORIZON': 15, 'N_SIM': 100, 'params': CartPoleParams.from_gym()
    }
    base_state = torch.tensor([0.0, 0.0, 0.2, 0.0], device=DEVICE)
    common_params['x0'] = base_state + torch.randn(common_params['BATCH_SIZE'], 4, device=DEVICE) * torch.tensor(
        [0.5, 0.5, 0.2, 0.2], device=DEVICE)
    jac_fn_batched = lambda x, u, dt: f_cartpole_jac_batched(x, u, dt, common_params['params'])

    results_analytic = run_simulation("analytic", jac_fn_batched, common_params)
    results_autodiff = run_simulation("auto_diff", None, common_params)

    avg_analytic = results_analytic['per_step_times'].mean()
    avg_autodiff = results_autodiff['per_step_times'].mean()
    print("\n" + "=" * 50)
    print("📊 RIEPILOGO PRESTAZIONI (CART-POLE) 📊")
    print("=" * 50)
    print(f"Tempo Medio Passo (Analitico Vettorizzato): {avg_analytic:.2f} ms")
    print(f"Tempo Medio Passo (Auto-Diff Vettorizzato): {avg_autodiff:.2f} ms")
    if avg_analytic < avg_autodiff:
        print(f" Il metodo analitico è stato {avg_autodiff / avg_analytic:.2f} volte più veloce.")
    else:
        print(f" Il metodo auto-diff è stato {avg_analytic / avg_autodiff:.2f} volte più veloce.")
    print("-" * 50)
    print(f"Errore Finale (Analitico): {results_analytic['final_error']:.4f} rad")
    print(f"Errore Finale (Auto-Diff): {results_autodiff['final_error']:.4f} rad")
    print("=" * 50)

    print("\nGenerazione della dashboard di confronto...")
    _plot_comparison(results_analytic, results_autodiff)


if __name__ == "__main__":
    main()