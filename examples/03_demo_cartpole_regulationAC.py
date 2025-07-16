import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ======================================================================================
# --- 1. IMPORT DEI COMPONENTI DEFINITI ESTERNAMENTE ---
# ======================================================================================

# Importa i componenti del controllore MPC differenziabile
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod

# Importa i componenti dell'agente AC-MPC che abbiamo definito
from ACMPC import ActorMPC
from ACMPC import CriticTransformer
from ACMPC import train


# ======================================================================================
# --- 2. DEFINIZIONE DEL TASK SPECIFICO (CART-POLE) ---
# ======================================================================================

@dataclass(frozen=True)
class CartPoleParams:
    m_c: float = 1.0
    m_p: float = 0.1
    l: float = 0.5
    g: float = 9.81


def f_cartpole(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    """Dinamica non lineare del Cart-Pole, gestisce il batching."""
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass, m_p_l = p.m_c + p.m_p, p.m_p * p.l
    temp = (force + m_p_l * omega ** 2 * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0 / 3.0 - p.m_p * cos_t ** 2 / total_mass))
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass
    return torch.stack((pos + vel * dt, vel + vel_dd * dt, theta + omega * dt, omega + theta_dd * dt), dim=-1)


class CartPoleEnv:
    """Wrapper per l'ambiente per compatibilità con il training loop."""

    def __init__(self, dyn_fn, reward_fn, dt, sim_horizon, device):
        self.dyn = dyn_fn
        self.reward_fn = reward_fn
        self.dt = dt
        self.sim_horizon = sim_horizon
        self.device = device
        self.state = None

    def reset(self):
        # Stato iniziale casuale vicino al punto di equilibrio instabile
        self.state = (torch.rand(4, device=self.device) - 0.5) * torch.tensor([0.1, 0.1, 0.4, 0.4], device=self.device)
        return self.state

    def step(self, action):
        prev_state = self.state
        self.state = self.dyn(self.state.unsqueeze(0), action.unsqueeze(0), self.dt).squeeze(0)
        reward = self.reward_fn(prev_state.unsqueeze(0), action.unsqueeze(0), self.state.unsqueeze(0)).item()
        # Condizione di fine episodio
        done = abs(self.state[0]) > 2.4 or abs(self.state[2]) > 0.209
        return self.state, reward, done


# ======================== FUNZIONI DI PLOTTING ========================
def _plot_comparison(xs_base, us_base, xs_acmpc, us_acmpc, dt, target, batch_size):
    """Visualizza i risultati del confronto."""
    N_TO_PLOT = min(5, batch_size)
    labels = ["Posizione [m]", "Velocità [m/s]", "Angolo [rad]", "Vel. Angolare [rad/s]"]
    time_state = torch.arange(xs_base.shape[1]) * dt
    time_ctrl = torch.arange(us_base.shape[1]) * dt
    colors = plt.cm.viridis(np.linspace(0, 1, N_TO_PLOT))

    fig, axs = plt.subplots(xs_base.shape[-1], 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Confronto Traiettorie: MPC Base (--) vs AC-MPC (-)")

    for i in range(xs_base.shape[-1]):
        for b in range(N_TO_PLOT):
            label_base = "MPC Base" if b == 0 else None
            label_acmpc = "AC-MPC" if b == 0 else None
            axs[i].plot(time_state, xs_base[b, :, i], color=colors[b], linestyle='--', label=label_base)
            axs[i].plot(time_state, xs_acmpc[b, :, i], color=colors[b], linestyle='-', label=label_acmpc)
        axs[i].axhline(target[i].item(), linestyle=":", color="k")
        axs[i].set_ylabel(labels[i]);
        axs[i].grid(True)

    fig.legend(loc='upper right')
    axs[-1].set_xlabel("Tempo [s]")

    plt.figure(figsize=(14, 6))
    plt.title("Confronto Comandi di Controllo")
    for b in range(N_TO_PLOT):
        plt.plot(time_ctrl, us_base[b, :, 0], color=colors[b], linestyle='--', label="MPC Base" if b == 0 else None)
        plt.plot(time_ctrl, us_acmpc[b, :, 0], color=colors[b], linestyle='-', label="AC-MPC" if b == 0 else None)
    plt.xlabel("Tempo [s]");
    plt.ylabel("Forza [N]");
    plt.grid(True);
    plt.legend()

    plt.show()


# ======================== 3. SCRIPT PRINCIPALE DI ORCHESTRAZIONE ========================
def main():
    # --- Configurazione Globale ---
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione Cart-Pole su dispositivo: {DEVICE}")

    # --- Iperparametri ---
    BATCH_SIZE_EVAL = 50
    DT = 0.05
    MPC_HORIZON = 20
    SIM_HORIZON_TRAIN = 100
    SIM_HORIZON_EVAL = 150
    TRAINING_STEPS = 100  # Aumentare per un training più approfondito
    CRITIC_HISTORY_LEN = 10

    # --- Setup Ambiente e Dinamica ---
    params = CartPoleParams()
    nx, nu = 4, 1
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)

    # --- Funzione di Reward (usata per training e per definire il costo base) ---
    Q_reward = torch.diag(torch.tensor([10.0, 1.0, 100.0, 1.0], device=DEVICE))
    R_reward = torch.diag(torch.tensor([0.1], device=DEVICE))

    def cartpole_reward_fn(states, actions, _):
        state_cost = torch.einsum('...i,ij,...j->...', states, Q_reward, states)
        action_cost = torch.einsum('...i,ij,...j->...', actions, R_reward, actions)
        return -(state_cost + action_cost)
    print("\n--- 1. Inizio Addestramento Agente AC-MPC ---")
    env_train = CartPoleEnv(dyn, cartpole_reward_fn, DT, SIM_HORIZON_TRAIN, DEVICE)
    actor_mpc = ActorMPC(nx, nu, MPC_HORIZON, DT, dyn, device=str(DEVICE))
    critic_mpc = CriticTransformer(nx, nu, CRITIC_HISTORY_LEN, MPC_HORIZON)
    critic_mpc.to(device=DEVICE, dtype=torch.get_default_dtype())
    train(
        env=env_train,
        actor=actor_mpc,
        critic=critic_mpc,
        reward_fn=cartpole_reward_fn,
        steps=TRAINING_STEPS,
        mpc_horizon=MPC_HORIZON
    )
    # --------------------

    actor_mpc.eval()
    print("--- Addestramento AC-MPC Completato ---\n")

    # --- 2. Setup del Controllore MPC Base ---
    print("--- 2. Setup Controllore MPC Base (con costi fissi) ---")
    x_target = torch.zeros(nx, device=DEVICE)
    C_base = torch.zeros(MPC_HORIZON, nx + nu, nx + nu, device=DEVICE)
    C_base[:, :nx, :nx] = Q_reward
    C_base[:, nx:, nx:] = R_reward
    c_base = torch.zeros(MPC_HORIZON, nx + nu, device=DEVICE)
    C_final_base = torch.zeros(nx + nu, nx + nu, device=DEVICE)
    C_final_base[:nx, :nx] = Q_reward * 10
    c_final_base = torch.zeros(nx + nu, device=DEVICE)
    cost_base = GeneralQuadCost(
        nx=nx, nu=nu, C=C_base, c=c_base, C_final=C_final_base, c_final=c_final_base,
        device=str(DEVICE), x_ref=x_target.repeat(MPC_HORIZON + 1, 1), u_ref=torch.zeros(MPC_HORIZON, nu, device=DEVICE)
    )
    mpc_base = DifferentiableMPCController(
        f_dyn=dyn, total_time=MPC_HORIZON * DT, step_size=DT, horizon=MPC_HORIZON,
        cost_module=cost_base, u_min=torch.tensor([-20.0]), u_max=torch.tensor([20.0]),
        N_sim=SIM_HORIZON_EVAL, device=str(DEVICE)
    )

    # --- 3. Esecuzione e Valutazione ---
    print("--- 3. Esecuzione e Valutazione dei Controllori ---")
    x0_eval = (torch.rand(BATCH_SIZE_EVAL, nx, device=DEVICE) - 0.5) * torch.tensor([1.0, 1.0, 0.5, 0.5], device=DEVICE)

    # Rollout del MPC Base
    print("Eseguendo il rollout per MPC Base...")
    Xs_base, Us_base = mpc_base.forward(x0_eval)

    # Rollout dell'AC-MPC addestrato
    print("Eseguendo il rollout per AC-MPC...")
    xs_acmpc_list, us_acmpc_list = [x0_eval], []
    x_current = x0_eval
    with torch.no_grad():  # Non serve calcolare gradienti durante la valutazione
        for _ in range(SIM_HORIZON_EVAL):
            action, _, _, _ = actor_mpc(x_current, deterministic=True)
            us_acmpc_list.append(action)
            x_current = dyn(x_current, action, DT)
            xs_acmpc_list.append(x_current)
    Xs_acmpc = torch.stack(xs_acmpc_list, dim=1)
    Us_acmpc = torch.stack(us_acmpc_list, dim=1)
    print("--- Valutazione Completata ---\n")

    # --- 4. Plot dei Risultati ---
    print("--- 4. Generazione Grafici di Confronto ---")
    _plot_comparison(Xs_base.cpu(), Us_base.cpu(), Xs_acmpc.cpu(), Us_acmpc.cpu(), DT, x_target.cpu(), BATCH_SIZE_EVAL)


if __name__ == "__main__":
    main()
