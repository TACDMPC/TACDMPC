import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import gym
from gym import spaces
import matplotlib.pyplot as plt
from torch import Tensor
# Importa le classi e le funzioni dai pacchetti del progetto
from ACMPC import ActorMPC
from ACMPC import CriticTransformer
from ACMPC import train
from DifferentialMPC import DifferentiableMPCController, GradMethod
from ACMPC import CheckpointManager

# --- Parametri di sistema e ambiente ---
NX, NU, DT = 2, 1, 0.1


def f_dyn_batched(x, u, dt=DT):
    p, v = torch.split(x, 1, dim=-1)
    u = u.to(p)
    p_new, v_new = p + v * dt, v + u * dt
    return torch.cat([p_new, v_new], dim=-1)


def f_dyn_jac_batched(x, u, dt=DT):
    B = x.shape[0]
    A = torch.tensor([[1, dt], [0, 1]], device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
    B_mat = torch.tensor([[0], [dt]], device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
    return A, B_mat


def reward_fn_batched(
        states: Tensor,
        actions: Tensor,
        next_states: Tensor
) -> Tensor:
    """
    Una funzione di reward più elaborata per la stabilizzazione del doppio integratore.

    Combina tre elementi:
    1.  Costo per la deviazione dalla posizione target (0.0).
    2.  Costo per la velocità non nulla (per incoraggiare a fermarsi).
    3.  Premio per il progresso (riduzione della distanza dal target).
    4.  Costo per lo sforzo di controllo.
    """
    device, dtype = states.device, states.dtype

    # --- 1. Definizione dei pesi e del target ---
    target_pos = 0.0
    target_vel = 0.0

    # Pesi per bilanciare i diversi obiettivi
    pos_weight = 1.5  # Penalizza molto l'errore di posizione
    vel_weight = 0.2  # Penalizza la velocità non nulla
    control_weight = 0.01  # Incoraggia azioni "morbide"
    progress_weight = 10.0  # Premia molto l'avvicinamento al target

    # --- 2. Calcolo dei costi (penalità) ---
    # Costo quadratico per l'errore di posizione e velocità
    pos_error = (states[..., 0] - target_pos) ** 2
    vel_error = (states[..., 1] - target_vel) ** 2

    # Costo per lo sforzo di controllo
    control_effort = torch.sum(actions ** 2, dim=-1)

    # Costo totale dello stato e del controllo correnti
    total_cost = (pos_weight * pos_error) + (vel_weight * vel_error) + (control_weight * control_effort)

    # --- 3. Calcolo del premio per il progresso ---
    # Distanza (quadratica) dal target prima e dopo l'azione
    dist_before = torch.sum(states ** 2, dim=-1)
    dist_after = torch.sum(next_states ** 2, dim=-1)

    # Il premio è proporzionale alla riduzione della distanza
    progress_reward = progress_weight * (dist_before - dist_after)

    # --- 4. Reward finale ---
    # La reward finale è il progresso fatto, meno il costo per essere nello stato attuale.
    # Questo spinge l'agente a raggiungere il target il più velocemente possibile.
    reward = progress_reward - total_cost

    return reward

class DoubleIntegratorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.state = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(NU,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(NX,), dtype=np.float32)

    def reset(self, initial_state=None):
        self.state = initial_state.astype(np.float32) if initial_state is not None else (
                    np.random.rand(NX) * 4 - 2).astype(np.float32)
        return self.state

    def step(self, action):
        state_tensor = torch.from_numpy(self.state).unsqueeze(0)
        action_tensor = torch.from_numpy(np.array(action)).unsqueeze(0)
        next_state_tensor = f_dyn_batched(state_tensor, action_tensor).squeeze(0)
        reward = reward_fn_batched(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0),
                                   next_state_tensor.unsqueeze(0)).item()
        self.state = next_state_tensor.numpy()
        return self.state, reward, False, {}


def env_factory():
    return DoubleIntegratorEnv()


def evaluate_and_plot(actor, env, device, file_prefix, title):
    print(f"📊 Eseguendo valutazione per: {title}")
    Path("plots").mkdir(exist_ok=True)
    state = env.reset(initial_state=np.array([2.0, 1.0]))
    states, actions = [state], []
    for _ in range(100):
        state_tensor = torch.from_numpy(state).to(device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): action, _, _, _ = actor(state_tensor, deterministic=True)
        action_np = action.cpu().numpy().flatten()
        state, _, _, _ = env.step(action_np)
        states.append(state)
        actions.append(action_np)
    states, actions = np.array(states), np.array(actions)
    time = np.arange(states.shape[0]) * DT

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Comportamento: {title}', fontsize=16)
    axs[0].plot(time, states[:, 0], 'b', label='Posizione (p)');
    axs[0].set_ylabel('Posizione');
    axs[0].legend();
    axs[0].grid(True, ls=':')
    axs[1].plot(time, states[:, 1], 'g', label='Velocità (v)');
    axs[1].set_ylabel('Velocità');
    axs[1].legend();
    axs[1].grid(True, ls=':')
    axs[2].plot(time[:-1], actions[:, 0], 'r', drawstyle='steps-post', label='Controllo (a)');
    axs[2].set_ylabel('Controllo');
    axs[2].set_xlabel('Tempo (s)');
    axs[2].legend();
    axs[2].grid(True, ls=':')
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.savefig(f"plots/{file_prefix}_trajectory.png");
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(states[:, 0], states[:, 1], 'b-o', markersize=3, label='Traiettoria')
    ax.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Inizio')
    ax.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='Fine')
    ax.set_title(f'Ritratto di Fase: {title}');
    ax.set_xlabel('Posizione (p)');
    ax.set_ylabel('Velocità (v)');
    ax.legend();
    ax.grid(True)
    plt.tight_layout();
    plt.savefig(f"plots/{file_prefix}_phase_portrait.png");
    plt.close(fig)
    print(f"Grafici salvati in 'plots/{file_prefix}_*.png'")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MPC_HORIZON, NUM_ENVS, EPISODE_LEN, TRAINING_STEPS, HISTORY_LEN = 5, 32, 50, 100, 25
    CHECKPOINT_DIR, RESUME_TRAINING = "./di_checkpoints", "latest"

    print(f"Inizio addestramento su dispositivo: {DEVICE}")
    print("=" * 50)

    checkpoint_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)


    class ActorMPCWithJacobian(ActorMPC):
        def __init__(self, nx, nu, horizon, dt, f_dyn, f_dyn_jac, device):
            super().__init__(nx, nu, horizon, dt, f_dyn, device)
            cost = self._create_placeholder_cost()
            self.mpc = DifferentiableMPCController(
                f_dyn=f_dyn, total_time=horizon * dt, step_size=dt, horizon=horizon, cost_module=cost,
                grad_method=GradMethod.ANALYTIC, f_dyn_jac=f_dyn_jac, device=device
            )


    actor = ActorMPCWithJacobian(
        nx=NX, nu=NU, horizon=MPC_HORIZON, dt=DT,
        f_dyn=f_dyn_batched, f_dyn_jac=f_dyn_jac_batched, device=DEVICE
    ).to(DEVICE, dtype=torch.float32)

    critic = CriticTransformer(
        state_dim=NX, action_dim=NU, history_len=HISTORY_LEN, pred_horizon=MPC_HORIZON,
        hidden_size=256, num_layers=4, num_heads=4
    ).to(DEVICE, dtype=torch.float32)

    evaluate_and_plot(actor, env_factory(), DEVICE, "before_training", "Policy Non Addestrata")

    print("\nInizio ciclo di training...\n")

    train(
        env_fn=env_factory, actor=actor, critic=critic, reward_fn=reward_fn_batched,
        steps=TRAINING_STEPS, num_envs=NUM_ENVS, episode_len=EPISODE_LEN,
        mpc_horizon=MPC_HORIZON, gamma=0.99, lam=0.97,
        mpve_gamma=0.99, mpve_weight=0.1,
        checkpoint_manager=checkpoint_manager, resume_from=RESUME_TRAINING,
        use_amp=True if DEVICE == "cuda" else False, seed=42
    )

    print("\n...Training completato.\n")

    print("Caricamento del modello migliore per la valutazione finale...")
    checkpoint_manager.load(DEVICE, resume_from="best", actor=actor)
    evaluate_and_plot(actor, env_factory(), DEVICE, "after_training", "Policy Addestrata (Best)")

    print("=" * 50)
    print("Processo completato.")