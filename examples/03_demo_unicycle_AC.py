# File: 03_demo_unicycle_AC.py

import torch
import numpy as np
import gym
from gym import spaces
import os
import math
from typing import Callable, List

# Importa i moduli della libreria ACMPC
from ACMPC import ActorMPC, CheckpointManager, CriticTransformer, train



# -----------------------------------------------------------------------------
# 1. DEFINIZIONE DELL'AMBIENTE GYM: UNICYCLE
# -----------------------------------------------------------------------------
# File: 03_demo_unicycle_AC.py (classe UnicycleEnv aggiornata)

class UnicycleEnv(gym.Env):
    """
    Ambiente Unicycle con reward DENSA per incoraggiare il progresso.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, trajectory: np.ndarray, dt: float = 0.1, episode_len: int = 500):
        super().__init__()
        self.dt = dt
        self.trajectory = trajectory
        self.max_steps = episode_len  # Usa la lunghezza dell'episodio passata

        # Stato e azione
        self.state = np.zeros(3, dtype=np.float32)  # x, y, theta
        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0]),  # v, omega
            high=np.array([5.0, 2.0]),
            dtype=np.float32
        )

        # L'osservazione include lo stato e i prossimi 2 waypoint
        self.observation_dim = 3 + 4
        obs_high = np.full(self.observation_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.current_step = 0
        self.waypoint_index = 0
        self.goal_radius = 0.5
        self.fail_radius = 5.0
        self.previous_dist_to_wp = 0.0  # Per calcolare la reward densa

    def _get_obs(self):
        wp1_idx = self.waypoint_index
        wp2_idx = min(wp1_idx + 1, len(self.trajectory) - 1)
        wp1 = self.trajectory[wp1_idx]
        wp2 = self.trajectory[wp2_idx]
        return np.concatenate([self.state, wp1, wp2]).astype(np.float32)

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        self.waypoint_index = 0

        # Inizializza la distanza per la reward densa
        current_wp = self.trajectory[self.waypoint_index]
        self.previous_dist_to_wp = np.linalg.norm(self.state[:2] - current_wp)

        info = {}
        return self._get_obs(), info


    def step(self, action: np.ndarray):
        v, omega = action[0], action[1]
        x, y, theta = self.state

        x += v * math.cos(theta) * self.dt
        y += v * math.sin(theta) * self.dt
        # Normalizza l'angolo tra [-pi, pi] per stabilità numerica
        theta = (theta + omega * self.dt + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([x, y, theta], dtype=np.float32)
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_steps

        # --- Logica della Reward Migliorata ---
        current_wp = self.trajectory[self.waypoint_index]
        dist_to_wp = np.linalg.norm(self.state[:2] - current_wp)

        # 1. Reward di progresso (componente densa, con peso maggiore)
        reward_progress = 1.5 * (self.previous_dist_to_wp - dist_to_wp)
        self.previous_dist_to_wp = dist_to_wp

        # 2. NUOVO: Reward di Orientamento
        angle_to_wp = math.atan2(current_wp[1] - y, current_wp[0] - x)
        angle_diff = abs(theta - angle_to_wp)
        # Bonus basato sul coseno della differenza di angolo (1 se allineato, -1 se opposto)
        reward_orientation = 0.2 * (math.cos(angle_diff))

        # 3. Penalità per azioni (leggermente ridotta)
        action_penalty = 0.005 * np.sum(np.square(action))

        # 4. NUOVO: Bonus per velocità positiva
        reward_velocity = 0.05 * v if v > 0.1 else -0.01

        # Reward totale
        reward = reward_progress + reward_orientation - action_penalty + reward_velocity

        # 5. Bonus/Malus per eventi sparsi (più impattanti)
        if dist_to_wp < self.goal_radius:
            reward += 20.0  # Bonus per aver raggiunto il waypoint
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.trajectory):
                terminated = True
                reward += 100.0  # Bonus finale per aver completato il percorso
            else:
                next_wp = self.trajectory[self.waypoint_index]
                self.previous_dist_to_wp = np.linalg.norm(self.state[:2] - next_wp)

        if dist_to_wp > self.fail_radius:
            reward -= 20.0  # Penalità per fallimento
            terminated = True

        info = {'waypoint_idx': self.waypoint_index, 'dist_to_wp': dist_to_wp}
        return self._get_obs(), float(reward), terminated, truncated, info

# Registra l'ambiente
gym.register(id='UnicycleTrack-v0', entry_point=__name__ + ':UnicycleEnv')


# -----------------------------------------------------------------------------
# 2. FUNZIONI PER L'MPC (DINAMICA, JACOBIANA E RICOMPENSA) DINAMICA PRESA ONLINE
# -----------------------------------------------------------------------------

def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """ Dinamica Unicycle (stato [x,y,theta], azione [v, omega]) """
    v, omega = u[..., 0], u[..., 1]
    theta = x[..., 2]
    new_x = x[..., 0] + v * torch.cos(theta) * dt
    new_y = x[..., 1] + v * torch.sin(theta) * dt
    new_theta = theta + omega * dt
    return torch.stack([new_x, new_y, new_theta], dim=-1)


def f_dyn_jac_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """ Jacobiana analitica della dinamica del Unicycle """
    B, dtype, device = x.shape[0], x.dtype, x.device
    v = u[..., 0]
    theta = x[..., 2]

    # Matrice A (df/dx)
    A = torch.zeros(B, 3, 3, dtype=dtype, device=device)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 2, 2] = 1.0
    A[:, 0, 2] = -v * torch.sin(theta) * dt
    A[:, 1, 2] = v * torch.cos(theta) * dt

    # Matrice B (df/du)
    B_mat = torch.zeros(B, 3, 2, dtype=dtype, device=device)
    B_mat[:, 0, 0] = torch.cos(theta) * dt
    B_mat[:, 1, 0] = torch.sin(theta) * dt
    B_mat[:, 2, 1] = dt

    return A, B_mat


def unicycle_reward_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Funzione di ricompensa per MPVE. Stima il progresso verso il waypoint.
    `state` qui è l'osservazione completa.
    """
    # Calcola la distanza dal prossimo waypoint (contenuto nell'osservazione)
    pos = state[..., :2]
    target_wp = state[..., 3:5]
    dist_to_wp = torch.linalg.norm(pos - target_wp, dim=-1)

    # Ricompensa basata sulla riduzione della distanza (incentivo denso)
    # Questa è una semplificazione della ricompensa sparsa per rendere il training più stabile
    # E' un proxy "denso" per la ricompensa "sparsa" dell'ambiente.
    # Un valore negativo alto significa che si sta avvicinando.
    return -dist_to_wp


# -----------------------------------------------------------------------------
# 3. SCRIPT DI TRAINING PRINCIPALE
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Parametri ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42 # a casual number
    CHECKPOINT_DIR = "./checkpoints_unicycle"
    DT = 0.1
    NX = 3  # Dimensione stato dinamica
    NU = 2  # Dimensione azione
    OBS_DIM = 7  # Dimensione osservazione (stato + 2 waypoint)
    MPC_HORIZON = 3
    EPISODE_LEN = 750
    TOTAL_TRAINING_STEPS = 30
    NUM_ENVS = 24
    PPO_EPOCHS = 3

    print(f"Utilizzando il dispositivo: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Definisce una traiettoria a forma di '8'
   # t = np.linspace(-np.pi, np.pi, 20)
    #trajectory = np.stack([5 * np.sin(t), 2.5 * np.sin(2 * t)], axis=1)
    trajectory = np.array([[5.0, 5.0]])
    # Funzioni per ambiente e dinamica
    env_fn = lambda: UnicycleEnv(trajectory=trajectory, dt=DT)
    bound_f_dyn = lambda x, u, _: f_dyn_torch(x, u, dt=DT)
    bound_f_dyn_jac = lambda x, u, _: f_dyn_jac_torch(x, u, dt=DT)

    # --- Inizializzazione di Attore e Critico ---
    actor = ActorMPC(
        nx=NX,
        nu=NU,
        observation_dim=OBS_DIM,  # Passiamo la dimensione dell'osservazione
        horizon=MPC_HORIZON,
        dt=DT,
        f_dyn=bound_f_dyn,
        f_dyn_jac=bound_f_dyn_jac,  # Forniamo la Jacobiana analitica
        device=DEVICE,
        grad_method='analytic'
    ).to(DEVICE)

    critic = CriticTransformer(
        state_dim=OBS_DIM,  # Il critico lavora sull'osservazione completa
        action_dim=NU,
        history_len=5,
        pred_horizon=MPC_HORIZON,
        hidden_size=256,
        num_layers=4,
        num_heads=4
    ).to(DEVICE)

    checkpoint_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)

    print(">>> Inizio del training per ACMPC su UnicycleTrack-v0 <<<")

    # --- Avvio del Training ---
    training_rewards = train(
        env_fn=env_fn,
        actor=actor,
        critic=critic,
        reward_fn=unicycle_reward_fn,  # Usa la nostra reward basata sulla distanza
        steps=TOTAL_TRAINING_STEPS,
        num_envs=NUM_ENVS,
        episode_len=EPISODE_LEN,
        mpc_horizon=MPC_HORIZON,
        ppo_epochs=PPO_EPOCHS,
        clip_param=0.2,
        entropy_coeff=0.01,
        mpve_coeff=0.1,
        checkpoint_manager=checkpoint_manager,
        resume_from="best",  # Carica il modello migliore se esiste
        use_amp=(DEVICE == "cuda"),
        seed=SEED
    )

    print(">>> Training completato! <<<")

    # --- Visualizzazione dei Risultati ---
    try:
        import matplotlib.pyplot as plt

        print("Visualizzazione della traiettoria finale...")

        # Carica il modello migliore
        checkpoint_manager.load(device=DEVICE, resume_from="best", actor=actor, critic=critic)

        env = UnicycleEnv(trajectory, DT)
        obs, _ = env.reset()

        history_x, history_y = [], []
        done = False

        while not done:
            history_x.append(env.state[0])
            history_y.append(env.state[1])

            obs_tensor = torch.from_numpy(obs).to(DEVICE).float()
            with torch.no_grad():
                action_tensor, _, _, _ = actor(obs_tensor, deterministic=True)

            action_np = action_tensor.cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

        plt.figure(figsize=(10, 6))
        # Plotta la traiettoria target
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'go--', label='Traiettoria Target', linewidth=2)
        # Plotta la traiettoria eseguita dal robot
        plt.plot(history_x, history_y, 'b-', label='Traiettoria Eseguita', linewidth=2)
        # Plotta i waypoint
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=8, label='Waypoints')
        plt.title("Tracking di Traiettoria con AC-MPC")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig("trajectory_unicycle.png")
        print("Grafico della traiettoria salvato in 'trajectory_unicycle.png'")

    except ImportError:
        print("Matplotlib non trovato. Salto la visualizzazione dei risultati.")