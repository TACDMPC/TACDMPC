import torch
import numpy as np
import gym
from gym import spaces
import os
from typing import Callable
from ACMPC import ActorMPC
from ACMPC import CriticTransformer
from ACMPC import train
from ACMPC import CheckpointManager

class DoubleIntegratorEnv(gym.Env):
    """
    Ambiente per un doppio integratore.
    Lo stato è [posizione, velocità] e l'azione è [forza/accelerazione].
    L'obiettivo è stabilizzare il sistema all'origine [0, 0].
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, dt: float = 0.05):
        super(DoubleIntegratorEnv, self).__init__()
        self.dt = dt
        self.max_steps = 250
        self.current_step = 0
        self.action_space = spaces.Box(low=np.array([-5.0]), high=np.array([5.0]), dtype=np.float32)
        high = np.array([10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.state = None

    def step(self, action):
        pos, vel = self.state
        force = np.clip(action, self.action_space.low, self.action_space.high)[0]
        new_vel = vel + force * self.dt
        new_pos = pos + new_vel * self.dt
        self.state = np.array([new_pos, new_vel], dtype=np.float32)
        self.current_step += 1
        pos_cost = 1.0 * (pos**2)
        vel_cost = 0.1 * (vel**2)
        action_cost = 0.01 * (action[0]**2)
        reward = - (pos_cost + vel_cost + action_cost)
        terminated = self.current_step >= self.max_steps
        truncated = False # In questo semplice caso, non c'è troncamento per altre ragioni
        info = {}
        return self.state, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-2.0, high=2.0, size=(2,)).astype(np.float32)
        self.current_step = 0
        info = {}
        return self.state, info

gym.register(
    id='DoubleIntegrator-v0',
    entry_point=__name__ + ':DoubleIntegratorEnv',
)

# -----------------------------------------------------------------------------
# 2. FUNZIONI PER L'MPC (DINAMICA, JACOBIANA E RICOMPENSA)
# -----------------------------------------------------------------------------
def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    pos, vel = x[..., 0:1], x[..., 1:2]
    new_vel = vel + u * dt
    new_pos = pos + new_vel * dt
    return torch.cat([new_pos, new_vel], dim=-1)

def f_dyn_jac_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.shape[0]
    device, dtype = x.device, x.dtype
    A = torch.tensor([[1, dt], [0, 1]], device=device, dtype=dtype)
    B = torch.tensor([[dt * dt], [dt]], device=device, dtype=dtype)
    return A.unsqueeze(0).expand(batch_size, -1, -1), B.unsqueeze(0).expand(batch_size, -1, -1)

def di_reward_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    pos_cost = 1.0 * state[..., 0]**2
    vel_cost = 0.1 * state[..., 1]**2
    action_cost = 0.01 * action.squeeze(-1)**2
    return -(pos_cost + vel_cost + action_cost)

# -----------------------------------------------------------------------------
# 3. SCRIPT DI TRAINING PRINCIPALE
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Parametri ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    CHECKPOINT_DIR = "./checkpoints_di"
    DT = 0.05
    NX = 2
    NU = 1
    MPC_HORIZON = 5
    EPISODE_LEN = 50
    TOTAL_TRAINING_STEPS = 250
    NUM_ENVS = 24
    PPO_EPOCHS = 3
    CLIP_PARAM = 0.2
    ENTROPY_COEFF = 0.01
    MPVE_COEFF = 0.1

    print(f"Utilizzando il dispositivo: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env_fn = lambda: gym.make('DoubleIntegrator-v0', dt=DT)

    bound_f_dyn = lambda x, u, _: f_dyn_torch(x, u, dt=DT)
    bound_f_dyn_jac = lambda x, u, _: f_dyn_jac_torch(x, u, dt=DT)

    actor = ActorMPC(
        nx=NX,
        nu=NU,
        horizon=MPC_HORIZON,
        dt=DT,
        f_dyn=bound_f_dyn,
        f_dyn_jac=bound_f_dyn_jac,
        device=DEVICE,
        grad_method='analytic'
    ).to(DEVICE)

    critic = CriticTransformer(
        state_dim=NX,
        action_dim=NU,
        history_len=10,
        pred_horizon=MPC_HORIZON,
        hidden_size=128,
        num_layers=3,
        num_heads=4
    ).to(DEVICE)

    checkpoint_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)

    print(">>> Inizio del training per ACMPC su DoubleIntegrator-v0 <<<")

    training_rewards = train(
        env_fn=env_fn,
        actor=actor,
        critic=critic,
        reward_fn=di_reward_fn,
        steps=TOTAL_TRAINING_STEPS,
        num_envs=NUM_ENVS,
        episode_len=EPISODE_LEN,
        mpc_horizon=MPC_HORIZON,
        ppo_epochs=PPO_EPOCHS,
        clip_param=CLIP_PARAM,
        entropy_coeff=ENTROPY_COEFF,
        mpve_coeff=MPVE_COEFF,
        checkpoint_manager=checkpoint_manager,
        resume_from="latest",
        use_amp=True if DEVICE == "cuda" else False,
        seed=SEED
    )

    print(">>> Training completato! <<<")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(training_rewards)
        plt.title("Ricompensa Media per Step di Training")
        plt.xlabel("Step di Training")
        plt.ylabel("Ricompensa Media")
        plt.grid(True)
        plt.savefig("training_rewards_di.png")
        print("Grafico delle ricompense salvato in 'training_rewards_di.png'")
    except ImportError:
        print("Matplotlib non trovato. Salto la visualizzazione dei risultati.")

