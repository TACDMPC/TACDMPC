

import torch
import numpy as np
import gym
from gym import spaces
import os

from ACMPC import ActorMPC, CheckpointManager, CriticTransformer, train


class ContinuousCartPoleEnv(gym.Env):
    """
    Un wrapper per l'ambiente CartPole-v1 che accetta azioni continue.
    L'azione continua rappresenta la forza applicata al carrello.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, dt: float = 0.02):
        self.env = gym.make('CartPole-v1')
        self.dt = dt
        self.max_steps = self.env._max_episode_steps

        # Definisce lo spazio delle azioni come una forza continua
        # La forza standard in CartPole è 10.0 N
        self.force_mag = 10.0
        self.action_space = spaces.Box(
            low=-self.force_mag,
            high=self.force_mag,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = self.env.observation_space
        self.state = None
        self.current_step = 0

    def step(self, action: np.ndarray):
        # Mappa l'azione continua (forza) all'azione discreta richiesta da CartPole-v1
        force = np.clip(action, self.action_space.low, self.action_space.high)[0]
        discrete_action = 1 if force > 0 else 0

        # Esegue un passo nell'ambiente originale, accettando i 5 valori restituiti
        self.state, reward, terminated, truncated, info = self.env.step(discrete_action)
        self.current_step += 1

        # Il file parallel_env.py si aspetta la nuova firma a 5 valori,
        # quindi li passiamo direttamente.
        return self.state, float(reward), terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            super().reset(seed=seed)

        self.state = self.env.reset(seed=seed)
        self.current_step = 0

        # La nuova API di gym.reset() può restituire una tupla (obs, info)
        if isinstance(self.state, tuple):
            info = self.state[1]
            self.state = self.state[0]
        else:
            info = {}

        return self.state, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

gym.register(
    id='ContinuousCartPole-v0',
    entry_point=__name__ + ':ContinuousCartPoleEnv',
)

def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Stato x: [pos, vel, angolo_p, vel_angolare_p]
    Azione u: [forza]
    """
    # Parametri fisici di CartPole-v1
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    cart_pos = x[..., 0]
    cart_vel = x[..., 1]
    pole_angle = x[..., 2]
    pole_vel = x[..., 3]
    force = u.squeeze(-1)
    costheta = torch.cos(pole_angle)
    sintheta = torch.sin(pole_angle)

    temp = (force + polemass_length * pole_vel.pow(2) * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / \
               (length * (4.0 / 3.0 - masspole * costheta.pow(2) / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    new_cart_pos = cart_pos + cart_vel * dt
    new_cart_vel = cart_vel + xacc * dt
    new_pole_angle = pole_angle + pole_vel * dt
    new_pole_vel = pole_vel + thetaacc * dt

    return torch.stack([new_cart_pos, new_cart_vel, new_pole_angle, new_pole_vel], dim=-1)


def cartpole_reward_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Funzione di ricompensa quadratica per stabilizzare il CartPole.
    """
    # Pesi per i diversi termini di costo
    W_pos = 1.0
    W_angle = 5.0
    W_vel = 0.1
    W_action = 0.01
    target_state = torch.zeros_like(state)
    cart_pos = state[..., 0]
    cart_vel = state[..., 1]
    pole_angle = state[..., 2]
    pole_vel = state[..., 3]
    pos_cost = W_pos * (cart_pos - target_state[..., 0]).pow(2)
    angle_cost = W_angle * (pole_angle - target_state[..., 2]).pow(2)
    vel_cost = W_vel * (cart_vel.pow(2) + pole_vel.pow(2))
    action_cost = W_action * action.squeeze(-1).pow(2)

    return -(pos_cost + angle_cost + vel_cost + action_cost)

if __name__ == '__main__':
    # --- Parametri ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    CHECKPOINT_DIR = "./checkpoints_cartpole"
    DT = 0.02  # Timestep dell'ambiente CartPole
    NX = 4  # Dimensione dello stato (pos, vel, angle, angle_vel)
    NU = 1  # Dimensione dell'azione (forza)
    MPC_HORIZON = 20
    EPISODE_LEN = 200
    TOTAL_TRAINING_STEPS = 2
    NUM_ENVS = 32
    PPO_EPOCHS = 2
    CLIP_PARAM = 0.2
    ENTROPY_COEFF = 0.01
    MPVE_COEFF = 0.1

    print(f"Utilizzando il dispositivo: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    env_fn = lambda: ContinuousCartPoleEnv(dt=DT)
    bound_f_dyn = lambda x, u, _: f_dyn_torch(x, u, dt=DT)
    actor = ActorMPC(
        nx=NX,
        nu=NU,
        horizon=MPC_HORIZON,
        dt=DT,
        f_dyn=bound_f_dyn,
        f_dyn_jac=None,
        device=DEVICE,
        grad_method='auto_diff'
    ).to(DEVICE)

    critic = CriticTransformer(
        state_dim=NX,
        action_dim=NU,
        history_len=1,  # non appesantiamo i cartpole con il transformer
        pred_horizon=MPC_HORIZON,
        hidden_size=128,
        num_layers=3,
        num_heads=4
    ).to(DEVICE)

    checkpoint_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)

    print(">>> Inizio del training per ACMPC su ContinuousCartPole-v0 <<<")
    training_rewards = train(
        env_fn=env_fn,
        actor=actor,
        critic=critic,
        reward_fn=cartpole_reward_fn,
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
        use_amp=(DEVICE == "cuda"),
        seed=SEED
    )

    print(">>> Training completato! <<<")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(training_rewards)
        plt.title("Ricompensa Media per Step di Training (CartPole)")
        plt.xlabel("Step di Training")
        plt.ylabel("Ricompensa Media")
        plt.grid(True)
        plt.savefig("training_rewards_cartpole.png")
        print("Grafico delle ricompense salvato in 'training_rewards_cartpole.png'")
    except ImportError:
        print("Matplotlib non trovato. Salto la visualizzazione dei risultati.")