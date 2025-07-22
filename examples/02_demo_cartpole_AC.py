import torch
import numpy as np
import gym
from gym import spaces
import os
import time
from typing import Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

from ACMPC import ActorMPC
from ACMPC import CriticTransformer
from ACMPC import CheckpointManager
from ACMPC import train

@dataclass(frozen=True)
class CartPoleParams:
    """Contiene i parametri fisici del sistema Cart-Pole."""
    m_c: float
    m_p: float
    l: float
    g: float


def f_cartpole_torch(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    """Dinamica non lineare del Cart-Pole, usata sia per la simulazione che per l'MPC."""
    pos, vel, theta, omega = x.split(1, dim=-1)
    force = u

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass = p.m_c + p.m_p
    m_p_l = p.m_p * p.l

    temp = (force + m_p_l * omega.pow(2) * sin_t) / total_mass
    theta_dd_num = (p.g * sin_t - cos_t * temp)
    theta_dd_den = (p.l * (4.0 / 3.0 - p.m_p * cos_t.pow(2) / total_mass))
    theta_dd = theta_dd_num / theta_dd_den
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass

    x_dot = torch.cat([vel, vel_dd, omega, theta_dd], dim=-1)
    return x + x_dot * dt

def cartpole_reward_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Funzione di reward per il task di regolazione."""
    W_pos, W_angle, W_vel, W_action = 10.0, 100.0, 10.0, 0.01
    cart_pos, cart_vel, pole_angle, pole_vel = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    pos_cost = W_pos * cart_pos.pow(2)
    angle_cost = W_angle * pole_angle.pow(2)
    vel_cost = W_vel * (cart_vel.pow(2) + pole_vel.pow(2))
    action_cost = W_action * action.squeeze(-1).pow(2)
    return -(pos_cost + angle_cost + vel_cost + action_cost)


class CustomCartPoleEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, params: CartPoleParams, dt: float = 0.05):
        super().__init__()
        self.dt = dt
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360  # ~0.209 rad
        self.max_steps = 500
        self.force_mag = 30.0
        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)

        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.current_step = 0
        self._render_env = None  # Ambiente per il solo rendering, creato al bisogno

    def step(self, action: np.ndarray):
        state_tensor = torch.from_numpy(self.state).to(self.device, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.from_numpy(action).to(self.device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            next_state_tensor = f_cartpole_torch(state_tensor, action_tensor, self.dt, self.params)

        self.state = next_state_tensor.squeeze(0).cpu().numpy()
        self.current_step += 1

        x, _, theta, _ = self.state
        terminated = bool(x < -self.x_threshold or x > self.x_threshold or
                          theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)
        truncated = bool(self.current_step >= self.max_steps)

        reward = cartpole_reward_fn(next_state_tensor.squeeze(0), action_tensor.squeeze(0)).item()

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        # Stato iniziale campionato da una distribuzione uniforme
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.current_step = 0
        if self._render_env:
            self._render_env.close()
            self._render_env = None
        return self.state, {}

    def render(self, mode='human'):
        if self._render_env is None:
            self._render_env = gym.make('CartPole-v1', render_mode=mode)
        self._render_env.unwrapped.state = self.state
        return self._render_env.render()

    def close(self):
        if self._render_env is not None:
            self._render_env.close()
            self._render_env = None

def run_evaluation(actor: ActorMPC, env_fn, num_episodes: int, episode_len: int, device: str) -> Dict:
    actor.eval()
    all_Xs, all_Us = [], []
    print(f"Esecuzione valutazione per {num_episodes} episodi...")
    for _ in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset()

        dt = env.dt
        episode_states, episode_actions = [obs], []

        for t_step in range(episode_len):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_tensor, _, _, _ = actor(obs_tensor, deterministic=True)

            action = action_tensor.squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            episode_states.append(obs)
            episode_actions.append(action)
            if terminated or truncated:
                break

        all_Xs.append(np.array(episode_states))
        all_Us.append(np.array(episode_actions))
        env.close()

    max_len_xs = max(len(x) for x in all_Xs)
    padded_Xs = [np.pad(x, ((0, max_len_xs - len(x)), (0, 0)), 'constant') for x in all_Xs]
    Xs_tensor = torch.from_numpy(np.array(padded_Xs)).float()

    max_len_us = max(len(u) for u in all_Us)
    padded_Us = [
        np.pad(u, ((0, max_len_us - len(u)), (0, 0)), 'constant', constant_values=0) if len(u) > 0 else np.zeros(
            (max_len_us, 1)) for u in all_Us]
    Us_tensor = torch.from_numpy(np.array(padded_Us)).float()

    final_states = np.array([ep[-1, :] for ep in all_Xs])
    mean_final_errors = np.mean(np.abs(final_states), axis=0)
    var_final_errors = np.var(np.abs(final_states), axis=0)

    return {'Xs': Xs_tensor, 'Us': Us_tensor, 'dt': dt,
            'metrics': {'mean_final_errors': mean_final_errors, 'var_final_errors': var_final_errors}}


def _plot_results(results: Dict, method_name: str = 'AC-MPC'):

    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f"Prestazioni Medie del Modello: {method_name} (Cart-Pole)", fontsize=20, weight='bold')

    dt = results['dt']
    mean_xs, std_xs = results['Xs'].mean(dim=0), results['Xs'].std(dim=0)
    mean_us, std_us = results['Us'].mean(dim=0), results['Us'].std(dim=0)

    time_ax_xs = torch.arange(mean_xs.shape[0]) * dt
    time_ax_us = torch.arange(mean_us.shape[0]) * dt

    ax = axs[0, 0]
    ax.plot(time_ax_xs, mean_xs[:, 0].cpu(), 'b-', label='Media (Posizione)')
    ax.fill_between(time_ax_xs, (mean_xs[:, 0] - std_xs[:, 0]).cpu(), (mean_xs[:, 0] + std_xs[:, 0]).cpu(), color='b',
                    alpha=0.2)
    ax.plot(time_ax_xs, mean_xs[:, 2].cpu(), 'g-', label='Media (Angolo)')
    ax.fill_between(time_ax_xs, (mean_xs[:, 2] - std_xs[:, 2]).cpu(), (mean_xs[:, 2] + std_xs[:, 2]).cpu(), color='g',
                    alpha=0.2)
    ax.axhline(0, color='r', linestyle='--', label='Target')
    ax.set_title("Stati: Posizione e Angolo"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel("Valore [m, rad]"), ax.grid(
        True), ax.legend()

    ax = axs[0, 1]
    ax.plot(time_ax_xs, mean_xs[:, 1].cpu(), 'c-', label='Media (Velocità x)')
    ax.fill_between(time_ax_xs, (mean_xs[:, 1] - std_xs[:, 1]).cpu(), (mean_xs[:, 1] + std_xs[:, 1]).cpu(), color='c',
                    alpha=0.2)
    ax.plot(time_ax_xs, mean_xs[:, 3].cpu(), 'm-', label='Media (Velocità Ang.)')
    ax.fill_between(time_ax_xs, (mean_xs[:, 3] - std_xs[:, 3]).cpu(), (mean_xs[:, 3] + std_xs[:, 3]).cpu(), color='m',
                    alpha=0.2)
    ax.axhline(0, color='r', linestyle='--', label='Target'), ax.set_title(
        "Stati: Velocità Lineare e Angolare"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel(
        "Valore [m/s, rad/s]"), ax.grid(True), ax.legend()

    ax = axs[1, 0]
    ax.plot(time_ax_us, mean_us[:, 0].cpu(), 'k-', label='Media (Forza)')
    ax.fill_between(time_ax_us, (mean_us[:, 0] - std_us[:, 0]).cpu(), (mean_us[:, 0] + std_us[:, 0]).cpu(), color='k',
                    alpha=0.2)
    ax.set_title("Input di Controllo (Forza)"), ax.set_xlabel("Tempo [s]"), ax.set_ylabel("Forza [N]"), ax.grid(
        True), ax.legend()

    ax = axs[1, 1]
    metrics = results['metrics']
    bar_labels = ['Posizione', 'Velocità', 'Angolo', 'Vel. Ang.']
    mean_errors, std_errors = metrics['mean_final_errors'], np.sqrt(metrics['var_final_errors'])
    ax.bar(bar_labels, mean_errors, yerr=std_errors, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
           alpha=0.8)
    ax.set_title("Errore Assoluto Medio Finale (tra Agenti)"), ax.set_ylabel("Errore Assoluto Medio"), ax.grid(True,
                                                                                                               axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("evaluation_results_cartpole_acmpc.png")
    print("\nGrafico con medie e dev. standard salvato in 'evaluation_results_cartpole_acmpc.png'")
    plt.show()


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED, CHECKPOINT_DIR = 42, "./checkpoints_cartpole_acmpc"
    DT, NX, NU = 0.05, 4, 1
    MPC_HORIZON, EPISODE_LEN = 15, 150
    TOTAL_TRAINING_STEPS, NUM_ENVS, PPO_EPOCHS = 40, 24, 3
    CLIP_PARAM, ENTROPY_COEFF, MPVE_COEFF = 0.2, 0.005, 0.1

    print(f"Utilizzando il dispositivo: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    cartpole_params = CartPoleParams(m_c=1.0, m_p=0.1, l=0.5, g=9.8)
    print(f"Parametri fisici del sistema: {cartpole_params}")

    env_fn = lambda: CustomCartPoleEnv(params=cartpole_params, dt=DT)
    bound_f_dyn = lambda x, u, _: f_cartpole_torch(x, u, DT, cartpole_params)

    actor = ActorMPC(nx=NX, nu=NU, horizon=MPC_HORIZON, dt=DT, f_dyn=bound_f_dyn, device=DEVICE).to(DEVICE)
    critic = CriticTransformer(state_dim=NX, action_dim=NU, history_len=1, pred_horizon=MPC_HORIZON,
                               hidden_size=128, num_layers=3, num_heads=4).to(DEVICE)
    checkpoint_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR, max_to_keep=3)
    print("\n>>> Inizio del training per AC-MPC su ambiente custom <<<")
    training_rewards = train(
        env_fn=env_fn, actor=actor, critic=critic, reward_fn=cartpole_reward_fn,
        steps=TOTAL_TRAINING_STEPS, num_envs=NUM_ENVS, episode_len=EPISODE_LEN,
        mpc_horizon=MPC_HORIZON, ppo_epochs=PPO_EPOCHS, clip_param=CLIP_PARAM,
        entropy_coeff=ENTROPY_COEFF, mpve_coeff=MPVE_COEFF,
        checkpoint_manager=checkpoint_manager, resume_from="latest",
        use_amp=(DEVICE == "cuda"), seed=SEED
    )
    print(">>> Training completato! <<<")

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(training_rewards)
        plt.title("Ricompensa Media per Step di Training (AC-MPC CartPole)")
        plt.xlabel("Step di Training"), plt.ylabel("Ricompensa Media"), plt.grid(True)
        plt.savefig("training_rewards_cartpole_acmpc.png")
        print("Grafico delle ricompense salvato in 'training_rewards_cartpole_acmpc.png'")
    except Exception as e:
        print(f"Impossibile generare il grafico delle ricompense: {e}")

    print("\n>>> Inizio della valutazione del modello addestrato <<<")
    eval_results = run_evaluation(
        actor=actor, env_fn=env_fn, num_episodes=50,
        episode_len=EPISODE_LEN * 2, device=DEVICE
    )

    metrics = eval_results['metrics']
    state_labels = ["Posizione", "Velocità", "Angolo", "Vel. Angolare"]
    print("\n>>> Metriche di Valutazione (calcolate sugli stati finali) <<<")
    print("-" * 70)
    print("Errore Assoluto Medio Finale per Stato:")
    for i, label in enumerate(state_labels): print(f"  - {label:<20}: {metrics['mean_final_errors'][i]:.4f}")
    print("-" * 70)
    print("Varianza dell'Errore Assoluto Finale per Stato:")
    for i, label in enumerate(state_labels): print(f"  - {label:<20}: {metrics['var_final_errors'][i]:.4f}")
    print("-" * 70)

    _plot_results(eval_results, method_name='AC-MPC (con Modello Dinamico)')