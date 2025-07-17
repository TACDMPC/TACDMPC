import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
from gym.spaces import Box
import os
from contextlib import redirect_stdout
import time
from typing import Optional

# =============================================================================
# PASSO 1: INTEGRAZIONE DELLE CLASSI E FUNZIONI NECESSARIE
# Ho integrato qui le definizioni delle classi e delle funzioni di training
# per applicare le correzioni necessarie in un unico script.
# =============================================================================

# --- Import dalle librerie dei tuoi file ---
# Assicurati che i file siano nella stessa cartella o nel PYTHONPATH
from ACMPC import ActorMPC
from ACMPC import CriticTransformer

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
from gym.spaces import Box
import os
from contextlib import redirect_stdout
import time
from typing import Optional

# =============================================================================
# PASSO 1: INTEGRAZIONE DELLE CLASSI E FUNZIONI NECESSARIE
# Ho integrato qui le definizioni delle classi e delle funzioni di training
# per applicare le correzioni necessarie in un unico script.
# =============================================================================

# --- Import dalle librerie dei tuoi file ---
# Assicurati che i file siano nella stessa cartella o nel PYTHONPATH

# --- Classi e funzioni del problema (Doppio Integratore) ---
def f_double_integrator_discrete(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Dinamica di un doppio integratore.
    Stato: [posizione, velocità]
    Controllo: [accelerazione]
    """
    pos, vel = x.unbind(dim=-1)
    acc = u.squeeze(-1)

    new_pos = pos + vel * dt
    new_vel = vel + acc * dt

    return torch.stack([new_pos, new_vel], dim=-1)


def reward_fn_double_integrator(states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                                target_pos: float) -> torch.Tensor:
    """
    Funzione di ricompensa per raggiungere una posizione target.
    """
    target_state = torch.tensor([target_pos, 0.0], device=states.device, dtype=states.dtype)

    pos_weight = 5.0
    vel_weight = 0.1
    # <<< MODIFICA: Ridotta la penalità sul controllo per incoraggiare azioni più forti >>>
    control_weight = 0.001

    state_error = pos_weight * (states[..., 0] - target_state[0]) ** 2 + vel_weight * (
                states[..., 1] - target_state[1]) ** 2
    control_cost = control_weight * torch.sum(actions ** 2, dim=-1)

    reward = -(state_error + control_cost)
    return reward


# --- Ambiente Gym Personalizzato per il Doppio Integratore ---
class DoubleIntegratorEnv(gym.Env):
    """Ambiente Gym per un sistema a doppio integratore."""

    def __init__(self, start_pos=-1.0, target_pos=1.0, dt=0.05, max_accel=2.0):
        super().__init__()
        self.dt = dt
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.max_accel = max_accel

        # <<< MODIFICA: Lo spazio delle azioni ora riflette la massima accelerazione >>>
        self.action_space = Box(low=-self.max_accel, high=self.max_accel, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.state = np.zeros(2, dtype=np.float32)
        self.max_steps = 200
        self.current_step = 0

    def reset(self):
        self.state = np.array([self.start_pos, 0.0], dtype=np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        # L'azione ora è direttamente l'accelerazione, che viene clippata ai limiti fisici
        acc = np.clip(action, self.action_space.low, self.action_space.high)[0]

        pos, vel = self.state
        new_pos = pos + vel * self.dt
        new_vel = vel + acc * self.dt
        self.state = np.array([new_pos, new_vel], dtype=np.float32)

        self.current_step += 1

        reward = -((self.state[0] - self.target_pos) ** 2)

        terminated = bool(abs(self.state[0]) > 5.0)
        truncated = bool(self.current_step >= self.max_steps)

        return self.state, reward, terminated, truncated, {}

    def close(self):
        pass


# --- Wrapper per l'ambiente Gym ---
class GymTensorWrapper:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.sim_horizon = self.env.max_steps

    def reset(self):
        reset_output, _ = self.env.reset()
        return torch.from_numpy(reset_output).to(device=self.device, dtype=torch.get_default_dtype())

    def step(self, action: torch.Tensor):
        action_np = action.cpu().detach().numpy()
        obs_np, reward, terminated, truncated, info = self.env.step(action_np)
        done = terminated or truncated
        obs_tensor = torch.from_numpy(obs_np).to(device=self.device, dtype=torch.get_default_dtype())
        return obs_tensor, reward, done

    def close(self):
        self.env.close()


# --- Funzioni di training (con la correzione applicata) ---
def rollout(env, actor):
    states, actions, rewards, log_probs, dists = [], [], [], [], []
    predicted_states_list, predicted_actions_list = [], []
    state = env.reset()
    device = state.device
    for _ in range(env.sim_horizon):
        action, log_prob, pred_states, pred_actions, dist = actor(state)
        new_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(torch.tensor([reward], device=device, dtype=torch.get_default_dtype()))
        log_probs.append(log_prob)
        dists.append(dist)
        predicted_states_list.append(pred_states)
        predicted_actions_list.append(pred_actions)
        state = new_state
        if done: break
    return (torch.stack(states), torch.stack(actions), torch.stack(rewards),
            torch.stack(log_probs), torch.stack(predicted_states_list),
            torch.stack(predicted_actions_list), dists)


def compute_gae_and_returns(rewards, values, gamma, lam):
    T = rewards.shape[0]
    values_extended = torch.cat([values, torch.zeros(1, device=values.device, dtype=values.dtype)], dim=0)
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values
    return advantages, returns


def train_ac_mpc(env, actor, critic, reward_fn, steps, mpc_horizon, use_amp, profile, mpve_weight, entropy_coeff):
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    device = next(actor.parameters()).device
    dtype = next(actor.parameters()).dtype
    token_dim = actor.nx + actor.nu
    history_len = critic.history_len
    training_rewards = []

    for step_idx in range(steps):
        states, actions, _, log_probs, pred_states, pred_actions, dists = rollout(env, actor)
        real_rewards = reward_fn(states[:-1], actions[:-1], states[1:])

        tokens = torch.cat([states.detach(), actions.detach()], dim=-1)
        padded_tokens = torch.cat([torch.zeros(history_len - 1, token_dim, device=device, dtype=dtype), tokens], dim=0)

        history_batch = padded_tokens.unfold(dimension=0, size=history_len, step=1).permute(0, 2, 1)

        values = critic(history_tokens=history_batch[:-1])
        advantages, returns = compute_gae_and_returns(real_rewards, values.detach(), gamma=0.99, lam=0.95)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        predicted_rewards = reward_fn(pred_states[:, :-1, :], pred_actions, pred_states[:, 1:, :])
        predicted_tokens = torch.cat([pred_states[:, :-1, :].detach(), pred_actions.detach()], dim=-1)

        with torch.no_grad():
            final_pred_input_seq = torch.cat([history_batch, predicted_tokens], dim=1)
            v_final_pred = critic(history_tokens=final_pred_input_seq).squeeze()

        mpve_targets = torch.zeros_like(predicted_rewards)
        for i in range(predicted_rewards.shape[0]):
            next_val = v_final_pred[i]
            for t in reversed(range(mpc_horizon)):
                target = predicted_rewards[i, t] + 0.99 * next_val
                mpve_targets[i, t] = target
                next_val = target

        predicted_values_list = []
        for t in range(len(history_batch)):
            history_at_t = history_batch[t].unsqueeze(0)
            for k in range(mpc_horizon):
                current_pred_seq = predicted_tokens[t, :k + 1, :].unsqueeze(0)
                input_for_critic = torch.cat([history_at_t, current_pred_seq], dim=1)
                value = critic(input_for_critic)
                predicted_values_list.append(value)

        if not predicted_values_list:
            loss_mpve = torch.tensor(0.0, device=device)
        else:
            predicted_values = torch.cat(predicted_values_list)
            loss_mpve = torch.nn.functional.mse_loss(predicted_values, mpve_targets.view(-1))

        entropy = torch.stack([d.entropy() for d in dists[:-1]]).mean()
        actor_loss = -(log_probs[:-1] * advantages).mean() - entropy_coeff * entropy
        critic_loss = nn.functional.mse_loss(values, returns.squeeze(-1)) + mpve_weight * loss_mpve

        total_loss = actor_loss + critic_loss

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)

        actor_opt.step()
        critic_opt.step()

        episode_reward = real_rewards.sum().item()
        training_rewards.append(episode_reward)
        if step_idx % 20 == 0:
            print(
                f"Step {step_idx}: Reward: {episode_reward:.2f}, ActorL: {actor_loss:.3f}, CriticL: {critic_loss:.3f}, Entropy: {entropy:.3f}")

    return training_rewards


def plot_results(total_rewards, final_states, final_actions, dt, target_pos):
    plt.figure(figsize=(12, 5))
    plt.plot(total_rewards)
    plt.title("Ricompensa per Episodio durante il Training")
    plt.xlabel("Passo di Training");
    plt.ylabel("Ricompensa Totale Episodio")
    plt.grid(True);
    plt.tight_layout();
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Traiettoria Finale della Policy Addestrata")
    time_axis = np.arange(final_states.shape[0]) * dt
    axs[0].plot(time_axis, final_states[:, 0].numpy(), label="Posizione [m]")
    axs[0].plot(time_axis, final_states[:, 1].numpy(), label="Velocità [m/s]")
    axs[0].axhline(target_pos, linestyle=":", color="k", label="Target Pos.")
    axs[0].set_ylabel("Stato");
    axs[0].grid(True);
    axs[0].legend()
    axs[1].plot(time_axis[:-1], final_actions[:, 0].numpy(), label="Accelerazione [m/s^2]", color='orange')
    axs[1].set_ylabel("Controllo");
    axs[1].grid(True);
    axs[1].legend()
    plt.xlabel(f"Tempo [s]");
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.show()


# =============================================================================
# PASSO 3: ESECUZIONE DELL'ESEMPIO
# =============================================================================
def main():
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione AC-MPC per Doppio Integratore su: {DEVICE}")

    DT, MPC_HORIZON, TRAIN_STEPS = 0.05, 20, 300
    NX, NU = 2, 1
    START_POS, TARGET_POS = -1.0, 1.0
    MAX_ACCEL = 2.0

    dyn_fn = lambda x, u, dt: f_double_integrator_discrete(x, u, dt)
    reward_fn = lambda s, a, ns: reward_fn_double_integrator(s, a, ns, TARGET_POS)

    base_env = DoubleIntegratorEnv(start_pos=START_POS, target_pos=TARGET_POS, dt=DT, max_accel=MAX_ACCEL)
    env = GymTensorWrapper(base_env, DEVICE)

    class PatchedActorMPC(ActorMPC):
        def __init__(self, u_min=None, u_max=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.U_prev = None
            # Imposta i limiti di controllo nell'MPC interno
            self.mpc.u_min = u_min
            self.mpc.u_max = u_max
            if hasattr(self.mpc, 'linearize_dynamics'):
                self.mpc.debug = False

        def forward(self, x, deterministic: bool = False):
            if x.ndim == 1: x = x.unsqueeze(0)
            batch_size = x.shape[0]
            assert batch_size == 1, "Questo esempio supporta solo batch size 1"

            C_batch, c_batch, C_final_batch, c_final_batch = self._generate_and_scale_costs(x)
            self.mpc.cost_module.C = C_batch.squeeze(0)
            self.mpc.cost_module.c = c_batch.squeeze(0)
            self.mpc.cost_module.C_final = C_final_batch.squeeze(0)
            self.mpc.cost_module.c_final = c_final_batch.squeeze(0)

            if self.U_prev is None:
                U_init = torch.zeros(batch_size, self.horizon, self.nu, device=self.device, dtype=self.dtype)
            else:
                U_init = torch.roll(self.U_prev, shifts=-1, dims=1)
                U_init[:, -1] = 0.0

            u_mpc, _ = self.mpc.solve_step(x, U_init)

            self.U_prev = self.mpc.U_last.detach()

            std = self.log_std.exp()
            dist = torch.distributions.Normal(u_mpc, std)
            action = u_mpc if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            predicted_states = self.mpc.X_last
            predicted_actions = self.mpc.U_last
            return (action.squeeze(0), log_prob.squeeze(0), predicted_states.squeeze(0), predicted_actions.squeeze(0),
                    dist)

    actor = PatchedActorMPC(
        nx=NX, nu=NU, horizon=MPC_HORIZON, dt=DT, f_dyn=dyn_fn, device=str(DEVICE),
        u_min=torch.tensor([-MAX_ACCEL], device=DEVICE, dtype=torch.get_default_dtype()),
        u_max=torch.tensor([MAX_ACCEL], device=DEVICE, dtype=torch.get_default_dtype())
    )

    critic = CriticTransformer(state_dim=NX, action_dim=NU, history_len=5, pred_horizon=MPC_HORIZON, hidden_size=32,
                               num_layers=2, num_heads=2).to(DEVICE)

    print("Avvio del training con l'implementazione robusta di AC-MPC...")
    training_rewards = train_ac_mpc(
        env=env, actor=actor, critic=critic, reward_fn=reward_fn,
        steps=TRAIN_STEPS, mpc_horizon=MPC_HORIZON, use_amp=False, profile=False,
        mpve_weight=0.1, entropy_coeff=0.01
    )

    print("\nEsecuzione di un rollout finale con la policy addestrata...")
    state = env.reset()
    final_states, final_actions = [state.cpu()], []
    for _ in range(env.sim_horizon):
        action, _, _, _, _ = actor(state, deterministic=True)
        final_actions.append(action.cpu().detach())
        state, _, done = env.step(action)
        final_states.append(state.cpu())
        if done: break
    final_states = torch.stack(final_states)
    final_actions = torch.stack(final_actions)

    print("Generazione dei grafici dei risultati...")
    plot_results(training_rewards, final_states, final_actions, DT, TARGET_POS)

    env.close()


if __name__ == "__main__":
    main()
