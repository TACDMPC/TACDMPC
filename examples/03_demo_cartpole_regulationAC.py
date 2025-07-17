import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout
import time

# Assumendo che ACMPC, CriticTransformer siano in un file ACMPC.py
from ACMPC import ActorMPC
from ACMPC import CriticTransformer


# =============================================================================
# PASSO 2: DEFINIZIONE SPECIFICA DELL'ESEMPIO (CART-POLE)
# =============================================================================

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
        return cls(m_c=float(env.unwrapped.masscart), m_p=float(env.unwrapped.masspole), l=float(env.unwrapped.length),
                   g=float(env.unwrapped.gravity))


def f_cartpole(x: torch.Tensor, u: torch.Tensor, p: CartPoleParams) -> torch.Tensor:
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass = p.m_c + p.m_p
    m_p_l = p.m_p * p.l
    temp = (force + m_p_l * omega ** 2 * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0 / 3.0 - p.m_p * cos_t ** 2 / total_mass))
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass
    return torch.stack([vel, vel_dd, omega, theta_dd], dim=-1)


def f_cartpole_discrete(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    force_magnitude = 10.0
    u_continuous = (u * 2 - 1) * force_magnitude
    x_dot = f_cartpole(x, u_continuous, p)
    return x + x_dot * dt


def reward_fn_cartpole(states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
    target_state = torch.tensor([0.0, 0.0, 0.0, 0.0], device=states.device, dtype=states.dtype)
    state_weights = torch.tensor([3.0, 0.1, 5.0, 0.1], device=states.device, dtype=states.dtype)
    control_weight = 0.001
    state_error = torch.sum(state_weights * (states - target_state) ** 2, dim=-1)
    control_cost = control_weight * torch.sum(actions ** 2, dim=-1)
    reward = -(state_error + control_cost)
    return reward


class GymTensorWrapper:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.sim_horizon = self.env.spec.max_episode_steps

    def reset(self, initial_state_numpy=None):
        reset_output = self.env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        if initial_state_numpy is not None:
            self.env.state = initial_state_numpy
            obs = initial_state_numpy
        return torch.from_numpy(obs).to(device=self.device, dtype=torch.get_default_dtype())

    def step(self, action: torch.Tensor):
        discrete_action = 1 if action.item() > 0 else 0
        step_output = self.env.step(discrete_action)
        if len(step_output) == 5:
            obs_np, reward, terminated, truncated, _ = step_output
            done = terminated or truncated
        else:
            obs_np, reward, done, _ = step_output
        obs_tensor = torch.from_numpy(obs_np).to(device=self.device, dtype=torch.get_default_dtype())
        return obs_tensor, reward, done

    def close(self):
        self.env.close()


def rollout(env, actor, mpc_horizon):
    states, actions, rewards, log_probs = [], [], [], []
    predicted_states_list, predicted_actions_list = [], []
    state = env.reset()
    device = state.device
    for _ in range(env.sim_horizon):
        action, log_prob, pred_states, pred_actions = actor(state)
        new_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(torch.tensor([reward], device=device, dtype=torch.get_default_dtype()))
        log_probs.append(log_prob)
        predicted_states_list.append(pred_states)
        predicted_actions_list.append(pred_actions)
        state = new_state
        if done: break
    return (torch.stack(states), torch.stack(actions), torch.stack(rewards),
            torch.stack(log_probs), torch.stack(predicted_states_list), torch.stack(predicted_actions_list))


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


def train_ac_mpc(env, actor, critic, reward_fn, steps, mpc_horizon, mpve_weight):
    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    device = next(actor.parameters()).device
    dtype = next(actor.parameters()).dtype
    history_len = critic.history_len
    token_dim = critic.token_dim
    training_rewards = []
    print("\n--- Inizio Training AC-MPC ---")

    for step_idx in range(steps):
        # 1. Rollout to collect data
        states, actions, _, log_probs, pred_states, pred_actions = rollout(env, actor, mpc_horizon)

        # 2. Calculate rewards and prepare data
        real_rewards = reward_fn(states[:-1], actions[:-1], states[1:])
        tokens = torch.cat([states.detach(), actions.detach()], dim=-1)
        padded_tokens = torch.cat([torch.zeros(history_len - 1, token_dim, device=device, dtype=dtype), tokens], dim=0)

        # FIX: Added .permute(0, 2, 1) to correct the tensor shape for the critic.
        history_batch = padded_tokens.unfold(dimension=0, size=history_len, step=1).permute(0, 2, 1)

        # This call now works because history_batch has the correct shape.
        values = critic(history_tokens=history_batch[:-1], predicted_tokens=None)

        # 3. Compute GAE and Returns
        advantages, returns = compute_gae_and_returns(real_rewards, values.detach(), gamma=0.99, lam=0.95)

        # 4. Calculate MPVE loss
        predicted_rewards = reward_fn(pred_states[:, :-1, :], pred_actions, pred_states[:, 1:, :])
        predicted_tokens = torch.cat([pred_states[:, :-1, :], pred_actions], dim=-1)

        with torch.no_grad():
            # Now that history_batch is correct, this concatenation works as intended.
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

        # 5. Calculate final losses
        actor_loss = -(log_probs[:-1] * advantages).mean()
        critic_loss_gae = torch.nn.functional.mse_loss(values, returns.squeeze(-1))
        critic_loss = critic_loss_gae + mpve_weight * loss_mpve

        # 6. Optimization Step
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        total_loss = actor_loss + critic_loss
        total_loss.backward()

        actor_opt.step()
        critic_opt.step()

        # 7. Logging
        episode_reward = real_rewards.sum().item()
        training_rewards.append(episode_reward)
        if step_idx % 10 == 0 or step_idx == steps - 1:
            print(
                f"  Passo {step_idx + 1}/{steps} -> Ricompensa: {episode_reward:.2f}, Actor Loss: {actor_loss:.3f}, Critic Loss: {critic_loss:.3f} (MPVE: {loss_mpve:.3f})")

    print("--- Training Completato ---\n")
    return training_rewards


# NUOVA FUNZIONE DI SIMULAZIONE
def run_simulation(env, actor, initial_state_numpy, sim_steps=50):
    """Esegue una singola simulazione e ne restituisce i dati."""
    print(f"Esecuzione simulazione per {sim_steps} passi...")
    state = env.reset(initial_state_numpy=initial_state_numpy)

    states_hist, actions_hist, timings = [state.cpu()], [], []

    for _ in range(sim_steps):
        start_time = time.perf_counter()
        action, _, _, _ = actor(state, deterministic=True)
        end_time = time.perf_counter()

        timings.append((end_time - start_time) * 1000)  # in ms

        actions_hist.append(action.cpu().detach())
        state, _, done = env.step(action)
        states_hist.append(state.cpu())

        if done:
            print("L'episodio è terminato prematuramente.")
            break

    return torch.stack(states_hist), torch.stack(actions_hist), np.array(timings)


# NUOVA FUNZIONE DI PLOTTING PER IL CONFRONTO
def plot_comparison(training_rewards, data_untrained, data_trained, dt):
    """Visualizza i risultati del training e il confronto pre/post addestramento."""
    states_untrained, actions_untrained, timings_untrained = data_untrained
    states_trained, actions_trained, timings_trained = data_trained

    # 1. Grafico della ricompensa durante il training
    plt.figure(figsize=(12, 5))
    plt.plot(training_rewards, 'b-o', markersize=4, label='Ricompensa per episodio')
    plt.title("📈 Ricompensa per Episodio durante il Training", fontsize=16)
    plt.xlabel("Passo di Training")
    plt.ylabel("Ricompensa Totale Episodio")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Grafici di confronto
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Confronto Performance Controllore: Prima vs. Dopo il Training", fontsize=18)
    time_axis = np.arange(len(states_trained)) * dt

    # ----- Colonna SINISTRA: NON ADDESTRATO -----
    axs[0, 0].set_title("Prima del Training (Non Addestrato)", fontsize=14)
    axs[0, 0].plot(np.arange(len(states_untrained)) * dt, states_untrained[:, 2].numpy(), label="Angolo Palo [rad]")
    axs[0, 0].plot(np.arange(len(states_untrained)) * dt, states_untrained[:, 0].numpy(),
                   label="Posizione Carrello [m]", linestyle='--')
    axs[0, 0].set_ylabel("Stato")
    axs[0, 0].grid(True);
    axs[0, 0].legend()
    axs[0, 0].axhline(0, color='black', linewidth=0.5, linestyle=':')

    axs[1, 0].plot(np.arange(len(actions_untrained)) * dt, actions_untrained[:, 0].numpy(), label="Forza [N]",
                   color='orange', drawstyle='steps-post')
    axs[1, 0].set_ylabel("Controllo")
    axs[1, 0].set_xlabel(f"Tempo [s]")
    axs[1, 0].grid(True);
    axs[1, 0].legend()

    # ----- Colonna DESTRA: ADDESTRATO -----
    axs[0, 1].set_title("Dopo il Training (Addestrato)", fontsize=14)
    axs[0, 1].plot(time_axis, states_trained[:, 2].numpy(), label="Angolo Palo [rad]")
    axs[0, 1].plot(time_axis, states_trained[:, 0].numpy(), label="Posizione Carrello [m]", linestyle='--')
    axs[0, 1].grid(True);
    axs[0, 1].legend()
    axs[0, 1].axhline(0, color='black', linewidth=0.5, linestyle=':')

    axs[1, 1].plot(np.arange(len(actions_trained)) * dt, actions_trained[:, 0].numpy(), label="Forza [N]",
                   color='orange', drawstyle='steps-post')
    axs[1, 1].set_xlabel(f"Tempo [s]")
    axs[1, 1].grid(True);
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3. Grafici aggiuntivi per analisi controllore addestrato
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Analisi Dettagliata del Controllore Addestrato", fontsize=16)

    # Errore e Derivata dell'errore (sull'angolo)
    error = -states_trained[:, 2].numpy()  # e = target - state = 0 - theta
    error_dot = -states_trained[:, 3].numpy()  # de/dt = -omega
    axs[0].plot(time_axis, error, label=r'Errore Angolo $e = -\theta$')
    axs[0].plot(time_axis, error_dot, label=r'Derivata Errore $\dot{e} = -\omega$')
    axs[0].set_title("📉 Errore e Derivata dell'Errore (Angolo)", fontsize=14)
    axs[0].set_xlabel("Tempo [s]")
    axs[0].set_ylabel("Valore")
    axs[0].grid(True);
    axs[0].legend()
    axs[0].axhline(0, color='black', linewidth=0.5, linestyle=':')

    # Tempo di esecuzione
    axs[1].plot(time_axis[:-1], timings_trained, label='Tempo di calcolo per step', color='green')
    axs[1].set_title(f"⏱️ Tempo di Esecuzione per Step (Avg: {np.mean(timings_trained):.2f} ms)", fontsize=14)
    axs[1].set_xlabel("Tempo [s]")
    axs[1].set_ylabel("Tempo [ms]")
    axs[1].grid(True);
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# PASSO 3: ESECUZIONE DELL'ESEMPIO
# =============================================================================
def main():
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione AC-MPC per Cart-Pole su dispositivo: {DEVICE}")

    # --- Parametri ---
    DT, MPC_HORIZON, TRAIN_STEPS = 0.05, 15, 250
    NX, NU = 4, 1
    INITIAL_LOG_STD = 0.0  # Valore più alto per esplorazione aggressiva (exp(0)=1.0)

    params = CartPoleParams.from_gym()
    dyn_fn = lambda x, u, dt: f_cartpole_discrete(x, u, dt, params)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        base_env = gym.make("CartPole-v1")
    env = GymTensorWrapper(base_env, DEVICE)

    # MODIFICA: Actor MPC con log_std personalizzabile
    class PatchedActorMPC(ActorMPC):
        def __init__(self, *args, initial_log_std=-2.0, **kwargs):
            super().__init__(*args, **kwargs)
            # Sovrascrivi il log_std per un'esplorazione più aggressiva
            self.log_std = torch.nn.Parameter(
                torch.ones(self.nu, device=self.device, dtype=self.dtype) * initial_log_std)
            if hasattr(self.mpc, 'linearize_dynamics'): self.mpc.debug = False

        def forward(self, x, deterministic: bool = False):
            if x.ndim == 1: x = x.unsqueeze(0)
            batch_size = x.shape[0]
            assert batch_size == 1, "PatchedActorMPC supporta solo batch size 1"
            C_batch, c_batch, C_final_batch, c_final_batch = self._generate_and_scale_costs(x)
            self.mpc.cost_module.C = C_batch.squeeze(0)
            self.mpc.cost_module.c = c_batch.squeeze(0)
            self.mpc.cost_module.C_final = C_final_batch.squeeze(0)
            self.mpc.cost_module.c_final = c_final_batch.squeeze(0)
            U_init = torch.zeros(batch_size, self.horizon, self.nu, device=self.device, dtype=self.dtype)
            u_mpc, _ = self.mpc.solve_step(x, U_init)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(u_mpc, std)
            action = u_mpc if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            predicted_states = self.mpc.X_last
            predicted_actions = self.mpc.U_last
            return (action.squeeze(0), log_prob.squeeze(0), predicted_states.squeeze(0), predicted_actions.squeeze(0))

    # --- Creazione Modelli ---
    actor = PatchedActorMPC(nx=NX, nu=NU, horizon=MPC_HORIZON, dt=DT, f_dyn=dyn_fn, device=str(DEVICE),
                            initial_log_std=INITIAL_LOG_STD)
    critic = CriticTransformer(state_dim=NX, action_dim=NU, history_len=15, pred_horizon=MPC_HORIZON, hidden_size=32,
                               num_layers=2, num_heads=2).to(DEVICE)

    # --- Stato iniziale per il test ---
    # Partiamo con una perturbazione sull'angolo per rendere il task non banale
    initial_state_np = np.array([0.0, 0.0, 0.3, 0.0])  # Angolo iniziale di ~17 gradi

    # --- ESECUZIONE E CONFRONTO ---

    # 1. Test con policy NON addestrata
    print("--- Test con Controllore NON Addestrato ---")
    data_untrained = run_simulation(env, actor, initial_state_np)

    # 2. Addestramento
    training_rewards = train_ac_mpc(
        env=env, actor=actor, critic=critic, reward_fn=reward_fn_cartpole,
        steps=TRAIN_STEPS, mpc_horizon=MPC_HORIZON, mpve_weight=0.1
    )

    # 3. Test con policy ADDESTRATA
    print("--- Test con Controllore Addestrato ---")
    data_trained = run_simulation(env, actor, initial_state_np)

    # 4. Generazione grafici di confronto
    print("Generazione grafici di confronto...")
    plot_comparison(training_rewards, data_untrained, data_trained, DT)

    env.close()


if __name__ == "__main__":
    main()

