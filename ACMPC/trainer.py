# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import time
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod
from Actor import ActorMPC
from Critic import CriticTransformer
from ReplayBuffer import SequenceReplayBuffer
from checkpoint import save_checkpoint, load_checkpoint

# =============================================================================
# 2. DEFINIZIONE DELL'AMBIENTE (dal tuo script di esempio)
# =============================================================================
ACTION_SCALE = 20.0 # Scala l'output di tanh [-1, 1] alla forza reale

@dataclass(frozen=True)
class CartPoleParams:
    m_c: float; m_p: float; l: float; g: float
    @classmethod
    def from_gym(cls):
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            env = gym.make("CartPole-v1")
        return cls(m_c=float(env.unwrapped.masscart), m_p=float(env.unwrapped.masspole),
                   l=float(env.unwrapped.length), g=float(env.unwrapped.gravity))

def f_cartpole_dyn(x, u, dt, p):
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass, m_p_l = p.m_c + p.m_p, p.m_p * p.l
    temp = (force + m_p_l * omega ** 2 * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0 / 3.0 - p.m_p * cos_t ** 2 / total_mass))
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass
    return torch.stack((pos + vel * dt, vel + vel_dd * dt, theta + omega * dt, omega + theta_dd * dt), dim=-1)

class CartPoleEnv:
    def __init__(self, dt=0.05, device="cpu"):
        self.params = CartPoleParams.from_gym()
        self.dt = dt
        self.device = device
        self.state = None
        self.dyn_func = lambda x, u, dt: f_cartpole_dyn(x, u, dt, self.params)

    def reset(self):
        self.state = torch.tensor([0.0, 0.0, 0.2, 0.0], device=self.device, dtype=torch.float64)
        return self.state

    def step(self, action: torch.Tensor):
        if self.state is None: raise RuntimeError("Chiamare reset() prima di step().")
        state_batch = self.state.unsqueeze(0)
        action_batch = action.to(self.device).unsqueeze(0)
        self.state = self.dyn_func(state_batch, action_batch, self.dt).squeeze(0)
        pos, _, theta, _ = torch.unbind(self.state)
        reward = torch.exp(-pos.abs()) + torch.exp(-theta.abs()) - 0.001 * (action**2).sum()
        done = bool(pos.abs() > 2.4 or theta.abs() > 0.6)
        return self.state, reward.item(), done

# =============================================================================
# 3. INIZIALIZZAZIONE E LOOP DI TRAINING (invariato)
# =============================================================================
# --- Parametri di Configurazione ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
NX, NU, HORIZON, DT = 4, 1, 15, 0.05
HISTORY_LEN = 10
BUFFER_SIZE, BATCH_SIZE = 50000, 256
GAMMA, TAU = 0.99, 0.005
LR_ACTOR, LR_CRITIC, LR_ALPHA = 3e-4, 3e-4, 3e-4
TARGET_ENTROPY = -float(NU)

# --- Inizializzazione Componenti ---
env = CartPoleEnv(dt=DT, device=DEVICE)
actor = ActorMPC(nx=NX, nu=NU, horizon=HORIZON, dt=DT, f_dyn=env.dyn_func, device=DEVICE)
critic_1 = CriticTransformer(nx=NX, nu=NU, history_len=HISTORY_LEN)
critic_2 = CriticTransformer(nx=NX, nu=NU, history_len=HISTORY_LEN)
critic_target_1 = CriticTransformer(nx=NX, nu=NU, history_len=HISTORY_LEN)
critic_target_2 = CriticTransformer(nx=NX, nu=NU, history_len=HISTORY_LEN)

critic_target_1.load_state_dict(critic_1.state_dict())
critic_target_2.load_state_dict(critic_2.state_dict())
actor.to(DEVICE); critic_1.to(DEVICE); critic_2.to(DEVICE)
critic_target_1.to(DEVICE); critic_target_2.to(DEVICE)

actor_optimizer = optim.Adam(actor.actor_net.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LR_CRITIC)
log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
alpha_optimizer = optim.Adam([log_alpha], lr=LR_ALPHA)
alpha = log_alpha.exp().item()

replay_buffer = SequenceReplayBuffer(BUFFER_SIZE, HISTORY_LEN, NX, NU, DEVICE)
start_episode = load_checkpoint(actor, critic_1, critic_2, actor_optimizer, critic_optimizer, "checkpoint.pth", DEVICE)


# =============================================================================
# 2. FUNZIONE DI AGGIORNAMENTO (SENZA PLACEHOLDER)
# =============================================================================
def update_step():
    # --- CORREZIONE QUI ---
    # Dichiara che 'alpha' si riferisce alla variabile globale
    global alpha

    if len(replay_buffer) < BATCH_SIZE + HISTORY_LEN:
        return

    history, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)

    # --- AGGIORNAMENTO CRITICI ---
    with torch.no_grad():
        U_init_dummy = torch.zeros(HORIZON, NU, device=DEVICE)
        next_actions_dist, next_log_probs = [], []

        # Calcoliamo le prossime azioni e log_probs per l'intero batch
        for i in range(BATCH_SIZE):
            na, nlp = actor.get_action(next_state[i], U_init_dummy)
            next_actions_dist.append(na)
            next_log_probs.append(nlp)

        next_actions = torch.stack(next_actions_dist)
        next_log_probs = torch.stack(next_log_probs).unsqueeze(1)

        next_history = torch.roll(history, shifts=-1, dims=1)
        next_history[:, -1, :] = torch.cat([next_state, next_actions], dim=-1)

        q_target1 = critic_target_1(next_history)
        q_target2 = critic_target_2(next_history)
        q_target_min = torch.min(q_target1, q_target2)
        q_target = reward + GAMMA * (1.0 - done) * (q_target_min - alpha * next_log_probs)

    q_current1 = critic_1(history)
    q_current2 = critic_2(history)
    critic_loss = nn.MSELoss()(q_current1, q_target) + nn.MSELoss()(q_current2, q_target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # --- AGGIORNAMENTO ATTORE E ALPHA ---
    # Ricostruiamo lo stato corrente dall'ultimo elemento della storia
    current_state_from_history = history[:, -1, :NX]
    actions_pred, log_probs_pred = [], []
    for i in range(BATCH_SIZE):
        ap, lpp = actor.get_action(current_state_from_history[i], U_init_dummy)
        actions_pred.append(ap)
        log_probs_pred.append(lpp)
    log_probs_pred = torch.stack(log_probs_pred).unsqueeze(1)

    q_actor1 = critic_1(history).detach()
    q_actor2 = critic_2(history).detach()
    actor_loss = (alpha * log_probs_pred - torch.min(q_actor1, q_actor2)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Aggiornamento alpha
    alpha_loss = -(log_alpha * (log_probs_pred.detach() + TARGET_ENTROPY)).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    alpha = log_alpha.exp().item()

    # --- AGGIORNAMENTO RETI TARGET ---
    with torch.no_grad():
        for target_param, param in zip(critic_target_1.parameters(), critic_1.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        for target_param, param in zip(critic_target_2.parameters(), critic_2.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


# =============================================================================
# 3. LOOP DI TRAINING PRINCIPALE
# =============================================================================
NUM_EPISODI = 5000
MAX_STEPS_PER_EPISODIO = 400
SAVE_EVERY_EPISODES = 20

print(f"🚀 Inizio addestramento su dispositivo: {DEVICE}")
total_steps = 0
for episode in range(start_episode, NUM_EPISODI):
    state = env.reset()
    episode_reward = 0
    episode_history = deque(maxlen=HISTORY_LEN)

    for t in range(MAX_STEPS_PER_EPISODIO):
        # Azione basata sullo stato corrente
        U_init = torch.zeros(HORIZON, NU, device=DEVICE)
        # ==================== MODIFICA DI DEBUG ====================
        print(f"Ep.{episode}, Step {t}: Chiamata a actor.get_action...")
        # =========================================================
        action, _ = actor.get_action(state, U_init, deterministic=False)
        action_np = action.detach().cpu().numpy()
        # ==================== MODIFICA DI DEBUG ====================
        print(f"Ep.{episode}, Step {t}: Azione ricevuta.")
        # =========================================================
        # Interazione con l'ambiente
        next_state, reward, done = env.step(action)
        episode_reward += reward
        total_steps += 1

        # Memorizzazione nel buffer
        replay_buffer.store(state.detach().cpu().numpy(), action_np, reward, next_state.detach().cpu().numpy(), done)

        state = next_state

        # ==================== MODIFICA DI DEBUG ====================
        print(f"Ep.{episode}, Step {t}: Chiamata a update_step...")
        # =========================================================
        update_step()
        # ==================== MODIFICA DI DEBUG ====================
        print(f"Ep.{episode}, Step {t}: update_step completato.")
        # =========================================================
        if done:
            break

    print(f"Episodio: {episode}, Step Totali: {total_steps}, Ricompensa: {episode_reward:.2f}, Alpha: {alpha:.4f}")

    # Salvataggio periodico del checkpoint
    if episode % SAVE_EVERY_EPISODES == 0:
        save_checkpoint(actor, critic_1, critic_2, actor_optimizer, critic_optimizer, episode, "checkpoint.pth")