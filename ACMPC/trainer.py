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
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
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
actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR, weight_decay=1e-4)
critic_optimizer = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LR_CRITIC)
log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
alpha_optimizer = optim.Adam([log_alpha], lr=LR_ALPHA)
alpha = log_alpha.exp().item()

replay_buffer = SequenceReplayBuffer(BUFFER_SIZE, HISTORY_LEN, NX, NU, DEVICE)
start_episode = load_checkpoint(actor, critic_1, critic_2, actor_optimizer, critic_optimizer, "checkpoint.pth", DEVICE)

# --------------------------- AMP helpers ------------------------------------
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F                                     # noqa: E402

# scaler inizializzati una-tantum (fuori dal training-loop)
critic_scaler: GradScaler = GradScaler(enabled=DEVICE.startswith("cuda"))
actor_scaler:  GradScaler = GradScaler(enabled=DEVICE.startswith("cuda"))

# ---------------------------------------------------------------------------
def update_step() -> None:
    global alpha
    if len(replay_buffer) < BATCH_SIZE + HISTORY_LEN:
        return

    history, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)
    dummy_U = torch.zeros(HORIZON, NU, device=DEVICE)
    with torch.no_grad():                                              # target-path
        next_a, next_logp = [], []
        for s in next_state:                                           # loop finché
            a_i, lp_i = actor.get_action(s, dummy_U)                   # get_action
            next_a.append(a_i), next_logp.append(lp_i)                 # non è batch-safe
        next_a     = torch.stack(next_a)                               # (B, nu)
        next_logp  = torch.stack(next_logp).unsqueeze(1)               # (B, 1)

        next_hist  = torch.roll(history, shifts=-1, dims=1)
        next_hist[:, -1, :] = torch.cat([next_state, next_a], dim=-1)

        q_t1 = critic_target_1(next_hist)
        q_t2 = critic_target_2(next_hist)
        q_target = reward + GAMMA * (1.0 - done) * (torch.min(q_t1, q_t2) - alpha * next_logp)

    with autocast(enabled=DEVICE.startswith("cuda")):                  # mixed-precision
        q1 = critic_1(history)
        q2 = critic_2(history)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    critic_optimizer.zero_grad(set_to_none=True)
    critic_scaler.scale(critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(critic_1.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(critic_2.parameters(), 1.0)
    critic_scaler.step(critic_optimizer)
    critic_scaler.update()
    s_t = history[:, -1, :NX]                                          # (B, nx)
    a_pi, logp_pi = [], []
    for s in s_t:                                                      # idem, non-batch
        a_i, lp_i = actor.get_action(s, dummy_U, deterministic=False)
        a_pi.append(a_i), logp_pi.append(lp_i)
    a_pi   = torch.stack(a_pi)                                         # (B, nu)
    logp_pi = torch.stack(logp_pi).unsqueeze(1)                        # (B, 1)
    hist_pi = history.clone()
    hist_pi[:, -1, NX:] = a_pi

    with autocast(enabled=DEVICE.startswith("cuda")):
        q1_pi = critic_1(hist_pi)
        q2_pi = critic_2(hist_pi)
        q_pi  = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * logp_pi - q_pi).mean()

    actor_optimizer.zero_grad(set_to_none=True)
    actor_scaler.scale(actor_loss).backward()
    torch.nn.utils.clip_grad_norm_(actor.actor_net.parameters(), 1.0)
    actor_scaler.step(actor_optimizer)
    actor_scaler.update()

    # ============================================================ temperature
    alpha_loss = -(log_alpha * (logp_pi.detach() + TARGET_ENTROPY)).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optimizer.step()
    alpha = log_alpha.exp().item()                                     # sync

    # ============================================================ soft-update
    with torch.no_grad():
        for tgt, src in zip(critic_target_1.parameters(), critic_1.parameters()):
            tgt.data.lerp_(src.data, TAU)
        for tgt, src in zip(critic_target_2.parameters(), critic_2.parameters()):
            tgt.data.lerp_(src.data, TAU)


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