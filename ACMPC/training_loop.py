# File: training_loop.py (versione integrale con tutte le correzioni)

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.func import vmap
from typing import Callable, Optional, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gym
import os
import time
import logging
from .parallel_env import ParallelEnvManager
from .actor import ActorMPC
from .critic_transformer import CriticTransformer
from .Checkpoint_Manager import CheckpointManager

def exemple_hover_reward_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    target_pos = torch.tensor([0.0, 0.0, 5.0], device=state.device)
    pos, vel = state[..., :3], state[..., 3:6]
    pos_error = torch.sum((pos - target_pos) ** 2, dim=-1)
    vel_error = torch.sum(vel ** 2, dim=-1)
    action_penalty = torch.sum(action ** 2, dim=-1)
    return - (pos_error + 0.1 * vel_error + 0.01 * action_penalty)


def compute_gae_and_returns(
        rewards: torch.Tensor, values: torch.Tensor, *, gamma: float = 0.99, lam: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    values_extended = torch.cat([values, values[-1:]], dim=0)
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[t] = last_advantage
    return advantages + values, advantages


def train(
        env_fn: Callable, actor: ActorMPC, critic: CriticTransformer,
        reward_fn: Callable, steps: int = 1000, *, num_envs: int = 24,
        episode_len: int = 250, mpc_horizon: int, ppo_epochs: int = 10,
        clip_param: float = 0.2, entropy_coeff: float = 0.01,
        mpve_coeff: float = 0.1, gamma: float = 0.99, lam: float = 0.97,
        checkpoint_manager: Optional[CheckpointManager] = None, resume_from: str = "latest",
        use_amp: bool = False, seed: int | None = None,
) -> List[float]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device = actor.device
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    scaler = GradScaler(enabled=use_amp)

    # Logica di caricamento corretta
    start_step_idx = 0
    if checkpoint_manager:
        start_step_idx = checkpoint_manager.load(
            device=device, resume_from=resume_from, actor=actor, critic=critic,
            actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, scaler=scaler
        )

    envs = ParallelEnvManager(env_fn, num_envs, device)
    obs_dim = envs.single_observation_space_shape[0]

    batched_compute_gae = vmap(compute_gae_and_returns, in_dims=(0, 0))
    batched_reward_fn = vmap(reward_fn)

    training_rewards = []

    for step_idx in range(start_step_idx, steps):
        step_start_time = time.time()
        logging.info(f"--- Inizio Step di Training {step_idx + 1}/{steps} (indice: {step_idx}) ---")

        # FASE 1: RACCOLTA DATI (ROLLOUT)
        # ... (Questa sezione rimane invariata) ...
        logging.info(f"Fase 1: Raccolta di {episode_len} campioni da {num_envs} ambienti...")
        rollout_start_time = time.time()
        states_buf = torch.zeros(num_envs, episode_len, obs_dim, device=device, dtype=actor.dtype)
        actions_buf = torch.zeros(num_envs, episode_len, actor.nu, device=device, dtype=actor.dtype)
        rewards_buf = torch.zeros(num_envs, episode_len, device=device, dtype=actor.dtype)
        log_probs_buf = torch.zeros(num_envs, episode_len, device=device, dtype=actor.dtype)
        pred_states_buf = torch.zeros(num_envs, episode_len, actor.horizon + 1, actor.nx, device=device,
                                      dtype=actor.dtype)
        pred_actions_buf = torch.zeros(num_envs, episode_len, actor.horizon, actor.nu, device=device, dtype=actor.dtype)
        current_states = envs.reset()
        with torch.no_grad():
            for t in range(episode_len):
                actions, log_probs, pred_states, pred_actions = actor(current_states, deterministic=False)
                next_states, rewards, dones, truncated, infos = envs.step(actions)
                states_buf[:, t], actions_buf[:, t] = current_states, actions
                rewards_buf[:, t], log_probs_buf[:, t] = rewards, log_probs
                pred_states_buf[:, t], pred_actions_buf[:, t] = pred_states, pred_actions
                for i in range(num_envs):
                    if dones[i] or truncated[i]:
                        reset_obs, _ = envs.envs[i].reset()
                        next_states[i] = torch.from_numpy(reset_obs).to(device, dtype=actor.dtype)
                current_states = next_states
        logging.info(f" Raccolta dati completata. (Durata: {time.time() - rollout_start_time:.2f}s)")

        # --- MODIFICA CRITICA: Ripristinata la chiamata corretta al critico ---
        # Il critico viene chiamato su un batch di stati individuali (seq_len=1)
        with torch.no_grad():
            values_buf_for_debug = critic(states_buf.view(-1, 1, obs_dim)).view(num_envs, episode_len)
        logging.debug(f"Shape stati raccolti: {states_buf.shape}")
        logging.debug(
            f"Reward media raccolta: {rewards_buf.mean().item():.4f} | Reward std: {rewards_buf.std().item():.4f}")
        logging.debug(f"Valori medi del critico (prima di GAE): {values_buf_for_debug.mean().item():.4f}")

        logging.info("Fase 2: Calcolo Valori e Vantaggi (GAE)...")
        gae_start_time = time.time()
        with torch.no_grad():
            # --- MODIFICA CRITICA: Ripristinata la chiamata corretta al critico ---
            values_buf = critic(states_buf.view(-1, 1, obs_dim)).view(num_envs, episode_len)
        returns, advantages = batched_compute_gae(rewards_buf, values_buf)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        logging.info(f" Calcolo GAE completato. (Durata: {time.time() - gae_start_time:.2f}s)")

        logging.info(f"Fase 3: Aggiornamento PPO ({ppo_epochs} epoche)...")
        ppo_start_time = time.time()
        b_states = states_buf.view(-1, obs_dim)
        b_actions = actions_buf.view(-1, actor.nu)
        b_log_probs_old, b_advantages, b_returns = log_probs_buf.view(-1), advantages.view(-1), returns.view(-1)
        b_pred_states = pred_states_buf.view(-1, actor.horizon + 1, actor.nx)
        b_pred_actions = pred_actions_buf.view(-1, actor.horizon, actor.nu)
        ppo_batch_size = b_states.shape[0]
        mini_batch_size = min(512, ppo_batch_size)

        for epoch in range(ppo_epochs):
            permutation = torch.randperm(ppo_batch_size)
            for start in range(0, ppo_batch_size, mini_batch_size):
                end = start + mini_batch_size
                indices = permutation[start:end]

                mb_states = b_states[indices]
                mb_actions = b_actions[indices]
                mb_log_probs_old = b_log_probs_old[indices]
                mb_advantages = b_advantages[indices]
                mb_returns = b_returns[indices]
                mb_pred_states = b_pred_states[indices]
                mb_pred_actions = b_pred_actions[indices]

                with autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=use_amp):
                    new_log_probs, entropy = actor.evaluate_actions(mb_states, mb_actions)
                    ratio = torch.exp(new_log_probs - mb_log_probs_old)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()

                    # --- MODIFICA CRITICA: Ripristinata la chiamata corretta con .unsqueeze(1) ---
                    new_values = critic(mb_states.unsqueeze(1)).squeeze()
                    critic_loss_standard = F.mse_loss(new_values, mb_returns)

                    # --- BLOCCO DI CALCOLO MPVE CORRETTO E ATTIVATO ---
                    # (Questa sezione è corretta e rimane invariata)
                    current_waypoints = mb_states[:, actor.nx:]
                    horizon_len = mb_pred_states.shape[1]
                    expanded_waypoints = current_waypoints.unsqueeze(1).expand(-1, horizon_len, -1)
                    mb_pred_observations = torch.cat([mb_pred_states, expanded_waypoints], dim=-1)
                    pred_rewards = batched_reward_fn(mb_pred_observations[:, :-1, :], mb_pred_actions)
                    with torch.no_grad():
                        v_h = critic(mb_pred_observations[:, -1, :].unsqueeze(1)).squeeze()
                        mpve_targets = torch.zeros_like(pred_rewards)
                        next_return = v_h
                        for t in reversed(range(actor.horizon)):
                            mpve_targets[:, t] = pred_rewards[:, t] + gamma * next_return
                            next_return = mpve_targets[:, t]
                    v_predicted = critic(mb_pred_observations[:, :-1, :]).squeeze(
                        -1)  # Squeeze per rimuovere l'ultima dim
                    mpve_loss = F.mse_loss(v_predicted, mpve_targets)
                    # --- FINE BLOCCO MPVE ---

                    critic_loss = critic_loss_standard + mpve_coeff * mpve_loss
                    total_loss = actor_loss + entropy_coeff * entropy_loss + critic_loss

                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()


                if epoch == 0 and start == 0:
                    scaler.unscale_(actor_optimizer)
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'))
                    scaler.unscale_(critic_optimizer)
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
                    logging.debug(f"--- PPO Mini-batch Report (step {step_idx + 1}) ---")
                    logging.debug(f"  ... (logging invariato) ...")

                scaler.step(actor_optimizer)
                scaler.step(critic_optimizer)
                scaler.update()

        logging.info(f" Aggiornamento PPO completato. (Durata: {time.time() - ppo_start_time:.2f}s)")

        mean_reward = rewards_buf.mean().item()
        training_rewards.append(mean_reward)
        logging.info(
            f"📊 Risultati Step: Reward Medio: {mean_reward:.3f} | Loss Attore: {actor_loss.item():.4f} | Loss Critico: {critic_loss.item():.4f}"
        )

        if checkpoint_manager:
            # Logica di salvataggio corretta (invariata da prima)
            with torch.no_grad():
                dummy_input_dim = actor.cost_map_net[0].in_features
                actor._update_cost_module(torch.zeros(1, dummy_input_dim, device=device, dtype=actor.dtype))

            checkpoint_manager.save(
                completed_step_idx=step_idx,
                actor=actor, critic=critic,
                actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                scaler=scaler, current_loss=critic_loss.item()
            )

        logging.info(f"--- Step {step_idx + 1} completato in {time.time() - step_start_time:.2f}s ---\n")

    envs.close()
    return training_rewards