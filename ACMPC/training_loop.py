import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.func import vmap
from typing import Callable, Optional

from .parallel_env import ParallelEnvManager
from .actor import ActorMPC
from .critic_transformer import CriticTransformer
from .Checkpoint_Manager import CheckpointManager


def compute_gae_and_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        *,
        gamma: float = 0.99,
        lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    values_extended = torch.cat([values, values[-1:]], dim=0)
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values
    return advantages, returns


def train(
        env_fn: Callable,
        actor: ActorMPC,
        critic: CriticTransformer,
        reward_fn: Callable,
        steps: int = 1000,
        *,
        num_envs: int = 8,
        episode_len: int = 250,
        mpc_horizon: int,
        gamma: float = 0.99,
        lam: float = 0.97,
        mpve_gamma: float = 0.99,
        mpve_weight: float = 0.1,
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume_from: str = "latest",
        use_amp: bool = False,
        seed: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device = actor.device
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)
    scaler = GradScaler(enabled=use_amp)

    start_step = 0
    if checkpoint_manager:
        start_step = checkpoint_manager.load(
            device=device, resume_from=resume_from, actor=actor, critic=critic,
            actor_optimizer=actor_opt, critic_optimizer=critic_opt, scaler=scaler,
        )

    envs = ParallelEnvManager(env_fn, num_envs, device)
    batched_compute_gae = vmap(compute_gae_and_returns, in_dims=(0, 0))

    for step_idx in range(start_step, steps):
        # Fase 1: Raccolta Dati
        states_buf = torch.zeros(num_envs, episode_len, actor.nx, device=device, dtype=actor.dtype)
        actions_buf = torch.zeros(num_envs, episode_len, actor.nu, device=device, dtype=actor.dtype)
        rewards_buf = torch.zeros(num_envs, episode_len, 1, device=device, dtype=actor.dtype)
        log_probs_buf = torch.zeros(num_envs, episode_len, device=device, dtype=actor.dtype)
        pred_states_buf = torch.zeros(num_envs, episode_len, mpc_horizon + 1, actor.nx, device=device,
                                      dtype=actor.dtype)
        pred_actions_buf = torch.zeros(num_envs, episode_len, mpc_horizon, actor.nu, device=device, dtype=actor.dtype)

        current_states = envs.reset()
        for t in range(episode_len):
            # Rimosso with torch.no_grad() per permettere la propagazione dei gradienti attraverso l'MPC
            actions, log_probs, pred_states, pred_actions = actor(current_states)

            # Usiamo .detach() per passare i tensori all'ambiente, che non richiede gradienti
            next_states, rewards, dones, infos = envs.step(actions.detach())

            states_buf[:, t] = current_states
            actions_buf[:, t] = actions
            rewards_buf[:, t] = rewards.unsqueeze(1) if rewards.ndim == 1 else rewards
            log_probs_buf[:, t] = log_probs
            pred_states_buf[:, t] = pred_states.detach()
            pred_actions_buf[:, t] = pred_actions.detach()
            current_states = next_states

        # Fase 2: Calcolo delle Loss
        with autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=use_amp):
            L, D_token = critic.history_len, critic.token_dim
            real_tokens = torch.cat([states_buf.detach(), actions_buf.detach()], dim=-1)
            padded_tokens = torch.cat(
                [torch.zeros(num_envs, L - 1, D_token, device=device, dtype=actor.dtype), real_tokens], dim=1)
            history_batch = padded_tokens.unfold(dimension=1, size=L, step=1).permute(0, 1, 3, 2).contiguous()

            with torch.no_grad():
                values_all_tokens = critic(history_batch.view(-1, L, D_token), use_causal_mask=True)
                values = values_all_tokens[:, -1].view(num_envs, episode_len, 1)

            advantages, returns = batched_compute_gae(rewards_buf, values)

            # --- MODIFICA CHIAVE: Ripristinata la chiamata corretta con 3 argomenti ---
            predicted_rewards = reward_fn(
                pred_states_buf[..., :-1, :],
                pred_actions_buf,
                pred_states_buf[..., 1:, :]
            )
            # --------------------------------------------------------------------------

            predicted_tokens = torch.cat([pred_states_buf[..., :-1, :], pred_actions_buf], dim=-1)
            full_sequence = torch.cat([history_batch, predicted_tokens], dim=2)
            full_sequence_reshaped = full_sequence.view(-1, L + mpc_horizon, D_token)

            all_values = critic(full_sequence_reshaped, use_causal_mask=True)
            predicted_values = all_values[:, L:]

            with torch.no_grad():
                mpve_targets = torch.zeros_like(predicted_rewards)
                next_val = all_values[:, -1].view(num_envs, episode_len)
                for k_rev in reversed(range(mpc_horizon)):
                    target = predicted_rewards[:, :, k_rev] + mpve_gamma * next_val
                    mpve_targets[:, :, k_rev] = target
                    next_val = target

            loss_mpve = torch.nn.functional.mse_loss(predicted_values, mpve_targets.view(-1, mpc_horizon))
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # log_probs_buf ora ha i gradienti, quindi la loss dell'attore funzionerà
            actor_loss = -(log_probs_buf * advantages.squeeze(-1)).mean()

            # Per la critic_loss_gae, il target `returns` non ha gradienti. Dobbiamo calcolare
            # i valori di nuovo con i gradienti per la backpropagation.
            values_with_grad_all_tokens = critic(history_batch.view(-1, L, D_token), use_causal_mask=True)
            values_with_grad = values_with_grad_all_tokens[:, -1].view(num_envs, episode_len, 1)
            critic_loss_gae = torch.nn.functional.mse_loss(values_with_grad, returns)

            critic_loss = critic_loss_gae + mpve_weight * loss_mpve

        # Fase 3: Ottimizzazione
        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)

        # Facciamo la backward separatamente per maggiore chiarezza e robustezza
        scaler.scale(actor_loss).backward(retain_graph=True)
        scaler.scale(critic_loss).backward()

        scaler.unscale_(actor_opt);
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        scaler.unscale_(critic_opt);
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        scaler.step(actor_opt);
        scaler.step(critic_opt)
        scaler.update()

        # Logging
        with torch.no_grad():
            mean_reward = rewards_buf.mean().item()
            value_mean = values.mean().item()
            value_std = values.std().item()
        print("-" * 70)
        print(
            f"Step {step_idx + 1}/{steps} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")
        print(f"  Metrica | Reward Medio: {mean_reward:.3f} | Valore Medio: {value_mean:.3f} (± {value_std:.3f})")
        print(f"  Losses  | GAE Loss: {critic_loss_gae.item():.4f} | MPVE Loss: {loss_mpve.item():.4f}")
        print("-" * 70)

        if checkpoint_manager:
            checkpoint_manager.save(
                step=step_idx + 1, actor=actor, critic=critic,
                actor_optimizer=actor_opt, critic_optimizer=critic_opt,
                scaler=scaler, current_loss=critic_loss.item()
            )
    envs.close()