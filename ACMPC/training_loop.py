import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.func import vmap
from typing import Callable, Optional

# Importazioni corrette (non relative)
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
            actor_optimizer=actor_opt, critic_optimizer=critic_opt
        )

    envs = ParallelEnvManager(env_fn, num_envs, device)
    batched_compute_gae = vmap(compute_gae_and_returns, in_dims=(0, 0))

    for step_idx in range(start_step, steps):
        # Fase 1: Raccolta Dati
        states_buf = torch.zeros(num_envs, episode_len, actor.nx, device=device, dtype=actor.dtype)
        actions_buf = torch.zeros(num_envs, episode_len, actor.nu, device=device, dtype=actor.dtype)
        rewards_buf = torch.zeros(num_envs, episode_len, 1, device=device, dtype=actor.dtype)

        current_states = envs.reset()
        with torch.no_grad():
            for t in range(episode_len):
                actions, _, _, _ = actor(current_states, deterministic=False)
                next_states, rewards, _, _ = envs.step(actions)
                states_buf[:, t] = current_states
                actions_buf[:, t] = actions
                rewards_buf[:, t] = rewards.unsqueeze(-1)
                current_states = next_states

        # Fase 2: Calcolo delle Loss
        # --- MODIFICA CHIAVE: Aggiunto bool() per sicurezza ---
        with autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=bool(use_amp)):
            L = critic.history_len
            tokens = torch.cat([states_buf, actions_buf], dim=-1)
            padded_tokens = F.pad(tokens, (0, 0, L - 1, 0))
            critic_input_windows = padded_tokens.unfold(dimension=1, size=L, step=1).permute(0, 1, 3, 2)

            with torch.no_grad():
                values_flat = critic(critic_input_windows.reshape(-1, L, critic.token_dim))
                values = values_flat[:, -1].view(num_envs, episode_len)

            _, returns = batched_compute_gae(rewards_buf.squeeze(-1), values)

            # Ricolleghiamo i grafi per il calcolo delle loss
            _, _, _, pred_actions = actor(states_buf.reshape(-1, actor.nx))
            actor_tokens = torch.cat([states_buf.reshape(-1, actor.nx), pred_actions[:, 0, :]], dim=-1)
            values_for_actor = critic(actor_tokens.unsqueeze(1))
            actor_loss = -values_for_actor.mean()

            values_new_flat = critic(critic_input_windows.reshape(-1, L, critic.token_dim))
            values_new = values_new_flat[:, -1].view(num_envs, episode_len)
            critic_loss = F.mse_loss(values_new, returns)

        # Fase 3: Ottimizzazione
        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)

        # Usiamo una loss totale per un backward più pulito, ma qui separiamo
        # per mantenere la logica precedente, che è comunque valida.
        scaler.scale(actor_loss).backward(retain_graph=True)
        scaler.scale(critic_loss).backward()

        scaler.unscale_(actor_opt)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        scaler.step(actor_opt)

        scaler.unscale_(critic_opt)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        scaler.step(critic_opt)

        scaler.update()

        # Logging
        with torch.no_grad():
            mean_reward = rewards_buf.mean().item()
        print("-" * 70)
        print(
            f"Step {step_idx + 1}/{steps} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")
        print(f"  Metrica | Reward Medio: {mean_reward:.3f}")
        print("-" * 70)

        if checkpoint_manager:
            checkpoint_manager.save(
                step=step_idx + 1, actor=actor, critic=critic,
                actor_optimizer=actor_opt, critic_optimizer=critic_opt,
                scaler=scaler, current_loss=critic_loss.item()
            )
    envs.close()