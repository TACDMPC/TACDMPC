"""Minimal PPO training loop for ActorMPC and CriticTransformer."""
import torch
import torch.optim as optim

from .actor import ActorMPC
from .critic_transformer import CriticTransformer


def rollout(env, actor: ActorMPC, critic: CriticTransformer, horizon: int):
    states = []
    actions = []
    rewards = []
    log_probs = []
    entropies = []
    values = []
    state = env.reset()
    hist = torch.zeros(1, critic.history_len, actor.nx + actor.nu, device=state.device)
    pred = torch.zeros(1, critic.pred_horizon, actor.nx + actor.nu, device=state.device)
    for _ in range(horizon):
        action, log_p, ent = actor(state, return_entropy=True)
        value = critic(state.unsqueeze(0), action.unsqueeze(0), hist, pred).squeeze(0)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(torch.tensor([reward], dtype=torch.float32, device=state.device))
        log_probs.append(log_p)
        entropies.append(ent)
        values.append(value)
        state = next_state
        if done:
            state = env.reset()
    with torch.no_grad():
        last_action, _ = actor(state)
        next_value = critic(state.unsqueeze(0), last_action.unsqueeze(0), hist, pred).squeeze(0)
    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(rewards),
        torch.stack(log_probs),
        torch.stack(entropies),
        torch.stack(values),
        next_value,
    )


def _compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor, gamma: float, lam: float):
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0.0
    v_next = next_value
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        v_next = values[t]
    return returns, advantages


def train(
    env,
    actor: ActorMPC,
    critic: CriticTransformer,
    steps: int = 100,
    horizon: int = 10,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.0,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
):
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)

    for _ in range(steps):
        (
            states,
            actions,
            rewards,
            log_probs,
            entropies,
            values,
            next_value,
        ) = rollout(env, actor, critic, horizon=horizon)

        returns, advantages = _compute_gae(
            rewards.squeeze(-1), values, next_value, gamma=gamma, lam=lam
        )
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * adv_norm.detach()).mean()
        if entropy_coef:
            actor_loss -= entropy_coef * entropies.mean()
        critic_loss = torch.nn.functional.mse_loss(values, returns)

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        if scheduler is not None:
            scheduler.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
