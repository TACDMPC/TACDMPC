"""Minimal PPO-style training loop leveraging differentiable MPC."""
from __future__ import annotations
import torch
from torch import nn, optim
from torch.distributions import Normal
from collections import deque
from ACMPC.actor import ActorMPC
from ACMPC.critic_transformer import CriticTransformer


def train(
    env,
    actor: ActorMPC,
    critic: CriticTransformer,
    *,
    episodes: int = 10,
    horizon: int = 200,
    gamma: float = 0.99,
    lr: float = 3e-4,
) -> None:
    device = next(actor.parameters()).device
    opt_actor = optim.Adam(actor.parameters(), lr=lr)
    opt_critic = optim.Adam(critic.parameters(), lr=lr)
    for ep in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        values = []
        states = deque(maxlen=critic.history_len)
        actions = deque(maxlen=critic.history_len)
        for t in range(horizon):
            hist_states = list(states) + [state.cpu().numpy()]
            hist_actions = list(actions) + [torch.zeros(actor.nu)]
            while len(hist_states) < critic.history_len:
                hist_states.insert(0, hist_states[0])
                hist_actions.insert(0, hist_actions[0])
            hist = torch.tensor(
                [list(s) + list(a) for s, a in zip(hist_states, hist_actions)],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            value = critic(hist)
            action, logp = actor(state)
            next_state, reward, done = env.step(action.detach())
            log_probs.append(logp)
            rewards.append(torch.tensor([reward], device=device))
            values.append(value)
            states.append(state.cpu())
            actions.append(action.detach().cpu())
            state = next_state
            if done:
                break
        returns = []
        R = torch.zeros(1, device=device)
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        opt_actor.zero_grad()
        opt_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        opt_actor.step()
        opt_critic.step()
        print(f"Episode {ep} reward {sum(r.item() for r in rewards):.2f}")
