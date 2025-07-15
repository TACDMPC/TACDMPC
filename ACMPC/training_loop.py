"""Esempio di training loop A2C semplificato."""

from __future__ import annotations
import torch
from torch import nn, optim
from dataclasses import dataclass

@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


def train_a2c(env, actor: nn.Module, critic: nn.Module, episodes: int = 10, gamma: float = 0.99):
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-4)
    for ep in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.get_default_dtype())
        done = False
        ep_reward = 0.0
        while not done:
            action, logp = actor(state)
            next_state, reward, done = env.step(action.detach())
            value = critic(
                state.unsqueeze(0),
                torch.zeros(actor.mpc.nu).unsqueeze(0),
                torch.zeros(1, 1, actor.nx),
                torch.zeros(1, 1, actor.mpc.nu),
                torch.zeros(1, actor.mpc.horizon, actor.nx),
                torch.zeros(1, actor.mpc.horizon, actor.mpc.nu),
            )
            next_value = critic(
                torch.tensor(next_state, dtype=state.dtype).unsqueeze(0),
                torch.zeros(actor.mpc.nu).unsqueeze(0),
                torch.zeros(1, 1, actor.nx),
                torch.zeros(1, 1, actor.mpc.nu),
                torch.zeros(1, actor.mpc.horizon, actor.nx),
                torch.zeros(1, actor.mpc.horizon, actor.mpc.nu),
            )
            advantage = reward + gamma * next_value - value
            actor_loss = -(logp * advantage.detach())
            critic_loss = advantage.pow(2)
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            opt_actor.step()
            opt_critic.step()
            state = torch.tensor(next_state, dtype=state.dtype)
            ep_reward += reward
        print(f"Episode {ep}: reward {ep_reward:.1f}")
